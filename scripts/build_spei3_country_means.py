# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Country-mean SPEI-3 from the Copernicus ERA5 drought grid, once, offline.

The drought path has no observation that reaches back before the HDX and
NMME ingests began, so the backcast resolves INCONCLUSIVE for most of its
window: a cell with fewer than ``drought.indicators.min_present_readings``
feeds that NAMED it is "not looked at", and one with fewer than
``min_answered_for_zero`` cannot rest a zero on absence. SPEI-3 is a
water-balance anomaly published monthly on a global grid since long before
either, so it is the observation those years are missing.

**This script is not on the resolution path and must never be called from
one.** Turning a raster into a country number is zonal statistics, which
the rulebook comment beside the ``tabular`` provider says does not belong
in a resolution run — it needs a scientific stack the pipeline does not
install, credentials the pipeline does not hold, and tens of gigabytes of
download. It is run by hand, its CSV is committed, and the rulebook's
existing ``tabular`` provider reads that file. Re-run it to extend the
series; nothing in CI runs it.

Two stages, deliberately separable so the slow half is done once:

    # 1. fetch the grid (needs a CDS account and ~/.cdsapirc)
    python -m scripts.build_spei3_country_means fetch \\
        --from 2016-01 --to 2026-08 --raw-dir data/spei3_raw

    # 2. reduce it to country means against the vendored boundaries
    python -m scripts.build_spei3_country_means reduce \\
        --raw-dir data/spei3_raw \\
        --out resolver/data/spei3_country_means.csv

Boundaries are the layer the rest of the machine already uses
(``resolver/hazard_resolution/data/ne_50m_admin_0_countries.slim.geojson.gz``),
restricted to the ISO3s in ``resolver/data/countries.csv``. A second
boundary source would give two answers for one country.

**A country smaller than a grid cell is sampled, not dropped.** A mask
built from cell CENTRES omits it entirely rather than skipping it, so it
is invisible in every skip counter — which is how NMME reported "175
countries produced, 1 skipped" against 252 in the run. Such a country
takes the value of the nearest cell centre and is labelled
``nearest_cell`` in the coverage report, so a reader can tell a sampled
country from a covered one.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
COUNTRIES_CSV = REPO_ROOT / "resolver" / "data" / "countries.csv"
DEFAULT_OUT = REPO_ROOT / "resolver" / "data" / "spei3_country_means.csv"

#: The Copernicus dataset and variable this reads. SPEI-3 is the
#: three-month Standardised Precipitation-Evapotranspiration Index: a
#: water-balance anomaly in sigma units, which is why the rulebook can
#: threshold it at -1.0 exactly as it thresholds the NMME anomaly.
CDS_DATASET = "derived-drought-historical-monthly"
CDS_VARIABLE = "standardised_precipitation_evapotranspiration_index_3_month"

#: The first month the series is built for. Before 2016 the drought path
#: has no indicator at all; this is the hole it exists to fill.
DEFAULT_START_YM = "2016-01"

#: How a country's value was arrived at.
COVERAGE_CELLS = "cells"
COVERAGE_NEAREST = "nearest_cell"
COVERAGE_NONE = "no_value"


# ---------------------------------------------------------------------------
# Month arithmetic (calendar months, never 30-day jumps)
# ---------------------------------------------------------------------------


def month_range(start_ym: str, end_ym: str) -> list[str]:
    """Every ``YYYY-MM`` from ``start_ym`` to ``end_ym`` inclusive.

    Stepped in CALENDAR months. Subtracting thirty days at a time is how
    an ACAPS window asked for Mar, Jan, Dec, Dec and never February.
    """

    start_y, start_m = (int(p) for p in start_ym.split("-"))
    end_y, end_m = (int(p) for p in end_ym.split("-"))
    out: list[str] = []
    year, month = start_y, start_m
    while (year, month) <= (end_y, end_m):
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return out


def previous_complete_month(today: dt.date | None = None) -> str:
    """The newest month that has ENDED.

    The current month is still being observed, and a partial month's
    anomaly is not the month's anomaly.
    """

    today = today or dt.date.today()
    first = today.replace(day=1)
    last_complete = first - dt.timedelta(days=1)
    return f"{last_complete.year:04d}-{last_complete.month:02d}"


# ---------------------------------------------------------------------------
# The zonal mean — a pure function over a plain grid
# ---------------------------------------------------------------------------


@dataclass
class Grid:
    """One month of gridded values on a regular lat/lon grid.

    ``values[i][j]`` is the value at ``(lats[i], lons[j])``; None (or NaN)
    is missing. Kept as plain sequences rather than an xarray object so
    the reduction below is testable without a scientific stack — which is
    what lets CI cover the rule that decides which countries get a value.
    """

    lats: Sequence[float]
    lons: Sequence[float]
    values: Sequence[Sequence[float | None]]


@dataclass
class CountryValue:
    iso3: str
    value: float | None
    coverage: str
    n_cells: int = 0


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def _cos_weight(lat: float) -> float:
    """Area weight for a cell centred at ``lat``.

    A degree of longitude is a shorter distance near the poles, so an
    unweighted mean over cell centres overstates the high-latitude end of
    a country. Never negative: a rounding error at the pole must not
    subtract a cell.
    """

    return max(0.0, math.cos(math.radians(lat)))


def shapely_contains(country: Any, lon: float, lat: float) -> bool:
    """Is ``(lon, lat)`` inside ``country``'s territory? The real predicate.

    Injected rather than called inline so the reduction below is a pure
    function of its arguments, and so the tests that pin the rule about
    small countries need no geometry stack to run. A test file that skips
    wholesale on a missing dependency is a test file whose failures nobody
    sees until CI.
    """

    from shapely.geometry import Point

    return bool(country.geom.contains(Point(lon, lat)))


def country_mean(
    grid: Grid,
    country: Any,
    *,
    nearest_when_uncovered: bool = True,
    contains: Any = None,
) -> CountryValue:
    """The area-weighted mean of ``grid`` over ``country``'s territory.

    Cells are attributed by their CENTRE, which is the ordinary rule and
    the one that loses small countries: a state smaller than a cell
    contains no centre and would come back empty. It is sampled from the
    nearest cell centre inside its own bounding box instead, and says so,
    because "absent from the mask" and "no data for this country" are
    different facts and only the second belongs in the output.
    """

    contains = contains or shapely_contains
    minx, miny, maxx, maxy = country.bounds
    total = 0.0
    weight_sum = 0.0
    n_cells = 0
    for i, lat in enumerate(grid.lats):
        if lat < miny or lat > maxy:
            continue
        row = grid.values[i]
        for j, lon in enumerate(grid.lons):
            if lon < minx or lon > maxx:
                continue
            value = row[j]
            if _is_missing(value):
                continue
            if not contains(country, lon, lat):
                continue
            weight = _cos_weight(lat)
            total += float(value) * weight
            weight_sum += weight
            n_cells += 1

    if weight_sum > 0:
        return CountryValue(country.iso3, total / weight_sum, COVERAGE_CELLS, n_cells)
    if not nearest_when_uncovered:
        return CountryValue(country.iso3, None, COVERAGE_NONE, 0)

    # No cell centre fell inside the territory. The country is smaller
    # than a cell, not missing from the world.
    sampled = _nearest_cell_value(grid, country)
    if sampled is None:
        return CountryValue(country.iso3, None, COVERAGE_NONE, 0)
    return CountryValue(country.iso3, sampled, COVERAGE_NEAREST, 0)


def _nearest_cell_value(grid: Grid, country: Any) -> float | None:
    """The value at the grid cell nearest the country's centroid, or None."""

    minx, miny, maxx, maxy = country.bounds
    clat = (miny + maxy) / 2.0
    clon = (minx + maxx) / 2.0
    best: tuple[float, float] | None = None
    for i, lat in enumerate(grid.lats):
        row = grid.values[i]
        for j, lon in enumerate(grid.lons):
            if _is_missing(row[j]):
                continue
            # Squared degrees, longitude scaled by latitude: good enough to
            # order candidates, and it needs no projection.
            dlat = lat - clat
            dlon = (lon - clon) * _cos_weight(clat)
            dist = dlat * dlat + dlon * dlon
            if best is None or dist < best[0]:
                best = (dist, float(row[j]))
    return None if best is None else best[1]


# ---------------------------------------------------------------------------
# Countries
# ---------------------------------------------------------------------------


def load_target_iso3s(path: Path | None = None) -> list[str]:
    """The ISO3s in ``resolver/data/countries.csv``, in file order.

    ``utf-8-sig``: the file carries a BOM, and reading it as plain utf-8
    puts an invisible character on the first column name.
    """

    src = path or COUNTRIES_CSV
    out: list[str] = []
    with open(src, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            iso3 = str(row.get("iso3") or "").strip().upper()
            if len(iso3) == 3:
                out.append(iso3)
    return out


def load_boundaries(iso3s: Iterable[str]) -> dict[str, Any]:
    """The vendored boundary layer, restricted to ``iso3s``.

    The SAME layer cyclone detection and the GDACS country fallback use.
    A second boundary source would give two answers for one country.
    """

    from resolver.hazard_resolution.geometry import load_country_geometries

    wanted = {str(i).strip().upper() for i in iso3s}
    return {k: v for k, v in load_country_geometries().items() if k in wanted}


# ---------------------------------------------------------------------------
# Reduce
# ---------------------------------------------------------------------------


@dataclass
class ReduceReport:
    months: list[str] = field(default_factory=list)
    rows: int = 0
    by_coverage: dict[str, int] = field(default_factory=dict)
    countries_never_valued: list[str] = field(default_factory=list)
    countries_without_boundary: list[str] = field(default_factory=list)
    months_missing: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "months": self.months,
            "rows": self.rows,
            "by_coverage": self.by_coverage,
            "countries_never_valued": self.countries_never_valued,
            "countries_without_boundary": self.countries_without_boundary,
            "months_missing": self.months_missing,
        }


def reduce_grids(
    grids: dict[str, Grid],
    countries: dict[str, Any],
    *,
    iso3s: Sequence[str],
    contains: Any = None,
) -> tuple[list[dict[str, Any]], ReduceReport]:
    """Country means for every month in ``grids``.

    Returns ``(rows, report)`` where a row is
    ``{"iso3", "ym", "value", "coverage", "n_cells"}``. A country with no
    value in a month writes NO row: an absent reading is unknown, and the
    rulebook entry is ``absence_means_no_drought: false`` precisely so
    that absence cannot be read as a quiet month.
    """

    report = ReduceReport(months=sorted(grids))
    report.countries_without_boundary = sorted(
        set(iso3s) - set(countries)
    )
    valued: set[str] = set()
    rows: list[dict[str, Any]] = []

    for ym in sorted(grids):
        grid = grids[ym]
        for iso3 in iso3s:
            country = countries.get(iso3)
            if country is None:
                continue
            result = country_mean(grid, country, contains=contains)
            report.by_coverage[result.coverage] = (
                report.by_coverage.get(result.coverage, 0) + 1
            )
            if result.value is None:
                continue
            valued.add(iso3)
            rows.append(
                {
                    "iso3": iso3,
                    "ym": ym,
                    "value": round(float(result.value), 4),
                    "coverage": result.coverage,
                    "n_cells": result.n_cells,
                }
            )

    report.rows = len(rows)
    report.countries_never_valued = sorted(set(countries) - valued)
    return rows, report


def write_csv(rows: Sequence[dict[str, Any]], out: Path) -> None:
    """Write the feed the ``tabular`` provider reads.

    ``iso3,ym,value`` are the columns that provider looks for; ``coverage``
    and ``n_cells`` ride along so a reader can tell a country whose value
    is a mean of two hundred cells from one sampled off the nearest.
    """

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["iso3", "ym", "value", "coverage", "n_cells"]
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["ym"], r["iso3"])):
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Reading the downloaded grid
# ---------------------------------------------------------------------------


def load_grids_from_dir(raw_dir: Path) -> dict[str, Grid]:
    """Every ``*.nc`` under ``raw_dir`` as ``{ym: Grid}``.

    Needs xarray + netCDF4, which is why it is quarantined here: nothing
    on the resolution path imports this module, so the pipeline never has
    to carry the dependency.
    """

    import numpy as np
    import xarray as xr

    grids: dict[str, Grid] = {}
    for path in sorted(raw_dir.glob("*.nc")):
        with xr.open_dataset(path) as ds:
            name = _spei_variable_name(ds)
            da = ds[name]
            lat_name = _coord_name(da, ("lat", "latitude"))
            lon_name = _coord_name(da, ("lon", "longitude"))
            time_name = _coord_name(da, ("time", "valid_time", "forecast_reference_time"))
            lats = [float(v) for v in da[lat_name].values]
            lons = [float(v) for v in da[lon_name].values]
            for stamp in da[time_name].values:
                ym = str(np.datetime_as_string(stamp, unit="M"))
                slab = da.sel({time_name: stamp})
                slab = slab.transpose(lat_name, lon_name)
                values = [
                    [None if np.isnan(v) else float(v) for v in row]
                    for row in np.asarray(slab.values, dtype="float64")
                ]
                grids[ym] = Grid(lats=lats, lons=lons, values=values)
    return grids


def _spei_variable_name(ds: Any) -> str:
    """The SPEI variable in a CDS NetCDF, whatever they called it.

    A fixed name answers "absent" for a variable that is right there, and
    CDS renames between dataset versions. The candidates are named and a
    miss says what the file actually held.
    """

    names = [str(n) for n in ds.data_vars]
    for candidate in names:
        low = candidate.lower()
        if "spei" in low:
            return candidate
    if len(names) == 1:
        return names[0]
    raise ValueError(
        f"no SPEI variable in {names!r} — name it explicitly if CDS has "
        "renamed it again"
    )


def _coord_name(da: Any, candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in da.coords or candidate in da.dims:
            return candidate
    raise ValueError(
        f"none of {list(candidates)} is a coordinate of the SPEI array; "
        f"it carries {list(da.coords)}"
    )


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def fetch_grids(raw_dir: Path, months: Sequence[str]) -> list[Path]:
    """Download one NetCDF per year from the CDS. Needs ``~/.cdsapirc``.

    Requested a year at a time: a request per month is thousands of jobs
    in the CDS queue for the same data, and the whole series in one
    request is a job large enough to be refused.
    """

    import cdsapi

    raw_dir.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    by_year: dict[str, list[str]] = {}
    for ym in months:
        year, month = ym.split("-")
        by_year.setdefault(year, []).append(month)

    written: list[Path] = []
    for year in sorted(by_year):
        target = raw_dir / f"spei3_{year}.nc"
        if target.exists():
            LOG.info("[spei3] %s already downloaded — skipping", target.name)
            written.append(target)
            continue
        LOG.info("[spei3] requesting %s (%d month(s))", year, len(by_year[year]))
        client.retrieve(
            CDS_DATASET,
            {
                "variable": CDS_VARIABLE,
                "year": year,
                "month": sorted(by_year[year]),
                "format": "netcdf",
            },
            str(target),
        )
        written.append(target)
    return written


# ---------------------------------------------------------------------------
# Coverage: what the feed is meant to move
# ---------------------------------------------------------------------------

#: The two reason codes this feed exists to reduce. A cell no indicator
#: NAMED is "not looked at" (`indicator_no_coverage`); a cell whose feeds
#: are too few to rest a zero on absence is INCONCLUSIVE
#: (`indicator_too_few_feeds_for_zero`). Both are drought cells the machine
#: declined to decide, and both are outside the occurrence base-rate
#: denominator — counting them as quiet years is what manufactured a
#: near-zero drought rate out of nine years no indicator covered.
COVERAGE_REASONS = (
    "indicator_no_coverage",
    "indicator_too_few_feeds_for_zero",
)

COVERAGE_SQL = """
SELECT
  json_extract_string(trigger_detail_json, '$.no_row_reason') AS reason_code,
  run_type,
  COUNT(*) AS cells
FROM haz_triggers
WHERE hazard = 'DR'
  AND json_extract_string(trigger_detail_json, '$.no_row_reason') IN ({reasons})
GROUP BY 1, 2
ORDER BY 1, 2
"""


def coverage_counts(con: Any) -> list[dict[str, Any]]:
    """DR cells the drought gate declined to decide, by reason and run type.

    Run this BEFORE wiring the feed and again after the next backcast:
    the difference is what the feed bought. Read-only.

    Split by ``run_type`` deliberately. The backcast is where the hole is —
    live months have HDX and NMME — so a fall concentrated in `backcast`
    is the feed doing its job, and one concentrated in `live` would mean
    something else changed.
    """

    reasons = ", ".join(f"'{r}'" for r in COVERAGE_REASONS)
    rows = con.execute(COVERAGE_SQL.format(reasons=reasons)).fetchall()
    return [
        {"reason_code": r[0], "run_type": r[1], "cells": int(r[2])} for r in rows
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download the SPEI-3 grid from the CDS")
    fetch.add_argument("--from", dest="start", default=DEFAULT_START_YM)
    fetch.add_argument("--to", dest="end", default=None)
    fetch.add_argument("--raw-dir", type=Path, required=True)

    reduce_p = sub.add_parser("reduce", help="grid -> country means CSV")
    reduce_p.add_argument("--raw-dir", type=Path, required=True)
    reduce_p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    reduce_p.add_argument("--from", dest="start", default=DEFAULT_START_YM)
    reduce_p.add_argument("--to", dest="end", default=None)
    reduce_p.add_argument("--report-out", type=Path, default=None)

    coverage = sub.add_parser(
        "coverage",
        help="count DR cells the drought gate declined to decide (read-only)",
    )
    coverage.add_argument("--db", required=True)
    coverage.add_argument("--report-out", type=Path, default=None)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.command == "coverage":
        import duckdb

        con = duckdb.connect(args.db, read_only=True)
        try:
            counts = coverage_counts(con)
        finally:
            con.close()
        total = sum(row["cells"] for row in counts)
        for row in counts:
            LOG.info(
                "[spei3] %-34s %-9s %d",
                row["reason_code"], row["run_type"], row["cells"],
            )
        LOG.info("[spei3] %d DR cell(s) undecided for want of an indicator", total)
        if args.report_out:
            args.report_out.parent.mkdir(parents=True, exist_ok=True)
            args.report_out.write_text(
                json.dumps({"total": total, "counts": counts}, indent=2),
                encoding="utf-8",
            )
        return 0

    end = args.end or previous_complete_month()
    months = month_range(args.start, end)

    if args.command == "fetch":
        written = fetch_grids(args.raw_dir, months)
        LOG.info("[spei3] %d file(s) in %s", len(written), args.raw_dir)
        return 0

    grids = load_grids_from_dir(args.raw_dir)
    if not grids:
        LOG.error("[spei3] no *.nc files in %s — run `fetch` first", args.raw_dir)
        return 1
    wanted = set(months)
    grids = {ym: g for ym, g in grids.items() if ym in wanted}

    iso3s = load_target_iso3s()
    countries = load_boundaries(iso3s)
    rows, report = reduce_grids(grids, countries, iso3s=iso3s)
    report.months_missing = sorted(wanted - set(grids))
    write_csv(rows, args.out)

    LOG.info(
        "[spei3] wrote %d row(s) for %d month(s) to %s",
        report.rows, len(report.months), args.out,
    )
    LOG.info("[spei3] coverage: %s", report.by_coverage)
    if report.countries_never_valued:
        # Named, never counted. A country that got no value in any month is
        # a hole a reader has to be able to see.
        LOG.warning(
            "[spei3] %d country/countries got no value in any month: %s",
            len(report.countries_never_valued),
            ",".join(report.countries_never_valued),
        )
    if report.countries_without_boundary:
        LOG.warning(
            "[spei3] %d ISO3(s) in countries.csv have no boundary in the "
            "vendored layer: %s",
            len(report.countries_without_boundary),
            ",".join(report.countries_without_boundary),
        )
    if report.months_missing:
        LOG.warning(
            "[spei3] %d requested month(s) are not in the grid: %s",
            len(report.months_missing), ",".join(report.months_missing),
        )
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(
            json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
