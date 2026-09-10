# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Country-mean SPEI-3 from the Copernicus ERA5 drought grid, on a schedule.

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
install and credentials the pipeline does not hold. So the CSV is the
product: this script writes it, ``.github/workflows/spei3_refresh.yml``
runs this script, and the rulebook's existing ``tabular`` provider reads
the committed file exactly as it would a remote feed. Nothing in a
Resolver Update, a forecast run or a backcast imports this module.

It used to say the download was "tens of gigabytes", which is what made
running it by hand sound like the only option. At 0.25 degrees the grid is
1440 x 721 cells; one variable, one accumulation period, twelve months of
float32 is roughly 50 MB a year, so 2016 to present is on the order of a
gigabyte. **That figure is arithmetic, not an observation** — the CDS was
unreachable from the environment this was written in — so the first real
run reports the bytes it actually downloaded and this paragraph is to be
corrected from it.

Five stages, each its own subcommand, deliberately separable so a run that
is cut short by the CDS queue can be resumed by the next one rather than
started again:

    # 1. what this run owes: the trailing revision window, plus any month
    #    the committed CSV is missing. Never a hardcoded range.
    python -m scripts.build_spei3_country_means plan \\
        --out resolver/data/spei3_country_means.csv --window-out window.json

    # 2. fetch the years that window touches (needs CDS credentials)
    python -m scripts.build_spei3_country_means fetch \\
        --window window.json --raw-dir spei3_raw --deadline-sec 16000

    # 3. reduce against the vendored boundaries and MERGE into a candidate
    python -m scripts.build_spei3_country_means reduce \\
        --raw-dir spei3_raw --window window.json \\
        --out resolver/data/spei3_country_means.csv \\
        --candidate-out candidate.csv --report-out reduce_report.json

    # 4. gate it. A wrong file is worse than a stale one, so this decides
    #    whether the candidate is allowed to become the committed feed.
    python -m scripts.build_spei3_country_means validate \\
        --candidate candidate.csv --against resolver/data/spei3_country_means.csv \\
        --window window.json --report-out coverage_report.json

    # 5. promote the candidate and write the status file the staleness
    #    check and the drought restale both read
    python -m scripts.build_spei3_country_means promote \\
        --candidate candidate.csv --out resolver/data/spei3_country_means.csv \\
        --status-out resolver/data/spei3_status.json

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

**Reduce MERGES, it never overwrites.** A monthly run fetches three or
four months; writing the CSV from only what it fetched would replace a
ten-year series with a quarter of one, and the next backcast would then
read the hole as an absence of measurement. Merging on ``(iso3, ym)`` is
also what makes the whole workflow resumable by construction: a partial
run is a smaller merge, never a truncation.
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
from typing import Any, Iterable, Mapping, Sequence

LOG = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
COUNTRIES_CSV = REPO_ROOT / "resolver" / "data" / "countries.csv"
DEFAULT_OUT = REPO_ROOT / "resolver" / "data" / "spei3_country_means.csv"

#: The Copernicus dataset and variable this reads. SPEI-3 is the
#: three-month Standardised Precipitation-Evapotranspiration Index: a
#: water-balance anomaly in sigma units, which is why the rulebook can
#: threshold it at -1.0 exactly as it thresholds the NMME anomaly.
CDS_DATASET = "derived-drought-historical-monthly"

#: The request the CDS actually accepts, as separate constants so a
#: rejection costs one line rather than a re-reading of the whole function.
#:
#: The dataset moved to the CDS on 30 October 2025 and the shape this
#: script shipped with is refused outright ("Request has not produced a
#: valid combination of values"). Five things changed: the variable name
#: dropped its ``_3_month`` suffix and the 3 moved into
#: ``accumulation_period``; ``product_type``, ``dataset_type`` and
#: ``version`` became mandatory; ``format`` became ``data_format``; and
#: months want leading zeros.
#:
#: ``reanalysis``, never ``ensemble_members``: the ensemble product carries
#: ten realisations and multiplies the download by ten for a country mean
#: that would come out the same.
#:
#: ``unarchived``, because :func:`load_grids_from_dir` globs for ``*.nc``
#: and hands the path to xarray — a zip named ``.nc`` fails in a way that
#: takes an afternoon to read.
#:
#: **This shape comes from the ECMWF forum thread announcing the release,
#: not from a call anybody here has made.** The authoritative version is
#: the "Show API request code" button on the dataset's own download form,
#: and if the first real run is refused the error body names the offending
#: key — which is why :func:`fetch_grids` logs the request it is about to
#: send.
CDS_VARIABLE = "standardised_precipitation_evapotranspiration_index"
CDS_ACCUMULATION_PERIOD = "3"
CDS_PRODUCT_TYPE = "reanalysis"
CDS_DATASET_TYPE = "consolidated_dataset"
CDS_VERSION = "1_0"
CDS_DATA_FORMAT = "netcdf"
CDS_DOWNLOAD_FORMAT = "unarchived"

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
            fh, fieldnames=list(CSV_FIELDS),
            # LF, not the csv module's default CRLF. This file is committed
            # and read in diffs, and every other CSV under resolver/data uses
            # LF; a lone CRLF file makes every review of it noisier for no
            # reason. It has never been written, so nothing is rewritten by
            # settling it now.
            lineterminator="\n",
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
# The committed CSV: read, merge, write
# ---------------------------------------------------------------------------

CSV_FIELDS = ("iso3", "ym", "value", "coverage", "n_cells")


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """The committed feed as rows, or ``[]`` when it does not exist yet.

    ``value`` and ``n_cells`` are kept as the STRINGS the file carries, not
    parsed and re-rendered. Re-rendering is how a merge that adds nothing
    still rewrites every line: a float round-trip is only byte-stable by
    luck, and a diff full of unchanged rows is a diff nobody reads. The
    validation gates parse what they need where they need it.
    """

    if not Path(path).exists():
        return []
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            iso3 = str(row.get("iso3") or "").strip().upper()
            ym = str(row.get("ym") or "").strip()
            if len(iso3) != 3 or not ym:
                continue
            out.append({
                "iso3": iso3,
                "ym": ym,
                "value": str(row.get("value") or "").strip(),
                "coverage": str(row.get("coverage") or "").strip(),
                "n_cells": str(row.get("n_cells") or "").strip(),
            })
    return out


def merge_rows(
    existing: Sequence[Mapping[str, Any]], incoming: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """``existing`` with ``incoming`` written over it, keyed ``(iso3, ym)``.

    This is the whole safety property of the scheduled producer. A monthly
    run reduces three or four months; writing the CSV from those alone
    would replace ten years of series with a quarter of one, and every
    drought cell in the months it deleted would go back to reading as a
    country nobody measured. So a month the run did not touch is left
    exactly as it was, byte for byte, and an empty ``incoming`` is a no-op.

    An incoming row REPLACES its key rather than being dropped: ERA5T
    values are revised to final ERA5 later, so the first version of a month
    is not its last, and the trailing revision window exists precisely to
    collect the revision.
    """

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for row in existing:
        merged[(str(row["iso3"]), str(row["ym"]))] = dict(row)
    for row in incoming:
        merged[(str(row["iso3"]), str(row["ym"]))] = dict(row)
    return sorted(merged.values(), key=lambda r: (r["ym"], r["iso3"]))


def months_in(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(row["ym"]) for row in rows}


def months_gaining_coverage(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Which months the merge actually changed.

    A month counts as having gained coverage when its rows differ at all:
    it is new, it gained a country, or a value in it moved. A revised value
    can move a drought verdict as surely as a new one can, so a revision
    counts — the point of this list is to say which months the resolution
    machine now has a different answer available for, and it is what the
    drought resume ledger is restaled against.

    A month whose rows are untouched is not in the list, which is what
    keeps a routine monthly append from re-walking ten years of history.
    """

    def index(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, str]]:
        out: dict[str, dict[str, str]] = {}
        for row in rows:
            out.setdefault(str(row["ym"]), {})[str(row["iso3"])] = str(row["value"])
        return out

    old, new = index(before), index(after)
    return sorted(ym for ym, values in new.items() if old.get(ym) != values)


# ---------------------------------------------------------------------------
# Planning: what this run owes
# ---------------------------------------------------------------------------

#: How many trailing months are re-fetched every run whatever the CSV says.
#: ERA5T runs about five days behind and its values are revised to final
#: ERA5 later, so the first version of a month is not its last. Four months
#: is a guess at how long a revision takes to settle and is the one number
#: here worth tuning once two cycles have been watched.
REVISION_WINDOW_MONTHS = 4


@dataclass
class Window:
    """The months a run will fetch and reduce, and why each is in the list."""

    months: list[str] = field(default_factory=list)
    revision_months: list[str] = field(default_factory=list)
    missing_months: list[str] = field(default_factory=list)
    start_ym: str = ""
    end_ym: str = ""
    full_rebuild: bool = False

    @property
    def years(self) -> list[str]:
        return sorted({ym.split("-")[0] for ym in self.months})

    def as_dict(self) -> dict[str, Any]:
        return {
            "months": self.months,
            "revision_months": self.revision_months,
            "missing_months": self.missing_months,
            "years": self.years,
            "start_ym": self.start_ym,
            "end_ym": self.end_ym,
            "full_rebuild": self.full_rebuild,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Window":
        return cls(
            months=[str(m) for m in payload.get("months") or []],
            revision_months=[str(m) for m in payload.get("revision_months") or []],
            missing_months=[str(m) for m in payload.get("missing_months") or []],
            start_ym=str(payload.get("start_ym") or ""),
            end_ym=str(payload.get("end_ym") or ""),
            full_rebuild=bool(payload.get("full_rebuild")),
        )


def plan_window(
    existing_months: Iterable[str],
    *,
    start_ym: str = DEFAULT_START_YM,
    end_ym: str | None = None,
    revision_months: int = REVISION_WINDOW_MONTHS,
    full_rebuild: bool = False,
) -> Window:
    """The months this run owes. Never a hardcoded range.

    Two halves, and the second is what makes the producer self-healing:

    * a trailing revision window, because ERA5T is revised later and the
      first version of a month is not its last;
    * every month inside the configured range that the committed CSV does
      not hold — so a month lost to a CDS job that timed out is picked up
      on the next cycle with nobody noticing it went missing.

    A closed year the CSV already covers is never re-fetched: the CSV is
    the durable artifact and the raw grids are disposable.
    """

    end = end_ym or previous_complete_month()
    every = month_range(start_ym, end)
    if full_rebuild:
        return Window(
            months=list(every), revision_months=[], missing_months=list(every),
            start_ym=start_ym, end_ym=end, full_rebuild=True,
        )
    held = {str(m) for m in existing_months}
    revision = every[-revision_months:] if revision_months > 0 else []
    missing = [ym for ym in every if ym not in held]
    months = sorted(set(revision) | set(missing))
    return Window(
        months=months,
        revision_months=sorted(set(revision)),
        missing_months=missing,
        start_ym=start_ym,
        end_ym=end,
        full_rebuild=False,
    )


# ---------------------------------------------------------------------------
# Validation: a wrong file is worse than a stale one
# ---------------------------------------------------------------------------

#: A SPEI value is an anomaly in sigma units. Roughly -5 .. +5 is the range
#: the index occupies; beyond it the number is a fill value read as data, a
#: unit that changed, or a variable that is not SPEI at all.
SANITY_SOFT_LIMIT = 5.0

#: And above this the candidate is refused outright. A single value of 1e20
#: (NetCDF's usual fill) reaching the feed would make the whole month read
#: as wet, and the rulebook thresholds at -1.0 sigma.
SANITY_HARD_LIMIT = 6.0

#: How far the sampled share may move between runs before the candidate is
#: refused. The share of countries resolved off the NEAREST cell rather
#: than from cells inside their own territory is a property of the grid
#: resolution and the boundary layer, not of the weather — so a jump means
#: one of those two moved, and a country mean computed against a boundary
#: layer that moved under it is not the quantity the previous rows hold.
NEAREST_SHARE_JUMP_LIMIT = 0.05


@dataclass
class Validation:
    """The gates, and what each one saw. ``ok`` is the only publish signal."""

    ok: bool = True
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    months_present: list[str] = field(default_factory=list)
    months_absent: dict[str, str] = field(default_factory=dict)
    countries_by_month: dict[str, int] = field(default_factory=dict)
    rows_by_month: dict[str, int] = field(default_factory=dict)
    nearest_share: float | None = None
    previous_nearest_share: float | None = None
    changed_months: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.ok = False
        self.failures.append(message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "failures": self.failures,
            "warnings": self.warnings,
            "months_present": self.months_present,
            "months_absent": self.months_absent,
            "countries_by_month": self.countries_by_month,
            "rows_by_month": self.rows_by_month,
            "nearest_share": self.nearest_share,
            "previous_nearest_share": self.previous_nearest_share,
            "changed_months": self.changed_months,
        }


def _float_or_none(text: Any) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(value) else value


def _nearest_share(rows: Sequence[Mapping[str, Any]]) -> float | None:
    total = sum(1 for r in rows if str(r.get("coverage") or ""))
    if not total:
        return None
    nearest = sum(1 for r in rows if str(r.get("coverage")) == COVERAGE_NEAREST)
    return nearest / total


def _per_month(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    """``(distinct countries per month, rows per month)``.

    The two are equal in a well-formed feed, since the key is ``(iso3, ym)``
    — and that is why both are counted. A month where they diverge holds a
    duplicated country, which the merge cannot produce and a hand-edited
    file can.
    """

    countries: dict[str, set[str]] = {}
    rows_per_month: dict[str, int] = {}
    for row in rows:
        ym = str(row["ym"])
        countries.setdefault(ym, set()).add(str(row["iso3"]))
        rows_per_month[ym] = rows_per_month.get(ym, 0) + 1
    return (
        {ym: len(iso3s) for ym, iso3s in sorted(countries.items())},
        dict(sorted(rows_per_month.items())),
    )


def validate_candidate(
    candidate: Sequence[Mapping[str, Any]],
    previous: Sequence[Mapping[str, Any]],
    *,
    window: Window | None = None,
    absent_reasons: Mapping[str, str] | None = None,
) -> Validation:
    """Decide whether ``candidate`` may become the committed feed.

    Fail closed. When any gate fails the committed CSV is left exactly as it
    was and the candidate is kept as an artifact instead: a wrong feed is
    worse than a stale one, because a stale one costs the months it does not
    cover and a wrong one poisons every month it does.

    The gates, and what each is guarding against:

    ``sanity``
        a fill value or a changed unit read as an anomaly. Anything above
        :data:`SANITY_HARD_LIMIT` refuses the candidate; the band between
        the soft and hard limits is warned about, because a real -5.2 sigma
        exists and is not a reason to publish nothing.
    ``window``
        a month the run asked for that is neither present nor explicitly
        recorded absent WITH a reason. "It is not there" and "the CDS job
        was still queued when the deadline expired" want different
        responses, and only the second is allowed to pass quietly.
    ``country count``
        a month that lost countries. The grid does not shrink; a month whose
        country count falls has been reduced against something other than
        the boundary layer the previous rows used.
    ``sampled share``
        the same fault caught from the other side — the share resolved off
        the nearest cell is a property of the grid and the boundaries, so a
        jump means one of them moved under us.
    ``row count``
        a month already present that came back with fewer rows. Merging
        cannot do this; only a reduction that silently lost cells can.
    """

    result = Validation()
    reasons = {str(k): str(v) for k, v in (absent_reasons or {}).items()}

    # -- sanity -----------------------------------------------------------
    extreme: list[str] = []
    soft: list[str] = []
    unparseable: list[str] = []
    for row in candidate:
        value = _float_or_none(row.get("value"))
        if value is None:
            unparseable.append(f"{row.get('iso3')}/{row.get('ym')}")
            continue
        if abs(value) > SANITY_HARD_LIMIT:
            extreme.append(f"{row['iso3']}/{row['ym']}={value:g}")
        elif abs(value) > SANITY_SOFT_LIMIT:
            soft.append(f"{row['iso3']}/{row['ym']}={value:g}")
    if extreme:
        result.fail(
            f"{len(extreme)} value(s) exceed +/-{SANITY_HARD_LIMIT:g} sigma, which "
            f"is a fill value or a changed unit rather than an anomaly: "
            + ", ".join(sorted(extreme)[:8])
        )
    if unparseable:
        result.fail(
            f"{len(unparseable)} row(s) carry no readable value: "
            + ", ".join(sorted(unparseable)[:8])
        )
    if soft:
        result.warnings.append(
            f"{len(soft)} value(s) sit outside +/-{SANITY_SOFT_LIMIT:g} sigma but "
            f"inside the hard limit: " + ", ".join(sorted(soft)[:8])
        )

    # -- the window the run asked for -------------------------------------
    present = months_in(candidate)
    result.months_present = sorted(present)
    if window is not None:
        unexplained: list[str] = []
        for ym in window.months:
            if ym in present:
                continue
            reason = reasons.get(ym, "")
            if reason:
                result.months_absent[ym] = reason
            else:
                unexplained.append(ym)
        if unexplained:
            result.fail(
                f"{len(unexplained)} requested month(s) are neither present nor "
                f"recorded absent with a reason: " + ",".join(unexplained[:12])
            )
        if result.months_absent:
            result.warnings.append(
                f"{len(result.months_absent)} requested month(s) are absent with a "
                "stated reason; the next run asks for them again: "
                + "; ".join(f"{k} ({v})" for k, v in sorted(result.months_absent.items())[:8])
            )

    # -- country and row counts per month ---------------------------------
    counts_now, rows_now = _per_month(candidate)
    counts_before, rows_before = _per_month(previous)
    result.countries_by_month = counts_now
    result.rows_by_month = rows_now
    shrunk = [
        f"{ym}: {counts_now.get(ym, 0)} < {before}"
        for ym, before in counts_before.items()
        if counts_now.get(ym, 0) < before
    ]
    if shrunk:
        result.fail(
            f"{len(shrunk)} month(s) lost countries against the committed feed: "
            + ", ".join(sorted(shrunk)[:8])
        )
    lost_rows = [
        f"{ym}: {rows_now.get(ym, 0)} < {before}"
        for ym, before in rows_before.items()
        if rows_now.get(ym, 0) < before
    ]
    if lost_rows and not shrunk:
        result.fail(
            f"{len(lost_rows)} month(s) already present came back with fewer rows: "
            + ", ".join(sorted(lost_rows)[:8])
        )

    # -- the sampled share ------------------------------------------------
    result.nearest_share = _nearest_share(candidate)
    result.previous_nearest_share = _nearest_share(previous)
    if result.nearest_share is not None and result.previous_nearest_share is not None:
        jump = abs(result.nearest_share - result.previous_nearest_share)
        if jump > NEAREST_SHARE_JUMP_LIMIT:
            result.fail(
                f"the share of countries sampled off the nearest cell moved "
                f"{result.previous_nearest_share:.1%} -> {result.nearest_share:.1%} "
                f"(limit {NEAREST_SHARE_JUMP_LIMIT:.1%}); the grid or the boundary "
                "layer has moved, so these means are not the quantity the "
                "committed rows hold"
            )

    result.changed_months = months_gaining_coverage(previous, candidate)
    return result


# ---------------------------------------------------------------------------
# The status file: how a quiet feed says it has gone quiet
# ---------------------------------------------------------------------------

DEFAULT_STATUS_OUT = REPO_ROOT / "resolver" / "data" / "spei3_status.json"

STATUS_OK = "ok"
STATUS_INCOMPLETE = "incomplete"
STATUS_FAILED = "failed"


def restale_token(months: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    """A stable name for one extension of the feed.

    The drought resume ledger is restaled ONCE per extension, and something
    has to say which extension has already been applied — otherwise the
    nightly backcast frees the same months every night, re-walks them, and
    frees them again. The token is over the months that gained coverage and
    the content of the rows in them, so a re-run that changes nothing
    produces the same token and applies nothing.
    """

    import hashlib

    wanted = set(months)
    payload = "|".join(
        f"{r['ym']}:{r['iso3']}:{r['value']}"
        for r in sorted(rows, key=lambda r: (r["ym"], r["iso3"]))
        if str(r["ym"]) in wanted
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{len(wanted)}m-{digest}"


def build_status(
    rows: Sequence[Mapping[str, Any]],
    *,
    status: str = STATUS_OK,
    changed_months: Sequence[str] = (),
    months_owed: Sequence[str] = (),
    run_id: str = "",
    failure: Mapping[str, Any] | None = None,
    previous: Mapping[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """What the feed currently covers, and whether anybody should worry.

    Written on every run of the producer, INCLUDING a run whose candidate
    failed its gates — the CSV is left alone there, but the failure is
    recorded, because a feed that quietly stops extending is the failure
    mode an unattended system is worst at noticing.

    ``last_success_run_id`` is carried forward from the previous status on a
    failed run rather than blanked: "the last good run" is a fact about the
    past and a failure does not change it.
    """

    prior = dict(previous or {})
    months = sorted(months_in(rows))
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("coverage") or COVERAGE_NONE)
        counts[key] = counts.get(key, 0) + 1
    payload: dict[str, Any] = {
        "feed": "spei3_country_means",
        "generated_at": generated_at
        or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "status": status,
        "newest_month": months[-1] if months else "",
        "oldest_month": months[0] if months else "",
        "months": len(months),
        "rows": len(rows),
        "countries": len({str(r["iso3"]) for r in rows}),
        "coverage": dict(sorted(counts.items())),
        "months_owed": sorted({str(m) for m in months_owed}),
        "last_success_run_id": prior.get("last_success_run_id", ""),
        "last_failure": prior.get("last_failure") or None,
    }
    if status == STATUS_OK and run_id:
        payload["last_success_run_id"] = str(run_id)
    if failure:
        payload["last_failure"] = {
            "run_id": str(run_id),
            "at": payload["generated_at"],
            **{str(k): v for k, v in failure.items()},
        }
    elif status == STATUS_OK:
        payload["last_failure"] = prior.get("last_failure") or None

    # The restale request. The producer cannot touch the canonical DB — it
    # is deliberately outside the pythia-resolver-db concurrency group, so
    # it can never cancel the nightly backcast or the monthly ingest — so
    # the committed status file is the channel, exactly as the CSV is, and
    # the backcast applies it. See resolver/hazard_resolution/backcast.py.
    changed = sorted({str(m) for m in changed_months})
    if changed:
        payload["restale"] = {
            "hazard": "DR",
            "token": restale_token(changed, rows),
            "months": changed,
            "requested_at": payload["generated_at"],
        }
    elif prior.get("restale"):
        # Nothing changed this run, so the previous request still stands
        # until the backcast has applied it.
        payload["restale"] = prior["restale"]
    return payload


def read_status(path: Path | None = None) -> dict[str, Any]:
    """The committed status file, or ``{}`` when there is none yet."""

    target = Path(path) if path is not None else DEFAULT_STATUS_OUT
    if not target.exists():
        return {}
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def cds_request(year: str, months: Sequence[str]) -> dict[str, Any]:
    """The request body for one year. A pure function, so a test can pin it.

    Pinned by a test rather than only exercised in production, because the
    keys are the whole of what makes this work: the pre-October-2025 shape
    is not merely suboptimal, it is REFUSED, and a refused request looks
    from the outside like an outage. See :data:`CDS_VARIABLE` for what
    changed and where the shape came from.
    """

    return {
        "variable": [CDS_VARIABLE],
        "accumulation_period": [CDS_ACCUMULATION_PERIOD],
        "product_type": [CDS_PRODUCT_TYPE],
        "dataset_type": CDS_DATASET_TYPE,
        "version": CDS_VERSION,
        "year": [str(year)],
        # Leading zeros: the CDS rejects a bare "1".
        "month": sorted(f"{int(m):02d}" for m in months),
        "data_format": CDS_DATA_FORMAT,
        "download_format": CDS_DOWNLOAD_FORMAT,
    }


@dataclass
class FetchResult:
    """What one fetch pass managed, and what it still owes."""

    written: list[str] = field(default_factory=list)
    skipped_present: list[str] = field(default_factory=list)
    years_owed: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)
    bytes_downloaded: int = 0
    deadline_hit: bool = False

    @property
    def complete(self) -> bool:
        return not self.years_owed and not self.failed

    def as_dict(self) -> dict[str, Any]:
        return {
            "written": self.written,
            "skipped_present": self.skipped_present,
            "years_owed": self.years_owed,
            "failed": self.failed,
            "bytes_downloaded": self.bytes_downloaded,
            "deadline_hit": self.deadline_hit,
            "complete": self.complete,
        }


def fetch_grids(
    raw_dir: Path,
    months: Sequence[str],
    *,
    deadline_sec: float | None = None,
    client: Any = None,
    now: Any = None,
) -> FetchResult:
    """Download one NetCDF per year from the CDS. Needs CDS credentials.

    Requested a year at a time: a request per month is thousands of jobs in
    the CDS queue for the same data, and the whole series in one request is
    a job large enough to be refused.

    **A CDS request is an asynchronous job and can sit queued for hours**,
    against a six-hour cap on an Actions job. So the pass carries a
    deadline and stops STARTING years once it expires, recording the rest as
    owed. A queued job is not a fault: the caller exits zero with status
    ``incomplete``, the next run resumes on exactly those years, and merging
    rather than overwriting is what makes that safe.

    The deadline bounds how many more jobs are started, not the one already
    in flight — the same contract the extraction budget has, for the same
    reason: there is no way to abandon a request half way that does not
    also abandon what it has paid for.
    """

    import time

    clock = now or time.monotonic
    started = clock()
    raw_dir.mkdir(parents=True, exist_ok=True)

    by_year: dict[str, list[str]] = {}
    for ym in months:
        year, month = str(ym).split("-")
        by_year.setdefault(year, []).append(month)

    result = FetchResult()
    if not by_year:
        LOG.info("[spei3] nothing to fetch — the committed feed already covers the window")
        return result

    if client is None:
        import cdsapi

        client = cdsapi.Client()

    for year in sorted(by_year):
        target = raw_dir / f"spei3_{year}.nc"
        if target.exists():
            LOG.info("[spei3] %s already downloaded — skipping", target.name)
            result.skipped_present.append(year)
            result.bytes_downloaded += target.stat().st_size
            continue
        if deadline_sec is not None and (clock() - started) >= deadline_sec:
            result.deadline_hit = True
            result.years_owed.append(year)
            continue
        request = cds_request(year, by_year[year])
        # Logged in full, on purpose: when the CDS refuses a request it
        # names the offending key, and the error and the request being in
        # the same log is the difference between a one-line fix and an
        # afternoon.
        LOG.info(
            "[spei3] requesting %s (%d month(s)) from %s: %s",
            year, len(by_year[year]), CDS_DATASET, json.dumps(request, sort_keys=True),
        )
        try:
            client.retrieve(CDS_DATASET, request, str(target))
        except Exception as exc:  # noqa: BLE001 - the reason is the payload
            # A failed year is owed, not fatal. The other years may well be
            # fine, and the run's own gates decide whether what it did get
            # is publishable.
            result.failed[year] = f"{type(exc).__name__}: {exc}"[:400]
            LOG.error("[spei3] %s failed: %s", year, result.failed[year])
            if target.exists():
                target.unlink()
            continue
        if not target.exists():
            result.failed[year] = "the client reported success and wrote no file"
            LOG.error("[spei3] %s: %s", year, result.failed[year])
            continue
        size = target.stat().st_size
        result.bytes_downloaded += size
        result.written.append(year)
        LOG.info("[spei3] %s -> %s (%.1f MB)", year, target.name, size / 1e6)

    if result.years_owed:
        LOG.warning(
            "[spei3] the %.0f-second deadline expired with %d year(s) still owed: "
            "%s — this run reports `incomplete` and the next one resumes on them",
            deadline_sec or 0, len(result.years_owed), ",".join(result.years_owed),
        )
    LOG.info(
        "[spei3] downloaded %.1f MB across %d year(s) (%d already present)",
        result.bytes_downloaded / 1e6, len(result.written), len(result.skipped_present),
    )
    return result


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



#: Countries with an occurrence base rate, per hazard. The measure of this
#: feed is not a green workflow; it is drought's coverage rising toward
#: flood's and cyclone's. DR stood at 80 countries against 252 for flood
#: and 237 for cyclone, because a cell nothing looked at is outside the
#: denominator and most drought cells were exactly that.
BASE_RATE_SQL = """
SELECT hazard, COUNT(DISTINCT iso3) AS countries, COUNT(*) AS rows
FROM haz_base_rates_occurrence
GROUP BY 1
ORDER BY 1
"""

#: What share of assessed drought cells produced a row. The acceptance
#: report's own denominator: cells assessed, never rows written.
RESOLUTION_RATE_SQL = """
SELECT
  t.hazard,
  COUNT(*) AS assessed,
  SUM(CASE WHEN r.iso3 IS NULL THEN 0 ELSE 1 END) AS resolved
FROM haz_triggers t
LEFT JOIN haz_resolutions r
  ON r.iso3 = t.iso3 AND r.hazard = t.hazard
 AND r.year = t.year AND r.month = t.month
GROUP BY 1
ORDER BY 1
"""


def base_rate_coverage(con: Any) -> list[dict[str, Any]]:
    """Countries covered by an occurrence base rate, per hazard. Read-only."""

    try:
        rows = con.execute(BASE_RATE_SQL).fetchall()
    except Exception as exc:  # noqa: BLE001 - a DB without the table
        LOG.warning("[spei3] base-rate coverage unavailable: %s", exc)
        return []
    return [
        {"hazard": r[0], "countries": int(r[1]), "rows": int(r[2])} for r in rows
    ]


def resolution_rates(con: Any) -> list[dict[str, Any]]:
    """Assessed cells and resolved cells per hazard. Read-only."""

    try:
        rows = con.execute(RESOLUTION_RATE_SQL).fetchall()
    except Exception as exc:  # noqa: BLE001 - a DB without the tables
        LOG.warning("[spei3] resolution rates unavailable: %s", exc)
        return []
    out: list[dict[str, Any]] = []
    for hazard, assessed, resolved in rows:
        assessed, resolved = int(assessed or 0), int(resolved or 0)
        out.append({
            "hazard": hazard,
            "assessed": assessed,
            "resolved": resolved,
            # A hazard with nothing assessed is "not assessed", never 0%.
            "rate": (resolved / assessed) if assessed else None,
        })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_end(end: str | None) -> str:
    return end or previous_complete_month()


def _cmd_plan(args: argparse.Namespace) -> int:
    existing = read_csv_rows(args.out)
    window = plan_window(
        months_in(existing),
        start_ym=args.start,
        end_ym=_resolve_end(args.end),
        revision_months=args.revision_months,
        full_rebuild=bool(args.full_rebuild),
    )
    LOG.info(
        "[spei3] the committed feed holds %d month(s); this run owes %d "
        "(%d revision, %d missing) across year(s) %s",
        len(months_in(existing)), len(window.months),
        len(window.revision_months), len(window.missing_months),
        ",".join(window.years) or "none",
    )
    if window.missing_months:
        # Named, not counted: a month missing for the third cycle running is
        # a different problem from one that appeared this morning.
        LOG.info(
            "[spei3] months the committed feed does not hold: %s",
            ",".join(window.missing_months[:36]),
        )
    if args.window_out:
        write_json(window.as_dict(), args.window_out)
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    if args.window:
        window = Window.from_dict(json.loads(Path(args.window).read_text("utf-8")))
        months = window.months
    else:
        months = month_range(args.start, _resolve_end(args.end))
    result = fetch_grids(
        args.raw_dir, months, deadline_sec=args.deadline_sec,
    )
    if args.result_out:
        write_json(result.as_dict(), args.result_out)
    # Exit 0 even when years are owed: a queued CDS job is not a fault, and
    # a red run here would be indistinguishable from a broken request.
    return 0


def _cmd_reduce(args: argparse.Namespace) -> int:
    grids = load_grids_from_dir(args.raw_dir)
    if args.window:
        wanted = set(
            Window.from_dict(
                json.loads(Path(args.window).read_text("utf-8"))
            ).months
        )
    else:
        wanted = set(month_range(args.start, _resolve_end(args.end)))
    grids = {ym: g for ym, g in grids.items() if ym in wanted}
    if not grids:
        # Not an error on its own: a run whose CDS jobs were all still
        # queued has nothing to reduce and nothing to publish, and the
        # committed feed is untouched either way.
        LOG.warning(
            "[spei3] no month of the window is in %s — nothing to merge; the "
            "committed feed is unchanged", args.raw_dir,
        )

    iso3s = load_target_iso3s()
    countries = load_boundaries(iso3s)
    incoming, report = reduce_grids(grids, countries, iso3s=iso3s)
    report.months_missing = sorted(wanted - set(grids))

    existing = read_csv_rows(args.out)
    merged = merge_rows(existing, incoming)
    candidate = args.candidate_out or args.out
    write_csv(merged, candidate)

    LOG.info(
        "[spei3] reduced %d month(s) into %d row(s); merged into %d row(s) "
        "across %d month(s) -> %s",
        len(report.months), report.rows, len(merged),
        len(months_in(merged)), candidate,
    )
    LOG.info("[spei3] coverage: %s", report.by_coverage)
    if report.countries_never_valued:
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
        payload = report.as_dict()
        payload["merged_rows"] = len(merged)
        payload["merged_months"] = sorted(months_in(merged))
        payload["changed_months"] = months_gaining_coverage(existing, merged)
        write_json(payload, args.report_out)
    return 0


def _absent_reasons(
    window: Window | None,
    reduce_report: Mapping[str, Any] | None,
    fetch_result: Mapping[str, Any] | None,
) -> dict[str, str]:
    """Why a requested month is absent, in the words of the stage that knows.

    A month with no reason fails the window gate, and it should: "the CDS
    job was still queued" and "the grid does not carry that month" want
    different responses, and only the first is allowed to pass quietly.
    """

    reasons: dict[str, str] = {}
    owed_years = {str(y) for y in (fetch_result or {}).get("years_owed") or []}
    failed_years = {str(y) for y in ((fetch_result or {}).get("failed") or {})}
    for ym in (window.months if window else []):
        year = ym.split("-")[0]
        if year in owed_years:
            reasons[ym] = "cds job outstanding when the deadline expired"
        elif year in failed_years:
            reasons[ym] = (
                "the cds request for this year failed: "
                + str(((fetch_result or {}).get("failed") or {}).get(year, ""))[:160]
            )
    for ym in (reduce_report or {}).get("months_missing") or []:
        reasons.setdefault(str(ym), "not present in the downloaded grid")
    return reasons


def _cmd_validate(args: argparse.Namespace) -> int:
    candidate = read_csv_rows(args.candidate)
    previous = read_csv_rows(args.against)
    window = (
        Window.from_dict(json.loads(Path(args.window).read_text("utf-8")))
        if args.window else None
    )
    reduce_report = (
        json.loads(Path(args.reduce_report).read_text("utf-8"))
        if args.reduce_report and Path(args.reduce_report).exists() else None
    )
    fetch_result = (
        json.loads(Path(args.fetch_result).read_text("utf-8"))
        if args.fetch_result and Path(args.fetch_result).exists() else None
    )
    result = validate_candidate(
        candidate, previous, window=window,
        absent_reasons=_absent_reasons(window, reduce_report, fetch_result),
    )
    for warning in result.warnings:
        LOG.warning("[spei3] %s", warning)
    for failure in result.failures:
        LOG.error("[spei3] GATE FAILED: %s", failure)
    if args.report_out:
        write_json(result.as_dict(), args.report_out)
    if result.ok:
        LOG.info(
            "[spei3] every gate passed: %d row(s), %d month(s), %d changed",
            len(candidate), len(result.months_present), len(result.changed_months),
        )
        return 0
    LOG.error(
        "[spei3] %d gate(s) failed — the committed feed is left exactly as it "
        "was. A wrong feed poisons every month it covers; a stale one only "
        "costs the months it does not.", len(result.failures),
    )
    return 1


def _cmd_promote(args: argparse.Namespace) -> int:
    """Move the validated candidate into place and write the status file.

    Split from ``validate`` so the workflow can write a status file on a
    FAILED run too. A feed that quietly stops extending is what an
    unattended system is worst at noticing, so the failure is recorded even
    though the CSV is not touched.
    """

    previous_status = read_status(args.status_out)
    if args.failed:
        rows = read_csv_rows(args.out)
        status = build_status(
            rows,
            status=STATUS_FAILED,
            months_owed=_owed_from(args),
            run_id=args.run_id,
            failure={"reason": args.failure_reason or "a validation gate failed"},
            previous=previous_status,
        )
        write_json(status, args.status_out)
        LOG.warning(
            "[spei3] recorded a failed run in %s; the committed feed still "
            "covers %s..%s", args.status_out,
            status["oldest_month"], status["newest_month"],
        )
        return 0

    candidate = read_csv_rows(args.candidate)
    previous = read_csv_rows(args.out)
    if not candidate:
        LOG.error("[spei3] the candidate is empty — refusing to promote it")
        return 1
    changed = months_gaining_coverage(previous, candidate)
    write_csv(candidate, args.out)
    owed = _owed_from(args)
    status = build_status(
        candidate,
        status=STATUS_INCOMPLETE if owed else STATUS_OK,
        changed_months=changed,
        months_owed=owed,
        run_id=args.run_id,
        previous=previous_status,
    )
    write_json(status, args.status_out)
    LOG.info(
        "[spei3] promoted %d row(s) covering %s..%s; %d month(s) gained "
        "coverage%s",
        len(candidate), status["oldest_month"], status["newest_month"],
        len(changed),
        f"; {len(owed)} month(s) still owed" if owed else "",
    )
    if changed:
        LOG.info(
            "[spei3] restale token %s requests a re-walk of DR month(s) %s",
            status["restale"]["token"], ",".join(changed[:36]),
        )
    return 0


def _owed_from(args: argparse.Namespace) -> list[str]:
    """Months this run could not cover, from the fetch result and the window."""

    if not (args.window and args.fetch_result):
        return []
    try:
        window = Window.from_dict(json.loads(Path(args.window).read_text("utf-8")))
        fetch_result = json.loads(Path(args.fetch_result).read_text("utf-8"))
    except (OSError, ValueError):
        return []
    owed_years = {str(y) for y in fetch_result.get("years_owed") or []}
    owed_years |= {str(y) for y in (fetch_result.get("failed") or {})}
    return [ym for ym in window.months if ym.split("-")[0] in owed_years]


def _cmd_coverage(args: argparse.Namespace) -> int:
    import duckdb

    con = duckdb.connect(args.db, read_only=True)
    try:
        counts = coverage_counts(con)
        base_rates = base_rate_coverage(con)
        rates = resolution_rates(con)
    finally:
        con.close()
    total = sum(row["cells"] for row in counts)
    for row in counts:
        LOG.info(
            "[spei3] %-34s %-9s %d",
            row["reason_code"], row["run_type"], row["cells"],
        )
    LOG.info("[spei3] %d DR cell(s) undecided for want of an indicator", total)
    for row in base_rates:
        LOG.info(
            "[spei3] %-4s occurrence base rates: %d country/countries, %d row(s)",
            row["hazard"], row["countries"], row["rows"],
        )
    for row in rates:
        rate = "not assessed" if row["rate"] is None else f"{row['rate']:.1%}"
        LOG.info(
            "[spei3] %-4s resolution rate: %s (%d of %d assessed cell(s))",
            row["hazard"], rate, row["resolved"], row["assessed"],
        )
    if args.report_out:
        write_json(
            {
                "total": total,
                "counts": counts,
                "base_rate_coverage": base_rates,
                "resolution_rates": rates,
            },
            args.report_out,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser(
        "plan", help="which months this run owes (never a hardcoded range)"
    )
    plan.add_argument("--out", type=Path, default=DEFAULT_OUT)
    plan.add_argument("--from", dest="start", default=DEFAULT_START_YM)
    plan.add_argument("--to", dest="end", default=None)
    plan.add_argument("--revision-months", type=int, default=REVISION_WINDOW_MONTHS)
    plan.add_argument("--full-rebuild", action="store_true")
    plan.add_argument("--window-out", type=Path, default=None)

    fetch = sub.add_parser("fetch", help="download the SPEI-3 grid from the CDS")
    fetch.add_argument("--from", dest="start", default=DEFAULT_START_YM)
    fetch.add_argument("--to", dest="end", default=None)
    fetch.add_argument("--raw-dir", type=Path, required=True)
    fetch.add_argument("--window", default="", help="a window.json from `plan`")
    fetch.add_argument(
        "--deadline-sec", type=float, default=None,
        help="stop STARTING CDS jobs after this many seconds; the rest are "
             "reported owed and the next run resumes on them",
    )
    fetch.add_argument("--result-out", type=Path, default=None)

    reduce_p = sub.add_parser("reduce", help="grid -> country means, MERGED into the feed")
    reduce_p.add_argument("--raw-dir", type=Path, required=True)
    reduce_p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    reduce_p.add_argument(
        "--candidate-out", type=Path, default=None,
        help="write the merge here instead of over --out, so it can be gated first",
    )
    reduce_p.add_argument("--from", dest="start", default=DEFAULT_START_YM)
    reduce_p.add_argument("--to", dest="end", default=None)
    reduce_p.add_argument("--window", default="")
    reduce_p.add_argument("--report-out", type=Path, default=None)

    validate = sub.add_parser(
        "validate", help="gate a candidate before it becomes the committed feed"
    )
    validate.add_argument("--candidate", type=Path, required=True)
    validate.add_argument("--against", type=Path, default=DEFAULT_OUT)
    validate.add_argument("--window", default="")
    validate.add_argument("--reduce-report", default="")
    validate.add_argument("--fetch-result", default="")
    validate.add_argument("--report-out", type=Path, default=None)

    promote = sub.add_parser(
        "promote", help="move a validated candidate into place; write the status file"
    )
    promote.add_argument("--candidate", type=Path, default=None)
    promote.add_argument("--out", type=Path, default=DEFAULT_OUT)
    promote.add_argument("--status-out", type=Path, default=DEFAULT_STATUS_OUT)
    promote.add_argument("--window", default="")
    promote.add_argument("--fetch-result", default="")
    promote.add_argument("--run-id", default="")
    promote.add_argument(
        "--failed", action="store_true",
        help="record a failed run in the status file and leave the feed alone",
    )
    promote.add_argument("--failure-reason", default="")

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
    handlers = {
        "plan": _cmd_plan,
        "fetch": _cmd_fetch,
        "reduce": _cmd_reduce,
        "validate": _cmd_validate,
        "promote": _cmd_promote,
        "coverage": _cmd_coverage,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
