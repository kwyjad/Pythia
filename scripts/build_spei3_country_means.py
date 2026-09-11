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
running it by hand sound like the only option. **Measured on the first live
run (34456535827): 95 MB zipped per year, eleven years in nine and a half
minutes, 1.05 GB in total.** That is a scheduled job rather than an
afternoon, and it is what the estimate here was replaced by — the earlier
figure of "roughly 50 MB a year" was arithmetic over a 0.25-degree grid and
came out half the real volume, so the run reports the bytes it downloads
and this paragraph is corrected from that rather than from a calculation.

The reduction is the other half of the feasibility question. **Measured on
run 34461670816: 125 months reduced into 29,625 rows in 56 minutes**, or
about 27 seconds a month over 237 countries, holding roughly 2.1 GB of grids
at peak. That fits the workflow's 350-minute budget and a 16 GB runner with
room, which is why the reduce reads every owed month in one pass instead of
streaming them; a wider window should re-measure rather than assume it scales
for free. (A synthetic-grid estimate beforehand said 17 seconds and was
optimistic by a third, because it scattered its missing cells at random while
a real country's bounding box is nearly all land — so nearly every cell
reaches the point-in-polygon test rather than one in three.)

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
import re
import shutil
import sys
import tempfile
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
#: ``zip``, because that is the only container this dataset serves. The
#: first live run (34456535827) asked for ``unarchived`` and the CDS
#: answered ``Download format not supported for this dataset. Defaulting
#: to zip.`` — then handed back an archive under the ``.nc`` name the
#: client had been told to write, which is exactly the failure the
#: ``unarchived`` request was chosen to avoid. Asking for what the dataset
#: actually serves keeps that warning out of every run's log, and
#: :func:`unpack_if_archive` sniffs the bytes regardless, because a file
#: extension is a claim about a file and not evidence about it.
#:
#: ``consolidated_dataset``, and this is what sets the feed's leading edge.
#: Copernicus publishes ERA5-Drought in two releases: the consolidated one,
#: built on final ERA5 and updated **2-3 months behind real time**, and an
#: intermediate one built on ERA5T and updated with **one month of delay**,
#: which the documentation describes as experimental and subject to change
#: before the official release. Measured on 2026-09-10 the consolidated
#: product served through 2026-05 against a previous complete month of
#: 2026-08 — exactly 3 behind, the documented worst case — and the fetch
#: correctly recorded 2026-06/07/08 as absent WITH a reason rather than as a
#: fault. So the feed stopping at 2026-05 is the product, not a defect here,
#: and ``resolver/diagnostics/feed_status.PRODUCT_LAG_MONTHS`` carries that
#: number so the staleness threshold is expressed against it.
#:
#: **A naming trap, and it is worth stating because the two axes are named
#: the opposite way round in the prose and in the API.** C3S presentation
#: material calls the LAG axis "product type" and the REALISATION axis
#: "dataset type". The API uses both terms in the other sense, and the API is
#: what is written here: ``product_type: reanalysis`` selects the realisation
#: (as against ``ensemble_members``), and ``dataset_type`` selects the lag
#: (consolidated as against intermediate). Write the request from the API's
#: vocabulary; the documentation's prose is for reading, not for copying.
#:
#: Also ignore any figure of five days for the lag. A C3S slide gives that
#: number and it disagrees with the dataset page, which states the lags
#: directly in both its overview text and its Data description table. The
#: dataset page is the source to trust, read on 2026-09-11: consolidated
#: 2-3 months, intermediate one month, and the dataset's own update date that
#: day was 2026-09-10, so it is actively maintained.
#:
#: Switching to the intermediate release would buy roughly two months of
#: leading edge — a feed reaching about 2026-08 rather than 2026-05, which
#: covers 2026-06, the newest month the backcast needs and the one the
#: consolidated product cannot supply. The merge-on-``(iso3, ym)`` write plus
#: the trailing revision window handle the upgrade with no migration: pull the
#: intermediate value now, and when the consolidated version of that month
#: appears the trailing window re-requests it and the merge replaces the row
#: with the better one. That behaviour is already built and tested, and this
#: is the case it was built for.
#:
#: **The literal is ``intermediate_dataset``, and it is now evidence rather
#: than a guess.** Run 34591286383 asked the CDS and two independent routes
#: agreed: the ``/constraints`` endpoint and the process description both name
#: exactly ``consolidated_dataset`` and ``intermediate_dataset``. The switch
#: still has to be made deliberately — see the two things below it must not
#: disturb — but there is no longer an unknown in it.
#:
#: It was NOT settled by reading around the problem, and the record of how it
#: was settled is worth keeping. Neither this sandbox nor a browser could do
#: it: the egress proxy denies cds.climate.copernicus.eu, and the Download
#: tab's "Show API request code" button only emits whatever the form currently
#: has selected. So the workflow asks, and there are two ways of asking.
#:
#: ``probe`` mode sends one deliberately invalid ``dataset_type`` and reads
#: the refusal. **It was run on 2026-09-11 (run 34587745561) and came back
#: empty**: the CDS answered "Request has not produced a valid combination of
#: values, please check your selection." with an echo of the request and no
#: list of accepted values. That settles something narrow — every mandatory
#: key was accepted, and the same body with ``consolidated_dataset`` fetched
#: eleven years in run 34472338247, so the refusal is about that one value —
#: but this dataset's validator complains about the whole COMBINATION rather
#: than about one key, so a wrong enum names no enum here.
#:
#: ``describe`` mode asks the service for its own schema instead of inferring
#: it from a complaint: the ``/constraints`` endpoint the web form itself
#: calls, and the process description's input schemas. See
#: :func:`describe_enum_values`. Take the literal from that output and only
#: then make the switch.
#:
#: Two things the switch must not disturb, recorded here while the reasoning
#: is fresh. ``REVISION_WINDOW_MONTHS`` has to be long enough to re-request a
#: month AFTER its consolidated version appears, which is roughly three months
#: later than the intermediate one — otherwise intermediate values become
#: permanent. And ``feed_status.PRODUCT_LAG_MONTHS`` describes whichever
#: product is requested, so it changes with this literal; keep it stated apart
#: from the missed-cycle tolerance, which is the mistake the version before it
#: made by reasoning from a lag the producer does not ask for.
#:
#: **The rest of this shape came from the ECMWF forum thread announcing
#: the release, and the first live run confirmed it**: eleven years were
#: accepted and served, 45 to 90 seconds each. If a future run is refused
#: the error body names the offending key — which is why
#: :func:`fetch_grids` logs the request it is about to send.
CDS_VARIABLE = "standardised_precipitation_evapotranspiration_index"
CDS_ACCUMULATION_PERIOD = "3"
CDS_PRODUCT_TYPE = "reanalysis"
CDS_DATASET_TYPE = "consolidated_dataset"
CDS_VERSION = "1_0"
CDS_DATA_FORMAT = "netcdf"
CDS_DOWNLOAD_FORMAT = "zip"

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
    #: Cells of this country's own territory the index saturated on, which
    #: are dropped like a NaN. Carried on the result rather than logged in
    #: place so the caller can report the total and name the country-months.
    n_saturated: int = 0


#: Above this magnitude the index has saturated rather than measured.
#:
#: SPEI is a fitted probability run through the inverse normal, and this
#: product is `SPEI3_genlogistic_...`: once the fitted probability reaches
#: the floating-point neighbourhood of 0 or 1 the inverse normal returns a
#: value near +/-8.2 whatever the water balance actually was. The first
#: full rebuild (run 34461670816) found exactly that shape — twelve values
#: at two magnitudes, `8.2095` and `-8.2221`, repeated to four decimals
#: across different countries and years, which no area-weighted mean of
#: real data produces.
#:
#: It happens where the fit degenerates rather than at random: Bahrain, a
#: country with so little rainfall that the distribution has almost no
#: variance to fit, and Palau, an island small enough to be sampled off one
#: nearest cell. A real SPEI extreme lives inside +/-4; -8.2 for Palau in
#: March is not a statement about Palau.
#:
#: So a saturated cell is treated as MISSING exactly as a NaN is, which is
#: also what keeps one such cell out of the area-weighted mean of a country
#: whose other cells are fine. The country-month then produces no row, and
#: the rulebook entry's `absence_means_no_drought: false` reads that as
#: unknown — which is the truth. :data:`SANITY_HARD_LIMIT` stays where it
#: is as the fill-value and changed-unit catcher.
SATURATION_ABS = 8.0


def _is_saturated(value: float) -> bool:
    """Has the index saturated rather than measured? See :data:`SATURATION_ABS`."""

    return abs(value) >= SATURATION_ABS


def _is_saturated_safely(value: Any) -> bool:
    """`_is_saturated` for a value that may not be a number at all."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return not math.isnan(number) and _is_saturated(number)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        number = float(value)
    except (TypeError, ValueError):
        return True
    if math.isnan(number):
        return True
    # Saturated is not measured. Counted by the caller rather than silently
    # folded in with the NaNs, because a rise in saturation means the fit is
    # degenerating somewhere new and that is worth seeing.
    return _is_saturated(number)


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
    n_saturated = 0
    for i, lat in enumerate(grid.lats):
        if lat < miny or lat > maxy:
            continue
        row = grid.values[i]
        for j, lon in enumerate(grid.lons):
            if lon < minx or lon > maxx:
                continue
            value = row[j]
            if _is_missing(value):
                # Counted only where the cell is actually this country's,
                # so the number means "cells this country lost" rather than
                # "cells in its bounding box".
                if value is not None and _is_saturated_safely(value):
                    if contains(country, lon, lat):
                        n_saturated += 1
                continue
            if not contains(country, lon, lat):
                continue
            weight = _cos_weight(lat)
            total += float(value) * weight
            weight_sum += weight
            n_cells += 1

    if weight_sum > 0:
        return CountryValue(
            country.iso3, total / weight_sum, COVERAGE_CELLS, n_cells, n_saturated
        )
    if not nearest_when_uncovered:
        return CountryValue(country.iso3, None, COVERAGE_NONE, 0, n_saturated)

    # No cell centre fell inside the territory. The country is smaller
    # than a cell, not missing from the world. `_nearest_cell_value` skips a
    # saturated cell for the same reason the loop above does, so a country
    # whose only cell has saturated comes back with no value rather than
    # with the saturation figure.
    sampled = _nearest_cell_value(grid, country)
    if sampled is None:
        return CountryValue(country.iso3, None, COVERAGE_NONE, 0, n_saturated)
    return CountryValue(country.iso3, sampled, COVERAGE_NEAREST, 0, n_saturated)


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
    #: Cells the index saturated on rather than measured (:data:`SATURATION_ABS`),
    #: and the country-months they cost a value. Reported rather than merely
    #: dropped: a rise means the fit is degenerating somewhere new.
    saturated_cells: int = 0
    saturated_country_months: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "months": self.months,
            "rows": self.rows,
            "by_coverage": self.by_coverage,
            "countries_never_valued": self.countries_never_valued,
            "countries_without_boundary": self.countries_without_boundary,
            "months_missing": self.months_missing,
            "saturated_cells": self.saturated_cells,
            "saturated_country_months": self.saturated_country_months,
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
            if result.n_saturated:
                report.saturated_cells += result.n_saturated
                report.saturated_country_months.append(
                    f"{iso3}/{ym}={result.n_saturated}"
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


#: What the first bytes of a file say it is. A ``.nc`` extension is a claim
#: about a file; these are evidence about it.
_MAGIC = {
    b"PK\x03\x04": "zip",
    b"CDF\x01": "netcdf3",
    b"CDF\x02": "netcdf3",
    b"\x89HDF": "netcdf4",
}


def sniff_container(path: Path) -> str:
    """``zip`` / ``netcdf3`` / ``netcdf4`` / ``unknown:<hex>`` from the bytes.

    The CDS writes to whatever filename the client hands it, so the
    extension records what we ASKED for and not what arrived. On the first
    live run every file was named ``spei3_YYYY.nc`` and every one of them
    was a zip, and the only symptom was xarray reporting no matching IO
    backend — a message about our installation, for a file that was never
    a NetCDF.
    """

    with path.open("rb") as handle:
        head = handle.read(8)
    for magic, name in _MAGIC.items():
        if head.startswith(magic):
            return name
    return "unknown:" + head.hex()


def unpack_if_archive(path: Path) -> list[Path]:
    """The NetCDF files behind ``path``, unpacking a zip in place.

    This dataset serves a zip whatever ``download_format`` asks for, so
    unpacking is the ordinary case rather than a fallback. Called from the
    fetch (so a downloaded year is normalised at once) AND from the read
    (so a raw directory left behind by an older run still reduces), because
    a cached grid is expensive and a re-download for a container mismatch
    buys nothing.

    Members keep the year in their name, since the year is what the fetch's
    resume check and the cache prune both read. The archive is removed once
    its members are on disk: two copies of a 95 MB year is a cache entry
    twice the size for no gain.
    """

    import zipfile

    kind = sniff_container(path)
    if kind != "zip":
        return [path]

    stem = path.stem
    written: list[Path] = []
    with zipfile.ZipFile(path) as archive:
        members = [m for m in archive.namelist() if not m.endswith("/")]
        wanted = [m for m in members if m.lower().endswith(".nc")]
        ignored = sorted(set(members) - set(wanted))
        if ignored:
            # Named rather than counted: a member we cannot read is either a
            # second product we do not want or the whole payload in a format
            # we did not ask for, and only its name says which.
            LOG.warning(
                "[spei3] %s carries %d member(s) this reads and %d it does not: %s",
                path.name, len(wanted), len(ignored), ", ".join(ignored[:6]),
            )
        if not wanted:
            raise ValueError(
                f"{path.name} is a zip carrying no .nc member "
                f"(members: {', '.join(members[:6]) or 'none'})"
            )
        for member in sorted(wanted):
            # `stem` already carries the year, so a member named `data.nc`
            # in two years' archives cannot collide.
            target = path.parent / f"{stem}__{Path(member).name}"
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            written.append(target)
            LOG.info(
                "[spei3] unpacked %s -> %s (%.1f MB)",
                path.name, target.name, target.stat().st_size / 1e6,
            )
    path.unlink()
    return written


def year_files(raw_dir: Path, year: str) -> list[Path]:
    """Everything on disk for one year, archived or unpacked.

    The fetch's resume check reads this rather than one filename, because a
    year that has been unpacked no longer has the name it was downloaded
    under and re-downloading it costs 95 MB to learn nothing.
    """

    return sorted(p for p in raw_dir.glob(f"spei3_{year}*") if p.is_file())


def load_grids_from_dir(raw_dir: Path) -> dict[str, Grid]:
    """Every ``*.nc`` under ``raw_dir`` as ``{ym: Grid}``.

    Needs xarray + netCDF4, which is why it is quarantined here: nothing
    on the resolution path imports this module, so the pipeline never has
    to carry the dependency.
    """

    import numpy as np
    import xarray as xr

    # Unpack before globbing: a cache restored from a run that predates
    # `unpack_if_archive` holds zips under `.nc` names, and re-downloading a
    # gigabyte to correct a container mismatch buys nothing.
    for candidate in sorted(raw_dir.glob("*.nc")):
        unpack_if_archive(candidate)

    grids: dict[str, Grid] = {}
    for path in sorted(raw_dir.glob("*.nc")):
        kind = sniff_container(path)
        if not kind.startswith("netcdf"):
            # Named, not swallowed: xarray's own complaint is about the IO
            # backends installed here, which sends the reader to the
            # requirements file for a problem in the download.
            raise ValueError(
                f"{path.name} is not a NetCDF file (first bytes say {kind!r}); "
                "the CDS writes to whatever filename it is given, so check "
                "`download_format` and the fetch log"
            )
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
#:
#: Under the CONSOLIDATED product this producer currently requests, the values
#: are built on final ERA5 and are not revised afterwards, so this window is
#: belt-and-braces: `plan` already re-asks for any month the CSV is missing,
#: and four months simply means a republished month is picked up too.
#:
#: It becomes LOAD-BEARING the moment the request moves to the intermediate
#: release. Those values are built on ERA5T, which is documented as
#: experimental and subject to change, and they are superseded by the
#: consolidated version of the same month roughly three months later — so the
#: window has to be long enough to re-request a month AFTER its consolidated
#: version appears, or the intermediate value becomes permanent. Four is not
#: enough of a margin for that on its own: the gap between the two releases is
#: about two months (one behind against three behind), so a window of four
#: leaves two months of slack for a missed cycle, and a missed cycle costs the
#: upgrade rather than the coverage. Widen it with the switch, and say what the
#: new number is reasoned from. See CDS_DATASET_TYPE for the whole argument.
#:
#: The comment this replaces reasoned from "ERA5T runs about five days
#: behind", which is a figure from a C3S slide that disagrees with the dataset
#: page AND describes a release this producer does not request.
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


# ---------------------------------------------------------------------------
# probe: ask the CDS what an enum accepts, instead of guessing
# ---------------------------------------------------------------------------
#
# The intermediate release's ``dataset_type`` literal is unknown.
# ``intermediate_dataset`` is likely by symmetry with ``consolidated_dataset``,
# and a guessed key is refused with the same message a misspelt variable name
# gets — so shipping it on a guess would turn a two-month gain into an outage
# that reads like one. Neither the sandbox this code was written in nor a
# browser is a good way to settle it: the proxy denies the CDS entirely, and
# the Download tab's "Show API request code" button only emits whatever the
# form currently has selected, which is a fiddly way to read one enum.
#
# The workflow can reach the CDS, so it asks. Probe mode submits ONE request
# with a deliberately invalid ``dataset_type``, catches the refusal, and prints
# the body, on the expectation that the CDS names the accepted values when it
# rejects an out-of-range key.
#
# **On this dataset it does not.** Run 34587745561 asked and got "Request has
# not produced a valid combination of values, please check your selection."
# with an echo of the request and no list at all: the validator here complains
# about the whole combination rather than about one key, so a deliberately
# wrong enum names no enum. The route that does answer is
# :func:`describe_enum_values`, which asks the service for its own schema.
#
# This one is kept regardless, and not out of sentiment. It is the only route
# that proves the request SHAPE is right — every mandatory key accepted, the
# refusal isolated to the one value changed — which is worth knowing on its
# own and is what rules out a missing key when a switch goes wrong. And other
# CDS datasets do name their values, so the trick still answers the next enum
# question elsewhere; enum questions recur, as this dataset showed when it
# moved ``format`` to ``data_format`` and pushed the accumulation period out
# of the variable name into a key of its own.

#: The value probe mode sends. Deliberately not a plausible guess — a probe
#: that happened to hit a REAL value would be a silent download rather than the
#: refusal it is here to read, and on this dataset a successful retrieve is
#: tens of megabytes.
PROBE_INVALID_VALUE = "pythia_probe_not_a_real_dataset_type"

#: Accepted-value lists are quoted in a few shapes across the CDS's error
#: bodies, so the reader tries several rather than pinning one. Finding none is
#: a REPORTED outcome: "the refusal does not list valid values" is the answer
#: that stops the switch, and dressing it up as a parse failure would invite
#: somebody to guess anyway.
_PROBE_VALUE_PATTERNS: tuple[str, ...] = (
    r"(?:allowed|accepted|permitted|valid|expected)\s+values?[^:\[]*[:\[]\s*(.+)",
    r"not\s+(?:a\s+)?valid[^.]*?\bone\s+of\s*[:\[]?\s*(.+)",
    r"must\s+be\s+one\s+of\s*[:\[]?\s*(.+)",
)

#: Anything shaped like a CDS enum literal: lowercase words joined by
#: underscores. Used only to pull the candidates out of whatever fragment the
#: patterns above isolated, so a changed sentence costs the sentence and not
#: the reading.
_PROBE_LITERAL = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+")


@dataclass
class ProbeResult:
    """What one deliberately-invalid request taught us about an enum."""

    key: str = "dataset_type"
    sent: dict[str, Any] = field(default_factory=dict)
    refused: bool = False
    error_type: str = ""
    body: str = ""
    values: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def settled(self) -> bool:
        """Did the refusal actually name the accepted values?

        The whole point of the probe. Unsettled means the switch does not
        proceed — see the note in ``probe_enum_values``.
        """

        return self.refused and bool(self.values)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "sent": self.sent,
            "refused": self.refused,
            "settled": self.settled,
            "error_type": self.error_type,
            "values": list(self.values),
            "body": self.body,
            "detail": self.detail,
        }


def parse_probe_refusal(body: str) -> list[str]:
    """The accepted values a CDS refusal names, in the order it names them.

    Pure, so the reading is tested against recorded error bodies rather than
    against the live service. An empty list is a legitimate answer and means
    the body named nothing — which is a finding, not a failure.
    """

    text = str(body or "")
    if not text.strip():
        return []
    for pattern in _PROBE_VALUE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        found = [
            value for value in _PROBE_LITERAL.findall(match.group(1))
            if value != PROBE_INVALID_VALUE
        ]
        if found:
            # Order preserved, duplicates dropped: the CDS lists these in its
            # own order and that order is worth keeping.
            seen: dict[str, None] = {}
            for value in found:
                seen.setdefault(value, None)
            return list(seen)
    return []


def probe_enum_values(
    key: str = "dataset_type",
    *,
    client: Any = None,
    invalid: str = PROBE_INVALID_VALUE,
) -> ProbeResult:
    """Send one invalid request and read the refusal.

    Fetches nothing, writes nothing, commits nothing: the request is built to
    be REFUSED, so there is no download to make. A request the CDS accepts is
    reported as unsettled rather than treated as success — an accepted probe
    means the invalid value was not invalid, and nothing has been learnt.
    """

    request = cds_request("2016", ["01"])
    request[key] = invalid
    result = ProbeResult(key=key, sent=dict(request))

    if client is None:  # pragma: no cover - needs CDS credentials
        import cdsapi

        client = cdsapi.Client(wait_until_complete=True, delete=False, quiet=False)

    target = Path(tempfile.gettempdir()) / "spei3_probe_should_never_be_written.nc"
    try:
        client.retrieve(CDS_DATASET, request, str(target))
    except Exception as exc:  # noqa: BLE001 - the refusal IS the result
        result.refused = True
        result.error_type = type(exc).__name__
        result.body = str(exc)
        result.values = parse_probe_refusal(result.body)
        if result.values:
            result.detail = (
                f"the CDS refused {key}={invalid!r} and named "
                f"{len(result.values)} accepted value(s)"
            )
        else:
            result.detail = (
                f"the CDS refused {key}={invalid!r} but did NOT list the accepted "
                "values, so this probe cannot settle the literal. Do not guess "
                "one: a wrong enum is refused exactly as a wrong variable name "
                "is, and shipping it would turn a two-month gain into an outage "
                "that reads like one."
            )
        return result

    # An accepted probe is not good news: it means the deliberately invalid
    # value was accepted, so either the key is ignored or the value is real.
    # Either way nothing was learnt, and a download may have happened.
    result.detail = (
        f"the CDS ACCEPTED {key}={invalid!r}, which it should not have. Either "
        f"{key} is being ignored for this dataset or the probe value collides "
        "with a real one; nothing was learnt about the accepted values."
    )
    if target.exists():
        try:
            target.unlink()
        except OSError:
            pass
    return result


# --------------------------------------------------------------------------
# Asking the CDS for the accepted values, rather than reading a refusal
# --------------------------------------------------------------------------
#
# The refusal probe above was the first way to ask and it came back empty.
# Run 34587745561 sent ``dataset_type=pythia_probe_not_a_real_dataset_type``
# in the live request shape and the CDS answered:
#
#     Request has not produced a valid combination of values, please check
#     your selection.
#
# followed by an echo of the request. It named no accepted values. That
# settles something narrow and useful — every mandatory key was accepted, and
# the same body with ``consolidated_dataset`` fetched eleven years in run
# 34472338247, so the refusal IS about that one value — but it does not settle
# what a valid value looks like. This dataset's validator complains about the
# whole COMBINATION rather than about one key, so a deliberately wrong enum
# buys nothing here.
#
# So ask the service for its own schema instead of inferring it from a
# complaint. Two routes, both read-only, taken in order:
#
#   1. ``POST {process}/constraints`` — the endpoint the web form calls on
#      every click. Given a partial request it returns the values still valid
#      for every key, which is exactly the question. A POST that queues no
#      job, downloads nothing and writes nothing: it is a read wearing a
#      verb.
#   2. ``GET {process}`` — the process description, whose input schemas carry
#      each key's full enum irrespective of the other selections.
#
# Both reached through the client this producer already builds, so there is no
# hand-rolled auth header to get wrong: ``cdsapi.Client()`` returns
# ecmwf-datastores' ``LegacyClient``, which holds a ``datastores.Client`` on
# ``.client`` carrying ``apply_constraints`` and ``get_process``. Borrowed
# under the same contract this repository uses for the GDACS connector's
# helpers: if the attribute is renamed upstream, say so plainly rather than
# failing obscurely.
#
# The constraints route is asked TWICE, and the pair is the point. Once with
# the rest of our request fixed, which answers "what may this key be in the
# request we actually send", and once with nothing fixed, which answers "what
# may this key be at all". A value present in the second list and absent from
# the first is a real and different finding: the release exists, but not in
# combination with something else we ask for.
#
# Finding nothing is still a reported outcome, for the same reason as in the
# refusal probe: "the service did not name the values" is the answer that
# stops the switch, and dressing it up as a parse failure would invite
# somebody to guess anyway.


#: Where a JSON Schema keeps its accepted values. The CDS spells the schema
#: key ``schema_`` in places (``schema`` collides with pydantic's own), so the
#: reader walks for the key by NAME rather than pinning one path — a changed
#: nesting then costs nothing.
_ENUM_CONTAINER_KEYS = ("enum", "oneOf", "anyOf")


def _enum_from_schema(node: Any) -> list[str]:
    """The accepted string values a JSON Schema fragment names, in order.

    Handles the plain ``enum`` list and the ``oneOf``/``anyOf`` form where
    each branch carries a ``const``. Anything else yields nothing, which the
    caller reports rather than papers over.
    """

    if not isinstance(node, dict):
        return []
    found: list[str] = []
    raw = node.get("enum")
    if isinstance(raw, list):
        found.extend(str(v) for v in raw if isinstance(v, (str, int)))
    for branch_key in ("oneOf", "anyOf"):
        branches = node.get(branch_key)
        if isinstance(branches, list):
            for branch in branches:
                if isinstance(branch, dict) and "const" in branch:
                    found.append(str(branch["const"]))
    # A schema may wrap the real thing (an array's ``items``, say).
    for nested in ("items", "schema", "schema_"):
        child = node.get(nested)
        if isinstance(child, dict) and not found:
            found.extend(_enum_from_schema(child))
    return _dedupe_preserving_order(found)


def _dedupe_preserving_order(values: Sequence[Any]) -> list[str]:
    """Order kept, duplicates dropped. The service's own order is worth having."""

    seen: dict[str, None] = {}
    for value in values:
        text = str(value)
        if text:
            seen.setdefault(text, None)
    return list(seen)


def parse_process_description(payload: Any, key: str) -> list[str]:
    """The accepted values a process description names for one key.

    Pure, so it is tested against recorded payload shapes rather than against
    the live service. Walks for a mapping ENTRY whose name is ``key`` and
    whose value carries a schema, because the CDS nests the schema under
    ``inputs[key]["schema_"]`` in some versions and ``["schema"]`` in others,
    and pinning one of them would answer "absent" for a value sitting right
    there — the ``PRAGMA table_info`` mistake in another costume.
    """

    hits: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for name, child in node.items():
                if name == key:
                    hits.extend(_enum_from_schema(child))
                walk(child)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(payload)
    return _dedupe_preserving_order(hits)


def parse_constraints_payload(payload: Any, key: str) -> list[str]:
    """The values a ``/constraints`` response says one key may still take.

    The documented shape is a flat mapping of key to a list of valid values.
    ``inputs`` nesting and a mapping-of-mappings are both tolerated, because
    the shape is the service's to change and the answer is worth more than the
    assumption.
    """

    if isinstance(payload, dict):
        for container in (payload, payload.get("inputs")):
            if not isinstance(container, dict):
                continue
            raw = container.get(key)
            if isinstance(raw, list):
                return _dedupe_preserving_order(raw)
            if isinstance(raw, dict):
                # Some surfaces return {value: something} rather than a list.
                return _dedupe_preserving_order(list(raw))
    return []


@dataclass
class DescribeRoute:
    """One attempt to ask the CDS, and what it said."""

    route: str = ""
    ok: bool = False
    values: list[str] = field(default_factory=list)
    error: str = ""
    excerpt: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "ok": self.ok,
            "values": list(self.values),
            "error": self.error,
            "excerpt": self.excerpt,
        }


@dataclass
class DescribeResult:
    """What the CDS's own schema says one enum accepts."""

    key: str = "dataset_type"
    routes: list[DescribeRoute] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    source_route: str = ""
    unconstrained: list[str] = field(default_factory=list)
    detail: str = ""

    @property
    def settled(self) -> bool:
        """Did any route actually name the accepted values?

        Unsettled means the switch does not proceed. Same rule as the refusal
        probe, and for the same reason.
        """

        return bool(self.values)

    @property
    def only_unconstrained(self) -> list[str]:
        """Values valid in general but NOT alongside the rest of our request.

        A finding in its own right: it means the release exists and something
        else we ask for excludes it, which is a different repair from the
        release not existing.
        """

        return [v for v in self.unconstrained if v not in self.values]

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "settled": self.settled,
            "values": list(self.values),
            "source_route": self.source_route,
            "unconstrained": list(self.unconstrained),
            "only_unconstrained": list(self.only_unconstrained),
            "routes": [route.as_dict() for route in self.routes],
            "detail": self.detail,
        }


#: Keys the ``/constraints`` endpoint refuses. They describe how a result is
#: DELIVERED rather than what is selected, so the form knows nothing about
#: them and says so with a 422 rather than ignoring them.
_CONSTRAINTS_NON_FORM_KEYS = ("data_format", "download_format")

#: How much of a payload to keep in the report. Enough for a human to read
#: the shape when the parser found nothing, bounded so a large process
#: description does not become the artifact.
_DESCRIBE_EXCERPT_CHARS = 4000


def _describe_api(client: Any) -> Any:
    """The datastores client underneath cdsapi's legacy wrapper.

    ``cdsapi.Client()`` returns ecmwf-datastores' ``LegacyClient``, which
    keeps the real client on ``.client``. Borrowed deliberately rather than
    rebuilt: hand-rolling the request means hand-rolling the ``PRIVATE-TOKEN``
    header, and a borrowed helper that disappears should say so.
    """

    api = getattr(client, "client", client)
    missing = [
        name for name in ("apply_constraints", "get_process")
        if not hasattr(api, name)
    ]
    if missing:
        raise AttributeError(
            "the CDS client exposes no "
            + " or ".join(missing)
            + f" (got {type(api).__name__}); ecmwf-datastores has moved the "
            "constraints API and this borrow needs re-pointing"
        )
    return api


def describe_enum_values(
    key: str = "dataset_type",
    *,
    client: Any = None,
) -> DescribeResult:
    """Ask the CDS what one request key accepts, and report what it said.

    Reads only. Queues no job, downloads nothing, writes nothing to the
    repository. Every route's outcome is recorded, including its failure,
    because "which way of asking failed" is most of the diagnosis when the
    answer does not arrive.
    """

    result = DescribeResult(key=key)

    if client is None:  # pragma: no cover - needs CDS credentials
        import cdsapi

        client = cdsapi.Client(wait_until_complete=True, delete=False, quiet=False)

    try:
        api = _describe_api(client)
    except AttributeError as exc:
        result.routes.append(DescribeRoute(route="client", error=str(exc)))
        result.detail = str(exc)
        return result

    # Route 1a: the constraints endpoint, with the rest of our request fixed.
    # The DELIVERY keys are dropped, not the selection ones. `/constraints`
    # answers about the form's own fields and rejects the rest outright: run
    # 34591286383 sent the whole request and got
    # "422 ... invalid parameter / invalid param 'data_format'", so that half
    # of the pair never answered and `only_unconstrained` came back empty for
    # want of an answer rather than because there was nothing to report.
    ours = {
        k: v for k, v in cds_request("2016", ["01"]).items()
        if k != key and k not in _CONSTRAINTS_NON_FORM_KEYS
    }
    for route_name, request in (
        (f"constraints(our request minus {key})", ours),
        ("constraints(nothing fixed)", {}),
    ):
        route = DescribeRoute(route=route_name)
        try:
            payload = api.apply_constraints(CDS_DATASET, request)
        except Exception as exc:  # noqa: BLE001 - a failed route is a finding
            route.error = f"{type(exc).__name__}: {exc}"
        else:
            route.ok = True
            route.values = parse_constraints_payload(payload, key)
            route.excerpt = json.dumps(payload, indent=2, sort_keys=True)[
                :_DESCRIBE_EXCERPT_CHARS
            ]
        result.routes.append(route)

    # Route 2: the process description's own input schema.
    route = DescribeRoute(route="process description")
    try:
        payload = api.get_process(CDS_DATASET).json
    except Exception as exc:  # noqa: BLE001 - a failed route is a finding
        route.error = f"{type(exc).__name__}: {exc}"
    else:
        route.ok = True
        route.values = parse_process_description(payload, key)
        route.excerpt = json.dumps(payload, indent=2, sort_keys=True)[
            :_DESCRIBE_EXCERPT_CHARS
        ]
    result.routes.append(route)

    # The answer is the first route that named anything, in the order asked:
    # the constrained list is the one that governs the request we send.
    for route in result.routes:
        if route.values and not result.values:
            result.values = list(route.values)
            result.source_route = route.route

    for route in result.routes:
        if route.route == "constraints(nothing fixed)":
            result.unconstrained = list(route.values)

    if result.values:
        result.detail = (
            f"the CDS named {len(result.values)} accepted value(s) for {key} "
            f"via {result.source_route}"
        )
        if result.only_unconstrained:
            result.detail += (
                "; and "
                + ", ".join(repr(v) for v in result.only_unconstrained)
                + " is valid in general but NOT alongside the rest of our "
                "request, so switching to it needs another key to move too"
            )
    else:
        asked = ", ".join(
            f"{route.route} ({'read' if route.ok else route.error})"
            for route in result.routes
        )
        result.detail = (
            f"no route named the accepted values for {key}. Asked: {asked}. Do "
            "not guess one: a wrong enum is refused exactly as a wrong "
            "variable name is, and shipping it would turn a two-month gain "
            "into an outage that reads like one."
        )
    return result


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
        # By year, not by that one filename: an unpacked year no longer has
        # the name it was downloaded under, and re-asking for it costs 95 MB
        # to learn nothing.
        held = year_files(raw_dir, year)
        if held:
            LOG.info(
                "[spei3] %s already downloaded — skipping (%s)",
                year, ", ".join(p.name for p in held),
            )
            result.skipped_present.append(year)
            result.bytes_downloaded += sum(p.stat().st_size for p in held)
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
        # The archive's own size is the download volume, so it is read
        # before unpacking replaces the file.
        size = target.stat().st_size
        result.bytes_downloaded += size
        result.written.append(year)
        LOG.info("[spei3] %s -> %s (%.1f MB)", year, target.name, size / 1e6)
        try:
            unpack_if_archive(target)
        except Exception as exc:  # noqa: BLE001 - the reason is the payload
            # A year that arrived and cannot be opened is owed, exactly as a
            # year that never arrived is: the difference matters to nobody
            # downstream, and leaving an unreadable file on disk would make
            # the next run's resume check skip it forever.
            result.failed[year] = f"{type(exc).__name__}: {exc}"[:400]
            LOG.error("[spei3] %s downloaded and could not be unpacked: %s",
                      year, result.failed[year])
            for stale in year_files(raw_dir, year):
                stale.unlink()
            result.written.remove(year)

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
    if report.saturated_cells:
        # INFO, not a warning: the index saturating over a hyper-arid country
        # or a one-cell island is the ordinary case for this product, and
        # warning about it every run is how a reader learns to skip the
        # warnings that matter. It is counted and named, so a RISE is still
        # visible — a country appearing here that did not before means the
        # fit has started degenerating somewhere new.
        LOG.info(
            "[spei3] the index saturated on %d cell(s) across %d country-month(s), "
            "dropped as unmeasured (|value| >= %g): %s",
            report.saturated_cells, len(report.saturated_country_months),
            SATURATION_ABS, ",".join(report.saturated_country_months[:20]),
        )
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


def _cmd_probe(args: argparse.Namespace) -> int:
    """Ask the CDS what an enum accepts. Read-only, and always exit 0.

    A probe is a question, and a question that goes red is one somebody stops
    asking. The verdict is in the report and the step summary; the exit code
    says only that the question was put.
    """

    result = probe_enum_values(args.key, invalid=args.invalid)
    LOG.info("probe %s: %s", args.key, result.detail)
    if result.values:
        LOG.info("accepted values for %s: %s", args.key, ", ".join(result.values))
    if result.body:
        LOG.info("the CDS said: %s", result.body)
    if args.report_out:
        write_json(result.as_dict(), args.report_out)
    return 0


def _cmd_describe(args: argparse.Namespace) -> int:
    """Ask the CDS's own schema what an enum accepts. Read-only, always exit 0.

    A question that goes red is a question somebody stops asking. The verdict
    is in the report and the step summary; the exit code says only that the
    question was put.
    """

    result = describe_enum_values(args.key)
    LOG.info("describe %s: %s", args.key, result.detail)
    for route in result.routes:
        LOG.info(
            "  route %-38s %s",
            route.route,
            ", ".join(route.values) if route.values
            else (route.error or "named no values"),
        )
    if args.report_out:
        write_json(result.as_dict(), args.report_out)
    return 0


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

    probe = sub.add_parser(
        "probe",
        help="ask the CDS what an enum accepts, by sending one invalid value",
    )
    probe.add_argument(
        "--key", default="dataset_type",
        help="the request key to probe; the CDS names the accepted values when "
             "it rejects an out-of-range one",
    )
    probe.add_argument(
        "--invalid", default=PROBE_INVALID_VALUE,
        help="the deliberately invalid value to send. Not a plausible guess: a "
             "probe that hits a REAL value downloads tens of megabytes instead "
             "of returning the refusal it exists to read",
    )
    probe.add_argument("--report-out", type=Path, default=None)

    describe = sub.add_parser(
        "describe",
        help="ask the CDS's own constraints and process description what an "
             "enum accepts (read-only)",
    )
    describe.add_argument(
        "--key", default="dataset_type",
        help="the request key to ask about",
    )
    describe.add_argument("--report-out", type=Path, default=None)

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
        "probe": _cmd_probe,
        "describe": _cmd_describe,
        "coverage": _cmd_coverage,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
