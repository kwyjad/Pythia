# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The SPEI-3 producer: the request it sends, the merge, and the gates.

``test_spei3_country_means.py`` covers the zonal reduction — the rule that
decides which countries get a value. This file covers the half that turns
that reduction into a scheduled producer, and the three things there that
can lose data silently:

* **the request**. The pre-October-2025 shape is not merely suboptimal, it
  is REFUSED, and a refused request looks from the outside like an outage.
  So the keys are pinned rather than only exercised in production.
* **the merge**. ``write_rows`` used to write the CSV from whatever the run
  had just reduced, so a monthly run that fetched three months would have
  replaced ten years of series with three months of rows the first time it
  ran on a schedule.
* **the gates**. A wrong feed poisons every month it covers; a stale one
  only costs the months it does not. Each gate is here with the fault it
  guards against.

**The CDS path is not tested and cannot be.** The environment this was
written in has no CDS account and cannot reach
``cds.climate.copernicus.eu``. There is no mocked "successful fetch" here
pretending otherwise: :func:`cds_request` is a pure function and its keys
are asserted, the deadline logic is exercised against an injected client,
and whether the CDS accepts the request is settled by the first real
``workflow_dispatch`` run.
"""

from __future__ import annotations

import ast
import datetime as dt
import json
import re
import sys
from pathlib import Path

import pytest

from scripts import build_spei3_country_means as spei

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spei3"

#: A fixed "today" for the tests that exercise the fetch. Which release a
#: month is asked from is a function of the calendar, so a test that lets
#: the real date decide is a test whose meaning changes in December.
#: 2026-09-12 is the day the switch was made: previous complete month
#: 2026-08, so the consolidated release reaches 2026-05 and everything after
#: it is intermediate.
TODAY = dt.date(2026, 9, 12)


def _rows(*triples):
    return [
        {"iso3": iso3, "ym": ym, "value": value, "coverage": "cells", "n_cells": "4"}
        for iso3, ym, value in triples
    ]


# ---------------------------------------------------------------------------
# The request the CDS actually accepts
# ---------------------------------------------------------------------------


def test_the_request_carries_every_key_the_cds_made_mandatory():
    """The five changes of 30 October 2025, each asserted by name.

    A missing one is not a degraded request; it is
    ``Request has not produced a valid combination of values``, which from
    the outside is indistinguishable from the dataset being down.
    """

    request = spei.cds_request("2026", ["1", "2", "12"])

    # The variable dropped its _3_month suffix and the 3 moved out.
    assert request["variable"] == [
        "standardised_precipitation_evapotranspiration_index"
    ]
    assert "_3_month" not in request["variable"][0]
    assert request["accumulation_period"] == ["3"]

    # Three keys that were absent and are mandatory.
    assert request["product_type"] == ["reanalysis"]
    assert request["dataset_type"] == "consolidated_dataset"
    assert request["version"] == "1_0"

    # `format` became `data_format`.
    assert request["data_format"] == "netcdf"
    assert "format" not in request
    # `zip` because that is the only container this dataset serves: run
    # 34456535827 asked for `unarchived` and was told "Download format not
    # supported for this dataset. Defaulting to zip." Asking for what is
    # served keeps that warning out of every run's log.
    assert request["download_format"] == "zip"


def test_months_are_zero_padded_because_the_cds_rejects_a_bare_one():
    assert spei.cds_request("2026", ["1", "2", "10"])["month"] == ["01", "02", "10"]


def test_the_ensemble_product_is_not_requested():
    """Ten realisations for a country mean that comes out the same."""

    assert spei.CDS_PRODUCT_TYPE == "reanalysis"
    assert "ensemble" not in spei.CDS_PRODUCT_TYPE


def test_a_year_is_one_request_not_twelve(tmp_path):
    """A request per month is thousands of queued jobs for the same data."""

    class _Client:
        def __init__(self):
            self.requests = []

        def retrieve(self, dataset, request, target):
            self.requests.append((dataset, request))
            Path(target).write_bytes(b"nc")

    client = _Client()
    spei.fetch_grids(
        tmp_path, [f"2026-{m:02d}" for m in range(1, 13)],
        client=client, today=TODAY,
    )
    # Two, because 2026 straddles the two releases — never twelve, which is
    # the fault this guards against. The months are split at the boundary
    # and nothing is asked for twice.
    assert len(client.requests) == 2
    by_type = {
        req["dataset_type"]: req["month"] for _, req in client.requests
    }
    assert by_type[spei.CDS_DATASET_TYPE_CONSOLIDATED] == [
        f"{m:02d}" for m in range(1, 6)
    ]
    assert by_type[spei.CDS_DATASET_TYPE_INTERMEDIATE] == [
        f"{m:02d}" for m in range(6, 13)
    ]


def test_a_closed_year_is_still_one_request(tmp_path):
    """Splitting a year is the boundary case, not the rule.

    Every year behind the consolidated lag is one request exactly as it was
    before the switch, which is nearly the whole record.
    """

    class _Client:
        def __init__(self):
            self.requests = []

        def retrieve(self, dataset, request, target):
            self.requests.append(request)
            Path(target).write_bytes(b"nc")

    client = _Client()
    spei.fetch_grids(
        tmp_path, [f"2019-{m:02d}" for m in range(1, 13)],
        client=client, today=TODAY,
    )
    assert len(client.requests) == 1
    assert client.requests[0]["dataset_type"] == spei.CDS_DATASET_TYPE_CONSOLIDATED


# ---------------------------------------------------------------------------
# The deadline: a queued CDS job is not a fault
# ---------------------------------------------------------------------------


def test_the_deadline_stops_starting_jobs_and_reports_the_rest_owed(tmp_path):
    """A CDS job can sit queued for hours against a six-hour job cap."""

    clock = iter([0.0, 0.0, 10_000.0, 10_000.0, 10_000.0, 10_000.0])

    class _Client:
        def __init__(self):
            self.years = []

        def retrieve(self, dataset, request, target):
            self.years.append(request["year"][0])
            Path(target).write_bytes(b"nc")

    client = _Client()
    result = spei.fetch_grids(
        tmp_path, ["2016-01", "2017-01", "2018-01"],
        deadline_sec=5_000, client=client, now=lambda: next(clock),
    )
    assert client.years == ["2016"]
    assert result.years_owed == ["2017", "2018"]
    assert result.deadline_hit is True
    assert result.complete is False


def test_a_year_already_downloaded_is_not_fetched_again(tmp_path):
    """The CSV is the durable artifact; a closed year never needs re-asking."""

    (tmp_path / "spei3_2016_consolidated.nc").write_bytes(b"already here")

    class _Client:
        def retrieve(self, dataset, request, target):  # pragma: no cover
            raise AssertionError("re-fetched a year already on disk")

    result = spei.fetch_grids(
        tmp_path, ["2016-01"], client=_Client(), today=TODAY,
    )
    assert result.skipped_present == ["2016 (consolidated)"]
    assert result.written == []


def test_an_unpacked_year_is_not_fetched_again(tmp_path):
    """The resume check reads the UNIT, not the name it was downloaded under.

    An unpacked unit is called ``spei3_2016_consolidated__data.nc``, so a
    check on the downloaded filename finds nothing and asks the CDS for
    95 MB it already holds — every run, forever.
    """

    (tmp_path / "spei3_2016_consolidated__unpacked.nc").write_bytes(
        b"CDF\x01already here"
    )

    class _Client:
        def retrieve(self, dataset, request, target):  # pragma: no cover
            raise AssertionError("re-fetched a year already unpacked on disk")

    result = spei.fetch_grids(
        tmp_path, ["2016-01"], client=_Client(), today=TODAY,
    )
    assert result.skipped_present == ["2016 (consolidated)"]


def test_one_release_never_answers_for_the_other(tmp_path):
    """The trap the tagged filenames exist to close.

    A year at the boundary needs both releases. With the resume check on the
    year alone, the consolidated file that landed first would answer for the
    intermediate request too, and the newest months — the whole point of the
    switch — would never be asked for again. That failure is permanent
    rather than merely wasteful.
    """

    (tmp_path / "spei3_2026_consolidated__data.nc").write_bytes(b"CDF\x01held")
    asked = []

    class _Client:
        def retrieve(self, dataset, request, target):
            asked.append(request["dataset_type"])
            Path(target).write_bytes(b"nc")

    result = spei.fetch_grids(
        tmp_path, ["2026-03", "2026-07"], client=_Client(), today=TODAY,
    )
    assert asked == [spei.CDS_DATASET_TYPE_INTERMEDIATE]
    assert result.skipped_present == ["2026 (consolidated)"]
    assert result.written == ["2026 (intermediate)"]


def test_a_failed_year_is_owed_not_fatal(tmp_path):
    """The other years may be fine, and the gates decide what is publishable."""

    class _Client:
        def retrieve(self, dataset, request, target):
            if request["year"] == ["2017"]:
                raise RuntimeError("invalid request / no valid combination")
            Path(target).write_bytes(b"nc")

    result = spei.fetch_grids(
        tmp_path, ["2016-01", "2017-01"], client=_Client(), today=TODAY,
    )
    assert result.written == ["2016 (consolidated)"]
    assert "2017" in result.failed
    assert "no valid combination" in result.failed["2017"]
    # Keyed by YEAR, because that is what the absent-reason reader and the
    # workflow's cache prune read, and a year is incomplete whichever of its
    # units failed — but the message names the release, so two failures of
    # one year are both readable.
    assert result.failed["2017"].startswith(spei.CDS_DATASET_TYPE_CONSOLIDATED)
    assert result.complete is False


def test_a_partial_file_from_a_failed_year_is_removed(tmp_path):
    """A truncated download must not be mistaken for a year already held."""

    class _Client:
        def retrieve(self, dataset, request, target):
            Path(target).write_bytes(b"half a file")
            raise RuntimeError("connection reset")

    spei.fetch_grids(tmp_path, ["2016-01"], client=_Client(), today=TODAY)
    assert not (tmp_path / "spei3_2016_consolidated.nc").exists()


# ---------------------------------------------------------------------------
# Saturation: the index stopping rather than measuring
#
# The first full rebuild (run 34461670816) reduced 125 months into 29,625
# rows and then failed its own value gate on twelve figures — at TWO
# magnitudes, 8.2095 and -8.2221, repeated to four decimals across different
# countries and different years. No area-weighted mean of real data does
# that. SPEI is a fitted probability run through the inverse normal, and this
# product is `SPEI3_genlogistic_...`: once the probability reaches the
# floating-point neighbourhood of 0 or 1 the inverse normal returns ~±8.2
# whatever the water balance was. It happened in Bahrain (almost no rainfall
# variance to fit) and Palau (one nearest cell), which is where a fit
# degenerates rather than anywhere at random.
# ---------------------------------------------------------------------------


#: The exact figures the run produced, kept as data so a future change to
#: the threshold has to argue with the evidence rather than with a number.
SATURATED_FROM_RUN_34461670816 = (8.2095, -8.2221)


def _grid_of(values: list[list[float | None]]) -> "spei.Grid":
    lats = [1.0 - i for i in range(len(values))]
    lons = [float(j) for j in range(len(values[0]))]
    return spei.Grid(lats=lats, lons=lons, values=values)


class _Box:
    """A country covering the whole test grid."""

    iso3 = "TST"
    bounds = (-99.0, -99.0, 99.0, 99.0)


def test_the_values_the_run_saturated_on_are_recognised():
    for value in SATURATED_FROM_RUN_34461670816:
        assert spei._is_saturated(value), value
        assert spei._is_missing(value), value


def test_a_real_extreme_is_not_treated_as_saturated():
    """A -4 sigma month is a drought, not an artefact. The gate's soft band
    exists for exactly these, and dropping them would be the fix eating the
    signal it was meant to protect."""

    for value in (-4.0, 3.9, -5.9219, 5.4996):
        assert not spei._is_saturated(value), value
        assert not spei._is_missing(value), value


def test_one_saturated_cell_does_not_poison_a_country_mean():
    """The half of this that matters for a large country.

    Bahrain is small; a country with two hundred good cells and one saturated
    one would otherwise carry the artefact into its mean, where nothing
    downstream could see it.
    """

    grid = _grid_of([[1.0, 8.2095], [1.0, 1.0]])
    result = spei.country_mean(grid, _Box(), contains=lambda c, lon, lat: True)

    assert result.coverage == spei.COVERAGE_CELLS
    assert result.value == pytest.approx(1.0)
    assert result.n_cells == 3
    assert result.n_saturated == 1


def test_a_country_whose_only_cell_saturated_gets_no_value():
    """Palau's case: one nearest cell, and it saturated.

    No value means no row, and the rulebook entry's
    `absence_means_no_drought: false` reads that as unknown — which is the
    truth. Writing -8.2 would have the drought gate read catastrophic drought
    in Palau every March.
    """

    grid = _grid_of([[-8.2221]])
    result = spei.country_mean(grid, _Box(), contains=lambda c, lon, lat: True)

    assert result.value is None
    assert result.coverage == spei.COVERAGE_NONE
    assert result.n_saturated == 1


def test_the_nearest_cell_fallback_will_not_serve_a_saturated_cell():
    """Otherwise the drop above is undone one line later."""

    grid = _grid_of([[8.2095, None], [None, 1.5]])
    # Nothing is inside the territory, so the nearest-cell path runs.
    result = spei.country_mean(grid, _Box(), contains=lambda c, lon, lat: False)

    assert result.value == pytest.approx(1.5)
    assert result.coverage == spei.COVERAGE_NEAREST


def test_saturated_cells_are_counted_and_named_not_silently_dropped():
    """A rise means the fit is degenerating somewhere new."""

    grids = {"2016-01": _grid_of([[8.2095, 1.0]])}
    countries = {"TST": _Box()}
    rows, report = spei.reduce_grids(
        grids, countries, iso3s=["TST"], contains=lambda c, lon, lat: True
    )

    assert report.saturated_cells == 1
    assert report.saturated_country_months == ["TST/2016-01=1"]
    assert report.as_dict()["saturated_cells"] == 1
    # The month still produces a row: one bad cell out of two is not a lost
    # country-month.
    assert [r["value"] for r in rows] == [1.0]


def test_the_hard_gate_stays_where_it_is():
    """Saturation handling is not a reason to widen the fill-value catcher.

    A 1e20 reaching the feed would make a whole month read as wet against a
    rulebook that thresholds at -1.0 sigma, and that is what the hard limit
    is for. It sits below the saturation figure on purpose: anything that
    somehow arrives saturated still fails the gate rather than being
    published.
    """

    assert spei.SANITY_HARD_LIMIT < spei.SATURATION_ABS
    assert spei.SANITY_SOFT_LIMIT < spei.SANITY_HARD_LIMIT


# ---------------------------------------------------------------------------
# The dependency the requirements file did not declare
#
# The second live run got past the container fix, unpacked 128 months and read
# every one of them, and then died on `ModuleNotFoundError: No module named
# 'yaml'` — nine minutes of download and 75 seconds of grid reading spent
# before the import fired. `yaml` is nothing this producer imports: the reduce
# borrows the vendored boundary layer from `resolver.hazard_resolution`, and
# reading a module from that package runs the package's __init__, which
# imports the rulebook, which imports yaml.
# ---------------------------------------------------------------------------


REQUIREMENTS = Path(__file__).resolve().parents[2] / "requirements-spei3.txt"

#: Import name -> the distribution that provides it. Only the ones this
#: chain can actually reach; a longer table would be a guess.
_OURS = {"resolver", "scripts", "pythia", "forecaster", "horizon_scanner", "interpreter", "sibyl"}

_DISTRIBUTION_FOR = {
    "yaml": "PyYAML",
    "shapely": "shapely",
    "xarray": "xarray",
    "netCDF4": "netCDF4",
    "numpy": "numpy",
    "cdsapi": "cdsapi",
    "pandas": "pandas",
    "duckdb": "duckdb",
    "requests": "requests",
}


def _declared_distributions() -> set[str]:
    names = set()
    for line in REQUIREMENTS.read_text("utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(re.split(r"[<>=!\[]", line, 1)[0].strip().lower())
    return names


def _module_level_third_party_imports(path: Path) -> set[str]:
    """Third-party modules imported when the file is merely IMPORTED.

    Distinct from the walk below, which sees function-local imports too: this
    producer quarantines its heavy imports inside the functions that need
    them, so "what does importing it cost" and "what can running it need" are
    different questions with different answers.
    """

    tree = ast.parse(path.read_text("utf-8"))
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return {n for n in found if n not in sys.stdlib_module_names and n not in _OURS}


def _third_party_imports(path: Path) -> set[str]:
    """Every third-party module name a file imports, local imports included."""

    tree = ast.parse(path.read_text("utf-8"))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return {
        name for name in found
        if name not in sys.stdlib_module_names and name not in _OURS
    }


#: The functions the workflow's five subcommands actually reach, each of
#: which quarantines a heavy import behind it. `_cmd_coverage` is
#: deliberately absent: it reads the canonical DuckDB, which this workflow
#: never downloads and must not, so its `duckdb` import is a need of a
#: command run elsewhere in the full environment.
#:
#: The two probe functions are here because the probe JOB runs in this
#: workflow too, off the same requirements file. Their `cdsapi` import is
#: already declared, so listing them changes nothing today — which is the
#: point of doing it now rather than after a probe run dies on a module the
#: producer's own path happens not to need.
_WORKFLOW_IMPORT_SITES = (
    "shapely_contains",
    "load_grids_from_dir",
    "fetch_grids",
    "probe_enum_values",
    "describe_enum_values",
)


def _local_third_party_imports(path: Path, functions: tuple[str, ...]) -> set[str]:
    """Third-party imports inside the named functions of one file."""

    tree = ast.parse(path.read_text("utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name not in functions:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                found.update(a.name.split(".")[0] for a in sub.names)
            elif isinstance(sub, ast.ImportFrom) and sub.module and sub.level == 0:
                found.add(sub.module.split(".")[0])
    return {n for n in found if n not in sys.stdlib_module_names}


def test_the_requirements_declare_everything_the_geometry_import_pulls_in():
    """The producer enters `resolver.hazard_resolution`; its __init__ runs.

    Walked statically rather than by importing, because the environment this
    test runs in HAS yaml — which is exactly why the gap survived into
    production, where the workflow installs `requirements-spei3.txt` and
    nothing else. The chain is the package __init__ plus the intra-package
    modules it reaches, one hop, which is as far as the __init__ goes.
    """

    package = Path(__file__).resolve().parents[1] / "hazard_resolution"
    needed = _third_party_imports(package / "__init__.py")
    tree = ast.parse((package / "__init__.py").read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "resolver.hazard_resolution."
        ):
            module = package / (node.module.split(".")[-1] + ".py")
            if module.exists():
                needed |= _third_party_imports(module)

    producer = Path(__file__).resolve().parents[2] / "scripts" / "build_spei3_country_means.py"
    needed |= _module_level_third_party_imports(producer)
    needed |= _local_third_party_imports(producer, _WORKFLOW_IMPORT_SITES)

    declared = _declared_distributions()
    missing = sorted(
        _DISTRIBUTION_FOR[name] for name in needed
        if name in _DISTRIBUTION_FOR
        and _DISTRIBUTION_FOR[name].lower() not in declared
    )
    assert not missing, (
        f"requirements-spei3.txt does not declare {missing}, which the reduce "
        "stage's import chain needs — the workflow installs only this file"
    )
    # And the one that actually bit, named so a future reader knows why a
    # YAML parser sits in a file otherwise made of scientific pins.
    assert "yaml" in needed
    assert "pyyaml" in declared


def test_the_geometry_borrow_is_the_only_thing_that_needs_yaml():
    """Not a stylistic point: it is why the fix is one line in one file.

    If the producer's own code needed yaml, or a second package did, the
    requirements file would be tracking two chains instead of one.
    """

    producer = Path(__file__).resolve().parents[2] / "scripts" / "build_spei3_country_means.py"
    assert "yaml" not in _module_level_third_party_imports(producer)
    assert "yaml" not in _local_third_party_imports(
        producer, _WORKFLOW_IMPORT_SITES + ("_cmd_coverage",)
    )


def test_the_producer_never_asks_the_pipeline_to_carry_its_stack():
    """The contract this whole arrangement exists to keep.

    `resolver/hazard_resolution/` reads a committed CSV through the rulebook's
    `tabular` provider. If it gained an xarray, netCDF4 or cdsapi import, the
    pipeline would have to install a scientific stack and a credentialled
    client for one external service to produce a file it only ever reads.

    **shapely is deliberately not in this set, and the brief that asked for
    the set named it.** `resolver/hazard_resolution/geometry.py` has imported
    shapely since August 2026 (commit fd54d89) — it is the cyclone detector's
    own boundary code, and shapely is declared in pyproject's `ingestion`
    extra and pinned in constraints-ci.txt. The resolution path cannot GAIN a
    dependency it already has, and asserting otherwise would fail on
    untouched code, which is how a guard gets switched off.
    """

    package = Path(__file__).resolve().parents[1] / "hazard_resolution"
    forbidden = {"xarray", "netCDF4", "cdsapi"}
    offenders = {}
    for module in sorted(package.rglob("*.py")):
        hit = _third_party_imports(module) & forbidden
        if hit:
            offenders[module.name] = sorted(hit)
    assert not offenders, (
        f"the resolution path must not import {sorted(forbidden)}: {offenders}"
    )


# ---------------------------------------------------------------------------
# The container: sniffed, never assumed
#
# Run 34456535827 fetched all eleven years and then failed to read one of
# them. The CDS said why in its own log — "Download format not supported for
# this dataset. Defaulting to zip." — and wrote the archive to the `.nc`
# filename the client had been handed, so xarray reported no matching IO
# backend: a message about our installation, for a file that was never a
# NetCDF.
# ---------------------------------------------------------------------------


def _zip_of(members: dict[str, bytes]) -> bytes:
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def test_a_zip_named_nc_is_recognised_as_a_zip(tmp_path):
    path = tmp_path / "spei3_2016.nc"
    path.write_bytes(_zip_of({"data.nc": b"CDF\x01payload"}))
    assert spei.sniff_container(path) == "zip"


def test_a_real_netcdf_is_recognised_and_left_alone(tmp_path):
    """The fixture is a genuine NetCDF, so the sniff must pass it through."""

    fixture = FIXTURES / "spei3_2016.nc"
    assert spei.sniff_container(fixture).startswith("netcdf")
    assert spei.unpack_if_archive(fixture) == [fixture]
    # Unchanged on disk: a passthrough must never delete its input.
    assert fixture.exists()


def test_a_file_that_is_neither_names_what_it_actually_is(tmp_path):
    path = tmp_path / "spei3_2016.nc"
    path.write_bytes(b"<html>Access denied")
    assert spei.sniff_container(path).startswith("unknown:")


def test_unpacking_replaces_the_archive_with_its_members(tmp_path):
    path = tmp_path / "spei3_2016.nc"
    path.write_bytes(_zip_of({"spei3_month.nc": b"CDF\x01one"}))

    written = spei.unpack_if_archive(path)

    assert [p.name for p in written] == ["spei3_2016__spei3_month.nc"]
    assert written[0].read_bytes() == b"CDF\x01one"
    # The archive goes: two copies of a 95 MB year doubles the cache entry
    # for no gain, and leaving it would make the loader try to open it again.
    assert not path.exists()


def test_members_keep_the_year_so_two_years_cannot_collide(tmp_path):
    """Both archives can legitimately carry a member called `data.nc`."""

    for year in ("2016", "2017"):
        target = tmp_path / f"spei3_{year}.nc"
        target.write_bytes(_zip_of({"data.nc": f"CDF\x01{year}".encode()}))
        spei.unpack_if_archive(target)

    names = sorted(p.name for p in tmp_path.glob("*.nc"))
    assert names == ["spei3_2016__data.nc", "spei3_2017__data.nc"]


def test_a_zip_carrying_no_netcdf_member_says_what_it_carried(tmp_path):
    path = tmp_path / "spei3_2016.nc"
    path.write_bytes(_zip_of({"data.grib": b"GRIB"}))

    with pytest.raises(ValueError) as excinfo:
        spei.unpack_if_archive(path)

    assert "data.grib" in str(excinfo.value)


def test_the_loader_unpacks_a_cached_archive_rather_than_refusing_it(tmp_path):
    """A cache from before this fix holds zips under `.nc` names.

    Re-downloading a gigabyte to correct a container mismatch buys nothing,
    so the read unpacks too — the fetch is not the only door in.
    """

    payload = (FIXTURES / "spei3_2016.nc").read_bytes()
    (tmp_path / "spei3_2016.nc").write_bytes(_zip_of({"grid.nc": payload}))

    grids = spei.load_grids_from_dir(tmp_path)

    assert grids, "the archived fixture should have been unpacked and read"


def test_the_loader_refuses_a_file_that_is_not_a_netcdf_and_says_so(tmp_path):
    """xarray's own message points at the requirements file, not the download."""

    (tmp_path / "spei3_2016.nc").write_bytes(b"<html>Access denied</html>")

    with pytest.raises(ValueError) as excinfo:
        spei.load_grids_from_dir(tmp_path)

    message = str(excinfo.value)
    assert "not a NetCDF" in message
    assert "download_format" in message


def test_a_downloaded_archive_is_unpacked_by_the_fetch_itself(tmp_path):
    """So the reduce sees NetCDF whichever container the CDS chose."""

    class _Client:
        def retrieve(self, dataset, request, target):
            Path(target).write_bytes(_zip_of({"data.nc": b"CDF\x01payload"}))

    result = spei.fetch_grids(tmp_path, ["2016-01"], client=_Client(), today=TODAY)

    assert result.written == ["2016 (consolidated)"]
    assert [p.name for p in tmp_path.glob("*.nc")] == [
        "spei3_2016_consolidated__data.nc"
    ]
    # The reported volume is the ARCHIVE's size: that is what came down the
    # wire, and it is the number the docstring's estimate is corrected from.
    assert result.bytes_downloaded > 0


def test_a_year_that_arrives_unreadable_is_owed_not_kept(tmp_path):
    """Leaving it on disk would make every later run's resume check skip it."""

    class _Client:
        def retrieve(self, dataset, request, target):
            Path(target).write_bytes(_zip_of({"data.grib": b"GRIB"}))

    result = spei.fetch_grids(tmp_path, ["2016-01"], client=_Client())

    assert result.written == []
    assert "2016" in result.failed
    assert list(tmp_path.glob("spei3_2016*")) == []


# ---------------------------------------------------------------------------
# Reduce MERGES: the fault that would have fired on the first scheduled run
# ---------------------------------------------------------------------------


def test_two_existing_months_plus_one_incoming_yields_three():
    existing = _rows(("AAA", "2016-01", "-1.0"), ("AAA", "2016-02", "-0.5"))
    incoming = _rows(("AAA", "2016-03", "0.25"))
    merged = spei.merge_rows(existing, incoming)
    assert [r["ym"] for r in merged] == ["2016-01", "2016-02", "2016-03"]


def test_an_incoming_month_replaces_its_rows_and_changes_no_others():
    """ERA5T is revised to final ERA5 later; the first version is not the last."""

    existing = _rows(
        ("AAA", "2016-01", "-1.0"), ("BBB", "2016-01", "0.2"),
        ("AAA", "2016-02", "-0.5"),
    )
    incoming = _rows(("AAA", "2016-01", "-1.4"))
    merged = {(r["iso3"], r["ym"]): r["value"] for r in spei.merge_rows(existing, incoming)}
    assert merged[("AAA", "2016-01")] == "-1.4"
    assert merged[("BBB", "2016-01")] == "0.2"
    assert merged[("AAA", "2016-02")] == "-0.5"
    assert len(merged) == 3


def test_an_empty_incoming_set_leaves_the_file_byte_identical(tmp_path):
    """The property the whole scheduled producer rests on.

    A run whose CDS jobs were all still queued has nothing to merge. It must
    not rewrite a line: a diff full of unchanged rows is a diff nobody reads,
    and a truncation is how ten years of series becomes three months of it.
    """

    source = FIXTURES / "committed_feed.csv"
    target = tmp_path / "feed.csv"
    target.write_bytes(source.read_bytes())
    before = target.read_bytes()

    spei.write_csv(spei.merge_rows(spei.read_csv_rows(target), []), target)
    assert target.read_bytes() == before


def test_a_reduce_of_one_month_does_not_truncate_ten_years(tmp_path):
    """End to end through the CLI's own merge, not just the helper."""

    feed = tmp_path / "feed.csv"
    spei.write_csv(
        _rows(*[("AAA", f"20{y:02d}-01", "-1.0") for y in range(16, 26)]), feed,
    )
    incoming = _rows(("AAA", "2026-01", "-0.2"))
    spei.write_csv(spei.merge_rows(spei.read_csv_rows(feed), incoming), feed)
    assert len(spei.read_csv_rows(feed)) == 11


def test_a_month_whose_rows_did_not_move_did_not_gain_coverage():
    """What the drought restale is scoped to."""

    before = _rows(("AAA", "2016-01", "-1.0"), ("AAA", "2016-02", "-0.5"))
    after = before + _rows(("AAA", "2016-03", "0.1"))
    assert spei.months_gaining_coverage(before, after) == ["2016-03"]


def test_a_revised_value_counts_as_gaining_coverage():
    """A revision can move a drought verdict as surely as a new month can."""

    before = _rows(("AAA", "2016-01", "-0.5"))
    after = _rows(("AAA", "2016-01", "-1.4"))
    assert spei.months_gaining_coverage(before, after) == ["2016-01"]


def test_a_new_country_in_an_existing_month_counts():
    before = _rows(("AAA", "2016-01", "-0.5"))
    after = before + _rows(("BBB", "2016-01", "-1.1"))
    assert spei.months_gaining_coverage(before, after) == ["2016-01"]


def test_nothing_changing_requests_no_restale():
    rows = _rows(("AAA", "2016-01", "-0.5"))
    assert spei.months_gaining_coverage(rows, list(rows)) == []


# ---------------------------------------------------------------------------
# Planning: never a hardcoded range
# ---------------------------------------------------------------------------


def test_the_window_is_the_revision_tail_plus_whatever_is_missing():
    window = spei.plan_window(
        ["2016-01", "2016-02"], start_ym="2016-01", end_ym="2016-06",
        revision_months=2,
    )
    # The tail (May, June) plus the months the CSV does not hold.
    assert window.months == ["2016-03", "2016-04", "2016-05", "2016-06"]
    assert window.revision_months == ["2016-05", "2016-06"]
    assert window.missing_months == ["2016-03", "2016-04", "2016-05", "2016-06"]


def test_a_covered_series_still_re_fetches_the_revision_tail():
    """ERA5T values are revised later, so a covered month is not a done one."""

    window = spei.plan_window(
        spei.month_range("2016-01", "2016-06"),
        start_ym="2016-01", end_ym="2016-06", revision_months=3,
    )
    assert window.months == ["2016-04", "2016-05", "2016-06"]
    assert window.missing_months == []


def test_a_hole_in_the_middle_is_picked_up_without_anybody_noticing():
    """The self-healing half: a month lost to a timed-out CDS job comes back."""

    held = [m for m in spei.month_range("2016-01", "2016-12") if m != "2016-07"]
    window = spei.plan_window(
        held, start_ym="2016-01", end_ym="2016-12", revision_months=1
    )
    assert "2016-07" in window.months


def test_a_full_rebuild_asks_for_the_whole_range():
    window = spei.plan_window(
        spei.month_range("2016-01", "2016-12"),
        start_ym="2016-01", end_ym="2016-03", full_rebuild=True,
    )
    assert window.months == ["2016-01", "2016-02", "2016-03"]
    assert window.full_rebuild is True


def test_the_window_names_only_the_years_it_touches():
    window = spei.plan_window(
        [], start_ym="2016-11", end_ym="2017-02", revision_months=0
    )
    assert window.years == ["2016", "2017"]


def test_a_window_round_trips_through_json():
    window = spei.plan_window([], start_ym="2016-01", end_ym="2016-02")
    assert spei.Window.from_dict(json.loads(json.dumps(window.as_dict()))).months == (
        window.months
    )


# ---------------------------------------------------------------------------
# The gates. Fail closed: a wrong feed is worse than a stale one.
# ---------------------------------------------------------------------------


def test_a_fill_value_read_as_an_anomaly_refuses_the_candidate():
    """1e20 is NetCDF's usual fill; the rulebook thresholds at -1.0 sigma."""

    result = spei.validate_candidate(_rows(("AAA", "2016-01", "1e20")), [])
    assert result.ok is False
    assert any("sigma" in f for f in result.failures)


def test_a_value_just_outside_five_sigma_warns_and_still_publishes():
    """A real -5.2 sigma exists and is not a reason to publish nothing."""

    result = spei.validate_candidate(_rows(("AAA", "2016-01", "-5.2")), [])
    assert result.ok is True
    assert any("5" in w for w in result.warnings)


def test_an_unreadable_value_refuses_the_candidate():
    result = spei.validate_candidate(_rows(("AAA", "2016-01", "n/a")), [])
    assert result.ok is False


def test_a_requested_month_that_is_absent_with_no_reason_fails():
    """"It is not there" and "the job was queued" want different responses."""

    window = spei.plan_window([], start_ym="2016-01", end_ym="2016-02")
    result = spei.validate_candidate(
        _rows(("AAA", "2016-01", "-1.0")), [], window=window
    )
    assert result.ok is False
    assert any("2016-02" in f for f in result.failures)


def test_a_requested_month_absent_with_a_stated_reason_passes():
    window = spei.plan_window([], start_ym="2016-01", end_ym="2016-02")
    result = spei.validate_candidate(
        _rows(("AAA", "2016-01", "-1.0")), [], window=window,
        absent_reasons={"2016-02": "cds job outstanding when the deadline expired"},
    )
    assert result.ok is True
    assert result.months_absent["2016-02"].startswith("cds job")


def test_a_month_that_lost_countries_fails():
    """The grid does not shrink. A month that loses countries was reduced
    against something other than the boundary layer the old rows used."""

    previous = _rows(("AAA", "2016-01", "-1.0"), ("BBB", "2016-01", "-0.5"))
    candidate = _rows(("AAA", "2016-01", "-1.0"))
    result = spei.validate_candidate(candidate, previous)
    assert result.ok is False
    assert any("lost countries" in f for f in result.failures)


def test_a_sampled_share_that_jumps_fails():
    """A jump means the grid or the boundary layer moved under us, so these
    means are not the quantity the committed rows hold."""

    previous = [
        {"iso3": f"C{i:02d}", "ym": "2016-01", "value": "-1.0",
         "coverage": "cells", "n_cells": "4"}
        for i in range(100)
    ]
    candidate = [dict(row) for row in previous]
    for row in candidate[:20]:
        row["coverage"] = "nearest_cell"
    result = spei.validate_candidate(candidate, previous)
    assert result.ok is False
    assert any("nearest cell" in f for f in result.failures)


def test_a_sampled_share_that_barely_moves_passes():
    previous = [
        {"iso3": f"C{i:02d}", "ym": "2016-01", "value": "-1.0",
         "coverage": "cells", "n_cells": "4"}
        for i in range(100)
    ]
    candidate = [dict(row) for row in previous]
    candidate[0]["coverage"] = "nearest_cell"
    assert spei.validate_candidate(candidate, previous).ok is True


def test_an_ordinary_monthly_append_passes_every_gate():
    """The gates must not refuse the case they exist to allow."""

    previous = spei.read_csv_rows(FIXTURES / "committed_feed.csv")
    incoming = [
        {"iso3": row["iso3"], "ym": "2016-03", "value": row["value"],
         "coverage": row["coverage"], "n_cells": row["n_cells"]}
        for row in previous if row["ym"] == "2016-02"
    ]
    candidate = spei.merge_rows(previous, incoming)
    window = spei.plan_window(
        spei.months_in(previous), start_ym="2016-01", end_ym="2016-03",
        revision_months=1,
    )
    result = spei.validate_candidate(candidate, previous, window=window)
    assert result.ok is True, result.failures
    assert result.changed_months == ["2016-03"]


# ---------------------------------------------------------------------------
# The status file, and the restale request it carries
# ---------------------------------------------------------------------------


def test_the_status_file_says_what_the_feed_covers():
    rows = spei.read_csv_rows(FIXTURES / "committed_feed.csv")
    status = spei.build_status(rows, run_id="123")
    assert status["oldest_month"] == "2016-01"
    assert status["newest_month"] == "2016-02"
    assert status["months"] == 2
    assert status["countries"] == 2
    assert status["coverage"] == {"cells": 2, "nearest_cell": 2}
    assert status["last_success_run_id"] == "123"


def test_a_restale_request_names_the_months_that_gained_coverage():
    rows = _rows(("AAA", "2016-01", "-1.0"), ("AAA", "2016-02", "-0.5"))
    status = spei.build_status(rows, changed_months=["2016-02"], run_id="1")
    assert status["restale"]["hazard"] == "DR"
    assert status["restale"]["months"] == ["2016-02"]
    assert status["restale"]["token"]


def test_a_run_that_changed_nothing_requests_nothing_new():
    rows = _rows(("AAA", "2016-01", "-1.0"))
    assert "restale" not in spei.build_status(rows, run_id="1")


def test_a_pending_request_survives_a_run_that_changed_nothing():
    """Until the backcast has applied it, the request still stands."""

    rows = _rows(("AAA", "2016-01", "-1.0"))
    first = spei.build_status(rows, changed_months=["2016-01"], run_id="1")
    second = spei.build_status(rows, run_id="2", previous=first)
    assert second["restale"] == first["restale"]


def test_the_same_extension_produces_the_same_token():
    """Which is what makes the restale one-shot rather than a treadmill."""

    rows = _rows(("AAA", "2016-01", "-1.0"))
    a = spei.restale_token(["2016-01"], rows)
    b = spei.restale_token(["2016-01"], list(rows))
    assert a == b


def test_a_different_extension_produces_a_different_token():
    rows = _rows(("AAA", "2016-01", "-1.0"), ("AAA", "2016-02", "-0.5"))
    assert spei.restale_token(["2016-01"], rows) != spei.restale_token(
        ["2016-01", "2016-02"], rows
    )


def test_a_failed_run_records_the_failure_and_keeps_the_last_good_run_id():
    """"The last good run" is a fact about the past; a failure does not move it."""

    rows = _rows(("AAA", "2016-01", "-1.0"))
    good = spei.build_status(rows, run_id="100")
    bad = spei.build_status(
        rows, status=spei.STATUS_FAILED, run_id="101",
        failure={"reason": "a gate failed"}, previous=good,
    )
    assert bad["status"] == "failed"
    assert bad["last_success_run_id"] == "100"
    assert bad["last_failure"]["run_id"] == "101"


def test_a_run_with_months_owed_says_it_is_incomplete():
    rows = _rows(("AAA", "2016-01", "-1.0"))
    status = spei.build_status(
        rows, status=spei.STATUS_INCOMPLETE, months_owed=["2016-02"], run_id="1",
    )
    assert status["status"] == "incomplete"
    assert status["months_owed"] == ["2016-02"]


# ---------------------------------------------------------------------------
# Reading the real NetCDF. The one test that needs the scientific stack, so
# it skips ITSELF rather than the file: a suite that skips wholesale on a
# missing dependency is a suite whose failures nobody sees until CI.
# ---------------------------------------------------------------------------


def test_the_loader_reads_the_committed_netcdf():
    pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")

    grids = spei.load_grids_from_dir(FIXTURES)
    assert sorted(grids) == ["2016-01", "2016-02"]
    grid = grids["2016-01"]
    assert len(grid.lats) == 4 and len(grid.lons) == 8
    # The NaN cell survives as missing, not as a zero: a value of zero is a
    # measurement and a missing cell is not.
    assert spei._is_missing(grid.values[0][0])
    assert grid.values[0][1] == pytest.approx(-1.9, abs=1e-4)


def test_the_variable_is_found_by_name_not_by_position():
    pytest.importorskip("xarray")
    import xarray as xr

    with xr.open_dataset(FIXTURES / "spei3_2016.nc") as ds:
        assert spei._spei_variable_name(ds) == "SPEI3"


def test_reduce_over_the_committed_netcdf_produces_country_means():
    """The reduction, the loader and the CSV writer in one pass."""

    pytest.importorskip("xarray")

    grids = spei.load_grids_from_dir(FIXTURES)

    class _Box:
        def __init__(self, iso3, bounds):
            self.iso3, self.bounds = iso3, bounds

    def contains(country, lon, lat):
        minx, miny, maxx, maxy = country.bounds
        return minx <= lon <= maxx and miny <= lat <= maxy

    countries = {
        # Spans several cells.
        "AAA": _Box("AAA", (0.0, 0.0, 3.0, 2.0)),
        # Smaller than a cell: no centre inside it, so it is SAMPLED.
        "BBB": _Box("BBB", (4.4, 1.4, 4.45, 1.45)),
    }
    rows, report = spei.reduce_grids(
        grids, countries, iso3s=["AAA", "BBB"], contains=contains,
    )
    by_key = {(r["iso3"], r["ym"]): r for r in rows}
    assert by_key[("AAA", "2016-01")]["coverage"] == "cells"
    assert by_key[("BBB", "2016-01")]["coverage"] == "nearest_cell"
    assert report.by_coverage["cells"] == 2
    assert report.by_coverage["nearest_cell"] == 2


# ---------------------------------------------------------------------------
# The producer must stay off the resolution path
# ---------------------------------------------------------------------------


def test_the_resolver_does_not_depend_on_the_producers_stack():
    """cdsapi, xarray and netCDF4 are the producer's, not the pipeline's.

    The rulebook's contract is that a Resolver Update reads a committed CSV.
    Adding a credentialled client and a scientific stack to the pipeline's
    dependency set to produce that CSV would give up the only thing the
    arrangement buys.
    """

    import subprocess
    import sys

    found = subprocess.run(
        [
            "grep", "-rnE", r"^\s*(import|from)\s+(cdsapi|xarray|netCDF4)\b",
            "--include=*.py", "resolver/hazard_resolution",
        ],
        capture_output=True, text=True, cwd=spei.REPO_ROOT,
    )
    assert found.stdout.strip() == "", (
        f"the resolution path imports the producer's stack:\n{found.stdout}"
    )


def test_the_producers_requirements_are_its_own_file():
    text = (spei.REPO_ROOT / "requirements-spei3.txt").read_text("utf-8")
    for package in ("cdsapi", "xarray", "netCDF4", "numpy", "shapely"):
        assert f"{package}==" in text, f"{package} is not pinned"


def test_the_producer_pins_match_the_resolver_constraints():
    """A skew here means CI green on a reduction production never performs."""

    producer = dict(
        line.split("==")
        for line in (spei.REPO_ROOT / "requirements-spei3.txt")
        .read_text("utf-8").splitlines()
        if "==" in line and not line.startswith("#")
    )
    constraints = dict(
        line.split("==")
        for line in (spei.REPO_ROOT / "constraints-ci.txt")
        .read_text("utf-8").splitlines()
        if "==" in line and not line.startswith("#")
    )
    for shared in ("xarray", "netCDF4", "numpy", "shapely"):
        assert producer[shared] == constraints[shared], (
            f"{shared} is pinned to {producer[shared]} for the producer and "
            f"{constraints[shared]} for the resolver tests"
        )


def test_the_workflow_runs_every_stage_of_the_producer():
    """A stage nothing calls is a stage that does not run."""

    text = (
        spei.REPO_ROOT / ".github" / "workflows" / "spei3_refresh.yml"
    ).read_text("utf-8")
    for stage in ("plan", "fetch", "reduce", "validate", "promote"):
        assert f"build_spei3_country_means {stage}" in text, f"{stage} is never run"


def _workflow_steps() -> list[dict]:
    import yaml

    workflow = yaml.safe_load(
        (spei.REPO_ROOT / ".github" / "workflows" / "spei3_refresh.yml")
        .read_text("utf-8")
    )
    return workflow["jobs"]["refresh"]["steps"]


def _step(name: str) -> dict:
    for step in _workflow_steps():
        if step.get("name") == name:
            return step
    raise AssertionError(f"no step called {name!r}")


def test_the_commit_step_survives_a_gate_failure():
    """On a failed gate `promote --failed` writes the status file and leaves
    the feed alone. That is the design; the commit step then has to add a
    path that does not exist.

    Run 34461670816 aborted there with `fatal: pathspec ... did not match any
    files` and exit 128, so the run went red carrying the COMMIT's message
    instead of the gate's, and the gate's own error step was skipped
    entirely. Six weeks of work and a 56-minute reduce reported as a git
    error.
    """

    # Comments stripped first: the step's own comment quotes the command it
    # replaced, and a bare substring test would match the explanation rather
    # than the code. (The concurrency-group test above learned this the same
    # way.)
    run = "\n".join(
        line for line in _step("Commit the feed")["run"].splitlines()
        if not line.strip().startswith("#")
    )
    assert 'git add -- "${FEED}" "${STATUS}"' not in run, (
        "a bare add of both paths aborts when the gate withheld the feed"
    )
    assert '[ -e "${path}" ]' in run
    assert "git add -- \"${path}\"" in run


def test_a_gate_failure_cannot_be_masked_by_a_later_step():
    """The gate is what a reader needs to be told about.

    Without `always()` any failure above this step skips it, and the run
    reports whatever failed last instead of the thing that decided nothing
    would be published.
    """

    step = _step("Fail if a gate failed")
    condition = str(step["if"])
    assert "always()" in condition
    assert "steps.gate.outcome" in condition


def test_the_workflow_is_not_in_the_canonical_db_concurrency_group():
    """It must never be able to cancel the nightly backcast or the ingest."""

    import yaml

    workflow = yaml.safe_load(
        (spei.REPO_ROOT / ".github" / "workflows" / "spei3_refresh.yml")
        .read_text("utf-8")
    )
    assert workflow["concurrency"]["group"] == "spei3-feed"
    assert workflow["concurrency"]["group"] != "pythia-resolver-db"


def test_the_workflow_avoids_the_days_the_monthly_chain_owns():
    """Crons here arrive hours late and the 27th and 28th are contested."""

    text = (
        spei.REPO_ROOT / ".github" / "workflows" / "spei3_refresh.yml"
    ).read_text("utf-8")
    crons = [
        line.split("cron:")[1].strip().strip('"').strip("'")
        for line in text.splitlines() if "cron:" in line
    ]
    assert crons
    for cron in crons:
        day_of_month = cron.split()[2]
        assert day_of_month not in ("27", "28")
        assert "27" not in day_of_month.split(",")
        assert "28" not in day_of_month.split(",")


# ---------------------------------------------------------------------------
# probe mode: ask the CDS what an enum accepts rather than guessing
# ---------------------------------------------------------------------------
#
# The intermediate release's `dataset_type` literal is unknown.
# `intermediate_dataset` is likely by symmetry, and a guessed enum is refused
# with the same message a misspelt variable name gets — so shipping the switch
# on a guess would turn a two-month gain in the feed's leading edge into an
# outage that reads like one. The CDS names accepted values when it rejects an
# out-of-range key, so the workflow asks it.


class _RefusingClient:
    """A CDS client that refuses, and records that it was asked exactly once."""

    def __init__(self, message):
        self.message = message
        self.calls = []

    def retrieve(self, dataset, request, target):
        self.calls.append((dataset, dict(request), target))
        raise RuntimeError(self.message)


@pytest.mark.parametrize("body", [
    "Invalid value for 'dataset_type'. Allowed values: "
    "['consolidated_dataset', 'intermediate_dataset']",
    "400 Client Error: dataset_type must be one of: consolidated_dataset, "
    "intermediate_dataset",
    "invalid request: 'dataset_type' is not a valid value, expected one of "
    "[consolidated_dataset, intermediate_dataset]",
])
def test_the_probe_reads_the_accepted_values_out_of_a_refusal(body):
    """Several shapes, because the wording is not ours to fix.

    A changed sentence should cost the sentence, not the reading, which is why
    the literals are pulled out of whatever fragment a pattern isolated rather
    than captured group by group.
    """

    assert spei.parse_probe_refusal(body) == [
        "consolidated_dataset", "intermediate_dataset",
    ]


def test_the_probe_value_itself_is_never_reported_as_an_accepted_one():
    """It appears in the body the CDS echoes back, and offering it as an
    answer would be the probe teaching itself its own invented literal."""

    body = (
        f"Invalid value {spei.PROBE_INVALID_VALUE!r} for 'dataset_type'. "
        "Allowed values: ['consolidated_dataset', 'intermediate_dataset']"
    )
    assert spei.PROBE_INVALID_VALUE not in spei.parse_probe_refusal(body)


def test_a_refusal_that_names_nothing_is_reported_and_never_guessed():
    """The one outcome that stops the switch. Saying so is the answer; a
    plausible-looking guess here is how a feed goes dark for a cycle."""

    client = _RefusingClient("Internal Server Error")
    result = spei.probe_enum_values(client=client)
    assert result.refused is True
    assert result.values == []
    assert result.settled is False
    assert "cannot settle" in result.detail
    assert "Do not guess" in result.detail


def test_the_probe_sends_one_request_and_asks_for_one_month():
    """It exists to be refused, so there is nothing to download — and a probe
    that quietly pulled tens of megabytes would not be a probe."""

    client = _RefusingClient("Allowed values: ['consolidated_dataset']")
    spei.probe_enum_values(client=client)
    assert len(client.calls) == 1
    _dataset, request, _target = client.calls[0]
    assert request["dataset_type"] == spei.PROBE_INVALID_VALUE
    assert request["year"] == ["2016"]
    assert request["month"] == ["01"]


def test_the_probe_borrows_the_real_request_shape():
    """Otherwise it would answer a question about a request nobody sends.

    Every mandatory key the live request carries has to be present and right,
    or the refusal names the missing key instead of the enum being probed.
    """

    client = _RefusingClient("Allowed values: ['consolidated_dataset']")
    spei.probe_enum_values(client=client)
    _dataset, request, _target = client.calls[0]
    live = spei.cds_request("2016", ["01"])
    for key, value in live.items():
        if key == "dataset_type":
            continue
        assert request[key] == value, key


def test_an_accepted_probe_is_unsettled_rather_than_a_success():
    """If the CDS accepts a value built to be invalid, either the key is being
    ignored or the value collides with a real one. Nothing was learnt, and
    reporting it as settled would be reporting a guess as a measurement."""

    class _Accepting:
        def retrieve(self, dataset, request, target):
            return None

    result = spei.probe_enum_values(client=_Accepting())
    assert result.refused is False
    assert result.settled is False
    assert "ACCEPTED" in result.detail


def test_probe_mode_writes_no_feed_and_no_status(tmp_path, monkeypatch):
    """Fetches nothing, writes nothing, commits nothing — the contract the
    workflow's separate job exists to keep."""

    client = _RefusingClient("Allowed values: ['consolidated_dataset']")
    real = spei.probe_enum_values
    monkeypatch.setattr(spei, "probe_enum_values",
                        lambda *a, **k: real(*a, client=client, **k))
    report = tmp_path / "probe.json"
    rc = spei.main(["probe", "--report-out", str(report)])
    assert rc == 0
    written = {p.name for p in tmp_path.iterdir()}
    assert written == {"probe.json"}, written
    assert json.loads(report.read_text("utf-8"))["values"] == ["consolidated_dataset"]


def test_the_probe_always_exits_zero_because_a_question_is_not_a_fault(tmp_path, monkeypatch):
    client = _RefusingClient("Internal Server Error")
    real = spei.probe_enum_values
    monkeypatch.setattr(spei, "probe_enum_values",
                        lambda *a, **k: real(*a, client=client, **k))
    assert spei.main(["probe", "--report-out", str(tmp_path / "p.json")]) == 0


def test_the_workflow_runs_the_probe_in_a_job_that_cannot_produce_a_feed():
    """A probe sharing the producer's job shares its checkout token, its cache
    and its commit step, and the whole point is that it touches none of them.
    """

    yaml = pytest.importorskip("yaml")
    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github" / "workflows" / "spei3_refresh.yml"
    )
    doc = yaml.safe_load(workflow.read_text("utf-8"))
    assert "probe" in doc["jobs"], "probe mode has no job of its own"
    probe_job = doc["jobs"]["probe"]
    # Dispatch only.
    assert "workflow_dispatch" in str(probe_job["if"])
    names = [str(step.get("name") or step.get("uses") or "") for step in probe_job["steps"]]
    for forbidden in ("Commit", "Promote", "Reduce", "Fetch the grid", "cache"):
        assert not any(forbidden.lower() in n.lower() for n in names), forbidden
    # And the producer stands aside while a probe runs, or one dispatch would
    # both ask the question and rebuild the feed.
    refresh_if = str(doc["jobs"]["refresh"]["if"])
    assert "inputs.probe" in refresh_if
    # NEGATED, not merely mentioning the input: a condition that referenced
    # `probe` without inverting it would read fine and run the producer on
    # every probe dispatch.
    assert "!(" in refresh_if, refresh_if


# ---------------------------------------------------------------------------
# describe mode: ask the CDS's own schema, because the refusal named nothing
# ---------------------------------------------------------------------------
#
# Probe mode ran on 2026-09-11 (run 34587745561) and came back empty: the CDS
# answered "Request has not produced a valid combination of values, please
# check your selection." with an echo of the request and no list of accepted
# values. This dataset's validator complains about the whole COMBINATION
# rather than about one key, so a deliberately wrong enum names no enum.
#
# So ask the service for its own schema. Two routes, both reads: the
# `/constraints` endpoint the web form itself calls, and the process
# description's input schemas. Both reached through the client this producer
# already builds, so there is no hand-rolled `PRIVATE-TOKEN` header to get
# wrong.


#: The body the live refusal probe actually returned, kept verbatim so the
#: test that says "this is why describe mode exists" rests on evidence.
_REAL_COMBINATION_REFUSAL = (
    "400 Client Error: Bad Request for url: "
    "https://cds.climate.copernicus.eu/api/retrieve/v1/processes/"
    "derived-drought-historical-monthly/execution\n"
    "invalid request\n"
    "Request has not produced a valid combination of values, please check "
    "your selection.\n"
    "{'accumulation_period': ['3'], 'data_format': 'netcdf', "
    "'dataset_type': 'pythia_probe_not_a_real_dataset_type', 'month': ['01'], "
    "'product_type': ['reanalysis'], 'variable': "
    "['standardised_precipitation_evapotranspiration_index'], "
    "'version': '1_0', 'year': ['2016']}"
)


class _DescribingClient:
    """A CDS client that answers the two read routes, and records the asking.

    Shaped like the real one: `cdsapi.Client()` returns ecmwf-datastores'
    LegacyClient, which keeps the datastores client on `.client`.
    """

    def __init__(self, constraints=None, process=None, constraints_error=None,
                 process_error=None):
        self._constraints = constraints if constraints is not None else {}
        self._process = process if process is not None else {}
        self._constraints_error = constraints_error
        self._process_error = process_error
        self.constraint_calls = []
        self.process_calls = []
        self.retrieved = []
        self.client = self

    def apply_constraints(self, dataset, request):
        self.constraint_calls.append((dataset, dict(request)))
        if self._constraints_error is not None:
            raise self._constraints_error
        if isinstance(self._constraints, list):
            return self._constraints[len(self.constraint_calls) - 1]
        return self._constraints

    def get_process(self, dataset):
        self.process_calls.append(dataset)
        if self._process_error is not None:
            raise self._process_error
        payload = self._process
        return type("_P", (), {"json": payload})()

    def retrieve(self, dataset, request, target):  # pragma: no cover
        self.retrieved.append((dataset, dict(request), target))
        raise AssertionError("describe mode must never retrieve anything")


def test_the_refusal_the_live_probe_got_names_nothing():
    """The finding that made this route necessary, pinned as evidence.

    If a future CDS release starts naming its values in that body, this test
    fails and the cheaper route is available again.
    """

    assert spei.parse_probe_refusal(_REAL_COMBINATION_REFUSAL) == []


def test_the_constraints_route_answers_the_question():
    """The documented shape: a flat mapping of key to the values still valid."""

    client = _DescribingClient(constraints={
        "dataset_type": ["consolidated_dataset", "intermediate_dataset"],
        "product_type": ["reanalysis", "ensemble_members"],
    })
    result = spei.describe_enum_values(client=client)
    assert result.settled is True
    assert result.values == ["consolidated_dataset", "intermediate_dataset"]
    assert "constraints" in result.source_route


def test_the_constraints_route_is_asked_with_our_request_and_with_nothing():
    """The pair is the point: one answers "what may this key be in the request
    we send", the other "what may it be at all"."""

    client = _DescribingClient(constraints=[
        {"dataset_type": ["consolidated_dataset"]},
        {"dataset_type": ["consolidated_dataset", "intermediate_dataset"]},
    ])
    result = spei.describe_enum_values(client=client)
    # The FIRST TWO calls are the pair. Later ones are the narrowing that
    # fires because this fixture puts a value in the only-unconstrained
    # position, which is its own test below.
    assert len(client.constraint_calls) >= 2
    _dataset, ours = client.constraint_calls[0]
    assert "dataset_type" not in ours, (
        "the key being asked about must not be pinned, or the answer is just "
        "the value we sent"
    )
    live = spei.cds_request("2016", ["01"])
    for key, value in live.items():
        if key == "dataset_type" or key in spei._CONSTRAINTS_NON_FORM_KEYS:
            continue
        assert ours[key] == value, key
    _dataset, unconstrained = client.constraint_calls[1]
    assert unconstrained == {}


def test_the_constrained_call_drops_the_delivery_keys_and_nothing_else():
    """`/constraints` answers about the form's fields and refuses the rest.

    Run 34591286383 sent the whole request and got "422 ... invalid parameter
    / invalid param 'data_format'", so that half of the pair never answered —
    and `only_unconstrained` then came back empty for want of an answer rather
    than because there was nothing to report, which is the worse failure of
    the two. The SELECTION keys must all survive, or the question stops being
    "is this value valid alongside what we ask for".
    """

    client = _DescribingClient(constraints={"dataset_type": ["consolidated_dataset"]})
    spei.describe_enum_values(client=client)
    _dataset, ours = client.constraint_calls[0]
    for delivery in spei._CONSTRAINTS_NON_FORM_KEYS:
        assert delivery not in ours, delivery
    for selection in (
        "variable", "accumulation_period", "product_type", "version",
        "year", "month",
    ):
        assert selection in ours, selection


def test_a_value_valid_only_unconstrained_is_reported_as_its_own_finding():
    """"The release exists but not with the rest of our request" and "the
    release does not exist" want different repairs, so they are not one
    answer."""

    client = _DescribingClient(constraints=[
        {"dataset_type": ["consolidated_dataset"]},
        {"dataset_type": ["consolidated_dataset", "intermediate_dataset"]},
    ])
    result = spei.describe_enum_values(client=client)
    assert result.values == ["consolidated_dataset"]
    assert result.unconstrained == [
        "consolidated_dataset", "intermediate_dataset",
    ]
    assert result.only_unconstrained == ["intermediate_dataset"]
    assert "NOT alongside" in result.detail


def test_the_process_description_answers_when_constraints_does_not():
    """A failed route is a finding, not the end of the asking."""

    client = _DescribingClient(
        constraints_error=RuntimeError("404 Client Error"),
        process={"inputs": {"dataset_type": {"schema_": {
            "enum": ["consolidated_dataset", "intermediate_dataset"],
        }}}},
    )
    result = spei.describe_enum_values(client=client)
    assert result.settled is True
    assert result.values == ["consolidated_dataset", "intermediate_dataset"]
    assert result.source_route == "process description"
    failed = [r for r in result.routes if r.error]
    assert len(failed) == 2, "both constraints attempts should record failure"
    assert "404" in failed[0].error


@pytest.mark.parametrize("payload", [
    {"inputs": {"dataset_type": {"schema_": {"enum": ["a_one", "b_two"]}}}},
    {"inputs": {"dataset_type": {"schema": {"enum": ["a_one", "b_two"]}}}},
    {"inputs": {"dataset_type": {"enum": ["a_one", "b_two"]}}},
    {"inputs": {"dataset_type": {"schema_": {"oneOf": [
        {"const": "a_one"}, {"const": "b_two"},
    ]}}}},
    {"inputs": {"dataset_type": {"schema_": {"type": "array", "items": {
        "enum": ["a_one", "b_two"],
    }}}}},
])
def test_the_process_description_reader_walks_for_the_key_not_a_path(payload):
    """The CDS spells the schema key `schema_` in places and `schema` in
    others, and wraps an array's values under `items`. Pinning one path would
    answer "absent" for a value sitting right there, which is the
    `PRAGMA table_info` mistake in another costume."""

    assert spei.parse_process_description(payload, "dataset_type") == [
        "a_one", "b_two",
    ]


def test_the_readers_report_nothing_rather_than_inventing_something():
    """An empty answer is the outcome that stops the switch. It has to be
    returnable."""

    assert spei.parse_constraints_payload({"product_type": ["x"]}, "dataset_type") == []
    assert spei.parse_process_description({"inputs": {}}, "dataset_type") == []
    assert spei.parse_constraints_payload(None, "dataset_type") == []
    assert spei.parse_process_description("not json", "dataset_type") == []


def test_no_route_naming_values_is_reported_and_never_guessed():
    """The whole discipline of this work order in one assertion."""

    client = _DescribingClient(constraints={}, process={})
    result = spei.describe_enum_values(client=client)
    assert result.settled is False
    assert result.values == []
    assert "Do not guess" in result.detail
    assert "no route named the accepted values" in result.detail


def test_every_route_is_recorded_including_the_ones_that_failed():
    """"Which way of asking failed" is most of the diagnosis when the answer
    does not arrive."""

    client = _DescribingClient(
        constraints_error=RuntimeError("403 Forbidden"),
        process_error=RuntimeError("500 Server Error"),
    )
    result = spei.describe_enum_values(client=client)
    assert [r.route for r in result.routes] == [
        "constraints(our request minus dataset_type)",
        "constraints(nothing fixed)",
        "process description",
    ]
    assert all(r.error for r in result.routes)
    assert "403" in result.detail and "500" in result.detail


def test_describe_mode_never_retrieves_anything():
    """A read is a read. The one method that downloads is the one it must not
    call — on this dataset a retrieve is tens of megabytes."""

    client = _DescribingClient(constraints={"dataset_type": ["consolidated_dataset"]})
    spei.describe_enum_values(client=client)
    assert client.retrieved == []


def test_a_renamed_borrow_says_so_rather_than_failing_obscurely():
    """`cdsapi.Client()` hands back a wrapper and the real client sits on
    `.client`. Borrowed on purpose — hand-rolling the request means
    hand-rolling the PRIVATE-TOKEN header — so a borrow that disappears has to
    name itself, exactly as the GDACS connector's helpers do."""

    class _Stranger:
        pass

    result = spei.describe_enum_values(client=_Stranger())
    assert result.settled is False
    assert "apply_constraints" in result.detail
    assert "re-pointing" in result.detail
    assert result.routes and result.routes[0].route == "client"


def test_describe_mode_writes_no_feed_and_no_status(tmp_path, monkeypatch):
    """It asks a question. A question that rewrote the feed would be a
    producer."""

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        spei, "describe_enum_values",
        lambda key="dataset_type", **kw: spei.DescribeResult(key=key),
    )
    report = tmp_path / "describe.json"
    rc = spei.main(["describe", "--report-out", str(report)])
    assert rc == 0
    written = {p.name for p in tmp_path.rglob("*") if p.is_file()}
    assert written == {"describe.json"}, written


def test_describe_mode_always_exits_zero_because_a_question_is_not_a_fault(
    tmp_path, monkeypatch,
):
    """A probe that goes red is a probe somebody stops running."""

    monkeypatch.setattr(
        spei, "describe_enum_values",
        lambda key="dataset_type", **kw: spei.DescribeResult(
            key=key, detail="no route named the accepted values",
        ),
    )
    assert spei.main(["describe", "--report-out", str(tmp_path / "r.json")]) == 0


def test_the_workflow_asks_both_questions_in_the_probe_job_only():
    """One checkbox, two questions, and neither of them touching the feed.

    The describe route is what can answer; the refusal probe is what proves
    the request shape is right. Both belong to the dispatch-only job.
    """

    import yaml

    doc = yaml.safe_load(
        (spei.REPO_ROOT / ".github" / "workflows" / "spei3_refresh.yml")
        .read_text("utf-8")
    )
    probe_steps = yaml.dump(doc["jobs"]["probe"]["steps"])
    assert "build_spei3_country_means describe" in probe_steps
    assert "build_spei3_country_means probe" in probe_steps
    refresh_steps = yaml.dump(doc["jobs"]["refresh"]["steps"])
    for asking in ("means describe", "means probe"):
        assert asking not in refresh_steps, (
            f"the producer job runs `{asking}`; asking a question and building "
            "a feed are different jobs for a reason"
        )


def test_the_probe_job_reports_even_when_a_route_fails():
    """Both asking steps write their own report and the summary renders
    whichever arrived. Without `always()` a failed first question would take
    the second one and the summary with it — and a probe whose output depends
    on nothing going wrong is not a diagnostic."""

    import yaml

    doc = yaml.safe_load(
        (spei.REPO_ROOT / ".github" / "workflows" / "spei3_refresh.yml")
        .read_text("utf-8")
    )
    steps = {s.get("name"): s for s in doc["jobs"]["probe"]["steps"] if s.get("name")}
    assert "always()" in str(steps["Probe dataset_type"].get("if"))
    assert "always()" in str(steps["Report what the CDS said"].get("if"))
    assert "always()" in str(steps["Upload the probe report"].get("if"))


def test_a_value_valid_only_unconstrained_is_narrowed_to_the_key_that_blocks_it():
    """Run 34591513359 put `intermediate_dataset` in exactly this position:
    allowed in general, refused alongside the rest of our request. "Something
    excludes it" is not a repair; the key's name is, and it is the difference
    between a switch that is one constant and one that has to vary per year.
    So the probe asks, one read per key, rather than reasoning about it.

    The fixture is shaped on what run 34591818890 actually found: `year` is
    the key, which says the intermediate release covers only the recent end of
    the record.
    """

    class _Selective:
        """Admits the value only when `year` is absent."""

        def __init__(self):
            self.client = self
            self.calls = []

        def apply_constraints(self, dataset, request):
            self.calls.append(dict(request))
            if not request:
                return {"dataset_type": ["consolidated_dataset", "intermediate_dataset"]}
            if "year" in request:
                return {"dataset_type": ["consolidated_dataset"]}
            return {"dataset_type": ["consolidated_dataset", "intermediate_dataset"]}

        def get_process(self, dataset):
            return type("_P", (), {"json": {}})()

    client = _Selective()
    result = spei.describe_enum_values(client=client)
    assert result.only_unconstrained == ["intermediate_dataset"]
    assert result.blocked_by == {"intermediate_dataset": ["year"]}
    assert "Dropping year admits 'intermediate_dataset'" in result.detail
    assert "vary that rather than replace one constant" in result.detail


def test_an_exclusion_no_single_key_explains_says_so():
    """A combination of two keys is a different repair from one, and naming
    the wrong one is worse than naming none."""

    class _Stubborn:
        def __init__(self):
            self.client = self

        def apply_constraints(self, dataset, request):
            if not request:
                return {"dataset_type": ["consolidated_dataset", "intermediate_dataset"]}
            return {"dataset_type": ["consolidated_dataset"]}

        def get_process(self, dataset):
            return type("_P", (), {"json": {}})()

    result = spei.describe_enum_values(client=_Stubborn())
    assert result.only_unconstrained == ["intermediate_dataset"]
    assert result.blocked_by == {}
    assert "No single key of ours explains" in result.detail
    assert "combination" in result.detail


def test_narrowing_costs_one_read_per_key_and_never_retrieves():
    """It is a read, and a bounded one: one call per key of ours, no more."""

    class _Counting:
        def __init__(self):
            self.client = self
            self.n = 0

        def apply_constraints(self, dataset, request):
            self.n += 1
            if not request:
                return {"dataset_type": ["consolidated_dataset", "intermediate_dataset"]}
            return {"dataset_type": ["consolidated_dataset"]}

        def get_process(self, dataset):
            return type("_P", (), {"json": {}})()

        def retrieve(self, *a, **k):  # pragma: no cover
            raise AssertionError("narrowing must never retrieve")

    client = _Counting()
    spei.describe_enum_values(client=client)
    ours = {
        k for k in spei.cds_request("2016", ["01"])
        if k != "dataset_type" and k not in spei._CONSTRAINTS_NON_FORM_KEYS
    }
    # Two pair calls, plus one per key of ours.
    assert client.n == 2 + len(ours)


def test_narrowing_is_skipped_when_nothing_needs_it():
    """The ordinary case must cost nothing extra."""

    class _Agreeing:
        def __init__(self):
            self.client = self
            self.n = 0

        def apply_constraints(self, dataset, request):
            self.n += 1
            return {"dataset_type": ["consolidated_dataset", "intermediate_dataset"]}

        def get_process(self, dataset):
            return type("_P", (), {"json": {}})()

    client = _Agreeing()
    result = spei.describe_enum_values(client=client)
    assert result.only_unconstrained == []
    assert result.blocked_by == {}
    assert client.n == 2


# ---------------------------------------------------------------------------
# The switch: which release a month is asked from, and what that costs
#
# `describe` mode settled the literal (`intermediate_dataset`, named by the
# constraints endpoint AND the process description in run 34591286383) and
# then settled the shape of the switch: with the rest of our request fixed
# the endpoint names `consolidated_dataset` ALONE, and run 34591818890
# narrowed the exclusion to `year`. So the intermediate release covers only
# the recent end of the record, and `dataset_type` is a decision per month
# rather than one constant.
# ---------------------------------------------------------------------------


def test_the_consolidated_release_is_asked_for_the_settled_end_of_the_record():
    """And the intermediate one only for what it cannot yet hold.

    The boundary is the consolidated release's documented worst-case lag,
    measured against the live feed on 2026-09-10: newest month served
    2026-05 against a previous complete month of 2026-08.
    """

    assert spei.dataset_type_for("2016-01", today=TODAY) == spei.CDS_DATASET_TYPE_CONSOLIDATED
    assert spei.dataset_type_for("2026-05", today=TODAY) == spei.CDS_DATASET_TYPE_CONSOLIDATED
    assert spei.dataset_type_for("2026-06", today=TODAY) == spei.CDS_DATASET_TYPE_INTERMEDIATE
    assert spei.dataset_type_for("2026-07", today=TODAY) == spei.CDS_DATASET_TYPE_INTERMEDIATE


def test_the_boundary_is_derived_from_the_lag_and_not_a_literal_month():
    """A hardcoded cutover is a switch that silently stops switching."""

    for today, boundary in (
        (dt.date(2026, 9, 12), "2026-05"),
        (dt.date(2027, 1, 3), "2026-09"),
        (dt.date(2027, 3, 31), "2026-11"),
    ):
        assert spei.dataset_type_for(boundary, today=today) == spei.CDS_DATASET_TYPE_CONSOLIDATED
        after = spei.shift_months(boundary, 1)
        assert spei.dataset_type_for(after, today=today) == spei.CDS_DATASET_TYPE_INTERMEDIATE


def test_the_request_names_the_release_it_was_asked_for():
    """Everything else about the two requests is identical, and must be.

    One enum separates them, which is exactly why the release-agreement gate
    exists: a wrong literal the CDS happened to accept would arrive looking
    like data rather than like an error.
    """

    consolidated = spei.cds_request(
        "2019", ["01"], dataset_type=spei.CDS_DATASET_TYPE_CONSOLIDATED
    )
    intermediate = spei.cds_request(
        "2026", ["07"], dataset_type=spei.CDS_DATASET_TYPE_INTERMEDIATE
    )
    assert consolidated["dataset_type"] == "consolidated_dataset"
    assert intermediate["dataset_type"] == "intermediate_dataset"
    assert {
        k: v for k, v in consolidated.items() if k not in ("dataset_type", "year", "month")
    } == {
        k: v for k, v in intermediate.items() if k not in ("dataset_type", "year", "month")
    }


def test_a_month_is_asked_from_one_release_and_never_both(tmp_path):
    """Two requests for one month is two CDS jobs for the same numbers."""

    asked: dict[str, list[str]] = {}

    class _Client:
        def retrieve(self, dataset, request, target):
            for month in request["month"]:
                asked.setdefault(f"{request['year'][0]}-{month}", []).append(
                    request["dataset_type"]
                )
            Path(target).write_bytes(b"nc")

    months = [f"2026-{m:02d}" for m in range(1, 9)]
    spei.fetch_grids(tmp_path, months, client=_Client(), today=TODAY)
    assert sorted(asked) == months
    assert all(len(v) == 1 for v in asked.values())


# ---------------------------------------------------------------------------
# Provenance, and the upgrade it makes self-healing
# ---------------------------------------------------------------------------


def _inside(country, lon, lat) -> bool:
    minx, miny, maxx, maxy = country.bounds
    return minx <= lon <= maxx and miny <= lat <= maxy


def test_every_reduced_row_carries_the_release_it_came_from():
    rows, _ = spei.reduce_grids(
        {
            "2026-03": spei.Grid(
                lats=[0.0], lons=[0.0], values=[[0.5]],
                dataset_type=spei.CDS_DATASET_TYPE_CONSOLIDATED,
            ),
            "2026-07": spei.Grid(
                lats=[0.0], lons=[0.0], values=[[0.5]],
                dataset_type=spei.CDS_DATASET_TYPE_INTERMEDIATE,
            ),
        },
        {"AAA": _Box()},
        iso3s=["AAA"],
        contains=_inside,
    )
    assert {r["ym"]: r["dataset_type"] for r in rows} == {
        "2026-03": spei.CDS_DATASET_TYPE_CONSOLIDATED,
        "2026-07": spei.CDS_DATASET_TYPE_INTERMEDIATE,
    }


def test_a_row_written_before_the_column_existed_reads_as_consolidated():
    """A true statement about this repository's history, not a guess.

    The producer had requested exactly one release before the switch, so a
    blank can only have come from it. The committed rows were migrated in
    the same commit; this covers a hand edit or an older tool.
    """

    assert spei.row_dataset_type({"dataset_type": ""}) == spei.CDS_DATASET_TYPE_CONSOLIDATED
    assert spei.row_dataset_type({}) == spei.CDS_DATASET_TYPE_CONSOLIDATED
    assert (
        spei.row_dataset_type({"dataset_type": "intermediate_dataset"})
        == spei.CDS_DATASET_TYPE_INTERMEDIATE
    )


def test_a_month_still_on_the_intermediate_release_is_asked_for_again():
    """The upgrade, and it is a fact about the file rather than a bet.

    An intermediate value is documented as experimental and subject to
    change. Leaving one in place once the consolidated version exists makes
    a provisional number permanent, and the trailing revision window is the
    wrong instrument for catching that: the upstream states its own lag as a
    RANGE, so a window is a guess about how long it will take.
    """

    held = [f"2026-{m:02d}" for m in range(1, 8)]
    window = spei.plan_window(
        held,
        start_ym="2026-01",
        end_ym="2026-07",
        revision_months=0,
        intermediate_months=["2026-04", "2026-06", "2026-07"],
        today=TODAY,
    )
    # 2026-04 can be upgraded; 2026-06 and 2026-07 are newer than the
    # consolidated release reaches, so they stay provisional this cycle.
    assert window.upgradeable_months == ["2026-04"]
    assert window.months == ["2026-04"]


def test_the_upgrade_does_not_depend_on_the_revision_window():
    """Which is the whole reason the column is on the row.

    A month that fell out of the window while still provisional is still
    upgraded, however long ago it was written.
    """

    window = spei.plan_window(
        [f"2025-{m:02d}" for m in range(1, 13)],
        start_ym="2025-01",
        end_ym="2025-12",
        revision_months=0,
        intermediate_months=["2025-02"],
        today=TODAY,
    )
    assert window.upgradeable_months == ["2025-02"]


def test_the_revision_window_outlives_the_gap_between_the_releases():
    """Six months, against a two-month gap plus room for missed cycles."""

    gap = spei.CONSOLIDATED_LAG_MONTHS - spei.INTERMEDIATE_LAG_MONTHS
    assert gap == 2
    assert spei.REVISION_WINDOW_MONTHS > gap + 1


def test_the_consolidated_grid_wins_where_both_releases_hold_a_month():
    """Which the cache makes routine rather than hypothetical.

    A month asked for as intermediate in September is asked for as
    consolidated in December, and the trailing years are kept between runs.
    """

    assert spei._supersedes(
        spei.CDS_DATASET_TYPE_CONSOLIDATED, spei.CDS_DATASET_TYPE_INTERMEDIATE
    )
    assert not spei._supersedes(
        spei.CDS_DATASET_TYPE_INTERMEDIATE, spei.CDS_DATASET_TYPE_CONSOLIDATED
    )


def test_the_release_is_read_off_the_filename():
    assert spei.dataset_type_of_file(
        Path("spei3_2026_intermediate__data.nc")
    ) == spei.CDS_DATASET_TYPE_INTERMEDIATE
    assert spei.dataset_type_of_file(
        Path("spei3_2026_consolidated.nc")
    ) == spei.CDS_DATASET_TYPE_CONSOLIDATED
    # Untagged: written before the producer asked for more than one release.
    assert spei.dataset_type_of_file(
        Path("spei3_2016__data.nc")
    ) == spei.CDS_DATASET_TYPE_CONSOLIDATED


# ---------------------------------------------------------------------------
# The gate the switch ships with
#
# Everything else about the two requests is identical, so the failure worth
# guarding against is narrow: the enum naming a different quantity. It would
# show up as a shifted mean or a rescaled spread, not as an error.
# ---------------------------------------------------------------------------


#: What the 29,625-row consolidated backfill of 2026-09-10 measured. Kept as
#: data so a future change to the band has to argue with the evidence.
BACKFILL_MEAN_AND_SD = (-0.086, 0.969)


def _release_rows(release: str, values):
    return [
        {
            "iso3": f"C{i:02d}", "ym": "2026-07", "value": f"{v:.4f}",
            "coverage": "cells", "n_cells": "10", "dataset_type": release,
        }
        for i, v in enumerate(values)
    ]


def test_the_committed_feed_matches_the_backfill_it_was_measured_from():
    """The reference the gate compares against, checked against the record."""

    rows = spei.read_csv_rows(
        Path(__file__).resolve().parents[2] / "resolver" / "data"
        / "spei3_country_means.csv"
    )
    assert len(rows) == 29625
    mean, sd = spei._mean_and_sd([float(r["value"]) for r in rows])
    expected_mean, expected_sd = BACKFILL_MEAN_AND_SD
    assert abs(mean - expected_mean) < 0.01
    assert abs(sd - expected_sd) < 0.01


def test_two_releases_that_agree_pass_the_gate():
    import random

    rng = random.Random(11)
    rows = _release_rows(
        spei.CDS_DATASET_TYPE_CONSOLIDATED,
        [rng.gauss(-0.086, 0.969) for _ in range(400)],
    ) + _release_rows(
        spei.CDS_DATASET_TYPE_INTERMEDIATE,
        [rng.gauss(-0.086, 0.969) for _ in range(200)],
    )
    report = spei.compare_releases(rows)
    assert report["verdict"] == "ok"
    assert report["intermediate_rows"] == 200


def test_a_release_serving_a_different_quantity_fails_the_gate():
    """A shifted mean and a rescaled spread are the two shapes of that fault."""

    import random

    rng = random.Random(11)
    consolidated = _release_rows(
        spei.CDS_DATASET_TYPE_CONSOLIDATED,
        [rng.gauss(-0.086, 0.969) for _ in range(400)],
    )
    shifted = consolidated + _release_rows(
        spei.CDS_DATASET_TYPE_INTERMEDIATE,
        [rng.gauss(2.5, 0.969) for _ in range(200)],
    )
    rescaled = consolidated + _release_rows(
        spei.CDS_DATASET_TYPE_INTERMEDIATE,
        [rng.gauss(-0.086, 0.05) for _ in range(200)],
    )
    assert spei.compare_releases(shifted)["verdict"] == "fail"
    assert spei.compare_releases(rescaled)["verdict"] == "fail"

    result = spei.validate_candidate(shifted, [])
    assert result.ok is False
    assert any("intermediate release" in f for f in result.failures)


def test_a_dry_month_does_not_trip_the_gate():
    """The band has to survive weather, or it is a gate nobody keeps on.

    A genuinely dry global month moves the mean by a few tenths. A whole
    sigma is not weather.
    """

    import random

    rng = random.Random(3)
    rows = _release_rows(
        spei.CDS_DATASET_TYPE_CONSOLIDATED,
        [rng.gauss(-0.086, 0.969) for _ in range(400)],
    ) + _release_rows(
        spei.CDS_DATASET_TYPE_INTERMEDIATE,
        [rng.gauss(-0.55, 1.05) for _ in range(200)],
    )
    assert spei.compare_releases(rows)["verdict"] == "ok"


def test_a_comparison_that_cannot_run_is_skipped_with_a_reason():
    """Never passed silently. This is the gate guarding its own change."""

    only_consolidated = _release_rows(
        spei.CDS_DATASET_TYPE_CONSOLIDATED, [0.1] * 400
    )
    report = spei.compare_releases(only_consolidated)
    assert report["verdict"] == "skipped"
    assert "no rows from the intermediate release" in report["reason"]

    thin = only_consolidated + _release_rows(
        spei.CDS_DATASET_TYPE_INTERMEDIATE, [0.1] * 5
    )
    report = spei.compare_releases(thin)
    assert report["verdict"] == "skipped"
    assert "below" in report["reason"]
    # And a skip never fails the candidate.
    assert spei.validate_candidate(thin, []).ok is True


def test_the_comparison_is_reported_whether_or_not_it_decided_anything():
    """The numbers are how a reader sees the switch landed."""

    rows = _release_rows(spei.CDS_DATASET_TYPE_CONSOLIDATED, [0.1, 0.2, 0.3])
    result = spei.validate_candidate(rows, [])
    assert result.release_comparison["consolidated_rows"] == 3
    assert "consolidated_mean" in result.release_comparison
    assert "release_comparison" in result.as_dict()


def test_a_cached_unit_that_does_not_cover_the_window_is_re_asked(tmp_path):
    """The hole the switch made reachable, closed.

    A unit's month-set is not fixed. A year at the boundary splits between
    the releases, and an operator re-dispatching with different ``from_ym``
    / ``to_ym`` changes it too. Skipping on the presence of a file alone
    would leave a month silently never fetched, recorded absent with a
    stated reason, on a green run — and the next run would skip it again for
    exactly the same reason.
    """

    asked = []

    class _Client:
        def retrieve(self, dataset, request, target):
            asked.append(sorted(request["month"]))
            Path(target).write_bytes(b"CDF\x01nc")

    first = spei.fetch_grids(
        tmp_path, ["2026-06", "2026-07"], client=_Client(), today=TODAY,
    )
    assert first.written == ["2026 (intermediate)"]
    assert asked == [["06", "07"]]

    # A later run wants a month the cached unit does not hold.
    second = spei.fetch_grids(
        tmp_path, ["2026-06", "2026-07", "2026-08"], client=_Client(), today=TODAY,
    )
    assert asked[-1] == ["06", "07", "08"]
    assert second.written == ["2026 (intermediate)"]
    assert second.skipped_present == []


def test_a_cached_unit_that_covers_the_window_is_not_re_asked(tmp_path):
    """Which is the whole point of the cache, and the case it is for."""

    class _Writing:
        def retrieve(self, dataset, request, target):
            Path(target).write_bytes(b"CDF\x01nc")

    spei.fetch_grids(tmp_path, ["2026-06", "2026-07"], client=_Writing(), today=TODAY)

    class _Refusing:
        def retrieve(self, dataset, request, target):  # pragma: no cover
            raise AssertionError("re-fetched a unit that already covers the window")

    result = spei.fetch_grids(
        tmp_path, ["2026-07"], client=_Refusing(), today=TODAY,
    )
    assert result.skipped_present == ["2026 (intermediate)"]


def test_a_cached_unit_from_before_the_manifest_answers_as_it_used_to(tmp_path):
    """A cache is far likelier to be complete than worth 95 MB on suspicion."""

    (tmp_path / "spei3_2016_consolidated__data.nc").write_bytes(b"CDF\x01held")

    class _Refusing:
        def retrieve(self, dataset, request, target):  # pragma: no cover
            raise AssertionError("re-fetched a unit cached before the manifest")

    result = spei.fetch_grids(
        tmp_path, ["2016-01", "2016-02"], client=_Refusing(), today=TODAY,
    )
    assert result.skipped_present == ["2016 (consolidated)"]
