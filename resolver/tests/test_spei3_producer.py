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

import datetime as dt
import json
from pathlib import Path

import pytest

from scripts import build_spei3_country_means as spei

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "spei3"


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

    # `format` became `data_format`, and the archive has to be unpacked.
    assert request["data_format"] == "netcdf"
    assert "format" not in request
    assert request["download_format"] == "unarchived"


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
        tmp_path, [f"2026-{m:02d}" for m in range(1, 13)], client=client,
    )
    assert len(client.requests) == 1
    assert client.requests[0][1]["month"] == [f"{m:02d}" for m in range(1, 13)]


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

    (tmp_path / "spei3_2016.nc").write_bytes(b"already here")

    class _Client:
        def retrieve(self, dataset, request, target):  # pragma: no cover
            raise AssertionError("re-fetched a year already on disk")

    result = spei.fetch_grids(tmp_path, ["2016-01"], client=_Client())
    assert result.skipped_present == ["2016"]
    assert result.written == []


def test_a_failed_year_is_owed_not_fatal(tmp_path):
    """The other years may be fine, and the gates decide what is publishable."""

    class _Client:
        def retrieve(self, dataset, request, target):
            if request["year"] == ["2017"]:
                raise RuntimeError("invalid request / no valid combination")
            Path(target).write_bytes(b"nc")

    result = spei.fetch_grids(tmp_path, ["2016-01", "2017-01"], client=_Client())
    assert result.written == ["2016"]
    assert "2017" in result.failed
    assert "no valid combination" in result.failed["2017"]
    assert result.complete is False


def test_a_partial_file_from_a_failed_year_is_removed(tmp_path):
    """A truncated download must not be mistaken for a year already held."""

    class _Client:
        def retrieve(self, dataset, request, target):
            Path(target).write_bytes(b"half a file")
            raise RuntimeError("connection reset")

    spei.fetch_grids(tmp_path, ["2016-01"], client=_Client())
    assert not (tmp_path / "spei3_2016.nc").exists()


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
