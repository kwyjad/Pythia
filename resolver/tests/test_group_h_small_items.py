# Pythia / Copyright (c) 2025 Kevin Wyjad
"""Group H of the run-33841370196 repairs: the small items.

One test per fault, each asserting the row or file the next Resolver
Update leaves behind rather than the log line it prints:

* the bundle's ACLED coverage query names the column the table has;
* ``run/step_timings.csv`` is filled from the jobs API;
* ``run/git.json`` says what the runner checked out, and untracked run
  outputs are not "dirty";
* a GDACS event that names no country is placed from its position in the
  CONNECTOR path, as the PA machine already did;
* an extraction answer cut off at the output ceiling is re-read once at a
  doubled budget and the ledger says which happened;
* a budget-bound BACKCAST cell is deferred, never NO_DATA, the resume
  ledger owes it, and the run summary says so.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from resolver.db import duckdb_io  # noqa: E402
from resolver.hazard_resolution import backcast as bc  # noqa: E402
from resolver.hazard_resolution import cell_ledger  # noqa: E402
from resolver.hazard_resolution import extract as extract_mod  # noqa: E402
from resolver.hazard_resolution import impact as impact_mod  # noqa: E402
from resolver.hazard_resolution.detect import (  # noqa: E402
    CountryTrigger,
    DetectionResult,
    write_triggers,
)
from resolver.hazard_resolution.schema import RUN_TYPE_BACKCAST, ensure_haz_schema  # noqa: E402
from resolver.tests.hazard_resolution_utils import (  # noqa: E402
    SYNTHETIC_COUNTRIES_GEOJSON,
    golden_documents,
    make_rulebook,
    seed_reliefweb_docs,
)
from scripts import build_resolver_debug_bundle as bundle  # noqa: E402

TODAY = dt.date(2026, 9, 5)
ISO3, YM, HAZARD = "PHL", "2024-03", "FL"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _builder(tmp_path: Path, db: Path | None = None, environ: dict | None = None):
    return bundle.BundleBuilder(
        out_path=tmp_path / "b.zip",
        db_path=db,
        diagnostics_dir=tmp_path / "diag",
        run_log_dir=None,
        staging=tmp_path / "staging",
        max_bytes=bundle.DEFAULT_MAX_BYTES,
        environ=environ or {},
    )


def _resolver_db(path: Path) -> None:
    con = duckdb.connect(str(path))
    duckdb_io.init_schema(con)
    con.execute(
        "INSERT INTO acled_monthly_fatalities (iso3, month, fatalities, source, updated_at) "
        "VALUES ('SOM', DATE '2026-06-01', 10, 'acled', now()), "
        "('SOM', DATE '2026-07-01', 12, 'acled', now()), "
        "('KEN', DATE '2026-07-01', 1, 'acled', now())"
    )
    con.close()


@pytest.fixture()
def rulebook():
    return make_rulebook()


@pytest.fixture()
def con():
    connection = duckdb.connect(":memory:")
    ensure_haz_schema(connection)
    return connection


# ---------------------------------------------------------------------------
# H1: MIN(ym) against a table whose column is `month`
# ---------------------------------------------------------------------------


def test_the_acled_coverage_query_names_the_column_the_table_has(tmp_path):
    db = tmp_path / "resolver.duckdb"
    _resolver_db(db)
    builder = _builder(tmp_path, db)
    sql = next(
        sql for name, _needs, sql in builder.DB_QUERIES
        if name == "acled_monthly_fatalities_coverage"
    )
    columns, rows = builder.query(sql)
    assert rows[0][:2] == ("2026-06", "2026-07"), rows
    assert rows[0][2] == 3 and rows[0][3] == 2
    assert not builder.problems, builder.problems


def test_every_bundle_query_is_checked_against_the_real_schema(tmp_path):
    """The check that would have caught MIN(ym): every query whose tables
    exist must run, and a binder error is a FAIL naming the query."""

    db = tmp_path / "resolver.duckdb"
    _resolver_db(db)
    builder = _builder(tmp_path, db)
    builder._check_every_db_query_names_real_columns()
    check = builder.checks[-1]
    assert check["verdict"] == "PASS", check

    # Break one query the way the September bundle had it broken.
    broken = _builder(tmp_path, db)
    broken.DB_QUERIES = tuple(
        (name, needs, sql.replace("strftime(MIN(month), '%Y-%m')", "MIN(ym)"))
        if name == "acled_monthly_fatalities_coverage" else (name, needs, sql)
        for name, needs, sql in builder.DB_QUERIES
    )
    broken._check_every_db_query_names_real_columns()
    check = broken.checks[-1]
    assert check["verdict"] == "FAIL"
    assert "acled_monthly_fatalities_coverage" in check["detail"]


# ---------------------------------------------------------------------------
# H2: step timings from the jobs API
# ---------------------------------------------------------------------------


_JOBS = {
    "jobs": [
        {"id": 1, "name": "Skip if a staged pipeline is in flight", "status": "completed",
         "runner_name": "r-a", "html_url": "https://x/1",
         "steps": [{"number": 1, "name": "gate", "status": "completed",
                    "conclusion": "success", "started_at": "2026-09-04T03:00:00Z",
                    "completed_at": "2026-09-04T03:00:05Z"}]},
        {"id": 2, "name": "Full pipeline backfill", "status": "in_progress",
         "runner_name": "r-b", "html_url": "https://x/2",
         "steps": [
             {"number": 1, "name": "Checkout repository", "status": "completed",
              "conclusion": "success", "started_at": "2026-09-04T03:01:00Z",
              "completed_at": "2026-09-04T03:01:30Z"},
             {"number": 2, "name": "Phase 1: connectors", "status": "completed",
              "conclusion": "success", "started_at": "2026-09-04T03:01:30Z",
              "completed_at": "2026-09-04T03:41:30Z"},
             {"number": 3, "name": "Build resolver debug bundle", "status": "in_progress",
              "conclusion": None, "started_at": "2026-09-04T03:41:31Z",
              "completed_at": None},
             {"number": 4, "name": "Upload diagnostics", "status": "pending",
              "conclusion": None, "started_at": None, "completed_at": None},
         ]},
    ]
}


def test_the_bundle_step_finds_its_own_job_among_several():
    job = bundle._pick_this_job(_JOBS["jobs"], {})
    assert job["name"] == "Full pipeline backfill"
    payload = bundle._step_timings_payload(job, "https://api/jobs")
    by_name = {s["name"]: s for s in payload["steps"]}
    assert by_name["Phase 1: connectors"]["duration_sec"] == 2400.0
    assert by_name["Build resolver debug bundle"]["duration_sec"] is None
    assert by_name["Upload diagnostics"]["status"] == "pending"


def test_step_timings_are_read_from_the_jobs_api_with_the_token(tmp_path, monkeypatch):
    import io
    import urllib.request

    seen: dict = {}

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, timeout=0):
        seen["url"] = request.full_url
        seen["auth"] = request.get_header("Authorization")
        return _Resp(json.dumps(_JOBS).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    env = {
        "GITHUB_RUN_ID": "33841370196", "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_REPOSITORY": "kwyjad/Pythia", "GITHUB_TOKEN": "ghs_test_token",
    }
    builder = _builder(tmp_path, environ=env)
    builder._step_timings()

    assert "/actions/runs/33841370196/attempts/1/jobs" in seen["url"]
    assert seen["auth"] == "Bearer ghs_test_token"
    csv_text = (tmp_path / "staging" / "run" / "step_timings.csv").read_text(encoding="utf-8")
    assert "Phase 1: connectors" in csv_text and "2400.0" in csv_text
    assert "github jobs api" in csv_text
    raw = json.loads((tmp_path / "staging" / "run" / "step_timings.json").read_text())
    assert raw["job_name"] == "Full pipeline backfill"
    # The token never lands in the bundle.
    assert "ghs_test_token" not in csv_text and "ghs_test_token" not in json.dumps(raw)


def test_step_timings_without_a_token_say_so_instead_of_shipping_empty(tmp_path):
    builder = _builder(tmp_path, environ={"GITHUB_RUN_ID": "1", "GITHUB_REPOSITORY": "k/P"})
    builder._step_timings()
    csv_text = (tmp_path / "staging" / "run" / "step_timings.csv").read_text(encoding="utf-8")
    assert "no GITHUB_TOKEN" in csv_text
    assert any("GITHUB_TOKEN" in p for p in builder.problems)


# ---------------------------------------------------------------------------
# H3: git.json tells the truth about the checkout
# ---------------------------------------------------------------------------


def test_git_state_counts_untracked_outputs_apart_from_tracked_changes(monkeypatch):
    head = "0937a4f0000000000000000000000000deadbeef"
    answers = {
        ("rev-parse", "HEAD"): head,
        ("rev-parse", "--abbrev-ref", "HEAD"): "main",
        ("status", "--porcelain"): "?? diagnostics/x.log\n?? data/resolver.duckdb\n",
        ("rev-parse", "--is-shallow-repository"): "false",
    }

    def fake_git(args):
        if args[0] == "log":
            return "0937a4f|2026-09-03|head commit\nabc1234|2026-09-02|older\n"
        return answers.get(tuple(args), "")

    monkeypatch.setattr(bundle, "_run_git", fake_git)
    info = bundle.BundleBuilder._git_state({"GITHUB_SHA": head, "GITHUB_RUN_ID": "1"})
    assert info["dirty"] is False
    assert info["n_untracked_files"] == 2
    assert info["head_matches_github_sha"] is True
    assert info["head_is_first_recent_commit"] is True
    assert info["shallow_clone"] is False

    answers[("status", "--porcelain")] = " M resolver/tools/enrich.py\n?? diag/x\n"
    info = bundle.BundleBuilder._git_state({"GITHUB_SHA": head})
    assert info["dirty"] is True and info["n_tracked_changes"] == 1


def test_the_commit_check_fails_when_head_is_not_the_dispatched_sha(tmp_path):
    builder = _builder(tmp_path, environ={"GITHUB_RUN_ID": "1"})
    builder.git_info = {
        "commit": "aaaa", "github_sha": "bbbb", "dirty": False,
        "n_untracked_files": 300, "tracked_changes": [], "n_tracked_changes": 0,
    }
    builder._check_bundle_records_the_checked_out_commit()
    assert builder.checks[-1]["verdict"] == "FAIL"

    builder.git_info["github_sha"] = "aaaa"
    builder._check_bundle_records_the_checked_out_commit()
    assert builder.checks[-1]["verdict"] == "PASS"
    assert "300 untracked" in builder.checks[-1]["detail"]


# ---------------------------------------------------------------------------
# H4: GDACS connector places a country-less event from its position
# ---------------------------------------------------------------------------


def test_the_connector_places_a_country_less_event_by_geometry(monkeypatch):
    from resolver.connectors.gdacs import GdacsConnector
    from resolver.hazard_resolution.geometry import load_country_geometries

    monkeypatch.setattr(
        "resolver.hazard_resolution.geometry.load_country_geometries",
        lambda *a, **k: load_country_geometries(SYNTHETIC_COUNTRIES_GEOJSON),
    )
    # Squareland (AAA) in the synthetic fixture spans lon 10..15, lat -2..2.
    lat, lon, iso3 = 0.5, 12.0, "AAA"

    connector = GdacsConnector()
    event = {
        "eventtype": "TC", "eventid": "1001273", "iso3": "", "iso3_list": [],
        "country": "", "lat": lat, "lon": lon, "population": 1000.0,
        "fromdate": dt.date(2026, 8, 20), "todate": dt.date(2026, 8, 22),
        "alertlevel": "Orange", "pub_date": dt.date(2026, 8, 22),
    }
    rows = connector._expand_to_country_months([event], {})
    assert rows, "the event was dropped although it had a position"
    assert rows[0]["iso3"] == iso3
    assert event["iso3_from_geometry"] is True
    assert connector.events_resolved_by_geometry == [f"TC-1001273:{iso3}"]
    assert connector.events_without_country == []


def test_an_unplaceable_event_is_still_dropped_and_named(monkeypatch):
    from resolver.connectors.gdacs import GdacsConnector

    monkeypatch.setattr("resolver.connectors.gdacs._geometry_resolver", lambda: None)
    connector = GdacsConnector()
    event = {
        "eventtype": "TC", "eventid": "1001315", "iso3": "", "iso3_list": [],
        "country": "", "lat": None, "lon": None, "population": 0.0,
        "fromdate": dt.date(2026, 8, 20), "todate": dt.date(2026, 8, 22),
        "alertlevel": "Green", "pub_date": dt.date(2026, 8, 22),
    }
    assert connector._expand_to_country_months([event], {}) == []
    assert connector.events_without_country == ["TC-1001315"]


# ---------------------------------------------------------------------------
# H5: an answer cut at the output ceiling is re-read once
# ---------------------------------------------------------------------------


def _truncating_seam(fail_times: int):
    """A seam whose first ``fail_times`` answers are truncated at max_tokens."""

    calls: list[dict] = []
    good = golden_documents()[0]
    response = json.dumps({
        "figures": [{
            "value": 1917, "unit": "people",
            "quote": good["body"][:60],
            "stated_by": "OCHA", "area": "national", "date": "2024-03-10",
            "cumulative_or_new": "cumulative",
        }]
    })

    def call(model_ref, prompt, rulebook, max_output_tokens=None):
        calls.append({"max_output_tokens": max_output_tokens})
        if len(calls) <= fail_times:
            return "", {"prompt_tokens": 5000, "completion_tokens": 2000}, (
                "Anthropic response truncated at max_tokens "
                "(raise PYTHIA_ANTHROPIC_SPD_MAX_TOKENS; adaptive thinking shares this budget)"
            )
        return response, {"prompt_tokens": 5000, "completion_tokens": 900}, ""

    return call, calls


def test_a_truncated_extraction_is_re_read_at_a_doubled_budget(con, rulebook):
    doc = golden_documents()[0]
    seed_reliefweb_docs(con, [doc])
    call, calls = _truncating_seam(fail_times=1)
    budget = extract_mod.ExtractionBudget(max_calls_per_month=100)
    result = extract_mod.extract_for_cell(
        con, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook, budget=budget,
        documents=[doc], call=call, model_ref="anthropic:test",
    )
    assert [c["max_output_tokens"] for c in calls] == [
        None, 2 * int(rulebook.get("extraction.max_output_tokens"))
    ]
    assert result.docs_truncated_retried == 1
    assert result.docs_truncated_after_retry == 0
    assert result.docs_failed == 0
    assert result.calls_made == 2  # both calls billed and counted
    status = con.execute(
        "SELECT status FROM haz_doc_extractions WHERE doc_id = ?", [doc["doc_id"]]
    ).fetchone()[0]
    assert status == "ok"


def test_a_document_too_long_even_at_the_doubled_budget_is_recorded_as_such(con, rulebook):
    doc = golden_documents()[0]
    seed_reliefweb_docs(con, [doc])
    call, calls = _truncating_seam(fail_times=2)
    budget = extract_mod.ExtractionBudget(max_calls_per_month=100)
    result = extract_mod.extract_for_cell(
        con, iso3=ISO3, ym=YM, hazard=HAZARD, rulebook=rulebook, budget=budget,
        documents=[doc], call=call, model_ref="anthropic:test",
    )
    assert len(calls) == 2
    assert result.docs_truncated_after_retry == 1
    assert result.docs_failed == 1
    error = con.execute(
        "SELECT error FROM haz_doc_extractions WHERE doc_id = ?", [doc["doc_id"]]
    ).fetchone()[0]
    assert error.startswith("truncated_after_retry_at_")


def test_the_rulebook_answer_budget_is_no_longer_two_thousand():
    from resolver.hazard_resolution.rulebook import load_rulebook

    assert int(load_rulebook().get("extraction.max_output_tokens")) >= 8000


def test_the_truncation_check_reads_the_ledger(tmp_path):
    db = tmp_path / "resolver.duckdb"
    con = duckdb.connect(str(db))
    ensure_haz_schema(con)
    started = "2026-09-04T03:00:00Z"
    con.execute(
        """
        INSERT INTO haz_doc_extractions
            (doc_id, iso3, year, month, hazard, model, prompt_version, status,
             figures_json, error, created_at)
        VALUES
            ('rw-1', 'CHL', 2026, 7, 'FL', 'm', 'v2', 'error', '[]',
             'Anthropic response truncated at max_tokens', TIMESTAMP '2026-09-04 03:30:00'),
            ('rw-2', 'CHL', 2026, 7, 'FL', 'm', 'v2', 'error', '[]',
             'truncated_after_retry_at_16000_output_tokens: max_tokens',
             TIMESTAMP '2026-09-04 03:31:00')
        """
    )
    con.close()
    builder = _builder(tmp_path, db, environ={"PYTHIA_RUN_STARTED_AT": started})
    builder._check_no_extraction_lost_to_truncation()
    check = builder.checks[-1]
    assert check["verdict"] == "FAIL" and "rw-1" in check["detail"]

    con = duckdb.connect(str(db))
    con.execute("DELETE FROM haz_doc_extractions WHERE doc_id = 'rw-1'")
    con.close()
    builder = _builder(tmp_path, db, environ={"PYTHIA_RUN_STARTED_AT": started})
    builder._check_no_extraction_lost_to_truncation()
    check = builder.checks[-1]
    assert check["verdict"] == "PASS" and "too long" in check["detail"]


# ---------------------------------------------------------------------------
# H6: a budget-bound backcast cell is deferred, never NO_DATA
# ---------------------------------------------------------------------------


def _trigger(con, rulebook, iso3: str, ym: str, run_type: str) -> None:
    result = DetectionResult(
        hazard=HAZARD, ym=ym, coverage_ok=True, coverage_note="test",
        n_points_month=0, n_points_qualifying=0, max_point_time=None,
    )
    result.rows.append(CountryTrigger(iso3=iso3, triggered=True, trigger_source="gdacs"))
    write_triggers(con, result, rulebook, run_type=run_type)


def test_a_backcast_cell_the_budget_cut_short_gets_no_row(con, rulebook):
    seed_reliefweb_docs(con, golden_documents())
    _trigger(con, rulebook, ISO3, YM, RUN_TYPE_BACKCAST)
    exhausted = extract_mod.ExtractionBudget(
        max_calls_per_month=3000, backcast_max_calls_per_month=2000,
        backcast_used_this_month=2000, run_type=RUN_TYPE_BACKCAST,
    )
    assert exhausted.exhausted
    from unittest import mock

    with mock.patch.object(extract_mod, "load_budget", return_value=exhausted):
        run = impact_mod.resolve_triggered_cells(
            con, ym=YM, hazard=HAZARD, iso3s=[ISO3], rulebook=rulebook,
            fetches={"emdat": {"ok": True}, "ifrc_go": {"ok": True},
                     "idmc_idu": {"ok": True}, "gdacs": {"ok": True}},
            today=TODAY, fetch_documents=False, run_type=RUN_TYPE_BACKCAST,
        )
    assert run.deferred_cells == [ISO3]
    assert run.extraction_binding_limit == "backcast share (2000)"
    assert con.execute("SELECT COUNT(*) FROM haz_resolutions").fetchone()[0] == 0
    detail = json.loads(
        con.execute("SELECT trigger_detail_json FROM haz_triggers").fetchone()[0]
    )
    assert detail["no_row_reason"] == cell_ledger.REASON_BUDGET_DEFERRED
    assert "backcast share (2000)" in detail["no_row_note"]

    counts = bc.month_counts(con, HAZARD, YM)
    assert counts["cells_deferred_for_budget"] == 1
    assert counts["deferred_cells"] == [ISO3]
    deferred, reason = bc.month_is_deferred(counts)
    assert deferred and "backcast share (2000)" in reason


def test_a_live_run_never_defers_it_writes_what_it_has(con, rulebook):
    seed_reliefweb_docs(con, golden_documents())
    _trigger(con, rulebook, ISO3, YM, "live")
    exhausted = extract_mod.ExtractionBudget(max_calls_per_month=0)
    from unittest import mock

    with mock.patch.object(extract_mod, "load_budget", return_value=exhausted):
        run = impact_mod.resolve_triggered_cells(
            con, ym=YM, hazard=HAZARD, iso3s=[ISO3], rulebook=rulebook,
            fetches={"emdat": {"ok": True}, "ifrc_go": {"ok": True},
                     "idmc_idu": {"ok": True}, "gdacs": {"ok": True}},
            today=TODAY, fetch_documents=False, run_type="live",
        )
    assert run.deferred_cells == []
    assert run.extraction_budget_capped is True
    assert con.execute("SELECT COUNT(*) FROM haz_resolutions").fetchone()[0] == 1


def test_the_resume_ledger_records_deferred_and_walks_only_the_owed_cells(con, rulebook):
    """Night one defers PHL for budget; night two walks exactly PHL."""

    calls: list[dict] = []

    def runner(*, ym: str, run_type: str, countries_filter=None, dry_run=False, **_):
        calls.append({"ym": ym, "countries": countries_filter})
        if dry_run:
            return 0
        _trigger(con, rulebook, "PHL", ym, run_type)
        if len(calls) == 1:
            # First night: the budget bound before PHL was read.
            from resolver.hazard_resolution import detect as detect_mod

            detect_mod.record_no_row_reason(
                con, hazard=HAZARD, iso3="PHL", ym=ym,
                reason=cell_ledger.REASON_BUDGET_DEFERRED,
                note="extraction budget bound at its backcast share (2000); resumes",
            )
        else:
            from resolver.hazard_resolution import resolutions as res_mod
            from resolver.tests.hazard_resolution_utils import silent_sweep_evidence

            res_mod.write_zero_resolution(
                con, iso3="PHL", year=int(ym[:4]), month=int(ym[5:]), hazard=HAZARD,
                evidence_of_absence=silent_sweep_evidence("PHL", ym),
                rulebook=rulebook, today=TODAY, run_type=run_type,
            )
        return 0

    first = bc.run_backcast(
        hazard_name="flood", rulebook=rulebook, con=con, today=TODAY,
        from_ym="2015-01", to_ym="2015-01", runner=runner,
    )
    assert first.months_budget_deferred == 1
    assert first.cells_deferred_for_budget == 1
    assert first.extraction_budget_bound is True
    assert first.extraction_binding_limit == "backcast share (2000)"
    assert first.months_failed == 0
    assert first.budget_policy == "checkpoint_and_resume"
    assert any("no cap was raised" in w for w in first.warnings)
    status, owed = con.execute(
        "SELECT status, deferred_cells FROM haz_backcast_progress WHERE ym = '2015-01'"
    ).fetchone()
    assert status == "deferred" and json.loads(owed) == ["PHL"]
    assert bc.completed_months(con, HAZARD) == set()
    assert bc.deferred_months(con, HAZARD) == {"2015-01": ["PHL"]}

    second = bc.run_backcast(
        hazard_name="flood", rulebook=rulebook, con=con, today=TODAY,
        from_ym="2015-01", to_ym="2015-01", runner=runner,
    )
    assert calls[1]["countries"] == ["PHL"]
    assert second.months_budget_deferred == 0
    assert second.extraction_budget_bound is False
    assert con.execute(
        "SELECT status FROM haz_backcast_progress WHERE ym = '2015-01'"
    ).fetchone()[0] == "ok"


def test_the_backcast_summary_names_the_policy(tmp_path, con, rulebook, monkeypatch):
    def runner(*, ym, run_type, countries_filter=None, dry_run=False, **_):
        _trigger(con, rulebook, "PHL", ym, run_type)
        from resolver.hazard_resolution import detect as detect_mod

        detect_mod.record_no_row_reason(
            con, hazard=HAZARD, iso3="PHL", ym=ym,
            reason=cell_ledger.REASON_BUDGET_DEFERRED,
            note="extraction budget bound at its backcast share (2000); resumes",
        )
        return 0

    run = bc.run_backcast(
        hazard_name="flood", rulebook=rulebook, con=con, today=TODAY,
        from_ym="2015-01", to_ym="2015-01", runner=runner,
    )
    assert run.extraction_budget_bound
    # The summary the workflow uploads carries the decision in words.
    out = tmp_path / "summary.json"
    monkeypatch.setattr(bc, "run_backcast", lambda **kw: run)
    rc = bc.main(["--hazard", "flood", "--summary-out", str(out)])
    assert rc == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["extraction_budget_bound"] is True
    assert summary["budget_policy"] == "checkpoint_and_resume"
    assert summary["months_budget_deferred"] == 1
    assert summary["cells_deferred_for_budget"] == 1
    assert summary["extraction_binding_limit"] == "backcast share (2000)"


def test_a_legacy_ledger_is_rebuilt_to_admit_deferred(tmp_path):
    db = tmp_path / "old.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        """
        CREATE TABLE haz_backcast_progress (
            hazard TEXT NOT NULL CHECK (hazard IN ('FL','DR','TC')),
            ym TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('ok','failed')),
            cells INTEGER NOT NULL DEFAULT 0,
            resolved_value INTEGER NOT NULL DEFAULT 0,
            resolved_zero INTEGER NOT NULL DEFAULT 0,
            no_data INTEGER NOT NULL DEFAULT 0,
            frozen_skipped INTEGER NOT NULL DEFAULT 0,
            extraction_calls INTEGER NOT NULL DEFAULT 0,
            extraction_cost_usd DOUBLE NOT NULL DEFAULT 0.0,
            duration_sec DOUBLE,
            error TEXT,
            ran_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (hazard, ym)
        )
        """
    )
    con.execute(
        "INSERT INTO haz_backcast_progress (hazard, ym, status, cells) VALUES ('FL','2015-01','ok',3)"
    )
    ensure_haz_schema(con, force=True)
    bc.record_month(con, hazard="FL", ym="2015-02", status="deferred",
                    counts={"cells": 2}, deferred_cells=["PHL", "VNM"])
    rows = con.execute(
        "SELECT ym, status, deferred_cells FROM haz_backcast_progress ORDER BY ym"
    ).fetchall()
    assert rows[0][:2] == ("2015-01", "ok")
    assert rows[1] == ("2015-02", "deferred", '["PHL", "VNM"]')
    # Idempotent: a second ensure leaves it alone.
    ensure_haz_schema(con, force=True)
    assert con.execute("SELECT COUNT(*) FROM haz_backcast_progress").fetchone()[0] == 2


def test_the_deferral_check_wants_the_ledger_to_owe_every_deferred_cell(tmp_path):
    db = tmp_path / "resolver.duckdb"
    con = duckdb.connect(str(db))
    ensure_haz_schema(con)
    rulebook = make_rulebook()
    _trigger(con, rulebook, "PHL", "2015-01", RUN_TYPE_BACKCAST)
    from resolver.hazard_resolution import detect as detect_mod

    detect_mod.record_no_row_reason(
        con, hazard=HAZARD, iso3="PHL", ym="2015-01",
        reason=cell_ledger.REASON_BUDGET_DEFERRED, note="bound",
    )
    # Ledger says the month is done: contradiction.
    bc.record_month(con, hazard=HAZARD, ym="2015-01", status="ok", counts={"cells": 1})
    con.close()
    builder = _builder(tmp_path, db)
    builder._check_backcast_deferral_is_recorded()
    assert builder.checks[-1]["verdict"] == "FAIL"
    assert "not deferred" in builder.checks[-1]["detail"]

    con = duckdb.connect(str(db))
    bc.record_month(con, hazard=HAZARD, ym="2015-01", status="deferred",
                    counts={"cells": 1}, deferred_cells=["PHL"])
    con.close()
    builder = _builder(tmp_path, db)
    builder._check_backcast_deferral_is_recorded()
    assert builder.checks[-1]["verdict"] == "PASS"
