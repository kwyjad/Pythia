# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The resolver debug bundle: does it answer the questions it exists for?

Seven questions define the bundle (see the module docstring of
``scripts/build_resolver_debug_bundle.py``). These tests assert the four that
can be checked without a network or a live run, plus the three properties
that decide whether the bundle is usable at all: it never leaks a secret, it
is always produced however little exists, and when the size ceiling binds it
drops code before evidence and says so.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from resolver.diagnostics import redaction, run_log
from scripts import build_resolver_debug_bundle as bundle

duckdb = pytest.importorskip("duckdb")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _make_db(path: Path) -> None:
    """A small database carrying every table the bundle's questions touch."""

    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE haz_triggers (
            iso3 TEXT, year INTEGER, month INTEGER, hazard TEXT,
            triggered BOOLEAN, trigger_source TEXT, trigger_detail_json TEXT,
            evidence_of_absence_json TEXT, run_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE haz_resolutions (
            iso3 TEXT, year INTEGER, month INTEGER, hazard TEXT, status TEXT,
            value DOUBLE, provenance_json TEXT, rule_fired TEXT,
            flagged BOOLEAN, provisional BOOLEAN, run_type TEXT,
            frozen_at TIMESTAMP, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE haz_impact_candidates (
            iso3 TEXT, year INTEGER, month INTEGER, hazard TEXT, value DOUBLE,
            value_type TEXT, source TEXT, source_ref TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE haz_doc_extractions (
            doc_id TEXT, iso3 TEXT, year INTEGER, month INTEGER, hazard TEXT,
            model TEXT, prompt_version TEXT, status TEXT, n_figures INTEGER,
            n_rejected INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER,
            cost_usd DOUBLE, doc_url TEXT, error TEXT, run_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE enso_state (
            fetch_date DATE, enso_phase TEXT, nino34_anomaly DOUBLE, oni DOUBLE,
            observation_date DATE, status TEXT
        )
        """
    )
    con.execute(
        "CREATE TABLE seasonal_tc_context_cache (iso3 TEXT, context_text TEXT, "
        "fetched_at TIMESTAMP)"
    )
    con.execute(
        """
        CREATE TABLE seasonal_tc_outlooks (
            basin TEXT, source TEXT, forecast_season TEXT, category TEXT,
            issue_date DATE, issue_date_key TEXT, issue_date_reason TEXT,
            season_reason TEXT, named_storms_forecast TEXT, named_storms_reason TEXT,
            raw_json TEXT, fetched_at TIMESTAMP, updated_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        CREATE TABLE conflict_forecasts (
            source TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT,
            lead_months INTEGER, forecast_issue_date DATE, value DOUBLE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE facts_resolved (
            iso3 TEXT, ym TEXT, hazard_code TEXT, metric TEXT, value DOUBLE,
            publisher TEXT, series_semantics TEXT, as_of_date DATE
        )
        """
    )

    # A cyclone month: one triggered cell resolved, one triggered cell with no
    # row (the ladder found nothing before the freeze deadline), one quiet cell
    # zeroed, and one quiet cell whose sweep could not be read.
    con.executemany(
        "INSERT INTO haz_triggers VALUES (?,?,?,?,?,?,?,?,?, CURRENT_TIMESTAMP)",
        [
            ("PHL", 2026, 8, "TC", True, "ibtracs", "{}", None, "live"),
            ("VNM", 2026, 8, "TC", True, "ibtracs", "{}", None, "live"),
            ("FJI", 2026, 8, "TC", False, "none", "{}", '{"reliefweb": {}}', "live"),
            ("TON", 2026, 8, "TC", False, "none", "{}", None, "live"),
        ],
    )
    con.executemany(
        "INSERT INTO haz_resolutions VALUES (?,?,?,?,?,?,?,?,?,?,?,NULL,CURRENT_TIMESTAMP)",
        [
            ("PHL", 2026, 8, "TC", "RESOLVED_VALUE", 40000.0, "{}", "ladder:emdat",
             False, True, "live"),
            ("FJI", 2026, 8, "TC", "RESOLVED_ZERO", 0.0, "{}", "zero:silent_sweep",
             False, True, "live"),
        ],
    )
    con.execute(
        "INSERT INTO haz_impact_candidates VALUES "
        "('PHL',2026,8,'TC',40000.0,'affected','emdat','disno-1')"
    )
    con.execute(
        "INSERT INTO haz_doc_extractions VALUES "
        "('rw-1','PHL',2026,8,'TC','haiku','v1','ok',2,1,900,120,0.009,"
        "'https://reliefweb.int/1',NULL,'live',CURRENT_TIMESTAMP)"
    )
    con.execute(
        "INSERT INTO haz_doc_extractions VALUES "
        "('rw-2','VNM',2026,8,'TC','haiku','v1','error',0,0,0,0,0.0,"
        "'https://reliefweb.int/2','timeout','backcast',CURRENT_TIMESTAMP)"
    )
    con.execute("INSERT INTO facts_resolved VALUES "
                "('PHL','2026-08','TC','affected',40000.0,'IFRC','new',DATE '2026-08-31')")
    con.close()


def _write_streams(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / f"{run_log.STREAM_HTTP}.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "connector": "resolver.connectors.acled_cast",
            "method": "GET",
            "url": "https://api.acleddata.com/cast/read?key=<redacted:sha256:deadbeef>&limit=5000",
            "status": 200, "elapsed_ms": 1200.0, "response_bytes": 51234,
            "redirects": 0, "from_cache": False, "error": None,
        }) + "\n")
        handle.write(json.dumps({
            "connector": "resolver.hazard_resolution.emdat",
            "method": "POST", "url": "https://api.emdat.be/v1",
            "status": 500, "elapsed_ms": 300.0, "response_bytes": 40,
            "redirects": 0, "from_cache": False, "error": None,
        }) + "\n")
    with open(directory / f"{run_log.STREAM_ENVELOPE}.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "connector": "resolver.connectors.acled_cast",
            "host": "api.acleddata.com", "path": "/cast/read",
            "url": "https://api.acleddata.com/cast/read",
            "status": 200,
            "envelope": {
                "json": True,
                "top_level": {
                    "status": 200, "success": True, "count": 9879,
                    "last_update": "2025-12-01",
                    "messages": "", "data_query_restrictions": "free tier",
                    "data": "list[9879]",
                },
                "n_rows": 9879,
                "columns": ["country", "month", "year", "timestamp"],
                "max_by_date_column": {"month": "12", "year": "2025",
                                       "timestamp": "2025-12-01T00:00:00"},
            },
        }) + "\n")
    with open(directory / f"{run_log.STREAM_CELLS}.jsonl", "w", encoding="utf-8") as handle:
        for iso3, outcome, reason in (
            ("VNM", "pending_skip", "pending_before_freeze"),
            ("TON", "no_row", "sweep_inconclusive"),
        ):
            handle.write(json.dumps({
                "stage": "ladder", "iso3": iso3, "hazard": "TC", "ym": "2026-08",
                "triggered": iso3 == "VNM", "write_outcome": outcome,
                "reason_code": reason, "rungs_unavailable": ["emdat"],
                "rungs_readable": [], "extraction": {}, "detail": {},
            }) + "\n")
    with open(directory / f"{run_log.STREAM_FIGURES}.jsonl", "w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "iso3": "PHL", "hazard": "TC", "ym": "2026-08", "outcome": "rejected",
            "doc_id": "rw-1", "value": 120000.0, "unit": "people",
            "reason": "exceeds_gdacs_exposure_ceiling",
            "ceiling": 2.0, "ceiling_multiplier": 3.0, "ceiling_source": "gdacs",
            "ceiling_source_ref": "TC-1001273",
            "ceiling_field": "gdacs.population -> payload.exposed_population",
            "quote": "some 120,000 people were affected",
        }) + "\n")


def _build(tmp_path: Path, **kwargs):
    out = tmp_path / "bundle.zip"
    manifest = bundle.build_bundle(
        out_path=out,
        db_path=kwargs.get("db_path"),
        diagnostics_dir=kwargs.get("diagnostics_dir", tmp_path / "missing"),
        run_log_dir=kwargs.get("run_log_dir"),
        max_bytes=kwargs.get("max_bytes", bundle.DEFAULT_MAX_BYTES),
        staging=tmp_path / "staging",
        environ=kwargs.get("environ", {}),
    )
    return out, manifest


@pytest.fixture()
def full_run(tmp_path: Path):
    db = tmp_path / "resolver.duckdb"
    _make_db(db)
    diagnostics = tmp_path / "diagnostics"
    (diagnostics / "ingestion").mkdir(parents=True)
    (diagnostics / "phase25_haz_cyclone.log").write_text(
        "2026-08-28 03:10:00 INFO resolver.hazard_resolution.impact PHL 2026-08 ladder\n"
        "2026-08-28 03:10:01 WARNING resolver.hazard_resolution.emdat "
        "ladder source emdat unavailable for TC 2026-08: HTTP 500\n"
        "2026-08-28 03:10:02 WARNING resolver.hazard_resolution.emdat "
        "ladder source emdat unavailable for TC 2026-07: HTTP 500\n"
        "2026-08-28 03:10:03 ERROR resolver.hazard_resolution.drought "
        "756 cells inconclusive\n",
        encoding="utf-8",
    )
    (diagnostics / "haz_run_cyclone.json").write_text(
        json.dumps({"hazard": "cyclone", "months": {"2026-08": {
            "cells": 4, "resolved_value": 1, "zeros": 1, "failures": 0}}}),
        encoding="utf-8",
    )
    (diagnostics / "haz_run_drought.json").write_text(
        json.dumps({"hazard": "drought", "months": {"2026-08": {
            "cells": 756, "resolved_value": 0, "resolved_zero": 0, "failures": 0}}}),
        encoding="utf-8",
    )
    (diagnostics / "db_signature_before.json").write_text(
        json.dumps({"required_counts": {"facts_resolved": 1},
                    "all_counts": {"facts_resolved": 1, "haz_resolutions": 0}}),
        encoding="utf-8",
    )
    (diagnostics / "ingestion" / "connectors_report.jsonl").write_text(
        json.dumps({"connector_id": "acled_client", "status": "ok",
                    "counts": {"fetched": 380000, "normalized": 1752, "written": 1752}})
        + "\n",
        encoding="utf-8",
    )
    streams = tmp_path / "runlog"
    _write_streams(streams)
    return {"db": db, "diagnostics": diagnostics, "streams": streams}


# --------------------------------------------------------------------------
# The seven questions
# --------------------------------------------------------------------------


def test_which_url_did_each_connector_call_and_how_old_was_the_data(tmp_path, full_run):
    """Q1: the URL, the response envelope and the arithmetic behind the vintage."""

    out, _ = _build(tmp_path, db_path=full_run["db"],
                    diagnostics_dir=full_run["diagnostics"],
                    run_log_dir=full_run["streams"])
    with zipfile.ZipFile(out) as zf:
        requests = zf.read("http/requests.jsonl").decode()
        envelope = zf.read(
            "http/envelopes/resolver.connectors.acled_cast.json"
        ).decode()
        by_connector = zf.read("http/requests_by_connector.csv").decode()

    assert "api.acleddata.com/cast/read" in requests
    assert "api.emdat.be" in requests
    # The half the connector discards, which is the half that says WHY a
    # vintage has stalled: a quota restriction and an upstream stop want
    # different fixes.
    assert "data_query_restrictions" in envelope
    assert "last_update" in envelope
    # The arithmetic behind "this data is N days old", not just the verdict.
    assert "max_by_date_column" in envelope
    assert "resolver.hazard_resolution.emdat" in by_connector
    assert "n_5xx" in by_connector


def test_what_thresholds_and_windows_was_the_run_using(tmp_path, full_run):
    """Q2: rulebook thresholds are copied, never inferred from behaviour."""

    out, _ = _build(
        tmp_path, db_path=full_run["db"], diagnostics_dir=full_run["diagnostics"],
        run_log_dir=full_run["streams"],
        environ={"GDACS_MONTHS": "3", "FEWSNET_MONTHS": "12", "PYTHIA_DB_URL": "duckdb:///x"},
    )
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        env = json.loads(zf.read("run/env_effective.json"))
        rulebook = zf.read("config/rulebook.yaml").decode()

    assert "config/rulebook.yaml" in names
    assert "config/workflows/resolver_update.yml" in names
    assert env["effective_windows"]["gdacs_months"] == "3"
    assert env["effective_windows"]["fewsnet_months"] == "12"
    machine = env["effective_windows"]["hazard_machine"]
    assert machine.get("freeze_days") is not None
    assert machine.get("ladder")
    assert "ceiling_multiplier" in rulebook


def test_for_any_unresolved_cell_why_did_it_produce_no_row(tmp_path, full_run):
    """Q3: the reason code, per cell — not a subtraction between summaries."""

    out, _ = _build(tmp_path, db_path=full_run["db"],
                    diagnostics_dir=full_run["diagnostics"],
                    run_log_dir=full_run["streams"])
    with zipfile.ZipFile(out) as zf:
        ledger = zf.read("hazard/cell_ledger.csv").decode()

    rows = {
        line.split(",")[0]: line
        for line in ledger.splitlines()
        if line and not line.startswith("#")
    }
    assert "PHL" in rows and "RESOLVED_VALUE" in rows["PHL"]
    # A triggered cell with no row, and the reason the calendar gives.
    assert "pending_before_freeze" in rows["VNM"]
    # A quiet cell whose sweep could not be read: unproven silence, never a zero.
    assert "sweep_inconclusive" in rows["TON"]
    # And a cell nothing accounted for would be named as a bug, not a silence.
    assert not any("unexplained_no_row" in line for line in rows.values())


def test_for_any_rejected_figure_what_ceiling_and_what_source_column(tmp_path, full_run):
    """Q4: value, ceiling, and the upstream field the ceiling came from.

    A ceiling of 2 against a reported 120,000 is a GDACS enrichment failure,
    not a mis-transcription, and only the source column says which.
    """

    out, _ = _build(tmp_path, db_path=full_run["db"],
                    diagnostics_dir=full_run["diagnostics"],
                    run_log_dir=full_run["streams"])
    with zipfile.ZipFile(out) as zf:
        figures = zf.read("hazard/figures_ledger.csv").decode()

    assert "exceeds_gdacs_exposure_ceiling" in figures
    assert "120000" in figures
    assert "gdacs.population -> payload.exposed_population" in figures
    assert "TC-1001273" in figures


def test_was_the_extraction_budget_exhausted_and_by_whom(tmp_path, full_run):
    """Q5: spend split by caller, so backcast and live draw are separable."""

    out, _ = _build(tmp_path, db_path=full_run["db"],
                    diagnostics_dir=full_run["diagnostics"],
                    run_log_dir=full_run["streams"])
    with zipfile.ZipFile(out) as zf:
        budget = zf.read("hazard/extraction_budget.csv").decode()
        calls = zf.read("hazard/extraction_calls.csv").decode()

    assert "live" in budget and "backcast" in budget
    assert "live reserve=" in budget  # the caps the spend is measured against
    # billed_calls, not calls: a call that never reached the provider spent
    # nothing and must cost nothing.
    assert "billed_calls" in budget
    assert "rw-2" in calls and "timeout" in calls


def test_which_code_and_config_produced_this_run(tmp_path, full_run):
    """Q6: the participating source files, verbatim, with a hashed index."""

    out, _ = _build(tmp_path, db_path=full_run["db"],
                    diagnostics_dir=full_run["diagnostics"],
                    run_log_dir=full_run["streams"])
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        index = zf.read("code/file_index.csv").decode()
        git = json.loads(zf.read("run/git.json"))

    assert "code/resolver/hazard_resolution/reconcile.py" in names
    # The log named emdat; selection follows the log rather than a fixed list.
    assert "code/resolver/hazard_resolution/emdat.py" in names
    assert "sha256_16" in index
    assert "commit" in git


def test_do_any_two_sources_in_the_run_contradict_each_other(tmp_path, full_run):
    """Q7: contradictions are asserted across the bundle, with both values."""

    db = full_run["db"]
    con = duckdb.connect(str(db))
    # A phase stated with no measurement behind it, and a TC narrative that
    # flatly disagrees with the stored phase. Both were live in August 2026.
    con.execute(
        "INSERT INTO enso_state VALUES "
        "(DATE '2026-08-01','Neutral',NULL,NULL,DATE '2026-08-01','fresh')"
    )
    con.execute(
        "INSERT INTO seasonal_tc_context_cache VALUES "
        "('PHL','Strong El Nino conditions persist across the basin', CURRENT_TIMESTAMP)"
    )
    # Two outlooks for one basin and season sharing an issue date.
    con.executemany(
        "INSERT INTO seasonal_tc_outlooks VALUES "
        "(?,?,?,?,?,?,NULL,NULL,?,NULL,?,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
        [
            ("NA", "TSR", "2026", "pre_season", "2026-05-20", "2026-05-20", "17",
             '{"issue_date": "2026-05-20"}'),
            ("NA", "TSR", "2026", "august_update", "2026-05-20", "2026-05-20", "19",
             '{"issue_date": "2026-05-20"}'),
        ],
    )
    con.execute(
        "INSERT INTO conflict_forecasts VALUES "
        "('ACLED_CAST','SOM','ACE','cast_total_events',1,DATE '2025-12-01',10.0)"
    )
    con.close()

    out, manifest = _build(tmp_path, db_path=db,
                           diagnostics_dir=full_run["diagnostics"],
                           run_log_dir=full_run["streams"])
    with zipfile.ZipFile(out) as zf:
        contradictions = zf.read("checks/contradictions.md").decode()
        reconciliation = zf.read("checks/reconciliation.md").decode()
        readme = zf.read("README.md").decode()

    failed = {c["name"] for c in manifest["checks"] if c["verdict"] == "FAIL"}
    assert "enso_phase_never_stated_without_a_measurement" in failed
    assert "enso_phase_matches_the_tc_context_narrative" in failed
    assert "no_two_tc_outlooks_share_a_basin_season_and_issue_date" in failed
    assert "no_stored_forecast_vintage_past_its_threshold" in failed
    # A hazard reporting failures=0 while resolving nothing at all.
    assert "no_hazard_reports_zero_failures_while_resolving_nothing" in failed
    assert "FAIL" in contradictions
    # The reconciliation names the semantics of each counter rather than
    # carrying a caveat that the totals are not comparable.
    assert "claimed written" in reconciliation and "table delta" in reconciliation
    # The README leads with what is wrong.
    assert "## What is wrong" in readme
    assert "enso_phase_never_stated_without_a_measurement" in readme


# --------------------------------------------------------------------------
# Properties that decide whether the bundle is usable at all
# --------------------------------------------------------------------------


def test_redaction_rejects_a_bundle_containing_a_known_secret(tmp_path, full_run, monkeypatch):
    """A bundle meant for a chat window cannot leak. The build must fail."""

    secret = "acled-live-token-ABCDEF0123456789"
    (full_run["diagnostics"] / "phase1_leak.log").write_text(
        f"authenticating with token={secret}\n", encoding="utf-8"
    )
    # Defeat the capture-time redaction so only the final scan can catch it —
    # that scan is the guarantee, and a test that lets redact_text do the work
    # would pass with the scan deleted.
    monkeypatch.setattr(bundle, "redact_text", lambda text, values=None: text)

    with pytest.raises(bundle.SecretLeak) as excinfo:
        _build(tmp_path, db_path=full_run["db"],
               diagnostics_dir=full_run["diagnostics"],
               run_log_dir=full_run["streams"],
               environ={"ACLED_ACCESS_KEY": secret})

    assert "logs/phase1_leak.log" in excinfo.value.hits
    # The leak report quotes the FINGERPRINT, never the value: a report that
    # quotes the leak is a second leak.
    assert secret not in str(excinfo.value.hits)
    assert excinfo.value.hits["logs/phase1_leak.log"] == [redaction.fingerprint(secret)]


def test_redaction_passes_when_the_secret_was_redacted_at_capture(tmp_path, full_run):
    secret = "acled-live-token-ABCDEF0123456789"
    (full_run["diagnostics"] / "phase1_leak.log").write_text(
        f"authenticating with token={secret}\n", encoding="utf-8"
    )
    out, manifest = _build(tmp_path, db_path=full_run["db"],
                           diagnostics_dir=full_run["diagnostics"],
                           run_log_dir=full_run["streams"],
                           environ={"ACLED_ACCESS_KEY": secret})
    assert manifest["redaction_scan"]["files_with_hits"] == {}
    with zipfile.ZipFile(out) as zf:
        log = zf.read("logs/phase1_leak.log").decode()
    assert secret not in log
    assert redaction.fingerprint(secret) in log


@pytest.mark.parametrize(
    "missing", ["db", "diagnostics", "streams", "everything"]
)
def test_the_builder_degrades_rather_than_failing(tmp_path, full_run, missing):
    """A diagnostic that fails when the run failed is never there when needed."""

    kwargs = {
        "db_path": full_run["db"],
        "diagnostics_dir": full_run["diagnostics"],
        "run_log_dir": full_run["streams"],
    }
    if missing in ("db", "everything"):
        kwargs["db_path"] = tmp_path / "no-such.duckdb"
    if missing in ("diagnostics", "everything"):
        kwargs["diagnostics_dir"] = tmp_path / "no-such-dir"
    if missing in ("streams", "everything"):
        kwargs["run_log_dir"] = None

    out, manifest = _build(tmp_path, **kwargs)
    assert out.is_file()
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    # The two things a reader needs whatever else is absent.
    assert "README.md" in names
    assert "manifest.json" in names
    assert "checks/contradictions.md" in names
    # And the absence is stated, not silent.
    if missing != "db":
        assert manifest["sections"]["db"]["ok"]
    else:
        assert any("database" in p for p in manifest["problems"])
    if missing in ("streams", "everything"):
        assert any("PYTHIA_RUN_LOG_DIR" in p for p in manifest["problems"])


def test_the_builder_survives_a_missing_phase_log(tmp_path, full_run):
    for log in full_run["diagnostics"].glob("*.log"):
        log.unlink()
    out, manifest = _build(tmp_path, db_path=full_run["db"],
                           diagnostics_dir=full_run["diagnostics"],
                           run_log_dir=full_run["streams"])
    with zipfile.ZipFile(out) as zf:
        index = zf.read("logs/log_index.md").decode()
        names = set(zf.namelist())
    assert "No phase logs were found" in index
    # Code selection falls back to the machine's own modules, so "which code
    # produced this run" still has an answer.
    assert "code/resolver/hazard_resolution/reconcile.py" in names
    assert manifest["sections"]["logs"]["ok"]


def test_the_size_ceiling_drops_code_before_evidence_and_records_it(tmp_path, full_run):
    """Truncate loudly, never silently — and evidence outranks source."""

    big = full_run["diagnostics"] / "phase3_big.log"
    big.write_text("\n".join(f"line {i} of an enormous log" for i in range(200_000)),
                   encoding="utf-8")

    out, manifest = _build(tmp_path, db_path=full_run["db"],
                           diagnostics_dir=full_run["diagnostics"],
                           run_log_dir=full_run["streams"],
                           max_bytes=60_000)

    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
    assert not any(n.startswith("code/") for n in names), "code/ should go first"
    assert "hazard/cell_ledger.csv" in names, "evidence must outlive code"
    assert any("code/" in entry for entry in manifest["dropped"])
    assert manifest["truncated"], "the truncation must be recorded, not silent"
    with zipfile.ZipFile(out) as zf:
        truncated = zf.read("logs/phase3_big.log").decode()
    assert "TRUNCATED" in truncated
    # The original size is stated, so a reader knows what they are missing.
    assert "original file was" in truncated


def test_a_truncated_stream_line_costs_one_record_not_the_file(tmp_path, full_run):
    path = full_run["streams"] / f"{run_log.STREAM_CELLS}.jsonl"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"stage": "ladder", "iso3": "KI')  # killed mid-write

    out, _ = _build(tmp_path, db_path=full_run["db"],
                    diagnostics_dir=full_run["diagnostics"],
                    run_log_dir=full_run["streams"])
    with zipfile.ZipFile(out) as zf:
        ledger = zf.read("hazard/cell_ledger.csv").decode()
    assert "sweep_inconclusive" in ledger  # the intact records survived


def test_normalise_db_path_handles_the_urls_the_workflow_actually_sets():
    # The workflow writes duckdb:///<github.workspace>/data/... and
    # github.workspace is already absolute, so the fourth slash is the path's.
    assert bundle.normalise_db_path("duckdb:////w/data/r.duckdb") == Path("/w/data/r.duckdb")
    assert bundle.normalise_db_path("duckdb:///a/b.duckdb") == Path("a/b.duckdb")
    assert bundle.normalise_db_path("data/resolver.duckdb") == Path("data/resolver.duckdb")
    assert bundle.normalise_db_path("") is None
    assert bundle.normalise_db_path(None) is None


def test_the_enso_ladder_must_have_two_live_ranks(tmp_path, full_run):
    """Run 33841370196 had one working index source and nothing to compare it to."""

    db = full_run["db"]
    con = duckdb.connect(str(db))
    con.execute("ALTER TABLE enso_state ADD COLUMN index_evidence_json TEXT")
    con.execute("ALTER TABLE enso_state ADD COLUMN row_kind TEXT")
    one_rank = json.dumps({
        "readings": [
            {"rank": 1, "ok": False, "error": "HTTP 400", "newest_observation": None},
            {"rank": 2, "ok": False, "error": "2046 days old", "newest_observation": "2021-01-27"},
            {"rank": 3, "ok": True, "newest_observation": "2026-07-01"},
        ]
    })
    con.execute(
        "INSERT INTO enso_state VALUES "
        "(DATE '2026-09-04','El Niño',1.8,1.8,DATE '2026-07-01','fresh',?, 'live')",
        [one_rank],
    )
    # A historical row with everything alive must not rescue a live run.
    con.execute(
        "INSERT INTO enso_state VALUES "
        "(DATE '1950-01-01','La Niña',-1.5,-1.5,DATE '1950-01-01','historical',?, 'historical')",
        [json.dumps({"readings": [{"rank": 1, "ok": True}, {"rank": 2, "ok": True}]})],
    )
    con.close()

    _out, manifest = _build(tmp_path, db_path=db,
                            diagnostics_dir=full_run["diagnostics"],
                            run_log_dir=full_run["streams"])
    checks = {c["name"]: c for c in manifest["checks"]}
    check = checks["enso_ladder_has_two_live_ranks_in_the_last_30_days"]
    assert check["verdict"] == "FAIL"
    assert "[3]" in check["left"]

    con = duckdb.connect(str(db))
    con.execute(
        "UPDATE enso_state SET index_evidence_json = ? WHERE fetch_date = DATE '2026-09-04'",
        [json.dumps({"readings": [
            {"rank": 1, "ok": True, "newest_observation": "2026-08-19"},
            {"rank": 2, "ok": False},
            {"rank": 3, "ok": True, "newest_observation": "2026-07-01"},
        ]})],
    )
    con.close()
    _out, manifest = _build(tmp_path / "again", db_path=db,
                            diagnostics_dir=full_run["diagnostics"],
                            run_log_dir=full_run["streams"])
    checks = {c["name"]: c for c in manifest["checks"]}
    assert checks["enso_ladder_has_two_live_ranks_in_the_last_30_days"]["verdict"] == "PASS"


def test_an_acled_web_page_beside_an_ok_connector_is_a_contradiction(tmp_path, full_run):
    """B1.3(d): a run cannot report ok while receiving ACLED's Unauthorized page."""

    from resolver.diagnostics import run_log

    streams = full_run["streams"]
    with open(streams / f"{run_log.STREAM_HTTP}.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "connector": "acled_client", "method": "GET",
            "url": "https://acleddata.com/api/acled/read?page=1",
            "status": 200, "content_type": "text/html; charset=UTF-8",
            "elapsed_ms": 12.0, "response_bytes": 5120, "redirects": 0,
        }) + "\n")
    report = full_run["diagnostics"] / "ingestion" / "connectors_report.jsonl"
    with open(report, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"connector_id": "acled_client", "status": "ok",
                                 "counts": {"written": 0}}) + "\n")

    _out, manifest = _build(tmp_path, db_path=full_run["db"],
                            diagnostics_dir=full_run["diagnostics"],
                            run_log_dir=streams)
    checks = {c["name"]: c for c in manifest["checks"]}
    check = checks["acled_html_responses_are_recorded_as_connector_failures"]
    assert check["verdict"] == "FAIL"
    assert "acled_client" in check["left"]


def test_a_conflict_forecast_reader_never_serves_a_past_target_month(tmp_path, full_run):
    db = full_run["db"]
    con = duckdb.connect(str(db))
    con.execute("ALTER TABLE conflict_forecasts ADD COLUMN target_month DATE")
    con.execute(
        "INSERT INTO conflict_forecasts VALUES "
        "('ACLED_CAST','SOM','ACE','cast_total_events',1,DATE '2025-12-01',10.0,DATE '2026-01-01')"
    )
    con.close()
    _out, manifest = _build(tmp_path, db_path=db,
                            diagnostics_dir=full_run["diagnostics"],
                            run_log_dir=full_run["streams"])
    checks = {c["name"]: c for c in manifest["checks"]}
    check = checks["no_conflict_forecast_served_has_a_target_month_in_the_past"]
    assert check["verdict"] == "PASS"
    assert "target_month" in check["detail"]


# --------------------------------------------------------------------------
# Group D (run 33841370196): the drought path's contradiction checks
# --------------------------------------------------------------------------


def _checks(tmp_path, db, full_run, sub="d"):
    _out, manifest = _build(tmp_path / sub, db_path=db,
                            diagnostics_dir=full_run["diagnostics"])
    return {c["name"]: c for c in manifest["checks"]}


def test_a_no_row_cell_with_a_reason_on_its_trigger_row_is_explained(tmp_path, full_run):
    """D1: 28,728 drought cells read unexplained_no_row because the reason
    lived only in the run stream, which the backcast never enables. The
    trigger row now carries it, so a bundle built from the DB alone knows."""

    db = full_run["db"]
    con = duckdb.connect(str(db))
    con.execute(
        "UPDATE haz_triggers SET trigger_detail_json = ? WHERE iso3 = 'TON'",
        [json.dumps({"no_row_reason": "sweep_inconclusive", "no_row_note": "HTTP 503"})],
    )
    con.execute(
        "INSERT INTO haz_triggers VALUES ('SOM', 2024, 3, 'DR', FALSE, 'none', ?, "
        "NULL, 'backcast', CURRENT_TIMESTAMP)",
        [json.dumps({"no_row_reason": "indicator_inconclusive", "assessed": False})],
    )
    con.close()
    # No run log at all: the DB is the only account.
    out, manifest = _build(tmp_path, db_path=db, diagnostics_dir=full_run["diagnostics"])
    with zipfile.ZipFile(out) as zf:
        ledger = zf.read("hazard/cell_ledger.csv").decode()
    rows = {l.split(",")[0]: l for l in ledger.splitlines() if l and not l.startswith("#")}
    assert "sweep_inconclusive" in rows["TON"]
    assert "indicator_inconclusive" in rows["SOM"] and "False" in rows["SOM"]
    # VNM has a trigger row, no resolution and no reason: the one bug left.
    assert "unexplained_no_row" in rows["VNM"]
    checks = {c["name"]: c for c in manifest["checks"]}
    assert checks["no_assessed_cell_lacks_a_reason_for_having_no_row"]["verdict"] == "FAIL"
    assert checks["no_assessed_cell_lacks_a_reason_for_having_no_row"]["left"].startswith("1 of")

    con = duckdb.connect(str(db))
    con.execute(
        "UPDATE haz_triggers SET trigger_detail_json = ? WHERE iso3 = 'VNM'",
        [json.dumps({"no_row_reason": "pending_before_freeze"})],
    )
    con.close()
    checks = _checks(tmp_path, db, full_run, "again")
    assert checks["no_assessed_cell_lacks_a_reason_for_having_no_row"]["verdict"] == "PASS"


def test_drought_severity_base_rates_must_exist_beside_assessed_drought_cells(tmp_path, full_run):
    """D1: 307 severity rows for FL and TC and not one for DR."""

    db = full_run["db"]
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE haz_base_rates_severity (iso3 TEXT, hazard TEXT, q10 DOUBLE, "
        "q25 DOUBLE, q50 DOUBLE, q75 DOUBLE, q90 DOUBLE, n_events INTEGER)"
    )
    con.execute(
        "INSERT INTO haz_triggers VALUES ('SOM', 2024, 3, 'DR', TRUE, 'drought_indicators', "
        "'{}', NULL, 'backcast', CURRENT_TIMESTAMP)"
    )
    con.execute(
        "INSERT INTO haz_resolutions VALUES ('SOM', 2024, 3, 'DR', 'RESOLVED_VALUE', "
        "140000.0, '{}', 'drought:ipc_phase3plus_delta', FALSE, FALSE, 'backcast', "
        "TIMESTAMP '2024-05-30', CURRENT_TIMESTAMP)"
    )
    con.close()
    checks = _checks(tmp_path, db, full_run)
    check = checks["drought_severity_base_rates_exist"]
    assert check["verdict"] == "FAIL"
    assert "0 DR severity rows" in check["left"]

    con = duckdb.connect(str(db))
    con.execute(
        "INSERT INTO haz_base_rates_severity VALUES ('SOM', 'DR', 1, 2, 3, 4, 5, 1)"
    )
    con.close()
    checks = _checks(tmp_path, db, full_run, "again")
    assert checks["drought_severity_base_rates_exist"]["verdict"] == "PASS"


def test_nmme_must_be_read_for_a_month_seasonal_forecasts_covers(tmp_path, full_run):
    """D2: 2,408 NMME rows merged, "no usable rows" logged for every month."""

    db = full_run["db"]
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE seasonal_forecasts (iso3 TEXT, variable TEXT, lead_months INTEGER, "
        "anomaly_value DOUBLE, forecast_issue_date DATE)"
    )
    con.execute(
        "INSERT INTO seasonal_forecasts VALUES ('SOM', 'prate', 1, -1.2, DATE '2026-07-08')"
    )
    con.execute(
        "CREATE TABLE haz_raw_drought_indicators (record_id TEXT, iso3 TEXT, ym TEXT, "
        "hazard TEXT, payload_json TEXT)"
    )
    con.execute(
        "INSERT INTO haz_triggers VALUES ('SOM', 2026, 8, 'DR', FALSE, 'none', "
        "'{}', NULL, 'live', CURRENT_TIMESTAMP)"
    )
    con.close()
    checks = _checks(tmp_path, db, full_run)
    check = checks["nmme_indicator_read_for_every_month_seasonal_forecasts_covers"]
    assert check["verdict"] == "FAIL"
    assert "2026-08" in check["detail"]

    con = duckdb.connect(str(db))
    con.execute(
        "INSERT INTO haz_raw_drought_indicators VALUES ('nmme_precip_anomaly-2026-08', NULL, "
        "'2026-08', 'DR', '{\"name\":\"nmme_precip_anomaly\",\"values\":{\"SOM\":-1.2}}')"
    )
    con.close()
    checks = _checks(tmp_path, db, full_run, "again")
    assert checks["nmme_indicator_read_for_every_month_seasonal_forecasts_covers"]["verdict"] == "PASS"


def test_a_drought_zero_on_one_feed_is_a_contradiction(tmp_path, full_run):
    """D3: 159 zeros a month rested on the one feed of three that answered."""

    db = full_run["db"]
    one_feed = json.dumps({"decision": {"delta": None, "indicators": {"readings": [
        {"name": "asap", "state": "unavailable"},
        {"name": "hdx_agricultural_stress", "state": "no_drought"},
        {"name": "nmme_precip_anomaly", "state": "unavailable"},
    ]}}})
    con = duckdb.connect(str(db))
    con.execute(
        "INSERT INTO haz_resolutions VALUES ('SOM', 2026, 8, 'DR', 'RESOLVED_ZERO', 0.0, ?, "
        "'drought_zero:no_indicator_signal+no_ipc_deterioration', FALSE, TRUE, 'live', "
        "TIMESTAMP '2026-10-30', CURRENT_TIMESTAMP)",
        [one_feed],
    )
    con.close()
    checks = _checks(tmp_path, db, full_run)
    check = checks["no_drought_zero_rests_on_fewer_feeds_than_the_rulebook_demands"]
    assert check["verdict"] == "FAIL"
    assert "SOM/2026-08 (1 feed(s))" in check["detail"]

    two_feeds = json.dumps({"decision": {"delta": None, "indicators": {
        "answered_count": 2, "readings": []}}})
    con = duckdb.connect(str(db))
    con.execute("UPDATE haz_resolutions SET provenance_json = ? WHERE hazard = 'DR'", [two_feeds])
    con.close()
    checks = _checks(tmp_path, db, full_run, "again")
    assert checks["no_drought_zero_rests_on_fewer_feeds_than_the_rulebook_demands"]["verdict"] == "PASS"


def test_no_drought_row_may_exist_for_a_month_still_in_progress(tmp_path, full_run):
    """D4: the 2026-09 pass wrote 159 zeros on 4 September."""

    import datetime as dt

    db = full_run["db"]
    today = dt.date.today()
    con = duckdb.connect(str(db))
    con.execute(
        "INSERT INTO haz_resolutions VALUES ('SOM', ?, ?, 'DR', 'RESOLVED_ZERO', 0.0, '{}', "
        "'drought_zero:no_indicator_signal+no_ipc_deterioration', FALSE, TRUE, 'live', "
        "NULL, CURRENT_TIMESTAMP)",
        [today.year, today.month],
    )
    con.close()
    checks = _checks(tmp_path, db, full_run)
    check = checks["no_drought_resolution_for_a_month_still_in_progress"]
    assert check["verdict"] == "FAIL"
    assert f"{today:%Y-%m}: 1" in check["detail"]

    con = duckdb.connect(str(db))
    con.execute("DELETE FROM haz_resolutions WHERE hazard = 'DR'")
    con.close()
    assert _checks(tmp_path, db, full_run, "again")[
        "no_drought_resolution_for_a_month_still_in_progress"]["verdict"] == "PASS"


# --------------------------------------------------------------------------
# Group E (run 33841370196): EM-DAT lockout, empty tables, future dates
# --------------------------------------------------------------------------


def test_a_publication_date_after_the_run_date_is_a_contradiction(tmp_path, full_run):
    db = full_run["db"]
    con = duckdb.connect(str(db))
    con.execute("ALTER TABLE facts_resolved ADD COLUMN publication_date VARCHAR")
    con.execute(
        "INSERT INTO facts_resolved (iso3, ym, hazard_code, metric, value, publisher, "
        "series_semantics, as_of_date, publication_date) VALUES "
        "('SWZ','2027-03','DR','phase3plus_projection',10.0,'IPC','stock',DATE '2027-03-01','2027-03-01')"
    )
    con.close()
    checks = _checks(tmp_path, db, full_run, "e1")
    check = checks["no_fact_carries_a_publication_date_after_the_run_date"]
    assert check["verdict"] == "FAIL"
    assert "facts_resolved: 1 rows" in check["detail"]


def test_freshness_measures_from_the_newest_value_that_has_happened(tmp_path, full_run):
    """A period end in 2027 must not make facts_resolved read as fresh."""

    db = full_run["db"]
    con = duckdb.connect(str(db))
    con.execute(
        "INSERT INTO facts_resolved (iso3, ym, hazard_code, metric, value, publisher, "
        "series_semantics, as_of_date) VALUES "
        "('SWZ','2027-03','DR','phase3plus_projection',10.0,'IPC','stock',DATE '2027-03-01')"
    )
    con.close()
    out, _manifest = _build(tmp_path / "e2", db_path=db, diagnostics_dir=full_run["diagnostics"])
    with zipfile.ZipFile(out) as zf:
        text = zf.read("db/freshness.csv").decode("utf-8")
    row = next(
        line for line in text.splitlines()
        if line.startswith("facts_resolved,as_of_date")
    )
    fields = row.split(",")
    # max_value is the newest value at or before now; the 2027 period end is
    # counted under n_future and named, never allowed to set the verdict.
    assert "2027" not in fields[2]
    assert fields[-2] == "1" and fields[-1].startswith("2027-03-01")


def test_a_table_this_workflow_writes_may_not_be_empty_after_the_run(tmp_path, full_run):
    db = full_run["db"]
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month DATE, fatalities BIGINT)")
    con.execute("CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, score_type TEXT, "
                "model_name TEXT, value DOUBLE, metric TEXT)")
    con.close()
    checks = _checks(tmp_path, db, full_run, "e3")
    check = checks["no_declared_active_table_is_empty_after_the_run"]
    assert check["verdict"] == "FAIL"
    assert "acled_monthly_fatalities" in check["detail"]
    # A table another workflow fills is reported, never failed on.
    assert "carried from another workflow" in check["detail"]
    assert "scores" in check["detail"]


def test_an_empty_calibration_weights_is_explained_in_the_writers_terms(tmp_path, full_run):
    db = full_run["db"]
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE calibration_weights (as_of_month TEXT, hazard_code TEXT, "
                "metric TEXT, model_name TEXT, weight DOUBLE)")
    con.execute("CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, score_type TEXT, "
                "model_name TEXT, value DOUBLE, metric TEXT)")
    con.execute("CREATE TABLE questions (question_id TEXT, hazard_code TEXT, metric TEXT, "
                "is_test BOOLEAN)")
    con.execute("CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, value DOUBLE)")
    con.execute("INSERT INTO questions VALUES ('SOM_ACE_FATALITIES_2026-08','ACE','FATALITIES',FALSE)")
    con.execute("INSERT INTO resolutions VALUES ('SOM_ACE_FATALITIES_2026-08',1,415.0)")
    con.execute("INSERT INTO scores VALUES ('SOM_ACE_FATALITIES_2026-08',1,'brier','gemini-3.5-flash',0.4,'FATALITIES')")
    con.close()
    checks = _checks(tmp_path, db, full_run, "e4")
    check = checks["no_declared_active_table_is_empty_after_the_run"]
    assert "calibration_weights (writer needs 20 resolved questions" in check["detail"]
    assert "ACE/FATALITIES at 1" in check["detail"]


def test_emdat_must_be_read_when_a_key_is_configured(tmp_path, full_run):
    """E1: a configured key, a ladder run, and an empty cache is a contradiction."""

    from resolver.diagnostics import run_log

    db = full_run["db"]
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE haz_raw_emdat (record_id TEXT, payload_json TEXT)")
    con.close()
    streams = full_run["streams"]
    with open(streams / "source_fetches.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "source": "emdat", "hazard": "TC", "ym": "2026-08", "ok": False,
            "records": 0, "inserted": 0, "served_from_cache": False,
            "failure_class": "auth_rejected",
            "error": 'EM-DAT GraphQL errors: [{"message": "Invalid key passed or insufficient user access"}]',
        }) + "\n")

    _out, manifest = _build(tmp_path / "e5", db_path=db,
                            diagnostics_dir=full_run["diagnostics"],
                            run_log_dir=streams,
                            environ={"EMDAT_API_KEY": "dead-key"})
    checks = {c["name"]: c for c in manifest["checks"]}
    check = checks["emdat_is_read_when_a_key_is_configured"]
    assert check["verdict"] == "FAIL"
    assert "rejected the key" in check["detail"] and "EMDAT_API_KEY" in check["detail"]
    assert "0 rows" in check["detail"]

    with zipfile.ZipFile(_out) as zf:
        fetches = zf.read("hazard/source_fetches.csv").decode("utf-8")
    assert "auth_rejected" in fetches
    _ = run_log


def test_emdat_check_is_skipped_without_a_key(tmp_path, full_run):
    checks = _checks(tmp_path, full_run["db"], full_run, "e6")
    assert checks["emdat_is_read_when_a_key_is_configured"]["verdict"] == "SKIP"
# Group F (run 33841370196): every outlook carries a date or a reason
# --------------------------------------------------------------------------


def test_an_outlook_with_neither_a_date_nor_a_reason_is_a_contradiction(tmp_path, full_run):
    db = full_run["db"]
    con = duckdb.connect(str(db))
    con.executemany(
        "INSERT INTO seasonal_tc_outlooks VALUES "
        "(?,?,?,?,?,?,?,NULL,NULL,NULL,'{}',CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
        [
            ("AUS", "BoM", "2025-26", "seasonal_outlook", "2025-10-01", "2025-10-01",
             "month_precision_only"),
            ("NIO", "IMD_RSMC_NewDelhi", "2026", "climatology_context", None,
             "undated:climatology_context_has_no_issue_date",
             "climatology_context_has_no_issue_date"),
            ("SP", "BoM", "2025-26", "seasonal_outlook", None, "undated:", None),
        ],
    )
    con.close()
    checks = _checks(tmp_path, db, full_run, "f1")
    check = checks["every_tc_outlook_has_a_parseable_issue_date_or_a_reason"]
    assert check["verdict"] == "FAIL"
    assert "1 undated with none" in check["left"]
    assert "SP/BoM" in check["detail"]


def test_an_undated_outlook_with_a_reason_passes(tmp_path, full_run):
    db = full_run["db"]
    con = duckdb.connect(str(db))
    con.execute(
        "INSERT INTO seasonal_tc_outlooks VALUES "
        "('NIO','IMD_RSMC_NewDelhi','2026','climatology_context',NULL,"
        "'undated:climatology_context_has_no_issue_date',"
        "'climatology_context_has_no_issue_date',NULL,NULL,NULL,'{}',"
        "CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)"
    )
    con.close()
    checks = _checks(tmp_path, db, full_run, "f2")
    assert checks["every_tc_outlook_has_a_parseable_issue_date_or_a_reason"]["verdict"] == "PASS"


def test_a_legacy_outlook_table_shape_fails_the_date_check(tmp_path, full_run):
    db = full_run["db"]
    con = duckdb.connect(str(db))
    con.execute("DROP TABLE seasonal_tc_outlooks")
    con.execute(
        "CREATE TABLE seasonal_tc_outlooks (basin TEXT, source TEXT, forecast_season TEXT, "
        "named_storms_forecast TEXT, category TEXT, raw_json TEXT, fetched_at TIMESTAMP)"
    )
    con.close()
    checks = _checks(tmp_path, db, full_run, "f3")
    assert checks["every_tc_outlook_has_a_parseable_issue_date_or_a_reason"]["verdict"] == "FAIL"


# --------------------------------------------------------------------------
# Group G (run 33841370196): the checks the review asked for all exist
# --------------------------------------------------------------------------


def test_the_six_requested_contradiction_checks_are_all_registered(tmp_path, full_run):
    """G3: each item on the review's list is a named check the bundle runs."""

    checks = _checks(tmp_path, full_run["db"], full_run, "g3")
    for name in (
        "no_assessed_cell_lacks_a_reason_for_having_no_row",       # unexplained_no_row = 0
        "no_declared_active_table_is_empty_after_the_run",          # active tables non-empty
        "no_fact_carries_a_publication_date_after_the_run_date",    # publication_date <= today
        "enso_ladder_has_two_live_ranks_in_the_last_30_days",       # two ENSO ranks alive
        "no_conflict_forecast_served_has_a_target_month_in_the_past",
        "no_accepted_figure_lies_outside_its_cell_reporting_window",
        "connectors_claiming_rows_touched_their_source_rows",       # G1 + G2
    ):
        assert name in checks, name
