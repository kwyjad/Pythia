# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The operational half of the debug bundle.

Cover for the six defects in the existing outputs and for every collector
added beside them. The shape most of these tests take is the shape of the
2026-09-01 run: a cycle that lost four provider batches, three member
forecasts and five months of CrisisWatch, and whose bundle reported
"LLM Calls: OK, 0 errors".

Two contracts are asserted throughout and are the reason the section
exists at all: a collector never fails the phase, and every file the
manifest names is in the zip.
"""

from __future__ import annotations

import csv
import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.debug_bundle import (  # noqa: E402
    anomalies as anomalies_mod,
    batch_lifecycle,
    code_snapshot,
    connector_freshness,
    manifest as manifest_mod,
    model_completeness,
    prompt_cache,
    prompt_prefixes,
    provider_objects,
    redaction,
    retry_report,
    workflow_logs,
)


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

def test_secret_env_values_are_fingerprinted_not_erased():
    # A constant mask cannot answer the question an incident actually asks:
    # is this the same key the last good run used?
    a = redaction.redact_env_value("OPENAI_API_KEY", "sk-aaaaaaaaaaaa")
    b = redaction.redact_env_value("OPENAI_API_KEY", "sk-bbbbbbbbbbbb")
    again = redaction.redact_env_value("OPENAI_API_KEY", "sk-aaaaaaaaaaaa")
    assert a.startswith("<redacted:sha256:") and a == again and a != b
    assert "sk-aaaa" not in a


def test_non_secret_env_values_survive():
    assert redaction.redact_env_value("PYTHIA_PROMPT_CACHE_ENABLED", "1") == "1"
    # A cache-partition label is not a credential, however it is spelled.
    assert redaction.redact_env_value("PYTHIA_PROMPT_CACHE_KEY", "pythia:spd") == "pythia:spd"


@pytest.mark.parametrize(
    "text",
    [
        "using key sk-ant-abcdefghijklmnopqrstuvwxyz012345",
        "Authorization: Bearer ghp_abcdefghijklmnopqrstuvwxyz01",
        "GEMINI_API_KEY=AIzaSyAbcdefghijklmnopqrstuvwxyz0123",
        "ACAPS_PASSWORD: hunter2hunter2hunter2",
    ],
)
def test_free_text_credentials_never_reach_the_artifact(text: str):
    out = redaction.redact_text(text)
    assert "<redacted:sha256:" in out
    for secret in ("sk-ant-abcdefghij", "ghp_abcdefghij", "AIzaSyAbcdefghij", "hunter2hunter2"):
        assert secret not in out


# ---------------------------------------------------------------------------
# Batch lifecycle
# ---------------------------------------------------------------------------

def _batch_db(tmp_path: Path):
    con = duckdb.connect(str(tmp_path / "b.duckdb"))
    con.execute(
        """
        CREATE TABLE llm_batches (
            batch_id TEXT, provider TEXT, provider_batch_id TEXT, family TEXT,
            run_id TEXT, hs_run_id TEXT, pipeline_id TEXT, stage TEXT,
            model_id TEXT, status TEXT, n_requests INTEGER, n_succeeded INTEGER,
            n_errored INTEGER, n_expired INTEGER, n_fallback_sync INTEGER,
            input_file_id TEXT, output_file_id TEXT, error_file_id TEXT,
            results_url TEXT, submitted_at TIMESTAMP, first_polled_at TIMESTAMP,
            ended_at TIMESTAMP, collected_at TIMESTAMP, error_text TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE llm_batch_requests (
            custom_id TEXT, batch_id TEXT, status TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE llm_calls (
            phase TEXT, call_type TEXT, provider TEXT, model_id TEXT,
            run_id TEXT, hs_run_id TEXT, cost_usd DOUBLE, usage_json TEXT
        )
        """
    )
    return con


def test_a_batch_that_yielded_nothing_is_reported_with_its_provider_error(tmp_path: Path):
    con = _batch_db(tmp_path)
    t0 = datetime(2026, 9, 1, 4, 0, 0)
    con.execute(
        "INSERT INTO llm_batches VALUES ('b1','openai','batch_x','spd_v2',"
        "'fc_1',NULL,'pl_1','fc_submit','gpt-5.6-sol','failed',105,0,0,105,105,"
        "'file-in',NULL,NULL,NULL,?,?,?,?,?)",
        [t0, t0 + timedelta(minutes=1), t0 + timedelta(minutes=2), t0 + timedelta(minutes=40),
         json.dumps({"state": "failed", "errors": [{"code": "invalid_request"}]})],
    )
    payload = batch_lifecycle.collect(con, hs_run_id=None, forecaster_run_id="fc_1")
    (record,) = payload["batches"]
    assert record["yielded_nothing"] is True
    assert record["n_fell_back_to_sync"] == 105
    assert record["provider_error_payload"]["errors"][0]["code"] == "invalid_request"
    assert record["input_file_id"] == "file-in"
    # The queue clock is submit -> ended, not submit -> collected: the
    # second includes however long the poller took to come back.
    assert record["queue_wall_seconds"] == 120.0
    assert record["seconds_to_first_poll"] == 60.0
    assert payload["totals"]["n_batches_yielded_nothing"] == 1


def test_lost_discount_counts_only_batchable_families(tmp_path: Path):
    con = _batch_db(tmp_path)
    con.execute(
        "INSERT INTO llm_batches VALUES ('b1','openai','x','spd_v2','fc_1',NULL,'pl_1',"
        "'fc','m','collected',1,1,0,0,0,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL)"
    )
    sync = json.dumps({"prompt_tokens": 100})
    batched = json.dumps({"prompt_tokens": 100, "service_tier": "batch"})
    con.execute(
        "INSERT INTO llm_calls VALUES ('spd_v2','spd_v2','openai','gpt-5.6-sol','fc_1',NULL,10.0,?)",
        [sync],
    )
    con.execute(
        "INSERT INTO llm_calls VALUES ('spd_v2','spd_v2','openai','gpt-5.6-sol','fc_1',NULL,4.0,?)",
        [batched],
    )
    # Scenario writing is never batched by this pipeline, so a synchronous
    # scenario call has lost nothing and must not be counted as a loss.
    con.execute(
        "INSERT INTO llm_calls VALUES ('scenario_v2','scenario_v2','google','gemini','fc_1',NULL,3.0,?)",
        [sync],
    )
    payload = batch_lifecycle.collect(con, hs_run_id=None, forecaster_run_id="fc_1")
    by_phase = {r["phase"]: r for r in payload["cost_by_phase_model"]}
    assert by_phase["spd_v2"]["lost_discount_usd"] == pytest.approx(5.0)
    assert by_phase["scenario_v2"]["lost_discount_usd"] == 0.0
    assert by_phase["scenario_v2"]["batchable_family"] is False
    # Both counterfactuals, both directions.
    assert by_phase["spd_v2"]["counterfactual_all_batch_usd"] == pytest.approx(9.0)
    assert by_phase["spd_v2"]["counterfactual_all_sync_usd"] == pytest.approx(18.0)
    assert payload["totals"]["lost_discount_usd"] == pytest.approx(5.0)


def test_batch_lifecycle_on_a_db_without_the_tables_says_so(tmp_path: Path):
    con = duckdb.connect(str(tmp_path / "empty.duckdb"))
    payload = batch_lifecycle.collect(con, hs_run_id="hs_1", forecaster_run_id=None)
    assert payload["batches"] == []
    assert "llm_batches" in payload["note"]


# ---------------------------------------------------------------------------
# Provider objects
# ---------------------------------------------------------------------------

def test_a_provider_fetch_failure_is_recorded_not_dropped(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def _boom(url, headers, timeout=30.0):
        return {"ok": False, "status_code": 404, "text": "No such batch"}

    monkeypatch.setattr(provider_objects, "_http_get", _boom)
    payload = provider_objects.collect(
        [{"batch_id": "b1", "provider": "openai", "provider_batch_id": "batch_x"}]
    )
    (entry,) = payload["objects"]
    # The status code and body are the difference between "it expired",
    # "the key is wrong" and "the network broke".
    assert entry["batch_object"]["status_code"] == 404
    assert "No such batch" in entry["batch_object"]["text"]


def test_a_batch_that_never_reached_the_provider_is_not_a_404():
    payload = provider_objects.collect(
        [{"batch_id": "b1", "provider": "openai", "provider_batch_id": None}]
    )
    assert "never reached the provider" in payload["objects"][0]["error"]


# ---------------------------------------------------------------------------
# Workflow logs
# ---------------------------------------------------------------------------

def test_an_oversized_log_is_cut_from_the_middle_and_says_so():
    text = ("head\n" * 200) + ("x" * 6_000_000) + ("tail\n" * 200)
    out = workflow_logs.truncate_middle(text, limit=1000)
    assert len(out.encode("utf-8")) < 2000
    assert out.startswith("head")
    assert out.rstrip().endswith("tail")
    assert "TRUNCATED" in out


def test_a_failed_log_fetch_writes_a_stub_and_keeps_going(tmp_path: Path, monkeypatch):
    class _Api(workflow_logs.GitHubApi):
        def get_json(self, path, params=None):
            return {"workflow_runs": []}

    monkeypatch.setenv("GITHUB_RUN_ID", "999")

    def _explode(run_id: str) -> bytes:
        raise RuntimeError("403 Forbidden for token ghp_abcdefghijklmnopqrstuvwx")

    index = workflow_logs.collect(
        tmp_path,
        pipeline_id="pl_1",
        api=_Api("t", "o/r"),
        fetch_logs=_explode,
    )
    (entry,) = index["runs"]
    assert entry["ok"] is False
    stub = (tmp_path / "workflow_logs" / Path(entry["file"]).name).read_text()
    assert "403 Forbidden" in stub
    # A credential echoed into a log is the one thing that must never reach
    # the artifact — including when the log is our own error message.
    assert "ghp_abcdefghijklmnopqrstuvwx" not in stub
    assert "ghp_abcdefghijklmnopqrstuvwx" not in entry["error"]


# ---------------------------------------------------------------------------
# Connector freshness
# ---------------------------------------------------------------------------

def _crisiswatch_db(tmp_path: Path):
    con = duckdb.connect(str(tmp_path / "cw.duckdb"))
    con.execute(
        """
        CREATE TABLE crisiswatch_entries (
            iso3 TEXT, arrow TEXT, alert_type TEXT, year INTEGER, month INTEGER
        )
        """
    )
    return con


def test_crisiswatch_reports_every_edition_not_only_the_latest(tmp_path: Path):
    con = _crisiswatch_db(tmp_path)
    # The September shape: February and June, and nothing between. "Latest
    # 2026-06" hid a four-month gap.
    con.execute("INSERT INTO crisiswatch_entries VALUES ('SOM','deteriorated','conflict_risk',2026,2)")
    con.execute("INSERT INTO crisiswatch_entries VALUES ('ETH','unchanged','',2026,6)")
    con.execute("INSERT INTO crisiswatch_entries VALUES ('SDN','improved','',2026,6)")
    detail = connector_freshness.crisiswatch_detail(con, questions=[], today=datetime(2026, 9, 1).date())
    assert detail["n_editions"] == 2
    assert [e["edition"] for e in detail["editions_present"]] == ["2026-02", "2026-06"]
    assert detail["edition_span"] == "2026-02 .. 2026-06"
    assert detail["arrow_counts"] == {"deteriorated": 1, "unchanged": 1, "improved": 1}
    assert detail["n_alerts"] == 1


def test_crisiswatch_names_the_ace_questions_forecast_without_an_arrow(tmp_path: Path):
    con = _crisiswatch_db(tmp_path)
    con.execute("INSERT INTO crisiswatch_entries VALUES ('SOM','deteriorated','',2026,6)")
    questions = [
        {"iso3": "SOM", "hazard_code": "ACE", "question_id": "SOM_ACE_PA_2026-10"},
        {"iso3": "TCD", "hazard_code": "ACE", "question_id": "TCD_ACE_PA_2026-10"},
        {"iso3": "TCD", "hazard_code": "ACE", "question_id": "TCD_ACE_FATALITIES_2026-10"},
        {"iso3": "PAK", "hazard_code": "FL", "question_id": "PAK_FL_PA_2026-10"},
    ]
    detail = connector_freshness.crisiswatch_detail(con, questions=questions)
    assert detail["ace_countries_forecast"] == 2
    assert detail["ace_countries_without_crisiswatch_row"] == 1
    assert detail["ace_countries_missing_iso3s"] == ["TCD"]
    # Two ACE questions for Chad went out with no conflict arrow in them.
    assert detail["n_ace_questions_without_crisiswatch"] == 2


def test_freshness_measures_the_observation_not_the_fetch(tmp_path: Path):
    con = duckdb.connect(str(tmp_path / "f.duckdb"))
    con.execute(
        "CREATE TABLE conflict_forecasts (iso3 TEXT, forecast_issue_date DATE, fetched_at TIMESTAMP)"
    )
    # Re-fetched today, but the vintage it carries is nine months old: this
    # is the ACLED CAST shape, and measuring from fetched_at reads it fresh.
    con.execute(
        "INSERT INTO conflict_forecasts VALUES ('SOM', DATE '2025-12-01', TIMESTAMP '2026-09-01 00:00:00')"
    )
    rows, _ = connector_freshness.collect(
        con, countries=["SOM"], run_start=datetime(2026, 9, 1).date()
    )
    row = next(r for r in rows if r["source"] == "Conflict forecasts")
    assert row["observation_column"] == "forecast_issue_date"
    assert row["age_days_at_run_start"] == 274
    assert row["status"] == "stale"
    # And the reader is told the prompt does label it.
    assert row["prompt_staleness_warning"] == "yes"


def test_every_source_resolves_a_real_observation_column(tmp_path: Path):
    """The freshness table is only useful if it reads the right column.

    A wrong column name does not raise: the collector falls through to
    fetched_at, or to nothing, and the row reads fresh or absent. That is
    the failure shape hardest to notice, so the real shapes are pinned
    here — reliefweb_reports dates rows with published_date, not date;
    acled_monthly_fatalities with month, not ym; the ACAPS tables with
    snapshot_date and entry_date; and crisiswatch_entries with no date
    column at all, only its (year, month) edition.
    """

    pythia_schema = pytest.importorskip("pythia.db.schema")
    con = duckdb.connect(str(tmp_path / "real.duckdb"))
    pythia_schema.ensure_schema(con)
    # These two are created by their writers, not by ensure_schema.
    con.execute(
        """
        CREATE TABLE reliefweb_reports (
            iso3 VARCHAR, report_id INTEGER, title VARCHAR, published_date VARCHAR,
            sources VARCHAR, disaster_types VARCHAR, themes VARCHAR,
            body_excerpt VARCHAR, url VARCHAR, fetched_at VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE acled_monthly_fatalities (
            iso3 VARCHAR, month DATE, fatalities BIGINT, source VARCHAR,
            updated_at TIMESTAMP
        )
        """
    )
    con.execute(
        "INSERT INTO reliefweb_reports VALUES "
        "('SOM',1,'t','2026-08-15T00:00:00+00:00','s','d','th','b','u','2026-09-01')"
    )
    con.execute("INSERT INTO acled_monthly_fatalities VALUES ('SOM', DATE '2026-07-01', 5, 'ACLED', now())")
    con.execute(
        "INSERT INTO acaps_inform_severity (iso3, crisis_id, snapshot_date, severity_score) "
        "VALUES ('SOM','c1', DATE '2026-06-01', 4.0)"
    )
    con.execute("INSERT INTO crisiswatch_entries (iso3, month, year, arrow) VALUES ('SOM',6,2026,'deteriorated')")
    con.execute(
        "INSERT INTO gdelt_conflict_indicators (iso3, event_date, total_events) "
        "VALUES ('SOM', DATE '2026-08-30', 10)"
    )
    con.execute(
        "INSERT INTO enso_state (fetch_date, observation_date, enso_phase) "
        "VALUES (DATE '2026-09-01', DATE '2026-08-20', 'El Nino')"
    )

    rows, _ = connector_freshness.collect(
        con, countries=["SOM"], run_start=datetime(2026, 9, 1).date()
    )
    by_source = {r["source"]: r for r in rows}
    expected_columns = {
        "ACLED fatalities": "month",
        "ReliefWeb": "published_date",
        "ACAPS INFORM Severity": "snapshot_date",
        "GDELT": "event_date",
        "ENSO": "observation_date",
        "CrisisWatch": "edition (year, month)",
    }
    for source, column in expected_columns.items():
        row = by_source[source]
        assert row["observation_column"] == column, source
        assert row["age_days_at_run_start"] != "", source
    # And the ages are the observations', not today's.
    assert by_source["ReliefWeb"]["age_days_at_run_start"] == 17
    assert by_source["ACLED fatalities"]["age_days_at_run_start"] == 62
    assert by_source["CrisisWatch"]["age_days_at_run_start"] == 62


# ---------------------------------------------------------------------------
# Prompt cache
# ---------------------------------------------------------------------------

def test_cache_report_separates_a_provider_that_cannot_cache_from_one_that_did_not(tmp_path: Path):
    con = duckdb.connect(str(tmp_path / "pc.duckdb"))
    con.execute(
        "CREATE TABLE llm_calls (phase TEXT, provider TEXT, model_id TEXT, run_id TEXT, usage_json TEXT)"
    )
    con.execute(
        "INSERT INTO llm_calls VALUES ('spd_v2','google','gemini-3.1-pro','fc_1',?)",
        [json.dumps({"prompt_tokens": 1000, "service_tier": "batch"})],
    )
    con.execute(
        "INSERT INTO llm_calls VALUES ('spd_v2','anthropic','claude-opus-5','fc_1',?)",
        [json.dumps({"prompt_tokens": 1000, "cache_read_input_tokens": 400})],
    )
    rows = prompt_cache.collect(con, predicate="run_id = ?", params=["fc_1"])
    by_provider = {r["provider"]: r for r in rows}
    assert by_provider["anthropic"]["cache_hit_rate_pct"] == 40.0
    assert by_provider["google"]["cache_hit_rate_pct"] == 0.0
    # Gemini batch mode is not eligible for the implicit cache; that is the
    # provider's design, not a defect, and the row must not read as one.
    assert "not eligible" in by_provider["google"]["note"]
    summary = prompt_cache.summarise(rows)
    assert summary["cache_hit_rate_pct"] == 20.0


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------

def test_retry_rows_count_attempts_backoff_and_reason_class(tmp_path: Path):
    con = duckdb.connect(str(tmp_path / "r.duckdb"))
    con.execute(
        "CREATE TABLE llm_calls (phase TEXT, provider TEXT, model_id TEXT, run_id TEXT, "
        "error_text TEXT, usage_json TEXT)"
    )
    con.execute(
        "INSERT INTO llm_calls VALUES ('spd_v2','openai','gpt-5.6-sol','fc_1','429 rate limit',?)",
        [json.dumps({"attempts_used": 3, "backoffs_sec": [1.0, 2.5]})],
    )
    con.execute(
        "INSERT INTO llm_calls VALUES ('spd_v2','openai','gpt-5.6-sol','fc_1',NULL,?)",
        [json.dumps({"attempts_used": 1, "backoffs_sec": []})],
    )
    (row,) = retry_report.collect(con, predicate="run_id = ?", params=["fc_1"])
    assert row["n_calls"] == 2
    assert row["n_attempts"] == 4
    assert row["n_calls_needing_retry"] == 1
    assert row["n_extra_attempts"] == 2
    assert row["total_backoff_sec"] == 3.5
    assert row["n_rate_limit"] == 1


def test_a_brave_breaker_trip_is_countable_from_its_sentinel(tmp_path: Path):
    con = duckdb.connect(str(tmp_path / "r2.duckdb"))
    con.execute("CREATE TABLE llm_calls (model_id TEXT, hs_run_id TEXT)")
    con.execute("INSERT INTO llm_calls VALUES ('grounding-breaker-tripped','hs_1')")
    con.execute("INSERT INTO llm_calls VALUES ('grounding-breaker-tripped','hs_1')")
    con.execute("INSERT INTO llm_calls VALUES ('grounding-failed','hs_1')")
    summary = retry_report.breaker_summary(con, predicate="hs_run_id = ?", params=["hs_1"])
    assert summary["brave_breaker_short_circuits"] == 2
    assert summary["no_backend_calls"] == 3


# ---------------------------------------------------------------------------
# Model completeness
# ---------------------------------------------------------------------------

def _forecasts_db(tmp_path: Path):
    con = duckdb.connect(str(tmp_path / "fc.duckdb"))
    con.execute(
        """
        CREATE TABLE forecasts_raw (
            run_id TEXT, question_id TEXT, model_name TEXT,
            month_index INTEGER, bucket_index INTEGER, status TEXT
        )
        """
    )
    return con


def test_a_member_that_wrote_no_forecast_is_visible_per_month(tmp_path: Path):
    con = _forecasts_db(tmp_path)
    for model in ("model-a", "model-b"):
        for month in range(1, 7):
            for bucket in range(1, 7):
                con.execute(
                    "INSERT INTO forecasts_raw VALUES ('fc_1','Q1',?,?,?,'ok')",
                    [model, month, bucket],
                )
    # The 2026-09-01 signature: an ok call, a parse failure, one
    # no_forecast row, and a health view still reporting 5/5.
    con.execute("INSERT INTO forecasts_raw VALUES ('fc_1','Q1','model-c',NULL,NULL,'no_forecast')")
    questions = [{"question_id": "Q1", "iso3": "SOM", "hazard_code": "ACE", "metric": "PA", "track": 1}]
    rows, rollup = model_completeness.collect(
        con,
        run_id="fc_1",
        questions=questions,
        expected_models=["model-a", "model-b", "model-c"],
        track2_model="track2_flash",
    )
    assert len(rows) == 18  # three models x six months
    missing = [r for r in rows if r["verdict"] != "ok"]
    assert len(missing) == 6
    assert {r["model_name"] for r in missing} == {"model-c"}
    assert rollup["n_cells_missing"] == 6
    assert rollup["by_model"] == {"model-c": 6}
    # Every one of the six months was aggregated from two members, not three.
    assert rollup["n_question_months_short"] == 6


def test_a_wrong_length_bucket_vector_counts_as_missing(tmp_path: Path):
    con = _forecasts_db(tmp_path)
    for month in range(1, 7):
        # Five buckets under the post-restructure six-bucket PA scheme: the
        # SPD chain rejects wrong-length vectors, so this is not a forecast.
        for bucket in range(1, 6):
            con.execute(
                "INSERT INTO forecasts_raw VALUES ('fc_1','Q1','model-a',?,?,'ok')", [month, bucket]
            )
    rows, rollup = model_completeness.collect(
        con,
        run_id="fc_1",
        questions=[{"question_id": "Q1", "metric": "PA", "track": 1}],
        expected_models=["model-a"],
        track2_model="track2_flash",
    )
    assert {r["verdict"] for r in rows} == {"wrong_bucket_count"}
    assert rollup["n_cells_missing"] == 6


# ---------------------------------------------------------------------------
# Prompt prefix deduplication
# ---------------------------------------------------------------------------

def test_a_shared_prefix_is_stored_once_and_the_prompt_still_reconstructs():
    prefix = "ROLE AND TASK\n" + ("You are a forecaster. " * 40) + "\n"
    rows = [
        {"phase": "spd_v2", "call_type": "spd_v2", "provider": "openai",
         "model_id": "m", "hazard_code": "ACE", "metric": "PA",
         "prompt_text": prefix + f"COUNTRY: {iso3}\n"}
        for iso3 in ("SOM", "ETH", "SDN")
    ]
    prefixes, assignment = prompt_prefixes.build_prefix_index(rows)
    assert len(prefixes) == 1
    saved = prompt_prefixes.savings(prefixes)
    assert saved["prefix_chars_avoided"] > 0

    for row in rows:
        record = prompt_prefixes.apply(dict(row), row, assignment)
        assert record["prompt_is_complete"] is False
        rebuilt = prefixes[record["prompt_prefix_sha256"]]["text"] + record["prompt_text"]
        assert rebuilt == row["prompt_text"]


def test_prompts_from_different_builders_never_share_a_prefix():
    shared = "COMMON HEADER\n" + ("x" * 500) + "\n"
    rows = [
        {"phase": "spd_v2", "call_type": "spd_v2", "provider": "openai", "model_id": "m",
         "hazard_code": "ACE", "metric": "PA", "prompt_text": shared + "a"},
        {"phase": "hs_triage", "call_type": "rc_pass_1", "provider": "google", "model_id": "g",
         "hazard_code": "ACE", "metric": "", "prompt_text": shared + "b"},
    ]
    prefixes, assignment = prompt_prefixes.build_prefix_index(rows)
    # One prompt each in two groups: cutting on text they happen to share
    # would put question-specific content into a "static" block.
    assert prefixes == {} and assignment == {}


# ---------------------------------------------------------------------------
# Anomalies
# ---------------------------------------------------------------------------

def test_anomalies_lead_with_failures_and_name_their_evidence():
    entries = anomalies_mod.build(
        health_checks=[
            {"subsystem": "LLM Calls", "status": "OK", "detail": "clean"},
            {"subsystem": "CrisisWatch", "status": "WARN", "detail": "stale"},
        ],
        batch_lifecycle={
            "batches": [
                {
                    "batch_id": "b1", "provider": "openai", "phase": "spd_v2",
                    "provider_state": "failed", "n_requests": 105, "yielded_nothing": True,
                }
            ],
            "totals": {"fallback_pct_of_requests": 100.0, "lost_discount_usd": 5.4},
        },
        file_names={"batch_lifecycle": "batch_lifecycle__fc_1.json"},
    )
    severities = [e["severity"] for e in entries]
    assert severities == sorted(severities, key=lambda s: {"fail": 0, "warn": 1, "info": 2}[s])
    fail = next(e for e in entries if e["severity"] == "fail")
    assert fail["evidence_file"] == "batch_lifecycle__fc_1.json"
    # An OK check is not an anomaly.
    assert not any(e["description"] == "clean" for e in entries)


def test_a_clean_run_produces_no_anomalies():
    assert anomalies_mod.build(health_checks=[{"subsystem": "x", "status": "OK", "detail": "y"}]) == []


# ---------------------------------------------------------------------------
# Code snapshot
# ---------------------------------------------------------------------------

def test_the_previous_production_sha_comes_from_the_db_not_a_guess(tmp_path: Path):
    con = duckdb.connect(str(tmp_path / "hs.duckdb"))
    con.execute(
        "CREATE TABLE hs_runs (hs_run_id TEXT, generated_at TIMESTAMP, git_sha TEXT, is_test BOOLEAN)"
    )
    con.execute("INSERT INTO hs_runs VALUES ('hs_20260801T000000', TIMESTAMP '2026-08-01', 'aaa', FALSE)")
    # A test run's commit is not the baseline any regression is measured
    # against, so it must never be picked as the previous production run.
    con.execute("INSERT INTO hs_runs VALUES ('hs_20260815T000000', TIMESTAMP '2026-08-15', 'bbb', TRUE)")
    con.execute("INSERT INTO hs_runs VALUES ('hs_20260901T000000', TIMESTAMP '2026-09-01', 'ccc', FALSE)")
    sha, run = code_snapshot.previous_production_git_sha(con, "hs_20260901T000000")
    assert (sha, run) == ("aaa", "hs_20260801T000000")


def test_code_snapshot_copies_the_named_files(tmp_path: Path):
    index = code_snapshot.collect(tmp_path, repo_root=Path(__file__).resolve().parents[2])
    copied = {entry["source"] for entry in index["files"]}
    assert "pythia/llm_batch.py" in copied
    assert "horizon_scanner/crisiswatch.py" in copied
    assert index["problems"] == []
    assert (tmp_path / "code_snapshot" / "commits_since_last_production_run.txt").exists()


# ---------------------------------------------------------------------------
# Manifest and zip
# ---------------------------------------------------------------------------

def test_every_file_the_manifest_names_is_in_the_zip(tmp_path: Path):
    from scripts.dump_pythia_debug_bundle import build_bundle_zips

    (tmp_path / "executive_summary__fc_1.md").write_text("# summary\n")
    (tmp_path / "question_metrics__fc_1.csv").write_text("a,b\n1,2\n")
    (tmp_path / "workflow_logs").mkdir()
    (tmp_path / "workflow_logs" / "poller__1.txt").write_text("log\n")
    (tmp_path / "code_snapshot").mkdir()
    (tmp_path / "code_snapshot" / "pythia__llm_batch.py").write_text("code\n")

    manifest = manifest_mod.build(
        tmp_path, hs_run_id="hs_1", forecaster_run_id="fc_1", pipeline_id="pl_1"
    )
    manifest_mod.write(tmp_path, manifest)
    result = build_bundle_zips(tmp_path, "fc_1")
    manifest_mod.annotate_archives(manifest, result)

    with zipfile.ZipFile(tmp_path / result["bundle_zip"]) as zf:
        names = set(zf.namelist())
    for entry in manifest["files"]:
        assert entry["archive"] == result["bundle_zip"]
        assert entry["name"] in names, f"{entry['name']} is named in the manifest but not in the zip"
    # The manifest must travel inside the bundle, not only beside it.
    assert "BUNDLE_MANIFEST.json" in names
    # Subdirectories keep their path; root files stay flat.
    assert "workflow_logs/poller__1.txt" in names
    assert "question_metrics__fc_1.csv" in names


def test_manifest_counts_records_and_describes_each_file(tmp_path: Path):
    (tmp_path / "question_metrics__fc_1.csv").write_text("a,b\n1,2\n3,4\n")
    manifest = manifest_mod.build(
        tmp_path, hs_run_id=None, forecaster_run_id="fc_1", pipeline_id=None
    )
    (entry,) = manifest["files"]
    assert entry["records"] == 2
    assert entry["description"]
    assert manifest["bundle_schema_version"] == manifest_mod.BUNDLE_SCHEMA_VERSION


def test_an_oversized_bundle_splits_the_logs_out_rather_than_dropping_them(tmp_path: Path):
    from scripts.dump_pythia_debug_bundle import build_bundle_zips

    (tmp_path / "executive_summary__fc_1.md").write_text("# summary\n")
    (tmp_path / "workflow_logs").mkdir()
    # Incompressible, so it survives DEFLATE and actually breaks the target.
    import os as _os

    (tmp_path / "workflow_logs" / "big.txt").write_bytes(_os.urandom(200_000))
    result = build_bundle_zips(tmp_path, "fc_1", size_target=50_000)
    assert result["split"] is True
    # A split bundle still names the logs in the manifest, and each entry
    # says which archive holds it — otherwise a reader looking for a log
    # concludes the collector failed rather than opening the second one.
    manifest = manifest_mod.annotate_archives(
        manifest_mod.build(tmp_path, hs_run_id=None, forecaster_run_id="fc_1", pipeline_id=None),
        result,
    )
    archives = {e["name"]: e["archive"] for e in manifest["files"]}
    assert archives["workflow_logs/big.txt"] == result["workflow_logs_zip"]
    assert archives["executive_summary__fc_1.md"] == result["bundle_zip"]
    with zipfile.ZipFile(tmp_path / result["bundle_zip"]) as zf:
        assert "workflow_logs/big.txt" not in zf.namelist()
    with zipfile.ZipFile(tmp_path / result["workflow_logs_zip"]) as zf:
        assert "workflow_logs/big.txt" in zf.namelist()
    assert "Nothing was dropped" in result["split_note"]


# ---------------------------------------------------------------------------
# A failing collector must never fail the phase
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("suffix", [".json", ".csv"])
def test_a_failing_collector_writes_a_stub_and_the_run_continues(tmp_path: Path, suffix: str):
    from scripts.dump_pythia_debug_bundle import _COLLECTOR_ERRORS, _run_collector

    _COLLECTOR_ERRORS.clear()
    path = tmp_path / f"thing{suffix}"

    def _boom():
        raise RuntimeError("upstream went away")

    assert _run_collector("thing", path, _boom) is None
    assert path.exists()
    text = path.read_text()
    assert "upstream went away" in text
    assert _COLLECTOR_ERRORS and _COLLECTOR_ERRORS[0]["collector"] == "thing"
    _COLLECTOR_ERRORS.clear()
