# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""/v1/sibyl/* routes against current and pre-October-2026 schemas.

The API serves whatever DB the release carries, so a column added to
sibyl_runs / sibyl_forecasts (time_capped, selection_pass) must read as its
neutral value on an older DB, never as a 500.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
duckdb = pytest.importorskip("duckdb")
yaml = pytest.importorskip("yaml")

from fastapi.testclient import TestClient  # noqa: E402

from pythia import config as pythia_config  # noqa: E402


def _client(tmp_path: Path, db_path: Path, monkeypatch) -> TestClient:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"app": {"db_url": f"duckdb:///{db_path}"}}))
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(cfg))
    monkeypatch.setenv("PYTHIA_DATA_BACKGROUND_SYNC", "0")
    pythia_config.load.cache_clear()
    import pythia.api.app as app_mod

    app_mod._READ_CON = None
    return TestClient(app_mod.app)


@pytest.fixture()
def reset_api():
    yield
    import pythia.api.app as app_mod

    app_mod._READ_CON = None
    pythia_config.load.cache_clear()


def _legacy_db(path: Path) -> None:
    """sibyl tables as they stood before time_capped / selection_pass."""
    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE sibyl_runs (
            sibyl_run_id TEXT, hs_run_id TEXT, as_of DATE, model TEXT, k INTEGER,
            max_steps INTEGER, aggregation TEXT, run_hard_cap_usd DOUBLE,
            budget_capped BOOLEAN, run_cost_usd DOUBLE, opus_cost_usd DOUBLE,
            brave_cost_usd DOUBLE, n_selected INTEGER, n_forecast INTEGER,
            n_skipped INTEGER, config_json TEXT, created_at TIMESTAMP,
            is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        """
        CREATE TABLE sibyl_forecasts (
            sibyl_run_id TEXT, run_id TEXT, question_id TEXT, iso3 TEXT,
            hazard_code TEXT, metric TEXT, track TEXT, status TEXT,
            skip_reason TEXT, as_of DATE, k INTEGER, aggregation TEXT,
            volatility_score DOUBLE, triage_score DOUBLE,
            pooled_quantiles_json TEXT, trials_json TEXT, bucket_probs_json TEXT,
            js_divergence_vs_standard DOUBLE, js_divergence_inter_trial DOUBLE,
            cost_usd DOUBLE, opus_cost_usd DOUBLE, brave_cost_usd DOUBLE,
            leakage_json TEXT, created_at TIMESTAMP, is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    con.execute(
        "INSERT INTO sibyl_runs VALUES ('sib_1','hs_1',DATE '2026-09-01','m',3,10,"
        "'linear_pool',40,FALSE,5,4,1,10,10,0,'{}',TIMESTAMP '2026-09-01 06:00',FALSE)"
    )
    con.execute(
        "INSERT INTO sibyl_forecasts (sibyl_run_id, question_id, iso3, hazard_code,"
        " metric, status, volatility_score, created_at, is_test) VALUES"
        " ('sib_1','ETH_ACE_FATALITIES_2026-10','ETH','ACE','FATALITIES','ok',0.4,"
        " TIMESTAMP '2026-09-01 06:00', FALSE)"
    )
    con.close()


def test_legacy_schema_reads_neutral_values(tmp_path, monkeypatch, reset_api):
    db = tmp_path / "legacy.duckdb"
    _legacy_db(db)
    c = _client(tmp_path, db, monkeypatch)

    summary = c.get("/v1/sibyl/summary").json()
    assert summary["run"]["time_capped"] is False
    assert summary["questions"][0]["selection_pass"] is None

    rows = c.get("/v1/sibyl/questions").json()["rows"]
    assert rows[0]["selection_pass"] is None


def test_current_schema_serves_time_capped_and_selection_pass(tmp_path, monkeypatch, reset_api):
    db = tmp_path / "current.duckdb"
    _legacy_db(db)
    con = duckdb.connect(str(db))
    con.execute("ALTER TABLE sibyl_runs ADD COLUMN time_capped BOOLEAN DEFAULT FALSE")
    con.execute("ALTER TABLE sibyl_forecasts ADD COLUMN selection_pass TEXT")
    con.execute("UPDATE sibyl_runs SET time_capped = TRUE")
    con.execute("UPDATE sibyl_forecasts SET selection_pass = 'floor'")
    con.close()
    c = _client(tmp_path, db, monkeypatch)

    summary = c.get("/v1/sibyl/summary").json()
    assert summary["run"]["time_capped"] is True
    assert summary["questions"][0]["selection_pass"] == "floor"
    assert c.get("/v1/sibyl/questions").json()["rows"][0]["selection_pass"] == "floor"


def test_calibration_endpoint_without_the_table(tmp_path, monkeypatch, reset_api):
    db = tmp_path / "legacy.duckdb"
    _legacy_db(db)
    c = _client(tmp_path, db, monkeypatch)
    r = c.get("/v1/sibyl/calibration")
    assert r.status_code == 200
    assert r.json()["has_advice_table"] is False and r.json()["rows"] == []


def test_calibration_endpoint_serves_the_newest_month(tmp_path, monkeypatch, reset_api):
    import json

    from sibyl.advice import build_rows
    from tests.test_sibyl_advice import _records

    db = tmp_path / "advice.duckdb"
    _legacy_db(db)
    con = duckdb.connect(str(db))
    from pythia.db.schema import ensure_sibyl_calibration_advice_table

    ensure_sibyl_calibration_advice_table(con)
    for month, scale, n in (("2026-10", 1.0, 6), ("2026-11", 0.2, 24)):
        for row in build_rows(_records(n, scale=scale), {}, as_of_month=month):
            con.execute(
                "INSERT INTO sibyl_calibration_advice VALUES (?, ?, ?, ?, ?, ?, ?, 'v', now())",
                [row["as_of_month"], row["hazard_code"], row["metric"], row["scope"],
                 row["n_questions"], row["advice"], json.dumps(row["findings"])],
            )
    con.close()
    c = _client(tmp_path, db, monkeypatch)

    body = c.get("/v1/sibyl/calibration").json()
    assert body["as_of_month"] == "2026-11" and body["months"] == ["2026-11", "2026-10"]
    group = next(r for r in body["rows"] if r["scope"] == "group")
    assert group["n_questions"] == 24 and group["advice"]
    assert group["diagnostics"]["coverage_10_90"]["n_questions"] == 24
    assert body["arm_comparison"]["status"] == "not yet"
    # Failure types (Oct 2026): absent from a generation written before.
    assert body["failure_types"] is None

    old = c.get("/v1/sibyl/calibration?as_of_month=2026-10").json()
    assert old["as_of_month"] == "2026-10"
    assert next(r for r in old["rows"] if r["scope"] == "group")["gate"] == "6 of 20 resolved questions"
    assert c.get("/v1/sibyl/calibration?as_of_month=bad").status_code == 422


def test_process_measures_and_month_vectors_on_both_schemas(tmp_path, monkeypatch, reset_api):
    """Oct 2026, Part 6: process measures read None on an older DB, and the
    question detail parses the reference and the published vectors by month."""
    import json

    db = tmp_path / "legacy.duckdb"
    _legacy_db(db)
    c = _client(tmp_path, db, monkeypatch)
    run = c.get("/v1/sibyl/summary").json()["run"]
    assert run["docs_per_trial"] is None and run["reference_weight"] is None
    detail = c.get("/v1/sibyl/question_detail",
                   params={"question_id": "ETH_ACE_FATALITIES_2026-10"}).json()
    assert detail["record"]["reference"] is None

    con = duckdb.connect(str(db))
    for col, typ in (("docs_per_trial", "DOUBLE"), ("reference_weight", "DOUBLE"),
                     ("reference_weight_source", "TEXT")):
        con.execute(f"ALTER TABLE sibyl_runs ADD COLUMN {col} {typ}")
    for col in ("reference_json", "final_by_month_json"):
        con.execute(f"ALTER TABLE sibyl_forecasts ADD COLUMN {col} TEXT")
    con.execute("UPDATE sibyl_runs SET docs_per_trial = 2.5, reference_weight = 0.75, "
                "reference_weight_source = 'fitted'")
    con.execute("UPDATE sibyl_forecasts SET reference_json = ?, final_by_month_json = ?",
                [json.dumps({"by_month": {"1": [0.5, 0.5], "6": [0.4, 0.6]}}),
                 json.dumps({"1": [0.6, 0.4], "6": [0.3, 0.7]})])
    con.close()
    import pythia.api.app as app_mod

    app_mod._READ_CON = None
    run = c.get("/v1/sibyl/summary").json()["run"]
    assert (run["docs_per_trial"], run["reference_weight"], run["reference_weight_source"]) == (
        2.5, 0.75, "fitted")
    rec = c.get("/v1/sibyl/question_detail",
                params={"question_id": "ETH_ACE_FATALITIES_2026-10"}).json()["record"]
    assert rec["reference"]["by_month"]["6"] == [0.4, 0.6]
    assert rec["final_by_month"]["1"] == [0.6, 0.4]


def test_summary_carries_the_shadow_arm_on_both_schemas(tmp_path, monkeypatch, reset_api):
    """Oct 2026, Part 7: the run's shadow fields read None on an older DB and
    the comparison says "not yet" rather than failing."""
    import json

    db = tmp_path / "legacy.duckdb"
    _legacy_db(db)
    c = _client(tmp_path, db, monkeypatch)
    body = c.get("/v1/sibyl/summary").json()
    assert body["run"]["shadow_status"] is None and body["run"]["shadow_cost_usd"] is None
    assert body["shadow"]["series"]["brier"]["status"] == "not_yet"

    con = duckdb.connect(str(db))
    con.execute("ALTER TABLE sibyl_runs ADD COLUMN shadow_status TEXT")
    con.execute("ALTER TABLE sibyl_forecasts ADD COLUMN shadow_json TEXT")
    con.execute("UPDATE sibyl_runs SET shadow_status = 'no_key'")
    con.execute("UPDATE sibyl_forecasts SET shadow_json = ?",
                [json.dumps({"status": "ok", "model": "openai:gpt-6-sol"})])
    con.close()
    import pythia.api.app as app_mod

    app_mod._READ_CON = None
    body = c.get("/v1/sibyl/summary").json()
    assert body["run"]["shadow_status"] == "no_key"
    assert body["shadow"]["model"] == "openai:gpt-6-sol"
    assert body["shadow"]["series"]["brier"] == {
        "status": "not_yet", "n_questions": 0, "min_questions": 20}
