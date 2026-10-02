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
