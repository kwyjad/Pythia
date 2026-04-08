# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for /v1/diagnostics/run_summary endpoint and PA metric KPI bug fix.

Tests:
- run_summary endpoint returns correct shape and counts
- PA metric FORECASTS count is non-zero when PA questions exist (regression)
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
import yaml

fastapi = pytest.importorskip("fastapi")
duckdb = pytest.importorskip("duckdb")

from fastapi.testclient import TestClient

from pythia import config as pythia_config
from pythia.api.app import app


def _write_config(tmp_path: Path, db_path: Path) -> Path:
    cfg = {"app": {"db_url": f"duckdb:///{db_path}"}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    db_path = tmp_path / "api-run-summary.duckdb"
    con = duckdb.connect(str(db_path), read_only=False)

    # --- tables ---
    con.execute("""
        CREATE TABLE questions (
            question_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            target_month TEXT,
            metric TEXT,
            hs_run_id TEXT,
            status TEXT,
            track INTEGER,
            is_test BOOLEAN DEFAULT FALSE
        )
    """)
    con.execute("""
        CREATE TABLE hs_triage (
            run_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            tier TEXT,
            triage_score DOUBLE,
            need_full_spd BOOLEAN,
            regime_change_likelihood DOUBLE,
            regime_change_level INTEGER,
            track INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_test BOOLEAN DEFAULT FALSE
        )
    """)
    con.execute("""
        CREATE TABLE forecasts_ensemble (
            question_id TEXT,
            run_id TEXT,
            month_index INTEGER,
            bucket_index INTEGER,
            probability DOUBLE,
            model_name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_test BOOLEAN DEFAULT FALSE
        )
    """)
    con.execute("""
        CREATE TABLE forecasts_raw (
            run_id TEXT,
            question_id TEXT,
            model_name TEXT,
            month_index INTEGER,
            bucket_index INTEGER,
            probability DOUBLE,
            ok BOOLEAN DEFAULT TRUE,
            is_test BOOLEAN DEFAULT FALSE
        )
    """)
    con.execute("""
        CREATE TABLE llm_calls (
            call_id TEXT,
            run_id TEXT,
            hs_run_id TEXT,
            phase TEXT,
            cost_usd DOUBLE,
            total_tokens INTEGER,
            error_text TEXT,
            status TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_test BOOLEAN DEFAULT FALSE
        )
    """)
    con.execute("""
        CREATE TABLE resolutions (
            question_id TEXT,
            horizon_m INTEGER,
            value DOUBLE,
            observed_month TEXT
        )
    """)

    # --- seed data ---
    hs_run = "hs_test_run_1"
    fc_run = "fc_test_run_1"

    # HS triage: 3 countries x 4 hazards, with some seasonal skips
    # Country A (AAA): ACE L2, DR L1, FL L0, TC seasonal skip
    # Country B (BBB): ACE L1, DR L0, FL L0, TC L0
    # Country C (CCC): ACE L0, DR L0, FL seasonal skip, TC seasonal skip
    triage_rows = [
        # (run_id, iso3, hazard, tier, score, need_spd, rc_likelihood, rc_level, track)
        (hs_run, "AAA", "ACE", "priority", 0.8, True, 0.45, 2, 1),
        (hs_run, "AAA", "DR", "priority", 0.6, True, 0.20, 1, 1),
        (hs_run, "AAA", "FL", "quiet", 0.1, False, 0.05, 0, 2),
        (hs_run, "AAA", "TC", "quiet", 0.0, False, None, None, None),   # seasonal skip
        (hs_run, "BBB", "ACE", "priority", 0.5, True, 0.18, 1, 1),
        (hs_run, "BBB", "DR", "quiet", 0.2, False, 0.08, 0, 2),
        (hs_run, "BBB", "FL", "quiet", 0.1, False, 0.04, 0, 2),
        (hs_run, "BBB", "TC", "quiet", 0.1, False, 0.02, 0, 2),
        (hs_run, "CCC", "ACE", "quiet", 0.1, False, 0.03, 0, 2),
        (hs_run, "CCC", "DR", "quiet", 0.1, False, 0.05, 0, 2),
        (hs_run, "CCC", "FL", "quiet", 0.0, False, None, None, None),   # seasonal skip
        (hs_run, "CCC", "TC", "quiet", 0.0, False, None, None, None),   # seasonal skip
    ]
    for row in triage_rows:
        con.execute(
            "INSERT INTO hs_triage (run_id, iso3, hazard_code, tier, triage_score, "
            "need_full_spd, regime_change_likelihood, regime_change_level, track) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            list(row),
        )

    # Questions: mix of metrics
    questions = [
        ("q_aaa_ace_fat", "AAA", "ACE", "FATALITIES", hs_run, 1),
        ("q_aaa_ace_pa", "AAA", "ACE", "PA", hs_run, 1),
        ("q_aaa_dr_ph3", "AAA", "DR", "PHASE3PLUS_IN_NEED", hs_run, 1),
        ("q_aaa_dr_eo", "AAA", "DR", "EVENT_OCCURRENCE", hs_run, 1),
        ("q_aaa_fl_pa", "AAA", "FL", "PA", hs_run, 2),
        ("q_aaa_fl_eo", "AAA", "FL", "EVENT_OCCURRENCE", hs_run, 2),
        ("q_bbb_ace_fat", "BBB", "ACE", "FATALITIES", hs_run, 1),
        ("q_bbb_ace_pa", "BBB", "ACE", "PA", hs_run, 1),
        ("q_bbb_dr_eo", "BBB", "DR", "EVENT_OCCURRENCE", hs_run, 2),
        ("q_bbb_fl_pa", "BBB", "FL", "PA", hs_run, 2),
    ]
    for qid, iso3, hc, metric, hsrun, track in questions:
        con.execute(
            "INSERT INTO questions (question_id, iso3, hazard_code, metric, "
            "target_month, hs_run_id, status, track) VALUES (?, ?, ?, ?, '2026-04', ?, 'active', ?)",
            [qid, iso3, hc, metric, hsrun, track],
        )

    # Forecasts ensemble (link questions to run)
    for qid, *_ in questions:
        con.execute(
            "INSERT INTO forecasts_ensemble (question_id, run_id, month_index, "
            "bucket_index, probability, model_name) VALUES (?, ?, 1, 1, 0.5, 'ensemble_bayesmc_v2')",
            [qid, fc_run],
        )

    # Forecasts raw (for model counting)
    for model in ["model_a", "model_b", "model_c"]:
        con.execute(
            "INSERT INTO forecasts_raw (run_id, question_id, model_name, "
            "month_index, bucket_index, probability, ok) VALUES (?, 'q_aaa_ace_fat', ?, 1, 1, 0.5, TRUE)",
            [fc_run, model],
        )

    # LLM calls
    llm_rows = [
        (fc_run, hs_run, "hs_triage", 1.50, 5000, None, "ok"),
        (fc_run, hs_run, "spd_v2", 3.20, 12000, None, "ok"),
        (fc_run, hs_run, "binary_v2", 0.80, 3000, None, "ok"),
        (fc_run, hs_run, "scenario_v2", 0.10, 500, None, "ok"),
        (fc_run, hs_run, "spd_v2", 0.00, 100, "timeout error", "error"),
    ]
    for run, hs, phase, cost, tokens, err, status in llm_rows:
        con.execute(
            "INSERT INTO llm_calls (run_id, hs_run_id, phase, cost_usd, "
            "total_tokens, error_text, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [run, hs, phase, cost, tokens, err, status],
        )

    con.close()

    config_path = _write_config(tmp_path, db_path)
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    pythia_config.load.cache_clear()

    import pythia.api.app as _app_mod
    _app_mod._READ_CON = None

    try:
        yield
    finally:
        _app_mod._READ_CON = None
        pythia_config.load.cache_clear()


def test_run_summary_shape(api_env: None) -> None:
    """run_summary returns expected shape and correct counts."""
    client = TestClient(app)
    resp = client.get(
        "/v1/diagnostics/run_summary",
        params={"forecaster_run_id": "fc_test_run_1"},
    )
    assert resp.status_code == 200
    data = resp.json()

    # Top-level keys
    assert data["run_id"] == "fc_test_run_1"
    assert data["hs_run_id"] == "hs_test_run_1"
    assert "coverage" in data
    assert "metrics" in data
    assert "rc_assessment" in data
    assert "tracks" in data
    assert "ensemble" in data
    assert "cost" in data
    assert "llm_health" in data


def test_run_summary_coverage(api_env: None) -> None:
    """Coverage funnel counts are correct."""
    client = TestClient(app)
    resp = client.get(
        "/v1/diagnostics/run_summary",
        params={"forecaster_run_id": "fc_test_run_1"},
    )
    data = resp.json()
    cov = data["coverage"]

    assert cov["countries_scanned"] == 3  # AAA, BBB, CCC
    assert cov["hazard_pairs_assessed"] == 9  # 12 total - 3 seasonal skips
    assert cov["seasonal_screenouts"] == 3  # TC for AAA, FL+TC for CCC
    assert cov["total_questions"] == 10
    assert cov["countries_with_forecasts"] == 2  # AAA, BBB
    assert cov["countries_no_questions"] == 1  # CCC


def test_run_summary_metrics(api_env: None) -> None:
    """Metrics breakdown lists correct question counts per metric."""
    client = TestClient(app)
    resp = client.get(
        "/v1/diagnostics/run_summary",
        params={"forecaster_run_id": "fc_test_run_1"},
    )
    metrics = {m["metric"]: m for m in resp.json()["metrics"]}

    assert metrics["FATALITIES"]["questions"] == 2
    assert metrics["PA"]["questions"] == 4
    assert metrics["EVENT_OCCURRENCE"]["questions"] == 3
    assert metrics["PHASE3PLUS_IN_NEED"]["questions"] == 1


def test_run_summary_rc_assessment(api_env: None) -> None:
    """RC assessment levels and by-hazard breakdown are correct."""
    client = TestClient(app)
    resp = client.get(
        "/v1/diagnostics/run_summary",
        params={"forecaster_run_id": "fc_test_run_1"},
    )
    rc = resp.json()["rc_assessment"]

    assert rc["total_assessed"] == 9
    assert rc["levels"]["L0"] == 6
    assert rc["levels"]["L1"] == 2  # AAA/DR, BBB/ACE
    assert rc["levels"]["L2"] == 1  # AAA/ACE
    assert rc["levels"]["L3"] == 0

    # L1+ rate = 3/9
    assert abs(rc["l1_plus_rate"] - 3 / 9) < 0.01

    # By hazard
    by_haz = {h["hazard_code"]: h for h in rc["by_hazard"]}
    assert by_haz["ACE"]["L2"] == 1
    assert by_haz["ACE"]["L1"] == 1
    assert by_haz["DR"]["L1"] == 1

    # Countries by level: L1 = AAA + BBB (2), L2 = AAA (1)
    assert rc["countries_by_level"]["L1"] >= 2
    assert rc["countries_by_level"]["L2"] >= 1


def test_run_summary_tracks(api_env: None) -> None:
    """Track split shows correct question counts."""
    client = TestClient(app)
    resp = client.get(
        "/v1/diagnostics/run_summary",
        params={"forecaster_run_id": "fc_test_run_1"},
    )
    tracks = resp.json()["tracks"]

    assert tracks["track1"]["questions"] == 6
    assert tracks["track2"]["questions"] == 4
    assert tracks["track1"]["models"] == 3  # model_a, model_b, model_c


def test_run_summary_cost(api_env: None) -> None:
    """Cost breakdown sums correctly."""
    client = TestClient(app)
    resp = client.get(
        "/v1/diagnostics/run_summary",
        params={"forecaster_run_id": "fc_test_run_1"},
    )
    cost = resp.json()["cost"]
    assert cost["total_usd"] == 5.60
    assert cost["total_tokens"] == 20600

    by_phase = {p["phase"]: p["cost_usd"] for p in cost["by_phase"]}
    assert by_phase["hs_triage"] == 1.50
    assert by_phase["spd_v2"] == 3.20
    assert by_phase["binary_v2"] == 0.80
    assert by_phase["scenario_v2"] == 0.10


def test_run_summary_llm_health(api_env: None) -> None:
    """LLM health reports errors correctly."""
    client = TestClient(app)
    resp = client.get(
        "/v1/diagnostics/run_summary",
        params={"forecaster_run_id": "fc_test_run_1"},
    )
    health = resp.json()["llm_health"]
    assert health["total_calls"] == 5
    assert health["errors"] == 1
    assert abs(health["error_rate"] - 0.2) < 0.01


def test_kpi_scopes_pa_forecasts_nonzero(api_env: None) -> None:
    """Regression: PA metric FORECASTS count must be non-zero when PA questions exist."""
    client = TestClient(app)
    resp = client.get(
        "/v1/diagnostics/kpi_scopes",
        params={"metric_scope": "PA", "forecaster_run_id": "fc_test_run_1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    selected = data["scopes"]["selected_run"]
    # 4 PA questions exist (q_aaa_ace_pa, q_aaa_fl_pa, q_bbb_ace_pa, q_bbb_fl_pa)
    assert selected["forecasts"] >= 4, (
        f"PA forecasts should be >= 4, got {selected['forecasts']}"
    )


def test_kpi_scopes_fatalities_nonzero(api_env: None) -> None:
    """FATALITIES metric FORECASTS count should be non-zero."""
    client = TestClient(app)
    resp = client.get(
        "/v1/diagnostics/kpi_scopes",
        params={"metric_scope": "FATALITIES", "forecaster_run_id": "fc_test_run_1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    selected = data["scopes"]["selected_run"]
    assert selected["forecasts"] >= 2, (
        f"FATALITIES forecasts should be >= 2, got {selected['forecasts']}"
    )
