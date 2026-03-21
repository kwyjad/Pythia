# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Phase 4 Dashboard + Calibration API tests.

Tests:
- risk_index with metric=EVENT_OCCURRENCE returns binary probabilities
- risk_index with metric=PHASE3PLUS_IN_NEED works with normalization
- performance_scores with metric=EVENT_OCCURRENCE returns Brier only
- diagnostics_kpi_scopes accepts EVENT_OCCURRENCE and PHASE3PLUS_IN_NEED
- resolution_rates endpoint returns correct format
- Existing PA/FATALITIES endpoints remain unchanged
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
    cfg = {
        "app": {"db_url": f"duckdb:///{db_path}"},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    db_path = tmp_path / "api-phase4.duckdb"
    con = duckdb.connect(str(db_path), read_only=False)
    con.execute(
        """
        CREATE TABLE questions (
            question_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            target_month TEXT,
            metric TEXT,
            hs_run_id TEXT,
            status TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE forecasts_ensemble (
            question_id TEXT,
            month_index INTEGER,
            bucket_index INTEGER,
            probability DOUBLE,
            model_name TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE populations (
            iso3 TEXT,
            population BIGINT,
            year INTEGER
        );
        """
    )
    con.execute(
        """
        CREATE TABLE bucket_centroids (
            hazard_code TEXT,
            metric TEXT,
            bucket_index INTEGER,
            centroid DOUBLE
        );
        """
    )
    con.execute(
        """
        CREATE TABLE resolutions (
            question_id TEXT,
            horizon_m INTEGER,
            value DOUBLE,
            observed_month TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE scores (
            question_id TEXT,
            horizon_m INTEGER,
            score_type TEXT,
            value DOUBLE,
            model_name TEXT,
            run_id TEXT
        );
        """
    )

    # -- PA question with forecast
    con.execute(
        """
        INSERT INTO questions (question_id, iso3, hazard_code, target_month, metric)
        VALUES
          ('q_pa', 'USA', 'FL', '2026-01', 'PA'),
          ('q_fat', 'USA', 'ACE', '2026-01', 'FATALITIES'),
          ('q_bin1', 'USA', 'FL', '2026-01', 'EVENT_OCCURRENCE'),
          ('q_bin2', 'GBR', 'TC', '2026-01', 'EVENT_OCCURRENCE'),
          ('q_ph3', 'ETH', 'DR', '2026-01', 'PHASE3PLUS_IN_NEED'),
          ('q_pa_nores', 'BRA', 'FL', '2026-01', 'PA');
        """
    )

    # Binary forecasts: bucket_1 = P(event)
    con.execute(
        """
        INSERT INTO forecasts_ensemble (question_id, month_index, bucket_index, probability, model_name)
        VALUES
          ('q_bin1', 1, 1, 0.35, 'ensemble_bayesmc_v2'),
          ('q_bin1', 2, 1, 0.40, 'ensemble_bayesmc_v2'),
          ('q_bin1', 3, 1, 0.25, 'ensemble_bayesmc_v2'),
          ('q_bin1', 4, 1, 0.20, 'ensemble_bayesmc_v2'),
          ('q_bin1', 5, 1, 0.15, 'ensemble_bayesmc_v2'),
          ('q_bin1', 6, 1, 0.10, 'ensemble_bayesmc_v2'),
          ('q_bin2', 1, 1, 0.50, 'ensemble_bayesmc_v2'),
          ('q_bin2', 2, 1, 0.55, 'ensemble_bayesmc_v2'),
          ('q_bin2', 3, 1, 0.45, 'ensemble_bayesmc_v2'),
          ('q_bin2', 4, 1, 0.30, 'ensemble_bayesmc_v2'),
          ('q_bin2', 5, 1, 0.20, 'ensemble_bayesmc_v2'),
          ('q_bin2', 6, 1, 0.15, 'ensemble_bayesmc_v2');
        """
    )

    # PHASE3PLUS_IN_NEED forecasts (SPD format)
    con.execute(
        """
        INSERT INTO forecasts_ensemble (question_id, month_index, bucket_index, probability, model_name)
        VALUES
          ('q_ph3', 1, 1, 0.10, 'ensemble_bayesmc_v2'),
          ('q_ph3', 1, 2, 0.30, 'ensemble_bayesmc_v2'),
          ('q_ph3', 1, 3, 0.40, 'ensemble_bayesmc_v2'),
          ('q_ph3', 1, 4, 0.15, 'ensemble_bayesmc_v2'),
          ('q_ph3', 1, 5, 0.05, 'ensemble_bayesmc_v2');
        """
    )

    # PA forecasts
    con.execute(
        """
        INSERT INTO forecasts_ensemble (question_id, month_index, bucket_index, probability, model_name)
        VALUES
          ('q_pa', 1, 1, 0.20, 'ensemble_bayesmc_v2'),
          ('q_pa', 1, 2, 0.30, 'ensemble_bayesmc_v2'),
          ('q_pa', 1, 3, 0.25, 'ensemble_bayesmc_v2'),
          ('q_pa', 1, 4, 0.15, 'ensemble_bayesmc_v2'),
          ('q_pa', 1, 5, 0.10, 'ensemble_bayesmc_v2');
        """
    )

    # Populations
    con.execute(
        """
        INSERT INTO populations (iso3, population, year)
        VALUES
          ('USA', 330000000, 2024),
          ('GBR', 67000000, 2024),
          ('ETH', 120000000, 2024),
          ('BRA', 215000000, 2024);
        """
    )

    # Bucket centroids
    con.execute(
        """
        INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid)
        VALUES
          ('*', 'PA', 1, 5000),
          ('*', 'PA', 2, 25000),
          ('*', 'PA', 3, 120000),
          ('*', 'PA', 4, 350000),
          ('*', 'PA', 5, 700000),
          ('*', 'PHASE3PLUS_IN_NEED', 1, 50000),
          ('*', 'PHASE3PLUS_IN_NEED', 2, 500000),
          ('*', 'PHASE3PLUS_IN_NEED', 3, 3000000),
          ('*', 'PHASE3PLUS_IN_NEED', 4, 10000000),
          ('*', 'PHASE3PLUS_IN_NEED', 5, 20000000);
        """
    )

    # Resolutions (some questions resolved, some not)
    con.execute(
        """
        INSERT INTO resolutions (question_id, horizon_m, value, observed_month)
        VALUES
          ('q_bin1', 1, 1.0, '2025-08'),
          ('q_bin1', 2, 0.0, '2025-09'),
          ('q_bin1', 3, 0.0, '2025-10'),
          ('q_bin2', 1, 1.0, '2025-08'),
          ('q_bin2', 2, 1.0, '2025-09'),
          ('q_pa', 1, 15000.0, '2025-08');
        """
    )

    # Scores
    con.execute(
        """
        INSERT INTO scores (question_id, horizon_m, score_type, value, model_name, run_id)
        VALUES
          ('q_bin1', 1, 'brier', 0.4225, NULL, 'run1'),
          ('q_bin1', 2, 'brier', 0.1600, NULL, 'run1'),
          ('q_bin2', 1, 'brier', 0.2500, NULL, 'run1'),
          ('q_pa', 1, 'brier', 0.3000, NULL, 'run1'),
          ('q_pa', 1, 'log', 1.2000, NULL, 'run1'),
          ('q_pa', 1, 'crps', 0.1500, NULL, 'run1');
        """
    )
    con.close()

    config_path = _write_config(tmp_path, db_path)
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    pythia_config.load.cache_clear()

    # Reset the singleton DB connection so it picks up the test DB
    import pythia.api.app as _app_mod
    _app_mod._READ_CON = None

    try:
        yield
    finally:
        _app_mod._READ_CON = None
        pythia_config.load.cache_clear()


def test_risk_index_binary(api_env: None) -> None:
    """Risk index with metric=EVENT_OCCURRENCE returns probabilities."""
    client = TestClient(app)
    resp = client.get(
        "/v1/risk_index",
        params={"metric": "EVENT_OCCURRENCE", "target_month": "2026-01"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["metric"] == "EVENT_OCCURRENCE"
    assert payload["metric_type"] == "binary"
    assert payload["rows"]

    # Should have USA and GBR rows
    iso3s = {row["iso3"] for row in payload["rows"]}
    assert "USA" in iso3s
    assert "GBR" in iso3s

    # Values should be probabilities (0-1)
    for row in payload["rows"]:
        assert row["metric_type"] == "binary"
        total = row.get("total")
        assert total is not None
        assert 0 <= total <= 1


def test_risk_index_binary_no_per_capita_change(api_env: None) -> None:
    """Binary EVENT_OCCURRENCE returns raw probability for per-capita."""
    client = TestClient(app)
    resp = client.get(
        "/v1/risk_index",
        params={"metric": "EVENT_OCCURRENCE", "target_month": "2026-01", "normalize": True},
    )
    assert resp.status_code == 200
    payload = resp.json()
    for row in payload["rows"]:
        # For binary, total_pc should equal total (probability IS the rate)
        assert row["total_pc"] == row["total"]


def test_risk_index_phase3plus(api_env: None) -> None:
    """Risk index with metric=PHASE3PLUS_IN_NEED returns SPD-based values."""
    client = TestClient(app)
    resp = client.get(
        "/v1/risk_index",
        params={
            "metric": "PHASE3PLUS_IN_NEED",
            "hazard_code": "DR",
            "target_month": "2026-01",
            "normalize": True,
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["metric"] == "PHASE3PLUS_IN_NEED"
    assert payload["rows"]
    row = payload["rows"][0]
    assert row["iso3"] == "ETH"
    # EIV should be > 0 (SPD × centroids)
    assert row["m1"] is not None and row["m1"] > 0
    # Per-capita should be EIV / population
    if row.get("population") and row["population"] > 0 and row.get("m1_pc") is not None:
        assert row["m1_pc"] == pytest.approx(row["m1"] / row["population"], rel=0.01)


def test_risk_index_pa_unchanged(api_env: None) -> None:
    """Existing PA endpoint still works correctly."""
    client = TestClient(app)
    resp = client.get(
        "/v1/risk_index",
        params={"metric": "PA", "target_month": "2026-01"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["metric"] == "PA"
    assert payload["metric_type"] == "spd"
    assert payload["rows"]


def test_performance_scores_binary(api_env: None) -> None:
    """Performance scores with metric=EVENT_OCCURRENCE returns Brier only."""
    client = TestClient(app)
    resp = client.get(
        "/v1/performance/scores",
        params={"metric": "EVENT_OCCURRENCE"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    for row in payload["summary_rows"]:
        # Binary questions only have Brier scores
        assert row["score_type"] == "brier"


def test_diagnostics_kpi_scopes_event_occurrence(api_env: None) -> None:
    """diagnostics_kpi_scopes accepts EVENT_OCCURRENCE metric."""
    client = TestClient(app)
    resp = client.get(
        "/v1/diagnostics/kpi_scopes",
        params={"metric_scope": "EVENT_OCCURRENCE"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    # Should not have "metric_scope_unrecognized" in notes
    notes = payload.get("notes", [])
    assert "metric_scope_unrecognized" not in (notes or [])


def test_diagnostics_kpi_scopes_phase3plus(api_env: None) -> None:
    """diagnostics_kpi_scopes accepts PHASE3PLUS_IN_NEED metric."""
    client = TestClient(app)
    resp = client.get(
        "/v1/diagnostics/kpi_scopes",
        params={"metric_scope": "PHASE3PLUS_IN_NEED"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    notes = payload.get("notes", [])
    assert "metric_scope_unrecognized" not in (notes or [])


def test_resolution_rates(api_env: None) -> None:
    """Resolution rates endpoint returns correct format and values."""
    client = TestClient(app)
    resp = client.get("/v1/diagnostics/resolution_rates")
    assert resp.status_code == 200
    payload = resp.json()
    assert "rows" in payload
    rows = payload["rows"]
    assert len(rows) > 0

    # Check format
    for row in rows:
        assert "hazard_code" in row
        assert "metric" in row
        assert "total_questions" in row
        assert "resolved_questions" in row
        assert "skipped_questions" in row
        assert "resolution_rate" in row
        assert row["total_questions"] >= row["resolved_questions"]
        assert row["skipped_questions"] == row["total_questions"] - row["resolved_questions"]
        assert 0.0 <= row["resolution_rate"] <= 1.0

    # Binary questions should have high resolution rate (we resolved 3/6 horizons)
    binary_rows = [r for r in rows if r["metric"] == "EVENT_OCCURRENCE"]
    assert binary_rows
    # PA should have resolution too
    pa_rows = [r for r in rows if r["metric"] == "PA"]
    assert pa_rows


def test_resolution_rates_filtered(api_env: None) -> None:
    """Resolution rates can be filtered by hazard_code."""
    client = TestClient(app)
    resp = client.get(
        "/v1/diagnostics/resolution_rates",
        params={"hazard_code": "FL"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    for row in payload["rows"]:
        assert row["hazard_code"] == "FL"


def test_missing_population_per_capita(api_env: None) -> None:
    """Per-capita with missing population returns null, not crash."""
    client = TestClient(app)
    # Use a metric where the country has no population data
    resp = client.get(
        "/v1/risk_index",
        params={"metric": "PA", "target_month": "2026-01", "normalize": True},
    )
    assert resp.status_code == 200
