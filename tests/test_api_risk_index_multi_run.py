"""Verify that the risk index uses only the latest forecaster run when
multiple runs exist for the same question_id, preventing double-counting."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
import yaml
from fastapi.testclient import TestClient

duckdb = pytest.importorskip("duckdb")

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
def multi_run_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    db_path = tmp_path / "multi-run.duckdb"
    con = duckdb.connect(str(db_path), read_only=False)

    con.execute("""
        CREATE TABLE questions (
            question_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            target_month TEXT,
            metric TEXT
        );
    """)
    con.execute("""
        CREATE TABLE forecasts_ensemble (
            run_id TEXT,
            question_id TEXT,
            month_index INTEGER,
            bucket_index INTEGER,
            probability DOUBLE,
            model_name TEXT,
            created_at TIMESTAMP
        );
    """)
    con.execute("""
        CREATE TABLE populations (
            iso3 TEXT,
            population BIGINT,
            year INTEGER
        );
    """)
    con.execute("""
        CREATE TABLE bucket_centroids (
            hazard_code TEXT,
            metric TEXT,
            bucket_index INTEGER,
            centroid DOUBLE
        );
    """)

    # One question
    con.execute("""
        INSERT INTO questions (question_id, iso3, hazard_code, target_month, metric)
        VALUES ('q1', 'USA', 'FL', '2026-03', 'PA');
    """)

    # Run 1: p=0.8 for bucket 2 (centroid 2.0)
    con.execute("""
        INSERT INTO forecasts_ensemble
            (run_id, question_id, month_index, bucket_index, probability, model_name, created_at)
        VALUES
            ('fc_1000001', 'q1', 1, 2, 0.8, 'ensemble_bayesmc_v2', '2026-03-01 10:00:00');
    """)

    # Run 2 (LATER): p=0.6 for bucket 2 (centroid 2.0)
    con.execute("""
        INSERT INTO forecasts_ensemble
            (run_id, question_id, month_index, bucket_index, probability, model_name, created_at)
        VALUES
            ('fc_1000002', 'q1', 1, 2, 0.6, 'ensemble_bayesmc_v2', '2026-03-15 10:00:00');
    """)

    con.execute("""
        INSERT INTO populations (iso3, population, year) VALUES ('USA', 1000000, 2024);
    """)
    con.execute("""
        INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid)
        VALUES
          ('*', 'PA', 1, 1.0),
          ('*', 'PA', 2, 2.0),
          ('*', 'PA', 3, 3.0),
          ('*', 'PA', 4, 4.0),
          ('*', 'PA', 5, 5.0);
    """)
    con.close()

    config_path = _write_config(tmp_path, db_path)
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    pythia_config.load.cache_clear()

    try:
        yield
    finally:
        pythia_config.load.cache_clear()


def test_risk_index_latest_run_only(multi_run_env: None) -> None:
    """With two runs, risk index should use only the latest run (fc_1000002)."""
    client = TestClient(app)

    resp = client.get(
        "/v1/risk_index",
        params={"metric": "PA", "horizon_m": 1, "target_month": "2026-03"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    rows = payload["rows"]
    assert len(rows) == 1
    usa = rows[0]
    assert usa["iso3"] == "USA"
    # Latest run (fc_1000002): p=0.6, centroid=2.0 → EIV = 0.6 * 2.0 = 1.2
    # NOT doubled: (0.8 + 0.6) * 2.0 = 2.8
    assert usa["m1"] == pytest.approx(1.2)


def test_risk_index_explicit_run_id(multi_run_env: None) -> None:
    """Passing forecaster_run_id should scope to that specific run."""
    client = TestClient(app)

    # Request the OLDER run explicitly
    resp = client.get(
        "/v1/risk_index",
        params={
            "metric": "PA",
            "horizon_m": 1,
            "target_month": "2026-03",
            "forecaster_run_id": "fc_1000001",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    rows = payload["rows"]
    assert len(rows) == 1
    usa = rows[0]
    # Older run (fc_1000001): p=0.8, centroid=2.0 → EIV = 0.8 * 2.0 = 1.6
    assert usa["m1"] == pytest.approx(1.6)
    assert payload["forecaster_run_id"] == "fc_1000001"
