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
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    db_path = tmp_path / "api-risk-index.duckdb"
    con = duckdb.connect(str(db_path), read_only=False)
    con.execute(
        """
        CREATE TABLE questions (
            question_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            target_month TEXT,
            metric TEXT
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
        INSERT INTO questions (question_id, iso3, hazard_code, target_month, metric)
        VALUES
          ('q1', 'USA', 'FL', '2026-01', 'PA'),
          ('q2', 'USA', 'ACE', '2026-01', 'FATALITIES'),
          ('q3', 'USA', 'TC', '2026-01', 'PA');
        """
    )
    con.execute(
        """
        INSERT INTO forecasts_ensemble (question_id, month_index, bucket_index, probability, model_name)
        VALUES
          ('q1', 1, 2, 1.0, 'ensemble_bayesmc_v2'),
          ('q1', 1, 5, 1.0, 'ensemble_mean_v2'),
          ('q3', 1, 5, 1.0, 'ensemble_bayesmc_v2'),
          ('q2', 1, 3, 1.0, 'ensemble_bayesmc_v2'),
          ('q2', 2, 4, 1.0, 'ensemble_bayesmc_v2');
        """
    )
    con.execute(
        """
        INSERT INTO populations (iso3, population, year)
        VALUES ('USA', 1000000, 2024);
        """
    )
    con.execute(
        """
        INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid)
        VALUES
          ('*', 'PA', 1, 1.0),
          ('*', 'PA', 2, 2.0),
          ('*', 'PA', 3, 3.0),
          ('*', 'PA', 4, 4.0),
          ('*', 'PA', 5, 5.0),
          ('*', 'FATALITIES', 1, 1.0),
          ('*', 'FATALITIES', 2, 2.0),
          ('*', 'FATALITIES', 3, 3.0),
          ('*', 'FATALITIES', 4, 4.0),
          ('*', 'FATALITIES', 5, 5.0);
        """
    )
    con.close()

    config_path = _write_config(tmp_path, db_path)
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    pythia_config.load.cache_clear()

    try:
        yield
    finally:
        pythia_config.load.cache_clear()


def test_risk_index_smoke(api_env: None) -> None:
    client = TestClient(app)

    resp = client.get(
        "/v1/risk_index",
        params={"metric": "PA", "horizon_m": 1, "normalize": True, "target_month": "2026-01"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["rows"]
    row = payload["rows"][0]
    assert row["total"] == pytest.approx(5.2)
    assert row["population"] == 1000000
    assert row["m1_pc"] == pytest.approx(row["m1"] / row["population"])
    assert row["total_pc"] == pytest.approx(row["total"] / row["population"])

    resp = client.get(
        "/v1/risk_index",
        params={
            "metric": "FATALITIES",
            "horizon_m": 1,
            "normalize": True,
            "target_month": "2026-01",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    row = payload["rows"][0]
    assert "population" in row
    assert "total_pc" in row
    assert row["total"] == pytest.approx(7.0)
