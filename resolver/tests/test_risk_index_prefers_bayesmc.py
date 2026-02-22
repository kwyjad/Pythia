from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
import yaml

pytest.importorskip("fastapi")
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
    db_path = tmp_path / "api-risk-index-bayesmc.duckdb"
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
        CREATE TABLE bucket_centroids (
            metric TEXT,
            hazard_code TEXT,
            bucket_index INTEGER,
            centroid DOUBLE
        );
        """
    )
    con.execute(
        """
        INSERT INTO questions (question_id, iso3, hazard_code, target_month, metric)
        VALUES ('q1', 'ETH', 'FL', '2026-01', 'PA');
        """
    )
    con.execute(
        """
        INSERT INTO forecasts_ensemble (question_id, month_index, bucket_index, probability, model_name)
        VALUES
          ('q1', 1, 2, 1.0, 'ensemble_bayesmc_v2'),
          ('q1', 1, 5, 1.0, 'ensemble_mean_v2');
        """
    )
    con.execute(
        """
        INSERT INTO bucket_centroids (metric, hazard_code, bucket_index, centroid)
        VALUES
          ('PA', '*', 1, 1.0),
          ('PA', '*', 2, 2.0),
          ('PA', '*', 3, 3.0),
          ('PA', '*', 4, 4.0),
          ('PA', '*', 5, 5.0);
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


def test_risk_index_prefers_bayesmc(api_env: None) -> None:
    client = TestClient(app)

    resp = client.get(
        "/v1/risk_index",
        params={"metric": "PA", "horizon_m": 1, "normalize": False, "target_month": "2026-01"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["rows"]
    row = payload["rows"][0]
    assert row["total"] == pytest.approx(2.0)
    assert row["m1"] == pytest.approx(2.0)
