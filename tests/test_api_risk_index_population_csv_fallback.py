from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
import yaml
from fastapi.testclient import TestClient

duckdb = pytest.importorskip("duckdb")

from pythia import config as pythia_config
from pythia.api import app as api_app
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
    db_path = tmp_path / "api-risk-index-population.duckdb"
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
            probability DOUBLE
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
        VALUES ('q1', 'USA', 'FL', '2026-01', 'PA');
        """
    )
    con.execute(
        """
        INSERT INTO forecasts_ensemble (question_id, month_index, bucket_index, probability)
        VALUES ('q1', 1, 2, 1.0);
        """
    )
    con.execute(
        """
        INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid)
        VALUES ('*', 'PA', 2, 2.0);
        """
    )
    con.close()

    population_path = tmp_path / "population.tsv"
    population_path.write_text(
        "iso3\tcountry_name\tpopulation\tas_of\tsource\n"
        "USA\tUnited States\t1,234,000\t2024\tTest\n",
        encoding="utf-8",
    )
    config_path = _write_config(tmp_path, db_path)
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("PYTHIA_POPULATION_CSV_PATH", str(population_path))
    pythia_config.load.cache_clear()
    api_app._POPULATION_BY_ISO3 = {}

    try:
        yield
    finally:
        pythia_config.load.cache_clear()
        api_app._POPULATION_BY_ISO3 = {}


def test_risk_index_population_csv_fallback(api_env: None) -> None:
    client = TestClient(app)

    resp = client.get(
        "/v1/risk_index",
        params={"metric": "PA", "horizon_m": 1, "normalize": True, "target_month": "2026-01"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["rows"]
    row = payload["rows"][0]
    assert row["population"] == 1234000
    assert row["m1_pc"] == pytest.approx(row["m1"] / row["population"])
