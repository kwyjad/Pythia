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
    db_path = tmp_path / "api.duckdb"
    con = duckdb.connect(str(db_path), read_only=False)
    con.execute(
        """
        CREATE TABLE questions (
            question_id TEXT,
            iso3 TEXT,
            target_month TEXT,
            metric TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE forecasts_ensemble (
            question_id TEXT,
            horizon_m INTEGER,
            class_bin TEXT,
            p DOUBLE
        );
        """
    )
    con.execute(
        """
        INSERT INTO questions (question_id, iso3, target_month, metric)
        VALUES
            ('q1', 'USA', '2026-01', 'PA'),
            ('q2', 'CAN', '2025-12', 'PA');
        """
    )
    con.execute(
        """
        INSERT INTO forecasts_ensemble (question_id, horizon_m, class_bin, p)
        VALUES ('q1', 1, '<10k', 1.0);
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


def test_risk_index_defaults_to_latest_forecasted_month(api_env: None) -> None:
    client = TestClient(app)

    resp = client.get("/v1/risk_index", params={"metric": "PA", "horizon_m": 1, "normalize": True})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["target_month"] == "2026-01"
    assert payload["rows"]


def test_risk_index_fallbacks_from_empty_month(api_env: None) -> None:
    client = TestClient(app)

    resp = client.get(
        "/v1/risk_index",
        params={"metric": "PA", "horizon_m": 1, "normalize": True, "target_month": "2025-12"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["target_month"] == "2026-01"
    assert payload["rows"]
