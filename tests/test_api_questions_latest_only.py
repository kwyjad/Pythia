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
            hazard_code TEXT,
            metric TEXT,
            target_month TEXT,
            hs_run_id TEXT,
            status TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE hs_runs (
            hs_run_id TEXT,
            created_at TIMESTAMP
        );
        """
    )
    con.execute(
        """
        CREATE TABLE forecasts_ensemble (
            question_id TEXT
        );
        """
    )
    con.execute(
        """
        INSERT INTO hs_runs (hs_run_id, created_at)
        VALUES
            ('hs_run_old', '2024-01-01 00:00:00'),
            ('hs_run_new', '2024-02-01 00:00:00');
        """
    )
    con.execute(
        """
        INSERT INTO questions (
            question_id,
            iso3,
            hazard_code,
            metric,
            target_month,
            hs_run_id,
            status
        )
        VALUES
            ('q_old', 'USA', 'TC', 'PIN', '2024-03', 'hs_run_old', 'open'),
            ('q_new', 'USA', 'TC', 'PIN', '2024-03', 'hs_run_new', 'open');
        """
    )
    con.execute(
        """
        INSERT INTO forecasts_ensemble (question_id)
        VALUES ('q_new');
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


def test_latest_only_questions_selects_latest_hs_run(api_env: None) -> None:
    client = TestClient(app)

    resp = client.get("/v1/questions", params={"latest_only": "true"})
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["question_id"] == "q_new"


def test_countries_endpoint_returns_counts(api_env: None) -> None:
    client = TestClient(app)

    resp = client.get("/v1/countries")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["rows"] == [
        {"iso3": "USA", "n_questions": 2, "n_forecasted": 1}
    ]
