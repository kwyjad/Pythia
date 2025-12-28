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
            hs_run_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE hs_runs (
            hs_run_id TEXT,
            generated_at TIMESTAMP
        );
        """
    )
    con.execute(
        """
        INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric)
        VALUES ('AFG_ACE_FATALITIES', 'hs_20251218T153908', 'AFG', 'ACE', 'FATALITIES');
        """
    )
    con.execute(
        """
        INSERT INTO hs_runs (hs_run_id, generated_at)
        VALUES ('hs_20251218T153908', '2025-12-18 15:39:08');
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


def test_question_bundle_uses_generated_at(api_env: None) -> None:
    client = TestClient(app)
    resp = client.get(
        "/v1/question_bundle",
        params={"question_id": "AFG_ACE_FATALITIES", "hs_run_id": "hs_20251218T153908"},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["question"]["question_id"] == "AFG_ACE_FATALITIES"


def test_question_bundle_missing_question(api_env: None) -> None:
    client = TestClient(app)
    resp = client.get("/v1/question_bundle", params={"question_id": "DOES_NOT_EXIST"})
    assert resp.status_code == 404
