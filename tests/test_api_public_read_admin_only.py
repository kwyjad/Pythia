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
        CREATE TABLE questions (status TEXT);
        CREATE TABLE forecasts_ensemble (question_id TEXT);
        CREATE TABLE resolutions (question_id TEXT);
        CREATE TABLE scores (question_id TEXT);
        CREATE TABLE hs_runs (run_id TEXT, created_at TIMESTAMP, meta TEXT);
        CREATE TABLE calibration_weights (as_of_month TEXT, created_at TIMESTAMP);
        """
    )
    con.close()

    config_path = _write_config(tmp_path, db_path)
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    pythia_config.load.cache_clear()

    def _fake_enqueue_run(_: list[str]) -> str:
        return "ui_run_test"

    monkeypatch.setattr("pythia.api.app.enqueue_run", _fake_enqueue_run)

    try:
        yield
    finally:
        pythia_config.load.cache_clear()



def test_public_reads_and_admin_run_require_token(api_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHIA_API_TOKEN", "x")
    monkeypatch.delenv("PYTHIA_API_KEY", raising=False)
    pythia_config.load.cache_clear()

    client = TestClient(app)

    resp = client.get("/v1/health")
    assert resp.status_code == 200

    resp = client.get("/v1/diagnostics/summary")
    assert resp.status_code != 401

    resp = client.post("/v1/run", json={"countries": []})
    assert resp.status_code == 401

    authed = TestClient(app, headers={"Authorization": "Bearer x"})
    resp = authed.post("/v1/run", json={"countries": []})
    assert resp.status_code == 200


def test_admin_run_fails_closed_without_token(api_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHIA_API_TOKEN", raising=False)
    monkeypatch.delenv("PYTHIA_API_KEY", raising=False)
    pythia_config.load.cache_clear()

    client = TestClient(app)
    resp = client.post("/v1/run", json={"countries": []})
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Admin token not configured"
