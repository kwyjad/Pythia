from __future__ import annotations

import sys
import types
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
        CREATE TABLE questions (status TEXT, is_test BOOLEAN DEFAULT FALSE);
        CREATE TABLE forecasts_ensemble (question_id TEXT, is_test BOOLEAN DEFAULT FALSE);
        CREATE TABLE resolutions (question_id TEXT);
        CREATE TABLE scores (question_id TEXT, is_test BOOLEAN DEFAULT FALSE);
        CREATE TABLE hs_runs (run_id TEXT, created_at TIMESTAMP, meta TEXT, is_test BOOLEAN DEFAULT FALSE);
        CREATE TABLE calibration_weights (as_of_month TEXT, created_at TIMESTAMP);
        """
    )
    con.close()

    config_path = _write_config(tmp_path, db_path)
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    pythia_config.load.cache_clear()

    def _fake_enqueue_run(_: list[str]) -> str:
        return "ui_run_test"

    # /v1/run imports pythia.pipeline.run lazily inside the handler (the real
    # module pulls in the full horizon_scanner tree); stub it in sys.modules so
    # the deferred import resolves to the fake without loading the pipeline.
    stub = types.ModuleType("pythia.pipeline.run")
    stub.enqueue_run = _fake_enqueue_run
    monkeypatch.setitem(sys.modules, "pythia.pipeline.run", stub)

    try:
        yield
    finally:
        pythia_config.load.cache_clear()



def test_public_reads_and_admin_run_require_token(api_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHIA_API_TOKEN", "x")
    monkeypatch.setenv("PYTHIA_ALLOW_INPROCESS_RUN", "1")
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


def test_admin_run_disabled_by_default(api_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Without PYTHIA_ALLOW_INPROCESS_RUN=1 the endpoint refuses even with a
    valid token — the in-process pipeline must never run on the memory-
    constrained API deployment unless explicitly enabled."""
    monkeypatch.setenv("PYTHIA_API_TOKEN", "x")
    monkeypatch.delenv("PYTHIA_ALLOW_INPROCESS_RUN", raising=False)
    pythia_config.load.cache_clear()

    authed = TestClient(app, headers={"Authorization": "Bearer x"})
    resp = authed.post("/v1/run", json={"countries": []})
    assert resp.status_code == 503
    assert "PYTHIA_ALLOW_INPROCESS_RUN" in resp.json()["detail"]


def test_admin_run_fails_closed_without_token(api_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYTHIA_API_TOKEN", raising=False)
    monkeypatch.delenv("PYTHIA_API_KEY", raising=False)
    pythia_config.load.cache_clear()

    client = TestClient(app)
    resp = client.post("/v1/run", json={"countries": []})
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Admin token not configured"
