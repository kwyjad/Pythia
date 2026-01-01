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
            status TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE llm_calls (
            question_id TEXT,
            created_at TIMESTAMP,
            phase TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE forecasts_ensemble (
            question_id TEXT,
            created_at TIMESTAMP
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
        "INSERT INTO hs_runs (hs_run_id, created_at) VALUES ('run-1', NOW())"
    )
    con.execute(
        """
        INSERT INTO questions (question_id, status)
        VALUES ('q1', 'active'), ('q2', 'active'), ('q3', 'inactive')
        """
    )
    con.execute(
        """
        INSERT INTO llm_calls (question_id, created_at, phase)
        VALUES
            ('q1', TIMESTAMP '2026-01-15 12:00:00', 'forecast'),
            ('q2', TIMESTAMP '2026-01-20 13:00:00', 'research'),
            ('q3', TIMESTAMP '2025-12-05 09:00:00', 'forecast')
        """
    )
    con.execute(
        """
        INSERT INTO forecasts_ensemble (question_id, created_at)
        VALUES
            ('q1', TIMESTAMP '2026-01-16 00:00:00'),
            ('q3', TIMESTAMP '2025-12-06 00:00:00')
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


def test_diagnostics_schema_tolerant(api_env: None) -> None:
    client = TestClient(app)

    resp = client.get("/v1/diagnostics/summary")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["questions_with_resolutions"] == 0
    assert payload["questions_with_scores"] == 0
    assert payload["questions_with_forecasts"] == 2
    assert payload["latest_hs_run"]["run_id"] == "run-1"


def test_kpi_scopes_schema_tolerant(api_env: None) -> None:
    client = TestClient(app)

    resp = client.get("/v1/diagnostics/kpi_scopes")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["default_scope"] == "latest_run"
    assert "latest_run" in payload["scopes"]
    assert "total_active" in payload["scopes"]
    assert "total_all" in payload["scopes"]

    latest = payload["scopes"]["latest_run"]
    total_active = payload["scopes"]["total_active"]
    total_all = payload["scopes"]["total_all"]

    assert isinstance(latest["questions"], int)
    assert isinstance(latest["questions_with_forecasts"], int)
    assert isinstance(total_active["questions"], int)
    assert isinstance(total_active["questions_with_forecasts"], int)
    assert isinstance(total_all["questions"], int)
    assert isinstance(total_all["questions_with_forecasts"], int)

    assert total_all["questions"] >= total_active["questions"]
    assert total_active["questions"] >= latest["questions"]
