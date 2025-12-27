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
            status TEXT,
            iso3 TEXT,
            target_month TEXT,
            metric TEXT
        );
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


def test_missing_optional_tables_are_ok(api_env: None) -> None:
    client = TestClient(app)

    resp = client.get("/v1/diagnostics/summary")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["questions_with_resolutions"] == 0
    assert payload["questions_with_scores"] == 0
    assert payload["questions_with_forecasts"] == 0

    resp = client.get(
        "/v1/resolutions",
        params={"iso3": "USA", "month": "2025-01", "metric": "PIN"},
    )
    assert resp.status_code == 200
    assert resp.json() == {"rows": []}
