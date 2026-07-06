# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Row caps on /v1/forecasts/ensemble and /v1/forecasts/history.

These endpoints materialize every row as a Python dict; an unbounded pull
of forecasts_ensemble could take hundreds of MB. A generous default cap
(200k) applies, an explicit ?limit= raises it, and truncation is flagged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

pytest.importorskip("fastapi")
duckdb = pytest.importorskip("duckdb")

from fastapi.testclient import TestClient

from pythia import config as pythia_config
import pythia.api.app as _app_mod
import pythia.api.routes.forecasts as forecasts_mod
from pythia.api.app import app

N_FORECAST_ROWS = 8


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    db_path = tmp_path / "api.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE questions (
          question_id TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT,
          target_month TEXT, run_id TEXT, is_test BOOLEAN DEFAULT FALSE
        );
        CREATE TABLE forecasts_ensemble (
          question_id TEXT, horizon_m INTEGER, class_bin TEXT, p DOUBLE,
          aggregator TEXT, ensemble_version TEXT, is_test BOOLEAN DEFAULT FALSE
        );
        CREATE TABLE hs_runs (run_id TEXT, created_at TIMESTAMP);
        """
    )
    con.execute(
        "INSERT INTO questions VALUES ('q1', 'ETH', 'ACE', 'FATALITIES', '2026-12', 'run1', FALSE)"
    )
    con.execute("INSERT INTO hs_runs VALUES ('run1', TIMESTAMP '2026-07-01 00:00:00')")
    for i in range(N_FORECAST_ROWS):
        con.execute(
            "INSERT INTO forecasts_ensemble VALUES ('q1', ?, ?, 0.1, 'mean', 'v2', FALSE)",
            [1 + i // 4, f"bin{i % 4}"],
        )
    con.close()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"app:\n  db_url: 'duckdb:///{db_path}'\n", encoding="utf-8")
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    pythia_config.load.cache_clear()
    _app_mod._READ_CON = None
    try:
        yield
    finally:
        _app_mod._READ_CON = None
        pythia_config.load.cache_clear()


def test_ensemble_default_cap_truncates_and_flags(api_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forecasts_mod, "_DEFAULT_ROW_CAP", 5)

    client = TestClient(app)
    resp = client.get("/v1/forecasts/ensemble", params={"latest_only": "false"})

    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["rows"]) == 5
    assert payload["truncated"] is True


def test_ensemble_explicit_limit_overrides_default(api_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forecasts_mod, "_DEFAULT_ROW_CAP", 5)

    client = TestClient(app)
    resp = client.get(
        "/v1/forecasts/ensemble",
        params={"latest_only": "false", "limit": N_FORECAST_ROWS},
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["rows"]) == N_FORECAST_ROWS
    assert "truncated" not in payload  # under the cap — response shape unchanged


def test_history_cap_truncates_and_flags(api_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(forecasts_mod, "_DEFAULT_ROW_CAP", 3)

    client = TestClient(app)
    params = {
        "iso3": "ETH",
        "hazard_code": "ACE",
        "metric": "FATALITIES",
        "target_month": "2026-12",
    }
    resp = client.get("/v1/forecasts/history", params=params)
    assert resp.status_code == 200
    payload = resp.json()
    assert len(payload["rows"]) == 3
    assert payload["truncated"] is True

    resp_full = client.get("/v1/forecasts/history", params={**params, "limit": 100})
    assert resp_full.status_code == 200
    assert len(resp_full.json()["rows"]) == N_FORECAST_ROWS


def test_limit_above_hard_max_rejected(api_env: None) -> None:
    client = TestClient(app)
    resp = client.get(
        "/v1/forecasts/ensemble",
        params={"latest_only": "false", "limit": forecasts_mod._MAX_ROW_CAP + 1},
    )
    assert resp.status_code == 422
