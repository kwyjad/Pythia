# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Download endpoints: temp-file serving must be byte-identical to the old
streaming path, release the heavy semaphore at build end (not stream end),
and clean up its temp files.

The export builders themselves are covered by resolver/tests; here they are
stubbed so these tests exercise only the serving path that changed.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest

pytest.importorskip("fastapi")
duckdb = pytest.importorskip("duckdb")

from fastapi.testclient import TestClient

from pythia import config as pythia_config
import pythia.api.app as _app_mod
import pythia.api.routes.downloads as downloads_mod
from pythia.api import core as _core
from pythia.api.app import app


def _fake_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ISO": ["ETH", "SOM", "AFG"],
            "value": [1.5, float("nan"), -3.25],
            "count": [10, 20, 0],
            # Note: no empty strings — Excel round-trips "" as NaN, which is an
            # openpyxl artifact shared with the old BytesIO path, not a serving bug.
            "note": ["a,b", 'quo"ted', "plain"],
        }
    )


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    db_path = tmp_path / "api.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE questions (question_id TEXT)")
    con.close()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"app:\n  db_url: 'duckdb:///{db_path}'\n", encoding="utf-8")
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))

    export_dir = tmp_path / "exports"
    monkeypatch.setenv("PYTHIA_EXPORT_TMP_DIR", str(export_dir))

    pythia_config.load.cache_clear()
    _app_mod._READ_CON = None
    try:
        yield export_dir
    finally:
        _app_mod._READ_CON = None
        pythia_config.load.cache_clear()


def test_forecasts_csv_bytes_identical(api_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = _fake_df()
    monkeypatch.setattr(downloads_mod, "build_forecast_spd_export", lambda con, include_test=False: df.copy())

    client = TestClient(app)
    resp = client.get("/v1/downloads/forecasts.csv")

    assert resp.status_code == 200
    assert resp.content == df.to_csv(index=False).encode()
    assert resp.headers["content-disposition"] == 'attachment; filename="pythia_forecasts_export.csv"'
    assert resp.headers["content-type"].startswith("text/csv")


def test_forecasts_xlsx_round_trip(api_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openpyxl")
    df = _fake_df()
    monkeypatch.setattr(downloads_mod, "build_forecast_spd_export", lambda con, include_test=False: df.copy())

    client = TestClient(app)
    resp = client.get("/v1/downloads/forecasts.xlsx")

    assert resp.status_code == 200
    assert resp.headers["content-disposition"] == 'attachment; filename="pythia_forecasts_export.xlsx"'
    round_trip = pd.read_excel(BytesIO(resp.content))
    pd.testing.assert_frame_equal(round_trip, df, check_dtype=False)


def test_semaphore_released_before_streaming(api_env: Path) -> None:
    """The new guarantee: the heavy semaphore is released when the build
    finishes, not when the client finishes downloading."""
    initial = _core._HEAVY_REQUEST_SEMAPHORE._value

    resp = _core._serve_export(_fake_df, "x.csv", build_error_detail="boom")

    # Released before the response object is even returned to the framework.
    assert _core._HEAVY_REQUEST_SEMAPHORE._value == initial
    # DataFrame was spilled to disk; the file exists until the background task runs.
    assert os.path.exists(resp.path)
    resp.background.func(*resp.background.args)
    assert not os.path.exists(resp.path)


def test_tmp_file_cleaned_after_request(api_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(downloads_mod, "build_forecast_spd_export", lambda con, include_test=False: _fake_df())

    client = TestClient(app)
    resp = client.get("/v1/downloads/forecasts.csv")
    assert resp.status_code == 200

    # TestClient runs background tasks synchronously after the response.
    leftovers = list(api_env.glob("*")) if api_env.exists() else []
    assert leftovers == []


def test_build_error_returns_500_and_releases_semaphore(api_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(con, include_test=False):
        raise ValueError("builder exploded")

    monkeypatch.setattr(downloads_mod, "build_forecast_spd_export", _boom)
    initial = _core._HEAVY_REQUEST_SEMAPHORE._value

    client = TestClient(app)
    resp = client.get("/v1/downloads/forecasts.csv")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to build forecast download export"
    assert _core._HEAVY_REQUEST_SEMAPHORE._value == initial
    leftovers = list(api_env.glob("*")) if api_env.exists() else []
    assert leftovers == []
