# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""/v1/version staleness probes must run once per DB version, not per request.

The dashboard homepage is force-dynamic and hits /v1/version on every page
view; the probes are 6 full-column MAX scans. They are cached keyed by the
DB file mtime, which the sync layer advances via os.replace on refresh.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest

pytest.importorskip("fastapi")
duckdb = pytest.importorskip("duckdb")

from fastapi.testclient import TestClient

from pythia import config as pythia_config
import pythia.api.app as _app_mod
from pythia.api.app import app


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Generator[Path, None, None]:
    db_path = tmp_path / "api.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE forecasts_ensemble (run_id TEXT, created_at TIMESTAMP)")
    con.execute("INSERT INTO forecasts_ensemble VALUES ('fc_1', TIMESTAMP '2026-07-01 10:00:00')")
    con.close()

    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"app:\n  db_url: 'duckdb:///{db_path}'\n", encoding="utf-8")
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    pythia_config.load.cache_clear()
    _app_mod._READ_CON = None
    _app_mod._VERSION_PROBE_CACHE = None

    monkeypatch.setattr(
        _app_mod,
        "maybe_sync_latest_db",
        lambda: {"db_sha256": "test", "latest_hs_created_at": "2026-07-01T09:00:00"},
    )

    try:
        yield db_path
    finally:
        _app_mod._READ_CON = None
        _app_mod._VERSION_PROBE_CACHE = None
        pythia_config.load.cache_clear()


def test_version_probes_cached_until_db_mtime_changes(api_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    real = _app_mod._compute_staleness_probes

    def _counting(include_test: bool = False):
        calls["n"] += 1
        return real(include_test)

    monkeypatch.setattr(_app_mod, "_compute_staleness_probes", _counting)
    client = TestClient(app)

    r1 = client.get("/v1/version")
    assert r1.status_code == 200
    assert r1.json()["latest_forecast_month"] == "2026-07"
    assert r1.json()["latest_forecast_run_id"] == "fc_1"
    assert calls["n"] == 1

    r2 = client.get("/v1/version")
    assert r2.status_code == 200
    assert r2.json()["latest_forecast_month"] == "2026-07"
    assert calls["n"] == 1  # served from cache — same DB version

    # A DB refresh (sync's os.replace advances the mtime) invalidates — once
    # the throttled refresh check runs and reopens the read connection (the
    # cache is keyed on the connection's open-time mtime, not the raw file
    # mtime, so probes are only recomputed when new data is actually visible).
    st = os.stat(api_env)
    os.utime(api_env, (st.st_atime, st.st_mtime + 5))
    _app_mod._LAST_SYNC_CHECK = None  # expire the 60s refresh throttle
    r3 = client.get("/v1/version")
    assert r3.status_code == 200
    assert calls["n"] == 2


def test_version_latest_data_at_combines_manifest_and_probes(api_env: Path) -> None:
    client = TestClient(app)
    resp = client.get("/v1/version")
    assert resp.status_code == 200
    payload = resp.json()
    # forecasts_ensemble created_at (10:00) beats the manifest HS timestamp (09:00).
    assert payload["latest_data_at"] == "2026-07-01T10:00:00"


def test_version_probes_not_pinned_by_throttled_refresh(api_env: Path) -> None:
    """A DB swap during the 60s refresh-throttle window must not pin the
    pre-swap probes under the new DB version: the first request after the
    swap still reads the old inode, and keying the cache on the raw file
    mtime would cache those stale probes under the new mtime forever."""
    import time as _time

    client = TestClient(app)
    r1 = client.get("/v1/version")
    assert r1.json()["latest_forecast_run_id"] == "fc_1"

    # Publish a new DB (sync's os.replace semantics) with fresher data.
    new_db = api_env.parent / "new.duckdb"
    con = duckdb.connect(str(new_db))
    con.execute("CREATE TABLE forecasts_ensemble (run_id TEXT, created_at TIMESTAMP)")
    con.execute("INSERT INTO forecasts_ensemble VALUES ('fc_2', TIMESTAMP '2026-07-02 10:00:00')")
    con.close()
    os.replace(new_db, api_env)
    st = os.stat(api_env)
    os.utime(api_env, (st.st_atime, st.st_mtime + 5))  # mtime strictly advances

    # Request inside the throttle window: connection still reads the old inode.
    _app_mod._LAST_SYNC_CHECK = _time.monotonic()
    r2 = client.get("/v1/version")
    assert r2.json()["latest_forecast_run_id"] == "fc_1"

    # Throttle expires -> connection reopens -> probes must show the new data.
    _app_mod._LAST_SYNC_CHECK = None
    r3 = client.get("/v1/version")
    assert r3.json()["latest_forecast_run_id"] == "fc_2"
