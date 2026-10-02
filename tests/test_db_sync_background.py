# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The API keeps itself current without waiting for a visitor (Oct 2026).

The 1 October 2026 forecasts were published green and the dashboard kept
showing September. The API only ever fetched the release when a request
arrived, downloaded inside that request while holding the sync lock, and
swapped in whatever bytes arrived. These tests pin the repairs: a background
loop that syncs and swaps on its own, requests that never queue behind a
download, a download refused before the swap when its hash or length is
wrong, and a disk shortfall named in words.
"""

from __future__ import annotations

import hashlib
import threading
import time

import pytest

from pythia.api import db_sync
from pythia.config import load as load_cfg


class _Resp:
    def __init__(self, *, json_data=None, content=b"", length=None):
        self._json = json_data
        self._content = content
        self.status_code = 200
        self.headers = {} if length is None else {"Content-Length": str(length)}

    def raise_for_status(self):
        return None

    def json(self):
        return self._json

    def iter_content(self, chunk_size=1024):
        yield self._content


@pytest.fixture
def env(tmp_path, monkeypatch):
    db_path = tmp_path / "resolver.duckdb"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"app:\n  db_url: 'duckdb:///{db_path}'\n")
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(cfg))
    load_cfg.cache_clear()
    monkeypatch.setenv("PYTHIA_DATA_REPO", "kwyjad/Pythia")
    monkeypatch.setenv("PYTHIA_DATA_RELEASE_TAG", "pythia-data-latest")
    monkeypatch.setenv("PYTHIA_DATA_SYNC_INTERVAL_S", "0")
    for name, value in (
        ("_LAST_MANIFEST", None),
        ("_LAST_SYNC_AT", None),
        ("_LAST_SYNC_ERROR", None),
        ("_LAST_DOWNLOADED_KEY", None),
        ("_LAST_FETCHED_KEY", None),
        ("_LATEST_RUNS", {}),
    ):
        monkeypatch.setattr(db_sync, name, value)
    monkeypatch.setattr(db_sync, "_refresh_latest_runs", lambda p: None)
    db_sync._DB_REFRESHED.clear()
    yield db_path
    db_sync.stop_background_sync()
    db_sync._DB_REFRESHED.clear()
    load_cfg.cache_clear()


def _serve(monkeypatch, manifest, content, length=None):
    def fake_get(url, headers=None, stream=False, timeout=None):
        if "manifest.json" in url:
            return _Resp(json_data=dict(manifest))
        return _Resp(content=content, length=length)

    monkeypatch.setattr(db_sync.requests, "get", fake_get)


def test_a_matching_sha256_is_swapped_in(env, monkeypatch):
    content = b"new-db"
    sha = hashlib.sha256(content).hexdigest()
    _serve(monkeypatch, {"db_sha256": sha}, content, length=len(content))
    db_sync.maybe_sync_latest_db()
    assert env.read_bytes() == content
    assert db_sync.get_sync_status()["in_sync"] is True


def test_a_sha256_mismatch_never_replaces_the_served_db(env, monkeypatch):
    env.write_bytes(b"old-good-db")
    _serve(monkeypatch, {"db_sha256": "0" * 64}, b"corrupt")
    with pytest.raises(db_sync.DbSyncError, match="does not match"):
        db_sync.maybe_sync_latest_db()
    assert env.read_bytes() == b"old-good-db"
    assert not env.with_suffix(".duckdb.tmp").exists()
    status = db_sync.get_sync_status()
    assert status["in_sync"] is not True
    assert "does not match" in status["last_error"]


def test_a_truncated_download_is_refused(env, monkeypatch):
    env.write_bytes(b"old-good-db")
    _serve(monkeypatch, {"db_sha256": "run-key"}, b"half", length=100)
    with pytest.raises(db_sync.DbSyncError, match="Truncated"):
        db_sync.maybe_sync_latest_db()
    assert env.read_bytes() == b"old-good-db"


def test_a_disk_too_small_for_two_copies_is_named(env, monkeypatch):
    env.write_bytes(b"old")
    _serve(monkeypatch, {"db_sha256": "k"}, b"x" * 10, length=800_000_000)

    class _Usage:
        free = 500_000_000

    monkeypatch.setattr(db_sync.shutil, "disk_usage", lambda p: _Usage())
    with pytest.raises(db_sync.DbSyncError, match="Insufficient disk space"):
        db_sync.maybe_sync_latest_db()
    assert "two copies" in db_sync.get_sync_status()["last_error"]
    assert env.read_bytes() == b"old"


def test_a_request_never_waits_behind_a_running_download(env, monkeypatch):
    db_sync._LAST_MANIFEST = {"db_sha256": "cached"}
    calls = []
    monkeypatch.setattr(db_sync, "fetch_manifest", lambda: calls.append(1) or {})
    assert db_sync._SYNC_LOCK.acquire()
    try:
        started = time.monotonic()
        result = db_sync.maybe_sync_latest_db()
        assert time.monotonic() - started < 1.0
    finally:
        db_sync._SYNC_LOCK.release()
    assert result["db_sha256"] == "cached"
    assert calls == []


def test_the_background_loop_syncs_and_swaps_without_a_request(env, monkeypatch):
    content = b"october-db"
    sha = hashlib.sha256(content).hexdigest()
    _serve(monkeypatch, {"db_sha256": sha, "latest_hs_run_id": "hs_20261001"}, content)
    monkeypatch.setenv("PYTHIA_DATA_SYNC_INTERVAL_S", "15")
    swapped = threading.Event()

    assert db_sync.start_background_sync(on_refresh=swapped.set) is True
    assert db_sync.background_sync_running()
    assert swapped.wait(5.0), "the loop never swapped the connection"
    assert env.read_bytes() == content
    # The loop consumed the flag, so a request will not swap a second time.
    assert db_sync.db_was_refreshed() is False


def test_background_sync_can_be_switched_off(env, monkeypatch):
    monkeypatch.setenv("PYTHIA_DATA_BACKGROUND_SYNC", "0")
    assert db_sync.start_background_sync() is False
    assert not db_sync.background_sync_running()


def test_background_sync_needs_a_release_to_poll(env, monkeypatch):
    monkeypatch.delenv("PYTHIA_DATA_REPO")
    assert db_sync.start_background_sync() is False


def test_a_crashing_sync_does_not_kill_the_loop(env, monkeypatch):
    attempts = []

    def boom(wait=False):
        attempts.append(1)
        raise RuntimeError("unexpected")

    monkeypatch.setattr(db_sync, "maybe_sync_latest_db", boom)
    monkeypatch.setattr(db_sync._BG_STOP, "wait", lambda t: len(attempts) >= 3)
    db_sync._BG_STOP.clear()
    db_sync._background_sync_loop(None, 15)
    assert len(attempts) == 3


def _make_db(path, value):
    import duckdb

    con = duckdb.connect(str(path))
    con.execute(f"CREATE TABLE marker AS SELECT {value} AS v")
    con.close()


def test_a_swap_serves_the_new_file_even_with_a_pooled_connection_open(env, monkeypatch):
    """The October 2026 trap, rebuilt.

    The API's startup schema pass "closed" a pythia.db.schema connection,
    which only returned it to the pool, still OPEN. DuckDB hands every new
    connection to a path the instance already open there, so each swap after
    a release download reopened onto the boot-time file and the dashboard
    served last month until the process restarted.
    """
    from pythia.api import core
    from pythia.db import schema

    _make_db(env, 1)
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{env}")
    pooled = schema.connect()
    pooled.close()  # back to the pool, still open: the trap
    monkeypatch.setattr(core, "_READ_CON", None)
    monkeypatch.setattr(core, "maybe_sync_latest_db", lambda *a, **k: None)
    try:
        con = core._ensure_read_connection()
        assert con.execute("SELECT v FROM marker").fetchone()[0] == 1

        new = env.with_name("incoming.duckdb")
        _make_db(new, 2)
        import os

        os.replace(new, env)
        assert core._swap_read_connection()
        assert core._READ_CON.execute("SELECT v FROM marker").fetchone()[0] == 2
    finally:
        if core._READ_CON is not None:
            core._READ_CON.close()
        monkeypatch.setattr(core, "_READ_CON", None)
        schema.close_pooled_connections()


def test_served_is_not_the_same_as_downloaded(env, monkeypatch):
    monkeypatch.setattr(db_sync, "_LAST_FETCHED_KEY", "new")
    monkeypatch.setattr(db_sync, "_LAST_DOWNLOADED_KEY", "new")
    monkeypatch.setattr(db_sync, "_SERVED_KEY", "old")
    status = db_sync.get_sync_status()
    assert status["in_sync"] is False
    assert status["served_key"] == "old"
