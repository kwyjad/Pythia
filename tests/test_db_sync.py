# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from pathlib import Path

import pytest
import requests

from pythia.api import db_sync
from pythia.config import load as load_cfg


class _FakeResponse:
    def __init__(self, *, json_data=None, content=b""):
        self._json_data = json_data
        self._content = content
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._json_data

    def iter_content(self, chunk_size=1024):
        yield self._content


def test_maybe_sync_latest_db_downloads_and_throttles(tmp_path, monkeypatch):
    db_path = tmp_path / "resolver.duckdb"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"app:\n  db_url: 'duckdb:///{db_path}'\n")

    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    load_cfg.cache_clear()

    monkeypatch.setenv("PYTHIA_DATA_REPO", "kwyjad/Pythia")
    monkeypatch.setenv("PYTHIA_DATA_RELEASE_TAG", "pythia-data-latest")
    monkeypatch.setenv("PYTHIA_DATA_DB_ASSET", "resolver.duckdb")
    monkeypatch.setenv("PYTHIA_DATA_MANIFEST_ASSET", "manifest.json")
    monkeypatch.setenv("PYTHIA_DATA_SYNC_INTERVAL_S", "60")

    manifest_payload = {
        "db_sha256": "abc123",
        "latest_hs_run_id": "run-1",
        "latest_hs_created_at": "2024-01-01T00:00:00Z",
    }

    calls = []

    def fake_get(url, headers=None, stream=False, timeout=None):
        calls.append(url)
        # URLs carry a ?t=<epoch> cache-buster, so match on substring.
        if "manifest.json" in url:
            return _FakeResponse(json_data=manifest_payload)
        if "resolver.duckdb" in url:
            return _FakeResponse(content=b"duckdb-bytes")
        raise AssertionError(f"Unexpected URL {url}")

    times = {"value": 0.0}

    def fake_monotonic():
        return times["value"]

    monkeypatch.setattr(db_sync.requests, "get", fake_get)
    monkeypatch.setattr(db_sync.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(db_sync, "_LAST_MANIFEST", None)
    monkeypatch.setattr(db_sync, "_LAST_SYNC_AT", None)

    manifest = db_sync.maybe_sync_latest_db()

    assert db_path.exists()
    assert db_path.read_bytes() == b"duckdb-bytes"
    assert manifest == db_sync.get_cached_manifest()
    assert manifest["db_sha256"] == "abc123"
    assert len(calls) == 2

    times["value"] = 30.0
    manifest_again = db_sync.maybe_sync_latest_db()
    assert manifest_again == manifest
    assert len(calls) == 2


def test_download_db_atomic_clears_stale_temp_directory(tmp_path, monkeypatch):
    """Regression: a leftover ``<dest>.tmp`` *directory* must not wedge sync.

    Reproduces the Render failure ``[Errno 21] Is a directory:
    'resolver.duckdb.tmp'`` where a crashed download left the temp path behind
    as a directory, causing every subsequent sync to fail and freezing the API
    on a stale DB.
    """
    db_path = tmp_path / "resolver.duckdb"

    # Wedged state: the temp path exists as a non-empty directory.
    stale = db_path.with_suffix(db_path.suffix + ".tmp")
    stale.mkdir()
    (stale / "leftover").write_text("junk")
    assert stale.is_dir()

    monkeypatch.setenv("PYTHIA_DATA_REPO", "kwyjad/Pythia")
    monkeypatch.setenv("PYTHIA_DATA_RELEASE_TAG", "pythia-data-latest")
    monkeypatch.setenv("PYTHIA_DATA_DB_ASSET", "resolver.duckdb")

    def fake_get(url, headers=None, stream=False, timeout=None):
        assert "resolver.duckdb" in url
        return _FakeResponse(content=b"fresh-db-bytes")

    monkeypatch.setattr(db_sync.requests, "get", fake_get)

    db_sync.download_db_atomic(db_path)

    assert db_path.is_file()
    assert db_path.read_bytes() == b"fresh-db-bytes"
    # The temp path was consumed by the atomic os.replace and is gone.
    assert not stale.exists()


def test_get_sync_status_reports_drift_on_failed_download(tmp_path, monkeypatch):
    """Models the Render incident: a healthy first sync, then a newer release
    whose DB download fails — ``in_sync`` must report drift, not stay green."""
    db_path = tmp_path / "resolver.duckdb"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"app:\n  db_url: 'duckdb:///{db_path}'\n")

    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
    load_cfg.cache_clear()

    monkeypatch.setenv("PYTHIA_DATA_REPO", "kwyjad/Pythia")
    monkeypatch.setenv("PYTHIA_DATA_RELEASE_TAG", "pythia-data-latest")
    monkeypatch.setenv("PYTHIA_DATA_SYNC_INTERVAL_S", "60")

    monkeypatch.setattr(db_sync, "_LAST_MANIFEST", None)
    monkeypatch.setattr(db_sync, "_LAST_SYNC_AT", None)
    monkeypatch.setattr(db_sync, "_LATEST_RUNS", None)
    monkeypatch.setattr(db_sync, "_LAST_DOWNLOADED_KEY", None)
    monkeypatch.setattr(db_sync, "_LAST_FETCHED_KEY", None)
    monkeypatch.setattr(db_sync, "_LAST_SYNC_ERROR", None)

    times = {"value": 0.0}
    monkeypatch.setattr(db_sync.time, "monotonic", lambda: times["value"])

    state = {"key": "key-1", "db_fails": False}

    def fake_get(url, headers=None, stream=False, timeout=None):
        if "manifest.json" in url:
            return _FakeResponse(
                json_data={"db_sha256": state["key"], "latest_hs_run_id": "run-1"}
            )
        if "resolver.duckdb" in url:
            if state["db_fails"]:
                raise requests.RequestException("boom")
            return _FakeResponse(content=b"db-bytes")
        raise AssertionError(f"Unexpected URL {url}")

    monkeypatch.setattr(db_sync.requests, "get", fake_get)

    # First sync succeeds -> in_sync True.
    db_sync.maybe_sync_latest_db()
    status = db_sync.get_sync_status()
    assert status["last_error"] is None
    assert status["in_sync"] is True
    assert status["last_ok_at"]

    # A newer release appears but its DB download now fails on every attempt.
    times["value"] = 120.0
    state["key"] = "key-2"
    state["db_fails"] = True
    with pytest.raises(db_sync.DbSyncError):
        db_sync.maybe_sync_latest_db()
    status = db_sync.get_sync_status()
    assert status["last_error"]
    assert status["manifest_key"] == "key-2"
    assert status["downloaded_key"] == "key-1"
    assert status["in_sync"] is False  # drift is visible, not silently green
