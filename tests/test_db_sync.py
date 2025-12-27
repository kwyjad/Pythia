# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from pathlib import Path

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
        if url.endswith("manifest.json"):
            return _FakeResponse(json_data=manifest_payload)
        if url.endswith("resolver.duckdb"):
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
