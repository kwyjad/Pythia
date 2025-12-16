# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
from pathlib import Path

from resolver.ingestion import dtm_client


def _reset_paths(tmp_path: Path) -> None:
    dtm_client.OUT_DIR = tmp_path
    dtm_client.OUT_PATH = tmp_path / "dtm.csv"
    dtm_client.HTTP_TRACE_PATH = dtm_client.OUT_DIR / "dtm_http.ndjson"
    dtm_client.META_PATH = dtm_client.OUT_PATH.with_suffix(dtm_client.OUT_PATH.suffix + ".meta.json")


def test_skip_writes_header_and_trace(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RESOLVER_SKIP_DTM", "1")
    _reset_paths(Path(tmp_path))

    # no strict-empty -> exit 0
    rc = dtm_client.main([])
    assert rc == 0

    csv_path = dtm_client.OUT_PATH
    assert csv_path.exists()
    with csv_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().strip()
        assert header
        assert ",".join(dtm_client.CANONICAL_HEADERS) == header

    http_trace = dtm_client.HTTP_TRACE_PATH
    assert http_trace.exists()
    with http_trace.open("r", encoding="utf-8") as handle:
        line = handle.readline().strip()
        assert line
        payload = json.loads(line)
        assert payload.get("offline") is True


def test_skip_strict_empty_exits_2(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("RESOLVER_SKIP_DTM", "1")
    _reset_paths(Path(tmp_path))
    rc = dtm_client.main(["--strict-empty"])
    assert rc == 2
