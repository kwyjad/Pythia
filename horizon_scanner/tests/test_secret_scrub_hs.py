# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Horizon Scanner writers and callers keep the Gemini key out of stored text.

``grounding_debug_json`` ships in the public release DB and is served by
``/v1/question_bundle``; a grounding backend's error used to quote the
request URL, which carried the key.
"""

from __future__ import annotations

import json

import pytest
import requests

KEY = "AIzaSyHSTESTKEY0123456789abcdefgh"


def test_crisiswatch_sends_the_gemini_key_in_a_header(monkeypatch):
    from horizon_scanner import crisiswatch

    calls: list[dict] = []

    def fake_post(url, *args, **kwargs):
        calls.append({"url": url, **kwargs})
        raise requests.ConnectionError(f"failed for url: {url}")

    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setattr(requests, "post", fake_post)
    crisiswatch._call_gemini_grounding("prompt", timeout_sec=1)

    assert calls, "no request was made"
    for call in calls:
        assert KEY not in call["url"]
        assert KEY not in json.dumps(call.get("params") or {})
        assert (call.get("headers") or {}).get("x-goog-api-key") == KEY


def test_a_grounding_pack_debug_payload_is_stored_without_the_key(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")

    from horizon_scanner.db_writer import log_hs_hazard_tail_packs_to_db
    from pythia.db.schema import ensure_schema
    from resolver.db import duckdb_io

    db_path = tmp_path / "packs.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")
    ensure_schema()

    pack = {
        "iso3": "som",
        "hazard_code": "fl",
        "query": "rc_grounding: flood",
        "markdown": "## Flood signals",
        "sources": [],
        "grounded": False,
        "grounding_debug": {
            "provider_error_message": f"exception: ConnectionError('url: /m?key={KEY}')",
        },
    }
    log_hs_hazard_tail_packs_to_db("hs_run_scrub", [pack], is_test=True)

    conn = duckdb_io.get_db(str(db_path))
    try:
        stored = conn.execute(
            "SELECT grounding_debug_json FROM hs_hazard_tail_packs WHERE hs_run_id = 'hs_run_scrub'"
        ).fetchone()[0]
    finally:
        duckdb_io.close_db(conn)

    assert KEY not in stored
    assert "<redacted>" in stored
    json.loads(stored)  # still valid JSON
