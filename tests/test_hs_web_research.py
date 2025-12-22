# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

import horizon_scanner.horizon_scanner as hs
from pythia.db.schema import connect, ensure_schema


def test_hs_country_report_persisted_when_web_research_enabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "test.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_ENABLED", "1")

    fake_pack = {
        "query": "Kenya outlook",
        "recency_days": 120,
        "structural_context": "struct",
        "recent_signals": ["signal-a"],
        "sources": [{"title": "Test", "url": "http://example.com"}],
        "grounded": True,
    }

    calls: dict[str, object] = {}

    def fake_fetch_evidence_pack(query: str, purpose: str, run_id: str | None = None, question_id: str | None = None):
        calls["query"] = query
        calls["purpose"] = purpose
        calls["run_id"] = run_id
        calls["question_id"] = question_id
        return fake_pack

    monkeypatch.setattr(hs, "fetch_evidence_pack", fake_fetch_evidence_pack)

    pack = hs._maybe_build_country_evidence_pack("hs-run", "KEN", "Kenya")

    assert pack is not None
    assert pack.get("markdown", "").startswith("# Evidence pack")

    con = connect(read_only=False)
    ensure_schema(con)
    rows = con.execute(
        "SELECT report_markdown, sources_json, grounded, grounding_debug_json, structural_context, recent_signals_json FROM hs_country_reports WHERE hs_run_id = ? AND iso3 = ?",
        ["hs-run", "KEN"],
    ).fetchall()
    con.close()

    assert len(rows) == 1
    stored_sources = json.loads(rows[0][1])
    assert stored_sources and stored_sources[0]["url"] == "http://example.com"
    assert rows[0][2] is True
    debug = json.loads(rows[0][3])
    assert isinstance(debug, dict)
    assert "struct" in (rows[0][4] or "")
    assert "signal-a" in json.loads(rows[0][5])[0]
