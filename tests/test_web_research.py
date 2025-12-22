# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.web_research import web_research
from pythia.web_research.backends.gemini_grounding import parse_gemini_grounding_response
from pythia.web_research.budget import BudgetGuard


def test_parse_gemini_grounding_sets_grounded_true():
    resp = {
        "candidates": [
            {
                "groundingMetadata": {
                    "webSearchQueries": ["foo"],
                    "groundingSupports": [
                        {
                            "support": {
                                "sourceUrl": "https://example.com",
                                "title": "Example",
                                "publisher": "News",
                                "publishedDate": "2025-01-01",
                                "summary": "Summary here",
                            }
                        }
                    ],
                }
            }
        ]
    }
    sources, grounded, debug = parse_gemini_grounding_response(resp)
    assert grounded is True
    assert len(sources) == 1
    assert sources[0].url == "https://example.com"
    assert debug["groundingSupports_count"] == 1


def test_budget_guard_blocks_followup_calls(monkeypatch, tmp_path):
    """Budget caps should block additional fetches without invoking backend."""

    # Reset budget state and configure tight caps
    BudgetGuard._STATE = {}
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_MAX_CALLS_PER_QUESTION", "1")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_CACHE", "0")

    # Use an in-memory DuckDB for logging
    db_path = tmp_path / "wr.duckdb"
    def _conn(read_only: bool = False):
        return duckdb.connect(str(db_path), read_only=read_only)

    monkeypatch.setattr(web_research, "connect", _conn)

    calls: list[str] = []

    def fake_fetch(**kwargs):
        calls.append(kwargs.get("query", ""))
        pack = web_research.EvidencePack(query=kwargs.get("query", ""), recency_days=120, backend="gemini")
        pack.grounded = True
        pack.sources = []
        return pack

    monkeypatch.setattr(web_research.gemini_grounding, "fetch_via_gemini", fake_fetch)

    first = web_research.fetch_evidence_pack("query one", purpose="hs", run_id="run-budget", question_id="Q1")
    assert first["error"] is None
    second = web_research.fetch_evidence_pack("query two", purpose="hs", run_id="run-budget", question_id="Q1")
    assert second["error"]["type"] == "budget_exceeded"
    assert len(calls) == 1
    # Ensure llm_calls logging used the provided DB
    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
    finally:
        con.close()
    assert rows == 2
