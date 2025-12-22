# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
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


def test_fetch_via_gemini_parses_grounding(monkeypatch):
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_ENABLED", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "candidates": [
                    {
                        "groundingMetadata": {
                            "groundingSupports": [
                                {
                                    "support": {
                                        "sourceUrl": "https://example.com/one",
                                        "title": "Example One",
                                        "publisher": "Example Publisher",
                                    }
                                }
                            ],
                            "groundingChunks": [
                                {
                                    "sourceUrl": "https://example.com/two",
                                    "title": "Second",
                                    "publisher": "Example Publisher",
                                }
                            ],
                            "webSearchQueries": ["example query"],
                        },
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "structural_context": "Line 1\nLine 2",
                                            "recent_signals": ["Signal 1", "Signal 2"],
                                            "notes": "note text",
                                        }
                                    )
                                }
                            ]
                        },
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 20,
                    "totalTokenCount": 30,
                },
            }

    monkeypatch.setattr(web_research.gemini_grounding.requests, "post", lambda *args, **kwargs: FakeResponse())

    pack = web_research.gemini_grounding.fetch_via_gemini(
        "test query",
        recency_days=120,
        include_structural=True,
        timeout_sec=30,
        max_results=5,
    )

    assert pack.grounded is True
    assert len(pack.sources) == 2
    assert pack.sources[0].url == "https://example.com/one"
    assert pack.sources[1].url == "https://example.com/two"
    assert pack.structural_context.startswith("Line 1")
    assert pack.recent_signals == ["Signal 1", "Signal 2"]
    assert pack.debug.get("usage", {}).get("total_tokens") == 30
