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
                    "groundingSupports": [{"support": {"sourceUrl": "https://example.com/support"}}],
                    "groundingChunks": [
                        {
                            "web": {
                                "uri": "https://example.com",
                                "title": "Example",
                            }
                        },
                        {
                            "web": {
                                "uri": "https://example.com",  # duplicate should be deduped
                                "title": "Example Duplicate",
                            }
                        },
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
    assert debug["groundingChunks_count"] == 2


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
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_MODEL_ID", "gemini-3-pro-preview")

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "candidates": [
                    {
                        "groundingMetadata": {
                            "groundingChunks": [
                                {
                                    "web": {"uri": "https://example.com/one", "title": "Example One"},
                                },
                                {
                                    "web": {"uri": "https://example.com/two", "title": "Second"},
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
    assert "gemini-3-pro-preview" in pack.debug.get("attempted_models", [])
    assert pack.debug.get("selected_model_id") == "gemini-3-pro-preview"
    assert pack.unverified_sources == []


def test_fetch_via_gemini_preserves_recent_signals_without_structural(monkeypatch):
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_ENABLED", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "candidates": [
                    {
                        "groundingMetadata": {
                            "groundingSupports": [{"support": {"sourceUrl": "https://example.com", "title": "Example"}}],
                            "groundingChunks": [{"web": {"uri": "https://example.com", "title": "Example"}}],
                            "webSearchQueries": ["example"],
                        },
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {
                                            "structural_context": "Background line 1",
                                            "recent_signals": ["Signal A", "Signal B"],
                                        }
                                    )
                                }
                            ]
                        },
                    }
                ],
                "usageMetadata": {},
            }

    monkeypatch.setattr(web_research.gemini_grounding.requests, "post", lambda *args, **kwargs: FakeResponse())

    pack = web_research.gemini_grounding.fetch_via_gemini(
        "test query",
        recency_days=120,
        include_structural=False,
        timeout_sec=30,
        max_results=5,
    )

    assert pack.structural_context == ""
    assert pack.recent_signals == ["Signal A", "Signal B"]
    assert pack.grounded is True


def test_fetch_via_gemini_extracts_unverified_urls(monkeypatch):
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_ENABLED", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": "Here are some leads: https://example.com/a and also https://example.com/b ."
                                }
                            ]
                        }
                    }
                ]
            }

    monkeypatch.setattr(web_research.gemini_grounding.requests, "post", lambda *args, **kwargs: FakeResponse())

    pack = web_research.gemini_grounding.fetch_via_gemini(
        "test query",
        recency_days=120,
        include_structural=True,
        timeout_sec=30,
        max_results=5,
    )

    assert pack.grounded is False
    assert pack.sources == []
    assert [src.url for src in pack.unverified_sources] == ["https://example.com/a", "https://example.com/b"]
    assert pack.debug.get("unverified_url_count") == 2


def test_fetch_via_openai_web_search_parses_sources(monkeypatch):
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_MODEL_ID", "gpt-4.1")

    class FakeResponse:
        def model_dump(self):
            return {
                "output": [
                    {
                        "type": "web_search_call",
                        "web_search_call": {
                            "action": {
                                "sources": [
                                    {"url": "https://example.com/a", "title": "Example A"},
                                    {"url": "https://example.com/b", "title": "Example B"},
                                ]
                            }
                        },
                    },
                    {
                        "type": "message",
                        "message": {
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps(
                                        {
                                            "structural_context": "Context line",
                                            "recent_signals": ["Sig 1"],
                                        }
                                    ),
                                    "annotations": [{"url": "https://example.com/c"}],
                                }
                            ]
                        },
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30, "web_search_requests": 1},
            }

    class FakeResponses:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return FakeResponse()

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.responses = FakeResponses()

    monkeypatch.setattr(web_research.openai_web_search, "OpenAI", FakeClient)

    pack = web_research.openai_web_search.fetch_via_openai_web_search(
        "test query",
        recency_days=120,
        include_structural=True,
        timeout_sec=30,
        max_results=5,
    )

    assert pack.backend == "openai"
    assert pack.grounded is True
    assert [s.url for s in pack.sources] == [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]
    assert pack.structural_context.startswith("Context line")
    assert pack.recent_signals == ["Sig 1"]
    assert pack.debug.get("provider") == "openai"
    assert pack.debug.get("model_id") == "gpt-4.1"
    assert pack.debug.get("usage", {}).get("web_search_requests") == 1
    assert pack.debug.get("n_verified_sources") == 3


def test_fetch_via_claude_web_search_parses_sources(monkeypatch):
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_ENABLED", "1")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_MODEL_ID", "claude-3-opus")

    class FakeResponse:
        def model_dump(self):
            return {
                "content": [
                    {
                        "type": "web_search_tool_result",
                        "results": [
                            {"url": "https://example.com/one", "title": "One", "page_age": "2d"},
                            {"url": "https://example.com/two", "title": "Two"},
                        ],
                    },
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "structural_context": "Line A",
                                        "recent_signals": ["Recent A"],
                                    }
                                ),
                            }
                        ],
                    },
                ],
                "usage": {"input_tokens": 11, "output_tokens": 21},
            }

    class FakeMessages:
        def create(self, **kwargs):
            self.kwargs = kwargs
            return FakeResponse()

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.messages = FakeMessages()

    monkeypatch.setattr(web_research.claude_web_search, "Anthropic", FakeClient)

    pack = web_research.claude_web_search.fetch_via_claude_web_search(
        "test query",
        recency_days=120,
        include_structural=True,
        timeout_sec=30,
        max_results=5,
    )

    assert pack.backend == "claude"
    assert pack.grounded is True
    assert [s.url for s in pack.sources] == ["https://example.com/one", "https://example.com/two"]
    assert pack.structural_context == "Line A"
    assert pack.recent_signals == ["Recent A"]
    assert pack.debug.get("provider") == "anthropic"
    assert pack.debug.get("model_id") == "claude-3-opus"
    assert pack.debug.get("usage", {}).get("total_tokens") == 32
    assert pack.debug.get("n_verified_sources") == 2


def test_auto_backend_prefers_openai_when_gemini_missing(monkeypatch):
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_BACKEND", "auto")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_CACHE", "0")

    gemini_pack = web_research.EvidencePack(query="q", recency_days=120, backend="gemini")
    gemini_pack.error = {"type": "grounding_missing", "message": "no grounding"}
    gemini_pack.grounded = False

    openai_pack = web_research.EvidencePack(query="q", recency_days=120, backend="openai")
    openai_pack.grounded = True
    openai_pack.sources = [web_research.EvidenceSource(title="Example", url="https://example.com")]
    claude_calls = {"count": 0}

    def _claude_fetch(**kwargs):
        claude_calls["count"] += 1
        pack = web_research.EvidencePack(query="q", recency_days=120, backend="claude")
        pack.error = {"type": "grounding_missing", "message": "no sources"}
        pack.grounded = False
        return pack

    monkeypatch.setattr(web_research.gemini_grounding, "fetch_via_gemini", lambda *args, **kwargs: gemini_pack)
    monkeypatch.setattr(web_research.openai_web_search, "fetch_via_openai_web_search", lambda *args, **kwargs: openai_pack)
    monkeypatch.setattr(web_research.claude_web_search, "fetch_via_claude_web_search", _claude_fetch)

    pack = web_research.fetch_evidence_pack("query", purpose="hs", run_id="run-auto-openai", question_id="Q-openai")

    assert pack["backend"] == "openai"
    assert pack["grounded"] is True
    assert pack["sources"][0]["url"] == "https://example.com"
    assert pack["debug"].get("selected_backend") == "openai"
    assert pack["debug"].get("attempted_backends") == ["gemini", "openai"]
    assert claude_calls["count"] == 0


def test_auto_backend_prefers_claude_when_openai_missing(monkeypatch):
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_BACKEND", "auto")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_CACHE", "0")

    gemini_pack = web_research.EvidencePack(query="q", recency_days=120, backend="gemini")
    gemini_pack.error = {"type": "grounding_missing", "message": "no grounding"}
    gemini_pack.grounded = False

    openai_pack = web_research.EvidencePack(query="q", recency_days=120, backend="openai")
    openai_pack.error = {"type": "grounding_missing", "message": "no sources"}
    openai_pack.grounded = False

    claude_pack = web_research.EvidencePack(query="q", recency_days=120, backend="claude")
    claude_pack.grounded = True
    claude_pack.sources = [web_research.EvidenceSource(title="Claude Source", url="https://claude.example.com")]

    monkeypatch.setattr(web_research.gemini_grounding, "fetch_via_gemini", lambda *args, **kwargs: gemini_pack)
    monkeypatch.setattr(web_research.openai_web_search, "fetch_via_openai_web_search", lambda *args, **kwargs: openai_pack)
    monkeypatch.setattr(web_research.claude_web_search, "fetch_via_claude_web_search", lambda *args, **kwargs: claude_pack)

    pack = web_research.fetch_evidence_pack("query", purpose="hs", run_id="run-auto-claude", question_id="Q-claude")

    assert pack["backend"] == "claude"
    assert pack["grounded"] is True
    assert pack["sources"][0]["url"] == "https://claude.example.com"
    assert pack["debug"].get("selected_backend") == "claude"
    assert pack["debug"].get("attempted_backends") == ["gemini", "openai", "claude"]


def test_fetch_via_gemini_retries_once_when_missing_grounding(monkeypatch):
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_ENABLED", "1")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    calls = []

    class FakeResponse:
        status_code = 200

        def json(self):
            # Deliberately no groundingMetadata to trigger retry
            return {
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {"structural_context": "", "recent_signals": ["Signal A"], "notes": ""}
                                    )
                                }
                            ]
                        }
                    }
                ]
            }

    def fake_post(url, params=None, json=None, timeout=None):
        calls.append({"url": url, "body": json})
        return FakeResponse()

    monkeypatch.setattr(web_research.gemini_grounding.requests, "post", fake_post)

    pack = web_research.gemini_grounding.fetch_via_gemini(
        "test query",
        recency_days=120,
        include_structural=True,
        timeout_sec=30,
        max_results=5,
    )

    assert pack.grounded is False
    assert pack.error["type"] == "grounding_missing"
    assert pack.debug.get("retry_used") is True
    assert pack.debug.get("retry_success") is False
    assert len(calls) == 2
    assert "You must use Google Search" in calls[1]["body"]["contents"][0]["parts"][0]["text"]


def test_web_research_logging_uses_provider_and_model(monkeypatch):
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_ENABLED", "1")
    called = {}

    class FakeConn:
        def close(self):
            pass

    def fake_connect(read_only: bool = False):
        return FakeConn()

    def fake_log_call(conn, **kwargs):
        called.update(kwargs)

    fake_pack = web_research.EvidencePack(query="q", recency_days=120, backend="gemini")
    fake_pack.debug = {
        "provider": "google",
        "selected_model_id": "gemini-3-flash-preview",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost_usd": 0.25},
    }

    monkeypatch.setattr(web_research, "connect", fake_connect)
    monkeypatch.setattr(web_research, "ensure_schema", lambda conn: None)
    monkeypatch.setattr(web_research, "log_web_research_call", fake_log_call)

    web_research._log_web_research(
        fake_pack,
        purpose="hs",
        run_id="run1",
        question_id="Q1",
        start_ms=int(time.time() * 1000),
        cached=False,
        success=True,
    )

    assert called.get("provider") == "google"
    assert called.get("model_id") == "gemini-3-flash-preview"
    assert called.get("model_name") == "Gemini Grounding"


def test_budget_guard_records_actual_cost(monkeypatch):
    BudgetGuard._STATE = {}
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_BUDGET_USD_PER_RUN", "1")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_CACHE", "0")

    fake_pack = web_research.EvidencePack(query="q", recency_days=120, backend="gemini")
    fake_pack.debug = {"usage": {"cost_usd": 2.0}}

    def fake_fetch(**kwargs):
        return fake_pack

    monkeypatch.setattr(web_research.gemini_grounding, "fetch_via_gemini", fake_fetch)
    monkeypatch.setattr(web_research, "_log_web_research", lambda *args, **kwargs: None)

    first = web_research.fetch_evidence_pack("query one", purpose="hs", run_id="budget-run", question_id="Q1")
    assert first["error"] is None

    second = web_research.fetch_evidence_pack("query two", purpose="hs", run_id="budget-run", question_id="Q1")
    assert second["error"]["type"] == "budget_exceeded"


def test_auto_backend_uses_fallback_when_configured(monkeypatch):
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_BACKEND", "auto")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_FALLBACK_BACKEND", "exa")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_CACHE", "0")

    gemini_pack = web_research.EvidencePack(query="q", recency_days=120, backend="gemini")
    gemini_pack.error = {"type": "grounding_missing", "message": "no grounding"}
    gemini_pack.grounded = False

    openai_pack = web_research.EvidencePack(query="q", recency_days=120, backend="openai")
    openai_pack.error = {"type": "grounding_missing", "message": "no sources"}
    openai_pack.grounded = False

    claude_pack = web_research.EvidencePack(query="q", recency_days=120, backend="claude")
    claude_pack.error = {"type": "grounding_missing", "message": "no sources"}
    claude_pack.grounded = False

    fallback_pack = web_research.EvidencePack(query="q", recency_days=120, backend="exa")
    fallback_pack.grounded = True
    fallback_pack.sources = [web_research.EvidenceSource(title="Example", url="https://example.com")]

    monkeypatch.setattr(web_research.gemini_grounding, "fetch_via_gemini", lambda *args, **kwargs: gemini_pack)
    monkeypatch.setattr(web_research.openai_web_search, "fetch_via_openai_web_search", lambda *args, **kwargs: openai_pack)
    monkeypatch.setattr(web_research.claude_web_search, "fetch_via_claude_web_search", lambda *args, **kwargs: claude_pack)
    monkeypatch.setattr(web_research, "_fetch_via_exa", lambda *args, **kwargs: fallback_pack)

    pack = web_research.fetch_evidence_pack("query", purpose="hs", run_id="run-auto", question_id="Q-auto")

    assert pack["backend"] == "exa"
    assert pack["grounded"] is True
    assert pack["sources"][0]["url"] == "https://example.com"
    assert pack["debug"].get("selected_backend") == "exa"
    assert pack["debug"].get("attempted_backends") == ["gemini", "openai", "claude", "exa"]


def test_auto_backend_without_fallback_sets_error(monkeypatch):
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_BACKEND", "auto")
    monkeypatch.delenv("PYTHIA_WEB_RESEARCH_FALLBACK_BACKEND", raising=False)
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_CACHE", "0")

    gemini_pack = web_research.EvidencePack(query="q", recency_days=120, backend="gemini")
    gemini_pack.error = {"type": "grounding_missing", "message": "no grounding"}
    gemini_pack.grounded = False

    openai_pack = web_research.EvidencePack(query="q", recency_days=120, backend="openai")
    openai_pack.error = {"type": "grounding_missing", "message": "no sources"}
    openai_pack.grounded = False

    claude_pack = web_research.EvidencePack(query="q", recency_days=120, backend="claude")
    claude_pack.error = {"type": "grounding_missing", "message": "no sources"}
    claude_pack.grounded = False

    monkeypatch.setattr(web_research.gemini_grounding, "fetch_via_gemini", lambda *args, **kwargs: gemini_pack)
    monkeypatch.setattr(web_research.openai_web_search, "fetch_via_openai_web_search", lambda *args, **kwargs: openai_pack)
    monkeypatch.setattr(web_research.claude_web_search, "fetch_via_claude_web_search", lambda *args, **kwargs: claude_pack)

    pack = web_research.fetch_evidence_pack("query", purpose="hs", run_id="run-auto", question_id="Q-auto")

    assert pack["backend"] == "gemini"
    assert pack["grounded"] is False
    assert pack["error"]["type"] == "no_backend_available"
    assert pack["debug"].get("attempted_backends") == ["gemini", "openai", "claude"]
    assert pack["debug"].get("auto_fallback_backend") is None
    assert pack["debug"].get("selected_backend") == ""
