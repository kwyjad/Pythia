# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
import pytest

from pythia.adversarial_check import (
    _build_adversarial_queries,
    format_adversarial_check_for_spd,
    run_adversarial_check,
    _aggregate_evidence_text,
    _aggregate_sources,
)


# ---------------------------------------------------------------------------
# Sample RC results
# ---------------------------------------------------------------------------

_RC_ACE_UP = {
    "likelihood": 0.75,
    "magnitude": 0.65,
    "direction": "up",
    "window": "month_1-2",
    "rationale_bullets": [
        "Troop buildup along northern border",
        "Ceasefire violations reported weekly",
    ],
    "trigger_signals": [
        {"signal": "troop movements near border", "timeframe_months": 2, "evidence_refs": []},
        {"signal": "arms shipments intercepted", "timeframe_months": 1, "evidence_refs": []},
    ],
    "valid": True,
    "status": "ok",
}

_RC_FL_UP = {
    "likelihood": 0.65,
    "magnitude": 0.55,
    "direction": "up",
    "window": "month_2",
    "rationale_bullets": ["Above-normal rainfall forecast"],
    "trigger_signals": [
        {"signal": "river gauge levels rising", "timeframe_months": 1, "evidence_refs": []},
    ],
    "valid": True,
    "status": "ok",
}

_RC_ACE_DOWN = {
    "likelihood": 0.60,
    "magnitude": 0.50,
    "direction": "down",
    "window": "month_1",
    "rationale_bullets": ["Peace talks progressing"],
    "trigger_signals": [],
    "valid": True,
    "status": "ok",
}

_RC_LOW = {
    "likelihood": 0.30,
    "magnitude": 0.20,
    "direction": "unclear",
    "window": "",
    "rationale_bullets": [],
    "trigger_signals": [],
    "valid": True,
    "status": "ok",
}

_SAMPLE_CHECK_RESULT = {
    "counter_evidence": [
        {
            "claim": "Peace talks between X and Y resumed on Feb 28",
            "source": "Reuters",
            "relevance": "Directly contradicts RC trigger of ceasefire collapse",
            "strength": "moderate",
        },
    ],
    "historical_analogs": [
        {
            "analog": "Similar troop buildup in 2019 did not lead to escalation",
            "outcome": "Tensions de-escalated after diplomatic intervention",
            "relevance": "Suggests current signals may not lead to predicted outcome",
        },
    ],
    "stabilizing_factors": [
        "International mediator presence (UN envoy active since Jan)",
        "Rainy season limits military mobility through April",
    ],
    "net_assessment": "moderate",
    "summary": "Moderate counter-evidence suggests diplomatic channels may prevent escalation",
    "sources": [{"title": "Reuters report", "url": "https://reuters.com/example", "date": "2025-02-28"}],
    "grounded": True,
}


# ---------------------------------------------------------------------------
# Tests for _build_adversarial_queries
# ---------------------------------------------------------------------------


class TestBuildAdversarialQueries:
    def test_ace_up_queries(self):
        queries = _build_adversarial_queries("Somalia", "SOM", "ACE", _RC_ACE_UP)
        assert len(queries) >= 2
        assert len(queries) <= 3
        # Should contain peace/de-escalation terms
        combined = " ".join(queries).lower()
        assert "somalia" in combined
        assert any(term in combined for term in ["peace", "ceasefire", "de-escalation"])

    def test_fl_up_queries(self):
        queries = _build_adversarial_queries("Bangladesh", "BGD", "FL", _RC_FL_UP)
        assert len(queries) >= 2
        combined = " ".join(queries).lower()
        assert "bangladesh" in combined
        assert any(term in combined for term in ["forecast", "revised", "improved", "preparedness"])

    def test_ace_down_queries(self):
        queries = _build_adversarial_queries("Ethiopia", "ETH", "ACE", _RC_ACE_DOWN)
        assert len(queries) >= 1
        combined = " ".join(queries).lower()
        assert "ethiopia" in combined
        assert any(term in combined for term in ["conflict", "resumption", "violence", "collapse"])

    def test_uses_trigger_signals(self):
        queries = _build_adversarial_queries("Somalia", "SOM", "ACE", _RC_ACE_UP)
        # Should have a trigger-specific query (3rd query)
        assert len(queries) == 3
        assert "troop movements near border" in queries[2].lower()

    def test_no_trigger_signals(self):
        queries = _build_adversarial_queries("Ethiopia", "ETH", "ACE", _RC_ACE_DOWN)
        # No trigger signals, should still have base queries
        assert len(queries) >= 1
        assert len(queries) <= 3

    def test_max_three_queries(self):
        rc = {**_RC_ACE_UP, "trigger_signals": [
            {"signal": f"signal {i}", "timeframe_months": 1, "evidence_refs": []}
            for i in range(5)
        ]}
        queries = _build_adversarial_queries("Somalia", "SOM", "ACE", rc)
        assert len(queries) <= 3


# ---------------------------------------------------------------------------
# Tests for run_adversarial_check
# ---------------------------------------------------------------------------


class TestRunAdversarialCheck:
    def test_returns_none_for_low_rc(self):
        result = run_adversarial_check(
            iso3="SOM",
            country_name="Somalia",
            hazard_code="ACE",
            rc_result=_RC_LOW,
            run_id="test-run-001",
        )
        assert result is None

    def test_returns_none_for_l1_rc(self):
        rc_l1 = {
            "likelihood": 0.45,
            "magnitude": 0.30,
            "direction": "up",
            "window": "month_1",
            "rationale_bullets": [],
            "trigger_signals": [],
            "valid": True,
            "status": "ok",
        }
        result = run_adversarial_check(
            iso3="SOM",
            country_name="Somalia",
            hazard_code="ACE",
            rc_result=rc_l1,
            run_id="test-run-001",
        )
        assert result is None

    def test_returns_inconclusive_when_no_evidence(self, monkeypatch):
        """When fetch_evidence_pack returns empty packs, should return inconclusive."""
        monkeypatch.setenv("PYTHIA_RETRIEVER_ENABLED", "1")
        monkeypatch.setenv("PYTHIA_RETRIEVER_MODEL_ID", "test-model")

        def fake_fetch(*args, **kwargs):
            return {"sources": [], "recent_signals": [], "structural_context": ""}

        monkeypatch.setattr(
            "pythia.adversarial_check.fetch_evidence_pack", fake_fetch
        )

        result = run_adversarial_check(
            iso3="SOM",
            country_name="Somalia",
            hazard_code="ACE",
            rc_result=_RC_ACE_UP,
            run_id="test-run-001",
        )
        assert result is not None
        assert result["net_assessment"] == "inconclusive"
        assert result["summary"] == "No adversarial evidence found in search results"
        assert result["counter_evidence"] == []

    def test_synthesis_with_mocked_evidence(self, monkeypatch):
        """Full flow with mocked fetch_evidence_pack and call_chat_ms."""
        monkeypatch.setenv("PYTHIA_RETRIEVER_ENABLED", "1")
        monkeypatch.setenv("PYTHIA_RETRIEVER_MODEL_ID", "test-model")

        def fake_fetch(*args, **kwargs):
            return {
                "sources": [{"title": "Reuters", "url": "https://reuters.com/test", "date": "2025-03-01"}],
                "recent_signals": ["Peace talks resumed between parties"],
                "structural_context": "Diplomatic efforts ongoing",
                "grounded": True,
            }

        monkeypatch.setattr(
            "pythia.adversarial_check.fetch_evidence_pack", fake_fetch
        )

        synthesis_result = {
            "counter_evidence": [
                {"claim": "Peace talks resumed", "source": "Reuters", "relevance": "Contradicts escalation", "strength": "moderate"}
            ],
            "historical_analogs": [],
            "stabilizing_factors": ["Active diplomacy"],
            "net_assessment": "moderate",
            "summary": "Moderate counter-evidence from diplomatic channels",
        }

        async def fake_call_chat_ms(spec, prompt, **kwargs):
            return json.dumps(synthesis_result), {"prompt_tokens": 100, "completion_tokens": 50}, ""

        monkeypatch.setattr(
            "pythia.adversarial_check.call_chat_ms", fake_call_chat_ms
        )

        result = run_adversarial_check(
            iso3="SOM",
            country_name="Somalia",
            hazard_code="ACE",
            rc_result=_RC_ACE_UP,
            run_id="test-run-001",
        )
        assert result is not None
        assert result["net_assessment"] == "moderate"
        assert len(result["counter_evidence"]) == 1
        assert result["counter_evidence"][0]["claim"] == "Peace talks resumed"
        assert result["grounded"] is True
        assert len(result["sources"]) == 1

    def test_returns_none_on_synthesis_failure(self, monkeypatch):
        """If LLM synthesis fails, should return None."""
        monkeypatch.setenv("PYTHIA_RETRIEVER_ENABLED", "1")
        monkeypatch.setenv("PYTHIA_RETRIEVER_MODEL_ID", "test-model")

        def fake_fetch(*args, **kwargs):
            return {
                "sources": [{"title": "Test", "url": "https://example.com", "date": "2025-01-01"}],
                "recent_signals": ["Some signal"],
                "structural_context": "Some context",
                "grounded": True,
            }

        monkeypatch.setattr(
            "pythia.adversarial_check.fetch_evidence_pack", fake_fetch
        )

        async def fake_call_chat_ms(spec, prompt, **kwargs):
            return "", {}, "API error"

        monkeypatch.setattr(
            "pythia.adversarial_check.call_chat_ms", fake_call_chat_ms
        )

        result = run_adversarial_check(
            iso3="SOM",
            country_name="Somalia",
            hazard_code="ACE",
            rc_result=_RC_ACE_UP,
            run_id="test-run-001",
        )
        assert result is None


# ---------------------------------------------------------------------------
# Tests for format_adversarial_check_for_spd
# ---------------------------------------------------------------------------


class TestFormatAdversarialCheckForSpd:
    def test_none_returns_empty(self):
        assert format_adversarial_check_for_spd(None) == ""

    def test_empty_dict_returns_empty(self):
        # Empty dict is falsy, treated same as None
        assert format_adversarial_check_for_spd({}) == ""

    def test_full_result_formatting(self):
        result = format_adversarial_check_for_spd(_SAMPLE_CHECK_RESULT, rc_level=3)
        assert "ADVERSARIAL EVIDENCE CHECK (RC Level 3" in result
        assert "Net assessment: moderate" in result
        assert "Counter-evidence:" in result
        assert "[moderate]" in result
        assert "Peace talks between X and Y resumed on Feb 28" in result
        assert "Source: Reuters" in result
        assert "Historical analogs:" in result
        assert "Similar troop buildup in 2019" in result
        assert "Stabilizing factors:" in result
        assert "International mediator presence" in result
        assert "INSTRUCTION:" in result

    def test_inconclusive_result(self):
        check = {
            "counter_evidence": [],
            "historical_analogs": [],
            "stabilizing_factors": [],
            "net_assessment": "inconclusive",
            "summary": "No adversarial evidence found",
        }
        result = format_adversarial_check_for_spd(check, rc_level=2)
        assert "inconclusive" in result
        assert "Counter-evidence:" not in result  # no counter-evidence section
        assert "INSTRUCTION:" in result


# ---------------------------------------------------------------------------
# Tests for evidence aggregation helpers
# ---------------------------------------------------------------------------


class TestAggregateHelpers:
    def test_aggregate_evidence_text_empty(self):
        result = _aggregate_evidence_text([])
        assert "no search evidence" in result

    def test_aggregate_evidence_text_with_data(self):
        packs = [
            {
                "structural_context": "Context one",
                "recent_signals": ["Signal A", "Signal B"],
                "sources": [{"title": "Src", "url": "https://example.com"}],
            },
        ]
        result = _aggregate_evidence_text(packs)
        assert "Search 1" in result
        assert "Context one" in result
        assert "Signal A" in result
        assert "Src" in result

    def test_aggregate_sources_deduplicates(self):
        packs = [
            {"sources": [{"title": "A", "url": "https://a.com", "date": "2025-01-01"}]},
            {"sources": [
                {"title": "A", "url": "https://a.com", "date": "2025-01-01"},
                {"title": "B", "url": "https://b.com", "date": "2025-01-02"},
            ]},
        ]
        result = _aggregate_sources(packs)
        assert len(result) == 2
        urls = {s["url"] for s in result}
        assert "https://a.com" in urls
        assert "https://b.com" in urls
