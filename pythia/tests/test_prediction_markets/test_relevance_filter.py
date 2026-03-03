# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for prediction_markets.relevance_filter module."""

import pytest

from pythia.prediction_markets.relevance_filter import (
    _parse_filter_response,
    filter_by_relevance_passthrough,
)
from pythia.prediction_markets.types import PredictionMarketQuestion


def _make_question(platform="metaculus", title="Test?", prob=0.5, volume=None, forecasters=None):
    return PredictionMarketQuestion(
        platform=platform,
        question_title=title,
        url=f"https://example.com/{title}",
        probability=prob,
        volume_usd=volume,
        num_forecasters=forecasters,
    )


def test_parse_filter_response_valid():
    candidates = [
        _make_question(title="Q1"),
        _make_question(title="Q2"),
        _make_question(title="Q3"),
    ]
    text = '[{"index": 1, "score": 9, "relevance_note": "very relevant"}, {"index": 3, "score": 7, "relevance_note": "somewhat relevant"}]'
    result = _parse_filter_response(text, candidates)
    assert result is not None
    assert len(result) == 2
    assert result[0].relevance_score == 9.0
    assert result[0].question_title == "Q1"
    assert result[1].relevance_score == 7.0


def test_parse_filter_response_filters_low_scores():
    candidates = [_make_question(title="Q1")]
    text = '[{"index": 1, "score": 3, "relevance_note": "not relevant"}]'
    result = _parse_filter_response(text, candidates)
    # Score 3 is below min_relevance_score of 6
    assert result is None or len(result) == 0


def test_parse_filter_response_invalid_json():
    candidates = [_make_question()]
    result = _parse_filter_response("not json", candidates)
    assert result is None


def test_passthrough_fallback():
    candidates = [
        _make_question(title="Low vol", volume=100, forecasters=5),
        _make_question(title="High vol", volume=100000, forecasters=200),
        _make_question(title="Mid vol", volume=50000, forecasters=50),
    ]
    result = filter_by_relevance_passthrough(candidates)
    assert len(result) <= 5
    # All should have default score
    for q in result:
        assert q.relevance_score == 5.0
        assert "LLM unavailable" in q.relevance_note


@pytest.mark.asyncio
async def test_filter_candidates_falls_back(monkeypatch):
    """When LLM is unavailable, falls back to passthrough."""
    from pythia.prediction_markets import relevance_filter

    async def _failing_llm(*args, **kwargs):
        return None

    monkeypatch.setattr(relevance_filter, "filter_by_relevance_llm", _failing_llm)

    candidates = [
        _make_question(title="Q1", volume=50000, forecasters=100),
        _make_question(title="Q2", volume=10000, forecasters=20),
    ]
    result = await relevance_filter.filter_candidates(
        question_text="Test question",
        country_name="Iran",
        hazard_name="armed conflict",
        forecast_start="2026-04-01",
        forecast_end="2026-09-30",
        candidates=candidates,
    )
    assert len(result) == 2
    assert all(q.relevance_score == 5.0 for q in result)
