# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for prediction_markets.query_generator module."""

import pytest

from pythia.prediction_markets.query_generator import (
    _parse_llm_response,
    generate_queries_keyword,
)


def test_keyword_generation_armed_conflict():
    queries = generate_queries_keyword("Iran", "IRN", "ACE", "2026-09-30")
    assert len(queries) >= 2
    assert any("iran" in q.lower() and "conflict" in q.lower() for q in queries)
    assert all("2026" in q for q in queries)


def test_keyword_generation_drought():
    queries = generate_queries_keyword("Ethiopia", "ETH", "DR", "2026-06-30")
    assert any("drought" in q.lower() for q in queries)
    assert any("famine" in q.lower() for q in queries)


def test_keyword_generation_flood():
    queries = generate_queries_keyword("Bangladesh", "BGD", "FL", "2026-08-31")
    assert any("flood" in q.lower() for q in queries)


def test_keyword_generation_no_year():
    queries = generate_queries_keyword("Syria", "SYR", "ACE", "")
    assert len(queries) >= 1
    # No year appended
    assert not any(q.endswith(" ") for q in queries)


def test_parse_llm_response_valid():
    text = '["Iran Israel military 2026", "Iran nuclear deal", "US Iran war"]'
    result = _parse_llm_response(text)
    assert result == ["Iran Israel military 2026", "Iran nuclear deal", "US Iran war"]


def test_parse_llm_response_with_wrapper():
    text = 'Here are the queries:\n["query1", "query2"]\nDone.'
    result = _parse_llm_response(text)
    assert result == ["query1", "query2"]


def test_parse_llm_response_invalid():
    text = "This is not JSON at all"
    result = _parse_llm_response(text)
    assert result is None


def test_parse_llm_response_empty_array():
    text = "[]"
    result = _parse_llm_response(text)
    # Empty list after filtering empty strings
    assert result == [] or result is None


def test_generate_queries_falls_back_to_keyword(monkeypatch):
    """When LLM is unavailable, falls back to keyword generation."""
    import asyncio
    from pythia.prediction_markets import query_generator

    # Make LLM call fail by mocking the import
    async def _failing_llm(*args, **kwargs):
        return None

    monkeypatch.setattr(query_generator, "generate_queries_llm", _failing_llm)

    queries = asyncio.run(query_generator.generate_queries(
        question_text="Armed conflict fatalities in Iran",
        country_name="Iran",
        iso3="IRN",
        hazard_code="ACE",
        hazard_name="armed conflict",
        forecast_start="2026-04-01",
        forecast_end="2026-09-30",
    ))
    assert len(queries) >= 1
    assert any("iran" in q.lower() for q in queries)
