# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for prediction_markets.retriever module."""

import pytest

from pythia.prediction_markets.retriever import (
    _deduplicate,
    _filter_by_date,
    _norm_title,
    _titles_similar,
)
from pythia.prediction_markets.types import MarketBundle, PredictionMarketQuestion


def _make_q(platform="metaculus", title="Test?", prob=0.5, close_date=None):
    return PredictionMarketQuestion(
        platform=platform,
        question_title=title,
        url=f"https://example.com/{title}",
        probability=prob,
        close_date=close_date,
    )


def test_norm_title():
    assert _norm_title("Will Iran's Nuclear Deal Succeed?") == "will iran s nuclear deal succeed"
    assert _norm_title("") == ""


def test_titles_similar():
    assert _titles_similar(
        "Will Israel strike Iran nuclear facilities?",
        "Will Israel strike Iran's nuclear facilities by 2026?",
    )
    assert not _titles_similar(
        "Will Israel strike Iran?",
        "Will Mars colony be established?",
    )


def test_deduplicate_keeps_higher_priority():
    q1 = _make_q("manifold", "Will Iran conflict escalate?")
    q2 = _make_q("metaculus", "Will Iran conflict escalate?")
    result = _deduplicate([q1, q2])
    assert len(result) == 1
    assert result[0].platform == "metaculus"  # Higher priority


def test_deduplicate_keeps_different_questions():
    q1 = _make_q("metaculus", "Will Iran conflict escalate?")
    q2 = _make_q("polymarket", "Will Mars colony be established?")
    result = _deduplicate([q1, q2])
    assert len(result) == 2


def test_filter_by_date_removes_expired():
    q1 = _make_q(close_date="2026-03-01T00:00:00Z")  # Before forecast
    q2 = _make_q(close_date="2026-07-01T00:00:00Z")  # During forecast
    q3 = _make_q(close_date=None)  # No close date — keep

    result = _filter_by_date([q1, q2, q3], "2026-04-01")
    assert len(result) == 2
    titles = [q.question_title for q in result]
    assert q2.question_title in titles
    assert q3.question_title in titles


def test_filter_by_date_no_forecast_start():
    q1 = _make_q(close_date="2025-01-01T00:00:00Z")
    result = _filter_by_date([q1], "")
    assert len(result) == 1  # No filtering when no forecast_start


@pytest.mark.asyncio
async def test_get_prediction_market_signals_disabled(monkeypatch):
    """Returns empty bundle when disabled."""
    monkeypatch.setenv("PYTHIA_PREDICTION_MARKETS_ENABLED", "0")

    # Need to reload config to pick up env change
    from pythia.prediction_markets.retriever import get_prediction_market_signals

    result = await get_prediction_market_signals(
        question_text="Test",
        country_name="Iran",
        iso3="IRN",
        hazard_code="ACE",
        hazard_name="armed conflict",
        forecast_start="2026-04-01",
        forecast_end="2026-09-30",
    )
    assert isinstance(result, MarketBundle)
    assert result.questions == []


@pytest.mark.asyncio
async def test_get_prediction_market_signals_never_raises(monkeypatch):
    """The retriever should never raise, even if everything fails."""
    monkeypatch.setenv("PYTHIA_PREDICTION_MARKETS_ENABLED", "1")

    # Mock config to return enabled
    import pythia.prediction_markets.config as pm_cfg
    monkeypatch.setattr(pm_cfg, "is_enabled", lambda: True)
    monkeypatch.setattr(pm_cfg, "is_platform_enabled", lambda p: True)

    # Mock query generator to raise
    import pythia.prediction_markets.query_generator as qg

    async def _exploding(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(qg, "generate_queries", _exploding)

    from pythia.prediction_markets.retriever import get_prediction_market_signals

    result = await get_prediction_market_signals(
        question_text="Test",
        country_name="Iran",
        iso3="IRN",
        hazard_code="ACE",
        hazard_name="armed conflict",
        forecast_start="2026-04-01",
        forecast_end="2026-09-30",
        timeout_sec=5.0,
    )
    assert isinstance(result, MarketBundle)
    assert len(result.errors) > 0


def test_config_loading(monkeypatch):
    """Config correctly reads enabled state from env."""
    monkeypatch.setenv("PYTHIA_PREDICTION_MARKETS_ENABLED", "1")
    from pythia.prediction_markets.config import is_enabled

    assert is_enabled() is True

    monkeypatch.setenv("PYTHIA_PREDICTION_MARKETS_ENABLED", "0")
    assert is_enabled() is False


def test_config_timeout(monkeypatch):
    """Config correctly reads timeout from env."""
    monkeypatch.setenv("PYTHIA_PREDICTION_MARKETS_TIMEOUT_SEC", "45")
    from pythia.prediction_markets.config import get_timeout_sec

    assert get_timeout_sec() == 45.0
