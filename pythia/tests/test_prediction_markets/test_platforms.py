# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for prediction_markets.platforms module (mocked HTTP)."""

import json
from unittest.mock import MagicMock, patch


# --- Metaculus tests ---


def _make_metaculus_question(
    qid=1, title="Test question?", prob=0.72, nr_forecasters=50, status="open"
):
    return {
        "id": qid,
        "title": title,
        "slug": "test-question",
        "status": status,
        "type": "forecast",
        "nr_forecasters": nr_forecasters,
        "scheduled_close_time": "2026-06-01T00:00:00Z",
        "scheduled_resolve_time": "2026-07-01T00:00:00Z",
        "question": {
            "type": "binary",
            "aggregations": {
                "recency_weighted": {
                    "latest": {"centers": [prob], "means": [prob]}
                }
            },
        },
    }


def test_metaculus_search_questions():
    from pythia.prediction_markets.platforms.metaculus import search_questions

    cached = [
        _make_metaculus_question(1, "Will Iran conflict escalate in 2026?", 0.35, 120),
        _make_metaculus_question(2, "Will Mars colony be established?", 0.05, 200),
        _make_metaculus_question(3, "Iran nuclear deal 2026", 0.60, 50),
    ]
    results = search_questions(["iran conflict"], cached_questions=cached)
    titles = [r.question_title for r in results]
    assert "Will Iran conflict escalate in 2026?" in titles
    assert "Will Mars colony be established?" not in titles


def test_metaculus_quality_filter(monkeypatch):
    monkeypatch.setenv("PYTHIA_PREDICTION_MARKETS_ENABLED", "1")
    from pythia.prediction_markets.platforms.metaculus import search_questions

    cached = [
        _make_metaculus_question(1, "Iran war?", 0.50, 5),  # below min_forecasters=10
        _make_metaculus_question(2, "Iran conflict?", 0.60, 50),
    ]
    results = search_questions(["iran"], cached_questions=cached)
    assert len(results) == 1
    assert results[0].question_title == "Iran conflict?"


def test_metaculus_parse_probability():
    from pythia.prediction_markets.platforms.metaculus import _extract_community_prediction

    q = _make_metaculus_question(prob=0.72)
    assert _extract_community_prediction(q) == 0.72


# --- Polymarket tests ---


def _mock_polymarket_response(markets):
    return {"events": [{"markets": markets}]}


def test_polymarket_search_markets():
    from pythia.prediction_markets.platforms.polymarket import search_markets

    market = {
        "question": "Will Israel strike Iran?",
        "slug": "will-israel-strike-iran",
        "outcomePrices": '["0.35", "0.65"]',
        "volume": "150000.0",
        "liquidity": "25000.0",
        "active": True,
        "closed": False,
        "endDate": "2026-06-30T00:00:00Z",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _mock_polymarket_response([market])

    with patch("pythia.prediction_markets.platforms.polymarket.requests.get", return_value=mock_resp):
        results = search_markets(["iran strike"])

    assert len(results) == 1
    assert results[0].platform == "polymarket"
    assert results[0].probability == 0.35
    assert results[0].volume_usd == 150000.0


def test_polymarket_quality_filter():
    from pythia.prediction_markets.platforms.polymarket import search_markets

    low_vol_market = {
        "question": "Low volume market?",
        "slug": "low-vol",
        "outcomePrices": '["0.50", "0.50"]',
        "volume": "500.0",  # Below $10k min
        "liquidity": "100.0",
        "active": True,
        "closed": False,
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = _mock_polymarket_response([low_vol_market])

    with patch("pythia.prediction_markets.platforms.polymarket.requests.get", return_value=mock_resp):
        results = search_markets(["test"])

    assert len(results) == 0


# --- Manifold tests ---


def test_manifold_search_markets():
    from pythia.prediction_markets.platforms.manifold import search_markets

    market = {
        "id": "abc123",
        "question": "Will Khamenei be Supreme Leader in 2027?",
        "slug": "will-khamenei-be-supreme-leader",
        "url": "https://manifold.markets/user/will-khamenei",
        "probability": 0.72,
        "uniqueBettorCount": 45,
        "totalLiquidity": 2100,
        "volume": 12500,
        "closeTime": 1735689600000,
        "isResolved": False,
        "outcomeType": "BINARY",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [market]

    with patch("pythia.prediction_markets.platforms.manifold.requests.get", return_value=mock_resp):
        results = search_markets(["Khamenei supreme leader"])

    assert len(results) == 1
    assert results[0].platform == "manifold"
    assert results[0].probability == 0.72
    assert results[0].num_forecasters == 45


def test_manifold_quality_filter():
    from pythia.prediction_markets.platforms.manifold import search_markets

    market = {
        "id": "xyz",
        "question": "Low quality market?",
        "probability": 0.50,
        "uniqueBettorCount": 3,  # Below min 10
        "totalLiquidity": 100,  # Below min 500
        "isResolved": False,
        "outcomeType": "BINARY",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [market]

    with patch("pythia.prediction_markets.platforms.manifold.requests.get", return_value=mock_resp):
        results = search_markets(["test"])

    assert len(results) == 0


def test_manifold_skip_resolved():
    from pythia.prediction_markets.platforms.manifold import search_markets

    market = {
        "id": "resolved1",
        "question": "Already resolved?",
        "probability": 1.0,
        "uniqueBettorCount": 100,
        "totalLiquidity": 5000,
        "isResolved": True,
        "outcomeType": "BINARY",
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [market]

    with patch("pythia.prediction_markets.platforms.manifold.requests.get", return_value=mock_resp):
        results = search_markets(["test"])

    assert len(results) == 0
