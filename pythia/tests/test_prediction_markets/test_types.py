# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for prediction_markets.types module."""

from pythia.prediction_markets.types import MarketBundle, PredictionMarketQuestion


def test_prediction_market_question_defaults():
    q = PredictionMarketQuestion(
        platform="metaculus",
        question_title="Test question?",
        url="https://metaculus.com/questions/123/",
    )
    assert q.probability is None
    assert q.relevance_score == 0.0
    assert q.question_type == "binary"


def test_prediction_market_question_to_dict():
    q = PredictionMarketQuestion(
        platform="polymarket",
        question_title="Will X happen?",
        url="https://polymarket.com/event/x",
        probability=0.65,
        volume_usd=50000.0,
        relevance_score=8.0,
        relevance_note="directly related",
    )
    d = q.to_dict()
    assert d["platform"] == "polymarket"
    assert d["probability"] == 0.65
    assert d["volume_usd"] == 50000.0
    assert d["relevance_score"] == 8.0


def test_market_bundle_empty():
    bundle = MarketBundle()
    assert bundle.questions == []
    assert bundle.to_prompt_text() == ""
    d = bundle.to_research_dict()
    assert d["questions"] == []
    assert isinstance(d["retrieval_timestamp"], str)


def test_market_bundle_to_prompt_text():
    q1 = PredictionMarketQuestion(
        platform="metaculus",
        question_title="Will conflict escalate?",
        url="https://metaculus.com/questions/1/",
        probability=0.72,
        num_forecasters=104,
        relevance_score=9.0,
        relevance_note="directly affects conflict forecast",
    )
    q2 = PredictionMarketQuestion(
        platform="manifold",
        question_title="Related question?",
        url="https://manifold.markets/u/q",
        probability=0.45,
        num_forecasters=30,
        volume_usd=5000.0,
        relevance_score=6.0,
        relevance_note="tangentially related",
    )
    bundle = MarketBundle(questions=[q2, q1])  # intentionally out of order
    text = bundle.to_prompt_text()

    # Should be sorted by relevance_score desc
    lines = text.split("\n")
    assert "[metaculus]" in lines[0]  # q1 has higher score
    assert "72%" in text
    assert "104 forecasters" in text
    assert "M$5,000 volume" in text


def test_market_bundle_to_research_dict():
    q = PredictionMarketQuestion(
        platform="polymarket",
        question_title="Test?",
        url="https://polymarket.com/event/test",
        probability=0.50,
        relevance_score=7.0,
    )
    bundle = MarketBundle(
        questions=[q],
        query_terms_used=["test query"],
        errors=["manifold: timeout"],
    )
    d = bundle.to_research_dict()
    assert len(d["questions"]) == 1
    assert d["questions"][0]["platform"] == "polymarket"
    assert d["query_terms_used"] == ["test query"]
    assert "manifold: timeout" in d["errors"]
