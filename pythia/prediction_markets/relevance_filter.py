# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Relevance filtering for prediction market candidates.

Filters and scores candidate prediction market questions for relevance to
a given Fred question using either an LLM (primary) or a passthrough
fallback (sorts by volume/liquidity).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pythia.prediction_markets.config import get_relevance_filter_config
from pythia.prediction_markets.types import PredictionMarketQuestion

logger = logging.getLogger(__name__)


def _build_filter_prompt(
    question_text: str,
    country_name: str,
    hazard_name: str,
    forecast_start: str,
    forecast_end: str,
    candidates: list[PredictionMarketQuestion],
) -> str:
    numbered_list = []
    for i, c in enumerate(candidates, 1):
        prob_str = f"{c.probability:.0%}" if c.probability is not None else "N/A"
        numbered_list.append(
            f"{i}. [{c.platform}] \"{c.question_title}\" — {prob_str}"
        )

    return (
        "You are filtering prediction market questions for relevance to a humanitarian forecast.\n\n"
        f"Fred question: \"{question_text}\"\n"
        f"Country: {country_name}\n"
        f"Hazard: {hazard_name}\n"
        f"Forecast period: {forecast_start} to {forecast_end}\n\n"
        "Below are candidate prediction market questions. Rate each 0-10 for relevance "
        "to the Fred question (10 = directly affects the forecast outcome, 0 = completely "
        "unrelated). A question is relevant if its outcome would meaningfully shift the "
        "probability distribution of the Fred question.\n\n"
        "Candidates:\n"
        + "\n".join(numbered_list)
        + "\n\n"
        "Return ONLY a JSON array of objects: "
        '[{"index": 1, "score": 8, "relevance_note": "brief reason"}, ...]\n'
        "Include only candidates with score >= 6."
    )


def _parse_filter_response(
    text: str, candidates: list[PredictionMarketQuestion]
) -> list[PredictionMarketQuestion] | None:
    """Parse LLM response and apply scores to candidates."""
    text = text.strip()
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(parsed, list):
        return None

    cfg = get_relevance_filter_config()
    min_score = float(cfg.get("min_relevance_score", 6))
    max_results = int(cfg.get("max_results", 5))

    scored: list[PredictionMarketQuestion] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        score = item.get("score")
        note = item.get("relevance_note", "")

        if not isinstance(idx, (int, float)):
            continue
        if not isinstance(score, (int, float)):
            continue

        idx = int(idx) - 1  # Convert to 0-indexed
        if idx < 0 or idx >= len(candidates):
            continue
        if float(score) < min_score:
            continue

        q = candidates[idx]
        q.relevance_score = float(score)
        q.relevance_note = str(note)[:200]
        scored.append(q)

    scored.sort(key=lambda x: -x.relevance_score)
    return scored[:max_results] if scored else None


async def filter_by_relevance_llm(
    question_text: str,
    country_name: str,
    hazard_name: str,
    forecast_start: str,
    forecast_end: str,
    candidates: list[PredictionMarketQuestion],
    run_id: str | None = None,
) -> list[PredictionMarketQuestion] | None:
    """Filter candidates using LLM-based relevance scoring.

    Returns None if the LLM call fails (caller should use passthrough fallback).
    """
    if not candidates:
        return []

    cfg = get_relevance_filter_config()
    model_id = cfg.get("model", "gemini-3-flash-preview")

    prompt = _build_filter_prompt(
        question_text, country_name, hazard_name,
        forecast_start, forecast_end, candidates,
    )

    try:
        from forecaster.providers import ModelSpec, call_chat_ms

        spec = ModelSpec(
            name="PM-RelevanceFilter",
            provider="google",
            model_id=model_id,
            active=True,
        )
        text, usage, error = await call_chat_ms(
            spec,
            prompt,
            temperature=0.1,
            prompt_key="prediction_markets.relevance_filter",
            prompt_version="1.0.0",
            component="prediction_markets",
            run_id=run_id,
        )
        if error:
            logger.warning("Relevance filter LLM error: %s", error)
            return None
        result = _parse_filter_response(text, candidates)
        if result is not None:
            return result
        logger.warning("Relevance filter LLM returned unparseable response")
        return None
    except Exception as exc:
        logger.warning("Relevance filter LLM call failed: %s", exc)
        return None


def filter_by_relevance_passthrough(
    candidates: list[PredictionMarketQuestion],
) -> list[PredictionMarketQuestion]:
    """Fallback: return candidates sorted by volume/liquidity without LLM scoring.

    Applies a default relevance_score of 5.0 and caps at max_results.
    """
    cfg = get_relevance_filter_config()
    max_results = int(cfg.get("max_results", 5))

    def _sort_key(q: PredictionMarketQuestion) -> float:
        vol = q.volume_usd if q.volume_usd is not None else 0
        fc = q.num_forecasters if q.num_forecasters is not None else 0
        return -(vol + fc * 100)  # Weight forecasters more

    sorted_candidates = sorted(candidates, key=_sort_key)
    result = sorted_candidates[:max_results]
    for q in result:
        q.relevance_score = 5.0
        q.relevance_note = "relevance not scored (LLM unavailable)"
    return result


async def filter_candidates(
    question_text: str,
    country_name: str,
    hazard_name: str,
    forecast_start: str,
    forecast_end: str,
    candidates: list[PredictionMarketQuestion],
    run_id: str | None = None,
) -> list[PredictionMarketQuestion]:
    """Filter candidates, trying LLM first with passthrough fallback.

    Always returns a list (possibly empty).
    """
    if not candidates:
        return []

    # Try LLM-based filtering
    result = await filter_by_relevance_llm(
        question_text, country_name, hazard_name,
        forecast_start, forecast_end, candidates,
        run_id=run_id,
    )
    if result is not None:
        return result

    # Fallback to passthrough
    logger.info("Using passthrough fallback for relevance filtering")
    return filter_by_relevance_passthrough(candidates)
