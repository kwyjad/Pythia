# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Main prediction market signal retriever.

Orchestrates query generation, parallel platform searches, deduplication,
relevance filtering, and bundle assembly. This is the primary entry point
for the prediction market integration.
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
from datetime import datetime, timezone
from typing import Any

from pythia.prediction_markets import config as pm_config
from pythia.prediction_markets.types import MarketBundle, PredictionMarketQuestion

logger = logging.getLogger(__name__)


def _norm_title(s: str) -> str:
    """Normalize title for deduplication comparisons."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _titles_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """Check if two question titles are similar enough to be duplicates."""
    na, nb = _norm_title(a), _norm_title(b)
    if not na or not nb:
        return False
    return difflib.SequenceMatcher(None, na, nb).ratio() >= threshold


def _deduplicate(
    questions: list[PredictionMarketQuestion],
) -> list[PredictionMarketQuestion]:
    """Remove near-duplicate questions across platforms.

    Keeps the version from the higher-priority platform:
    metaculus > polymarket > manifold.
    """
    platform_priority = {"metaculus": 0, "polymarket": 1, "manifold": 2}
    # Sort by platform priority so we keep higher-priority versions
    questions.sort(key=lambda q: platform_priority.get(q.platform, 99))

    kept: list[PredictionMarketQuestion] = []
    for q in questions:
        is_dup = False
        for existing in kept:
            if _titles_similar(q.question_title, existing.question_title):
                is_dup = True
                break
        if not is_dup:
            kept.append(q)
    return kept


def _filter_by_date(
    questions: list[PredictionMarketQuestion],
    forecast_start: str,
) -> list[PredictionMarketQuestion]:
    """Remove questions that close before the forecast period starts."""
    if not forecast_start:
        return questions

    try:
        raw = forecast_start.replace("Z", "+00:00")
        start_dt = datetime.fromisoformat(raw)
        # Ensure timezone-aware for comparison
        if start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return questions

    kept: list[PredictionMarketQuestion] = []
    for q in questions:
        if not q.close_date:
            kept.append(q)
            continue
        try:
            close_raw = q.close_date.replace("Z", "+00:00")
            close_dt = datetime.fromisoformat(close_raw)
            if close_dt.tzinfo is None:
                close_dt = close_dt.replace(tzinfo=timezone.utc)
            if close_dt >= start_dt:
                kept.append(q)
        except (ValueError, TypeError):
            kept.append(q)
    return kept


def _search_metaculus(queries: list[str]) -> list[PredictionMarketQuestion]:
    """Search Metaculus using cached index."""
    from pythia.prediction_markets.cache import get_or_refresh_cache
    from pythia.prediction_markets.platforms.metaculus import search_questions

    cached = get_or_refresh_cache()
    return search_questions(queries, cached_questions=cached)


def _search_polymarket(queries: list[str]) -> list[PredictionMarketQuestion]:
    """Search Polymarket."""
    from pythia.prediction_markets.platforms.polymarket import search_markets

    return search_markets(queries)


def _search_manifold(queries: list[str]) -> list[PredictionMarketQuestion]:
    """Search Manifold Markets."""
    from pythia.prediction_markets.platforms.manifold import search_markets

    return search_markets(queries)


async def _search_all_platforms(
    queries: list[str],
) -> tuple[list[PredictionMarketQuestion], list[str]]:
    """Search all enabled platforms in parallel. Returns (results, errors)."""
    all_results: list[PredictionMarketQuestion] = []
    errors: list[str] = []

    tasks: list[tuple[str, Any]] = []
    if pm_config.is_platform_enabled("metaculus"):
        tasks.append(("metaculus", asyncio.to_thread(_search_metaculus, queries)))
    if pm_config.is_platform_enabled("polymarket"):
        tasks.append(("polymarket", asyncio.to_thread(_search_polymarket, queries)))
    if pm_config.is_platform_enabled("manifold"):
        tasks.append(("manifold", asyncio.to_thread(_search_manifold, queries)))

    if not tasks:
        return [], ["all platforms disabled"]

    # Run all platform searches in parallel
    coros = [t[1] for t in tasks]
    results = await asyncio.gather(*coros, return_exceptions=True)

    for (platform, _), result in zip(tasks, results):
        if isinstance(result, Exception):
            msg = f"{platform}: {result!r}"
            logger.warning("Platform search failed: %s", msg)
            errors.append(msg)
        elif isinstance(result, list):
            logger.info("Platform %s returned %d results", platform, len(result))
            all_results.extend(result)
        else:
            errors.append(f"{platform}: unexpected result type {type(result)}")

    return all_results, errors


async def get_prediction_market_signals(
    question_text: str,
    country_name: str,
    iso3: str,
    hazard_code: str,
    hazard_name: str,
    forecast_start: str,
    forecast_end: str,
    max_results: int = 5,
    timeout_sec: float | None = None,
    run_id: str | None = None,
) -> MarketBundle:
    """Query Metaculus, Polymarket, and Manifold for prediction market questions
    related to the given Fred question.

    Returns a formatted evidence bundle. Returns empty bundle if no relevant
    questions found or all APIs fail. Never raises.

    Args:
        question_text: Natural language Fred question.
        country_name: Full country name (e.g., "Iran").
        iso3: ISO 3166-1 alpha-3 code (e.g., "IRN").
        hazard_code: Hazard code (e.g., "ACE", "DR", "FL").
        hazard_name: Human-readable hazard name (e.g., "armed conflict").
        forecast_start: ISO date string for forecast window start.
        forecast_end: ISO date string for forecast window end.
        max_results: Maximum number of results to return.
        timeout_sec: Overall timeout. If None, uses config value.
        run_id: Forecaster run ID for LLM call logging.
    """
    if timeout_sec is None:
        timeout_sec = pm_config.get_timeout_sec()

    # Check master switch
    if not pm_config.is_enabled():
        return MarketBundle()

    async def _inner() -> MarketBundle:
        errors: list[str] = []

        # 1. Generate queries
        from pythia.prediction_markets.query_generator import generate_queries

        queries = await generate_queries(
            question_text, country_name, iso3, hazard_code, hazard_name,
            forecast_start, forecast_end, run_id=run_id,
        )
        logger.info(
            "Generated %d queries for %s/%s: %s",
            len(queries), iso3, hazard_code, queries,
        )

        # 2. Search all platforms in parallel
        raw_results, platform_errors = await _search_all_platforms(queries)
        errors.extend(platform_errors)

        if not raw_results:
            return MarketBundle(
                query_terms_used=queries,
                errors=errors,
            )

        # 3. Deduplicate
        deduped = _deduplicate(raw_results)

        # 4. Filter by date
        date_filtered = _filter_by_date(deduped, forecast_start)

        if not date_filtered:
            return MarketBundle(
                query_terms_used=queries,
                errors=errors,
            )

        # 5. Relevance filtering
        from pythia.prediction_markets.relevance_filter import filter_candidates

        filtered = await filter_candidates(
            question_text, country_name, hazard_name,
            forecast_start, forecast_end, date_filtered,
            run_id=run_id,
        )

        return MarketBundle(
            questions=filtered[:max_results],
            query_terms_used=queries,
            errors=errors,
        )

    try:
        return await asyncio.wait_for(_inner(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        logger.warning(
            "Prediction market retrieval timed out after %.1fs for %s/%s",
            timeout_sec, iso3, hazard_code,
        )
        return MarketBundle(errors=[f"timeout after {timeout_sec}s"])
    except Exception as exc:
        logger.error(
            "Prediction market retrieval failed for %s/%s: %s",
            iso3, hazard_code, exc,
        )
        return MarketBundle(errors=[f"retrieval error: {exc!r}"])
