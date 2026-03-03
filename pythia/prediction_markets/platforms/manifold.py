# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Manifold Markets API client for prediction market signal retrieval."""

from __future__ import annotations

import logging
from typing import Any

import requests

from pythia.prediction_markets.config import get_platform_config
from pythia.prediction_markets.types import PredictionMarketQuestion

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://api.manifold.markets/v0"
_REQUEST_TIMEOUT = 12  # seconds per HTTP request


def _get_config() -> dict[str, Any]:
    return get_platform_config("manifold")


def _api_url() -> str:
    return _get_config().get("api_url", _DEFAULT_API_URL)


def _min_unique_bettors() -> int:
    return int(_get_config().get("min_unique_bettors", 10))


def _min_liquidity_mana() -> float:
    return float(_get_config().get("min_liquidity_mana", 500))


def _parse_market(market: dict[str, Any]) -> PredictionMarketQuestion | None:
    """Parse a Manifold market object into a PredictionMarketQuestion."""
    question = (market.get("question") or market.get("title") or "").strip()
    if not question:
        return None

    # Skip resolved markets
    if market.get("isResolved", False):
        return None

    # Quality filters
    bettors = market.get("uniqueBettorCount")
    if isinstance(bettors, (int, float)):
        bettors = int(bettors)
        if bettors < _min_unique_bettors():
            return None
    else:
        bettors = None

    liquidity = market.get("totalLiquidity")
    if isinstance(liquidity, (int, float)):
        liquidity = float(liquidity)
        if liquidity < _min_liquidity_mana():
            return None
    else:
        liquidity = None

    # Extract probability
    prob = market.get("probability")
    if isinstance(prob, (int, float)):
        prob = float(prob)
        if prob > 1.0:
            prob = prob / 100.0
    else:
        prob = None

    # Build URL
    url = market.get("url", "")
    if not url:
        slug = market.get("slug", "")
        creator = market.get("creatorUsername", "")
        if slug and creator:
            url = f"https://manifold.markets/{creator}/{slug}"

    # Volume
    volume = market.get("volume")
    if isinstance(volume, (int, float)):
        volume = float(volume)
    else:
        volume = None

    # Close time (Manifold uses epoch ms)
    close_date = None
    close_time = market.get("closeTime")
    if isinstance(close_time, (int, float)):
        try:
            from datetime import datetime, timezone

            close_date = datetime.fromtimestamp(
                close_time / 1000, tz=timezone.utc
            ).isoformat()
        except (OSError, ValueError):
            pass

    # Question type
    outcome_type = (market.get("outcomeType") or "").upper()
    if outcome_type == "BINARY":
        q_type = "binary"
    elif outcome_type in ("MULTIPLE_CHOICE", "FREE_RESPONSE"):
        q_type = "multiple_choice"
    elif outcome_type in ("NUMERIC", "PSEUDO_NUMERIC"):
        q_type = "numeric"
    else:
        q_type = "binary"

    return PredictionMarketQuestion(
        platform="manifold",
        question_title=question,
        url=url,
        probability=prob,
        num_forecasters=bettors,
        volume_usd=volume,  # Mana, not USD
        close_date=close_date,
        resolve_date=None,
        question_type=q_type,
    )


def search_markets(queries: list[str]) -> list[PredictionMarketQuestion]:
    """Search Manifold Markets for markets matching the given queries.

    Uses the search-markets endpoint — best text search of the three platforms.
    """
    base_url = _api_url()
    results: list[PredictionMarketQuestion] = []
    seen_ids: set[str] = set()

    for query in queries:
        try:
            resp = requests.get(
                f"{base_url}/search-markets",
                params={
                    "term": query,
                    "sort": "liquidity",
                    "filter": "open",
                    "limit": 20,
                },
                timeout=_REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.warning(
                    "Manifold search HTTP %d for query '%s'",
                    resp.status_code,
                    query,
                )
                continue
            markets = resp.json()
        except Exception as exc:
            logger.warning("Manifold search failed for query '%s': %s", query, exc)
            continue

        if not isinstance(markets, list):
            continue

        for market in markets:
            if not isinstance(market, dict):
                continue

            market_id = market.get("id", "")
            if market_id and market_id in seen_ids:
                continue
            if market_id:
                seen_ids.add(market_id)

            parsed = _parse_market(market)
            if parsed is not None:
                results.append(parsed)

    return results
