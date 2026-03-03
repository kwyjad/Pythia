# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Polymarket Gamma API client for prediction market signal retrieval."""

from __future__ import annotations

import json
import logging
from typing import Any

import requests

from pythia.prediction_markets.config import get_platform_config
from pythia.prediction_markets.types import PredictionMarketQuestion

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://gamma-api.polymarket.com"
_REQUEST_TIMEOUT = 12  # seconds per HTTP request


def _get_config() -> dict[str, Any]:
    return get_platform_config("polymarket")


def _api_url() -> str:
    return _get_config().get("api_url", _DEFAULT_API_URL)


def _min_volume_usd() -> float:
    return float(_get_config().get("min_volume_usd", 10000))


def _min_liquidity_usd() -> float:
    return float(_get_config().get("min_liquidity_usd", 5000))


def _parse_outcome_prices(raw: Any) -> float | None:
    """Parse outcomePrices field to get P(Yes).

    outcomePrices is a JSON string like '["0.35", "0.65"]'.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
    if isinstance(raw, list) and len(raw) >= 1:
        try:
            val = float(raw[0])
            if 0.0 <= val <= 1.0:
                return val
        except (TypeError, ValueError):
            pass
    return None


def _parse_market(market: dict[str, Any]) -> PredictionMarketQuestion | None:
    """Parse a Polymarket market object into a PredictionMarketQuestion."""
    question = (market.get("question") or market.get("title") or "").strip()
    if not question:
        return None

    # Skip closed/inactive markets
    if not market.get("active", True):
        return None
    if market.get("closed", False):
        return None

    prob = _parse_outcome_prices(market.get("outcomePrices"))

    # Parse volume and liquidity
    volume = None
    try:
        volume = float(market.get("volume", 0) or 0)
    except (TypeError, ValueError):
        pass

    liquidity = None
    try:
        liquidity = float(market.get("liquidity", 0) or 0)
    except (TypeError, ValueError):
        pass

    # Quality filters
    if volume is not None and volume < _min_volume_usd():
        return None
    if liquidity is not None and liquidity < _min_liquidity_usd():
        return None

    slug = market.get("slug", "")
    market_id = market.get("id", "")
    if slug:
        url = f"https://polymarket.com/event/{slug}"
    elif market_id:
        url = f"https://polymarket.com/event/{market_id}"
    else:
        url = ""

    # Determine question type from outcomes
    outcomes = market.get("outcomes")
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except (json.JSONDecodeError, TypeError):
            outcomes = None
    q_type = "binary"
    if isinstance(outcomes, list) and len(outcomes) > 2:
        q_type = "multiple_choice"

    return PredictionMarketQuestion(
        platform="polymarket",
        question_title=question,
        url=url,
        probability=prob,
        num_forecasters=None,  # Polymarket doesn't expose trader count per market
        volume_usd=volume,
        close_date=market.get("endDate"),
        resolve_date=None,
        question_type=q_type,
    )


def search_markets(queries: list[str]) -> list[PredictionMarketQuestion]:
    """Search Polymarket for markets matching the given queries.

    Uses the public-search endpoint for text search.
    """
    base_url = _api_url()
    results: list[PredictionMarketQuestion] = []
    seen_titles: set[str] = set()

    for query in queries:
        try:
            resp = requests.get(
                f"{base_url}/public-search",
                params={"query": query},
                timeout=_REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.warning(
                    "Polymarket search HTTP %d for query '%s'",
                    resp.status_code,
                    query,
                )
                continue
            data = resp.json()
        except Exception as exc:
            logger.warning("Polymarket search failed for query '%s': %s", query, exc)
            continue

        # Parse events → nested markets
        events = data.get("events", [])
        if not isinstance(events, list):
            continue

        for event in events:
            if not isinstance(event, dict):
                continue
            markets = event.get("markets", [])
            if not isinstance(markets, list):
                # Event itself might be a market
                markets = [event]
            for market in markets:
                if not isinstance(market, dict):
                    continue
                parsed = _parse_market(market)
                if parsed is None:
                    continue
                # Deduplicate by normalized title
                norm_title = parsed.question_title.lower().strip()
                if norm_title in seen_titles:
                    continue
                seen_titles.add(norm_title)
                results.append(parsed)

    return results
