# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Metaculus API client for prediction market signal retrieval."""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from pythia.prediction_markets.config import get_platform_config
from pythia.prediction_markets.types import PredictionMarketQuestion

logger = logging.getLogger(__name__)

_DEFAULT_API_URL = "https://www.metaculus.com/api2/questions/"
_REQUEST_TIMEOUT = 15  # seconds per HTTP request
_PAGE_SIZE = 200


def _get_config() -> dict[str, Any]:
    return get_platform_config("metaculus")


def _api_url() -> str:
    return _get_config().get("api_url", _DEFAULT_API_URL)


def _min_forecasters() -> int:
    return int(_get_config().get("min_forecasters", 10))


def _auth_headers() -> dict[str, str]:
    token = os.getenv("METACULUS_API_TOKEN", "").strip()
    if token:
        return {"Authorization": f"Token {token}"}
    return {}


def _extract_community_prediction(question_obj: dict[str, Any]) -> float | None:
    """Extract community prediction probability from Metaculus question object."""
    q = question_obj.get("question")
    if not isinstance(q, dict):
        return None
    aggs = q.get("aggregations")
    if not isinstance(aggs, dict):
        return None
    # Prefer recency_weighted, fall back to metaculus_prediction
    for agg_key in ("recency_weighted", "metaculus_prediction"):
        agg = aggs.get(agg_key)
        if not isinstance(agg, dict):
            continue
        latest = agg.get("latest")
        if not isinstance(latest, dict):
            continue
        centers = latest.get("centers")
        if isinstance(centers, list) and centers:
            try:
                val = float(centers[0])
                if 0.0 <= val <= 1.0:
                    return val
            except (TypeError, ValueError):
                continue
    return None


def _extract_question_type(question_obj: dict[str, Any]) -> str:
    """Extract question type from Metaculus question object."""
    q = question_obj.get("question")
    if isinstance(q, dict):
        qtype = q.get("type", "")
        if qtype in ("binary", "numeric", "multiple_choice"):
            return qtype
    raw_type = question_obj.get("type", "")
    if raw_type == "forecast":
        return "binary"
    return str(raw_type) if raw_type else "binary"


def _parse_question(obj: dict[str, Any]) -> PredictionMarketQuestion | None:
    """Parse a Metaculus API question object into a PredictionMarketQuestion."""
    title = obj.get("title", "").strip()
    if not title:
        return None

    qid = obj.get("id")
    slug = obj.get("slug", "")
    if qid:
        url = f"https://www.metaculus.com/questions/{qid}/{slug}/"
    else:
        return None

    status = obj.get("status", "")
    if status not in ("open", ""):
        return None

    prob = _extract_community_prediction(obj)
    nr_forecasters = obj.get("nr_forecasters")
    if isinstance(nr_forecasters, (int, float)):
        nr_forecasters = int(nr_forecasters)
    else:
        nr_forecasters = None

    return PredictionMarketQuestion(
        platform="metaculus",
        question_title=title,
        url=url,
        probability=prob,
        num_forecasters=nr_forecasters,
        volume_usd=None,
        close_date=obj.get("scheduled_close_time"),
        resolve_date=obj.get("scheduled_resolve_time"),
        question_type=_extract_question_type(obj),
    )


def fetch_open_questions(
    max_pages: int = 20,
) -> list[dict[str, Any]]:
    """Fetch all open forecast questions from Metaculus API (for cache building).

    Returns raw question objects.
    """
    url = _api_url()
    headers = _auth_headers()
    all_questions: list[dict[str, Any]] = []
    offset = 0

    for _ in range(max_pages):
        params = {
            "status": "open",
            "type": "forecast",
            "order_by": "-activity",
            "limit": _PAGE_SIZE,
            "offset": offset,
        }
        try:
            resp = requests.get(
                url, params=params, headers=headers, timeout=_REQUEST_TIMEOUT
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Metaculus API fetch failed at offset %d: %s", offset, exc)
            break

        results = data.get("results", [])
        if not isinstance(results, list) or not results:
            break

        all_questions.extend(results)
        offset += _PAGE_SIZE

        # Check if there are more pages
        if not data.get("next"):
            break

    return all_questions


def search_questions(
    queries: list[str],
    cached_questions: list[dict[str, Any]] | None = None,
) -> list[PredictionMarketQuestion]:
    """Search Metaculus questions by keyword matching against cached or fetched questions.

    Args:
        queries: Search query strings.
        cached_questions: Pre-fetched question objects (from cache). If None,
            falls back to a limited API fetch.
    """
    if cached_questions is None:
        # Fallback: fetch a limited set via API
        cached_questions = fetch_open_questions(max_pages=3)

    min_fc = _min_forecasters()
    query_terms = set()
    for q in queries:
        for word in q.lower().split():
            word = word.strip()
            if len(word) > 2:
                query_terms.add(word)

    if not query_terms:
        return []

    results: list[PredictionMarketQuestion] = []
    seen_ids: set[int] = set()

    for obj in cached_questions:
        title = (obj.get("title") or "").lower()
        if not title:
            continue

        # Check if any query term appears in the title
        matched = sum(1 for term in query_terms if term in title)
        if matched < 1:
            continue

        # Quality filter
        nr = obj.get("nr_forecasters")
        if isinstance(nr, (int, float)) and int(nr) < min_fc:
            continue

        qid = obj.get("id")
        if qid in seen_ids:
            continue
        seen_ids.add(qid)

        parsed = _parse_question(obj)
        if parsed is not None:
            results.append(parsed)

    return results
