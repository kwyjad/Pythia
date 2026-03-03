# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Metaculus question index cache.

Stores all open Metaculus forecast questions as a local JSON file for fast
keyword matching at query time. Refreshed when the cache is older than
the configured TTL (default 24 hours).
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from pythia.prediction_markets.config import get_platform_config

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = "data"
_DEFAULT_CACHE_FILENAME = "metaculus_index.json"


def _cache_ttl_seconds() -> float:
    cfg = get_platform_config("metaculus")
    hours = float(cfg.get("cache_ttl_hours", 24))
    return hours * 3600


def _cache_path() -> Path:
    cache_dir = os.getenv("PYTHIA_PM_CACHE_DIR", _DEFAULT_CACHE_DIR)
    return Path(cache_dir) / _DEFAULT_CACHE_FILENAME


def load_cache() -> list[dict[str, Any]] | None:
    """Load cached Metaculus questions if cache exists and is fresh.

    Returns None if cache is missing, stale, or corrupted.
    """
    path = _cache_path()
    if not path.exists():
        return None

    try:
        mtime = path.stat().st_mtime
        age = time.time() - mtime
        if age > _cache_ttl_seconds():
            logger.debug("Metaculus cache stale (%.0fs old, ttl=%.0fs)", age, _cache_ttl_seconds())
            return None

        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        questions = data.get("questions", [])
        if not isinstance(questions, list):
            return None
        logger.debug("Loaded %d questions from Metaculus cache", len(questions))
        return questions
    except Exception as exc:
        logger.warning("Failed to load Metaculus cache: %s", exc)
        return None


def save_cache(questions: list[dict[str, Any]]) -> None:
    """Save Metaculus question index to cache file."""
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Store minimal fields to keep cache small
        compact = []
        for q in questions:
            compact.append({
                "id": q.get("id"),
                "title": q.get("title", ""),
                "slug": q.get("slug", ""),
                "type": q.get("type", ""),
                "status": q.get("status", ""),
                "nr_forecasters": q.get("nr_forecasters"),
                "scheduled_close_time": q.get("scheduled_close_time"),
                "scheduled_resolve_time": q.get("scheduled_resolve_time"),
                "question": q.get("question"),  # contains aggregations
            })

        payload = {
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "count": len(compact),
            "questions": compact,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        logger.info("Saved %d questions to Metaculus cache at %s", len(compact), path)
    except Exception as exc:
        logger.warning("Failed to save Metaculus cache: %s", exc)


def get_or_refresh_cache() -> list[dict[str, Any]]:
    """Get cached Metaculus questions, refreshing from API if stale.

    Returns an empty list if both cache and API fetch fail.
    """
    cached = load_cache()
    if cached is not None:
        return cached

    # Refresh from API
    try:
        from pythia.prediction_markets.platforms.metaculus import fetch_open_questions

        logger.info("Refreshing Metaculus question index cache...")
        questions = fetch_open_questions(max_pages=20)
        if questions:
            save_cache(questions)
            return questions
        logger.warning("Metaculus API returned no questions")
        return []
    except Exception as exc:
        logger.warning("Failed to refresh Metaculus cache: %s", exc)
        return []
