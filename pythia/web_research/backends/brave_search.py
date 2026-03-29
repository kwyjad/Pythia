# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Brave Search API backend for web research grounding.

Provides deterministic direct web search via the Brave Web Search API.
Used as the primary grounding backend for RC and triage grounding calls.

Auth: ``BRAVE_SEARCH_API_KEY`` env var (X-Subscription-Token header).
Pricing: $5 per 1,000 queries.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Set

import requests

from pythia.web_research.types import EvidencePack, EvidenceSource

logger = logging.getLogger(__name__)

# Brave Search API endpoint
_BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Cost per query in USD ($5 / 1000 queries)
BRAVE_COST_PER_QUERY = 0.005

# Maximum signal lines to extract from snippets
MAX_SIGNAL_LINES = 8


# ---------------------------------------------------------------------------
# Query templates per hazard
# ---------------------------------------------------------------------------

_HAZARD_QUERY_TEMPLATES: Dict[str, List[str]] = {
    "ACE": [
        "{country} armed conflict violence displacement {year}",
        "{country} conflict security humanitarian crisis",
        "{country} conflict casualties attacks latest",
    ],
    "DR": [
        "{country} drought food insecurity famine {year}",
        "{country} food crisis humanitarian aid",
        "{country} food security situation update",
    ],
    "FL": [
        "{country} flooding displacement humanitarian {year}",
        "{country} floods damage humanitarian response",
        "{country} flood disaster latest",
    ],
    "TC": [
        "{country} cyclone typhoon hurricane {year}",
        "{country} tropical storm damage humanitarian",
    ],
}


def build_brave_queries(
    hazard_code: str,
    country_name: str,
    year: int | str | None = None,
) -> List[str]:
    """Build 2-3 search queries for a hazard using Brave Search templates.

    Parameters
    ----------
    hazard_code : str
        Hazard code (ACE, DR, FL, TC).
    country_name : str
        Human-readable country name.
    year : int or str or None
        Current year for temporal anchoring (default: current year).

    Returns
    -------
    list[str]
        2-3 query strings.
    """
    if year is None:
        year = datetime.now().year
    templates = _HAZARD_QUERY_TEMPLATES.get(
        hazard_code.upper(),
        ["{country} humanitarian crisis {year}", "{country} disaster latest"],
    )
    return [
        t.format(country=country_name, year=year)
        for t in templates
    ]


def _map_freshness(recency_days: int) -> str:
    """Map recency_days to Brave freshness parameter.

    Brave freshness values: pd (past day), pw (past week), pm (past month),
    py (past year).
    """
    if recency_days <= 1:
        return "pd"
    if recency_days <= 7:
        return "pw"
    if recency_days <= 31:
        return "pm"
    return "py"


def _run_single_query(
    query: str,
    api_key: str,
    freshness: str,
    timeout_sec: int,
    count: int = 10,
) -> tuple[List[Dict[str, Any]], int]:
    """Execute a single Brave Search API query.

    Returns
    -------
    tuple[list[dict], int]
        (results list, HTTP status code)
    """
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params: Dict[str, Any] = {
        "q": query,
        "count": count,
        "freshness": freshness,
        "extra_snippets": True,
        "text_decorations": False,
    }

    try:
        resp = requests.get(
            _BRAVE_SEARCH_URL,
            headers=headers,
            params=params,
            timeout=timeout_sec,
        )
        if resp.status_code != 200:
            return [], resp.status_code
        data = resp.json()
        web_results = data.get("web", {}).get("results", [])
        return web_results, resp.status_code
    except Exception as exc:
        logger.debug("Brave Search query failed: %s", exc)
        return [], 599


def fetch_via_brave_search(
    query: str,
    *,
    recency_days: int,
    include_structural: bool,
    timeout_sec: int,
    max_results: int,
    hazard_code: str | None = None,
    country_name: str | None = None,
) -> EvidencePack:
    """Fetch web research evidence via Brave Search API.

    When *hazard_code* and *country_name* are provided, runs 2-3 hazard-specific
    query formulations and deduplicates results by URL. Otherwise, runs a single
    query using the provided *query* string.

    Parameters
    ----------
    query : str
        Primary search query.
    recency_days : int
        Recency filter in days.
    include_structural : bool
        Whether to extract structural context from snippets.
    timeout_sec : int
        HTTP timeout per request.
    max_results : int
        Maximum results to return.
    hazard_code : str or None
        Hazard code for multi-query formulations.
    country_name : str or None
        Country name for multi-query formulations.

    Returns
    -------
    EvidencePack
        Evidence pack with sources, signals, and debug info.
    """
    pack = EvidencePack(query=query, recency_days=recency_days, backend="brave")

    api_key = os.getenv("BRAVE_SEARCH_API_KEY", "").strip()
    if not api_key:
        pack.error = {"type": "missing_api_key", "message": "BRAVE_SEARCH_API_KEY not set"}
        pack.debug = {"error": "missing_api_key"}
        return pack

    freshness = _map_freshness(recency_days)
    start_time = time.time()

    # Build query list: multi-query for hazard grounding, single query otherwise
    if hazard_code and country_name:
        queries = build_brave_queries(hazard_code, country_name)
    else:
        queries = [query]

    # Run all queries and collect results
    all_results: List[Dict[str, Any]] = []
    seen_urls: Set[str] = set()
    total_queries = 0
    last_status_code = 0

    for q in queries:
        results, status_code = _run_single_query(q, api_key, freshness, timeout_sec, count=max_results)
        total_queries += 1
        last_status_code = status_code
        for r in results:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)

    elapsed_ms = int((time.time() - start_time) * 1000)

    # Convert to EvidenceSource list
    sources: List[EvidenceSource] = []
    for r in all_results[:max_results]:
        url = r.get("url", "")
        title = r.get("title", url)
        description = r.get("description", "")
        # Brave returns extra_snippets as a list of additional snippet strings
        extra_snippets = r.get("extra_snippets") or []
        summary = description
        if extra_snippets:
            summary = description + " " + " ".join(extra_snippets[:2])
        publisher = r.get("meta_url", {}).get("hostname", "") if isinstance(r.get("meta_url"), dict) else ""
        page_age = r.get("page_age", "")

        sources.append(EvidenceSource(
            title=title,
            url=url,
            publisher=publisher,
            date=page_age if page_age else None,
            summary=summary.strip(),
        ))

    grounded = bool(sources)

    # Build recent_signals from top snippets
    recent_signals: List[str] = []
    for src in sources[:MAX_SIGNAL_LINES]:
        if src.summary:
            # Truncate long snippets to first 200 chars
            snippet = src.summary[:200].strip()
            if snippet:
                recent_signals.append(snippet)

    # Build structural_context from top results
    structural_context = ""
    if include_structural and sources:
        context_parts: List[str] = []
        for src in sources[:5]:
            if src.summary:
                context_parts.append(f"- {src.title}: {src.summary[:150]}")
        structural_context = "\n".join(context_parts[:MAX_SIGNAL_LINES])

    # Cost: $5 per 1000 queries
    cost_usd = total_queries * BRAVE_COST_PER_QUERY

    usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "web_search_requests": total_queries,
        "elapsed_ms": elapsed_ms,
        "cost_usd": cost_usd,
    }

    pack.sources = sources
    pack.grounded = grounded
    pack.structural_context = structural_context if include_structural else ""
    pack.recent_signals = recent_signals
    pack.unverified_sources = []

    pack.debug = {
        "provider": "brave",
        "model_id": "brave-web-search",
        "selected_model_id": "brave-web-search",
        "grounding_backend": "brave_search",
        "max_results": max_results,
        "status_code": last_status_code,
        "n_sources": len(sources),
        "n_verified_sources": len(sources),
        "total_queries": total_queries,
        "queries_used": queries,
        "freshness": freshness,
        "usage": usage,
        "fetched_at": datetime.utcnow().isoformat(),
    }

    if not grounded:
        pack.error = {"type": "no_results", "message": "Brave Search returned no results"}

    return pack
