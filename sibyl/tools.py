# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl agent tools: open-web search and page fetch.

Exactly two tools are exposed to the agent — ``brave_search`` (via the
shared Brave wrapper, with its circuit breaker and rate limiting) and
``fetch_url``. The structured Pythia connectors are deliberately NOT
available: the independence of the two tracks is the point. A disabled,
config-gated extension point exists for authoritative live lookups
(``LIVE_LOOKUPS_ENABLED``, default off).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

import requests

from pythia.web_research.backends.brave_search import fetch_via_brave_search
from pythia.web_research.types import EvidenceSource

from sibyl.config import (
    BRAVE_MAX_RESULTS,
    BRAVE_TIMEOUT_SEC,
    FETCH_URL_MAX_CHARS,
    FETCH_URL_TIMEOUT_SEC,
    SEARCH_WINDOW_DAYS,
)
from sibyl.leakage import (
    LeakageStats,
    date_range_freshness,
    filter_sources,
    is_backtest,
    is_blocked_url,
    snippet_leaks,
)

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Uniform result envelope handed back to the agent loop."""

    tool: str
    ok: bool
    text: str  # what the agent sees
    cost_usd: float = 0.0
    sources: List[EvidenceSource] = field(default_factory=list)
    leakage: LeakageStats = field(default_factory=LeakageStats)
    error: Optional[str] = None


def brave_search(query: str, as_of: date, *, today: Optional[date] = None) -> ToolResult:
    """Date-filtered web search through the shared Brave wrapper.

    The freshness date-range always ends at *as_of* (harmless live, a hard
    cap in backtest); leakage post-filtering runs on the results.
    """
    freshness = date_range_freshness(as_of, SEARCH_WINDOW_DAYS)
    pack = fetch_via_brave_search(
        query,
        recency_days=SEARCH_WINDOW_DAYS,
        include_structural=False,
        timeout_sec=BRAVE_TIMEOUT_SEC,
        max_results=BRAVE_MAX_RESULTS,
        freshness_override=freshness,
    )
    cost = 0.0
    try:
        cost = float((pack.debug.get("usage") or {}).get("cost_usd", 0.0))
    except (TypeError, ValueError):
        cost = 0.0

    if pack.error and not pack.sources:
        err_type = (pack.error or {}).get("type", "unknown")
        return ToolResult(
            tool="brave_search",
            ok=False,
            text=f"[search failed: {err_type}] No results for query: {query}",
            cost_usd=cost,
            error=err_type,
        )

    kept, stats = filter_sources(pack.sources, as_of, today=today)
    if not kept:
        return ToolResult(
            tool="brave_search",
            ok=True,
            text=(
                "No usable results (all results were filtered out by the "
                f"as-of/leakage controls). Query: {query}"
            ),
            cost_usd=cost,
            leakage=stats,
        )

    lines = [f"Search results for: {query} (window ending {as_of.isoformat()})"]
    for i, src in enumerate(kept, start=1):
        date_str = f" [{src.date}]" if src.date else ""
        summary = (src.summary or "").strip()
        lines.append(f"{i}. {src.title}{date_str}\n   {src.url}\n   {summary}")
    return ToolResult(
        tool="brave_search",
        ok=True,
        text="\n".join(lines),
        cost_usd=cost,
        sources=kept,
        leakage=stats,
    )


def _html_to_text(html: str) -> str:
    """Extract readable text from HTML via BeautifulSoup (repo-standard)."""
    from bs4 import BeautifulSoup  # deferred: not needed for search-only runs

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def fetch_url(url: str, as_of: date, *, today: Optional[date] = None) -> ToolResult:
    """Fetch a page the agent found via search and return readable text.

    Blocked resolution-source URLs are refused; in backtest the extracted
    text goes through the snippet leak classifier before being returned.
    """
    stats = LeakageStats(total_retrieved=1)
    if is_blocked_url(url):
        stats.dropped_blocked_domain = 1
        return ToolResult(
            tool="fetch_url",
            ok=False,
            text=(
                "This URL belongs to a resolution data source and is blocked "
                "for Sibyl. Rely on open-web reporting instead."
            ),
            leakage=stats,
            error="blocked_domain",
        )

    try:
        resp = requests.get(
            url,
            timeout=FETCH_URL_TIMEOUT_SEC,
            headers={"User-Agent": "Mozilla/5.0 (compatible; PythiaSibyl/1.0)"},
        )
    except requests.RequestException as exc:
        return ToolResult(
            tool="fetch_url", ok=False,
            text=f"[fetch failed: {type(exc).__name__}] {url}",
            leakage=stats, error=type(exc).__name__,
        )
    if resp.status_code != 200:
        return ToolResult(
            tool="fetch_url", ok=False,
            text=f"[fetch failed: HTTP {resp.status_code}] {url}",
            leakage=stats, error=f"http_{resp.status_code}",
        )

    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "html" in content_type or resp.text.lstrip()[:1] == "<":
        text = _html_to_text(resp.text)
    else:
        text = resp.text
    text = (text or "").strip()[:FETCH_URL_MAX_CHARS]
    if not text:
        return ToolResult(
            tool="fetch_url", ok=False,
            text=f"[fetch returned no readable text] {url}",
            leakage=stats, error="empty",
        )

    if is_backtest(as_of, today=today) and snippet_leaks(text[:2000], as_of):
        stats.dropped_post_asof = 1
        stats.notes.append(f"fetched page dated after asOf dropped: {url}")
        logger.info("sibyl.leakage: dropped fetched page post-asOf: %s", url)
        return ToolResult(
            tool="fetch_url", ok=False,
            text=(
                "The fetched page contains material dated after the "
                f"forecast as-of date ({as_of.isoformat()}) and was withheld "
                "by the backtest leakage controls."
            ),
            leakage=stats, error="post_asof",
        )

    return ToolResult(
        tool="fetch_url", ok=True,
        text=f"Content of {url} (truncated to {FETCH_URL_MAX_CHARS} chars):\n{text}",
        leakage=stats,
    )
