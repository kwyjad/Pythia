# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""
market_snapshot.py — Standalone Manifold prediction market snapshot utility.

Extracted from forecaster/research.py. Fetches the best-matching Manifold
prediction market for a given question title and returns structured data
for injection into the SPD v2 prompt.

Dependencies: requests, difflib, re, math, logging (all stdlib except requests).
"""
from __future__ import annotations

import difflib
import logging
import math
import re
from typing import Any, Optional

import requests

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MARKET_SIMILARITY_THRESHOLD = 0.55

# ---------------------------------------------------------------------------
# Text similarity helpers
# ---------------------------------------------------------------------------


def _norm_for_similarity(s: str) -> str:
    """Lowercase alphanumeric string for similarity comparisons."""
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _title_similarity(a: str, b: str) -> float:
    """Blend token overlap and sequence ratio for rough similarity."""
    na, nb = _norm_for_similarity(a), _norm_for_similarity(b)
    if not na or not nb:
        return 0.0
    toks_a = set(na.split())
    toks_b = set(nb.split())
    if not toks_a or not toks_b:
        token_score = 0.0
    else:
        token_score = len(toks_a & toks_b) / len(toks_a | toks_b)
    seq_score = difflib.SequenceMatcher(None, na, nb).ratio()
    # Weighted average; emphasize sequence ratio but keep token overlap
    return 0.65 * seq_score + 0.35 * token_score


# ---------------------------------------------------------------------------
# Probability extraction helpers
# ---------------------------------------------------------------------------


def _find_numeric_value(obj: Any) -> Optional[float]:
    """Recursively search for the first numeric value in a nested structure."""
    if isinstance(obj, (int, float)):
        val = float(obj)
        if math.isnan(val):
            return None
        return val
    if isinstance(obj, dict):
        preferred_keys = (
            "p_yes",
            "probability",
            "p",
            "yes",
            "value",
            "q2",
            "median",
        )
        for key in preferred_keys:
            if key in obj and isinstance(obj[key], (int, float)):
                val = float(obj[key])
                if math.isnan(val):
                    continue
                return val
        for val in obj.values():
            got = _find_numeric_value(val)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for item in obj:
            got = _find_numeric_value(item)
            if got is not None:
                return got
    return None


def _format_percent(prob: float) -> str:
    return f"{max(0.0, min(prob, 1.0)) * 100:.1f}%"


# ---------------------------------------------------------------------------
# Manifold snapshot
# ---------------------------------------------------------------------------


def _manifold_snapshot(
    query_title: str, timeout: float = 12.0
) -> tuple[Optional[dict[str, Any]], list[str]]:
    """Fetch and score a single best Manifold market match.

    Returns (market_dict, debug_lines) where market_dict is
    ``{platform, title, url, prob}`` or ``None``.
    """
    url = "https://api.manifold.markets/v0/search-markets"
    params = {"term": query_title, "limit": 40}
    debug: list[str] = []
    try:
        resp = requests.get(url, params=params, timeout=timeout)
    except Exception as exc:
        debug.append(f"Manifold: request error {exc!r}")
        return None, debug
    if resp.status_code != 200:
        debug.append(f"Manifold: HTTP {resp.status_code}")
        return None, debug
    try:
        markets = resp.json()
    except Exception as exc:
        debug.append(f"Manifold: JSON error {exc!r}")
        return None, debug
    if not isinstance(markets, list):
        debug.append("Manifold: unexpected payload (not a list)")
        return None, debug
    best: tuple[float, dict[str, Any]] = (0.0, {})
    for market in markets:
        if not isinstance(market, dict):
            continue
        if market.get("isResolved"):
            continue
        if (market.get("outcomeType") or "").upper() != "BINARY":
            continue
        title = market.get("question") or market.get("title") or ""
        score = _title_similarity(query_title, title)
        if score > best[0]:
            best = (score, market)
    if not best[1]:
        debug.append("Manifold: no open binary results in response")
        return None, debug
    if best[0] < _MARKET_SIMILARITY_THRESHOLD:
        best_title = ""
        if isinstance(best[1], dict):
            best_title = best[1].get("question") or best[1].get("title") or ""
        debug.append(
            f"Manifold: best score {best[0]:.2f} below threshold "
            f"{_MARKET_SIMILARITY_THRESHOLD:.2f} for '{best_title or '(none)'}'"
        )
        return None, debug
    chosen = best[1]
    prob = chosen.get("probability")
    if prob is None:
        prob = chosen.get("p")
    if isinstance(prob, (int, float)):
        prob = float(prob)
        if prob > 1:
            prob /= 100.0
    else:
        debug.append("Manifold: match missing probability field")
        return None, debug
    market_url = chosen.get("url")
    if not market_url:
        slug = chosen.get("slug") or ""
        creator = chosen.get("creatorUsername") or ""
        if slug and creator:
            market_url = f"https://manifold.markets/{creator}/{slug}"
    debug.append(
        f"Manifold: matched '{(chosen.get('question') or chosen.get('title') or '')[:80]}' "
        f"(score {best[0]:.2f}, {prob * 100:.1f}%)"
    )
    return {
        "platform": "Manifold",
        "title": chosen.get("question") or chosen.get("title") or "",
        "url": market_url or "",
        "prob": prob,
    }, debug


def _collect_market_snapshots(
    query_title: str, timeout: float = 12.0
) -> tuple[str, dict[str, bool], list[str]]:
    """Return markdown snippet + meta flags for market matches plus debug info."""
    matches: list[dict[str, Any]] = []
    found = {"manifold": False}
    debug_lines: list[str] = []

    m2, dbg2 = _manifold_snapshot(query_title, timeout=timeout)
    if dbg2:
        debug_lines.extend(dbg2)
    if m2:
        matches.append(m2)
        found["manifold"] = True

    if not matches:
        if not debug_lines:
            debug_lines.append("Market snapshots: no matches")
        return "", found, debug_lines

    lines = [
        "### Market Snapshots (community forecasts)",
    ]
    for item in matches:
        title = item.get("title") or "(untitled)"
        item_url = item.get("url") or ""
        prob = item.get("prob")
        prob_text = _format_percent(prob) if isinstance(prob, (int, float)) else "N/A"
        if item_url:
            question_txt = f"[{title}]({item_url})"
        else:
            question_txt = title
        lines.append(
            f"- **{item.get('platform', '?')}**: {question_txt} — Community forecast: {prob_text}"
        )

    matched_names = sorted(name for name, present in found.items() if present)
    if matched_names:
        debug_lines.append("Market snapshots: found " + ", ".join(matched_names))
    else:
        debug_lines.append("Market snapshots: found none")

    return "\n".join(lines), found, debug_lines


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_market_snapshot(
    query_title: str,
    timeout: float = 12.0,
) -> dict | None:
    """Fetch the best-matching Manifold prediction market for a question.

    Returns a dict with keys: platform, title, url, prob (0-1 float),
    or None if no match found.
    """
    if not query_title or not query_title.strip():
        return None
    try:
        snapshot, debug = _manifold_snapshot(query_title, timeout=timeout)
        for line in debug:
            LOG.debug(line)
        return snapshot
    except Exception as exc:
        LOG.debug("fetch_market_snapshot failed: %s", exc)
        return None


def format_market_snapshot_for_spd(snapshot: dict | None) -> str:
    """Format a market snapshot for injection into the SPD v2 prompt.

    Returns a compact text block or empty string.
    """
    if not snapshot:
        return ""
    title = snapshot.get("title") or "(untitled)"
    url = snapshot.get("url") or ""
    prob = snapshot.get("prob")
    prob_text = _format_percent(prob) if isinstance(prob, (int, float)) else "N/A"
    platform = snapshot.get("platform", "Manifold")
    if url:
        return f"**{platform}**: [{title}]({url}) — {prob_text}"
    return f"**{platform}**: {title} — {prob_text}"
