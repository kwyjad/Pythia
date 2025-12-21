# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

from pythia.web_research.types import EvidencePack, EvidenceSource


def _extract_sources_from_grounding(gm: Dict[str, Any]) -> List[EvidenceSource]:
    sources: List[EvidenceSource] = []
    supports = gm.get("groundingSupports") or []
    for support in supports:
        src = support.get("support") or support
        url = src.get("sourceUrl") or src.get("uri") or ""
        title = src.get("title") or src.get("description") or url
        publisher = src.get("publisher") or src.get("site") or ""
        published = src.get("publishedDate") or src.get("date")
        summary = src.get("summary") or ""
        if url:
            sources.append(EvidenceSource(title=title or url, url=url, publisher=publisher, date=published, summary=summary))

    chunks = gm.get("groundingChunks") or []
    for chunk in chunks:
        # groundingChunks may contain inline source urls/titles
        url = chunk.get("sourceUrl") or ""
        if not url:
            continue
        title = chunk.get("title") or chunk.get("content") or url
        publisher = chunk.get("publisher") or ""
        published = chunk.get("publishedDate") or None
        sources.append(EvidenceSource(title=title, url=url, publisher=publisher, date=published, summary=""))

    return sources


def parse_gemini_grounding_response(resp: Dict[str, Any]) -> Tuple[List[EvidenceSource], bool, Dict[str, Any]]:
    """Parse a Gemini response dict for grounding metadata and sources."""

    debug: Dict[str, Any] = {}
    candidates = resp.get("candidates") or []
    if not candidates:
        return [], False, debug

    gm = candidates[0].get("groundingMetadata") or {}
    debug["webSearchQueries"] = gm.get("webSearchQueries") or []
    debug["groundingSupports_count"] = len(gm.get("groundingSupports", []))
    debug["groundingChunks_count"] = len(gm.get("groundingChunks", []))

    sources = _extract_sources_from_grounding(gm) if gm else []
    grounded = bool(gm)
    return sources, grounded, debug


def fetch_via_gemini(
    query: str,
    *,
    recency_days: int,
    include_structural: bool,
    timeout_sec: int,
    max_results: int,
) -> EvidencePack:
    """
    Placeholder Gemini grounding fetcher.

    This function intentionally avoids making external calls in test environments.
    In production, plug in the actual Gemini client and pass its raw response
    through `parse_gemini_grounding_response`.
    """

    # In CI/test environments we keep this offline-safe.
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pack = EvidencePack(query=query, recency_days=recency_days, backend="gemini")
        pack.debug = {"error": "missing_api_key"}
        return pack

    # Real call is left as a future integration; for now return an empty grounded pack.
    pack = EvidencePack(query=query, recency_days=recency_days, backend="gemini")
    pack.structural_context = "" if not include_structural else "Structural context unavailable (placeholder)."
    pack.recent_signals = []
    pack.grounded = False
    pack.debug = {
        "webSearchQueries": [],
        "groundingSupports_count": 0,
        "groundingChunks_count": 0,
        "max_results": max_results,
        "fetched_at": datetime.utcnow().isoformat(),
    }
    return pack
