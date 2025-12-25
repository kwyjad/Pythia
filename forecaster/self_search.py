# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

from pythia.web_research import fetch_evidence_pack


def model_self_search_enabled() -> bool:
    return os.getenv("PYTHIA_MODEL_SELF_SEARCH_ENABLED", "0") == "1"


def self_search_enabled() -> bool:
    return (
        os.getenv("PYTHIA_FORECASTER_SELF_SEARCH", "0") == "1"
        and os.getenv("PYTHIA_SPD_WEB_SEARCH_ENABLED", "0") == "1"
        and model_self_search_enabled()
    )


def self_search_limits() -> Tuple[int, int]:
    """Return (max_calls_per_model, max_sources)."""

    try:
        max_calls = int(os.getenv("PYTHIA_FORECASTER_SELF_SEARCH_MAX_CALLS_PER_MODEL", "1"))
    except Exception:
        max_calls = 1
    try:
        max_sources = int(os.getenv("PYTHIA_FORECASTER_SELF_SEARCH_MAX_SOURCES", "6"))
    except Exception:
        max_sources = 6
    return max_calls, max_sources


def extract_self_search_query(text: str) -> Optional[str]:
    """
    Return the query string if the model requested more evidence using the
    NEED_WEB_EVIDENCE escape hatch; otherwise None.
    """

    if not text:
        return None
    m = re.search(r"NEED_WEB_EVIDENCE:\s*(.+)", str(text), flags=re.I)
    if not m:
        return None
    query = m.group(1).strip()
    return query or None


def trim_sources(pack: Dict[str, Any], max_sources: int) -> Dict[str, Any]:
    sources = pack.get("sources") or []
    if isinstance(sources, list) and len(sources) > max_sources:
        sources = sources[:max_sources]
    trimmed = dict(pack or {})
    trimmed["sources"] = sources if isinstance(sources, list) else []
    return trimmed


def append_evidence_to_prompt(prompt: str, evidence: Dict[str, Any]) -> str:
    evidence_text = json.dumps(evidence or {}, ensure_ascii=False, indent=2)
    return (
        f"{prompt}\n\nSELF-SEARCH RESULTS (recent signals prioritized; structural context is background only):\n"
        f"```json\n{evidence_text}\n```\n"
        "Use this evidence and return the required JSON output. Do not request more evidence again."
    )


def append_retriever_evidence_to_prompt(prompt: str, evidence: Dict[str, Any]) -> str:
    evidence_text = json.dumps(evidence or {}, ensure_ascii=False, indent=2)
    return (
        f"{prompt}\n\nRETRIEVER EVIDENCE PACK (recent signals prioritized; structural context is background only):\n"
        f"```json\n{evidence_text}\n```\n"
        "Use this evidence and return the required JSON output. Do not request more evidence."
    )


def run_self_search(
    query: str,
    *,
    run_id: Optional[str],
    question_id: Optional[str],
    iso3: Optional[str],
    hazard_code: Optional[str],
    purpose: str = "forecast_self_search",
) -> Dict[str, Any]:
    """Execute a single web-research call for the given query."""

    return fetch_evidence_pack(
        query,
        purpose=purpose,
        run_id=f"{run_id or 'forecast'}__self_search",
        question_id=question_id,
    )


def combine_usage(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    combined = dict(base or {})
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        combined[key] = int(base.get(key) or 0) + int(extra.get(key) or 0)
    combined["cost_usd"] = float(base.get("cost_usd") or 0.0) + float(extra.get("cost_usd") or 0.0)
    combined["elapsed_ms"] = max(int(base.get("elapsed_ms") or 0), int(extra.get("elapsed_ms") or 0))
    if "self_search" in extra:
        combined["self_search"] = extra["self_search"]
    return combined
