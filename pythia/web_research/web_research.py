# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from pythia.db.schema import connect, ensure_schema
from pythia.web_research.budget import BudgetGuard, BudgetExceededError
from pythia.web_research.cache import get as cache_get, set as cache_set
from pythia.web_research.types import EvidencePack, EvidenceSource
from pythia.web_research.backends import gemini_grounding
from pythia.db.util import log_web_research_call


class WebResearchError(Exception):
    """Raised for unexpected web research failures."""


class WebResearchBudgetError(WebResearchError):
    """Raised when budget or call caps are hit."""


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)) or default)
    except Exception:
        return default


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _is_enabled() -> bool:
    return os.getenv("PYTHIA_WEB_RESEARCH_ENABLED", "0") == "1"


def _build_cache_key(query: str, recency_days: int, backend: str) -> str:
    h = hashlib.sha256()
    h.update(f"{query}|{recency_days}|{backend}".encode("utf-8"))
    return h.hexdigest()


def _pack_from_cache(query: str, cached: Dict[str, Any], *, backend: str, recency_days: int) -> EvidencePack:
    pack = EvidencePack(query=query, recency_days=recency_days, backend=backend)
    pack_data = dict(cached or {})
    pack.structural_context = pack_data.get("structural_context", "")
    pack.recent_signals = pack_data.get("recent_signals") or []
    pack.grounded = bool(pack_data.get("grounded"))
    pack.debug = pack_data.get("debug") or {}
    pack.sources = [EvidenceSource(**src) for src in pack_data.get("sources", []) if isinstance(src, dict)]
    pack.error = pack_data.get("error")
    pack.retrieved_at = pack_data.get("retrieved_at", pack.retrieved_at)
    if pack.debug and isinstance(pack.debug, dict):
        usage = pack.debug.get("usage")
        if isinstance(usage, dict):
            pack.debug["usage"] = {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd") if usage.get(k) is not None}
    return pack


def fetch_evidence_pack(
    query: str,
    purpose: str,
    run_id: Optional[str] = None,
    question_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified entrypoint for web research evidence packs.

    Returns a stable dict with the evidence pack shape. When disabled or budget
    caps are hit, a structured error is returned.
    """

    backend = _env_str("PYTHIA_WEB_RESEARCH_BACKEND", "gemini").lower()
    recency_days = _env_int("PYTHIA_WEB_RESEARCH_RECENCY_DAYS", 120)
    include_structural = os.getenv("PYTHIA_WEB_RESEARCH_INCLUDE_STRUCTURAL", "1") != "0"
    timeout_sec = _env_int("PYTHIA_WEB_RESEARCH_TIMEOUT_SEC", 60)
    max_results = _env_int("PYTHIA_WEB_RESEARCH_MAX_RESULTS", 10)

    start_ms = int(time.time() * 1000)
    guard = BudgetGuard(run_id or "default")

    if not _is_enabled():
        pack = EvidencePack(query=query, recency_days=recency_days, backend=backend)
        pack.error = {"type": "disabled", "message": "web research disabled via PYTHIA_WEB_RESEARCH_ENABLED"}
        _log_web_research(pack, purpose, run_id, question_id, start_ms, cached=False, success=False)
        return pack.to_dict()

    try:
        guard.check_and_reserve(cost_estimate=0.0)
    except BudgetExceededError as exc:
        pack = EvidencePack(query=query, recency_days=recency_days, backend=backend)
        pack.error = {"type": "budget_exceeded", "message": str(exc)}
        _log_web_research(pack, purpose, run_id, question_id, start_ms, cached=False, success=False)
        return pack.to_dict()
    except Exception as exc:
        pack = EvidencePack(query=query, recency_days=recency_days, backend=backend)
        pack.error = {"type": "budget_error", "message": str(exc)}
        _log_web_research(pack, purpose, run_id, question_id, start_ms, cached=False, success=False)
        return pack.to_dict()

    cache_key = _build_cache_key(query, recency_days, backend)
    cached_value = cache_get(cache_key)
    if cached_value:
        pack = _pack_from_cache(query, cached_value, backend=backend, recency_days=recency_days)
        _log_web_research(pack, purpose, run_id, question_id, start_ms, cached=True, success=True)
        return pack.to_dict()

    try:
        if backend == "gemini":
            pack = gemini_grounding.fetch_via_gemini(
                query,
                recency_days=recency_days,
                include_structural=include_structural,
                timeout_sec=timeout_sec,
                max_results=max_results,
            )
        else:
            pack = EvidencePack(query=query, recency_days=recency_days, backend=backend)
            pack.error = {"type": "unsupported_backend", "message": f"backend={backend} not implemented"}
    except BudgetExceededError as exc:
        pack = EvidencePack(query=query, recency_days=recency_days, backend=backend)
        pack.error = {"type": "budget_exceeded", "message": str(exc)}
    except Exception as exc:
        pack = EvidencePack(query=query, recency_days=recency_days, backend=backend)
        pack.error = {"type": "unexpected_error", "message": str(exc)[:500]}

    cache_set(cache_key, pack.to_dict())
    _log_web_research(pack, purpose, run_id, question_id, start_ms, cached=False, success=pack.error is None)
    return pack.to_dict()


def _log_web_research(
    pack: EvidencePack,
    purpose: str,
    run_id: Optional[str],
    question_id: Optional[str],
    start_ms: int,
    *,
    cached: bool,
    success: bool,
) -> None:
    """Best-effort audit logging to llm_calls."""

    try:
        elapsed_ms = int(time.time() * 1000) - int(start_ms)
        usage_json = {
            "grounded": bool(pack.grounded),
            "n_sources": len(pack.sources),
            "cached": cached,
        }
        if pack.debug:
            usage_json["debug"] = pack.debug
        if pack.error:
            usage_json["error"] = pack.error
        if pack.debug and isinstance(pack.debug, dict):
            usage = pack.debug.get("usage")
            if isinstance(usage, dict):
                usage_json.update({
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "cost_usd": usage.get("cost_usd", 0.0),
                })

        con = connect(read_only=False)
        ensure_schema(con)
        log_web_research_call(
            con,
            component="web_research",
            phase=_resolve_phase(purpose),
            provider="web_research",
            model_name=pack.backend or "unknown",
            model_id=pack.backend or "unknown",
            run_id=run_id,
            question_id=question_id,
            prompt_text=pack.query,
            response_text=json.dumps(pack.to_dict(), ensure_ascii=False),
            parsed_json=pack.to_dict(),
            usage=usage_json,
            elapsed_ms=elapsed_ms,
            error_text=pack.error.get("message") if pack.error else None,
            success=success and not pack.error,
        )
        con.close()
    except Exception:
        return


def _resolve_phase(purpose: str) -> str:
    purpose_norm = (purpose or "").lower()
    if "hs" in purpose_norm:
        return "hs_web_research"
    if "forecast" in purpose_norm:
        return "forecast_web_research"
    return "research_web_research"
