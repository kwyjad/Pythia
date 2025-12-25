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
import importlib

from pythia.db.schema import connect, ensure_schema
from pythia.web_research.budget import BudgetGuard, BudgetExceededError
from pythia.web_research.cache import get as cache_get, set as cache_set
from pythia.web_research.types import EvidencePack, EvidenceSource
from pythia.db.util import log_web_research_call
from forecaster.providers import estimate_cost_usd


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


def _summarize_attempt(pack: EvidencePack) -> Dict[str, Any]:
    return {
        "backend": pack.backend,
        "grounded": bool(pack.sources),
        "n_sources": len(pack.sources),
        "n_unverified_sources": len(getattr(pack, "unverified_sources", []) or []),
        "error": pack.error,
    }


def _normalize_attempted_backends(value: Any, default_backend: str, pack: EvidencePack | None = None) -> list[Dict[str, Any]]:
    attempts: list[Dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                attempts.append(
                    {
                        "backend": item.get("backend") or default_backend,
                        "grounded": bool(item.get("grounded")) or bool(item.get("n_sources")),
                        "n_sources": int(item.get("n_sources", 0) or 0),
                        "n_unverified_sources": int(item.get("n_unverified_sources", 0) or 0),
                        "error": item.get("error"),
                    }
                )
            elif isinstance(item, str):
                attempts.append(
                    {"backend": item, "grounded": False, "n_sources": 0, "n_unverified_sources": 0, "error": None}
                )
    if not attempts and pack is not None:
        attempts.append(_summarize_attempt(pack))
    elif not attempts and default_backend:
        attempts.append({"backend": default_backend, "grounded": False, "n_sources": 0, "n_unverified_sources": 0, "error": None})
    return attempts


def _build_cache_key(query: str, recency_days: int, backend: str, model_id: str | None) -> str:
    h = hashlib.sha256()
    model_tag = model_id or ""
    h.update(f"{query}|{recency_days}|{backend}|{model_tag}".encode("utf-8"))
    return h.hexdigest()


def _fetch_via_exa(query: str, *, recency_days: int, timeout_sec: int, max_results: int) -> EvidencePack:
    """Skeleton Exa backend, opt-in via EXA_API_KEY."""

    pack = EvidencePack(query=query, recency_days=recency_days, backend="exa")
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        pack.error = {"type": "missing_api_key", "message": "EXA_API_KEY not set"}
        pack.debug = {"error": "missing_api_key"}
        return pack

    pack.error = {"type": "backend_unimplemented", "message": "Exa backend is not yet implemented"}
    pack.debug = {"error": "backend_unimplemented"}
    return pack


def _fetch_via_perplexity(query: str, *, recency_days: int, timeout_sec: int, max_results: int) -> EvidencePack:
    """Skeleton Perplexity backend, opt-in via PERPLEXITY_API_KEY."""

    pack = EvidencePack(query=query, recency_days=recency_days, backend="perplexity")
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        pack.error = {"type": "missing_api_key", "message": "PERPLEXITY_API_KEY not set"}
        pack.debug = {"error": "missing_api_key"}
        return pack

    pack.error = {"type": "backend_unimplemented", "message": "Perplexity backend is not yet implemented"}
    pack.debug = {"error": "backend_unimplemented"}
    return pack


def _pack_from_cache(query: str, cached: Dict[str, Any], *, backend: str, recency_days: int) -> EvidencePack:
    pack = EvidencePack(query=query, recency_days=recency_days, backend=backend)
    pack_data = dict(cached or {})
    pack.backend = pack_data.get("backend", backend)
    pack.structural_context = pack_data.get("structural_context", "")
    pack.recent_signals = pack_data.get("recent_signals") or []
    pack.sources = [EvidenceSource(**src) for src in pack_data.get("sources", []) if isinstance(src, dict)]
    pack.unverified_sources = [EvidenceSource(**src) for src in pack_data.get("unverified_sources", []) if isinstance(src, dict)]
    pack.grounded = bool(pack.sources)
    pack.debug = pack_data.get("debug") or {}
    pack.error = pack_data.get("error")
    pack.retrieved_at = pack_data.get("retrieved_at", pack.retrieved_at)
    if pack.debug and isinstance(pack.debug, dict):
        usage = pack.debug.get("usage")
        if isinstance(usage, dict):
            pack.debug["usage"] = {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens", "cost_usd") if usage.get(k) is not None}
        pack.debug["attempted_backends"] = _normalize_attempted_backends(pack.debug.get("attempted_backends"), pack.backend, pack)
        pack.debug.setdefault("selected_backend", pack.backend if pack.sources else pack.debug.get("selected_backend") or pack.backend)
    return pack


def _load_backend_module(module_name: str):
    try:
        module = importlib.import_module(f"pythia.web_research.backends.{module_name}")
        # Cache on globals for test monkeypatch compatibility
        globals()[module_name] = module
        return module, None
    except ImportError as exc:
        return None, str(exc)


def fetch_evidence_pack(
    query: str,
    purpose: str,
    run_id: Optional[str] = None,
    question_id: Optional[str] = None,
    hs_run_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Unified entrypoint for web research evidence packs.

    Returns a stable dict with the evidence pack shape. When disabled or budget
    caps are hit, a structured error is returned.
    """

    retriever_enabled = os.getenv("PYTHIA_RETRIEVER_ENABLED", "0") == "1"
    retriever_model_id = _env_str("PYTHIA_RETRIEVER_MODEL_ID", "")
    backend = _env_str("PYTHIA_WEB_RESEARCH_BACKEND", "gemini").lower()
    recency_days = _env_int("PYTHIA_WEB_RESEARCH_RECENCY_DAYS", 120)
    include_structural = os.getenv("PYTHIA_WEB_RESEARCH_INCLUDE_STRUCTURAL", "1") != "0"
    timeout_sec = _env_int("PYTHIA_WEB_RESEARCH_TIMEOUT_SEC", 60)
    max_results = _env_int("PYTHIA_WEB_RESEARCH_MAX_RESULTS", 10)
    fallback_backend = _env_str("PYTHIA_WEB_RESEARCH_FALLBACK_BACKEND", "").lower()
    if retriever_enabled:
        backend = "gemini"
        fallback_backend = ""
        if not model_id:
            model_id = retriever_model_id or None

    start_ms = int(time.time() * 1000)
    guard = BudgetGuard(run_id or "default")

    if not _is_enabled():
        pack = EvidencePack(query=query, recency_days=recency_days, backend=backend)
        pack.error = {"type": "disabled", "message": "web research disabled via PYTHIA_WEB_RESEARCH_ENABLED"}
        _log_web_research(pack, purpose, run_id, question_id, hs_run_id, start_ms, cached=False, success=False)
        return pack.to_dict()

    try:
        guard.check_and_reserve(cost_estimate=0.0)
    except BudgetExceededError as exc:
        pack = EvidencePack(query=query, recency_days=recency_days, backend=backend)
        pack.error = {"type": "budget_exceeded", "message": str(exc)}
        _log_web_research(pack, purpose, run_id, question_id, hs_run_id, start_ms, cached=False, success=False)
        return pack.to_dict()
    except Exception as exc:
        pack = EvidencePack(query=query, recency_days=recency_days, backend=backend)
        pack.error = {"type": "budget_error", "message": str(exc)}
        _log_web_research(pack, purpose, run_id, question_id, hs_run_id, start_ms, cached=False, success=False)
        return pack.to_dict()

    cache_key = _build_cache_key(query, recency_days, backend, model_id)
    cached_value = cache_get(cache_key)
    if cached_value:
        pack = _pack_from_cache(query, cached_value, backend=backend, recency_days=recency_days)
        if isinstance(pack.debug, dict):
            pack.debug.setdefault("attempted_backends", [pack.backend])
            pack.debug.setdefault("selected_backend", pack.backend if pack.sources else "")
            pack.debug.setdefault("n_verified_sources", len(pack.sources))
        _log_web_research(pack, purpose, run_id, question_id, hs_run_id, start_ms, cached=True, success=True)
        return pack.to_dict()

    try:
        if backend == "auto":
            attempted: list[Dict[str, Any]] = []
            selected_pack: Optional[EvidencePack] = None

            def _record_attempt(p: EvidencePack) -> None:
                attempted.append(_summarize_attempt(p))

            for backend_name, module_name, func_name in (
                ("gemini", "gemini_grounding", "fetch_via_gemini"),
                ("openai", "openai_web_search", "fetch_via_openai_web_search"),
                ("claude", "claude_web_search", "fetch_via_claude_web_search"),
            ):
                module, import_err = _load_backend_module(module_name)
                if module is None:
                    candidate = EvidencePack(query=query, recency_days=recency_days, backend=backend_name)
                    candidate.error = {"type": "missing_dependency", "message": import_err or f"backend {backend_name} not available"}
                    _record_attempt(candidate)
                    if selected_pack is None:
                        selected_pack = candidate
                    continue

                backend_fn = getattr(module, func_name)
                backend_kwargs = {
                    "recency_days": recency_days,
                    "include_structural": include_structural,
                    "timeout_sec": timeout_sec,
                    "max_results": max_results,
                }
                if backend_name == "gemini":
                    backend_kwargs["model_id"] = model_id
                candidate = backend_fn(query, **backend_kwargs)
                candidate.grounded = bool(candidate.sources)
                _record_attempt(candidate)
                if candidate.sources:
                    selected_pack = candidate
                    break
                if not selected_pack:
                    selected_pack = candidate

            pack = selected_pack

            if pack and not pack.sources and fallback_backend:
                fallback_pack: Optional[EvidencePack] = None
                if fallback_backend == "exa":
                    fallback_pack = _fetch_via_exa(
                        query,
                        recency_days=recency_days,
                        timeout_sec=timeout_sec,
                        max_results=max_results,
                    )
                elif fallback_backend == "perplexity":
                    fallback_pack = _fetch_via_perplexity(
                        query,
                        recency_days=recency_days,
                        timeout_sec=timeout_sec,
                        max_results=max_results,
                    )
                else:
                    fallback_pack = EvidencePack(query=query, recency_days=recency_days, backend=fallback_backend)
                    fallback_pack.error = {"type": "unsupported_backend", "message": f"backend={fallback_backend} not implemented"}

                if fallback_pack:
                    fallback_pack.grounded = bool(fallback_pack.sources)
                    _record_attempt(fallback_pack)
                    if fallback_pack.sources:
                        pack = fallback_pack
                    else:
                        pack = fallback_pack if pack is None else pack

            if pack is None:
                pack = EvidencePack(query=query, recency_days=recency_days, backend="unknown")
                pack.error = {"type": "no_backend_available", "message": "no backend attempts were made"}

            pack.debug = {
                **(pack.debug or {}),
                "attempted_backends": attempted,
                "selected_backend": pack.backend,
                "n_verified_sources": len(pack.sources),
                "auto_attempts": attempted,
            }
            if not pack.sources and not pack.error:
                pack.error = {"type": "grounding_missing", "message": "no verified sources from any backend"}
        elif backend == "gemini":
            module, import_err = _load_backend_module("gemini_grounding")
            if module is None:
                pack = EvidencePack(query=query, recency_days=recency_days, backend=backend)
                pack.error = {"type": "missing_dependency", "message": import_err or "gemini backend not available"}
            else:
                pack = module.fetch_via_gemini(
                    query,
                    recency_days=recency_days,
                    include_structural=include_structural,
                    timeout_sec=timeout_sec,
                    max_results=max_results,
                    model_id=model_id,
                )
        elif backend == "openai":
            module, import_err = _load_backend_module("openai_web_search")
            if module is None:
                pack = EvidencePack(query=query, recency_days=recency_days, backend=backend)
                pack.error = {"type": "missing_dependency", "message": import_err or "openai backend not available"}
            else:
                pack = module.fetch_via_openai_web_search(
                    query,
                    recency_days=recency_days,
                    include_structural=include_structural,
                    timeout_sec=timeout_sec,
                    max_results=max_results,
                )
        elif backend == "claude":
            module, import_err = _load_backend_module("claude_web_search")
            if module is None:
                pack = EvidencePack(query=query, recency_days=recency_days, backend=backend)
                pack.error = {"type": "missing_dependency", "message": import_err or "claude backend not available"}
            else:
                pack = module.fetch_via_claude_web_search(
                    query,
                    recency_days=recency_days,
                    include_structural=include_structural,
                    timeout_sec=timeout_sec,
                    max_results=max_results,
                )
        elif backend == "exa":
            pack = _fetch_via_exa(
                query,
                recency_days=recency_days,
                timeout_sec=timeout_sec,
                max_results=max_results,
            )
        elif backend == "perplexity":
            pack = _fetch_via_perplexity(
                query,
                recency_days=recency_days,
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

    pack.grounded = bool(pack.sources)
    if not isinstance(pack.debug, dict):
        pack.debug = {}
    pack.debug["attempted_backends"] = _normalize_attempted_backends(pack.debug.get("attempted_backends"), pack.backend, pack)
    pack.debug.setdefault("selected_backend", pack.backend)
    pack.debug.setdefault("n_verified_sources", len(pack.sources))

    try:
        usage = pack.debug.get("usage") if isinstance(pack.debug, dict) else {}
        cost_usd = 0.0
        if isinstance(usage, dict):
            cost_usd = float(usage.get("cost_usd") or usage.get("total_cost_usd") or 0.0)
        guard.record_actual(cost_usd=cost_usd)
    except Exception:
        pass

    cache_set(cache_key, pack.to_dict())
    _log_web_research(pack, purpose, run_id, question_id, hs_run_id, start_ms, cached=False, success=pack.error is None)
    return pack.to_dict()


def _log_web_research(
    pack: EvidencePack,
    purpose: str,
    run_id: Optional[str],
    question_id: Optional[str],
    hs_run_id: Optional[str],
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
            "n_verified_sources": len(pack.sources),
            "cached": cached,
            "elapsed_ms": elapsed_ms,
        }
        if isinstance(pack.debug, dict):
            if "attempted_backends" in pack.debug:
                usage_json["attempted_backends"] = pack.debug.get("attempted_backends")
            if "selected_backend" in pack.debug:
                usage_json["selected_backend"] = pack.debug.get("selected_backend")
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
                    "total_cost_usd": usage.get("total_cost_usd", usage.get("cost_usd", 0.0)),
                })

        con = connect(read_only=False)
        ensure_schema(con)
        provider = ""
        model_id = pack.backend or "unknown"
        model_name = pack.backend or "unknown"
        if isinstance(pack.debug, dict):
            provider = str(pack.debug.get("provider") or provider)
            model_id = str(pack.debug.get("selected_model_id") or pack.debug.get("model_id") or model_id)
        if not provider:
            provider = "google" if pack.backend == "gemini" else pack.backend or "unknown"
        if os.getenv("PYTHIA_RETRIEVER_ENABLED", "0") == "1" and pack.backend == "gemini":
            model_id = model_id or os.getenv("PYTHIA_RETRIEVER_MODEL_ID") or "gemini-2.5-flash-lite"
        if pack.backend == "gemini":
            model_name = "Gemini Grounding"
        if "cost_usd" not in usage_json or float(usage_json.get("cost_usd") or 0.0) == 0.0:
            usage_json["cost_usd"] = estimate_cost_usd(model_id, usage_json)
        log_web_research_call(
            con,
            component="web_research",
            phase=_resolve_phase(purpose),
            provider=provider,
            model_name=model_name,
            model_id=model_id,
            run_id=run_id,
            hs_run_id=hs_run_id,
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
