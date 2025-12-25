# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
import os
from datetime import datetime
import re
from typing import Any, Dict, List, Tuple, Set

import requests

from pythia.web_research.types import EvidencePack, EvidenceSource


def _extract_sources_from_grounding(gm: Dict[str, Any]) -> List[EvidenceSource]:
    sources: List[EvidenceSource] = []
    seen_urls: Set[str] = set()
    chunks = gm.get("groundingChunks") or []
    for chunk in chunks:
        web = chunk.get("web") or {}
        url = web.get("uri") or ""
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = web.get("title") or url
        sources.append(EvidenceSource(title=title, url=url, publisher="", date=None, summary=""))

    return sources


def parse_gemini_grounding_response(resp: Dict[str, Any]) -> Tuple[List[EvidenceSource], bool, Dict[str, Any]]:
    """Parse a Gemini response dict for grounding metadata and sources."""

    debug: Dict[str, Any] = {
        "response_has_candidates": False,
        "grounding_metadata_present": False,
        "groundingSupports_count": 0,
        "groundingChunks_count": 0,
    }
    candidates = resp.get("candidates") or []
    if not candidates:
        return [], False, debug
    debug["response_has_candidates"] = True

    gm = candidates[0].get("groundingMetadata") or {}
    debug["grounding_metadata_present"] = bool(gm)
    debug["webSearchQueries"] = gm.get("webSearchQueries") or []
    debug["groundingSupports_count"] = len(gm.get("groundingSupports", []))
    debug["groundingChunks_count"] = len(gm.get("groundingChunks", []))

    sources = _extract_sources_from_grounding(gm) if gm else []
    grounded = bool(gm and sources)
    return sources, grounded, debug


def fetch_via_gemini(
    query: str,
    *,
    recency_days: int,
    include_structural: bool,
    timeout_sec: int,
    max_results: int,
    model_id: str | None = None,
) -> EvidencePack:
    """Fetch web research evidence via Gemini with Google Search grounding."""

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    env_model_id = (model_id or os.getenv("PYTHIA_WEB_RESEARCH_MODEL_ID") or "").strip()
    model_candidates: List[str] = []
    if env_model_id:
        model_candidates.append(env_model_id)
    for default_model in ("gemini-3-flash-preview", "gemini-3-pro-preview"):
        if default_model not in model_candidates:
            model_candidates.append(default_model)

    pack = EvidencePack(query=query, recency_days=recency_days, backend="gemini")

    if not api_key:
        pack.error = {"type": "missing_api_key", "message": "GEMINI_API_KEY / GOOGLE_API_KEY not set"}
        pack.debug = {"error": "missing_api_key"}
        return pack

    prompt = (
        "You are a research assistant using Google Search grounding. "
        "Return strictly JSON with this shape:\n"
        "{\n  \"structural_context\": \"max 8 lines\",\n"
        "  \"recent_signals\": [\"<=8 bullets, last 120 days\"],\n  \"notes\": \"optional\"\n}"
        "\n- Focus on authoritative, recent sources (last "
        f"{recency_days} days).\n- Do not include URLs in the JSON text.\n"
        f"Query: {query}"
    )

    force_search_prompt = (
        prompt
        + "\n\nYou must use Google Search for this request. If you do not search, return { \"recent_signals\": [], \"structural_context\": \"\" }."
    )

    gen_cfg = {"temperature": 0.2, "maxOutputTokens": 768}

    def _build_body(prompt_text: str) -> Dict[str, Any]:
        return {
            "contents": [{"parts": [{"text": prompt_text}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": gen_cfg,
        }

    last_errors: List[str] = []
    attempted_shapes: List[str] = []
    attempted_models: List[str] = []
    response_data: Dict[str, Any] = {}
    used_attempt = ""
    status_code = 0
    selected_model_id = ""
    provider_name = "google"
    retry_used = False
    retry_success = False

    def _post(url: str, body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        try:
            resp = requests.post(url, params={"key": api_key}, json=body, timeout=timeout_sec)
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, {"error_text": (resp.text or "")[:800]}
        except Exception as exc:  # pragma: no cover - defensive
            return 599, {"error_text": f"exception: {exc!r}"}

    for model_id in model_candidates:
        attempted_models.append(model_id)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"

        for attempt_name, body in (
            ("google_search", _build_body(prompt)),
            ("google_search_force", _build_body(force_search_prompt)),
        ):
            if attempt_name == "google_search_force":
                retry_used = True
            attempted_shapes.append(attempt_name)
            status_code, data = _post(url, body)
            response_data = data
            used_attempt = attempt_name
            sources, grounded, debug_meta = parse_gemini_grounding_response(data)
            if status_code == 200 and grounded:
                selected_model_id = model_id
                if attempt_name == "google_search_force":
                    retry_success = True
                break
            # capture short error for debugging and try next shape
            msg = ""
            if isinstance(data, dict):
                if "error" in data and isinstance(data["error"], dict):
                    msg = data["error"].get("message", "") or ""
                if not msg:
                    msg = data.get("error_text", "")
            if not msg:
                msg = f"status={status_code} grounded={grounded}"
            last_errors.append(f"{model_id}/{attempt_name}: {msg[:200]}")
            # Only retry when first attempt succeeded but lacked grounding metadata
            if attempt_name == "google_search" and status_code == 200 and not grounded:
                continue
            # Otherwise move to next model
            break

        if status_code == 200 and grounded:
            break

    sources, grounded, debug_meta = parse_gemini_grounding_response(response_data)

    text_blob = ""
    extra_debug: Dict[str, Any] = {}
    for cand in (response_data.get("candidates") or []):
        content = cand.get("content") or {}
        for part in content.get("parts") or []:
            t = part.get("text")
            if t:
                text_blob = str(t).strip()
                break
        if text_blob:
            break

    structural_context = ""
    recent_signals: List[str] = []
    unverified_sources: List[EvidenceSource] = []
    if text_blob:
        try:
            parsed = json.loads(text_blob)
            if include_structural:
                structural_context = str(parsed.get("structural_context", "") or "")
            signals_raw = parsed.get("recent_signals") or []
            if isinstance(signals_raw, list):
                recent_signals = [str(x) for x in signals_raw if str(x).strip()][:8]
            if structural_context:
                lines = structural_context.splitlines()
                structural_context = "\n".join(lines[:8]).strip()
            if include_structural:
                extra_debug["notes"] = parsed.get("notes")
        except Exception:
            extra_debug["raw_text"] = text_blob
        if not grounded:
            urls = re.findall(r"https?://[^\s\]\"')>]+", text_blob)
            seen: set[str] = set()
            for url in urls:
                if url in seen:
                    continue
                seen.add(url)
                unverified_sources.append(EvidenceSource(title=url, url=url))
            if unverified_sources:
                extra_debug["unverified_url_count"] = len(unverified_sources)

    usage_meta = response_data.get("usageMetadata") or {}
    usage = {
        "prompt_tokens": int(usage_meta.get("promptTokenCount", 0) or 0),
        "completion_tokens": int(usage_meta.get("candidatesTokenCount", 0) or 0),
        "total_tokens": int(usage_meta.get("totalTokenCount", 0) or 0),
    }
    if grounded and usage["total_tokens"] == 0:
        approx_total = max(len(prompt.split()) + len(text_blob.split()), 1)
        usage["prompt_tokens"] = usage["prompt_tokens"] or max(len(prompt.split()), 1)
        usage["completion_tokens"] = usage["completion_tokens"] or max(len(text_blob.split()), 1)
        usage["total_tokens"] = approx_total

    pack.sources = sources
    pack.grounded = grounded
    provider_error_message = None
    if isinstance(response_data, dict):
        err_obj = response_data.get("error")
        if isinstance(err_obj, dict):
            provider_error_message = err_obj.get("message")
        if not provider_error_message:
            provider_error_message = response_data.get("error_text")

    pack.structural_context = structural_context if include_structural else ""
    pack.recent_signals = recent_signals
    pack.unverified_sources = unverified_sources
    pack.debug = {
        **debug_meta,
        **extra_debug,
        "attempted_shapes": attempted_shapes,
        "attempted_models": attempted_models,
        "selected_model_id": selected_model_id or (attempted_models[-1] if attempted_models else ""),
        "used_attempt": used_attempt,
        "status_code": status_code,
        "usage": usage,
        "model_id": selected_model_id or (attempted_models[-1] if attempted_models else ""),
        "provider": provider_name,
        "max_results": max_results,
        "fetched_at": datetime.utcnow().isoformat(),
        "retry_used": retry_used,
        "retry_success": retry_success,
    }
    if last_errors:
        pack.debug["last_errors"] = last_errors[:5]
    if provider_error_message:
        pack.debug["provider_error_message"] = provider_error_message

    if not grounded:
        msg = "no grounding metadata returned" if not last_errors else "; ".join(last_errors[:3])
        error_type = "grounding_missing"
        if provider_error_message and status_code != 200:
            error_type = "provider_error"
        pack.error = {"type": error_type, "message": msg}

    return pack
