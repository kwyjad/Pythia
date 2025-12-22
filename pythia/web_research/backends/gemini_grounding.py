# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import requests

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
    """Fetch web research evidence via Gemini with Google Search grounding."""

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_id = (os.getenv("PYTHIA_WEB_RESEARCH_MODEL_ID") or "gemini-3-flash-preview").strip()

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

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
    gen_cfg = {"temperature": 0.2, "maxOutputTokens": 768, "responseMimeType": "application/json"}

    contents = [{"parts": [{"text": prompt}]}]
    attempts: List[Tuple[str, Dict[str, Any]]] = [
        (
            "googleSearchRetrieval",
            {
                "contents": contents,
                "tools": [{"googleSearchRetrieval": {}}],
                "generationConfig": gen_cfg,
            },
        ),
        (
            "google_search",
            {
                "contents": contents,
                "tools": [{"google_search": {}}],
                "generationConfig": gen_cfg,
            },
        ),
    ]

    last_errors: List[str] = []
    attempted_shapes: List[str] = []
    response_data: Dict[str, Any] = {}
    used_attempt = ""
    status_code = 0

    def _post(body: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        try:
            resp = requests.post(url, params={"key": api_key}, json=body, timeout=timeout_sec)
            try:
                return resp.status_code, resp.json()
            except Exception:
                return resp.status_code, {"error_text": (resp.text or "")[:800]}
        except Exception as exc:  # pragma: no cover - defensive
            return 599, {"error_text": f"exception: {exc!r}"}

    for attempt_name, body in attempts:
        attempted_shapes.append(attempt_name)
        status_code, data = _post(body)
        response_data = data
        used_attempt = attempt_name
        sources, grounded, debug_meta = parse_gemini_grounding_response(data)
        if status_code == 200 and grounded:
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
        last_errors.append(f"{attempt_name}: {msg[:200]}")
        if status_code == 200 and not grounded:
            # Try next attempt shape to elicit grounding metadata
            continue
        if status_code != 200:
            continue

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
    pack.structural_context = structural_context if include_structural else ""
    pack.recent_signals = recent_signals if include_structural else []
    pack.debug = {
        **debug_meta,
        **extra_debug,
        "attempted_shapes": attempted_shapes,
        "used_attempt": used_attempt,
        "status_code": status_code,
        "usage": usage,
        "model_id": model_id,
        "max_results": max_results,
        "fetched_at": datetime.utcnow().isoformat(),
    }

    if not grounded:
        msg = "no grounding metadata returned" if not last_errors else "; ".join(last_errors[:3])
        pack.error = {"type": "grounding_missing", "message": msg}

    if response_data.get("error"):
        err_msg = response_data.get("error", {}).get("message")
        if err_msg:
            pack.error = {"type": "provider_error", "message": err_msg}

    return pack
