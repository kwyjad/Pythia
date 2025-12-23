# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple, Set

from openai import OpenAI

from pythia.web_research.types import EvidencePack, EvidenceSource


def _response_to_dict(resp: Any) -> Dict[str, Any]:
    if isinstance(resp, dict):
        return resp
    for attr in ("model_dump", "to_dict"):
        fn = getattr(resp, attr, None)
        if callable(fn):
            try:
                return fn()  # type: ignore[call-arg]
            except Exception:
                pass
    if hasattr(resp, "json") and callable(resp.json):
        try:
            return json.loads(resp.json())  # type: ignore[arg-type]
        except Exception:
            return {}
    return {}


def _parse_sources_from_output(output: List[Dict[str, Any]]) -> List[EvidenceSource]:
    sources: List[EvidenceSource] = []
    seen: Set[str] = set()

    for item in output:
        if item.get("type") == "web_search_call":
            action = (item.get("web_search_call") or {}).get("action") or {}
            for src in action.get("sources") or []:
                url = src.get("url") or src.get("uri") or ""
                if not url or url in seen:
                    continue
                seen.add(url)
                title = src.get("title") or url
                summary = src.get("summary") or ""
                publisher = src.get("source") or ""
                sources.append(EvidenceSource(title=title, url=url, publisher=publisher, summary=summary))
        elif item.get("type") == "message":
            message = item.get("message") or {}
            for content in message.get("content") or []:
                if content.get("type") != "output_text":
                    continue
                for ann in content.get("annotations") or []:
                    url = ann.get("url") or ""
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    title = ann.get("title") or url
                    sources.append(EvidenceSource(title=title, url=url, publisher="", summary=""))

    return sources


def _parse_structured_text(output: List[Dict[str, Any]], include_structural: bool) -> Tuple[str, List[str], Dict[str, Any]]:
    structural_context = ""
    recent_signals: List[str] = []
    extra_debug: Dict[str, Any] = {}

    for item in output:
        if item.get("type") != "message":
            continue
        message = item.get("message") or {}
        for content in message.get("content") or []:
            if content.get("type") != "output_text":
                continue
            text = content.get("text") or ""
            if not text:
                continue
            try:
                parsed = json.loads(text)
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
                return structural_context, recent_signals, extra_debug
            except Exception:
                extra_debug["raw_text"] = text
                return structural_context, recent_signals, extra_debug
    return structural_context, recent_signals, extra_debug


def fetch_via_openai_web_search(
    query: str,
    *,
    recency_days: int,
    include_structural: bool,
    timeout_sec: int,
    max_results: int,
) -> EvidencePack:
    """Fetch web research evidence via OpenAI Responses web_search."""

    pack = EvidencePack(query=query, recency_days=recency_days, backend="openai")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pack.error = {"type": "missing_api_key", "message": "OPENAI_API_KEY not set"}
        pack.debug = {"error": "missing_api_key"}
        return pack

    model_id = (os.getenv("PYTHIA_WEB_RESEARCH_MODEL_ID") or "gpt-4.1").strip()
    fallback_model = "gpt-4.1-mini"
    if not model_id:
        model_id = "gpt-4.1"

    prompt = (
        "You are a research assistant using OpenAI web_search. "
        "Return strictly JSON with this shape:\n"
        "{\n  \"structural_context\": \"max 8 lines\",\n"
        "  \"recent_signals\": [\"<=8 bullets, last 120 days\"],\n  \"notes\": \"optional\"\n}"
        "\n- Focus on authoritative, recent sources (last "
        f"{recency_days} days).\n- Do not include URLs in the JSON text.\n"
        f"Query: {query}"
    )

    client = OpenAI(api_key=api_key)
    used_model = model_id
    data: Dict[str, Any] = {}
    provider_error_message = None

    def _create(model: str) -> Dict[str, Any]:
        resp = client.responses.create(
            model=model,
            input=prompt,
            tools=[{"type": "web_search"}],
            max_output_tokens=800,
            include=["web_search_call.action.sources"],
        )
        return _response_to_dict(resp)

    for candidate_model in (model_id, fallback_model):
        used_model = candidate_model
        try:
            data = _create(candidate_model)
            break
        except Exception as exc:  # pragma: no cover - network failures
            provider_error_message = str(exc)
            data = {}
            continue

    output = data.get("output") or []
    sources = _parse_sources_from_output(output)
    grounded = bool(sources)
    structural_context, recent_signals, extra_debug = _parse_structured_text(output, include_structural)

    usage_raw = data.get("usage") or {}
    usage = {
        "prompt_tokens": int(usage_raw.get("input_tokens", 0) or 0),
        "completion_tokens": int(usage_raw.get("output_tokens", 0) or 0),
        "total_tokens": int(usage_raw.get("total_tokens", 0) or 0),
        "web_search_requests": usage_raw.get("web_search_requests"),
    }

    pack.sources = sources
    pack.grounded = grounded
    pack.structural_context = structural_context if include_structural else ""
    pack.recent_signals = recent_signals
    pack.unverified_sources = []

    pack.debug = {
        **extra_debug,
        "provider": "openai",
        "model_id": used_model,
        "selected_model_id": used_model,
        "max_results": max_results,
        "status_code": 200 if data else 0,
        "usage": usage,
        "fetched_at": datetime.utcnow().isoformat(),
    }
    if provider_error_message:
        pack.debug["provider_error_message"] = provider_error_message

    if not grounded:
        pack.error = {"type": "grounding_missing", "message": "no web_search sources returned"}
    return pack
