# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Set

from anthropic import Anthropic

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
    return {}


def _parse_sources_from_content(content: List[Dict[str, Any]]) -> List[EvidenceSource]:
    sources: List[EvidenceSource] = []
    seen: Set[str] = set()
    for item in content:
        if item.get("type") == "web_search_tool_result":
            for result in item.get("results") or []:
                url = result.get("url") or ""
                if not url or url in seen:
                    continue
                seen.add(url)
                title = result.get("title") or url
                sources.append(EvidenceSource(title=title, url=url, publisher="", date=result.get("page_age")))
        elif item.get("type") == "tool_result":
            name = item.get("name") or ""
            if name != "web_search":
                continue
            for result in item.get("results") or []:
                url = result.get("url") or ""
                if not url or url in seen:
                    continue
                seen.add(url)
                title = result.get("title") or url
                sources.append(EvidenceSource(title=title, url=url, publisher="", date=result.get("page_age")))
    return sources


def fetch_via_claude_web_search(
    query: str,
    *,
    recency_days: int,
    include_structural: bool,
    timeout_sec: int,
    max_results: int,
) -> EvidencePack:
    """Fetch web research evidence via Claude web search tool."""

    pack = EvidencePack(query=query, recency_days=recency_days, backend="claude")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pack.error = {"type": "missing_api_key", "message": "ANTHROPIC_API_KEY not set"}
        pack.debug = {"error": "missing_api_key"}
        return pack

    model_id = (os.getenv("PYTHIA_WEB_RESEARCH_MODEL_ID") or "claude-3-7-sonnet-20250219").strip()
    if not model_id:
        model_id = "claude-3-7-sonnet-20250219"

    prompt = (
        "You are a research assistant using the Claude web_search tool. "
        "Return strictly JSON with this shape:\n"
        "{\n  \"structural_context\": \"max 8 lines\",\n"
        "  \"recent_signals\": [\"<=8 bullets, last 120 days\"],\n  \"notes\": \"optional\"\n}"
        "\n- Focus on authoritative, recent sources (last "
        f"{recency_days} days).\n- Do not include URLs in the JSON text.\n"
        f"Query: {query}"
    )

    client = Anthropic(api_key=api_key, timeout=timeout_sec)
    data: Dict[str, Any] = {}
    provider_error_message = None

    try:
        resp = client.messages.create(
            model=model_id,
            max_output_tokens=800,
            system=prompt,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
            tool_choice={"type": "tool", "name": "web_search"},
        )
        data = _response_to_dict(resp)
    except Exception as exc:  # pragma: no cover - network failures
        provider_error_message = str(exc)

    content = data.get("content") or []
    sources = _parse_sources_from_content(content)
    grounded = bool(sources)

    structural_context = ""
    recent_signals: List[str] = []
    extra_debug: Dict[str, Any] = {}
    for item in content:
        if item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            if block.get("type") != "text":
                continue
            text = block.get("text") or ""
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
                break
            except Exception:
                extra_debug["raw_text"] = text
                break

    usage_raw = data.get("usage") or {}
    usage = {
        "prompt_tokens": int(usage_raw.get("input_tokens", 0) or 0),
        "completion_tokens": int(usage_raw.get("output_tokens", 0) or 0),
        "total_tokens": int(usage_raw.get("input_tokens", 0) or 0) + int(usage_raw.get("output_tokens", 0) or 0),
    }

    pack.sources = sources
    pack.grounded = grounded
    pack.structural_context = structural_context if include_structural else ""
    pack.recent_signals = recent_signals

    pack.debug = {
        **extra_debug,
        "provider": "anthropic",
        "model_id": model_id,
        "selected_model_id": model_id,
        "max_results": max_results,
        "usage": usage,
        "fetched_at": datetime.utcnow().isoformat(),
    }
    if provider_error_message:
        pack.debug["provider_error_message"] = provider_error_message
        pack.error = {"type": "web_search_unavailable", "message": provider_error_message}
    elif not grounded:
        pack.error = {"type": "grounding_missing", "message": "no web_search_tool_result sources returned"}

    return pack
