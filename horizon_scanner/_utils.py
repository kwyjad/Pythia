# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Shared helpers for the Horizon Scanner triage and regime-change modules."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Model resolution
# ---------------------------------------------------------------------------

def resolve_hs_model() -> str:
    """Return the Gemini model ID for Horizon Scanner calls."""
    from forecaster.providers import GEMINI_MODEL_ID

    model_id = (GEMINI_MODEL_ID or "").strip()
    if model_id:
        return model_id
    return os.getenv("HS_MODEL_ID", "gemini-3-flash-preview")


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_response(raw: str) -> dict[str, Any]:
    """Parse a JSON response, handling markdown fences and partial JSON."""

    s = (raw or "").strip()
    if not s:
        raise ValueError("empty response")

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    fenced_json = re.search(r"```json\s*(.*?)\s*```", s, flags=re.S | re.I)
    if fenced_json:
        return json.loads(fenced_json.group(1).strip())

    fenced_any = re.search(r"```\s*(.*?)\s*```", s, flags=re.S)
    if fenced_any:
        return json.loads(fenced_any.group(1).strip())

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(s[start : end + 1])

    raise json.JSONDecodeError("could not locate JSON object", s, 0)


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------

def coerce_score_or_none(raw: Any) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN check
        return None
    return value


def coerce_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    cleaned: list[str] = []
    for item in raw:
        value = str(item).strip()
        if value:
            cleaned.append(value)
    return cleaned


# ---------------------------------------------------------------------------
# Merge helpers
# ---------------------------------------------------------------------------

def merge_unique(values_a: list[str], values_b: list[str], limit: int = 6) -> list[str]:
    merged = sorted({value for value in values_a + values_b if value})
    return merged[:limit]


def merge_unique_signals(
    signals_a: list[dict[str, Any]],
    signals_b: list[dict[str, Any]],
    limit: int = 6,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in signals_a + signals_b:
        if not isinstance(entry, dict):
            continue
        key = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        merged.append(entry)
        if len(merged) >= limit:
            break
    merged.sort(key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    return merged[:limit]


# ---------------------------------------------------------------------------
# Error / status helpers
# ---------------------------------------------------------------------------

def short_error(raw: str | None, limit: int = 200) -> str:
    if not raw:
        return ""
    text = str(raw).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def status_from_error(error_text: str | None) -> str:
    if not error_text:
        return "ok"
    lowered = str(error_text).lower()
    if "cooldown active" in lowered:
        return "cooldown"
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "parse failed" in lowered or "json" in lowered:
        return "parse_error"
    if "empty response" in lowered:
        return "empty_response"
    return "provider_error"


# ---------------------------------------------------------------------------
# JSON repair via fallback model
# ---------------------------------------------------------------------------

async def repair_json_response(
    raw_text: str,
    *,
    run_id: str | None = None,
    fallback_specs: list,
    component: str = "HorizonScanner",
    prompt_key: str = "hs.json_repair",
) -> tuple[dict[str, Any], dict[str, Any], str, Any]:
    """Attempt to repair malformed JSON using a fallback LLM.

    Returns (parsed_obj, usage, error_str, model_spec_or_None).
    """
    from forecaster import providers as _providers

    if not raw_text:
        return {}, {}, "empty response", None
    repair_prompt = (
        "Convert the following into valid JSON ONLY. No prose. Preserve keys/values. "
        "Output a single JSON object.\n\n"
        f"{raw_text}"
    )
    for spec in fallback_specs:
        text, usage, error = await _providers.call_chat_ms(
            spec,
            repair_prompt,
            temperature=0.0,
            prompt_key=prompt_key,
            prompt_version="1.0.0",
            component=component,
            run_id=run_id,
        )
        if error:
            continue
        try:
            obj = parse_json_response(text)
        except Exception as exc:  # noqa: BLE001
            return {}, usage or {}, f"repair parse failed: {type(exc).__name__}: {exc}", spec
        return obj, usage or {}, "", spec
    return {}, {}, "repair failed", None
