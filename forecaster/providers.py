# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations
"""Forecaster provider adapters.

This module exposes a unified async interface for calling the LLM providers used by
Pythia's forecaster. Providers and model choices are controlled through
``pythia/config.yaml`` so operators can toggle models without code changes.

Each provider call is logged to the ``llm_calls`` table (best-effort) with token
usage, latency, and estimated cost so we can monitor spend across runs.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import duckdb
import httpx
import requests

from pythia.config import load as load_cfg
from pythia.db.util import write_llm_call
from pythia.llm_profiles import get_current_models, get_current_profile


# Default timeouts (seconds); can be overridden via env
GPT5_CALL_TIMEOUT_SEC = float(os.getenv("GPT5_CALL_TIMEOUT_SEC", "300"))
GEMINI_CALL_TIMEOUT_SEC = float(os.getenv("GEMINI_CALL_TIMEOUT_SEC", "300"))
GROK_CALL_TIMEOUT_SEC = float(os.getenv("GROK_CALL_TIMEOUT_SEC", "300"))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ProviderResult:
    text: str
    usage: Dict[str, int]
    cost_usd: float
    model_id: str
    error: Optional[str] = None


@dataclass
class ModelSpec:
    name: str
    provider: str  # "openai" | "anthropic" | "google" | "xai"
    model_id: str
    weight: float = 1.0
    active: bool = True
    purpose: Optional[str] = None


llm_semaphore = asyncio.Semaphore(int(os.getenv("LLM_MAX_CONCURRENCY", "4")))


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_DEFAULT_PROVIDER_CONFIG: Dict[str, Dict[str, Any]] = {
    "openai": {
        "enabled": True,
        "model": "",
        "env_key": "OPENAI_API_KEY",
        "display_name": "OpenAI",
    },
    "anthropic": {
        "enabled": True,
        "model": "",
        "env_key": "ANTHROPIC_API_KEY",
        "display_name": "Claude",
    },
    "google": {
        "enabled": True,
        "model": "",
        "env_key": "GEMINI_API_KEY",
        "display_name": "Gemini",
    },
    "xai": {
        "enabled": True,
        "model": "",
        "env_key": "XAI_API_KEY",
        "display_name": "Grok",
    },
}

_cfg = load_cfg()
_app_cfg = _cfg.get("app", {}) if isinstance(_cfg, dict) else {}
_forecaster_cfg = _cfg.get("forecaster", {}) if isinstance(_cfg, dict) else {}
_provider_overrides = _forecaster_cfg.get("providers", {}) if isinstance(_forecaster_cfg, dict) else {}


def _merge_provider_config() -> Dict[str, Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    # start with defaults so we always have sane values
    for key, defaults in _DEFAULT_PROVIDER_CONFIG.items():
        merged[key] = dict(defaults)
    # apply overrides from config.yaml
    for key, override in _provider_overrides.items():
        if not isinstance(override, dict):
            continue
        base = merged.setdefault(key, {})
        for ok, ov in override.items():
            base[ok] = ov

    profile_models: Dict[str, str] = {}
    try:
        profile_models = get_current_models()
    except Exception:
        profile_models = {}

    for provider_name, entry in merged.items():
        model = str(entry.get("model") or "").strip()
        if not model and profile_models:
            profile_model = profile_models.get(provider_name)
            if profile_model:
                entry["model"] = profile_model
    return merged


_provider_config = _merge_provider_config()


def _provider_display_name(provider: str, model_id: str, cfg: Dict[str, Any]) -> str:
    explicit = cfg.get("display_name")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    base_names = {
        "openai": "OpenAI",
        "anthropic": "Claude",
        "google": "Gemini",
        "xai": "Grok",
    }
    base = base_names.get(provider, provider.title())
    if not model_id:
        return base
    return f"{base}-{model_id.replace('/', '-')}"


def _resolve_timeout(env_name: str, fallback: float | int | None, default: float) -> float:
    candidate = os.getenv(env_name)
    if candidate:
        try:
            return max(1.0, float(candidate))
        except Exception:
            pass
    if fallback is not None:
        try:
            return max(1.0, float(fallback))
        except Exception:
            pass
    return default


_DB_URL = str(_app_cfg.get("db_url", "")).strip()


def _duckdb_path(db_url: str) -> Optional[str]:
    if not db_url:
        return None
    if db_url.startswith("duckdb:///"):
        return db_url[len("duckdb:///"):]
    return db_url


_DB_PATH = _duckdb_path(_DB_URL)

if _DB_PATH and _DB_PATH not in {":memory:", ""}:
    db_dir = os.path.dirname(_DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)


# prompt metadata used for logging; fall back to a sensible default
_FORECASTER_PROMPT_VERSION = str(_forecaster_cfg.get("prompt_version", "1.0.0"))


_PROVIDER_STATES: Dict[str, Dict[str, Any]] = {}
_MODEL_SPECS: List[ModelSpec] = []

for provider_name, cfg_entry in _provider_config.items():
    env_key = str(cfg_entry.get("env_key", "")).strip()
    api_key = os.getenv(env_key, "").strip() if env_key else ""
    model_id = str(cfg_entry.get("model", "")).strip()
    enabled_flag = bool(cfg_entry.get("enabled", False))
    weight = float(cfg_entry.get("weight", 1.0) or 1.0)
    display_name = _provider_display_name(provider_name, model_id, cfg_entry)
    active = bool(enabled_flag and api_key and model_id)

    _PROVIDER_STATES[provider_name] = {
        "api_key": api_key,
        "model": model_id,
        "env_key": env_key,
        "enabled": enabled_flag,
        "active": active,
        "display_name": display_name,
        "weight": weight,
    }

    _MODEL_SPECS.append(
        ModelSpec(
            name=display_name,
            provider=provider_name,
            model_id=model_id,
            weight=weight,
            active=active,
        )
    )


KNOWN_MODELS: List[str] = [spec.name for spec in _MODEL_SPECS]
DEFAULT_ENSEMBLE: List[ModelSpec] = [spec for spec in _MODEL_SPECS if spec.active]


def summarize_model_specs(specs: List[ModelSpec]) -> str:
    """Return a stable, non-secret summary of model specs."""

    parts: List[str] = []
    for ms in specs:
        parts.append(
            f"{ms.provider}:{ms.model_id}"
            f"({'active' if ms.active else 'inactive'},w={getattr(ms, 'weight', 1.0)})"
        )
    return ", ".join(parts)


def default_ensemble_summary() -> str:
    """Summarize the current default ensemble without secrets."""

    return summarize_model_specs(DEFAULT_ENSEMBLE)


# backwards-compatible aliases reused elsewhere in the forecaster package
_OPENAI_STATE = _PROVIDER_STATES.get("openai", {})
OPENAI_MODEL_ID = _OPENAI_STATE.get("model", "")
_GEMINI_STATE = _PROVIDER_STATES.get("google", {})
GEMINI_MODEL_ID = _GEMINI_STATE.get("model", "")
_XAI_STATE = _PROVIDER_STATES.get("xai", {})
GROK_MODEL_ID = _XAI_STATE.get("model", "")

_OPENAI_API_KEY = _OPENAI_STATE.get("api_key", "")
_ANTHROPIC_API_KEY = _PROVIDER_STATES.get("anthropic", {}).get("api_key", "")
_GEMINI_API_KEY = _GEMINI_STATE.get("api_key", "")
_XAI_API_KEY = _XAI_STATE.get("api_key", "")

_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
_XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").strip()

_OPENAI_TIMEOUT = _resolve_timeout("OPENAI_CALL_TIMEOUT_SEC", GPT5_CALL_TIMEOUT_SEC, 60.0)
_ANTHROPIC_TIMEOUT = _resolve_timeout("ANTHROPIC_CALL_TIMEOUT_SEC", GPT5_CALL_TIMEOUT_SEC, 60.0)
_GEMINI_TIMEOUT = _resolve_timeout("GEMINI_CALL_TIMEOUT_SEC", GEMINI_CALL_TIMEOUT_SEC, 60.0)
_XAI_TIMEOUT = _resolve_timeout("XAI_CALL_TIMEOUT_SEC", GROK_CALL_TIMEOUT_SEC, 60.0)

_ANTHROPIC_VERSION = os.getenv("ANTHROPIC_API_VERSION", "2023-06-01")
_ANTHROPIC_MAX_OUTPUT = int(os.getenv("ANTHROPIC_MAX_OUTPUT_TOKENS", "2048") or 2048)


_http_client: Optional[httpx.AsyncClient] = None


def _get_or_client() -> httpx.AsyncClient:
    """Return a shared async HTTP client for provider calls."""

    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=30.0)
    return _http_client


# ---------------------------------------------------------------------------
# Usage / cost helpers
# ---------------------------------------------------------------------------

# Cost per 1,000 tokens for known models (USD). Keys should match ModelSpec.model_id
# entries so we can estimate costs directly from provider usage metadata.
MODEL_PRICES_PER_1K: Dict[str, tuple[float, float]] = {
    # Budget / testing models
    "gpt-5-nano": (0.00005, 0.00040),
    "openai/gpt-5-nano": (0.00005, 0.00040),
    "gemini-2.5-flash-lite": (0.00005, 0.00015),
    "google/gemini-2.5-flash-lite": (0.00005, 0.00015),
    "claude-haiku-4-5-20251001": (0.00100, 0.00500),
    "anthropic/claude-haiku-4-5-20251001": (0.00100, 0.00500),
    "grok-4-1-fast-reasoning": (0.00030, 0.00050),
    "xai/grok-4-1-fast-reasoning": (0.00030, 0.00050),

    # Production / frontier models
    "gpt-5.1": (0.00125, 0.01000),
    "openai/gpt-5.1": (0.00125, 0.01000),
    "gemini-3-pro-preview": (0.00200, 0.01200),
    "google/gemini-3-pro-preview": (0.00200, 0.01200),
    "claude-opus-4-5-20251101": (0.00500, 0.02500),
    "anthropic/claude-opus-4-5-20251101": (0.00500, 0.02500),
    "grok-4-0709": (0.00300, 0.01500),
    "xai/grok-4-0709": (0.00300, 0.01500),
}

_MODEL_PRICES: Optional[Dict[str, Dict[str, float]]] = None


def _load_model_prices() -> Dict[str, Dict[str, float]]:
    global _MODEL_PRICES
    if _MODEL_PRICES is not None:
        return _MODEL_PRICES
    data = os.getenv("MODEL_COSTS_JSON", "").strip()
    if not data:
        _MODEL_PRICES = {}
        return _MODEL_PRICES
    try:
        _MODEL_PRICES = json.loads(data)
    except Exception:
        _MODEL_PRICES = {}
    return _MODEL_PRICES


def usage_to_dict(usage_obj: Any) -> Dict[str, int]:
    base = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if usage_obj is None:
        return base
    try:
        prompt_tokens = getattr(usage_obj, "prompt_tokens", None)
        completion_tokens = getattr(usage_obj, "completion_tokens", None)
        total_tokens = getattr(usage_obj, "total_tokens", None)
        if isinstance(usage_obj, dict):
            prompt_tokens = usage_obj.get("prompt_tokens", usage_obj.get("input_tokens"))
            completion_tokens = usage_obj.get("completion_tokens", usage_obj.get("output_tokens"))
            total_tokens = usage_obj.get("total_tokens")
        base["prompt_tokens"] = int(prompt_tokens or 0)
        base["completion_tokens"] = int(completion_tokens or 0)
        if total_tokens is None:
            total_tokens = base["prompt_tokens"] + base["completion_tokens"]
        base["total_tokens"] = int(total_tokens or 0)
    except Exception:
        return base
    return base


def estimate_cost_usd(model_id: str, usage: Dict[str, int]) -> float:
    if not usage or not isinstance(usage, dict):
        return 0.0

    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)

    def _resolve_price_tuple(mid: str) -> Optional[tuple[float, float]]:
        if not mid:
            return None
        original = str(mid).strip()
        if not original:
            return None
        normalized = original.lower()
        prices: Optional[tuple[float, float]] = MODEL_PRICES_PER_1K.get(normalized)
        if not prices:
            alt_ids = [normalized.replace("/", "-"), normalized.split("/", 1)[-1]]
            for alt in alt_ids:
                if alt in MODEL_PRICES_PER_1K:
                    prices = MODEL_PRICES_PER_1K[alt]
                    break
        if prices:
            return float(prices[0]), float(prices[1])

        # Fallback to JSON overrides if provided via MODEL_COSTS_JSON
        dynamic_prices = _load_model_prices()
        price_entry = (
            dynamic_prices.get(normalized)
            or dynamic_prices.get(original)
            or dynamic_prices.get(normalized.replace("/", "-"))
            or dynamic_prices.get(normalized.split("/", 1)[-1])
            or {}
        )
        try:
            prompt_rate = float(price_entry.get("prompt", 0.0))
            completion_rate = float(price_entry.get("completion", 0.0))
            return prompt_rate, completion_rate
        except Exception:
            return None

    price_tuple = _resolve_price_tuple(model_id)
    if not price_tuple:
        return 0.0

    input_cost_per_1k, output_cost_per_1k = price_tuple

    if prompt_tokens or completion_tokens:
        input_cost = (prompt_tokens / 1000.0) * input_cost_per_1k
        output_cost = (completion_tokens / 1000.0) * output_cost_per_1k
        return float(input_cost + output_cost)

    return float((total_tokens / 1000.0) * input_cost_per_1k)


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------


def call_openai(prompt: str, model: str, temperature: float) -> ProviderResult:
    if not _OPENAI_API_KEY:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error="missing OPENAI_API_KEY")
    try:
        resp = requests.post(
            f"{_OPENAI_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {_OPENAI_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": float(temperature),
            },
            timeout=_OPENAI_TIMEOUT,
        )
    except Exception as exc:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error=f"OpenAI error: {exc}")

    try:
        payload = resp.json()
    except Exception:
        payload = {}

    if not resp.ok:
        message = ""
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                message = str(err.get("message", ""))
            elif isinstance(err, str):
                message = err
        if not message:
            message = resp.text[:400]
        return ProviderResult("", usage_to_dict(None), 0.0, model, error=f"OpenAI HTTP {resp.status_code}: {message}")

    text = ""
    if isinstance(payload, dict):
        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            if isinstance(message, dict):
                text = str(message.get("content", "")).strip()
    usage = usage_to_dict(payload.get("usage") if isinstance(payload, dict) else {})
    return ProviderResult(text=text, usage=usage, cost_usd=0.0, model_id=model)


def call_anthropic(prompt: str, model: str, temperature: float) -> ProviderResult:
    if not _ANTHROPIC_API_KEY:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error="missing ANTHROPIC_API_KEY")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": _ANTHROPIC_API_KEY,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": _ANTHROPIC_MAX_OUTPUT,
        "temperature": float(temperature),
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=_ANTHROPIC_TIMEOUT)
    except Exception as exc:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error=f"Anthropic request error: {exc}")

    try:
        payload = resp.json()
    except Exception:
        payload = {}

    if not resp.ok:
        message = ""
        if isinstance(payload, dict):
            message = payload.get("error", {}).get("message") if isinstance(payload.get("error"), dict) else payload.get("error")
            if not isinstance(message, str):
                message = ""
        if not message:
            message = resp.text[:400]
        return ProviderResult("", usage_to_dict(None), 0.0, model, error=f"Anthropic HTTP {resp.status_code}: {message}")

    text = ""
    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, list) and content:
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
            text = "".join(parts).strip()
    usage_raw = {}
    if isinstance(payload, dict):
        usage_raw = payload.get("usage") or {}
    usage = usage_to_dict({
        "prompt_tokens": usage_raw.get("input_tokens", 0),
        "completion_tokens": usage_raw.get("output_tokens", 0),
        "total_tokens": usage_raw.get("input_tokens", 0) + usage_raw.get("output_tokens", 0),
    })
    return ProviderResult(text=text, usage=usage, cost_usd=0.0, model_id=model)


def call_google(prompt: str, model: str, temperature: float) -> ProviderResult:
    if not _GEMINI_API_KEY:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error="missing GEMINI_API_KEY")
    api_model = model.split("/", 1)[-1] if "/" in model else model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{api_model}:generateContent?key={_GEMINI_API_KEY}"
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": float(temperature)},
    }
    try:
        resp = requests.post(url, json=body, timeout=_GEMINI_TIMEOUT)
    except Exception as exc:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error=f"Gemini request error: {exc}")

    try:
        payload = resp.json()
    except Exception:
        payload = {}

    if resp.status_code != 200:
        message = ""
        if isinstance(payload, dict):
            error_obj = payload.get("error", {})
            if isinstance(error_obj, dict):
                message = str(error_obj.get("message", ""))
            elif isinstance(error_obj, str):
                message = error_obj
        if not message:
            message = resp.text[:400]
        return ProviderResult("", usage_to_dict(None), 0.0, model, error=f"Gemini HTTP {resp.status_code}: {message}")

    text = ""
    if isinstance(payload, dict):
        try:
            text = payload["candidates"][0]["content"]["parts"][0].get("text", "").strip()
        except Exception:
            text = payload.get("text", "") or ""
    usage_meta = payload.get("usageMetadata") if isinstance(payload, dict) else {}
    usage = usage_to_dict({
        "prompt_tokens": usage_meta.get("promptTokenCount", 0),
        "completion_tokens": usage_meta.get("candidatesTokenCount", 0),
        "total_tokens": usage_meta.get("totalTokenCount", 0),
    })
    return ProviderResult(text=text, usage=usage, cost_usd=0.0, model_id=model)


def call_xai(prompt: str, model: str, temperature: float) -> ProviderResult:
    if not _XAI_API_KEY:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error="missing XAI_API_KEY")
    headers = {"Authorization": f"Bearer {_XAI_API_KEY}", "Content-Type": "application/json"}
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": float(temperature)}
    try:
        resp = requests.post(f"{_XAI_BASE_URL.rstrip('/')}/chat/completions", headers=headers, json=body, timeout=_XAI_TIMEOUT)
    except Exception as exc:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error=f"xAI request error: {exc}")

    try:
        payload = resp.json()
    except Exception:
        try:
            payload = json.loads(resp.text)
        except Exception:
            payload = {}

    if not resp.ok:
        message = ""
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                for key in ("message", "error"):
                    msg = err.get(key)
                    if isinstance(msg, str) and msg:
                        message = msg
                        break
            elif isinstance(err, str):
                message = err
            if not message:
                for key in ("message", "detail", "error_message"):
                    msg = payload.get(key)
                    if isinstance(msg, str) and msg:
                        message = msg
                        break
        if not message:
            message = resp.text[:400]
        return ProviderResult("", usage_to_dict(None), 0.0, model, error=f"xAI HTTP {resp.status_code}: {message}")

    text = ""
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        text = content.strip()
    usage_raw = payload.get("usage") if isinstance(payload, dict) else {}
    usage = usage_to_dict({
        "prompt_tokens": usage_raw.get("prompt_tokens", 0),
        "completion_tokens": usage_raw.get("completion_tokens", 0),
        "total_tokens": usage_raw.get("total_tokens")
        or usage_raw.get("totalTokens")
        or (usage_raw.get("prompt_tokens", 0) or 0) + (usage_raw.get("completion_tokens", 0) or 0),
    })
    return ProviderResult(text=text, usage=usage, cost_usd=0.0, model_id=model)


def _call_provider_sync(provider: str, prompt: str, model: str, temperature: float) -> ProviderResult:
    p = (provider or "").lower()
    if p == "openai":
        return call_openai(prompt, model, temperature)
    if p == "anthropic":
        return call_anthropic(prompt, model, temperature)
    if p in {"google", "gemini"}:
        return call_google(prompt, model, temperature)
    if p in {"xai", "grok"}:
        return call_xai(prompt, model, temperature)
    return ProviderResult("", usage_to_dict(None), 0.0, model, error=f"unsupported provider {provider}")


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def _log_llm_call(
    component: str,
    model: str,
    prompt_key: str,
    prompt_version: str,
    usage: Dict[str, int],
    cost: float,
    latency_ms: int,
    success: bool,
) -> None:
    if not _DB_PATH:
        return

    try:
        llm_profile = get_current_profile()
    except Exception:
        llm_profile = None

    hs_run_id = os.getenv("PYTHIA_HS_RUN_ID")
    ui_run_id = os.getenv("PYTHIA_UI_RUN_ID")
    forecaster_run_id = os.getenv("PYTHIA_FORECASTER_RUN_ID")
    conn = None
    try:
        conn = duckdb.connect(_DB_PATH, read_only=False)
        write_llm_call(
            conn,
            component=component,
            model=model,
            prompt_key=prompt_key,
            version=prompt_version,
            usage=usage,
            cost=cost,
            latency_ms=latency_ms,
            success=success,
            llm_profile=llm_profile,
            hs_run_id=hs_run_id,
            ui_run_id=ui_run_id,
            forecaster_run_id=forecaster_run_id,
        )
    except Exception:
        pass
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Public async API used by ensemble.py
# ---------------------------------------------------------------------------

async def call_chat_ms(
    ms: ModelSpec,
    prompt: str,
    temperature: float = 0.2,
    *,
    prompt_key: str = "forecaster.forecast",
    prompt_version: Optional[str] = None,
    component: str = "Forecaster",
) -> tuple[str, Dict[str, int], str]:
    """Call the configured provider for a model spec and return (text, usage, error)."""

    if not ms.active:
        return "", usage_to_dict(None), f"provider {ms.provider} inactive"

    start = time.time()
    result: Optional[ProviderResult] = None
    error: Optional[str] = None

    try:
        async with llm_semaphore:
            result = await asyncio.to_thread(_call_provider_sync, ms.provider, prompt, ms.model_id, temperature)
    except Exception as exc:  # pragma: no cover - unexpected runtime errors
        error = f"provider call error: {exc}"

    if result is None:
        result = ProviderResult("", usage_to_dict(None), 0.0, ms.model_id, error=error or "unknown error")
    elif result.error:
        error = result.error

    elapsed_ms = int((time.time() - start) * 1000)
    usage = result.usage or usage_to_dict(None)
    cost = result.cost_usd if result.cost_usd else estimate_cost_usd(ms.model_id, usage)
    _log_llm_call(
        component=component,
        model=ms.model_id,
        prompt_key=prompt_key,
        prompt_version=prompt_version or _FORECASTER_PROMPT_VERSION,
        usage=usage,
        cost=cost,
        latency_ms=elapsed_ms,
        success=not bool(error),
    )

    if error:
        return "", usage, error
    return result.text or "", usage, ""


# ---------------------------------------------------------------------------
# Gemini helper for research fallback (legacy API)
# ---------------------------------------------------------------------------

async def _call_google(
    prompt_text: str,
    model: Optional[str] = None,
    timeout: float = 120.0,
    temperature: float = 0.3,
) -> str:
    del timeout  # compatibility; real timeout controlled via environment/config
    if not _PROVIDER_STATES.get("google", {}).get("active"):
        return ""
    model_id = (model or GEMINI_MODEL_ID or "").strip()
    if not model_id:
        return ""
    result = await asyncio.to_thread(call_google, prompt_text, model_id, temperature)
    if result.error:
        return ""
    return result.text
