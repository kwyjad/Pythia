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
import requests

try:  # pragma: no cover - defensive import for environments without openai
    from openai import AsyncOpenAI, OpenAI
except Exception:  # pragma: no cover - openai package missing
    AsyncOpenAI = None  # type: ignore
    OpenAI = None  # type: ignore

from pythia.config import load as load_cfg
from pythia.db.util import write_llm_call

from .config import (
    GEMINI_CALL_TIMEOUT_SEC,
    GPT5_CALL_TIMEOUT_SEC,
    GROK_CALL_TIMEOUT_SEC,
)


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


llm_semaphore = asyncio.Semaphore(int(os.getenv("LLM_MAX_CONCURRENCY", "4")))


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_DEFAULT_PROVIDER_CONFIG: Dict[str, Dict[str, Any]] = {
    "openai": {
        "enabled": True,
        "model": "gpt-5.1-pro",
        "env_key": "OPENAI_API_KEY",
        "display_name": "OpenAI-gpt-5.1-pro",
    },
    "anthropic": {
        "enabled": True,
        "model": "claude-opus-4.5",
        "env_key": "ANTHROPIC_API_KEY",
        "display_name": "Claude-opus-4.5",
    },
    "google": {
        "enabled": True,
        "model": "gemini-3-pro",
        "env_key": "GEMINI_API_KEY",
        "display_name": "Gemini-3-pro",
    },
    "xai": {
        "enabled": True,
        "model": "grok-4.1",
        "env_key": "XAI_API_KEY",
        "display_name": "Grok-4.1",
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


# backwards-compatible aliases reused elsewhere in the forecaster package
_OPENAI_STATE = _PROVIDER_STATES.get("openai", {})
OPENAI_MODEL_ID = _OPENAI_STATE.get("model", "")
OPENROUTER_FALLBACK_ID = OPENAI_MODEL_ID  # legacy name kept for topic_classify
_GEMINI_STATE = _PROVIDER_STATES.get("google", {})
GEMINI_MODEL_ID = _GEMINI_STATE.get("model", "")
_XAI_STATE = _PROVIDER_STATES.get("xai", {})
GROK_MODEL_ID = _XAI_STATE.get("model", "")

_OPENAI_API_KEY = _OPENAI_STATE.get("api_key", "")
_ANTHROPIC_API_KEY = _PROVIDER_STATES.get("anthropic", {}).get("api_key", "")
_GEMINI_API_KEY = _GEMINI_STATE.get("api_key", "")
_XAI_API_KEY = _XAI_STATE.get("api_key", "")

_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip() or None
_XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1/chat/completions").strip()

_OPENAI_TIMEOUT = _resolve_timeout("OPENAI_CALL_TIMEOUT_SEC", GPT5_CALL_TIMEOUT_SEC, 60.0)
_ANTHROPIC_TIMEOUT = _resolve_timeout("ANTHROPIC_CALL_TIMEOUT_SEC", GPT5_CALL_TIMEOUT_SEC, 60.0)
_GEMINI_TIMEOUT = _resolve_timeout("GEMINI_CALL_TIMEOUT_SEC", GEMINI_CALL_TIMEOUT_SEC, 60.0)
_XAI_TIMEOUT = _resolve_timeout("XAI_CALL_TIMEOUT_SEC", GROK_CALL_TIMEOUT_SEC, 60.0)

_ANTHROPIC_VERSION = os.getenv("ANTHROPIC_API_VERSION", "2023-06-01")
_ANTHROPIC_MAX_OUTPUT = int(os.getenv("ANTHROPIC_MAX_OUTPUT_TOKENS", "2048") or 2048)


# ---------------------------------------------------------------------------
# Usage / cost helpers
# ---------------------------------------------------------------------------

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
    prices = _load_model_prices()
    if not usage or not isinstance(usage, dict):
        return 0.0
    price_entry = (
        prices.get(model_id)
        or prices.get(model_id.replace("/", "-"))
        or prices.get(model_id.split("/", 1)[-1])
        or {}
    )
    try:
        prompt_rate = float(price_entry.get("prompt", 0.0))
        completion_rate = float(price_entry.get("completion", 0.0))
        return (
            (usage.get("prompt_tokens", 0) / 1000.0) * prompt_rate
            + (usage.get("completion_tokens", 0) / 1000.0) * completion_rate
        )
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Provider client helpers
# ---------------------------------------------------------------------------

_openai_client_sync: Optional[OpenAI] = None
_openai_client_async: Optional[AsyncOpenAI] = None


def _get_openai_client() -> Optional[OpenAI]:
    global _openai_client_sync
    if OpenAI is None or not _OPENAI_API_KEY:
        return None
    if _openai_client_sync is None:
        _openai_client_sync = OpenAI(api_key=_OPENAI_API_KEY, base_url=_OPENAI_BASE_URL, timeout=_OPENAI_TIMEOUT)
    return _openai_client_sync


def _get_or_client() -> Optional[AsyncOpenAI]:  # legacy name used by topic_classify
    global _openai_client_async
    if AsyncOpenAI is None or not _OPENAI_API_KEY:
        return None
    if _openai_client_async is None:
        try:
            _openai_client_async = AsyncOpenAI(
                api_key=_OPENAI_API_KEY,
                base_url=_OPENAI_BASE_URL,
                timeout=_OPENAI_TIMEOUT,
            )
        except Exception:  # pragma: no cover - defensive, we just disable async usage
            return None
    return _openai_client_async


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------


def call_openai(prompt: str, model: str, temperature: float) -> ProviderResult:
    if not _OPENAI_API_KEY:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error="missing OPENAI_API_KEY")
    client = _get_openai_client()
    if client is None:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error="openai client unavailable")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = usage_to_dict(getattr(resp, "usage", None))
        return ProviderResult(text=text, usage=usage, cost_usd=0.0, model_id=model)
    except Exception as exc:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error=f"OpenAI error: {exc}")


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
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={_GEMINI_API_KEY}"
    body = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
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
        resp = requests.post(_XAI_BASE_URL, headers=headers, json=body, timeout=XAI_TIMEOUT)
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
