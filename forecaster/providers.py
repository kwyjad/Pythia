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
import logging
import os
import random
import re
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
    retry_after: Optional[float] = None


@dataclass
class ModelSpec:
    name: str
    provider: str  # "openai" | "anthropic" | "google" | "xai"
    model_id: str
    weight: float = 1.0
    active: bool = True
    purpose: Optional[str] = None


_MAX_LLM_CONCURRENCY = int(os.getenv("PYTHIA_LLM_CONCURRENCY", os.getenv("LLM_MAX_CONCURRENCY", "18")))
_LLM_SEMAPHORES: Dict[int, asyncio.Semaphore] = {}
_HTTP_CLIENTS_BY_LOOP: Dict[int, httpx.AsyncClient] = {}


def _parse_retry_after(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    try:
        value = float(raw)
        return max(0.0, value)
    except Exception:
        return None


def _get_llm_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    key = id(loop)
    sem = _LLM_SEMAPHORES.get(key)
    if sem is None:
        sem = asyncio.Semaphore(_MAX_LLM_CONCURRENCY)
        _LLM_SEMAPHORES[key] = sem
    return sem


def get_llm_semaphore() -> asyncio.Semaphore:
    """Return a semaphore scoped to the current event loop."""

    return _get_llm_semaphore()


# ---------------------------------------------------------------------------
# Provider failure tracking (per run)
# ---------------------------------------------------------------------------

LOGGER = logging.getLogger(__name__)

_PROVIDER_FAILURE_THRESHOLD = int(
    os.getenv("PYTHIA_PROVIDER_FAILURE_THRESHOLD", os.getenv("PROVIDER_FAILURE_THRESHOLD", "6") or 6)
)
_PROVIDER_COOLDOWN_SECONDS = float(os.getenv("PYTHIA_PROVIDER_COOLDOWN_SECONDS", "60") or 60.0)
_PROVIDER_RESET_ON_SUCCESS = os.getenv("PYTHIA_PROVIDER_RESET_ON_SUCCESS", "1") != "0"
_RUN_PROVIDER_STATE: Dict[str, Dict[str, Dict[str, float]]] = {}


def _resolve_run_key(run_id: str | None = None) -> str:
    for candidate in (
        run_id,
        os.getenv("PYTHIA_FORECASTER_RUN_ID"),
        os.getenv("PYTHIA_HS_RUN_ID"),
        os.getenv("PYTHIA_UI_RUN_ID"),
    ):
        if candidate and str(candidate).strip():
            return str(candidate).strip()
    return "default"


def reset_provider_failures_for_run(run_id: str | None = None) -> None:
    key = _resolve_run_key(run_id)
    _RUN_PROVIDER_STATE.pop(key, None)


def _provider_state_for_run(provider: str, run_id: str | None = None) -> Dict[str, float]:
    key = _resolve_run_key(run_id)
    run_state = _RUN_PROVIDER_STATE.setdefault(key, {})
    state = run_state.setdefault(
        provider,
        {"consecutive_failures": 0.0, "cooldown_until_ts": 0.0},
    )
    return state


def _note_provider_failure(provider: str, run_id: str | None = None) -> Dict[str, float]:
    state = _provider_state_for_run(provider, run_id)
    failures = int(state.get("consecutive_failures", 0)) + 1
    state["consecutive_failures"] = float(failures)
    LOGGER.debug("Provider failure count incremented: provider=%s failures=%s", provider, failures)
    if failures >= _PROVIDER_FAILURE_THRESHOLD:
        cooldown_until = time.time() + _PROVIDER_COOLDOWN_SECONDS
        state["cooldown_until_ts"] = cooldown_until
        LOGGER.warning(
            "Provider cooldown started: provider=%s failures=%s until=%s",
            provider,
            failures,
            cooldown_until,
        )
    return state


def _note_provider_success(provider: str, run_id: str | None = None) -> None:
    if not _PROVIDER_RESET_ON_SUCCESS:
        return
    state = _provider_state_for_run(provider, run_id)
    had_failures = int(state.get("consecutive_failures", 0)) > 0
    had_cooldown = float(state.get("cooldown_until_ts", 0.0)) > 0.0
    if had_failures or had_cooldown:
        state["consecutive_failures"] = 0.0
        state["cooldown_until_ts"] = 0.0
        LOGGER.info("Provider failure counters reset: provider=%s", provider)


def _provider_failures_for_run(provider: str, run_id: str | None = None) -> int:
    state = _provider_state_for_run(provider, run_id)
    return int(state.get("consecutive_failures", 0))


def is_provider_disabled_for_run(provider: str, run_id: str | None = None) -> bool:
    state = _provider_state_for_run(provider, run_id)
    return time.time() < float(state.get("cooldown_until_ts", 0.0))


def disabled_providers_for_run(run_id: str | None = None) -> List[str]:
    key = _resolve_run_key(run_id)
    run_state = _RUN_PROVIDER_STATE.get(key, {})
    now = time.time()
    return sorted(
        [
            provider
            for provider, state in run_state.items()
            if now < float(state.get("cooldown_until_ts", 0.0))
        ]
    )


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

# Provider env-key registry — only needs updating when adding a *new provider*
# (which also requires a new call_* function in _call_provider_sync).
_PROVIDER_ENV_KEYS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
    "kimi": "KIMI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

_cfg = load_cfg()
_app_cfg = _cfg.get("app", {}) if isinstance(_cfg, dict) else {}
_forecaster_cfg = _cfg.get("forecaster", {}) if isinstance(_cfg, dict) else {}


def _provider_display_name(provider: str, model_id: str, cfg: Dict[str, Any] | None = None) -> str:
    if cfg:
        explicit = cfg.get("display_name")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
    base_names = {
        "openai": "OpenAI",
        "anthropic": "Claude",
        "google": "Gemini",
        "xai": "Grok",
        "kimi": "Kimi",
        "deepseek": "DeepSeek",
    }
    base = base_names.get(provider, provider.title())
    if not model_id:
        return base
    if provider == "google":
        mid = model_id.lower()
        if "flash" in mid:
            return "Gemini Flash"
        if "pro" in mid:
            return "Gemini Pro"
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


# --- Populate _PROVIDER_STATES from env-key registry ---
_PROVIDER_STATES: Dict[str, Dict[str, Any]] = {}
for _prov_name, _env_key in _PROVIDER_ENV_KEYS.items():
    _api_key = os.getenv(_env_key, "").strip()
    _PROVIDER_STATES[_prov_name] = {
        "api_key": _api_key,
        "env_key": _env_key,
        "enabled": True,
        "weight": 1.0,
    }


def _parse_blocked_providers() -> set[str]:
    raw = os.getenv("PYTHIA_BLOCK_PROVIDERS", "") or ""
    blocked: set[str] = set()
    for part in raw.split(","):
        p = part.strip().lower()
        if p:
            blocked.add(p)
    return blocked


_BLOCKED_PROVIDERS: set[str] = _parse_blocked_providers()


def _apply_provider_block(specs: List[ModelSpec]) -> List[ModelSpec]:
    if not _BLOCKED_PROVIDERS:
        return list(specs)
    return [spec for spec in specs if spec.provider not in _BLOCKED_PROVIDERS]


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


# SPD ensemble helpers
def _make_model_spec(provider: str, model_id: str, *, purpose: Optional[str] = None) -> ModelSpec:
    provider_l = (provider or "").strip().lower()
    name = _provider_display_name(provider_l, model_id)
    state = _PROVIDER_STATES.get(provider_l, {})
    weight = float(state.get("weight", 1.0) or 1.0)
    api_key_present = bool(state.get("api_key"))
    active = bool(api_key_present and model_id)
    return ModelSpec(
        name=name,
        provider=provider_l,
        model_id=model_id,
        weight=weight,
        active=active,
        purpose=purpose,
    )


def parse_ensemble_specs(spec_str: str | None) -> List[ModelSpec]:
    """
    Parse a comma-separated provider:model_id list into ModelSpecs.

    Each ModelSpec is active only if the provider has an API key configured and a
    non-empty model_id. Duplicate providers are allowed.
    """

    if not spec_str:
        return []

    specs: List[ModelSpec] = []
    for raw_part in spec_str.split(","):
        part = raw_part.strip()
        if not part or ":" not in part:
            continue
        provider, model_id = part.split(":", 1)
        provider = provider.strip().lower()
        model_id = model_id.strip()
        if not provider or not model_id:
            continue
        specs.append(_make_model_spec(provider, model_id))

    return _apply_provider_block(specs)


def _load_ensemble_from_config() -> List[ModelSpec]:
    """Read the active profile's ``ensemble`` list from config.yaml.

    Falls back to the legacy ``forecaster.providers`` format or an empty list.
    """

    try:
        from pythia.llm_profiles import get_ensemble_list
        ensemble_list = get_ensemble_list()
    except Exception:
        ensemble_list = []

    if ensemble_list:
        spec_str = ",".join(str(e) for e in ensemble_list)
        return parse_ensemble_specs(spec_str)

    # Legacy fallback: read from forecaster.providers (if present)
    legacy_providers = _forecaster_cfg.get("providers", {}) if isinstance(_forecaster_cfg, dict) else {}
    if isinstance(legacy_providers, dict) and legacy_providers:
        parts: List[str] = []
        for prov, entry in legacy_providers.items():
            if isinstance(entry, dict):
                model = str(entry.get("model", "")).strip()
                if model:
                    parts.append(f"{prov}:{model}")
        if parts:
            return parse_ensemble_specs(",".join(parts))

    return []


# --- Build model lists from config ---
_config_ensemble: List[ModelSpec] = _load_ensemble_from_config()

# Populate _PROVIDER_STATES with model info from ensemble (first model per provider)
for _ms in _config_ensemble:
    _state = _PROVIDER_STATES.setdefault(_ms.provider, {
        "api_key": "", "env_key": _PROVIDER_ENV_KEYS.get(_ms.provider, ""), "enabled": True, "weight": 1.0,
    })
    if "model" not in _state:
        _state["model"] = _ms.model_id
        _state["display_name"] = _ms.name

_MODEL_SPECS: List[ModelSpec] = list(_config_ensemble)
KNOWN_MODELS: List[str] = [spec.name for spec in _MODEL_SPECS]
DEFAULT_ENSEMBLE: List[ModelSpec] = _apply_provider_block([spec for spec in _MODEL_SPECS if spec.active])

# SPD ensemble: env var override takes precedence, otherwise use config ensemble
SPD_ENSEMBLE_OVERRIDE: List[ModelSpec] = parse_ensemble_specs(os.getenv("PYTHIA_SPD_ENSEMBLE_SPECS", ""))


def _apply_spd_google_model_override(specs: List[ModelSpec]) -> List[ModelSpec]:
    override = (os.getenv("PYTHIA_SPD_GOOGLE_MODEL_ID") or "").strip()
    if not override:
        return specs

    updated: List[ModelSpec] = []
    seen: set[tuple[str, str]] = set()
    for ms in specs:
        if ms.provider == "google":
            ms = ModelSpec(
                name=ms.name,
                provider=ms.provider,
                model_id=override,
                weight=ms.weight,
                active=bool(ms.active and override),
                purpose=ms.purpose,
            )
        key = (ms.provider, ms.model_id)
        if key in seen:
            continue
        seen.add(key)
        updated.append(ms)
    return updated


SPD_ENSEMBLE: List[ModelSpec] = _apply_spd_google_model_override(
    SPD_ENSEMBLE_OVERRIDE or list(DEFAULT_ENSEMBLE)
)

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
_KIMI_STATE = _PROVIDER_STATES.get("kimi", {})
_KIMI_API_KEY = _KIMI_STATE.get("api_key", "")
_DEEPSEEK_STATE = _PROVIDER_STATES.get("deepseek", {})
_DEEPSEEK_API_KEY = _DEEPSEEK_STATE.get("api_key", "")

_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
_XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1").strip()
_KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1").strip()
_DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip()

_OPENAI_TIMEOUT = _resolve_timeout("OPENAI_CALL_TIMEOUT_SEC", GPT5_CALL_TIMEOUT_SEC, 60.0)
_ANTHROPIC_TIMEOUT = _resolve_timeout("ANTHROPIC_CALL_TIMEOUT_SEC", GPT5_CALL_TIMEOUT_SEC, 60.0)
_GEMINI_TIMEOUT = _resolve_timeout("GEMINI_CALL_TIMEOUT_SEC", GEMINI_CALL_TIMEOUT_SEC, 60.0)
_XAI_TIMEOUT = _resolve_timeout("XAI_CALL_TIMEOUT_SEC", GROK_CALL_TIMEOUT_SEC, 60.0)
_KIMI_TIMEOUT = _resolve_timeout("KIMI_CALL_TIMEOUT_SEC", None, 300.0)
_DEEPSEEK_TIMEOUT = _resolve_timeout("DEEPSEEK_CALL_TIMEOUT_SEC", None, 300.0)

_ANTHROPIC_VERSION = os.getenv("ANTHROPIC_API_VERSION", "2023-06-01")
_ANTHROPIC_MAX_OUTPUT = int(os.getenv("ANTHROPIC_MAX_OUTPUT_TOKENS", "2048") or 2048)


def _get_or_client() -> httpx.AsyncClient:
    """Return a shared async HTTP client for provider calls."""

    loop = asyncio.get_running_loop()
    key = id(loop)
    client = _HTTP_CLIENTS_BY_LOOP.get(key)
    if client is None:
        client = httpx.AsyncClient(timeout=30.0)
        _HTTP_CLIENTS_BY_LOOP[key] = client
    return client


# ---------------------------------------------------------------------------
# Usage / cost helpers
# ---------------------------------------------------------------------------

# Cost per 1,000 tokens for known models (USD). Loaded from pythia/model_costs.json
# so that adding a new model's cost only requires editing a JSON file, not Python code.

def _load_model_costs_json() -> Dict[str, tuple[float, float]]:
    """Load model cost data from ``pythia/model_costs.json``."""
    import pathlib

    try:
        import pythia
        costs_path = pathlib.Path(pythia.__file__).parent / "model_costs.json"
    except Exception:
        costs_path = pathlib.Path(__file__).parent.parent / "pythia" / "model_costs.json"
    try:
        with open(costs_path) as f:
            raw = json.load(f)
    except Exception:
        return {}
    result: Dict[str, tuple[float, float]] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if isinstance(value, (list, tuple)) and len(value) == 2:
            try:
                result[key] = (float(value[0]), float(value[1]))
            except (ValueError, TypeError):
                continue
    return result


MODEL_PRICES_PER_1K: Dict[str, tuple[float, float]] = _load_model_costs_json()

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
        retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
        message = ""
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                message = str(err.get("message", ""))
            elif isinstance(err, str):
                message = err
        if not message:
            message = resp.text[:400]
        return ProviderResult(
            "",
            usage_to_dict(None),
            0.0,
            model,
            error=f"OpenAI HTTP {resp.status_code}: {message}",
            retry_after=retry_after,
        )

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
        retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
        message = ""
        if isinstance(payload, dict):
            message = payload.get("error", {}).get("message") if isinstance(payload.get("error"), dict) else payload.get("error")
            if not isinstance(message, str):
                message = ""
        if not message:
            message = resp.text[:400]
        return ProviderResult(
            "",
            usage_to_dict(None),
            0.0,
            model,
            error=f"Anthropic HTTP {resp.status_code}: {message}",
            retry_after=retry_after,
        )

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


def call_google(
    prompt: str,
    model: str,
    temperature: float,
    *,
    timeout_sec: Optional[float] = None,
    thinking_level: Optional[str] = None,
) -> ProviderResult:
    if not _GEMINI_API_KEY:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error="missing GEMINI_API_KEY")
    api_model = model.split("/", 1)[-1] if "/" in model else model
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{api_model}:generateContent?key={_GEMINI_API_KEY}"
    generation_config: Dict[str, Any] = {"temperature": float(temperature)}
    if thinking_level and api_model.lower().startswith("gemini-3-"):
        generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    try:
        resp = requests.post(
            url,
            json=body,
            timeout=timeout_sec if timeout_sec is not None else _GEMINI_TIMEOUT,
        )
    except Exception as exc:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error=f"Gemini request error: {exc}")

    try:
        payload = resp.json()
    except Exception:
        payload = {}

    if resp.status_code != 200:
        retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
        message = ""
        if isinstance(payload, dict):
            error_obj = payload.get("error", {})
            if isinstance(error_obj, dict):
                message = str(error_obj.get("message", ""))
            elif isinstance(error_obj, str):
                message = error_obj
        if not message:
            message = resp.text[:400]
        return ProviderResult(
            "",
            usage_to_dict(None),
            0.0,
            model,
            error=f"Gemini HTTP {resp.status_code}: {message}",
            retry_after=retry_after,
        )

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
        retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
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
        return ProviderResult(
            "",
            usage_to_dict(None),
            0.0,
            model,
            error=f"xAI HTTP {resp.status_code}: {message}",
            retry_after=retry_after,
        )

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


def _call_openai_compatible(
    prompt: str,
    model: str,
    temperature: float,
    *,
    api_key: str,
    base_url: str,
    timeout: float,
    provider_label: str,
) -> ProviderResult:
    """Shared implementation for OpenAI-compatible APIs (Kimi, DeepSeek, etc.)."""
    if not api_key:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error=f"missing {provider_label} API key")
    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": float(temperature),
            },
            timeout=timeout,
        )
    except Exception as exc:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error=f"{provider_label} error: {exc}")

    try:
        payload = resp.json()
    except Exception:
        payload = {}

    if not resp.ok:
        retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
        message = ""
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                message = str(err.get("message", ""))
            elif isinstance(err, str):
                message = err
        if not message:
            message = resp.text[:400]
        return ProviderResult(
            "",
            usage_to_dict(None),
            0.0,
            model,
            error=f"{provider_label} HTTP {resp.status_code}: {message}",
            retry_after=retry_after,
        )

    text = ""
    if isinstance(payload, dict):
        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message") or {}
            if isinstance(message, dict):
                text = str(message.get("content", "")).strip()
    usage = usage_to_dict(payload.get("usage") if isinstance(payload, dict) else {})
    return ProviderResult(text=text, usage=usage, cost_usd=0.0, model_id=model)


def call_kimi(prompt: str, model: str, temperature: float) -> ProviderResult:
    return _call_openai_compatible(
        prompt, model, temperature,
        api_key=_KIMI_API_KEY,
        base_url=_KIMI_BASE_URL,
        timeout=_KIMI_TIMEOUT,
        provider_label="Kimi",
    )


def call_deepseek(prompt: str, model: str, temperature: float) -> ProviderResult:
    return _call_openai_compatible(
        prompt, model, temperature,
        api_key=_DEEPSEEK_API_KEY,
        base_url=_DEEPSEEK_BASE_URL,
        timeout=_DEEPSEEK_TIMEOUT,
        provider_label="DeepSeek",
    )


def _call_provider_sync(
    provider: str,
    prompt: str,
    model: str,
    temperature: float,
    *,
    timeout_sec: Optional[float] = None,
    thinking_level: Optional[str] = None,
) -> ProviderResult:
    p = (provider or "").lower()
    if p == "openai":
        return call_openai(prompt, model, temperature)
    if p == "anthropic":
        return call_anthropic(prompt, model, temperature)
    if p in {"google", "gemini"}:
        return call_google(prompt, model, temperature, timeout_sec=timeout_sec, thinking_level=thinking_level)
    if p in {"xai", "grok"}:
        return call_xai(prompt, model, temperature)
    if p == "kimi":
        return call_kimi(prompt, model, temperature)
    if p == "deepseek":
        return call_deepseek(prompt, model, temperature)
    return ProviderResult("", usage_to_dict(None), 0.0, model, error=f"unsupported provider {provider}")


def _extract_status_code(error: str) -> Optional[int]:
    patterns = (
        r"HTTP\s*(\d{3})",
        r"status(?:\s*code)?\s*(\d{3})",
    )
    for pat in patterns:
        match = re.search(pat, error, flags=re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                return None
    return None


def _is_timeout_error(error: Optional[str]) -> bool:
    if not error:
        return False
    lower_err = error.lower()
    return "timeout" in lower_err or "timed out" in lower_err


def _should_retry_provider_error(
    error: Optional[str],
    retry_after_hint: Optional[float] = None,
    *,
    purpose: Optional[str] = None,
    allow_timeout_retry: bool = True,
) -> tuple[bool, Optional[float]]:
    if not error:
        return False, None

    if _is_timeout_error(error):
        if purpose == "hs_triage":
            return True, retry_after_hint
        if allow_timeout_retry:
            return True, retry_after_hint
        return False, None

    lower_err = error.lower()
    transient_keywords = (
        "connection reset",
        "connection aborted",
        "remote end closed",
        "temporarily unavailable",
        "transport error",
        "connection closed without response",
    )
    for kw in transient_keywords:
        if kw in lower_err:
            return True, retry_after_hint

    status_code = _extract_status_code(error)
    if status_code == 429:
        return True, retry_after_hint
    if status_code is not None and 500 <= status_code < 600:
        return True, retry_after_hint

    return False, None


def _format_provider_exception(exc: Exception) -> str:
    loop_id: Optional[int] = None
    sem_id: Optional[int] = None
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        sem = _get_llm_semaphore()
        sem_id = id(sem)
    except Exception:
        pass
    loop_info = ""
    if loop_id is not None:
        loop_info = f" [loop_id={loop_id}"
        if sem_id is not None:
            loop_info += f" sem_id={sem_id}"
        loop_info += "]"
    return f"provider call error: {exc}{loop_info}"


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
    run_id: str | None = None,
) -> tuple[str, Dict[str, int], str]:
    """Call the configured provider for a model spec and return (text, usage, error)."""

    if not ms.active:
        return "", usage_to_dict(None), f"provider {ms.provider} inactive"

    run_key = _resolve_run_key(run_id)
    state = _provider_state_for_run(ms.provider, run_key)
    cooldown_until = float(state.get("cooldown_until_ts", 0.0))
    now = time.time()
    if cooldown_until and now >= cooldown_until:
        if state.get("cooldown_until_ts"):
            LOGGER.info(
                "Provider cooldown ended: provider=%s run_id=%s",
                ms.provider,
                run_key,
            )
        state["cooldown_until_ts"] = 0.0
    if is_provider_disabled_for_run(ms.provider, run_key):
        LOGGER.info(
            "Provider cooldown active; skipping call: provider=%s run_id=%s",
            ms.provider,
            run_key,
        )
        usage = usage_to_dict(None)
        usage["cooldown_active"] = True
        usage["cooldown_until_ts"] = float(state.get("cooldown_until_ts", 0.0))
        usage["provider_failures_in_run"] = _provider_failures_for_run(ms.provider, run_key)
        usage["attempts_used"] = 0
        usage["backoffs_sec"] = []
        return "", usage, (
            f"provider {ms.provider} cooldown active for run {run_key} until {state.get('cooldown_until_ts')}"
        )

    start = time.time()
    spd_google = ms.provider == "google" and ms.purpose == "spd_v2"
    hs_triage = ms.purpose == "hs_triage"
    thinking_level: Optional[str] = None
    timeout_sec: Optional[float] = None
    hs_timeout_sec: Optional[float] = None
    hs_max_retry_after_sec: Optional[float] = None
    hs_fail_fast_on_retry_after = False
    hs_max_attempts: Optional[int] = None
    hs_usage: Dict[str, Any] = {}
    backoffs_sec: list[float] = []
    if spd_google:
        model_id_lower = ms.model_id.lower()
        if "gemini-3-flash" in model_id_lower:
            thinking_level = (os.getenv("PYTHIA_GOOGLE_SPD_THINKING_LEVEL_FLASH", "low") or "").strip()
            if not thinking_level:
                thinking_level = None
            timeout_sec = _resolve_timeout("PYTHIA_GOOGLE_SPD_TIMEOUT_FLASH_SEC", None, 90.0)
        elif "gemini-3-pro" in model_id_lower:
            thinking_level = (os.getenv("PYTHIA_GOOGLE_SPD_THINKING_LEVEL_PRO", "") or "").strip()
            if not thinking_level:
                thinking_level = None
            timeout_sec = _resolve_timeout("PYTHIA_GOOGLE_SPD_TIMEOUT_PRO_SEC", None, 120.0)
        try:
            max_attempts = max(1, int(os.getenv("PYTHIA_GOOGLE_SPD_RETRIES", "1") or 1))
        except Exception:
            max_attempts = 1
    else:
        max_attempts = max(1, int(os.getenv("PYTHIA_LLM_RETRIES", "3") or 3))
    if hs_triage:
        try:
            hs_max_attempts = max(1, int(os.getenv("PYTHIA_HS_LLM_MAX_ATTEMPTS", "3") or 3))
        except Exception:
            hs_max_attempts = 3
        try:
            hs_max_retry_after_sec = max(
                0.0, float(os.getenv("PYTHIA_HS_LLM_MAX_RETRY_AFTER_SEC", "10") or 10)
            )
        except Exception:
            hs_max_retry_after_sec = 10.0
        hs_fail_fast_on_retry_after = os.getenv("PYTHIA_HS_LLM_FAIL_FAST_ON_RETRY_AFTER", "1") == "1"
        if ms.provider == "google":
            hs_timeout_sec = _resolve_timeout("PYTHIA_HS_GEMINI_TIMEOUT_SEC", None, 120.0)
            timeout_sec = hs_timeout_sec
        if hs_max_attempts is not None:
            max_attempts = min(max_attempts, hs_max_attempts)
        if hs_timeout_sec is not None:
            hs_usage["hs_timeout_sec"] = hs_timeout_sec
    attempt = 0
    result: Optional[ProviderResult] = None
    error: Optional[str] = None

    while attempt < max_attempts:
        attempt += 1
        try:
            async with _get_llm_semaphore():
                call_task = asyncio.to_thread(
                    _call_provider_sync,
                    ms.provider,
                    prompt,
                    ms.model_id,
                    temperature,
                    timeout_sec=timeout_sec,
                    thinking_level=thinking_level,
                )
                if timeout_sec is not None:
                    result = await asyncio.wait_for(call_task, timeout=timeout_sec)
                else:
                    result = await call_task
        except asyncio.TimeoutError:
            error = f"timeout after {timeout_sec}s"
            result = ProviderResult("", usage_to_dict(None), 0.0, ms.model_id, error=error)
        except Exception as exc:  # pragma: no cover - unexpected runtime errors
            error = _format_provider_exception(exc)
            result = ProviderResult("", usage_to_dict(None), 0.0, ms.model_id, error=error)
        else:
            error = result.error if result and result.error else None

        retry_after_hint = result.retry_after if result else None
        allow_timeout_retry = os.getenv("PYTHIA_LLM_RETRY_TIMEOUTS", "1") != "0"
        should_retry, retry_after = _should_retry_provider_error(
            error,
            retry_after_hint,
            purpose=ms.purpose,
            allow_timeout_retry=allow_timeout_retry,
        )
        if hs_triage and retry_after_hint is not None:
            hs_usage["retry_after_hint_sec"] = retry_after_hint
            if hs_max_retry_after_sec is not None and retry_after_hint > hs_max_retry_after_sec:
                hs_usage["retry_after_capped"] = True
                if hs_fail_fast_on_retry_after:
                    hs_usage["retry_after_used_sec"] = 0.0
                    should_retry = False
                else:
                    retry_after = hs_max_retry_after_sec
        if not should_retry or attempt >= max_attempts:
            break

        if retry_after is not None:
            backoff = min(20.0, float(retry_after))
        else:
            backoff = min(20.0, 1.0 * (2 ** (attempt - 1)))
        backoff += random.uniform(0.0, 0.5)
        if hs_triage and hs_max_retry_after_sec is not None:
            if backoff > hs_max_retry_after_sec:
                backoff = hs_max_retry_after_sec
                hs_usage["retry_after_capped"] = True
            hs_usage["retry_after_used_sec"] = backoff
        backoffs_sec.append(backoff)
        await asyncio.sleep(backoff)

    if result is None:
        result = ProviderResult("", usage_to_dict(None), 0.0, ms.model_id, error=error or "unknown error")

    error = result.error if result and result.error else None

    elapsed_ms = int((time.time() - start) * 1000)
    usage = result.usage or usage_to_dict(None)
    usage["attempts_used"] = attempt
    usage["backoffs_sec"] = backoffs_sec
    if hs_usage:
        usage.update(hs_usage)
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
        if spd_google and _is_timeout_error(error):
            return "", usage, error
        state = _note_provider_failure(ms.provider, run_key)
        usage["provider_failures_in_run"] = int(state.get("consecutive_failures", 0))
        cooldown_until_ts = float(state.get("cooldown_until_ts", 0.0))
        if cooldown_until_ts > time.time():
            usage["cooldown_active"] = True
            usage["cooldown_until_ts"] = cooldown_until_ts
        return "", usage, error
    _note_provider_success(ms.provider, run_key)
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
