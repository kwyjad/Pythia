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


# ---------------------------------------------------------------------------
# Credit-retry: long-pause retry for billing/quota exhaustion errors
# ---------------------------------------------------------------------------

class ProviderBillingError(Exception):
    """Raised when a provider returns a billing/quota-exhaustion error (distinct from rate-limiting)."""

    def __init__(self, provider: str, message: str, status_code: int | None = None):
        self.provider = provider
        self.status_code = status_code
        super().__init__(message)


_CREDIT_RETRY_CONFIG: dict[str, dict[str, int]] = {
    "openai":    {"pause_sec": 900, "max_retries": 3},   # 15 min × 3
    "anthropic": {"pause_sec": 300, "max_retries": 3},   # 5 min × 3
    "google":    {"pause_sec": 600, "max_retries": 3},   # 10 min × 3
}


def _credit_retry_config_for(provider: str) -> tuple[int, int] | None:
    """Return (pause_sec, max_retries) for a provider, with env-var overrides.

    Returns None for providers not in ``_CREDIT_RETRY_CONFIG``.
    """
    p = (provider or "").lower()
    base = _CREDIT_RETRY_CONFIG.get(p)
    if base is None:
        return None
    p_upper = p.upper()
    pause = int(os.getenv(f"PYTHIA_CREDIT_RETRY_PAUSE_{p_upper}", str(base["pause_sec"])))
    max_retries = int(os.getenv(f"PYTHIA_CREDIT_RETRY_MAX_{p_upper}", str(base["max_retries"])))
    return pause, max_retries


def _is_billing_error(provider: str, error_text: str, status_code: int | None = None) -> bool:
    """Detect billing/quota-exhaustion errors, distinct from transient rate limits.

    Conservative: returns False when uncertain.
    """
    if not error_text:
        return False
    p = (provider or "").lower()
    lower = error_text.lower()
    sc = status_code

    if p == "openai":
        # OpenAI billing: 429 + quota/billing keywords, but NOT rate-limit 429s
        if sc == 429 and "rate limit" not in lower:
            return "quota" in lower or "billing" in lower or "insufficient_quota" in lower
        return False

    if p == "anthropic":
        # Anthropic billing: 400 or 403 + credit/billing/blocked keywords
        if sc in (400, 403):
            return "insufficient" in lower or "billing" in lower or "blocked" in lower
        return False

    if p in ("google", "gemini"):
        # Google billing: 429 + RESOURCE_EXHAUSTED + quota/billing
        # Excludes RPM/TPM rate limits ("Too Many Requests" without RESOURCE_EXHAUSTED)
        if sc == 429 and "resource_exhausted" in lower:
            return "quota" in lower or "billing" in lower
        return False

    return False


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
    provider: str  # "openai" | "anthropic" | "google"
    model_id: str
    weight: float = 1.0
    active: bool = True
    purpose: Optional[str] = None
    temperature: Optional[float] = None  # None = use caller default (0.2)
    # None = don't send. OpenAI: reasoning_effort; Google: thinkingLevel
    # ("off"|"low"|"medium"|"high"); Anthropic (effort-capable models only,
    # see _ANTHROPIC_EFFORT_PREFIXES): output_config.effort
    # ("low"|"medium"|"high"|"xhigh"|"max").
    thinking: Optional[str] = None


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
}

_cfg = load_cfg()
_app_cfg = _cfg.get("app", {}) if isinstance(_cfg, dict) else {}
_forecaster_cfg = _cfg.get("forecaster", {}) if isinstance(_cfg, dict) else {}


def _provider_display_name(provider: str, model_id: str, cfg: Dict[str, Any] | None = None) -> str:
    """Return the display name stored as ``model_name`` in forecasts_raw / scores / llm_calls.

    Uses the specific model id (e.g. ``gemini-3.5-flash``, ``gpt-5.6-sol``) so
    dashboards, downloads, and calibration weights always reference a clear,
    unambiguous model — never a generic family label like "Gemini" or
    "Gemini Flash". An explicit ``display_name`` in config still wins.
    """
    if cfg:
        explicit = cfg.get("display_name")
        if isinstance(explicit, str) and explicit.strip():
            return explicit.strip()
    if model_id:
        return model_id.replace("/", "-")
    return provider.title()


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

    Entry parsing (registry aliases, ``provider:model_id`` strings, and
    ``{provider, model_id}`` dicts) lives in
    ``pythia.llm_profiles.get_ensemble_resolved``.

    Falls back to the legacy ``forecaster.providers`` format or an empty list.
    """

    try:
        from pythia.llm_profiles import get_ensemble_resolved
        ensemble_list = get_ensemble_resolved()
    except Exception:
        ensemble_list = []

    if ensemble_list:
        specs: List[ModelSpec] = []
        for entry in ensemble_list:
            provider = str(entry.get("provider", "")).strip().lower()
            model_id = str(entry.get("model_id", "")).strip()
            if not provider or not model_id:
                continue
            ms = _make_model_spec(provider, model_id)

            temp_val = entry.get("temperature")
            if temp_val is not None:
                try:
                    ms.temperature = float(temp_val)
                except (ValueError, TypeError):
                    pass

            thinking_val = entry.get("thinking")
            if isinstance(thinking_val, str) and thinking_val.strip():
                ms.thinking = thinking_val.strip().lower()

            specs.append(ms)

        return _apply_provider_block(specs)

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
            # NOTE: this rebuild drops temperature/thinking BY DESIGN (the
            # PYTHIA_GOOGLE_SPD_THINKING_LEVEL_* env fallbacks cover
            # override runs — documented in CLAUDE.md). The NAME must
            # follow the override though: display names are the model id
            # itself, and keeping the old name would attribute
            # forecasts_raw/scores/calibration weights to a model that
            # never ran.
            ms = ModelSpec(
                name=override,
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

_OPENAI_API_KEY = _OPENAI_STATE.get("api_key", "")
_ANTHROPIC_API_KEY = _PROVIDER_STATES.get("anthropic", {}).get("api_key", "")
_GEMINI_API_KEY = _GEMINI_STATE.get("api_key", "")

_OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()

# OpenAI models that reject a custom temperature (HTTP 400 if sent): the
# GPT-5 reasoning family (gpt-5, gpt-5.4, gpt-5.6-sol, gpt-5.6-luna, ...).
# Prefix-matched so a lineup bump (e.g. gpt-5.5) can't silently reintroduce
# temperature on a path that doesn't set reasoning_effort (the hs_fallback
# JSON-repair spec carries no thinking value).
_OPENAI_NO_TEMPERATURE_PREFIXES = ("gpt-5",)


def _openai_drops_temperature(model: str) -> bool:
    # Case-insensitive like the Anthropic guard: an env-supplied id with any
    # capitalization ("GPT-5.6-sol" via PYTHIA_SPD_ENSEMBLE_SPECS) would
    # otherwise send temperature and 400 on every call.
    return (model or "").lower().startswith(_OPENAI_NO_TEMPERATURE_PREFIXES)
# Anthropic models that reject sampling params outright (HTTP 400 if sent):
# the Opus 4.7+/Sonnet 5+/Fable family removed temperature/top_p/top_k.
#
# These are LITERAL id prefixes, not families: "claude-opus-4-8" does NOT cover
# "claude-opus-5". Every Anthropic lineup bump must add its own entry here, or
# the SPD Claude member AND every Sibyl step 400 on the first call.
_ANTHROPIC_NO_TEMPERATURE_PREFIXES = (
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable",
)
# Anthropic models that accept the GA effort knob (``output_config.effort``,
# no beta header; ``budget_tokens`` is REMOVED on Opus 5 — 400 if sent).
# Same literal-prefix contract as the temperature guard above, and gated for
# the same reason in reverse: grounding_claude is pinned to Haiku 4.5, which
# REJECTS ``effort`` — an ungated emit would 400 that role on its first call.
# Deliberately narrower than the temperature tuple: a missing entry is a
# safe no-op (the level is silently not sent), a wrong entry is a 400 on
# every call, so only generations verified to take ``effort`` are listed.
# Accepted levels: low | medium | high | xhigh | max ("high" is the model
# default). ModelSpec.thinking values pass through verbatim.
_ANTHROPIC_EFFORT_PREFIXES = (
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable",
    "claude-mythos",
)
_OPENAI_TIMEOUT = _resolve_timeout("OPENAI_CALL_TIMEOUT_SEC", GPT5_CALL_TIMEOUT_SEC, 60.0)
_ANTHROPIC_TIMEOUT = _resolve_timeout("ANTHROPIC_CALL_TIMEOUT_SEC", GPT5_CALL_TIMEOUT_SEC, 60.0)
_GEMINI_TIMEOUT = _resolve_timeout("GEMINI_CALL_TIMEOUT_SEC", GEMINI_CALL_TIMEOUT_SEC, 60.0)

_ANTHROPIC_VERSION = os.getenv("ANTHROPIC_API_VERSION", "2023-06-01")
_ANTHROPIC_MAX_OUTPUT = int(os.getenv("ANTHROPIC_MAX_OUTPUT_TOKENS", "16384") or 16384)
# SPD/binary ceiling. Opus 5 runs adaptive thinking BY DEFAULT (Opus 4.8 did not
# unless asked), and thinking tokens are charged against the same max_tokens
# budget as the answer — so a 16k cap that comfortably held an SPD JSON on 4.8
# can now truncate mid-object, which surfaces only as a parse failure.
_ANTHROPIC_SPD_MAX_OUTPUT_DEFAULT = 32768
_ANTHROPIC_SPD_MAX_OUTPUT = int(
    os.getenv("PYTHIA_ANTHROPIC_SPD_MAX_TOKENS", str(max(_ANTHROPIC_MAX_OUTPUT, _ANTHROPIC_SPD_MAX_OUTPUT_DEFAULT)))
    or _ANTHROPIC_SPD_MAX_OUTPUT_DEFAULT
)


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

# Cost per 1,000,000 (1M) tokens for known models (USD). Loaded from
# pythia/model_costs.json so that adding a new model's cost only requires
# editing a JSON file, not Python code.

def _load_model_costs_raw() -> Dict[str, Any]:
    """Load the raw ``pythia/model_costs.json`` mapping (per-1M USD rates).

    Two value forms are accepted per model id (backward compatible):

    * legacy 2-array: ``[input, output]``
    * object form:    ``{"input": .., "output": .., "cached_input": ..,
                         "cache_write_5m": .., "cache_write_1h": ..}``
      (cache fields optional — provider defaults fill the gaps, see
      ``resolve_price_detail``)
    """
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
    return raw if isinstance(raw, dict) else {}


def _parse_model_costs(raw: Dict[str, Any]) -> tuple[Dict[str, tuple[float, float]], Dict[str, Dict[str, float]]]:
    """Split the raw cost table into (input, output) tuples + full detail records."""

    pairs: Dict[str, tuple[float, float]] = {}
    details: Dict[str, Dict[str, float]] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        try:
            if isinstance(value, (list, tuple)) and len(value) == 2:
                pairs[key] = (float(value[0]), float(value[1]))
            elif isinstance(value, dict) and "input" in value and "output" in value:
                pairs[key] = (float(value["input"]), float(value["output"]))
                detail: Dict[str, float] = {}
                for field in ("cached_input", "cache_write_5m", "cache_write_1h"):
                    if value.get(field) is not None:
                        detail[field] = float(value[field])
                if detail:
                    details[key] = detail
        except (ValueError, TypeError):
            continue
    return pairs, details


def _load_model_costs_json() -> Dict[str, tuple[float, float]]:
    """Load model cost data from ``pythia/model_costs.json`` (per-1M rates)."""

    pairs, _ = _parse_model_costs(_load_model_costs_raw())
    return pairs


_MODEL_PRICE_PAIRS, _MODEL_PRICE_CACHE_DETAILS = _parse_model_costs(_load_model_costs_raw())
MODEL_PRICES_PER_1M: Dict[str, tuple[float, float]] = _MODEL_PRICE_PAIRS

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


# Cache/batch usage fields preserved by usage_to_dict beyond the three core
# token counts. Int-valued keys are coerced; string keys pass through verbatim.
# These flow into llm_calls.usage_json via the loggers' passthrough and are
# priced by compute_cost_split_usd — dropping one here silently overbills the
# ledger (cached tokens billed at the full input rate).
_USAGE_PASSTHROUGH_INT_KEYS = (
    "cache_read_input_tokens",      # Anthropic: tokens served from prompt cache
    "cache_creation_input_tokens",  # Anthropic: tokens written to prompt cache
    "cached_tokens",                # OpenAI prompt_tokens_details / Gemini cachedContentTokenCount
    "thoughts_tokens",              # Gemini thoughtsTokenCount (already folded into completion_tokens)
)
_USAGE_PASSTHROUGH_STR_KEYS = (
    "service_tier",                 # "batch" marks Batch-API results (50% pricing)
    "batch_id",
    "provider_batch_id",
)


def usage_to_dict(usage_obj: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
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
        if isinstance(usage_obj, dict):
            for key in _USAGE_PASSTHROUGH_INT_KEYS:
                if usage_obj.get(key) is not None:
                    try:
                        base[key] = int(usage_obj[key] or 0)
                    except (TypeError, ValueError):
                        continue
            for key in _USAGE_PASSTHROUGH_STR_KEYS:
                value = usage_obj.get(key)
                if value:
                    base[key] = str(value)
    except Exception:
        return base
    return base


def resolve_price_per_1m(model_id: str) -> Optional[tuple[float, float]]:
    """Return (input, output) USD cost per 1,000,000 (1M) tokens for *model_id*.

    Single lookup path into pythia/model_costs.json (plus the MODEL_COSTS_JSON
    env override). Accepts plain ids ("gpt-5.4") and provider-prefixed forms
    ("openai/gpt-5.4"). Returns None when the model has no cost entry.
    """

    if not model_id:
        return None
    original = str(model_id).strip()
    if not original:
        return None
    normalized = original.lower()
    prices: Optional[tuple[float, float]] = MODEL_PRICES_PER_1M.get(normalized)
    if not prices:
        alt_ids = [normalized.replace("/", "-"), normalized.split("/", 1)[-1]]
        for alt in alt_ids:
            if alt in MODEL_PRICES_PER_1M:
                prices = MODEL_PRICES_PER_1M[alt]
                break
    if prices:
        return float(prices[0]), float(prices[1])

    # Fallback to JSON overrides if provided via MODEL_COSTS_JSON
    # (same convention: {"prompt": ..., "completion": ...} in USD per 1M tokens)
    dynamic_prices = _load_model_prices()
    price_entry = (
        dynamic_prices.get(normalized)
        or dynamic_prices.get(original)
        or dynamic_prices.get(normalized.replace("/", "-"))
        or dynamic_prices.get(normalized.split("/", 1)[-1])
        or {}
    )
    if not price_entry:
        return None
    try:
        prompt_rate = float(price_entry.get("prompt", 0.0))
        completion_rate = float(price_entry.get("completion", 0.0))
        return prompt_rate, completion_rate
    except Exception:
        return None


# Provider-default cache pricing multipliers (fraction of the input rate),
# used when model_costs.json has no explicit cached-rate fields. Verified
# against provider pricing pages 2026-07: Anthropic cache reads 0.1x, 5-min
# cache writes 1.25x, 1h writes 2x; OpenAI gpt-5-family cached input 0.1x
# (no write premium — caching is automatic); Gemini implicit cached tokens
# 0.25x (no write premium).
_CACHE_RATE_DEFAULTS = (
    ("claude", {"cached_input": 0.10, "cache_write_5m": 1.25, "cache_write_1h": 2.0}),
    ("gpt", {"cached_input": 0.10, "cache_write_5m": 1.0, "cache_write_1h": 1.0}),
    ("gemini", {"cached_input": 0.25, "cache_write_5m": 1.0, "cache_write_1h": 1.0}),
)

# Batch API discount: flat 50% off all token charges on OpenAI, Anthropic and
# Gemini batch endpoints alike. Applied when usage carries service_tier="batch".
BATCH_PRICE_MULTIPLIER = 0.5


def resolve_price_detail(model_id: str) -> Optional[Dict[str, float]]:
    """Return the full per-1M price record for *model_id*.

    Keys: ``input``, ``output``, ``cached_input``, ``cache_write_5m``,
    ``cache_write_1h`` (all USD per 1M tokens). Cache fields come from the
    object form in model_costs.json when present, otherwise from
    provider-default multipliers on the input rate. Returns None when the
    model has no cost entry at all (same semantics as resolve_price_per_1m).
    """

    pair = resolve_price_per_1m(model_id)
    if not pair:
        return None
    input_rate, output_rate = pair

    normalized = str(model_id or "").strip().lower()
    detail: Dict[str, float] = {}
    for key in (normalized, normalized.replace("/", "-"), normalized.split("/", 1)[-1]):
        if key in _MODEL_PRICE_CACHE_DETAILS:
            detail = dict(_MODEL_PRICE_CACHE_DETAILS[key])
            break

    bare_model = normalized.split("/", 1)[-1]
    multipliers: Dict[str, float] = {"cached_input": 1.0, "cache_write_5m": 1.0, "cache_write_1h": 1.0}
    for prefix, defaults in _CACHE_RATE_DEFAULTS:
        if bare_model.startswith(prefix):
            multipliers = dict(defaults)
            break

    return {
        "input": input_rate,
        "output": output_rate,
        "cached_input": detail.get("cached_input", input_rate * multipliers["cached_input"]),
        "cache_write_5m": detail.get("cache_write_5m", input_rate * multipliers["cache_write_5m"]),
        "cache_write_1h": detail.get("cache_write_1h", input_rate * multipliers["cache_write_1h"]),
    }


def compute_cost_split_usd(model_id: str, usage: Dict[str, Any]) -> tuple[float, float, float]:
    """Return (input_cost, output_cost, total_cost) USD for a usage dict.

    The single cost formula shared by ``estimate_cost_usd`` (providers) and
    ``_compute_costs_for_usage`` (llm_logging) so the two can never drift.

    Invariant: ``prompt_tokens`` is the TOTAL prompt size (adapters normalize
    Anthropic's cache-exclusive ``input_tokens`` back to the total). Cached
    portions are priced at the cached rate, Anthropic cache writes at the
    write-premium rate (5-min TTL assumed — the only TTL Pythia requests),
    and ``service_tier == "batch"`` halves everything.
    """

    if not usage or not isinstance(usage, dict):
        return 0.0, 0.0, 0.0

    prices = resolve_price_detail(model_id)
    if not prices:
        return 0.0, 0.0, 0.0

    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
    cached = int(usage.get("cached_tokens", 0) or 0)

    if not (prompt_tokens or completion_tokens):
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        total = (total_tokens / 1_000_000.0) * prices["input"]
        if str(usage.get("service_tier") or "") == "batch":
            total *= BATCH_PRICE_MULTIPLIER
        return float(total), 0.0, float(total)

    uncached = max(0, prompt_tokens - cache_read - cache_creation - cached)
    input_cost = (
        (uncached / 1_000_000.0) * prices["input"]
        + ((cached + cache_read) / 1_000_000.0) * prices["cached_input"]
        # Writes are priced at the 5m rate BY DESIGN: Pythia only ever
        # requests {"type": "ephemeral"} (5m TTL). If a 1h-TTL cache is
        # ever added, this line must branch on the TTL or 1h writes will
        # underbill 1.6x — cache_write_1h exists in model_costs.json /
        # resolve_price_detail for that future, not for today's math.
        + (cache_creation / 1_000_000.0) * prices["cache_write_5m"]
    )
    output_cost = (completion_tokens / 1_000_000.0) * prices["output"]

    if str(usage.get("service_tier") or "") == "batch":
        input_cost *= BATCH_PRICE_MULTIPLIER
        output_cost *= BATCH_PRICE_MULTIPLIER

    return float(input_cost), float(output_cost), float(input_cost + output_cost)


def estimate_cost_usd(model_id: str, usage: Dict[str, int]) -> float:
    _, _, total = compute_cost_split_usd(model_id, usage)
    return total


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------


def _is_gemini_3_family(model_id: str) -> bool:
    """True for Gemini 3.x model ids (gemini-3-*, gemini-3.1-*, gemini-3.5-*, ...)."""

    return (model_id or "").lower().startswith(("gemini-3-", "gemini-3."))


def _google_model_family(model_id: str) -> Optional[str]:
    """Classify a Gemini 3.x model id as "flash" or "pro".

    Used for the per-family SPD timeout / thinking-level env fallbacks.
    Substring-based so version bumps (gemini-3-flash-preview ->
    gemini-3.5-flash) don't silently disable the env overrides.
    """

    mid = (model_id or "").lower()
    if not _is_gemini_3_family(mid):
        return None
    if "flash" in mid:
        return "flash"
    if "pro" in mid:
        return "pro"
    return None


def build_openai_body(
    prompt: str,
    model: str,
    temperature: float,
    *,
    reasoning_effort: Optional[str] = None,
    prompt_cache_key: Optional[str] = None,
) -> dict:
    """Build the exact /v1/chat/completions request body.

    Shared by the sync path (call_openai) and the Batch API layer
    (pythia.llm_batch) so batch payloads are byte-identical to sync payloads.
    Never rebuild this dict at a call site — params silently drop at that seam.
    """

    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if reasoning_effort and reasoning_effort not in ("none", "off"):
        body["reasoning_effort"] = reasoning_effort
    elif not _openai_drops_temperature(model):
        body["temperature"] = float(temperature)
    # Routing hint only (caching itself is automatic for >=1024-token
    # prefixes); gated so legacy requests stay byte-identical when off.
    if prompt_cache_key and _prompt_cache_enabled():
        body["prompt_cache_key"] = prompt_cache_key
    return body


def _openai_usage_from_payload(payload: Any) -> Dict[str, Any]:
    """Normalize an OpenAI chat-completions usage payload (incl. cached tokens)."""

    usage_raw = payload.get("usage") if isinstance(payload, dict) else None
    usage = usage_to_dict(usage_raw)
    if isinstance(usage_raw, dict):
        details = usage_raw.get("prompt_tokens_details")
        if isinstance(details, dict) and details.get("cached_tokens"):
            try:
                usage["cached_tokens"] = int(details["cached_tokens"] or 0)
            except (TypeError, ValueError):
                pass
    return usage


def call_openai(
    prompt: str,
    model: str,
    temperature: float,
    *,
    reasoning_effort: Optional[str] = None,
    prompt_cache_key: Optional[str] = None,
) -> ProviderResult:
    if not _OPENAI_API_KEY:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error="missing OPENAI_API_KEY")
    body = build_openai_body(
        prompt,
        model,
        temperature,
        reasoning_effort=reasoning_effort,
        prompt_cache_key=prompt_cache_key,
    )
    try:
        resp = requests.post(
            f"{_OPENAI_BASE_URL.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {_OPENAI_API_KEY}", "Content-Type": "application/json"},
            json=body,
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
    usage = _openai_usage_from_payload(payload)
    return ProviderResult(text=text, usage=usage, cost_usd=0.0, model_id=model)


def _anthropic_stop_reason_error(payload: Any) -> Optional[str]:
    """Return an error string when Anthropic's ``stop_reason`` means "no usable answer".

    Two HTTP-200 outcomes produce empty or partial content and would otherwise be
    indistinguishable from a successful empty response:

    * ``refusal`` — a safety classifier declined the request. ``stop_details``
      carries the category (e.g. ``cyber``), which is worth logging: Pythia
      forecasts armed conflict and fatalities, so this is a plausible outcome
      rather than a theoretical one.
    * ``max_tokens`` — the answer hit the output ceiling. From Opus 5 adaptive
      thinking shares that ceiling, so this is the truncation signal.

    Returns ``None`` for every other stop reason (including ``end_turn``).
    """

    if not isinstance(payload, dict):
        return None
    stop_reason = payload.get("stop_reason")
    if stop_reason == "refusal":
        category = ""
        details = payload.get("stop_details")
        if isinstance(details, dict):
            category = str(details.get("category") or "").strip()
        suffix = f" (category={category})" if category else ""
        return f"Anthropic refusal: request declined by safety classifiers{suffix}"
    if stop_reason == "max_tokens":
        return (
            "Anthropic response truncated at max_tokens "
            "(raise PYTHIA_ANTHROPIC_SPD_MAX_TOKENS; adaptive thinking shares this budget)"
        )
    return None


def _prompt_cache_enabled() -> bool:
    """Gate for explicit prompt-cache markers (PYTHIA_PROMPT_CACHE_ENABLED).

    Only affects Anthropic cache_control blocks and the OpenAI
    prompt_cache_key routing hint; Gemini implicit caching is automatic and
    has no request knob. Default OFF (byte-identical legacy requests).
    """

    return os.getenv("PYTHIA_PROMPT_CACHE_ENABLED", "0").strip().lower() in ("1", "true", "yes")


# Minimum prompt-prefix size a cache-marked span must reach before we attach
# cache_control. Anthropic silently declines to cache a shorter prefix (no
# error, just cache_creation_input_tokens: 0), so a marker below the minimum
# is dead weight rather than a cost — but refusing to mark a span that WOULD
# have cached is lost money, which is what a single conservative constant was
# doing: it was set for a 1024-token minimum while claude-opus-5 (the only
# Anthropic model on the forecast path) caches from 512, so every binary_v2
# prefix (~760 tokens) was silently left unmarked.
#
# The minimum is per-model and NOT monotonic across generations — 512 on the
# newest models, still 4096 on Opus 4.6/4.5 and Haiku 4.5 — so one constant is
# wrong in one direction or the other for every model it doesn't describe.
# Keyed by literal id prefix, like the sampling-param guards above; a new
# Anthropic model matching nothing falls back to the most conservative value,
# which caches less but never mis-marks.
_ANTHROPIC_CACHE_MIN_TOKENS: tuple[tuple[str, int], ...] = (
    ("claude-opus-5", 512),
    ("claude-fable-5", 512),
    ("claude-mythos-5", 512),
    ("claude-opus-4-8", 1024),
    ("claude-sonnet-5", 1024),
    ("claude-sonnet-4-6", 1024),
    ("claude-sonnet-4-5", 1024),
    ("claude-opus-4-7", 2048),
    ("claude-opus-4-6", 4096),
    ("claude-opus-4-5", 4096),
    ("claude-haiku-4-5", 4096),
)
_ANTHROPIC_CACHE_MIN_TOKENS_DEFAULT = 4096

# ~4 chars/token. The comparison is on characters because the body is built
# before any tokenizer runs and count_tokens would be a network round trip per
# request on the hot path.
_ANTHROPIC_CHARS_PER_TOKEN = 4


def _anthropic_cache_min_chars(model: str) -> int:
    """Smallest cache-marked span worth a breakpoint, in characters.

    ``PYTHIA_ANTHROPIC_CACHE_MIN_CHARS`` overrides the table outright — the
    escape hatch for a model we don't yet list.
    """

    override = os.getenv("PYTHIA_ANTHROPIC_CACHE_MIN_CHARS", "").strip()
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    model_l = (model or "").lower()
    for prefix, min_tokens in _ANTHROPIC_CACHE_MIN_TOKENS:
        if model_l.startswith(prefix):
            return min_tokens * _ANTHROPIC_CHARS_PER_TOKEN
    return _ANTHROPIC_CACHE_MIN_TOKENS_DEFAULT * _ANTHROPIC_CHARS_PER_TOKEN


def build_anthropic_body(
    prompt: str,
    model: str,
    temperature: float,
    *,
    purpose: str | None = None,
    cache_segments: Optional[List[tuple]] = None,
    cache_ttl: Optional[str] = None,
    max_tokens_override: Optional[int] = None,
    thinking_level: Optional[str] = None,
) -> dict:
    """Build the exact /v1/messages request body.

    Shared by the sync path (call_anthropic) and the Batch API layer
    (pythia.llm_batch: each batch item's ``params`` is exactly this body) so
    batch payloads are byte-identical to sync payloads. Carries the SPD/binary
    max_tokens switch and the no-sampling-params model rule.

    ``cache_segments`` is an ordered list of ``(text, is_breakpoint)`` tuples
    whose concatenation must equal ``prompt``. When provided AND
    PYTHIA_PROMPT_CACHE_ENABLED is on, the user message is sent as multiple
    text blocks with ``cache_control: {type: ephemeral}`` on breakpoint blocks
    (max 4 per request, enforced here). Otherwise the plain single-string
    message is sent — byte-identical to the legacy request.

    ``cache_ttl`` sets the cache entry's lifetime: ``None`` (default) leaves
    the bare ``{"type": "ephemeral"}`` marker, i.e. Anthropic's 5-minute TTL
    and 1.25x write premium; ``"1h"`` buys an hour at a 2x write premium. The
    batch path passes "1h" because a provider batch routinely outlives five
    minutes — the 2026-08-01 SPD batch ran 41 — so an entry written by the
    first item is gone long before the last item is processed. Only the TTL
    differs between sync and batch bodies; the prompt bytes are identical.
    """

    # Use higher max_tokens for SPD/binary forecast calls to avoid truncation.
    # sibyl_step gets the same raised ceiling: Sibyl runs Opus 5 (adaptive
    # thinking shares max_tokens) and each step answer carries a full 7-level
    # quantile JSON plus prose — the 16k default truncates exactly like SPD.
    max_tokens = _ANTHROPIC_MAX_OUTPUT
    if purpose in ("spd_v2", "binary_v2", "sibyl_step"):
        max_tokens = _ANTHROPIC_SPD_MAX_OUTPUT
    # An explicit caller budget wins over the purpose-derived defaults —
    # the hazard-extraction path passes the rulebook's
    # extraction.max_output_tokens here, so that knob is config, not prose.
    if max_tokens_override:
        max_tokens = int(max_tokens_override)

    content: Any = prompt
    if cache_segments and _prompt_cache_enabled():
        min_chars = _anthropic_cache_min_chars(model)
        cache_control: dict = {"type": "ephemeral"}
        if cache_ttl:
            cache_control["ttl"] = cache_ttl
        blocks: List[dict] = []
        marked = 0
        chars_so_far = 0
        for seg in cache_segments:
            text, is_breakpoint = seg[0], bool(seg[1])
            if not text:
                continue
            chars_so_far += len(text)
            block: dict = {"type": "text", "text": text}
            # Only mark breakpoints where the cumulative prefix is plausibly
            # above THIS model's cacheable minimum, and never more than 4.
            if is_breakpoint and marked < 4 and chars_so_far >= min_chars:
                block["cache_control"] = dict(cache_control)
                marked += 1
            blocks.append(block)
        if blocks:
            content = blocks

    body: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": float(temperature),
        "messages": [{"role": "user", "content": content}],
    }
    # Opus 4.7+ family models reject sampling params with HTTP 400.
    if model.lower().startswith(_ANTHROPIC_NO_TEMPERATURE_PREFIXES):
        body.pop("temperature", None)
    # Explicit thinking depth (ModelSpec.thinking / resolve_request_params),
    # emitted only for models verified to accept the effort knob — see
    # _ANTHROPIC_EFFORT_PREFIXES. Omitted entirely when unset, so every
    # existing body stays byte-identical. Thinking tokens share the
    # max_tokens budget: a caller raising effort above the default should
    # also raise max_tokens_override.
    if (
        thinking_level
        and thinking_level not in ("off", "none")
        and model.lower().startswith(_ANTHROPIC_EFFORT_PREFIXES)
    ):
        body["output_config"] = {"effort": str(thinking_level)}
    return body


def _anthropic_usage_from_payload(payload: Any) -> Dict[str, Any]:
    """Normalize an Anthropic usage payload.

    Anthropic's ``input_tokens`` EXCLUDES cache reads/writes; Pythia's invariant
    is that ``prompt_tokens`` = total prompt size, so the cache counts are added
    back and preserved as separate fields for cache-aware pricing.
    """

    usage_raw = payload.get("usage") if isinstance(payload, dict) else None
    if not isinstance(usage_raw, dict):
        usage_raw = {}
    input_tokens = int(usage_raw.get("input_tokens") or 0)
    output_tokens = int(usage_raw.get("output_tokens") or 0)
    cache_read = int(usage_raw.get("cache_read_input_tokens") or 0)
    cache_creation = int(usage_raw.get("cache_creation_input_tokens") or 0)
    prompt_tokens = input_tokens + cache_read + cache_creation
    usage_dict: Dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": prompt_tokens + output_tokens,
    }
    if cache_read:
        usage_dict["cache_read_input_tokens"] = cache_read
    if cache_creation:
        usage_dict["cache_creation_input_tokens"] = cache_creation
    return usage_to_dict(usage_dict)


def call_anthropic(
    prompt: str,
    model: str,
    temperature: float,
    *,
    purpose: str | None = None,
    cache_segments: Optional[List[tuple]] = None,
    max_tokens_override: Optional[int] = None,
    timeout_sec: Optional[float] = None,
    thinking_level: Optional[str] = None,
) -> ProviderResult:
    if not _ANTHROPIC_API_KEY:
        return ProviderResult("", usage_to_dict(None), 0.0, model, error="missing ANTHROPIC_API_KEY")
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": _ANTHROPIC_API_KEY,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    body = build_anthropic_body(
        prompt, model, temperature, purpose=purpose, cache_segments=cache_segments,
        max_tokens_override=max_tokens_override, thinking_level=thinking_level,
    )
    try:
        resp = requests.post(
            url, headers=headers, json=body, timeout=timeout_sec or _ANTHROPIC_TIMEOUT
        )
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
                # Only "text" blocks are the answer. Adaptive thinking (on by
                # default from Opus 5) also emits "thinking" blocks — skipping
                # them here is deliberate, not an oversight.
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
            text = "".join(parts).strip()

    usage = _anthropic_usage_from_payload(payload)

    # A safety refusal or a thinking-induced truncation both come back as HTTP 200
    # with empty or partial content. Without naming them, the member simply drops
    # out of the ensemble with no error recorded anywhere. Usage is preserved on
    # the error result — a truncated call still burned tokens and must still be
    # costed.
    stop_error = _anthropic_stop_reason_error(payload)
    if stop_error:
        return ProviderResult("", usage, 0.0, model, error=stop_error)

    return ProviderResult(text=text, usage=usage, cost_usd=0.0, model_id=model)


def build_google_body(
    prompt: str,
    model: str,
    temperature: float,
    *,
    thinking_level: Optional[str] = None,
) -> dict:
    """Build the exact generateContent request body.

    Shared by the sync path (call_google) and the Batch API layer
    (pythia.llm_batch) so batch payloads are byte-identical to sync payloads.
    Carries the Gemini-3.x-only thinkingConfig gate.
    """

    api_model = model.split("/", 1)[-1] if "/" in model else model
    generation_config: Dict[str, Any] = {"temperature": float(temperature)}
    # thinkingLevel is supported by the Gemini 3.x families (gemini-3-*,
    # gemini-3.1-*, gemini-3.5-*); older 2.5 models use a different knob
    # (thinkingBudget) and would reject this config.
    if thinking_level and _is_gemini_3_family(api_model):
        generation_config["thinkingConfig"] = {"thinkingLevel": thinking_level}
    return {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }


def _google_usage_from_payload(payload: Any) -> Dict[str, Any]:
    """Normalize a Gemini usageMetadata payload (incl. implicit-cache tokens)."""

    usage_meta = payload.get("usageMetadata") if isinstance(payload, dict) else None
    if not isinstance(usage_meta, dict):
        usage_meta = {}
    # thoughtsTokenCount (thinking tokens) is billed at the OUTPUT rate but
    # reported SEPARATELY from candidatesTokenCount — fold it into
    # completion_tokens or every thinking call underbills its output spend
    # (both Gemini SPD members and all HS RC/triage calls run thinking).
    completion = int(usage_meta.get("candidatesTokenCount") or 0)
    thoughts = int(usage_meta.get("thoughtsTokenCount") or 0)
    usage_dict: Dict[str, Any] = {
        "prompt_tokens": usage_meta.get("promptTokenCount", 0),
        "completion_tokens": completion + thoughts,
        "total_tokens": usage_meta.get("totalTokenCount", 0),
    }
    if thoughts:
        usage_dict["thoughts_tokens"] = thoughts
    if usage_meta.get("cachedContentTokenCount"):
        usage_dict["cached_tokens"] = usage_meta.get("cachedContentTokenCount")
    return usage_to_dict(usage_dict)


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
    body = build_google_body(prompt, model, temperature, thinking_level=thinking_level)
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
    usage = _google_usage_from_payload(payload)
    return ProviderResult(text=text, usage=usage, cost_usd=0.0, model_id=model)


def resolve_request_params(ms: "ModelSpec", temperature: float) -> tuple[float, Optional[str]]:
    """Resolve (effective_temperature, thinking_level) for a model spec.

    The single source of the per-model param rules used by call_chat_ms AND
    the Batch API layer (build_body_for_spec): ModelSpec.thinking from
    config wins; Google SPD members fall back to the
    PYTHIA_GOOGLE_SPD_THINKING_LEVEL_* env vars; empty/off/none → None;
    ModelSpec.temperature overrides the call-site temperature.
    """

    spd_google = ms.provider == "google" and ms.purpose == "spd_v2"
    thinking_level: Optional[str] = None
    if ms.thinking and ms.thinking not in ("off", "none"):
        thinking_level = ms.thinking
    elif spd_google:
        google_family = _google_model_family(ms.model_id)
        if google_family == "flash":
            thinking_level = (os.getenv("PYTHIA_GOOGLE_SPD_THINKING_LEVEL_FLASH", "low") or "").strip()
        elif google_family == "pro":
            thinking_level = (os.getenv("PYTHIA_GOOGLE_SPD_THINKING_LEVEL_PRO", "") or "").strip()
    if thinking_level in ("", "off", "none"):
        thinking_level = None
    effective_temperature = ms.temperature if ms.temperature is not None else temperature
    return effective_temperature, thinking_level


def _batch_prompt_cache_enabled() -> bool:
    """Gate for cache_control blocks in BATCH bodies (code default OFF).

    Anthropic cache hits inside a batch are best-effort — items fan out
    across nodes, and a cache entry only becomes readable once the first
    response has begun, so co-scheduled items cannot share one. The write
    premium, by contrast, is charged whether or not a read ever lands. That
    asymmetry is why this is a flag rather than always-on.

    Set to "1" in pythia_pipeline_stage.yml since 2026-08-03: with it off the
    hit rate on the batched path was 0% BY CONSTRUCTION (the 2026-08-01 run
    logged zero cache reads across all 99 Anthropic calls), so the caution
    could never be tested. The downside if reads don't materialize is the
    write premium on the shared prefix alone — ~20.6k tokens, about $0.06 per
    cycle. Verify with batch_economics.py / post_run_diagnostics.py, both of
    which already read cache_read_input_tokens; revert this flag to "0" if
    reads stay at zero.

    OpenAI's automatic-cache discount does not apply to its Batch API at all,
    so prompt_cache_key is never emitted in batch bodies regardless.
    """

    return os.getenv("PYTHIA_BATCH_PROMPT_CACHE", "0").strip().lower() in ("1", "true", "yes")


# A provider batch routinely outlives Anthropic's 5-minute default TTL (the
# 2026-08-01 SPD batch ran 41 minutes end to end), which would leave the
# prefix written by the first item expired before the last item is scheduled.
# 1h costs a 2x write premium instead of 1.25x — ~$0.04 more per cycle at
# current prefix sizes, against a read that would otherwise never land.
# Set PYTHIA_BATCH_CACHE_TTL=none to fall back to the 5-minute default.
# Read per call, not at import, so it stays monkeypatchable and can't be
# frozen by import order — same contract as _batch_prompt_cache_enabled().
def _batch_cache_ttl() -> Optional[str]:
    raw = (os.getenv("PYTHIA_BATCH_CACHE_TTL", "") or "").strip()
    if raw.lower() == "none":
        return None
    return raw or "1h"


def build_body_for_spec(
    ms: "ModelSpec",
    prompt: str,
    temperature: float = 0.2,
    *,
    cache_prefix: Optional[str] = None,
    prompt_cache_key: Optional[str] = None,
) -> dict:
    """Build the exact provider request body a sync call_chat_ms would send.

    Single entry point for the Batch API layer: dispatches to the shared
    per-provider body builders with the same resolved params as the sync
    path, so a batch item's body is byte-identical to the sync request.

    The cache args keep sync/batch parity now that the sync SPD path sends
    cache_control segments: with PYTHIA_BATCH_PROMPT_CACHE unset (code
    default) batch bodies stay plain-string (see _batch_prompt_cache_enabled);
    ``prompt_cache_key`` is accepted for signature parity but deliberately
    never emitted (no OpenAI batch discount exists for it). When batch caching
    IS on, the only divergence from the sync body is the cache_control TTL —
    the prompt bytes are unchanged, so a replayed batch result is still the
    answer to the same prompt a sync call would have sent.
    """

    del prompt_cache_key  # never emitted in batch bodies — see docstring
    effective_temperature, thinking_level = resolve_request_params(ms, temperature)
    provider = (ms.provider or "").lower()
    if provider == "openai":
        return build_openai_body(
            prompt, ms.model_id, effective_temperature, reasoning_effort=thinking_level
        )
    if provider == "anthropic":
        cache_segments = None
        if (
            _batch_prompt_cache_enabled()
            and cache_prefix
            and prompt.startswith(cache_prefix)
        ):
            cache_segments = [(cache_prefix, True), (prompt[len(cache_prefix):], False)]
        return build_anthropic_body(
            prompt, ms.model_id, effective_temperature, purpose=ms.purpose,
            cache_segments=cache_segments,
            cache_ttl=_batch_cache_ttl() if cache_segments else None,
            thinking_level=thinking_level,
        )
    if provider in {"google", "gemini"}:
        return build_google_body(
            prompt, ms.model_id, effective_temperature, thinking_level=thinking_level
        )
    raise ValueError(f"unsupported provider for batch body: {ms.provider}")


def _call_provider_sync(
    provider: str,
    prompt: str,
    model: str,
    temperature: float,
    *,
    timeout_sec: Optional[float] = None,
    thinking_level: Optional[str] = None,
    purpose: str | None = None,
    cache_segments: Optional[List[tuple]] = None,
    prompt_cache_key: Optional[str] = None,
    max_tokens_override: Optional[int] = None,
) -> ProviderResult:
    p = (provider or "").lower()
    if p == "openai":
        return call_openai(
            prompt, model, temperature,
            reasoning_effort=thinking_level,
            prompt_cache_key=prompt_cache_key,
        )
    if p == "anthropic":
        return call_anthropic(
            prompt, model, temperature, purpose=purpose, cache_segments=cache_segments,
            max_tokens_override=max_tokens_override, timeout_sec=timeout_sec,
            thinking_level=thinking_level,
        )
    if p in {"google", "gemini"}:
        return call_google(prompt, model, temperature, timeout_sec=timeout_sec, thinking_level=thinking_level)
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
    log_call: bool = True,
    cache_segments: Optional[List[tuple]] = None,
    prompt_cache_key: Optional[str] = None,
    timeout_sec: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
) -> tuple[str, Dict[str, int], str]:
    """Call the configured provider for a model spec and return (text, usage, error).

    log_call: pass False from call sites that write their own rich llm_calls
    row (log_hs_llm_call / log_forecaster_llm_call). The generic row written
    here has no phase/iso3/run linkage, so leaving it on next to a rich row
    double-counts the call's cost in phase-less aggregations (the Costs page
    "other" bucket) — see the July 2026 telemetry fix.

    cache_segments / prompt_cache_key: optional prompt-cache plumbing
    (Anthropic cache_control content blocks / OpenAI routing hint). Both are
    inert unless PYTHIA_PROMPT_CACHE_ENABLED=1; providers that don't use a
    given knob ignore it. ``"".join(seg[0] for seg in cache_segments)`` must
    equal ``prompt`` — the plain string is what retries and logging use.

    timeout_sec / max_output_tokens: explicit per-call overrides. A caller
    timeout wins over the purpose-derived defaults below and is enforced
    both at the async layer (asyncio.wait_for) and, for Anthropic, at the
    transport. max_output_tokens currently applies to Anthropic bodies only
    (the hazard-extraction path passes the rulebook's declared budget so
    that knob is real config, not prose); other providers keep their
    purpose-derived ceilings.
    """

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
    caller_timeout = timeout_sec is not None
    hs_timeout_sec: Optional[float] = None
    hs_max_retry_after_sec: Optional[float] = None
    hs_fail_fast_on_retry_after = False
    hs_max_attempts: Optional[int] = None
    hs_usage: Dict[str, Any] = {}
    backoffs_sec: list[float] = []

    # Per-model temperature/thinking resolution — shared with the Batch API
    # layer via resolve_request_params so batch bodies can't drift.
    effective_temperature, thinking_level = resolve_request_params(ms, temperature)

    if spd_google:
        google_family = _google_model_family(ms.model_id)
        if not caller_timeout:
            if google_family == "flash":
                timeout_sec = _resolve_timeout("PYTHIA_GOOGLE_SPD_TIMEOUT_FLASH_SEC", None, 300.0)
            elif google_family == "pro":
                timeout_sec = _resolve_timeout("PYTHIA_GOOGLE_SPD_TIMEOUT_PRO_SEC", None, 300.0)
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
            if not caller_timeout:
                timeout_sec = hs_timeout_sec
        if hs_max_attempts is not None:
            max_attempts = min(max_attempts, hs_max_attempts)
        if hs_timeout_sec is not None:
            hs_usage["hs_timeout_sec"] = hs_timeout_sec
    attempt = 0
    result: Optional[ProviderResult] = None
    error: Optional[str] = None

    # Credit-retry outer loop: wraps the transient-retry inner loop
    credit_cfg = _credit_retry_config_for(ms.provider)
    credit_max_retries = credit_cfg[1] if credit_cfg else 0
    credit_pause_sec = credit_cfg[0] if credit_cfg else 0
    credit_retries_used = 0
    credit_retry_pauses: list[float] = []
    billing_error_detected = False
    # Token spend of attempts discarded by a retry — previously only the
    # FINAL attempt's usage was costed, so N−1 retried calls were billed by
    # the provider but never reached the ledger.
    retried_prompt_tokens = 0
    retried_completion_tokens = 0

    for credit_attempt in range(credit_max_retries + 1):
        # Reset inner loop state for each credit-retry attempt
        attempt = 0
        result = None
        error = None
        backoffs_sec = []

        while attempt < max_attempts:
            attempt += 1
            try:
                async with _get_llm_semaphore():
                    call_task = asyncio.to_thread(
                        _call_provider_sync,
                        ms.provider,
                        prompt,
                        ms.model_id,
                        effective_temperature,
                        timeout_sec=timeout_sec,
                        thinking_level=thinking_level,
                        purpose=ms.purpose,
                        cache_segments=cache_segments,
                        prompt_cache_key=prompt_cache_key,
                        max_tokens_override=max_output_tokens,
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

            # This attempt's result is about to be discarded for a retry —
            # bank its token usage so the final ledger row still counts it.
            if result is not None and result.usage:
                retried_prompt_tokens += int(result.usage.get("prompt_tokens") or 0)
                retried_completion_tokens += int(result.usage.get("completion_tokens") or 0)

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

        # Inner loop done — check if we succeeded or hit a billing error
        if result and not result.error:
            if credit_retries_used > 0:
                total_pause = sum(credit_retry_pauses)
                LOGGER.info(
                    "[CREDIT_RETRY] %s recovered after %.0fs pause (%d credit retries)",
                    ms.provider, total_pause, credit_retries_used,
                )
            break

        # Check for billing error
        err_text = error or (result.error if result else "") or ""
        err_status = _extract_status_code(err_text)
        if _is_billing_error(ms.provider, err_text, err_status):
            billing_error_detected = True
            if credit_attempt < credit_max_retries:
                credit_retries_used += 1
                LOGGER.warning(
                    "[CREDIT_RETRY] %s billing error detected. Pausing %ds before retry %d/%d. Error: %s",
                    ms.provider, credit_pause_sec, credit_retries_used, credit_max_retries,
                    err_text[:200],
                )
                credit_retry_pauses.append(float(credit_pause_sec))
                await asyncio.sleep(credit_pause_sec)
                continue
            else:
                total_pause = sum(credit_retry_pauses)
                LOGGER.error(
                    "[CREDIT_RETRY] %s billing error persists after %d credit retries (%.0fs total pause). Giving up.",
                    ms.provider, credit_max_retries, total_pause,
                )
        break  # Not a billing error, or retries exhausted

    if result is None:
        result = ProviderResult("", usage_to_dict(None), 0.0, ms.model_id, error=error or "unknown error")

    error = result.error if result and result.error else None

    elapsed_ms = int((time.time() - start) * 1000)
    usage = result.usage or usage_to_dict(None)
    usage["attempts_used"] = attempt
    usage["backoffs_sec"] = backoffs_sec
    if retried_prompt_tokens or retried_completion_tokens:
        # Fold discarded-attempt spend into the billed totals (the provider
        # charged for those calls); keep the breakdown for diagnostics.
        usage["retried_prompt_tokens"] = retried_prompt_tokens
        usage["retried_completion_tokens"] = retried_completion_tokens
        usage["prompt_tokens"] = int(usage.get("prompt_tokens") or 0) + retried_prompt_tokens
        usage["completion_tokens"] = int(usage.get("completion_tokens") or 0) + retried_completion_tokens
        usage["total_tokens"] = (
            int(usage.get("total_tokens") or 0)
            + retried_prompt_tokens
            + retried_completion_tokens
        )
    if credit_retries_used > 0 or billing_error_detected:
        usage["credit_retries_used"] = credit_retries_used
        usage["credit_retry_pauses_sec"] = credit_retry_pauses
        usage["billing_error_detected"] = True
    if hs_usage:
        usage.update(hs_usage)
    cost = result.cost_usd if result.cost_usd else estimate_cost_usd(ms.model_id, usage)
    if log_call:
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
        if "Anthropic refusal:" in error or "truncated at max_tokens" in error:
            # Content-driven outcomes (safety classifier / output ceiling),
            # not provider health — feeding them to the breaker would let 6
            # consecutive refusals trip a whole-provider cooldown and drop
            # the Claude member from unrelated questions.
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
