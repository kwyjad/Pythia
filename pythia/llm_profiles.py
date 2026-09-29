# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Central LLM model resolution.

All model choices flow through this module:

- ``llm.models`` in config.yaml is the MODEL REGISTRY: one alias per model
  family (e.g. ``gpt: openai:gpt-6-sol``). Swapping a family means editing that
  one line (plus a cost entry in pythia/model_costs.json).
- ``llm.profiles.<profile>.ensemble`` lists SPD ensemble members, each
  referencing a registry alias via ``model:`` (legacy ``provider:model_id``
  strings and ``{provider, model_id}`` dicts still work).
- ``llm.profiles.<profile>.roles`` maps every non-ensemble purpose (HS triage
  passes, RC passes, Track 2, scenario writer, grounding fallbacks, ...) to a
  registry alias or explicit ``provider:model_id`` ref.

Resolution order at every call site: purpose-specific env var (if any) >
config role > ``_ROLE_FALLBACKS`` below. ``_ROLE_FALLBACKS`` is the single
code-level fallback table — call sites must not contain model-id literals.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from pythia.config import load as load_cfg


# Last-resort defaults, used only when config.yaml is missing/unreadable or a
# role is absent from it. Keep in sync with the roles block in config.yaml.
_ROLE_FALLBACKS: Dict[str, str] = {
    "hs_default": "google:gemini-3.5-flash",
    "hs_triage_pass1": "google:gemini-3.5-flash",
    "hs_triage_pass2": "google:gemini-3.5-flash",
    "rc_pass1": "google:gemini-3.5-flash",
    "rc_pass2": "google:gemini-3.5-flash",
    "hs_fallback": "openai:gpt-6-sol",
    "track2_spd": "google:gemini-3.5-flash",
    "scenario_writer": "google:gemini-3.5-flash",
    "grounding_gemini": "google:gemini-2.5-flash",
    "grounding_openai": "openai:gpt-6-sol",
    "grounding_openai_fallback": "openai:gpt-6-luna",
    "grounding_claude": "anthropic:claude-haiku-4-5-20251001",
    "crisiswatch": "google:gemini-2.5-flash",
    "hazard_extraction": "anthropic:claude-haiku-4-5-20251001",
    "interpreter": "anthropic:claude-opus-5-5",
}


def get_current_profile() -> str:
    """
    Resolve the active LLM profile.

    Priority:
      1) PYTHIA_LLM_PROFILE env var (if set)
      2) pythia.config.yaml: llm.profile
      3) default "prod"
    """

    env_profile = os.getenv("PYTHIA_LLM_PROFILE")
    if env_profile:
        return env_profile.strip()

    cfg = load_cfg()
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    profile = (llm_cfg.get("profile") or "prod").strip()
    return profile


def _get_llm_cfg() -> Dict[str, Any]:
    cfg = load_cfg()
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    return llm_cfg if isinstance(llm_cfg, dict) else {}


def _get_profile_data() -> Dict[str, Any]:
    profiles = _get_llm_cfg().get("profiles") or {}
    profile_data = profiles.get(get_current_profile(), {})
    return profile_data if isinstance(profile_data, dict) else {}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_model_registry() -> Dict[str, str]:
    """Return the ``llm.models`` alias registry: {alias: "provider:model_id"}."""

    models = _get_llm_cfg().get("models") or {}
    registry: Dict[str, str] = {}
    if isinstance(models, dict):
        for alias, ref in models.items():
            if isinstance(ref, str) and ref.strip():
                registry[str(alias).strip()] = ref.strip()
    return registry


def resolve_model_ref(ref: Optional[str]) -> Optional[str]:
    """Resolve a registry alias or explicit ref to ``provider:model_id``.

    Returns None for empty input or an alias that isn't in the registry.
    """

    if not isinstance(ref, str):
        return None
    ref = ref.strip()
    if not ref:
        return None
    if ":" in ref:
        return ref
    resolved = get_model_registry().get(ref)
    if isinstance(resolved, str) and ":" in resolved:
        return resolved.strip()
    return None


def split_model_ref(ref: str, default_provider: str = "google") -> Tuple[str, str]:
    """Split ``provider:model_id`` into (provider, model_id).

    A bare model id is attributed to *default_provider* (legacy env-var
    values like ``HS_MODEL_ID=gemini-3.5-flash``).
    """

    if ":" in ref:
        provider, model_id = ref.split(":", 1)
        return provider.strip().lower(), model_id.strip()
    return default_provider, ref.strip()


def get_role_model(role: str) -> str:
    """Resolve a role name to ``provider:model_id``.

    Priority: profile ``roles`` block > legacy flat profile key >
    ``_ROLE_FALLBACKS``. Purpose-specific env vars are applied by call
    sites BEFORE calling this (env always wins).
    """

    try:
        profile_data = _get_profile_data()
        roles = profile_data.get("roles")
        raw = roles.get(role) if isinstance(roles, dict) else None
        if not raw:
            raw = profile_data.get(role)  # legacy flat purpose key
        resolved = resolve_model_ref(raw if isinstance(raw, str) else None)
        if resolved:
            return resolved
    except Exception:
        pass
    return _ROLE_FALLBACKS.get(role, "")


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def get_ensemble_list() -> list:
    """Return the raw ensemble list from the active profile, or empty list."""

    ensemble = _get_profile_data().get("ensemble")
    return ensemble if isinstance(ensemble, list) else []


def get_ensemble_resolved() -> List[Dict[str, Any]]:
    """Return the ensemble normalized to dicts with registry aliases resolved.

    Each item: {"provider": str, "model_id": str, "temperature"?: float,
    "thinking"?: str}. Entries that fail to resolve are dropped.

    Accepted entry formats:
      - ``{"model": "<alias-or-provider:model_id>", ...params}`` (preferred)
      - ``"provider:model_id"`` string (legacy)
      - ``{"provider": ..., "model_id": ..., ...params}`` (legacy)
    """

    resolved: List[Dict[str, Any]] = []
    for entry in get_ensemble_list():
        provider = model_id = ""
        params: Dict[str, Any] = {}
        if isinstance(entry, str):
            ref = resolve_model_ref(entry)
            if ref:
                provider, model_id = split_model_ref(ref)
        elif isinstance(entry, dict):
            if entry.get("model"):
                ref = resolve_model_ref(str(entry.get("model")))
                if ref:
                    provider, model_id = split_model_ref(ref)
            else:
                provider = str(entry.get("provider", "")).strip().lower()
                model_id = str(entry.get("model_id", "")).strip()
            params = {
                k: v
                for k, v in entry.items()
                if k not in {"model", "provider", "model_id"}
            }
        if not provider or not model_id:
            continue
        resolved.append({"provider": provider, "model_id": model_id, **params})
    return resolved


def get_current_models() -> Dict[str, str]:
    """
    Return the current profile's model IDs per provider (first model per
    provider in the ensemble).

    Example return:
      {"openai": "gpt-6-sol", "google": "gemini-3.1-pro-preview",
       "anthropic": "claude-opus-5-5"}
    """

    models: Dict[str, str] = {}
    for entry in get_ensemble_resolved():
        provider = entry["provider"]
        if provider not in models:
            models[provider] = entry["model_id"]
    if models:
        return models

    # Legacy format: flat provider→model_id dict
    profile_data = _get_profile_data()
    if profile_data and not profile_data.get("ensemble"):
        return {
            str(k): str(v)
            for k, v in profile_data.items()
            if v and isinstance(v, (str, int, float)) and k not in {"roles"}
        }
    return {}


def get_purpose_model(purpose: str) -> str | None:
    """Legacy accessor for purpose-specific models (e.g. ``hs_fallback``).

    Now resolves through the roles block + registry; returns
    ``provider:model_id`` or None.
    """

    return get_role_model(purpose) or None


def get_model_families() -> Dict[str, List[str]]:
    """Return ``llm.model_families``: family key -> model ids, oldest first.

    A family is one model line across versions (every Claude Opus, every
    GPT frontier model). Calibration uses it to carry a retired version's
    track record to its successor as a prior, so a version bump does not
    start the new model from nothing.
    """

    fams = _get_llm_cfg().get("model_families")
    out: Dict[str, List[str]] = {}
    if isinstance(fams, dict):
        for key, ids in fams.items():
            if isinstance(ids, list):
                out[str(key)] = [str(i).strip() for i in ids if str(i).strip()]
    return out


def model_family(model_id: Optional[str]) -> Optional[str]:
    """The family key a model id belongs to, or None."""

    if not model_id:
        return None
    mid = str(model_id).strip()
    for key, ids in get_model_families().items():
        if mid in ids:
            return key
    return None


def family_predecessors(model_id: Optional[str]) -> List[str]:
    """Earlier versions in the model's family, newest first."""

    fam = model_family(model_id)
    if not fam:
        return []
    ids = get_model_families().get(fam, [])
    idx = ids.index(str(model_id).strip())
    return list(reversed(ids[:idx]))


def voting_ensemble_model_ids() -> List[str]:
    """Model ids of the ensemble members that vote (shadow members excluded)."""

    out: List[str] = []
    for entry in get_ensemble_resolved():
        shadow = entry.get("shadow")
        if shadow is True or str(shadow).strip().lower() in {"true", "1", "yes"}:
            continue
        out.append(entry["model_id"])
    return out
