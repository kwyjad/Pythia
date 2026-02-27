# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import os
from typing import Dict

from pythia.config import load as load_cfg


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


def get_current_models() -> Dict[str, str]:
    """
    Return the current profile's model IDs per provider.

    Reads from the ``ensemble`` list in the active profile (new format) or
    falls back to the legacy flat provider→model_id mapping.

    Example return:
      {"openai": "gpt-5.1", "google": "gemini-3-flash-preview",
       "anthropic": "claude-sonnet-4-6"}
    """

    cfg = load_cfg()
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    profiles = llm_cfg.get("profiles") or {}
    profile = get_current_profile()
    profile_data = profiles.get(profile, {})

    # New format: ensemble list of "provider:model_id" strings
    ensemble = profile_data.get("ensemble") if isinstance(profile_data, dict) else None
    if isinstance(ensemble, list):
        models: Dict[str, str] = {}
        for entry in ensemble:
            entry_str = str(entry).strip()
            if ":" in entry_str:
                provider, model_id = entry_str.split(":", 1)
                provider = provider.strip().lower()
                model_id = model_id.strip()
                if provider and model_id and provider not in models:
                    models[provider] = model_id  # first model per provider
        return models

    # Legacy format: flat provider→model_id dict
    if isinstance(profile_data, dict):
        return {str(k): str(v) for k, v in profile_data.items() if v}
    return {}


def get_ensemble_list() -> list:
    """Return the raw ensemble list from the active profile, or empty list."""

    cfg = load_cfg()
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    profiles = llm_cfg.get("profiles") or {}
    profile = get_current_profile()
    profile_data = profiles.get(profile, {})
    if isinstance(profile_data, dict):
        ensemble = profile_data.get("ensemble")
        if isinstance(ensemble, list):
            return ensemble
    return []


def get_purpose_model(purpose: str) -> str | None:
    """Read a purpose-specific model override from the active profile.

    E.g. ``get_purpose_model("hs_fallback")`` reads
    ``llm.profiles.prod.hs_fallback`` → ``"openai:gpt-5.1"``.
    """

    cfg = load_cfg()
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    profiles = llm_cfg.get("profiles") or {}
    profile = get_current_profile()
    profile_data = profiles.get(profile, {})
    if isinstance(profile_data, dict):
        val = profile_data.get(purpose)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return None
