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

    Example:
      {"openai": "gpt-5.1", "google": "gemini-3-flash-preview",
       "anthropic": "claude-opus-4-5-20251101", "xai": "grok-4-0709"}
    """

    cfg = load_cfg()
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    profiles = llm_cfg.get("profiles") or {}
    profile = get_current_profile()
    models = profiles.get(profile, {})
    return {str(k): str(v) for k, v in models.items() if v}
