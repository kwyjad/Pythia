# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import os
from typing import Any


def _load_pm_config() -> dict[str, Any]:
    """Load prediction_markets config from pythia/config.yaml."""
    try:
        from pythia.config import load

        cfg = load()
        return cfg.get("prediction_markets", {}) if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def is_enabled() -> bool:
    """Check if prediction markets retriever is enabled."""
    env = os.getenv("PYTHIA_PREDICTION_MARKETS_ENABLED", "").strip()
    if env:
        return env in ("1", "true", "yes")
    return bool(_load_pm_config().get("enabled", False))


def get_timeout_sec() -> float:
    """Get overall retrieval timeout in seconds."""
    env = os.getenv("PYTHIA_PREDICTION_MARKETS_TIMEOUT_SEC", "").strip()
    if env:
        try:
            return max(1.0, float(env))
        except (TypeError, ValueError):
            pass
    return float(_load_pm_config().get("timeout_sec", 30))


def get_platform_config(platform: str) -> dict[str, Any]:
    """Get config for a specific platform (metaculus, polymarket, manifold)."""
    cfg = _load_pm_config()
    platforms = cfg.get("platforms", {})
    if not isinstance(platforms, dict):
        return {}
    return platforms.get(platform, {}) if isinstance(platforms.get(platform), dict) else {}


def is_platform_enabled(platform: str) -> bool:
    """Check if a specific platform is enabled."""
    if not is_enabled():
        return False
    pcfg = get_platform_config(platform)
    return bool(pcfg.get("enabled", True))


def get_query_generation_config() -> dict[str, Any]:
    """Get query generation config."""
    cfg = _load_pm_config()
    qg = cfg.get("query_generation", {})
    return qg if isinstance(qg, dict) else {}


def get_relevance_filter_config() -> dict[str, Any]:
    """Get relevance filter config."""
    cfg = _load_pm_config()
    rf = cfg.get("relevance_filter", {})
    return rf if isinstance(rf, dict) else {}
