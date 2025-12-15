# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Helper utilities for authenticating against the DTM API."""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

LOG = logging.getLogger(__name__)


def _mask_key(key: str) -> str:
    if len(key) <= 4:
        return "***"
    return f"...{key[-4:]}"


def get_dtm_api_key() -> Optional[str]:
    """Return the configured DTM API key or subscription key if available."""

    primary = (os.getenv("DTM_API_KEY") or "").strip()
    fallback = (os.getenv("DTM_SUBSCRIPTION_KEY") or "").strip()

    if primary:
        LOG.info("Using DTM API key ending with %s", _mask_key(primary))
        return primary

    if fallback:
        LOG.info("Using DTM subscription key ending with %s", _mask_key(fallback))
        return fallback

    LOG.debug("No DTM API credentials found in environment")
    return None


def build_discovery_header_variants(key: str) -> List[Dict[str, str]]:
    """Return the header permutations commonly used by the DTM gateway."""

    token = (key or "").strip()
    if not token:
        return []
    variants: List[Dict[str, str]] = []
    primary = {"Ocp-Apim-Subscription-Key": token}
    variants.append(primary)
    alternate = {"X-API-Key": token}
    # Avoid duplicating variants when the headers would be identical (unlikely but safe)
    if alternate.keys() != primary.keys():
        variants.append(alternate)
    return variants
