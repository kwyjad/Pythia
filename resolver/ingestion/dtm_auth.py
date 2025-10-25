"""Helper utilities for authenticating against the DTM API."""
from __future__ import annotations

import logging
import os
from typing import Optional

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
