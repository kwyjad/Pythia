"""Helper utilities for authenticating against the DTM API."""
from __future__ import annotations

import logging
import os

LOG = logging.getLogger(__name__)


def get_dtm_api_key() -> str | None:
    """Return the configured DTM API subscription key if available."""

    key = (os.getenv("DTM_API_KEY") or "").strip()
    if not key:
        LOG.debug("DTM_API_KEY not present in environment")
        return None

    masked = f"...{key[-4:]}" if len(key) > 4 else "***"
    LOG.info("Using DTM_API_KEY ending with %s", masked)
    return key
