"""Helper utilities for authenticating against the DTM API."""
from __future__ import annotations

import logging
import os

LOG = logging.getLogger(__name__)


def get_dtm_api_key() -> str:
    """Return the required DTM API subscription key.

    Raises:
        RuntimeError: If the ``DTM_API_KEY`` environment variable is missing or empty.
    """

    raw = os.getenv("DTM_API_KEY")
    api_key = raw.strip() if raw else ""
    if not api_key:
        message = (
            "DTM_API_KEY not set. Please add it to GitHub secrets or the local env. "
            "Without it, discovery and data fetch will fail."
        )
        LOG.error(message)
        raise RuntimeError(message)

    masked = f"...{api_key[-4:]}" if len(api_key) > 4 else "***"
    LOG.info("Using DTM_API_KEY ending with %s", masked)
    return api_key


def check_api_key_configured() -> bool:
    """Return ``True`` when ``DTM_API_KEY`` is present and non-empty."""

    return bool(os.getenv("DTM_API_KEY", "").strip())
