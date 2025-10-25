"""Helper utilities for authenticating against the DTM API."""
from __future__ import annotations

import logging
import os

LOG = logging.getLogger(__name__)


def get_dtm_api_key() -> str:
    """Return the DTM API subscription key.

    Raises:
        RuntimeError: If the environment variable is unset or empty.
    """

    key = (os.getenv("DTM_API_KEY") or "").strip()
    if not key:
        msg = (
            "DTM_API_KEY not set. Please add it to GitHub secrets or local env. "
            "Without it, discovery and data fetch will fail."
        )
        LOG.error(msg)
        raise RuntimeError(msg)

    masked = f"...{key[-4:]}" if len(key) > 4 else "***"
    LOG.info("Using DTM_API_KEY ending with %s", masked)
    return key
