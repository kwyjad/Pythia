"""Helper utilities for authenticating against the DTM API."""
from __future__ import annotations

import logging
import os
from typing import Dict, Optional

LOG = logging.getLogger(__name__)


def get_dtm_api_key() -> Optional[str]:
    """Return the DTM API subscription key from environment variables.

    Returns:
        The API key from DTM_API_KEY environment variable, or None if not set.
    """
    api_key = os.environ.get("DTM_API_KEY", "").strip()

    if not api_key:
        LOG.warning(
            "DTM_API_KEY environment variable not set. "
            "Register at https://dtm-apim-portal.iom.int/ and "
            "subscribe to API-V3 to get a subscription key."
        )
        return None

    LOG.info("DTM API key found (length: %d characters)", len(api_key))
    return api_key


def get_auth_headers() -> Dict[str, str]:
    """Return authentication headers for DTM API requests.

    Returns:
        Dictionary with Ocp-Apim-Subscription-Key header.

    Raises:
        RuntimeError: If DTM_API_KEY environment variable is not set.
    """
    api_key = get_dtm_api_key()

    if not api_key:
        raise RuntimeError(
            "DTM authentication failed: DTM_API_KEY environment variable not set. "
            "Register at https://dtm-apim-portal.iom.int/ and "
            "subscribe to API-V3 to get a subscription key."
        )

    return {"Ocp-Apim-Subscription-Key": api_key}


def check_api_key_configured() -> bool:
    """Check if DTM API key is configured (for diagnostics).

    Returns:
        True if key is set and non-empty, False otherwise.
    """
    return bool(os.environ.get("DTM_API_KEY", "").strip())
