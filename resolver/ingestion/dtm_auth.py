"""Helper utilities for authenticating against the DTM API."""
from __future__ import annotations

import logging
import os
from typing import Dict

LOG = logging.getLogger(__name__)


def get_dtm_api_key() -> str:
    """Return the DTM API subscription key from environment variables.

    Returns:
        The API key from DTM_API_KEY environment variable.

    Raises:
        RuntimeError: If DTM_API_KEY environment variable is not set.
    """
    api_key = os.environ.get("DTM_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "DTM authentication failed: DTM_API_KEY environment variable not set. "
            "Register at https://dtm-apim-portal.iom.int/ and subscribe to API-V3 "
            "to get a subscription key."
        )

    LOG.debug("DTM API key found (length: %d)", len(api_key))
    return api_key


def get_auth_headers() -> Dict[str, str]:
    """Return authentication headers for DTM API requests.

    Returns:
        Dictionary with Ocp-Apim-Subscription-Key header.
    """
    api_key = get_dtm_api_key()
    return {"Ocp-Apim-Subscription-Key": api_key}
