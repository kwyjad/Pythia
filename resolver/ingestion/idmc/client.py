"""Client shims for the offline-only IDMC connector skeleton."""
from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import pandas as pd

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "fixtures")


def fetch_offline(skip_network: bool = True) -> Dict[str, pd.DataFrame]:
    """Load miniature CSV fixtures for the connector."""

    monthly = pd.read_csv(os.path.join(FIXTURES_DIR, "sample_monthly.csv"))
    annual = pd.read_csv(os.path.join(FIXTURES_DIR, "sample_annual.csv"))
    return {"monthly_flow": monthly, "stock": annual}


def fetch(
    cfg,
    skip_network: bool = False,
    soft_timeouts: bool = True,  # noqa: ARG001 - future compatibility
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Return fixtures and a tiny diagnostics payload.

    Network access is intentionally disabled in the skeleton connector.
    """

    if skip_network:
        return fetch_offline(True), {"mode": "offline", "http": {"requests": 0}}

    # Future online integration will live here.
    return fetch_offline(True), {"mode": "offline-forced", "http": {"requests": 0}}
