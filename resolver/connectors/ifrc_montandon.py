# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IFRC Montandon (GO API) connector wrapper.

Delegates fetching to the existing ``resolver.ingestion.ifrc_go_client``
module, which queries the IFRC GO Admin v2 API for field reports, appeals,
and situation reports, then returns rows already in canonical format.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import pandas as pd

from .protocol import CANONICAL_COLUMNS
from .validate import empty_canonical

LOG = logging.getLogger(__name__)


class IfrcMontandonConnector:
    """Fetch IFRC GO data and return a canonical DataFrame."""

    name: str = "ifrc_montandon"

    def fetch_and_normalize(self) -> pd.DataFrame:
        """Run the IFRC GO ingestion pipeline and return canonical rows.

        Calls ``ifrc_go_client.collect_rows()`` which handles API
        pagination, hazard detection, metric extraction, and row assembly.
        The result is already in the 21-column canonical format.
        """
        if os.getenv("RESOLVER_INGESTION_MODE") == "stubs":
            LOG.info("[ifrc_montandon] stubs mode — skipping live API")
            return empty_canonical()

        from resolver.ingestion.ifrc_go_client import collect_rows, COLUMNS

        try:
            rows: List[List[str]] = collect_rows()
        except Exception as exc:
            LOG.warning("[ifrc_montandon] fetch failed: %s", exc)
            return empty_canonical()

        if not rows:
            LOG.info("[ifrc_montandon] no rows collected")
            return empty_canonical()

        df = pd.DataFrame(rows, columns=COLUMNS)

        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        df = df[CANONICAL_COLUMNS].copy()
        LOG.info("[ifrc_montandon] produced %d canonical rows", len(df))
        return df
