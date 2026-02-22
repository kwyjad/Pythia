# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ACLED connector wrapper.

Delegates all fetching and normalisation to the existing
``resolver.ingestion.acled_client`` module, then returns a canonical
DataFrame for the pipeline orchestrator.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import pandas as pd

from .protocol import CANONICAL_COLUMNS
from .validate import empty_canonical

LOG = logging.getLogger(__name__)


class AcledConnector:
    """Fetch ACLED conflict data and return a canonical DataFrame."""

    name: str = "acled"

    def fetch_and_normalize(self) -> pd.DataFrame:
        """Run the ACLED ingestion pipeline and return canonical rows.

        Calls ``acled_client.collect_rows()`` which handles auth, API
        pagination, onset detection, and row assembly.  The result is
        already in CANONICAL_HEADERS format (a superset of
        ``CANONICAL_COLUMNS``).
        """
        from resolver.ingestion.acled_client import collect_rows, CANONICAL_HEADERS

        rows: List[Dict[str, Any]] = collect_rows()
        if not rows:
            LOG.info("[acled] no rows collected")
            return empty_canonical()

        df = pd.DataFrame(rows, columns=CANONICAL_HEADERS)

        # ACLED's CANONICAL_HEADERS is a superset of CANONICAL_COLUMNS
        # (it includes publication_date, publisher, etc. — all of which
        # are present in our protocol).  Select exactly the canonical set.
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        df = df[CANONICAL_COLUMNS].copy()
        LOG.info("[acled] produced %d canonical rows", len(df))
        return df
