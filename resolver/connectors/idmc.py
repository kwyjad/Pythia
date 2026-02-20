# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IDMC connector wrapper.

Delegates fetching to the ``resolver.ingestion.idmc`` package, then maps
its 6-column export format to the full canonical schema.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .protocol import CANONICAL_COLUMNS
from .validate import empty_canonical

LOG = logging.getLogger(__name__)

# Default hazard metadata for IDMC displacement data.
_HAZARD_CODE = "DI"  # Displacement (internal)
_HAZARD_LABEL = "Internal displacement"
_HAZARD_CLASS = "displacement"
_PUBLISHER = "IDMC"
_SOURCE_TYPE = "agency"
_UNIT = "persons"


class IdmcConnector:
    """Fetch IDMC displacement data and return a canonical DataFrame."""

    name: str = "idmc"

    def fetch_and_normalize(self) -> pd.DataFrame:
        """Run the IDMC ingestion pipeline and return canonical rows.

        Uses the IDMC client to fetch data, then builds resolution-ready
        facts via ``idmc.export.build_resolution_ready_facts``.  The
        IDMC export schema has 6 columns; we map these to the full
        21-column canonical schema, filling defaults for provenance
        fields that IDMC does not natively provide.
        """
        from resolver.ingestion.idmc.client import IdmcClient
        from resolver.ingestion.idmc.config import load as load_idmc_config
        from resolver.ingestion.idmc.export import build_resolution_ready_facts
        from resolver.ingestion.idmc.normalize import normalize_all

        config = load_idmc_config()
        client = IdmcClient(config)

        try:
            raw = client.fetch()
        except Exception as exc:
            LOG.warning("[idmc] fetch failed: %s", exc)
            return empty_canonical()

        if raw.empty:
            LOG.info("[idmc] no raw rows fetched")
            return empty_canonical()

        normalized = normalize_all(raw, config)
        if normalized.empty:
            LOG.info("[idmc] normalisation produced 0 rows")
            return empty_canonical()

        facts = build_resolution_ready_facts(normalized)
        if facts.empty:
            LOG.info("[idmc] resolution-ready facts empty")
            return empty_canonical()

        return self._to_canonical(facts)

    @staticmethod
    def _to_canonical(facts: pd.DataFrame) -> pd.DataFrame:
        """Map the 6-column IDMC facts to the 21-column canonical schema."""

        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        canonical = pd.DataFrame(
            {
                "event_id": (
                    facts["iso3"].astype(str)
                    + "-IDMC-"
                    + facts["metric"].astype(str)
                    + "-"
                    + facts["as_of_date"].astype(str)
                ),
                "country_name": "",  # enrichment step will fill from registry
                "iso3": facts["iso3"],
                "hazard_code": _HAZARD_CODE,
                "hazard_label": _HAZARD_LABEL,
                "hazard_class": _HAZARD_CLASS,
                "metric": facts["metric"],
                "series_semantics": facts["series_semantics"],
                "value": facts["value"],
                "unit": _UNIT,
                "as_of_date": facts["as_of_date"],
                "publication_date": now_iso[:10],
                "publisher": _PUBLISHER,
                "source_type": _SOURCE_TYPE,
                "source_url": "",
                "doc_title": "IDMC displacement data",
                "definition_text": "",
                "method": "api",
                "confidence": "high",
                "revision": "1",
                "ingested_at": now_iso,
            }
        )

        canonical = canonical[CANONICAL_COLUMNS].copy()
        LOG.info("[idmc] produced %d canonical rows", len(canonical))
        return canonical
