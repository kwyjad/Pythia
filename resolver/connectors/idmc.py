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

# IDMC CONFLICT displacement only, as hazard ACE (Oct 2026). Until then this
# connector wrote every IDMC figure as hazard DI, all causes summed, which is
# the same fault the adapter path had: typhoon evacuations read as conflict.
_HAZARD_LABEL = "Armed conflict — internal displacement"
_HAZARD_CLASS = "human-induced"
_PUBLISHER = "IDMC"
_SOURCE_TYPE = "agency"
_UNIT = "persons"


class IdmcConnector:
    """Fetch IDMC conflict displacement and return a canonical DataFrame."""

    name: str = "idmc"

    def fetch_and_normalize(self) -> pd.DataFrame:
        """Read the IDU route, keep conflict displacement, map to canonical.

        Delegates to :mod:`resolver.ingestion.idmc_conflict`, the one place
        that splits IDMC records by cause, so this path and the Resolver
        Update path cannot write different series.
        """
        import datetime as _dt
        import os

        from resolver.ingestion import idmc_conflict as ic

        credential = ic._client_id()
        if credential is None:
            LOG.warning("[idmc] no IDMC client id; nothing written")
            return empty_canonical()
        months = int(os.getenv(ic.MONTHS_ENV, "") or ic.DEFAULT_MONTHS)
        first, last = ic.month_window(_dt.date.today(), months)
        try:
            records = ic._default_get(
                os.getenv(ic.URL_ENV, "").strip() or ic.DEFAULT_IDU_ALL_URL,
                {"client_id": credential[0]},
                ic.REQUEST_TIMEOUT_SEC,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("[idmc] fetch failed: %s", type(exc).__name__)
            return empty_canonical()
        flows, report = ic.conflict_monthly_flows(records or [], first, last)
        LOG.info("[idmc] conflict rows=%s excluded_people=%s",
                 report.get("rows"), report.get("excluded_people"))
        if flows.empty:
            return empty_canonical()
        facts = ic.staging_frame(flows)
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
                "hazard_code": "ACE",
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
                "doc_title": "IDMC conflict displacement data",
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
