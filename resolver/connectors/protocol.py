# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Connector protocol — the contract every data source must satisfy."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

# Every connector must return a DataFrame with exactly these columns.
# The orchestrator derives ``ym`` from ``as_of_date`` before writing to DuckDB.
CANONICAL_COLUMNS: list[str] = [
    "event_id",
    "country_name",
    "iso3",
    "hazard_code",
    "hazard_label",
    "hazard_class",
    "metric",
    "series_semantics",
    "value",
    "unit",
    "as_of_date",
    "publication_date",
    "publisher",
    "source_type",
    "source_url",
    "doc_title",
    "definition_text",
    "method",
    "confidence",
    "revision",
    "ingested_at",
]


@runtime_checkable
class Connector(Protocol):
    """Protocol that every Resolver data-source connector must satisfy.

    Connectors are self-contained: they fetch data, normalise it, and
    return a canonical DataFrame.  The orchestrator (``run_pipeline.py``)
    handles DuckDB writes, precedence, and delta computation.
    """

    name: str

    def fetch_and_normalize(self) -> pd.DataFrame:
        """Fetch data from the source and return a canonical DataFrame.

        The returned DataFrame **must** contain exactly ``CANONICAL_COLUMNS``
        (in any order).  Columns that are not meaningful for the source
        should be present but may contain empty strings.

        Returns an empty DataFrame with the correct columns when no data
        is available.
        """
        ...
