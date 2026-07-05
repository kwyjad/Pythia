# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Connector protocol — the contract every data source must satisfy.

The contract vs the database schema
-----------------------------------
``CANONICAL_COLUMNS`` (21 columns) is the *connector output contract*, not
the ``facts_resolved`` table schema. The table has more columns
(~27 in ``resolver/db/schema.sql``) because the pipeline adds derived and
provenance fields after validation:

- ``enrich()`` fills defaults (metric/unit/series_semantics/event_id) and
  ``derive_ym()`` adds ``ym`` from ``as_of_date``;
- the precedence engine adds ``precedence_tier`` / ``provenance_source`` /
  ``as_of`` and passes 13 metadata fields through from the winning row;
- the DB writer stamps ``created_at`` (and ``is_test`` where applicable).

Connectors must never emit those derived columns themselves.

The ``extra_columns`` convention (supplementary columns)
--------------------------------------------------------
A connector MAY return columns beyond the canonical 21 when the source has
genuinely source-specific data that downstream consumers need (the canonical
example: GDACS ``alertlevel`` — Green/Orange/Red — which resolutions and
prompts read, and which is NULL for every other source).

Rules for a supplementary column:

1. It must be declared to validation: pass ``extra_columns=["alertlevel"]``
   to :func:`resolver.connectors.validate.validate_canonical` when calling
   it directly. The ``run_pipeline`` orchestrator auto-detects non-canonical
   columns and passes them through, so pipeline runs never reject them —
   the declaration matters for direct/strict validation call sites.
2. It must exist as a nullable column in ``facts_resolved``
   (``resolver/db/schema.sql``) — the pipeline preserves it end-to-end and
   the write fails if the table lacks it. Add the column to the schema in
   the same PR that adds the connector output.
3. It must be NULL/empty for sources that don't produce it; consumers must
   treat NULL as "not applicable", never as a default value.
4. Do NOT reuse a supplementary column for a second source with different
   semantics — add a new column instead.

Current supplementary columns: ``alertlevel`` (GDACS only).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

# Every connector must return a DataFrame with exactly these columns
# (plus optional declared supplementary columns — see module docstring).
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
