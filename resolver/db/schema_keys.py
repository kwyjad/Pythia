"""Canonical key definitions for DuckDB resolver tables."""

from __future__ import annotations

FACTS_RESOLVED_KEY_COLUMNS = [
    "event_id",
    "iso3",
    "hazard_code",
    "metric",
    "as_of_date",
    "publication_date",
    "source_id",
    "series_semantics",
    "ym",
]

FACTS_DELTAS_KEY_COLUMNS = [
    "event_id",
    "iso3",
    "hazard_code",
    "metric",
    "as_of_date",
    "publication_date",
    "source_id",
    "ym",
]

EMDAT_PA_KEY_COLUMNS = [
    "iso3",
    "ym",
    "shock_type",
]

UX_RESOLVED = "ux_facts_resolved_series"
UX_DELTAS = "ux_facts_deltas_series"

__all__ = [
    "FACTS_RESOLVED_KEY_COLUMNS",
    "FACTS_DELTAS_KEY_COLUMNS",
    "EMDAT_PA_KEY_COLUMNS",
    "UX_RESOLVED",
    "UX_DELTAS",
]
