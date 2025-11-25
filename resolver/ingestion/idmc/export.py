"""Adapters for resolution-ready facts exports from IDMC normalised data."""
from __future__ import annotations

from typing import Iterable, List

import pandas as pd

FACT_COLUMNS: List[str] = [
    "iso3",
    "as_of_date",
    "metric",
    "value",
    "series_semantics",
    "source",
]

# Columns expected by downstream exporters when staging ``flow.csv``.
FLOW_EXPORT_COLUMNS: List[str] = [
    "iso3",
    "as_of_date",
    "metric",
    "value",
    "series_semantics",
    "source",
]

FLOW_METRIC = "new_displacements"
FLOW_SERIES_SEMANTICS = "new"


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return ``frame`` restricted to ``columns`` in a deterministic order."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing required columns: {', '.join(sorted(missing))}")
    return frame.loc[:, list(columns)].copy()


def build_resolution_ready_facts(normalized: pd.DataFrame) -> pd.DataFrame:
    """Return resolution-ready facts derived from the normalised payload."""

    if normalized.empty:
        return pd.DataFrame(columns=FACT_COLUMNS)

    facts = _ensure_columns(normalized, FACT_COLUMNS)

    # Normalise basic types and guard against malformed values.
    facts["iso3"] = facts["iso3"].astype(str).str.strip().str.upper()
    facts["as_of_date"] = facts["as_of_date"].astype(str).str.strip()
    facts["metric"] = facts["metric"].astype(str).str.strip()
    facts["series_semantics"] = (
        facts["series_semantics"].fillna("").astype(str).str.strip()
    )
    facts["source"] = facts["source"].fillna("").astype(str).str.strip()
    facts["value"] = pd.to_numeric(facts["value"], errors="coerce")

    facts = facts.dropna(subset=["iso3", "as_of_date", "metric", "value"])

    # Enforce expected semantics for the IDMC flow export.
    facts.loc[:, "metric"] = FLOW_METRIC
    facts.loc[:, "series_semantics"] = FLOW_SERIES_SEMANTICS

    facts = (
        facts.sort_values(["iso3", "as_of_date", "metric", "series_semantics"])
        .drop_duplicates(["iso3", "as_of_date", "metric", "series_semantics"], keep="first")
        .reset_index(drop=True)
    )
    return facts


def summarise_facts(facts: pd.DataFrame) -> dict[str, object]:
    """Return lightweight diagnostics for a resolution-ready facts frame."""

    summary = {
        "rows": int(facts.shape[0]),
        "metrics": [],
        "series_semantics": [],
        "countries": [],
        "as_of_dates": [],
    }
    if facts.empty:
        return summary

    summary["metrics"] = sorted(
        facts["metric"].dropna().astype(str).str.strip().unique().tolist()
    )
    summary["series_semantics"] = sorted(
        facts["series_semantics"].dropna().astype(str).str.strip().unique().tolist()
    )
    summary["countries"] = sorted(
        facts["iso3"].dropna().astype(str).str.strip().str.upper().unique().tolist()
    )
    summary["as_of_dates"] = sorted(
        facts["as_of_date"].dropna().astype(str).str.strip().unique().tolist()
    )
    return summary
