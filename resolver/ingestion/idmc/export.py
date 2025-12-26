# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Adapters for resolution-ready facts exports from IDMC normalised data."""
from __future__ import annotations

import json
import os
import re
from typing import Iterable, List

import pandas as pd

from .normalize import derive_ym

YM_REGEX = re.compile(r"^\d{4}-\d{2}$")

FACT_COLUMNS: List[str] = [
    "iso3",
    "as_of_date",
    "metric",
    "value",
    "series_semantics",
    "source",
    "ym",
    "record_id",
]

# Columns expected by downstream exporters when staging ``flow.csv``.
FLOW_EXPORT_COLUMNS: List[str] = [
    "iso3",
    "as_of_date",
    "metric",
    "value",
    "series_semantics",
    "source",
    "ym",
    "record_id",
]

FLOW_METRIC = "new_displacements"
FLOW_SERIES_SEMANTICS = "new"


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Return ``frame`` restricted to ``columns`` in a deterministic order."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"missing required columns: {', '.join(sorted(missing))}")
    return frame.loc[:, list(columns)].copy()


def _normalize_ym(facts: pd.DataFrame) -> pd.Series:
    if "ym" in facts.columns:
        ym_series = facts["ym"].astype("string").fillna("").str.strip()
    else:
        ym_series = pd.Series([""] * len(facts), index=facts.index, dtype="string")

    if "as_of_date" in facts.columns:
        as_of_series = pd.to_datetime(facts["as_of_date"], errors="coerce", utc=False)
        ym_from_asof = as_of_series.dt.strftime("%Y-%m")
        ym_series = ym_series.mask(ym_series == "", ym_from_asof)

    missing = ym_series.isna() | ym_series.astype(str).str.strip().eq("")
    if missing.any():
        derived = facts.loc[missing].apply(lambda row: derive_ym(row.to_dict()), axis=1)
        ym_series = ym_series.astype("object")
        ym_series.loc[missing] = derived
        ym_series = ym_series.astype("string")

    valid = ym_series.str.match(YM_REGEX)
    return ym_series.where(valid)


def _ensure_record_id(facts: pd.DataFrame) -> pd.Series:
    candidate_cols = ["record_id", "source_id", "event_id", "id", "url", "source_url"]
    candidate = pd.Series([pd.NA] * len(facts), index=facts.index, dtype="string")
    for column in candidate_cols:
        if column in facts.columns:
            series = facts[column].astype("string").fillna("").str.strip()
            candidate = candidate.where(candidate.fillna("").astype(str).str.strip() != "", series)

    def _series_or_empty(name: str) -> pd.Series:
        if name in facts.columns:
            return facts[name].astype("string").fillna("").str.strip()
        return pd.Series([""] * len(facts), index=facts.index, dtype="string")

    iso = _series_or_empty("iso3")
    metric = _series_or_empty("metric")
    ym = _series_or_empty("ym")
    source = _series_or_empty("source")
    value = facts["value"] if "value" in facts.columns else pd.Series([pd.NA] * len(facts), index=facts.index)
    value_text = value.astype("string").fillna("").str.strip()

    fallback = iso + "-" + metric + "-" + ym + "-" + value_text + "-" + source
    candidate = candidate.astype("string").fillna("").str.strip()
    return candidate.where(candidate != "", fallback)


def _collect_drop_examples(frame: pd.DataFrame, mask: pd.Series) -> list[dict[str, str]]:
    if frame.empty or not mask.any():
        return []
    examples: list[dict[str, str]] = []
    columns = ["record_id", "source_id", "url", "iso3", "as_of_date", "metric"]
    for _, row in frame.loc[mask].head(3).iterrows():
        entry: dict[str, str] = {}
        for column in columns:
            if column in frame.columns:
                value = row.get(column)
                if value is None or pd.isna(value):
                    continue
                text = str(value).strip()
                if text:
                    entry[column] = text
        if entry:
            examples.append(entry)
    return examples


def _write_month_drop_diagnostics(dropped_count: int, examples: list[dict[str, str]]) -> str:
    path = os.path.join("diagnostics", "idmc_month_drop.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"dropped_count": int(dropped_count), "examples": examples}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path


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

    facts["ym"] = _normalize_ym(facts)
    facts["record_id"] = _ensure_record_id(facts)

    missing_ym = facts["ym"].isna() | facts["ym"].astype(str).str.strip().eq("")
    examples = _collect_drop_examples(facts, missing_ym)
    dropped_count = int(missing_ym.sum())
    if dropped_count:
        facts = facts.loc[~missing_ym].copy()
    _write_month_drop_diagnostics(dropped_count, examples)

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
