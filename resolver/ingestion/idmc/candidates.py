# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Precedence candidate adapter for IDMC normalized data."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict

import pandas as pd

CANDIDATE_COLS = [
    "iso3",
    "as_of_date",
    "metric",
    "value",
    "source_system",
    "collection_type",
    "coverage",
    "freshness_days",
    "origin_iso3",
    "destination_iso3",
    "method_note",
    "series",
    "indicator",
    "indicator_kind",
    "qa_rank",
]

_METRIC_MAP: Dict[str, str] = {
    "new_displacements": "internal_displacement_new",
}
_METHOD_NOTE = "IDU preliminary; ~180d rolling window"


def _now_utc_date() -> pd.Timestamp:
    """Return today's UTC date as a pandas timestamp."""

    return pd.Timestamp(datetime.now(timezone.utc).date())


def _map_metric(value: str | None) -> str | None:
    if value is None:
        return None
    return _METRIC_MAP.get(str(value).strip())


def to_candidates_from_normalized(df_norm: pd.DataFrame) -> pd.DataFrame:
    """Convert IDMC normalized rows into precedence candidate schema."""

    if df_norm.empty:
        return pd.DataFrame(columns=CANDIDATE_COLS)

    df = df_norm.copy()

    df["iso3"] = df["iso3"].astype(str).str.upper().str.strip()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce", utc=False)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["metric"] = df["metric"].map(_map_metric)

    df = df.dropna(subset=["iso3", "as_of_date", "value", "metric"])
    if df.empty:
        return pd.DataFrame(columns=CANDIDATE_COLS)

    out = pd.DataFrame(
        {
            "iso3": df["iso3"],
            "as_of_date": df["as_of_date"],
            "metric": df["metric"],
            "value": df["value"],
            "source_system": "IDMC",
            "collection_type": "curated_event",
            "coverage": "national",
            "origin_iso3": pd.Series([None] * len(df)),
            "destination_iso3": pd.Series([None] * len(df)),
            "method_note": _METHOD_NOTE,
            "series": "IDU",
            "indicator": pd.Series([None] * len(df)),
            "indicator_kind": "explicit_flow",
            "qa_rank": 3,
        }
    )

    today = _now_utc_date().normalize()
    as_of_dates = out["as_of_date"].dt.tz_localize(None)
    out["freshness_days"] = (today - as_of_dates).dt.days

    out = out[CANDIDATE_COLS]

    out = out.sort_values(["iso3", "as_of_date", "metric"], kind="mergesort")
    out = out.drop_duplicates(["iso3", "as_of_date", "metric", "series"], keep="last")
    out = out.reset_index(drop=True)

    return out
