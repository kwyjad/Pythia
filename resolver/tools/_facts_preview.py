# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from typing import List

import pandas as pd


def _month_end_from_ym(series: pd.Series) -> pd.Series:
    """Return ISO month-end strings derived from YYYY-MM values."""

    ym = series.astype("string").fillna("").str.strip()
    month_start = pd.to_datetime(ym + "-01", errors="coerce", utc=False)
    month_end = month_start + pd.offsets.MonthEnd(0)
    return month_end.dt.strftime("%Y-%m-%d")

CANONICAL_PREVIEW_COLUMNS: List[str] = [
    "iso3",
    "ym",
    "as_of_date",
    "hazard_code",
    "metric",
    "value",
]


def enforce_canonical_preview(df: pd.DataFrame | None) -> pd.DataFrame:
    """Return a frame limited to the canonical preview schema.

    The validator invoked by freeze_snapshot expects the preview CSV to contain
    the columns listed in ``CANONICAL_PREVIEW_COLUMNS`` and no additional
    connector-specific fields. This helper normalizes the incoming frame,
    ensuring each required column is present and lightly coerced into the
    expected shape (e.g., ISO codes upper-cased, hazard codes lower-cased, and
    date strings formatted as ``YYYY-MM-DD``).
    """

    if df is None or df.empty:
        return pd.DataFrame(columns=CANONICAL_PREVIEW_COLUMNS)

    frame = df.copy()
    for column in CANONICAL_PREVIEW_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame["iso3"] = (
        frame["iso3"].astype("string").fillna("").str.strip().str.upper()
    )
    frame["ym"] = frame["ym"].astype("string").fillna("").str.strip()

    parsed_as_of = pd.to_datetime(frame["as_of_date"], errors="coerce", utc=False)
    existing_str = frame["as_of_date"].astype("string").fillna("").str.strip()
    iso_as_of = parsed_as_of.dt.strftime("%Y-%m-%d")
    frame["as_of_date"] = iso_as_of.where(parsed_as_of.notna(), existing_str)

    ym_month_end = _month_end_from_ym(frame["ym"])
    parsed_month_prefix = parsed_as_of.dt.strftime("%Y-%m")
    needs_alignment = parsed_as_of.isna() | (parsed_month_prefix != frame["ym"])
    frame.loc[needs_alignment & ym_month_end.notna(), "as_of_date"] = ym_month_end[
        needs_alignment & ym_month_end.notna()
    ]

    frame["hazard_code"] = (
        frame["hazard_code"].astype("string").fillna("").str.strip().str.lower()
    )
    frame["metric"] = frame["metric"].astype("string").fillna("").str.strip()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

    return frame[CANONICAL_PREVIEW_COLUMNS]
