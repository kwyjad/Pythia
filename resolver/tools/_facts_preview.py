from __future__ import annotations

from typing import List

import pandas as pd

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

    parsed_as_of = pd.to_datetime(frame["as_of_date"], errors="coerce")
    iso_as_of = parsed_as_of.dt.strftime("%Y-%m-%d")
    fallback_as_of = frame["as_of_date"].astype("string").fillna("")
    frame["as_of_date"] = iso_as_of.where(parsed_as_of.notna(), fallback_as_of)

    frame["hazard_code"] = (
        frame["hazard_code"].astype("string").fillna("").str.strip().str.lower()
    )
    frame["metric"] = frame["metric"].astype("string").fillna("").str.strip()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

    return frame[CANONICAL_PREVIEW_COLUMNS]
