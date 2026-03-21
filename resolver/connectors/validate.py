# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Schema validation for canonical connector output."""

from __future__ import annotations

import pandas as pd

from .protocol import CANONICAL_COLUMNS


def validate_canonical(
    df: pd.DataFrame,
    *,
    source: str = "unknown",
    extra_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Assert that *df* conforms to the canonical connector schema.

    Parameters
    ----------
    extra_columns:
        Optional list of additional columns that are allowed beyond
        CANONICAL_COLUMNS (e.g. ``["alertlevel"]`` for GDACS).

    Raises ``ValueError`` with a clear message on any violation.
    Returns *df* unchanged (for chaining).
    """
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{source}] canonical DataFrame missing columns: {missing}"
        )

    allowed = set(CANONICAL_COLUMNS) | set(extra_columns or [])
    extra = [c for c in df.columns if c not in allowed]
    if extra:
        raise ValueError(
            f"[{source}] canonical DataFrame has unexpected columns: {extra}"
        )

    if not df.empty:
        numeric_vals = pd.to_numeric(df["value"], errors="coerce")
        bad_count = numeric_vals.isna().sum() - df["value"].isna().sum()
        if bad_count > 0:
            raise ValueError(
                f"[{source}] {bad_count} rows have non-numeric 'value'"
            )

        iso3_lens = df["iso3"].dropna().astype(str).str.strip().str.len()
        bad_iso = (iso3_lens != 3).sum()
        if bad_iso > 0:
            raise ValueError(
                f"[{source}] {bad_iso} rows have iso3 that is not 3 characters"
            )

    return df


def empty_canonical() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical column set."""
    return pd.DataFrame(columns=CANONICAL_COLUMNS)
