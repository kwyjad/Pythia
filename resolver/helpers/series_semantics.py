# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Utilities for normalising ``series_semantics`` values across the codebase."""

# Final enforcement happens in ``resolver.db.duckdb_io._canonicalize_semantics``,
# which collapses values to the table-specific policies when writing to DuckDB.

from __future__ import annotations

import logging

import pandas as pd

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:  # pragma: no cover - avoid duplicate handlers
    LOGGER.addHandler(logging.NullHandler())

CANONICAL_MAP = {
    "stock": "stock",
    "stock_estimate": "stock_estimate",
    "stock estimate": "stock_estimate",
    "stock-estimate": "stock_estimate",
    "stock_est.": "stock_estimate",
    "stock est.": "stock_estimate",
    "new": "new",
    "delta": "new",
    "deltas": "new",
}

KNOWN_ALIASES = {
    "stock_est": "stock_estimate",
    "stockest": "stock_estimate",
    "stockestimates": "stock_estimate",
    "snapshot": "stock",
    "inventory": "stock",
}

_CANONICAL_VALUES = {"new", "stock", "stock_estimate"}


def _iter_normalised(values: pd.Series) -> tuple[pd.Series, set[str]]:
    cleaned = values.astype(str).str.strip()
    lowered = cleaned.str.lower()
    normalised = []
    unknown: set[str] = set()
    for original, lowered_value in zip(cleaned, lowered, strict=False):
        canonical = CANONICAL_MAP.get(lowered_value)
        if canonical is None:
            canonical = KNOWN_ALIASES.get(lowered_value)
        if canonical is None and lowered_value in _CANONICAL_VALUES:
            canonical = lowered_value
        if canonical is None and not original:
            canonical = ""
        if canonical is None:
            unknown.add(lowered_value)
            normalised.append(original)
        else:
            normalised.append(canonical)
    return pd.Series(normalised, index=values.index, dtype="string"), unknown


def normalize_series_semantics(
    frame: pd.DataFrame, *, column: str = "series_semantics"
) -> pd.DataFrame:
    """Return a copy of ``frame`` with canonicalised ``series_semantics`` values."""

    if column not in frame.columns:
        return frame

    normalised = frame.copy()
    series = normalised[column].astype("string").fillna("")
    mapped, unknown = _iter_normalised(series)
    normalised[column] = mapped
    if unknown:
        LOGGER.warning(
            "series_semantics normalisation left unknown values: %s",
            sorted(value for value in unknown if value),
        )
    return normalised


__all__ = ["CANONICAL_MAP", "normalize_series_semantics"]
