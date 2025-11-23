"""Helpers to export IDMC normalised data to resolution-ready facts files."""
from __future__ import annotations

import os
from typing import Final

import pandas as pd

FACTS_COLS: Final[list[str]] = [
    "iso3",
    "as_of_date",
    "metric",
    "value",
    "series_semantics",
    "source",
]

# ``stock.csv`` shares the same column contract as the normalised facts export.
STOCK_EXPORT_COLUMNS: Final[list[str]] = FACTS_COLS

# Absolute staging path used by downstream tooling when probing for ``stock.csv``.
STOCK_STAGING_PATH: Final[str] = os.path.join("resolver", "staging", "idmc", "stock.csv")


def ensure_dir(path: str) -> None:
    """Ensure ``path`` exists, creating directories as required."""

    os.makedirs(path, exist_ok=True)


def to_facts(df_norm: pd.DataFrame) -> pd.DataFrame:
    """Map normalised IDMC rows to the resolver facts schema."""

    if df_norm.empty:
        return pd.DataFrame(columns=FACTS_COLS)

    missing = [column for column in FACTS_COLS if column not in df_norm.columns]
    if missing:
        raise KeyError(
            "missing required columns: " + ", ".join(sorted(set(missing)))
        )

    out = df_norm.loc[:, FACTS_COLS].copy()

    # Canonical IDMC facts must always report source="IDMC".
    out["source"] = "IDMC"

    out["iso3"] = out["iso3"].astype(str).str.upper().str.strip()
    out["metric"] = out["metric"].astype(str)
    out["series_semantics"] = out["series_semantics"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    out = out.dropna(subset=["iso3", "as_of_date", "metric", "value"])

    out = (
        out.sort_values("value")
        .drop_duplicates(["iso3", "as_of_date", "metric"], keep="last")
        .reset_index(drop=True)
    )
    return out


def write_facts_csv(df_facts: pd.DataFrame, out_dir: str) -> str:
    """Write ``df_facts`` to ``out_dir`` as CSV and return the file path."""

    ensure_dir(out_dir)
    path = os.path.join(out_dir, "idmc_facts_flow.csv")
    df_facts.to_csv(path, index=False)
    return path


def write_facts_parquet(df_facts: pd.DataFrame, out_dir: str) -> str:
    """Write ``df_facts`` to ``out_dir`` as Parquet and return the file path.

    Returns an empty string if Parquet support is unavailable.
    """

    ensure_dir(out_dir)
    path = os.path.join(out_dir, "idmc_facts_flow.parquet")
    try:
        df_facts.to_parquet(path, index=False)
        return path
    except Exception:  # pragma: no cover - optional dependency
        return ""
