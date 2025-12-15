# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IFRC GO → canonical normalizer."""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pandas as pd

from .base import BaseAdapter, CANONICAL_COLUMNS, LOGGER


def _parse_iso_date(value: str | None) -> dt.date | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return dt.date.fromisoformat(text[:10])
    except ValueError:
        return None


class IFRCAdapter(BaseAdapter):
    """Normalizer for IFRC GO staging CSVs."""

    canonical_source = "ifrc_go"
    raw_slug = "ifrc_go"

    def load(self, raw_path: Path) -> pd.DataFrame:  # type: ignore[override]
        return pd.read_csv(raw_path, dtype=str, keep_default_na=False)

    def map(self, frame: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        df = frame.copy()

        # ------------------------------------------------------------------
        # Parse dates and filter by resolver window
        # ------------------------------------------------------------------
        df["as_of_date"] = pd.to_datetime(df.get("as_of_date"), errors="coerce", format="%Y-%m-%d")
        invalid_dates = int(df["as_of_date"].isna().sum())
        if invalid_dates:
            LOGGER.warning("ifrc_go: dropping %s rows with invalid as_of_date", invalid_dates)
        df = df.dropna(subset=["as_of_date"]).copy()

        start = _parse_iso_date(os.getenv("RESOLVER_START_ISO"))
        end = _parse_iso_date(os.getenv("RESOLVER_END_ISO"))
        if start:
            before = len(df)
            df = df.loc[df["as_of_date"] >= pd.Timestamp(start)].copy()
            LOGGER.info("ifrc_go: filtered rows outside start %s → %s → %s", start, before, len(df))
        if end:
            before = len(df)
            df = df.loc[df["as_of_date"] <= pd.Timestamp(end)].copy()
            LOGGER.info("ifrc_go: filtered rows outside end %s → %s → %s", end, before, len(df))

        if df.empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        df["as_of_date"] = df["as_of_date"].dt.to_period("M").dt.to_timestamp("M")
        df["as_of_date"] = df["as_of_date"].dt.strftime("%Y-%m-%d")

        # ------------------------------------------------------------------
        # Numeric coercion for values
        # ------------------------------------------------------------------
        cleaned_value = (
            df.get("value", "")
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
        )
        numeric_value = pd.to_numeric(cleaned_value, errors="coerce")
        dropped = int(numeric_value.isna().sum())
        if dropped:
            LOGGER.warning("ifrc_go: dropping %s rows with non-numeric value", dropped)
        df = df.loc[~numeric_value.isna()].copy()
        df.loc[:, "value"] = numeric_value.loc[df.index].astype(float)

        if df.empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        # ------------------------------------------------------------------
        # Assemble canonical columns
        # ------------------------------------------------------------------
        canonical = pd.DataFrame(
            {
                "event_id": df.get("event_id", ""),
                "country_name": df.get("country_name", ""),
                "iso3": df.get("iso3", "").str.upper(),
                "hazard_code": df.get("hazard_code", "").str.upper(),
                "hazard_label": df.get("hazard_label", ""),
                "hazard_class": df.get("hazard_class", ""),
                "metric": df.get("metric", ""),
                "unit": df.get("unit", ""),
                "as_of_date": df.get("as_of_date"),
                "value": df.get("value"),
                "series_semantics": df.get("series_semantics", ""),
                "source": self.canonical_slug,
            }
        )

        text_columns = [col for col in CANONICAL_COLUMNS if col not in {"value"}]
        for col in text_columns:
            canonical.loc[:, col] = canonical[col].fillna("").astype(str)

        canonical.loc[:, "value"] = canonical["value"].astype(float)
        return canonical



__all__ = ["IFRCAdapter"]
