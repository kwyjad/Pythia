# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ACLED staging CSV → canonical normalizer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import BaseAdapter, CANONICAL_COLUMNS, LOGGER


# Map ACLED-internal metric names to the canonical names expected by
# downstream consumers (compute_resolutions.py, forecaster, etc.).
_METRIC_MAP: dict[str, str] = {
    "fatalities_battle_month": "fatalities",
}


class ACLEDAdapter(BaseAdapter):
    """Normalizer for ACLED staging CSVs (21-column format)."""

    canonical_source = "acled"
    raw_slug = "acled"

    def load(self, raw_path: Path) -> pd.DataFrame:  # type: ignore[override]
        return pd.read_csv(raw_path, dtype=str, keep_default_na=False)

    def map(self, frame: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        df = frame.copy()

        if df.empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        # ------------------------------------------------------------------
        # as_of_date: ACLED writes YYYY-MM strings → convert to month-end
        # ------------------------------------------------------------------
        raw_dates = df.get("as_of_date", pd.Series(dtype=str)).astype(str).str.strip()
        parsed = pd.to_datetime(raw_dates, format="%Y-%m", errors="coerce")
        invalid = int(parsed.isna().sum())
        if invalid:
            LOGGER.warning("acled: dropping %s rows with invalid as_of_date", invalid)
        df = df.loc[~parsed.isna()].copy()
        parsed = parsed.dropna()
        if df.empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        # Month-end: add MonthEnd offset, then format
        month_end = parsed + pd.offsets.MonthEnd(0)
        df["as_of_date"] = month_end.dt.strftime("%Y-%m-%d")

        # ------------------------------------------------------------------
        # value: numeric coercion
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
            LOGGER.warning("acled: dropping %s rows with non-numeric value", dropped)
        df = df.loc[~numeric_value.isna()].copy()
        df.loc[:, "value"] = numeric_value.loc[df.index].astype(float)

        if df.empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        # ------------------------------------------------------------------
        # metric: remap to canonical names for downstream consumers
        # ------------------------------------------------------------------
        metric_series = df.get("metric", pd.Series(dtype=str)).astype(str).str.strip()
        df["metric"] = metric_series.replace(_METRIC_MAP)

        # ------------------------------------------------------------------
        # Assemble canonical output
        # ------------------------------------------------------------------
        canonical = pd.DataFrame(
            {
                "event_id": df.get("event_id", ""),
                "country_name": df.get("country_name", ""),
                "iso3": df.get("iso3", "").str.upper(),
                "hazard_code": df.get("hazard_code", "").str.upper(),
                "hazard_label": df.get("hazard_label", ""),
                "hazard_class": df.get("hazard_class", ""),
                "metric": df["metric"],
                "unit": df.get("unit", ""),
                "as_of_date": df["as_of_date"],
                "value": df["value"],
                "series_semantics": "new",
                "source": self.canonical_slug,
            }
        )

        text_columns = [col for col in CANONICAL_COLUMNS if col != "value"]
        for col in text_columns:
            canonical[col] = canonical[col].fillna("").astype(str)

        canonical["value"] = pd.to_numeric(canonical["value"], errors="coerce").astype(float)
        return canonical


__all__ = ["ACLEDAdapter"]
