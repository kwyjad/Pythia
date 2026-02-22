# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IDMC staging CSV → canonical normalizer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .base import BaseAdapter, CANONICAL_COLUMNS, LOGGER


class IDMCAdapter(BaseAdapter):
    """Normalizer for IDMC flow staging CSVs (6-column format).

    IDMC outputs staging data to ``idmc/flow.csv`` (a subdirectory), so
    ``resolve_raw_path`` is overridden to look there instead of a flat file.
    """

    canonical_source = "idmc"
    raw_slug = None  # custom resolution below

    def resolve_raw_path(self, input_dir: Path) -> Path:
        """Look for ``idmc/flow.csv`` inside *input_dir*."""

        input_dir = input_dir.expanduser().resolve()

        # Primary: subdirectory layout used by idmc/cli.py
        primary = input_dir / "idmc" / "flow.csv"
        if primary.exists():
            return primary

        # Fallback: flat file (e.g. test fixtures or manual staging)
        flat = input_dir / "idmc.csv"
        if flat.exists():
            LOGGER.debug("idmc: using flat file fallback %s", flat)
            return flat

        raise FileNotFoundError(
            f"No IDMC staging CSV found in {input_dir} "
            f"(tried {primary} and {flat})"
        )

    def load(self, raw_path: Path) -> pd.DataFrame:  # type: ignore[override]
        return pd.read_csv(raw_path, dtype=str, keep_default_na=False)

    def map(self, frame: pd.DataFrame) -> pd.DataFrame:  # type: ignore[override]
        df = frame.copy()

        if df.empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        # ------------------------------------------------------------------
        # as_of_date validation — IDMC already writes YYYY-MM-DD month-end
        # ------------------------------------------------------------------
        raw_dates = df.get("as_of_date", pd.Series(dtype=str)).astype(str).str.strip()
        parsed = pd.to_datetime(raw_dates, errors="coerce")
        invalid = int(parsed.isna().sum())
        if invalid:
            LOGGER.warning("idmc: dropping %s rows with invalid as_of_date", invalid)
        df = df.loc[~parsed.isna()].copy()
        parsed = parsed.dropna()
        if df.empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        df["as_of_date"] = parsed.dt.strftime("%Y-%m-%d")

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
            LOGGER.warning("idmc: dropping %s rows with non-numeric value", dropped)
        df = df.loc[~numeric_value.isna()].copy()
        df.loc[:, "value"] = numeric_value.loc[df.index].astype(float)

        if df.empty:
            return pd.DataFrame(columns=CANONICAL_COLUMNS)

        # ------------------------------------------------------------------
        # Assemble canonical output
        # ------------------------------------------------------------------
        canonical = pd.DataFrame(
            {
                "event_id": "",
                "country_name": "",
                "iso3": df.get("iso3", "").str.upper(),
                "hazard_code": "IDU",
                "hazard_label": "Internal Displacement",
                "hazard_class": "displacement",
                "metric": df.get("metric", "new_displacements"),
                "unit": "persons",
                "as_of_date": df["as_of_date"],
                "value": df["value"],
                "series_semantics": df.get("series_semantics", "new"),
                "source": self.canonical_slug,
            },
            index=df.index,
        )

        text_columns = [col for col in CANONICAL_COLUMNS if col != "value"]
        for col in text_columns:
            canonical[col] = canonical[col].fillna("").astype(str)

        canonical["value"] = pd.to_numeric(canonical["value"], errors="coerce").astype(float)
        return canonical


__all__ = ["IDMCAdapter"]
