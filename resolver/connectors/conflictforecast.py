# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""conflictforecast.org (Mueller/Rauh) conflict forecast connector.

Fetches monthly conflict probability and violence intensity forecasts
from the conflictforecast.org Backendless API. Returns a DataFrame
matching the ``conflict_forecasts`` DuckDB table schema.

Data source:
    - API: Backendless file service
    - Auth: None required (open access)
    - Update cadence: Monthly
    - License: CC-BY (cite Mueller, Rauh & Seimon 2024)
    - Reference: "Introducing a global dataset on conflict forecasts
      and news topics," Data & Policy, vol. 6.
"""

from __future__ import annotations

import io
import logging
import math
import re
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

LOG = logging.getLogger(__name__)

_API_BASE = (
    "https://api.backendless.com"
    "/C177D0DC-B3D5-818C-FF1E-1CC11BC69600"
    "/C5F2917E-C2F6-4F7D-9063-69555274134E"
    "/services/fileService"
)
_TIMEOUT = 60

# File name patterns to match (case-insensitive).
# These map to (metric_name, lead_months).
_FILE_PATTERNS: List[Tuple[str, str, int]] = [
    (r"armed.?conflict.*3", "cf_armed_conflict_risk_3m", 3),
    (r"armed.?conflict.*12", "cf_armed_conflict_risk_12m", 12),
    (r"violence.?intensity.*3", "cf_violence_intensity_3m", 3),
]


class ConflictForecastOrgConnector:
    """Fetch conflictforecast.org country-level forecasts."""

    name: str = "conflictforecast_org"

    def fetch_forecasts(self) -> pd.DataFrame:
        """Fetch the latest Armed Conflict and Violence Intensity CSVs.

        Returns a DataFrame with columns matching the ``conflict_forecasts``
        table: source, iso3, hazard_code, metric, lead_months, value,
        forecast_issue_date, target_month, model_version.
        """
        try:
            file_listing = self._get_file_listing()
        except Exception as exc:
            LOG.warning("[conflictforecast_org] file listing failed: %s", exc)
            return pd.DataFrame()

        if not file_listing:
            LOG.info("[conflictforecast_org] empty file listing")
            return pd.DataFrame()

        issue_date = self._derive_issue_date(file_listing)
        all_rows: List[Dict[str, Any]] = []

        for pattern, metric, lead_months in _FILE_PATTERNS:
            url = self._find_file_url(file_listing, pattern)
            if not url:
                LOG.warning(
                    "[conflictforecast_org] no file matching '%s' in listing",
                    pattern,
                )
                continue

            try:
                df_csv = self._download_csv(url)
            except Exception as exc:
                LOG.warning(
                    "[conflictforecast_org] failed to download %s: %s",
                    metric, exc,
                )
                continue

            rows = self._transform_csv(df_csv, metric, lead_months, issue_date)
            all_rows.extend(rows)

        if not all_rows:
            LOG.info("[conflictforecast_org] no rows produced")
            return pd.DataFrame()

        df = pd.DataFrame(all_rows)
        LOG.info("[conflictforecast_org] produced %d forecast rows", len(df))
        return df

    # ------------------------------------------------------------------
    # API interaction
    # ------------------------------------------------------------------

    def _get_file_listing(self) -> List[Dict[str, Any]]:
        """Call the get-latest-file-listing endpoint."""
        url = f"{_API_BASE}/get-latest-file-listing"
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return []

    @staticmethod
    def _find_file_url(
        listing: List[Dict[str, Any]], pattern: str
    ) -> Optional[str]:
        """Find the publicUrl for a file matching the given name pattern."""
        regex = re.compile(pattern, re.IGNORECASE)
        for entry in listing:
            name = entry.get("name") or ""
            if regex.search(name):
                return entry.get("publicUrl") or ""
        return None

    @staticmethod
    def _download_csv(url: str) -> pd.DataFrame:
        """Download a CSV from the given URL and return a DataFrame."""
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        return pd.read_csv(io.StringIO(resp.text))

    @staticmethod
    def _derive_issue_date(listing: List[Dict[str, Any]]) -> date:
        """Derive forecast issue date from the file listing metadata.

        The listing entries may contain a 'created' or 'date' field, or
        we can infer from file names (which often contain MM-YYYY).
        Falls back to today's date.
        """
        for entry in listing:
            name = entry.get("name") or ""
            # Try to extract MM-YYYY from file name
            match = re.search(r"(\d{2})-(\d{4})", name)
            if match:
                month, year = int(match.group(1)), int(match.group(2))
                if 1 <= month <= 12 and 2020 <= year <= 2030:
                    return date(year, month, 1)

            # Try 'created' timestamp
            created = entry.get("created")
            if created and isinstance(created, (int, float)):
                from datetime import datetime
                try:
                    dt = datetime.fromtimestamp(created / 1000)  # ms epoch
                    return dt.date().replace(day=1)
                except Exception:
                    pass

        return date.today().replace(day=1)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    @staticmethod
    def _transform_csv(
        df: pd.DataFrame,
        metric: str,
        lead_months: int,
        issue_date: date,
    ) -> List[Dict[str, Any]]:
        """Transform a downloaded CSV into conflict_forecasts rows.

        The CSVs typically have columns like: country, iso3 (or iso, isocode),
        and a value column. We detect the ISO3 and value columns dynamically.
        """
        if df.empty:
            return []

        # Detect ISO3 column (case-insensitive)
        iso3_col = None
        for col in df.columns:
            if col.lower() in ("iso3", "iso", "isocode", "iso_code", "isoab"):
                iso3_col = col
                break
        if iso3_col is None:
            LOG.warning(
                "[conflictforecast_org] no ISO3 column found in CSV "
                "(columns: %s)", list(df.columns),
            )
            return []

        # Detect value column: use the last numeric column that isn't the
        # ISO column or a country-name column.
        value_col = None
        skip_cols = {iso3_col.lower(), "country", "name", "country_name", "region"}
        for col in reversed(list(df.columns)):
            if col.lower() in skip_cols:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                value_col = col
                break

        if value_col is None:
            # Try coercing columns to numeric
            for col in reversed(list(df.columns)):
                if col.lower() in skip_cols:
                    continue
                try:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                    if df[col].notna().any():
                        value_col = col
                        break
                except Exception:
                    continue

        if value_col is None:
            LOG.warning(
                "[conflictforecast_org] no numeric value column found in CSV "
                "(columns: %s)", list(df.columns),
            )
            return []

        # Compute target month
        target_year = issue_date.year + (issue_date.month + lead_months - 1) // 12
        target_month_num = (issue_date.month + lead_months - 1) % 12 + 1
        target_month = date(target_year, target_month_num, 1)

        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            iso3 = str(row.get(iso3_col, "")).strip().upper()
            if not iso3 or len(iso3) != 3:
                continue

            val = row.get(value_col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                continue

            try:
                val = float(val)
            except (ValueError, TypeError):
                continue

            rows.append({
                "source": "conflictforecast_org",
                "iso3": iso3,
                "hazard_code": "AC",
                "metric": metric,
                "lead_months": lead_months,
                "value": val,
                "forecast_issue_date": issue_date,
                "target_month": target_month,
                "model_version": "conflictforecast_org",
            })

        return rows
