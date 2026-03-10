# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""VIEWS (Uppsala/PRIO) conflict forecast connector.

Fetches country-month state-based conflict forecasts from the VIEWS
Early Warning System API. Returns a DataFrame matching the
``conflict_forecasts`` DuckDB table schema.

Data source:
    - API: https://api.viewsforecasting.org
    - Model: fatalities003 (current production model)
    - Auth: None required (open access)
    - Update cadence: Monthly (~mid-month)
    - License: Open access, cite Hegre et al. and the VIEWS project.

Key variables:
    - main_mean: predicted fatalities (natural scale, ensemble mean)
    - main_dich: P(≥25 battle-related deaths) per country-month
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

LOG = logging.getLogger(__name__)

_API_BASE = "https://api.viewsforecasting.org"
_TIMEOUT = 60
_PAGESIZE = 1000
_MAX_LEAD_MONTHS = 6


class ViewsConnector:
    """Fetch VIEWS fatalities003 cm/sb forecasts."""

    name: str = "views"

    def fetch_forecasts(self) -> pd.DataFrame:
        """Fetch the latest VIEWS cm/sb forecasts, filtered to 6 lead months.

        Returns a DataFrame with columns matching the ``conflict_forecasts``
        table: source, iso3, hazard_code, metric, lead_months, value,
        forecast_issue_date, target_month, model_version.
        """
        try:
            run_id = self._detect_run_id()
            records = self._fetch_all_pages(run_id)
        except Exception as exc:
            LOG.warning("[views] fetch failed: %s", exc)
            return pd.DataFrame()

        if not records:
            LOG.info("[views] no records returned from API")
            return pd.DataFrame()

        model_version = self._extract_model_version(run_id)
        issue_date = self._derive_issue_date(records)

        rows = self._transform(records, issue_date, model_version)

        if not rows:
            LOG.info("[views] no rows after transformation")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        LOG.info("[views] produced %d forecast rows", len(df))
        return df

    # ------------------------------------------------------------------
    # API interaction
    # ------------------------------------------------------------------

    def _fetch_all_pages(self, run_id: str = "current") -> List[Dict[str, Any]]:
        """Paginate through the VIEWS cm/sb endpoint."""
        all_records: List[Dict[str, Any]] = []
        url: Optional[str] = f"{_API_BASE}/{run_id}/cm/sb"
        page = 1

        while url:
            LOG.debug("[views] fetching page %d: %s", page, url)
            resp = requests.get(
                url,
                params={"page": page, "pagesize": _PAGESIZE} if page == 1 else None,
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            body = resp.json()

            data = body.get("data") or body.get("results")
            if data is None:
                # Try treating body as list directly, or extract from top-level
                if isinstance(body, list):
                    data = body
                else:
                    # VIEWS API nests data under the pagination wrapper
                    data = [
                        v for k, v in body.items()
                        if isinstance(v, list) and k not in (
                            "next_page", "prev_page", "page_count",
                            "page_cur", "row_count", "start_date",
                            "end_date", "model_tree",
                        )
                    ]
                    if data:
                        data = data[0]
                    else:
                        # Records are embedded directly — flatten non-meta keys
                        meta_keys = {
                            "next_page", "prev_page", "page_count",
                            "page_cur", "row_count", "start_date",
                            "end_date", "model_tree",
                        }
                        records_from_body = []
                        for k, v in body.items():
                            if k not in meta_keys and isinstance(v, dict):
                                records_from_body.append(v)
                        data = records_from_body if records_from_body else []

            if isinstance(data, list):
                all_records.extend(data)
            elif isinstance(data, dict):
                all_records.extend(data.values())

            next_page = body.get("next_page", "") if isinstance(body, dict) else ""
            if next_page and isinstance(next_page, str) and next_page.strip():
                url = next_page
                page += 1
            else:
                url = None

        return all_records

    def _detect_run_id(self) -> str:
        """Fetch the latest run ID from the API root."""
        try:
            resp = requests.get(_API_BASE, timeout=_TIMEOUT)
            resp.raise_for_status()
            body = resp.json()
            runs = body.get("runs", []) if isinstance(body, dict) else body
            if isinstance(runs, list):
                # Pick the latest fatalities003 run, falling back to older models
                for prefix in ("fatalities003", "fatalities002", "fatalities001"):
                    matches = [r for r in runs if isinstance(r, str) and r.startswith(prefix)]
                    if matches:
                        return sorted(matches)[-1]
        except Exception as exc:
            LOG.debug("[views] could not detect run_id: %s", exc)
        return "current"

    @staticmethod
    def _extract_model_version(run_id: str) -> str:
        """Extract model name from run_id like 'fatalities003_2025_12_t01'."""
        parts = run_id.split("_")
        return parts[0] if parts else "fatalities003"

    @staticmethod
    def _derive_issue_date(records: List[Dict[str, Any]]) -> date:
        """Derive forecast issue date from the earliest target month minus 1.

        VIEWS forecasts start from the month after the last observed data.
        The issue date is approximately the month of that last observation.
        """
        min_year, min_month = 9999, 12
        for rec in records:
            y = rec.get("year")
            m = rec.get("month")
            if y is not None and m is not None:
                if (int(y), int(m)) < (min_year, min_month):
                    min_year, min_month = int(y), int(m)

        if min_year == 9999:
            return date.today().replace(day=1)

        # Issue date is the month before the first forecast month
        if min_month == 1:
            return date(min_year - 1, 12, 1)
        return date(min_year, min_month - 1, 1)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _transform(
        self,
        records: List[Dict[str, Any]],
        issue_date: date,
        model_version: str,
    ) -> List[Dict[str, Any]]:
        """Transform raw API records into conflict_forecasts rows.

        Emits two rows per country per lead month:
        - views_predicted_fatalities (main_mean, natural scale)
        - views_p_gte25_brd (main_dich, probability 0-1)
        """
        # Determine base month (issue_date) to compute lead_months
        base_year, base_month = issue_date.year, issue_date.month

        rows: List[Dict[str, Any]] = []
        for rec in records:
            iso3 = (rec.get("isoab") or "").strip().upper()
            if not iso3 or len(iso3) != 3:
                continue

            year = rec.get("year")
            month = rec.get("month")
            if year is None or month is None:
                continue
            year, month = int(year), int(month)

            # Compute lead months from base
            lead = (year - base_year) * 12 + (month - base_month)
            if lead < 1 or lead > _MAX_LEAD_MONTHS:
                continue

            target = date(year, month, 1)

            # Predicted fatalities (natural scale)
            main_mean = rec.get("main_mean")
            if main_mean is not None:
                try:
                    val = float(main_mean)
                    if not math.isnan(val):
                        rows.append({
                            "source": "VIEWS",
                            "iso3": iso3,
                            "hazard_code": "AC",
                            "metric": "views_predicted_fatalities",
                            "lead_months": lead,
                            "value": val,
                            "forecast_issue_date": issue_date,
                            "target_month": target,
                            "model_version": model_version,
                        })
                except (ValueError, TypeError):
                    pass

            # P(≥25 BRD)
            main_dich = rec.get("main_dich")
            if main_dich is not None:
                try:
                    val = float(main_dich)
                    if not math.isnan(val):
                        rows.append({
                            "source": "VIEWS",
                            "iso3": iso3,
                            "hazard_code": "AC",
                            "metric": "views_p_gte25_brd",
                            "lead_months": lead,
                            "value": val,
                            "forecast_issue_date": issue_date,
                            "target_month": target,
                            "model_version": model_version,
                        })
                except (ValueError, TypeError):
                    pass

        return rows
