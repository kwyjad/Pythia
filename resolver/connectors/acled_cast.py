# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ACLED CAST (Conflict Alert System Tool) forecast connector.

Fetches monthly country-level conflict event count forecasts from the
ACLED CAST API.  Returns a DataFrame matching the ``conflict_forecasts``
DuckDB table schema.

Data source:
    - API: https://acleddata.com/api/cast/read
    - Auth: ACLED OAuth2 (shared with the ACLED event connector)
    - Update cadence: First/second Thursday of each month
    - Variables: total_forecast, battles_forecast, erv_forecast, vac_forecast
    - Spatial: admin1 level (aggregated to country here)
    - Temporal: 6 months ahead

CAST is the only conflict forecast source that disaggregates by event type
(battles vs. explosions/remote violence vs. violence against civilians).
It predicts event *counts*, not fatalities.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

LOG = logging.getLogger(__name__)

_API_URL = "https://acleddata.com/api/cast/read"
_TIMEOUT = 90
_PAGE_SIZE = 5000
_MAX_PAGES = 20  # safety valve: 100,000 rows max
_MAX_LEAD_MONTHS = 6

_MONTH_MAP: Dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

# CAST-specific country name overrides that supplement the default aliases
# in resolver.ingestion.utils.iso_normalize.
_CAST_ALIASES: Dict[str, str] = {
    "Republic of Congo": "COG",
    "Congo": "COG",
    "Ivory Coast": "CIV",
    "Burma": "MMR",
    "Burma/Myanmar": "MMR",
    "eSwatini": "SWZ",
    "Eswatini": "SWZ",
    "Palestine": "PSE",
    "Occupied Palestinian Territory": "PSE",
    "Somaliland": "SOM",
    "Western Sahara": "ESH",
    "Czechia": "CZE",
    "East Timor": "TLS",
    "Timor-Leste": "TLS",
    "Cabo Verde": "CPV",
    "Cape Verde": "CPV",
    "Türkiye": "TUR",
    "Turkey": "TUR",
    "Republic of Korea": "KOR",
    "South Korea": "KOR",
    "North Korea": "PRK",
    "Korea, South": "KOR",
    "Korea, North": "PRK",
    "Guinea": "GIN",
}

_METRIC_MAP: Dict[str, str] = {
    "total_forecast": "cast_total_events",
    "battles_forecast": "cast_battles_events",
    "erv_forecast": "cast_erv_events",
    "vac_forecast": "cast_vac_events",
}

_FORECAST_FIELDS = list(_METRIC_MAP.keys())


class AcledCastConnector:
    """Fetch ACLED CAST country-level event count forecasts."""

    name: str = "acled_cast"

    def fetch_forecasts(self) -> pd.DataFrame:
        """Fetch the latest ACLED CAST forecasts, aggregated to country level.

        Returns a DataFrame with columns matching the ``conflict_forecasts``
        table: source, iso3, hazard_code, metric, lead_months, value,
        forecast_issue_date, target_month, model_version.
        """
        try:
            records = self._fetch_all_records()
        except Exception as exc:
            LOG.warning("[acled_cast] fetch failed: %s", exc)
            return pd.DataFrame()

        if not records:
            LOG.info("[acled_cast] no records returned from API")
            return pd.DataFrame()

        issue_date = self._derive_issue_date(records)
        aggregated = self._aggregate_to_country(records)

        if aggregated.empty:
            LOG.info("[acled_cast] no data after aggregation")
            return pd.DataFrame()

        rows = self._transform(aggregated, issue_date)

        if not rows:
            LOG.info("[acled_cast] no rows after transformation")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        LOG.info("[acled_cast] produced %d forecast rows", len(df))
        return df

    # ------------------------------------------------------------------
    # API interaction
    # ------------------------------------------------------------------

    def _fetch_all_records(self) -> List[Dict[str, Any]]:
        """Paginate through the ACLED CAST endpoint."""
        from resolver.ingestion.acled_auth import get_auth_header

        headers = get_auth_header()
        headers["Accept"] = "application/json"
        all_records: List[Dict[str, Any]] = []

        for page in range(1, _MAX_PAGES + 1):
            LOG.debug("[acled_cast] fetching page %d", page)
            params: Dict[str, Any] = {
                "limit": _PAGE_SIZE,
                "page": page,
            }
            resp = requests.get(
                _API_URL,
                params=params,
                headers=headers,
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            body = resp.json()

            # Extract data array from response
            data: list = []
            if isinstance(body, dict):
                data = body.get("data", [])
                if not isinstance(data, list):
                    data = []
            elif isinstance(body, list):
                data = body

            if not data:
                break

            all_records.extend(data)
            LOG.debug("[acled_cast] page %d returned %d records", page, len(data))

            if len(data) < _PAGE_SIZE:
                break  # last page

            # Brief courtesy sleep between pages
            time.sleep(0.5)

        LOG.info("[acled_cast] fetched %d total records", len(all_records))
        return all_records

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _aggregate_to_country(
        self, records: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Aggregate admin1-level CAST records to country level.

        Groups by (country, month, year) and sums forecast fields.
        Converts month names to numbers.
        """
        rows: List[Dict[str, Any]] = []
        unmapped_countries: set[str] = set()

        for rec in records:
            country = (rec.get("country") or "").strip()
            month_str = (rec.get("month") or "").strip().lower()
            year = rec.get("year")

            if not country or not month_str or year is None:
                continue

            month_num = _MONTH_MAP.get(month_str)
            if month_num is None:
                LOG.debug("[acled_cast] unknown month name: %r", month_str)
                continue

            row: Dict[str, Any] = {
                "country": country,
                "month_num": month_num,
                "year": int(year),
            }
            for field in _FORECAST_FIELDS:
                val = rec.get(field)
                try:
                    row[field] = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    row[field] = 0.0

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Sum admin1 forecasts to country level
        agg_cols = {f: "sum" for f in _FORECAST_FIELDS}
        grouped = (
            df.groupby(["country", "month_num", "year"], as_index=False)
            .agg(agg_cols)
        )

        # Resolve ISO3 codes
        from resolver.ingestion.utils.iso_normalize import to_iso3

        iso3_codes = []
        for country_name in grouped["country"]:
            iso3 = to_iso3(country_name, _CAST_ALIASES)
            if not iso3:
                unmapped_countries.add(country_name)
            iso3_codes.append(iso3)

        grouped["iso3"] = iso3_codes

        if unmapped_countries:
            LOG.warning(
                "[acled_cast] %d countries could not be mapped to ISO3: %s",
                len(unmapped_countries),
                sorted(unmapped_countries),
            )

        # Drop rows without ISO3
        grouped = grouped[grouped["iso3"].notna()].copy()
        return grouped

    # ------------------------------------------------------------------
    # Issue date
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_issue_date(records: List[Dict[str, Any]]) -> date:
        """Derive forecast issue date from the timestamp field.

        Takes the maximum timestamp across all records and normalizes
        to the 1st of its month.  Falls back to today's date.
        """
        max_dt: Optional[datetime] = None

        for rec in records:
            ts = rec.get("timestamp")
            if ts is None:
                continue
            try:
                if isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(float(ts))
                elif isinstance(ts, str):
                    # Try ISO format first, then epoch string
                    try:
                        dt = datetime.fromisoformat(
                            ts.replace("Z", "+00:00")
                        )
                    except ValueError:
                        dt = datetime.fromtimestamp(float(ts))
                else:
                    continue
                if max_dt is None or dt > max_dt:
                    max_dt = dt
            except (ValueError, TypeError, OSError):
                continue

        if max_dt:
            return max_dt.date().replace(day=1)
        return date.today().replace(day=1)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _transform(
        self,
        aggregated: pd.DataFrame,
        issue_date: date,
    ) -> List[Dict[str, Any]]:
        """Transform country-aggregated data into conflict_forecasts rows.

        Emits 4 rows per country per lead month (one per metric).
        """
        base_year, base_month = issue_date.year, issue_date.month
        rows: List[Dict[str, Any]] = []

        for _, rec in aggregated.iterrows():
            iso3 = rec.get("iso3")
            if not iso3 or not isinstance(iso3, str) or len(iso3) != 3:
                continue

            year = int(rec["year"])
            month = int(rec["month_num"])

            # Compute lead months from issue date
            lead = (year - base_year) * 12 + (month - base_month)
            if lead < 1 or lead > _MAX_LEAD_MONTHS:
                continue

            target = date(year, month, 1)

            for src_field, metric_name in _METRIC_MAP.items():
                val = rec.get(src_field)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    continue
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    continue

                rows.append(
                    {
                        "source": "ACLED_CAST",
                        "iso3": iso3.upper(),
                        "hazard_code": "AC",
                        "metric": metric_name,
                        "lead_months": lead,
                        "value": val,
                        "forecast_issue_date": issue_date,
                        "target_month": target,
                        "model_version": "cast",
                    }
                )

        return rows
