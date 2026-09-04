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

What the September 2026 review settled, verified against the live API with
a valid token on 2026-09-04 (see CLAUDE.md, run 33841370196, Group B):

* ``year`` filters the TARGET year; ``year=2026`` returns 9,879 records on
  two pages, every one stamped 2025-12-10, and ``year=2027`` returns none.
  The feed is frozen upstream; the connector, credentials and pagination
  all work. The stale vintage is therefore FLAGGED on every row it writes
  (``conflict_forecasts.is_stale_vintage`` / ``vintage_age_days``) and the
  prompt readers serve only rows whose target month has not passed.
* The unfiltered fallback is GONE. The unfiltered endpoint pages the
  archive oldest first (page 1 is 2023-24 rows), so a fallback written to
  avoid regressing to zero data would have ingested two-year-old forecasts
  and reported success. An empty result stays empty.
* Pagination loses nothing today (9,879 rows, 9,879 distinct (country,
  admin1, month, year) keys), but ``_aggregate_to_country`` SUMS across
  every record, so a paging change that repeated a row across a boundary
  would double the counts silently. Records are deduplicated on that key
  first and the drop count is logged beside the raw count.
* A vintage carries the months it carries. December 2025 has five target
  months, January to May 2026; June is absent from the API, not lost here.
  The target months found are logged and reported, never inferred from
  ``_MAX_LEAD_MONTHS``.
* Every country-month that leaves the pipeline between aggregation and the
  written rows is named, with its reason (no ISO3, lead outside 1..6).
"""

from __future__ import annotations

import logging
import math
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

LOG = logging.getLogger(__name__)

_API_URL = "https://acleddata.com/api/cast/read"
_TIMEOUT = 90
_PAGE_SIZE = 5000
_MAX_PAGES = 20  # safety valve: 100,000 rows max
_MAX_LEAD_MONTHS = 6
# Warn loudly when the newest fetched forecast is older than this. CAST is
# issued monthly (first/second Thursday), so a healthy feed is < ~35 days old;
# 45 mirrors the conflict-forecast staleness threshold used elsewhere.
_STALENESS_WARN_DAYS = 45

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

    def __init__(self) -> None:
        #: What the last fetch found, for the run summary. Filled by
        #: fetch_forecasts; read by resolver.tools.fetch_conflict_forecasts.
        self.summary: Dict[str, Any] = {}

    def fetch_forecasts(self) -> pd.DataFrame:
        """Fetch the latest ACLED CAST forecasts, aggregated to country level.

        Returns a DataFrame with columns matching the ``conflict_forecasts``
        table: source, iso3, hazard_code, metric, lead_months, value,
        forecast_issue_date, target_month, model_version.
        """
        self.summary = {}
        try:
            # `year` filters the forecast TARGET year, and CAST issues
            # forecasts up to six months ahead — so late in the year the
            # freshest issue's target months spill into next year. Fetch
            # BOTH target years and merge. There is deliberately NO
            # unfiltered fallback: that endpoint pages the archive oldest
            # first, and older data presented as current is worse than none.
            current_year = date.today().year
            records = self._fetch_all_records(year=current_year)
            try:
                next_records = self._fetch_all_records(year=current_year + 1)
            except Exception as exc:  # pragma: no cover - defensive
                next_records = []
                LOG.warning(
                    "[acled_cast] year=%d pull failed (%s) — continuing with "
                    "year=%d records only",
                    current_year + 1, exc, current_year,
                )
            # Never trust an ACLED request-side filter — verify by each
            # record's OWN year field so an ignored/buggy `year` param can't
            # double-count the same rows across the two pulls.
            records += [
                rec for rec in next_records
                if str(rec.get("year", "")).strip() == str(current_year + 1)
            ]
        except Exception as exc:
            LOG.error(
                "[acled_cast] fetch failed — CAST forecasts will be MISSING "
                "from this ingest cycle: %s", exc,
            )
            self.summary = {"error": str(exc)[:300]}
            return pd.DataFrame()

        if not records:
            LOG.info(
                "[acled_cast] no records for target years %d or %d — writing "
                "nothing (an empty result stays empty; there is no unfiltered "
                "fallback because it serves the archive oldest first)",
                current_year, current_year + 1,
            )
            self.summary = {"records_fetched": 0}
            return pd.DataFrame()

        issue_date = self._derive_issue_date(records)
        age_days = (date.today() - issue_date).days
        self.summary["records_fetched"] = len(records)
        self.summary["issue_date"] = issue_date.isoformat()
        self.summary["vintage_age_days"] = age_days
        if age_days > _STALENESS_WARN_DAYS:
            LOG.warning(
                "[acled_cast] latest forecast issue date %s is %d days old "
                "(> %d) — the ACLED CAST API is not serving current forecasts. "
                "Every row written carries is_stale_vintage=true, and the "
                "prompt readers serve only target months that have not "
                "passed. Escalate to ACLED; the connector and auth are "
                "working (it fetched %d records).",
                issue_date.isoformat(), age_days, _STALENESS_WARN_DAYS,
                len(records),
            )
        aggregated = self._aggregate_to_country(records)

        if aggregated.empty:
            LOG.info("[acled_cast] no data after aggregation")
            return pd.DataFrame()

        rows = self._transform(aggregated, issue_date)

        if not rows:
            LOG.info("[acled_cast] no rows after transformation")
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        target_months = sorted({str(t) for t in df["target_month"]})
        country_months = df[["iso3", "target_month"]].drop_duplicates()
        self.summary["target_months"] = target_months
        self.summary["countries"] = int(df["iso3"].nunique())
        self.summary["country_months_written"] = int(len(country_months))
        # Which countries carry fewer target months than the vintage does:
        # 88 countries x 5 months is 440, and 438 was what the September run
        # produced. A country with a month missing upstream is named here.
        per_country = country_months.groupby("iso3").size()
        gaps = {
            str(iso3): int(len(target_months) - n)
            for iso3, n in per_country.items() if n < len(target_months)
        }
        self.summary["countries_with_month_gaps"] = gaps
        LOG.info(
            "[acled_cast] produced %d forecast rows: %d countries x target "
            "months %s (%d country-months; %d expected if every country "
            "carried every month; countries with gaps: %s)",
            len(df), self.summary["countries"], target_months,
            len(country_months), self.summary["countries"] * len(target_months),
            gaps or "none",
        )
        if len(target_months) < _MAX_LEAD_MONTHS:
            LOG.warning(
                "[acled_cast] the %s vintage carries %d target month(s) %s, "
                "not the documented %d — the missing months are absent from "
                "the API, not dropped here",
                issue_date.isoformat(), len(target_months), target_months,
                _MAX_LEAD_MONTHS,
            )
        return df

    # ------------------------------------------------------------------
    # API interaction
    # ------------------------------------------------------------------

    def _fetch_all_records(
        self, year: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Paginate through the ACLED CAST endpoint.

        When *year* is provided, ACLED's documented ``year`` filter restricts
        the response to that forecast year. Note ``year`` filters the forecast
        *target* year, so a persistently stale ``forecast_issue_date`` even
        with the filter set indicates upstream staleness, not a query bug.
        """
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
            if year is not None:
                params["year"] = year
            resp = requests.get(
                _API_URL,
                params=params,
                headers=headers,
                timeout=_TIMEOUT,
            )
            # The single ACLED error path: an HTML page (the gateway's
            # "Unauthorized" page comes with a 200) raises a named failure
            # and is never read as an empty page of records.
            from resolver.ingestion.acled_auth import parse_json_response

            body = parse_json_response(resp, what="CAST read")

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

        # Deduplicate on the admin1 key BEFORE the SUM below. ACLED orders
        # by month and splits a month across the page boundary; today that
        # repeats nothing (9,879 rows, 9,879 distinct keys on 2026-09-04),
        # but a paging change that repeated a row would double every count
        # silently. The distinct-key count is logged beside the raw count
        # so "fetched N records" can never mean N rows of which some are
        # the same row twice.
        seen: set[tuple] = set()
        deduplicated: List[Dict[str, Any]] = []
        for rec in records:
            key = (
                (rec.get("country") or "").strip(),
                (rec.get("admin1") or "").strip(),
                (rec.get("month") or "").strip().lower(),
                str(rec.get("year") or "").strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(rec)
        dropped = len(records) - len(deduplicated)
        self.summary["distinct_admin1_keys"] = len(deduplicated)
        self.summary["duplicate_records_dropped"] = dropped
        LOG.info(
            "[acled_cast] %d records fetched, %d distinct (country, admin1, "
            "month, year) keys, %d duplicate record(s) dropped before "
            "aggregation",
            len(records), len(deduplicated), dropped,
        )
        if dropped:
            LOG.warning(
                "[acled_cast] ACLED's paging repeated %d record(s) across a "
                "boundary — the SUM would have doubled them", dropped,
            )

        for rec in deduplicated:
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

        self.summary["country_months_aggregated"] = int(len(grouped))
        if unmapped_countries:
            lost = grouped[grouped["iso3"].isna()]
            named = sorted(
                f"{r.country} {int(r.year)}-{int(r.month_num):02d}"
                for r in lost.itertuples()
            )
            self.summary["country_months_dropped_no_iso3"] = named
            LOG.warning(
                "[acled_cast] %d countries could not be mapped to ISO3 (%s); "
                "%d country-month(s) dropped: %s",
                len(unmapped_countries), sorted(unmapped_countries),
                len(named), named,
            )
        else:
            self.summary["country_months_dropped_no_iso3"] = []

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
                # Epoch timestamps are read in UTC: datetime.fromtimestamp
                # without a zone is runner-local, and a runner in another
                # zone would file a 23:00 UTC issue under the next day.
                if isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
                elif isinstance(ts, str):
                    # Try ISO format first, then epoch string
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                    except ValueError:
                        dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
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
        dropped_lead: List[str] = []
        dropped_iso3: List[str] = []

        for _, rec in aggregated.iterrows():
            iso3 = rec.get("iso3")
            year = int(rec["year"])
            month = int(rec["month_num"])
            if not iso3 or not isinstance(iso3, str) or len(iso3) != 3:
                dropped_iso3.append(f"{rec.get('country')} {year}-{month:02d}")
                continue

            # Compute lead months from issue date
            lead = (year - base_year) * 12 + (month - base_month)
            if lead < 1 or lead > _MAX_LEAD_MONTHS:
                dropped_lead.append(f"{iso3} {year}-{month:02d} (lead {lead})")
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

        # A country-month that vanishes silently is the same class of fault
        # as a figure that vanishes silently: name each one and its reason.
        self.summary["country_months_dropped_by_lead_filter"] = dropped_lead
        if dropped_iso3:
            self.summary.setdefault("country_months_dropped_no_iso3", [])
            self.summary["country_months_dropped_no_iso3"] += dropped_iso3
        if dropped_lead:
            LOG.warning(
                "[acled_cast] %d country-month(s) dropped by the lead filter "
                "(outside 1..%d months from the %s issue): %s",
                len(dropped_lead), _MAX_LEAD_MONTHS, issue_date.isoformat(),
                dropped_lead,
            )
        return rows
