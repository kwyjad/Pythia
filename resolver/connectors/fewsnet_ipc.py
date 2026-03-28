# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""FEWS NET IPC connector — fetches IPC Phase 3+ population estimates from
the FEWS NET Data Warehouse.

Endpoint: https://fdw.fews.net/api/ipcpopulationsize.csv
No authentication required.

Two scenarios are kept:
- "Current Situation" → metric ``phase3plus_in_need`` (used for resolution)
- "Most Likely"       → metric ``phase3plus_projection`` (context for prompts)

ENV:
    FEWSNET_MONTHS         — months of history to fetch (default 12; 120 for
                             backfill to 2016)
    FEWSNET_REQUEST_DELAY  — seconds between retries (default 1.0)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .protocol import CANONICAL_COLUMNS
from .validate import empty_canonical, validate_canonical

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENDPOINT = "https://fdw.fews.net/api/ipcpopulationsize.csv"

_SCENARIO_METRIC: dict[str, str] = {
    "Current Situation": "phase3plus_in_need",
    "Most Likely": "phase3plus_projection",
}

_COUNTRIES_CSV = Path(__file__).resolve().parent.parent / "data" / "countries.csv"
_FEWSNET_COUNTRIES_JSON = (
    Path(__file__).resolve().parent.parent / "data" / "fewsnet_countries.json"
)


def _load_iso3_to_name() -> dict[str, str]:
    """Load ISO3 → country_name from countries.csv."""
    mapping: dict[str, str] = {}
    try:
        df = pd.read_csv(_COUNTRIES_CSV, usecols=["country_name", "iso3"])
        for _, row in df.iterrows():
            mapping[str(row["iso3"]).strip().upper()] = str(row["country_name"]).strip()
    except Exception as exc:
        LOG.warning("[fewsnet_ipc] failed to load countries.csv: %s", exc)
    return mapping


def _iso2_to_iso3(code: str) -> str | None:
    """Convert an ISO-2 country code to ISO-3 using pycountry."""
    try:
        import pycountry
    except ImportError:
        LOG.error("[fewsnet_ipc] pycountry not installed — cannot convert ISO2→ISO3")
        return None

    code = code.strip().upper()
    if not code or len(code) != 2:
        return None

    try:
        country = pycountry.countries.get(alpha_2=code)
        if country:
            return country.alpha_3
    except Exception:
        pass

    return None


def _build_session() -> requests.Session:
    """Build a requests session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _write_country_list(iso3_codes: list[str]) -> None:
    """Write the list of ISO3 codes with FEWS NET IPC data."""
    try:
        codes = sorted(set(iso3_codes))
        _FEWSNET_COUNTRIES_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(_FEWSNET_COUNTRIES_JSON, "w", encoding="utf-8") as fh:
            json.dump(codes, fh, indent=2)
        LOG.info(
            "[fewsnet_ipc] wrote %d countries to %s",
            len(codes),
            _FEWSNET_COUNTRIES_JSON,
        )
    except Exception as exc:
        LOG.warning("[fewsnet_ipc] failed to write country list: %s", exc)


# ---------------------------------------------------------------------------
# FewsnetIpcConnector
# ---------------------------------------------------------------------------


class FewsnetIpcConnector:
    """Fetch FEWS NET IPC Phase 3+ population data and return a canonical DataFrame."""

    name: str = "fewsnet_ipc"

    def fetch_and_normalize(self) -> pd.DataFrame:
        """Fetch IPC population data and return canonical rows."""
        months_back = int(os.getenv("FEWSNET_MONTHS", "12"))
        delay = float(os.getenv("FEWSNET_REQUEST_DELAY", "1.0"))

        today = date.today()
        total_months = today.year * 12 + today.month - months_back
        start_year = total_months // 12
        start_month = total_months % 12
        if start_month == 0:
            start_month = 12
            start_year -= 1
        start_date = date(start_year, start_month, 1)

        LOG.info(
            "[fewsnet_ipc] fetching IPC data from %s (months_back=%d)",
            start_date.isoformat(),
            months_back,
        )

        # Fetch CSV
        session = _build_session()
        params = {
            "start_date": start_date.isoformat(),
            "format": "csv",
        }

        try:
            resp = session.get(_ENDPOINT, params=params, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            LOG.error("[fewsnet_ipc] fetch failed: %s", exc)
            return empty_canonical()

        if delay > 0:
            time.sleep(delay)

        # Parse CSV (UTF-8 BOM)
        try:
            raw = resp.content.decode("utf-8-sig")
            df_raw = pd.read_csv(StringIO(raw))
        except Exception as exc:
            LOG.error("[fewsnet_ipc] CSV parse failed: %s", exc)
            return empty_canonical()

        if df_raw.empty:
            LOG.warning("[fewsnet_ipc] empty CSV response")
            return empty_canonical()

        LOG.info("[fewsnet_ipc] fetched %d raw rows, columns: %s",
                 len(df_raw), list(df_raw.columns))

        # Filter to wanted scenarios
        df_raw = df_raw[df_raw["scenario_name"].isin(_SCENARIO_METRIC.keys())].copy()
        if df_raw.empty:
            LOG.warning("[fewsnet_ipc] no Current Situation or Most Likely rows")
            return empty_canonical()

        LOG.info(
            "[fewsnet_ipc] %d rows after scenario filter, "
            "unique (country_code, scenario_name) pairs: %d",
            len(df_raw),
            df_raw.groupby(["country_code", "scenario_name"]).ngroups,
        )

        # Convert ISO2 → ISO3
        df_raw["iso3"] = df_raw["country_code"].apply(_iso2_to_iso3)
        bad_iso = df_raw["iso3"].isna().sum()
        if bad_iso > 0:
            LOG.warning(
                "[fewsnet_ipc] dropping %d rows with unconvertible ISO2 codes",
                bad_iso,
            )
            df_raw = df_raw.dropna(subset=["iso3"])

        if df_raw.empty:
            return empty_canonical()

        # Write country list (all ISO3 codes that have any data)
        all_iso3 = df_raw["iso3"].unique().tolist()
        _write_country_list(all_iso3)

        # Derive ym from projection_start
        df_raw["projection_start"] = pd.to_datetime(
            df_raw["projection_start"], errors="coerce"
        )
        df_raw["reporting_date"] = pd.to_datetime(
            df_raw["reporting_date"], errors="coerce"
        )
        df_raw["projection_end"] = pd.to_datetime(
            df_raw["projection_end"], errors="coerce"
        )
        df_raw["ym"] = df_raw["projection_start"].dt.strftime("%Y-%m")

        n_before_dedup = len(df_raw)
        unique_ym = df_raw["ym"].nunique()
        LOG.info(
            "[fewsnet_ipc] before dedup: %d rows, %d unique ym values, "
            "%d unique (iso3, scenario) pairs",
            n_before_dedup,
            unique_ym,
            df_raw.groupby(["iso3", "scenario_name"]).ngroups,
        )

        # Deduplicate: keep latest reporting_date per
        # (iso3, scenario, ym, projection_end).  Earlier versions deduped
        # on (iso3, scenario, ym) alone, which was too aggressive — FEWS NET
        # publishes overlapping analysis windows that share the same
        # projection_start but differ in projection_end (e.g. Oct-Dec vs
        # Oct-Mar).  Including projection_end preserves distinct windows.
        df_raw["_proj_end_str"] = df_raw["projection_end"].dt.strftime("%Y-%m-%d")
        df_raw = df_raw.sort_values("reporting_date", ascending=False)
        df_raw = df_raw.drop_duplicates(
            subset=["iso3", "scenario_name", "ym", "_proj_end_str"], keep="first"
        )
        df_raw = df_raw.drop(columns=["_proj_end_str"])

        LOG.info(
            "[fewsnet_ipc] %d rows after dedup (%d countries, dropped %d dupes)",
            len(df_raw),
            df_raw["iso3"].nunique(),
            n_before_dedup - len(df_raw),
        )

        # Map metric
        df_raw["metric"] = df_raw["scenario_name"].map(_SCENARIO_METRIC)

        # Load country name lookup
        iso3_to_name = _load_iso3_to_name()

        now_utc = datetime.now(timezone.utc).isoformat()

        # Build canonical DataFrame
        rows = []
        for _, r in df_raw.iterrows():
            iso3 = str(r["iso3"])
            scenario = str(r["scenario_name"])
            as_of = (
                r["projection_end"].strftime("%Y-%m-%d")
                if pd.notna(r.get("projection_end"))
                else r["reporting_date"].strftime("%Y-%m-%d")
                if pd.notna(r.get("reporting_date"))
                else ""
            )
            pub_date = (
                r["reporting_date"].strftime("%Y-%m-%d")
                if pd.notna(r.get("reporting_date"))
                else ""
            )

            rows.append(
                {
                    "event_id": "",
                    "country_name": iso3_to_name.get(iso3, ""),
                    "iso3": iso3,
                    "hazard_code": "DR",
                    "hazard_label": "Drought",
                    "hazard_class": "natural",
                    "metric": str(r["metric"]),
                    "series_semantics": "stock",
                    "value": r["value"],
                    "unit": "persons",
                    "as_of_date": as_of,
                    "publication_date": pub_date,
                    "publisher": "FEWS NET",
                    "source_type": "ipc_classification",
                    "source_url": "https://fdw.fews.net",
                    "doc_title": scenario,
                    "definition_text": (
                        f"IPC Phase 3+ (Crisis or worse) population estimate "
                        f"from FEWS NET — {scenario}"
                    ),
                    "method": "ipc_phase_classification",
                    "confidence": "high",
                    "revision": "",
                    "ingested_at": now_utc,
                }
            )

        if not rows:
            return empty_canonical()

        df = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
        return validate_canonical(df, source="fewsnet_ipc")
