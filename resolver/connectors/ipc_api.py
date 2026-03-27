# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IPC API connector — fetches IPC Phase 3+ population estimates from
the IPC public API (api.ipcinfo.org).

Supplements FEWS NET data for countries not covered by FEWS NET.  Countries
already in ``fewsnet_countries.json`` are excluded to prevent overlap.

Endpoint: https://api.ipcinfo.org/population?format=json
Requires API key (``IPC_API_KEY`` env var).

Two period types are kept:
- ``ipc_period = "A"`` (Current)   → metric ``phase3plus_in_need`` (resolution)
- ``ipc_period = "P"`` (Projected) → metric ``phase3plus_projection`` (prompt context)

ENV:
    IPC_API_KEY            — API key for api.ipcinfo.org (required)
    IPC_API_MONTHS         — months of history to fetch (default 24)
    IPC_API_REQUEST_DELAY  — seconds between retries (default 1.0)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import date, datetime, timezone
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

_API_BASE = "https://api.ipcinfo.org"
_POPULATION_ENDPOINT = f"{_API_BASE}/population"

# IPC period codes → canonical metric names (mirrors FEWS NET mapping)
_PERIOD_METRIC: dict[str, str] = {
    "A": "phase3plus_in_need",      # Current Situation → used for resolution
    "P": "phase3plus_projection",   # First Projection → prompt context
}

_PERIOD_LABELS: dict[str, str] = {
    "A": "Current Situation",
    "P": "First Projection",
}

_COUNTRIES_CSV = Path(__file__).resolve().parent.parent / "data" / "countries.csv"
_FEWSNET_COUNTRIES_JSON = (
    Path(__file__).resolve().parent.parent / "data" / "fewsnet_countries.json"
)
_IPC_COUNTRIES_JSON = (
    Path(__file__).resolve().parent.parent / "data" / "ipc_countries.json"
)


def _load_iso3_to_name() -> dict[str, str]:
    """Load ISO3 → country_name from countries.csv."""
    mapping: dict[str, str] = {}
    try:
        df = pd.read_csv(_COUNTRIES_CSV, usecols=["country_name", "iso3"])
        for _, row in df.iterrows():
            mapping[str(row["iso3"]).strip().upper()] = str(row["country_name"]).strip()
    except Exception as exc:
        LOG.warning("[ipc_api] failed to load countries.csv: %s", exc)
    return mapping


def _load_fewsnet_country_list() -> set[str]:
    """Load FEWS NET-monitored country ISO3 codes (to exclude from IPC)."""
    try:
        with open(_FEWSNET_COUNTRIES_JSON, encoding="utf-8") as fh:
            codes = json.load(fh)
        return {c.upper() for c in codes}
    except FileNotFoundError:
        LOG.warning(
            "[ipc_api] %s not found — cannot exclude FEWS NET countries; "
            "returning all IPC countries",
            _FEWSNET_COUNTRIES_JSON,
        )
        return set()
    except Exception as exc:
        LOG.warning("[ipc_api] failed to load fewsnet_countries.json: %s", exc)
        return set()


def _iso2_to_iso3(code: str) -> str | None:
    """Convert an ISO-2 country code to ISO-3 using pycountry."""
    try:
        import pycountry
    except ImportError:
        LOG.error("[ipc_api] pycountry not installed — cannot convert ISO2→ISO3")
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


def _parse_period_dates(date_str: str) -> tuple[str, str]:
    """Parse IPC API period date strings into (start_iso, end_iso).

    The IPC API returns dates in ``current_period_dates`` and
    ``projected_period_dates`` as human-readable ranges like:
        "Jun 2025 - Sep 2025"
        "June 2025 - September 2025"
        "2025-06-01 - 2025-09-30"
        "Jun 2025"  (single month)

    Returns ``("", "")`` if parsing fails.
    """
    if not date_str or not isinstance(date_str, str):
        return ("", "")

    date_str = date_str.strip()

    # Split on common separators: " - ", " to ", " – " (en-dash).
    # Require at least one space on each side so ISO dates (2025-06-01) are not broken.
    parts = re.split(r"\s+[-–]\s+|\s+to\s+", date_str, maxsplit=1)

    results: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            dt = pd.to_datetime(part, format="mixed", dayfirst=False)
            results.append(dt.strftime("%Y-%m-%d"))
        except Exception:
            # Try common IPC formats explicitly
            for fmt in ("%b %Y", "%B %Y", "%Y-%m-%d", "%Y-%m", "%d/%m/%Y"):
                try:
                    dt = datetime.strptime(part, fmt)
                    results.append(dt.strftime("%Y-%m-%d"))
                    break
                except ValueError:
                    continue

    if len(results) == 0:
        return ("", "")
    if len(results) == 1:
        return (results[0], results[0])
    return (results[0], results[1])


def _write_country_list(iso3_codes: list[str]) -> None:
    """Write the list of ISO3 codes with IPC API data."""
    try:
        codes = sorted(set(iso3_codes))
        _IPC_COUNTRIES_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(_IPC_COUNTRIES_JSON, "w", encoding="utf-8") as fh:
            json.dump(codes, fh, indent=2)
        LOG.info(
            "[ipc_api] wrote %d countries to %s",
            len(codes),
            _IPC_COUNTRIES_JSON,
        )
    except Exception as exc:
        LOG.warning("[ipc_api] failed to write country list: %s", exc)


# ---------------------------------------------------------------------------
# IpcApiConnector
# ---------------------------------------------------------------------------


class IpcApiConnector:
    """Fetch IPC Phase 3+ population data from api.ipcinfo.org and return
    a canonical DataFrame."""

    name: str = "ipc_api"

    def fetch_and_normalize(self) -> pd.DataFrame:
        """Fetch IPC population data and return canonical rows."""
        api_key = os.getenv("IPC_API_KEY", "").strip()
        if not api_key:
            LOG.warning("[ipc_api] IPC_API_KEY not set — skipping")
            return empty_canonical()

        months_back = int(os.getenv("IPC_API_MONTHS", "24"))
        delay = float(os.getenv("IPC_API_REQUEST_DELAY", "1.0"))

        # Compute start year — IPC API accepts year integers, not YYYY-MM
        today = date.today()
        start_year = today.year - (months_back // 12)
        if months_back % 12 >= today.month:
            start_year -= 1
        start_year = max(start_year, 2015)  # IPC API data starts at 2015
        end_year = today.year

        LOG.info(
            "[ipc_api] fetching IPC data for years %d–%d (months_back=%d)",
            start_year,
            end_year,
            months_back,
        )

        # Fetch from IPC API — key and year range as query parameters
        session = _build_session()
        params: dict[str, str | int] = {
            "key": api_key,
            "start": start_year,
            "end": end_year,
        }

        try:
            resp = session.get(_POPULATION_ENDPOINT, params=params, timeout=120)
            resp.raise_for_status()
        except requests.RequestException as exc:
            LOG.error("[ipc_api] fetch failed: %s", exc)
            return empty_canonical()

        if delay > 0:
            time.sleep(delay)

        # Parse JSON response
        try:
            payload = resp.json()
        except ValueError as exc:
            LOG.error("[ipc_api] JSON parse failed: %s", exc)
            return empty_canonical()

        if not payload:
            LOG.warning("[ipc_api] empty response from IPC API")
            return empty_canonical()

        # The response is a list of analysis records
        records = payload if isinstance(payload, list) else [payload]
        if not records:
            LOG.warning("[ipc_api] no records in IPC API response")
            return empty_canonical()

        # Log sample record for schema debugging
        if records and isinstance(records[0], dict):
            r0 = records[0]
            LOG.info(
                "[ipc_api] sample record keys: %s",
                sorted(r0.keys()),
            )
            LOG.info(
                "[ipc_api] sample dates: country=%s, current_period_dates=%r, "
                "projected_period_dates=%r, analysis_date=%r, p3plus=%r, "
                "p3plus_projected=%r",
                r0.get("country"),
                r0.get("current_period_dates"),
                r0.get("projected_period_dates"),
                r0.get("analysis_date"),
                r0.get("p3plus"),
                r0.get("p3plus_projected"),
            )

        LOG.info("[ipc_api] fetched %d raw records", len(records))

        # Load FEWS NET countries for exclusion
        fewsnet_countries = _load_fewsnet_country_list()

        # Parse records into flat rows.
        # Each IPC API record contains BOTH current and projected data:
        #   p3plus / current_period_dates   → phase3plus_in_need  (period "A")
        #   p3plus_projected / projected_period_dates → phase3plus_projection (period "P")
        parsed_rows: list[dict] = []
        _date_parse_failures = 0

        for rec in records:
            if not isinstance(rec, dict):
                continue

            # Extract country code (try multiple field names)
            country_code = (
                rec.get("country")
                or rec.get("country_code")
                or rec.get("iso2")
                or ""
            )
            if isinstance(country_code, dict):
                country_code = country_code.get("iso2", "") or country_code.get("code", "")
            country_code = str(country_code).strip()

            # Convert ISO2 → ISO3 (IPC API uses ISO2)
            if len(country_code) == 2:
                iso3 = _iso2_to_iso3(country_code)
            elif len(country_code) == 3:
                iso3 = country_code.upper()
            else:
                continue

            if not iso3:
                continue

            # Exclude FEWS NET countries
            if iso3 in fewsnet_countries:
                continue

            # Analysis date (common to both current and projected)
            analysis_date = str(
                rec.get("analysis_date", "")
                or rec.get("date", "")
                or rec.get("created_date", "")
                or ""
            ).strip()

            # --- Current Situation row (period "A") ---
            current_value = rec.get("p3plus")
            if current_value is None:
                # Fallback: sum phase3 + phase4 + phase5 populations
                p3 = rec.get("phase3_population") or rec.get("phase3") or 0
                p4 = rec.get("phase4_population") or rec.get("phase4") or 0
                p5 = rec.get("phase5_population") or rec.get("phase5") or 0
                try:
                    current_value = int(p3 or 0) + int(p4 or 0) + int(p5 or 0)
                except (TypeError, ValueError):
                    current_value = 0
            if current_value is not None:
                try:
                    cv = float(current_value)
                except (TypeError, ValueError):
                    cv = 0.0
                if cv > 0:
                    current_dates_raw = str(rec.get("current_period_dates", "") or "").strip()
                    c_start, c_end = _parse_period_dates(current_dates_raw)
                    if c_start:
                        parsed_rows.append({
                            "iso3": iso3,
                            "ipc_period": "A",
                            "value": cv,
                            "period_start": c_start,
                            "period_end": c_end or c_start,
                            "analysis_date": analysis_date,
                        })
                    else:
                        _date_parse_failures += 1

            # --- First Projection row (period "P") ---
            projected_value = rec.get("p3plus_projected")
            if projected_value is not None:
                try:
                    pv = float(projected_value)
                except (TypeError, ValueError):
                    pv = 0.0
                if pv > 0:
                    proj_dates_raw = str(rec.get("projected_period_dates", "") or "").strip()
                    p_start, p_end = _parse_period_dates(proj_dates_raw)
                    if p_start:
                        parsed_rows.append({
                            "iso3": iso3,
                            "ipc_period": "P",
                            "value": pv,
                            "period_start": p_start,
                            "period_end": p_end or p_start,
                            "analysis_date": analysis_date,
                        })
                    else:
                        _date_parse_failures += 1

        if _date_parse_failures > 0:
            LOG.warning(
                "[ipc_api] %d rows skipped due to unparseable period dates",
                _date_parse_failures,
            )

        if not parsed_rows:
            LOG.warning("[ipc_api] no valid rows after parsing")
            return empty_canonical()

        df_raw = pd.DataFrame(parsed_rows)

        LOG.info(
            "[ipc_api] parsed %d rows (%d countries, excluding %d FEWS NET countries)",
            len(df_raw),
            df_raw["iso3"].nunique(),
            len(fewsnet_countries),
        )

        # Write IPC country list
        all_iso3 = df_raw["iso3"].unique().tolist()
        _write_country_list(all_iso3)

        # Derive ym from period_start
        df_raw["period_start_dt"] = pd.to_datetime(
            df_raw["period_start"], errors="coerce"
        )
        df_raw["analysis_date_dt"] = pd.to_datetime(
            df_raw["analysis_date"], errors="coerce"
        )
        df_raw["period_end_dt"] = pd.to_datetime(
            df_raw["period_end"], errors="coerce"
        )
        df_raw["ym"] = df_raw["period_start_dt"].dt.strftime("%Y-%m")

        # Drop rows without a parseable period_start
        bad_ym = df_raw["ym"].isna().sum()
        if bad_ym > 0:
            LOG.warning("[ipc_api] dropping %d rows with unparseable period_start", bad_ym)
            df_raw = df_raw.dropna(subset=["ym"])

        if df_raw.empty:
            return empty_canonical()

        # Map metric from period code
        df_raw["metric"] = df_raw["ipc_period"].map(_PERIOD_METRIC)

        # Deduplicate: keep latest analysis_date per (iso3, ipc_period, ym)
        df_raw = df_raw.sort_values("analysis_date_dt", ascending=False, na_position="last")
        df_raw = df_raw.drop_duplicates(
            subset=["iso3", "ipc_period", "ym"], keep="first"
        )

        LOG.info(
            "[ipc_api] %d rows after dedup (%d countries)",
            len(df_raw),
            df_raw["iso3"].nunique(),
        )

        # Load country name lookup
        iso3_to_name = _load_iso3_to_name()
        now_utc = datetime.now(timezone.utc).isoformat()

        # Build canonical DataFrame
        rows = []
        for _, r in df_raw.iterrows():
            iso3 = str(r["iso3"])
            period = str(r["ipc_period"])
            period_label = _PERIOD_LABELS.get(period, period)

            as_of = (
                r["period_end_dt"].strftime("%Y-%m-%d")
                if pd.notna(r.get("period_end_dt"))
                else r["analysis_date_dt"].strftime("%Y-%m-%d")
                if pd.notna(r.get("analysis_date_dt"))
                else ""
            )
            pub_date = (
                r["analysis_date_dt"].strftime("%Y-%m-%d")
                if pd.notna(r.get("analysis_date_dt"))
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
                    "publisher": "IPC",
                    "source_type": "ipc_classification",
                    "source_url": "https://www.ipcinfo.org",
                    "doc_title": period_label,
                    "definition_text": (
                        f"IPC Phase 3+ (Crisis or worse) population estimate "
                        f"from IPC API — {period_label}"
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
        return validate_canonical(df, source="ipc_api")
