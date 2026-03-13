# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""HDX Signals connector — OCHA automated crisis monitoring.

Downloads the hdx_signals.csv from the HDX CKAN API and caches it locally.
At prompt build time, queries the cached CSV for signals relevant to a given
country and hazard code.

HDX Signals monitors seven humanitarian datasets and generates automated
alerts when statistically significant negative changes are detected in a
location.  Signals are only generated when significant negative changes
exceed thresholds, and a signal for a location + indicator is suppressed
for 6 months after the last signal at the same or higher alert level.

Public API
----------
- :func:`fetch_and_cache` — download CSV from HDX, save to local cache
- :func:`ensure_cache_fresh` — refresh cache if stale (default: 7 days)
- :func:`get_signals_for_country` — filter cached CSV for a country + hazard
- :func:`format_hdx_signals_for_prompt` — build prompt text block
- :func:`clear_cache` — clear in-memory parsed CSV
"""

from __future__ import annotations

import csv
import io
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = _PROJECT_ROOT / "data" / "hdx_signals"
CACHE_FILE = CACHE_DIR / "hdx_signals.csv"

# Direct download URL (stable, from CKAN resource metadata).
_HDX_SIGNALS_CSV_URL = (
    "https://data.humdata.org/dataset/"
    "464950af-0d57-47ae-a1fa-b4413b0adaf7/resource/"
    "49056751-af41-430e-92e8-81c4b0e5b38f/download/hdx_signals.csv"
)

# HDX CKAN API for dataset discovery (fallback if direct URL fails).
_HDX_CKAN_BASE = "https://data.humdata.org/api/3/action"

# ---------------------------------------------------------------------------
# Indicator ↔ hazard mappings
# ---------------------------------------------------------------------------

# HDX Signals indicator_id → set of Pythia hazard codes.
INDICATOR_TO_HAZARDS: dict[str, set[str]] = {
    "acled_conflict": {"ACE"},
    "idmc_displacement_conflict": {"ACE"},
    "idmc_displacement_disaster": {"FL", "TC", "DR", "HW"},
    "ipc_food_insecurity": {"DR"},
    "jrc_agricultural_hotspots": {"DR", "HW"},
    "wfp_market_monitor": {"DR"},
    "acaps_inform_severity": {"ACE", "FL", "TC", "DR", "HW"},  # cross-cutting
}

# Reverse mapping: Pythia hazard code → list of relevant indicator_ids.
HAZARD_TO_INDICATORS: dict[str, list[str]] = {}
for _ind, _hazards in INDICATOR_TO_HAZARDS.items():
    for _h in _hazards:
        HAZARD_TO_INDICATORS.setdefault(_h, []).append(_ind)

# Human-readable display names for prompt output.
INDICATOR_DISPLAY_NAMES: dict[str, str] = {
    "acled_conflict": "Conflict Events (ACLED)",
    "idmc_displacement_conflict": "Conflict-Driven Displacement (IDMC)",
    "idmc_displacement_disaster": "Disaster-Driven Displacement (IDMC)",
    "ipc_food_insecurity": "Food Insecurity (IPC)",
    "jrc_agricultural_hotspots": "Agricultural Production Hotspot (JRC)",
    "wfp_market_monitor": "Food Market Prices (WFP)",
    "acaps_inform_severity": "INFORM Severity Index (ACAPS)",
}

# ---------------------------------------------------------------------------
# In-memory cache for parsed CSV rows (populated on first access per run).
# ---------------------------------------------------------------------------

_SIGNALS_CACHE: list[dict[str, str]] | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_and_cache() -> Path | None:
    """Download hdx_signals.csv from HDX and cache locally.

    Returns the path to the cached file on success, None on failure.
    """
    import requests  # lazy import to avoid load-time cost

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Try direct URL first (fastest path).
    url = _HDX_SIGNALS_CSV_URL
    try:
        resp = requests.get(url, timeout=60, allow_redirects=True)
        resp.raise_for_status()
    except Exception as exc:
        log.warning("Direct HDX Signals download failed: %s — trying CKAN API", exc)
        url = _resolve_csv_url_via_ckan()
        if not url:
            return None
        try:
            resp = requests.get(url, timeout=60, allow_redirects=True)
            resp.raise_for_status()
        except Exception as exc2:
            log.error("HDX Signals CSV download failed: %s", exc2)
            return None

    log.info(
        "HDX Signals: download complete — status %d, content length %d bytes",
        resp.status_code,
        len(resp.content),
    )
    if len(resp.content) < 100:
        log.warning(
            "HDX Signals: downloaded file is suspiciously small (%d bytes)",
            len(resp.content),
        )

    CACHE_FILE.write_bytes(resp.content)
    log.info(
        "HDX Signals: cached %d KB to %s",
        len(resp.content) // 1024,
        CACHE_FILE,
    )

    # Invalidate in-memory cache so next access re-parses.
    global _SIGNALS_CACHE
    _SIGNALS_CACHE = None

    return CACHE_FILE


def ensure_cache_fresh(max_age_hours: int = 168) -> bool:
    """Refresh the cache if it is older than *max_age_hours* (default 7 days).

    Returns True if the cache is fresh after the operation.
    """
    log.info("HDX Signals: checking cache freshness...")
    if CACHE_FILE.exists():
        age = datetime.now() - datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
        if age.total_seconds() < max_age_hours * 3600:
            rows = _load_csv()
            log.info("HDX Signals: cache is fresh, %d rows", len(rows))
            return True

    result = fetch_and_cache()
    if result is not None:
        rows = _load_csv()
        log.info("HDX Signals: cache refreshed, %d rows", len(rows))
    return result is not None


def get_signals_for_country(
    iso3: str,
    hazard_code: str,
    max_age_days: int = 180,
) -> list[dict[str, str]]:
    """Return active signals for a country + hazard from the cached CSV.

    Parameters
    ----------
    iso3 : ISO3 country code.
    hazard_code : Pythia hazard code (ACE, FL, TC, DR, HW).
    max_age_days : Only return signals from the last N days (default 180).

    Returns
    -------
    List of signal dicts sorted by campaign_date descending, then alert_level
    (High concern before Medium concern).
    """
    rows = _load_csv()
    if not rows:
        return []

    relevant_indicators = set(HAZARD_TO_INDICATORS.get(hazard_code, []))
    if not relevant_indicators:
        return []

    cutoff = datetime.now() - timedelta(days=max_age_days)
    results: list[dict[str, str]] = []

    for row in rows:
        if row.get("iso3") != iso3:
            continue
        if row.get("indicator_id") not in relevant_indicators:
            continue

        # Parse campaign_date for recency filtering.
        raw_date = (row.get("campaign_date") or "").strip()
        if not raw_date:
            continue
        try:
            cd = datetime.strptime(raw_date[:10], "%Y-%m-%d")
        except ValueError:
            continue
        if cd < cutoff:
            continue

        results.append(row)

    # Sort: most recent first, then High concern before Medium concern.
    def _sort_key(r: dict[str, str]) -> tuple[str, int]:
        d = (r.get("campaign_date") or "")[:10]
        # "High concern" < "Medium concern" alphabetically, but we want
        # High first, so invert: High → 0, Medium → 1.
        level_order = 0 if "high" in (r.get("alert_level") or "").lower() else 1
        return (d, level_order)

    results.sort(key=_sort_key, reverse=True)
    # Fix: reverse=True flips both fields.  Re-sort so that within the same
    # date, High comes first (level_order 0 before 1 → ascending).
    results.sort(
        key=lambda r: (
            (r.get("campaign_date") or "")[:10],
            0 if "high" in (r.get("alert_level") or "").lower() else 1,
        ),
        reverse=False,
    )
    # Now reverse so most-recent date is first, and within same date High is
    # before Medium.
    results.sort(
        key=lambda r: (r.get("campaign_date") or "")[:10],
        reverse=True,
    )

    return results


def format_hdx_signals_for_prompt(
    iso3: str,
    hazard_code: str,
    max_age_days: int = 180,
    max_signals: int = 8,
) -> str:
    """Build the HDX Signals evidence section for a country + hazard.

    Returns an empty string if no relevant signals are found.
    """
    signals = get_signals_for_country(iso3, hazard_code, max_age_days=max_age_days)
    if not signals:
        log.debug(
            "HDX Signals: no signals for %s/%s (this is normal for most countries)",
            iso3,
            hazard_code,
        )
        return ""

    signals = signals[:max_signals]

    lines: list[str] = [
        "## HDX Signals (OCHA Automated Crisis Monitoring)\n",
    ]

    for s in signals:
        indicator_id = s.get("indicator_id", "")
        display_name = INDICATOR_DISPLAY_NAMES.get(indicator_id, indicator_id)
        alert_level = (s.get("alert_level") or "").strip()
        date_str = (s.get("date") or s.get("campaign_date") or "")[:10]
        value = (s.get("value") or "").strip()

        lines.append(f"### [{alert_level}] {display_name} — {date_str}")

        # Disaster displacement disclaimer.
        if indicator_id == "idmc_displacement_disaster":
            lines.append(
                "Note: This displacement signal covers all disaster types "
                f"and is not specific to {hazard_code}."
            )

        # Prefer summary_long → summary_short → further_information.
        summary = _pick_summary(s)
        if summary:
            lines.append(summary)

        if value:
            lines.append(f"Indicator value: {value}")

        source_url = (s.get("source_url") or "").strip()
        if source_url:
            lines.append(f"Source: {source_url}")

        lines.append("")  # blank line between signals

    lines.append(
        "NOTE: HDX Signals are generated by OCHA's automated monitoring "
        "system when statistically significant negative changes are detected "
        "in key humanitarian datasets. A signal indicates a notable "
        "deterioration, not a prediction. Signals are suppressed for 6 months "
        "after the last alert for the same location-indicator pair, so "
        "absence of a signal does not mean absence of concern."
    )

    return "\n".join(lines)


def clear_cache() -> None:
    """Clear the in-memory parsed CSV cache (for testing or new HS run)."""
    global _SIGNALS_CACHE
    _SIGNALS_CACHE = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_csv() -> list[dict[str, str]]:
    """Load and cache the parsed CSV rows in memory."""
    global _SIGNALS_CACHE
    if _SIGNALS_CACHE is not None:
        return _SIGNALS_CACHE

    if not CACHE_FILE.exists():
        log.debug("HDX Signals cache file not found: %s", CACHE_FILE)
        return []

    try:
        text = CACHE_FILE.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(text))
        _SIGNALS_CACHE = list(reader)
        unique_countries = len({r.get("iso3") for r in _SIGNALS_CACHE if r.get("iso3")})
        log.info(
            "HDX Signals: loaded %d rows from cache, %d unique countries",
            len(_SIGNALS_CACHE),
            unique_countries,
        )
        return _SIGNALS_CACHE
    except Exception as exc:
        log.warning("HDX Signals CSV parse failed: %s", exc)
        return []


def _pick_summary(row: dict[str, str]) -> str:
    """Return the best available summary text from a signal row."""
    for key in ("summary_long", "summary_short", "further_information"):
        val = (row.get(key) or "").strip()
        if val:
            return val
    return ""


def _resolve_csv_url_via_ckan() -> str | None:
    """Fallback: discover the hdx_signals.csv URL via CKAN package_show."""
    import requests

    try:
        resp = requests.get(
            f"{_HDX_CKAN_BASE}/package_show",
            params={"id": "hdx-signals"},
            timeout=30,
        )
        resp.raise_for_status()
        dataset = resp.json().get("result", {})
        for resource in dataset.get("resources", []):
            name = (resource.get("name") or "").lower()
            if "hdx_signals" in name and "metadata" not in name and "dictionary" not in name:
                return resource.get("url")
    except Exception as exc:
        log.error("CKAN package_show fallback failed: %s", exc)

    return None
