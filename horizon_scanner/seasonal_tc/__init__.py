# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Seasonal tropical cyclone forecast pipeline.

Provides cached seasonal TC outlooks from TSR, NOAA CPC, and BoM for
injection into TC prompts.  The heavy scraping is done offline via
:mod:`seasonal_tc_runner`; this module exposes lightweight readers that
load the cached JSON and return prompt-ready text.

Usage (from the Pythia pipeline)::

    from horizon_scanner.seasonal_tc import get_seasonal_tc_context_for_country

    ctx = get_seasonal_tc_context_for_country("PHL")  # -> NWP forecasts
    ctx = get_seasonal_tc_context_for_country("HTI")  # -> ATL forecasts
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output path
# ---------------------------------------------------------------------------

_OUTPUT_DIR = Path(__file__).parent / "output"
_JSON_PATH = _OUTPUT_DIR / "seasonal_tc_forecasts.json"

# Max age in seconds before we warn (120 days).
_MAX_AGE_SECONDS = 120 * 24 * 60 * 60

# ---------------------------------------------------------------------------
# Country-to-basin mapping
# ---------------------------------------------------------------------------
# Maps ISO3 codes to the TC basin(s) that affect each country.
# Countries may appear in multiple basins (e.g., Central American countries
# are affected by both ATL and ENP).

_BASIN_TO_COUNTRIES: dict[str, list[str]] = {
    # North Atlantic
    "ATL": [
        # Caribbean
        "ATG", "BHS", "BRB", "BLZ", "CUB", "DMA", "DOM", "GRD", "HTI",
        "JAM", "KNA", "LCA", "VCT", "TTO", "ABW", "CUW", "SXM",
        # Central America (Atlantic coast)
        "GTM", "HND", "NIC", "CRI", "PAN",
        # US Gulf/East Coast
        "USA",
        # Mexico (Gulf/Caribbean coast)
        "MEX",
        # Bermuda
        "BMU",
        # West Africa (Cabo Verde region)
        "CPV",
        # Other Atlantic-exposed
        "VEN", "COL", "GUY", "SUR",
    ],
    # Eastern North Pacific
    "ENP": [
        "MEX",  # Pacific coast
        "GTM", "SLV", "HND", "NIC", "CRI", "PAN",  # Central America Pacific
    ],
    # Central Pacific
    "CP": [
        "USA",  # Hawaii
    ],
    # Northwest Pacific
    "NWP": [
        "PHL", "JPN", "CHN", "VNM", "TWN", "KOR", "PRK",
        "MHL", "FSM", "PLW", "GUM",  # Micronesia/Guam
        "HKG", "MAC", "MMR", "THA", "LAO", "KHM",
    ],
    # Australian Region
    "AUS": [
        "AUS",
        "IDN",  # south of equator
        "TLS",  # Timor-Leste
    ],
    # South Pacific
    "SP": [
        "FJI", "VUT", "TON", "WSM", "ASM",
        "NCL", "SLB", "PNG",
        "COK", "NIU", "TUV", "WLF",
        "NZL",
    ],
    # South-West Indian Ocean
    "SWI": [
        "MDG", "MOZ", "MUS", "REU", "COM", "MYT",
        "TZA", "KEN",
        "ZWE", "MWI",  # inland but affected by remnant TC moisture
    ],
    # North Indian Ocean
    "NIO": [
        "IND", "BGD", "MMR", "LKA",
        "OMN", "YEM", "PAK",
        "MDV", "SOM",
    ],
}

# Invert to ISO3 -> list of basin codes.
COUNTRY_TO_BASINS: dict[str, list[str]] = {}
for _basin, _countries in _BASIN_TO_COUNTRIES.items():
    for _iso3 in _countries:
        COUNTRY_TO_BASINS.setdefault(_iso3, []).append(_basin)


# ---------------------------------------------------------------------------
# Cache reader
# ---------------------------------------------------------------------------

def _load_forecasts() -> list[dict]:
    """Load the cached forecasts JSON, returning an empty list on failure."""
    if not _JSON_PATH.exists():
        log.warning(
            "Seasonal TC forecasts file not found at %s. "
            "Run `python -m horizon_scanner.seasonal_tc.seasonal_tc_runner` to generate it.",
            _JSON_PATH,
        )
        return []

    # Check staleness.
    age = time.time() - _JSON_PATH.stat().st_mtime
    if age > _MAX_AGE_SECONDS:
        log.warning(
            "Seasonal TC forecasts are %.0f days old (file: %s). "
            "Consider re-running the pipeline.",
            age / 86400,
            _JSON_PATH,
        )

    try:
        return json.loads(_JSON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Failed to read seasonal TC forecasts: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_seasonal_tc_context(basin_code: str) -> str:
    """Return prompt context text for the given TC basin.

    Reads from the cached ``seasonal_tc_forecasts.json`` and returns the
    concatenated ``prompt_context`` blocks for all forecasts matching
    *basin_code*.  Returns an empty string if no data is available.
    """
    forecasts = _load_forecasts()
    if not forecasts:
        return ""

    blocks = []
    for f in forecasts:
        if f.get("basin") == basin_code and f.get("prompt_context"):
            blocks.append(f["prompt_context"])

    return "\n\n".join(blocks)


def get_seasonal_tc_context_for_country(iso3: str) -> str:
    """Return prompt context text for the TC basin(s) relevant to a country.

    Tries the DB cache first (``seasonal_tc_context_cache``).  If the DB
    has fresh data, returns it immediately.  Otherwise falls back to the
    on-disk cached JSON file.
    """
    iso3 = iso3.upper()

    # --- DB-first path ---
    db_text = load_seasonal_tc_context_from_db(iso3)
    if db_text:
        return db_text

    # --- Fallback: cached JSON file ---
    basins = COUNTRY_TO_BASINS.get(iso3, [])
    if not basins:
        return ""

    forecasts = _load_forecasts()
    if not forecasts:
        return ""

    blocks = []
    seen = set()
    for f in forecasts:
        basin = f.get("basin", "")
        ctx = f.get("prompt_context", "")
        if basin in basins and ctx:
            # Deduplicate identical blocks (same forecast from same source).
            key = (f.get("source", ""), basin, f.get("forecast_type", ""))
            if key not in seen:
                seen.add(key)
                blocks.append(ctx)

    return "\n\n".join(blocks)


def format_seasonal_tc_for_spd(context: str) -> Optional[str]:
    """Format seasonal TC context for SPD prompt injection.

    Returns a labeled section string, or *None* if *context* is empty.
    """
    if not context:
        return None
    return (
        "SEASONAL TC FORECASTS (pre-scraped from TSR, NOAA CPC, BoM):\n"
        + context
    )


# ---------------------------------------------------------------------------
# DB store / load helpers
# ---------------------------------------------------------------------------

def store_seasonal_tc_context_cache(iso3: str, text: str) -> bool:
    """Write a country's seasonal TC context to the DB cache.

    Uses INSERT OR REPLACE so repeated calls for the same iso3 overwrite
    the previous row.  Returns True on success, False on failure.
    Non-fatal on DB errors.
    """
    try:
        from pythia.db.schema import connect, ensure_schema  # lazy import

        con = connect()
        ensure_schema(con)
        con.execute(
            """
            INSERT OR REPLACE INTO seasonal_tc_context_cache
                (iso3, context_text, fetched_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            [iso3.upper(), text],
        )
        return True
    except Exception as exc:
        log.warning("store_seasonal_tc_context_cache(%s): %s", iso3, exc)
        return False


def load_seasonal_tc_context_from_db(
    iso3: str, max_age_days: int = 90
) -> Optional[str]:
    """Read cached seasonal TC context for a country from the DB.

    Returns *None* if no row exists or if the cached row is older than
    *max_age_days*.  Non-fatal on DB errors.
    """
    try:
        from pythia.db.schema import connect, ensure_schema  # lazy import

        con = connect()
        ensure_schema(con)
        rows = con.execute(
            """
            SELECT context_text, fetched_at
            FROM seasonal_tc_context_cache
            WHERE iso3 = ?
            """,
            [iso3.upper()],
        ).fetchall()
        if not rows:
            return None
        context_text, fetched_at = rows[0]
        # Check staleness
        if fetched_at is not None:
            from datetime import datetime, timezone

            age_days = (
                datetime.now(timezone.utc) - fetched_at.replace(tzinfo=timezone.utc)
            ).total_seconds() / 86400
            if age_days > max_age_days:
                log.debug(
                    "seasonal_tc_context_cache for %s is %.0f days old (max %d)",
                    iso3, age_days, max_age_days,
                )
                return None
        return context_text or None
    except Exception as exc:
        log.warning("load_seasonal_tc_context_from_db(%s): %s", iso3, exc)
        return None


def store_seasonal_tc_outlooks(outlooks: list[dict]) -> int:
    """Persist raw seasonal TC outlook dicts to the DB.

    Takes the list of forecast dicts produced by
    :func:`seasonal_tc_runner.collect_all` (after deduplication).
    Returns the number of rows successfully stored.
    Non-fatal on DB errors.
    """
    if not outlooks:
        return 0
    try:
        import json as _json
        from pythia.db.schema import connect, ensure_schema  # lazy import

        con = connect()
        ensure_schema(con)
        stored = 0
        for f in outlooks:
            try:
                con.execute(
                    """
                    INSERT OR REPLACE INTO seasonal_tc_outlooks
                        (basin, source, forecast_season,
                         named_storms_forecast, category, raw_json, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    [
                        f.get("basin", ""),
                        f.get("source", ""),
                        f.get("forecast_season", f.get("season", "")),
                        f.get("named_storms_forecast", f.get("named_storms", "")),
                        f.get("category", f.get("forecast_type", "")),
                        _json.dumps(f, default=str),
                    ],
                )
                stored += 1
            except Exception as row_exc:
                log.warning("store_seasonal_tc_outlooks row error: %s", row_exc)
        log.info("store_seasonal_tc_outlooks: stored %d / %d outlooks", stored, len(outlooks))
        return stored
    except Exception as exc:
        log.warning("store_seasonal_tc_outlooks: %s", exc)
        return 0
