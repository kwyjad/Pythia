# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""GDACS connector — fetches disaster events via the GDACS JSON search API
and static RSS feeds.

Data sources (the ``rss.aspx?profile=ARCHIVE`` endpoint returns only a
generic info item — it does NOT return event data):

1. **JSON search API** (``gdacsapi/api/events/geteventlist/SEARCH``) —
   works for any date range, returns event metadata including
   ``affectedcountries`` with ISO3 codes, ``alertlevel``, dates.

2. **Static RSS feeds** (``xml/rss_fl_3m.xml``, ``xml/rss_tc_3m.xml``) —
   cover the last 3 months and include ``gdacs:population`` data.
   The drought feed (``rss_dr_3m.xml``) returns 404.

3. **Per-event RSS** (``datareport/resources/{TYPE}/{ID}/rss_{ID}.xml``) —
   available for any event, includes ``gdacs:population`` data.

Strategy:
- For the default 3-month window: fetch static RSS feeds for FL/TC (fast,
  1 request per hazard type, includes population data).  DR always uses
  the JSON search API because the DR RSS feed returns 404.
- For historical backfill (>3 months): use JSON search API for event
  discovery, then fetch per-event RSS for population data.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from calendar import monthrange
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

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

# JSON search API — works for any date range
_SEARCH_API = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"

# Static RSS feeds — 3-month window, include population data
_STATIC_RSS: dict[str, str] = {
    "FL": "https://www.gdacs.org/xml/rss_fl_3m.xml",
    "TC": "https://www.gdacs.org/xml/rss_tc_3m.xml",
    # DR feed (rss_dr_3m.xml) returns 404 as of 2026-03
}

# Per-event RSS pattern — available for any event, has population data
_EVENT_RSS_PATTERN = (
    "https://www.gdacs.org/datareport/resources/{type}/{eventid}/rss_{eventid}.xml"
)

# Per-event JSON, the API's own route. The datareport tree above is a
# published-report artefact and answers 403 for an event GDACS never wrote a
# report for, which is a statement about the REPORT and not about the event —
# the API still describes it. Asked once, only after a refusal.
_EVENT_DATA_PATTERN = (
    "https://www.gdacs.org/gdacsapi/api/events/geteventdata"
    "?eventtype={type}&eventid={eventid}"
)

#: Population-shaped fields in a ``geteventdata`` impacts entry, lowercased.
#: GDACS names its measures as the field: ``pop39`` and ``pop74`` are the
#: populations inside the 39 kt and 74 kt wind envelopes of a cyclone, and
#: ``popaffected`` is the general affected count. They are handed to
#: :func:`parse_gdacs_population` as the UNIT for that reason — the same
#: rule that reads ``Pop74`` off the RSS reads them here.
_EVENT_DATA_POPULATION_FIELDS: frozenset[str] = frozenset(
    {"pop39", "pop74", "popaffected"}
)

# GDACS event types we care about
_WANTED_TYPES = {"DR", "FL", "TC"}

# Map GDACS eventtype -> Pythia hazard_code (verified against shocks.csv)
_HAZARD_MAP: dict[str, str] = {
    "DR": "DR",
    "FL": "FL",
    "TC": "TC",
}

# Hazard metadata from shocks.csv
_HAZARD_LABEL: dict[str, str] = {
    "DR": "Drought",
    "FL": "Flood",
    "TC": "Tropical Cyclone",
}
_HAZARD_CLASS: dict[str, str] = {
    "DR": "natural",
    "FL": "natural",
    "TC": "natural",
}

# Alert level -> confidence mapping
_CONFIDENCE_MAP: dict[str, str] = {
    "Red": "high",
    "Orange": "medium",
    "Green": "low",
}

# XML namespaces used in the GDACS RSS feeds
_NS: dict[str, str] = {
    "gdacs": "http://www.gdacs.org",
    "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
    # GeoRSS is the second place a GDACS item states its position, as a
    # single "lat lon" string. Without it registered, an item carrying only
    # georss:point parsed to no coordinates at all.
    "georss": "http://www.georss.org/georss",
}

# Countries CSV path
_COUNTRIES_CSV = Path(__file__).resolve().parent.parent / "data" / "countries.csv"

# Approximate 2024 population estimates (World Bank / UN) for population-
# weighted allocation of multi-country events.  Keyed by ISO3.
# We only need countries that are plausibly affected by DR/FL/TC events.
# Values are approximate and will be refined later.
_POPULATION: dict[str, int] = {
    "AFG": 42_200_000, "ALB": 2_800_000, "DZA": 45_600_000,
    "AGO": 36_700_000, "ARG": 46_700_000, "ARM": 2_800_000,
    "AUS": 26_400_000, "AUT": 9_100_000, "AZE": 10_200_000,
    "BHS": 410_000, "BHR": 1_500_000, "BGD": 173_000_000,
    "BRB": 282_000, "BLR": 9_200_000, "BEL": 11_700_000,
    "BLZ": 410_000, "BEN": 13_700_000, "BTN": 790_000,
    "BOL": 12_400_000, "BIH": 3_200_000, "BWA": 2_600_000,
    "BRA": 216_400_000, "BRN": 450_000, "BGR": 6_500_000,
    "BFA": 23_300_000, "BDI": 13_200_000, "KHM": 17_400_000,
    "CMR": 28_600_000, "CAN": 40_100_000, "CPV": 600_000,
    "CAF": 5_600_000, "TCD": 18_300_000, "CHL": 19_800_000,
    "CHN": 1_425_700_000, "COL": 52_100_000, "COM": 850_000,
    "COG": 6_100_000, "COD": 102_300_000, "CRI": 5_200_000,
    "CIV": 28_900_000, "HRV": 3_900_000, "CUB": 11_200_000,
    "CYP": 1_300_000, "CZE": 10_900_000, "DNK": 5_900_000,
    "DJI": 1_100_000, "DMA": 73_000, "DOM": 11_300_000,
    "ECU": 18_200_000, "EGY": 112_700_000, "SLV": 6_400_000,
    "GNQ": 1_700_000, "ERI": 3_700_000, "EST": 1_400_000,
    "SWZ": 1_200_000, "ETH": 126_500_000, "FJI": 930_000,
    "FIN": 5_600_000, "FRA": 68_200_000, "GAB": 2_400_000,
    "GMB": 2_700_000, "GEO": 3_700_000, "DEU": 84_500_000,
    "GHA": 34_100_000, "GRC": 10_300_000, "GRD": 126_000,
    "GTM": 18_100_000, "GIN": 14_200_000, "GNB": 2_100_000,
    "GUY": 810_000, "HTI": 11_700_000, "HND": 10_400_000,
    "HUN": 9_600_000, "ISL": 380_000, "IND": 1_441_700_000,
    "IDN": 277_500_000, "IRN": 89_200_000, "IRQ": 44_500_000,
    "IRL": 5_200_000, "ISR": 9_800_000, "ITA": 58_900_000,
    "JAM": 2_800_000, "JPN": 123_300_000, "JOR": 11_300_000,
    "KAZ": 19_800_000, "KEN": 55_100_000, "KIR": 132_000,
    "KWT": 4_300_000, "KGZ": 7_000_000, "LAO": 7_600_000,
    "LVA": 1_800_000, "LBN": 5_500_000, "LSO": 2_300_000,
    "LBR": 5_400_000, "LBY": 6_900_000, "LTU": 2_800_000,
    "LUX": 670_000, "MDG": 30_300_000, "MWI": 20_900_000,
    "MYS": 34_300_000, "MDV": 520_000, "MLI": 23_300_000,
    "MLT": 540_000, "MHL": 42_000, "MRT": 4_900_000,
    "MUS": 1_300_000, "MEX": 130_900_000, "FSM": 115_000,
    "MDA": 2_600_000, "MNG": 3_400_000, "MNE": 620_000,
    "MAR": 37_800_000, "MOZ": 33_900_000, "MMR": 54_600_000,
    "NAM": 2_600_000, "NRU": 13_000, "NPL": 30_900_000,
    "NLD": 17_700_000, "NZL": 5_200_000, "NIC": 7_000_000,
    "NER": 27_200_000, "NGA": 223_800_000, "PRK": 26_200_000,
    "MKD": 1_800_000, "NOR": 5_500_000, "OMN": 4_700_000,
    "PAK": 240_500_000, "PLW": 18_000, "PSE": 5_400_000,
    "PAN": 4_400_000, "PNG": 10_400_000, "PRY": 6_900_000,
    "PER": 34_400_000, "PHL": 117_300_000, "POL": 36_800_000,
    "PRT": 10_400_000, "QAT": 2_700_000, "ROU": 19_100_000,
    "RUS": 144_200_000, "RWA": 14_100_000, "KNA": 48_000,
    "LCA": 180_000, "VCT": 103_000, "WSM": 225_000,
    "STP": 230_000, "SAU": 36_900_000, "SEN": 17_700_000,
    "SRB": 6_600_000, "SYC": 108_000, "SLE": 8_600_000,
    "SGP": 6_000_000, "SVK": 5_400_000, "SVN": 2_100_000,
    "SLB": 740_000, "SOM": 18_100_000, "ZAF": 60_400_000,
    "KOR": 51_700_000, "SSD": 11_100_000, "ESP": 48_000_000,
    "LKA": 22_200_000, "SDN": 48_100_000, "SUR": 620_000,
    "SWE": 10_600_000, "CHE": 8_800_000, "SYR": 22_900_000,
    "TWN": 23_900_000, "TJK": 10_100_000, "TZA": 65_500_000,
    "THA": 72_000_000, "TLS": 1_400_000, "TGO": 9_100_000,
    "TON": 107_000, "TTO": 1_500_000, "TUN": 12_500_000,
    "TUR": 85_800_000, "TKM": 6_500_000, "TUV": 11_000,
    "UGA": 48_600_000, "UKR": 37_000_000, "ARE": 10_000_000,
    "GBR": 67_700_000, "USA": 339_900_000, "URY": 3_400_000,
    "UZB": 35_600_000, "VUT": 330_000, "VEN": 28_400_000,
    "VNM": 99_500_000, "YEM": 34_400_000, "ZMB": 20_600_000,
    "ZWE": 16_700_000,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_countries() -> tuple[dict[str, str], dict[str, str]]:
    """Load countries.csv and return (name_to_iso3, iso3_to_name) mappings.

    Names are lowercased for case-insensitive lookup.
    """
    name_to_iso3: dict[str, str] = {}
    iso3_to_name: dict[str, str] = {}
    try:
        df = pd.read_csv(_COUNTRIES_CSV)
        for _, row in df.iterrows():
            name = str(row["country_name"]).strip()
            iso3 = str(row["iso3"]).strip().upper()
            name_to_iso3[name.lower()] = iso3
            iso3_to_name[iso3] = name
    except Exception as exc:
        LOG.warning("[gdacs] failed to load countries.csv: %s", exc)
    return name_to_iso3, iso3_to_name


#: Statuses the per-event RSS fetch retries. 403 is here because GDACS
#: answers a rate-limited caller with one, and urllib3's retry list does not
#: carry it — so 532 of 1,050 refusals in run 33946954189 were never retried.
#: Statuses worth asking again about. **403 is deliberately absent.** It was
#: added on the reading that GDACS answers a throttled caller 403 and that
#: the ladder would ride it out. Two runs disproved that. Run 33946954189
#: refused 532 of 1,050 requests; run 34124705852 cut volume by 73% and the
#: refusal rate barely moved, 79.6% to 76% — a rate limit eases when the
#: rate falls and this did not. And of 291 distinct events, 153 were served
#: on the FIRST request and 138 were refused on all four attempts: whatever
#: decides a refusal decides it per event and does not change its mind.
#: Retrying a 403 four times spends four requests to learn what one already
#: said, and the answer to a refusing source is fewer requests and a cache,
#: not a harder retry. 429 stays — that is the code a source uses when it
#: means "slow down", and it is worth obeying.
_RETRYABLE_ENRICH_STATUS = frozenset({429, 500, 502, 503, 504})

#: Refused, recorded, and asked once. The event keeps whatever the
#: cache already holds for it.
_REFUSAL_STATUS = frozenset({401, 403, 451})

#: One worker, paced. Six with no delay drew the refusals; two with a
#: quarter of a second did not fix them, because volume was never the whole
#: story. The pace is now held by a process-wide bucket rather than by the
#: per-worker sleep, so it survives an operator raising the worker count.
_DEFAULT_ENRICH_WORKERS = 1
_DEFAULT_ENRICH_DELAY = 0.25
_DEFAULT_ENRICH_ATTEMPTS = 4
#: Minimum seconds between ANY two per-event requests from this process.
_DEFAULT_ENRICH_MIN_INTERVAL = 2.0
#: A wall-clock ceiling on one enrichment pass, or 0 for none. A pace slow
#: enough to be polite is slow enough to outrun a step budget: 3,272 events
#: at one request every two seconds is 109 minutes against a 60-minute
#: reset step. Rather than choose between a rude pace and a killed step,
#: the pass stops asking and says how many it left — the events it did not
#: reach keep whatever the cache holds, and the next run asks for those.
_DEFAULT_ENRICH_MAX_SECONDS = 0.0


class _RateLimiter:
    """A process-wide floor on the interval between requests.

    The per-worker sleep paces one worker and multiplies by however many
    there are; this does not. It is deliberately the crudest possible
    limiter — one lock, one timestamp — because the thing it must never do
    is fail open under contention.
    """

    def __init__(self, min_interval: float) -> None:
        self._min_interval = max(0.0, float(min_interval))
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            self._next_allowed = max(now, self._next_allowed) + self._min_interval
        if wait > 0:
            time.sleep(wait)


def _enrich_max_seconds() -> float:
    try:
        raw = os.getenv("GDACS_ENRICH_MAX_SECONDS", "")
        return max(0.0, float(raw)) if raw else _DEFAULT_ENRICH_MAX_SECONDS
    except ValueError:
        return _DEFAULT_ENRICH_MAX_SECONDS


def _enrich_min_interval() -> float:
    try:
        raw = os.getenv("GDACS_ENRICH_MIN_INTERVAL_SEC", "")
        return max(0.0, float(raw)) if raw else _DEFAULT_ENRICH_MIN_INTERVAL
    except ValueError:
        return _DEFAULT_ENRICH_MIN_INTERVAL


#: Rebuilt per fetch by :func:`reset_exposure_memo`, so a test that changes
#: the interval is not held to the first test's pace.
_ENRICH_LIMITER = _RateLimiter(_enrich_min_interval())


def _enrich_attempts() -> int:
    try:
        return max(1, int(os.getenv("GDACS_ENRICH_ATTEMPTS", "") or _DEFAULT_ENRICH_ATTEMPTS))
    except ValueError:
        return _DEFAULT_ENRICH_ATTEMPTS


def _enrich_delay(caller_delay: float | None) -> float:
    """Seconds a worker waits between its own per-event requests.

    ``GDACS_ENRICH_DELAY`` wins where it is set, so an operator can slow the
    enrichment without touching the discovery delay the caller passes — the
    two used to be one number, so tuning discovery moved the enrichment rate
    with it. Otherwise the caller's own value stands, zero included: a test
    that asks for no delay must get none.
    """

    raw = os.getenv("GDACS_ENRICH_DELAY", "")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return max(0.0, float(caller_delay or 0.0))


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with jitter, capped.

    The jitter is the point: a pool of workers that all sleep the same
    interval and wake together reproduces the burst that drew the refusal.
    """

    base = min(_DEFAULT_ENRICH_DELAY * (2 ** attempt), 30.0)
    return base * (0.5 + random.random())


#: What this connector calls itself to GDACS. An outbound request asks for
#: what it wants: until Sept 2026 this session sent no User-Agent and no
#: Accept at all, so every call went out as ``python-requests/2.x``, which is
#: the shape a bot filter is freest to refuse. This repo has been refused for
#: exactly that twice already — bom.gov.au 403s a generic agent, and noaa.gov
#: refused ``PythiaBot/1.0``.
_USER_AGENT = os.getenv("GDACS_USER_AGENT", "").strip() or (
    "Mozilla/5.0 (compatible; Pythia/1.0; +https://fredforecaster.org)"
)


#: One event's exposure, remembered for the life of THIS PROCESS.
#:
#: Keyed ``(eventtype, eventid)`` -> ``(best RSS episode | None, refusal
#: status | None)``. Deliberately NOT a cache of upstream state: it exists
#: because the PA machine walks the same events once per hazard-month pass
#: and an event's exposure cannot change between two passes of one run. It
#: is process-scoped, so the next run asks GDACS again, and
#: ``reset_exposure_memo()`` clears it — a module-level store that outlives
#: a test would serve the first test's events to every later one, which is
#: the trap the vendored-boundary loader documents.
_EXPOSURE_MEMO: dict[tuple[str, str], tuple[dict[str, Any] | None, int | None]] = {}
_EXPOSURE_MEMO_LOCK = threading.Lock()


def reset_exposure_memo() -> None:
    """Forget every remembered per-event exposure. For tests and long runs."""

    global _ENRICH_LIMITER
    with _EXPOSURE_MEMO_LOCK:
        _EXPOSURE_MEMO.clear()
    _ENRICH_LIMITER = _RateLimiter(_enrich_min_interval())


#: Property names a GDACS search feature might state an exposure under.
#: Checked in order; the first that parses to a positive number wins. This
#: is a probe, not a promise — if none of them is ever present the run says
#: so and the per-event fetch carries on as before.
_BULK_POPULATION_KEYS = (
    "population",
    "populationexposed",
    "exposedpopulation",
    "affectedpopulation",
    "poptotal",
)

#: Property keys seen on search features this run, so the next run's log
#: answers "is there a bulk route" from evidence. Reset per fetch.
_SEARCH_PROPERTY_KEYS: set[str] = set()
_SEARCH_PROPERTY_LOCK = threading.Lock()


def reset_search_property_keys() -> None:
    with _SEARCH_PROPERTY_LOCK:
        _SEARCH_PROPERTY_KEYS.clear()


def observed_search_property_keys() -> list[str]:
    with _SEARCH_PROPERTY_LOCK:
        return sorted(_SEARCH_PROPERTY_KEYS)


def _note_search_properties(props: dict[str, Any]) -> None:
    if not isinstance(props, dict):
        return
    with _SEARCH_PROPERTY_LOCK:
        _SEARCH_PROPERTY_KEYS.update(str(k) for k in props)


def _population_from_properties(props: dict[str, Any]) -> tuple[float, str]:
    """``(exposure, field name)`` from a search feature, or ``(0.0, "")``.

    GDACS nests some numbers under ``severitydata``, so that is searched
    too. A non-positive figure is GDACS declining to say and is treated as
    absent, exactly as it is everywhere else in this connector.
    """

    if not isinstance(props, dict):
        return 0.0, ""
    candidates: list[tuple[str, Any]] = []
    for key in _BULK_POPULATION_KEYS:
        if key in props:
            candidates.append((key, props[key]))
    severity = props.get("severitydata")
    if isinstance(severity, dict):
        for key in _BULK_POPULATION_KEYS:
            if key in severity:
                candidates.append((f"severitydata.{key}", severity[key]))
    for name, raw in candidates:
        # A JSON field carries no separate unit attribute, so the RSS unit
        # parser has nothing to work with here. A value that is not plainly
        # a number is left alone rather than guessed at: an unrecognised
        # figure is UNKNOWN everywhere else in this connector and must be
        # here too, or the bulk route becomes a way to smuggle one in.
        try:
            value = float(str(raw).replace(",", "").strip())
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value, f"search:{name}"
    return 0.0, ""


def _build_session() -> requests.Session:
    """Build a requests session with retry logic and honest headers."""
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": _USER_AGENT,
            "Accept": "application/rss+xml, application/xml, text/xml, application/json;q=0.9, */*;q=0.8",
            "Accept-Language": "en",
        }
    )
    return session


def _month_range(start: date, end: date):
    """Yield (year, month) tuples from start to end inclusive."""
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


def _last_day(year: int, month: int) -> date:
    """Return the last day of the given month."""
    return date(year, month, monthrange(year, month)[1])


def _overlapping_months(from_date: date, to_date: date) -> list[tuple[int, int]]:
    """Return list of (year, month) tuples that the event overlaps."""
    months = []
    for y, m in _month_range(from_date, to_date):
        months.append((y, m))
    return months


def _parse_date(text: str | None) -> date | None:
    """Parse a date string from GDACS XML or JSON."""
    if not text:
        return None
    text = text.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d",
                "%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except Exception:
        return None


def _text(element: ET.Element | None, tag: str, ns: dict[str, str] = _NS) -> str | None:
    """Extract text from a child element, or None."""
    if element is None:
        return None
    child = element.find(tag, ns)
    if child is not None and child.text:
        return child.text.strip()
    return None


def _attr(element: ET.Element | None, tag: str, attr: str,
          ns: dict[str, str] = _NS) -> str | None:
    """Extract an attribute from a child element."""
    if element is None:
        return None
    child = element.find(tag, ns)
    if child is not None:
        return child.get(attr)
    return None


#: ``gdacs:population`` unit strings that mean the value is already a count of
#: people. GDACS labels the MEASURE, not a multiplier: the public fixtures
#: read ``unit="Pop74"`` (people under Category 1 winds or higher, TC),
#: ``unit="Population in 100km"`` (EQ), ``unit=""`` (DR, value 0), and the
#: flood feed ``unit="people"``. Compared case-insensitively after trimming.
_POPULATION_PEOPLE_UNITS = frozenset({
    "", "people", "persons", "population", "pop74", "pop_total", "population in 100km",
})
#: Multiplicative unit words, in case a feed ever states exposure in
#: thousands or millions: the bare number would then be 1,000x or
#: 1,000,000x too small, which is exactly the shape of a ceiling of 5.
_POPULATION_MULTIPLIERS: dict[str, float] = {
    "thousand": 1_000.0, "thousands": 1_000.0, "k": 1_000.0,
    "million": 1_000_000.0, "millions": 1_000_000.0, "m": 1_000_000.0,
    "mln": 1_000_000.0,
}


def parse_gdacs_population(
    value: str | None, unit: str | None, text: str | None = None
) -> tuple[float | None, dict[str, Any]]:
    """The exposed population as a count of PEOPLE, or None when it cannot be read.

    Returns ``(people, detail)``. ``detail`` carries the raw ``value``,
    ``unit`` and element text verbatim plus the outcome, so the stored record
    says what GDACS actually published — the September 2026 figures ledger
    showed ceilings of 5, 20 and 955 and nothing in the artifact could say
    whether the feed had said "5 people", "5 thousand" or "5 Million".

    * a unit in ``_POPULATION_PEOPLE_UNITS``: the value is people;
    * a multiplicative unit ("Million", "Thousand"): the value is scaled;
    * a unit that starts with "population" or "pop" (GDACS names measures
      that way): people, recorded as ``assumed_people``;
    * anything else: UNKNOWN (None), logged, never the bare number.
    """

    raw_value = "" if value is None else str(value).strip()
    raw_unit = "" if unit is None else str(unit).strip()
    detail: dict[str, Any] = {
        "raw_value": raw_value,
        "raw_unit": raw_unit,
        "text": (text or "").strip()[:200],
        "outcome": "",
        "multiplier": 1.0,
    }
    if not raw_value:
        detail["outcome"] = "no_value"
        return None, detail
    try:
        number = float(raw_value.replace(",", ""))
    except (TypeError, ValueError):
        detail["outcome"] = "value_not_a_number"
        return None, detail
    key = raw_unit.lower()
    if key in _POPULATION_PEOPLE_UNITS:
        detail["outcome"] = "people"
        return number, detail
    if key in _POPULATION_MULTIPLIERS:
        detail["multiplier"] = _POPULATION_MULTIPLIERS[key]
        detail["outcome"] = "scaled"
        return number * _POPULATION_MULTIPLIERS[key], detail
    if key.startswith("population") or key.startswith("pop"):
        detail["outcome"] = "assumed_people"
        return number, detail
    detail["outcome"] = "unrecognised_unit"
    LOG.warning(
        "[gdacs] gdacs:population unit %r is not recognised (value=%r, text=%r) — "
        "the exposure is UNKNOWN for this event, not %s",
        raw_unit, raw_value, detail["text"][:80], raw_value,
    )
    return None, detail


def _population_candidates(node: Any, out: dict[str, Any]) -> None:
    """Collect every population-shaped field anywhere under ``node``.

    A recursive walk rather than a fixed path, deliberately. The impacts
    array is where these fields live, but the exact nesting is the vendor's
    to change and a fixed path answers "absent" for a field that is right
    there — which is the shape of every silent connector failure in this
    repository. The walk finds the field wherever GDACS puts it, and the
    keys actually seen are recorded where it finds nothing, so a changed
    shape is settled by evidence rather than by re-reading this function.
    """

    if isinstance(node, dict):
        for key, value in node.items():
            if str(key).strip().lower() in _EVENT_DATA_POPULATION_FIELDS:
                out.setdefault(str(key).strip().lower(), value)
            else:
                _population_candidates(value, out)
    elif isinstance(node, list):
        for item in node:
            _population_candidates(item, out)


def parse_geteventdata_population(
    payload: Any,
) -> tuple[float | None, dict[str, Any]]:
    """The exposed population from a ``geteventdata`` body, or None.

    Returns ``(people, detail)``. ``detail`` names the field that answered,
    every population-shaped field seen with its raw value, and — when none
    answered — the top-level keys the body carried, so a body that changed
    shape says so in the next run's bundle instead of reading as an event
    with no exposure.

    Where several fields answer, the LARGEST wins. The figure is used as an
    upper bound on plausible impact, and a bound set too low rejects correct
    figures: that is the fault the ceiling multiplier was raised to 3.0 to
    end, and picking the 74 kt envelope over the 39 kt one would reintroduce
    it. Every candidate is recorded, so the choice is auditable.
    """

    detail: dict[str, Any] = {
        "outcome": "",
        "field": "",
        "candidates": {},
        "keys_seen": [],
    }
    impacts = None
    if isinstance(payload, dict):
        impacts = payload.get("impacts")
        if impacts is None and isinstance(payload.get("properties"), dict):
            impacts = payload["properties"].get("impacts")
    # Prefer the impacts array the API documents; fall back to the whole
    # body, because a field moved one level up is still the field.
    found: dict[str, Any] = {}
    if impacts is not None:
        _population_candidates(impacts, found)
    if not found:
        _population_candidates(payload, found)

    detail["candidates"] = {k: str(v)[:40] for k, v in found.items()}
    best_value: float | None = None
    best_field = ""
    for field_name, raw in found.items():
        # The field name IS the unit: `pop74` is in the people set and the
        # rest start with "pop", which parse_gdacs_population reads as
        # people. A value it cannot read is UNKNOWN, never the bare number.
        people, _ = parse_gdacs_population(
            None if raw is None else str(raw), field_name
        )
        if people is None or people <= 0:
            continue
        if best_value is None or people > best_value:
            best_value, best_field = people, field_name

    if best_value is None:
        detail["outcome"] = "no_population_field"
        if isinstance(payload, dict):
            detail["keys_seen"] = sorted(str(k) for k in payload)[:40]
        LOG.warning(
            "[gdacs] geteventdata carried no readable population field "
            "(looked for %s; saw %s) — the exposure is UNKNOWN for this "
            "event, not zero",
            ",".join(sorted(_EVENT_DATA_POPULATION_FIELDS)),
            ",".join(detail["keys_seen"]) or "nothing",
        )
        return None, detail

    detail["outcome"] = "ok"
    detail["field"] = best_field
    return best_value, detail


# ---------------------------------------------------------------------------
# GdacsConnector
# ---------------------------------------------------------------------------


def _geometry_resolver():
    """The PA machine's boundary loader, or None when it cannot be imported.

    ``resolver.hazard_resolution.gdacs`` borrows THIS module's helpers, so the
    import is made lazily at the one call site rather than at module scope,
    and its absence costs the fallback, never the connector.
    """

    try:
        from resolver.hazard_resolution.gdacs import _CountryGeometries

        return _CountryGeometries()
    except Exception as exc:  # noqa: BLE001 - the boundaries are optional here
        LOG.warning("[gdacs] geometry fallback unavailable: %s", exc)
        return None


def _iso3s_by_geometry(event: dict[str, Any], geometries) -> list[str]:
    """Countries near the event's own point, nearest first; [] when unplaceable.

    Delegates to :func:`resolver.hazard_resolution.gdacs._iso3s_near_event`
    so the connector and the machine attribute a country-less event by ONE
    rule and one distance (500 km). Never raises.
    """

    if geometries is None:
        return []
    try:
        from resolver.hazard_resolution.gdacs import _iso3s_near_event

        hazard = _HAZARD_MAP.get(str(event.get("eventtype") or ""), str(event.get("eventtype")))
        return list(_iso3s_near_event(event, hazard, geometries))
    except Exception as exc:  # noqa: BLE001
        LOG.warning(
            "[gdacs] geometry fallback failed for %s/%s: %s",
            event.get("eventtype"), event.get("eventid"), exc,
        )
        return []


def _item_point(item: ET.Element) -> tuple[float | None, float | None]:
    """(lat, lon) for a GDACS RSS item, or (None, None).

    Two spellings, both standard and both used by GDACS: ``geo:lat`` /
    ``geo:long`` as separate elements, and ``georss:point`` as one
    space-separated "lat lon" string. The pair is returned in the same
    (lat, lon) order and under the same keys the JSON path already uses,
    so the geometry resolver needs no second code path.
    """

    lat_text = _text(item, "geo:lat")
    lon_text = _text(item, "geo:long") or _text(item, "geo:lon")
    if lat_text and lon_text:
        try:
            return float(lat_text), float(lon_text)
        except (TypeError, ValueError):
            pass
    point = _text(item, "georss:point")
    if point:
        parts = point.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                return float(parts[0]), float(parts[1])
            except (TypeError, ValueError):
                pass
    return None, None


def _feature_point(feature: dict[str, Any], props: dict[str, Any]) -> tuple[float | None, float | None]:
    """(lat, lon) for a GDACS GeoJSON feature, or (None, None).

    Reads the feature geometry first and falls back to the latitude and
    longitude GDACS also puts in the properties block. Only a Point is used:
    the centroid of a cyclone track or a flood polygon is not a landfall, and
    guessing one would attribute an event to whichever country happened to be
    nearest the middle of it.
    """

    geometry = feature.get("geometry") or {}
    if str(geometry.get("type") or "").lower() == "point":
        coords = geometry.get("coordinates") or []
        if len(coords) >= 2:
            try:
                # GeoJSON is (lon, lat).
                return float(coords[1]), float(coords[0])
            except (TypeError, ValueError):
                pass
    try:
        lat = props.get("latitude")
        lon = props.get("longitude")
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    except (TypeError, ValueError):
        pass
    return None, None


class GdacsConnector:
    """Fetch GDACS disaster events and return a canonical DataFrame."""

    name: str = "gdacs"

    @staticmethod
    def _today() -> date:
        """Return today's date (extracted for testability)."""
        return date.today()

    def fetch_and_normalize(self) -> pd.DataFrame:
        """Fetch GDACS events and return canonical rows."""
        delay = float(os.getenv("GDACS_REQUEST_DELAY", "1.0"))
        months_back = int(os.getenv("GDACS_MONTHS", "3"))

        end_date = self._today()

        # Calculate start_date by subtracting months_back from end_date
        y, m = end_date.year, end_date.month
        m -= months_back
        while m <= 0:
            m += 12
            y -= 1
        start_date = date(y, m, 1)
        LOG.info("[gdacs] fetching %d months: %s to %s", months_back, start_date, end_date)
        name_to_iso3, iso3_to_name = _load_countries()
        session = _build_session()

        # Step 1: Fetch all events
        raw_events = self._fetch_all_events(
            session, start_date, end_date, delay, name_to_iso3,
        )
        if not raw_events:
            LOG.info("[gdacs] no events fetched")
            return empty_canonical()

        # Step 2: Deduplicate by (eventtype, eventid) keeping latest episode
        events = self._deduplicate(raw_events)

        # Step 3: Expand to country-month rows with population split
        rows = self._expand_to_country_months(events, name_to_iso3)
        if not rows:
            LOG.info("[gdacs] no country-month rows after expansion")
            return empty_canonical()

        # Step 4: Aggregate by (iso3, hazard_code, year, month)
        agg_df = self._aggregate(rows)

        # Step 5: Apply no-event logic (TC zero-fill)
        agg_df = self._apply_no_event_logic(agg_df, start_date, end_date, name_to_iso3)

        # Step 6: Map to canonical columns
        df = self._to_canonical(agg_df, iso3_to_name)

        LOG.info("[gdacs] produced %d canonical rows", len(df))
        return validate_canonical(df, source="gdacs", extra_columns=["alertlevel"])

    # -----------------------------------------------------------------------
    # Fetching — two strategies based on window size
    # -----------------------------------------------------------------------

    def _fetch_all_events(
        self,
        session: requests.Session,
        start_date: date,
        end_date: date,
        delay: float,
        name_to_iso3: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Fetch events using the best available data source.

        DR (drought) always uses the JSON search API because the static
        DR RSS feed (``rss_dr_3m.xml``) returns 404.  FL and TC use
        static RSS feeds when the window is ≤3 months (fast, includes
        population data) and the JSON API otherwise.
        """
        use_rss = os.getenv("GDACS_FORCE_RSS", "").strip().lower() in ("1", "true")
        use_json = os.getenv("GDACS_FORCE_JSON", "").strip().lower() in ("1", "true")

        if use_json:
            return self._fetch_via_json_api(
                session, start_date, end_date, delay, name_to_iso3,
            )
        if use_rss:
            return self._fetch_via_static_rss(session, name_to_iso3)

        # Auto-detect: use RSS for <=3 months, JSON API for longer
        months_span = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        if months_span <= 3:
            LOG.info("[gdacs] using static RSS feeds for FL/TC (<=3 month window)")
            events = self._fetch_via_static_rss(session, name_to_iso3)
            if events:
                # Filter to date range (RSS feeds cover exactly 3 months)
                events = [
                    e for e in events
                    if e["todate"] >= start_date and e["fromdate"] <= end_date
                ]
            else:
                LOG.warning("[gdacs] static RSS returned 0 FL/TC events")

            # DR always via JSON API (DR RSS feed returns 404)
            LOG.info("[gdacs] fetching DR events via JSON search API")
            dr_events = self._fetch_via_json_api(
                session, start_date, end_date, delay, name_to_iso3,
                event_types=["DR"],
            )
            LOG.info("[gdacs] JSON API returned %d DR events", len(dr_events))
            return (events or []) + dr_events

        LOG.info("[gdacs] using JSON search API (>3 month window)")
        return self._fetch_via_json_api(
            session, start_date, end_date, delay, name_to_iso3,
        )

    # -----------------------------------------------------------------------
    # Strategy 1: Static RSS feeds (fast, 3-month window, has population)
    # -----------------------------------------------------------------------

    def _fetch_via_static_rss(
        self,
        session: requests.Session,
        name_to_iso3: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Fetch from static RSS feeds (FL 3m, TC 3m)."""
        all_events: list[dict[str, Any]] = []
        for etype in _STATIC_RSS:
            all_events.extend(
                self.fetch_static_rss_for_type(session, etype, name_to_iso3)
            )
        return all_events

    def fetch_static_rss_for_type(
        self,
        session: requests.Session,
        etype: str,
        name_to_iso3: dict[str, str],
    ) -> list[dict[str, Any]]:
        """One static feed's events, or [] when it cannot be read.

        Split out so the PA machine can borrow it: the 3-month feeds carry
        ``gdacs:population`` for every event they list, in ONE request, and
        the machine was paying a per-event request for figures already on
        the table. Never raises — a feed that cannot be read costs the
        events it would have covered and nothing else.
        """

        url = _STATIC_RSS.get(etype)
        if not url:
            return []
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 404:
                LOG.warning("[gdacs] static RSS 404 for %s: %s", etype, url)
                return []
            resp.raise_for_status()
            events = self._parse_rss(resp.content, name_to_iso3)
            LOG.info("[gdacs] static RSS %s: %d events", etype, len(events))
            return events
        except Exception as exc:  # noqa: BLE001 - a feed decides nothing on its own
            LOG.warning("[gdacs] error fetching static RSS %s: %s", etype, exc)
            return []

    def _parse_rss(
        self,
        xml_bytes: bytes,
        name_to_iso3: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Parse RSS XML and extract relevant event records."""
        events: list[dict[str, Any]] = []
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError as exc:
            LOG.warning("[gdacs] XML parse error: %s", exc)
            return events

        for item in root.iter("item"):
            try:
                event = self._parse_item(item, name_to_iso3)
                if event is not None:
                    events.append(event)
            except Exception as exc:
                LOG.warning("[gdacs] error parsing item: %s", exc)
        return events

    def _parse_item(
        self,
        item: ET.Element,
        name_to_iso3: dict[str, str],
    ) -> dict[str, Any] | None:
        """Parse a single RSS <item> into an event dict, or None if filtered."""
        eventtype = _text(item, "gdacs:eventtype")
        if not eventtype or eventtype not in _WANTED_TYPES:
            return None

        eventid = _text(item, "gdacs:eventid")
        if not eventid:
            return None

        # Population exposed. The element carries value AND unit attributes
        # and a descriptive text; all three are kept verbatim on the event
        # so the stored record can answer what GDACS actually said. A value
        # that cannot be read as people is UNKNOWN (0.0 here, the connector's
        # long-standing "no figure" convention), never the bare number.
        pop_element = item.find("gdacs:population", _NS)
        pop_value = _attr(item, "gdacs:population", "value")
        pop_unit = _attr(item, "gdacs:population", "unit")
        pop_text = (pop_element.text or "") if pop_element is not None else ""
        parsed_population, population_detail = parse_gdacs_population(
            pop_value, pop_unit, pop_text
        )
        population = parsed_population if parsed_population is not None else 0.0

        # ISO3 — try direct field first, then resolve from country name
        iso3 = _text(item, "gdacs:iso3")
        country = _text(item, "gdacs:country")
        if not iso3 and country:
            iso3 = name_to_iso3.get(country.lower())

        # Dates
        fromdate = _parse_date(_text(item, "gdacs:fromdate"))
        todate = _parse_date(_text(item, "gdacs:todate"))
        if not fromdate:
            return None
        if not todate:
            todate = fromdate

        # Alert info
        alertlevel = _text(item, "gdacs:alertlevel") or "Green"
        alertscore = _text(item, "gdacs:alertscore")

        # Position. The JSON discovery path sets lat/lon from the feature
        # geometry; the RSS path did not read them at all, so an event GDACS
        # names no country for — a cyclone still over open ocean, which is
        # most of them at discovery — had nothing for the geometry resolver
        # to work with and was dropped outright. Seventeen TC events went
        # that way in run 33946954189.
        lat, lon = _item_point(item)

        # Publication date (use pubDate if available)
        pub_date_text = None
        pub_el = item.find("pubDate")
        if pub_el is not None and pub_el.text:
            pub_date_text = pub_el.text.strip()
        pub_date = _parse_date(pub_date_text) or todate

        return {
            "eventtype": eventtype,
            "eventid": eventid,
            "population": population,
            "population_unit": pop_unit or "",
            "population_text": population_detail["text"],
            "population_raw": population_detail["raw_value"],
            "population_parse": population_detail["outcome"],
            "iso3": iso3,
            "country": country,
            "lat": lat,
            "lon": lon,
            "fromdate": fromdate,
            "todate": todate,
            "alertlevel": alertlevel,
            "alertscore": alertscore,
            "pub_date": pub_date,
        }

    # -----------------------------------------------------------------------
    # Strategy 2: JSON search API (any date range) + per-event RSS
    # -----------------------------------------------------------------------

    def _fetch_via_json_api(
        self,
        session: requests.Session,
        start_date: date,
        end_date: date,
        delay: float,
        name_to_iso3: dict[str, str],
        event_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch events via the JSON search API, then enrich with per-event RSS.

        Parameters
        ----------
        event_types:
            Optional subset of event types to query (e.g. ``["DR"]``).
            Defaults to all wanted types (FL, TC, DR).
        """
        # Step 1: Discover events via JSON search API (chunked by quarter)
        json_events = self._search_events(
            session, start_date, end_date, delay, event_types=event_types,
        )
        LOG.info("[gdacs] JSON API returned %d events", len(json_events))
        if not json_events:
            return []

        # Step 2: Fetch per-event RSS for population data
        events = self._enrich_with_population(
            session, json_events, delay, name_to_iso3,
        )
        return events

    def _search_events(
        self,
        session: requests.Session,
        start_date: date,
        end_date: date,
        delay: float,
        event_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the GDACS JSON search API in quarterly chunks."""
        all_events: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, int]] = set()

        # Chunk into quarters to avoid hitting response limits
        chunk_start = start_date
        while chunk_start <= end_date:
            # End of quarter (3 months from start)
            cy, cm = chunk_start.year, chunk_start.month
            cm += 3
            while cm > 12:
                cm -= 12
                cy += 1
            chunk_end = min(date(cy, cm, 1), end_date)

            try:
                events = self._search_chunk(
                    session, chunk_start, chunk_end, event_types=event_types,
                )
                for ev in events:
                    key = (ev["eventtype"], ev["eventid"])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_events.append(ev)
            except Exception as exc:
                LOG.warning(
                    "[gdacs] JSON API error for %s to %s: %s",
                    chunk_start, chunk_end, exc,
                )

            # Advance to next chunk — if chunk_end reached end_date, stop.
            if chunk_end >= end_date:
                break
            chunk_start = chunk_end
            if delay > 0:
                time.sleep(delay)

        return all_events

    def _search_chunk(
        self,
        session: requests.Session,
        from_date: date,
        to_date: date,
        event_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the JSON search API for a single date chunk."""
        params = {
            "eventlist": ";".join(event_types or sorted(_WANTED_TYPES)),
            "fromDate": from_date.isoformat(),
            "toDate": to_date.isoformat(),
            "alertlevel": "Green;Orange;Red",
        }
        resp = session.get(_SEARCH_API, params=params, timeout=60)
        resp.raise_for_status()
        if not resp.text.strip():
            LOG.debug("[gdacs] empty response for %s to %s", from_date, to_date)
            return []
        data = resp.json()

        events: list[dict[str, Any]] = []
        features = data.get("features", [])
        for feat in features:
            props = feat.get("properties", {})
            eventtype = props.get("eventtype", "")
            if eventtype not in _WANTED_TYPES:
                continue

            # Extract affected countries with ISO3 codes
            affected = props.get("affectedcountries", [])
            iso3_list = [
                c.get("iso3", "").strip().upper()
                for c in affected
                if c.get("iso3", "").strip()
            ]
            # Fallback to primary iso3 field
            primary_iso3 = (props.get("iso3") or "").strip().upper()
            if not iso3_list and primary_iso3:
                iso3_list = [primary_iso3]

            fromdate = _parse_date(props.get("fromdate"))
            todate = _parse_date(props.get("todate"))
            if not fromdate:
                continue
            if not todate:
                todate = fromdate

            # Keep the feature's own coordinates. GDACS names no country for
            # an event whose affectedcountries list is empty — routine for a
            # tropical cyclone still over open ocean — and without a position
            # such an event can only be dropped. With one it can be resolved
            # against the boundaries the PA machine already ships.
            lat, lon = _feature_point(feat, props)

            # WHAT THE BULK ROUTE ACTUALLY CARRIES. The per-event RSS costs
            # one request per event and is where every refusal came from; if
            # the search response already states an exposure, the whole
            # enrichment pass is unnecessary. The code assumed it does not
            # and recorded nothing, so the assumption could never be checked
            # — a connector's response envelope is evidence the connector
            # itself discards. The property keys are noted once per run, and
            # a population-shaped field is USED when one is present.
            _note_search_properties(props)
            bulk_population, bulk_field = _population_from_properties(props)

            events.append({
                "eventtype": eventtype,
                "eventid": props.get("eventid"),
                "iso3": primary_iso3 or (iso3_list[0] if iso3_list else ""),
                "iso3_list": iso3_list,
                "country": props.get("country", ""),
                "lat": lat,
                "lon": lon,
                "fromdate": fromdate,
                "todate": todate,
                "alertlevel": props.get("alertlevel", "Green"),
                "alertscore": props.get("alertscore"),
                # 0.0 unless the search response stated one, in which case
                # the per-event fetch is not needed at all.
                "population": bulk_population,
                "population_enriched": bulk_population > 0,
                "population_source": "search" if bulk_population > 0 else "",
                "population_parse": bulk_field,
                "pub_date": _parse_date(props.get("datemodified")) or todate,
            })

        return events

    def _enrich_one_event(
        self,
        session: requests.Session,
        ev: dict[str, Any],
        name_to_iso3: dict[str, str],
    ) -> dict[str, Any]:
        """Fetch one event's per-event RSS and merge population data in place.

        Errors (404, network, parse) are tolerated — the event is returned
        unenriched with ``population`` left at its discovery-time value, and
        marked ``population_enriched = False`` so a caller can tell an event
        GDACS declined to describe from one it described as zero.

        **A refusal is retried.** In run 33946954189, 532 of 1,050 per-event
        fetches came back HTTP 403 — the refusals starting after about
        thirteen successes and continuing for most of the run, which is rate
        limiting rather than a permission problem. urllib3's retry list does
        not carry 403, so every one of those events lost its exposure figure
        and the cells that depended on it reconciled with no upper bound at
        all. The backoff is exponential with jitter, because a fleet of
        workers retrying in lockstep is the same burst again.
        """
        etype = ev["eventtype"]
        eid = ev["eventid"]
        ev.setdefault("population_enriched", False)

        # An event's exposure does not change during a run, but the machine
        # asks for it once per hazard-month pass: run 34081262443 made 2,678
        # per-event requests for 294 distinct events — up to 24 for one event
        # across six passes at four attempts each — and 2,132 of them were
        # refused. Answering the second and later passes from memory is not a
        # cache of upstream state; it is not asking the same question six
        # times in one process. A refusal is remembered too: re-running the
        # whole retry ladder against a host that has just refused it four
        # times spends four more requests to learn the same thing, and the
        # raw cache's carry-forward is what supplies the figure meanwhile.
        memo_key = (str(etype), str(eid))
        remembered = _EXPOSURE_MEMO.get(memo_key)
        if remembered is not None:
            return self._apply_exposure(ev, remembered)

        best, refused = self._fetch_event_exposure(session, etype, eid, name_to_iso3)
        with _EXPOSURE_MEMO_LOCK:
            _EXPOSURE_MEMO[memo_key] = (best, refused)
        return self._apply_exposure(ev, (best, refused))

    @staticmethod
    def _apply_exposure(
        ev: dict[str, Any],
        outcome: tuple[dict[str, Any] | None, int | None],
    ) -> dict[str, Any]:
        """Merge one event's fetched exposure onto ``ev``.

        Split out of the fetch so a remembered outcome and a fresh one reach
        the event by exactly the same path.
        """

        best, refused = outcome
        if refused is not None:
            ev["population_refused"] = int(refused)
            return ev
        if not best:
            return ev
        ev["population"] = best["population"]
        for key in (
            "population_unit", "population_text",
            "population_raw", "population_parse",
            # A figure served from the cache must say so, and say when it
            # was read. A stale figure is worth more than none, and only
            # worth anything if it is labelled.
            "population_source", "population_cached_at",
        ):
            ev[key] = best.get(key, ev.get(key, ""))
        ev.setdefault("population_source", "live")
        if not ev.get("population_source"):
            ev["population_source"] = "live"
        # Also update iso3/country if the RSS has better data
        if best.get("iso3") and not ev.get("iso3"):
            ev["iso3"] = best["iso3"]
        if best.get("country") and not ev.get("country"):
            ev["country"] = best["country"]
        if best.get("lat") is not None and ev.get("lat") is None:
            ev["lat"] = best.get("lat")
        if best.get("lon") is not None and ev.get("lon") is None:
            ev["lon"] = best.get("lon")
        ev["population_enriched"] = True
        return ev

    def _fetch_event_exposure(
        self,
        session: requests.Session,
        etype: str,
        eid: Any,
        name_to_iso3: dict[str, str],
    ) -> tuple[dict[str, Any] | None, int | None]:
        """``(best RSS episode or None, refusal status or None)`` for one event."""

        url = _EVENT_RSS_PATTERN.format(type=etype, eventid=eid)
        attempts = _enrich_attempts()

        for attempt in range(1, attempts + 1):
            try:
                # Every per-event request in this process passes here, so
                # the pace holds whatever the worker count is.
                _ENRICH_LIMITER.acquire()
                resp = session.get(url, timeout=30)
                if resp.status_code == 404:
                    LOG.debug("[gdacs] per-event RSS 404 for %s/%s", etype, eid)
                    return None, None
                if resp.status_code in _RETRYABLE_ENRICH_STATUS and attempt < attempts:
                    pause = _backoff_seconds(attempt)
                    LOG.debug(
                        "[gdacs] per-event RSS %d for %s/%s — attempt %d of %d, "
                        "sleeping %.2fs",
                        resp.status_code, etype, eid, attempt, attempts, pause,
                    )
                    time.sleep(pause)
                    continue
                if resp.status_code in _RETRYABLE_ENRICH_STATUS:
                    LOG.debug(
                        "[gdacs] per-event RSS %d for %s/%s after %d attempts — "
                        "the event keeps its discovery-time population",
                        resp.status_code, etype, eid, attempts,
                    )
                    return None, int(resp.status_code)
                if resp.status_code in _REFUSAL_STATUS:
                    # The datareport tree is refusing, and asked once — the
                    # answer to a refusing source is fewer requests, not a
                    # harder retry. But a 403 there is a statement about a
                    # published REPORT that does not exist, not about the
                    # event, and the API describes the event regardless. So
                    # one request to the API's own route before giving up.
                    LOG.debug(
                        "[gdacs] per-event RSS %d for %s/%s — not retried, "
                        "asking geteventdata once",
                        resp.status_code, etype, eid,
                    )
                    episode = self._fetch_event_data_exposure(session, etype, eid)
                    if episode is not None:
                        return episode, None
                    return None, int(resp.status_code)
                resp.raise_for_status()

                # Parse the per-event RSS to get population
                rss_events = self._parse_rss(resp.content, name_to_iso3)
                if not rss_events:
                    return None, None
                # Take the latest episode (highest todate)
                best = max(
                    rss_events,
                    key=lambda e: (e["todate"], e.get("pub_date") or e["todate"]),
                )
                return best, None
            except Exception as exc:
                if attempt < attempts:
                    pause = _backoff_seconds(attempt)
                    LOG.debug(
                        "[gdacs] error fetching RSS for %s/%s (attempt %d of %d): "
                        "%s — sleeping %.2fs",
                        etype, eid, attempt, attempts, exc, pause,
                    )
                    time.sleep(pause)
                    continue
                LOG.debug("[gdacs] error fetching RSS for %s/%s: %s", etype, eid, exc)
                return None, None

        return None, None

    def _fetch_event_data_exposure(
        self,
        session: requests.Session,
        etype: str,
        eid: Any,
    ) -> dict[str, Any] | None:
        """One request to ``geteventdata``, or None. Never raises.

        The datareport RSS is a published-report artefact: GDACS answers 403
        for an event it never wrote a report for, which says nothing about
        whether the event happened or how many people it exposed. The API's
        own per-event route still describes it, and it is a different host
        path with a different refusal, so a refusal there is not evidence
        that this one will refuse.

        Asked exactly once, under the same process-wide pace as every other
        per-event request. A failure returns None and the caller records the
        original refusal, so an event GDACS genuinely will not describe is
        still counted as refused rather than silently carrying no figure.
        """

        url = _EVENT_DATA_PATTERN.format(type=etype, eventid=eid)
        try:
            _ENRICH_LIMITER.acquire()
            resp = session.get(url, timeout=30)
            if resp.status_code != 200:
                LOG.debug(
                    "[gdacs] geteventdata %d for %s/%s", resp.status_code, etype, eid
                )
                return None
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001 - a fallback never costs the run
            LOG.debug("[gdacs] geteventdata failed for %s/%s: %s", etype, eid, exc)
            return None

        people, detail = parse_geteventdata_population(payload)
        if people is None or people <= 0:
            return None
        LOG.debug(
            "[gdacs] geteventdata answered for %s/%s: %s = %s",
            etype, eid, detail["field"], people,
        )
        return {
            "population": people,
            # The field name is the unit, and is kept verbatim: a ceiling
            # of 2 against a reported 40,000 is an enrichment failure, and
            # only the source column says which field produced it.
            "population_unit": detail["field"],
            "population_text": "",
            "population_raw": str(detail["candidates"].get(detail["field"], "")),
            "population_parse": detail["outcome"],
            # Which endpoint answered. Without it a figure the datareport
            # route refused and the API supplied is indistinguishable from
            # one the datareport route served.
            "population_source": "geteventdata",
        }

    def _enrich_with_population(
        self,
        session: requests.Session,
        events: list[dict[str, Any]],
        delay: float,
        name_to_iso3: dict[str, str],
        *,
        workers: int | None = None,
        min_interval: float | None = None,
        max_seconds: float | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch per-event RSS to get population data for each event.

        With ``GDACS_ENRICH_WORKERS`` > 1 (default 6) the per-event fetches
        run on a thread pool — each worker keeps its own ``requests.Session``
        (sessions are not thread-safe) and still sleeps ``delay`` between its
        own requests, so the effective request rate is roughly
        ``workers / (delay + response_time)``.  The 2026-07-08 reset run
        enriched 3,272 events strictly sequentially at ~1.3s each (72 min);
        the pool brings that to ~10 min without hammering GDACS.
        ``GDACS_ENRICH_WORKERS=1`` preserves the exact sequential behavior.
        """
        global _ENRICH_LIMITER
        total = len(events)
        # An explicit argument wins (the PA machine passes the rulebook's
        # values, which is what makes `flood.gdacs.enrich_workers` mean
        # something — it was validated and read by nothing until Sept 2026,
        # so lowering it did nothing at all). Then the env, then the default.
        if workers is None:
            workers = int(
                os.getenv("GDACS_ENRICH_WORKERS", "") or _DEFAULT_ENRICH_WORKERS
            )
        workers = max(1, int(workers))
        if min_interval is not None:
            _ENRICH_LIMITER = _RateLimiter(float(min_interval))
        budget = _enrich_max_seconds() if max_seconds is None else float(max_seconds)
        deadline = (time.monotonic() + budget) if budget > 0 else None
        unasked = 0
        # The enrichment delay has its OWN knob now (GDACS_ENRICH_DELAY), so
        # an operator can slow the per-event fetches without touching
        # discovery. Unset, the caller's value stands.
        delay = _enrich_delay(delay)

        if workers == 1 or total <= 1:
            enriched: list[dict[str, Any]] = []
            for i, ev in enumerate(events):
                if deadline is not None and time.monotonic() >= deadline:
                    # Stop asking, keep the events. A memo hit costs no
                    # request, so this only ever cuts short the genuinely
                    # unknown ones — which the next run asks for.
                    unasked = total - i
                    enriched.extend(events[i:])
                    break
                enriched.append(self._enrich_one_event(session, ev, name_to_iso3))
                if (i + 1) % 50 == 0:
                    LOG.info("[gdacs] enriched %d/%d events", i + 1, total)
                if delay > 0:
                    time.sleep(delay)
            self._log_enrichment_outcome(enriched, unasked=unasked)
            return enriched

        LOG.info("[gdacs] enriching %d events with %d workers", total, workers)
        thread_local = threading.local()

        overrun = threading.Event()

        def _worker(ev: dict[str, Any]) -> dict[str, Any]:
            if deadline is not None and time.monotonic() >= deadline:
                overrun.set()
                return ev
            worker_session = getattr(thread_local, "session", None)
            if worker_session is None:
                worker_session = _build_session()
                thread_local.session = worker_session
            result = self._enrich_one_event(worker_session, ev, name_to_iso3)
            if delay > 0:
                time.sleep(delay)
            return result

        enriched = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker, ev) for ev in events]
            for done, future in enumerate(as_completed(futures)):
                enriched.append(future.result())
                if (done + 1) % 50 == 0:
                    LOG.info("[gdacs] enriched %d/%d events", done + 1, total)

        if overrun.is_set():
            unasked = sum(
                1 for e in enriched
                if not e.get("population_enriched") and not e.get("population_refused")
            )
        self._log_enrichment_outcome(enriched, unasked=unasked)
        return enriched

    @staticmethod
    def _log_enrichment_outcome(
        events: list[dict[str, Any]], *, unasked: int = 0
    ) -> None:
        """Say how many events GDACS actually described.

        A refusal rate is the only way to tell "no exposure figure exists"
        from "we were throttled": the run that prompted this counted 532
        refusals in 1,050 requests and reported nothing but a per-event
        DEBUG line nobody reads at INFO.
        """

        refused = [e for e in events if e.get("population_refused")]
        enriched = sum(1 for e in events if e.get("population_enriched"))
        if unasked:
            LOG.warning(
                "[gdacs] the enrichment pass ran out of time with %d event(s) "
                "unasked — they keep whatever the cache holds and the next "
                "run asks for them; raise GDACS_ENRICH_MAX_SECONDS or lower "
                "GDACS_ENRICH_MIN_INTERVAL_SEC if this persists",
                unasked,
            )
        if refused:
            statuses = sorted({int(e["population_refused"]) for e in refused})
            LOG.warning(
                "[gdacs] %d of %d per-event fetches were refused after retries "
                "(status %s) — those events keep no exposure figure, so the "
                "cells that need one reconcile with no upper bound",
                len(refused), len(events),
                ",".join(str(s) for s in statuses),
            )
        LOG.info(
            "[gdacs] enrichment: %d of %d events carry an exposure figure",
            enriched, len(events),
        )

    # -----------------------------------------------------------------------
    # Deduplication
    # -----------------------------------------------------------------------

    def _deduplicate(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only the latest episode per (eventtype, eventid)."""
        best: dict[tuple[str, str], dict[str, Any]] = {}
        for ev in events:
            key = (ev["eventtype"], str(ev["eventid"]))
            existing = best.get(key)
            if existing is None:
                best[key] = ev
            else:
                # Keep the one with the later todate (or pub_date as tiebreaker)
                ev_sort = (ev["todate"], ev.get("pub_date") or ev["todate"])
                ex_sort = (existing["todate"], existing.get("pub_date") or existing["todate"])
                if ev_sort > ex_sort:
                    best[key] = ev
        return list(best.values())

    # -----------------------------------------------------------------------
    # Country-month expansion
    # -----------------------------------------------------------------------

    def _expand_to_country_months(
        self,
        events: list[dict[str, Any]],
        name_to_iso3: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Expand events to per-country per-month rows."""
        rows: list[dict[str, Any]] = []
        geometries = None
        self.events_resolved_by_geometry: list[str] = []
        self.events_without_country: list[str] = []
        for ev in events:
            iso3_list = self._resolve_countries(ev, name_to_iso3)
            if not iso3_list:
                # GDACS names no country for an event still over open ocean
                # (routine for a tropical cyclone), and the row it produces
                # was read by no cell. The PA machine already resolves such
                # an event from its own position against the vendored
                # boundaries; the connector path now does the same, so a
                # storm sitting 200 km off a coast lands on that coast's
                # country-month in facts_resolved as it does in haz_raw_gdacs.
                if geometries is None:
                    geometries = _geometry_resolver()
                iso3_list = _iso3s_by_geometry(ev, geometries)
                if iso3_list:
                    ev["iso3_from_geometry"] = True
                    self.events_resolved_by_geometry.append(
                        f"{ev['eventtype']}-{ev['eventid']}:{','.join(iso3_list[:3])}"
                    )
            if not iso3_list:
                self.events_without_country.append(f"{ev['eventtype']}-{ev['eventid']}")
                LOG.warning(
                    "[gdacs] cannot resolve country for event %s/%s (country=%r, "
                    "lat=%r, lon=%r) — dropped",
                    ev["eventtype"], ev["eventid"], ev.get("country"),
                    ev.get("lat"), ev.get("lon"),
                )
                continue

            months = _overlapping_months(ev["fromdate"], ev["todate"])
            pop_shares = self._population_split(iso3_list, ev["population"])

            for iso3, pop_value in pop_shares.items():
                for ym in months:
                    rows.append({
                        "iso3": iso3,
                        "hazard_code": _HAZARD_MAP[ev["eventtype"]],
                        "year": ym[0],
                        "month": ym[1],
                        "value": pop_value,
                        "alertlevel": ev["alertlevel"],
                        "todate": ev["todate"],
                    })
        if self.events_resolved_by_geometry:
            LOG.info(
                "[gdacs] %d event(s) named no country and were placed from their "
                "position: %s",
                len(self.events_resolved_by_geometry),
                "; ".join(self.events_resolved_by_geometry[:20]),
            )
        if self.events_without_country:
            # INFO, not WARNING. A tropical cyclone over open ocean names no
            # country and sits too far from any coast to be placed, which is
            # the feed describing a storm at sea rather than a fault. It is
            # counted and named so a rise is still visible, but a warning
            # every run for the ordinary case is how a reader learns to skip
            # the warnings that matter.
            LOG.info(
                "[gdacs] %d event(s) were over open ocean or otherwise too far "
                "from land to place, and produced no row: %s",
                len(self.events_without_country),
                ",".join(self.events_without_country[:30]),
            )
        return rows

    def _resolve_countries(
        self,
        event: dict[str, Any],
        name_to_iso3: dict[str, str],
    ) -> list[str]:
        """Resolve an event to a list of ISO3 codes."""
        # JSON API events may carry a pre-extracted iso3_list
        iso3_list = event.get("iso3_list")
        if iso3_list and all(len(c) == 3 for c in iso3_list):
            return iso3_list

        iso3 = event.get("iso3")
        if iso3 and len(iso3) == 3:
            # Could be comma-separated for multi-country events
            codes = [c.strip().upper() for c in iso3.split(",") if c.strip()]
            if all(len(c) == 3 for c in codes):
                return codes

        # Try country name lookup (may be comma-separated)
        country = event.get("country")
        if country:
            parts = [p.strip() for p in country.split(",")]
            resolved = []
            for p in parts:
                code = name_to_iso3.get(p.lower())
                if code:
                    resolved.append(code)
            if resolved:
                return resolved

        return []

    def _population_split(
        self,
        iso3_list: list[str],
        total_population: float,
    ) -> dict[str, float]:
        """Split population value across countries by population weight."""
        if len(iso3_list) == 1:
            return {iso3_list[0]: total_population}

        pops = {c: _POPULATION.get(c, 1_000_000) for c in iso3_list}
        total_pop = sum(pops.values())
        if total_pop == 0:
            equal_share = total_population / len(iso3_list)
            return {c: equal_share for c in iso3_list}

        return {c: total_population * (p / total_pop) for c, p in pops.items()}

    # -----------------------------------------------------------------------
    # Aggregation
    # -----------------------------------------------------------------------

    def _aggregate(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        """Group by (iso3, hazard_code, year, month) and SUM values."""
        df = pd.DataFrame(rows)
        agg = df.groupby(["iso3", "hazard_code", "year", "month"], as_index=False).agg(
            value=("value", "sum"),
            alertlevel=("alertlevel", lambda x: max(
                x, key=lambda a: {"Red": 3, "Orange": 2, "Green": 1}.get(a, 0)
            )),
            todate=("todate", "max"),
        )
        return agg

    # -----------------------------------------------------------------------
    # No-event logic
    # -----------------------------------------------------------------------

    def _apply_no_event_logic(
        self,
        agg_df: pd.DataFrame,
        start_date: date,
        end_date: date,
        name_to_iso3: dict[str, str],
    ) -> pd.DataFrame:
        """Apply zero-fill for TC; FL and DR get no zero-fill."""
        # Get all countries that have ANY TC event in the dataset
        tc_rows = agg_df[agg_df["hazard_code"] == "TC"]
        tc_countries = set(tc_rows["iso3"].unique()) if not tc_rows.empty else set()

        if not tc_countries:
            return agg_df

        # Build the full set of (country, year, month) for TC
        all_months = list(_month_range(start_date, end_date))
        zero_rows: list[dict[str, Any]] = []

        # Existing TC (iso3, year, month) combos
        tc_existing = set()
        if not tc_rows.empty:
            for _, r in tc_rows.iterrows():
                tc_existing.add((r["iso3"], r["year"], r["month"]))

        for iso3 in tc_countries:
            for y, m in all_months:
                if (iso3, y, m) not in tc_existing:
                    zero_rows.append({
                        "iso3": iso3,
                        "hazard_code": "TC",
                        "year": y,
                        "month": m,
                        "value": 0.0,
                        "alertlevel": "Green",
                        "todate": _last_day(y, m),
                    })

        if zero_rows:
            zero_df = pd.DataFrame(zero_rows)
            agg_df = pd.concat([agg_df, zero_df], ignore_index=True)

        return agg_df

    # -----------------------------------------------------------------------
    # Canonical mapping
    # -----------------------------------------------------------------------

    def _to_canonical(
        self,
        agg_df: pd.DataFrame,
        iso3_to_name: dict[str, str],
    ) -> pd.DataFrame:
        """Map aggregated rows to canonical format plus event_occurrence rows.

        Produces TWO rows per (iso3, hazard_code, year, month):
        1. metric="in_need" — population exposed (existing behaviour)
        2. metric="event_occurrence" — binary 1/0 based on alertlevel

        Also carries ``alertlevel`` as a supplementary column (not part of
        CANONICAL_COLUMNS) so it can flow into facts_resolved.
        """
        now_utc = datetime.now(timezone.utc).isoformat()

        records: list[dict[str, str]] = []
        alertlevels: list[str] = []

        for _, row in agg_df.iterrows():
            iso3 = row["iso3"]
            hazard_code = row["hazard_code"]
            year = int(row["year"])
            month = int(row["month"])
            value = row["value"]
            alertlevel = row.get("alertlevel", "Green")
            todate = row.get("todate")

            as_of = _last_day(year, month)
            pub_date = todate if isinstance(todate, date) else as_of
            confidence = _CONFIDENCE_MAP.get(alertlevel, "low")
            country_name = iso3_to_name.get(iso3, "")
            hazard_label = _HAZARD_LABEL.get(hazard_code, "")
            pub_date_str = pub_date.isoformat() if isinstance(pub_date, date) else str(pub_date)

            # --- Row 1: population exposed (metric="in_need") ---
            records.append({
                "event_id": "",
                "country_name": country_name,
                "iso3": iso3,
                "hazard_code": hazard_code,
                "hazard_label": hazard_label,
                "hazard_class": _HAZARD_CLASS.get(hazard_code, "natural"),
                "metric": "in_need",
                "series_semantics": "stock",
                "value": str(value),
                "unit": "persons",
                "as_of_date": as_of.isoformat(),
                "publication_date": pub_date_str,
                "publisher": "GDACS / JRC",
                "source_type": "satellite_derived",
                "source_url": "https://www.gdacs.org",
                "doc_title": "",
                "definition_text": (
                    f"Population exposed to {hazard_label.lower()} conditions "
                    f"as estimated by GDACS using GHSL population overlay"
                ),
                "method": "ghsl_exposure_overlay",
                "confidence": confidence,
                "revision": "",
                "ingested_at": now_utc,
            })
            alertlevels.append(alertlevel)

            # --- Row 2: binary event occurrence ---
            event_value = 1 if alertlevel in ("Orange", "Red") else 0
            records.append({
                "event_id": "",
                "country_name": country_name,
                "iso3": iso3,
                "hazard_code": hazard_code,
                "hazard_label": hazard_label,
                "hazard_class": _HAZARD_CLASS.get(hazard_code, "natural"),
                "metric": "event_occurrence",
                "series_semantics": "stock",
                "value": str(event_value),
                "unit": "binary",
                "as_of_date": as_of.isoformat(),
                "publication_date": pub_date_str,
                "publisher": "GDACS / JRC",
                "source_type": "satellite_derived",
                "source_url": "https://www.gdacs.org",
                "doc_title": "",
                "definition_text": (
                    f"Binary indicator of {hazard_label.lower()} event occurrence "
                    f"based on GDACS alert level (1=Orange/Red, 0=Green/none)"
                ),
                "method": "gdacs_alertlevel_threshold",
                "confidence": confidence,
                "revision": "",
                "ingested_at": now_utc,
            })
            alertlevels.append(alertlevel)

        df = pd.DataFrame(records, columns=CANONICAL_COLUMNS)

        # Add alertlevel as a supplementary column (not part of canonical
        # schema, but will be written to facts_resolved if the column exists).
        df["alertlevel"] = alertlevels

        return df
