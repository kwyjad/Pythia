# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""GDACS RSS archive connector.

Fetches disaster events (drought, flood, tropical cyclone) from the GDACS
RSS archive endpoint, extracts population-exposed figures, and returns a
canonical DataFrame for the Resolver pipeline.
"""

from __future__ import annotations

import logging
import os
import time
import xml.etree.ElementTree as ET
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

_BASE_URL = "http://gdacs.org/rss.aspx"

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

# XML namespaces used in the GDACS RSS feed
_NS: dict[str, str] = {
    "gdacs": "http://www.gdacs.org",
    "geo": "http://www.w3.org/2003/01/geo/wgs84_pos#",
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
    """Parse a date string from GDACS XML."""
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


# ---------------------------------------------------------------------------
# GdacsConnector
# ---------------------------------------------------------------------------


class GdacsConnector:
    """Fetch GDACS disaster events and return a canonical DataFrame."""

    name: str = "gdacs"

    @staticmethod
    def _today() -> date:
        """Return today's date (extracted for testability)."""
        return date.today()

    def fetch_and_normalize(self) -> pd.DataFrame:
        """Fetch GDACS RSS archive and return canonical rows."""
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
        return validate_canonical(df, source="gdacs")

    # -----------------------------------------------------------------------
    # Fetching
    # -----------------------------------------------------------------------

    def _fetch_all_events(
        self,
        session: requests.Session,
        start_date: date,
        end_date: date,
        delay: float,
        name_to_iso3: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Fetch RSS archive month-by-month and extract events."""
        events: list[dict[str, Any]] = []
        for y, m in _month_range(start_date, end_date):
            chunk_start = date(y, m, 1)
            chunk_end = _last_day(y, m)
            # Clamp to actual range
            if chunk_start < start_date:
                chunk_start = start_date
            if chunk_end > end_date:
                chunk_end = end_date

            try:
                chunk_events = self._fetch_month(
                    session, chunk_start, chunk_end, name_to_iso3,
                )
                events.extend(chunk_events)
            except Exception as exc:
                LOG.warning("[gdacs] error fetching %04d-%02d: %s", y, m, exc)

            if delay > 0:
                time.sleep(delay)

        return events

    def _fetch_month(
        self,
        session: requests.Session,
        from_date: date,
        to_date: date,
        name_to_iso3: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Fetch a single month chunk from the GDACS RSS archive."""
        params = {
            "profile": "ARCHIVE",
            "fromarchive": "true",
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
        }
        resp = session.get(_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        return self._parse_rss(resp.content, name_to_iso3)

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

        # Population exposed
        pop_value = _attr(item, "gdacs:population", "value")
        population = 0.0
        if pop_value:
            try:
                population = float(pop_value)
            except (ValueError, TypeError):
                pass

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
            "iso3": iso3,
            "country": country,
            "fromdate": fromdate,
            "todate": todate,
            "alertlevel": alertlevel,
            "alertscore": alertscore,
            "pub_date": pub_date,
        }

    # -----------------------------------------------------------------------
    # Deduplication
    # -----------------------------------------------------------------------

    def _deduplicate(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep only the latest episode per (eventtype, eventid)."""
        best: dict[tuple[str, str], dict[str, Any]] = {}
        for ev in events:
            key = (ev["eventtype"], ev["eventid"])
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
        for ev in events:
            iso3_list = self._resolve_countries(ev, name_to_iso3)
            if not iso3_list:
                LOG.warning(
                    "[gdacs] cannot resolve country for event %s/%s (country=%r)",
                    ev["eventtype"], ev["eventid"], ev.get("country"),
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
        return rows

    def _resolve_countries(
        self,
        event: dict[str, Any],
        name_to_iso3: dict[str, str],
    ) -> list[str]:
        """Resolve an event to a list of ISO3 codes."""
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
        """Map aggregated rows to the 21-column canonical format."""
        now_utc = datetime.now(timezone.utc).isoformat()

        records: list[dict[str, str]] = []
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
                "publication_date": pub_date.isoformat() if isinstance(pub_date, date) else str(pub_date),
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

        return pd.DataFrame(records, columns=CANONICAL_COLUMNS)
