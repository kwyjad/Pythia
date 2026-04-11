# Pythia / Copyright (c) 2025 Kevin Wyjad
"""GDELT conflict indicators connector.

Fetches GDELT 1.0 daily event exports (no auth, no cost) and computes
per-country conflict-intensity indicators based on CAMEO event codes.

The computed indicators are:

* Total events / day
* Material conflict share (QuadClass == 4)
* Verbal conflict share (QuadClass == 3)
* Tier 1 direct violence (CAMEO root 18, 19, 20)
* Tier 2 escalatory posture (CAMEO root 15, 17)
* Tier 3 political deterioration (CAMEO root 14, 16)
* Average Goldstein score (cooperation/conflict -10..+10)
* Average tone of conflict articles

These are stored as daily per-country rows in ``gdelt_conflict_indicators``
and later aggregated into rolling 7/30/60 day windows at prompt build time.

GDELT indicators are media-derived: they measure *reporting intensity*, not
ground-truth events.  They are used as a sentiment/attention thermometer
alongside ACLED ground-truth data in ACE prompts.

Data source: http://data.gdeltproject.org/events/{YYYYMMDD}.export.CSV.zip
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import zipfile
from collections import Counter
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, Optional

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GDELT_BASE_URL = "http://data.gdeltproject.org/events"
GDELT_REQUEST_TIMEOUT = 60
GDELT_USER_AGENT = "Pythia-GDELT-Connector/1.0 (+https://github.com/kwyjad/Pythia)"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# CAMEO event code tiers
# ---------------------------------------------------------------------------

# CAMEO EventRootCode strings (2-char zero-padded) for each tier.
# See https://www.gdeltproject.org/data/lookups/CAMEO.eventcodes.txt
TIER_1_CODES: frozenset[str] = frozenset({"18", "19", "20"})
TIER_2_CODES: frozenset[str] = frozenset({"15", "17"})
TIER_3_CODES: frozenset[str] = frozenset({"14", "16"})

ALL_CONFLICT_ROOT_CODES: frozenset[str] = TIER_1_CODES | TIER_2_CODES | TIER_3_CODES


# ---------------------------------------------------------------------------
# GDELT 1.0 column schema
# ---------------------------------------------------------------------------

# GDELT 1.0 event export has 58 unnamed tab-delimited columns.
# See http://data.gdeltproject.org/documentation/GDELT-Data_Format_Codebook.pdf
_GDELT_COLUMNS: list[str] = [
    "GLOBALEVENTID",        # 0
    "SQLDATE",              # 1
    "MonthYear",            # 2
    "Year",                 # 3
    "FractionDate",         # 4
    "Actor1Code",           # 5
    "Actor1Name",           # 6
    "Actor1CountryCode",    # 7
    "Actor1KnownGroupCode", # 8
    "Actor1EthnicCode",     # 9
    "Actor1Religion1Code",  # 10
    "Actor1Religion2Code",  # 11
    "Actor1Type1Code",      # 12
    "Actor1Type2Code",      # 13
    "Actor1Type3Code",      # 14
    "Actor2Code",           # 15
    "Actor2Name",           # 16
    "Actor2CountryCode",    # 17
    "Actor2KnownGroupCode", # 18
    "Actor2EthnicCode",     # 19
    "Actor2Religion1Code",  # 20
    "Actor2Religion2Code",  # 21
    "Actor2Type1Code",      # 22
    "Actor2Type2Code",      # 23
    "Actor2Type3Code",      # 24
    "IsRootEvent",          # 25
    "EventCode",            # 26
    "EventBaseCode",        # 27
    "EventRootCode",        # 28
    "QuadClass",            # 29
    "GoldsteinScale",       # 30
    "NumMentions",          # 31
    "NumSources",           # 32
    "NumArticles",          # 33
    "AvgTone",              # 34
    "Actor1Geo_Type",       # 35
    "Actor1Geo_FullName",   # 36
    "Actor1Geo_CountryCode",# 37
    "Actor1Geo_ADM1Code",   # 38
    "Actor1Geo_Lat",        # 39
    "Actor1Geo_Long",       # 40
    "Actor1Geo_FeatureID",  # 41
    "Actor2Geo_Type",       # 42
    "Actor2Geo_FullName",   # 43
    "Actor2Geo_CountryCode",# 44
    "Actor2Geo_ADM1Code",   # 45
    "Actor2Geo_Lat",        # 46
    "Actor2Geo_Long",       # 47
    "Actor2Geo_FeatureID",  # 48
    "ActionGeo_Type",       # 49
    "ActionGeo_FullName",   # 50
    "ActionGeo_CountryCode",# 51
    "ActionGeo_ADM1Code",   # 52
    "ActionGeo_Lat",        # 53
    "ActionGeo_Long",       # 54
    "ActionGeo_FeatureID",  # 55
    "DATEADDED",            # 56
    "SOURCEURL",            # 57
]


# ---------------------------------------------------------------------------
# FIPS-to-ISO3 mapping
# ---------------------------------------------------------------------------

# GDELT uses FIPS 10-4 country codes in ActionGeo_CountryCode.  Pythia uses
# ISO-3166-1 alpha-3.  This static mapping covers all GDELT-active countries.
# Reference: https://www.gdeltproject.org/data/lookups/FIPS.country.txt
# Only countries that Pythia forecasts are strictly required; additional
# entries are harmless.
_FIPS_TO_ISO3: Dict[str, str] = {
    "AF": "AFG", "AL": "ALB", "AG": "DZA", "AN": "AND", "AO": "AGO",
    "AC": "ATG", "AR": "ARG", "AM": "ARM", "AS": "AUS", "AU": "AUT",
    "AJ": "AZE", "BF": "BHS", "BA": "BHR", "BG": "BGD", "BB": "BRB",
    "BO": "BLR", "BE": "BEL", "BH": "BLZ", "BN": "BEN", "BT": "BTN",
    "BL": "BOL", "BK": "BIH", "BC": "BWA", "BR": "BRA", "BX": "BRN",
    "BU": "BGR", "UV": "BFA", "BM": "MMR", "BY": "BDI", "CB": "KHM",
    "CM": "CMR", "CA": "CAN", "CV": "CPV", "CT": "CAF", "CD": "TCD",
    "CI": "CHL", "CH": "CHN", "CO": "COL", "CN": "COM", "CF": "COG",
    "CG": "COD", "CS": "CRI", "IV": "CIV", "HR": "HRV", "CU": "CUB",
    "CY": "CYP", "EZ": "CZE", "DA": "DNK", "DJ": "DJI", "DO": "DMA",
    "DR": "DOM", "EC": "ECU", "EG": "EGY", "ES": "SLV", "EK": "GNQ",
    "ER": "ERI", "EN": "EST", "ET": "ETH", "FJ": "FJI", "FI": "FIN",
    "FR": "FRA", "GB": "GAB", "GA": "GMB", "GG": "GEO", "GM": "DEU",
    "GH": "GHA", "GR": "GRC", "GJ": "GRD", "GT": "GTM", "GV": "GIN",
    "PU": "GNB", "GY": "GUY", "HA": "HTI", "HO": "HND", "HU": "HUN",
    "IC": "ISL", "IN": "IND", "ID": "IDN", "IR": "IRN", "IZ": "IRQ",
    "EI": "IRL", "IS": "ISR", "IT": "ITA", "JM": "JAM", "JA": "JPN",
    "JO": "JOR", "KZ": "KAZ", "KE": "KEN", "KR": "KIR", "KN": "PRK",
    "KS": "KOR", "KV": "XKX", "KU": "KWT", "KG": "KGZ", "LA": "LAO",
    "LG": "LVA", "LE": "LBN", "LT": "LSO", "LI": "LBR", "LY": "LBY",
    "LS": "LIE", "LH": "LTU", "LU": "LUX", "MK": "MKD", "MA": "MDG",
    "MI": "MWI", "MY": "MYS", "MV": "MDV", "ML": "MLI", "MT": "MLT",
    "RM": "MHL", "MR": "MRT", "MP": "MUS", "MX": "MEX", "FM": "FSM",
    "MD": "MDA", "MN": "MCO", "MG": "MNG", "MJ": "MNE", "MO": "MAR",
    "MZ": "MOZ", "WA": "NAM", "NR": "NRU", "NP": "NPL", "NL": "NLD",
    "NZ": "NZL", "NU": "NIC", "NG": "NER", "NI": "NGA", "NO": "NOR",
    "MU": "OMN", "PK": "PAK", "PS": "PLW", "PM": "PAN", "PP": "PNG",
    "PA": "PRY", "PE": "PER", "RP": "PHL", "PL": "POL", "PO": "PRT",
    "QA": "QAT", "RO": "ROU", "RS": "RUS", "RW": "RWA", "SC": "KNA",
    "ST": "LCA", "VC": "VCT", "WS": "WSM", "SM": "SMR", "TP": "STP",
    "SA": "SAU", "SG": "SEN", "RI": "SRB", "SE": "SYC", "SL": "SLE",
    "SN": "SGP", "LO": "SVK", "SI": "SVN", "BP": "SLB", "SO": "SOM",
    "SF": "ZAF", "OD": "SSD", "SP": "ESP", "CE": "LKA", "SU": "SDN",
    "NS": "SUR", "WZ": "SWZ", "SW": "SWE", "SZ": "CHE", "SY": "SYR",
    "TW": "TWN", "TI": "TJK", "TZ": "TZA", "TH": "THA", "TT": "TLS",
    "TO": "TGO", "TN": "TON", "TD": "TTO", "TS": "TUN", "TU": "TUR",
    "TX": "TKM", "TV": "TUV", "UG": "UGA", "UP": "UKR", "AE": "ARE",
    "UK": "GBR", "US": "USA", "UY": "URY", "UZ": "UZB", "NH": "VUT",
    "VE": "VEN", "VM": "VNM", "YM": "YEM", "ZA": "ZMB", "ZI": "ZWE",
}


def _load_fips_to_iso3() -> Dict[str, str]:
    """Return the FIPS -> ISO3 mapping dict."""
    return dict(_FIPS_TO_ISO3)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def fetch_gdelt_daily_events(target_date: date) -> Optional[list[dict[str, Any]]]:
    """Download and parse one day of GDELT 1.0 events.

    Returns a list of event dicts (only the fields we care about) or None on
    failure.  GDELT daily exports are ~5-10 MB zipped and contain ~100-300k
    events globally.
    """
    date_str = target_date.strftime("%Y%m%d")
    url = f"{GDELT_BASE_URL}/{date_str}.export.CSV.zip"
    headers = {"User-Agent": GDELT_USER_AGENT}

    try:
        resp = requests.get(url, headers=headers, timeout=GDELT_REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        logger.warning("[gdelt] fetch failed for %s: %s", date_str, exc)
        return None

    if resp.status_code == 404:
        logger.debug("[gdelt] no export available for %s (404)", date_str)
        return None
    if resp.status_code != 200:
        logger.warning("[gdelt] HTTP %d for %s", resp.status_code, date_str)
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            names = zf.namelist()
            if not names:
                logger.warning("[gdelt] empty zip for %s", date_str)
                return None
            with zf.open(names[0]) as fh:
                raw = fh.read().decode("utf-8", errors="replace")
    except zipfile.BadZipFile as exc:
        logger.warning("[gdelt] bad zip for %s: %s", date_str, exc)
        return None

    # Only keep the fields we need to keep memory small.
    keep_indices = {
        1: "SQLDATE",
        26: "EventCode",
        28: "EventRootCode",
        29: "QuadClass",
        30: "GoldsteinScale",
        33: "NumArticles",
        34: "AvgTone",
        51: "ActionGeo_CountryCode",
    }
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 58:
            continue
        row: dict[str, Any] = {}
        for idx, key in keep_indices.items():
            row[key] = parts[idx]
        rows.append(row)

    logger.debug("[gdelt] parsed %d events for %s", len(rows), date_str)
    return rows


# ---------------------------------------------------------------------------
# Compute indicators
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def compute_country_indicators(
    events: Iterable[dict[str, Any]],
    fips_to_iso3: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate raw events into per-country indicators.

    Returns a dict keyed by ISO3 with indicator sub-dicts.  Events whose
    ActionGeo_CountryCode does not map to a known ISO3 are silently dropped.
    """
    mapping = fips_to_iso3 or _FIPS_TO_ISO3
    by_country: Dict[str, Dict[str, Any]] = {}

    for event in events:
        fips = (event.get("ActionGeo_CountryCode") or "").strip().upper()
        if not fips:
            continue
        iso3 = mapping.get(fips)
        if not iso3:
            continue

        bucket = by_country.setdefault(
            iso3,
            {
                "total_events": 0,
                "material_conflict_events": 0,
                "verbal_conflict_events": 0,
                "tier1_events": 0,
                "tier2_events": 0,
                "tier3_events": 0,
                "_goldstein_sum": 0.0,
                "_goldstein_n": 0,
                "_tone_conflict_sum": 0.0,
                "_tone_conflict_n": 0,
                "_top_codes": Counter(),
            },
        )

        bucket["total_events"] += 1

        quad = _safe_int(event.get("QuadClass"))
        if quad == 4:
            bucket["material_conflict_events"] += 1
        elif quad == 3:
            bucket["verbal_conflict_events"] += 1

        root = (event.get("EventRootCode") or "").strip()
        if root in TIER_1_CODES:
            bucket["tier1_events"] += 1
        elif root in TIER_2_CODES:
            bucket["tier2_events"] += 1
        elif root in TIER_3_CODES:
            bucket["tier3_events"] += 1

        gold = _safe_float(event.get("GoldsteinScale"))
        if gold is not None:
            bucket["_goldstein_sum"] += gold
            bucket["_goldstein_n"] += 1

        if quad in (3, 4):
            tone = _safe_float(event.get("AvgTone"))
            if tone is not None:
                bucket["_tone_conflict_sum"] += tone
                bucket["_tone_conflict_n"] += 1

        if root in (TIER_1_CODES | TIER_2_CODES):
            code = (event.get("EventCode") or "").strip()
            if code:
                bucket["_top_codes"][code] += 1

    # Finalize: convert sums/counts into averages and drop scratch fields.
    out: Dict[str, Dict[str, Any]] = {}
    for iso3, b in by_country.items():
        avg_gold = (
            b["_goldstein_sum"] / b["_goldstein_n"]
            if b["_goldstein_n"] > 0
            else None
        )
        avg_tone = (
            b["_tone_conflict_sum"] / b["_tone_conflict_n"]
            if b["_tone_conflict_n"] > 0
            else None
        )
        top_codes = b["_top_codes"].most_common(5)
        out[iso3] = {
            "total_events": b["total_events"],
            "material_conflict_events": b["material_conflict_events"],
            "verbal_conflict_events": b["verbal_conflict_events"],
            "tier1_events": b["tier1_events"],
            "tier2_events": b["tier2_events"],
            "tier3_events": b["tier3_events"],
            "avg_goldstein": avg_gold,
            "avg_tone_conflict": avg_tone,
            "top_codes_json": json.dumps(top_codes),
        }
    return out


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _get_connection(db_url: Optional[str] = None):
    from pythia.db.schema import connect, ensure_schema

    if db_url:
        # Callers passing an explicit URL get a dedicated connection to that
        # file rather than the process-default pool.
        import duckdb

        path = db_url
        if path.startswith("duckdb:///"):
            path = path[len("duckdb:///") :]
        con = duckdb.connect(path, read_only=False)
    else:
        con = connect()
    ensure_schema(con)
    return con


def _store_day(
    con,
    event_date: date,
    indicators_by_country: Dict[str, Dict[str, Any]],
    is_test: bool = False,
) -> int:
    if not indicators_by_country:
        return 0

    rows = []
    for iso3, ind in indicators_by_country.items():
        rows.append(
            (
                iso3,
                event_date,
                int(ind.get("total_events", 0)),
                int(ind.get("material_conflict_events", 0)),
                int(ind.get("verbal_conflict_events", 0)),
                int(ind.get("tier1_events", 0)),
                int(ind.get("tier2_events", 0)),
                int(ind.get("tier3_events", 0)),
                ind.get("avg_goldstein"),
                ind.get("avg_tone_conflict"),
                ind.get("top_codes_json"),
                is_test,
            )
        )

    con.executemany(
        """
        INSERT OR REPLACE INTO gdelt_conflict_indicators (
            iso3, event_date, total_events, material_conflict_events,
            verbal_conflict_events, tier1_events, tier2_events, tier3_events,
            avg_goldstein, avg_tone_conflict, top_codes_json, is_test
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        rows,
    )
    return len(rows)


def fetch_and_store_gdelt_indicators(
    days: Optional[int] = None,
    db_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch the last N days of GDELT events and persist per-country indicators.

    Returns a summary dict with counts of days processed, rows stored, and
    skipped days.
    """
    days = days if days is not None else _env_int("GDELT_EVENTS_DAYS", 90)
    delay = _env_float("GDELT_EVENTS_REQUEST_DELAY", 1.0)

    from pythia.test_mode import is_test_mode

    is_test = is_test_mode()

    con = _get_connection(db_url)
    try:
        today = datetime.utcnow().date()
        # GDELT 1.0 daily exports are published once per day (next day). Start
        # from yesterday to avoid 404s.
        start_offset = 1
        summary = {
            "days_requested": days,
            "days_fetched": 0,
            "days_skipped": 0,
            "rows_stored": 0,
        }
        for i in range(days):
            target = today - timedelta(days=start_offset + i)
            events = fetch_gdelt_daily_events(target)
            if events is None:
                summary["days_skipped"] += 1
            else:
                indicators = compute_country_indicators(events)
                stored = _store_day(con, target, indicators, is_test=is_test)
                summary["days_fetched"] += 1
                summary["rows_stored"] += stored
            if delay > 0:
                time.sleep(delay)

        logger.info(
            "[gdelt] fetch complete: %d days fetched, %d skipped, %d rows stored",
            summary["days_fetched"],
            summary["days_skipped"],
            summary["rows_stored"],
        )
        return summary
    finally:
        try:
            con.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_gdelt_conflict_indicators(
    iso3: str,
    db_url: Optional[str] = None,
    lookback_days: int = 90,
) -> Optional[Dict[str, Any]]:
    """Load GDELT indicators for a country and compute rolling summaries.

    Returns a dict with keys used by ``format_gdelt_for_prompt``, or None if
    no data is available.
    """
    if not iso3:
        return None
    iso3 = iso3.upper()

    con = _get_connection(db_url)
    try:
        start = (datetime.utcnow().date() - timedelta(days=lookback_days))
        rows = con.execute(
            """
            SELECT event_date, total_events, material_conflict_events,
                   verbal_conflict_events, tier1_events, tier2_events,
                   tier3_events, avg_goldstein, avg_tone_conflict, top_codes_json
            FROM gdelt_conflict_indicators
            WHERE iso3 = ? AND event_date >= ?
            ORDER BY event_date ASC
            """,
            [iso3, start],
        ).fetchall()
    finally:
        try:
            con.close()
        except Exception:
            pass

    if not rows:
        return None

    # Build day records.
    days: list[dict[str, Any]] = []
    for r in rows:
        days.append(
            {
                "event_date": r[0],
                "total_events": int(r[1] or 0),
                "material_conflict_events": int(r[2] or 0),
                "verbal_conflict_events": int(r[3] or 0),
                "tier1_events": int(r[4] or 0),
                "tier2_events": int(r[5] or 0),
                "tier3_events": int(r[6] or 0),
                "avg_goldstein": r[7],
                "avg_tone_conflict": r[8],
                "top_codes_json": r[9],
            }
        )

    def _window(daylist: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
        return daylist[-n:] if len(daylist) >= 1 else []

    def _share(window: list[dict[str, Any]], key: str) -> Optional[float]:
        total = sum(d["total_events"] for d in window)
        if total == 0:
            return None
        return sum(d[key] for d in window) / total

    def _avg(window: list[dict[str, Any]], key: str) -> Optional[float]:
        vals = [d[key] for d in window if d[key] is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    w7 = _window(days, 7)
    w30 = _window(days, 30)
    w60 = _window(days, 60)

    mc_7 = _share(w7, "material_conflict_events")
    mc_30 = _share(w30, "material_conflict_events")
    mc_60 = _share(w60, "material_conflict_events")

    vi_key_sum_7 = sum(d["tier1_events"] + d["tier2_events"] for d in w7)
    vi_key_sum_30 = sum(d["tier1_events"] + d["tier2_events"] for d in w30)
    total_7 = sum(d["total_events"] for d in w7)
    total_30 = sum(d["total_events"] for d in w30)
    vi_7 = (vi_key_sum_7 / total_7) if total_7 > 0 else None
    vi_30 = (vi_key_sum_30 / total_30) if total_30 > 0 else None

    gold_7 = _avg(w7, "avg_goldstein")
    gold_30 = _avg(w30, "avg_goldstein")
    tone_7 = _avg(w7, "avg_tone_conflict")
    tone_30 = _avg(w30, "avg_tone_conflict")

    # Trend direction: compare 7d material-conflict share vs 30d baseline.
    trend = "flat"
    trend_desc = ""
    if mc_7 is not None and mc_30 is not None:
        delta = mc_7 - mc_30
        if delta > 0.05:
            trend = "rising"
            trend_desc = f"7d {mc_7*100:.1f}% vs 30d {mc_30*100:.1f}%"
        elif delta < -0.05:
            trend = "falling"
            trend_desc = f"7d {mc_7*100:.1f}% vs 30d {mc_30*100:.1f}%"
        else:
            trend_desc = f"7d {mc_7*100:.1f}% vs 30d {mc_30*100:.1f}%"

    # Top escalatory codes in the last 7 days (merge top_codes_json across days).
    merged = Counter()
    for d in w7:
        tc = d.get("top_codes_json")
        if not tc:
            continue
        try:
            for code, n in json.loads(tc):
                merged[code] += int(n)
        except Exception:
            continue
    top_codes = merged.most_common(5)

    start_date = days[0]["event_date"]
    end_date = days[-1]["event_date"]

    return {
        "iso3": iso3,
        "start_date": start_date,
        "end_date": end_date,
        "days_covered": len(days),
        "total_events": sum(d["total_events"] for d in days),
        "material_conflict_share_7d": mc_7,
        "material_conflict_share_30d": mc_30,
        "material_conflict_share_60d": mc_60,
        "violence_intensity_7d": vi_7,
        "violence_intensity_30d": vi_30,
        "avg_goldstein_7d": gold_7,
        "avg_goldstein_30d": gold_30,
        "avg_tone_conflict_7d": tone_7,
        "avg_tone_conflict_30d": tone_30,
        "trend": trend,
        "trend_description": trend_desc,
        "top_escalatory_codes": top_codes,
    }


# ---------------------------------------------------------------------------
# Format for prompt
# ---------------------------------------------------------------------------


def _fmt_pct(v: Optional[float]) -> str:
    return f"{v*100:.1f}%" if v is not None else "n/a"


def _fmt_num(v: Optional[float], fmt: str = "{:.1f}") -> str:
    return fmt.format(v) if v is not None else "n/a"


def format_gdelt_for_prompt(iso3: str, country_name: Optional[str] = None) -> str:
    """Render the GDELT indicator text block for prompt injection.

    Returns an empty string if no data is available for the country.
    """
    try:
        data = load_gdelt_conflict_indicators(iso3)
    except Exception as exc:
        logger.debug("[gdelt] load failed for %s: %s", iso3, exc)
        return ""

    if not data:
        return ""

    cname = country_name or iso3
    top_codes_text = ", ".join(f"{c}({n})" for c, n in data["top_escalatory_codes"]) or "none"

    lines = [
        f"## GDELT Media Conflict Indicators ({cname}, {data['start_date']} to {data['end_date']})",
        f"- Events monitored: {data['total_events']} ({data['days_covered']} days)",
        f"- Material Conflict share: {_fmt_pct(data['material_conflict_share_7d'])} "
        f"(30d avg: {_fmt_pct(data['material_conflict_share_30d'])}, "
        f"60d avg: {_fmt_pct(data['material_conflict_share_60d'])})",
        f"- Violence intensity (T1+T2): {_fmt_pct(data['violence_intensity_7d'])} "
        f"(30d avg: {_fmt_pct(data['violence_intensity_30d'])})",
        f"- Trend: {data['trend']}"
        + (f" ({data['trend_description']})" if data["trend_description"] else ""),
        f"- Avg Goldstein Scale: {_fmt_num(data['avg_goldstein_7d'])} "
        f"(30d avg: {_fmt_num(data['avg_goldstein_30d'])})",
        f"- Top escalatory codes (last 7d): {top_codes_text}",
        f"- Avg Tone of conflict articles: {_fmt_num(data['avg_tone_conflict_7d'])} "
        f"(30d avg: {_fmt_num(data['avg_tone_conflict_30d'])})",
        "NOTE: GDELT indicators are media-derived (not ground truth). They "
        "measure reporting intensity, not verified events. Use as a "
        "sentiment/attention thermometer alongside ACLED ground truth.",
    ]
    return "\n".join(lines)


def format_gdelt_for_rc_prompt(iso3: str, country_name: Optional[str] = None) -> str:
    """Variant of ``format_gdelt_for_prompt`` with RC-specific guidance."""
    base = format_gdelt_for_prompt(iso3, country_name=country_name)
    if not base:
        return ""
    guidance = (
        "\nRC USE: If GDELT shows a rising share of material conflict events "
        "or deteriorating Goldstein scores relative to the 30/60-day average, "
        "and this aligns with ACLED base rate trends or conflict forecast "
        "signals, this strengthens the case for higher RC likelihood."
    )
    return base + guidance


__all__ = [
    "TIER_1_CODES",
    "TIER_2_CODES",
    "TIER_3_CODES",
    "fetch_gdelt_daily_events",
    "compute_country_indicators",
    "fetch_and_store_gdelt_indicators",
    "load_gdelt_conflict_indicators",
    "format_gdelt_for_prompt",
    "format_gdelt_for_rc_prompt",
    "_load_fips_to_iso3",
]
