# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ICG CrisisWatch connector — monthly conflict monitoring data.

Fetches the latest ICG CrisisWatch data via Gemini grounding search:
  1. "On the Horizon" flags (conflict risks + resolution opportunities)
  2. Global Overview arrows (deteriorated/improved/unchanged for ~70 countries)

Since crisisgroup.org is behind Cloudflare (returns 403 to programmatic
fetches), the Gemini queries avoid ``site:`` operators and rely on Google's
cached snippets instead.

Fallback: if Gemini returns no data, loads from a local JSON cache file
at ``horizon_scanner/data/crisiswatch_latest.json``.

Called ONCE per HS run (not per-country). Results are cached in-memory
for the run duration and persisted to the ``crisiswatch_entries`` DuckDB
table for cross-run access (triage, SPD).

Usage::

    data = fetch_crisiswatch()
    if data:
        entry = data.get("SOM")
        if entry:
            print(entry["arrow"], entry["summary"])
"""

from __future__ import annotations

import calendar
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

_CURRENT_DIR = Path(__file__).parent
_DATA_DIR = _CURRENT_DIR / "data"
_FALLBACK_PATH = _DATA_DIR / "crisiswatch_latest.json"

# ---------------------------------------------------------------------------
# In-memory cache (per HS run)
# ---------------------------------------------------------------------------
_CACHE: Dict[str, Any] | None = None

# ---------------------------------------------------------------------------
# ICG country name -> ISO3 mapping
# ---------------------------------------------------------------------------
# ICG uses its own country naming conventions. This dict maps all known
# CrisisWatch country names to ISO3 codes. Unmatched names are logged
# as warnings so they can be added here.
_ICG_COUNTRY_ISO3: dict[str, str] = {
    "Afghanistan": "AFG",
    "Albania": "ALB",
    "Algeria": "DZA",
    "Angola": "AGO",
    "Armenia": "ARM",
    "Azerbaijan": "AZE",
    "Bahrain": "BHR",
    "Bangladesh": "BGD",
    "Belarus": "BLR",
    "Benin": "BEN",
    "Bolivia": "BOL",
    "Bosnia and Herzegovina": "BIH",
    "Bosnia-Herzegovina": "BIH",
    "Brazil": "BRA",
    "Burkina Faso": "BFA",
    "Burundi": "BDI",
    "Cambodia": "KHM",
    "Cameroon": "CMR",
    "Central African Republic": "CAF",
    "Chad": "TCD",
    "Chile": "CHL",
    "China": "CHN",
    "Colombia": "COL",
    "Comoros": "COM",
    "Congo": "COG",
    "Costa Rica": "CRI",
    "Cote d'Ivoire": "CIV",
    "Croatia": "HRV",
    "Cuba": "CUB",
    "Cyprus": "CYP",
    "Democratic Republic of Congo": "COD",
    "Democratic Republic of the Congo": "COD",
    "DRC": "COD",
    "Djibouti": "DJI",
    "Dominican Republic": "DOM",
    "Ecuador": "ECU",
    "Egypt": "EGY",
    "El Salvador": "SLV",
    "Equatorial Guinea": "GNQ",
    "Eritrea": "ERI",
    "Eswatini": "SWZ",
    "Ethiopia": "ETH",
    "Fiji": "FJI",
    "Gabon": "GAB",
    "Gambia": "GMB",
    "Georgia": "GEO",
    "Ghana": "GHA",
    "Guatemala": "GTM",
    "Guinea": "GIN",
    "Guinea Conakry": "GIN",
    "Guinea-Bissau": "GNB",
    "Guyana": "GUY",
    "Haiti": "HTI",
    "Honduras": "HND",
    "India": "IND",
    "Indonesia": "IDN",
    "Iran": "IRN",
    "Iraq": "IRQ",
    "Israel": "ISR",
    "Israel/Palestine": "ISR",
    "Israel-Palestine": "ISR",
    "Jordan": "JOR",
    "Kazakhstan": "KAZ",
    "Kenya": "KEN",
    "Kosovo": "XKX",
    "Kyrgyzstan": "KGZ",
    "Laos": "LAO",
    "Lebanon": "LBN",
    "Lesotho": "LSO",
    "Liberia": "LBR",
    "Libya": "LBY",
    "Madagascar": "MDG",
    "Malawi": "MWI",
    "Malaysia": "MYS",
    "Maldives": "MDV",
    "Mali": "MLI",
    "Mauritania": "MRT",
    "Mexico": "MEX",
    "Moldova": "MDA",
    "Mongolia": "MNG",
    "Morocco": "MAR",
    "Morocco/Western Sahara": "MAR",
    "Mozambique": "MOZ",
    "Myanmar": "MMR",
    "Nagorno-Karabakh": "AZE",
    "Namibia": "NAM",
    "Nepal": "NPL",
    "Nicaragua": "NIC",
    "Niger": "NER",
    "Nigeria": "NGA",
    "North Korea": "PRK",
    "North Macedonia": "MKD",
    "Pakistan": "PAK",
    "Palestine": "PSE",
    "Palestinian Territories": "PSE",
    "Panama": "PAN",
    "Papua New Guinea": "PNG",
    "Peru": "PER",
    "Philippines": "PHL",
    "Poland": "POL",
    "Republic of Congo": "COG",
    "Russia": "RUS",
    "Rwanda": "RWA",
    "Saudi Arabia": "SAU",
    "Senegal": "SEN",
    "Serbia": "SRB",
    "Sierra Leone": "SLE",
    "Solomon Islands": "SLB",
    "Somalia": "SOM",
    "Somaliland": "SOM",
    "South Africa": "ZAF",
    "South Korea": "KOR",
    "South Sudan": "SSD",
    "Sri Lanka": "LKA",
    "Sudan": "SDN",
    "Suriname": "SUR",
    "Syria": "SYR",
    "Taiwan": "TWN",
    "Taiwan Strait": "TWN",
    "Tajikistan": "TJK",
    "Tanzania": "TZA",
    "Thailand": "THA",
    "Timor-Leste": "TLS",
    "Togo": "TGO",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "Turkmenistan": "TKM",
    "Uganda": "UGA",
    "Ukraine": "UKR",
    "United Arab Emirates": "ARE",
    "Uzbekistan": "UZB",
    "Venezuela": "VEN",
    "Vietnam": "VNM",
    "Western Sahara": "ESH",
    "Yemen": "YEM",
    "Zambia": "ZMB",
    "Zimbabwe": "ZWE",
}

# Case-insensitive lookup version (built once).
_ICG_LOOKUP: dict[str, str] = {k.upper(): v for k, v in _ICG_COUNTRY_ISO3.items()}


def _resolve_iso3(country_name: str) -> str | None:
    """Resolve an ICG country name to ISO3. Returns None if unknown."""
    if not country_name:
        return None
    key = country_name.strip().upper()
    iso3 = _ICG_LOOKUP.get(key)
    if iso3:
        return iso3
    # Try substring matches for composite names like "Israel/Palestine"
    for k, v in _ICG_LOOKUP.items():
        if key in k or k in key:
            return v
    return None


# ---------------------------------------------------------------------------
# Gemini grounding fetch
# ---------------------------------------------------------------------------


def _fetch_on_the_horizon(
    year: int, month_name: str
) -> list[dict[str, Any]]:
    """Gemini call #1: fetch ICG "On the Horizon" flags."""
    try:
        from pythia.web_research.backends.gemini_grounding import fetch_via_gemini
    except ImportError:
        log.debug("Gemini grounding unavailable — skipping On the Horizon fetch.")
        return []

    query = f'ICG CrisisWatch "on the horizon" conflict risks {month_name} {year} countries'
    custom_prompt = f"""\
You are a research assistant. Search for the latest ICG CrisisWatch \
"On the Horizon" section for {month_name} {year}.

"On the Horizon" is a monthly feature by the International Crisis Group \
that highlights ~3 CONFLICT RISKS and ~1 RESOLUTION OPPORTUNITY expected \
to emerge or escalate in the next 3-6 months.

Find the most recent "On the Horizon" and extract:
1. Each country or situation flagged
2. Whether it is flagged as "Conflict Risk" or "Resolution Opportunity"
3. A brief (1-2 sentence) summary of why it was flagged

Return a JSON object with this structure:
{{
  "month": "Month Year",
  "conflict_risks": [
    {{"country": "...", "summary": "..."}}
  ],
  "resolution_opportunities": [
    {{"country": "...", "summary": "..."}}
  ]
}}

If you cannot find any "On the Horizon" content, return: \
{{"month": "unknown", "conflict_risks": [], "resolution_opportunities": []}}
"""

    try:
        evidence = fetch_via_gemini(
            query=query,
            recency_days=45,
            include_structural=False,
            timeout_sec=30,
            max_results=5,
            custom_prompt=custom_prompt,
        )
        if hasattr(evidence, "to_dict"):
            pack = evidence.to_dict()
        else:
            pack = dict(evidence)

        raw_text = pack.get("markdown") or pack.get("raw_text") or ""
        log.info(
            "CrisisWatch On-the-Horizon: raw response length=%d",
            len(raw_text),
        )
        return _parse_horizon_response(raw_text)
    except Exception as exc:
        log.warning("On the Horizon Gemini fetch failed: %s", exc)
        return []


def _fetch_global_overview(
    year: int, month_name: str
) -> list[dict[str, Any]]:
    """Gemini call #2: fetch ICG CrisisWatch Global Overview arrows."""
    try:
        from pythia.web_research.backends.gemini_grounding import fetch_via_gemini
    except ImportError:
        log.debug("Gemini grounding unavailable — skipping Global Overview fetch.")
        return []

    query = (
        f"ICG CrisisWatch {month_name} {year} "
        f"deteriorated improved countries global overview"
    )
    custom_prompt = f"""\
You are a research assistant. Search for the latest ICG CrisisWatch \
Global Overview for {month_name} {year} (or the most recent month available).

The ICG CrisisWatch Global Overview categorizes ~70 countries into three \
groups each month:
- DETERIORATED SITUATIONS (arrow down) — conflict/crisis worsened
- IMPROVED SITUATIONS (arrow up) — conflict/crisis improved
- UNCHANGED SITUATIONS — no significant change

For each country mentioned, extract:
1. The country name
2. The arrow direction: "deteriorated", "improved", or "unchanged"
3. A brief (1-2 sentence) summary of the key development

Return a JSON object:
{{
  "month": "Month Year",
  "countries": [
    {{"country": "...", "arrow": "deteriorated|improved|unchanged", "summary": "..."}}
  ]
}}

Include ALL countries you can find, not just deteriorated ones. \
If you cannot find the Global Overview, return: \
{{"month": "unknown", "countries": []}}
"""

    try:
        evidence = fetch_via_gemini(
            query=query,
            recency_days=45,
            include_structural=False,
            timeout_sec=30,
            max_results=5,
            custom_prompt=custom_prompt,
        )
        if hasattr(evidence, "to_dict"):
            pack = evidence.to_dict()
        else:
            pack = dict(evidence)

        raw_text = pack.get("markdown") or pack.get("raw_text") or ""
        log.info(
            "CrisisWatch Global Overview: raw response length=%d",
            len(raw_text),
        )
        return _parse_overview_response(raw_text)
    except Exception as exc:
        log.warning("Global Overview Gemini fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------


def _parse_horizon_response(text: str) -> list[dict[str, Any]]:
    """Parse the Gemini response to extract On-the-Horizon flags."""
    results: list[dict[str, Any]] = []
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        return results
    try:
        data = json.loads(json_match.group())
        for entry in data.get("conflict_risks", []):
            country = (entry.get("country") or "").strip()
            if country:
                results.append({
                    "country": country,
                    "alert_type": "conflict_risk",
                    "summary": (entry.get("summary") or "").strip(),
                })
        for entry in data.get("resolution_opportunities", []):
            country = (entry.get("country") or "").strip()
            if country:
                results.append({
                    "country": country,
                    "alert_type": "resolution_opportunity",
                    "summary": (entry.get("summary") or "").strip(),
                })
    except json.JSONDecodeError:
        log.warning("CrisisWatch: failed to parse On-the-Horizon JSON")
    return results


def _parse_overview_response(text: str) -> list[dict[str, Any]]:
    """Parse the Gemini response to extract Global Overview arrows."""
    results: list[dict[str, Any]] = []
    json_match = re.search(r"\{[\s\S]*\}", text)
    if not json_match:
        return results
    try:
        data = json.loads(json_match.group())
        for entry in data.get("countries", []):
            country = (entry.get("country") or "").strip()
            arrow = (entry.get("arrow") or "").strip().lower()
            if country and arrow in ("deteriorated", "improved", "unchanged"):
                results.append({
                    "country": country,
                    "arrow": arrow,
                    "summary": (entry.get("summary") or "").strip(),
                })
    except json.JSONDecodeError:
        log.warning("CrisisWatch: failed to parse Global Overview JSON")
    return results


# ---------------------------------------------------------------------------
# Fallback JSON cache
# ---------------------------------------------------------------------------


def _load_fallback_json() -> Dict[str, Any] | None:
    """Load CrisisWatch data from local fallback JSON file."""
    if not _FALLBACK_PATH.exists():
        log.debug("CrisisWatch fallback file not found at %s", _FALLBACK_PATH)
        return None
    try:
        data = json.loads(_FALLBACK_PATH.read_text(encoding="utf-8"))
        entries = data.get("entries", [])
        if not entries:
            log.debug("CrisisWatch fallback file is empty or has no entries.")
            return None
        log.info(
            "CrisisWatch: loaded %d entries from fallback JSON (month=%s)",
            len(entries), data.get("month", "unknown"),
        )
        result: Dict[str, Any] = {}
        for entry in entries:
            iso3 = (entry.get("iso3") or "").strip().upper()
            if not iso3:
                country = entry.get("country", "")
                iso3 = _resolve_iso3(country) or ""
            if iso3:
                result[iso3] = {
                    "country": entry.get("country", ""),
                    "iso3": iso3,
                    "arrow": entry.get("arrow", ""),
                    "alert_type": entry.get("alert_type", ""),
                    "summary": entry.get("summary", ""),
                    "month": data.get("month", ""),
                    "year": data.get("year", 0),
                }
        return result if result else None
    except Exception as exc:
        log.warning("CrisisWatch fallback load failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# DuckDB persistence
# ---------------------------------------------------------------------------


def store_crisiswatch_entries(entries: Dict[str, Any]) -> None:
    """Write CrisisWatch entries to the crisiswatch_entries DuckDB table."""
    if not entries:
        return
    try:
        from pythia.db.schema import connect, ensure_schema
    except ImportError:
        log.debug("Pythia DB helpers unavailable — skipping CrisisWatch store.")
        return

    try:
        con = connect(read_only=False)
    except Exception as exc:
        log.warning("Could not connect to DuckDB for CrisisWatch store: %s", exc)
        return

    now = datetime.now(timezone.utc).isoformat()
    stored = 0
    try:
        ensure_schema(con)
        for iso3, entry in entries.items():
            month_str = entry.get("month", "")
            year = entry.get("year", 0)
            # Parse month number from month name string like "March 2026"
            month_num = _month_num_from_str(month_str)
            if not month_num and not year:
                continue
            con.execute(
                """
                INSERT OR REPLACE INTO crisiswatch_entries
                    (iso3, month, year, arrow, alert_type, summary,
                     country_name, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    iso3.upper(),
                    month_num or 0,
                    year or 0,
                    entry.get("arrow", ""),
                    entry.get("alert_type", ""),
                    entry.get("summary", ""),
                    entry.get("country", ""),
                    now,
                ],
            )
            stored += 1
        log.info("CrisisWatch: stored %d entries in DuckDB", stored)
    except Exception as exc:
        log.warning("CrisisWatch DuckDB store failed: %s", exc)
    finally:
        try:
            con.close()
        except Exception:
            pass


def load_crisiswatch_for_country(iso3: str) -> Dict[str, Any] | None:
    """Load the most recent CrisisWatch entry for a country from DuckDB."""
    try:
        from pythia.db.schema import connect
    except ImportError:
        return None

    try:
        con = connect(read_only=True)
    except Exception:
        return None

    try:
        row = con.execute(
            """
            SELECT iso3, month, year, arrow, alert_type, summary,
                   country_name, fetched_at
            FROM crisiswatch_entries
            WHERE iso3 = ?
            ORDER BY year DESC, month DESC
            LIMIT 1
            """,
            [iso3.upper()],
        ).fetchone()
        if not row:
            return None
        month_num = row[1]
        year = row[2]
        month_name = calendar.month_name[month_num] if 1 <= month_num <= 12 else ""
        return {
            "country": row[6] or "",
            "iso3": row[0],
            "arrow": row[3] or "",
            "alert_type": row[4] or "",
            "summary": row[5] or "",
            "month": f"{month_name} {year}" if month_name else "",
            "year": year,
        }
    except Exception as exc:
        log.debug("CrisisWatch DB load failed for %s: %s", iso3, exc)
        return None
    finally:
        try:
            con.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def format_crisiswatch_for_prompt(
    iso3: str, crisiswatch_data: Dict[str, Any] | None = None
) -> str | None:
    """Format CrisisWatch entry for injection into RC/triage/SPD prompts.

    If *crisiswatch_data* (keyed by ISO3) is provided, uses it directly.
    Otherwise loads from DuckDB.
    """
    entry = None
    if crisiswatch_data:
        entry = crisiswatch_data.get(iso3.upper())
    if entry is None:
        entry = load_crisiswatch_for_country(iso3)
    if not entry:
        return None

    arrow = (entry.get("arrow") or "").strip()
    alert_type = (entry.get("alert_type") or "").strip()
    summary = (entry.get("summary") or "").strip()
    country = entry.get("country") or iso3
    month = entry.get("month") or ""

    if not arrow and not alert_type and not summary:
        return None

    parts = [f'ICG CRISISWATCH — {country} ({month}):']

    if arrow:
        arrow_label = arrow.capitalize()
        parts.append(f"Arrow: {arrow_label}")

    if alert_type:
        if alert_type == "conflict_risk":
            parts.append(
                'Alert: CONFLICT RISK — ICG has flagged this country as an '
                '"On the Horizon" conflict risk for the coming months. '
                "This is a strong expert signal."
            )
        elif alert_type == "resolution_opportunity":
            parts.append(
                'Alert: RESOLUTION OPPORTUNITY — ICG has flagged this '
                'country as an "On the Horizon" resolution opportunity.'
            )

    if summary:
        parts.append(f"Context: {summary}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Backwards-compat wrapper
# ---------------------------------------------------------------------------


def get_horizon_countries(
    crisiswatch_data: Dict[str, Any] | None,
) -> dict[str, str]:
    """Extract {COUNTRY_NAME.upper(): formatted_note} for flagged countries.

    Backwards-compatible wrapper matching the old ``crisiswatch_horizon.py``
    API. Only includes entries with a non-empty ``alert_type`` (i.e., countries
    that appear in "On the Horizon").
    """
    if not crisiswatch_data:
        return {}
    result: dict[str, str] = {}
    for iso3, entry in crisiswatch_data.items():
        alert_type = (entry.get("alert_type") or "").strip()
        if not alert_type:
            continue
        name = (entry.get("country") or "").strip()
        if not name:
            continue
        risk_label = (
            "Conflict Risk" if alert_type == "conflict_risk"
            else "Resolution Opportunity"
        )
        summary = entry.get("summary", "")
        note = (
            f'ICG "ON THE HORIZON" FLAG: International Crisis Group has '
            f"flagged {name} as a {risk_label} in their latest monthly "
            f'"On the Horizon" assessment. This is a strong expert signal — '
            f"ICG is very selective about what they flag here.\n"
            f"Context: {summary}"
        )
        result[name.upper()] = note
    return result


# ---------------------------------------------------------------------------
# Main fetch entry point
# ---------------------------------------------------------------------------


def fetch_crisiswatch(
    year: int | None = None,
    month: int | None = None,
) -> Dict[str, Any] | None:
    """Fetch CrisisWatch data via Gemini grounding, with fallback to JSON.

    Returns a dict keyed by ISO3 (uppercased), where each value is a
    CrisisWatch entry dict with keys: country, iso3, arrow, alert_type,
    summary, month, year.

    Called ONCE per HS run; result is cached in-memory.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    if year is None:
        year = datetime.now().year
    if month is None:
        month = datetime.now().month

    month_name = calendar.month_name[month]

    # --- Phase 1: Two Gemini grounding calls ---
    horizon_entries = _fetch_on_the_horizon(year, month_name)
    overview_entries = _fetch_global_overview(year, month_name)

    log.info(
        "CrisisWatch fetch: horizon=%d entries, overview=%d entries",
        len(horizon_entries), len(overview_entries),
    )

    # Merge results. Overview entries are the base; horizon entries add
    # alert_type flags on top.
    merged: Dict[str, Any] = {}
    unmatched: list[str] = []

    # First, process overview entries (arrow + summary for all countries).
    for entry in overview_entries:
        country = entry["country"]
        iso3 = _resolve_iso3(country)
        if not iso3:
            unmatched.append(country)
            continue
        merged[iso3] = {
            "country": country,
            "iso3": iso3,
            "arrow": entry.get("arrow", ""),
            "alert_type": "",
            "summary": entry.get("summary", ""),
            "month": f"{month_name} {year}",
            "year": year,
        }

    # Then overlay horizon entries (add alert_type; update summary if richer).
    for entry in horizon_entries:
        country = entry["country"]
        iso3 = _resolve_iso3(country)
        if not iso3:
            unmatched.append(country)
            continue
        if iso3 in merged:
            merged[iso3]["alert_type"] = entry.get("alert_type", "")
            # Keep horizon summary if the overview one is empty.
            horizon_summary = entry.get("summary", "")
            if horizon_summary and not merged[iso3]["summary"]:
                merged[iso3]["summary"] = horizon_summary
        else:
            merged[iso3] = {
                "country": country,
                "iso3": iso3,
                "arrow": "",
                "alert_type": entry.get("alert_type", ""),
                "summary": entry.get("summary", ""),
                "month": f"{month_name} {year}",
                "year": year,
            }

    if unmatched:
        log.warning(
            "CrisisWatch: %d unmatched country names: %s",
            len(unmatched), sorted(set(unmatched)),
        )

    # --- Phase 2: Fallback to JSON if Gemini returned nothing ---
    if not merged:
        log.info("CrisisWatch: Gemini returned no data, trying fallback JSON.")
        merged = _load_fallback_json() or {}

    if merged:
        log.info(
            "CrisisWatch: %d countries resolved. "
            "Deteriorated=%d, Improved=%d, Unchanged=%d, Alerts=%d",
            len(merged),
            sum(1 for e in merged.values() if e.get("arrow") == "deteriorated"),
            sum(1 for e in merged.values() if e.get("arrow") == "improved"),
            sum(1 for e in merged.values() if e.get("arrow") == "unchanged"),
            sum(1 for e in merged.values() if e.get("alert_type")),
        )
        # Persist to DuckDB for triage/SPD phases.
        store_crisiswatch_entries(merged)
    else:
        log.warning("CrisisWatch: no data available from any source.")

    _CACHE = merged if merged else None
    return _CACHE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _month_num_from_str(month_str: str) -> int:
    """Extract month number from a string like 'March 2026'. Returns 0 on failure."""
    if not month_str:
        return 0
    for i, name in enumerate(calendar.month_name):
        if name and name.lower() in month_str.lower():
            return i
    return 0


def clear_cache() -> None:
    """Clear the in-memory cache (for testing or new HS run)."""
    global _CACHE
    _CACHE = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Standalone CLI for testing/debugging the CrisisWatch connector."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Fetch ICG CrisisWatch data via Gemini grounding and display results.",
    )
    parser.add_argument("--year", type=int, default=None, help="CrisisWatch edition year (default: current)")
    parser.add_argument("--month", type=int, default=None, help="CrisisWatch edition month (default: current)")
    parser.add_argument("--output", type=str, default=None, help="Write full result dict as JSON to file")
    parser.add_argument("--prompt-context", type=str, default=None, metavar="ISO3",
                        help="Print formatted prompt text for a specific country (e.g. AFG)")
    parser.add_argument("--store", action="store_true", help="Persist entries to DuckDB")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Fetch ---
    data = fetch_crisiswatch(year=args.year, month=args.month)

    if data is None:
        print(
            "\n[ERROR] CrisisWatch returned no data.\n"
            "Possible causes:\n"
            "  1. GEMINI_API_KEY / GOOGLE_API_KEY not set or invalid\n"
            "  2. Both Gemini grounding calls returned empty/unparseable results\n"
            "  3. Fallback JSON at horizon_scanner/data/crisiswatch_latest.json is empty\n"
            "\nRe-run with --verbose to see detailed Gemini request/response logging.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Summary to stderr ---
    arrows = [e.get("arrow", "") for e in data.values()]
    n_deteriorated = sum(1 for a in arrows if a == "deteriorated")
    n_improved = sum(1 for a in arrows if a == "improved")
    n_unchanged = sum(1 for a in arrows if a == "unchanged")
    n_no_arrow = sum(1 for a in arrows if not a)
    n_alerts = sum(1 for e in data.values() if e.get("alert_type"))
    print(
        f"\n=== CrisisWatch Summary ===\n"
        f"  Total countries : {len(data)}\n"
        f"  Deteriorated    : {n_deteriorated}\n"
        f"  Improved        : {n_improved}\n"
        f"  Unchanged       : {n_unchanged}\n"
        f"  No arrow        : {n_no_arrow}\n"
        f"  Alerts          : {n_alerts}\n",
        file=sys.stderr,
    )

    # --- Prompt context for a single country ---
    if args.prompt_context:
        iso3 = args.prompt_context.upper()
        text = format_crisiswatch_for_prompt(iso3, crisiswatch_data=data)
        if text:
            print(text)
        else:
            print(f"[WARN] No CrisisWatch entry for {iso3}.", file=sys.stderr)
            print(f"Available ISO3 codes: {', '.join(sorted(data.keys()))}", file=sys.stderr)
            sys.exit(1)
        return

    # --- JSON output ---
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"Wrote {len(data)} entries to {out_path}", file=sys.stderr)

    # --- Store to DuckDB ---
    if args.store:
        entries_list = list(data.values())
        store_crisiswatch_entries(entries_list)
        print(f"Stored {len(entries_list)} entries to crisiswatch_entries table.", file=sys.stderr)

    # --- Default: compact table to stdout ---
    if not args.output and not args.store:
        fmt = "{:<6s} {:<14s} {:<22s} {}"
        print(fmt.format("ISO3", "ARROW", "ALERT", "SUMMARY"))
        print(fmt.format("----", "-----", "-----", "-------"))
        for iso3 in sorted(data.keys()):
            e = data[iso3]
            arrow = e.get("arrow") or "-"
            alert = e.get("alert_type") or "-"
            summary = (e.get("summary") or "")[:80]
            print(fmt.format(iso3, arrow, alert, summary))


if __name__ == "__main__":
    main()
