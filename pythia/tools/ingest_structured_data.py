# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Ingest per-country structured data into DuckDB — bulk-fetch redesign.

**Problem solved**: The v1 script looped over ~190 countries, making
~20-30 ACAPS API calls *per country* (with 1 s inter-page sleeps),
resulting in 6+ hour runtimes that exceed GitHub Actions' limit.

**Strategy**: Fetch each source *globally* (all countries in one sweep),
then partition by ISO3 and store.  Where an API doesn't support global
fetch (IPC), fall back to per-country calls with concurrency.

Expected runtime: ~5-10 minutes (down from 6+ hours).

Usage:
    python -m pythia.tools.ingest_structured_data
    python -m pythia.tools.ingest_structured_data --sources acaps reliefweb nmme
    python -m pythia.tools.ingest_structured_data --sources conflict ipc
    python -m pythia.tools.ingest_structured_data --sources views acledcast
    python -m pythia.tools.ingest_structured_data --iso3 AFG,SYR,YEM
    python -m pythia.tools.ingest_structured_data --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import requests

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

_SOURCE_GROUPS: dict[str, list[str]] = {
    "views": ["views_forecasts"],
    "conflictforecast": ["conflictforecast_forecasts"],
    "acledcast": ["acledcast_forecasts"],
    "acaps_inform_severity": ["acaps_inform_severity"],
    "acaps_risk_radar": ["acaps_risk_radar"],
    "acaps_daily_monitoring": ["acaps_daily_monitoring"],
    "acaps_humanitarian_access": ["acaps_humanitarian_access"],
    "ipc": ["ipc_phases"],
    "reliefweb": ["reliefweb_reports"],
    "acled_political": ["acled_political_events"],
    "nmme": ["nmme_seasonal_forecasts"],
    "gdacs": ["gdacs_population_exposed"],
}

# Convenience aliases that expand to multiple source groups.
_SOURCE_ALIASES: dict[str, list[str]] = {
    "acaps": [
        "acaps_inform_severity",
        "acaps_risk_radar",
        "acaps_daily_monitoring",
        "acaps_humanitarian_access",
    ],
    "conflict": ["views", "conflictforecast", "acledcast"],
}

ALL_SOURCE_NAMES = sorted(_SOURCE_GROUPS.keys())
ALL_ALIAS_NAMES = sorted(_SOURCE_ALIASES.keys())

# ---------------------------------------------------------------------------
# Country list loader
# ---------------------------------------------------------------------------

_COUNTRIES_CSV = (
    Path(__file__).resolve().parents[2] / "resolver" / "data" / "countries.csv"
)


def _load_iso3_list(override: str | None = None) -> list[str]:
    """Return a sorted list of ISO-3 country codes."""
    if override:
        return sorted(
            {c.strip().upper() for c in override.split(",") if c.strip()}
        )
    if not _COUNTRIES_CSV.exists():
        LOG.error("Countries file not found: %s", _COUNTRIES_CSV)
        sys.exit(1)
    iso3s: list[str] = []
    with open(_COUNTRIES_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            code = row.get("iso3", "").strip().upper()
            if code:
                iso3s.append(code)
    return sorted(set(iso3s))


# ===================================================================
# Shared ACAPS infrastructure (auth, pagination)
# ===================================================================

ACAPS_API_BASE = "https://api.acaps.org"
_acaps_token: str | None = None


def _get_acaps_token(force_refresh: bool = False) -> str | None:
    global _acaps_token
    if _acaps_token and not force_refresh:
        return _acaps_token
    username = os.getenv("ACAPS_USERNAME", "").strip()
    password = os.getenv("ACAPS_PASSWORD", "").strip()
    if not username or not password:
        LOG.warning("ACAPS credentials not set — skipping ACAPS sources.")
        return None
    try:
        resp = requests.post(
            f"{ACAPS_API_BASE}/api/v1/token-auth/",
            json={"username": username, "password": password},
            timeout=30,
        )
        if resp.status_code != 200:
            LOG.warning("ACAPS token auth HTTP %d", resp.status_code)
            return None
        _acaps_token = resp.json().get("token")
        return _acaps_token
    except Exception as exc:
        LOG.warning("ACAPS token request failed: %s", exc)
        return None


def _fetch_paginated_global(
    endpoint: str,
    params: dict | None = None,
    max_pages: int = 50,
    token: str | None = None,
) -> list[dict]:
    """Fetch ALL pages from an ACAPS endpoint without iso3 filter."""
    if token is None:
        token = _get_acaps_token()
    if not token:
        return []

    headers = {"Authorization": f"Token {token}"}
    url: str | None = f"{ACAPS_API_BASE}{endpoint}"
    all_results: list[dict] = []
    retried_auth = False

    for page_num in range(1, max_pages + 1):
        if url is None:
            break
        try:
            resp = requests.get(
                url,
                params=params if page_num == 1 else None,
                headers=headers,
                timeout=90,
            )
        except requests.RequestException as exc:
            LOG.warning("ACAPS API request failed: %s", exc)
            return all_results

        if resp.status_code == 401 and not retried_auth:
            retried_auth = True
            new_token = _get_acaps_token(force_refresh=True)
            if new_token:
                headers = {"Authorization": f"Token {new_token}"}
                try:
                    resp = requests.get(
                        url,
                        params=params if page_num == 1 else None,
                        headers=headers,
                        timeout=90,
                    )
                except requests.RequestException as exc:
                    LOG.warning("ACAPS retry failed: %s", exc)
                    return all_results

        if resp.status_code != 200:
            LOG.warning(
                "ACAPS HTTP %d for %s (page %d)", resp.status_code, endpoint, page_num
            )
            return all_results

        try:
            body = resp.json()
        except ValueError:
            LOG.warning("ACAPS non-JSON response for %s", endpoint)
            return all_results

        results = body.get("results") or []
        if isinstance(results, list):
            all_results.extend(results)

        url = body.get("next")
        if url and page_num < max_pages:
            time.sleep(0.5)  # lighter sleep for bulk fetch

    LOG.info(
        "ACAPS %s: fetched %d records across %d page(s)",
        endpoint, len(all_results), min(page_num, max_pages),
    )
    return all_results


# ===================================================================
# Bulk fetchers — one function per source, returns {iso3: data}
# ===================================================================


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_access_score(record: dict) -> float | None:
    """Compute overall humanitarian access score from a record.

    Tries ``overall_score``, ``access_score``, ``score`` first.  Falls
    back to computing the mean of indicator values (I1–I9, P1–P2, etc.)
    which is how ACAPS derives the overall score on their website.
    """
    direct = _safe_float(
        record.get("overall_score")
        or record.get("access_score")
        or record.get("score")
    )
    if direct is not None:
        return direct

    # Collect numeric indicator values matching I\d+ or P\d+ keys.
    indicator_vals: list[float] = []
    for key, val in record.items():
        if re.fullmatch(r"[IP]\d+", key):
            v = _safe_float(val)
            if v is not None:
                indicator_vals.append(v)

    if not indicator_vals:
        return None

    return round(sum(indicator_vals) / len(indicator_vals), 4)


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _month_labels_back(n: int) -> list[str]:
    today = date.today()
    labels: list[str] = []
    for i in range(n):
        d = today.replace(day=1) - timedelta(days=i * 30)
        labels.append(d.strftime("%b%Y"))
    return labels


def _pick_country_crisis(results: list[dict]) -> dict:
    if len(results) == 1:
        return results[0]
    for r in results:
        if r.get("country_level"):
            return r

    def _score(r: dict) -> float:
        return _safe_float(
            r.get("severity_index_score")
            or r.get("severity_score")
            or r.get("score")
        ) or 0.0

    return max(results, key=_score)


# ----- ACAPS INFORM Severity (bulk) -----

def _bulk_fetch_inform_severity(
    countries: set[str],
) -> dict[str, dict]:
    """Fetch INFORM Severity for all countries in one sweep.

    Strategy: fetch the global snapshot (no iso3 filter), then the global
    country-log, group by iso3, and build per-country dicts.
    """
    token = _get_acaps_token()
    if not token:
        return {}

    # 1. Find the latest available snapshot month
    snapshot_data: list[dict] = []
    snapshot_date: str | None = None
    for label in _month_labels_back(3):
        data = _fetch_paginated_global(
            f"/api/v1/inform-severity-index/{label}/",
            max_pages=20,
            token=token,
        )
        if data:
            snapshot_data = data
            snapshot_date = label
            LOG.info("INFORM Severity: using snapshot %s (%d records)", label, len(data))
            break

    if not snapshot_data:
        LOG.warning("INFORM Severity: no snapshot found")
        return {}

    # Group snapshot by iso3
    by_country: dict[str, list[dict]] = defaultdict(list)
    for rec in snapshot_data:
        iso3_raw = rec.get("iso3") or ""
        if isinstance(iso3_raw, list):
            iso3_raw = iso3_raw[0] if iso3_raw else ""
        iso3 = str(iso3_raw).strip().upper()
        if iso3 and iso3 in countries:
            by_country[iso3].append(rec)

    # 2. Fetch global country-log (trend data)
    log_data = _fetch_paginated_global(
        "/api/v1/inform-severity-index/country-log/",
        max_pages=30,
        token=token,
    )
    trend_by_country: dict[str, list[dict]] = defaultdict(list)
    for entry in log_data:
        iso3_raw = entry.get("iso3") or ""
        if isinstance(iso3_raw, list):
            iso3_raw = iso3_raw[0] if iso3_raw else ""
        iso3 = str(iso3_raw).strip().upper()
        entry_date = entry.get("date", "")
        entry_score = _safe_float(entry.get("value"))
        if iso3 and iso3 in countries and entry_date and entry_score is not None:
            trend_by_country[iso3].append({"date": entry_date, "score": entry_score})

    # 3. Fetch global dimension data for top indicators
    top_indicators_by_country: dict[str, list[dict]] = defaultdict(list)
    for dim_name, dim_path in [
        ("impact", "impact-of-crisis"),
        ("conditions", "conditions-of-people-affected"),
        ("complexity", "complexity"),
    ]:
        dim_data = _fetch_paginated_global(
            f"/api/v1/inform-severity-index/{dim_path}/{snapshot_date}/",
            max_pages=20,
            token=token,
        )
        for ind in dim_data:
            iso3_raw = ind.get("iso3") or ""
            if isinstance(iso3_raw, list):
                iso3_raw = iso3_raw[0] if iso3_raw else ""
            iso3 = str(iso3_raw).strip().upper()
            fig = _safe_float(ind.get("figure"))
            if iso3 and iso3 in countries and fig is not None and fig >= 4.0:
                top_indicators_by_country[iso3].append({
                    "indicator": ind.get("indicator", ""),
                    "figure": fig,
                    "dimension": dim_name,
                })

    # 4. Assemble per-country dicts
    results: dict[str, dict] = {}
    for iso3, records in by_country.items():
        snapshot = _pick_country_crisis(records)

        severity_score = _safe_float(
            snapshot.get("severity_index_score")
            or snapshot.get("severity_score")
            or snapshot.get("score")
        )
        severity_category = (
            snapshot.get("severity_index_category")
            or snapshot.get("severity_category")
            or snapshot.get("category")
            or ""
        )
        impact_score = _safe_float(
            snapshot.get("impact_score")
            or snapshot.get("impact_of_the_crisis")
        )
        conditions_score = _safe_float(
            snapshot.get("conditions_score")
            or snapshot.get("conditions_of_people_affected")
        )
        complexity_score = _safe_float(
            snapshot.get("complexity_score") or snapshot.get("complexity")
        )

        # Trend
        trend_entries = sorted(
            trend_by_country.get(iso3, []), key=lambda e: e["date"]
        )
        if len(trend_entries) > 6:
            trend_entries = trend_entries[-6:]

        delta_1m = None
        delta_3m = None
        if severity_score is not None and len(trend_entries) >= 2:
            delta_1m = round(severity_score - trend_entries[-1]["score"], 2)
        if severity_score is not None and len(trend_entries) >= 4:
            delta_3m = round(severity_score - trend_entries[-3]["score"], 2)

        # Top indicators
        indicators = top_indicators_by_country.get(iso3, [])
        indicators.sort(key=lambda x: x["figure"], reverse=True)
        indicators = indicators[:5]

        results[iso3] = {
            "iso3": iso3,
            "crisis_id": snapshot.get("crisis_id", ""),
            "crisis_name": snapshot.get("crisis", "") or snapshot.get("crisis_name", ""),
            "severity_score": severity_score,
            "severity_category": severity_category,
            "impact_score": impact_score,
            "conditions_score": conditions_score,
            "complexity_score": complexity_score,
            "snapshot_date": snapshot_date,
            "trend_6m": trend_entries,
            "delta_1m": delta_1m,
            "delta_3m": delta_3m,
            "top_indicators": indicators,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    LOG.info("INFORM Severity: built data for %d countries", len(results))
    return results


# ----- ACAPS Risk Radar (bulk) -----

def _bulk_fetch_risk_radar(
    countries: set[str],
) -> dict[str, dict]:
    """Fetch Risk Radar for all countries in one sweep."""
    token = _get_acaps_token()
    if not token:
        return {}

    all_risks = _fetch_paginated_global(
        "/api/v1/risk-radar/risk-radar/",
        params={"status": "Active"},
        max_pages=20,
        token=token,
    )

    # Fallback to risk-list if empty
    used_fallback = False
    if not all_risks:
        all_risks = _fetch_paginated_global(
            "/api/v1/risk-list/",
            params={"status": "Active"},
            max_pages=20,
            token=token,
        )
        used_fallback = bool(all_risks)

    # Group by iso3
    risks_by_country: dict[str, list[dict]] = defaultdict(list)
    risk_ids_to_fetch: list[str] = []

    for r in all_risks:
        iso3_raw = r.get("iso3") or ""
        if isinstance(iso3_raw, list):
            iso3_raw = iso3_raw[0] if iso3_raw else ""
        iso3 = str(iso3_raw).strip().upper()
        if not iso3 or iso3 not in countries:
            continue

        risk_id = str(r.get("risk_id", r.get("id", "")))
        rationale = str(r.get("rationale", "") or "")
        if len(rationale) > 500:
            rationale = rationale[:500].rsplit(" ", 1)[0] + "..."

        risk_entry: dict[str, Any] = {
            "risk_id": risk_id,
            "title": r.get("risk_title", "") or r.get("title", ""),
            "risk_type": r.get("risk_type", ""),
            "risk_level": r.get("risk_level", ""),
            "probability": r.get("probability", ""),
            "impact": _safe_int(r.get("impact")),
            "risk_trend": r.get("risk_trend", "") or r.get("trend", ""),
            "expected_exposure": r.get("expected_exposure", ""),
            "triggers_summary": r.get("triggers_summary", "")
            or r.get("triggers", ""),
            "rationale": rationale,
            "triggers": [],
        }
        risks_by_country[iso3].append(risk_entry)
        if risk_id and not used_fallback:
            risk_ids_to_fetch.append(risk_id)

    # Bulk-fetch all triggers in one sweep (no iso3 filter)
    if risk_ids_to_fetch and not used_fallback:
        all_triggers = _fetch_paginated_global(
            "/api/v1/risk-radar/trigger-list/",
            max_pages=30,
            token=token,
        )
        # Index triggers by risk_id
        triggers_by_risk: dict[str, list[dict]] = defaultdict(list)
        for t in all_triggers:
            rid = str(t.get("risk") or t.get("risk_id") or "")
            if rid:
                desc = str(t.get("description", "") or "")
                if len(desc) > 200:
                    desc = desc[:200].rsplit(" ", 1)[0] + "..."
                triggers_by_risk[rid].append({
                    "trigger_id": str(t.get("trigger_id", "")),
                    "title": t.get("title", ""),
                    "completion_rate": t.get("completion_rate", ""),
                    "trend": t.get("trend", ""),
                    "description": desc,
                })

        # Attach triggers to their risks
        for iso3, risks in risks_by_country.items():
            for risk in risks:
                rid = risk.get("risk_id", "")
                risk["triggers"] = triggers_by_risk.get(rid, [])

    # Build result dicts
    level_order = {"Low": 0, "Medium": 1, "High": 2}
    results: dict[str, dict] = {}
    for iso3, risks in risks_by_country.items():
        highest = "None"
        for risk in risks:
            rl = risk.get("risk_level", "")
            if level_order.get(rl, -1) > level_order.get(highest, -1):
                highest = rl

        results[iso3] = {
            "iso3": iso3,
            "risks": risks,
            "total_active_risks": len(risks),
            "highest_risk_level": highest if risks else "None",
            "source": "risk-list" if used_fallback else "risk-radar",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }

    LOG.info("Risk Radar: built data for %d countries", len(results))
    return results


# ----- ACAPS Daily Monitoring (bulk) -----

def _bulk_fetch_daily_monitoring(
    countries: set[str],
    days_back: int = 30,
) -> dict[str, list[dict]]:
    """Fetch Daily Monitoring for all countries in one sweep."""
    token = _get_acaps_token()
    if not token:
        return {}

    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    all_entries = _fetch_paginated_global(
        "/api/v1/daily-monitoring/",
        params={
            "_internal_filter_date_gte": start_date.isoformat(),
            "_internal_filter_date_lte": end_date.isoformat(),
        },
        max_pages=30,
        token=token,
    )

    by_country: dict[str, list[dict]] = defaultdict(list)
    for r in all_entries:
        iso3_raw = r.get("iso3") or ""
        if isinstance(iso3_raw, list):
            iso3_raw = iso3_raw[0] if iso3_raw else ""
        iso3 = str(iso3_raw).strip().upper()
        if not iso3 or iso3 not in countries:
            continue

        dev = str(r.get("latest_developments", "") or "")
        if len(dev) > 400:
            dev = dev[:400].rsplit(" ", 1)[0] + "..."

        by_country[iso3].append({
            "entry_id": str(r.get("entry_id", r.get("id", ""))),
            "date": r.get("date", ""),
            "latest_developments": dev,
            "source": r.get("source", ""),
            "weekly_pick": bool(r.get("selected_weekly_pick", False)),
        })

    # Sort and limit per country
    for iso3 in by_country:
        by_country[iso3].sort(
            key=lambda e: e.get("date", ""), reverse=True
        )
        by_country[iso3] = by_country[iso3][:20]

    LOG.info("Daily Monitoring: data for %d countries", len(by_country))
    return dict(by_country)


# ----- ACAPS Humanitarian Access (bulk) -----

def _access_category(score: float) -> str:
    if score < 1:
        return "Very Low"
    if score < 2:
        return "Low"
    if score < 3:
        return "Medium"
    if score < 4:
        return "High"
    return "Very High"


def _bulk_fetch_humanitarian_access(
    countries: set[str],
) -> dict[str, dict]:
    """Fetch Humanitarian Access for all countries.

    Tries month by month (most recent first) until we find data.
    Unlike the per-country version that tries 12 months per country,
    we fetch each month globally and accumulate.
    """
    token = _get_acaps_token()
    if not token:
        return {}

    results: dict[str, dict] = {}
    remaining = set(countries)
    _logged_ha_keys = False

    for label in _month_labels_back(12):
        if not remaining:
            break

        data = _fetch_paginated_global(
            f"/api/v1/humanitarian-access/{label}/",
            max_pages=10,
            token=token,
        )

        # Group by iso3
        by_country: dict[str, list[dict]] = defaultdict(list)
        for rec in data:
            if not _logged_ha_keys:
                LOG.info("ACAPS Humanitarian Access sample record keys: %s", list(rec.keys())[:20])
                indicator_keys = [k for k in rec if re.fullmatch(r"[IP]\d+", k)]
                indicator_vals = {k: rec[k] for k in indicator_keys}
                sample_score = _compute_access_score(rec)
                LOG.info(
                    "ACAPS Humanitarian Access sample indicators: %s, computed score: %s",
                    indicator_vals, sample_score,
                )
                _logged_ha_keys = True

            iso3_raw = (
                rec.get("iso3")
                or rec.get("iso")
                or rec.get("country_iso3")
                or rec.get("country_code")
                or rec.get("country_iso")
                or ""
            )
            if not iso3_raw and isinstance(rec.get("country"), dict):
                iso3_raw = rec["country"].get("iso3") or ""
            if isinstance(iso3_raw, list):
                iso3_raw = iso3_raw[0] if iso3_raw else ""
            iso3 = str(iso3_raw).strip().upper()
            if iso3 and iso3 in remaining:
                by_country[iso3].append(rec)

        for iso3, records in by_country.items():
            record = _pick_country_crisis(records)
            score = _compute_access_score(record)
            if score is None:
                continue

            stale = False
            try:
                snap_date = datetime.strptime(label, "%b%Y")
                months_old = (
                    (date.today().year - snap_date.year) * 12
                    + date.today().month - snap_date.month
                )
                stale = months_old > 8
            except (ValueError, TypeError):
                pass

            results[iso3] = {
                "iso3": iso3,
                "crisis_id": record.get("crisis_id", ""),
                "access_score": score,
                "access_category": _access_category(score),
                "snapshot_date": label,
                "stale": stale,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            remaining.discard(iso3)

    LOG.info("Humanitarian Access: data for %d countries", len(results))
    return results


# ----- ReliefWeb (bulk) -----

def _bulk_fetch_reliefweb(
    countries: set[str],
    days_back: int = 45,
    max_per_country: int = 15,
) -> dict[str, list[dict]]:
    """Fetch ReliefWeb reports for ALL countries in bulk API calls.

    ReliefWeb supports up to limit=1000 per request. We fetch all recent
    reports globally (no country filter), then group by country.
    """
    from html.parser import HTMLParser

    class _HTMLStripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self._parts: list[str] = []

        def handle_data(self, data: str):
            self._parts.append(data)

        def get_text(self) -> str:
            return "".join(self._parts)

    def strip_html(html: str) -> str:
        s = _HTMLStripper()
        s.feed(html)
        return s.get_text()

    since = datetime.now(timezone.utc) - timedelta(days=days_back)
    since_iso = since.strftime("%Y-%m-%dT00:00:00+00:00")
    appname = "UNICEF-Resolver-P1L1T6"

    # Fetch in pages of 1000
    all_reports: list[dict] = []
    offset = 0
    page_size = 1000
    max_pages = 10  # safety: 10,000 reports max

    for page in range(max_pages):
        payload = {
            "filter": {
                "operator": "AND",
                "conditions": [
                    {"field": "date.created", "value": {"from": since_iso}},
                ],
            },
            "fields": {
                "include": [
                    "title",
                    "date.created",
                    "source.name",
                    "primary_country.iso3",
                    "disaster_type.name",
                    "theme.name",
                    "body",
                    "url",
                ],
            },
            "sort": ["date.created:desc"],
            "limit": page_size,
            "offset": offset,
        }

        url = "https://api.reliefweb.int/v1/reports"
        headers = {"Accept": "application/json"}
        params = {"appname": appname}

        try:
            resp = requests.post(
                url, json=payload, headers=headers, params=params, timeout=60,
            )
            if resp.status_code == 429:
                LOG.warning("ReliefWeb 429 — waiting 5s")
                time.sleep(5)
                resp = requests.post(
                    url, json=payload, headers=headers, params=params, timeout=60,
                )
            resp.raise_for_status()
        except requests.RequestException as exc:
            LOG.warning("ReliefWeb bulk fetch failed (page %d): %s", page, exc)
            break

        data = resp.json().get("data", [])
        if not data:
            break

        all_reports.extend(data)
        LOG.info("ReliefWeb: fetched page %d (%d reports)", page + 1, len(data))

        if len(data) < page_size:
            break  # last page

        offset += page_size
        time.sleep(0.5)  # courtesy

    LOG.info("ReliefWeb: %d total reports fetched globally", len(all_reports))

    # Group by country
    now_iso = datetime.now(timezone.utc).isoformat()
    by_country: dict[str, list[dict]] = defaultdict(list)

    for item in all_reports:
        fields = item.get("fields", {})
        pc = fields.get("primary_country") or {}
        iso3 = ""
        if isinstance(pc, dict):
            iso3_raw = pc.get("iso3") or ""
            if isinstance(iso3_raw, list):
                iso3_raw = iso3_raw[0] if iso3_raw else ""
            iso3 = str(iso3_raw).strip().upper()
        elif isinstance(pc, list) and pc:
            iso3_raw = pc[0].get("iso3") or ""
            if isinstance(iso3_raw, list):
                iso3_raw = iso3_raw[0] if iso3_raw else ""
            iso3 = str(iso3_raw).strip().upper()

        if not iso3 or iso3 not in countries:
            continue

        raw_body = fields.get("body", "") or ""
        plain_body = strip_html(raw_body).strip()
        if len(plain_body) > 500:
            plain_body = plain_body[:500].rsplit(" ", 1)[0] + " …"

        sources = [s.get("name", "") for s in (fields.get("source") or [])]
        disaster_types = [
            d.get("name", "") for d in (fields.get("disaster_type") or [])
        ]
        themes = [t.get("name", "") for t in (fields.get("theme") or [])]

        date_obj = fields.get("date") or {}
        published = (
            date_obj.get("created", "") if isinstance(date_obj, dict) else ""
        )

        by_country[iso3].append({
            "report_id": item.get("id"),
            "iso3": iso3,
            "title": fields.get("title", ""),
            "published_date": published,
            "sources": json.dumps(sources),
            "disaster_types": json.dumps(disaster_types),
            "themes": json.dumps(themes),
            "body_excerpt": plain_body,
            "url": fields.get("url", ""),
            "fetched_at": now_iso,
        })

    # Sort and limit per country
    for iso3 in by_country:
        by_country[iso3].sort(
            key=lambda r: r.get("published_date", ""), reverse=True
        )
        by_country[iso3] = by_country[iso3][:max_per_country]

    LOG.info("ReliefWeb: data for %d countries", len(by_country))
    return dict(by_country)


# ----- IPC (per-country, but concurrent) -----

def _bulk_fetch_ipc(
    countries: set[str],
    max_workers: int = 4,
) -> dict[str, dict]:
    """Fetch IPC phases — per-country (API requires it), but concurrent."""
    api_key = os.getenv("IPC_API_KEY", "").strip()
    if not api_key:
        LOG.warning("IPC_API_KEY not set — skipping IPC ingestion.")
        return {}

    from pythia.ipc_phases import fetch_ipc_phases

    results: dict[str, dict] = {}

    def _fetch_one(iso3: str) -> tuple[str, dict | None]:
        try:
            return iso3, fetch_ipc_phases(iso3)
        except Exception as exc:
            LOG.warning("IPC fetch failed for %s: %s", iso3, exc)
            return iso3, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, c): c for c in countries}
        for future in as_completed(futures):
            iso3, data = future.result()
            if data:
                results[iso3] = data

    LOG.info("IPC: data for %d countries", len(results))
    return results


# ===================================================================
# ACLED Political — per-country via pythia.acled_political
# ===================================================================


def _bulk_fetch_acled_political(
    countries: set[str],
    max_workers: int = 4,
) -> dict[str, list[dict]]:
    """Fetch ACLED political events — per-country, concurrent."""
    from pythia.acled_political import fetch_acled_political_events

    results: dict[str, list[dict]] = {}

    def _fetch_one(iso3: str) -> tuple[str, list[dict]]:
        try:
            return iso3, fetch_acled_political_events(iso3)
        except Exception as exc:
            LOG.warning("ACLED political fetch failed for %s: %s", iso3, exc)
            return iso3, []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, c): c for c in countries}
        for future in as_completed(futures):
            iso3, data = future.result()
            if data:
                results[iso3] = data

    LOG.info("ACLED political: data for %d countries", len(results))
    return results


# ===================================================================
# NMME — delegates to resolver.tools.ingest_nmme (global, not per-country)
# ===================================================================


def _bulk_fetch_nmme(dry_run: bool = False) -> dict[str, Any]:
    """Run the NMME seasonal-forecast pipeline (fetch + store).

    NMME is a global dataset (not per-country), so we delegate entirely
    to ``resolver.tools.ingest_nmme.main`` which handles its own DB writes.
    Returns a sentinel dict so the orchestrator sees non-empty data.

    Failures are caught and logged as warnings (non-fatal) because NMME
    FTP files are published ~9th-10th of each month and may not yet be
    available.
    """
    from resolver.tools.ingest_nmme import main as nmme_main

    argv: list[str] = []
    if dry_run:
        argv.append("--dry-run")
    try:
        nmme_main(argv)
    except Exception as exc:
        LOG.warning(
            "NMME ingestion failed (FTP files may not be published yet — non-fatal): %s",
            exc,
        )
        return {}
    # Return a sentinel so the orchestrator treats this as "has data".
    return {"__nmme_done__": True}


# ===================================================================
# GDACS — delegate to resolver.connectors.gdacs / resolver.tools.run_pipeline
# ===================================================================


def _bulk_fetch_gdacs(dry_run: bool = False) -> dict[str, Any]:
    """Run the GDACS connector via the Resolver pipeline.

    The GDACS connector fetches disaster population-exposure data (FL, DR,
    TC) from the GDACS RSS archive.  It writes to ``facts_resolved`` and
    ``facts_deltas`` through the standard Resolver pipeline.
    """
    if dry_run:
        LOG.info("[gdacs] dry-run — skipping GDACS fetch")
        return {"__gdacs_done__": True}

    try:
        from resolver.tools.run_pipeline import run_pipeline

        db_url = os.getenv("RESOLVER_DB_URL") or None
        result = run_pipeline(connectors=["gdacs"], db_url=db_url)
        LOG.info(
            "[gdacs] pipeline complete: %d facts, %d resolved, %d deltas",
            result.total_facts,
            result.resolved_rows,
            result.delta_rows,
        )
        if result.total_facts == 0:
            LOG.warning("[gdacs] no data returned from GDACS connector")
            return {}
    except Exception as exc:
        LOG.error("[gdacs] pipeline failed: %s", exc, exc_info=True)
        return {}

    return {"__gdacs_done__": True}


# ===================================================================
# Conflict forecasts — delegate to resolver.tools.fetch_conflict_forecasts
# ===================================================================

# Labels that handle their own DB writes and return sentinel dicts.
_SELF_STORING_LABELS = frozenset([
    "nmme_seasonal_forecasts",
    "views_forecasts",
    "conflictforecast_forecasts",
    "gdacs_population_exposed",
    "acledcast_forecasts",
])

# Map from our internal labels to the source names used by
# resolver.tools.fetch_conflict_forecasts.
_CONFLICT_SOURCE_MAP = {
    "views_forecasts": "views",
    "conflictforecast_forecasts": "conflictforecast_org",
    "acledcast_forecasts": "acled_cast",
}


def _bulk_fetch_conflict(label: str, dry_run: bool = False) -> dict[str, Any]:
    """Run a single conflict forecast source (fetch + store).

    Delegates to ``resolver.tools.fetch_conflict_forecasts.fetch_and_store``
    which handles its own DB writes.  Returns a sentinel dict on success.
    """
    from resolver.tools.fetch_conflict_forecasts import fetch_and_store

    source_name = _CONFLICT_SOURCE_MAP[label]
    counts = fetch_and_store(sources=[source_name], dry_run=dry_run)
    total = sum(counts.values())
    if total == 0:
        LOG.warning("[ingest] conflict source %s returned 0 rows", source_name)
        return {}
    return {f"__{label}_done__": True}


# ===================================================================
# Storage helpers — call the existing store functions
# ===================================================================


def _store_all(
    label: str,
    data_by_country: dict[str, Any],
    dry_run: bool = False,
) -> dict[str, int]:
    """Store bulk-fetched data using existing per-country store functions.

    Returns {"success": N, "empty": N, "fail": N}.
    """
    stats = {"success": 0, "empty": 0, "fail": 0}

    if not data_by_country:
        return stats

    if dry_run:
        stats["success"] = len(data_by_country)
        return stats

    for iso3, data in data_by_country.items():
        try:
            if label == "acaps_inform_severity":
                from pythia.acaps import store_inform_severity
                store_inform_severity(iso3, data)
            elif label == "acaps_risk_radar":
                from pythia.acaps import store_risk_radar
                store_risk_radar(iso3, data)
            elif label == "acaps_daily_monitoring":
                from pythia.acaps import store_daily_monitoring
                store_daily_monitoring(iso3, data)
            elif label == "acaps_humanitarian_access":
                from pythia.acaps import store_humanitarian_access
                store_humanitarian_access(iso3, data)
            elif label == "ipc_phases":
                from pythia.ipc_phases import store_ipc_phases
                store_ipc_phases(iso3, data)
            elif label == "reliefweb_reports":
                from horizon_scanner.reliefweb import store_reliefweb_reports
                store_reliefweb_reports(iso3, data)
            elif label == "acled_political_events":
                from pythia.acled_political import store_acled_political_events
                store_acled_political_events(iso3, data)
            elif label in _SELF_STORING_LABELS:
                # These sources handle their own DB writes in their
                # bulk-fetch functions (_bulk_fetch_nmme, _bulk_fetch_conflict).
                stats["success"] += 1
                continue
            else:
                LOG.error("Unknown label for storage: %s", label)
                stats["fail"] += 1
                continue

            stats["success"] += 1
        except Exception as exc:
            LOG.error("Store failed for %s / %s: %s", iso3, label, exc)
            stats["fail"] += 1

    return stats


# ===================================================================
# Main orchestrator
# ===================================================================


def ingest(
    *,
    iso3_override: str | None = None,
    sources: Sequence[str] | None = None,
    dry_run: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run the bulk structured-data ingestion.

    Returns a nested dict: { source_label: { "success": N, ... } }
    """
    countries_list = _load_iso3_list(iso3_override)
    countries = set(countries_list)
    LOG.info("[ingest] %d countries to process", len(countries))

    if sources is None:
        sources = ALL_SOURCE_NAMES
    # Expand aliases (acaps -> 4 ACAPS sources, conflict -> 3 conflict sources)
    expanded: list[str] = []
    for s in sources:
        if s in _SOURCE_ALIASES:
            expanded.extend(_SOURCE_ALIASES[s])
        else:
            expanded.append(s)
    sources = list(dict.fromkeys(expanded))  # dedupe, preserve order
    unknown = [s for s in sources if s not in _SOURCE_GROUPS]
    if unknown:
        raise ValueError(f"Unknown source group(s): {unknown}")

    labels: list[str] = []
    for src in sources:
        labels.extend(_SOURCE_GROUPS[src])

    LOG.info("[ingest] sources: %s (labels: %s)", list(sources), labels)
    if dry_run:
        LOG.info("[ingest] DRY RUN — data fetched but NOT stored")

    stats: dict[str, dict[str, int]] = {
        lbl: {"success": 0, "fail": 0, "empty": 0} for lbl in labels
    }

    # --- Phase 1: Bulk-fetch all sources concurrently ---
    # Each source group runs in its own thread for I/O parallelism.
    bulk_data: dict[str, dict[str, Any]] = {}

    def _run_source(label: str) -> tuple[str, dict[str, Any]]:
        LOG.info("[ingest] starting bulk fetch: %s", label)
        t0 = time.monotonic()
        try:
            if label == "acaps_inform_severity":
                result = _bulk_fetch_inform_severity(countries)
            elif label == "acaps_risk_radar":
                result = _bulk_fetch_risk_radar(countries)
            elif label == "acaps_daily_monitoring":
                result = _bulk_fetch_daily_monitoring(countries)
            elif label == "acaps_humanitarian_access":
                result = _bulk_fetch_humanitarian_access(countries)
            elif label == "ipc_phases":
                result = _bulk_fetch_ipc(countries)
            elif label == "reliefweb_reports":
                result = _bulk_fetch_reliefweb(countries)
            elif label == "nmme_seasonal_forecasts":
                result = _bulk_fetch_nmme(dry_run)
            elif label == "acled_political_events":
                result = _bulk_fetch_acled_political(countries)
            elif label == "gdacs_population_exposed":
                result = _bulk_fetch_gdacs(dry_run)
            elif label in _CONFLICT_SOURCE_MAP:
                result = _bulk_fetch_conflict(label, dry_run)
            else:
                LOG.error("Unknown label: %s", label)
                result = {}
        except Exception as exc:
            LOG.error("[ingest] bulk fetch %s failed: %s", label, exc, exc_info=True)
            result = {}

        elapsed = time.monotonic() - t0
        LOG.info(
            "[ingest] %s: fetched %d countries in %.1fs",
            label, len(result), elapsed,
        )
        return label, result

    # Run ACAPS sources sequentially (shared auth token, rate limits)
    # but run everything else concurrently (conflict forecasts, IPC, ReliefWeb, NMME)
    acaps_labels = [l for l in labels if l.startswith("acaps_")]
    other_labels = [l for l in labels if not l.startswith("acaps_")]

    with ThreadPoolExecutor(max_workers=max(3, len(other_labels))) as pool:
        futures = []

        # Submit a single future that runs all ACAPS sources sequentially
        def _run_acaps_sequential() -> list[tuple[str, dict]]:
            results = []
            for lbl in acaps_labels:
                results.append(_run_source(lbl))
            return results

        if acaps_labels:
            futures.append(pool.submit(_run_acaps_sequential))

        # Submit other sources individually
        for lbl in other_labels:
            futures.append(pool.submit(_run_source, lbl))

        for future in as_completed(futures):
            result = future.result()
            if isinstance(result, list):
                # ACAPS sequential batch
                for lbl, data in result:
                    bulk_data[lbl] = data
            else:
                lbl, data = result
                bulk_data[lbl] = data

    # --- Phase 2: Store all data ---
    LOG.info("[ingest] === Storage phase ===")
    for lbl in labels:
        data = bulk_data.get(lbl, {})
        if not data:
            stats[lbl]["empty"] = len(countries)
            continue

        s = _store_all(lbl, data, dry_run=dry_run)
        stats[lbl] = s

        # Count countries with no data as "empty"
        covered = s["success"] + s["fail"]
        stats[lbl]["empty"] = len(countries) - covered

    # ---- Summary ----
    print("\n===== Ingestion Summary =====")
    if dry_run:
        print("MODE: DRY RUN (no data written)")
    print(f"Countries processed: {len(countries)}")
    print()

    print("Per-source results:")
    for lbl in labels:
        s = stats[lbl]
        print(
            f"  {lbl:40s}  ok={s['success']:4d}  "
            f"empty={s['empty']:4d}  fail={s['fail']:4d}"
        )

    total_ok = sum(s["success"] for s in stats.values())
    total_fail = sum(s["fail"] for s in stats.values())
    print(f"\nTotals: success={total_ok}  failure={total_fail}")

    return stats


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Bulk-ingest structured data (conflict forecasts, ACAPS, IPC, "
            "ReliefWeb, NMME) into DuckDB for the Horizon Scanner pipeline."
        ),
    )
    parser.add_argument(
        "--iso3",
        default=None,
        help="Comma-separated ISO-3 codes (default: all from countries.csv)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help=(
            "Which source groups to ingest (default: all). "
            "Sources: " + ", ".join(ALL_SOURCE_NAMES) + ". "
            "Aliases: " + ", ".join(
                f"{k} (={'+'.join(v)})" for k, v in sorted(_SOURCE_ALIASES.items())
            ) + ". "
            "Accepts space-separated, comma-separated, or mixed."
        ),
    )
    parser.add_argument(
        "--db",
        dest="db_url",
        default=None,
        help="Override PYTHIA_DB_URL",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but do not write to DB",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Normalise --sources: accept comma-separated, space-separated, or mixed.
    # e.g. "acaps,ipc,reliefweb" or "acaps, ipc" or "acaps ipc reliefweb"
    if args.sources:
        normalised: list[str] = []
        for token in args.sources:
            normalised.extend(part.strip() for part in token.split(",") if part.strip())
        # Expand aliases at the CLI level so validation works.
        expanded: list[str] = []
        for s in normalised:
            if s in _SOURCE_ALIASES:
                expanded.extend(_SOURCE_ALIASES[s])
            elif s in _SOURCE_GROUPS:
                expanded.append(s)
            else:
                parser.error(
                    f"invalid source: {s}. "
                    f"Choose from: {', '.join(ALL_SOURCE_NAMES)} "
                    f"(aliases: {', '.join(ALL_ALIAS_NAMES)})"
                )
        args.sources = list(dict.fromkeys(expanded))  # dedupe

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.db_url:
        os.environ["PYTHIA_DB_URL"] = args.db_url

    try:
        stats = ingest(
            iso3_override=args.iso3,
            sources=args.sources,
            dry_run=args.dry_run,
        )
        total_fail = sum(s["fail"] for s in stats.values())
        if total_fail:
            LOG.warning("[ingest] completed with %d failure(s)", total_fail)
        else:
            LOG.info("[ingest] completed successfully")
    except Exception as exc:
        LOG.error("Fatal: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
    