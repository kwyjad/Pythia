# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ACAPS unified data connector.

Pulls four datasets from the ACAPS API through a shared authentication and
pagination layer:

1. **INFORM Severity Index** -- crisis severity scores with trend data
2. **Risk Radar** -- forward-looking risk assessments with triggers
3. **Daily Monitoring** -- analyst-curated situational updates
4. **Humanitarian Access** -- access constraint scores

All four datasets share the same API base URL, auth mechanism, and pagination
pattern.  This module handles all of them with shared infrastructure.

Public API
----------
INFORM Severity:
- :func:`fetch_inform_severity` -- fetch from ACAPS API
- :func:`store_inform_severity` -- upsert into DuckDB
- :func:`load_inform_severity` -- load from DuckDB
- :func:`format_inform_severity_for_prompt` -- triage / RC prompt block
- :func:`format_inform_severity_for_spd` -- compact SPD prompt block

Risk Radar:
- :func:`fetch_risk_radar` -- fetch from ACAPS API
- :func:`store_risk_radar` -- upsert into DuckDB
- :func:`load_risk_radar` -- load from DuckDB
- :func:`format_risk_radar_for_prompt` -- triage / RC prompt block
- :func:`format_risk_radar_for_spd` -- compact SPD prompt block

Daily Monitoring:
- :func:`fetch_daily_monitoring` -- fetch from ACAPS API
- :func:`store_daily_monitoring` -- upsert into DuckDB
- :func:`load_daily_monitoring` -- load from DuckDB
- :func:`format_daily_monitoring_for_prompt` -- triage prompt block
- :func:`format_daily_monitoring_for_spd` -- compact SPD prompt block

Humanitarian Access:
- :func:`fetch_humanitarian_access` -- fetch from ACAPS API
- :func:`store_humanitarian_access` -- upsert into DuckDB
- :func:`load_humanitarian_access` -- load from DuckDB
- :func:`format_humanitarian_access_for_prompt` -- triage prompt block
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)

ACAPS_API_BASE = "https://api.acaps.org"

_MAX_RETRIES = 2
_RATIONALE_MAX_LEN = 500
_TRIGGER_DESC_MAX_LEN = 200
_DEVELOPMENTS_MAX_LEN = 400

# Module-level token cache
_acaps_token: str | None = None


# ============================================================
# Shared infrastructure (auth, pagination, DB URL)
# ============================================================


def _db_url() -> str:
    """Resolve the Pythia DuckDB URL."""
    url = os.getenv("PYTHIA_DB_URL", "").strip()
    if url:
        return url
    try:
        from pythia.config import load as load_config
        cfg = load_config()
        url = str((cfg.get("app") or {}).get("db_url", "")).strip()
        if url:
            return url
    except Exception:
        pass
    from resolver.db.duckdb_io import DEFAULT_DB_URL
    return DEFAULT_DB_URL


def _get_acaps_token(force_refresh: bool = False) -> str | None:
    """Get or refresh ACAPS API token from env credentials.

    Parameters
    ----------
    force_refresh
        If True, ignore the cached token and re-authenticate.

    Returns
    -------
    str | None
        The API token, or None if credentials are missing or auth fails.
    """
    global _acaps_token
    if _acaps_token and not force_refresh:
        return _acaps_token

    username = os.getenv("ACAPS_USERNAME", "").strip()
    password = os.getenv("ACAPS_PASSWORD", "").strip()
    if not username or not password:
        log.warning(
            "ACAPS credentials not set (ACAPS_USERNAME / ACAPS_PASSWORD) "
            "-- skipping ACAPS data fetch."
        )
        return None

    try:
        resp = requests.post(
            f"{ACAPS_API_BASE}/api/v1/token-auth/",
            json={"username": username, "password": password},
            timeout=30,
        )
    except requests.RequestException as exc:
        log.warning("ACAPS token request failed: %s", exc)
        return None

    if resp.status_code != 200:
        log.warning("ACAPS token auth returned HTTP %d", resp.status_code)
        return None

    try:
        token = resp.json().get("token")
    except (ValueError, AttributeError):
        log.warning("ACAPS token response was not valid JSON")
        return None

    if not token:
        log.warning("ACAPS token response did not contain a token")
        return None

    _acaps_token = token
    return _acaps_token


def _fetch_paginated(
    endpoint: str,
    params: dict | None = None,
    max_pages: int = 10,
    token: str | None = None,
) -> list[dict]:
    """Fetch all pages from an ACAPS API endpoint.

    Parameters
    ----------
    endpoint
        API path (e.g. ``/api/v1/risk-radar/risk-radar/``).
    params
        Query parameters for the first request.
    max_pages
        Maximum pages to fetch (safety limit).
    token
        API token.  If None, will attempt to obtain one.

    Returns
    -------
    list[dict]
        Concatenated ``results`` from all pages.  Empty list on error.
    """
    if token is None:
        token = _get_acaps_token()
    if token is None:
        return []

    headers = {"Authorization": f"Token {token}"}

    # Build full URL for first request
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
                timeout=60,
            )
        except requests.RequestException as exc:
            log.warning("ACAPS API request failed: %s", exc)
            return all_results

        # Retry once on 401 (re-authenticate)
        if resp.status_code == 401 and not retried_auth:
            retried_auth = True
            new_token = _get_acaps_token(force_refresh=True)
            if new_token:
                token = new_token
                headers = {"Authorization": f"Token {token}"}
                # Retry same page
                try:
                    resp = requests.get(
                        url,
                        params=params if page_num == 1 else None,
                        headers=headers,
                        timeout=60,
                    )
                except requests.RequestException as exc:
                    log.warning("ACAPS API retry failed: %s", exc)
                    return all_results

        if resp.status_code != 200:
            log.warning("ACAPS API HTTP %d for %s", resp.status_code, endpoint)
            return all_results

        try:
            body = resp.json()
        except ValueError:
            log.warning("ACAPS API response was not valid JSON")
            return all_results

        results = body.get("results") or []
        if isinstance(results, list):
            all_results.extend(results)

        url = body.get("next")

        # Rate-limit: 1s sleep between page requests
        if url and page_num < max_pages:
            time.sleep(1)

    return all_results


def _current_month_label() -> str:
    """Return the current month as ``MmmYYYY`` (e.g. ``Mar2026``)."""
    return date.today().strftime("%b%Y")


def _month_labels_back(n: int) -> list[str]:
    """Return a list of ``MmmYYYY`` labels going back *n* months.

    The list is ordered most-recent-first (index 0 = current month).
    """
    today = date.today()
    labels: list[str] = []
    for i in range(n):
        d = today.replace(day=1) - timedelta(days=i * 30)
        labels.append(d.strftime("%b%Y"))
    return labels


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to *max_len* chars, breaking at a word boundary."""
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "..."


def _access_category(score: float) -> str:
    """Map an access score (0-5) to a category label."""
    if score < 1:
        return "Very Low"
    if score < 2:
        return "Low"
    if score < 3:
        return "Medium"
    if score < 4:
        return "High"
    return "Very High"


# ============================================================
# INFORM Severity Index
# ============================================================


def fetch_inform_severity(
    iso3: str,
    months_back: int = 6,
) -> dict | None:
    """Fetch INFORM Severity data for a country from the ACAPS API.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.
    months_back
        How many months of trend history to fetch.

    Returns
    -------
    dict | None
        Structured severity data, or None on error / missing data.
    """
    token = _get_acaps_token()
    if token is None:
        return None

    if isinstance(iso3, list):
        iso3 = iso3[0] if iso3 else ""
    iso3 = str(iso3).strip().upper()

    # --- Fetch current snapshot (try current month, then fall back) ---
    snapshot = None
    snapshot_date = None
    for label in _month_labels_back(3):
        results = _fetch_paginated(
            f"/api/v1/inform-severity-index/{label}/",
            params={"iso3": iso3},
            max_pages=1,
            token=token,
        )
        if results:
            snapshot = _pick_country_crisis(results)
            snapshot_date = label
            break

    if snapshot is None:
        log.info("No INFORM Severity data found for %s", iso3)
        return None

    severity_score = _safe_float(snapshot.get("severity_index_score")
                                 or snapshot.get("severity_score")
                                 or snapshot.get("score"))
    severity_category = (snapshot.get("severity_index_category")
                         or snapshot.get("severity_category")
                         or snapshot.get("category")
                         or "")

    # --- Fetch dimension scores if available in snapshot ---
    impact_score = _safe_float(snapshot.get("impact_score")
                               or snapshot.get("impact_of_the_crisis"))
    conditions_score = _safe_float(
        snapshot.get("conditions_score")
        or snapshot.get("conditions_of_people_affected")
    )
    complexity_score = _safe_float(snapshot.get("complexity_score")
                                   or snapshot.get("complexity"))

    crisis_id = snapshot.get("crisis_id", "")
    crisis_name = snapshot.get("crisis", "") or snapshot.get("crisis_name", "")

    # --- Fetch country-log for trend data ---
    trend_entries: list[dict] = []
    log_results = _fetch_paginated(
        "/api/v1/inform-severity-index/country-log/",
        params={"iso3": iso3},
        max_pages=5,
        token=token,
    )
    for entry in log_results:
        entry_date = entry.get("date", "")
        entry_score = _safe_float(entry.get("value"))
        if entry_date and entry_score is not None:
            trend_entries.append({"date": entry_date, "score": entry_score})

    # Sort by date ascending
    trend_entries.sort(key=lambda e: e["date"])

    # Keep only last months_back entries
    if len(trend_entries) > months_back:
        trend_entries = trend_entries[-months_back:]

    # Compute deltas
    delta_1m = None
    delta_3m = None
    if severity_score is not None and len(trend_entries) >= 2:
        delta_1m = round(severity_score - trend_entries[-1]["score"], 2)
    if severity_score is not None and len(trend_entries) >= 4:
        delta_3m = round(severity_score - trend_entries[-3]["score"], 2)

    # --- Fetch top indicators from dimension endpoints ---
    top_indicators: list[dict] = []
    for dim_name, dim_path in [
        ("impact", "impact-of-crisis"),
        ("conditions", "conditions-of-people-affected"),
        ("complexity", "complexity"),
    ]:
        dim_results = _fetch_paginated(
            f"/api/v1/inform-severity-index/{dim_path}/{snapshot_date}/",
            params={"iso3": iso3},
            max_pages=1,
            token=token,
        )
        for ind in dim_results:
            fig = _safe_float(ind.get("figure"))
            if fig is not None and fig >= 4.0:
                top_indicators.append({
                    "indicator": ind.get("indicator", ""),
                    "figure": fig,
                    "dimension": dim_name,
                })

    # Sort by figure descending, keep top 5
    top_indicators.sort(key=lambda x: x["figure"], reverse=True)
    top_indicators = top_indicators[:5]

    return {
        "iso3": iso3,
        "crisis_id": crisis_id,
        "crisis_name": crisis_name,
        "severity_score": severity_score,
        "severity_category": severity_category,
        "impact_score": impact_score,
        "conditions_score": conditions_score,
        "complexity_score": complexity_score,
        "snapshot_date": snapshot_date,
        "trend_6m": trend_entries,
        "delta_1m": delta_1m,
        "delta_3m": delta_3m,
        "top_indicators": top_indicators,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _pick_country_crisis(results: list[dict]) -> dict:
    """Pick the best crisis record for a country.

    Prefers country-level aggregates; falls back to highest severity.
    """
    if len(results) == 1:
        return results[0]

    # Look for country-level record
    for r in results:
        if r.get("country_level"):
            return r

    # Fall back to highest severity score
    def _score(r: dict) -> float:
        return _safe_float(
            r.get("severity_index_score")
            or r.get("severity_score")
            or r.get("score")
        ) or 0.0
    return max(results, key=_score)


def store_inform_severity(
    iso3: str,
    data: dict,
    db_url: str | None = None,
) -> None:
    """Upsert INFORM Severity data into DuckDB.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.
    data
        Severity dict as returned by :func:`fetch_inform_severity`.
    db_url
        Optional DuckDB URL override.
    """
    if not data:
        return

    try:
        from pythia.db.schema import connect, ensure_schema
    except Exception:
        log.debug("Pythia DB helpers unavailable -- skipping store.")
        return

    try:
        con = connect(read_only=False)
    except Exception:
        log.warning("Could not connect to DuckDB for INFORM Severity store.")
        return

    try:
        ensure_schema(con)
        now = data.get("fetched_at", datetime.now(timezone.utc).isoformat())

        con.execute(
            """
            INSERT OR REPLACE INTO acaps_inform_severity
                (iso3, crisis_id, snapshot_date, severity_score,
                 severity_category, impact_score, conditions_score,
                 complexity_score, crisis_name, top_indicators_json,
                 fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                iso3.upper(),
                data.get("crisis_id", ""),
                data.get("snapshot_date", ""),
                data.get("severity_score"),
                data.get("severity_category", ""),
                data.get("impact_score"),
                data.get("conditions_score"),
                data.get("complexity_score"),
                data.get("crisis_name", ""),
                json.dumps(data.get("top_indicators", [])),
                now,
            ],
        )

        # Store trend entries
        for entry in data.get("trend_6m", []):
            con.execute(
                """
                INSERT OR REPLACE INTO acaps_inform_severity_trend
                    (iso3, snapshot_date, score, fetched_at)
                VALUES (?, ?, ?, ?)
                """,
                [iso3.upper(), entry["date"], entry["score"], now],
            )
    except Exception as exc:
        log.warning("Failed to store INFORM Severity for %s: %s", iso3, exc)
    finally:
        con.close()


def load_inform_severity(
    iso3: str,
    db_url: str | None = None,
) -> dict | None:
    """Load INFORM Severity data from DuckDB.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.
    db_url
        Optional DuckDB URL override.

    Returns
    -------
    dict | None
        Severity data with computed deltas, or None if no data.
    """
    try:
        from resolver.db.duckdb_io import get_db, close_db
    except Exception:
        log.debug("DuckDB helpers unavailable -- skipping INFORM load.")
        return None

    db_url = db_url or _db_url()

    try:
        con = get_db(db_url)
    except Exception:
        log.debug("Could not connect to DuckDB at %s", db_url)
        return None

    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        ]
        if "acaps_inform_severity" not in tables:
            return None

        row = con.execute(
            """
            SELECT crisis_id, snapshot_date, severity_score,
                   severity_category, impact_score, conditions_score,
                   complexity_score, crisis_name, top_indicators_json,
                   fetched_at
            FROM acaps_inform_severity
            WHERE iso3 = ?
            ORDER BY snapshot_date DESC
            LIMIT 1
            """,
            [iso3.upper()],
        ).fetchone()

        if not row:
            return None

        (crisis_id, snapshot_date, severity_score, severity_category,
         impact_score, conditions_score, complexity_score, crisis_name,
         top_indicators_json, fetched_at) = row

        # Load trend data
        trend_rows = con.execute(
            """
            SELECT snapshot_date, score
            FROM acaps_inform_severity_trend
            WHERE iso3 = ?
            ORDER BY snapshot_date ASC
            """,
            [iso3.upper()],
        ).fetchall()

        trend_6m = [{"date": r[0], "score": r[1]} for r in trend_rows]

        # Compute deltas
        delta_1m = None
        delta_3m = None
        if severity_score is not None and len(trend_6m) >= 2:
            delta_1m = round(severity_score - trend_6m[-1]["score"], 2)
        if severity_score is not None and len(trend_6m) >= 4:
            delta_3m = round(severity_score - trend_6m[-3]["score"], 2)

        try:
            top_indicators = json.loads(top_indicators_json or "[]")
        except (ValueError, TypeError):
            top_indicators = []

        return {
            "iso3": iso3.upper(),
            "crisis_id": crisis_id,
            "crisis_name": crisis_name,
            "severity_score": severity_score,
            "severity_category": severity_category,
            "impact_score": impact_score,
            "conditions_score": conditions_score,
            "complexity_score": complexity_score,
            "snapshot_date": snapshot_date,
            "trend_6m": trend_6m,
            "delta_1m": delta_1m,
            "delta_3m": delta_3m,
            "top_indicators": top_indicators,
            "fetched_at": fetched_at,
        }
    except Exception as exc:
        log.warning("Failed to load INFORM Severity for %s: %s", iso3, exc)
        return None
    finally:
        close_db(con)


def format_inform_severity_for_prompt(data: dict | None) -> str:
    """Format INFORM Severity as a text block for RC / triage prompts.

    Parameters
    ----------
    data
        Severity dict from :func:`load_inform_severity`.

    Returns
    -------
    str
        Formatted prompt section, or empty string if no data.
    """
    if not data or data.get("severity_score") is None:
        return ""

    iso3 = data.get("iso3", "")
    score = data["severity_score"]
    cat = data.get("severity_category", "")
    snap = data.get("snapshot_date", "")
    impact = data.get("impact_score")
    conditions = data.get("conditions_score")
    complexity = data.get("complexity_score")
    d1 = data.get("delta_1m")
    d3 = data.get("delta_3m")
    indicators = data.get("top_indicators", [])

    parts: list[str] = [
        f"INFORM SEVERITY INDEX ({iso3}):",
        f"Overall: {score}/5.0 ({cat}) | Snapshot: {snap}",
    ]

    dim_parts: list[str] = []
    if impact is not None:
        dim_parts.append(f"Impact {impact}/5")
    if conditions is not None:
        dim_parts.append(f"Conditions {conditions}/5")
    if complexity is not None:
        dim_parts.append(f"Complexity {complexity}/5")
    if dim_parts:
        parts.append(f"Dimensions: {' | '.join(dim_parts)}")

    trend_parts: list[str] = []
    if d1 is not None:
        trend_parts.append(f"{d1:+.1f} vs last month")
    if d3 is not None:
        trend_parts.append(f"{d3:+.1f} vs 3 months ago")
    if trend_parts:
        parts.append(f"Trend: {', '.join(trend_parts)}")

    if indicators:
        ind_strs = [f"{i['indicator']} ({i['figure']})" for i in indicators[:3]]
        parts.append(f"Top indicators: {', '.join(ind_strs)}")

    parts.append(
        "\nINFORM Severity is an institutional composite of 31 indicators. "
        "Use it as a severity benchmark. Large positive deltas (>0.3/month) "
        "suggest rapid deterioration that should be reflected in your "
        "assessment."
    )

    return "\n".join(parts)


def format_inform_severity_for_spd(data: dict | None) -> str:
    """Compact INFORM Severity block for SPD prompts.

    Parameters
    ----------
    data
        Severity dict from :func:`load_inform_severity`.

    Returns
    -------
    str
        Compact prompt section, or empty string if no data.
    """
    if not data or data.get("severity_score") is None:
        return ""

    iso3 = data.get("iso3", "")
    score = data["severity_score"]
    cat = data.get("severity_category", "")
    d1 = data.get("delta_1m")
    d3 = data.get("delta_3m")

    d1_str = f"{d1:+.1f}" if d1 is not None else "n/a"
    d3_str = f"{d3:+.1f}" if d3 is not None else "n/a"

    return (
        f"INFORM SEVERITY ({iso3}): {score}/5.0 ({cat}) "
        f"[D1m: {d1_str}, D3m: {d3_str}]"
    )


# ============================================================
# Risk Radar
# ============================================================


def fetch_risk_radar(
    iso3: str,
    include_triggers: bool = True,
) -> dict | None:
    """Fetch ACAPS Risk Radar data for a country.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.
    include_triggers
        Whether to fetch individual trigger details per risk.

    Returns
    -------
    dict | None
        Structured risk data, or None on auth error.  A country with zero
        active risks returns a valid dict with ``risks: []``.
    """
    token = _get_acaps_token()
    if token is None:
        return None

    if isinstance(iso3, list):
        iso3 = iso3[0] if iso3 else ""
    iso3 = str(iso3).strip().upper()
    used_fallback = False

    # Try Risk Radar endpoint first
    results = _fetch_paginated(
        "/api/v1/risk-radar/risk-radar/",
        params={"iso3": iso3, "status": "Active"},
        max_pages=3,
        token=token,
    )

    # Fallback to original Risk List if Risk Radar is empty
    if not results:
        results = _fetch_paginated(
            "/api/v1/risk-list/",
            params={"iso3": iso3, "status": "Active"},
            max_pages=3,
            token=token,
        )
        used_fallback = bool(results)

    risks: list[dict] = []
    for r in results:
        risk_id = str(r.get("risk_id", r.get("id", "")))

        rationale = str(r.get("rationale", "") or "")
        rationale = _truncate(rationale, _RATIONALE_MAX_LEN)

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

        # Fetch individual triggers if requested
        if include_triggers and risk_id and not used_fallback:
            trigger_results = _fetch_paginated(
                "/api/v1/risk-radar/trigger-list/",
                params={"risk": risk_id},
                max_pages=2,
                token=token,
            )
            for t in trigger_results:
                desc = str(t.get("description", "") or "")
                desc = _truncate(desc, _TRIGGER_DESC_MAX_LEN)
                risk_entry["triggers"].append({
                    "trigger_id": str(t.get("trigger_id", "")),
                    "title": t.get("title", ""),
                    "completion_rate": t.get("completion_rate", ""),
                    "trend": t.get("trend", ""),
                    "description": desc,
                })

        risks.append(risk_entry)

    # Determine highest risk level
    level_order = {"Low": 0, "Medium": 1, "High": 2}
    highest = "None"
    for risk in risks:
        rl = risk.get("risk_level", "")
        if level_order.get(rl, -1) > level_order.get(highest, -1):
            highest = rl

    return {
        "iso3": iso3,
        "risks": risks,
        "total_active_risks": len(risks),
        "highest_risk_level": highest if risks else "None",
        "source": "risk-list" if used_fallback else "risk-radar",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def store_risk_radar(
    iso3: str,
    data: dict,
    db_url: str | None = None,
) -> None:
    """Upsert Risk Radar data into DuckDB.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.
    data
        Risk dict as returned by :func:`fetch_risk_radar`.
    db_url
        Optional DuckDB URL override.
    """
    if not data or not data.get("risks"):
        return

    try:
        from pythia.db.schema import connect, ensure_schema
    except Exception:
        log.debug("Pythia DB helpers unavailable -- skipping store.")
        return

    try:
        con = connect(read_only=False)
    except Exception:
        log.warning("Could not connect to DuckDB for Risk Radar store.")
        return

    try:
        ensure_schema(con)
        now = data.get("fetched_at", datetime.now(timezone.utc).isoformat())

        for risk in data["risks"]:
            con.execute(
                """
                INSERT OR REPLACE INTO acaps_risk_radar
                    (iso3, risk_id, risk_title, risk_type, risk_level,
                     probability, impact, risk_trend, expected_exposure,
                     triggers_summary, rationale_excerpt, triggers_json,
                     status, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    iso3.upper(),
                    risk.get("risk_id", ""),
                    risk.get("title", ""),
                    risk.get("risk_type", ""),
                    risk.get("risk_level", ""),
                    risk.get("probability", ""),
                    risk.get("impact", 0),
                    risk.get("risk_trend", ""),
                    risk.get("expected_exposure", ""),
                    risk.get("triggers_summary", ""),
                    risk.get("rationale", ""),
                    json.dumps(risk.get("triggers", [])),
                    "Active",
                    now,
                ],
            )
    except Exception as exc:
        log.warning("Failed to store Risk Radar for %s: %s", iso3, exc)
    finally:
        con.close()


def load_risk_radar(
    iso3: str,
    db_url: str | None = None,
) -> dict | None:
    """Load Risk Radar data from DuckDB.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.
    db_url
        Optional DuckDB URL override.

    Returns
    -------
    dict | None
        Risk data dict, or None if table doesn't exist / DB unavailable.
    """
    try:
        from resolver.db.duckdb_io import get_db, close_db
    except Exception:
        log.debug("DuckDB helpers unavailable -- skipping Risk Radar load.")
        return None

    db_url = db_url or _db_url()

    try:
        con = get_db(db_url)
    except Exception:
        log.debug("Could not connect to DuckDB at %s", db_url)
        return None

    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        ]
        if "acaps_risk_radar" not in tables:
            return None

        rows = con.execute(
            """
            SELECT risk_id, risk_title, risk_type, risk_level,
                   probability, impact, risk_trend, expected_exposure,
                   triggers_summary, rationale_excerpt, triggers_json,
                   status, fetched_at
            FROM acaps_risk_radar
            WHERE iso3 = ? AND status = 'Active'
            ORDER BY risk_level DESC
            """,
            [iso3.upper()],
        ).fetchall()

        risks: list[dict] = []
        for row in rows:
            (risk_id, risk_title, risk_type, risk_level, probability,
             impact, risk_trend, expected_exposure, triggers_summary,
             rationale_excerpt, triggers_json, status, fetched_at) = row

            try:
                triggers = json.loads(triggers_json or "[]")
            except (ValueError, TypeError):
                triggers = []

            risks.append({
                "risk_id": risk_id,
                "title": risk_title,
                "risk_type": risk_type,
                "risk_level": risk_level,
                "probability": probability,
                "impact": impact,
                "risk_trend": risk_trend,
                "expected_exposure": expected_exposure,
                "triggers_summary": triggers_summary,
                "rationale": rationale_excerpt,
                "triggers": triggers,
            })

        level_order = {"Low": 0, "Medium": 1, "High": 2}
        highest = "None"
        for risk in risks:
            rl = risk.get("risk_level", "")
            if level_order.get(rl, -1) > level_order.get(highest, -1):
                highest = rl

        return {
            "iso3": iso3.upper(),
            "risks": risks,
            "total_active_risks": len(risks),
            "highest_risk_level": highest if risks else "None",
        }
    except Exception as exc:
        log.warning("Failed to load Risk Radar for %s: %s", iso3, exc)
        return None
    finally:
        close_db(con)


def format_risk_radar_for_prompt(data: dict | None) -> str:
    """Format Risk Radar as a text block for RC / triage prompts.

    Parameters
    ----------
    data
        Risk dict from :func:`load_risk_radar`.

    Returns
    -------
    str
        Formatted prompt section, or empty string if no data / no risks.
    """
    if not data:
        return ""

    iso3 = data.get("iso3", "")
    risks = data.get("risks", [])
    total = data.get("total_active_risks", 0)
    highest = data.get("highest_risk_level", "None")

    if total == 0:
        return ""

    parts: list[str] = [
        f"ACAPS RISK RADAR ({iso3}):",
        f"{total} active risks identified by ACAPS analysts. "
        f"Highest level: {highest}.",
    ]

    for i, risk in enumerate(risks, 1):
        rl = risk.get("risk_level", "?")
        title = risk.get("title", "Untitled")
        rtype = risk.get("risk_type", "")
        prob = risk.get("probability", "?")
        impact = risk.get("impact", "?")
        trend = risk.get("risk_trend", "?")
        exposure = risk.get("expected_exposure", "")
        trig_sum = risk.get("triggers_summary", "")

        parts.append(f"\nRisk {i} [{rl}]: {title}")
        parts.append(
            f"  Type: {rtype} | Probability: {prob} | "
            f"Impact: {impact}/5"
        )
        parts.append(f"  Trend: {trend} | Exposure: {exposure}")
        if trig_sum:
            parts.append(f"  Triggers: {trig_sum}")

        triggers = risk.get("triggers", [])
        if triggers:
            parts.append("  Key trigger status:")
            for t in triggers[:3]:
                t_title = t.get("title", "?")
                t_comp = t.get("completion_rate", "?")
                t_trend = t.get("trend", "?")
                parts.append(
                    f"    - {t_title}: {t_comp} complete, trend {t_trend}"
                )

    parts.append(
        "\nACAPS risks are expert forward-looking assessments for the next "
        "6 months. Compare with your own RC assessment -- significant "
        "disagreements should be noted and reasoned about. Trigger "
        "completion rates indicate how close specific triggers are to "
        "firing."
    )

    return "\n".join(parts)


def format_risk_radar_for_spd(data: dict | None) -> str:
    """Compact Risk Radar block for SPD prompts.

    Parameters
    ----------
    data
        Risk dict from :func:`load_risk_radar`.

    Returns
    -------
    str
        Compact prompt section, or empty string if no data / no risks.
    """
    if not data or not data.get("risks"):
        return ""

    iso3 = data.get("iso3", "")
    risks = data.get("risks", [])
    total = data.get("total_active_risks", 0)
    highest = data.get("highest_risk_level", "None")

    parts: list[str] = [
        f"ACAPS RISKS ({iso3}): {total} active. Highest: {highest}.",
    ]

    for risk in risks:
        rl = risk.get("risk_level", "?")
        title = risk.get("title", "?")
        prob = risk.get("probability", "?")
        impact = risk.get("impact", "?")
        trend = risk.get("risk_trend", "?")

        triggers = risk.get("triggers", [])
        trig_strs: list[str] = []
        for t in triggers[:2]:
            t_title = t.get("title", "?")
            t_comp = t.get("completion_rate", "?")
            trig_strs.append(f"{t_title} ({t_comp})")

        line = f"- [{rl}] {title} (P:{prob}, I:{impact}/5, Trend:{trend})"
        parts.append(line)
        if trig_strs:
            parts.append(f"  Triggers: {', '.join(trig_strs)}")

    return "\n".join(parts)


# ============================================================
# Daily Monitoring
# ============================================================


def fetch_daily_monitoring(
    iso3: str,
    days_back: int = 30,
    max_entries: int = 20,
    weekly_picks_only: bool = False,
) -> list[dict] | None:
    """Fetch ACAPS Daily Monitoring entries for a country.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.
    days_back
        How many days of history to fetch.
    max_entries
        Maximum entries to return.
    weekly_picks_only
        If True, only return analyst-curated weekly picks.

    Returns
    -------
    list[dict] | None
        List of monitoring entries, or None on auth error.
    """
    token = _get_acaps_token()
    if token is None:
        return None

    if isinstance(iso3, list):
        iso3 = iso3[0] if iso3 else ""
    iso3 = str(iso3).strip().upper()
    end_date = date.today()
    start_date = end_date - timedelta(days=days_back)

    params: dict[str, Any] = {
        "iso3": iso3,
        "_internal_filter_date_gte": start_date.isoformat(),
        "_internal_filter_date_lte": end_date.isoformat(),
    }
    if weekly_picks_only:
        params["selected_weekly_pick"] = "true"

    results = _fetch_paginated(
        "/api/v1/daily-monitoring/",
        params=params,
        max_pages=3,
        token=token,
    )

    entries: list[dict] = []
    for r in results:
        dev = str(r.get("latest_developments", "") or "")
        dev = _truncate(dev, _DEVELOPMENTS_MAX_LEN)

        entries.append({
            "entry_id": str(r.get("entry_id", r.get("id", ""))),
            "date": r.get("date", ""),
            "latest_developments": dev,
            "source": r.get("source", ""),
            "weekly_pick": bool(r.get("selected_weekly_pick", False)),
        })

    # Sort by date descending
    entries.sort(key=lambda e: e.get("date", ""), reverse=True)

    # Limit
    entries = entries[:max_entries]

    return entries if entries else None


def store_daily_monitoring(
    iso3: str,
    entries: list[dict],
    db_url: str | None = None,
) -> None:
    """Upsert Daily Monitoring entries into DuckDB.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.
    entries
        Entry dicts as returned by :func:`fetch_daily_monitoring`.
    db_url
        Optional DuckDB URL override.
    """
    if not entries:
        return

    try:
        from pythia.db.schema import connect, ensure_schema
    except Exception:
        log.debug("Pythia DB helpers unavailable -- skipping store.")
        return

    try:
        con = connect(read_only=False)
    except Exception:
        log.warning("Could not connect to DuckDB for Daily Monitoring store.")
        return

    try:
        ensure_schema(con)
        now = datetime.now(timezone.utc).isoformat()

        for entry in entries:
            con.execute(
                """
                INSERT OR REPLACE INTO acaps_daily_monitoring
                    (iso3, entry_id, entry_date, latest_developments,
                     source, weekly_pick, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    iso3.upper(),
                    entry.get("entry_id", ""),
                    entry.get("date", ""),
                    entry.get("latest_developments", ""),
                    entry.get("source", ""),
                    entry.get("weekly_pick", False),
                    now,
                ],
            )
    except Exception as exc:
        log.warning("Failed to store Daily Monitoring for %s: %s", iso3, exc)
    finally:
        con.close()


def load_daily_monitoring(
    iso3: str,
    max_entries: int = 15,
    db_url: str | None = None,
) -> list[dict] | None:
    """Load Daily Monitoring entries from DuckDB.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.
    max_entries
        Maximum entries to return.
    db_url
        Optional DuckDB URL override.

    Returns
    -------
    list[dict] | None
        Entry dicts ordered by date descending, or None if no data.
    """
    try:
        from resolver.db.duckdb_io import get_db, close_db
    except Exception:
        log.debug("DuckDB helpers unavailable -- skipping monitoring load.")
        return None

    db_url = db_url or _db_url()

    try:
        con = get_db(db_url)
    except Exception:
        log.debug("Could not connect to DuckDB at %s", db_url)
        return None

    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        ]
        if "acaps_daily_monitoring" not in tables:
            return None

        rows = con.execute(
            """
            SELECT entry_id, entry_date, latest_developments,
                   source, weekly_pick
            FROM acaps_daily_monitoring
            WHERE iso3 = ?
            ORDER BY entry_date DESC
            LIMIT ?
            """,
            [iso3.upper(), max_entries],
        ).fetchall()

        if not rows:
            return None

        entries: list[dict] = []
        for row in rows:
            entries.append({
                "entry_id": row[0],
                "date": row[1],
                "latest_developments": row[2],
                "source": row[3],
                "weekly_pick": bool(row[4]),
            })

        return entries
    except Exception as exc:
        log.warning("Failed to load Daily Monitoring for %s: %s", iso3, exc)
        return None
    finally:
        close_db(con)


def format_daily_monitoring_for_prompt(entries: list[dict] | None) -> str:
    """Format Daily Monitoring as a text block for triage prompts.

    Parameters
    ----------
    entries
        Entry dicts from :func:`load_daily_monitoring`.

    Returns
    -------
    str
        Formatted prompt section, or empty string if no data.
    """
    if not entries:
        return ""

    # Infer iso3 from entries if possible (all same country)
    total = len(entries)

    parts: list[str] = [
        f"ACAPS DAILY MONITORING (last 30 days):",
        f"{total} entries. Weekly picks marked with *.",
    ]

    for entry in entries:
        pick = "* " if entry.get("weekly_pick") else "  "
        entry_date = entry.get("date", "?")
        dev = entry.get("latest_developments", "")
        parts.append(f"{pick}[{entry_date}] {dev}")

    parts.append(
        "\nThese are analyst-curated situational updates from ACAPS "
        "daily monitoring."
    )

    return "\n".join(parts)


def format_daily_monitoring_for_spd(entries: list[dict] | None) -> str:
    """Compact Daily Monitoring block for SPD prompts.

    Only includes weekly picks, max 5 entries.

    Parameters
    ----------
    entries
        Entry dicts from :func:`load_daily_monitoring`.

    Returns
    -------
    str
        Compact prompt section, or empty string if no data.
    """
    if not entries:
        return ""

    # Prefer weekly picks; fall back to all entries
    picks = [e for e in entries if e.get("weekly_pick")]
    display = (picks or entries)[:5]

    parts: list[str] = ["ACAPS MONITORING:"]
    for entry in display:
        entry_date = entry.get("date", "?")
        dev = entry.get("latest_developments", "")
        # Shorten for SPD
        if len(dev) > 150:
            dev = dev[:150].rsplit(" ", 1)[0] + "..."
        parts.append(f"- {entry_date}: {dev}")

    return "\n".join(parts)


# ============================================================
# Humanitarian Access
# ============================================================


def fetch_humanitarian_access(
    iso3: str,
) -> dict | None:
    """Fetch ACAPS Humanitarian Access data for a country.

    Tries the current month first, then steps back month by month until
    data is found (updates are infrequent, approximately every 6 months).

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.

    Returns
    -------
    dict | None
        Structured access data, or None on error / no data.
    """
    token = _get_acaps_token()
    if token is None:
        return None

    if isinstance(iso3, list):
        iso3 = iso3[0] if iso3 else ""
    iso3 = str(iso3).strip().upper()

    # Try month by month going back up to 12 months
    for label in _month_labels_back(12):
        results = _fetch_paginated(
            f"/api/v1/humanitarian-access/{label}/",
            params={"iso3": iso3},
            max_pages=1,
            token=token,
        )
        if results:
            record = _pick_country_crisis(results)
            score = _safe_float(
                record.get("overall_score")
                or record.get("access_score")
                or record.get("score")
            )
            if score is None:
                continue

            crisis_id = record.get("crisis_id", "")

            # Determine staleness (>8 months old)
            stale = False
            try:
                # Parse MmmYYYY
                snap_date = datetime.strptime(label, "%b%Y")
                months_old = (
                    (date.today().year - snap_date.year) * 12
                    + date.today().month - snap_date.month
                )
                stale = months_old > 8
            except (ValueError, TypeError):
                pass

            return {
                "iso3": iso3,
                "crisis_id": crisis_id,
                "access_score": score,
                "access_category": _access_category(score),
                "snapshot_date": label,
                "stale": stale,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }

    log.info("No Humanitarian Access data found for %s", iso3)
    return None


def store_humanitarian_access(
    iso3: str,
    data: dict,
    db_url: str | None = None,
) -> None:
    """Upsert Humanitarian Access data into DuckDB.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.
    data
        Access dict as returned by :func:`fetch_humanitarian_access`.
    db_url
        Optional DuckDB URL override.
    """
    if not data:
        return

    try:
        from pythia.db.schema import connect, ensure_schema
    except Exception:
        log.debug("Pythia DB helpers unavailable -- skipping store.")
        return

    try:
        con = connect(read_only=False)
    except Exception:
        log.warning("Could not connect to DuckDB for Humanitarian Access store.")
        return

    try:
        ensure_schema(con)
        now = data.get("fetched_at", datetime.now(timezone.utc).isoformat())

        con.execute(
            """
            INSERT OR REPLACE INTO acaps_humanitarian_access
                (iso3, crisis_id, snapshot_date, access_score,
                 access_category, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                iso3.upper(),
                data.get("crisis_id", ""),
                data.get("snapshot_date", ""),
                data.get("access_score"),
                data.get("access_category", ""),
                now,
            ],
        )
    except Exception as exc:
        log.warning(
            "Failed to store Humanitarian Access for %s: %s", iso3, exc,
        )
    finally:
        con.close()


def load_humanitarian_access(
    iso3: str,
    db_url: str | None = None,
) -> dict | None:
    """Load Humanitarian Access data from DuckDB.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.
    db_url
        Optional DuckDB URL override.

    Returns
    -------
    dict | None
        Access data dict, or None if no data.
    """
    try:
        from resolver.db.duckdb_io import get_db, close_db
    except Exception:
        log.debug("DuckDB helpers unavailable -- skipping access load.")
        return None

    db_url = db_url or _db_url()

    try:
        con = get_db(db_url)
    except Exception:
        log.debug("Could not connect to DuckDB at %s", db_url)
        return None

    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            ).fetchall()
        ]
        if "acaps_humanitarian_access" not in tables:
            return None

        row = con.execute(
            """
            SELECT crisis_id, snapshot_date, access_score,
                   access_category, fetched_at
            FROM acaps_humanitarian_access
            WHERE iso3 = ?
            ORDER BY snapshot_date DESC
            LIMIT 1
            """,
            [iso3.upper()],
        ).fetchone()

        if not row:
            return None

        crisis_id, snapshot_date, access_score, access_category, fetched_at = row

        # Compute staleness
        stale = False
        try:
            snap_date = datetime.strptime(snapshot_date, "%b%Y")
            months_old = (
                (date.today().year - snap_date.year) * 12
                + date.today().month - snap_date.month
            )
            stale = months_old > 8
        except (ValueError, TypeError):
            pass

        return {
            "iso3": iso3.upper(),
            "crisis_id": crisis_id,
            "access_score": access_score,
            "access_category": access_category,
            "snapshot_date": snapshot_date,
            "stale": stale,
            "fetched_at": fetched_at,
        }
    except Exception as exc:
        log.warning(
            "Failed to load Humanitarian Access for %s: %s", iso3, exc,
        )
        return None
    finally:
        close_db(con)


def format_humanitarian_access_for_prompt(data: dict | None) -> str:
    """Format Humanitarian Access as a text block for triage prompts.

    Parameters
    ----------
    data
        Access dict from :func:`load_humanitarian_access`.

    Returns
    -------
    str
        Formatted prompt section, or empty string if no data.
    """
    if not data or data.get("access_score") is None:
        return ""

    iso3 = data.get("iso3", "")
    score = data["access_score"]
    cat = data.get("access_category", "")
    snap = data.get("snapshot_date", "")
    stale = data.get("stale", False)
    stale_note = " (DATA MAY BE STALE)" if stale else ""

    return (
        f"HUMANITARIAN ACCESS ({iso3}): Score {score}/5.0 ({cat}) "
        f"[as of {snap}]{stale_note}\n"
        "High access constraints mean worse humanitarian outcomes "
        "and less reliable data."
    )


# ============================================================
# Helpers
# ============================================================


def _safe_float(value: Any) -> float | None:
    """Convert a value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int:
    """Convert a value to int, defaulting to 0."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
