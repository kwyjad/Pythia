# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IPC Food Security Phase Classification connector.

Fetches current and projected IPC phase population data from the IPC API,
stores it in DuckDB, and provides formatted text blocks for prompt injection.

IPC classifies food insecurity on a 5-phase scale (Minimal → Famine).
Phase 3+ ("Crisis or worse") is the primary metric for humanitarian need.

Public API
----------
- :func:`fetch_ipc_phases` — fetch from the IPC API
- :func:`store_ipc_phases` — upsert into DuckDB
- :func:`load_ipc_phases` — load from DuckDB
- :func:`format_ipc_for_prompt` — full triage / RC prompt block
- :func:`format_ipc_for_spd` — compact SPD prompt block with calibration check
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

IPC_API_BASE = "https://api.ipcinfo.org"

_STALENESS_DAYS = 180
_MAX_RETRIES = 2

# IPC uses ISO alpha-2 country codes; Pythia uses ISO3.
# Mapping covers countries typically analysed by IPC / Cadre Harmonisé.
_ISO3_TO_ISO2: dict[str, str] = {
    "AFG": "AF", "AGO": "AO", "BDI": "BI", "BEN": "BJ", "BFA": "BF",
    "BGD": "BD", "CAF": "CF", "CMR": "CM", "COD": "CD", "COG": "CG",
    "CIV": "CI", "DJI": "DJ", "ERI": "ER", "ETH": "ET", "GHA": "GH",
    "GIN": "GN", "GMB": "GM", "GTM": "GT", "GNB": "GW", "HTI": "HT",
    "HND": "HN", "IRQ": "IQ", "KEN": "KE", "LAO": "LA", "LBN": "LB",
    "LBR": "LR", "LBY": "LY", "MDG": "MG", "MLI": "ML", "MMR": "MM",
    "MOZ": "MZ", "MRT": "MR", "MWI": "MW", "NER": "NE", "NGA": "NG",
    "NPL": "NP", "PAK": "PK", "PNG": "PG", "PSE": "PS", "RWA": "RW",
    "SDN": "SD", "SEN": "SN", "SLE": "SL", "SLV": "SV", "SOM": "SO",
    "SSD": "SS", "SWZ": "SZ", "TCD": "TD", "TGO": "TG", "TZA": "TZ",
    "UGA": "UG", "UKR": "UA", "YEM": "YE", "ZAF": "ZA", "ZMB": "ZM",
    "ZWE": "ZW",
}


# ---------------------------------------------------------------------------
# DB URL resolution (mirrors pythia/acled_political.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso3_to_iso2(iso3: str) -> str | None:
    """Convert ISO 3166-1 alpha-3 to alpha-2.  Returns None if unknown."""
    return _ISO3_TO_ISO2.get(iso3.upper())


def _safe_int(value: Any) -> int:
    """Convert a value to int, defaulting to 0."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _compute_trend(
    current_phase3plus: int | None,
    projected_phase3plus: int | None,
) -> str:
    """Return 'worsening', 'improving', or 'stable'."""
    if current_phase3plus is None or projected_phase3plus is None:
        return "stable"
    if projected_phase3plus > current_phase3plus:
        return "worsening"
    if projected_phase3plus < current_phase3plus:
        return "improving"
    return "stable"


def _is_stale(analysis_date_str: str | None) -> bool:
    """Return True if the analysis date is more than _STALENESS_DAYS old."""
    if not analysis_date_str:
        return False
    try:
        analysis_date = datetime.strptime(
            analysis_date_str[:10], "%Y-%m-%d",
        ).date()
        return (date.today() - analysis_date).days > _STALENESS_DAYS
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# API fetching
# ---------------------------------------------------------------------------

def _get_api_key() -> str | None:
    """Read the IPC API key from the environment."""
    key = os.getenv("IPC_API_KEY", "").strip()
    return key or None


def _fetch_json(
    path: str,
    params: dict[str, Any] | None = None,
) -> Any | None:
    """GET an IPC API endpoint with simple retry on transient errors.

    Returns the parsed JSON response or None on failure.
    """
    url = f"{IPC_API_BASE}{path}"

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
        except requests.RequestException as exc:
            log.warning(
                "IPC API network error (attempt %d/%d): %s",
                attempt, _MAX_RETRIES, exc,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            return None

        if resp.status_code == 429 or resp.status_code >= 500:
            log.warning(
                "IPC API HTTP %d (attempt %d/%d)",
                resp.status_code, attempt, _MAX_RETRIES,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            return None

        if resp.status_code != 200:
            log.warning("IPC API HTTP %d for %s", resp.status_code, path)
            return None

        try:
            return resp.json()
        except ValueError:
            log.warning("IPC API response was not valid JSON for %s", path)
            return None

    return None


def _parse_population_response(
    payload: Any,
    iso3: str,
    include_projections: bool,
) -> dict | None:
    """Parse the /population response into a structured dict.

    The IPC API returns a list of analysis objects.  Each object contains
    phase population fields with suffixes indicating the period type:
    - no suffix → current
    - ``_projected`` → first projection
    - ``_second_projected`` → second projection

    We take the most recent analysis (last in list or highest id) and
    extract current + projected phase populations.
    """
    if not payload:
        return None

    # The response may be a list of analysis records or a single dict.
    records = payload if isinstance(payload, list) else [payload]
    if not records:
        return None

    # Pick the most recent analysis — typically the last element, but
    # sort by analysis id or date for safety.
    record = records[-1]

    analysis_id = str(record.get("id") or record.get("analysis_id") or "")
    analysis_date = str(record.get("date") or record.get("analysis_date") or "")
    analysis_period = str(record.get("period") or record.get("analysis_period") or "")
    projection_period = str(
        record.get("projected_period")
        or record.get("projection_period")
        or ""
    ) or None

    total_pop = _safe_int(
        record.get("total_population")
        or record.get("population")
        or record.get("estimated_population")
    )

    # --- Current phase populations ---
    c1 = _safe_int(record.get("phase1_population") or record.get("current_phase1"))
    c2 = _safe_int(record.get("phase2_population") or record.get("current_phase2"))
    c3 = _safe_int(record.get("phase3_population") or record.get("current_phase3"))
    c4 = _safe_int(record.get("phase4_population") or record.get("current_phase4"))
    c5 = _safe_int(record.get("phase5_population") or record.get("current_phase5"))
    c3plus = c3 + c4 + c5

    # If total_pop is zero, try to sum the phases
    if total_pop == 0:
        total_pop = c1 + c2 + c3plus

    c3plus_pct = round(c3plus / total_pop * 100, 1) if total_pop > 0 else 0.0

    current = {
        "phase1": c1,
        "phase2": c2,
        "phase3": c3,
        "phase4": c4,
        "phase5": c5,
        "phase3plus": c3plus,
        "phase3plus_pct": c3plus_pct,
    }

    # --- Projected phase populations ---
    projected = None
    if include_projections:
        p3 = _safe_int(
            record.get("phase3_population_projected")
            or record.get("projected_phase3")
        )
        p4 = _safe_int(
            record.get("phase4_population_projected")
            or record.get("projected_phase4")
        )
        p5 = _safe_int(
            record.get("phase5_population_projected")
            or record.get("projected_phase5")
        )
        p3plus = p3 + p4 + p5

        if p3plus > 0:
            p3plus_pct = round(p3plus / total_pop * 100, 1) if total_pop > 0 else 0.0
            projected = {
                "phase3": p3,
                "phase4": p4,
                "phase5": p5,
                "phase3plus": p3plus,
                "phase3plus_pct": p3plus_pct,
                "projection_period": projection_period,
            }

    # --- Phase 5 areas ---
    areas_in_phase5: list[str] = []
    areas_raw = record.get("areas_in_phase5") or record.get("phase5_areas")
    if isinstance(areas_raw, list):
        areas_in_phase5 = [str(a) for a in areas_raw if a]
    elif isinstance(areas_raw, str) and areas_raw.strip():
        try:
            parsed = json.loads(areas_raw)
            if isinstance(parsed, list):
                areas_in_phase5 = [str(a) for a in parsed if a]
        except (json.JSONDecodeError, TypeError):
            areas_in_phase5 = [s.strip() for s in areas_raw.split(",") if s.strip()]

    # --- Trend ---
    proj_p3plus = projected["phase3plus"] if projected else None
    trend = _compute_trend(c3plus, proj_p3plus)

    stale = _is_stale(analysis_date if analysis_date else None)

    return {
        "iso3": iso3.upper(),
        "analysis_id": analysis_id,
        "analysis_date": analysis_date,
        "analysis_period": analysis_period,
        "projection_period": projection_period,
        "total_population": total_pop,
        "current": current,
        "projected": projected,
        "trend": trend,
        "areas_in_phase5": areas_in_phase5,
        "stale": stale,
    }


# ---------------------------------------------------------------------------
# Public: fetch
# ---------------------------------------------------------------------------

def fetch_ipc_phases(
    iso3: str,
    include_projections: bool = True,
) -> dict | None:
    """Fetch the most recent IPC acute analysis for a country.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.
    include_projections
        Whether to include projected phase populations.

    Returns
    -------
    dict | None
        Structured IPC data dict, or ``None`` if unavailable.
    """
    api_key = _get_api_key()
    if not api_key:
        log.warning("IPC_API_KEY not set — cannot fetch IPC phases.")
        return None

    iso2 = _iso3_to_iso2(iso3)
    if not iso2:
        log.debug("No ISO2 mapping for %s — skipping IPC fetch.", iso3)
        return None

    params: dict[str, Any] = {
        "country": iso2,
        "type": "A",
        "key": api_key,
    }

    payload = _fetch_json("/population", params=params)
    if payload is None:
        return None

    return _parse_population_response(payload, iso3, include_projections)


# ---------------------------------------------------------------------------
# DuckDB storage
# ---------------------------------------------------------------------------

def store_ipc_phases(
    iso3: str,
    ipc_data: dict,
    db_url: str | None = None,
) -> None:
    """Upsert IPC phase data into DuckDB.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.
    ipc_data
        Dict as returned by :func:`fetch_ipc_phases`.
    db_url
        Optional DuckDB URL override.
    """
    if not ipc_data:
        return

    try:
        from pythia.db.schema import connect, ensure_schema
    except Exception:
        log.debug("Pythia DB helpers unavailable — skipping IPC store.")
        return

    try:
        con = connect(read_only=False)
    except Exception:
        log.warning("Could not connect to DuckDB for IPC phases store.")
        return

    try:
        ensure_schema(con)
        now = datetime.now(timezone.utc).isoformat()

        current = ipc_data.get("current") or {}
        projected = ipc_data.get("projected") or {}
        areas = ipc_data.get("areas_in_phase5") or []

        con.execute(
            """
            INSERT OR REPLACE INTO ipc_phases
                (iso3, analysis_id, analysis_date, analysis_period,
                 projection_period, total_population,
                 current_phase1, current_phase2, current_phase3,
                 current_phase4, current_phase5,
                 current_phase3plus, current_phase3plus_pct,
                 projected_phase3, projected_phase4, projected_phase5,
                 projected_phase3plus, projected_phase3plus_pct,
                 projected_period, trend, areas_in_phase5, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                iso3.upper(),
                ipc_data.get("analysis_id", ""),
                ipc_data.get("analysis_date", ""),
                ipc_data.get("analysis_period", ""),
                ipc_data.get("projection_period") or "",
                ipc_data.get("total_population", 0),
                current.get("phase1", 0),
                current.get("phase2", 0),
                current.get("phase3", 0),
                current.get("phase4", 0),
                current.get("phase5", 0),
                current.get("phase3plus", 0),
                current.get("phase3plus_pct", 0.0),
                projected.get("phase3", 0),
                projected.get("phase4", 0),
                projected.get("phase5", 0),
                projected.get("phase3plus", 0),
                projected.get("phase3plus_pct", 0.0),
                projected.get("projection_period") or "",
                ipc_data.get("trend", "stable"),
                json.dumps(areas) if areas else "[]",
                now,
            ],
        )
    except Exception as exc:
        log.warning("Failed to store IPC phases for %s: %s", iso3, exc)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_ipc_phases(
    iso3: str,
    db_url: str | None = None,
) -> dict | None:
    """Load the most recent IPC analysis for a country from DuckDB.

    Parameters
    ----------
    iso3
        ISO 3166-1 alpha-3 country code.
    db_url
        Optional DuckDB URL override.

    Returns
    -------
    dict | None
        Structured IPC data dict, or ``None`` if no data available.
    """
    try:
        from resolver.db.duckdb_io import get_db, close_db
    except Exception:
        log.debug("DuckDB helpers unavailable — skipping IPC phases load.")
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
        if "ipc_phases" not in tables:
            return None

        row = con.execute(
            """
            SELECT analysis_id, analysis_date, analysis_period,
                   projection_period, total_population,
                   current_phase1, current_phase2, current_phase3,
                   current_phase4, current_phase5,
                   current_phase3plus, current_phase3plus_pct,
                   projected_phase3, projected_phase4, projected_phase5,
                   projected_phase3plus, projected_phase3plus_pct,
                   projected_period, trend, areas_in_phase5
            FROM ipc_phases
            WHERE iso3 = ?
            ORDER BY analysis_date DESC
            LIMIT 1
            """,
            [iso3.upper()],
        ).fetchone()
    except Exception as exc:
        log.warning("Failed to load IPC phases for %s: %s", iso3, exc)
        return None
    finally:
        close_db(con)

    if not row:
        return None

    (
        analysis_id, analysis_date, analysis_period,
        projection_period, total_population,
        c1, c2, c3, c4, c5, c3plus, c3plus_pct,
        p3, p4, p5, p3plus, p3plus_pct,
        projected_period, trend, areas_raw,
    ) = row

    # Reconstruct nested dict
    current = {
        "phase1": _safe_int(c1),
        "phase2": _safe_int(c2),
        "phase3": _safe_int(c3),
        "phase4": _safe_int(c4),
        "phase5": _safe_int(c5),
        "phase3plus": _safe_int(c3plus),
        "phase3plus_pct": float(c3plus_pct) if c3plus_pct else 0.0,
    }

    projected = None
    p3_int = _safe_int(p3)
    p4_int = _safe_int(p4)
    p5_int = _safe_int(p5)
    p3plus_int = _safe_int(p3plus)
    if p3plus_int > 0:
        projected = {
            "phase3": p3_int,
            "phase4": p4_int,
            "phase5": p5_int,
            "phase3plus": p3plus_int,
            "phase3plus_pct": float(p3plus_pct) if p3plus_pct else 0.0,
            "projection_period": projected_period or projection_period or None,
        }

    # Parse areas
    areas_in_phase5: list[str] = []
    if areas_raw:
        try:
            parsed = json.loads(areas_raw)
            if isinstance(parsed, list):
                areas_in_phase5 = [str(a) for a in parsed if a]
        except (json.JSONDecodeError, TypeError):
            pass

    stale = _is_stale(analysis_date)

    return {
        "iso3": iso3.upper(),
        "analysis_id": analysis_id or "",
        "analysis_date": analysis_date or "",
        "analysis_period": analysis_period or "",
        "projection_period": projection_period or None,
        "total_population": _safe_int(total_population),
        "current": current,
        "projected": projected,
        "trend": trend or "stable",
        "areas_in_phase5": areas_in_phase5,
        "stale": stale,
    }


# ---------------------------------------------------------------------------
# Prompt formatters
# ---------------------------------------------------------------------------

def format_ipc_for_prompt(ipc_data: dict | None) -> str:
    """Format IPC data as a full text block for triage / RC prompts.

    Parameters
    ----------
    ipc_data
        Dict from :func:`load_ipc_phases` or :func:`fetch_ipc_phases`.

    Returns
    -------
    str
        Formatted prompt section, or empty string if no data.
    """
    if not ipc_data:
        return ""

    iso3 = ipc_data.get("iso3", "")
    current = ipc_data.get("current") or {}
    projected = ipc_data.get("projected")
    stale = ipc_data.get("stale", False)

    parts: list[str] = []
    parts.append(f"IPC FOOD SECURITY CLASSIFICATION ({iso3}):")

    if stale:
        parts.append("[WARNING: IPC ANALYSIS >6 MONTHS OLD]")

    analysis_period = ipc_data.get("analysis_period", "")
    projection_period = ipc_data.get("projection_period") or ""
    period_line = f"Analysis: {analysis_period}"
    if projection_period:
        period_line += f" | Projection: {projection_period}"
    parts.append(period_line)

    parts.append("")
    parts.append("Current situation:")
    parts.append(f"  Phase 3 (Crisis):    {current.get('phase3', 0):,} people")
    parts.append(f"  Phase 4 (Emergency): {current.get('phase4', 0):,} people")
    parts.append(f"  Phase 5 (Famine):    {current.get('phase5', 0):,} people")
    parts.append(
        f"  Total Phase 3+:      {current.get('phase3plus', 0):,} "
        f"({current.get('phase3plus_pct', 0.0):.1f}% of analyzed population)"
    )

    if projected:
        proj_period = projected.get("projection_period") or projection_period
        parts.append("")
        parts.append(f"Projected situation ({proj_period}):")
        parts.append(
            f"  Phase 3+: {projected.get('phase3plus', 0):,} "
            f"({projected.get('phase3plus_pct', 0.0):.1f}%)"
        )
        delta = projected.get("phase3plus", 0) - current.get("phase3plus", 0)
        trend = ipc_data.get("trend", "stable")
        parts.append(f"  Trend: {trend} (Phase 3+ {delta:+,} people vs current)")

    areas = ipc_data.get("areas_in_phase5") or []
    if areas:
        parts.append("")
        parts.append(f"FAMINE (Phase 5) detected in: {', '.join(areas)}")

    parts.append("")
    parts.append(
        "IPC projections are institutional expert estimates. They represent the\n"
        "humanitarian community's consensus on expected food security outcomes.\n"
        "Treat Phase 3+ population counts as calibration anchors for PA forecasts\n"
        "involving food insecurity, drought, and conflict-driven displacement."
    )

    return "\n".join(parts)


def format_ipc_for_spd(ipc_data: dict | None) -> str:
    """Compact IPC block with calibration check for SPD prompts.

    Parameters
    ----------
    ipc_data
        Dict from :func:`load_ipc_phases` or :func:`fetch_ipc_phases`.

    Returns
    -------
    str
        Compact prompt section with calibration instruction, or empty string.
    """
    if not ipc_data:
        return ""

    iso3 = ipc_data.get("iso3", "")
    current = ipc_data.get("current") or {}
    projected = ipc_data.get("projected")
    stale = ipc_data.get("stale", False)

    c3plus = current.get("phase3plus", 0)
    c3plus_pct = current.get("phase3plus_pct", 0.0)

    parts: list[str] = []

    header = f"IPC PHASES ({iso3}): Current Phase 3+: {c3plus:,} ({c3plus_pct:.1f}%)"
    if projected:
        p3plus = projected.get("phase3plus", 0)
        p3plus_pct = projected.get("phase3plus_pct", 0.0)
        trend = ipc_data.get("trend", "stable")
        header += f" | Projected: {p3plus:,} ({p3plus_pct:.1f}%) [{trend}]"
    parts.append(header)

    if stale:
        parts.append("[WARNING: IPC ANALYSIS >6 MONTHS OLD]")

    analysis_period = ipc_data.get("analysis_period", "")
    projection_period = ipc_data.get("projection_period") or ""
    period_line = f"Analysis period: {analysis_period}"
    if projection_period:
        period_line += f" -> Projection: {projection_period}"
    parts.append(period_line)

    areas = ipc_data.get("areas_in_phase5") or []
    if areas:
        parts.append(f"FAMINE (Phase 5) areas: {', '.join(areas)}")

    # Calibration check — only if projected data exists
    if projected:
        proj_p3plus = projected.get("phase3plus", 0)
        proj_period = projected.get("projection_period") or projection_period
        parts.append("")
        parts.append(
            f"CALIBRATION CHECK: IPC projects {proj_p3plus:,} people in "
            f"Phase 3+ for {proj_period}.\n"
            "If your PA forecast for overlapping months implies significantly fewer "
            "people affected than\nIPC Phase 3+ estimates, reconcile the discrepancy "
            "or explain why your estimate differs\n"
            "(e.g., different metric scope, PA vs food insecurity)."
        )

    return "\n".join(parts)
