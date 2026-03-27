# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Unified food security data loader and prompt formatters.

Abstracts over FEWS NET and IPC data sources, both stored in
``facts_resolved`` with metric ``phase3plus_in_need`` (resolution)
and ``phase3plus_projection`` (prompt context).

Routing:
  1. Try FEWS NET (publisher='FEWS NET') — primary for ~48 monitored countries
  2. If no FEWS NET data, try IPC (publisher='IPC') — supplementary
  3. Return None if neither has data

Public API
----------
- :func:`load_food_security` — load from DuckDB ``facts_resolved``
- :func:`format_food_security_for_prompt` — full text block for RC/triage prompts
- :func:`format_food_security_for_spd` — compact block for SPD prompts
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_SOURCE_URLS = {
    "FEWS NET": "FEWS NET Data Warehouse (fdw.fews.net)",
    "IPC": "IPC API (api.ipcinfo.org)",
}


def load_food_security(
    iso3: str,
    db_url: str | None = None,
) -> dict[str, Any] | None:
    """Load the most recent food security data for *iso3*.

    Tries FEWS NET first (publisher='FEWS NET'), then IPC (publisher='IPC').
    Returns a unified dict with a ``source`` field indicating the data origin.
    """
    result = _try_source(iso3, "FEWS NET", db_url)
    if result:
        return result

    result = _try_source(iso3, "IPC", db_url)
    return result


def format_food_security_for_prompt(data: dict[str, Any] | None) -> str:
    """Format food security data as a full text block for RC/triage prompts.

    Source-aware: header shows FEWS NET or IPC depending on data origin.
    """
    if not data:
        return ""

    iso3 = data.get("iso3", "")
    source = data.get("source", "FEWS NET")
    source_detail = _SOURCE_URLS.get(source, source)

    lines = [
        f"FOOD SECURITY (IPC Phase 3+) — {iso3}:",
        f"Source: {source_detail}",
        f"Current Situation: {data['current_phase3plus']:,} people in Phase 3+ "
        f"(Crisis or worse) as of {data['current_as_of']}",
    ]

    if data.get("projected_phase3plus") is not None:
        lines.append(
            f"Most Likely Projection: {data['projected_phase3plus']:,} people "
            f"in Phase 3+ as of {data['projected_as_of']}"
        )
        delta = data["projected_phase3plus"] - data["current_phase3plus"]
        lines.append(f"Trend: {data['trend']} (Phase 3+ {delta:+,} people vs current)")

    if data.get("stale"):
        lines.append(f"[WARNING: {source} data >6 months old — treat with caution]")

    lines.append("")
    lines.append(
        f"{source} Phase 3+ estimates represent the humanitarian community's "
        "consensus on food insecurity outcomes. Treat as calibration anchors "
        "for PA forecasts involving food insecurity, drought, and "
        "conflict-driven displacement."
    )

    return "\n".join(lines)


def format_food_security_for_spd(data: dict[str, Any] | None) -> str:
    """Format food security data as a compact block for SPD prompts.

    Source-aware: header shows FEWS NET or IPC depending on data origin.
    """
    if not data:
        return ""

    iso3 = data.get("iso3", "")
    source = data.get("source", "FEWS NET")

    parts = [f"{source} IPC PHASES ({iso3}):"]
    current_str = f"Current Phase 3+: {data['current_phase3plus']:,} (as of {data['current_as_of']})"

    if data.get("projected_phase3plus") is not None:
        current_str += f" | Projected: {data['projected_phase3plus']:,} [{data['trend']}]"

    parts.append(current_str)

    if data.get("stale"):
        parts.append(f"[WARNING: {source} data >6 months old]")

    if data.get("projected_phase3plus") is not None:
        parts.append(
            f"CALIBRATION CHECK: {source} projects {data['projected_phase3plus']:,} "
            f"people in Phase 3+ for {data['projected_as_of']}. If your PA forecast "
            f"for overlapping months implies significantly fewer people affected, "
            f"reconcile the discrepancy or explain why."
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _try_source(
    iso3: str,
    publisher: str,
    db_url: str | None = None,
) -> dict[str, Any] | None:
    """Query facts_resolved for a specific publisher's Phase 3+ data."""
    try:
        from pythia.db import get_connection
    except Exception:
        logger.debug("Cannot import pythia.db — skipping food security load")
        return None

    try:
        con = get_connection(db_url)
        rows = con.execute(
            """
            SELECT value, as_of_date, metric, series_semantics
            FROM facts_resolved
            WHERE iso3 = ?
              AND hazard_code = 'DR'
              AND metric IN ('phase3plus_in_need', 'phase3plus_projection')
              AND publisher = ?
            ORDER BY as_of_date DESC
            """,
            [iso3.upper(), publisher],
        ).fetchall()
    except Exception as exc:
        logger.debug("Food security query failed for %s (publisher=%s): %s", iso3, publisher, exc)
        return None

    if not rows:
        return None

    current_row = None
    projected_row = None
    for row in rows:
        value, as_of_date, metric, _semantics = row
        if metric == "phase3plus_in_need" and current_row is None:
            current_row = row
        elif metric == "phase3plus_projection" and projected_row is None:
            projected_row = row
        if current_row and projected_row:
            break

    if current_row is None:
        return None

    current_value = current_row[0]
    current_date = current_row[1]
    current_ym = _format_ym(current_date)

    result: dict[str, Any] = {
        "iso3": iso3.upper(),
        "source": publisher,
        "current_phase3plus": int(current_value) if current_value is not None else 0,
        "current_as_of": current_ym,
        "projected_phase3plus": None,
        "projected_as_of": None,
        "trend": "stable",
        "stale": _is_stale(current_date),
    }

    if projected_row is not None:
        proj_value = projected_row[0]
        proj_date = projected_row[1]
        result["projected_phase3plus"] = int(proj_value) if proj_value is not None else None
        result["projected_as_of"] = _format_ym(proj_date)

        if result["projected_phase3plus"] is not None and result["current_phase3plus"]:
            delta = result["projected_phase3plus"] - result["current_phase3plus"]
            if delta > 0:
                result["trend"] = "worsening"
            elif delta < 0:
                result["trend"] = "improving"

    return result


def _format_ym(dt: Any) -> str:
    """Convert a date/datetime/string to YYYY-MM format."""
    if dt is None:
        return "unknown"
    if isinstance(dt, str):
        return dt[:7] if len(dt) >= 7 else dt
    if hasattr(dt, "strftime"):
        return dt.strftime("%Y-%m")
    return str(dt)[:7]


def _is_stale(dt: Any) -> bool:
    """Return True if *dt* is more than 6 months old."""
    if dt is None:
        return True
    try:
        if isinstance(dt, str):
            from datetime import date as _date
            parts = dt[:10].split("-")
            dt = _date(int(parts[0]), int(parts[1]), int(parts[2]))
        now = datetime.now(timezone.utc).date()
        if hasattr(dt, "date"):
            dt = dt.date()
        delta_days = (now - dt).days
        return delta_days > 180
    except Exception:
        return True
