# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""FEWS NET Food Security (IPC Phase 3+) data loader and prompt formatters.

Reads FEWS NET Phase 3+ data from ``facts_resolved`` (ingested by the
Resolver's ``fewsnet_ipc`` connector) and provides formatted text blocks
for injection into RC, triage, and SPD prompts.

Public API
----------
- :func:`load_fewsnet_food_security` — load from DuckDB ``facts_resolved``
- :func:`format_fewsnet_for_prompt` — full text block for RC / triage prompts
- :func:`format_fewsnet_for_spd` — compact block for SPD prompts
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def load_fewsnet_food_security(
    iso3: str,
    db_url: str | None = None,
) -> dict[str, Any] | None:
    """Load the most recent FEWS NET Phase 3+ data for *iso3*.

    Queries ``facts_resolved`` for metrics ``phase3plus_in_need`` (Current
    Situation) and ``phase3plus_projection`` (Most Likely), returning the
    most recent row for each.

    Returns a structured dict or *None* if no data is available.
    """
    try:
        from pythia.db import get_connection
    except Exception:
        logger.debug("Cannot import pythia.db — skipping FEWS NET load")
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
            ORDER BY as_of_date DESC
            """,
            [iso3.upper()],
        ).fetchall()
    except Exception as exc:
        logger.debug("FEWS NET query failed for %s: %s", iso3, exc)
        return None

    if not rows:
        return None

    # Take the most recent row for each metric
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

    # Format as_of_date to YYYY-MM
    current_ym = _format_ym(current_date)

    result: dict[str, Any] = {
        "iso3": iso3.upper(),
        "source": "FEWS NET",
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


def format_fewsnet_for_prompt(data: dict[str, Any] | None) -> str:
    """Format FEWS NET data as a full text block for RC / triage prompts."""
    if not data:
        return ""

    iso3 = data.get("iso3", "")
    lines = [
        f"FEWS NET FOOD SECURITY (IPC Phase 3+) — {iso3}:",
        "Source: FEWS NET Data Warehouse (fdw.fews.net)",
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
        lines.append("[WARNING: FEWS NET data >6 months old — treat with caution]")

    lines.append("")
    lines.append(
        "FEWS NET Phase 3+ estimates represent the humanitarian community's "
        "consensus on food insecurity outcomes. Treat as calibration anchors "
        "for PA forecasts involving food insecurity, drought, and "
        "conflict-driven displacement."
    )

    return "\n".join(lines)


def format_fewsnet_for_spd(data: dict[str, Any] | None) -> str:
    """Format FEWS NET data as a compact block for SPD prompts."""
    if not data:
        return ""

    iso3 = data.get("iso3", "")
    parts = [f"FEWS NET IPC PHASES ({iso3}):"]
    current_str = f"Current Phase 3+: {data['current_phase3plus']:,} (as of {data['current_as_of']})"

    if data.get("projected_phase3plus") is not None:
        current_str += f" | Projected: {data['projected_phase3plus']:,} [{data['trend']}]"

    parts.append(current_str)

    if data.get("stale"):
        parts.append("[WARNING: FEWS NET data >6 months old]")

    if data.get("projected_phase3plus") is not None:
        parts.append(
            f"CALIBRATION CHECK: FEWS NET projects {data['projected_phase3plus']:,} "
            f"people in Phase 3+ for {data['projected_as_of']}. If your PA forecast "
            f"for overlapping months implies significantly fewer people affected, "
            f"reconcile the discrepancy or explain why."
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
