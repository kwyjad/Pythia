# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Resolver history / base-rate loaders and prompt formatters (moved verbatim from cli.py)."""

from __future__ import annotations

import csv
import functools
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import duckdb

from pythia.db.schema import connect

from .aggregators import _pythia_db_path_from_config
from .month_utils import _parse_month_key, _sanitize_month_series


COUNTRIES_CSV = Path(__file__).resolve().parents[1] / "resolver" / "data" / "countries.csv"


NATURAL_HAZARD_CODES = {"FL", "DR", "TC", "HW"}


IDMC_HZ_MAP = {"ACE", "DI"}


def _load_idmc_conflict_flow_history_summary(
    con,
    iso3: str,
    hazard_code: str,
) -> Dict[str, Any]:
    """
    Best-effort IDMC/DTM conflict displacement history for use in _build_history_summary.

    Reads per-month flows from facts_deltas for conflict hazards:
      - metric ∈ {new_displacements, idp_displacement_new_dtm, idp_displacement_flow_idmc}
      - series_semantics = 'new'
      - iso3 and hazard_code filtered.

    Returns a summary dict with:
      - source: "IDMC"
      - history_length_months: number of months with non-zero flow
      - recent_mean: mean over the last up-to-6 months
      - recent_max: max over the full window
      - trend: crude "increasing" / "decreasing" / "flat" based on first vs last month
      - last_6m_values: list of {"ym", "value"} for up to the last 6 months
      - data_quality: "medium" (we’re using delta flows only)
      - notes: short explanation of what we used.
    """
    iso3_up = (iso3 or "").upper().strip()
    hz_up = (hazard_code or "").upper().strip()

    if not iso3_up or not hz_up:
        return {
            "source": "IDMC",
            "history_length_months": 0,
            "recent_mean": None,
            "recent_max": None,
            "trend": "uncertain",
            "last_6m_values": [],
            "data_quality": "low",
            "notes": "Missing iso3 or hazard_code for IDMC conflict PA history.",
        }

    try:
        rows = con.execute(
            """
            SELECT
                ym,
                SUM(
                    CASE
                        WHEN lower(metric) = 'new_displacements'
                             THEN COALESCE(value_new, 0)
                        WHEN lower(metric) = 'idp_displacement_new_dtm'
                             THEN COALESCE(value_new, 0)
                        WHEN lower(metric) = 'idp_displacement_flow_idmc'
                             THEN COALESCE(value_new, 0)
                        ELSE 0
                    END
                ) AS flow_value
            FROM facts_deltas
            WHERE upper(iso3) = ?
              AND COALESCE(NULLIF(upper(hazard_code), ''), 'ACE') IN (?, 'IDU')
              AND lower(series_semantics) = 'new'
              AND lower(metric) IN (
                'new_displacements',
                'idp_displacement_new_dtm',
                'idp_displacement_flow_idmc'
              )
            GROUP BY ym
            ORDER BY ym
            """,
            [iso3_up, hz_up],
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "IDMC conflict PA history query failed for %s/%s: %s",
            iso3_up,
            hz_up,
            exc,
        )
        return {
            "source": "IDMC",
            "history_length_months": 0,
            "recent_mean": None,
            "recent_max": None,
            "trend": "uncertain",
            "last_6m_values": [],
            "data_quality": "low",
            "notes": f"IDMC history unavailable due to query error: {type(exc).__name__}",
        }

    logging.debug("IDMC displacement query for %s/%s: %d rows", iso3_up, hz_up, len(rows))

    if not rows:
        return {
            "source": "IDMC",
            "history_length_months": 0,
            "recent_mean": None,
            "recent_max": None,
            "trend": "uncertain",
            "last_6m_values": [],
            "data_quality": "low",
            "notes": "No IDMC/DTM displacement flow history found in facts_deltas.",
        }

    month_to_value: Dict[str, float] = {}
    for ym_val, flow_val in rows:
        try:
            v = float(flow_val or 0.0)
        except Exception:
            v = 0.0
        if v == 0:
            continue
        month_to_value[str(ym_val)] = v

    month_to_value, dropped_future, unparseable_keys = _sanitize_month_series(month_to_value)
    if not month_to_value:
        summary_empty: Dict[str, Any] = {
            "source": "IDMC",
            "history_length_months": 0,
            "recent_mean": None,
            "recent_max": None,
            "trend": "uncertain",
            "last_6m_values": [],
            "data_quality": "low",
            "notes": "IDMC displacement flows found but all were zero or filtered out.",
        }
        sanity_empty: Dict[str, Any] = {}
        if dropped_future:
            sanity_empty["dropped_future_months"] = dropped_future
        if unparseable_keys:
            sanity_empty["unparseable_month_keys"] = unparseable_keys
        if sanity_empty:
            summary_empty["_sanity"] = sanity_empty
        return summary_empty

    ordered_items = sorted(month_to_value.items())
    history: list[Dict[str, Any]] = [{"ym": k, "value": v} for k, v in ordered_items]
    values: list[float] = [float(v) for _k, v in ordered_items]

    n = len(history)
    last_6 = history[-6:]
    recent_window = values[-6:]
    recent_mean = sum(recent_window) / len(recent_window) if recent_window else None
    recent_max = max(values) if values else None

    trend = "flat"
    if len(values) >= 2:
        if values[-1] > values[0]:
            trend = "increasing"
        elif values[-1] < values[0]:
            trend = "decreasing"

    summary: Dict[str, Any] = {
        "source": "IDMC",
        "history_length_months": n,
        "recent_mean": recent_mean,
        "recent_max": recent_max,
        "trend": trend,
        "last_6m_values": last_6,
        "data_quality": "medium",
        "notes": (
            "IDMC/DTM monthly displacement flows (facts_deltas) used as the conflict PA base rate."
        ),
    }
    sanity: Dict[str, Any] = {}
    if dropped_future:
        sanity["dropped_future_months"] = dropped_future
    if unparseable_keys:
        sanity["unparseable_month_keys"] = unparseable_keys
    if sanity:
        summary["_sanity"] = sanity
    return summary


_HAZARD_DISPLAY_NAMES: Dict[str, str] = {
    "FL": "Flood",
    "DR": "Drought",
    "TC": "Tropical Cyclone",
    "HW": "Heat Wave",
    "ACE": "Armed Conflict",
    "DI": "Displacement Inflow",
}


def _build_gdacs_event_history(
    iso3: str,
    hazard_code: str,
) -> Dict[str, Any] | None:
    """Build GDACS event occurrence history for natural hazard base rate support.

    Queries facts_resolved for event_occurrence metric (binary 1/0) and
    alertlevel (Green/Orange/Red) for the given country + hazard.

    Only applicable for FL, DR, TC hazards.

    Returns None if no GDACS data exists for this country-hazard.
    """
    iso3_up = (iso3 or "").upper().strip()
    hz_up = (hazard_code or "").upper().strip()

    if hz_up not in ("FL", "DR", "TC"):
        return None

    con = connect(read_only=True)
    try:
        rows = con.execute(
            """
            SELECT ym, value, alertlevel
            FROM facts_resolved
            WHERE upper(iso3) = ?
              AND upper(hazard_code) = ?
              AND lower(metric) = 'event_occurrence'
            ORDER BY ym
            """,
            [iso3_up, hz_up],
        ).fetchall()
    except Exception:
        return None
    finally:
        con.close()

    if not rows:
        return None

    # Parse into structured records
    now_ym = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m")
    events: list[dict] = []
    for ym_raw, value, alertlevel in rows:
        ym = str(ym_raw)[:7] if ym_raw else ""
        if not ym or ym > now_ym:
            continue
        try:
            v = float(value or 0)
        except (TypeError, ValueError):
            v = 0.0
        events.append({
            "ym": ym,
            "occurred": v >= 1.0,
            "alertlevel": str(alertlevel or "").strip() or None,
        })

    if not events:
        return None

    events_sorted = sorted(events, key=lambda e: e["ym"])

    # Compute summary stats
    total_months = len(events_sorted)
    event_months = sum(1 for e in events_sorted if e["occurred"])
    event_rate = event_months / total_months if total_months > 0 else 0.0

    # Data range
    first_ym = events_sorted[0]["ym"]
    last_ym = events_sorted[-1]["ym"]

    # Alert level distribution (among event months only)
    alert_counts: Dict[str, int] = {}
    for e in events_sorted:
        if e["occurred"] and e["alertlevel"]:
            level = e["alertlevel"]
            alert_counts[level] = alert_counts.get(level, 0) + 1

    # Seasonal frequency: group by calendar month
    by_cal_month: Dict[int, Dict[str, int]] = {
        m: {"total": 0, "events": 0} for m in range(1, 13)
    }
    for e in events_sorted:
        parsed = _parse_month_key(e["ym"])
        if parsed:
            by_cal_month[parsed.month]["total"] += 1
            if e["occurred"]:
                by_cal_month[parsed.month]["events"] += 1

    seasonal: Dict[int, Dict[str, Any]] = {}
    for cal_month in range(1, 13):
        total = by_cal_month[cal_month]["total"]
        evts = by_cal_month[cal_month]["events"]
        seasonal[cal_month] = {
            "years_observed": total,
            "years_with_event": evts,
            "frequency_pct": round(evts / total * 100, 0) if total > 0 else 0.0,
        }

    # Recent 12 months
    recent_12 = events_sorted[-12:] if len(events_sorted) >= 12 else events_sorted[:]

    return {
        "type": "gdacs_event_history",
        "iso3": iso3_up,
        "hazard_code": hz_up,
        "data_range": f"{first_ym} to {last_ym}",
        "total_months": total_months,
        "event_months": event_months,
        "event_rate_pct": round(event_rate * 100, 1),
        "alert_distribution": alert_counts,
        "seasonal": seasonal,
        "recent_12": recent_12,
    }


def _format_base_rate_for_prompt(
    history_summary: Dict[str, Any],
    forecast_keys: list[str],
    iso3: str = "",
    hazard_code: str = "",
) -> str:
    """Format a base-rate dict into a human-readable prompt block.

    Handles seasonal_profile, conflict_trajectory, no_base_rate, and
    legacy dict formats (falls back to JSON dump).
    """
    summary_type = history_summary.get("type", "")
    iso3_up = (iso3 or "").upper().strip()
    hz_up = (hazard_code or "").upper().strip()

    country_name = _load_country_names().get(iso3_up, iso3_up)
    hazard_display = _HAZARD_DISPLAY_NAMES.get(hz_up, hz_up)

    # Determine which calendar months the forecast covers
    forecast_months: list[int] = []
    for k in forecast_keys:
        parsed = _parse_month_key(k)
        if parsed:
            forecast_months.append(parsed.month)

    _MONTH_NAMES = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December",
    }

    def _fmt(n: Any) -> str:
        """Format a number with comma separators, or return 'N/A'."""
        if n is None:
            return "N/A"
        try:
            return f"{int(n):,}"
        except (TypeError, ValueError):
            return str(n)

    if summary_type == "fewsnet_phase3":
        if history_summary.get("error"):
            # Loader-level failure (e.g. DB unavailable) — say so explicitly
            # instead of rendering a false "0% coverage" history.
            return (
                "RESOLVER HISTORY (FEWS NET IPC Phase 3+, Current Situation):\n"
                "History unavailable this run (data loader error — not evidence "
                "of missing FEWS NET coverage). Treat the base rate as uncertain."
            )
        source = history_summary.get("source", "FEWSNET_IPC")
        observed = history_summary.get("observed_months", 0)
        total = history_summary.get("history_length_months", 36)
        coverage = history_summary.get("coverage_pct", 0)
        recent_mean = history_summary.get("recent_mean")
        recent_max = history_summary.get("recent_max")
        trend = history_summary.get("trend", "unknown")
        trend_pct = history_summary.get("trend_pct")
        data_quality = history_summary.get("data_quality", "unknown")
        last_6m = history_summary.get("last_6m_values", [])

        lines = [
            "RESOLVER HISTORY (FEWS NET IPC Phase 3+, Current Situation):",
            f"Phase 3+ population reported in {observed} of the last {total} months ({coverage:.0f}% coverage).",
        ]

        # Latest value
        latest = None
        for entry in reversed(last_6m):
            if entry.get("value") is not None:
                latest = entry
                break
        if latest:
            lines.append(f"Latest value: {latest['ym']}: {_fmt(latest['value'])}")

        if recent_mean is not None:
            lines.append(f"Recent 6-month average (observed only): {_fmt(recent_mean)}")
        if recent_max is not None:
            lines.append(f"Peak (recent 6 months): {_fmt(recent_max)}")

        trend_str = trend
        if trend_pct is not None:
            trend_str = f"{trend} ({'+' if trend_pct > 0 else ''}{trend_pct:.0f}% over 12 months)"
        lines.append(f"Trend: {trend_str}")
        lines.append(f"Data quality: {data_quality} ({coverage:.0f}% monthly coverage)")

        lines.append("")
        lines.append("Recent values (null = no FEWS NET assessment that month):")

        # Format last 6 months in 2 rows of 3
        for i in range(0, len(last_6m), 3):
            chunk = last_6m[i:i+3]
            parts = []
            for entry in chunk:
                val = "null" if entry.get("value") is None else _fmt(entry["value"])
                parts.append(f"  {entry['ym']}: {val:<12}")
            lines.append(" | ".join(parts))

        lines.append("")
        lines.append(
            "Note: Months marked 'null' mean FEWS NET did not publish a Current "
            "Situation assessment for this country that month. This does NOT mean "
            "zero food insecurity. Your forecast should interpolate through gaps "
            "using the surrounding observed values and seasonal patterns."
        )

        return "\n".join(lines)

    if summary_type == "seasonal_profile":
        source = history_summary.get("source", "IFRC")
        data_range = history_summary.get("data_range", "unknown")
        years = history_summary.get("years_of_data", 0)
        months_data = history_summary.get("months", {})

        lines = [
            f"BASE RATE: {country_name} {hazard_display} ({hz_up}) people affected "
            f"— {source} data from {data_range} ({years} years)",
            "",
            "Seasonal profile for your forecast months:",
        ]

        # Find the longest month name for alignment
        month_names_for_forecast = [_MONTH_NAMES.get(m, str(m)) for m in forecast_months]
        max_name_len = max((len(n) for n in month_names_for_forecast), default=10)

        for cal_month in forecast_months:
            month_name = _MONTH_NAMES.get(cal_month, str(cal_month))
            m_data = months_data.get(cal_month) or months_data.get(str(cal_month)) or {}
            n_obs = m_data.get("n_observations", 0)
            lines.append(
                f"  {month_name + ':':<{max_name_len + 1}} "
                f"min={_fmt(m_data.get('min', 0))}, "
                f"max={_fmt(m_data.get('max', 0))}, "
                f"avg={_fmt(m_data.get('mean', 0))}, "
                f"median={_fmt(m_data.get('median', 0))} "
                f"({n_obs} obs)"
            )

        lines.append("")
        lines.append(
            "Note: These are historical monthly values across all available years. "
            "Use as your prior anchor, then update with current signals."
        )
        return "\n".join(lines)

    if summary_type == "conflict_trajectory":
        fat = history_summary.get("fatalities", {})
        disp = history_summary.get("displacements", {})

        lines = [
            f"BASE RATE: {country_name} {hazard_display} ({hz_up}) — recent trajectory",
        ]

        # Fatalities block
        lines.append("")
        if fat.get("last_month") is not None:
            last_f = fat["last_month"]
            trend_pct = fat.get("trend_pct")
            trend_dir = fat.get("trend_direction", "unknown")

            if trend_pct == "new_activity":
                trend_str = "new activity (no prior baseline)"
            elif trend_pct is not None:
                trend_str = f"{trend_dir} ({'+' if trend_pct > 0 else ''}{trend_pct}% vs prior 3-month window)"
            else:
                trend_str = "insufficient data for trend"

            lines.append(f"Fatalities ({fat.get('source', 'ACLED')}):")
            lines.append(f"  Last month ({last_f.get('ym', '?')}): {_fmt(last_f.get('value'))}")
            lines.append(f"  3-month avg: {_fmt(fat.get('trailing_3m_avg'))}/month")
            lines.append(f"  Trend: {trend_str}")
        else:
            note = fat.get("note", "No ACLED fatalities data available.")
            lines.append(f"Fatalities (ACLED): {note}")

        # Displacements block
        lines.append("")
        if disp.get("last_month") is not None:
            last_d = disp["last_month"]
            trend_pct = disp.get("trend_pct")
            trend_dir = disp.get("trend_direction", "unknown")

            if trend_pct == "new_activity":
                trend_str = "new activity (no prior baseline)"
            elif trend_pct is not None:
                trend_str = f"{trend_dir} ({'+' if trend_pct > 0 else ''}{trend_pct}% vs prior 3-month window)"
            else:
                trend_str = "insufficient data for trend"

            lines.append(f"Displacement ({disp.get('source', 'IDMC')}):")
            lines.append(f"  Last month ({last_d.get('ym', '?')}): {_fmt(last_d.get('value'))} new displacements")
            lines.append(f"  3-month avg: {_fmt(disp.get('trailing_3m_avg'))}/month")
            lines.append(f"  Trend: {trend_str}")
        else:
            note = disp.get("note", "No IDMC displacement data available.")
            lines.append(f"Displacement (IDMC): {note}")

        lines.append("")
        lines.append(
            "Note: Conflict base rates reflect recent trajectory, not seasonal patterns. "
            "Use as your prior anchor."
        )
        return "\n".join(lines)

    if summary_type == "no_base_rate":
        note = history_summary.get("note", "No base rate available for this hazard.")
        return (
            f"BASE RATE: {country_name} {hazard_display} ({hz_up})\n"
            f"{note}"
        )

    # Legacy fallback: dump as JSON (backward compat for any unrecognised type)
    return (
        "Resolver history summary (Resolver is one imperfect source; "
        "ACLED strong, IDMC short, IFRC Montandon may be sparse):\n"
        "```json\n"
        f"{json.dumps(history_summary, default=str, indent=2)}\n"
        "```"
    )


def _load_ifrc_pa_history(
    iso3: str,
    hazard_code: str,
    *,
    months: int = 36,
) -> Tuple[str, Dict[str, Any]]:
    """
    Load a 36-month IFRC Montandon 'people affected' history for a given
    ISO3 + Pythia hazard code.

    Queries facts_resolved (where IFRC Montandon connector data lands)
    for natural hazard PA metrics.
    """

    hz = (hazard_code or "").upper()

    try:
        con = duckdb.connect(_pythia_db_path_from_config(), read_only=True)
    except Exception:
        return "", {"error": "missing_db", "history_rows_detail": [], "summary_text": ""}

    try:
        rows = con.execute(
            """
            SELECT ym, value, COALESCE(source_id, '') AS source_id
            FROM facts_resolved
            WHERE iso3 = ?
              AND hazard_code = ?
              AND lower(metric) IN ('affected', 'in_need', 'pa')
            ORDER BY ym DESC
            LIMIT ?
            """,
            [iso3, hz, months],
        ).fetchall()
    except Exception as exc:
        con.close()
        return "", {
            "error": f"ifrc_query_error:{type(exc).__name__}",
            "history_rows_detail": [],
            "summary_text": "",
        }

    con.close()

    if not rows:
        return "", {"error": "no_history", "history_rows_detail": [], "summary_text": ""}

    history: List[Dict[str, Any]] = []
    values: List[float] = []
    for ym, pa_val, source_id in rows:
        ym_str = str(ym)
        v = float(pa_val or 0.0)
        history.append(
            {
                "ym": ym_str,
                "value": v,
                "source": "IFRC",
                "source_id": source_id,
            }
        )
        values.append(v)

    history_for_table = list(reversed(history))

    summary_lines: List[str] = [
        "### IFRC Montandon people affected — 36-month history (Resolver)",
        "",
        f"- Months available: {len(history)}",
        f"- Min monthly PA: {min(values):,.0f}",
        f"- Max monthly PA: {max(values):,.0f}",
    ]
    summary_lines.append("")
    summary_lines.append("| Month | People affected |")
    summary_lines.append("|-------|-----------------|")
    for row in history_for_table:
        summary_lines.append(f"| {row['ym']} | {row['value']:,.0f} |")
    summary = "\n".join(summary_lines)

    return summary, {
        "error": "",
        "history_rows_detail": history,
        "summary_text": summary,
    }


def _load_idmc_pa_history(
    iso3: str,
    hazard_code: str,
    *,
    months: int = 36,
) -> Tuple[str, Dict[str, Any]]:
    """
    Load a 36-month IDMC/DTM displacement history for conflict/displacement
    questions.

    Combines:
      - flows from facts_deltas (new_displacements, idp_displacement_new_dtm),
      - stocks from facts_resolved (idp_displacement_stock_dtm).

    We group by ym and sum flows; stocks are carried as-is per ym.
    """

    hz = (hazard_code or "").upper()
    if hz not in IDMC_HZ_MAP:
        return "", {"error": "no_mapping", "history_rows_detail": [], "summary_text": ""}

    try:
        con = duckdb.connect(_pythia_db_path_from_config(), read_only=True)
    except Exception:
        return "", {"error": "missing_db", "history_rows_detail": [], "summary_text": ""}

    try:
        flow_rows = con.execute(
            """
            SELECT
                ym,
                SUM(CASE WHEN lower(metric) = 'new_displacements'
                         THEN COALESCE(value_new, 0) ELSE 0 END) AS idmc_flow,
                SUM(CASE WHEN lower(metric) = 'idp_displacement_new_dtm'
                         THEN COALESCE(value_new, 0) ELSE 0 END) AS dtm_flow
            FROM facts_deltas
            WHERE iso3 = ?
              AND lower(series_semantics) = 'new'
              AND lower(metric) IN ('new_displacements', 'idp_displacement_new_dtm')
            GROUP BY ym
            ORDER BY ym DESC
            LIMIT ?
            """,
            [iso3, months],
        ).fetchall()

        stock_rows = con.execute(
            """
            SELECT ym, value AS stock_value
            FROM facts_resolved
            WHERE iso3 = ?
              AND lower(metric) = 'idp_displacement_stock_dtm'
            ORDER BY ym DESC
            LIMIT ?
            """,
            [iso3, months],
        ).fetchall()
    except Exception as exc:
        con.close()
        return "", {
            "error": f"idmc_query_error:{type(exc).__name__}",
            "history_rows_detail": [],
            "summary_text": "",
        }

    if not flow_rows and not stock_rows:
        con.close()
        return "", {"error": "no_history", "history_rows_detail": [], "summary_text": ""}

    stock_by_ym = {str(ym): float(stock or 0.0) for ym, stock in stock_rows}

    history: List[Dict[str, Any]] = []
    chosen_flow_values: List[float] = []
    stock_values: List[float] = []

    for ym, idmc_flow, dtm_flow in flow_rows:
        ym_str = str(ym)
        idmc_v = float(idmc_flow or 0.0)
        dtm_v = float(dtm_flow or 0.0)
        if idmc_v:
            chosen = idmc_v
            source = "IDMC"
        elif dtm_v:
            chosen = dtm_v
            source = "DTM"
        else:
            chosen = 0.0
            source = ""

        stock_v = stock_by_ym.get(ym_str, 0.0)

        history.append(
            {
                "ym": ym_str,
                "value": chosen,
                "value_flow": chosen,
                "value_stock": stock_v,
                "source": source or "IDMC/DTM",
                "flow_idmc": idmc_v,
                "flow_dtm": dtm_v,
            }
        )
        chosen_flow_values.append(chosen)
        stock_values.append(stock_v)

    con.close()

    if not history and not stock_values:
        return "", {"error": "no_history", "history_rows_detail": [], "summary_text": ""}

    history_for_table = list(reversed(history))

    summary_lines: List[str] = [
        "### IDMC/DTM displacement — 36-month history (Resolver)",
        "",
        f"- Months available: {len(history)}",
    ]
    if chosen_flow_values:
        summary_lines.append(
            f"- Flow (new displacements) min={min(chosen_flow_values):,.0f}, max={max(chosen_flow_values):,.0f}"
        )
    if stock_values:
        summary_lines.append(
            f"- Stock (IDPs, DTM) min={min(stock_values):,.0f}, max={max(stock_values):,.0f}"
        )
    summary_lines.append(
        "- Source selection: IDMC is used when available for a given month; DTM is used only when IDMC is missing."
    )
    summary_lines.append("")
    summary_lines.append("| Month | Flow (people displaced) | Stock (IDPs, DTM) | Source |")
    summary_lines.append("|-------|--------------------------|-------------------|--------|")
    for row in history_for_table:
        summary_lines.append(
            f"| {row['ym']} | {row['value_flow']:,.0f} | {row['value_stock']:,.0f} | {row['source']} |"
        )
    summary = "\n".join(summary_lines)

    return summary, {
        "error": "",
        "history_rows_detail": history,
        "summary_text": summary,
    }


def _load_acled_fatalities_history(
    iso3: str,
    *,
    months: int = 36,
) -> Tuple[str, Dict[str, Any]]:
    """
    Load a 36-month ACLED fatalities history for conflict questions.

    First try facts_resolved with metric='fatalities'; fall back to
    db.acled_monthly_fatalities if needed.
    """

    try:
        con = duckdb.connect(_pythia_db_path_from_config(), read_only=True)
    except Exception:
        return "", {"error": "missing_db", "history_rows_detail": [], "summary_text": ""}

    try:
        rows = con.execute(
            """
            SELECT ym, value, source_type
            FROM facts_resolved
            WHERE iso3 = ?
              AND lower(metric) = 'fatalities'
            ORDER BY ym DESC
            LIMIT ?
            """,
            [iso3, months],
        ).fetchall()
    except Exception as exc:
        rows = []
        facts_err = f"facts_query_error:{type(exc).__name__}"
    else:
        facts_err = ""

    history: List[Dict[str, Any]] = []
    values: List[float] = []

    if rows:
        for ym, val, source_type in rows:
            ym_str = str(ym)
            v = float(val or 0)
            history.append({"ym": ym_str, "value": v, "source": source_type or "acled"})
            values.append(v)
    else:
        try:
            rows2 = con.execute(
                """
                SELECT strftime(month, '%Y-%m') AS ym, fatalities, source
                FROM acled_monthly_fatalities
                WHERE iso3 = ?
                ORDER BY month DESC
                LIMIT ?
                """,
                [iso3, months],
            ).fetchall()
        except Exception as exc:
            con.close()
            return "", {
                "error": facts_err or f"acled_query_error:{type(exc).__name__}",
                "history_rows_detail": [],
                "summary_text": "",
            }

        if not rows2:
            con.close()
            return "", {"error": "no_rows", "history_rows_detail": [], "summary_text": ""}

        for ym_str, fatalities, src in rows2:
            v = float(fatalities or 0)
            history.append({"ym": ym_str, "value": v, "source": src or "acled"})
            values.append(v)

    con.close()

    values_rev = list(reversed(values))
    summary_lines = [
        "### ACLED conflict fatalities — 36-month history (Resolver)",
        "",
        f"- Months available: {len(history)}",
        f"- Min monthly fatalities: {min(values_rev):,.0f}",
        f"- Max monthly fatalities: {max(values_rev):,.0f}",
    ]
    summary = "\n".join(summary_lines)

    return summary, {
        "error": "",
        "history_rows_detail": history,
        "summary_text": summary,
    }


def _load_pa_history_block(
    iso3: str,
    hazard_code: str,
    *,
    metric: str,
    months: int = 36,
) -> tuple[str, Dict[str, Any]]:
    """
    Dispatch to the appropriate history loader based on metric + hazard.

    - For metric='FATALITIES' → ACLED fatalities history.
    - For metric='PA' and conflict/displacement hazards (ACO/ACE/CU/DI) →
      IDMC/DTM displacement history.
    - For metric='PA' and natural hazards (FL/DR/TC/HW) → IFRC Montandon PA history.
    """

    hz = (hazard_code or "").upper()
    m = (metric or "").upper()

    if m == "FATALITIES":
        return _load_acled_fatalities_history(iso3, months=months)

    if m == "PA" and hz in IDMC_HZ_MAP:
        return _load_idmc_pa_history(iso3, hazard_code, months=months)

    if m == "PA" and hz in NATURAL_HAZARD_CODES:
        return _load_ifrc_pa_history(iso3, hazard_code, months=months)

    return "", {"error": "no_mapping", "history_rows_detail": [], "summary_text": ""}


@functools.lru_cache(maxsize=1)
def _load_country_names() -> dict[str, str]:
    names: dict[str, str] = {}
    try:
        with COUNTRIES_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                iso3 = (row.get("iso3") or "").strip().upper()
                name = (row.get("country_name") or "").strip()
                if iso3 and name:
                    names[iso3] = name
    except Exception:
        return {}
    return names
