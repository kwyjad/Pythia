# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations
"""
cli.py — Forecaster runner (Pythia-only question sources)

WHAT THIS FILE DOES (high level, in plain English)
--------------------------------------------------
- Fetches Pythia Horizon Scanner questions (or local test questions).
- For each question:
  1) Runs the RESEARCH step to build a compact research brief.
  2) Classifies the question (primary/secondary topic + "strategic?" score).
     - If it's strategic *and* the question is binary, we try GTMC1.
  3) Builds a forecasting prompt and asks each LLM model in your ensemble for a forecast.
  4) Aggregates model outputs with a Bayesian Monte Carlo layer ("BMC"); optionally fuses GTMC1 for binary.
  5) Records *everything* into ONE wide CSV row via io_logs.write_unified_row(...).

- Additionally, it runs an **ablation** pass ("no-research") so you can quantify the
  value of the research component. Those results are logged into dedicated CSV columns.

- It also logs three ensemble **variants** for diagnostics:
  (a) no_gtmc1            → BMC aggregation without the GTMC1 signal,
  (b) uniform_weights     → treat all LLMs equally,
  (c) no_bmc_no_gtmc1     → a very simple average of model outputs (no BMC, no GTMC1).
"""

import argparse
import asyncio
import csv
import functools
import importlib
import importlib.util
import json
import os
import re
import logging
from collections import Counter
from urllib.parse import urlparse
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from contextlib import ExitStack
from typing import Any, Dict, List, Optional, Set, Tuple
import inspect

import duckdb
import numpy as np

from pathlib import Path
from pythia.db.schema import connect, ensure_schema
from pythia.db.schema import connect as pythia_connect
from pythia.web_research import fetch_evidence_pack
from forecaster.hs_utils import load_hs_triage_entry
from forecaster.self_search import (
    append_evidence_to_prompt,
    append_retriever_evidence_to_prompt,
    combine_usage,
    extract_self_search_query,
    model_self_search_enabled,
    run_self_search,
    self_search_enabled,
    self_search_limits,
    trim_sources,
)

LOG = logging.getLogger(__name__)

MAX_RESEARCH_WORKERS = int(os.getenv("FORECASTER_RESEARCH_MAX_WORKERS", "6"))
MAX_SPD_WORKERS = int(os.getenv("FORECASTER_SPD_MAX_WORKERS", "6"))
COUNTRIES_CSV = Path(__file__).resolve().parents[1] / "resolver" / "data" / "countries.csv"


_DEFAULT_ENSEMBLE_LOGGED = False


def _maybe_log_default_ensemble() -> None:
    global _DEFAULT_ENSEMBLE_LOGGED
    if _DEFAULT_ENSEMBLE_LOGGED:
        return
    if os.getenv("PYTHIA_DEBUG_MODELS", "0") != "1":
        return

    _DEFAULT_ENSEMBLE_LOGGED = True
    try:
        from forecaster.providers import DEFAULT_ENSEMBLE, default_ensemble_summary

        LOG.info(
            "[debug] DEFAULT_ENSEMBLE models=%d | %s",
            len(DEFAULT_ENSEMBLE),
            default_ensemble_summary(),
        )
    except Exception:  # noqa: BLE001
        LOG.exception("[debug] Failed to summarize DEFAULT_ENSEMBLE")


def _safe_json_loads(text: str) -> Any:
    """
    Best-effort JSON loader for LLM responses.

    - Strips ``` / ```json fences if present.
    - Tries to parse the whole string.
    - If that fails, tries the first {...} block.
    Raises json.JSONDecodeError if all attempts fail.
    """
    if text is None:
        raise json.JSONDecodeError("Empty text", "", 0)

    s = str(text).strip()

    # Strip simple markdown fences (``` or ```json)
    if s.startswith("```"):
        lines = s.splitlines()
        # Drop opening fence
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        # Drop closing fence if present
        if lines and lines[-1].lstrip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    # First attempt: whole string
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Second attempt: first {...} block
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = s[start : end + 1]
            return json.loads(candidate)
        # Re-raise original error
        raise


def _json_dumps_for_db(obj: Any, **kwargs: Any) -> str:
    """
    JSON-encode helper for DB payloads that may contain non-JSON-native
    Python objects (e.g., date). Unknown types are stringified.
    """

    return json.dumps(obj, default=str, **kwargs)


@dataclass
class QuestionRunSummary:
    question_id: str
    iso3: str
    hazard_code: str
    metric: str
    month_count: int = 0
    buckets_per_month: int = 0
    ensemble_rows: int = 0
    raw_rows: int = 0
    ev_min: Optional[float] = None
    ev_max: Optional[float] = None
    models: Dict[str, Dict[str, float]] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PythiaQuestion:
    """
    Minimal representation of a forecastable question in Pythia, as loaded from the
    `questions` table. This is used by the epoch-aware loader.
    """

    question_id: str
    hs_run_id: Optional[str]
    iso3: str
    hazard_code: str
    metric: str
    target_month: Optional[str]
    window_start_date: Optional[str]
    window_end_date: Optional[str]
    wording: str
    status: str
    pythia_metadata_json: Optional[str]
    scenario_ids_json: Optional[str] = None


def _dedupe_sources_by_url(sources: list[Any]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for src in sources:
        if isinstance(src, dict):
            url = str(src.get("url") or "").strip()
            if not url:
                continue
            if url in seen_urls:
                continue
            deduped.append(src)
            seen_urls.add(url)
        elif isinstance(src, str):
            url = src.strip()
            if not url:
                continue
            if url in seen_urls:
                continue
            deduped.append({"title": url, "url": url})
            seen_urls.add(url)
    return deduped


_PYTHIA_CFG_LOAD = None
if importlib.util.find_spec("pythia.config") is not None:
    _PYTHIA_CFG_LOAD = getattr(importlib.import_module("pythia.config"), "load", None)

try:
    from pythia.llm_profiles import get_current_models as _get_llm_profile_models
except Exception:
    _get_llm_profile_models = None  # type: ignore


# Hazard codes for which GTMC1 is relevant (adjust as needed for your schema)
CONFLICT_HAZARD_CODES = {"ACE", "ACO"}

# SPD buckets for Pythia PA/PIN forecasts (order must match prompts & aggregation)
SPD_CLASS_BINS_PA = [
    "<10k",
    "10k-<50k",
    "50k-<250k",
    "250k-<500k",
    ">=500k",
]

# SPD buckets for conflict fatalities forecasts (per month)
SPD_CLASS_BINS_FATALITIES = [
    "<5",
    "5-<25",
    "25-<100",
    "100-<500",
    ">=500",
]

# Backwards-compatibility alias; PA remains the default bucket scheme
SPD_CLASS_BINS = SPD_CLASS_BINS_PA

HZ_QUERY_MAP = {
    # Natural hazards
    "FL": "FLOOD",
    "DR": "DROUGHT",
    "TC": "TROPICAL_CYCLONE",
    "HW": "HEAT_WAVE",

    # Conflict / displacement (PA)
    "ACO": "CONFLICT",
    "ACE": "CONFLICT",
    "CU": "CONFLICT",
    "DI": "DISPLACEMENT",
}

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


# ---------------------------------------------------------------------------
# Hazard-specific base-rate builders (replace generic JSON dump in SPD v2)
# ---------------------------------------------------------------------------

_HAZARD_DISPLAY_NAMES: Dict[str, str] = {
    "FL": "Flood",
    "DR": "Drought",
    "TC": "Tropical Cyclone",
    "HW": "Heat Wave",
    "ACE": "Armed Conflict",
    "DI": "Displacement Inflow",
}


def _build_natural_hazard_seasonal_profile(
    iso3: str,
    hazard_code: str,
) -> Dict[str, Any]:
    """Build a calendar-month seasonal profile from facts_resolved for natural hazards.

    Queries ALL available data for the given iso3 + hazard_code where
    metric IN ('affected', 'in_need', 'pa'). Groups by calendar month (1-12)
    and computes min/max/mean/median across all available years.
    """
    iso3_up = (iso3 or "").upper().strip()
    hz_up = (hazard_code or "").upper().strip()

    empty: Dict[str, Any] = {
        "type": "seasonal_profile",
        "source": "IFRC",
        "data_range": "",
        "years_of_data": 0,
        "months": {m: {"min": 0, "max": 0, "mean": 0, "median": 0, "n_observations": 0} for m in range(1, 13)},
    }

    if not iso3_up or not hz_up:
        return empty

    con = connect(read_only=True)
    try:
        try:
            rows = con.execute(
                """
                SELECT ym, value
                FROM facts_resolved
                WHERE iso3 = ?
                  AND hazard_code = ?
                  AND lower(metric) IN ('affected', 'in_need', 'pa')
                ORDER BY ym
                """,
                [iso3_up, hz_up],
            ).fetchall()
        except Exception as exc:
            logging.warning(
                "Seasonal profile query failed for %s/%s: %s", iso3_up, hz_up, exc
            )
            return empty

        if not rows:
            return empty

        # Parse rows, filter future months, group by calendar month
        now_month = datetime.utcnow().strftime("%Y-%m")
        by_cal_month: Dict[int, list[float]] = {m: [] for m in range(1, 13)}
        all_yms: list[str] = []

        for ym_raw, value in rows:
            parsed = _parse_month_key(str(ym_raw))
            if parsed is None:
                continue
            month_key = parsed.strftime("%Y-%m")
            if month_key > now_month:
                continue
            try:
                v = float(value or 0)
            except (TypeError, ValueError):
                v = 0.0
            by_cal_month[parsed.month].append(v)
            all_yms.append(month_key)

        if not all_yms:
            return empty

        all_yms_sorted = sorted(all_yms)
        data_range = f"{all_yms_sorted[0]} to {all_yms_sorted[-1]}"

        # Compute distinct years
        years_set = {ym[:4] for ym in all_yms_sorted}
        years_of_data = len(years_set)

        months_result: Dict[int, Dict[str, Any]] = {}
        for cal_month in range(1, 13):
            vals = by_cal_month[cal_month]
            if not vals:
                months_result[cal_month] = {
                    "min": 0, "max": 0, "mean": 0, "median": 0, "n_observations": 0,
                }
            else:
                arr = np.array(vals)
                months_result[cal_month] = {
                    "min": round(float(np.min(arr))),
                    "max": round(float(np.max(arr))),
                    "mean": round(float(np.mean(arr))),
                    "median": round(float(np.median(arr))),
                    "n_observations": len(vals),
                }

        return {
            "type": "seasonal_profile",
            "source": "IFRC",
            "data_range": data_range,
            "years_of_data": years_of_data,
            "months": months_result,
        }
    finally:
        con.close()


def _build_conflict_base_rate(
    iso3: str,
    hazard_code: str,
) -> Dict[str, Any]:
    """Build a conflict trajectory base rate from ACLED fatalities and IDMC displacements.

    Queries the most recent 6 months of data from each source, computes
    trailing 3-month averages and trend direction.
    """
    iso3_up = (iso3 or "").upper().strip()
    hz_up = (hazard_code or "").upper().strip()

    if hz_up == "DI":
        return {
            "type": "no_base_rate",
            "note": (
                "DI (displacement inflow) has no Resolver base rate; rely on HS + research "
                "and exogenous neighbour shocks."
            ),
        }

    def _compute_trajectory(
        rows: list[tuple],
        source_name: str,
    ) -> Dict[str, Any]:
        """Compute trajectory stats from (ym, value) rows sorted by ym desc (most recent first)."""
        if not rows:
            return {
                "source": source_name,
                "last_month": None,
                "trailing_3m_avg": None,
                "prior_3m_avg": None,
                "trend_pct": None,
                "trend_direction": None,
                "last_6m": [],
                "note": f"No {source_name} data available for this country.",
            }

        # rows should be sorted ascending by ym
        last_6 = [{"ym": str(ym), "value": round(float(val or 0))} for ym, val in rows]
        values = [entry["value"] for entry in last_6]

        last_month_entry = last_6[-1]
        # trailing 3m = last 3 months, prior 3m = months 4-6
        trailing_3m_vals = values[-3:] if len(values) >= 3 else values
        prior_3m_vals = values[-6:-3] if len(values) >= 6 else values[:max(0, len(values) - 3)]

        trailing_3m_avg = round(sum(trailing_3m_vals) / len(trailing_3m_vals)) if trailing_3m_vals else None
        prior_3m_avg = round(sum(prior_3m_vals) / len(prior_3m_vals)) if prior_3m_vals else None

        trend_pct: Any = None
        trend_direction: Any = None
        if trailing_3m_avg is not None and prior_3m_avg is not None:
            if prior_3m_avg == 0:
                if trailing_3m_avg > 0:
                    trend_pct = "new_activity"
                    trend_direction = "escalating"
                else:
                    trend_pct = 0.0
                    trend_direction = "stable"
            else:
                pct = ((trailing_3m_avg - prior_3m_avg) / prior_3m_avg) * 100
                trend_pct = round(pct, 1)
                if pct > 10:
                    trend_direction = "escalating"
                elif pct < -10:
                    trend_direction = "de-escalating"
                else:
                    trend_direction = "stable"

        return {
            "source": source_name,
            "last_month": last_month_entry,
            "trailing_3m_avg": trailing_3m_avg,
            "prior_3m_avg": prior_3m_avg,
            "trend_pct": trend_pct,
            "trend_direction": trend_direction,
            "last_6m": last_6,
        }

    con = connect(read_only=True)
    try:
        # --- Fatalities from ACLED ---
        fatalities_data: Dict[str, Any]
        try:
            fat_rows = con.execute(
                """
                SELECT month, fatalities
                FROM acled_monthly_fatalities
                WHERE iso3 = ?
                ORDER BY month DESC
                LIMIT 6
                """,
                [iso3_up],
            ).fetchall()
            # Reverse to ascending order
            fat_rows = list(reversed(fat_rows))
            # Normalise month keys to YYYY-MM strings
            fat_rows = [(str(r[0])[:7] if len(str(r[0])) >= 7 else str(r[0]), r[1]) for r in fat_rows]
            fatalities_data = _compute_trajectory(fat_rows, "ACLED")
        except Exception as exc:
            logging.warning("ACLED fatalities query failed for %s: %s", iso3_up, exc)
            fatalities_data = {
                "source": "ACLED",
                "last_month": None, "trailing_3m_avg": None,
                "prior_3m_avg": None, "trend_pct": None,
                "trend_direction": None, "last_6m": [],
                "note": f"ACLED data unavailable: {type(exc).__name__}",
            }

        # --- Displacements from IDMC via facts_deltas ---
        displacements_data: Dict[str, Any]
        try:
            disp_rows = con.execute(
                """
                SELECT ym, SUM(COALESCE(value_new, 0)) AS flow_value
                FROM facts_deltas
                WHERE upper(iso3) = ?
                  AND COALESCE(NULLIF(upper(hazard_code), ''), 'ACE') IN (?, 'IDU')
                  AND lower(series_semantics) = 'new'
                  AND (
                      lower(source_id) IN ('idmc', 'idmc_idu')
                      OR lower(metric) IN (
                          'new_displacements',
                          'idp_displacement_new_dtm',
                          'idp_displacement_flow_idmc'
                      )
                  )
                GROUP BY ym
                ORDER BY ym DESC
                LIMIT 6
                """,
                [iso3_up, hz_up],
            ).fetchall()
            logging.debug("IDMC displacement query for %s/%s: %d rows", iso3_up, hz_up, len(disp_rows))
            disp_rows = list(reversed(disp_rows))
            disp_rows = [(str(r[0])[:7] if len(str(r[0])) >= 7 else str(r[0]), r[1]) for r in disp_rows]
            displacements_data = _compute_trajectory(disp_rows, "IDMC")
        except Exception as exc:
            logging.warning("IDMC displacement query failed for %s/%s: %s", iso3_up, hz_up, exc)
            displacements_data = {
                "source": "IDMC",
                "last_month": None, "trailing_3m_avg": None,
                "prior_3m_avg": None, "trend_pct": None,
                "trend_direction": None, "last_6m": [],
                "note": f"IDMC data unavailable: {type(exc).__name__}",
            }

        return {
            "type": "conflict_trajectory",
            "fatalities": fatalities_data,
            "displacements": displacements_data,
        }
    finally:
        con.close()


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


def _build_history_summary(iso3: str, hazard_code: str, metric: str) -> Dict[str, Any]:
    """Build Resolver history summary per hazard/metric rules.

    Dispatches to hazard-specific builders that return structured dicts
    with a ``type`` field (seasonal_profile, conflict_trajectory, or
    no_base_rate).  The ``source`` key is preserved for backward compat.
    """
    hz = (hazard_code or "").upper()
    m = (metric or "").upper()

    if hz == "DI":
        return {
            "type": "no_base_rate",
            "source": "NONE",
            "note": (
                "DI (displacement inflow) has no Resolver base rate; rely on HS + research "
                "and exogenous neighbour shocks."
            ),
        }

    # Natural hazards — seasonal profile
    if m == "PA" and hz in NATURAL_HAZARD_CODES:
        result = _build_natural_hazard_seasonal_profile(iso3, hz)
        # Preserve backward-compat "source" key at top level
        result.setdefault("source", "IFRC")
        return result

    # Conflict hazards — PA uses IDMC displacement history
    if hz == "ACE" and m == "PA":
        con = connect(read_only=True)
        try:
            return _load_idmc_conflict_flow_history_summary(con, iso3, hz)
        finally:
            con.close()

    # Conflict hazards — FATALITIES uses trajectory
    if hz == "ACE" and m == "FATALITIES":
        result = _build_conflict_base_rate(iso3, hz)
        result.setdefault("source", "ACLED")
        return result

    return {
        "type": "no_base_rate",
        "source": "NONE",
        "note": "No usable Resolver history for this hazard/metric.",
    }



def _extract_pythia_meta(post: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract Pythia-specific metadata attached by _load_pythia_questions(...).

    Returns a dict with keys:
      - iso3
      - hazard_code
      - metric
      - target_month
    Missing fields are normalized to "".
    """
    return {
        "iso3": str(post.get("pythia_iso3") or "").upper(),
        "hazard_code": str(post.get("pythia_hazard_code") or "").upper(),
        "metric": str(post.get("pythia_metric") or "").upper(),
        "target_month": str(post.get("pythia_target_month") or ""),
    }


def _infer_resolution_source(hazard_code: str, metric: str) -> str:
    hz = (hazard_code or "").upper()
    mt = (metric or "").upper()

    if mt == "FATALITIES" and hz in {"ACE"}:
        return "ACLED"
    if mt == "PA" and hz in {"ACE"}:
        return "IDMC"
    if mt == "PA" and hz in {"DR", "FL", "TC", "HW"}:
        return "IFRC"
    if mt == "PA" and hz == "DI":
        return "NONE"
    if mt == "EVENT_OCCURRENCE" and hz in {"FL", "DR", "TC"}:
        return "GDACS"
    return "NONE"



def _record_no_forecast(
    run_id: str,
    question_id: str,
    iso3: str,
    hazard_code: str,
    metric: str,
    reason: str,
    *,
    model_name: str = "ensemble",
    raw_debug_written: bool = False,
    raw_debug_tag: str | None = None,
    raw_debug_text: str | None = None,
) -> None:
    """Persist a no-forecast outcome with an explanation."""

    if (not raw_debug_written) and raw_debug_tag and raw_debug_text is not None:
        try:
            _write_spd_raw_text(run_id, question_id, raw_debug_tag, raw_debug_text)
        except Exception:
            pass

    con = connect(read_only=False)
    try:
        con.execute(
            """
            INSERT INTO forecasts_raw (
              run_id, question_id, model_name, month_index, bucket_index,
              probability, ok, elapsed_ms, cost_usd, prompt_tokens, completion_tokens,
              total_tokens, status, spd_json, human_explanation
            ) VALUES (?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 'no_forecast', NULL, ?)
            """,
            [run_id, question_id, model_name, reason],
        )

        con.execute(
            """
            INSERT INTO forecasts_ensemble (
              run_id, question_id, iso3, hazard_code, metric, model_name,
              month_index, bucket_index, probability, ev_value, weights_profile, created_at,
              status, human_explanation
            ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, 'ensemble', CURRENT_TIMESTAMP, 'no_forecast', ?)
            """,
            [run_id, question_id, iso3, hazard_code, metric, model_name, reason],
        )
    finally:
        con.close()


def _safe_json_load(s: str):
    try:
        import json as _json
        return _json.loads(s)
    except Exception:
        return None


def _as_dict(obj: Any) -> Dict[str, Any]:
    """Return a dict from obj. If obj is a JSON string, parse it. Else {}."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, (str, bytes)):
        try:
            parsed = json.loads(obj)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _must_dict(name: str, obj: Any) -> Dict[str, Any]:
    d = _as_dict(obj)
    if not d:
        # Keep this very explicit so CI logs are helpful and not a vague AttributeError
        raise RuntimeError(f"{name} is not a dict after coercion (type={type(obj).__name__})")
    return d


def _coerce_date(val: Any) -> Optional[date]:
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val).date()
        except Exception:
            return None
    return None


def _parse_month_key(s: str) -> Optional[date]:
    """
    Parse a month key into a date anchored to the first of the month.

    Supported formats:
      - YYYY-MM
      - YYYY-MM-DD (day component is discarded)
    """
    if not isinstance(s, str):
        return None

    text = s.strip()
    if not text:
        return None

    try:
        if len(text) == 7:
            dt = datetime.strptime(text, "%Y-%m")
            return date(dt.year, dt.month, 1)
        dt = datetime.fromisoformat(text)
        return date(dt.year, dt.month, 1)
    except Exception:
        try:
            parts = text.split("-")
            if len(parts) >= 2:
                year = int(parts[0])
                month = int(parts[1])
                return date(year, month, 1)
        except Exception:
            return None
    return None


def _sanitize_month_series(
    month_to_value: Dict[str, Any],
) -> tuple[Dict[str, Any], list[str], list[str]]:
    """
    Remove entries with month keys later than the current month.

    Returns (cleaned_dict, dropped_future_months, unparseable_month_keys).
    """
    now_month = datetime.utcnow().strftime("%Y-%m")
    cleaned: Dict[str, Any] = {}
    dropped: list[str] = []
    unparseable: list[str] = []

    for key in sorted(month_to_value.keys()):
        val = month_to_value[key]
        if str(key).strip() == "":
            unparseable.append(str(key))
            continue
        parsed = _parse_month_key(str(key))
        if parsed is None:
            unparseable.append(str(key))
            continue

        month_key = parsed.strftime("%Y-%m")
        if month_key > now_month:
            dropped.append(str(key))
            continue

        cleaned[str(key)] = val

    return cleaned, dropped, unparseable


def _build_month_labels(start_date: Optional[date], horizon_months: int = 6) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    if not isinstance(start_date, date):
        return labels
    y, m = start_date.year, start_date.month
    for idx in range(1, horizon_months + 1):
        labels[idx] = date(y, m, 1).strftime("%B %Y")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return labels


# ---- Forecaster internals (all relative imports) --------------------------------
from .config import ist_iso
from .prompts import (
    build_spd_prompt_v2,
    merge_evidence_packs,
)
from .binary_prompts import (
    build_binary_event_prompt,
    build_binary_base_rate,
    parse_binary_response,
)
from .scenario_writer import run_scenarios_for_run
from horizon_scanner.seasonal_context import CLIMATE_HAZARDS, load_seasonal_forecasts
from .providers import (
    DEFAULT_ENSEMBLE,
    GEMINI_MODEL_ID,
    ModelSpec,
    _PROVIDER_STATES,
    SPD_ENSEMBLE,
    _get_or_client,
    call_chat_ms,
    disabled_providers_for_run,
    is_provider_disabled_for_run,
    get_llm_semaphore,
    estimate_cost_usd,
    parse_ensemble_specs,
    reset_provider_failures_for_run,
)
from .ensemble import (
    EnsembleResult,
    MemberOutput,
    _normalize_spd_keys,
    _load_bucket_centroids_db,
    run_ensemble_binary,
    run_ensemble_mcq,
    run_ensemble_numeric,
    run_ensemble_spd,
    SPD_BUCKET_CENTROIDS_PA,
    SPD_BUCKET_CENTROIDS_FATALITIES,
    sanitize_mcq_vector,
)
from .aggregate import (
    SPD_BUCKET_CENTROIDS_DEFAULT,
    aggregate_binary,
    aggregate_mcq,
    aggregate_numeric,
    aggregate_spd,
    aggregate_spd_v2_mean,
    aggregate_spd_v2_bayesmc,
)
from .llm_logging import log_forecaster_llm_call

# --- Corrected seen_guard import ---
try:
    from . import seen_guard
except ImportError as e:
    print(f"[warn] seen_guard not available ({e!r}); continuing without duplicate protection.")
    seen_guard = None

from . import GTMC1

# --- seen_guard import shim (ensures a callable filter_unseen_posts exists) ---
try:
    try:
        # When cli.py is executed as a module
        from .seen_guard import SeenGuard  # type: ignore
    except Exception:
        # When cli.py is executed as a script from repo root
        from seen_guard import SeenGuard  # type: ignore

    _sg = SeenGuard()

    def filter_unseen_posts(posts):
        # Adapter to old call-site name; calls the actual class method.
        return _sg.filter_fresh_posts(posts)

except Exception as e:
    print(f"[seen_guard] disabled ({e}); processing all posts returned.")
    def filter_unseen_posts(posts):
        return posts
# --- end seen_guard import shim ---



# Unified CSV helpers (single file)
from .io_logs import ensure_unified_csv, write_unified_row, write_human_markdown, finalize_and_commit

# --------------------------------------------------------------------------------
# Small utility helpers (safe JSON, timing, clipping, etc.)
# --------------------------------------------------------------------------------

# --- SeenGuard wiring (robust to different shapes/APIs) -----------------------
def _load_seen_guard():
    """
    Try to load a SeenGuard instance from seen_guard.py in a robust way.
    Will look for common instance names and fall back to constructing SeenGuard.
    Returns: guard instance or None
    """
    try:
        import seen_guard as sg_mod
    except Exception:
        return None

    # Prefer a ready-made instance exported from the module
    for attr in ("_GUARD", "GUARD", "guard"):
        guard = getattr(sg_mod, attr, None)
        if guard is not None:
            return guard

    # Fallback: instantiate if class is available
    try:
        SG = getattr(sg_mod, "SeenGuard", None)
        if SG is not None:
            cooldown = int(os.getenv("SEEN_COOLDOWN_HOURS", "24"))
            path = os.getenv("SEEN_GUARD_PATH", "forecast_logs/state/seen_forecasts.jsonl")
            return SG(Path(path), cooldown_hours=cooldown)
    except Exception:
        pass

    return None


def _apply_seen_guard(guard, posts):
    """
    Call the first matching method on guard to filter posts.
    Accepts either a return of (posts, dup_count) or just posts.
    """
    if not guard or not posts:
        return posts, 0

    candidates = [
        "filter_fresh_posts",
        "filter_unseen_posts",
        "filter_posts",
        "filter_recent_posts",
        "filter_new_posts",
        "filter",  # very generic, last
    ]

    last_err = None
    for name in candidates:
        if hasattr(guard, name):
            fn = getattr(guard, name)
            try:
                # Try simple positional call
                result = fn(posts)
            except TypeError:
                # Try kwargs form if implemented that way
                try:
                    result = fn(posts=posts)
                except Exception as e:
                    last_err = e
                    continue
            except Exception as e:
                last_err = e
                continue

            # Normalize return
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list):
                return result
            if isinstance(result, list):
                return result, 0

            # Unexpected return shape; treat as no-op
            return posts, 0

    # If we got here, no callable matched or all failed
    if last_err:
        raise last_err
    return posts, 0
# ----------------------------------------------------------------------------- 

# Time in milliseconds since start_time
def _ms(start_time: float) -> int:
    return int(round((time.time() - start_time) * 1000))


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _sanitize_markdown_chunks(chunks: List[Any]) -> List[str]:
    """Return a list of strings suitable for ``"\n\n".join(...)``.

    The markdown builder collects many diagnostic entries, some of which
    originate from optional integrations (GTMC1 raw dumps, prediction-market
    lookups, etc.).  When any of those helpers return ``None`` we previously
    propagated the ``None`` directly into the markdown list.  Later, when we
    attempted to join the chunks we hit ``TypeError: sequence item X: expected
    str instance, NoneType found``.  This helper drops ``None`` entries and
    coerces any remaining values to strings so the join is always safe.
    """

    sanitized: List[str] = []
    for chunk in chunks:
        if chunk is None:
            continue
        if isinstance(chunk, str):
            sanitized.append(chunk)
            continue
        try:
            sanitized.append(str(chunk))
        except Exception:
            # If ``str(chunk)`` itself fails we silently drop the entry; the
            # surrounding debug output already makes it clear something odd
            # happened, and failing to write the human log is worse.
            continue
    return sanitized


def _pythia_db_url_from_config() -> str:
    """
    Best-effort helper to read the Pythia DuckDB URL from config or env.

    Priority:
      1. pythia.db.schema.get_db_url (if available)
      2. app.db_url from pythia.config
      3. PYTHIA_DB_URL environment variable
      4. default duckdb:///data/resolver.duckdb

    This helper is intentionally kept for backward compatibility with tests
    that monkeypatch it to point to a temporary DuckDB file.
    """

    try:
        from pythia.db.schema import get_db_url

        url = get_db_url()
        if url:
            return url
    except Exception:
        pass

    if _PYTHIA_CFG_LOAD is not None:
        try:
            cfg = _PYTHIA_CFG_LOAD()
            app_cfg = cfg.get("app", {}) if isinstance(cfg, dict) else {}
            db_url = str(app_cfg.get("db_url", "")).strip()
            if db_url:
                return db_url
        except Exception:
            pass

    env_url = os.getenv("PYTHIA_DB_URL", "").strip()
    if env_url:
        return env_url

    return "duckdb:///data/resolver.duckdb"


def _pythia_db_path_from_config() -> str:
    """Return a filesystem path for the configured DuckDB database."""

    db_url = _pythia_db_url_from_config()
    if db_url.startswith("duckdb:///"):
        return db_url.replace("duckdb:///", "", 1)
    return db_url


def _select_hs_run_id_for_forecast(
    con: duckdb.DuckDBPyConnection,
    explicit: Optional[str] = None,
) -> Optional[str]:
    """
    Choose which HS epoch (hs_run_id) to use for SPD v2 forecasting.

    - If `explicit` is provided (CLI --hs-run-id), use that.
    - Else, prefer the latest hs_run_id from hs_runs (generated_at DESC).
    - If hs_runs is empty, fall back to the latest run_id from hs_triage.
    - If nothing found, return None.
    """
    if explicit:
        return explicit

    # Prefer hs_runs if available
    row = con.execute(
        """
        SELECT hs_run_id
        FROM hs_runs
        ORDER BY generated_at DESC
        LIMIT 1
        """
    ).fetchone()
    if row and row[0]:
        return row[0]

    # Fallback: latest run_id in hs_triage
    row = con.execute(
        """
        SELECT run_id
        FROM hs_triage
        ORDER BY rowid DESC
        LIMIT 1
        """
    ).fetchone()
    if row and row[0]:
        return row[0]

    return None


def _write_spd_ensemble_to_db(
    *,
    run_id: str,
    question_id: str,
    spd_main: Dict[str, List[float]],
    metric: str,
    hazard_code: str,
    iso3: str = "",
    ev_main: Optional[Dict[str, Any]] = None,
    weights_profile: str = "",
) -> None:
    """
    Persist SPD ensemble into forecasts_ensemble.

    spd_main: dict like {"month_1": [p1..p5], ..., "month_6": [p1..p5]}
    ev_main:  dict like {"month_1": ev_value, ...} (optional)
    """

    metric_up = (metric or "").upper()
    hz_up = (hazard_code or "").upper()

    if metric_up == "FATALITIES" and (hz_up.startswith("CONFLICT") or hz_up in CONFLICT_HAZARD_CODES):
        class_bins = SPD_CLASS_BINS_FATALITIES
    else:
        class_bins = SPD_CLASS_BINS_PA

    from .ensemble import _normalize_spd_keys  # local import to avoid cycles

    spd_main = _normalize_spd_keys(spd_main, n_months=6, n_buckets=len(class_bins))

    try:
        import duckdb
    except Exception as exc:  # noqa: BLE001
        print(
            f"[warn] duckdb is required to write SPD ensemble (question_id={question_id}): {type(exc).__name__}: {exc}"
        )
        return

    db_url = _pythia_db_url_from_config()
    db_path = db_url[len("duckdb:///") :] if db_url.startswith("duckdb:///") else db_url

    try:
        con = duckdb.connect(db_path)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[warn] Failed to open DB for SPD ensemble write (question_id={question_id}): {type(exc).__name__}: {exc}"
        )
        return

    try:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS forecasts_ensemble (
                horizon_m INTEGER,
                class_bin VARCHAR,
                p DOUBLE,
                run_id TEXT,
                question_id TEXT,
                iso3 TEXT,
                hazard_code TEXT,
                metric TEXT,
                model_name TEXT,
                month_index INTEGER,
                bucket_index INTEGER,
                probability DOUBLE,
                ev_value DOUBLE,
                weights_profile TEXT,
                created_at TIMESTAMP
            );
            """
        )

        con.execute(
            "DELETE FROM forecasts_ensemble WHERE question_id = ? AND run_id = ?;",
            [question_id, run_id],
        )

        for month_idx in range(1, 7):
            key = f"month_{month_idx}"
            probs = spd_main.get(key) or []
            if not isinstance(probs, (list, tuple)):
                continue
            ev_val = None
            if ev_main and key in ev_main:
                try:
                    ev_val = float(ev_main[key])
                except Exception:
                    ev_val = None

            for bucket_idx, prob in enumerate(probs, start=1):
                class_bin = class_bins[bucket_idx - 1] if 0 <= bucket_idx - 1 < len(class_bins) else str(bucket_idx)
                try:
                    con.execute(
                        """
                        INSERT INTO forecasts_ensemble (
                            run_id,
                            question_id,
                            model_name,
                            metric,
                            hazard_code,
                            horizon_m,
                            class_bin,
                            p,
                            created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
                        """,
                        [
                            run_id,
                            question_id,
                            "ensemble",
                            metric_up,
                            hz_up,
                            month_idx,
                            class_bin,
                            float(prob),
                        ],
                    )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"[warn] Failed to write forecasts_ensemble row for q={question_id} month={month_idx}: {exc}"
                    )
    finally:
        try:
            con.close()
        except Exception:
            pass


def _write_spd_raw_to_db(
    *,
    run_id: str,
    question_id: str,
    iso3: str,
    hazard_code: str,
    metric: str,
    ens_res: EnsembleResult,
) -> None:
    """
    Write per-model SPD forecasts into forecasts_raw for this question.

    Uses PA buckets for PA metrics and conflict fatalities buckets for metric="FATALITIES"
    on conflict hazards.
    """

    metric_up = (metric or "").upper()
    hz_up = (hazard_code or "").upper()

    if metric_up == "FATALITIES" and (hz_up.startswith("CONFLICT") or hz_up in CONFLICT_HAZARD_CODES):
        class_bins = SPD_CLASS_BINS_FATALITIES
    else:
        class_bins = SPD_CLASS_BINS_PA

    try:
        con = connect(read_only=False)
        ensure_schema(con)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[warn] Failed to open DB for SPD raw write (question_id={question_id}): {type(exc).__name__}: {exc}"
        )
        return

    try:
        for m in ens_res.members:
            model_name = getattr(m, "name", "")
            try:
                con.execute(
                    "DELETE FROM forecasts_raw WHERE run_id = ? AND question_id = ? AND model_name = ?;",
                    [run_id, question_id, model_name],
                )
            except Exception:
                pass

            ok = bool(getattr(m, "ok", False))
            elapsed_ms = getattr(m, "elapsed_ms", 0) or 0
            cost_usd = getattr(m, "cost_usd", 0.0) or 0.0
            prompt_tokens = getattr(m, "prompt_tokens", 0) or 0
            completion_tokens = getattr(m, "completion_tokens", 0) or 0
            total_tokens = getattr(m, "total_tokens", prompt_tokens + completion_tokens) or 0

            if not isinstance(getattr(m, "parsed", None), dict):
                continue

            parsed = _normalize_spd_keys(m.parsed, n_months=6, n_buckets=len(class_bins))
            for month_idx in range(1, 7):
                key = f"month_{month_idx}"
                probs = parsed.get(key) or []
                if not isinstance(probs, (list, tuple)):
                    continue
                for bucket_idx, prob in enumerate(probs, start=1):
                    try:
                        cb = class_bins[bucket_idx - 1] if 0 <= bucket_idx - 1 < len(class_bins) else str(bucket_idx)
                        con.execute(
                            """
                            INSERT INTO forecasts_raw (
                                run_id,
                                question_id,
                                model_name,
                                month_index,
                                bucket_index,
                                probability,
                                ok,
                                elapsed_ms,
                                cost_usd,
                                prompt_tokens,
                                completion_tokens,
                                total_tokens,
                                horizon_m,
                                class_bin,
                                p
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                            """,
                            [
                                run_id,
                                question_id,
                                model_name,
                                month_idx,
                                bucket_idx,
                                float(prob),
                                ok,
                                elapsed_ms,
                                cost_usd,
                                prompt_tokens,
                                completion_tokens,
                                total_tokens,
                                month_idx,
                                cb,
                                float(prob),
                            ],
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[warn] Failed to write forecasts_raw row for q={question_id} month={month_idx}: {exc}"
                        )
    finally:
        try:
            con.close()
        except Exception:
            pass


def _count_spd_rows(run_id: str, question_id: str) -> tuple[int, int]:
    """Return (ensemble_rows, raw_rows) counts for the SPD question in DuckDB."""

    try:
        con = connect(read_only=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] failed to count SPD rows for {question_id}: {type(exc).__name__}: {exc}")
        return 0, 0

    try:
        ens_n = con.execute(
            "SELECT COUNT(*) FROM forecasts_ensemble WHERE run_id = ? AND question_id = ?",
            [run_id, question_id],
        ).fetchone()[0]
        raw_n = con.execute(
            "SELECT COUNT(*) FROM forecasts_raw WHERE run_id = ? AND question_id = ?",
            [run_id, question_id],
        ).fetchone()[0]
        return int(ens_n or 0), int(raw_n or 0)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] failed to count SPD rows for {question_id}: {type(exc).__name__}: {exc}")
        return 0, 0
    finally:
        try:
            con.close()
        except Exception:
            pass


def _spd_class_bins_for(metric: str, hazard_code: str) -> list[str]:
    metric_up = (metric or "").upper()
    hz_up = (hazard_code or "").upper()
    if metric_up == "FATALITIES" and (hz_up.startswith("CONFLICT") or hz_up in CONFLICT_HAZARD_CODES):
        return SPD_CLASS_BINS_FATALITIES
    return SPD_CLASS_BINS_PA


def _write_spd_members_v2_to_db(
    *,
    run_id: str,
    question_row: Any,
    specs_used: list[ModelSpec],
    per_model_spds: list[dict[str, list[float]]],
    raw_calls: list[dict[str, object]],
    resolution_source: str,
) -> None:
    """Persist SPD v2 member SPDs into forecasts_raw without touching ensemble rows."""

    qid = str(question_row.get("question_id") or "")
    hz = str(question_row.get("hazard_code") or "").upper()
    metric = str(question_row.get("metric") or "").upper()
    class_bins = _spd_class_bins_for(metric, hz)
    bucket_count = len(class_bins)

    try:
        con = connect(read_only=False)
        ensure_schema(con)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Failed to open DB for SPD member write (question_id={qid}): {type(exc).__name__}: {exc}")
        return

    def _model_for_index(i: int) -> ModelSpec | None:
        try:
            rc = raw_calls[i]
            ms = rc.get("model_spec") if isinstance(rc, dict) else None
            if isinstance(ms, ModelSpec):
                return ms
        except Exception:
            pass
        if 0 <= i < len(specs_used):
            return specs_used[i]
        return None

    def _month_indices(spd: dict[str, list[float]]) -> list[tuple[int, str]]:
        parsed = []
        for key in spd.keys():
            if isinstance(key, str) and key.startswith("month_"):
                try:
                    idx = int(key.split("_", 1)[1])
                    parsed.append((idx, key))
                    continue
                except Exception:
                    pass
            dt = _parse_month_key(str(key))
            if dt:
                parsed.append((int(dt.strftime("%Y%m")), key))
            else:
                parsed.append((0, key))
        parsed.sort(key=lambda t: (t[0], str(t[1])))
        numbered = []
        for new_idx, (_score, key) in enumerate(parsed[:6], start=1):
            numbered.append((new_idx, key))
        return numbered

    # Pre-compute safe model names to avoid collisions when providers share a label.
    def _safe_names() -> list[str]:
        base_names: list[str] = []
        specs_for_index: list[ModelSpec | None] = []
        for idx in range(len(per_model_spds)):
            ms = _model_for_index(idx)
            specs_for_index.append(ms)
            base_names.append(getattr(ms, "name", f"model_{idx}"))
        counts = Counter(base_names)
        seen: set[str] = set()
        safe: list[str] = []
        for idx, base in enumerate(base_names):
            ms = specs_for_index[idx]
            name = base
            if counts.get(base, 0) > 1:
                suffix = getattr(ms, "model_id", None) or getattr(ms, "provider", None) or f"dup{idx}"
                name = f"{base} ({suffix})"
            # ensure uniqueness even if suffix repeats
            dedupe_counter = 2
            candidate = name
            while candidate in seen:
                candidate = f"{name}#{dedupe_counter}"
                dedupe_counter += 1
            seen.add(candidate)
            safe.append(candidate)
        return safe

    safe_names_for_idx = _safe_names()

    try:
        for idx, model_spd in enumerate(per_model_spds):
            ms = _model_for_index(idx)
            model_name = safe_names_for_idx[idx] if idx < len(safe_names_for_idx) else getattr(ms, "name", f"model_{idx}")
            usage = {}
            try:
                usage = raw_calls[idx].get("usage") if isinstance(raw_calls[idx], dict) else {}
            except Exception:
                usage = {}
            elapsed_ms = int((usage or {}).get("elapsed_ms") or 0)
            cost_usd = float((usage or {}).get("cost_usd") or 0.0)
            prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
            completion_tokens = int((usage or {}).get("completion_tokens") or 0)
            total_tokens = int((usage or {}).get("total_tokens") or prompt_tokens + completion_tokens or 0)

            try:
                con.execute(
                    "DELETE FROM forecasts_raw WHERE run_id = ? AND question_id = ? AND model_name = ?;",
                    [run_id, qid, model_name],
                )
            except Exception:
                pass

            if not isinstance(model_spd, dict) or not model_spd:
                con.execute(
                    """
                    INSERT INTO forecasts_raw (
                        run_id, question_id, model_name, month_index, bucket_index,
                        probability, ok, elapsed_ms, cost_usd, prompt_tokens,
                        completion_tokens, total_tokens, status, spd_json, human_explanation
                    ) VALUES (?, ?, ?, NULL, NULL, NULL, NULL, ?, ?, ?, ?, ?, 'no_forecast', ?, ?)
                    """,
                    [
                        run_id,
                        qid,
                        model_name,
                        elapsed_ms,
                        cost_usd,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        json.dumps({"resolution_source": resolution_source, "spds": {}, "member_source": "spd_v2_member_call"}),
                        "No SPD returned for this model.",
                    ],
                )
                continue

            spd_json_payload: dict[str, object] = {
                "resolution_source": resolution_source,
                "member_source": "spd_v2_member_call",
                "spds": {},
            }
            ordered_months = _month_indices(model_spd)
            if len(ordered_months) < 6:
                next_idx = len(ordered_months) + 1
                while len(ordered_months) < 6:
                    ordered_months.append((next_idx, f"month_{next_idx}"))
                    next_idx += 1
            for month_index, month_key in ordered_months[:6]:
                probs_raw = model_spd.get(month_key) or []
                probs_vec = sanitize_mcq_vector(list(probs_raw), n_options=bucket_count)
                if len(probs_vec) != bucket_count:
                    probs_vec = [1.0 / bucket_count] * bucket_count
                spd_json_payload["spds"][month_key] = {"probs": probs_vec}
                for bucket_index, prob in enumerate(probs_vec[:bucket_count], start=1):
                    cb = class_bins[bucket_index - 1] if 0 <= bucket_index - 1 < len(class_bins) else str(bucket_index)
                    con.execute(
                        """
                        INSERT INTO forecasts_raw (
                            run_id, question_id, model_name, month_index, bucket_index,
                            probability, ok, elapsed_ms, cost_usd, prompt_tokens,
                            completion_tokens, total_tokens, status, spd_json, human_explanation,
                            horizon_m, class_bin, p
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ok', ?, NULL, ?, ?, ?)
                        """,
                        [
                            run_id,
                            qid,
                            model_name,
                            month_index,
                            bucket_index,
                            float(prob),
                            True,
                            elapsed_ms,
                            cost_usd,
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                            json.dumps(spd_json_payload),
                            month_index,
                            cb,
                            float(prob),
                        ],
                    )
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Failed to write SPD member rows for q={qid}: {exc}")
    finally:
        try:
            con.close()
        except Exception:
            pass

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


def _safe_row_get(row: Any, index: int, name: str) -> Any:
    """
    Safely get a column from a duckdb row that might be:
      - a mapping-like object (supports row[name]), or
      - a tuple/list (positional access only).

    - index: positional index of the column in the SELECT clause.
    - name: column name to try first when row is dict-like.

    This keeps our loader robust whether duckdb returns duckdb.Row or plain tuples.
    """
    try:
        return row[name]  # type: ignore[index]
    except Exception:
        return row[index]


def _row_to_pythia_question(row: Any) -> PythiaQuestion:
    """
    Convert a questions row (duckdb.Row or tuple) into a PythiaQuestion.

    Assumes the SELECT order used in _load_pythia_questions:
      0: question_id
      1: hs_run_id
      2: scenario_ids_json
      3: iso3
      4: hazard_code
      5: metric
      6: target_month
      7: window_start_date
      8: window_end_date
      9: wording
     10: status
     11: pythia_metadata_json
    """

    return PythiaQuestion(
        question_id=_safe_row_get(row, 0, "question_id"),
        hs_run_id=_safe_row_get(row, 1, "hs_run_id"),
        iso3=_safe_row_get(row, 3, "iso3"),
        hazard_code=_safe_row_get(row, 4, "hazard_code"),
        metric=_safe_row_get(row, 5, "metric"),
        target_month=_safe_row_get(row, 6, "target_month"),
        window_start_date=_safe_row_get(row, 7, "window_start_date"),
        window_end_date=_safe_row_get(row, 8, "window_end_date"),
        wording=_safe_row_get(row, 9, "wording"),
        status=_safe_row_get(row, 10, "status"),
        pythia_metadata_json=_safe_row_get(row, 11, "pythia_metadata_json"),
        scenario_ids_json=_safe_row_get(row, 2, "scenario_ids_json"),
    )


def _pythia_question_to_post(question: PythiaQuestion) -> Optional[dict]:
    meta = _as_dict(question.pythia_metadata_json or {})
    if meta.get("source") == "demo":
        return None

    qid = question.question_id
    iso3 = (question.iso3 or "").upper()
    hz = (question.hazard_code or "").upper()
    metric = question.metric or "PA"
    target_month = question.target_month or ""
    wording = question.wording or ""
    hs_run_id = question.hs_run_id

    scenario_ids = _safe_json_load(question.scenario_ids_json or "[]") or []

    question_block = {
        "id": qid,
        "title": wording,
        "type": "spd",
        "possibilities": {"type": "spd"},
    }

    return {
        "id": qid,
        "question": question_block,
        "description": "",
        "pythia_iso3": iso3,
        "pythia_hazard_code": hz,
        "pythia_metric": metric,
        "pythia_target_month": target_month,
        "pythia_status": question.status,
        "pythia_hs_run_id": hs_run_id,
        "pythia_scenario_ids": scenario_ids,
        "pythia_window_start_date": question.window_start_date,
        "pythia_window_end_date": question.window_end_date,
        "pythia_metadata": meta,
        "created_time_iso": datetime.utcnow().isoformat(),
    }


def _load_pythia_questions(
    limit: Optional[int] = None, iso3_filter: Optional[Set[str]] = None
) -> List[PythiaQuestion]:
    """
    Load questions for Pythia runs with epoch-aware gating:

    - Exclude ACO entirely.
    - Restrict iso3 to iso3_filter if provided; otherwise default to iso3 values
      present in hs_triage. If hs_triage is empty and no filter is provided,
      fall back to all iso3s present in the questions table.
    - For each (iso3, hazard_code, metric), prefer HS-generated questions
      (hs_run_id not null). Among HS questions, only those with the *latest*
      hs_run_id for that (iso3, hazard_code) are kept.
    - Only when no HS-driven questions exist for a triple, fall back to legacy
      static questions (hs_run_id is null/empty), still excluding ACO.
    """

    con = pythia_connect(read_only=True)

    if iso3_filter:
        iso3_allowed = {code.upper() for code in iso3_filter if code}
    else:
        rows = con.execute(
            "SELECT DISTINCT iso3 FROM hs_triage ORDER BY iso3"
        ).fetchall()
        iso3_allowed = {_safe_row_get(r, 0, "iso3") for r in rows}

    if not iso3_allowed:
        LOG.warning(
            "No iso3s found in hs_triage and no iso3 filter provided; "
            "falling back to all iso3s in questions."
        )
        rows = con.execute(
            "SELECT DISTINCT iso3 FROM questions ORDER BY iso3"
        ).fetchall()
        iso3_allowed = {_safe_row_get(r, 0, "iso3") for r in rows}

    if not iso3_allowed:
        LOG.warning("No iso3s available for Pythia loader; returning empty set.")
        con.close()
        return []

    iso3_params = sorted(iso3_allowed)
    placeholders = ",".join(["?"] * len(iso3_params))

    LOG.info("Pythia loader iso3_allowed: %s", iso3_params)

    hs_latest_sql = f"""
        SELECT
            iso3,
            hazard_code,
            MAX(run_id) AS hs_run_id
        FROM hs_triage
        WHERE iso3 IN ({placeholders})
        GROUP BY iso3, hazard_code
    """
    hs_latest_rows = con.execute(hs_latest_sql, iso3_params).fetchall()
    latest_hs_by_iso_hz: Dict[Tuple[str, str], str] = {}
    for row in hs_latest_rows:
        iso3 = _safe_row_get(row, 0, "iso3")
        hz = _safe_row_get(row, 1, "hazard_code")
        hs_run_id = _safe_row_get(row, 2, "hs_run_id")
        key = (iso3, hz)
        latest_hs_by_iso_hz[key] = hs_run_id

    hs_questions: List[PythiaQuestion] = []
    if latest_hs_by_iso_hz:
        hs_all_sql = f"""
            SELECT
                question_id, hs_run_id, scenario_ids_json,
                iso3, hazard_code, metric,
                target_month, window_start_date, window_end_date,
                wording, status, pythia_metadata_json
            FROM questions
            WHERE status = 'active'
              AND hs_run_id IS NOT NULL
              AND hs_run_id <> ''
              AND UPPER(COALESCE(hazard_code, '')) <> 'ACO'
              AND iso3 IN ({placeholders})
        """
        hs_all_rows = con.execute(hs_all_sql, iso3_params).fetchall()
        for row in hs_all_rows:
            q = _row_to_pythia_question(row)
            key_iso_hz = (q.iso3, q.hazard_code)
            latest_hs = latest_hs_by_iso_hz.get(key_iso_hz)
            if latest_hs is not None and q.hs_run_id == latest_hs:
                hs_questions.append(q)

    hs_triples = {(q.iso3, q.hazard_code, q.metric) for q in hs_questions}

    legacy_sql = f"""
        SELECT
            question_id, hs_run_id, scenario_ids_json,
            iso3, hazard_code, metric,
            target_month, window_start_date, window_end_date,
            wording, status, pythia_metadata_json
        FROM questions
        WHERE status = 'active'
          AND (hs_run_id IS NULL OR hs_run_id = '')
          AND UPPER(COALESCE(hazard_code, '')) <> 'ACO'
          AND iso3 IN ({placeholders})
    """
    legacy_rows = con.execute(legacy_sql, iso3_params).fetchall()
    legacy_questions: List[PythiaQuestion] = []
    for row in legacy_rows:
        q = _row_to_pythia_question(row)
        triple = (q.iso3, q.hazard_code, q.metric)
        if triple in hs_triples:
            continue
        legacy_questions.append(q)

    con.close()

    all_questions = hs_questions + legacy_questions

    if limit is not None and limit > 0 and len(all_questions) > limit:
        all_questions = all_questions[:limit]

    selected_ids = sorted({q.question_id for q in all_questions})
    LOG.info(
        "Pythia loader selected %d questions: %s",
        len(selected_ids),
        selected_ids,
    )

    LOG.info(
        "Pythia question loader: %d HS-driven questions (latest per triple), %d legacy fallback questions (ACO excluded).",
        len(hs_questions),
        len(legacy_questions),
    )

    if not all_questions:
        LOG.warning(
            "Pythia question loader returned an empty set. Check HS runs, iso3 filter, and the questions table."
        )

    return all_questions


def _load_research_json(run_id: str, question_id: str) -> Optional[Dict[str, Any]]:
    con = connect(read_only=True)
    try:
        row = con.execute(
            """
            SELECT research_json
            FROM question_research
            WHERE run_id = ? AND question_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [run_id, question_id],
        ).fetchone()
        if not row:
            return None
        return json.loads(row[0])
    finally:
        con.close()


def _load_question_evidence_pack(run_id: str, question_id: str) -> Optional[Dict[str, Any]]:
    con = connect(read_only=True)
    try:
        row = con.execute(
            """
            SELECT question_evidence_json
            FROM question_research
            WHERE run_id = ? AND question_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [run_id, question_id],
        ).fetchone()
        if not row or not row[0]:
            return None
        return json.loads(row[0])
    finally:
        con.close()


def _persist_question_evidence_pack(
    run_id: str,
    question_id: str,
    iso3: str,
    hazard_code: str,
    metric: str,
    question_evidence_pack: Dict[str, Any],
) -> None:
    con = connect(read_only=False)
    try:
        ensure_schema(con)
        con.execute(
            """
            INSERT INTO question_research
              (run_id, question_id, iso3, hazard_code, metric, research_json, hs_evidence_json, question_evidence_json, merged_evidence_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                question_id,
                iso3,
                hazard_code,
                metric,
                _json_dumps_for_db({"note": "retriever_only"}),
                _json_dumps_for_db({}),
                _json_dumps_for_db(question_evidence_pack or {}),
                _json_dumps_for_db(question_evidence_pack or {}),
            ],
        )
    finally:
        con.close()

def _load_hs_country_evidence_pack(hs_run_id: str, iso3: str) -> Optional[Dict[str, Any]]:
    """Load HS country evidence pack (markdown + sources) from DuckDB."""

    iso3_up = (iso3 or "").upper()
    if not hs_run_id or not iso3_up:
        return None

    con = connect(read_only=False)
    try:
        ensure_schema(con)
        row = con.execute(
            """
            SELECT report_markdown, sources_json, grounded, grounding_debug_json, structural_context, recent_signals_json
            FROM hs_country_reports
            WHERE hs_run_id = ? AND iso3 = ?
            LIMIT 1
            """,
            [hs_run_id, iso3_up],
        ).fetchone()
    finally:
        con.close()

    if not row:
        return None

    markdown = row[0] or ""
    sources_raw = row[1] or "[]"
    grounded_val = row[2] if len(row) > 2 else False
    grounding_debug_raw = row[3] if len(row) > 3 else "{}"
    structural_context = row[4] if len(row) > 4 else ""
    recent_signals_raw = row[5] if len(row) > 5 else "[]"
    try:
        sources = json.loads(sources_raw)
    except Exception:
        sources = []
    try:
        grounding_debug = json.loads(grounding_debug_raw)
    except Exception:
        grounding_debug = {}
    try:
        recent_signals = json.loads(recent_signals_raw)
    except Exception:
        recent_signals = []

    return {
        "markdown": markdown,
        "sources": sources if isinstance(sources, list) else [],
        "structural_context": structural_context or "",
        "recent_signals": recent_signals if isinstance(recent_signals, list) else [],
        "grounded": bool(grounded_val) or bool(sources),
        "debug": grounding_debug if isinstance(grounding_debug, dict) else {},
    }


def _truncate_tail_pack_signals(signals: list[str], max_bullets: int) -> list[str]:
    buckets = {
        "TRIGGER": [],
        "DAMPENER": [],
        "BASELINE": [],
        "OTHER": [],
    }
    for item in signals:
        if not isinstance(item, str):
            continue
        text = item.strip()
        if not text:
            continue
        upper = text.upper()
        if upper.startswith("TRIGGER"):
            buckets["TRIGGER"].append(text)
        elif upper.startswith("DAMPENER"):
            buckets["DAMPENER"].append(text)
        elif upper.startswith("BASELINE"):
            buckets["BASELINE"].append(text)
        else:
            buckets["OTHER"].append(text)

    ordered: list[str] = []
    for key in ("TRIGGER", "DAMPENER", "BASELINE", "OTHER"):
        ordered.extend(buckets[key])

    truncated = ordered[: max(0, max_bullets)]
    LOG.debug(
        "Tail pack signals truncated: in=%d out=%d trigger=%d dampener=%d baseline=%d other=%d",
        len(signals),
        len(truncated),
        len(buckets["TRIGGER"]),
        len(buckets["DAMPENER"]),
        len(buckets["BASELINE"]),
        len(buckets["OTHER"]),
    )
    return truncated


def _load_hs_hazard_tail_pack(
    hs_run_id: str,
    iso3: str,
    hazard_code: str,
    max_signals: int | None = None,
) -> Optional[Dict[str, Any]]:
    iso3_up = (iso3 or "").upper()
    hazard_up = (hazard_code or "").upper()
    if not hs_run_id or not iso3_up or not hazard_up:
        return None

    con = connect(read_only=True)
    try:
        try:
            table_info = con.execute("PRAGMA table_info('hs_hazard_tail_packs')").fetchall()
        except Exception:
            LOG.debug("HS hazard tail pack table missing for %s/%s", iso3_up, hazard_up)
            return None
        if not table_info:
            return None
        row = con.execute(
            """
            SELECT report_markdown, sources_json, grounded, grounding_debug_json, structural_context,
                   recent_signals_json, rc_level, rc_score, rc_direction, rc_window, query, created_at
            FROM hs_hazard_tail_packs
            WHERE hs_run_id = ? AND upper(iso3) = upper(?) AND upper(hazard_code) = upper(?)
            ORDER BY created_at DESC NULLS LAST
            LIMIT 1
            """,
            [hs_run_id, iso3_up, hazard_up],
        ).fetchone()
    finally:
        con.close()

    if not row:
        return None

    sources_raw = row[1] or "[]"
    grounded_val = row[2] if len(row) > 2 else False
    grounding_debug_raw = row[3] if len(row) > 3 else "{}"
    structural_context = row[4] if len(row) > 4 else ""
    recent_signals_raw = row[5] if len(row) > 5 else "[]"
    rc_level = row[6] if len(row) > 6 else None
    rc_score = row[7] if len(row) > 7 else None
    rc_direction = row[8] if len(row) > 8 else None
    rc_window = row[9] if len(row) > 9 else None
    query = row[10] if len(row) > 10 else ""
    created_at = row[11] if len(row) > 11 else None

    try:
        sources = json.loads(sources_raw)
    except Exception:
        sources = []
    try:
        grounding_debug = json.loads(grounding_debug_raw)
    except Exception:
        grounding_debug = {}
    try:
        recent_signals = json.loads(recent_signals_raw)
    except Exception:
        recent_signals = []

    if max_signals is not None and isinstance(recent_signals, list):
        recent_signals = _truncate_tail_pack_signals(recent_signals, max_signals)

    return {
        "query": query or "",
        "report_markdown": row[0] or "",
        "sources": sources if isinstance(sources, list) else [],
        "structural_context": structural_context or "",
        "recent_signals": recent_signals if isinstance(recent_signals, list) else [],
        "grounded": bool(grounded_val) or bool(sources),
        "debug": {
            "rc_level": rc_level,
            "rc_score": rc_score,
            "rc_direction": rc_direction,
            "rc_window": rc_window,
            "created_at": str(created_at) if created_at is not None else None,
            "grounding_debug": grounding_debug if isinstance(grounding_debug, dict) else {},
            "truncated_to": max_signals if max_signals is not None else None,
        },
    }


def _load_structured_data(
    iso3: str,
    hazard_code: str,
    hs_run_id: str | None = None,
    rc_level: int | None = None,
) -> Dict[str, Any]:
    """Load all structured connector data for SPD prompt injection.

    Returns a dict with keys matching what ``build_spd_prompt_v2()``
    expects in its ``structured_data`` parameter. Each source is loaded
    independently; failures are silently skipped.
    """
    sd: Dict[str, Any] = {}

    try:
        from horizon_scanner.reliefweb import load_reliefweb_reports
        rw = load_reliefweb_reports(iso3)
        if rw:
            sd["reliefweb_reports"] = rw
    except Exception:
        pass

    if hazard_code in ("ACE", "DI"):
        try:
            from pythia.acled_political import load_acled_political_events
            pol = load_acled_political_events(iso3)
            if pol:
                sd["acled_political_events"] = pol
        except Exception:
            pass

    try:
        from pythia.ipc_phases import load_ipc_phases
        ipc = load_ipc_phases(iso3)
        if ipc:
            sd["ipc_phases"] = ipc
    except Exception:
        pass

    try:
        from pythia.acaps import load_inform_severity
        inform = load_inform_severity(iso3)
        if inform:
            sd["inform_severity"] = inform
    except Exception:
        pass

    try:
        from pythia.acaps import load_risk_radar
        risks = load_risk_radar(iso3)
        if risks:
            sd["acaps_risk_radar"] = risks
    except Exception:
        pass

    try:
        from pythia.acaps import load_daily_monitoring
        monitoring = load_daily_monitoring(iso3)
        if monitoring:
            sd["acaps_monitoring"] = monitoring
    except Exception:
        pass

    # Conflict forecasts (ACE only)
    if hazard_code.upper() == "ACE":
        try:
            from horizon_scanner.conflict_forecasts import load_conflict_forecasts, format_conflict_forecasts_for_research
            forecasts = load_conflict_forecasts(iso3)
            if forecasts:
                sd["conflict_forecasts"] = format_conflict_forecasts_for_research(forecasts)
        except Exception:
            pass

    # ICG CrisisWatch (ACE only)
    if hazard_code.upper() == "ACE":
        try:
            from horizon_scanner.crisiswatch import format_crisiswatch_for_prompt
            cw_text = format_crisiswatch_for_prompt(iso3)
            if cw_text:
                sd["crisiswatch"] = cw_text
        except Exception:
            pass

    # HDX Signals (all hazards)
    try:
        from horizon_scanner.hdx_signals import format_hdx_signals_for_prompt
        hdx_text = format_hdx_signals_for_prompt(iso3, hazard_code)
        if hdx_text:
            sd["hdx_signals"] = hdx_text
    except Exception:
        pass

    # Seasonal TC forecasts (TC only)
    if hazard_code.upper() == "TC":
        try:
            from horizon_scanner.seasonal_tc import get_seasonal_tc_context_for_country
            stc = get_seasonal_tc_context_for_country(iso3)
            if stc:
                sd["seasonal_tc_context"] = stc
        except Exception:
            pass

    # ENSO context (all climate hazards)
    if hazard_code.upper() in {"TC", "FL", "DR", "HW"}:
        try:
            from horizon_scanner.enso import get_enso_prompt_context
            enso_ctx = get_enso_prompt_context()
            if enso_ctx:
                sd["enso_context"] = enso_ctx
        except Exception:
            pass

    # Adversarial check (RC L2+ only) — load from DB if available
    if (rc_level is not None and rc_level >= 2) and hs_run_id:
        try:
            con = connect(read_only=True)
            try:
                row = con.execute(
                    """
                    SELECT payload_json
                    FROM hs_adversarial_checks
                    WHERE hs_run_id = ? AND upper(iso3) = upper(?) AND upper(hazard_code) = upper(?)
                    ORDER BY created_at DESC NULLS LAST
                    LIMIT 1
                    """,
                    [hs_run_id, iso3.upper(), hazard_code.upper()],
                ).fetchone()
            finally:
                con.close()
            if row and row[0]:
                adv = json.loads(row[0])
                if adv:
                    sd["adversarial_check"] = adv
        except Exception:
            pass

    # Load RC/triage grounding evidence if available
    if hs_run_id:
        try:
            tail_pack = _load_hs_hazard_tail_pack(hs_run_id, iso3, hazard_code)
            if tail_pack:
                sd["hazard_grounding"] = tail_pack
        except Exception:
            pass

    return sd


def _build_question_evidence_query(question_row: duckdb.Row, wording: str) -> str:
    iso3 = (question_row.get("iso3") or "").upper()
    hazard = (question_row.get("hazard_code") or "").upper()
    metric = (question_row.get("metric") or "").upper()

    hazard_label = HZ_QUERY_MAP.get(hazard, hazard)
    target_month = question_row.get("target_month") or ""
    window_end = question_row.get("window_end_date") or ""

    timeframe = ""
    if target_month:
        timeframe = f" starting {target_month}"
    elif window_end:
        timeframe = f" through {window_end}"

    country = _country_label(iso3)
    return (
        f"{country} {hazard_label} {metric} outlook{timeframe} — gather recent signals (last 120 days) "
        "and concise structural drivers (max 8 lines). "
        f"Question focus: {wording or ''}"
    )


def _build_question_evidence_queries(
    question_row: duckdb.Row, wording: str, hs_entry: dict[str, Any]
) -> list[str]:
    base_query = _build_question_evidence_query(question_row, wording)
    tier = str(hs_entry.get("tier") or "").lower()
    if tier != "priority":
        return [base_query]

    iso3 = (question_row.get("iso3") or "").upper()
    hazard = (question_row.get("hazard_code") or "").upper()
    metric = (question_row.get("metric") or "").upper()
    hazard_label = HZ_QUERY_MAP.get(hazard, hazard)
    target_month = question_row.get("target_month") or ""
    window_end = question_row.get("window_end_date") or ""

    timeframe = ""
    if target_month:
        timeframe = f" starting {target_month}"
    elif window_end:
        timeframe = f" through {window_end}"

    country = _country_label(iso3)
    targeted_query = (
        f"{country} {hazard_label} {metric} hazard-specific outlook{timeframe} — "
        "focus on targeted drivers, regime shifts, and near-term signals."
    )
    return [base_query, targeted_query]


def _retriever_enabled() -> bool:
    return os.getenv("PYTHIA_RETRIEVER_ENABLED", "0") == "1"


def _retriever_model_id() -> str | None:
    model_id = (os.getenv("PYTHIA_RETRIEVER_MODEL_ID") or "").strip()
    return model_id or None


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


def _country_label(iso3: str | None) -> str:
    iso3_val = (iso3 or "").strip().upper()
    if not iso3_val:
        return ""
    name = _load_country_names().get(iso3_val)
    if name:
        return f"{name} ({iso3_val})"
    return f"Unknown Country ({iso3_val})"


def _merge_question_evidence_packs(
    packs: list[dict[str, Any]],
    queries: list[str],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for pack in packs:
        merged = merge_evidence_packs(merged, pack, max_sources=10, max_signals=12)

    recency_days = None
    for pack in packs:
        if pack.get("recency_days"):
            recency_days = pack.get("recency_days")
            break

    error_obj = None
    for pack in packs:
        if pack.get("error"):
            error_obj = pack.get("error")
            break

    merged_pack = {
        "query": " | ".join(queries),
        "recency_days": recency_days or 120,
        "structural_context": merged.get("structural_context", ""),
        "recent_signals": merged.get("recent_signals", []),
        "sources": merged.get("sources", []),
        "unverified_sources": merged.get("unverified_sources", []),
        "grounded": merged.get("grounded", False),
        "debug": {"retriever_queries": queries, "retriever_enabled": _retriever_enabled()},
    }
    if error_obj:
        merged_pack["error"] = error_obj
    return merged_pack


def _build_spd_web_search_query(
    *,
    iso3: str | None,
    hazard_code: str | None,
    metric: str | None,
    target_month: str | None,
    wording: str | None,
) -> str:
    iso3_val = (iso3 or "").upper()
    hazard = (hazard_code or "").upper()
    metric_val = (metric or "").upper()
    hazard_label = HZ_QUERY_MAP.get(hazard, hazard)
    timeframe = f" starting {target_month}" if target_month else ""
    wording_val = wording or ""
    country = _country_label(iso3_val)
    return (
        f"{country} {hazard_label} {metric_val} outlook{timeframe} — gather recent signals (last 120 days) "
        "and concise structural drivers (max 8 lines). "
        f"Question focus: {wording_val}"
    )


def _sanitize_sources(sources: Any) -> list[str]:
    """Return a deduped, order-stable list of real URLs."""

    clean: list[str] = []
    if not sources:
        return clean

    placeholders = {"...", "url", "url1", "url2", "url3"}
    for src in sources:
        if isinstance(src, dict):
            url = src.get("url") or ""
        else:
            url = src
        url = str(url or "").strip()
        if not url or url in placeholders:
            continue
        if not (url.startswith("http://") or url.startswith("https://")):
            continue
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if not host:
            continue
        if any(bad in host for bad in ("example.com", "example.org", "localhost", "placeholder")):
            continue
        if host in {"127.0.0.1", "0.0.0.0"}:
            continue
        if url not in clean:
            clean.append(url)
    return clean


def _render_verified_sources_block(sources: list[dict[str, Any]] | None, max_sources: int) -> str:
    if not sources:
        return ""
    lines = ["", "VERIFIED SOURCES (web_search_call.action.sources):"]
    count = 0
    for src in sources:
        if count >= max_sources:
            break
        if not isinstance(src, dict):
            continue
        url = str(src.get("url") or src.get("uri") or "").strip()
        if not url:
            continue
        title = str(src.get("title") or url).strip()
        lines.append(f"- {title} — {url}")
        count += 1
    if count == 0:
        return ""
    return "\n".join(lines)


def _normalize_and_enforce_grounding(research_json: Dict[str, Any], merged_pack: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize groundedness: only verified retrieval sources can set grounded=true."""

    research = dict(research_json or {})
    merged = merged_pack or {}

    verified_urls = _sanitize_sources([s.get("url") if isinstance(s, dict) else s for s in (merged.get("sources") or [])])
    pack_unverified = _sanitize_sources(
        [s.get("url") if isinstance(s, dict) else s for s in (merged.get("unverified_sources") or [])]
    )
    model_urls = _sanitize_sources(research.get("sources") or [])
    model_unverified = _sanitize_sources(research.get("unverified_sources") or [])

    unverified_urls: list[str] = []
    for url in pack_unverified + model_urls + model_unverified:
        if url in verified_urls or url in unverified_urls:
            continue
        unverified_urls.append(url)
        if len(unverified_urls) >= 24:
            break

    combined_sources: list[str] = []
    for url in verified_urls + unverified_urls:
        if url not in combined_sources:
            combined_sources.append(url)
        if len(combined_sources) >= 24:
            break

    research["verified_sources"] = verified_urls
    research["unverified_sources"] = unverified_urls
    research["sources"] = combined_sources
    research["grounded"] = bool(verified_urls)
    research["_grounding_enforced"] = True
    research["_grounding_verified_sources_count"] = len(verified_urls)
    research["_grounding_sources_count"] = len(combined_sources)
    return research


def _question_needs_spd(run_id: str, question_row: duckdb.Row) -> bool:
    iso3 = (question_row.get("iso3") or "").upper()
    hazard_code = (question_row.get("hazard_code") or "").upper()
    hs_run_id = question_row.get("hs_run_id") or run_id
    triage = load_hs_triage_entry(hs_run_id, iso3, hazard_code)
    if not triage:
        return True
    return bool(triage.get("need_full_spd", False))


def _should_run_research(run_id: str, question_row: duckdb.Row) -> bool:
    return _question_needs_spd(run_id, question_row)



async def _call_spd_model(
    prompt: str,
    *,
    run_id: str | None = None,
    question_id: str | None = None,
    iso3: str | None = None,
    hazard_code: str | None = None,
    metric: str | None = None,
    target_month: str | None = None,
    wording: str | None = None,
) -> tuple[str, Dict[str, Any], Optional[str], ModelSpec]:
    """Async wrapper for the SPD LLM call for v2 pipeline."""

    ms_default: ModelSpec | None = None
    for candidate in SPD_ENSEMBLE:
        if getattr(candidate, "active", False):
            ms_default = candidate
            break
        if ms_default is None:
            ms_default = candidate

    ms = (
        ModelSpec(**ms_default.__dict__)
        if ms_default is not None
        else ModelSpec(name="Gemini", provider="google", model_id=GEMINI_MODEL_ID, active=True, purpose="spd_v2")
    )
    return await _call_spd_model_for_spec(
        ms,
        prompt,
        run_id=run_id,
        question_id=question_id,
        iso3=iso3,
        hazard_code=hazard_code,
        metric=metric,
        target_month=target_month,
        wording=wording,
    )


async def _call_spd_model_for_spec(
    ms: ModelSpec,
    prompt: str,
    *,
    run_id: str | None = None,
    question_id: str | None = None,
    iso3: str | None = None,
    hazard_code: str | None = None,
    metric: str | None = None,
    target_month: str | None = None,
    wording: str | None = None,
    **_kwargs,
) -> tuple[str, Dict[str, Any], Optional[str], ModelSpec]:
    """Async wrapper for the SPD LLM call for a given model spec with self-search support."""

    if ms.purpose != "spd_v2":
        ms = ModelSpec(
            name=ms.name,
            provider=ms.provider,
            model_id=ms.model_id,
            weight=ms.weight,
            active=ms.active,
            purpose="spd_v2",
        )

    prompt_with_evidence = prompt
    if (
        model_self_search_enabled()
        and os.getenv("PYTHIA_SPD_WEB_SEARCH_ENABLED", "0") == "1"
        and ms.provider in {"openai", "anthropic"}
    ):
        query = _build_spd_web_search_query(
            iso3=iso3,
            hazard_code=hazard_code,
            metric=metric,
            target_month=target_month,
            wording=wording,
        )
        try:
            recency_days = int(os.getenv("PYTHIA_WEB_RESEARCH_RECENCY_DAYS", "120"))
        except Exception:
            recency_days = 120
        include_structural = os.getenv("PYTHIA_WEB_RESEARCH_INCLUDE_STRUCTURAL", "1") != "0"
        try:
            timeout_sec = int(os.getenv("PYTHIA_WEB_RESEARCH_TIMEOUT_SEC", "60"))
        except Exception:
            timeout_sec = 60
        try:
            max_results = int(os.getenv("PYTHIA_WEB_RESEARCH_MAX_RESULTS", "10"))
        except Exception:
            max_results = 10
        _, max_sources = self_search_limits()
        pack: dict[str, Any] | None = None
        error_text = None
        try:
            if ms.provider == "openai":
                from pythia.web_research.backends import openai_web_search

                pack = openai_web_search.fetch_via_openai_web_search(
                    query,
                    recency_days=recency_days,
                    include_structural=include_structural,
                    timeout_sec=timeout_sec,
                    max_results=max_results,
                ).to_dict()
            elif ms.provider == "anthropic":
                from pythia.web_research.backends import claude_web_search

                pack = claude_web_search.fetch_via_claude_web_search(
                    query,
                    recency_days=recency_days,
                    include_structural=include_structural,
                    timeout_sec=timeout_sec,
                    max_results=max_results,
                ).to_dict()
        except Exception as exc:  # noqa: BLE001
            error_text = f"forecast_web_research_error: {exc}"
            pack = {
                "query": query,
                "recency_days": recency_days,
                "grounded": False,
                "sources": [],
                "structural_context": "",
                "recent_signals": [],
                "error": {"type": "exception", "message": error_text},
            }

        if pack:
            trimmed_pack = trim_sources(pack, max_sources)
            prompt_with_evidence = append_evidence_to_prompt(prompt, trimmed_pack)
            verified_block = _render_verified_sources_block(
                trimmed_pack.get("sources") if isinstance(trimmed_pack, dict) else None,
                max_sources=min(3, max_sources),
            )
            if verified_block:
                prompt_with_evidence = f"{prompt_with_evidence}\n{verified_block}"
            usage = (pack.get("debug") or {}).get("usage") or {}
            error_obj = pack.get("error") or {}
            if not error_text:
                error_text = error_obj.get("message") if isinstance(error_obj, dict) else None
            if isinstance(error_obj, dict) and error_obj.get("code"):
                code_text = f"error_code={error_obj.get('code')}"
                if error_text:
                    error_text = f"{error_text} ({code_text})"
                else:
                    error_text = code_text
            await log_forecaster_llm_call(
                run_id=run_id or "forecast",
                question_id=question_id or "unknown",
                iso3=iso3,
                hazard_code=hazard_code,
                metric=metric,
                model_spec=ms,
                prompt_text=query,
                response_text=json.dumps(trimmed_pack, ensure_ascii=False),
                usage=usage,
                error_text=error_text,
                phase="forecast_web_research",
                call_type="forecast_web_research",
            )
    if (
        model_self_search_enabled()
        and os.getenv("PYTHIA_SPD_GOOGLE_WEB_SEARCH_ENABLED", "0") == "1"
        and ms.provider == "google"
    ):
        query = _build_spd_web_search_query(
            iso3=iso3,
            hazard_code=hazard_code,
            metric=metric,
            target_month=target_month,
            wording=wording,
        )
        try:
            recency_days = int(os.getenv("PYTHIA_WEB_RESEARCH_RECENCY_DAYS", "120"))
        except Exception:
            recency_days = 120
        include_structural = os.getenv("PYTHIA_WEB_RESEARCH_INCLUDE_STRUCTURAL", "1") != "0"
        try:
            timeout_sec = int(os.getenv("PYTHIA_WEB_RESEARCH_TIMEOUT_SEC", "60"))
        except Exception:
            timeout_sec = 60
        try:
            max_results = int(os.getenv("PYTHIA_WEB_RESEARCH_MAX_RESULTS", "10"))
        except Exception:
            max_results = 10
        _, max_sources = self_search_limits()
        pack: dict[str, Any] | None = None
        error_text = None
        try:
            from pythia.web_research.backends import gemini_grounding

            model_override = (os.getenv("PYTHIA_SPD_GOOGLE_MODEL_ID") or "gemini-2.5-pro").strip()
            pack = gemini_grounding.fetch_via_gemini(
                query,
                recency_days=recency_days,
                include_structural=include_structural,
                timeout_sec=timeout_sec,
                max_results=max_results,
                model_id=model_override or None,
            ).to_dict()
        except Exception as exc:  # noqa: BLE001
            error_text = f"forecast_web_research_error: {exc}"
            pack = {
                "query": query,
                "recency_days": recency_days,
                "grounded": False,
                "sources": [],
                "structural_context": "",
                "recent_signals": [],
                "error": {"type": "exception", "message": error_text},
            }

        if pack:
            trimmed_pack = trim_sources(pack, max_sources)
            prompt_with_evidence = append_evidence_to_prompt(prompt_with_evidence, trimmed_pack)
            verified_block = _render_verified_sources_block(
                trimmed_pack.get("sources") if isinstance(trimmed_pack, dict) else None,
                max_sources=min(3, max_sources),
            )
            if verified_block:
                prompt_with_evidence = f"{prompt_with_evidence}\n{verified_block}"
            usage = (pack.get("debug") or {}).get("usage") or {}
            error_obj = pack.get("error") or {}
            if not error_text:
                error_text = error_obj.get("message") if isinstance(error_obj, dict) else None
            await log_forecaster_llm_call(
                run_id=run_id or "forecast",
                question_id=question_id or "unknown",
                iso3=iso3,
                hazard_code=hazard_code,
                metric=metric,
                model_spec=ms,
                prompt_text=query,
                response_text=json.dumps(trimmed_pack, ensure_ascii=False),
                usage=usage,
                error_text=error_text,
                phase="forecast_web_research",
                call_type="forecast_web_research",
            )

    start = time.time()
    try:
        text, usage, error = await call_chat_ms(
            ms,
            prompt_with_evidence,
            temperature=0.2,
            prompt_key="spd.v2",
            prompt_version="1.0.0",
            component="Forecaster",
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.time() - start) * 1000)
        return "", {"elapsed_ms": elapsed_ms}, f"provider call error: {exc}", ms

    usage = dict(usage or {})
    usage.setdefault("elapsed_ms", int((time.time() - start) * 1000))

    query = extract_self_search_query(text or "")
    max_calls, max_sources = self_search_limits()

    if not query:
        return text, usage, error, ms

    if not self_search_enabled() or max_calls <= 0:
        usage["self_search"] = {
            "attempted": False,
            "requested": True,
            "succeeded": False,
            "query": query,
            "reason": "self_search_disabled",
        }
        return text, usage, error or "self_search_disabled", ms

    self_search_meta: dict[str, Any] = {
        "attempted": True,
        "requested": True,
        "query": query,
        "succeeded": False,
        "n_sources": 0,
    }

    evidence_pack: dict[str, Any] | None = None
    try:
        raw_pack = run_self_search(
            query,
            run_id=run_id,
            question_id=question_id,
            iso3=iso3,
            hazard_code=hazard_code,
            purpose="forecast_self_search",
        )
        evidence_pack = trim_sources(raw_pack or {}, max_sources)
        self_search_meta["succeeded"] = not evidence_pack.get("error")
        self_search_meta["n_sources"] = len(evidence_pack.get("sources") or [])
    except Exception as exc:  # noqa: BLE001
        evidence_pack = {"error": str(exc)}
        self_search_meta["succeeded"] = False
        self_search_meta["error"] = str(exc)

    usage["self_search"] = self_search_meta

    prompt_with_evidence = append_evidence_to_prompt(prompt, evidence_pack or {})

    try:
        text2, usage2, error2 = await call_chat_ms(
            ms,
            prompt_with_evidence,
            temperature=0.2,
            prompt_key="spd.v2.self_search",
            prompt_version="1.0.0",
            component="Forecaster",
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.time() - start) * 1000)
        self_search_meta["succeeded"] = False
        return text, usage, f"self_search_call_error: {exc}", ms

    usage2 = dict(usage2 or {})
    usage2.setdefault("elapsed_ms", int((time.time() - start) * 1000))
    usage2["self_search"] = self_search_meta

    query_second = extract_self_search_query(text2 or "")
    if query_second:
        return text2, combine_usage(usage, usage2), "self_search_second_request", ms

    return text2, combine_usage(usage, usage2), error2, ms


async def _call_spd_members_v2(
    prompt: str,
    specs: list[ModelSpec],
    *,
    run_id: str | None = None,
    question_id: str | None = None,
    iso3: str | None = None,
    hazard_code: str | None = None,
    metric: str | None = None,
    target_month: str | None = None,
    wording: str | None = None,
) -> tuple[
    list[dict[str, list[float]]],
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
]:
    """
    Call active SPD v2 members once and parse per-model SPDs.

    Returns:
      per_model_spds: list of {month: [probs]} dicts (may be empty per model)
      aggregated_usage: summed tokens/cost, max elapsed across models
      raw_calls: [{"model_spec", "text", "usage", "error"}]
      ensemble_meta: {"n_models_active", "n_models_called", "n_models_ok", "failed_providers", "partial_ensemble", "skipped_providers"}
    """

    specs_active = [ms for ms in specs if ms.active]
    skipped_providers = sorted(
        {ms.provider for ms in specs_active if is_provider_disabled_for_run(ms.provider, run_id)}
    )
    specs_used = [ms for ms in specs_active if ms.provider not in skipped_providers]
    ensemble_meta = {
        "n_models_active": len(specs_active),
        "n_models_called": len(specs_used),
        "n_models_ok": 0,
        "failed_providers": [],
        "partial_ensemble": bool(skipped_providers),
        "skipped_providers": skipped_providers,
    }

    if not specs_used:
        return [], {}, [], ensemble_meta

    tasks = [
        _call_spd_model_for_spec(
            ms,
            prompt,
            run_id=run_id,
            question_id=question_id,
            iso3=iso3,
            hazard_code=hazard_code,
            metric=metric,
            target_month=target_month,
            wording=wording,
        )
        for ms in specs_used
    ]
    call_results = await asyncio.gather(*tasks)

    per_model_spds: list[dict[str, list[float]]] = []
    raw_calls: list[dict[str, object]] = []

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cost = 0.0
    max_elapsed_ms = 0

    model_success: list[tuple[str, bool]] = []

    for text, usage, error, ms_val in call_results:
        usage = usage or {}
        raw_calls.append(
            {
                "model_spec": ms_val,
                "text": text or "",
                "usage": usage,
                "error": error,
            }
        )

        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens_val = int(usage.get("total_tokens") or 0)
        cost_usd_val = float(usage.get("cost_usd") or 0.0)
        elapsed_ms_val = int(usage.get("elapsed_ms") or 0)

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens += total_tokens_val
        total_cost += cost_usd_val
        max_elapsed_ms = max(max_elapsed_ms, elapsed_ms_val)

        model_spd: dict[str, list[float]] = {}

        if error or not text or not str(text).strip():
            per_model_spds.append(model_spd)
            model_success.append((getattr(ms_val, "provider", ""), False))
            continue

        try:
            spd_obj = _safe_json_loads(text)
        except Exception:  # noqa: BLE001
            per_model_spds.append(model_spd)
            model_success.append((getattr(ms_val, "provider", ""), False))
            continue

        if not isinstance(spd_obj, dict):
            per_model_spds.append(model_spd)
            model_success.append((getattr(ms_val, "provider", ""), False))
            continue

        spds = spd_obj.get("spds")
        if not isinstance(spds, dict):
            per_model_spds.append(model_spd)
            model_success.append((getattr(ms_val, "provider", ""), False))
            continue

        for month, payload in spds.items():
            if not isinstance(payload, dict):
                continue
            probs = payload.get("probs")
            if not isinstance(probs, list):
                continue
            vec = sanitize_mcq_vector(list(probs), n_options=5)
            model_spd[str(month)] = vec

        per_model_spds.append(model_spd)
        model_success.append((getattr(ms_val, "provider", ""), bool(model_spd)))

    aggregated_usage: dict[str, object] = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens or (total_prompt_tokens + total_completion_tokens),
        "cost_usd": total_cost,
        "elapsed_ms": max_elapsed_ms,
    }

    failed_providers = sorted({provider for provider, ok in model_success if not ok and provider})
    n_models_ok = sum(1 for _, ok in model_success if ok)
    ensemble_meta = {
        "n_models_active": len(specs_active),
        "n_models_called": len(specs_used),
        "n_models_ok": n_models_ok,
        "failed_providers": failed_providers,
        "partial_ensemble": n_models_ok < len(specs_active),
        "skipped_providers": skipped_providers,
    }

    return per_model_spds, aggregated_usage, raw_calls, ensemble_meta


async def _call_spd_members_v2_compat(
    prompt: str,
    specs: list[ModelSpec],
    *,
    run_id: str | None = None,
    question_id: str | None = None,
    iso3: str | None = None,
    hazard_code: str | None = None,
    metric: str | None = None,
    target_month: str | None = None,
    wording: str | None = None,
) -> tuple[list[dict[str, list[float]]], dict[str, object], list[dict[str, object]], dict[str, object]]:
    """
    Compatibility wrapper to avoid passing unsupported kwargs to monkeypatched callables in tests.
    """

    fn = _call_spd_members_v2
    try:
        sig = inspect.signature(fn)
        kwargs: dict[str, object] = {}
        if "run_id" in sig.parameters:
            kwargs["run_id"] = run_id
        if "question_id" in sig.parameters:
            kwargs["question_id"] = question_id
        if "iso3" in sig.parameters:
            kwargs["iso3"] = iso3
        if "hazard_code" in sig.parameters:
            kwargs["hazard_code"] = hazard_code
        if "metric" in sig.parameters:
            kwargs["metric"] = metric
        if "target_month" in sig.parameters:
            kwargs["target_month"] = target_month
        if "wording" in sig.parameters:
            kwargs["wording"] = wording
        return await fn(prompt, specs, **kwargs)
    except Exception:
        return await fn(prompt, specs, run_id=run_id)


async def _call_spd_model_compat(
    prompt: str,
    *,
    run_id: str | None = None,
    question_id: str | None = None,
    iso3: str | None = None,
    hazard_code: str | None = None,
    metric: str | None = None,
    target_month: str | None = None,
    wording: str | None = None,
) -> tuple[str, dict[str, Any], Optional[str], ModelSpec]:
    """
    Compatibility wrapper to avoid passing unsupported kwargs to monkeypatched callables in tests.
    """

    fn = _call_spd_model
    try:
        sig = inspect.signature(fn)
        kwargs: dict[str, object] = {}
        for key, value in {
            "run_id": run_id,
            "question_id": question_id,
            "iso3": iso3,
            "hazard_code": hazard_code,
            "metric": metric,
            "target_month": target_month,
            "wording": wording,
        }.items():
            if key in sig.parameters:
                kwargs[key] = value
        return await fn(prompt, **kwargs)
    except Exception:
        return await fn(prompt)


async def _call_spd_ensemble_v2(
    prompt: str,
    *,
    specs: list[ModelSpec] | None = None,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    """Thin wrapper around the current SPD v2 call path for diagnostics."""

    specs = specs or SPD_ENSEMBLE

    tasks = [_call_spd_model_for_spec(ms, prompt) for ms in specs if ms.active]
    if not tasks:
        return {}, {}, []

    call_results = await asyncio.gather(*tasks)

    raw_calls = [
        {
            "model_spec": ms_val,
            "text": text_val or "",
            "usage": usage_val,
            "error": error_val,
        }
        for text_val, usage_val, error_val, ms_val in call_results
    ]

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cost = 0.0
    max_elapsed_ms = 0

    month_sums: dict[str, list[float]] = {}
    month_counts: dict[str, int] = {}

    for text, usage, error, _ms in call_results:
        usage = usage or {}

        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens_val = int(usage.get("total_tokens") or 0)
        cost_usd_val = float(usage.get("cost_usd") or 0.0)
        elapsed_ms_val = int(usage.get("elapsed_ms") or 0)

        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens += total_tokens_val
        total_cost += cost_usd_val
        max_elapsed_ms = max(max_elapsed_ms, elapsed_ms_val)

        if error or not text or not str(text).strip():
            continue

        try:
            spd_obj = _safe_json_loads(text)
        except Exception:  # noqa: BLE001
            continue

        if not isinstance(spd_obj, dict):
            continue

        spds = spd_obj.get("spds")
        if not isinstance(spds, dict):
            continue

        for month, payload in spds.items():
            if not isinstance(payload, dict):
                continue
            probs = payload.get("probs")
            if not isinstance(probs, list):
                continue

            vec = sanitize_mcq_vector(list(probs), n_options=5)
            if month not in month_sums:
                month_sums[month] = [0.0 for _ in vec]
                month_counts[month] = 0

            if len(vec) != len(month_sums[month]):
                vec = vec[: len(month_sums[month])]

            for idx, val in enumerate(vec):
                month_sums[month][idx] += float(val)
            month_counts[month] += 1

    aggregated_usage: dict[str, object] = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens or (total_prompt_tokens + total_completion_tokens),
        "cost_usd": total_cost,
        "elapsed_ms": max_elapsed_ms,
    }

    if not month_sums:
        return {}, aggregated_usage, raw_calls

    spds_v2: dict[str, dict[str, object]] = {}
    for month, sums in month_sums.items():
        count = month_counts.get(month, 0)
        if count <= 0:
            continue
        avg_vec = [val / float(count) for val in sums]
        spds_v2[str(month)] = {"probs": sanitize_mcq_vector(avg_vec, n_options=len(avg_vec))}

    spd_obj: dict[str, object] = {"spds": spds_v2}

    return spd_obj, aggregated_usage, raw_calls


def _month_label_from_target(target_month: str, month_index: int) -> str | None:
    """Return YYYY-MM label offset by ``month_index`` months from ``target_month``."""

    try:
        base_month = datetime.strptime(target_month, "%Y-%m")
    except ValueError:
        return None

    total_months = (base_month.year * 12) + (base_month.month - 1) + month_index
    year = total_months // 12
    month = (total_months % 12) + 1

    return f"{year:04d}-{month:02d}"


def _is_calendar_month_key(key: str) -> bool:
    try:
        import re as _re  # local import to keep global imports minimal

        return bool(_re.match(r"^\d{4}-\d{2}$", str(key).strip()))
    except Exception:
        return False


def _parse_month_offset_key(key: str) -> int | None:
    """
    Parse month offset keys like 'month_0', 'month_1', 'm0', 'm1'.
    Returns the integer offset or None if not recognized.
    """
    if not isinstance(key, str):
        return None
    k = key.strip().lower()
    if k.startswith("month_") and k[6:].isdigit():
        try:
            return int(k[6:])
        except Exception:
            return None
    if k.startswith("m") and k[1:].isdigit():
        try:
            return int(k[1:])
        except Exception:
            return None
    return None


def _add_months(ym: str, offset: int) -> str:
    """Return YYYY-MM shifted by offset months (offset can be negative)."""
    parts = str(ym or "").split("-")
    if len(parts) != 2:
        return ""
    try:
        y = int(parts[0])
        m = int(parts[1])
    except Exception:
        return ""
    total_months = (y * 12 + (m - 1)) + int(offset)
    year = total_months // 12
    month = (total_months % 12) + 1
    return f"{year:04d}-{month:02d}"


def _expected_months(target_month: str, n: int = 6) -> list[str]:
    if not target_month:
        return []
    return [_add_months(target_month, i) for i in range(n)]


def _build_bayesmc_spd_obj(
    per_model_spds: list[dict[str, list[float]]],
    *,
    target_month: str | None,
    specs_used: list[ModelSpec],
) -> tuple[dict[str, object], dict[str, Any]]:
    spd_by_month, diag = aggregate_spd_v2_bayesmc(
        per_model_spds,
        n_buckets=5,
        prior_alpha=0.1,
        weights_by_model=None,
        model_names=[ms.name for ms in specs_used],
    )

    if not isinstance(diag, dict):
        diag = {"status": "unknown"}

    if not spd_by_month:
        diag.setdefault("status", "no_evidence_all_months")
        return {}, diag

    keys = list(spd_by_month.keys())
    normalized: dict[str, list[float]] = {}

    if keys and all(_is_calendar_month_key(k) for k in keys):
        normalized = dict(spd_by_month)
    elif any(_parse_month_offset_key(k) is not None for k in keys):
        if not target_month:
            diag["status"] = "missing_target_month"
            return {}, diag
        for k, vec in spd_by_month.items():
            offset = _parse_month_offset_key(k)
            if offset is None:
                continue
            ym = _add_months(target_month, offset)
            if ym:
                normalized[ym] = vec
        for k, vec in spd_by_month.items():
            if _is_calendar_month_key(k):
                normalized[str(k)] = vec
    else:
        diag["status"] = "unknown_month_labels"
        diag["sample_keys"] = sorted([str(k) for k in keys][:5])
        return {}, diag

    if target_month:
        expected_months = _expected_months(target_month, 6)
        missing_months = [m for m in expected_months if m not in normalized]
        if missing_months:
            diag["status"] = "insufficient_month_coverage"
            diag["missing_months"] = missing_months
            return {}, diag

    spd_obj: dict[str, object] = {"spds": {m: {"probs": vec} for m, vec in normalized.items()}}
    spd_obj["bayesmc_diag"] = diag

    return spd_obj, diag


async def _call_spd_bayesmc_v2(
    prompt: str,
    *,
    run_id: str,
    question_id: str,
    hs_run_id: str | None,
    target_month: str | None = None,
    specs: list[ModelSpec] | None = None,
    iso3: str | None = None,
    hazard_code: str | None = None,
    metric: str | None = None,
    wording: str | None = None,
) -> tuple[
    dict[str, object],
    dict[str, object],
    list[dict[str, object]],
    dict[str, object],
    list[dict[str, list[float]]],
    list[ModelSpec],
]:
    """
    BayesMC-backed SPD v2 bridge.

    Returns:
      - spd_obj: {"spds": {month_key: {"probs": [...]}}}
      - aggregated_usage: aggregated usage across members
      - raw_calls: list of member call summaries (model_spec, text, usage, error)
      - ensemble_meta: {"n_models_active", "n_models_called", "n_models_ok", "failed_providers", "partial_ensemble", "skipped_providers"}

    BayesMC emits month_* labels; when provided, ``target_month`` anchors those labels
    to YYYY-MM calendar months for compatibility with SPD v2 compare artifacts.
    """
    specs = specs or DEFAULT_ENSEMBLE
    specs_used = [ms for ms in specs if ms.active]

    if not specs_used:
        ensemble_meta: dict[str, object] = {
            "n_models_active": 0,
            "n_models_called": 0,
            "n_models_ok": 0,
            "failed_providers": [],
            "partial_ensemble": False,
            "skipped_providers": disabled_providers_for_run(run_id),
        }
        return {}, {}, [], ensemble_meta, [], []

    per_model_spds, aggregated_usage, raw_calls, ensemble_meta = await _call_spd_members_v2_compat(
        prompt,
        specs_used,
        run_id=run_id,
        question_id=question_id,
        iso3=iso3,
        hazard_code=hazard_code,
        metric=metric,
        target_month=target_month,
        wording=wording,
    )
    member_raw_by_model_id: dict[str, str] = {}
    for rc in raw_calls:
        try:
            ms = rc.get("model_spec")
            if isinstance(ms, ModelSpec):
                member_raw_by_model_id[ms.model_id] = str(rc.get("text") or "")
        except Exception:
            continue

    spd_obj, _diag = _build_bayesmc_spd_obj(
        per_model_spds, target_month=target_month, specs_used=specs_used
    )
    try:
        ensemble_meta["bayesmc_diag"] = _diag
    except Exception:
        pass

    if spd_obj:
        _attach_ensemble_meta(spd_obj, ensemble_meta)

    if not spd_obj:
        return {}, aggregated_usage, raw_calls, ensemble_meta, per_model_spds, specs_used

    return spd_obj, aggregated_usage, raw_calls, ensemble_meta, per_model_spds, specs_used


def _format_ensemble_meta(ensemble_meta: dict[str, object]) -> str:
    n_models_active = int(ensemble_meta.get("n_models_active") or 0)
    n_models_ok = int(ensemble_meta.get("n_models_ok") or 0)
    n_models_called = int(ensemble_meta.get("n_models_called") or n_models_active)
    failed = ensemble_meta.get("failed_providers") or []
    skipped = ensemble_meta.get("skipped_providers") or []
    parts = [f"ok={n_models_ok}/{n_models_active}"]
    if n_models_called != n_models_active:
        parts.append(f"called={n_models_called}")
    if skipped:
        parts.append(f"skipped={skipped}")
    parts.append(f"failed={failed}")
    return "ensemble_meta: " + " ".join(parts)


def _append_ensemble_meta(reason: str, ensemble_meta: str) -> str:
    reason = (reason or "").strip()
    ensemble_meta = (ensemble_meta or "").strip()
    if not ensemble_meta:
        return reason
    if not reason:
        return ensemble_meta
    return f"{reason} | {ensemble_meta}"


def _attach_ensemble_meta(spd_obj: dict[str, object], ensemble_meta: dict[str, object]) -> None:
    spd_obj["ensemble_meta"] = ensemble_meta

    suffix = _format_ensemble_meta(ensemble_meta)
    human_explanation = spd_obj.get("human_explanation")
    if isinstance(human_explanation, str) and human_explanation.strip():
        spd_obj["human_explanation"] = f"{human_explanation}\n{suffix}"
    else:
        spd_obj["human_explanation"] = suffix


def _write_spd_raw_text(run_id: str, question_id: str, tag: str, raw_text: str) -> Path:
    out_dir = Path("debug") / "spd_raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}__{question_id}_{tag}.txt"
    out_path.write_text(raw_text or "", encoding="utf-8")
    return out_path


def _write_spd_raw_debug(run_id: str, qid: str, suffix: str, text: str) -> Path:
    return _write_spd_raw_text(run_id, qid, suffix, text)


def _calls_summary(calls: list[dict[str, object]]) -> list[str]:
    out: list[str] = []
    for c in calls:
        ms = c.get("model_spec")
        if isinstance(ms, ModelSpec):
            out.append(f"{ms.provider}:{ms.model_id}")
    return out


def _specs_summary(specs: list[ModelSpec]) -> list[str]:
    out: list[str] = []
    for ms in specs:
        out.append(f"{ms.provider}:{ms.model_id}{'' if ms.active else '(inactive)'}")
    return out


def _provider_debug_snapshot() -> dict[str, object]:
    """Return provider active/credential status for diagnostics."""

    snapshot: dict[str, object] = {}
    try:
        for name in sorted(_PROVIDER_STATES.keys()):
            state = _PROVIDER_STATES.get(name, {})
            snapshot[name] = {
                "active": bool(state.get("active")),
                "has_api_key": bool(state.get("api_key")),
                "model": state.get("model", ""),
            }
    except Exception:  # noqa: BLE001
        snapshot = {"error": "provider_snapshot_failed"}
    return snapshot


def _spd_side_status(
    spd_obj: dict[str, object] | None, raw_calls: list[dict[str, object]], specs: list[ModelSpec]
) -> dict[str, object]:
    """
    Return status metadata for one SPD path.
    """

    active_specs = [ms for ms in specs if ms.active]
    if not specs and raw_calls:
        return {
            "status": "specs_empty",
            "reason": "spec list empty; inferred models from raw_calls",
            "n_calls": len(raw_calls),
            "n_active_specs": 0,
            "n_models_in_calls": sum(
                1 for c in raw_calls if isinstance(c.get("model_spec"), ModelSpec)
            ),
        }

    if not active_specs:
        return {
            "status": "no_active_models",
            "reason": "spec list has no active models",
            "n_calls": len(raw_calls),
            "n_active_specs": 0,
        }

    if not spd_obj:
        n_errors = sum(1 for c in raw_calls if c.get("error"))
        return {
            "status": "missing_spds",
            "reason": "no aggregated SPD object produced",
            "n_calls": len(raw_calls),
            "n_active_specs": len(active_specs),
            "n_errors": n_errors,
        }

    vecs = _spd_v2_to_month_vectors(spd_obj)
    if not vecs:
        return {
            "status": "missing_spds",
            "reason": "spd_obj had no usable month vectors",
            "n_calls": len(raw_calls),
            "n_active_specs": len(active_specs),
        }

    return {
        "status": "ok",
        "reason": "",
        "n_calls": len(raw_calls),
        "n_active_specs": len(active_specs),
        "n_months": len(vecs),
    }


def _has_v2_spds(spd_obj: dict[str, object] | None) -> bool:
    """Return True if the SPD v2 payload has any usable month vectors."""

    vecs = _spd_v2_to_month_vectors(spd_obj)
    return bool(vecs)


def _select_spd_specs_for_run() -> tuple[list[ModelSpec], str]:
    """
    Choose active SPD specs, preferring SPD_ENSEMBLE but falling back to DEFAULT_ENSEMBLE.
    Returns (active_specs, source_label).
    """

    specs = list(SPD_ENSEMBLE)
    active = [ms for ms in specs if getattr(ms, "active", False)]
    if active:
        return active, "SPD_ENSEMBLE"

    specs = list(DEFAULT_ENSEMBLE)
    active = [ms for ms in specs if getattr(ms, "active", False)]
    if active:
        return active, "DEFAULT_ENSEMBLE"

    return [], "none"


def _spd_v2_to_month_vectors(spd_obj: dict[str, object] | None) -> dict[str, list[float]]:
    """
    Convert SPD v2 object {"spds": {month: {"probs": [...]}}} into {month: [..]}.
    Defensive: ignores malformed entries.
    """
    if not isinstance(spd_obj, dict):
        return {}
    spds = spd_obj.get("spds")
    if not isinstance(spds, dict):
        return {}

    out: dict[str, list[float]] = {}
    for month, payload in spds.items():
        if not isinstance(payload, dict):
            continue
        probs = payload.get("probs")
        if not isinstance(probs, list) or not probs:
            continue
        try:
            vec = [float(x) for x in probs]
        except Exception:  # noqa: BLE001
            continue
        out[str(month)] = vec
    return out


def _compare_spd_vectors(
    a: dict[str, list[float]],
    b: dict[str, list[float]],
) -> dict[str, object]:
    """
    Compare two month->prob-vector mappings.

    Produces per-month:
      - max_abs_diff
      - l1_diff
      - length mismatch info
    and overall summary stats.
    """
    months_a = set(a.keys())
    months_b = set(b.keys())
    months_union = sorted(months_a | months_b)

    per_month: dict[str, object] = {}
    max_max_abs = 0.0
    max_l1 = 0.0

    for m in months_union:
        va = a.get(m)
        vb = b.get(m)
        if va is None or vb is None:
            per_month[m] = {
                "present_in": "a_only" if vb is None else "b_only",
            }
            continue

        la, lb = len(va), len(vb)
        n = min(la, lb)
        diffs = [abs(va[i] - vb[i]) for i in range(n)]
        max_abs = max(diffs) if diffs else 0.0
        l1 = sum(diffs)

        entry: dict[str, object] = {
            "max_abs_diff": max_abs,
            "l1_diff": l1,
            "len_a": la,
            "len_b": lb,
        }
        if la != lb:
            entry["length_mismatch"] = True
        per_month[m] = entry

        max_max_abs = max(max_max_abs, max_abs)
        max_l1 = max(max_l1, l1)

    return {
        "months_a": sorted(months_a),
        "months_b": sorted(months_b),
        "months_union": months_union,
        "per_month": per_month,
        "summary": {
            "n_months_a": len(months_a),
            "n_months_b": len(months_b),
            "max_max_abs_diff": max_max_abs,
            "max_l1_diff": max_l1,
        },
    }


def _write_spd_compare_artifact(run_id: str, qid: str, payload: dict[str, object]) -> None:
    base = os.getenv("PYTHIA_SPD_COMPARE_DIR", "debug/spd_compare")
    out_dir = Path(base)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}__{qid}.json"
    out_path.write_text(_json_dumps_for_db(payload), encoding="utf-8")



async def _run_research_for_question(run_id: str, question_row: duckdb.Row) -> None:
    """DEPRECATED: question-level web research replaced by structured data injection."""
    LOG.info("Skipping deprecated question-level web research (replaced by structured data injection)")
    qid = question_row["question_id"]
    iso3 = question_row["iso3"]
    hz = question_row["hazard_code"]
    metric = question_row["metric"]
    wording = question_row.get("wording") or question_row.get("title") or ""
    try:
        resolution_source = _infer_resolution_source(hz, metric)

        hs_run_id = question_row["hs_run_id"] or run_id
        hs_entry = load_hs_triage_entry(hs_run_id, iso3, hz)
        history_summary = _build_history_summary(iso3, hz, metric)
        resolver_features = history_summary

        hs_evidence_pack: dict[str, Any] | None = None
        question_evidence_pack: dict[str, Any] | None = None
        hazard_tail_pack: dict[str, Any] | None = None
        # DEPRECATED: web research via fetch_evidence_pack is replaced by structured
        # data injection. The retriever code below is kept for reference but bypassed.
        retriever_enabled = False  # was: _retriever_enabled()
        if retriever_enabled or os.getenv("PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED", "0") == "1":
            try:
                hs_evidence_pack = _load_hs_country_evidence_pack(hs_run_id, iso3)
            except Exception as exc:  # noqa: BLE001
                logging.warning("HS evidence pack load failed for %s: %s", iso3, exc)
                hs_evidence_pack = None

            try:
                question_queries = _build_question_evidence_queries(question_row, wording, hs_entry)
                retriever_model_id = _retriever_model_id() if retriever_enabled else None
                packs: list[dict[str, Any]] = []
                for query in question_queries:
                    pack = fetch_evidence_pack(
                        query,
                        purpose="research_question_pack",
                        run_id=run_id,
                        question_id=qid,
                        hs_run_id=question_row.get("hs_run_id"),
                        model_id=retriever_model_id if retriever_enabled else None,
                    )
                    packs.append(pack)
                if packs:
                    question_evidence_pack = _merge_question_evidence_packs(packs, question_queries)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Question evidence pack fetch failed for %s: %s", qid, exc)
                question_evidence_pack = None
            try:
                hazard_tail_pack = _load_hs_hazard_tail_pack(hs_run_id, iso3, hz)
            except Exception as exc:  # noqa: BLE001
                logging.warning("HS hazard tail pack load failed for %s/%s: %s", iso3, hz, exc)
                hazard_tail_pack = None

        merged_evidence_pack = merge_evidence_packs(hs_evidence_pack, question_evidence_pack)
        if hazard_tail_pack:
            if hs_evidence_pack is None:
                hs_evidence_pack = {}
            hs_evidence_pack["hazard_tail_pack"] = hazard_tail_pack
            merged_evidence_pack = merge_evidence_packs(
                merged_evidence_pack,
                {
                    **hazard_tail_pack,
                    "recent_signals": (hazard_tail_pack.get("recent_signals") or [])[:12],
                },
                max_sources=12,
                max_signals=12,
            )
            merged_evidence_pack["hs_hazard_tail_pack"] = hazard_tail_pack

        # Inject NMME seasonal outlook for climate hazards.
        if hz in CLIMATE_HAZARDS:
            try:
                _seasonal = load_seasonal_forecasts(iso3)
                if _seasonal:
                    if merged_evidence_pack is None:
                        merged_evidence_pack = {}
                    merged_evidence_pack["nmme_seasonal_outlook"] = _seasonal
            except Exception as _exc:
                logging.debug("Seasonal forecast load failed for %s: %s", iso3, _exc)

        # Prediction market signals (if enabled).
        _pm_bundle = None
        try:
            from pythia.prediction_markets.config import is_enabled as _pm_enabled

            if _pm_enabled():
                from pythia.prediction_markets.retriever import get_prediction_market_signals

                _country_name = _load_country_names().get(iso3, iso3)
                _hz_name = HZ_QUERY_MAP.get(hz, hz).replace("_", " ").lower()
                _pm_bundle = await get_prediction_market_signals(
                    question_text=wording,
                    country_name=_country_name,
                    iso3=iso3,
                    hazard_code=hz,
                    hazard_name=_hz_name,
                    forecast_start=str(question_row.get("window_start_date", "")),
                    forecast_end=str(question_row.get("window_end_date", "")),
                    run_id=run_id,
                )
        except Exception as _pm_exc:
            logging.warning("Prediction market retrieval failed for %s: %s", qid, _pm_exc)

        # Build minimal research dict (Researcher LLM removed — evidence now
        # flows via structured_data connectors and HS grounding).
        research: dict[str, Any] = {
            "note": "researcher_removed_v2",
            "sources": [],
            "grounded": False,
        }

        # Inject prediction market signals from dedicated retriever.
        if _pm_bundle is not None and _pm_bundle.questions:
            research["prediction_market_signals"] = _pm_bundle.to_research_dict()

        # Inject NMME seasonal outlook from merged evidence.
        if merged_evidence_pack and "nmme_seasonal_outlook" in merged_evidence_pack:
            research["nmme_seasonal_outlook"] = merged_evidence_pack["nmme_seasonal_outlook"]

        # Manifold snapshot fallback if dedicated PM retriever returned nothing.
        if not research.get("prediction_market_signals"):
            try:
                from pythia.market_snapshot import fetch_market_snapshot

                _snapshot = fetch_market_snapshot(wording)
                if _snapshot:
                    research["prediction_market_signals"] = {
                        "questions": [{
                            "platform": _snapshot["platform"],
                            "title": _snapshot["title"],
                            "url": _snapshot["url"],
                            "probability": _snapshot["prob"],
                        }],
                    }
            except Exception as _ms_exc:  # noqa: BLE001
                logging.debug("Manifold snapshot fallback failed for %s: %s", qid, _ms_exc)

        con = connect(read_only=False)
        try:
            ensure_schema(con)
            con.execute(
                """
                INSERT INTO question_research
                  (run_id, question_id, iso3, hazard_code, metric, research_json, hs_evidence_json, question_evidence_json, merged_evidence_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    qid,
                    iso3,
                    hz,
                    metric,
                    _json_dumps_for_db(research),
                    _json_dumps_for_db(hs_evidence_pack or {}),
                    _json_dumps_for_db(question_evidence_pack or {}),
                    _json_dumps_for_db(merged_evidence_pack or {}),
                ],
            )
        finally:
            con.close()
    except Exception as exc:  # noqa: BLE001
        logging.error("Research v2 hard failure for %s: %s", qid, exc)
        return


def _write_spd_outputs(
    run_id: str,
    question_row: Any,
    spd_obj: Dict[str, Any],
    *,
    resolution_source: str,
    usage: Dict[str, Any],
    model_name: str = "ensemble",
) -> None:
    rec = _normalize_question_row_for_spd(question_row)

    qid = rec["question_id"]
    iso3 = rec["iso3"]
    hz = rec["hazard_code"]
    metric = rec["metric"]
    bucket_labels = SPD_CLASS_BINS_FATALITIES if metric.upper() == "FATALITIES" else SPD_CLASS_BINS_PA

    spds = spd_obj.get("spds") if isinstance(spd_obj, dict) else None
    if not isinstance(spds, dict):
        return

    human_explanation = spd_obj.get("human_explanation") if isinstance(spd_obj, dict) else None
    human_explanation = human_explanation or ""

    spd_payload = dict(spd_obj)
    spd_payload.setdefault("resolution_source", resolution_source)

    con = connect(read_only=False)
    try:
        con.execute(
            "DELETE FROM forecasts_raw WHERE run_id = ? AND question_id = ? AND model_name = ?;",
            [run_id, qid, model_name],
        )
        con.execute(
            "DELETE FROM forecasts_ensemble WHERE run_id = ? AND question_id = ? AND model_name = ?;",
            [run_id, qid, model_name],
        )
        for month_idx, (month_label, payload) in enumerate(sorted(spds.items()), start=1):
            probs = payload.get("probs") if isinstance(payload, dict) else None
            if not probs:
                continue
            for bucket_index, prob in enumerate(list(probs)[: len(bucket_labels)], start=1):
                cb = bucket_labels[bucket_index - 1] if 0 <= bucket_index - 1 < len(bucket_labels) else str(bucket_index)
                con.execute(
                    """
                    INSERT INTO forecasts_raw (
                        run_id, question_id, model_name, month_index, bucket_index,
                        probability, ok, elapsed_ms, cost_usd, prompt_tokens, completion_tokens,
                        total_tokens, status, spd_json, human_explanation,
                        horizon_m, class_bin, p
                    ) VALUES (?, ?, ?, ?, ?, ?, TRUE, ?, ?, ?, ?, ?, 'ok', ?, ?, ?, ?, ?)
                    """,
                    [
                        run_id,
                        qid,
                        model_name,
                        month_idx,
                        bucket_index,
                        float(prob),
                        usage.get("elapsed_ms"),
                        usage.get("cost_usd"),
                        usage.get("prompt_tokens"),
                        usage.get("completion_tokens"),
                        usage.get("total_tokens"),
                        _json_dumps_for_db(spd_payload),
                        human_explanation,
                        month_idx,
                        cb,
                        float(prob),
                    ],
                )
                con.execute(
                    """
                    INSERT INTO forecasts_ensemble (
                        run_id, question_id, iso3, hazard_code, metric, model_name,
                        month_index, bucket_index, probability, ev_value, weights_profile, created_at,
                        status, human_explanation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'ensemble', CURRENT_TIMESTAMP, 'ok', ?)
                    """,
                    [
                        run_id,
                        qid,
                        iso3,
                        hz,
                        metric,
                        model_name,
                        month_idx,
                        bucket_index,
                        float(prob),
                        human_explanation,
                    ],
                )
    finally:
        con.close()


def _normalize_question_row_for_spd(question_row: Any) -> Dict[str, Any]:
    """
    Normalise a question row (dict, duckdb.Row, or tuple/list) into a plain dict.

    This keeps `_run_spd_for_question` robust when called from:
      - The main pipeline (where questions are dict-like records)
      - Unit tests that use `con.execute(...).fetchone()` (tuple rows)
    """

    if isinstance(question_row, Mapping):
        return dict(question_row)

    try:
        _ = question_row["question_id"]  # type: ignore[index]
        if hasattr(question_row, "keys"):
            keys = list(question_row.keys())  # type: ignore[attr-defined]
            return {k: question_row[k] for k in keys}  # type: ignore[index]
    except Exception:
        pass

    if isinstance(question_row, (tuple, list)):
        if len(question_row) < 12:
            raise TypeError(
                f"Unsupported questions row shape for SPD; expected >=12 columns, got {len(question_row)}"
            )
        return {
            "question_id": question_row[0],
            "hs_run_id": question_row[1],
            "scenario_ids_json": question_row[2],
            "iso3": question_row[3],
            "hazard_code": question_row[4],
            "metric": question_row[5],
            "target_month": question_row[6],
            "window_start_date": question_row[7],
            "window_end_date": question_row[8],
            "wording": question_row[9],
            "status": question_row[10],
            "pythia_metadata_json": question_row[11],
        }

    raise TypeError(
        f"Unsupported question_row type {type(question_row)!r}; expected Mapping, duckdb.Row, or tuple/list."
    )


def _first_target_month(target_months: Any) -> str | None:
    """Return the first target month string from a string or iterable, if present."""

    if isinstance(target_months, str) and target_months.strip():
        return target_months.strip()

    if isinstance(target_months, (list, tuple)):
        for month_val in target_months:
            if isinstance(month_val, str) and month_val.strip():
                return month_val.strip()

    return None


# ---------------------------------------------------------------------------
# Binary event forecast support (EVENT_OCCURRENCE questions)
# ---------------------------------------------------------------------------

def _write_binary_outputs(
    run_id: str,
    question_row: Any,
    month_probs: dict[str, float],
    *,
    resolution_source: str,
    usage: dict[str, Any],
    model_name: str = "ensemble",
) -> None:
    """Write binary forecasts using SPD storage convention.

    Convention: bucket_1 = P(yes), bucket_2 = P(no) = 1 - P(yes),
    buckets 3-5 = 0.  This avoids schema changes while keeping binary
    forecasts readable by the scoring pipeline.
    """
    rec = _normalize_question_row_for_spd(question_row)
    qid = rec["question_id"]
    iso3 = rec["iso3"]
    hz = rec["hazard_code"]
    metric = rec["metric"]

    # Binary bucket labels (informational only)
    bucket_labels = ["P(event)", "P(no_event)", "unused_3", "unused_4", "unused_5"]

    con = connect(read_only=False)
    try:
        con.execute(
            "DELETE FROM forecasts_raw WHERE run_id = ? AND question_id = ? AND model_name = ?;",
            [run_id, qid, model_name],
        )
        con.execute(
            "DELETE FROM forecasts_ensemble WHERE run_id = ? AND question_id = ? AND model_name = ?;",
            [run_id, qid, model_name],
        )
        for month_idx, month_label in enumerate(sorted(month_probs.keys()), start=1):
            p_yes = float(month_probs[month_label])
            # bucket_1 = P(yes), bucket_2 = P(no), buckets 3-5 = 0
            probs = [p_yes, 1.0 - p_yes, 0.0, 0.0, 0.0]
            for bucket_index, prob in enumerate(probs, start=1):
                cb = bucket_labels[bucket_index - 1]
                con.execute(
                    """
                    INSERT INTO forecasts_raw (
                        run_id, question_id, model_name, month_index, bucket_index,
                        probability, ok, elapsed_ms, cost_usd, prompt_tokens, completion_tokens,
                        total_tokens, status, spd_json, human_explanation,
                        horizon_m, class_bin, p
                    ) VALUES (?, ?, ?, ?, ?, ?, TRUE, ?, ?, ?, ?, ?, 'ok', ?, ?, ?, ?, ?)
                    """,
                    [
                        run_id, qid, model_name, month_idx, bucket_index,
                        float(prob),
                        usage.get("elapsed_ms"),
                        usage.get("cost_usd"),
                        usage.get("prompt_tokens"),
                        usage.get("completion_tokens"),
                        usage.get("total_tokens"),
                        _json_dumps_for_db({"binary": True, "p_yes": p_yes, "resolution_source": resolution_source}),
                        "",
                        month_idx,
                        cb,
                        float(prob),
                    ],
                )
                con.execute(
                    """
                    INSERT INTO forecasts_ensemble (
                        run_id, question_id, iso3, hazard_code, metric, model_name,
                        month_index, bucket_index, probability, ev_value, weights_profile, created_at,
                        status, human_explanation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'ensemble', CURRENT_TIMESTAMP, 'ok', ?)
                    """,
                    [
                        run_id, qid, iso3, hz, metric, model_name,
                        month_idx, bucket_index, float(prob), "",
                    ],
                )
    finally:
        con.close()


async def _run_binary_forecast_for_question(
    run_id: str,
    question_row: Any,
    *,
    track: int = 1,
) -> None:
    """Run binary event forecast for an EVENT_OCCURRENCE question.

    Uses the binary prompt builder, calls the same LLM models, parses
    the response as per-month probabilities, aggregates via weighted
    average, and stores using the bucket convention.
    """
    rec = _normalize_question_row_for_spd(question_row)
    qid = rec.get("question_id")
    iso3 = (rec.get("iso3") or "").upper()
    hz = (rec.get("hazard_code") or "").upper()
    metric = rec.get("metric") or "EVENT_OCCURRENCE"
    wording = rec.get("wording") or ""

    try:
        resolution_source = _infer_resolution_source(hz, metric)
        hs_run_id = rec.get("hs_run_id") or run_id
        hs_entry = load_hs_triage_entry(hs_run_id, iso3, hz)
        structured_data = _load_structured_data(iso3, hz, hs_run_id=hs_run_id)

        # Build binary base rate from facts_resolved
        base_rate = build_binary_base_rate(iso3, hz)

        # Load current GDACS alerts
        current_alerts: list[dict] = []
        try:
            alert_con = connect(read_only=True)
            alert_rows = alert_con.execute(
                """
                SELECT ym, value, alertlevel
                FROM facts_resolved
                WHERE upper(iso3) = ? AND upper(hazard_code) = ?
                  AND lower(metric) = 'event_occurrence'
                ORDER BY ym DESC LIMIT 6
                """,
                [iso3, hz],
            ).fetchall()
            alert_con.close()
            for arow in alert_rows:
                current_alerts.append({
                    "ym": str(arow[0]),
                    "value": float(arow[1] or 0),
                    "alertlevel": str(arow[2] or ""),
                })
        except Exception:
            pass

        target_months = rec.get("target_months") or rec.get("target_month")
        window_start = rec.get("window_start_date")

        prompt = build_binary_event_prompt(
            question={
                "question_id": qid,
                "iso3": iso3,
                "hazard_code": hz,
                "metric": metric,
                "country_name": iso3,
                "window_start_date": window_start,
                "wording": wording,
            },
            base_rate=base_rate,
            current_alerts=current_alerts,
            structured_data=structured_data,
            hs_triage_entry=hs_entry,
            today=date.today().isoformat() if date else str(datetime.now().date()),
        )

        # Select model specs based on track
        if track == 2:
            specs = [TRACK2_MODEL_SPEC]
            model_name = "track2_flash"
        else:
            specs_active, _ = _select_spd_specs_for_run()
            if not specs_active:
                _record_no_forecast(
                    run_id, qid or "", iso3, hz, metric,
                    "binary: no active model specs",
                    model_name="ensemble_mean_v2",
                )
                return
            specs = specs_active
            model_name = "ensemble_mean_v2"

        target_month = _first_target_month(target_months)

        # Call models — reuse the same model calling infrastructure
        per_model_spds, usage, raw_calls, ensemble_meta = (
            await _call_spd_members_v2_compat(
                prompt,
                specs,
                run_id=run_id,
                question_id=qid,
                iso3=iso3,
                hazard_code=hz,
                metric=metric,
                target_month=target_month,
                wording=wording,
            )
        )

        # Log LLM calls
        for call in raw_calls:
            ms = call.get("model_spec")
            if not isinstance(ms, ModelSpec):
                continue
            await log_forecaster_llm_call(
                run_id=run_id,
                question_id=qid,
                iso3=iso3,
                hazard_code=hz,
                metric=metric,
                model_spec=ms,
                prompt_text=prompt,
                response_text=str(call.get("text") or ""),
                usage=call.get("usage") or {},
                error_text=str(call.get("error")) if call.get("error") else None,
                phase="binary_v2",
                call_type="binary_v2",
                hs_run_id=hs_run_id,
            )

        # Parse binary responses from raw model outputs
        all_model_probs: list[dict[str, float]] = []
        for call in raw_calls:
            raw_text = str(call.get("text") or "")
            if not raw_text:
                continue
            parsed = parse_binary_response(raw_text)
            if parsed:
                all_model_probs.append(parsed)

        if not all_model_probs:
            _record_no_forecast(
                run_id, qid or "", iso3, hz, metric,
                "binary: no valid responses parsed",
                model_name=model_name,
            )
            return

        # Aggregate: simple weighted average across models per month
        all_months: set[str] = set()
        for mp in all_model_probs:
            all_months.update(mp.keys())

        aggregated: dict[str, float] = {}
        for month in sorted(all_months):
            probs = [mp[month] for mp in all_model_probs if month in mp]
            if probs:
                aggregated[month] = sum(probs) / len(probs)

        if not aggregated:
            _record_no_forecast(
                run_id, qid or "", iso3, hz, metric,
                "binary: no months in aggregated result",
                model_name=model_name,
            )
            return

        # Write binary outputs
        _write_binary_outputs(
            run_id,
            question_row,
            aggregated,
            resolution_source=resolution_source,
            usage=usage or {},
            model_name=model_name,
        )

        # Also write BayesMC aggregation for Track 1
        if track == 1 and len(all_model_probs) > 1:
            bayesmc_aggregated: dict[str, float] = {}
            for month in sorted(all_months):
                probs = [mp[month] for mp in all_model_probs if month in mp]
                if not probs:
                    continue
                # Use aggregate_binary from aggregate.py
                from .ensemble import EnsembleResult, MemberOutput
                members = [
                    MemberOutput(name=f"model_{i}", ok=True, parsed=p, raw="")
                    for i, p in enumerate(probs)
                ]
                ens_res = EnsembleResult(members=members)
                agg_p, _ = aggregate_binary(ens_res)
                bayesmc_aggregated[month] = agg_p

            if bayesmc_aggregated:
                _write_binary_outputs(
                    run_id,
                    question_row,
                    bayesmc_aggregated,
                    resolution_source=resolution_source,
                    usage=usage or {},
                    model_name="ensemble_bayesmc_v2",
                )

    except Exception:
        LOG.exception("Binary forecast failed for %s", qid)
        _record_no_forecast(
            run_id, qid or "", iso3, hz, metric,
            "binary: exception during forecast",
            model_name="ensemble_mean_v2" if track == 1 else "track2_flash",
        )


TRACK2_MODEL_SPEC = ModelSpec(
    name="track2_flash",
    provider="google",
    model_id="gemini-3-flash-preview",
    weight=1.0,
    active=True,
    purpose="track2_spd",
)


async def _run_track2_spd_for_question(run_id: str, question_row: Any) -> None:
    """Run single-model SPD forecast for Track 2 questions (Gemini Flash only)."""
    rec = _normalize_question_row_for_spd(question_row)
    qid = rec.get("question_id")
    iso3 = (rec.get("iso3") or "").upper()
    hz = (rec.get("hazard_code") or "").upper()
    metric = rec.get("metric") or "PA"
    wording = rec.get("wording") or rec.get("title") or ""

    # Binary questions use a separate pipeline
    if metric.upper() == "EVENT_OCCURRENCE":
        await _run_binary_forecast_for_question(run_id, question_row, track=2)
        return

    try:
        resolution_source = _infer_resolution_source(hz, metric)
        hs_run_id = rec.get("hs_run_id") or run_id
        hs_entry = load_hs_triage_entry(hs_run_id, iso3, hz)
        history_summary = _build_history_summary(iso3, hz, metric)
        research_json = _load_research_json(run_id, qid)
        if not isinstance(research_json, dict):
            research_json = {
                "note": "missing_research_v2",
                "base_rate_hint": "Use Resolver history + HS triage + model prior.",
                "sources": [],
                "grounded": False,
            }
        else:
            research_json = dict(research_json)

        target_months = rec.get("target_months") or rec.get("target_month")
        target_month = _first_target_month(target_months)

        # Inject NMME seasonal outlook for climate hazards.
        if hz in CLIMATE_HAZARDS and "nmme_seasonal_outlook" not in research_json:
            try:
                _seasonal = load_seasonal_forecasts(iso3)
                if _seasonal:
                    research_json["nmme_seasonal_outlook"] = _seasonal
            except Exception as _exc:
                logging.debug("Seasonal forecast load failed for %s: %s", iso3, _exc)

        question_evidence_pack = _load_question_evidence_pack(run_id, qid) if qid else None
        structured_data = _load_structured_data(iso3, hz, hs_run_id=hs_run_id)
        prompt = build_spd_prompt_v2(
            question={
                "question_id": qid,
                "iso3": iso3,
                "hazard_code": hz,
                "metric": metric,
                "resolution_source": resolution_source,
                "wording": wording,
                "target_months": target_months,
            },
            history_summary=history_summary,
            hs_triage_entry=hs_entry,
            research_json=research_json,
            structured_data=structured_data,
            model_name=TRACK2_MODEL_SPEC.name,
        )
        if question_evidence_pack:
            prompt = append_retriever_evidence_to_prompt(prompt, question_evidence_pack)

        # Single model call with Gemini Flash
        per_model_spds, usage, raw_calls, ensemble_meta = (
            await _call_spd_members_v2_compat(
                prompt,
                [TRACK2_MODEL_SPEC],
                run_id=run_id,
                question_id=qid,
                iso3=iso3,
                hazard_code=hz,
                metric=metric,
                target_month=target_month,
                wording=wording,
            )
        )

        # Log LLM calls
        for call in raw_calls:
            ms = call.get("model_spec")
            if not isinstance(ms, ModelSpec):
                continue
            await log_forecaster_llm_call(
                run_id=run_id,
                question_id=qid,
                iso3=iso3,
                hazard_code=hz,
                metric=metric,
                model_spec=ms,
                prompt_text=prompt,
                response_text=str(call.get("text") or ""),
                usage=call.get("usage") or {},
                error_text=str(call.get("error")) if call.get("error") else None,
                phase="spd_v2",
                call_type="spd_v2",
                hs_run_id=hs_run_id,
            )

        if not per_model_spds:
            _record_no_forecast(
                run_id, qid, iso3, hz, metric,
                "track2: no SPD from single model",
                model_name="track2_flash",
            )
            return

        # Use mean aggregation (effectively identity for single model)
        spd_mean = aggregate_spd_v2_mean(per_model_spds)
        spd_obj = {"spds": {m: {"probs": vec} for m, vec in spd_mean.items()}}
        _attach_ensemble_meta(spd_obj, ensemble_meta)

        if _has_v2_spds(spd_obj):
            _write_spd_outputs(
                run_id,
                question_row,
                spd_obj,
                resolution_source=resolution_source,
                usage=usage or {},
                model_name="track2_flash",
            )
        else:
            _record_no_forecast(
                run_id, qid, iso3, hz, metric,
                "track2: missing spds from single model",
                model_name="track2_flash",
            )

        # Also write per-model raw data
        _write_spd_members_v2_to_db(
            run_id=run_id,
            question_row=rec,
            specs_used=[TRACK2_MODEL_SPEC],
            per_model_spds=per_model_spds,
            raw_calls=raw_calls,
            resolution_source=resolution_source,
        )

    except Exception:
        LOG.exception("Track 2 SPD failed for %s", qid)
        _record_no_forecast(
            run_id, qid or "", iso3, hz, metric,
            "track2: exception during SPD",
            model_name="track2_flash",
        )


async def _run_spd_for_question(run_id: str, question_row: Any) -> None:
    _maybe_log_default_ensemble()

    rec = _normalize_question_row_for_spd(question_row)
    member_spds_snapshot: list[dict[str, list[float]]] | None = None
    member_specs_snapshot: list[ModelSpec] | None = None
    member_raw_calls_snapshot: list[dict[str, object]] | None = None
    members_written = False

    qid = rec.get("question_id")
    iso3 = (rec.get("iso3") or "").upper()
    hz = (rec.get("hazard_code") or "").upper()
    metric = rec.get("metric") or "PA"
    wording = rec.get("wording") or rec.get("title") or ""

    # Binary questions use a separate pipeline
    if metric.upper() == "EVENT_OCCURRENCE":
        await _run_binary_forecast_for_question(run_id, question_row, track=1)
        return

    try:
        resolution_source = _infer_resolution_source(hz, metric)

        hs_run_id = rec.get("hs_run_id") or run_id
        hs_entry = load_hs_triage_entry(hs_run_id, iso3, hz)
        history_summary = _build_history_summary(iso3, hz, metric)
        research_json = _load_research_json(run_id, qid)
        if not isinstance(research_json, dict):
            research_json = {
                "note": "missing_research_v2",
                "base_rate_hint": "Use Resolver history + HS triage + model prior.",
                "sources": [],
                "grounded": False,
            }
        else:
            research_json = dict(research_json)

        def _coerce_float(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _coerce_int(value: Any) -> int | None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        rc_level = _coerce_int(hs_entry.get("regime_change_level"))
        rc_score = _coerce_float(hs_entry.get("regime_change_score"))
        if rc_level is None:
            rc_payload = hs_entry.get("regime_change")
            if isinstance(rc_payload, dict):
                rc_score = rc_score if rc_score is not None else _coerce_float(rc_payload.get("score"))
                rc_level = _coerce_int(rc_payload.get("level"))

        rc_elevated = False
        if rc_level is not None:
            rc_elevated = rc_level >= 2
        elif rc_score is not None:
            rc_elevated = rc_score >= 0.30

        if rc_elevated:
            max_signals = int(os.getenv("PYTHIA_SPD_TAIL_PACKS_MAX_SIGNALS", "12"))
            try:
                tail_pack = _load_hs_hazard_tail_pack(
                    hs_run_id,
                    iso3,
                    hz,
                    max_signals=max_signals,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.debug("SPD tail pack load failed for %s/%s: %s", iso3, hz, exc)
                tail_pack = None
            if tail_pack:
                research_json["hs_hazard_tail_pack"] = tail_pack
                merged_sources = _dedupe_sources_by_url(
                    (research_json.get("sources") or []) + (tail_pack.get("sources") or [])
                )
                research_json["sources"] = merged_sources
                LOG.debug(
                    "SPD tail pack injected for %s/%s: bullets=%d sources=%d",
                    iso3,
                    hz,
                    len(tail_pack.get("recent_signals") or []),
                    len(merged_sources),
                )

        # Inject NMME seasonal outlook for climate hazards.
        if hz in CLIMATE_HAZARDS and "nmme_seasonal_outlook" not in research_json:
            try:
                _seasonal = load_seasonal_forecasts(iso3)
                if _seasonal:
                    research_json["nmme_seasonal_outlook"] = _seasonal
            except Exception as _exc:
                logging.debug("Seasonal forecast load failed for %s: %s", iso3, _exc)

        target_months = rec.get("target_months") or rec.get("target_month")
        target_month = _first_target_month(target_months)

        question_evidence_pack = _load_question_evidence_pack(run_id, qid) if qid else None
        # DEPRECATED: question-level web research via fetch_evidence_pack is replaced
        # by structured data injection (conflict forecasts, ReliefWeb, HDX Signals,
        # HS grounding evidence). The retriever code below is kept for reference but
        # bypassed — structured data now flows via _load_structured_data().
        if False and question_evidence_pack is None and _retriever_enabled():  # noqa: SIM223
            try:
                question_queries = _build_question_evidence_queries(rec, wording, hs_entry)
                retriever_model_id = _retriever_model_id()
                packs: list[dict[str, Any]] = []
                for query in question_queries:
                    pack = fetch_evidence_pack(
                        query,
                        purpose="research_question_pack",
                        run_id=run_id,
                        question_id=qid,
                        hs_run_id=hs_run_id,
                        model_id=retriever_model_id if _retriever_enabled() else None,
                    )
                    packs.append(pack)
                if packs:
                    question_evidence_pack = _merge_question_evidence_packs(packs, question_queries)
                    if qid:
                        _persist_question_evidence_pack(
                            run_id,
                            qid,
                            iso3,
                            hz,
                            metric,
                            question_evidence_pack,
                        )
            except Exception as exc:  # noqa: BLE001
                logging.warning("SPD retriever evidence pack fetch failed for %s: %s", qid, exc)
                question_evidence_pack = None

        structured_data = _load_structured_data(
            iso3, hz, hs_run_id=hs_run_id, rc_level=rc_level,
        )
        prompt = build_spd_prompt_v2(
            question={
                "question_id": qid,
                "iso3": iso3,
                "hazard_code": hz,
                "metric": metric,
                "resolution_source": resolution_source,
                "wording": wording,
                "target_months": target_months,
            },
            history_summary=history_summary,
            hs_triage_entry=hs_entry,
            research_json=research_json,
            structured_data=structured_data,
        )
        if question_evidence_pack:
            prompt = append_retriever_evidence_to_prompt(prompt, question_evidence_pack)

        dual_run = os.getenv("PYTHIA_SPD_V2_DUAL_RUN", "0") == "1"
        use_bayesmc = os.getenv("PYTHIA_SPD_V2_USE_BAYESMC", "0") == "1"
        write_both = os.getenv("PYTHIA_SPD_V2_WRITE_BOTH", "0") == "1"

        if dual_run:
            specs_active, specs_source = _select_spd_specs_for_run()
            specs_used_summary = _specs_summary(specs_active)
            n_active_specs = len(specs_active)
            specs_source_len = (
                len(SPD_ENSEMBLE)
                if specs_source == "SPD_ENSEMBLE"
                else len(DEFAULT_ENSEMBLE)
                if specs_source == "DEFAULT_ENSEMBLE"
                else len(SPD_ENSEMBLE) + len(DEFAULT_ENSEMBLE)
            )
            if not specs_active:
                reason = (
                    f"{specs_source} has no active model specs"
                    if specs_source != "none"
                    else "SPD_ENSEMBLE and DEFAULT_ENSEMBLE have no active model specs"
                )
                diff = {
                    "error": "no_active_models",
                    "reason": reason,
                    "n_specs": specs_source_len,
                    "n_active_specs": n_active_specs,
                    "specs_source": specs_source,
                    "providers": _provider_debug_snapshot(),
                }
                payload = {
                    "run_id": run_id,
                    "question_id": qid,
                    "iso3": iso3,
                    "hazard_code": hz,
                    "metric": metric,
                    "hs_run_id": hs_run_id,
                    "write_path": "bayesmc" if use_bayesmc else "v2_ensemble",
                    "specs_used": specs_used_summary,
                    "v2_ensemble": {
                        "models": [],
                        "usage": {},
                        "status_info": {
                            "status": "no_active_models",
                            "reason": reason,
                            "n_calls": 0,
                            "n_active_specs": n_active_specs,
                        },
                    },
                    "bayesmc": {
                        "models": [],
                        "usage": {},
                        "status_info": {
                            "status": "no_active_models",
                            "reason": reason,
                            "n_calls": 0,
                            "n_active_specs": n_active_specs,
                        },
                    },
                    "specs_source": specs_source,
                    "diff": diff,
                }
                _write_spd_compare_artifact(run_id, qid, payload)
                return
            try:
                per_model_spds, aggregated_usage, raw_calls, ensemble_meta = (
                    await _call_spd_members_v2_compat(
                        prompt,
                        specs_active,
                        run_id=run_id,
                        question_id=qid,
                        iso3=iso3,
                        hazard_code=hz,
                        metric=metric,
                        target_month=target_month,
                        wording=wording,
                    )
                )
                if member_spds_snapshot is None:
                    member_spds_snapshot = per_model_spds
                    member_specs_snapshot = specs_active
                    member_raw_calls_snapshot = raw_calls
                spd_mean = aggregate_spd_v2_mean(per_model_spds)
                spd_v2 = {"spds": {m: {"probs": vec} for m, vec in spd_mean.items()}}
                _attach_ensemble_meta(spd_v2, ensemble_meta)

                spd_bm, diag_bm = _build_bayesmc_spd_obj(
                    per_model_spds, target_month=target_month, specs_used=specs_active
                )
                if spd_bm:
                    spd_bm.setdefault("bayesmc_diag", diag_bm)
                    _attach_ensemble_meta(spd_bm, ensemble_meta)

                vec_v2 = _spd_v2_to_month_vectors(spd_v2)
                vec_bm = _spd_v2_to_month_vectors(spd_bm)

                diff = _compare_spd_vectors(vec_v2, vec_bm)

                vectors: dict[str, dict[str, object]] = {}
                for m in diff["months_union"]:
                    va = vec_v2.get(m)
                    vb = vec_bm.get(m)
                    entry: dict[str, object] = {}

                    if va is not None:
                        entry["v2_probs"] = va
                        if len(va) > 0:
                            top_idx = int(max(range(len(va)), key=lambda i: va[i]))
                            entry["v2_top"] = {"idx": top_idx + 1, "p": va[top_idx]}
                        entry["v2_sum"] = sum(va)

                    if vb is not None:
                        entry["bayesmc_probs"] = vb
                        if len(vb) > 0:
                            top_idx = int(max(range(len(vb)), key=lambda i: vb[i]))
                            entry["bayesmc_top"] = {"idx": top_idx + 1, "p": vb[top_idx]}
                        entry["bayesmc_sum"] = sum(vb)

                    vectors[m] = entry

                payload: dict[str, object] = {
                    "run_id": run_id,
                    "question_id": qid,
                    "iso3": iso3,
                    "hazard_code": hz,
                    "metric": metric,
                    "hs_run_id": hs_run_id,
                    "write_path": "bayesmc" if use_bayesmc else "v2_ensemble",
                    "specs_used": specs_used_summary,
                    "v2_ensemble": {
                        "models": _calls_summary(raw_calls),
                        "usage": aggregated_usage,
                        "status_info": _spd_side_status(spd_v2, raw_calls, specs_active),
                    },
                    "bayesmc": {
                        "models": _calls_summary(raw_calls),
                        "usage": aggregated_usage,
                        "status_info": _spd_side_status(spd_bm, raw_calls, specs_active),
                    },
                    "diff": diff,
                    "vectors": vectors,
                }
                _write_spd_compare_artifact(run_id, qid, payload)
            except Exception:  # noqa: BLE001
                LOG.exception("[debug] SPD dual-run compare failed for %s", qid)

        # CI contract (forecaster/tests/test_spd.py):
        # - On successful SPD v2 run: must write >=1 row to forecasts_ensemble for (run_id, question_id).
        # - Must also log >=1 row to llm_calls with call_type='spd_v2' for (run_id, question_id).
        # - If response JSON is missing 'spds': must record no_forecast w/ reason containing 'missing spds'
        #   and write debug/spd_raw/{run_id}__{question_id}_missing_spds.txt.

        spd_obj: Dict[str, Any] | None = None
        usage: Dict[str, Any] = {}
        text = ""
        raw_calls: list[dict[str, object]] = []

        if write_both:
            specs_active, specs_source = _select_spd_specs_for_run()

            per_model_spds, usage, raw_calls, ensemble_meta = await _call_spd_members_v2_compat(
                prompt,
                specs_active,
                run_id=run_id,
                question_id=qid,
                iso3=iso3,
                hazard_code=hz,
                metric=metric,
                target_month=target_month,
                wording=wording,
            )
            if member_spds_snapshot is None:
                member_spds_snapshot = per_model_spds
                member_specs_snapshot = specs_active
                member_raw_calls_snapshot = raw_calls

            if (not raw_calls) and (not per_model_spds):
                reason = (
                    f"no active ensemble models ({specs_source})"
                    if specs_source != "none"
                    else "no active ensemble models (SPD_ENSEMBLE and DEFAULT_ENSEMBLE inactive)"
                )
                _record_no_forecast(
                    run_id,
                    qid,
                    iso3,
                    hz,
                    metric,
                    reason,
                    model_name="ensemble_mean_v2",
                )
                _record_no_forecast(
                    run_id,
                    qid,
                    iso3,
                    hz,
                    metric,
                    reason,
                    model_name="ensemble_bayesmc_v2",
                )
                return

            specs_used_for_bayesmc = specs_active
            if not specs_used_for_bayesmc and raw_calls:
                inferred: list[ModelSpec] = []
                for rc in raw_calls:
                    ms = rc.get("model_spec") if isinstance(rc, dict) else None
                    if isinstance(ms, ModelSpec):
                        inferred.append(ms)
                if inferred:
                    specs_used_for_bayesmc = inferred
                    specs_source = "inferred_from_raw_calls"

            if not members_written:
                _write_spd_members_v2_to_db(
                    run_id=run_id,
                    question_row=rec,
                    specs_used=member_specs_snapshot or specs_active,
                    per_model_spds=member_spds_snapshot or per_model_spds,
                    raw_calls=member_raw_calls_snapshot or raw_calls,
                    resolution_source=resolution_source,
                )
                members_written = True

            raw_texts = [str(rc.get("text") or "") for rc in raw_calls if isinstance(rc, dict)]
            debug_written = False

            logged_any_spd_call = False
            for call in raw_calls:
                ms = call.get("model_spec")
                if not isinstance(ms, ModelSpec):
                    continue
                await log_forecaster_llm_call(
                    run_id=run_id,
                    question_id=qid,
                    iso3=iso3,
                    hazard_code=hz,
                    metric=metric,
                    model_spec=ms,
                    prompt_text=prompt,
                    response_text=str(call.get("text") or ""),
                    usage=call.get("usage") or {},
                    error_text=str(call.get("error")) if call.get("error") else None,
                    phase="spd_v2",
                    call_type="spd_v2",
                    hs_run_id=hs_run_id,
                )
                logged_any_spd_call = True

            if not logged_any_spd_call:
                fallback = raw_calls[0] if raw_calls else {}
                await log_forecaster_llm_call(
                    run_id=run_id,
                    question_id=qid,
                    iso3=iso3,
                    hazard_code=hz,
                    metric=metric,
                    model_spec=fallback.get("model_spec")
                    if isinstance(fallback.get("model_spec"), ModelSpec)
                    else None,
                    prompt_text=prompt,
                    response_text=str(fallback.get("text") or ""),
                    usage=fallback.get("usage") or {},
                    error_text=(
                        str(fallback.get("error"))
                        if fallback.get("error")
                        else "bayesmc: no ensemble members"
                    ),
                    phase="spd_v2",
                    call_type="spd_v2",
                    hs_run_id=hs_run_id,
                )

            ensemble_meta_str = _format_ensemble_meta(ensemble_meta)
            if specs_source:
                ensemble_meta_str = f"{ensemble_meta_str} | specs_source={specs_source}"

            spd_mean = aggregate_spd_v2_mean(per_model_spds)
            spd_mean_obj = {"spds": {m: {"probs": vec} for m, vec in spd_mean.items()}}
            if _has_v2_spds(spd_mean_obj):
                _attach_ensemble_meta(spd_mean_obj, ensemble_meta)

            spd_bm_obj, diag_bm = _build_bayesmc_spd_obj(
                per_model_spds, target_month=target_month, specs_used=specs_used_for_bayesmc
            )
            if _has_v2_spds(spd_bm_obj):
                spd_bm_obj.setdefault("bayesmc_diag", diag_bm)
                _attach_ensemble_meta(spd_bm_obj, ensemble_meta)

            missing_months: list[str] = []
            if isinstance(diag_bm, dict):
                missing_months = diag_bm.get("missing_months") or []

            mean_has_spds = _has_v2_spds(spd_mean_obj)
            bayesmc_has_spds = _has_v2_spds(spd_bm_obj)

            if not bayesmc_has_spds and mean_has_spds:
                fallback_diag: dict[str, object] = {
                    "status": "fallback_to_mean",
                    "original_bayesmc_status": diag_bm.get("status") if isinstance(diag_bm, dict) else None,
                }
                if missing_months:
                    fallback_diag["missing_months"] = missing_months
                explanation_parts = [
                    "fallback_to_mean: BayesMC produced no SPD; wrote mean SPD as fallback.",
                    f"bayesmc_status={fallback_diag.get('original_bayesmc_status')}",
                ]
                if missing_months:
                    explanation_parts.append(f"missing_months={missing_months}")
                spd_bm_obj = {
                    "spds": spd_mean_obj.get("spds"),
                    "bayesmc_diag": fallback_diag,
                    "human_explanation": " ".join(explanation_parts),
                }
                _attach_ensemble_meta(spd_bm_obj, ensemble_meta)
                bayesmc_has_spds = _has_v2_spds(spd_bm_obj)

            debug_tag = "insufficient_month_coverage" if missing_months else "missing_spds"
            debug_payload = raw_texts[0] if raw_texts else ""
            if not debug_payload:
                debug_payload = json.dumps(
                    {
                        "missing_months": missing_months,
                        "bayesmc_diag": diag_bm,
                        "ensemble_meta": ensemble_meta,
                    }
                )

            insufficient_coverage = int(ensemble_meta.get("n_models_ok") or 0) < 2
            coverage_reason = "insufficient ensemble coverage" if insufficient_coverage else "missing spds"

            if mean_has_spds:
                _write_spd_outputs(
                    run_id,
                    question_row,
                    spd_mean_obj,
                    resolution_source=resolution_source,
                    usage=usage or {},
                    model_name="ensemble_mean_v2",
                )
            else:
                reason_mean = _append_ensemble_meta(coverage_reason, ensemble_meta_str)
                _write_spd_raw_text(run_id, qid, debug_tag, debug_payload)
                debug_written = True
                _record_no_forecast(
                    run_id,
                    qid,
                    iso3,
                    hz,
                    metric,
                    reason_mean,
                    model_name="ensemble_mean_v2",
                    raw_debug_written=debug_written,
                )

            if bayesmc_has_spds:
                _write_spd_outputs(
                    run_id,
                    question_row,
                    spd_bm_obj,
                    resolution_source=resolution_source,
                    usage=usage or {},
                    model_name="ensemble_bayesmc_v2",
                )
            else:
                reason_bm = coverage_reason
                tag = debug_tag
                if missing_months:
                    reason_bm = f"BayesMC produced insufficient month coverage; missing_months={missing_months}"
                    tag = "insufficient_month_coverage"
                reason_bm = _append_ensemble_meta(reason_bm, ensemble_meta_str)
                _write_spd_raw_text(run_id, qid, tag, debug_payload)
                debug_written = True
                _record_no_forecast(
                    run_id,
                    qid,
                    iso3,
                    hz,
                    metric,
                    reason_bm,
                    model_name="ensemble_bayesmc_v2",
                    raw_debug_written=debug_written,
                )

            return

        if use_bayesmc:
            specs_active, specs_source = _select_spd_specs_for_run()
            spd_obj, usage, raw_calls, ensemble_meta, per_model_spds_bm, specs_used_bm = await _call_spd_bayesmc_v2(
                prompt,
                run_id=run_id,
                question_id=qid,
                hs_run_id=hs_run_id,
                target_month=target_month,
                specs=specs_active,
                iso3=iso3,
                hazard_code=hz,
                metric=metric,
                wording=wording,
            )
            if member_spds_snapshot is None:
                member_spds_snapshot = per_model_spds_bm
                member_specs_snapshot = specs_used_bm
                member_raw_calls_snapshot = raw_calls

            if not members_written:
                _write_spd_members_v2_to_db(
                    run_id=run_id,
                    question_row=rec,
                    specs_used=member_specs_snapshot or specs_used_bm,
                    per_model_spds=member_spds_snapshot or per_model_spds_bm,
                    raw_calls=member_raw_calls_snapshot or raw_calls,
                    resolution_source=resolution_source,
                )
                members_written = True
            raw_texts = [str(rc.get("text") or "") for rc in raw_calls if isinstance(rc, dict)]
            text = json.dumps(spd_obj)

            ensemble_meta_str = _format_ensemble_meta(ensemble_meta)
            if specs_source:
                ensemble_meta_str = f"{ensemble_meta_str} | specs_source={specs_source}"
            bayesmc_diag = {}
            try:
                diag_from_meta = ensemble_meta.get("bayesmc_diag")
                if isinstance(diag_from_meta, dict):
                    bayesmc_diag = diag_from_meta
            except Exception:
                bayesmc_diag = {}

            # If BayesMC yields no SPD months, treat as missing spds (same contract as v2 path).
            if int((ensemble_meta or {}).get("n_models_ok") or 0) < 2:
                if raw_texts and raw_texts[0].strip():
                    _write_spd_raw_text(run_id, qid, "missing_spds", raw_texts[0])
                    reason = "missing spds"
                else:
                    diag = {
                        "status": "no_active_models_or_no_calls",
                        "reason": "BayesMC had <2 ok models and no raw model text captured",
                        "n_raw_calls": len(raw_calls or []),
                        "ensemble_meta": ensemble_meta or {},
                    }
                    _write_spd_raw_text(run_id, qid, "no_active_models", json.dumps(diag))
                    reason = "no active ensemble models"

                reason = _append_ensemble_meta(reason, ensemble_meta_str)
                _record_no_forecast(
                    run_id,
                    qid,
                    iso3,
                    hz,
                    metric,
                    reason,
                    model_name="ensemble_bayesmc_v2",
                    raw_debug_written=True,
                )
                return

            if not spd_obj:
                first_text = ""
                if raw_texts:
                    first_text = raw_texts[0]
                elif text:
                    first_text = str(text)
                missing_months = []
                if isinstance(bayesmc_diag, dict):
                    missing_months = bayesmc_diag.get("missing_months") or []
                if missing_months:
                    reason = f"BayesMC produced insufficient month coverage; missing_months={missing_months}"
                    diag_text = json.dumps(
                        {
                            "missing_months": missing_months,
                            "bayesmc_diag": bayesmc_diag,
                        }
                    )
                    _write_spd_raw_text(
                        run_id,
                        qid,
                        "insufficient_month_coverage",
                        diag_text if diag_text else first_text,
                    )
                else:
                    reason = "missing spds"
                    _write_spd_raw_text(run_id, qid, "missing_spds", first_text)

                reason = _append_ensemble_meta(reason, ensemble_meta_str)
                _record_no_forecast(
                    run_id,
                    qid,
                    iso3,
                    hz,
                    metric,
                    reason,
                    model_name="ensemble_bayesmc_v2",
                    raw_debug_written=True,
                )
                return

            logged_any_spd_call = False

            for call in raw_calls:
                ms = call.get("model_spec")
                if not isinstance(ms, ModelSpec):
                    continue
                await log_forecaster_llm_call(
                    run_id=run_id,
                    question_id=qid,
                    iso3=iso3,
                    hazard_code=hz,
                    metric=metric,
                    model_spec=ms,
                    prompt_text=prompt,
                    response_text=str(call.get("text") or ""),
                    usage=call.get("usage") or {},
                    error_text=str(call.get("error")) if call.get("error") else None,
                    phase="spd_v2",
                    call_type="spd_v2",
                    hs_run_id=hs_run_id,
                )
                logged_any_spd_call = True

            if not logged_any_spd_call:
                fallback = raw_calls[0] if raw_calls else {}
                await log_forecaster_llm_call(
                    run_id=run_id,
                    question_id=qid,
                    iso3=iso3,
                    hazard_code=hz,
                    metric=metric,
                    model_spec=fallback.get("model_spec")
                    if isinstance(fallback.get("model_spec"), ModelSpec)
                    else None,
                    prompt_text=prompt,
                    response_text=str(fallback.get("text") or ""),
                    usage=fallback.get("usage") or {},
                    error_text=(
                        str(fallback.get("error"))
                        if fallback.get("error")
                        else "bayesmc: no ensemble members"
                    ),
                    phase="spd_v2",
                    call_type="spd_v2",
                    hs_run_id=hs_run_id,
                )
        else:
            text, usage, error, ms = await _call_spd_model_compat(
                prompt,
                run_id=run_id,
                question_id=qid,
                iso3=iso3,
                hazard_code=hz,
                metric=metric,
                target_month=target_month,
                wording=wording,
            )

            await log_forecaster_llm_call(
                run_id=run_id,
                question_id=qid,
                iso3=iso3,
                hazard_code=hz,
                metric=metric,
                model_spec=ms,
                prompt_text=prompt,
                response_text=text or "",
                usage=usage,
                error_text=str(error) if error else None,
                phase="spd_v2",
                hs_run_id=hs_run_id,
            )

            if error or not text or not text.strip():
                _record_no_forecast(
                    run_id,
                    qid,
                    iso3,
                    hz,
                    metric,
                    f"LLM error or empty response: {error or 'no text'}",
                )
                return

            try:
                spd_obj = _safe_json_loads(text)
            except json.JSONDecodeError as exc:
                raw_dir = Path("debug/spd_raw")
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / f"{run_id}__{qid}.txt"
                raw_path.write_text(text or "", encoding="utf-8")
                logging.error(
                    "SPD JSON decode failed for %s: %s (saved to %s)", qid, exc, raw_path
                )
                _record_no_forecast(run_id, qid, iso3, hz, metric, f"bad SPD JSON: {exc}")
                return

        if not isinstance(spd_obj, dict) or "spds" not in spd_obj:
            first_text = text or ""
            if raw_calls:
                first_text = str(raw_calls[0].get("text") or "")
            raw_path = _write_spd_raw_text(run_id, qid, "missing_spds", first_text)
            LOG.error(
                "SPD JSON missing 'spds' key for %s (saved raw text to %s)",
                qid,
                raw_path,
            )
            _record_no_forecast(
                run_id,
                qid,
                iso3,
                hz,
                metric,
                "missing spds",
                raw_debug_written=True,
            )
            return

        spds = spd_obj.get("spds") or {}
        if not spds:
            raw_dir = Path("debug/spd_raw")
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f"{run_id}__{qid}_empty_spds.txt"
            raw_path.write_text(text or "", encoding="utf-8")
            LOG.error(
                "SPD JSON contained empty 'spds' for %s (saved raw text to %s)",
                qid,
                raw_path,
            )
            _record_no_forecast(run_id, qid, iso3, hz, metric, "empty spds")
            return

        _write_spd_outputs(
            run_id,
            question_row,
            spd_obj,
            resolution_source=resolution_source,
            usage=usage or {},
            model_name="ensemble_bayesmc_v2" if use_bayesmc else "ensemble",
        )

    except Exception:
        LOG.exception("SPD v2 failed for question_id=%s", qid)

def _maybe_dump_raw_gtmc1(content: str, *, run_id: str, question_id: str) -> Optional[str]:
    """
    If PYTHIA_DEBUG_RAW=1, write the raw LLM JSON-ish text we received for the
    GTMC1 actor table to a file in gtmc_logs/ and return the path. Otherwise None.
    """
    if os.getenv("PYTHIA_DEBUG_RAW", "0") != "1":
        return None
    try:
        os.makedirs("gtmc_logs", exist_ok=True)
        path = os.path.join("gtmc_logs", f"{run_id}_q{question_id}_actors_raw.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
    except Exception:
        return None

# --------------------------------------------------------------------------------
# Calibration weights loader (optional legacy file fallback).
# --------------------------------------------------------------------------------

def _load_calibration_weights_file() -> Dict[str, Any]:
    path = os.getenv("CALIB_WEIGHTS_PATH", "")
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_calibration_weights_db(
    hazard_code: str,
    metric: str,
) -> Optional[Dict[str, float]]:
    try:
        from resolver.db import duckdb_io
    except Exception:
        return None

    hz = (hazard_code or "").upper().strip()
    mt = (metric or "").upper().strip()
    if not hz or not mt:
        return None

    db_url = _pythia_db_url_from_config() or os.getenv("RESOLVER_DB_URL", "").strip()
    if not db_url:
        return None

    conn = None
    try:
        conn = duckdb_io.get_db(db_url)
    except Exception:
        return None

    try:
        row = conn.execute(
            """
            SELECT as_of_month
            FROM calibration_weights
            WHERE hazard_code = ? AND metric = ?
            ORDER BY as_of_month DESC
            LIMIT 1
            """,
            [hz, mt],
        ).fetchone()
        if not row:
            return None
        as_of_month = str(row[0])

        rows = conn.execute(
            """
            SELECT model_name, weight
            FROM calibration_weights
            WHERE hazard_code = ? AND metric = ? AND as_of_month = ?
            ORDER BY COALESCE(model_name, '')
            """,
            [hz, mt, as_of_month],
        ).fetchall()
        if not rows:
            return None

        weights: Dict[str, float] = {}
        for model_name, weight in rows:
            if model_name is None:
                continue
            weights[str(model_name)] = float(weight)
        if not weights:
            return None

        if os.getenv("PYTHIA_DEBUG_DB", "0") == "1":
            print(
                "[forecaster] loaded calibration weights for hazard="
                f"{hz} metric={mt} as_of={as_of_month}: {weights}"
            )
        return weights
    except Exception:
        return None
    finally:
        try:
            duckdb_io.close_db(conn)
        except Exception:
            pass


def _expected_spd_model_ids() -> list[str]:
    spec_override = os.getenv("PYTHIA_SPD_ENSEMBLE_SPECS", "").strip()
    specs = parse_ensemble_specs(spec_override) if spec_override else SPD_ENSEMBLE
    model_ids = sorted({spec.model_id for spec in specs if getattr(spec, "model_id", "")})
    return list(model_ids)


def _load_calibration_advice_db(
    hazard_code: str,
    metric: str,
) -> Optional[str]:
    try:
        from resolver.db import duckdb_io
    except Exception:
        return None

    hz = (hazard_code or "").upper().strip()
    mt = (metric or "").upper().strip()
    if not hz or not mt:
        return None

    db_url = _pythia_db_url_from_config() or os.getenv("RESOLVER_DB_URL", "").strip()
    if not db_url:
        return None

    conn = None
    try:
        conn = duckdb_io.get_db(db_url)
    except Exception:
        return None

    try:
        row = conn.execute(
            """
            SELECT advice
            FROM calibration_advice
            WHERE hazard_code = ? AND metric = ?
            ORDER BY as_of_month DESC
            LIMIT 1
            """,
            [hz, mt],
        ).fetchone()
        if not row:
            return None
        advice = row[0]
        return str(advice)
    except Exception:
        return None
    finally:
        try:
            duckdb_io.close_db(conn)
        except Exception:
            pass

def _choose_weights_for_question(
    calib: Dict[str, Any],
    class_primary: str,
    qtype: str,
    ensemble_specs: List[ModelSpec] | None = None,
) -> Tuple[Dict[str, float], str]:
    model_names = [ms.name for ms in (ensemble_specs or DEFAULT_ENSEMBLE)]
    # 1) class-conditional
    try:
        by_class = calib.get("by_class", {})
        w = by_class.get(class_primary or "", {}).get(qtype, {})
        if isinstance(w, dict) and w:
            out = {m: float(w.get(m, 0.0)) for m in model_names}
            s = sum(out.values()) or 0.0
            if s > 0:
                return out, f"class_conditional:{class_primary}:{qtype}"
    except Exception:
        pass
    # 2) global
    try:
        glob = calib.get("global", {})
        w = glob.get(qtype, {})
        if isinstance(w, dict) and w:
            out = {m: float(w.get(m, 0.0)) for m in model_names}
            s = sum(out.values()) or 0.0
            if s > 0:
                return out, f"global:{qtype}"
    except Exception:
        pass
    # 3) uniform
    return ({m: 1.0 for m in model_names}, "uniform")

# --------------------------------------------------------------------------------
# Shape helpers
# --------------------------------------------------------------------------------

def _get_possibilities(q: dict) -> dict:
    return (q.get("possibilities") or q.get("range") or {})

def _get_options_list(q: dict) -> List[str]:
    if isinstance(q.get("options"), list):
        out = []
        for opt in q["options"]:
            if isinstance(opt, dict):
                out.append(str(opt.get("label") or opt.get("name") or ""))
            else:
                out.append(str(opt))
        return out
    poss = _get_possibilities(q)
    if isinstance(poss.get("options"), list):
        return [str(x.get("name") if isinstance(x, dict) else x) for x in poss["options"]]
    if isinstance(poss.get("scale", {}).get("options"), list):
        return [str(x.get("name") if isinstance(x, dict) else x) for x in poss["scale"]["options"]]
    return []

def _is_discrete(q: dict) -> bool:
    poss = _get_possibilities(q)
    q_type = (poss.get("type") or q.get("type") or "").lower()
    if q_type == "discrete":
        return True
    if q_type == "numeric" and isinstance(poss.get("scale", {}).get("values"), list):
        return True
    return False

def _discrete_values(q: dict) -> List[float]:
    poss = _get_possibilities(q)
    values = poss.get("scale", {}).get("values") or poss.get("values")
    if not values:
        return []
    return [float(v) for v in values]

# --------------------------------------------------------------------------------
# Simple, no-BMC fallback aggregators for the diagnostic variant "no_bmc_no_gtmc1"
# --------------------------------------------------------------------------------

def _simple_average_binary(members: List[MemberOutput]) -> Optional[float]:
    vals = [float(m.parsed) for m in members if m.ok and isinstance(m.parsed, (int, float))]
    if not vals:
        return None
    return float(np.mean([_clip01(v) for v in vals]))

def _simple_average_mcq(members: List[MemberOutput], n_opts: int) -> Optional[List[float]]:
    vecs: List[List[float]] = []
    for m in members:
        if m.ok and isinstance(m.parsed, list) and len(m.parsed) == n_opts:
            v = np.asarray(m.parsed, dtype=float)
            v = np.clip(v, 0.0, 1.0)
            s = float(v.sum())
            if s > 0:
                vecs.append((v / s).tolist())
    if not vecs:
        return None
    mean = np.mean(np.asarray(vecs), axis=0)
    mean = np.clip(mean, 1e-9, 1.0)
    mean = mean / float(mean.sum())
    return mean.tolist()

def _simple_average_numeric(members: List[MemberOutput]) -> Optional[Dict[str, float]]:
    p10s, p50s, p90s = [], [], []
    for m in members:
        if m.ok and isinstance(m.parsed, dict):
            d = m.parsed
            if "P10" in d and "P90" in d:
                p10s.append(float(d["P10"]))
                p90s.append(float(d["P90"]))
                p50s.append(float(d.get("P50", 0.5*(float(d["P10"]) + float(d["P90"])))))
    if not p10s:
        return None
    return {
        "P10": float(np.mean(p10s)),
        "P50": float(np.mean(p50s)) if p50s else 0.5 * (float(np.mean(p10s)) + float(np.mean(p90s))),
        "P90": float(np.mean(p90s)),
    }

# ==============================================================================
# CLI entrypoint
# ==============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forecaster runner")
    p.add_argument(
        "--mode",
        default="pythia",
        choices=["pythia", "test_questions"],
        help=(
            "Question source: 'pythia' (DuckDB questions table) or "
            "'test_questions' (local JSON)."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Max questions to forecast. If <= 0, no limit is applied (all questions "
            "from the current HS epoch)."
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=(
            "Optional batch size for SPD v2. If > 0, Research/SPD/Scenario will "
            "run in batches of this many questions; otherwise all questions are "
            "processed in a single batch."
        ),
    )
    p.add_argument("--purpose", default="ad_hoc", help="String tag recorded in CSV/logs")
    p.add_argument(
        "--questions-file",
        default="data/test_questions.json",
        help="When --mode test_questions, path to JSON payload",
    )
    p.add_argument(
        "--iso3",
        type=str,
        default="",
        help=(
            "Optional comma-separated list of ISO3 codes to forecast "
            "(e.g. 'ETH,SOM'). If omitted, SPD v2 defaults to all countries "
            "present in hs_triage for the chosen HS epoch."
        ),
    )
    p.add_argument(
        "--hs-run-id",
        type=str,
        default="",
        help=(
            "Optional HS run id (hs_run_id) to use as epoch for SPD v2. "
            "If omitted, the latest hs_run_id in hs_runs/hs_triage is used."
        ),
    )
    return p.parse_args()

def main() -> None:
    args = _parse_args()
    print("🚀 Forecaster ensemble starting…")
    print(f"Mode: {args.mode} | Limit: {args.limit} | Purpose: {args.purpose}")

    def _run_v2_pipeline():
        ensure_schema()

        # Parse iso3 filter from CLI
        iso3_filter: Optional[set[str]] = None
        if args.iso3:
            iso3_filter = {
                code.strip().upper()
                for code in args.iso3.split(",")
                if code.strip()
            } or None

        # Determine HS epoch (hs_run_id) to use
        con = connect(read_only=True)
        try:
            hs_run_id = _select_hs_run_id_for_forecast(
                con,
                explicit=(args.hs_run_id or None),
            )
            if not hs_run_id:
                print("[fatal] No HS epoch (hs_run_id) found; cannot run SPD v2.")
                return

            print(f"[v2] Using HS epoch hs_run_id={hs_run_id}")
            rows = con.execute(
                """
                SELECT DISTINCT iso3
                FROM hs_triage
                WHERE run_id = ?
                ORDER BY iso3
                """,
                [hs_run_id],
            ).fetchall()
            epoch_iso3s = {row[0] for row in rows}

            if iso3_filter:
                allowed_iso3s = (
                    epoch_iso3s & iso3_filter if epoch_iso3s else iso3_filter
                )
            else:
                allowed_iso3s = epoch_iso3s

            if not allowed_iso3s:
                print(
                    f"[fatal] No iso3s to forecast for hs_run_id={hs_run_id} (allowed_iso3s empty)."
                )
                return

            cols = [
                "question_id",
                "hs_run_id",
                "iso3",
                "hazard_code",
                "metric",
                "target_month",
                "window_start_date",
                "window_end_date",
                "wording",
                "track",
            ]
            placeholders = ",".join(["?"] * len(allowed_iso3s))
            # Check if track column exists (backward compat with older DBs)
            q_cols = {r[0] for r in con.execute("DESCRIBE questions").fetchall()}
            track_expr = "track" if "track" in q_cols else "NULL AS track"
            sql = f"""
                SELECT
                    question_id, hs_run_id, iso3, hazard_code, metric,
                    target_month, window_start_date, window_end_date, wording,
                    {track_expr}
                FROM questions
                WHERE status = 'active'
                  AND hs_run_id = ?
                  AND iso3 IN ({placeholders})
                  AND UPPER(COALESCE(hazard_code, '')) <> 'ACO'
                ORDER BY iso3, hazard_code, metric, target_month, question_id
            """
            params: List[Any] = [hs_run_id] + list(allowed_iso3s)

            if args.limit and args.limit > 0:
                sql += "\n                LIMIT ?"
                params.append(args.limit)

            raw_rows = con.execute(sql, params).fetchall()
        finally:
            con.close()

        questions = [
            dict(
                zip(
                    [
                        "question_id",
                        "hs_run_id",
                        "iso3",
                        "hazard_code",
                        "metric",
                        "target_month",
                        "window_start_date",
                        "window_end_date",
                        "wording",
                        "track",
                    ],
                    row,
                )
            )
            for row in raw_rows
        ]

        if not questions:
            print("[fatal] No questions selected for SPD v2; nothing to forecast.")
            return

        run_id = f"fc_{int(time.time())}"
        os.environ["PYTHIA_FORECASTER_RUN_ID"] = run_id
        reset_provider_failures_for_run(run_id)
        n_track1 = sum(1 for q in questions if q.get("track") == 1)
        n_track2 = sum(1 for q in questions if q.get("track") == 2)
        print(f"[v2] run_id={run_id} | questions={len(questions)} (track1={n_track1}, track2={n_track2})")

        async def _run_v2_pipeline_async() -> None:
            research_sem = asyncio.Semaphore(MAX_RESEARCH_WORKERS)
            spd_sem = asyncio.Semaphore(MAX_SPD_WORKERS)
            question_start_ms: dict[str, int] = {}
            expected_model_ids = _expected_spd_model_ids()

            def _record_question_run_metrics(q: dict, *, start_ms: int) -> None:
                qid = str(q.get("question_id") or "")
                if not qid:
                    return
                finished_at = datetime.utcnow()
                wall_ms = int(time.time() * 1000) - int(start_ms)
                started_at = datetime.utcfromtimestamp(start_ms / 1000.0)
                cost_usd = 0.0
                ok_model_ids: list[str] = []
                con = pythia_connect(read_only=False)
                try:
                    ensure_schema(con)
                    row = con.execute(
                        """
                        SELECT COALESCE(SUM(cost_usd), 0.0)
                        FROM llm_calls
                        WHERE run_id = ? AND question_id = ?
                        """,
                        [run_id, qid],
                    ).fetchone()
                    if row:
                        cost_usd = float(row[0] or 0.0)
                    ok_rows = con.execute(
                        """
                        SELECT DISTINCT model_id
                        FROM llm_calls
                        WHERE run_id = ?
                          AND question_id = ?
                          AND call_type = 'spd_v2'
                          AND (error_text IS NULL OR error_text = '')
                          AND model_id IS NOT NULL
                          AND model_id <> ''
                        ORDER BY model_id
                        """,
                        [run_id, qid],
                    ).fetchall()
                    ok_model_ids = [str(r[0]) for r in ok_rows if r and r[0]]
                    missing_model_ids = sorted(set(expected_model_ids) - set(ok_model_ids))
                    con.execute(
                        "DELETE FROM question_run_metrics WHERE run_id = ? AND question_id = ?",
                        [run_id, qid],
                    )
                    con.execute(
                        """
                        INSERT INTO question_run_metrics (
                            run_id, question_id, iso3, hazard_code, metric,
                            started_at_utc, finished_at_utc, wall_ms, cost_usd,
                            n_spd_models_expected, n_spd_models_ok, missing_model_ids_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            run_id,
                            qid,
                            (q.get("iso3") or ""),
                            (q.get("hazard_code") or ""),
                            (q.get("metric") or ""),
                            started_at,
                            finished_at,
                            wall_ms,
                            cost_usd,
                            len(expected_model_ids),
                            len(ok_model_ids),
                            json.dumps(missing_model_ids, ensure_ascii=False),
                        ],
                    )
                finally:
                    con.close()

            # DEPRECATED: _run_research_for_question is no longer called. Scheduled for removal.
            async def _research_task(q: dict) -> None:
                qid = str(q.get("question_id") or "")
                iso3 = (q.get("iso3") or "").upper()
                hz = (q.get("hazard_code") or "").upper()
                metric = q.get("metric") or "PA"
                # Persist minimal placeholder — no web search calls
                try:
                    con = connect(read_only=False)
                    ensure_schema(con)
                    con.execute(
                        """
                        INSERT INTO question_research
                          (run_id, question_id, iso3, hazard_code, metric, research_json,
                           hs_evidence_json, question_evidence_json, merged_evidence_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [run_id, qid, iso3, hz, metric,
                         '{"note": "researcher_retired"}', '{}', '{}', '{}'],
                    )
                    con.close()
                except Exception:
                    pass

            async def _spd_task(q: dict) -> None:
                qid = str(q.get("question_id") or "")
                if qid and qid not in question_start_ms:
                    question_start_ms[qid] = int(time.time() * 1000)
                if not _question_needs_spd(run_id, q):
                    start_ms = question_start_ms.get(qid or "")
                    if start_ms is not None:
                        _record_question_run_metrics(q, start_ms=start_ms)
                    return
                async with spd_sem:
                    track_val = q.get("track")
                    if track_val == 2:
                        await _run_track2_spd_for_question(run_id, q)
                    else:
                        await _run_spd_for_question(run_id, q)
                if not qid:
                    return
                start_ms = question_start_ms.get(qid)
                if start_ms is None:
                    return
                _record_question_run_metrics(q, start_ms=start_ms)

            if args.batch_size and args.batch_size > 0:
                batch_size = max(1, args.batch_size)
            else:
                batch_size = len(questions) or 1

            for start_idx in range(0, len(questions), batch_size):
                batch = questions[start_idx : start_idx + batch_size]
                print(
                    f"[v2] Processing batch {start_idx // batch_size + 1} "
                    f"({len(batch)} question(s)) / total {len(questions)}"
                )

                await asyncio.gather(*(_research_task(q) for q in batch))
                await asyncio.gather(*(_spd_task(q) for q in batch))

        asyncio.run(_run_v2_pipeline_async())

        run_scenarios_for_run(run_id)

    try:
        _run_v2_pipeline()
    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"[fatal] {type(e).__name__}: {str(e)[:200]}")
        raise


if __name__ == "__main__":
    main()
