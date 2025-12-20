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
import importlib
import importlib.util
import json
import os
import re
import logging
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
from forecaster.hs_utils import load_hs_triage_entry

LOG = logging.getLogger(__name__)

MAX_RESEARCH_WORKERS = int(os.getenv("FORECASTER_RESEARCH_MAX_WORKERS", "6"))
MAX_SPD_WORKERS = int(os.getenv("FORECASTER_SPD_MAX_WORKERS", "6"))


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

_PYTHIA_CFG_LOAD = None
if importlib.util.find_spec("pythia.config") is not None:
    _PYTHIA_CFG_LOAD = getattr(importlib.import_module("pythia.config"), "load", None)

try:
    from pythia.llm_profiles import get_current_models as _get_llm_profile_models
except Exception:
    _get_llm_profile_models = None  # type: ignore


# Hazard codes for which GTMC1 is relevant (adjust as needed for your schema)
CONFLICT_HAZARD_CODES = {
    "CONFLICT",
    "POLITICAL_VIOLENCE",
    "CIVIL_CONFLICT",
    "URBAN_CONFLICT",
}

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
    # EM-DAT natural hazards
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

EMDAT_SHOCK_MAP = {
    "FL": "flood",
    "DR": "drought",
    "TC": "tropical cyclone",
    "HW": "heat wave",
}

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
              AND COALESCE(NULLIF(upper(hazard_code), ''), 'ACE') = ?
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


def _map_hazard_to_emdat_shock(hazard_code: str) -> str:
    hz = hazard_code.upper()
    if hz == "FL":
        return "flood"
    if hz == "DR":
        return "drought"
    if hz == "TC":
        return "tropical cyclone"
    if hz == "HW":
        return "heat wave"
    return hz.lower()


def _build_history_summary(iso3: str, hazard_code: str, metric: str) -> Dict[str, Any]:
    """Build Resolver history summary per hazard/metric rules."""

    hz = (hazard_code or "").upper()
    m = (metric or "").upper()

    if hz == "DI":
        return {
            "source": "NONE",
            "history_length_months": 0,
            "recent_mean": None,
            "recent_max": None,
            "trend": "uncertain",
            "last_6m_values": [],
            "data_quality": "low",
            "notes": (
                "DI (displacement inflow) has no Resolver base rate; rely on HS + research "
                "and exogenous neighbour shocks."
            ),
        }

    con = connect(read_only=True)
    try:

        if m == "FATALITIES":
            try:
                rows = con.execute(
                    """
                    SELECT month, fatalities
                    FROM acled_monthly_fatalities
                    WHERE iso3 = ?
                    ORDER BY month
                    """,
                    [iso3],
                ).fetchall()
            except Exception as exc:
                logging.warning(
                    "ACLED history query failed for %s: %s", iso3, exc
                )
                return {
                    "source": "ACLED",
                    "history_length_months": 0,
                    "recent_mean": None,
                    "recent_max": None,
                    "trend": "uncertain",
                    "last_6m_values": [],
                    "data_quality": "low",
                    "notes": f"ACLED history unavailable: {type(exc).__name__}",
                }

            if not rows:
                return {
                    "source": "ACLED",
                    "history_length_months": 0,
                    "recent_mean": None,
                    "recent_max": None,
                    "trend": "uncertain",
                    "last_6m_values": [],
                    "data_quality": "low",
                    "notes": "No ACLED history for this country/hazard.",
                }

            month_to_value = {str(r[0]): r[1] for r in rows}
            month_to_value, dropped_future, unparseable_keys = _sanitize_month_series(month_to_value)
            if not month_to_value:
                summary_empty: Dict[str, Any] = {
                    "source": "ACLED",
                    "history_length_months": 0,
                    "recent_mean": None,
                    "recent_max": None,
                    "trend": "uncertain",
                    "last_6m_values": [],
                    "data_quality": "low",
                    "notes": "ACLED history was filtered out (future or invalid months).",
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
            vals = [int(v) for _k, v in ordered_items]
            n = len(vals)
            recent = vals[-min(n, 6):]
            trend = "uncertain"
            if len(recent) >= 2:
                if recent[-1] > recent[0]:
                    trend = "up"
                elif recent[-1] < recent[0]:
                    trend = "down"
                else:
                    trend = "flat"

            summary: Dict[str, Any] = {
                "source": "ACLED",
                "history_length_months": n,
                "recent_mean": sum(recent) / len(recent),
                "recent_max": max(recent),
                "trend": trend,
                "last_6m_values": [
                    {"ym": ym, "value": int(val)} for ym, val in ordered_items[-min(n, 6) :]
                ],
                "data_quality": "high",
                "notes": "ACLED coverage is relatively strong for this country/hazard.",
            }
            sanity: Dict[str, Any] = {}
            if dropped_future:
                sanity["dropped_future_months"] = dropped_future
            if unparseable_keys:
                sanity["unparseable_month_keys"] = unparseable_keys
            if sanity:
                summary["_sanity"] = sanity
            return summary
            sanity: Dict[str, Any] = {}
            if dropped_future:
                sanity["dropped_future_months"] = dropped_future
            if unparseable_keys:
                sanity["unparseable_month_keys"] = unparseable_keys
            if sanity:
                return_summary["_sanity"] = sanity
            return return_summary

        if m == "PA" and hz in {"ACE"}:
            try:
                summary = _load_idmc_conflict_flow_history_summary(con, iso3, hz)
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "IDMC conflict PA summary helper failed for %s/%s: %s",
                    iso3,
                    hz,
                    exc,
                )
                summary = {
                    "source": "IDMC",
                    "history_length_months": 0,
                    "recent_mean": None,
                    "recent_max": None,
                    "trend": "uncertain",
                    "last_6m_values": [],
                    "data_quality": "low",
                    "notes": f"IDMC history helper crashed: {type(exc).__name__}",
                }

            return summary

        if m == "PA" and hz in {"FL", "DR", "TC", "HW"}:
            shock = _map_hazard_to_emdat_shock(hazard_code)
            try:
                rows = con.execute(
                    """
                    SELECT ym, pa
                    FROM emdat_pa
                    WHERE iso3 = ?
                      AND shock_type = ?
                    ORDER BY ym
                    """,
                    [iso3, shock],
                ).fetchall()
            except Exception as exc:
                logging.warning(
                    "EM-DAT history query failed for %s/%s: %s", iso3, shock, exc
                )
                return {
                    "source": "EM-DAT",
                    "history_length_months": 0,
                    "recent_mean": None,
                    "recent_max": None,
                    "trend": "uncertain",
                    "last_6m_values": [],
                    "data_quality": "low",
                    "notes": f"EM-DAT history unavailable: {type(exc).__name__}",
                }

            if not rows:
                return {
                    "source": "EM-DAT",
                    "history_length_months": 0,
                    "recent_mean": None,
                    "recent_max": None,
                    "trend": "uncertain",
                    "last_6m_values": [],
                    "data_quality": "low",
                    "notes": "No reliable EM-DAT history for this country/hazard; treat base rate as unknown.",
                }

            month_to_value = {str(r[0]): r[1] for r in rows}
            month_to_value, dropped_future, unparseable_keys = _sanitize_month_series(month_to_value)
            if not month_to_value:
                summary_empty = {
                    "source": "EM-DAT",
                    "history_length_months": 0,
                    "recent_mean": None,
                    "recent_max": None,
                    "trend": "uncertain",
                    "last_6m_values": [],
                    "data_quality": "low",
                    "notes": "EM-DAT history was filtered out (future or invalid months).",
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
            vals = [float(v) for _k, v in ordered_items]
            n = len(vals)
            recent = vals[-min(n, 6):]
            trend = "uncertain"
            if len(recent) >= 2:
                if recent[-1] > recent[0]:
                    trend = "up"
                elif recent[-1] < recent[0]:
                    trend = "down"
                else:
                    trend = "flat"

            summary = {
                "source": "EM-DAT",
                "history_length_months": n,
                "recent_mean": sum(recent) / len(recent),
                "recent_max": max(recent),
                "trend": trend,
                "last_6m_values": [
                    {"ym": ym, "value": float(val)} for ym, val in ordered_items[-min(n, 6) :]
                ],
                "data_quality": "medium",
                "notes": "EM-DAT often only records large disasters; treat as a noisy base-rate signal.",
            }
            sanity: Dict[str, Any] = {}
            if dropped_future:
                sanity["dropped_future_months"] = dropped_future
            if unparseable_keys:
                sanity["unparseable_month_keys"] = unparseable_keys
            if sanity:
                summary["_sanity"] = sanity
            return summary

        return {
            "source": "NONE",
            "history_length_months": 0,
            "recent_mean": None,
            "recent_max": None,
            "trend": "uncertain",
            "last_6m_values": [],
            "data_quality": "low",
            "notes": "No usable Resolver history for this hazard/metric.",
        }
    finally:
        con.close()


def _infer_resolution_source(hazard_code: str, metric: str) -> str:
    hz = (hazard_code or "").upper()
    mt = (metric or "").upper()

    if mt == "FATALITIES" and hz in {"ACE"}:
        return "ACLED"
    if mt == "PA" and hz in {"ACE"}:
        return "IDMC"
    if mt == "PA" and hz in {"DR", "FL", "TC", "HW"}:
        return "EM-DAT"
    if mt == "PA" and hz == "DI":
        return "NONE"
    return "NONE"


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


def _advise_poetry_lock_if_needed():
    # Dev convenience: if Poetry complains about a stale lock, print the fix.
    import os
    if os.getenv("CI"):
        return  # CI already handles regeneration
    # Lightweight hint only; we don't try to run Poetry here.
    os.environ.setdefault("PYTHIA_LOCK_HINT_SHOWN", "0")


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
        parsed = _parse_month_key(str(key))
        if parsed is None:
            unparseable.append(str(key))
            cleaned[str(key)] = val
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
    build_binary_prompt,
    build_numeric_prompt,
    build_mcq_prompt,
    build_spd_prompt,
    build_spd_prompt_fatalities,
    build_spd_prompt_pa,
    build_research_prompt_v2,
    build_spd_prompt_v2,
)
from .scenario_writer import run_scenarios_for_run
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
from .research import run_research_async

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

def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

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
                            p
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
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
        con.execute(
            "DELETE FROM forecasts_raw WHERE run_id = ? AND question_id = ?;",
            [run_id, question_id],
        )

        for m in ens_res.members:
            model_name = getattr(m, "name", "")
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
                                total_tokens
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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


def _load_emdat_pa_history(
    iso3: str,
    hazard_code: str,
    *,
    months: int = 36,
) -> Tuple[str, Dict[str, Any]]:
    """
    Load a 36-month EM-DAT 'people affected' history for a given ISO3 +
    Pythia hazard code.

    Uses the emdat_pa table as the single source of truth for monthly PA
    values.
    """

    hz = (hazard_code or "").upper()
    shock_type = EMDAT_SHOCK_MAP.get(hz, hz.lower())

    try:
        con = duckdb.connect(_pythia_db_path_from_config(), read_only=True)
    except Exception:
        return "", {"error": "missing_db", "history_rows_detail": [], "summary_text": ""}

    try:
        rows = con.execute(
            """
            SELECT ym, pa, shock_type, COALESCE(source_id, '') AS source_id
            FROM emdat_pa
            WHERE iso3 = ?
              AND lower(shock_type) = ?
            ORDER BY ym DESC
            LIMIT ?
            """,
            [iso3, shock_type.lower(), months],
        ).fetchall()
    except Exception as exc:
        con.close()
        return "", {
            "error": f"emdat_query_error:{type(exc).__name__}",
            "history_rows_detail": [],
            "summary_text": "",
        }

    con.close()

    if not rows:
        return "", {"error": "no_history", "history_rows_detail": [], "summary_text": ""}

    history: List[Dict[str, Any]] = []
    values: List[float] = []
    for ym, pa_val, shock_type_val, source_id in rows:
        ym_str = str(ym)
        v = float(pa_val or 0.0)
        history.append(
            {
                "ym": ym_str,
                "value": v,
                "source": "EM-DAT",
                "shock_type": shock_type_val,
                "source_id": source_id,
            }
        )
        values.append(v)

    history_for_table = list(reversed(history))

    summary_lines: List[str] = [
        "### EM-DAT people affected — 36-month history (Resolver)",
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
    - For metric='PA' and natural hazards (FL/DR/TC/HW) → EM-DAT PA history.
    """

    hz = (hazard_code or "").upper()
    m = (metric or "").upper()

    if m == "FATALITIES":
        return _load_acled_fatalities_history(iso3, months=months)

    if m == "PA" and hz in IDMC_HZ_MAP:
        return _load_idmc_pa_history(iso3, hazard_code, months=months)

    if m == "PA" and hz in EMDAT_SHOCK_MAP:
        return _load_emdat_pa_history(iso3, hazard_code, months=months)

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


def _question_needs_spd(run_id: str, question_row: duckdb.Row) -> bool:
    iso3 = (question_row.get("iso3") or "").upper()
    hazard_code = (question_row.get("hazard_code") or "").upper()
    hs_run_id = question_row.get("hs_run_id") or run_id
    triage = load_hs_triage_entry(hs_run_id, iso3, hazard_code)
    if not triage:
        return True
    return bool(triage.get("need_full_spd", False))


async def _call_research_model(prompt: str, *, run_id: str | None = None) -> tuple[str, Dict[str, Any], Optional[str], ModelSpec]:
    """Async wrapper for the research LLM call for v2 pipeline."""

    ms = ModelSpec(
        name="Gemini",
        provider="google",
        model_id=GEMINI_MODEL_ID,
        active=True,
        purpose="research_v2",
    )
    start = time.time()
    try:
        text, usage, error = await call_chat_ms(
            ms,
            prompt,
            temperature=0.3,
            prompt_key="research.v2",
            prompt_version="1.0.0",
            component="Researcher",
            run_id=run_id,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.time() - start) * 1000)
        return "", {"elapsed_ms": elapsed_ms}, f"provider call error: {exc}", ms

    usage = dict(usage or {})
    usage.setdefault("elapsed_ms", int((time.time() - start) * 1000))
    return text, usage, error, ms


async def _call_spd_model(prompt: str, *, run_id: str | None = None) -> tuple[str, Dict[str, Any], Optional[str], ModelSpec]:
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
    return await _call_spd_model_for_spec(ms, prompt, run_id=run_id)


async def _call_spd_model_for_spec(
    ms: ModelSpec, prompt: str, *, run_id: str | None = None, **_kwargs
) -> tuple[str, Dict[str, Any], Optional[str], ModelSpec]:
    """Async wrapper for the SPD LLM call for a given model spec."""

    start = time.time()
    try:
        text, usage, error = await call_chat_ms(
            ms,
            prompt,
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
    return text, usage, error, ms


async def _call_spd_members_v2(
    prompt: str, specs: list[ModelSpec], *, run_id: str | None = None
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

    tasks = [_call_spd_model_for_spec(ms, prompt, run_id=run_id) for ms in specs_used]
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
) -> tuple[
    dict[str, object], dict[str, object], list[dict[str, object]], dict[str, object]
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
    specs = specs or SPD_ENSEMBLE
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
        return {}, {}, [], ensemble_meta

    per_model_spds, aggregated_usage, raw_calls, ensemble_meta = await _call_spd_members_v2(
        prompt, specs_used, run_id=run_id
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
        return {}, aggregated_usage, raw_calls, ensemble_meta

    return spd_obj, aggregated_usage, raw_calls, ensemble_meta


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

        prompt = build_research_prompt_v2(
            question={
                "question_id": qid,
                "iso3": iso3,
                "hazard_code": hz,
                "metric": metric,
                "resolution_source": resolution_source,
                "wording": wording,
            },
            hs_triage_entry=hs_entry,
            resolver_features=resolver_features,
            model_info={},
        )

        text, usage, error, ms = await _call_research_model(prompt, run_id=run_id)

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
            phase="research_v2",
            hs_run_id=hs_run_id,
        )

        if error or not text or not text.strip():
            logging.error("Research v2 returned error/empty for %s: %s", qid, error)
            return

        try:
            research = _safe_json_loads(text)
        except json.JSONDecodeError as exc:
            raw_dir = Path("debug/research_raw")
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = raw_dir / f"{run_id}__{qid}.txt"
            raw_path.write_text(text or "", encoding="utf-8")
            logging.error(
                "Research JSON decode failed for %s: %s (saved to %s)",
                qid,
                exc,
                raw_path,
            )
            return
        con = connect(read_only=False)
        try:
            con.execute(
                """
                INSERT INTO question_research
                  (run_id, question_id, iso3, hazard_code, metric, research_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [run_id, qid, iso3, hz, metric, _json_dumps_for_db(research)],
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
        for month_idx, (month_label, payload) in enumerate(sorted(spds.items()), start=1):
            probs = payload.get("probs") if isinstance(payload, dict) else None
            if not probs:
                continue
            for bucket_index, prob in enumerate(list(probs)[: len(bucket_labels)], start=1):
                con.execute(
                    """
                    INSERT INTO forecasts_raw (
                        run_id, question_id, model_name, month_index, bucket_index,
                        probability, ok, elapsed_ms, cost_usd, prompt_tokens, completion_tokens,
                        total_tokens, status, spd_json, human_explanation
                    ) VALUES (?, ?, ?, ?, ?, ?, TRUE, ?, ?, ?, ?, ?, 'ok', ?, ?)
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


async def _run_spd_for_question(run_id: str, question_row: Any) -> None:
    _maybe_log_default_ensemble()

    rec = _normalize_question_row_for_spd(question_row)

    qid = rec.get("question_id")
    iso3 = (rec.get("iso3") or "").upper()
    hz = (rec.get("hazard_code") or "").upper()
    metric = rec.get("metric") or "PA"
    wording = rec.get("wording") or rec.get("title") or ""

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
            }

        target_months = rec.get("target_months") or rec.get("target_month")
        target_month = _first_target_month(target_months)

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
        )

        dual_run = os.getenv("PYTHIA_SPD_V2_DUAL_RUN", "0") == "1"
        use_bayesmc = os.getenv("PYTHIA_SPD_V2_USE_BAYESMC", "0") == "1"
        write_both = os.getenv("PYTHIA_SPD_V2_WRITE_BOTH", "0") == "1"

        if dual_run:
            specs = list(SPD_ENSEMBLE)
            specs_used_summary = _specs_summary(specs)
            specs_active = [ms for ms in specs if getattr(ms, "active", False)]
            n_active_specs = len(specs_active)
            if not specs or n_active_specs == 0:
                diff = {
                    "error": "no_active_models",
                    "reason": "SPD_ENSEMBLE has no active model specs",
                    "n_specs": len(specs),
                    "n_active_specs": n_active_specs,
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
                            "reason": "spec list empty or inactive",
                            "n_calls": 0,
                            "n_active_specs": n_active_specs,
                        },
                    },
                    "bayesmc": {
                        "models": [],
                        "usage": {},
                        "status_info": {
                            "status": "no_active_models",
                            "reason": "spec list empty or inactive",
                            "n_calls": 0,
                            "n_active_specs": n_active_specs,
                        },
                    },
                    "diff": diff,
                }
                _write_spd_compare_artifact(run_id, qid, payload)
                return
            try:
                per_model_spds, aggregated_usage, raw_calls, ensemble_meta = (
                    await _call_spd_members_v2(prompt, specs_active, run_id=run_id)
                )
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
            specs = list(SPD_ENSEMBLE)
            specs_active = [ms for ms in specs if getattr(ms, "active", False)]

            if not specs_active:
                _record_no_forecast(
                    run_id,
                    qid,
                    iso3,
                    hz,
                    metric,
                    "no active ensemble models",
                    model_name="ensemble_mean_v2",
                )
                _record_no_forecast(
                    run_id,
                    qid,
                    iso3,
                    hz,
                    metric,
                    "no active ensemble models",
                    model_name="ensemble_bayesmc_v2",
                )
                return

            per_model_spds, usage, raw_calls, ensemble_meta = await _call_spd_members_v2(
                prompt, specs_active, run_id=run_id
            )

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

            if int(ensemble_meta.get("n_models_ok") or 0) < 2:
                reason = _append_ensemble_meta("insufficient ensemble coverage", ensemble_meta_str)
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

            spd_mean = aggregate_spd_v2_mean(per_model_spds)
            spd_mean_obj = {"spds": {m: {"probs": vec} for m, vec in spd_mean.items()}}
            _attach_ensemble_meta(spd_mean_obj, ensemble_meta)

            spd_bm_obj, diag_bm = _build_bayesmc_spd_obj(
                per_model_spds, target_month=target_month, specs_used=specs_active
            )
            if spd_bm_obj:
                spd_bm_obj.setdefault("bayesmc_diag", diag_bm)
                _attach_ensemble_meta(spd_bm_obj, ensemble_meta)

            if spd_mean_obj:
                _write_spd_outputs(
                    run_id,
                    question_row,
                    spd_mean_obj,
                    resolution_source=resolution_source,
                    usage=usage or {},
                    model_name="ensemble_mean_v2",
                )
            else:
                reason_mean = _append_ensemble_meta("missing spds", ensemble_meta_str)
                _record_no_forecast(
                    run_id,
                    qid,
                    iso3,
                    hz,
                    metric,
                    reason_mean,
                    model_name="ensemble_mean_v2",
                )

            reason_bm = "missing spds"
            if isinstance(diag_bm, dict):
                reason_bm = str(diag_bm.get("status") or reason_bm)

            if spd_bm_obj:
                _write_spd_outputs(
                    run_id,
                    question_row,
                    spd_bm_obj,
                    resolution_source=resolution_source,
                    usage=usage or {},
                    model_name="ensemble_bayesmc_v2",
                )
            else:
                reason_bm = _append_ensemble_meta(reason_bm, ensemble_meta_str)
                missing_months = []
                if isinstance(diag_bm, dict):
                    missing_months = diag_bm.get("missing_months") or []
                    if missing_months:
                        reason_bm = f"BayesMC produced insufficient month coverage; missing_months={missing_months}"
                raw_texts = [str(rc.get("text") or "") for rc in raw_calls if isinstance(rc, dict) and rc.get("text")]
                debug_payload = json.dumps(
                    {
                        "missing_months": missing_months,
                        "bayesmc_diag": diag_bm,
                    }
                )
                if not debug_payload and raw_texts:
                    debug_payload = raw_texts[0]
                if missing_months:
                    _write_spd_raw_text(run_id, qid, "insufficient_month_coverage", debug_payload)
                else:
                    _write_spd_raw_text(run_id, qid, "missing_spds", debug_payload)
                _record_no_forecast(
                    run_id,
                    qid,
                    iso3,
                    hz,
                    metric,
                    reason_bm,
                    model_name="ensemble_bayesmc_v2",
                    raw_debug_written=True,
                )

            return

        if use_bayesmc:
            spd_obj, usage, raw_calls, ensemble_meta = await _call_spd_bayesmc_v2(
                prompt,
                run_id=run_id,
                question_id=qid,
                hs_run_id=hs_run_id,
                target_month=target_month,
            )
            raw_texts = [str(rc.get("text") or "") for rc in raw_calls if isinstance(rc, dict)]
            text = json.dumps(spd_obj)

            ensemble_meta_str = _format_ensemble_meta(ensemble_meta)
            bayesmc_diag = {}
            try:
                diag_from_meta = ensemble_meta.get("bayesmc_diag")
                if isinstance(diag_from_meta, dict):
                    bayesmc_diag = diag_from_meta
            except Exception:
                bayesmc_diag = {}

            # If BayesMC yields no SPD months, treat as missing spds (same contract as v2 path).
            if int((ensemble_meta or {}).get("n_models_ok") or 0) < 2:
                first_text = ""
                if raw_texts:
                    first_text = raw_texts[0]
                elif text:
                    first_text = str(text)

                _write_spd_raw_text(run_id, qid, "missing_spds", first_text)

                reason = _append_ensemble_meta("missing spds", ensemble_meta_str)
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
            text, usage, error, ms = await _call_spd_model(prompt)

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

# --------------------------------------------------------------------------------
# Core orchestration for ONE question → produce a single CSV row
# --------------------------------------------------------------------------------

async def _run_one_question_body(
    post: dict,
    *,
    run_id: str,
    purpose: str,
    calib: Dict[str, Any],
    seen_guard_state: Dict[str, Any],
    seen_guard_run_report: Optional[Dict[str, Any]] = None,
    summary: Optional[QuestionRunSummary] = None,
) -> None:
    t_start_total = time.time()
    _maybe_log_default_ensemble()
    _post_original = post
    try:
    
        post = _must_dict("post", post)
        q = _must_dict("q", post.get("question"))

        required = ("title", "type")
        missing = [k for k in required if not str(q.get(k, "")).strip()]
        if missing:
            raise RuntimeError(f"question payload missing required keys: {missing}")

        # Metaculus posts use integer IDs; Pythia Horizon Scanner posts use hex string IDs.
        # Try to coerce to int for backwards compatibility, but fall back to 0 so hex IDs don't crash.
        raw_post_id = post.get("id") or post.get("post_id") or 0
        try:
            post_id = int(raw_post_id)
        except (TypeError, ValueError):
            post_id = 0
        question_id_raw = q.get("id") or post.get("id") or post.get("post_id") or ""
        question_id = str(question_id_raw)

        # Derive hs_run_id for this question, if available
        try:
            hs_run_id = (
                post.get("hs_run_id")
                or post.get("pythia_hs_run_id")
                or q.get("hs_run_id")
                or None
            )
        except Exception:
            hs_run_id = None

        seen_guard_enabled = bool(seen_guard_state.get("enabled", False))
        seen_guard_lock_acquired = seen_guard_state.get("lock_acquired")
        seen_guard_lock_error = str(seen_guard_state.get("lock_error") or "")

        title = str(q.get("title") or post.get("title") or "").strip()
        url = str(post.get("question_url") or "")
        qtype = (q.get("type") or "binary").strip()
        description = str(post.get("description") or q.get("description") or "")
        criteria = str(q.get("resolution_criteria") or q.get("fine_print") or q.get("resolution") or "")
        units = q.get("unit") or q.get("units") or ""
        tournament_id = post.get("pythia_hs_run_id") or ""
    
        # Options / discrete values
        options = _get_options_list(q)
        n_options = len(options) if qtype == "multiple_choice" else 0
        discrete_values = _discrete_values(q) if qtype in ("numeric", "discrete") and _is_discrete(q) else []
        ev_main: Optional[Dict[str, Any]] = None
        spd_bucket_labels_used: Optional[List[str]] = None
        spd_bucket_centroids_used: Optional[List[float]] = None
        spd_centroid_source = ""

        pmeta = _extract_pythia_meta(post)
        pythia_meta_full = _as_dict(post.get("pythia_metadata") or {})
        metric_up = (pmeta.get("metric") or "").upper()

        hz_code = (pmeta.get("hazard_code") or "").upper()
        hz_query = HZ_QUERY_MAP.get(hz_code, hz_code)

        resolution_source = str(pythia_meta_full.get("resolution_source") or "")
        hazard_label = str(pythia_meta_full.get("hazard_label") or hz_code)

        # Treat hazards mapped to CONFLICT (e.g., ACO, ACE, CU) as conflict, plus
        # anything in the legacy CONFLICT_HAZARD_CODES set.
        hz_is_conflict = bool(
            hz_query
            and (
                hz_query in CONFLICT_HAZARD_CODES
                or hz_query.startswith("CONFLICT")
            )
        )

        # Safety net: if this question is resolved using ACLED, treat as conflict.
        if not hz_is_conflict and "ACLED" in resolution_source.upper():
            hz_is_conflict = True
        window_start_date = _coerce_date(post.get("pythia_window_start_date"))
        window_end_date = _coerce_date(post.get("pythia_window_end_date"))
        month_labels = _build_month_labels(window_start_date, horizon_months=6)
        today_date = date.today()

        # ------------------ 1) Research step (LLM brief + sources appended) ---------
        t0 = time.time()
        research_text, research_meta = await run_research_async(
            run_id=run_id,
            question_id=str(question_id),
            title=title,
            description=description,
            criteria=criteria,
            qtype=qtype,
            options=options if qtype == "multiple_choice" else None,
            units=str(units) if units else None,
            slug=f"q{question_id}",
        )

        # Normalize meta to a dict so downstream `.get(...)` calls never crash.
        research_meta = _as_dict(research_meta)

        t_research_ms = _ms(t0)

        # Supplement research with PA history when we have iso3 + hazard_code
        pa_block = ""
        pa_meta: Dict[str, Any] = {}
        if pmeta.get("iso3") and pmeta.get("hazard_code"):
            try:
                pa_block, pa_meta = _load_pa_history_block(
                    pmeta["iso3"],
                    pmeta["hazard_code"],
                    months=36,
                    metric=metric_up,
                )
            except TypeError as exc:
                # Backwards compatibility for monkeypatched stubs in tests that do not
                # accept the metric kwarg.
                if "unexpected keyword argument" in str(exc) and "'metric'" in str(exc):
                    pa_block, pa_meta = _load_pa_history_block(
                        pmeta["iso3"],
                        pmeta["hazard_code"],
                        months=36,
                    )
                else:
                    raise
            if pa_block:
                research_text = f"{research_text}\n\n{pa_block}"

        history_rows = pa_meta.get("history_rows_detail") or []
        summary_text = pa_meta.get("summary_text") or ""
        error_code = pa_meta.get("error") or ""
        history_len = len(history_rows)
        snapshot_start = history_rows[-1]["ym"] if history_rows else ""
        snapshot_end = history_rows[0]["ym"] if history_rows else ""

        # Merge PA meta into research_meta under a clear prefix
        for key, value in pa_meta.items():
            if key.startswith("pa_history_"):
                research_meta[key] = value
            else:
                research_meta[f"pa_history_{key}"] = value

        # Research text notes for missing history
        if pmeta.get("iso3") and pmeta.get("hazard_code") and not history_rows:
            research_text = (
                f"{research_text}\n\n"
                "**Resolver history note:** No 36-month history was found in Resolver "
                f"for {pmeta.get('iso3','')}/{pmeta.get('hazard_code','')} ({metric_up}). Keep your prior heavily anchored to "
                "the historical base rate and make this explicit in your reasoning."
            )

        if os.getenv("PYTHIA_DEBUG_DB", "0") == "1" and pmeta.get("iso3") and pmeta.get("hazard_code"):
            print(
                f"[history_debug] iso3={pmeta.get('iso3')} hz={pmeta.get('hazard_code')} "
                f"metric={metric_up} error={error_code} n_rows={history_len}"
            )

        if qtype == "spd" and pmeta.get("iso3") and pmeta.get("hazard_code"):
            history_rows_out: list[Dict[str, Any]] = []
            if isinstance(history_rows, list):
                for item in history_rows:
                    if not isinstance(item, dict):
                        continue
                    history_rows_out.append(
                        {
                            "ym": str(item.get("ym") or ""),
                            "value": item.get("value"),
                            "source": str(item.get("source") or ""),
                        }
                    )

            snapshot_start = history_rows_out[-1]["ym"] if history_rows_out else snapshot_start
            snapshot_end = history_rows_out[0]["ym"] if history_rows_out else snapshot_end

            context_extra = {
                "history_len": len(history_rows_out),
                "summary_text": summary_text,
                "error": error_code,
            }

            try:
                con = connect(read_only=False)
                ensure_schema(con)
                con.execute(
                    "DELETE FROM question_context WHERE run_id = ? AND question_id = ?;",
                    [run_id, str(question_id)],
                )
                con.execute(
                    """
                    INSERT INTO question_context (
                        run_id,
                        question_id,
                        iso3,
                        hazard_code,
                        metric,
                        snapshot_start_month,
                        snapshot_end_month,
                        pa_history_json,
                        context_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    [
                        run_id,
                        str(question_id),
                        pmeta.get("iso3", "").upper(),
                        pmeta.get("hazard_code", "").upper(),
                        metric_up,
                        snapshot_start,
                        snapshot_end,
                        _json_dumps_for_db(history_rows_out, ensure_ascii=False),
                        _json_dumps_for_db(context_extra, ensure_ascii=False),
                    ],
                )
                con.close()
            except Exception as ctx_exc:  # noqa: BLE001
                print(
                    f"[warn] Failed to write question_context for Q{question_id}: {type(ctx_exc).__name__}: {ctx_exc}"
                )

        calib_advice_text = None
        if hz_code and metric_up:
            calib_advice_text = _load_calibration_advice_db(hz_code, metric_up)

        if calib_advice_text:
            research_text = (
                f"{research_text}\n\n## Calibration guidance for this hazard/metric\n{calib_advice_text}"
            )


        # ------------------ 2) Hazard-based "classification" (for GTMC1 gate) -----
        is_conflict_hazard = hz_is_conflict

        # We still publish classifier-like fields to keep CSV schema stable,
        # but they are now cheap deterministic values.
        class_primary = hz_code or ""
        class_secondary = ""
        is_strategic = is_conflict_hazard
        strategic_score = 1.0 if is_conflict_hazard else 0.0
        classifier_source = "hazard_code"
        classifier_rationale = (
            "hazard_code in CONFLICT_HAZARD_CODES"
            if is_conflict_hazard
            else "non-conflict hazard or missing hazard_code"
        )
        classifier_cost = 0.0

        # ------------------ 3) Optional GTMC1 (binary + conflict hazards only) ------
        # NOTE: for now, GTMC1 still only runs for binary questions. When we
        # move to SPD questions, we may relax the qtype guard.
        gtmc1_active = bool(is_conflict_hazard and qtype == "binary")
        actors_table: Optional[List[Dict[str, Any]]] = None
        gtmc1_signal: Dict[str, Any] = {}
        gtmc1_policy_sentence: str = ""
        t_gtmc1_ms = 0
    
        # Raw-dump debugging fields (only populated on failure / deactivation)
        gtmc1_raw_dump_path: str = ""
        gtmc1_raw_excerpt: str = ""
        gtmc1_raw_reason: str = ""

        if gtmc1_active:
            try:
                # Use the same async OpenAI client as other calls; model comes from config
                from .config import OPENAI_API_KEY
                from .providers import _get_or_client  # async OpenAI client

                client = _get_or_client()
                if client is None or not OPENAI_API_KEY:
                    gtmc1_active = False
                else:
                    prompt = f"""You are a research analyst preparing inputs for a Bruce Bueno de Mesquita-style
    game-theoretic bargaining model (BDM/Scholz). Identify actors and quantitative inputs on four dimensions.
    TITLE:
    {title}
    CONTEXT:
    {description}
    LATEST RESEARCH:
    {research_text}
    INSTRUCTIONS
    1) Define a POLICY CONTINUUM 0–100 for this question:
       0 = outcome least favorable to YES resolution; 100 = most favorable to YES resolution.
    2) Identify 3–8 ACTORS that materially influence the outcome (government, opposition, factions,
       mediators, veto players, firms, unions, external patrons).
    3) For each actor, provide:
       - "position" (0–100)
       - "capability" (0–100)
       - "salience" (0–100)
       - "risk_threshold" (0.00–0.10)
    4) OUTPUT STRICT JSON ONLY; NO commentary; schema:
    {{
      "policy_continuum": "Short one-sentence description of the 0–100 axis.",
      "actors": [
        {{"name":"Government","position":62,"capability":70,"salience":80,"risk_threshold":0.04}},
        {{"name":"Opposition","position":35,"capability":60,"salience":85,"risk_threshold":0.05}}
      ]
    }}
    Constraints: All numbers within ranges; 3–8 total actors; valid JSON.
    """
                    t_gt0 = time.time()
                    profile_models = {}
                    if _get_llm_profile_models is not None:
                        try:
                            profile_models = _get_llm_profile_models()
                        except Exception:
                            profile_models = {}
                    default_gtmc1_model = profile_models.get("openai", "gpt-5.1-pro")
                    async with get_llm_semaphore():
                        resp = await client.chat.completions.create(
                            model=os.getenv("GTMC1_MODEL_ID", default_gtmc1_model),
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                        )
                    text = (resp.choices[0].message.content or "").strip()
                    raw_text_for_debug = text  # keep exactly what the LLM sent
                    try:
                        data = json.loads(re.sub(r"^```json\s*|\s*```$", "", text, flags=re.S))
                    except Exception:
                        data = {}
                        gtmc1_active = False
                        gtmc1_raw_reason = "json_parse_error"
                        # Dump raw if requested; otherwise keep a short excerpt for the human log
                        gtmc1_raw_dump_path = _maybe_dump_raw_gtmc1(raw_text_for_debug, run_id=run_id, question_id=question_id) or ""
                        if not gtmc1_raw_dump_path:
                            # Use the same limit used elsewhere for model raw content
                            MAX_RAW = int(os.getenv("HUMAN_LOG_MODEL_RAW_MAX_CHARS", "5000"))
                            gtmc1_raw_excerpt = raw_text_for_debug[:MAX_RAW]
    
                    actors = data.get("actors") or []
                    gtmc1_policy_sentence = str(data.get("policy_continuum") or "").strip()
                    cleaned: List[Dict[str, Any]] = []
                    for a in actors:
                        try:
                            nm = str(a.get("name") or "").strip()
                            pos = float(a.get("position")); cap = float(a.get("capability"))
                            sal = float(a.get("salience")); thr = float(a.get("risk_threshold"))
                            if not nm: continue
                            if not (0.0 <= pos <= 100.0): continue
                            if not (0.0 <= cap <= 100.0): continue
                            if not (0.0 <= sal <= 100.0): continue
                            if not (0.0 <= thr <= 0.10): continue
                            cleaned.append({
                                "name": nm, "position": pos, "capability": cap,
                                "salience": sal, "risk_threshold": thr
                            })
                        except Exception:
                            continue
                    if len(cleaned) >= 3:
                        actors_table = cleaned
                        gtmc1_signal, _df_like = await asyncio.to_thread(
                            GTMC1.run_monte_carlo_from_actor_table,
                            actor_rows=actors_table,
                            num_runs=60,
                            log_dir="gtmc_logs",
                            run_slug=f"q{question_id}",
                        )
                    else:
                        gtmc1_active = False
                        gtmc1_raw_reason = "actors_lt_3"
                        gtmc1_raw_dump_path = _maybe_dump_raw_gtmc1(raw_text_for_debug, run_id=run_id, question_id=question_id) or ""
                        if not gtmc1_raw_dump_path:
                            MAX_RAW = int(os.getenv("HUMAN_LOG_MODEL_RAW_MAX_CHARS", "5000"))
                            gtmc1_raw_excerpt = raw_text_for_debug[:MAX_RAW]
            except Exception:
                gtmc1_active = False
                t_gtmc1_ms = 0

        # If GTMC1 succeeded, append a short summary to the research bundle
        if gtmc1_active and gtmc1_signal:
            try:
                prob_yes = gtmc1_signal.get("gtmc1_prob") \
                    or gtmc1_signal.get("prob_yes") \
                    or gtmc1_signal.get("exceedance_ge_50")
                coal_rate = gtmc1_signal.get("coalition_rate")
                disp = gtmc1_signal.get("dispersion")

                lines = [
                    "## GTMC1 scenario analysis (bargaining model)",
                    "",
                    f"- Policy continuum: {gtmc1_policy_sentence or '(not specified)'}",
                ]
                if prob_yes is not None:
                    lines.append(f"- GTMC1-estimated probability of YES-aligned outcome: {float(prob_yes):.2f}")
                if coal_rate is not None:
                    lines.append(f"- Coalition formation rate in simulations: {float(coal_rate):.2f}")
                if disp is not None:
                    lines.append(f"- Dispersion of actor positions (0–1): {float(disp):.2f}")

                gtmc1_block = "\n".join(lines)
                research_text = f"{research_text}\n\n{gtmc1_block}"
            except Exception:
                # On any formatting error, keep GTMC1 out of the research bundle but don't crash.
                pass
    
        # ------------------ 4) Build main prompts (WITH research) -------------------
        if qtype == "binary":
            main_prompt = build_binary_prompt(title, description, research_text, criteria)
        elif qtype == "multiple_choice":
            main_prompt = build_mcq_prompt(title, options, description, research_text, criteria)
        elif qtype == "spd":
            if metric_up == "FATALITIES" and hz_is_conflict:
                main_prompt = build_spd_prompt_fatalities(
                    question_title=title,
                    iso3=pmeta.get("iso3", ""),
                    hazard_code=hz_code,
                    hazard_label=hazard_label,
                    metric=metric_up,
                    background=description,
                    research_text=research_text,
                    resolution_source=resolution_source,
                    window_start_date=window_start_date,
                    window_end_date=window_end_date,
                    month_labels=month_labels,
                    today=today_date,
                    criteria=criteria,
                )
            else:
                main_prompt = build_spd_prompt_pa(
                    question_title=title,
                    iso3=pmeta.get("iso3", ""),
                    hazard_code=hz_code,
                    hazard_label=hazard_label,
                    metric=metric_up,
                    background=description,
                    research_text=research_text,
                    resolution_source=resolution_source,
                    window_start_date=window_start_date,
                    window_end_date=window_end_date,
                    month_labels=month_labels,
                    today=today_date,
                    criteria=criteria,
                )
        else:
            main_prompt = build_numeric_prompt(title, str(units or ""), description, research_text, criteria)
    
        # ------------------ 5) Ensemble calls (WITH research) -----------------------
        ensemble_specs = SPD_ENSEMBLE if qtype == "spd" else DEFAULT_ENSEMBLE
        t0 = time.time()
        if qtype == "binary":
            ens_res = await run_ensemble_binary(main_prompt, ensemble_specs)
        elif qtype == "multiple_choice":
            ens_res = await run_ensemble_mcq(main_prompt, n_options, ensemble_specs)
        elif qtype == "spd":
            ens_res = await run_ensemble_spd(
                main_prompt,
                ensemble_specs,
                run_id=run_id,
                question_id=str(question_id),
                hs_run_id=hs_run_id,
            )
        else:
            ens_res = await run_ensemble_numeric(main_prompt, ensemble_specs)
        t_ensemble_ms = _ms(t0)
    
        # ------------------ 6) Choose calibration weights & aggregate ---------------
        calib_weights_map: Dict[str, float] = {}
        weights_profile = "uniform"

        if qtype == "spd" and hz_code and metric_up:
            db_weights = _load_calibration_weights_db(hz_code, metric_up)
            if db_weights:
                calib_weights_map = db_weights
                weights_profile = f"db:{hz_code}:{metric_up}"

        if not calib_weights_map:
            calib_weights_map, weights_profile = _choose_weights_for_question(
                _load_calibration_weights_file(),
                class_primary=class_primary,
                qtype=qtype,
                ensemble_specs=ensemble_specs,
            )
    
        # MAIN aggregation (with optional GTMC1 for binary)
        if qtype == "binary":
            final_main, bmc_summary = aggregate_binary(ens_res, gtmc1_signal if gtmc1_active else None, calib_weights_map)
        elif qtype == "multiple_choice":
            vec_main, bmc_summary = aggregate_mcq(ens_res, n_options, calib_weights_map)
            final_main = {options[i]: vec_main[i] for i in range(n_options)} if n_options else {}
        elif qtype == "spd":
            bucket_labels = SPD_CLASS_BINS_PA
            if metric_up == "FATALITIES":
                bucket_labels = SPD_CLASS_BINS_FATALITIES

            metric_up_local = (metric_up or "").upper()
            if metric_up_local == "PA":
                default_centroids = SPD_BUCKET_CENTROIDS_PA
                default_source_label = "default_pa"
            elif metric_up_local == "FATALITIES":
                default_centroids = SPD_BUCKET_CENTROIDS_FATALITIES
                default_source_label = "default_fatalities"
            else:
                default_centroids = SPD_BUCKET_CENTROIDS_DEFAULT
                default_source_label = "default_generic"

            bucket_centroids_db = None
            centroid_source = default_source_label
            if pmeta.get("hazard_code") and pmeta.get("metric"):
                bucket_centroids_db = _load_bucket_centroids_db(
                    hazard_code=pmeta["hazard_code"],
                    metric=pmeta["metric"],
                    class_bins=bucket_labels,
                )

            if bucket_centroids_db is not None:
                bucket_centroids = bucket_centroids_db
                centroid_source = "db"
            else:
                bucket_centroids = default_centroids

            spd_bucket_labels_used = list(bucket_labels)
            spd_bucket_centroids_used = list(bucket_centroids)
            spd_centroid_source = centroid_source

            try:
                # Normal SPD aggregation path
                spd_main, ev_dict, bmc_summary = aggregate_spd(
                    ens_res,
                    weights=calib_weights_map,
                    bucket_centroids=bucket_centroids,
                )
                from .ensemble import _normalize_spd_keys  # local import to avoid cycles

                spd_main = _normalize_spd_keys(spd_main, n_months=6, n_buckets=len(bucket_labels))
                final_main = spd_main
                ev_main = ev_dict
            except KeyError as exc:
                # Contain schema/parse bugs like KeyError('\n     "month_1"') and fall back.
                try:
                    raw_keys = set()
                    ens_obj = locals().get("ens_res")
                    # We don't need the exact class; just look for .members
                    if hasattr(ens_obj, "members"):
                        for _m in ens_obj.members:
                            if isinstance(_m, MemberOutput) and isinstance(_m.parsed, dict):
                                raw_keys.update(str(k) for k in _m.parsed.keys())
                except Exception:
                    raw_keys = set()

                offending = str(exc)
                print(
                    f"[spd] KeyError during SPD aggregation for question_id={question_id!r}: {offending!r}. "
                    f"raw_spd_keys={sorted(raw_keys)!r}. Falling back to uniform SPD across 6 months."
                )

                # Build a conservative uniform SPD: all buckets equal for all 6 months.
                n_buckets = len(bucket_labels)
                uniform_vec = [1.0 / float(n_buckets)] * n_buckets
                spd_main = {f"month_{i}": list(uniform_vec) for i in range(1, 7)}

                # No meaningful expected values if aggregation failed; keep it empty.
                ev_main = {}

                # Tag BMC summary so we can see this in CSV / logs.
                bmc_summary = {
                    "method": "spd_keyerror_fallback",
                    "error": offending,
                }

                final_main = spd_main
        else:
            quantiles_main, bmc_summary = aggregate_numeric(ens_res, calib_weights_map)
            final_main = dict(quantiles_main)

        if summary is not None and qtype == "spd" and isinstance(final_main, dict):
            summary.month_count = len([k for k in final_main.keys() if str(k).startswith("month_")])
            first_month_vec = next(iter(final_main.values()), [])
            if isinstance(first_month_vec, list):
                summary.buckets_per_month = len(first_month_vec)
            if isinstance(ev_main, dict) and ev_main:
                ev_vals = [float(v) for v in ev_main.values() if v is not None]
                if ev_vals:
                    summary.ev_min = min(ev_vals)
                    summary.ev_max = max(ev_vals)
            if isinstance(ens_res, EnsembleResult):
                for m in ens_res.members:
                    summary.models[m.name] = {
                        "ok": 1.0 if m.ok else 0.0,
                        "elapsed_ms": float(getattr(m, "elapsed_ms", 0) or 0),
                        "cost_usd": float(getattr(m, "cost_usd", 0.0) or 0.0),
                        "total_tokens": float(getattr(m, "total_tokens", 0) or 0),
                    }

        # If this is a Pythia SPD question, write ensemble SPD into DuckDB
        if qtype == "spd" and isinstance(final_main, dict):
            # Heuristic: presence of Pythia metadata marks Pythia mode
            if "pythia_iso3" in post or "pythia_hazard_code" in post:
                try:
                    _write_spd_ensemble_to_db(
                        run_id=run_id,
                        question_id=str(question_id),
                        iso3=pmeta.get("iso3", ""),
                        hazard_code=pmeta.get("hazard_code", ""),
                        metric=pmeta.get("metric", ""),
                        spd_main=final_main,
                        ev_main=ev_main,
                        weights_profile=weights_profile,
                    )
                except Exception as exc:
                    print(f"[warn] Failed to write SPD ensemble to DB for question {question_id}: {exc}")
                try:
                    _write_spd_raw_to_db(
                        run_id=run_id,
                        question_id=str(question_id),
                        iso3=pmeta.get("iso3", ""),
                        hazard_code=pmeta.get("hazard_code", ""),
                        metric=pmeta.get("metric", ""),
                        ens_res=ens_res,
                    )
                except Exception as exc:
                    print(f"[warn] Failed to write SPD RAW to DB for question {question_id}: {exc}")

                if summary is not None:
                    ens_n, raw_n = _count_spd_rows(run_id, str(question_id))
                    summary.ensemble_rows = ens_n
                    summary.raw_rows = raw_n

        bmc_summary = _as_dict(bmc_summary)

        # ------------------ 7) Diagnostic variants (WITH research) ------------------
        # NOTE: aggregate_binary now ignores gtmc1_signal; these variants are retained
        # only for schema continuity and weight-comparison diagnostics.
        if qtype == "binary":
            v_nogtmc1, _ = aggregate_binary(ens_res, None, calib_weights_map)
            v_uniform, _ = aggregate_binary(
                ens_res,
                gtmc1_signal if gtmc1_active else None,
                {m.name: 1.0 for m in ensemble_specs},
            )
            v_simple = _simple_average_binary(ens_res.members)
        elif qtype == "multiple_choice":
            v_nogtmc1_vec, _ = aggregate_mcq(ens_res, n_options, calib_weights_map)
            v_nogtmc1 = {options[i]: v_nogtmc1_vec[i] for i in range(n_options)} if n_options else {}
            v_uniform_vec, _ = aggregate_mcq(ens_res, n_options, {m.name: 1.0 for m in ensemble_specs})
            v_uniform = {options[i]: v_uniform_vec[i] for i in range(n_options)} if n_options else {}
            v_simple_vec = _simple_average_mcq(ens_res.members, n_options)
            v_simple = {options[i]: v_simple_vec[i] for i in range(n_options)} if (n_options and v_simple_vec) else {}
        elif qtype == "spd":
            v_nogtmc1, ev_nogtmc1, _ = aggregate_spd(
                ens_res,
                weights=calib_weights_map,
                bucket_centroids=bucket_centroids,
            )
            v_uniform, ev_uniform, _ = aggregate_spd(
                ens_res,
                weights={m.name: 1.0 for m in ensemble_specs},
                bucket_centroids=bucket_centroids,
            )
            v_simple = v_nogtmc1
        else:
            v_nogtmc1, _ = aggregate_numeric(ens_res, calib_weights_map)
            v_uniform, _ = aggregate_numeric(ens_res, {m.name: 1.0 for m in ensemble_specs})
            v_simple = _simple_average_numeric(ens_res.members) or {}
    
        # ------------------ 8) Ablation pass: NO RESEARCH ---------------------------
        if qtype == "spd":
            # Skip ablation for SPD to avoid doubling LLM cost
            ab_main = final_main
            ab_uniform = final_main
            ab_simple = final_main
        elif qtype == "binary":
            ab_prompt = build_binary_prompt(title, description, "", criteria)
            ens_res_ab = await run_ensemble_binary(ab_prompt, ensemble_specs)
            ab_main, _ = aggregate_binary(ens_res_ab, None, calib_weights_map)
            ab_uniform, _ = aggregate_binary(ens_res_ab, None, {m.name: 1.0 for m in ensemble_specs})
            ab_simple = _simple_average_binary(ens_res_ab.members)
        elif qtype == "multiple_choice":
            ab_prompt = build_mcq_prompt(title, options, description, "", criteria)
            ens_res_ab = await run_ensemble_mcq(ab_prompt, n_options, ensemble_specs)
            ab_vec, _ = aggregate_mcq(ens_res_ab, n_options, calib_weights_map)
            ab_main = {options[i]: ab_vec[i] for i in range(n_options)} if n_options else {}
            ab_uniform_vec, _ = aggregate_mcq(ens_res_ab, n_options, {m.name: 1.0 for m in ensemble_specs})
            ab_uniform = {options[i]: ab_uniform_vec[i] for i in range(n_options)} if n_options else {}
            ab_simple_vec = _simple_average_mcq(ens_res_ab.members, n_options)
            ab_simple = {options[i]: ab_simple_vec[i] for i in range(n_options)} if (n_options and ab_simple_vec) else {}
        else:
            ab_prompt = build_numeric_prompt(title, str(units or ""), description, "", criteria)
            ens_res_ab = await run_ensemble_numeric(ab_prompt, ensemble_specs)
            ab_main, _ = aggregate_numeric(ens_res_ab, calib_weights_map)
            ab_uniform, _ = aggregate_numeric(ens_res_ab, {m.name: 1.0 for m in ensemble_specs})
            ab_simple = _simple_average_numeric(ens_res_ab.members) or {}
    
        # ------------------ 9) Build ONE wide CSV row and write it ------------------
        ensure_unified_csv()
    
        row: Dict[str, Any] = {
            # Run metadata
            "run_id": run_id,
            "run_time_iso": ist_iso(),
            "purpose": purpose,
            "git_sha": os.getenv("GIT_SHA", ""),
            "config_profile": "default",
            "weights_profile": "class_calibration",
            "llm_models_json": [
                {"name": ms.name, "provider": ms.provider, "model_id": ms.model_id, "weight": ms.weight}
                for ms in ensemble_specs
            ],
    
            # Question metadata
            "question_id": str(question_id),
            "question_url": url,
            "question_title": title,
            "question_type": qtype,
            "tournament_id": tournament_id if isinstance(tournament_id, str) else str(tournament_id),
            "created_time_iso": post.get("creation_time") or q.get("creation_time") or "",
            "closes_time_iso": post.get("close_time") or q.get("close_time") or "",
            "resolves_time_iso": post.get("scheduled_resolve_time") or q.get("scheduled_resolve_time") or "",
    
            # Classification
            "class_primary": class_primary,
            "class_secondary": class_secondary or "",
            "is_strategic": str(is_strategic),
            "strategic_score": f"{strategic_score:.3f}",
            "classifier_source": classifier_source,
            "classifier_rationale": classifier_rationale,
    
            # Research
            "research_llm": research_meta.get("research_llm", ""),
            "research_source": research_meta.get("research_source", ""),
            "research_query": research_meta.get("research_query", ""),
            "research_n_raw": str(research_meta.get("research_n_raw", "")),
            "research_n_kept": str(research_meta.get("research_n_kept", "")),
            "research_cached": research_meta.get("research_cached", ""),
            "research_error": research_meta.get("research_error", ""),
    
    
            # Options/values
            "n_options": str(n_options if qtype == "multiple_choice" else 0),
            "options_json": options if qtype == "multiple_choice" else "",
            "discrete_values_json": discrete_values if (qtype in ("numeric", "discrete") and discrete_values) else "",
        }

        if qtype == "spd":
            if spd_bucket_labels_used is not None:
                row["spd_bucket_labels"] = spd_bucket_labels_used
            if spd_bucket_centroids_used is not None:
                row["spd_bucket_centroids"] = spd_bucket_centroids_used
            if spd_centroid_source:
                row["spd_centroid_source"] = spd_centroid_source

        row["seen_guard_triggered"] = (
            "1"
            if seen_guard_enabled and bool(seen_guard_lock_acquired)
            else ("0" if seen_guard_enabled else "")
        )
    
        # Per-model outputs
        for i, ms in enumerate(ensemble_specs):
            mo: Optional[MemberOutput] = None
            if isinstance(ens_res, EnsembleResult) and i < len(ens_res.members):
                mo = ens_res.members[i]
    
            ok = bool(mo and mo.ok)
            row[f"model_ok__{ms.name}"] = "1" if ok else "0"
            row[f"model_time_ms__{ms.name}"] = str(getattr(mo, "elapsed_ms", 0) or "")
    
            if ok and mo is not None:
                if qtype == "binary" and isinstance(mo.parsed, (float, int)):
                    row[f"binary_prob__{ms.name}"] = f"{_clip01(float(mo.parsed)):.6f}"
                elif qtype == "multiple_choice" and isinstance(mo.parsed, list):
                    row[f"mcq_json__{ms.name}"] = mo.parsed
                elif qtype == "spd" and isinstance(mo.parsed, dict):
                    row[f"spd_json__{ms.name}"] = mo.parsed
                elif qtype in ("numeric", "discrete") and isinstance(mo.parsed, dict):
                    p10 = _safe_float(mo.parsed.get("P10"))
                    p50 = _safe_float(mo.parsed.get("P50"))
                    p90 = _safe_float(mo.parsed.get("P90"))
                    if p10 is not None: row[f"numeric_p10__{ms.name}"] = f"{p10:.6f}"
                    if p50 is not None: row[f"numeric_p50__{ms.name}"] = f"{p50:.6f}"
                    if p90 is not None: row[f"numeric_p90__{ms.name}"] = f"{p90:.6f}"
    
            row[f"cost_usd__{ms.name}"] = f"{getattr(mo,'cost_usd',0.0):.6f}" if mo else ""
    
        # Ensemble (main)
        if qtype == "binary" and isinstance(final_main, float):
            row["binary_prob__ensemble"] = f"{_clip01(final_main):.6f}"
        elif qtype == "multiple_choice" and isinstance(final_main, dict):
            row["mcq_json__ensemble"] = final_main
            for j in range(min(15, n_options)):
                row[f"mcq_{j+1}__ensemble"] = f"{_clip01(float(final_main.get(options[j], 0.0))):.6f}"
        elif qtype == "spd" and isinstance(final_main, dict):
            row["spd_json__ensemble"] = final_main
            if isinstance(ev_main, dict):
                row["spd_ev_json__ensemble"] = ev_main
        elif qtype in ("numeric", "discrete") and isinstance(final_main, dict):
            for k in ("P10", "P50", "P90"):
                if k in final_main:
                    row[f"numeric_{k.lower()}__ensemble"] = f"{float(final_main[k]):.6f}"
    
        # Variants (WITH research)
        def _fill_variant(tag: str, val: Any):
            if qtype == "binary" and isinstance(val, float):
                row[f"binary_prob__ensemble_{tag}"] = f"{_clip01(val):.6f}"
            elif qtype == "multiple_choice" and isinstance(val, dict):
                row[f"mcq_json__ensemble_{tag}"] = val
            elif qtype == "spd" and isinstance(val, dict):
                row[f"spd_json__ensemble_{tag}"] = val
            elif qtype in ("numeric", "discrete") and isinstance(val, dict):
                for k in ("P10", "P50", "P90"):
                    if k in val:
                        row[f"numeric_{k.lower()}__ensemble_{tag}"] = f"{float(val[k]):.6f}"
    
        _fill_variant("no_gtmc1", v_nogtmc1)
        _fill_variant("uniform_weights", v_uniform)
        if qtype == "binary":
            _fill_variant("no_bmc_no_gtmc1", v_simple)  # float for binary
        else:
            _fill_variant("no_bmc_no_gtmc1", v_simple if isinstance(v_simple, dict) else v_simple)
    
        # Ablation (NO research)
        row["ablation_no_research"] = "1"
        if qtype == "binary" and isinstance(ab_main, float):
            row["binary_prob__ensemble_no_research"] = f"{_clip01(ab_main):.6f}"
        elif qtype == "multiple_choice" and isinstance(ab_main, dict):
            row["mcq_json__ensemble_no_research"] = ab_main
            for j in range(min(15, n_options)):
                row[f"mcq_{j+1}__ensemble_no_research"] = f"{_clip01(float(ab_main.get(options[j], 0.0))):.6f}"
        elif qtype == "spd" and isinstance(ab_main, dict):
            row["spd_json__ensemble_no_research"] = ab_main
        elif qtype in ("numeric", "discrete") and isinstance(ab_main, dict):
            for k in ("P10", "P50", "P90"):
                if k in ab_main:
                    row[f"numeric_{k.lower()}__ensemble_no_research"] = f"{float(ab_main[k]):.6f}"
    
        def _fill_ablation_variant(tag: str, val: Any):
            if qtype == "binary" and isinstance(val, float):
                row[f"binary_prob__ensemble_no_research_{tag}"] = f"{_clip01(val):.6f}"
            elif qtype == "multiple_choice" and isinstance(val, dict):
                row[f"mcq_json__ensemble_no_research_{tag}"] = val
            elif qtype == "spd" and isinstance(val, dict):
                row[f"spd_json__ensemble_no_research_{tag}"] = val
            elif qtype in ("numeric", "discrete") and isinstance(val, dict):
                for k in ("P10", "P50", "P90"):
                    if k in val:
                        row[f"numeric_{k.lower()}__ensemble_no_research_{tag}"] = f"{float(val[k]):.6f}"
    
        _fill_ablation_variant("no_gtmc1", ab_main)
        _fill_ablation_variant("uniform_weights", ab_uniform)
        _fill_ablation_variant("no_bmc_no_gtmc1", ab_simple if isinstance(ab_simple, dict) else ({"P50": ab_simple} if isinstance(ab_simple, float) else ab_simple))

        # Diagnostics, timings, weights used
        gtmc1_signal = _as_dict(gtmc1_signal)
        row.update({
            "gtmc1_active": "1" if gtmc1_active else "0",
            "actors_cached": "0",
            "gtmc1_actor_count": str(len(actors_table) if actors_table else 0),
            "gtmc1_coalition_rate": (gtmc1_signal.get("coalition_rate") if gtmc1_signal else ""),
            "gtmc1_exceedance_ge_50": (gtmc1_signal.get("exceedance_ge_50") if gtmc1_signal else ""),
            "gtmc1_dispersion": (gtmc1_signal.get("dispersion") if gtmc1_signal else ""),
            "gtmc1_median_rounds": (gtmc1_signal.get("median_rounds") if gtmc1_signal else ""),
            "gtmc1_num_runs": (gtmc1_signal.get("num_runs") if gtmc1_signal else ""),
            "gtmc1_policy_sentence": gtmc1_policy_sentence or "",
            "gtmc1_signal_json": gtmc1_signal or "",
    
            "bmc_summary_json": "",
    
            "cdf_steps_clamped": "",
            "cdf_upper_open_adjusted": "",
            "prob_sum_renormalized": "",

            "t_research_ms": str(t_research_ms),
            "t_ensemble_ms": str(t_ensemble_ms),
            "t_gtmc1_ms": str(t_gtmc1_ms),
            "t_total_ms": str(_ms(t_start_total)),

            "resolved": "",
            "resolved_time_iso": "",
            "resolved_outcome_label": "",
            "resolved_value": "",
            "score_brier": "",
            "score_log": "",
            "score_crps": "",
    
            "score_brier__no_research": "",
            "score_log__no_research": "",
            "score_crps__no_research": "",
    
            "weights_profile_applied": weights_profile,
            "weights_per_model_json": calib_weights_map,
            "dedupe_hash": "",
            "seen_guard_triggered": "",
        })
    
        # Human-readable markdown log
        MAX_RAW_CHARS = int(os.getenv("HUMAN_LOG_MODEL_RAW_MAX_CHARS","5000"))
        RESEARCH_MAX = int(os.getenv("HUMAN_LOG_RESEARCH_MAX_CHARS","20000"))
        md = []
        md.append(f"# {title} (QID: {question_id})")
        md.append(f"- Type: {qtype}")
        md.append(f"- URL: {url}")
        md.append(f"- Classifier: {class_primary} | strategic={is_strategic} (score={strategic_score:.2f})")
        md.append("### SeenGuard")
        lock_status = "n/a"
        if seen_guard_enabled:
            lock_status = "acquired" if seen_guard_lock_acquired else "not_acquired"
        md.append(f"- enabled={seen_guard_enabled} | lock_status={lock_status}")
        if seen_guard_run_report:
            before = seen_guard_run_report.get("before")
            skipped = seen_guard_run_report.get("skipped")
            after = seen_guard_run_report.get("after")
            md.append(f"- run_filter: before={before} | skipped={skipped} | after={after}")
            if seen_guard_run_report.get("error"):
                md.append(f"- filter_error={seen_guard_run_report['error']}")
        debug_note = "lock disabled"
        if seen_guard_enabled:
            debug_note = "lock acquired" if seen_guard_lock_acquired else "lock fallback"
        if seen_guard_lock_error:
            debug_note += f" | error={seen_guard_lock_error}"
        md.append(f"- debug_note={debug_note}")
    
        md.append("## Research (summary)")
        md.append((research_text or "").strip()[:RESEARCH_MAX])
        # Research (debug)
        try:
            _r_src   = research_meta.get("research_source","")
            _r_llm   = research_meta.get("research_llm","")
            _r_q     = research_meta.get("research_query","")
            _r_raw   = research_meta.get("research_n_raw","")
            _r_kept  = research_meta.get("research_n_kept","")
            _r_cache = research_meta.get("research_cached","")
            _r_err   = research_meta.get("research_error","")
            md.append("### Research (debug)")
            _r_cost  = research_meta.get("research_cost_usd", 0.0)
            md.append(
                f"- source={_r_src} | llm={_r_llm} | cached={_r_cache} | "
                f"n_raw={_r_raw} | n_kept={_r_kept} | cost=${float(_r_cost):.6f}"
            )
    
            if _r_q:
                md.append(f"- query: {_r_q}")
            if _r_err:
                md.append(f"- error: {_r_err}")
        except Exception:
            pass
    
        # --- GTMC1 (debug) --------------------------------------------------------
        try:
            md.append("### GTMC1 (debug)")
            # Basic flags
            md.append(f"- strategic_class={is_strategic} | strategic_score={strategic_score:.2f} | source={classifier_source}")
            md.append(f"- gtmc1_active={gtmc1_active} | qtype={qtype} | t_ms={t_gtmc1_ms}")
    
            # Actor extraction outcome
            _n_actors = len(actors_table) if actors_table else 0
            md.append(f"- actors_parsed={_n_actors}")
    
            # Key Monte Carlo outputs (if any)
            _sig = gtmc1_signal or {}
            _ex = _sig.get("exceedance_ge_50")
            _coal = _sig.get("coalition_rate")
            _med = _sig.get("median_of_final_medians")
            _disp = _sig.get("dispersion")
    
            md.append(f"- exceedance_ge_50={_ex} | coalition_rate={_coal} | median={_med} | dispersion={_disp}")
            _runs_csv = _sig.get("runs_csv")
            if _runs_csv:
                md.append(f"- runs_csv={_runs_csv}")
            _meta_json = _sig.get("meta_json")
            if _meta_json:
                md.append(f"- meta_json={_meta_json}")
    
            # If GTMC1 was expected but didn’t apply, say why (best effort).
            if is_conflict_hazard and qtype == "binary" and not gtmc1_active:
                md.append("- note=GTMC1 gate opened (conflict hazard) but deactivated later (client/JSON/actors<3).")
            # If we captured raw (on failure), surface it.
            if gtmc1_raw_reason:
                md.append(f"- raw_reason={gtmc1_raw_reason}")
            if gtmc1_raw_dump_path or gtmc1_raw_excerpt:
                md.append("### GTMC1 (raw)")
                if gtmc1_raw_dump_path:
                    md.append(f"- raw_file={gtmc1_raw_dump_path}")
                if gtmc1_raw_excerpt:
                    md.append("```json")
                    md.append(gtmc1_raw_excerpt)
                    md.append("```")
        except Exception as _gtmc1_dbg_ex:
            md.append(f"- gtmc1_debug_error={type(_gtmc1_dbg_ex).__name__}: {str(_gtmc1_dbg_ex)[:200]}")
        # --------------------------------------------------------------------------
    
        # --- GTMC1 (actors used) ---------------------------------------------------
        # Show the actual table we fed into GTMC1 so you can audit inputs later.
        if gtmc1_active and actors_table:
            try:
                md.append("### GTMC1 (actors used)")
                md.append("| Actor | Position | Capability | Salience | Risk thresh |")
                md.append("|---|---:|---:|---:|---:|")
                for a in actors_table:
                    md.append(
                        f"| {a['name']} | {float(a['position']):.0f} | "
                        f"{float(a['capability']):.0f} | {float(a['salience']):.0f} | "
                        f"{float(a['risk_threshold']):.3f} |"
                    )
            except Exception as _gtmc1_tbl_ex:
                md.append(f"- actors_table_render_error={type(_gtmc1_tbl_ex).__name__}: {str(_gtmc1_tbl_ex)[:160]}")
    
        # --- Ensemble outputs (compact) --------------------------------------------
        try:
            md.append("### Ensemble (model outputs)")
            for m in ens_res.members:
                if not isinstance(m, MemberOutput):
                    continue
                _line = f"- {m.name}: ok={m.ok} t_ms={getattr(m,'elapsed_ms',0)}"
                if qtype == "binary" and m.ok and isinstance(m.parsed, (float, int)):
                    _line += f" p={_clip01(float(m.parsed)):.4f}"
                elif qtype == "multiple_choice" and m.ok and isinstance(m.parsed, list):
                    # just show top-3
                    try:
                        vec = [float(x) for x in m.parsed]
                        idxs = np.argsort(vec)[::-1][:3]
                        _line += " top3=" + ", ".join([f"{options[i]}:{_clip01(vec[i]):.3f}" for i in idxs])
                    except Exception:
                        pass
                elif qtype == "spd" and m.ok and isinstance(m.parsed, dict):
                    try:
                        m1 = m.parsed.get("month_1")
                        if isinstance(m1, list) and len(m1) >= 3:
                            _line += " month_1=" + ", ".join([f"{float(x):.2f}" for x in m1[:3]]) + " …"
                    except Exception:
                        pass
                elif qtype in ("numeric", "discrete") and m.ok and isinstance(m.parsed, dict):
                    p10 = _safe_float(m.parsed.get("P10"))
                    p50 = _safe_float(m.parsed.get("P50"))
                    p90 = _safe_float(m.parsed.get("P90"))
                    if p10 is not None and p90 is not None:
                        if p50 is None:
                            p50 = 0.5 * (p10 + p90)
                        _line += f" P10={p10:.3f}, P50={p50:.3f}, P90={p90:.3f}"
                md.append(_line)
        except Exception as _ens_dbg_ex:
            md.append(f"- ensemble_debug_error={type(_ens_dbg_ex).__name__}: {str(_ens_dbg_ex)[:200]}")
    
        # --- Per-model details: reasoning + usage/cost --------------------------------
        try:
            MODEL_RAW_MAX = int(os.getenv("HUMAN_LOG_MODEL_RAW_MAX_CHARS", "5000"))
            md.append("")
            md.append("### Per-model (raw + usage/cost)")
    
            for m in ens_res.members:
                if not isinstance(m, MemberOutput):
                    continue
                md.append(f"#### {m.name}")
                md.append(
                    f"- ok={m.ok} | t_ms={getattr(m,'elapsed_ms',0)} | "
                    f"tokens: prompt={getattr(m,'prompt_tokens',0)}, "
                    f"completion={getattr(m,'completion_tokens',0)}, "
                    f"total={getattr(m,'total_tokens',0)} | "
                    f"cost=${float(getattr(m,'cost_usd',0.0)):.6f}"
                )
                if getattr(m, "error", None):
                    md.append(f"- error={str(m.error)[:240]}")
                if getattr(m, "raw_text", None):
                    raw = (m.raw_text or "").strip()
                    if raw:
                        md.append("```md")
                        md.append(raw[:MODEL_RAW_MAX])
                        md.append("```")
        except Exception as _pm_ex:
            md.append(f"- per_model_dump_error={type(_pm_ex).__name__}: {str(_pm_ex)[:200]}")
    
        # --- Aggregation summary (BMC) ---------------------------------------------
        try:
            md.append("### Aggregation (BMC)")
            # Make the BMC summary JSON-safe and also visible in the human log
            bmc_json = {}
            if isinstance(bmc_summary, dict):
                # strip large arrays already removed; copy select keys if present
                for k in ("mean", "var", "std", "n_evidence", "p10", "p50", "p90"):
                    if k in bmc_summary:
                        bmc_json[k] = bmc_summary[k]
            # Put a human line:
            if qtype == "binary" and isinstance(final_main, float):
                md.append(f"- final_probability={_clip01(final_main):.4f}")
            elif qtype == "multiple_choice" and isinstance(final_main, dict):
                # show top-3
                items = sorted(final_main.items(), key=lambda kv: kv[1], reverse=True)[:3]
                md.append("- final_top3=" + ", ".join([f"{k}:{_clip01(float(v)):.3f}" for k, v in items]))
            elif qtype == "spd" and isinstance(final_main, dict):
                md.append("### SPD Forecast (5 buckets × 6 months)")
                for m_idx in range(1, 7):
                    key = f"month_{m_idx}"
                    probs = final_main.get(key)
                    if not isinstance(probs, list) or len(probs) != 5:
                        continue
                    line = " | ".join(f"{float(p):.2f}" for p in probs)
                    md.append(f"- {key}: {line}")
                if isinstance(ev_main, dict):
                    md.append("### Expected people affected (per month)")
                    for key, val in sorted(ev_main.items()):
                        md.append(f"- {key}: {float(val):,.0f}")
            elif qtype in ("numeric", "discrete") and isinstance(final_main, dict):
                _p10 = final_main.get("P10"); _p50 = final_main.get("P50"); _p90 = final_main.get("P90")
                md.append(f"- final_quantiles: P10={_p10}, P50={_p50}, P90={_p90}")
            md.append(f"- bmc_summary={_json_dumps_for_db(bmc_json)}")
        except Exception as _bmc_dbg_ex:
            md.append(f"- bmc_debug_error={type(_bmc_dbg_ex).__name__}: {str(_bmc_dbg_ex)[:200]}")
    
        # --------------------------------------------------------------------------
        # Attach BMC summary into CSV row (JSON), then persist both CSV + human log
        # --------------------------------------------------------------------------
        try:
            if isinstance(bmc_summary, dict):
                row["bmc_summary_json"] = {k: v for k, v in bmc_summary.items() if k != "samples"}
        except Exception:
            # keep whatever default is in row already
            pass
    
        # Write human-readable markdown file
        try:
            safe_md = _sanitize_markdown_chunks(md)
            if len(safe_md) < len(md):
                dropped = len(md) - len(safe_md)
                print(f"[warn] Dropped {dropped} non-string markdown line(s) for Q{question_id}.")
            write_human_markdown(run_id=run_id, question_id=question_id, content="\n\n".join(safe_md))
        except Exception as _md_ex:
            print(f"[warn] failed to write human markdown for Q{question_id}: {type(_md_ex).__name__}: {str(_md_ex)[:180]}")
    
        # Finally, write the unified CSV row
        write_unified_row(row)
        print("✔ logged to forecasts.csv")
        return
    
    
    except Exception as _e:
        if summary is not None:
            summary.error = f"{type(_e).__name__}: {str(_e)[:200]}"
        _post_t = type(_post_original).__name__
        try:
            _q_t = type(q).__name__
        except Exception:
            _q_t = "unknown"
        try:
            _cls_t = type(cls_info).__name__  # may be undefined earlier; that's fine
        except Exception:
            _cls_t = "unknown"

        _err_t = type(_e).__name__
        _err_msg = str(_e)[:200]

        # Detect SPD questions
        is_spd = False
        try:
            poss = (q.get("possibilities") or {}) if isinstance(q, dict) else {}
            qt = (poss.get("type") or (q.get("type") if isinstance(q, dict) else "") or "").lower()
            is_spd = (qt == "spd")
        except Exception:
            is_spd = False

        spd_keys = None
        if is_spd:
            try:
                keys_set = set()
                ens_obj = locals().get("ens_res")
                if hasattr(ens_obj, "members"):
                    for _m in getattr(ens_obj, "members", []):
                        if isinstance(_m, MemberOutput) and isinstance(_m.parsed, dict):
                            keys_set.update(_m.parsed.keys())
                final_obj = locals().get("final_main")
                if isinstance(final_obj, dict):
                    keys_set.update(final_obj.keys())
                spd_keys = sorted(list(keys_set)) if keys_set else []
            except Exception:
                spd_keys = None

        # Core log message
        try:
            msg = (
                f"[error] run_one_question internal failure "
                f"(post_type={_post_t}, q_type={_q_t}, cls_info_type={_cls_t}): "
                f"{_err_t}: {_err_msg}"
            )
            if spd_keys is not None:
                msg += f" | spd_keys={spd_keys!r}"
            print(msg)
            traceback.print_exc()
        except Exception:
            pass

        # --- SPD soft-fail toggle ---
        hard_fail = os.getenv("PYTHIA_SPD_HARD_FAIL", "0") == "1"

        if is_spd and isinstance(_e, KeyError) and not hard_fail:
            print(
                f"[spd] soft-fail KeyError in SPD question; "
                f"skipping question without raising (post_type={_post_t}, q_type={_q_t})."
            )
            return

        raise RuntimeError(
            f"run_one_question failed (post={_post_t}, q={_q_t}, cls_info={_cls_t})"
        ) from _e


async def run_one_question(
    post: dict,
    *,
    run_id: str,
    purpose: str,
    calib: Dict[str, Any],
    seen_guard_run_report: Optional[Dict[str, Any]] = None,
    summary: Optional[QuestionRunSummary] = None,
) -> None:
    post = _must_dict("post", post)
    q = _as_dict(post.get("question"))
    question_id_raw = q.get("id") or post.get("id") or post.get("post_id") or ""
    question_id = str(question_id_raw)

    seen_guard_state: Dict[str, Any] = {
        "enabled": bool(seen_guard),
        "lock_acquired": None,
        "lock_error": "",
    }

    lock_stack = ExitStack()
    try:
        if seen_guard:
            try:
                acquired = lock_stack.enter_context(seen_guard.lock(question_id))
                seen_guard_state["lock_acquired"] = bool(acquired)
                if not acquired:
                    print(f"[seen_guard] QID {question_id} is locked by another process; skipping.")
                    return
            except Exception as _sg_lock_ex:
                seen_guard_state["lock_error"] = f"{type(_sg_lock_ex).__name__}: {str(_sg_lock_ex)[:160]}"
                seen_guard_state["lock_acquired"] = False
                print(f"[seen_guard] lock error for QID {question_id}: {seen_guard_state['lock_error']}")

        await _run_one_question_body(
            post=post,
            run_id=run_id,
            purpose=purpose,
            calib=calib,
            seen_guard_state=seen_guard_state,
            seen_guard_run_report=seen_guard_run_report,
            summary=summary,
        )
    finally:
        lock_stack.close()


# ==============================================================================
# Top-level runner (fetch posts, iterate, and commit logs)
# ==============================================================================

async def run_job(
    mode: str,
    limit: int,
    purpose: str,
    *,
    questions_file: str = "data/test_questions.json",
    iso3_filter: Optional[Set[str]] = None,
) -> None:
    """
    Fetch a batch of posts and process them one by one.
    Supports:
      - mode="pythia": reads Horizon Scanner questions from DuckDB
      - mode="test_questions": reads local JSON of test posts

    iso3_filter can be used to restrict Pythia runs to specific ISO3 codes.
    """
    # --- local imports to keep this function self-contained ---------------
    import os, json, inspect
    from pathlib import Path

    def _istamp():
        # Use Istanbul-tz stamp from config if available, else UTC-ish fallback
        try:
            from .config import IST_TZ
            from datetime import datetime
            return datetime.now(IST_TZ).strftime("%Y%m%d-%H%M%S")
        except Exception:
            from datetime import datetime, timezone
            return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    run_id = _istamp()
    os.environ["PYTHIA_FORECASTER_RUN_ID"] = run_id
    reset_provider_failures_for_run(run_id)
    print("----------------------------------------------------------------------------------------")

    run_summaries: List[QuestionRunSummary] = []
    hs_run_ids: set[str] = set()

    iso3_filter_normalized = (
        {code.upper() for code in iso3_filter} if iso3_filter else None
    )

    # --- load helpers from this module scope --------------------------------
    # They already exist below in this file; just reference them:
    #   ensure_unified_csv(), run_one_question(...), _load_calibration_weights_file()
    #   finalize_and_commit()

    # --- load questions ------------------------------------------------------
    posts: List[dict] = []
    fetch_limit = max(1, limit)

    if mode == "test_questions":
        qfile = Path(questions_file)
        if not qfile.exists():
            raise FileNotFoundError(f"Questions file not found: {qfile}")

        with qfile.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            posts = data
        elif isinstance(data, dict):
            posts = data.get("results") or data.get("posts") or []
        else:
            posts = []

        print(f"[info] Loaded {len(posts)} test post(s) from {qfile.as_posix()}.")
    elif mode == "pythia":
        print("[info] Loading Pythia questions from DuckDB...")
        questions_loaded = _load_pythia_questions(
            fetch_limit, iso3_filter=iso3_filter_normalized
        )
        posts = []
        for q in questions_loaded:
            post = _pythia_question_to_post(q)
            if post:
                posts.append(post)
        print(
            f"[info] Loaded {len(posts)} question(s) from DuckDB (Pythia mode)."
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # --- SeenGuard wiring (handles both package and top-level) ---------------
    def _load_seen_guard():
        """
        Try to import a SeenGuard instance/class from forecaster.seen_guard or seen_guard.
        Return an instance or None.
        """
        sg_mod = None
        # Prefer relative (inside package)
        try:
            from . import seen_guard as _sg
            sg_mod = _sg
        except Exception:
            # Fall back to absolute names
            for modname in ("forecaster.seen_guard", "seen_guard"):
                try:
                    sg_mod = importlib.import_module(modname)
                    break
                except Exception:
                    continue

        if sg_mod is None:
            return None

        # If module exposes a ready-made instance, use it
        for attr in ("_GUARD", "GUARD", "guard"):
            guard = getattr(sg_mod, attr, None)
            if guard is not None:
                return guard

        # Else instantiate SeenGuard(csv_path/state_file/lock_dir via env defaults)
        SG = getattr(sg_mod, "SeenGuard", None)
        if SG is not None and inspect.isclass(SG):
            try:
                return SG()  # it reads env defaults internally
            except Exception:
                return None

        return None

    def _apply_seen_guard(guard, posts_list):
        """
        Call whichever filter method exists; normalize return to (posts, dup_count).
        """
        if guard is None or not posts_list:
            return posts_list, 0

        candidates = [
            "filter_unseen_posts",     # your current API
            "filter_fresh_posts",      # earlier suggestion
            "filter_posts",
            "filter_recent_posts",
            "filter_new_posts",
            "filter",                  # very generic, last
        ]
        last_err = None
        for name in candidates:
            if hasattr(guard, name):
                fn = getattr(guard, name)
                try:
                    # most APIs: fn(posts)
                    result = fn(posts_list)
                except TypeError:
                    try:
                        # named arg fallback
                        result = fn(posts=posts_list)
                    except Exception as e:
                        last_err = e
                        continue
                except Exception as e:
                    last_err = e
                    continue

                # normalize return
                if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], list):
                    return result
                if isinstance(result, list):
                    # compute naive dup_count
                    return result, max(0, len(posts_list) - len(result))

                # unexpected shape → treat as no-op
                return posts_list, 0

        if last_err:
            raise last_err
        return posts_list, 0

    # Try to activate seen guard
    seen_guard_run_report: Dict[str, Any] = {
        "enabled": False,
        "before": len(posts),
        "after": len(posts),
        "skipped": 0,
        "error": "",
    }
    try:
        guard = _load_seen_guard()
        if guard is None:
            print("[seen_guard] not active; processing all posts returned.")
        else:
            seen_guard_run_report["enabled"] = True
            seen_guard_run_report["before"] = len(posts)
            before = len(posts)
            posts, dup_count = _apply_seen_guard(guard, posts)
            after = len(posts)
            if not isinstance(dup_count, int):
                dup_count = max(0, before - after)
            seen_guard_run_report["skipped"] = int(dup_count)
            seen_guard_run_report["after"] = after
            print(f"[seen_guard] {dup_count} duplicate(s) skipped; {after} fresh post(s) remain.")
    except Exception as _sg_ex:
        seen_guard_run_report["error"] = f"{type(_sg_ex).__name__}: {str(_sg_ex)[:200]}"
        print(f"[seen_guard] disabled due to error: {type(_sg_ex).__name__}: {str(_sg_ex)[:200]}")

    # Ensure CSV exists before we start
    ensure_unified_csv()

    # Process each post
    if not posts:
        print("[info] No posts to process.")
        try:
            finalize_and_commit()
            print("[logs] finalize_and_commit: done")
        except Exception as e:
            print(f"[warn] finalize_and_commit failed: {type(e).__name__}: {str(e)[:180]}")
        if os.getenv("PYTHIA_LOG_SUMMARY", "1") == "1":
            print("----------------------------------------------------------------------------------------")
            print("[summary] Forecaster Pythia run summary:")
            print(f"[summary] run_id={run_id} | hs_run_ids=none")
            print(f"[summary] Questions processed: {len(run_summaries)}")
        return

    batch = posts[: max(1, limit)]
    for idx, raw_post in enumerate(batch, start=1):
        post = raw_post
        if not isinstance(post, dict):
            print(
                f"[error] Skipping entry #{idx}: unexpected post type "
                f"{type(raw_post).__name__}"
            )
            continue

        q = post.get("question") or {}
        qid = q.get("id") or post.get("id") or "?"
        title = (q.get("title") or post.get("title") or "").strip()
        pmeta = _extract_pythia_meta(post)
        summary = QuestionRunSummary(
            question_id=str(qid),
            iso3=pmeta.get("iso3", ""),
            hazard_code=pmeta.get("hazard_code", ""),
            metric=(pmeta.get("metric") or "").upper(),
        )
        hs_run_id = str(post.get("pythia_hs_run_id") or "").strip()
        if hs_run_id:
            hs_run_ids.add(hs_run_id)
        print("")
        print("----------------------------------------------------------------------------------------")
        print(f"[{idx}/{len(batch)}] ❓ {title}  (QID: {qid})")
        try:
            await run_one_question(
                post,
                run_id=run_id,
                purpose=purpose,
                calib=_load_calibration_weights_file(),
                seen_guard_run_report=seen_guard_run_report,
                summary=summary,
            )
        except Exception as e:
            print(f"[error] run_one_question failed for QID {qid}: {type(e).__name__}: {str(e)[:200]}")
            if summary is not None and not summary.error:
                summary.error = f"{type(e).__name__}: {str(e)[:200]}"
        finally:
            run_summaries.append(summary)

    # Commit logs to git if configured
    try:
        finalize_and_commit()
        print("[logs] finalize_and_commit: done")
    except Exception as e:
        print(f"[warn] finalize_and_commit failed: {type(e).__name__}: {str(e)[:180]}")

    if os.getenv("PYTHIA_LOG_SUMMARY", "1") == "1":
        print("----------------------------------------------------------------------------------------")
        print("[summary] Forecaster Pythia run summary:")
        hs_ids = ", ".join(sorted(hs_run_ids)) if hs_run_ids else "none"
        print(f"[summary] run_id={run_id} | hs_run_ids={hs_ids}")
        print(f"[summary] Questions processed: {len(run_summaries)}")
        for s in sorted(run_summaries, key=lambda _s: _s.question_id):
            status = "OK" if not s.error else f"ERROR: {s.error}"
            model_strs = []
            for name, stats in sorted(s.models.items()):
                model_strs.append(
                    f"{name}(ok={int(stats.get('ok', 0))}, tokens={int(stats.get('total_tokens', 0))}, "
                    f"cost=${float(stats.get('cost_usd', 0.0)):.4f})"
                )
            models_joined = "; ".join(model_strs) if model_strs else "none"

            ev_span = ""
            if s.ev_min is not None and s.ev_max is not None:
                ev_span = f" EV[{s.ev_min:,.0f}–{s.ev_max:,.0f}]"

            print(
                f"[summary] Q {s.question_id} | {s.iso3}/{s.hazard_code}/{s.metric} | "
                f"months={s.month_count}x{s.buckets_per_month} | "
                f"ens_rows={s.ensemble_rows} raw_rows={s.raw_rows}{ev_span} | "
                f"models: {models_joined} | {status}"
            )


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
    try:
        _advise_poetry_lock_if_needed()
    except Exception:
        pass
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
            ]
            placeholders = ",".join(["?"] * len(allowed_iso3s))
            sql = f"""
                SELECT
                    question_id, hs_run_id, iso3, hazard_code, metric,
                    target_month, window_start_date, window_end_date, wording
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
        print(f"[v2] run_id={run_id} | questions={len(questions)}")

        async def _run_v2_pipeline_async() -> None:
            research_sem = asyncio.Semaphore(MAX_RESEARCH_WORKERS)
            spd_sem = asyncio.Semaphore(MAX_SPD_WORKERS)

            async def _research_task(q: dict) -> None:
                async with research_sem:
                    await _run_research_for_question(run_id, q)

            async def _spd_task(q: dict) -> None:
                if not _question_needs_spd(run_id, q):
                    return
                async with spd_sem:
                    await _run_spd_for_question(run_id, q)

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
