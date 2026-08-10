# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Compute forecast deviation-from-base-rate rows (``forecast_deviation``).

One row per (run, question, aggregate model): how far the published forecast
moved away from the base-rate anchor the forecaster was shown at prompt time
(``pythia.tools.base_rate_spd``), plus expected-impact values for ranking.

Deterministic — all arithmetic here, none in any LLM. The interpreter module
explains these numbers; it never computes them.

Metrics per row:
- ``js_vs_baserate``: Jensen-Shannon divergence between the forecast SPD
  (averaged across the six window months) and the base-rate SPD. Reuses
  ``_js_divergence`` from ``generate_calibration_advice`` — one implementation
  in the repo, natural-log, range [0, ln 2].
- ``log_ev_ratio``: ``ln(EV_forecast / EV_baserate)`` — signed, so direction
  is legible. Binary questions use ``ln(p_forecast / p_baserate)``.
- ``eiv_nominal``: expected impact value over the window using the surge blend
  the risk index uses (``max + alpha * (sum - max)``, alpha=0.1) with seed
  centroids from ``pythia.buckets.centroids_for``. NULL for binary questions
  (EVENT_OCCURRENCE has no impact centroid).
- ``eiv_per_100k``: population-normalised, from ``resolver/data/population.csv``
  (the same registry ``/v1/risk_index?normalize=true`` reads; env override
  ``PYTHIA_POPULATION_CSV_PATH``).

A question whose (hazard, metric) pair has no base rate gets NO row — an
anchor is never silently invented (``baserate_source`` records provenance for
every row that is written, and ``baserate_json`` stores the anchor itself so
the report is reproducible).

``is_test`` inherits from the question row (None-and-inherit; never a
hardcoded False).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pythia.buckets import centroids_for, labels_for

from interpreter import config as _interp_config
from interpreter import gating as _gating
from pythia.config import load as load_cfg
from pythia.tools.base_rate_spd import base_rate_spd, _as_of_ym
from pythia.tools.compute_scores import _load_spd
from resolver.db import duckdb_io

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())

# Surge blending weight — matches the /v1/risk_index default so the
# interpreter's impact numbers agree with the dashboard's.
EIV_SURGE_ALPHA = 0.1

# Floor for expected values entering the log ratio (same idiom as the EIV
# scoring in compute_scores: ln(0) is not a number anyone can read).
EV_FLOOR = 1.0

NUM_MONTHS = 6

# The aggregates a deviation row is computed for. Derived from the calibration
# exclusion list (single-sourced), minus the legacy 'ensemble' name.
try:
    from pythia.tools.compute_calibration_pythia import AGGREGATE_MODEL_NAMES as _AGG
    DEVIATION_MODEL_NAMES = frozenset(_AGG) - {"ensemble"}
except Exception:  # pragma: no cover - defensive literal fallback
    DEVIATION_MODEL_NAMES = frozenset(
        {"ensemble_mean_v2", "ensemble_bayesmc_v2", "track2_flash", "sibyl"}
    )


def _js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence (reuses the calibration-advice helper)."""
    import numpy as np

    from pythia.tools.generate_calibration_advice import (  # noqa: PLC0415
        _js_divergence as _jsd,
    )

    return float(_jsd(np.asarray(p, dtype=float), np.asarray(q, dtype=float)))


def _utcnow_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Population registry (same source as pythia/api/core.py, stdlib parsing)
# ---------------------------------------------------------------------------

_POPULATION_CACHE: Dict[str, int] = {}
_POPULATION_LOADED = False


def _population(iso3: str) -> Optional[int]:
    global _POPULATION_LOADED
    if not _POPULATION_LOADED:
        _POPULATION_LOADED = True
        try:
            env_path = os.getenv("PYTHIA_POPULATION_CSV_PATH")
            if env_path:
                path = Path(env_path)
            else:
                repo_root = Path(__file__).resolve().parents[2]
                path = repo_root / "resolver" / "data" / "population.csv"
            if path.exists():
                with path.open("r", encoding="utf-8-sig", newline="") as fh:
                    reader = csv.DictReader(fh)
                    fields = {
                        (name or "").strip().lower(): name
                        for name in (reader.fieldnames or [])
                    }
                    iso_col = fields.get("iso3")
                    pop_col = fields.get("population")
                    if iso_col and pop_col:
                        for row in reader:
                            iso = str(row.get(iso_col) or "").strip().upper()
                            raw = (
                                str(row.get(pop_col) or "")
                                .strip()
                                .replace(",", "")
                                .replace(" ", "")
                            )
                            if not iso or not raw:
                                continue
                            try:
                                value = int(float(raw))
                            except ValueError:
                                continue
                            if value > 0:
                                _POPULATION_CACHE[iso] = value
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Population registry unavailable: %s", exc)
    return _POPULATION_CACHE.get((iso3 or "").upper())


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

FORECAST_DEVIATION_DDL = """
CREATE TABLE IF NOT EXISTS forecast_deviation (
    run_id TEXT,
    question_id TEXT,
    model_name TEXT,
    iso3 TEXT,
    hazard_code TEXT,
    metric TEXT,
    score_family TEXT,
    js_vs_baserate DOUBLE,
    log_ev_ratio DOUBLE,
    eiv_nominal DOUBLE,
    eiv_per_100k DOUBLE,
    -- Expected EXCESS: how many more than history would suggest. This is
    -- what the report ranks by, because the ratio is dominated by whichever
    -- anchor sits closest to zero.
    eiv_baserate DOUBLE,
    excess_nominal DOUBLE,
    excess_per_100k DOUBLE,
    -- P(>= the metric's action threshold) per horizon, JSON list in window
    -- order, and how many observations the anchor rests on.
    exceedances_json TEXT,
    baserate_n_obs INTEGER,
    baserate_source TEXT,
    baserate_json TEXT,
    is_test BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT now()
)
"""


# Columns added after the table shipped. The DDL is CREATE TABLE IF NOT
# EXISTS, so an existing canonical DB would never gain them; the same
# add-column-if-missing idiom the resolution machine uses.
_COLUMN_MIGRATIONS = (
    ("eiv_baserate", "DOUBLE"),
    ("excess_nominal", "DOUBLE"),
    ("excess_per_100k", "DOUBLE"),
    ("exceedances_json", "TEXT"),
    ("baserate_n_obs", "INTEGER"),
)


# base_rate_spd counts observations under a different key per method
# (n_months_used, n_months_observed, n_severity_values, ...). One normalised
# reading, because the thin-anchor rule needs a single number and the caller
# should not have to know which anchor method ran.
_N_OBS_KEYS = (
    "n_months_used",
    "n_months_observed",
    "n_pa_months_all",
    "n_occurrence_obs",
    "n_severity_values",
)


def _baserate_n_obs(detail: Any) -> Optional[int]:
    """How many historical observations the anchor rests on, or None."""
    if not isinstance(detail, dict):
        return None
    for key in _N_OBS_KEYS:
        value = detail.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _ensure_table(conn) -> None:
    conn.execute(FORECAST_DEVIATION_DDL)
    for name, sql_type in _COLUMN_MIGRATIONS:
        try:
            conn.execute(
                f"ALTER TABLE forecast_deviation ADD COLUMN IF NOT EXISTS "
                f"{name} {sql_type}"
            )
        except Exception as exc:  # noqa: BLE001 - a fresh table already has them
            LOGGER.debug("forecast_deviation migration %s: %s", name, exc)


# ---------------------------------------------------------------------------
# Per-question metric computation (pure helpers, unit-tested directly)
# ---------------------------------------------------------------------------

def _surge_blend(monthly: Sequence[float], alpha: float = EIV_SURGE_ALPHA) -> float:
    if not monthly:
        return 0.0
    mx = max(monthly)
    return mx + alpha * (sum(monthly) - mx)


def _mean_spd(monthly_spds: Sequence[Sequence[float]]) -> List[float]:
    k = len(monthly_spds[0])
    out = [0.0] * k
    for spd in monthly_spds:
        for i, p in enumerate(spd):
            out[i] += float(p)
    n = float(len(monthly_spds))
    return [v / n for v in out]


def spd_deviation_metrics(
    monthly_spds: Sequence[Sequence[float]],
    base_probs: Sequence[float],
    base_probs_by_month: Optional[Sequence[Sequence[float]]],
    metric: str,
) -> Dict[str, float]:
    """Deviation metrics for an SPD question.

    ``monthly_spds``: the forecast's per-month bucket vectors (any subset of
    the six window months). ``base_probs_by_month``, when given, must align
    positionally with ``monthly_spds``.
    """
    centroids = centroids_for(metric)
    mean_forecast = _mean_spd(monthly_spds)
    js = _js_divergence(mean_forecast, base_probs)

    monthly_eiv_f = [
        sum(p * c for p, c in zip(spd, centroids)) for spd in monthly_spds
    ]
    if base_probs_by_month:
        base_monthly = list(base_probs_by_month)
    else:
        base_monthly = [list(base_probs)] * len(monthly_spds)
    monthly_eiv_b = [
        sum(p * c for p, c in zip(spd, centroids)) for spd in base_monthly
    ]

    eiv_f = _surge_blend(monthly_eiv_f)
    eiv_b = _surge_blend(monthly_eiv_b)
    log_ev_ratio = math.log(max(eiv_f, EV_FLOOR) / max(eiv_b, EV_FLOOR))

    return {
        "js_vs_baserate": js,
        "log_ev_ratio": log_ev_ratio,
        "eiv_nominal": eiv_f,
        "eiv_baserate": eiv_b,
    }


def binary_deviation_metrics(
    monthly_p: Sequence[float],
    base_p: float,
    base_p_by_month: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Deviation metrics for a binary EVENT_OCCURRENCE question."""
    p_f = sum(monthly_p) / len(monthly_p)
    if base_p_by_month:
        p_b = sum(base_p_by_month) / len(base_p_by_month)
    else:
        p_b = base_p
    js = _js_divergence([p_f, 1.0 - p_f], [p_b, 1.0 - p_b])
    eps = 1e-6
    log_ev_ratio = math.log(max(p_f, eps) / max(p_b, eps))
    return {
        "js_vs_baserate": js,
        "log_ev_ratio": log_ev_ratio,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _anchor_ym(window_start: Any, target_month: Any) -> Optional[str]:
    """The question's window_start month; falls back to target_month - 5."""
    try:
        if window_start:
            return _as_of_ym(str(window_start)[:7])
    except ValueError:
        pass
    try:
        if target_month:
            ym = _as_of_ym(str(target_month)[:7])
            y = int(ym[:4])
            m = int(ym[5:7]) - (NUM_MONTHS - 1)
            y += (m - 1) // 12
            m = (m - 1) % 12 + 1
            return f"{y:04d}-{m:02d}"
    except ValueError:
        pass
    return None


def compute_deviation(db_url: str, run_id: Optional[str] = None) -> int:
    """Compute and write forecast_deviation rows. Returns rows written."""
    if not duckdb_io.DUCKDB_AVAILABLE:
        raise RuntimeError(duckdb_io.duckdb_unavailable_reason())
    conn = duckdb_io.get_db(db_url or duckdb_io.DEFAULT_DB_URL)
    try:
        _ensure_table(conn)

        for table in ("forecasts_raw", "questions"):
            try:
                conn.execute(f"SELECT 1 FROM {table} LIMIT 0")
            except Exception:
                LOGGER.info("compute_deviation: table %s not found; nothing to do.", table)
                return 0

        model_list = ",".join(f"'{m}'" for m in sorted(DEVIATION_MODEL_NAMES))
        if run_id is not None:
            rid_clause, rid_params = "AND fr.run_id = ?", [run_id]
        else:
            rid_clause, rid_params = "", []

        pairs = conn.execute(
            f"""
            SELECT DISTINCT fr.run_id, fr.question_id, fr.model_name,
                   q.iso3, q.hazard_code, upper(q.metric) AS metric,
                   q.window_start_date, q.target_month,
                   COALESCE(q.is_test, FALSE) AS is_test
            FROM forecasts_raw fr
            JOIN questions q ON q.question_id = fr.question_id
            WHERE fr.model_name IN ({model_list})
              {rid_clause}
            ORDER BY fr.run_id, fr.question_id, fr.model_name
            """,
            rid_params,
        ).fetchall()

        if not pairs:
            LOGGER.info("compute_deviation: no aggregate forecasts to process.")
            return 0

        # Base rate cache: one lookup per question, shared across models.
        base_cache: Dict[str, Tuple[list, str, dict]] = {}
        rows_out: List[tuple] = []
        run_ids_touched: set = set()
        n_no_baserate = 0
        now = _utcnow_naive()

        for (rid, qid, model_name, iso3, hazard_code, metric,
             window_start, target_month, is_test) in pairs:
            anchor = _anchor_ym(window_start, target_month)
            if anchor is None:
                LOGGER.warning(
                    "compute_deviation: %s has no window_start/target_month; skipping.", qid
                )
                continue

            if qid not in base_cache:
                base_cache[qid] = base_rate_spd(conn, iso3, hazard_code, metric, anchor)
            base_probs, base_source, base_detail = base_cache[qid]
            if not base_probs:
                n_no_baserate += 1
                LOGGER.info(
                    "compute_deviation: no base rate for %s (%s/%s): %s",
                    qid, hazard_code, metric,
                    base_detail.get("reason", "unknown"),
                )
                continue

            score_family = str(base_detail.get("score_family") or "spd")
            is_binary = (metric or "").upper() == "EVENT_OCCURRENCE"

            probs_by_month_map = base_detail.get("probs_by_month") or {}

            if is_binary:
                p_rows = conn.execute(
                    """
                    SELECT month_index, probability FROM forecasts_raw
                    WHERE question_id = ? AND model_name = ? AND bucket_index = 1
                      AND (run_id = ? OR (? IS NULL AND run_id IS NULL))
                    ORDER BY month_index
                    """,
                    [qid, model_name, rid, rid],
                ).fetchall()
                monthly_p = [float(p) for _, p in p_rows if p is not None]
                if not monthly_p:
                    continue
                base_p_by_month = None
                if probs_by_month_map:
                    base_p_by_month = [v[0] for v in probs_by_month_map.values()]
                metrics_d = binary_deviation_metrics(
                    monthly_p, base_probs[0], base_p_by_month
                )
                eiv_nominal = None
                eiv_per_100k = None
                eiv_baserate = None
                excess_nominal = None
                excess_per_100k = None
                # A binary question's P(event) IS its exceedance; there is no
                # size threshold to clear, only a probability.
                exceedances = [float(x) for x in monthly_p]
            else:
                class_bins = labels_for(metric)
                if not class_bins:
                    continue
                monthly_spds = []
                for month_index in range(1, NUM_MONTHS + 1):
                    spd = _load_spd(
                        conn,
                        question_id=qid,
                        horizon_m=month_index,
                        class_bins=class_bins,
                        model_name=model_name,
                        run_id=rid,
                        _has_run_id=True,
                    )
                    if spd:
                        monthly_spds.append(spd)
                if not monthly_spds:
                    continue
                metrics_d = spd_deviation_metrics(
                    monthly_spds, base_probs, None, metric
                )
                eiv_nominal = metrics_d["eiv_nominal"]
                eiv_baserate = metrics_d["eiv_baserate"]
                pop = _population(iso3)
                eiv_per_100k = (
                    (eiv_nominal / pop) * 100_000.0 if pop else None
                )
                # Expected EXCESS, the report's ordering: how many more than
                # history would suggest. Both sides use the same surge blend,
                # so the difference is one summary applied to two
                # distributions rather than two different summaries.
                excess_nominal = eiv_nominal - eiv_baserate
                excess_per_100k = (
                    (excess_nominal / pop) * 100_000.0 if pop else None
                )
                # P(>= the action threshold) per horizon. The threshold is
                # per country as well as per metric: a flat floor would
                # exclude every small state.
                threshold = _gating.materiality_threshold(
                    metric, pop,
                    absolute=_interp_config.threshold_for_metric(metric),
                    population_share=_interp_config.population_share_for_metric(metric),
                )
                exceedances = _gating.horizon_exceedances(
                    monthly_spds, metric, threshold
                )

            rows_out.append((
                rid, qid, model_name, iso3, hazard_code, metric,
                score_family,
                metrics_d["js_vs_baserate"],
                metrics_d["log_ev_ratio"],
                eiv_nominal, eiv_per_100k,
                eiv_baserate, excess_nominal, excess_per_100k,
                json.dumps(exceedances),
                _baserate_n_obs(base_detail),
                base_source,
                json.dumps({"probs": base_probs, "detail": base_detail}, default=str),
                bool(is_test), now,
            ))
            run_ids_touched.add(rid)

        # Idempotent per-run replace.
        for rid in run_ids_touched:
            if rid is None:
                conn.execute("DELETE FROM forecast_deviation WHERE run_id IS NULL")
            else:
                conn.execute("DELETE FROM forecast_deviation WHERE run_id = ?", [rid])
        if rows_out:
            conn.executemany(
                """
                INSERT INTO forecast_deviation
                    (run_id, question_id, model_name, iso3, hazard_code, metric,
                     score_family, js_vs_baserate, log_ev_ratio, eiv_nominal,
                     eiv_per_100k, eiv_baserate, excess_nominal, excess_per_100k,
                     exceedances_json, baserate_n_obs,
                     baserate_source, baserate_json, is_test, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_out,
            )
        LOGGER.info(
            "compute_deviation: wrote %d rows across %d run(s); "
            "%d question(s) had no base-rate anchor (skipped, never invented).",
            len(rows_out), len(run_ids_touched), n_no_baserate,
        )
        return len(rows_out)
    finally:
        duckdb_io.close_db(conn)


def _get_db_url_from_config() -> str:
    cfg = load_cfg()
    app_cfg = cfg.get("app", {}) if isinstance(cfg, dict) else {}
    db_url = str(app_cfg.get("db_url", "")).strip()
    return db_url or duckdb_io.DEFAULT_DB_URL


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute forecast deviation-from-base-rate rows.",
    )
    parser.add_argument("--db-url", default=None, help="DuckDB URL (default: from config).")
    parser.add_argument(
        "--run-id", default=None,
        help="Restrict to one forecaster run id (default: all runs with aggregate rows).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    db_url = args.db_url or _get_db_url_from_config()
    compute_deviation(db_url=db_url, run_id=args.run_id)


if __name__ == "__main__":
    main()
