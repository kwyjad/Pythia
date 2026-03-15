# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Generate concrete, hazard-specific calibration advice from scoring data.

Reads the scores, resolutions, forecasts_ensemble and forecasts_raw tables,
computes calibration findings per hazard/metric, and writes prompt-ready
advice text to the calibration_advice table.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pythia.config import load as load_cfg
from pythia.tools.compute_scores import (
    PA_THRESHOLDS,
    FATAL_THRESHOLDS,
    SPD_CLASS_BINS_PA,
    SPD_CLASS_BINS_FATALITIES,
    _bucket_index,
)
from resolver.db import duckdb_io

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())

MIN_QUESTIONS = 20
MIN_QUESTIONS_PER_MODEL = 10
MAX_ADVICE_CHARS = 3800

HAZARD_LABELS = {
    "ACE": "ARMED CONFLICT",
    "FL": "FLOOD",
    "DR": "DROUGHT",
    "TC": "TROPICAL CYCLONE",
    "HW": "HEATWAVE",
    "DI": "DISPLACEMENT INFLOW",
}

METRIC_LABELS = {
    "FATALITIES": "FATALITIES",
    "PA": "PEOPLE AFFECTED",
}

# Hand-written seed advice for bootstrapping before enough scoring data exists.
SEED_ADVICE: Dict[Tuple[str, str], str] = {
    ("ACE", "FATALITIES"): (
        "TAIL COVERAGE:\n"
        "- Historical pattern: conflict fatality spikes (bucket 4-5, >=100 fatalities) "
        "occur ~15-20% of months in active-conflict countries.\n"
        "- Known bias: models assign <5% to top buckets even when RC is elevated.\n"
        "- ACTION: For active conflict zones, ensure bucket 4+5 combined >= 10%. "
        "When RC >= L2, bucket 4+5 combined should be >= 15%.\n\n"
        "BUCKET CALIBRATION:\n"
        "- Bucket 1 (<5 fatalities) is typically overweighted by 10-15pp in "
        "active conflict zones.\n"
        "- ACTION: In countries with recent conflict history, do not assign >60% "
        "to bucket 1 unless there is strong de-escalation evidence.\n\n"
        "HORIZON DIFFERENTIATION:\n"
        "- Conflict fatalities show meaningful month-to-month variation. Later "
        "horizons (months 4-6) should carry wider distributions.\n"
        "- ACTION: Avoid identical SPDs across all 6 months. Widen uncertainty "
        "for months 4-6 relative to months 1-2."
    ),
    ("ACE", "PA"): (
        "TAIL COVERAGE:\n"
        "- Displacement events reaching 250k+ (buckets 4-5) occur in ~5-10% "
        "of country-months for ACE hazard.\n"
        "- ACTION: Ensure non-trivial tail mass (>=3% combined for buckets 4-5) "
        "unless the country has no recent displacement.\n\n"
        "BUCKET CALIBRATION:\n"
        "- The 10k-50k range (bucket 2) is systematically under-predicted in "
        "countries with ongoing low-level displacement.\n"
        "- ACTION: In countries with established IDP populations, give bucket 2 "
        "at least 15% probability."
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _table_exists(conn: Any, name: str) -> bool:
    try:
        conn.execute(f"PRAGMA table_info('{name}')").fetchall()
        return True
    except Exception:
        return False


def _row_count(conn: Any, name: str) -> int:
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] or 0
    except Exception:
        return 0


def _get_db_url_from_config() -> str:
    cfg = load_cfg()
    app_cfg = cfg.get("app", {}) if isinstance(cfg, dict) else {}
    db_url = str(app_cfg.get("db_url", "")).strip()
    if not db_url:
        db_url = duckdb_io.DEFAULT_DB_URL
        LOGGER.warning("app.db_url missing; falling back to %s", db_url)
    else:
        LOGGER.info("Using app.db_url from config: %s", db_url)
    return db_url


def _ensure_findings_json_column(conn: Any) -> None:
    """Ensure calibration_advice table has all required columns."""
    existing = set()
    try:
        for row in conn.execute("PRAGMA table_info('calibration_advice')").fetchall():
            existing.add(str(row[1]).lower())
    except Exception:
        return

    for col, col_type in [
        ("findings_json", "TEXT"),
        ("model_name", "TEXT"),
        ("advice_version", "TEXT"),
    ]:
        if col not in existing:
            try:
                conn.execute(
                    f"ALTER TABLE calibration_advice ADD COLUMN {col} {col_type}"
                )
            except Exception:
                pass

    # Backfill NULL model_name to sentinel
    if "model_name" in existing or "model_name" not in existing:
        try:
            conn.execute(
                "UPDATE calibration_advice SET model_name = '__shared__' WHERE model_name IS NULL"
            )
        except Exception:
            pass


def _class_bins_for_metric(metric: str) -> List[str]:
    """Return ordered class_bin labels for the given metric."""
    m = (metric or "").upper()
    if m == "FATALITIES":
        return list(SPD_CLASS_BINS_FATALITIES)
    return list(SPD_CLASS_BINS_PA)


def _tail_bins(metric: str) -> Tuple[str, str]:
    """Return the two class_bin labels corresponding to buckets 4 and 5."""
    m = (metric or "").upper()
    if m == "FATALITIES":
        return ("100-<500", ">=500")
    return ("250k-<500k", ">=500k")


def _bucket_case_sql(metric: str) -> str:
    """Generate a SQL CASE expression to bucketify resolutions.value."""
    m = (metric or "").upper()
    if m == "FATALITIES":
        return """
            CASE
                WHEN r.value < 5 THEN 1
                WHEN r.value < 25 THEN 2
                WHEN r.value < 100 THEN 3
                WHEN r.value < 500 THEN 4
                ELSE 5
            END
        """
    return """
        CASE
            WHEN r.value < 10000 THEN 1
            WHEN r.value < 50000 THEN 2
            WHEN r.value < 250000 THEN 3
            WHEN r.value < 500000 THEN 4
            ELSE 5
        END
    """


def _bin_to_bucket_num(class_bin: str, metric: str) -> Optional[int]:
    """Map a class_bin label to bucket number (1-based)."""
    bins = _class_bins_for_metric(metric)
    try:
        return bins.index(class_bin) + 1
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def _discover_hazard_metric_pairs(
    conn: Any,
) -> List[Tuple[str, str]]:
    """Find all (hazard_code, metric) pairs with resolved questions."""
    sql = """
        SELECT DISTINCT upper(q.hazard_code) AS hc, upper(q.metric) AS m
        FROM questions q
        JOIN resolutions r ON r.question_id = q.question_id
        WHERE upper(q.metric) IN ('PA', 'FATALITIES')
        ORDER BY hc, m
    """
    rows = conn.execute(sql).fetchall()
    return [(r[0], r[1]) for r in rows]


def _count_resolved(conn: Any, hazard_code: str, metric: str) -> int:
    """Count distinct resolved questions for a hazard/metric pair."""
    sql = """
        SELECT COUNT(DISTINCT r.question_id)
        FROM resolutions r
        JOIN questions q ON q.question_id = r.question_id
        WHERE upper(q.hazard_code) = ? AND upper(q.metric) = ?
    """
    row = conn.execute(sql, [hazard_code.upper(), metric.upper()]).fetchone()
    return row[0] if row else 0


def _compute_tail_coverage(
    conn: Any, hazard_code: str, metric: str,
) -> Optional[Dict[str, Any]]:
    """Compare ensemble tail probabilities vs actual resolution rates."""
    hz = hazard_code.upper()
    m = metric.upper()
    tail_bin_1, tail_bin_2 = _tail_bins(m)
    bucket_case = _bucket_case_sql(m)

    sql = f"""
        WITH resolved_with_bucket AS (
            SELECT
                r.question_id,
                r.horizon_m,
                {bucket_case} AS resolved_bucket
            FROM resolutions r
            JOIN questions q ON q.question_id = r.question_id
            WHERE upper(q.hazard_code) = ?
              AND upper(q.metric) = ?
        ),
        ensemble_tail_probs AS (
            SELECT
                fe.question_id,
                fe.horizon_m,
                SUM(CASE WHEN fe.class_bin IN (?, ?) THEN fe.p ELSE 0 END) AS tail_prob
            FROM forecasts_ensemble fe
            JOIN questions q ON q.question_id = fe.question_id
            WHERE upper(q.hazard_code) = ?
              AND upper(q.metric) = ?
              AND fe.class_bin IS NOT NULL
              AND fe.p IS NOT NULL
            GROUP BY fe.question_id, fe.horizon_m
        )
        SELECT
            COUNT(*) AS n_forecasts,
            AVG(et.tail_prob) AS avg_assigned_tail,
            SUM(CASE WHEN et.tail_prob < 0.05 THEN 1 ELSE 0 END)::DOUBLE
                / GREATEST(COUNT(*), 1) AS frac_starved,
            SUM(CASE WHEN rb.resolved_bucket >= 4 THEN 1 ELSE 0 END)::DOUBLE
                / GREATEST(COUNT(*), 1) AS actual_tail_rate
        FROM ensemble_tail_probs et
        JOIN resolved_with_bucket rb
            ON rb.question_id = et.question_id AND rb.horizon_m = et.horizon_m
    """
    try:
        row = conn.execute(
            sql, [hz, m, tail_bin_1, tail_bin_2, hz, m]
        ).fetchone()
    except Exception as exc:
        LOGGER.warning("Tail coverage query failed for %s/%s: %s", hz, m, exc)
        return None

    if not row or row[0] == 0:
        return None

    return {
        "n_forecasts": row[0],
        "avg_assigned_tail": float(row[1] or 0),
        "frac_starved": float(row[2] or 0),
        "actual_tail_rate": float(row[3] or 0),
    }


def _compute_bucket_calibration(
    conn: Any, hazard_code: str, metric: str,
) -> Optional[List[Dict[str, Any]]]:
    """Per-bucket reliability: mean assigned probability vs resolution rate."""
    hz = hazard_code.upper()
    m = metric.upper()
    bins = _class_bins_for_metric(m)
    bucket_case = _bucket_case_sql(m)

    sql = f"""
        WITH resolved_with_bucket AS (
            SELECT
                r.question_id,
                r.horizon_m,
                {bucket_case} AS resolved_bucket
            FROM resolutions r
            JOIN questions q ON q.question_id = r.question_id
            WHERE upper(q.hazard_code) = ?
              AND upper(q.metric) = ?
        )
        SELECT
            fe.class_bin,
            AVG(fe.p) AS mean_assigned,
            AVG(CASE WHEN rb.resolved_bucket = (
                CASE fe.class_bin
                    {' '.join(f"WHEN '{b}' THEN {i+1}" for i, b in enumerate(bins))}
                END
            ) THEN 1.0 ELSE 0.0 END) AS actual_rate,
            COUNT(*) AS n_samples
        FROM forecasts_ensemble fe
        JOIN resolved_with_bucket rb
            ON rb.question_id = fe.question_id AND rb.horizon_m = fe.horizon_m
        JOIN questions q ON q.question_id = fe.question_id
        WHERE upper(q.hazard_code) = ?
          AND upper(q.metric) = ?
          AND fe.class_bin IS NOT NULL
          AND fe.p IS NOT NULL
        GROUP BY fe.class_bin
        ORDER BY fe.class_bin
    """
    try:
        rows = conn.execute(sql, [hz, m, hz, m]).fetchall()
    except Exception as exc:
        LOGGER.warning("Bucket calibration query failed for %s/%s: %s", hz, m, exc)
        return None

    if not rows:
        return None

    result = []
    for row in rows:
        cb = str(row[0])
        if cb not in bins:
            continue
        result.append({
            "class_bin": cb,
            "mean_assigned": float(row[1] or 0),
            "actual_rate": float(row[2] or 0),
            "n_samples": int(row[3] or 0),
        })

    # Sort by bin order
    bin_order = {b: i for i, b in enumerate(bins)}
    result.sort(key=lambda x: bin_order.get(x["class_bin"], 99))
    return result if result else None


def _compute_per_model_brier(
    conn: Any, hazard_code: str, metric: str,
) -> Optional[Dict[str, Any]]:
    """Per-model and ensemble Brier scores from the scores table."""
    hz = hazard_code.upper()
    m = metric.upper()

    sql = """
        SELECT
            COALESCE(s.model_name, '__ensemble__') AS mn,
            AVG(s.value) AS avg_brier,
            COUNT(*) AS n_scores
        FROM scores s
        JOIN questions q ON q.question_id = s.question_id
        WHERE s.score_type = 'brier'
          AND upper(q.hazard_code) = ?
          AND upper(q.metric) = ?
        GROUP BY mn
        ORDER BY avg_brier ASC
    """
    try:
        rows = conn.execute(sql, [hz, m]).fetchall()
    except Exception as exc:
        LOGGER.warning("Per-model Brier query failed for %s/%s: %s", hz, m, exc)
        return None

    if not rows:
        return None

    models = []
    ensemble_brier = None
    for row in rows:
        name = str(row[0])
        brier = float(row[1] or 0)
        n = int(row[2] or 0)
        if name == "__ensemble__":
            ensemble_brier = brier
        else:
            models.append({"name": name, "brier": brier, "n": n})

    if not models:
        return None

    models.sort(key=lambda x: x["brier"])
    return {
        "ensemble_brier": ensemble_brier,
        "best": models[0],
        "worst": models[-1],
        "all_models": models,
    }


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two probability vectors."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # Ensure valid distributions
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    # KL(p||m) + KL(q||m)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def _compute_month_position_bias(
    conn: Any, hazard_code: str, metric: str,
    n_buckets: int = 5,
) -> Optional[Dict[str, Any]]:
    """Check if models differentiate SPDs across horizons 1-6.

    Uses Jensen-Shannon divergence between month-1 and month-6 average SPDs
    (full distribution check), plus legacy bucket-5 stdev for compatibility.
    """
    hz = hazard_code.upper()
    m = metric.upper()

    if not _table_exists(conn, "forecasts_raw"):
        return None

    sql = """
        SELECT
            fr.month_index,
            fr.bucket_index,
            AVG(fr.probability) AS avg_prob
        FROM forecasts_raw fr
        JOIN questions q ON q.question_id = fr.question_id
        WHERE upper(q.hazard_code) = ?
          AND upper(q.metric) = ?
          AND fr.month_index BETWEEN 1 AND 6
          AND fr.bucket_index BETWEEN 1 AND ?
          AND fr.probability IS NOT NULL
        GROUP BY fr.month_index, fr.bucket_index
        ORDER BY fr.month_index, fr.bucket_index
    """
    try:
        rows = conn.execute(sql, [hz, m, n_buckets]).fetchall()
    except Exception as exc:
        LOGGER.warning("Month position query failed for %s/%s: %s", hz, m, exc)
        return None

    if not rows:
        return None

    # Build per-month SPD vectors
    month_spds: Dict[int, List[float]] = {}
    for month_idx, bucket_idx, avg_prob in rows:
        mi = int(month_idx)
        bi = int(bucket_idx)
        if mi not in month_spds:
            month_spds[mi] = [0.0] * n_buckets
        if 1 <= bi <= n_buckets:
            month_spds[mi][bi - 1] = float(avg_prob or 0)

    # Legacy bucket-5 stdev
    b5_probs = [month_spds[mi][4] for mi in sorted(month_spds.keys()) if mi in month_spds]
    b5_stdev = float(np.std(b5_probs)) if len(b5_probs) >= 2 else 0.0

    # JSD between month 1 and month 6
    jsd = None
    if 1 in month_spds and 6 in month_spds:
        jsd = _js_divergence(
            np.array(month_spds[1]),
            np.array(month_spds[6]),
        )

    flat = (jsd is not None and jsd < 0.005) or (jsd is None and b5_stdev < 0.02)

    return {
        "by_month_b5": {mi: month_spds[mi][4] for mi in sorted(month_spds.keys())},
        "stdev": b5_stdev,
        "jsd_m1_m6": jsd,
        "mean_top_bucket_prob": sum(b5_probs) / len(b5_probs) if b5_probs else 0.0,
        "flat": flat,
    }


def _compute_per_model_bucket_calibration(
    conn: Any, hazard_code: str, metric: str, model_name: str,
) -> Optional[List[Dict[str, Any]]]:
    """Per-bucket reliability for a single model (from forecasts_raw)."""
    hz = hazard_code.upper()
    m = metric.upper()
    bins = _class_bins_for_metric(m)
    bucket_case = _bucket_case_sql(m)

    sql = f"""
        WITH resolved_with_bucket AS (
            SELECT
                r.question_id,
                r.horizon_m,
                {bucket_case} AS resolved_bucket
            FROM resolutions r
            JOIN questions q ON q.question_id = r.question_id
            WHERE upper(q.hazard_code) = ?
              AND upper(q.metric) = ?
        )
        SELECT
            fr.bucket_index,
            AVG(fr.probability) AS mean_assigned,
            AVG(CASE WHEN rb.resolved_bucket = fr.bucket_index
                 THEN 1.0 ELSE 0.0 END) AS actual_rate,
            COUNT(*) AS n_samples
        FROM forecasts_raw fr
        JOIN resolved_with_bucket rb
            ON rb.question_id = fr.question_id AND rb.horizon_m = fr.month_index
        JOIN questions q ON q.question_id = fr.question_id
        WHERE upper(q.hazard_code) = ?
          AND upper(q.metric) = ?
          AND fr.model_name = ?
          AND fr.probability IS NOT NULL
        GROUP BY fr.bucket_index
        ORDER BY fr.bucket_index
    """
    try:
        rows = conn.execute(sql, [hz, m, hz, m, model_name]).fetchall()
    except Exception as exc:
        LOGGER.warning(
            "Per-model bucket calibration failed for %s/%s/%s: %s",
            hz, m, model_name, exc,
        )
        return None

    if not rows:
        return None

    result = []
    for row in rows:
        bucket_idx = int(row[0])
        if bucket_idx < 1 or bucket_idx > len(bins):
            continue
        result.append({
            "bucket_index": bucket_idx,
            "class_bin": bins[bucket_idx - 1],
            "mean_assigned": float(row[1] or 0),
            "actual_rate": float(row[2] or 0),
            "n_samples": int(row[3] or 0),
        })

    result.sort(key=lambda x: x["bucket_index"])
    return result if result else None


def _compute_per_model_tail_coverage(
    conn: Any, hazard_code: str, metric: str, model_name: str,
) -> Optional[Dict[str, Any]]:
    """Tail coverage (buckets 4-5) for a single model from forecasts_raw."""
    hz = hazard_code.upper()
    m = metric.upper()
    bucket_case = _bucket_case_sql(m)

    sql = f"""
        WITH resolved_with_bucket AS (
            SELECT
                r.question_id,
                r.horizon_m,
                {bucket_case} AS resolved_bucket
            FROM resolutions r
            JOIN questions q ON q.question_id = r.question_id
            WHERE upper(q.hazard_code) = ?
              AND upper(q.metric) = ?
        ),
        model_tail_probs AS (
            SELECT
                fr.question_id,
                fr.month_index AS horizon_m,
                SUM(CASE WHEN fr.bucket_index >= 4 THEN fr.probability ELSE 0 END) AS tail_prob
            FROM forecasts_raw fr
            JOIN questions q ON q.question_id = fr.question_id
            WHERE upper(q.hazard_code) = ?
              AND upper(q.metric) = ?
              AND fr.model_name = ?
              AND fr.probability IS NOT NULL
            GROUP BY fr.question_id, fr.month_index
        )
        SELECT
            COUNT(*) AS n_forecasts,
            AVG(mt.tail_prob) AS avg_assigned_tail,
            SUM(CASE WHEN mt.tail_prob < 0.05 THEN 1 ELSE 0 END)::DOUBLE
                / GREATEST(COUNT(*), 1) AS frac_starved,
            SUM(CASE WHEN rb.resolved_bucket >= 4 THEN 1 ELSE 0 END)::DOUBLE
                / GREATEST(COUNT(*), 1) AS actual_tail_rate
        FROM model_tail_probs mt
        JOIN resolved_with_bucket rb
            ON rb.question_id = mt.question_id AND rb.horizon_m = mt.horizon_m
    """
    try:
        row = conn.execute(sql, [hz, m, hz, m, model_name]).fetchone()
    except Exception as exc:
        LOGGER.warning(
            "Per-model tail coverage failed for %s/%s/%s: %s",
            hz, m, model_name, exc,
        )
        return None

    if not row or row[0] == 0:
        return None

    return {
        "n_forecasts": row[0],
        "avg_assigned_tail": float(row[1] or 0),
        "frac_starved": float(row[2] or 0),
        "actual_tail_rate": float(row[3] or 0),
    }


def _compute_per_model_horizon_diff(
    conn: Any, hazard_code: str, metric: str, model_name: str,
    n_buckets: int = 5,
) -> Optional[Dict[str, Any]]:
    """Check horizon differentiation for a single model using JS divergence."""
    hz = hazard_code.upper()
    m = metric.upper()

    sql = """
        SELECT
            fr.month_index,
            fr.bucket_index,
            AVG(fr.probability) AS avg_prob
        FROM forecasts_raw fr
        JOIN questions q ON q.question_id = fr.question_id
        WHERE upper(q.hazard_code) = ?
          AND upper(q.metric) = ?
          AND fr.model_name = ?
          AND fr.month_index BETWEEN 1 AND 6
          AND fr.bucket_index BETWEEN 1 AND ?
          AND fr.probability IS NOT NULL
        GROUP BY fr.month_index, fr.bucket_index
        ORDER BY fr.month_index, fr.bucket_index
    """
    try:
        rows = conn.execute(sql, [hz, m, model_name, n_buckets]).fetchall()
    except Exception as exc:
        LOGGER.warning(
            "Per-model horizon diff failed for %s/%s/%s: %s",
            hz, m, model_name, exc,
        )
        return None

    if not rows:
        return None

    # Build per-month SPD vectors
    month_spds: Dict[int, List[float]] = {}
    for month_idx, bucket_idx, avg_prob in rows:
        mi = int(month_idx)
        bi = int(bucket_idx)
        if mi not in month_spds:
            month_spds[mi] = [0.0] * n_buckets
        if 1 <= bi <= n_buckets:
            month_spds[mi][bi - 1] = float(avg_prob or 0)

    if 1 not in month_spds or 6 not in month_spds:
        # Fall back to bucket-5 stdev if we don't have both endpoints
        return None

    spd_m1 = np.array(month_spds[1])
    spd_m6 = np.array(month_spds[6])
    jsd = _js_divergence(spd_m1, spd_m6)

    # Also compute legacy bucket-5 stdev for backwards compatibility
    b5_probs = [month_spds[mi][4] for mi in sorted(month_spds.keys()) if mi in month_spds]
    b5_stdev = float(np.std(b5_probs)) if b5_probs else 0.0

    return {
        "jsd_m1_m6": jsd,
        "b5_stdev": b5_stdev,
        "flat": jsd < 0.005,  # JSD threshold for "effectively identical"
        "month_spds": {str(mi): spd for mi, spd in sorted(month_spds.items())},
    }


# ---------------------------------------------------------------------------
# Part D — Longitudinal feedback diagnostic
# ---------------------------------------------------------------------------

MIN_MONTHS_FOR_IMPACT = 6


def _compute_advice_impact(
    conn: Any, hazard_code: str, metric: str,
) -> Optional[Dict[str, Any]]:
    """Check if model Brier scores improved after advice was first injected.

    Compares average Brier in the 3 months before first advice vs the most
    recent 3 months. Only activates when >= 6 months of data exist after
    the first advice date.
    """
    hz = hazard_code.upper()
    m = metric.upper()

    # Find earliest advice date for this hazard/metric
    sql_first = """
        SELECT MIN(as_of_month) FROM calibration_advice
        WHERE hazard_code = ? AND metric = ? AND model_name = '__shared__'
          AND as_of_month > '2000-01'
    """
    try:
        row = conn.execute(sql_first, [hz, m]).fetchone()
    except Exception:
        return None

    if not row or not row[0]:
        return None

    first_advice_month = row[0]

    # Check we have enough months since first advice
    sql_months = """
        SELECT COUNT(DISTINCT r.observed_month)
        FROM resolutions r
        JOIN questions q ON q.question_id = r.question_id
        WHERE upper(q.hazard_code) = ? AND upper(q.metric) = ?
          AND r.observed_month >= ?
    """
    try:
        row = conn.execute(sql_months, [hz, m, first_advice_month]).fetchone()
    except Exception:
        return None

    if not row or (row[0] or 0) < MIN_MONTHS_FOR_IMPACT:
        return None

    # Compare pre-advice vs post-advice Brier per model
    sql_compare = """
        SELECT
            s.model_name,
            AVG(CASE WHEN r.observed_month < ? THEN s.value END) AS pre_brier,
            AVG(CASE WHEN r.observed_month >= ? THEN s.value END) AS post_brier,
            COUNT(CASE WHEN r.observed_month < ? THEN 1 END) AS n_pre,
            COUNT(CASE WHEN r.observed_month >= ? THEN 1 END) AS n_post
        FROM scores s
        JOIN questions q ON q.question_id = s.question_id
        JOIN resolutions r ON r.question_id = s.question_id AND r.horizon_m = s.horizon_m
        WHERE s.score_type = 'brier'
          AND upper(q.hazard_code) = ?
          AND upper(q.metric) = ?
          AND s.model_name IS NOT NULL
        GROUP BY s.model_name
    """
    try:
        rows = conn.execute(
            sql_compare,
            [first_advice_month, first_advice_month,
             first_advice_month, first_advice_month, hz, m],
        ).fetchall()
    except Exception:
        return None

    if not rows:
        return None

    models = []
    for row in rows:
        name, pre, post, n_pre, n_post = row
        if pre is not None and post is not None and n_pre >= 3 and n_post >= 3:
            models.append({
                "name": name,
                "pre_brier": float(pre),
                "post_brier": float(post),
                "delta": float(post) - float(pre),
                "n_pre": int(n_pre),
                "n_post": int(n_post),
            })

    if not models:
        return None

    return {
        "first_advice_month": first_advice_month,
        "models": models,
    }


def _compute_rc_conditional(
    conn: Any, hazard_code: str, metric: str,
) -> Optional[Dict[str, Any]]:
    """Compare tail mass at RC L0 vs RC L2+."""
    hz = hazard_code.upper()
    m = metric.upper()

    if not _table_exists(conn, "hs_triage"):
        return None

    tail_bin_1, tail_bin_2 = _tail_bins(m)

    sql = """
        WITH forecast_tail AS (
            SELECT
                fe.question_id,
                fe.horizon_m,
                SUM(CASE WHEN fe.class_bin IN (?, ?) THEN fe.p ELSE 0 END) AS tail_prob
            FROM forecasts_ensemble fe
            JOIN questions q ON q.question_id = fe.question_id
            WHERE upper(q.hazard_code) = ?
              AND upper(q.metric) = ?
              AND fe.class_bin IS NOT NULL
              AND fe.p IS NOT NULL
            GROUP BY fe.question_id, fe.horizon_m
        )
        SELECT
            CASE WHEN ht.regime_change_level >= 2 THEN 'rc_elevated'
                 ELSE 'rc_normal' END AS rc_group,
            AVG(ft.tail_prob) AS avg_tail,
            COUNT(*) AS n
        FROM forecast_tail ft
        JOIN questions q ON q.question_id = ft.question_id
        JOIN hs_triage ht ON ht.run_id = q.hs_run_id
            AND ht.iso3 = q.iso3
            AND ht.hazard_code = q.hazard_code
        WHERE ht.regime_change_level IS NOT NULL
        GROUP BY rc_group
    """
    try:
        rows = conn.execute(sql, [tail_bin_1, tail_bin_2, hz, m]).fetchall()
    except Exception as exc:
        LOGGER.warning("RC-conditional query failed for %s/%s: %s", hz, m, exc)
        return None

    if not rows:
        return None

    result: Dict[str, Any] = {}
    for row in rows:
        group = str(row[0])
        result[group] = {"avg_tail": float(row[1] or 0), "n": int(row[2] or 0)}

    if not result:
        return None
    return result


def _compute_eiv_accuracy(
    conn: Any, hazard_code: str, metric: str,
) -> Optional[Dict[str, Any]]:
    """EIV accuracy stats from eiv_scores table."""
    hz = hazard_code.upper()
    m = metric.upper()

    n_resolved = _count_resolved(conn, hazard_code, metric)
    if n_resolved < MIN_QUESTIONS:
        return None

    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS n,
                AVG(log_ratio_err) AS mean_log_ratio_err,
                MEDIAN(log_ratio_err) AS median_log_ratio_err,
                AVG(CASE WHEN within_20pct THEN 1.0 ELSE 0.0 END) AS frac_within_20,
                AVG(eiv_forecast) AS mean_eiv,
                AVG(actual_value) AS mean_actual
            FROM eiv_scores e
            JOIN questions q ON q.question_id = e.question_id
            WHERE upper(q.hazard_code) = ?
              AND upper(e.metric) = ?
              AND e.model_name = '__ensemble__'
            """,
            [hz, m],
        ).fetchone()
    except Exception as exc:
        LOGGER.warning("EIV accuracy query failed for %s/%s: %s", hz, m, exc)
        return None

    if not row or row[0] == 0:
        return None

    return {
        "n_scored": int(row[0]),
        "mean_log_ratio_err": float(row[1] or 0),
        "median_log_ratio_err": float(row[2] or 0),
        "frac_within_20pct": float(row[3] or 0),
        "mean_eiv_forecast": float(row[4] or 0),
        "mean_actual": float(row[5] or 0),
        "systematic_bias": "over" if (row[4] or 0) > (row[5] or 0) * 1.2 else (
            "under" if (row[4] or 0) < (row[5] or 0) * 0.8 else "neutral"
        ),
    }


def _compute_centroid_drift(
    conn: Any, hazard_code: str, metric: str,
) -> Optional[Dict[str, Any]]:
    """Compare current centroids to defaults to detect drift."""
    from pythia.buckets import get_bucket_specs as _get_specs

    hz = hazard_code.upper()
    m = metric.upper()
    specs = list(_get_specs(m))
    if not specs:
        return None

    try:
        rows = conn.execute(
            """
            SELECT bucket_index, centroid, as_of_month
            FROM bucket_centroids
            WHERE (upper(hazard_code) = ? OR hazard_code = '*')
              AND upper(metric) = ?
              AND as_of_month IS NOT NULL
            ORDER BY hazard_code DESC, as_of_month DESC
            """,
            [hz, m],
        ).fetchall()
    except Exception:
        return None

    if not rows:
        return None  # No EMA-updated centroids yet

    # Get the latest centroid per bucket
    latest_by_bucket: Dict[int, Dict[str, Any]] = {}
    for bi, c, aom in rows:
        bi = int(bi)
        if bi not in latest_by_bucket:
            latest_by_bucket[bi] = {"centroid": float(c), "as_of_month": aom}

    drifts = []
    for spec in specs:
        if spec.idx in latest_by_bucket:
            current = latest_by_bucket[spec.idx]["centroid"]
            default = float(spec.centroid)
            pct_change = ((current - default) / max(default, 1.0)) * 100.0
            drifts.append({
                "bucket": spec.idx,
                "label": spec.label,
                "default_centroid": default,
                "current_centroid": current,
                "pct_change": round(pct_change, 1),
                "as_of_month": latest_by_bucket[spec.idx]["as_of_month"],
            })

    if not drifts:
        return None

    return {
        "hazard_code": hz,
        "metric": m,
        "bucket_drifts": drifts,
        "max_abs_pct_drift": max(abs(d["pct_change"]) for d in drifts),
    }


def _compute_views_benchmark(
    conn: Any, hazard_code: str, metric: str,
) -> Optional[Dict[str, Any]]:
    """Load ViEWS Brier score for comparison against ensemble/models."""
    hz = hazard_code.upper()
    m = metric.upper()

    # Only relevant for ACE/FATALITIES (ViEWS only forecasts fatalities)
    if hz != "ACE" or m != "FATALITIES":
        return None

    sql = """
        SELECT
            AVG(s.value) AS avg_brier,
            COUNT(*) AS n_scores
        FROM scores s
        JOIN questions q ON q.question_id = s.question_id
        WHERE s.score_type = 'brier'
          AND s.model_name = '__ext_views'
          AND upper(q.hazard_code) = ?
          AND upper(q.metric) = ?
    """
    try:
        row = conn.execute(sql, [hz, m]).fetchone()
    except Exception:
        return None

    if not row or not row[0] or row[1] < 5:
        return None

    return {
        "avg_brier": float(row[0]),
        "n_scores": int(row[1]),
    }


# ---------------------------------------------------------------------------
# Advice formatting
# ---------------------------------------------------------------------------

def _format_advice(
    findings: Dict[str, Any],
    hazard_code: str,
    metric: str,
    as_of_month: str,
    n_resolved: int,
) -> str:
    """Format calibration findings into prompt-ready advice text."""
    hz_label = HAZARD_LABELS.get(hazard_code, hazard_code)
    m_label = METRIC_LABELS.get(metric, metric)
    key = f"{hazard_code}/{metric}"
    lines: List[str] = []

    header = f"{hz_label} \u2014 {m_label} ({key}, {as_of_month}, n={n_resolved} resolved):"
    if hazard_code == "*":
        header = f"OVERALL ENSEMBLE ({as_of_month}):"
    lines.append(header)
    lines.append("")

    # Tail coverage
    tc = findings.get("tail_coverage")
    if tc:
        avg_tail = tc["avg_assigned_tail"] * 100
        actual = tc["actual_tail_rate"] * 100
        frac_starved = tc["frac_starved"] * 100

        lines.append(
            f"TAIL: Ensemble assigns avg {avg_tail:.1f}% to buckets 4-5; "
            f"actual resolution rate is {actual:.1f}%."
        )
        if actual > avg_tail + 3:
            gap = actual - avg_tail
            target = max(avg_tail + gap * 0.6, actual * 0.7)
            lines.append(
                f"ACTION: Increase bucket 4+5 combined probability to at least "
                f"{target:.0f}% unless you have strong evidence against tail outcomes."
            )
        elif avg_tail > actual + 5:
            lines.append(
                "ACTION: Tail probabilities may be slightly overweighted. "
                "Verify with reference class before assigning high tail mass."
            )
        else:
            lines.append("Tail calibration is reasonable. No correction needed.")
        lines.append("")

    # Bucket calibration
    bc = findings.get("bucket_calibration")
    if bc:
        lines.append("CALIBRATION BY BUCKET:")
        worst_gap = 0.0
        worst_bucket = ""
        for entry in bc:
            assigned = entry["mean_assigned"] * 100
            actual = entry["actual_rate"] * 100
            gap = assigned - actual
            direction = "overconfident" if gap > 2 else ("underconfident" if gap < -2 else "well calibrated")
            suffix = f"by {abs(gap):.0f}pp" if abs(gap) > 2 else ""
            lines.append(
                f"  {entry['class_bin']:>12s}: assigned {assigned:.0f}%, "
                f"actual {actual:.0f}% -> {direction} {suffix}".rstrip()
            )
            if abs(gap) > abs(worst_gap):
                worst_gap = gap
                worst_bucket = entry["class_bin"]
        if abs(worst_gap) > 3:
            if worst_gap > 0:
                lines.append(
                    f"ACTION: Reduce probability on bucket {worst_bucket} by ~{abs(worst_gap):.0f}pp "
                    "and redistribute to under-predicted buckets."
                )
            else:
                lines.append(
                    f"ACTION: Increase probability on bucket {worst_bucket} by ~{abs(worst_gap):.0f}pp."
                )
        lines.append("")

    # Per-model Brier
    mb = findings.get("per_model_brier")
    if mb:
        parts = []
        if mb.get("best"):
            parts.append(f"{mb['best']['name']} (Brier {mb['best']['brier']:.3f}, best)")
        if mb.get("worst"):
            parts.append(f"{mb['worst']['name']} (Brier {mb['worst']['brier']:.3f}, worst)")
        if parts:
            lines.append(f"MODELS: {', '.join(parts)}.")
        if mb.get("ensemble_brier") is not None:
            lines.append(
                f"Ensemble mean Brier: {mb['ensemble_brier']:.3f}."
            )
        lines.append("")

    # Month position bias (with JSD)
    mpb = findings.get("month_position_bias")
    if mpb:
        jsd = mpb.get("jsd_m1_m6")
        if mpb["flat"]:
            if jsd is not None:
                lines.append(
                    f"HORIZONS: Month-1 vs month-6 SPDs are nearly identical "
                    f"(JS divergence = {jsd:.4f}, bucket-5 stdev = {mpb['stdev']*100:.1f}pp)."
                )
            else:
                lines.append(
                    f"HORIZONS: Models show <{mpb['stdev']*100:.1f}pp variation in "
                    f"top-bucket probability across months 1-6 (nearly flat)."
                )
            lines.append(
                "ACTION: Differentiate across horizons. Widen uncertainty for "
                "later months (4-6); do not copy month 1 SPD to all months."
            )
        else:
            if jsd is not None:
                lines.append(
                    f"HORIZONS: Good differentiation (JS divergence = {jsd:.4f}, "
                    f"bucket-5 stdev = {mpb['stdev']*100:.1f}pp)."
                )
            else:
                lines.append(
                    f"HORIZONS: Models differentiate across months "
                    f"(stdev {mpb['stdev']*100:.1f}pp). Good horizon differentiation."
                )
        lines.append("")

    # RC-conditional
    rc = findings.get("rc_conditional")
    if rc:
        normal = rc.get("rc_normal", {})
        elevated = rc.get("rc_elevated", {})
        if normal and elevated:
            norm_tail = normal["avg_tail"] * 100
            elev_tail = elevated["avg_tail"] * 100
            lines.append(
                f"RC-CONDITIONAL: At RC L0-1, avg tail mass = {norm_tail:.1f}%. "
                f"At RC L2+, avg tail mass = {elev_tail:.1f}%."
            )
            if elev_tail < norm_tail + 5:
                lines.append(
                    "ACTION: When RC >= L2, tail mass should be meaningfully "
                    "higher than at RC L0. Ensure at least 5pp increase in "
                    "bucket 4+5 combined probability."
                )
            else:
                lines.append("RC-conditional adjustment looks appropriate.")
            lines.append("")

    # Advice impact (longitudinal feedback)
    ai = findings.get("advice_impact")
    if ai:
        lines.append(f"LONGITUDINAL (since advice first injected {ai['first_advice_month']}):")
        for model in ai["models"]:
            direction = "improved" if model["delta"] < -0.01 else (
                "worsened" if model["delta"] > 0.01 else "unchanged"
            )
            lines.append(
                f"  {model['name']}: Brier {model['pre_brier']:.3f} -> "
                f"{model['post_brier']:.3f} ({direction}, "
                f"n_pre={model['n_pre']}, n_post={model['n_post']})"
            )
        lines.append("")

    # EIV accuracy (Phase 3)
    eiv = findings.get("eiv_accuracy")
    if eiv:
        frac = eiv["frac_within_20pct"] * 100
        lines.append(
            f"EIV ACCURACY: Ensemble EIV predictions are within 20% of actual "
            f"values {frac:.0f}% of the time (n={eiv['n_scored']}). "
            f"Mean log-ratio error: {eiv['mean_log_ratio_err']:.3f}. "
            f"Systematic bias: {eiv['systematic_bias']}."
        )
        if eiv["systematic_bias"] == "over":
            lines.append(
                "ACTION: EIV forecasts systematically overestimate impact. "
                "Shift probability mass toward lower buckets."
            )
        elif eiv["systematic_bias"] == "under":
            lines.append(
                "ACTION: EIV forecasts systematically underestimate impact. "
                "Shift probability mass toward higher buckets."
            )
        lines.append("")

    # Centroid drift (Phase 3)
    cd = findings.get("centroid_drift")
    if cd:
        drift_parts = []
        for d in cd["bucket_drifts"]:
            drift_parts.append(
                f"bucket {d['bucket']} ({d['label']}) shifted {d['pct_change']:+.1f}%"
            )
        lines.append(
            f"CENTROID DRIFT: {', '.join(drift_parts)}. "
            f"Largest drift: {cd['max_abs_pct_drift']:.1f}%."
        )
        lines.append("")

    # ViEWS benchmark
    vb = findings.get("views_benchmark")
    if vb:
        lines.append(
            f"EXTERNAL BENCHMARK: ViEWS fatality forecasts scored Brier "
            f"{vb['avg_brier']:.3f} (n={vb['n_scores']}) on this hazard/metric."
        )
        # Compare to ensemble
        mb = findings.get("per_model_brier")
        if mb and mb.get("ensemble_brier") is not None:
            ens_brier = mb["ensemble_brier"]
            if vb["avg_brier"] < ens_brier - 0.02:
                lines.append(
                    "ViEWS outperforms the Pythia ensemble. Treat ViEWS conflict "
                    "forecasts as a strong prior — anchor toward their predictions "
                    "and require specific evidence to deviate."
                )
            elif vb["avg_brier"] > ens_brier + 0.02:
                lines.append(
                    "Pythia ensemble outperforms ViEWS. Use ViEWS forecasts as "
                    "one input among several, not as a strong anchor."
                )
            else:
                lines.append(
                    "ViEWS and Pythia ensemble perform similarly. Use ViEWS "
                    "as a credible cross-check."
                )
        lines.append("")

    text = "\n".join(lines).strip()
    if len(text) > MAX_ADVICE_CHARS:
        text = text[:MAX_ADVICE_CHARS - 20] + "\n...[truncated]"
    return text


def _format_per_model_advice(
    model_name: str,
    findings: Dict[str, Any],
    hazard_code: str,
    metric: str,
    as_of_month: str,
    n_scored: int,
    shared_findings: Dict[str, Any],
) -> str:
    """Format model-specific calibration advice."""
    hz_label = HAZARD_LABELS.get(hazard_code, hazard_code)
    m_label = METRIC_LABELS.get(metric, metric)
    lines: List[str] = []

    lines.append(
        f"MODEL-SPECIFIC CALIBRATION ({model_name}) — "
        f"{hz_label}/{m_label} ({as_of_month}, n={n_scored} scored):"
    )
    lines.append("")

    # 1. Relative standing
    mb = findings.get("model_brier")
    if mb:
        lines.append(
            f"YOUR PERFORMANCE: Brier score {mb['brier']:.3f} "
            f"(rank {mb['rank']} of {mb['total_models']}). "
            f"Best: {mb['best_name']} at {mb['best_brier']:.3f}."
        )
        lines.append("")

    # 2. Per-model tail coverage
    tc = findings.get("tail_coverage")
    if tc:
        avg_tail = tc["avg_assigned_tail"] * 100
        actual = tc["actual_tail_rate"] * 100
        lines.append(
            f"YOUR TAIL: You assign avg {avg_tail:.1f}% to buckets 4-5; "
            f"actual rate is {actual:.1f}%."
        )
        if actual > avg_tail + 3:
            gap = actual - avg_tail
            target = max(avg_tail + gap * 0.6, actual * 0.7)
            lines.append(
                f"ACTION: Increase your bucket 4+5 combined to at least {target:.0f}%."
            )
        elif avg_tail > actual + 5:
            lines.append(
                "ACTION: You may be over-assigning tail mass. Verify with base rates."
            )
        lines.append("")

    # 3. Per-model bucket calibration
    bc = findings.get("bucket_calibration")
    if bc:
        lines.append("YOUR CALIBRATION BY BUCKET:")
        worst_gap = 0.0
        worst_bucket = ""
        for entry in bc:
            assigned = entry["mean_assigned"] * 100
            actual = entry["actual_rate"] * 100
            gap = assigned - actual
            direction = (
                "overconfident" if gap > 2
                else ("underconfident" if gap < -2 else "well calibrated")
            )
            suffix = f"by {abs(gap):.0f}pp" if abs(gap) > 2 else ""
            lines.append(
                f"  Bucket {entry['bucket_index']:d} ({entry['class_bin']:>12s}): "
                f"you assign {assigned:.0f}%, actual {actual:.0f}% "
                f"-> {direction} {suffix}".rstrip()
            )
            if abs(gap) > abs(worst_gap):
                worst_gap = gap
                worst_bucket = entry["class_bin"]
        if abs(worst_gap) > 3:
            if worst_gap > 0:
                lines.append(
                    f"ACTION: Reduce your probability on bucket {worst_bucket} "
                    f"by ~{abs(worst_gap):.0f}pp."
                )
            else:
                lines.append(
                    f"ACTION: Increase your probability on bucket {worst_bucket} "
                    f"by ~{abs(worst_gap):.0f}pp."
                )
        lines.append("")

    # 4. Per-model horizon differentiation
    hd = findings.get("horizon_diff")
    if hd:
        jsd = hd["jsd_m1_m6"]
        if hd["flat"]:
            lines.append(
                f"YOUR HORIZONS: Your month-1 vs month-6 SPDs are nearly identical "
                f"(JS divergence = {jsd:.4f})."
            )
            lines.append(
                "ACTION: Widen uncertainty for months 4-6. Do not copy month 1 to all months."
            )
        else:
            lines.append(
                f"YOUR HORIZONS: Good differentiation (JS divergence = {jsd:.4f})."
            )
        lines.append("")

    # ViEWS comparison for this model
    vb = shared_findings.get("views_benchmark")
    mb = findings.get("model_brier")
    if vb and mb:
        model_brier = mb["brier"]
        views_brier = vb["avg_brier"]
        if views_brier < model_brier - 0.03:
            lines.append(
                f"NOTE: ViEWS (Brier {views_brier:.3f}) outperforms you "
                f"(Brier {model_brier:.3f}) on this hazard. Give strong weight "
                f"to ViEWS conflict forecasts when they are provided."
            )
        elif model_brier < views_brier - 0.03:
            lines.append(
                f"NOTE: You (Brier {model_brier:.3f}) outperform ViEWS "
                f"(Brier {views_brier:.3f}). Use ViEWS as supplementary context, "
                f"not as a primary anchor."
            )
        lines.append("")

    text = "\n".join(lines).strip()
    if len(text) > MAX_ADVICE_CHARS:
        text = text[:MAX_ADVICE_CHARS - 20] + "\n...[truncated]"
    return text


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def _upsert_advice(
    conn: Any,
    as_of_month: str,
    hazard_code: str,
    metric: str,
    advice_text: str,
    findings: Dict[str, Any],
    model_name: str = "__shared__",
    advice_version: str = "v1",
) -> None:
    """Write (or replace) a calibration_advice row."""
    conn.execute(
        "DELETE FROM calibration_advice "
        "WHERE as_of_month = ? AND hazard_code = ? AND metric = ? AND model_name = ?",
        [as_of_month, hazard_code, metric, model_name],
    )

    findings_safe = {}
    for k, v in findings.items():
        try:
            json.dumps(v)
            findings_safe[k] = v
        except (TypeError, ValueError):
            findings_safe[k] = str(v)

    conn.execute(
        """
        INSERT OR REPLACE INTO calibration_advice
            (as_of_month, hazard_code, metric, model_name, advice, findings_json,
             advice_version, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            as_of_month,
            hazard_code,
            metric,
            model_name,
            advice_text,
            json.dumps(findings_safe),
            advice_version,
            datetime.utcnow(),
        ],
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_calibration_advice(
    db_url: str,
    as_of: Optional[date] = None,
    seed_defaults: bool = False,
    advice_version: str = "v1",
) -> None:
    """Generate and write calibration advice for all hazard/metric pairs."""
    if as_of is None:
        as_of = date.today()
    as_of_month = as_of.strftime("%Y-%m")

    if not duckdb_io.DUCKDB_AVAILABLE:
        raise RuntimeError(duckdb_io.duckdb_unavailable_reason())

    conn = duckdb_io.get_db(db_url or duckdb_io.DEFAULT_DB_URL)

    try:
        _ensure_findings_json_column(conn)

        # Seed defaults if requested (uses sentinel date so real data supersedes)
        if seed_defaults:
            _seed_default_advice(conn)

        # Check required tables
        for table in ("scores", "resolutions", "questions", "forecasts_ensemble"):
            if not _table_exists(conn, table):
                LOGGER.info(
                    "generate_calibration_advice: table %s not found; skipping.",
                    table,
                )
                return

        if _row_count(conn, "scores") == 0:
            LOGGER.info("generate_calibration_advice: scores table is empty; skipping.")
            return

        pairs = _discover_hazard_metric_pairs(conn)
        if not pairs:
            LOGGER.info("generate_calibration_advice: no resolved hazard/metric pairs found.")
            return

        total_written = 0
        all_model_briers: List[Dict[str, Any]] = []

        for hazard_code, metric in pairs:
            n_resolved = _count_resolved(conn, hazard_code, metric)
            if n_resolved < MIN_QUESTIONS:
                LOGGER.info(
                    "Skipping %s/%s: only %d resolved questions (need %d).",
                    hazard_code, metric, n_resolved, MIN_QUESTIONS,
                )
                continue

            findings: Dict[str, Any] = {}
            findings["tail_coverage"] = _compute_tail_coverage(conn, hazard_code, metric)
            findings["bucket_calibration"] = _compute_bucket_calibration(
                conn, hazard_code, metric,
            )
            findings["per_model_brier"] = _compute_per_model_brier(
                conn, hazard_code, metric,
            )
            findings["month_position_bias"] = _compute_month_position_bias(
                conn, hazard_code, metric,
            )
            findings["rc_conditional"] = _compute_rc_conditional(
                conn, hazard_code, metric,
            )

            # ViEWS external benchmark (ACE/FATALITIES only)
            findings["views_benchmark"] = _compute_views_benchmark(
                conn, hazard_code, metric,
            )

            # Longitudinal feedback diagnostic (Part D)
            findings["advice_impact"] = _compute_advice_impact(conn, hazard_code, metric)

            # EIV accuracy and centroid drift (Phase 3 — auto-activates)
            if _table_exists(conn, "eiv_scores") and _row_count(conn, "eiv_scores") > 0:
                findings["eiv_accuracy"] = _compute_eiv_accuracy(
                    conn, hazard_code, metric,
                )
                findings["centroid_drift"] = _compute_centroid_drift(
                    conn, hazard_code, metric,
                )

            advice_text = _format_advice(
                findings, hazard_code, metric, as_of_month, n_resolved,
            )
            _upsert_advice(
                conn, as_of_month, hazard_code, metric, advice_text, findings,
                advice_version=advice_version,
            )
            total_written += 1

            if findings.get("per_model_brier"):
                all_model_briers.append(findings["per_model_brier"])

            LOGGER.info(
                "Wrote calibration advice for %s/%s (%d chars).",
                hazard_code, metric, len(advice_text),
            )

            # --- Per-model advice (after shared advice) ---
            model_brier_data = findings.get("per_model_brier")
            if model_brier_data and model_brier_data.get("all_models"):
                all_models = model_brier_data["all_models"]
                sorted_models = sorted(all_models, key=lambda x: x["brier"])

                for model_info in all_models:
                    mname = model_info["name"]
                    n_scored = model_info.get("n", 0)

                    if n_scored < MIN_QUESTIONS_PER_MODEL:
                        LOGGER.info(
                            "Skipping per-model advice for %s on %s/%s: "
                            "only %d scored (need %d).",
                            mname, hazard_code, metric,
                            n_scored, MIN_QUESTIONS_PER_MODEL,
                        )
                        continue

                    model_findings: Dict[str, Any] = {}

                    # Relative standing
                    rank = next(
                        (i + 1 for i, m in enumerate(sorted_models) if m["name"] == mname),
                        None,
                    )
                    model_findings["model_brier"] = {
                        "brier": model_info["brier"],
                        "rank": rank,
                        "total_models": len(all_models),
                        "best_name": sorted_models[0]["name"],
                        "best_brier": sorted_models[0]["brier"],
                    }

                    # Per-model tail coverage
                    model_findings["tail_coverage"] = _compute_per_model_tail_coverage(
                        conn, hazard_code, metric, mname,
                    )

                    # Per-model bucket calibration
                    model_findings["bucket_calibration"] = _compute_per_model_bucket_calibration(
                        conn, hazard_code, metric, mname,
                    )

                    # Per-model horizon differentiation (JSD-based)
                    model_findings["horizon_diff"] = _compute_per_model_horizon_diff(
                        conn, hazard_code, metric, mname,
                    )

                    model_advice = _format_per_model_advice(
                        model_name=mname,
                        findings=model_findings,
                        hazard_code=hazard_code,
                        metric=metric,
                        as_of_month=as_of_month,
                        n_scored=n_scored,
                        shared_findings=findings,
                    )

                    _upsert_advice(
                        conn, as_of_month, hazard_code, metric,
                        model_advice, model_findings,
                        model_name=mname,
                        advice_version=advice_version,
                    )
                    total_written += 1

                    LOGGER.info(
                        "Wrote per-model advice for %s on %s/%s (%d chars).",
                        mname, hazard_code, metric, len(model_advice),
                    )

        # Write global row
        global_findings = _compute_global_findings(all_model_briers)
        if global_findings:
            global_advice = _format_advice(
                global_findings, "*", "*", as_of_month, 0,
            )
            _upsert_advice(
                conn, as_of_month, "*", "*", global_advice, global_findings,
                advice_version=advice_version,
            )
            total_written += 1

        LOGGER.info(
            "generate_calibration_advice: wrote %d advice rows for as_of_month=%s.",
            total_written, as_of_month,
        )

    finally:
        duckdb_io.close_db(conn)


def _compute_global_findings(
    all_model_briers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate across all hazard/metrics for global stats."""
    if not all_model_briers:
        return {}

    # Aggregate per-model scores across hazard/metrics
    model_scores: Dict[str, List[float]] = {}
    ensemble_scores: List[float] = []

    for mb in all_model_briers:
        if mb.get("ensemble_brier") is not None:
            ensemble_scores.append(mb["ensemble_brier"])
        for model in mb.get("all_models", []):
            model_scores.setdefault(model["name"], []).append(model["brier"])

    if not model_scores:
        return {}

    model_avgs = {
        name: sum(scores) / len(scores) for name, scores in model_scores.items()
    }
    sorted_models = sorted(model_avgs.items(), key=lambda x: x[1])

    result: Dict[str, Any] = {
        "per_model_brier": {
            "best": {"name": sorted_models[0][0], "brier": sorted_models[0][1]},
            "worst": {"name": sorted_models[-1][0], "brier": sorted_models[-1][1]},
            "ensemble_brier": (
                sum(ensemble_scores) / len(ensemble_scores)
                if ensemble_scores else None
            ),
            "all_models": [
                {"name": n, "brier": b} for n, b in sorted_models
            ],
        }
    }
    return result


def _seed_default_advice(conn: Any) -> None:
    """Insert hand-written seed advice with sentinel date (superseded by real data)."""
    seed_month = "2000-01"

    for (hz, m), advice_text in SEED_ADVICE.items():
        try:
            existing = conn.execute(
                """
                SELECT 1 FROM calibration_advice
                WHERE as_of_month = ? AND hazard_code = ? AND metric = ?
                """,
                [seed_month, hz, m],
            ).fetchone()
            if existing:
                continue
            conn.execute(
                """
                INSERT INTO calibration_advice
                    (as_of_month, hazard_code, metric, advice, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [seed_month, hz, m, advice_text, datetime.utcnow()],
            )
            LOGGER.info("Seeded default advice for %s/%s.", hz, m)
        except Exception as exc:
            LOGGER.warning("Failed to seed advice for %s/%s: %s", hz, m, exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate concrete calibration advice from scoring data.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="DuckDB URL (e.g. duckdb:///data/resolver.duckdb)",
    )
    parser.add_argument(
        "--seed-defaults",
        action="store_true",
        default=False,
        help="Insert hand-written seed advice for bootstrapping.",
    )
    parser.add_argument(
        "--advice-version",
        default="v1",
        help="Version tag for this advice generation run (for A/B testing).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    db_url = args.db_url or _get_db_url_from_config()
    generate_calibration_advice(
        db_url=db_url,
        seed_defaults=args.seed_defaults,
        advice_version=args.advice_version,
    )


if __name__ == "__main__":
    main()
