# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Score ViEWS conflict forecasts against Pythia resolutions.

Converts ViEWS point forecasts into synthetic 5-bucket SPDs, then scores
them using the same Brier/log/CRPS functions as compute_scores.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pythia.config import load as load_cfg
from pythia.tools.compute_scores import (
    FATAL_THRESHOLDS,
    SPD_CLASS_BINS_FATALITIES,
    _brier,
    _log_score,
    _crps_like,
    _bucket_index,
)
from resolver.db import duckdb_io


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())

# Sentinel model_name for ViEWS in the scores table.
# Prefixed with '__ext_' to distinguish from Pythia ensemble members.
VIEWS_MODEL_NAME = "__ext_views"

# Log-normal sigma parameter controlling spread around the point forecast.
# This should be calibrated empirically once enough resolutions exist.
# Initial value of 1.0 gives moderate spread.
# Too narrow -> overconfident SPDs -> high Brier on tail events.
# Too wide -> uninformative SPDs -> moderate Brier everywhere.
LOGNORMAL_SIGMA = 1.0

# Minimum point forecast value for log-normal conversion.
# Values below this are treated as "near-zero" and get a spike on bucket 1.
MIN_POINT_FOR_LOGNORMAL = 0.5


def _table_exists(conn, name: str) -> bool:
    try:
        conn.execute(f"PRAGMA table_info('{name}')").fetchall()
        return True
    except Exception:
        return False


def _lognorm_cdf(x: float, mu: float, sigma: float) -> float:
    """Log-normal CDF without scipy dependency."""
    if x <= 0:
        return 0.0
    z = (math.log(x) - mu) / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def point_to_spd_fatalities(
    point_forecast: float,
    sigma: float = LOGNORMAL_SIGMA,
) -> List[float]:
    """Convert a ViEWS fatalities point forecast into a 5-bucket SPD.

    Uses a log-normal distribution centered on the point forecast to
    compute probability mass in each FATALITIES bucket.

    For near-zero forecasts (< MIN_POINT_FOR_LOGNORMAL), returns a
    distribution heavily concentrated on bucket 1.

    Args:
        point_forecast: Expected fatalities (continuous, non-negative).
        sigma: Log-normal sigma (spread parameter). Higher = more uncertain.

    Returns:
        List of 5 probabilities summing to 1.0, one per bucket.
    """
    thresholds = FATAL_THRESHOLDS  # [0, 5, 25, 100, 500, inf]
    n_buckets = len(thresholds) - 1  # 5

    pf = max(float(point_forecast), 0.0)

    # Near-zero: spike on bucket 1
    if pf < MIN_POINT_FOR_LOGNORMAL:
        spd = [0.0] * n_buckets
        spd[0] = 0.90
        spd[1] = 0.05
        spd[2] = 0.03
        spd[3] = 0.015
        spd[4] = 0.005
        return spd

    # Log-normal: mu chosen so that E[X] = pf
    # For log-normal: E[X] = exp(mu + sigma^2/2)
    # So: mu = ln(pf) - sigma^2/2
    mu = math.log(pf) - (sigma ** 2) / 2.0

    spd = []
    for i in range(n_buckets):
        lo = thresholds[i]
        hi = thresholds[i + 1]

        if hi == float("inf"):
            # Last bucket: P(X >= lo)
            p = 1.0 - _lognorm_cdf(lo, mu, sigma)
        elif lo == 0.0:
            # First bucket: P(X < hi)
            p = _lognorm_cdf(hi, mu, sigma)
        else:
            # Middle bucket: P(lo <= X < hi)
            p = _lognorm_cdf(hi, mu, sigma) - _lognorm_cdf(lo, mu, sigma)

        spd.append(max(p, 1e-9))  # Floor to avoid log(0) in scoring

    # Normalize
    total = sum(spd)
    return [p / total for p in spd]


def _load_views_forecast_pairs(conn) -> List[Dict]:
    """Load matched pairs with latest ViEWS forecast per question/horizon.

    Returns list of dicts with keys:
        question_id, horizon_m, iso3, resolved_value, metric,
        views_value, views_issue_date, views_model_version
    """
    sql = """
        WITH ranked_views AS (
            SELECT
                cf.iso3,
                cf.lead_months,
                strftime(cf.target_month, '%Y-%m') AS target_ym,
                cf.value,
                cf.forecast_issue_date,
                cf.model_version,
                ROW_NUMBER() OVER (
                    PARTITION BY cf.iso3, cf.lead_months, strftime(cf.target_month, '%Y-%m')
                    ORDER BY cf.forecast_issue_date DESC
                ) AS rn
            FROM conflict_forecasts cf
            WHERE cf.source = 'views'
              AND upper(cf.metric) = 'FATALITIES'
        )
        SELECT
            q.question_id,
            r.horizon_m,
            q.iso3,
            r.value AS resolved_value,
            upper(q.metric) AS metric,
            r.observed_month,
            rv.value AS views_value,
            rv.forecast_issue_date AS views_issue_date,
            rv.model_version AS views_model_version
        FROM questions q
        JOIN resolutions r
            ON q.question_id = r.question_id
        JOIN hs_runs h
            ON q.hs_run_id = h.hs_run_id
        JOIN ranked_views rv
            ON upper(rv.iso3) = upper(q.iso3)
            AND rv.lead_months = r.horizon_m
            AND rv.target_ym = r.observed_month
            AND rv.rn = 1
        WHERE upper(q.hazard_code) = 'ACE'
          AND upper(q.metric) = 'FATALITIES'
        ORDER BY q.question_id, r.horizon_m
    """
    rows = conn.execute(sql).fetchall()

    pairs = []
    for row in rows:
        pairs.append({
            "question_id": row[0],
            "horizon_m": row[1],
            "iso3": row[2],
            "resolved_value": float(row[3]),
            "metric": row[4],
            "observed_month": row[5],
            "views_value": float(row[6]),
            "views_issue_date": str(row[7]) if row[7] else None,
            "views_model_version": str(row[8]) if row[8] else None,
        })

    return pairs


def score_views(db_url: str, sigma: float = LOGNORMAL_SIGMA) -> None:
    """Score ViEWS forecasts against Pythia resolutions.

    For each matched (question, horizon) pair:
    1. Convert ViEWS point forecast -> synthetic SPD via log-normal
    2. Score the SPD with Brier, log, CRPS (same functions as compute_scores.py)
    3. Write to `scores` table with model_name = '__ext_views'
    """
    if not duckdb_io.DUCKDB_AVAILABLE:
        raise RuntimeError(duckdb_io.duckdb_unavailable_reason())

    conn = duckdb_io.get_db(db_url or duckdb_io.DEFAULT_DB_URL)

    try:
        # Ensure scores table exists (should already from compute_scores)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scores (
              question_id TEXT,
              horizon_m INTEGER,
              metric TEXT,
              score_type TEXT,
              model_name TEXT,
              value DOUBLE,
              created_at TIMESTAMP DEFAULT now()
            )
            """
        )

        # Ensure views_scored_forecasts table for audit trail
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS views_scored_forecasts (
                question_id     TEXT,
                horizon_m       INTEGER,
                views_value     DOUBLE,
                views_issue_date TEXT,
                views_model_version TEXT,
                synthetic_spd   TEXT,
                sigma_used      DOUBLE,
                resolved_value  DOUBLE,
                resolved_bucket INTEGER,
                brier           DOUBLE,
                log_score       DOUBLE,
                crps            DOUBLE,
                created_at      TIMESTAMP DEFAULT now(),
                PRIMARY KEY (question_id, horizon_m)
            )
            """
        )

        # Check prerequisites
        for table in ("conflict_forecasts", "questions", "resolutions"):
            if not _table_exists(conn, table):
                LOGGER.info(
                    "score_views: table %s not found; nothing to do.", table,
                )
                return

        pairs = _load_views_forecast_pairs(conn)
        LOGGER.info("Found %d ViEWS<>Pythia matched forecast pairs.", len(pairs))

        if not pairs:
            LOGGER.info("No ViEWS forecasts matched to resolved questions; nothing to score.")
            return

        n_scored = 0

        for pair in pairs:
            qid = pair["question_id"]
            hm = pair["horizon_m"]
            views_val = pair["views_value"]
            resolved_val = pair["resolved_value"]
            metric = pair["metric"]

            # Convert point forecast to SPD
            spd = point_to_spd_fatalities(views_val, sigma=sigma)

            # Find resolved bucket
            j = _bucket_index(resolved_val, metric)
            if j is None:
                continue

            # Score
            brier = _brier(spd, j)
            log_s = _log_score(spd, j)
            crps = _crps_like(spd, j)

            # Write scores (same table as Pythia models, distinguished by model_name)
            conn.execute(
                """
                DELETE FROM scores
                WHERE question_id = ? AND horizon_m = ? AND metric = ?
                  AND model_name = ?
                """,
                [qid, hm, metric, VIEWS_MODEL_NAME],
            )
            now = datetime.utcnow()
            conn.executemany(
                """
                INSERT INTO scores
                    (question_id, horizon_m, metric, score_type, model_name, value, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (qid, hm, metric, "brier", VIEWS_MODEL_NAME, brier, now),
                    (qid, hm, metric, "log", VIEWS_MODEL_NAME, log_s, now),
                    (qid, hm, metric, "crps", VIEWS_MODEL_NAME, crps, now),
                ],
            )

            # Write audit trail
            conn.execute(
                """
                INSERT OR REPLACE INTO views_scored_forecasts
                    (question_id, horizon_m, views_value, views_issue_date,
                     views_model_version, synthetic_spd, sigma_used,
                     resolved_value, resolved_bucket, brier, log_score, crps, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    qid, hm, views_val,
                    pair["views_issue_date"],
                    pair["views_model_version"],
                    json.dumps([round(p, 6) for p in spd]),
                    sigma,
                    resolved_val, j,
                    brier, log_s, crps,
                    now,
                ],
            )

            n_scored += 1

        LOGGER.info(
            "score_views: scored %d ViEWS forecasts (sigma=%.2f, model_name='%s').",
            n_scored, sigma, VIEWS_MODEL_NAME,
        )

    finally:
        duckdb_io.close_db(conn)


def optimize_sigma(
    db_url: str,
    sigma_range: Tuple[float, float] = (0.3, 2.5),
    n_steps: int = 20,
) -> float:
    """Find the sigma that minimizes mean Brier score for ViEWS.

    Loads all matched pairs, scores each with different sigma values,
    and returns the sigma with the lowest mean Brier.
    """
    conn = duckdb_io.get_db(db_url or duckdb_io.DEFAULT_DB_URL)

    try:
        pairs = _load_views_forecast_pairs(conn)
        if not pairs:
            LOGGER.warning("No matched pairs for sigma optimization.")
            return LOGNORMAL_SIGMA

        lo, hi = sigma_range
        step = (hi - lo) / n_steps
        best_sigma = LOGNORMAL_SIGMA
        best_mean_brier = float("inf")

        for i in range(n_steps + 1):
            sigma = lo + i * step
            brier_sum = 0.0
            n_valid = 0

            for pair in pairs:
                spd = point_to_spd_fatalities(pair["views_value"], sigma=sigma)
                j = _bucket_index(pair["resolved_value"], pair["metric"])
                if j is None:
                    continue
                brier_sum += _brier(spd, j)
                n_valid += 1

            if n_valid > 0:
                mean_brier = brier_sum / n_valid
                if mean_brier < best_mean_brier:
                    best_mean_brier = mean_brier
                    best_sigma = sigma

        LOGGER.info(
            "Optimal sigma: %.3f (mean Brier = %.4f over %d pairs).",
            best_sigma, best_mean_brier, len(pairs),
        )
        return best_sigma

    finally:
        duckdb_io.close_db(conn)


def _get_db_url_from_config() -> str:
    cfg = load_cfg()
    app_cfg = cfg.get("app", {}) if isinstance(cfg, dict) else {}
    db_url = str(app_cfg.get("db_url", "")).strip()
    if not db_url:
        db_url = duckdb_io.DEFAULT_DB_URL
    return db_url


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score ViEWS conflict forecasts against Pythia resolutions.",
    )
    parser.add_argument(
        "--db-url",
        default=None,
        help="DuckDB URL (default: from pythia.config).",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=LOGNORMAL_SIGMA,
        help=f"Log-normal sigma for point-to-SPD conversion (default: {LOGNORMAL_SIGMA}).",
    )
    parser.add_argument(
        "--optimize-sigma",
        action="store_true",
        default=False,
        help="Find optimal sigma via grid search before scoring.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    db_url = args.db_url or _get_db_url_from_config()

    sigma = args.sigma
    if args.optimize_sigma:
        sigma = optimize_sigma(db_url)
        LOGGER.info("Using optimized sigma=%.3f", sigma)

    score_views(db_url=db_url, sigma=sigma)


if __name__ == "__main__":
    main()
