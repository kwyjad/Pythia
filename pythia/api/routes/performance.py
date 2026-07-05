# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Calibration and performance-score routes:
/v1/calibration/weights, /v1/calibration/advice, /v1/performance/scores.

Endpoint functions moved verbatim from pythia.api.app (July 2026
decomposition); shared helpers come from pythia.api.core.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query

from pythia.api.core import (
    _con,
    _execute,
    _rows_from_cursor,
    _table_exists,
    _table_has_columns,
    _test_filter,
)

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/v1/calibration/weights")
def get_calibration_weights(
    hazard_code: Optional[str] = Query(None),
    metric: Optional[str] = Query(None),
    as_of_month: Optional[str] = Query(None, description="YYYY-MM; if omitted, use latest"),
):
    """
    Return calibration weights per model for the given hazard_code/metric/as_of_month.

    If as_of_month is omitted, we use the latest as_of_month in calibration_weights
    (optionally filtered by hazard_code/metric).
    """
    con = _con()

    # Resolve as_of_month if not given
    params: dict = {}
    where_bits = []

    if hazard_code:
        where_bits.append("hazard_code = :hazard_code")
        params["hazard_code"] = hazard_code.upper()
    if metric:
        where_bits.append("metric = :metric")
        params["metric"] = metric.upper()

    base_where = ""
    if where_bits:
        base_where = "WHERE " + " AND ".join(where_bits)

    if not as_of_month:
        # Pick latest as_of_month given any hazard/metric filters
        sql_latest = f"""
          SELECT as_of_month
          FROM calibration_weights
          {base_where}
          ORDER BY as_of_month DESC
          LIMIT 1
        """
        row = _execute(con, sql_latest, params).fetchone()
        if not row:
            return {"found": False, "as_of_month": None, "rows": []}
        as_of_month = row[0]

    # Now fetch rows for this as_of_month + filters
    params["as_of_month"] = as_of_month
    where_full = ["as_of_month = :as_of_month"]
    if hazard_code:
        where_full.append("hazard_code = :hazard_code")
    if metric:
        where_full.append("metric = :metric")

    sql = """
      SELECT
        as_of_month,
        hazard_code,
        metric,
        model_name,
        weight,
        n_questions,
        n_samples,
        avg_brier,
        avg_log,
        avg_crps,
        created_at
      FROM calibration_weights
    """
    sql += " WHERE " + " AND ".join(where_full)
    sql += " ORDER BY hazard_code, metric, model_name"

    rows = _rows_from_cursor(_execute(con, sql, params))

    if not rows:
        return {"found": False, "as_of_month": as_of_month, "rows": []}

    # We return rows, plus the resolved as_of_month for convenience
    return {
        "found": True,
        "as_of_month": as_of_month,
        "rows": rows,
    }


@router.get("/v1/calibration/advice")
def get_calibration_advice(
    hazard_code: Optional[str] = Query(None),
    metric: Optional[str] = Query(None),
    as_of_month: Optional[str] = Query(None, description="YYYY-MM; if omitted, use latest"),
):
    """
    Return calibration advice text per (hazard_code, metric, as_of_month).

    - If hazard_code/metric are omitted, returns advice for all rows at the chosen as_of_month.
    - If as_of_month is omitted, uses the latest as_of_month present in calibration_advice
      (optionally filtered by hazard_code/metric).
    """
    con = _con()

    params: dict = {}
    where_bits = []

    if hazard_code:
        where_bits.append("hazard_code = :hazard_code")
        params["hazard_code"] = hazard_code.upper()
    if metric:
        where_bits.append("metric = :metric")
        params["metric"] = metric.upper()

    base_where = ""
    if where_bits:
        base_where = "WHERE " + " AND ".join(where_bits)

    if not as_of_month:
        sql_latest = f"""
          SELECT as_of_month
          FROM calibration_advice
          {base_where}
          ORDER BY as_of_month DESC
          LIMIT 1
        """
        row = _execute(con, sql_latest, params).fetchone()
        if not row:
            return {"found": False, "as_of_month": None, "rows": []}
        as_of_month = row[0]

    params["as_of_month"] = as_of_month
    where_full = ["as_of_month = :as_of_month"]
    if hazard_code:
        where_full.append("hazard_code = :hazard_code")
    if metric:
        where_full.append("metric = :metric")

    sql = """
      SELECT
        as_of_month,
        hazard_code,
        metric,
        advice,
        created_at
      FROM calibration_advice
    """
    sql += " WHERE " + " AND ".join(where_full)
    sql += " ORDER BY hazard_code, metric"

    rows = _rows_from_cursor(_execute(con, sql, params))

    if not rows:
        return {"found": False, "as_of_month": as_of_month, "rows": []}

    return {
        "found": True,
        "as_of_month": as_of_month,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Performance scores
# ---------------------------------------------------------------------------


@router.get("/v1/performance/scores")
def performance_scores(
    metric: Optional[str] = Query(None, description="PA or FATALITIES"),
    track: Optional[int] = Query(None, description="Filter by track (1 or 2)"),
    include_test: bool = Query(False),
):
    """Return aggregated scoring metrics (Brier, Log, CRPS) for the performance page.

    Returns two result sets:
    - ``summary_rows``: scores aggregated across all runs, grouped by
      (hazard_code, metric, score_type, model_name).  Powers the Total,
      By Hazard, and By Model views.
    - ``run_rows``: ensemble-only scores grouped by HS run (and hazard/metric/
      score_type).  Powers the By Run view.

    Each row also carries a derived ``score_family`` field: ``'binary'`` for
    EVENT_OCCURRENCE questions (Brier range 0-1) and ``'spd'`` for multiclass
    questions (PA / FATALITIES / PHASE3PLUS_IN_NEED, Brier range 0-2).  These
    two families are on different scales and MUST NEVER be averaged together —
    the dashboard renders them as separate columns/KPIs.
    """
    con = _con()

    if not _table_exists(con, "scores") or not _table_exists(con, "questions"):
        return {"summary_rows": [], "run_rows": [], "track_counts": {"track1": 0, "track2": 0}}

    params: dict = {}
    metric_filter = ""
    if metric:
        metric_filter = "AND UPPER(q.metric) = UPPER(:metric)"
        params["metric"] = metric.upper()

    track_filter = ""
    q_cols = {r[0] for r in con.execute("DESCRIBE questions").fetchall()}
    has_track = "track" in q_cols
    if track and has_track:
        track_filter = "AND q.track = :track"
        params["track"] = track

    _tf = _test_filter(include_test, "s")

    # Track counts for KPI cards (always unfiltered by track).
    # ``total`` counts every distinct scored question regardless of track
    # (including legacy questions that pre-date the Track 1/2 split, where
    # q.track IS NULL). track1/track2 are the breakdown for questions that
    # do carry a track tag.
    track_counts = {"track1": 0, "track2": 0, "total": 0}
    try:
        tc_filter = metric_filter  # respect metric filter but not track filter
        total_row = _execute(con, f"""
            SELECT COUNT(DISTINCT s.question_id) AS n
            FROM scores s
            JOIN questions q ON q.question_id = s.question_id
            WHERE 1=1 {tc_filter}{_tf}
        """, {k: v for k, v in params.items() if k != "track"}).fetchone()
        if total_row and total_row[0] is not None:
            track_counts["total"] = int(total_row[0])
    except Exception:
        logger.debug("Failed to compute total scored question count")
    if has_track:
        try:
            tc_filter = metric_filter  # respect metric filter but not track filter
            tc_rows = _execute(con, f"""
                SELECT q.track, COUNT(DISTINCT s.question_id) AS n
                FROM scores s
                JOIN questions q ON q.question_id = s.question_id
                WHERE q.track IS NOT NULL {tc_filter}{_tf}
                GROUP BY q.track
            """, {k: v for k, v in params.items() if k != "track"}).fetchall()
            for t, n in tc_rows:
                if t == 1:
                    track_counts["track1"] = int(n)
                elif t == 2:
                    track_counts["track2"] = int(n)
        except Exception:
            logger.debug("Failed to compute track counts for performance")

    # Query 1 -- summary rows (all models including ensemble)
    sql_summary = f"""
        SELECT
          q.hazard_code,
          UPPER(q.metric) AS metric,
          CASE WHEN UPPER(q.metric) = 'EVENT_OCCURRENCE' THEN 'binary' ELSE 'spd' END AS score_family,
          s.score_type,
          s.model_name,
          COUNT(*) AS n_samples,
          COUNT(DISTINCT s.question_id) AS n_questions,
          AVG(s.value) AS avg_value,
          MEDIAN(s.value) AS median_value
        FROM scores s
        JOIN questions q ON q.question_id = s.question_id
        WHERE 1=1 {metric_filter} {track_filter}{_tf}
        GROUP BY q.hazard_code, UPPER(q.metric),
                 CASE WHEN UPPER(q.metric) = 'EVENT_OCCURRENCE' THEN 'binary' ELSE 'spd' END,
                 s.score_type, s.model_name
        ORDER BY q.hazard_code, UPPER(q.metric), s.score_type,
                 s.model_name NULLS FIRST
    """
    summary_rows = _rows_from_cursor(_execute(con, sql_summary, params))

    # Query 2 -- per-run rows (all models, including named ensembles)
    # Group by forecaster run (scores.run_id) when available so that scores
    # are attributed to the run that *produced* the forecasts, not the HS run
    # that created the questions (which may belong to a different epoch).
    has_hs_runs = _table_exists(con, "hs_runs")
    has_scores_run_id = _table_has_columns(con, "scores", ["run_id"])
    has_run_provenance = _table_exists(con, "run_provenance")

    if has_scores_run_id and has_hs_runs and has_run_provenance:
        sql_runs = f"""
            SELECT
              s.run_id AS forecaster_run_id,
              COALESCE(rp.hs_run_id, q.hs_run_id) AS hs_run_id,
              STRFTIME(MAX(h.generated_at), '%Y-%m-%d') AS run_date,
              q.hazard_code,
              UPPER(q.metric) AS metric,
              CASE WHEN UPPER(q.metric) = 'EVENT_OCCURRENCE' THEN 'binary' ELSE 'spd' END AS score_family,
              s.score_type,
              s.model_name,
              COUNT(*) AS n_samples,
              COUNT(DISTINCT s.question_id) AS n_questions,
              AVG(s.value) AS avg_value,
              MEDIAN(s.value) AS median_value
            FROM scores s
            JOIN questions q ON q.question_id = s.question_id
            LEFT JOIN run_provenance rp ON s.run_id = rp.forecaster_run_id
            LEFT JOIN hs_runs h
              ON COALESCE(rp.hs_run_id, q.hs_run_id) = h.hs_run_id
            WHERE 1=1 {metric_filter} {track_filter}{_tf}
            GROUP BY s.run_id,
                     COALESCE(rp.hs_run_id, q.hs_run_id),
                     q.hazard_code, UPPER(q.metric),
                     CASE WHEN UPPER(q.metric) = 'EVENT_OCCURRENCE' THEN 'binary' ELSE 'spd' END,
                     s.score_type, s.model_name
            ORDER BY run_date DESC NULLS LAST, q.hazard_code, s.score_type,
                     s.model_name NULLS FIRST
        """
    elif has_hs_runs:
        sql_runs = f"""
            SELECT
              NULL AS forecaster_run_id,
              q.hs_run_id,
              STRFTIME(MAX(h.generated_at), '%Y-%m-%d') AS run_date,
              q.hazard_code,
              UPPER(q.metric) AS metric,
              CASE WHEN UPPER(q.metric) = 'EVENT_OCCURRENCE' THEN 'binary' ELSE 'spd' END AS score_family,
              s.score_type,
              s.model_name,
              COUNT(*) AS n_samples,
              COUNT(DISTINCT s.question_id) AS n_questions,
              AVG(s.value) AS avg_value,
              MEDIAN(s.value) AS median_value
            FROM scores s
            JOIN questions q ON q.question_id = s.question_id
            LEFT JOIN hs_runs h ON q.hs_run_id = h.hs_run_id
            WHERE 1=1 {metric_filter} {track_filter}{_tf}
            GROUP BY q.hs_run_id, q.hazard_code, UPPER(q.metric),
                     CASE WHEN UPPER(q.metric) = 'EVENT_OCCURRENCE' THEN 'binary' ELSE 'spd' END,
                     s.score_type, s.model_name
            ORDER BY run_date DESC NULLS LAST, q.hazard_code, s.score_type,
                     s.model_name NULLS FIRST
        """
    else:
        sql_runs = f"""
            SELECT
              NULL AS forecaster_run_id,
              q.hs_run_id,
              NULL AS run_date,
              q.hazard_code,
              UPPER(q.metric) AS metric,
              CASE WHEN UPPER(q.metric) = 'EVENT_OCCURRENCE' THEN 'binary' ELSE 'spd' END AS score_family,
              s.score_type,
              s.model_name,
              COUNT(*) AS n_samples,
              COUNT(DISTINCT s.question_id) AS n_questions,
              AVG(s.value) AS avg_value,
              MEDIAN(s.value) AS median_value
            FROM scores s
            JOIN questions q ON q.question_id = s.question_id
            WHERE 1=1 {metric_filter} {track_filter}{_tf}
            GROUP BY q.hs_run_id, q.hazard_code, UPPER(q.metric),
                     CASE WHEN UPPER(q.metric) = 'EVENT_OCCURRENCE' THEN 'binary' ELSE 'spd' END,
                     s.score_type, s.model_name
            ORDER BY q.hs_run_id DESC, q.hazard_code, s.score_type,
                     s.model_name NULLS FIRST
        """
    run_rows = _rows_from_cursor(_execute(con, sql_runs, params))

    return {
        "summary_rows": summary_rows,
        "run_rows": run_rows,
        "track_counts": track_counts,
    }
