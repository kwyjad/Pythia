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
import statistics
from typing import Any, Dict, List, Optional

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

# Single-sourced from sibyl.config (import-light: os only — safe for the
# memory-constrained API process; see test_api_lazy_pipeline_import.py). This is
# the same preference the /v1/sibyl endpoints use to pick the standard-track
# baseline, so the head-to-head compares Sibyl against the same reference model.
from sibyl.config import STANDARD_MODEL_PREFERENCE as _STANDARD_MODEL_PREFERENCE

# The stable model_name under which Sibyl writes its pooled SPD into
# forecasts_raw / forecasts_ensemble (and therefore scores).
_SIBYL_MODEL_NAME = "sibyl"

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


# ---------------------------------------------------------------------------
# Sibyl vs. main-pipeline head-to-head comparison
# ---------------------------------------------------------------------------


def _family_of(metric: Optional[str]) -> str:
    """Brier family for a metric — 'binary' for EVENT_OCCURRENCE, else 'spd'.

    The two families are on different scales (binary Brier 0-1, multiclass SPD
    Brier 0-2) and must never be averaged together.
    """
    return "binary" if (metric or "").upper() == "EVENT_OCCURRENCE" else "spd"


def _mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def _median(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def _aggregate_pairs(pairs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate paired scores into {score_family: {score_type: stats}}.

    Never blends score families. For each (family, score_type) group we report
    the paired means/deltas across all (question, horizon) samples, plus a
    question-level win rate (each side's score averaged per question first, so
    long-window questions don't dominate). All three score types are
    lower-is-better, so a negative ``mean_delta`` (= sibyl − standard) means
    Sibyl is better.
    """
    out: Dict[str, Dict[str, Any]] = {}
    # group pairs by (family, score_type)
    groups: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for p in pairs:
        fam = p["score_family"]
        st = p["score_type"]
        groups.setdefault(fam, {}).setdefault(st, []).append(p)

    for fam, by_type in groups.items():
        out[fam] = {}
        for st, rows in by_type.items():
            deltas = [
                r["sibyl_value"] - r["standard_value"]
                for r in rows
                if r["sibyl_value"] is not None and r["standard_value"] is not None
            ]
            # Question-level win rate: average each side per question first.
            per_q: Dict[str, Dict[str, List[float]]] = {}
            for r in rows:
                q = per_q.setdefault(r["question_id"], {"s": [], "b": []})
                if r["sibyl_value"] is not None:
                    q["s"].append(r["sibyl_value"])
                if r["standard_value"] is not None:
                    q["b"].append(r["standard_value"])
            sibyl_wins = standard_wins = ties = 0
            for q in per_q.values():
                s_avg = _mean(q["s"])
                b_avg = _mean(q["b"])
                if s_avg is None or b_avg is None:
                    continue
                if s_avg < b_avg:
                    sibyl_wins += 1
                elif b_avg < s_avg:
                    standard_wins += 1
                else:
                    ties += 1
            decided = sibyl_wins + standard_wins + ties
            out[fam][st] = {
                "n_pairs": len(rows),
                "n_questions": len({r["question_id"] for r in rows}),
                "sibyl_mean": _mean([r["sibyl_value"] for r in rows]),
                "standard_mean": _mean([r["standard_value"] for r in rows]),
                "mean_delta": _mean(deltas),
                "median_delta": _median(deltas),
                "sibyl_wins": sibyl_wins,
                "standard_wins": standard_wins,
                "ties": ties,
                "win_rate": (sibyl_wins / decided) if decided else None,
            }
    return out


def _by_hazard_metric(pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per (hazard, metric) Brier rollup on the covered set (Brier only)."""
    groups: Dict[tuple, List[Dict[str, Any]]] = {}
    for p in pairs:
        if p["score_type"] != "brier":
            continue
        key = (p["hazard_code"], p["metric"], p["score_family"])
        groups.setdefault(key, []).append(p)
    rows: List[Dict[str, Any]] = []
    for (hz, metric, fam), rs in sorted(groups.items(), key=lambda kv: kv[0]):
        sibyl_brier = _mean([r["sibyl_value"] for r in rs])
        standard_brier = _mean([r["standard_value"] for r in rs])
        rows.append({
            "hazard_code": hz,
            "metric": metric,
            "score_family": fam,
            "n_questions": len({r["question_id"] for r in rs}),
            "sibyl_brier": sibyl_brier,
            "standard_brier": standard_brier,
            "delta": (sibyl_brier - standard_brier)
            if (sibyl_brier is not None and standard_brier is not None)
            else None,
        })
    return rows


@router.get("/v1/performance/sibyl_comparison")
def sibyl_comparison(
    include_test: bool = Query(False),
    baseline: Optional[str] = Query(
        None,
        description="Standard model_name to compare Sibyl against; default is "
        "best-available per question (bayesmc > mean > track2_flash).",
    ),
    sibyl_run_id: Optional[str] = Query(
        None, description="Restrict to questions from one Sibyl run."
    ),
):
    """Fair head-to-head between the Sibyl deep-research track and the main
    pipeline, restricted to the exact (question, horizon, score_type) set that
    Sibyl actually forecast (Sibyl only covers the top-volatility subset, so an
    all-questions average would be unfair).

    Tolerates pre-Sibyl / empty DBs and the current post-DB-reset state where no
    scores exist yet: returns ``has_sibyl:false`` with an empty ``pairs`` list
    but still populates ``runs`` (from ``sibyl_runs``) so the UI can show a
    "Sibyl ran, awaiting resolutions" state.
    """
    con = _con()

    empty: Dict[str, Any] = {
        "has_sibyl": False,
        "baseline_used": None,
        "available_baselines": [],
        "pairs": [],
        "aggregate": {},
        "by_hazard_metric": [],
        "runs": [],
    }

    # runs are useful even with zero scores (coverage/cost/budget KPIs).
    runs = _load_sibyl_runs(con, include_test, sibyl_run_id)
    empty["runs"] = runs

    if not _table_exists(con, "scores") or not _table_exists(con, "questions"):
        return empty

    _tf_s = _test_filter(include_test, "s")
    pref = list(_STANDARD_MODEL_PREFERENCE)
    # Trusted, code-controlled constants — safe to inline. Kept single-sourced
    # from sibyl.config so the baseline set never drifts from the Sibyl track.
    pref_in = ", ".join("'" + m.replace("'", "''") + "'" for m in pref)

    # Which standard aggregates actually have scores? Only offer real baselines.
    avail = {
        r[0]
        for r in _execute(
            con,
            f"SELECT DISTINCT model_name FROM scores s "
            f"WHERE s.model_name IN ({pref_in}){_tf_s}",
        ).fetchall()
        if r and r[0]
    }
    available_baselines = [m for m in pref if m in avail]

    # Is Sibyl scored at all?
    has_sibyl = bool(
        _execute(
            con,
            f"SELECT 1 FROM scores s WHERE s.model_name = '{_SIBYL_MODEL_NAME}'"
            f"{_tf_s} LIMIT 1",
        ).fetchall()
    )
    if not has_sibyl or not available_baselines:
        return empty

    # Resolve the baseline the caller asked for (must be a real, available one).
    baseline_used: Optional[str] = None
    baseline_filter = ""
    params: Dict[str, Any] = {}
    if baseline and baseline in avail:
        baseline_used = baseline
        baseline_filter = "AND s.model_name = :baseline"
        params["baseline"] = baseline
    else:
        baseline_used = "best_available"

    # Optional restriction to one Sibyl run's question set.
    sib_run_filter = ""
    if sibyl_run_id and _table_exists(con, "sibyl_forecasts"):
        sib_run_filter = (
            "AND s.question_id IN (SELECT question_id FROM sibyl_forecasts "
            "WHERE sibyl_run_id = :srid)"
        )
        params["srid"] = sibyl_run_id

    # Per-question Sibyl metadata (one row per question; latest run, or the
    # requested run) for the divergence / cost / volatility visuals.
    has_sf = _table_exists(con, "sibyl_forecasts")
    if has_sf:
        _tf_sf = _test_filter(include_test, "sf")
        sf_run_filter = "AND sf.sibyl_run_id = :srid" if sibyl_run_id else ""
        meta_cte = f"""
        , sf_pick AS (
          SELECT sf.question_id,
                 sf.js_divergence_vs_standard,
                 sf.js_divergence_inter_trial,
                 sf.volatility_score,
                 sf.cost_usd,
                 ROW_NUMBER() OVER (
                   PARTITION BY sf.question_id ORDER BY sf.created_at DESC
                 ) AS rn
          FROM sibyl_forecasts sf
          WHERE 1=1 {sf_run_filter}{_tf_sf}
        )
        """
        meta_select = (
            "sf.js_divergence_vs_standard, sf.js_divergence_inter_trial, "
            "sf.volatility_score, sf.cost_usd"
        )
        meta_join = "LEFT JOIN sf_pick sf ON sf.question_id = sib.question_id AND sf.rn = 1"
    else:
        meta_cte = ""
        meta_select = (
            "NULL AS js_divergence_vs_standard, NULL AS js_divergence_inter_trial, "
            "NULL AS volatility_score, NULL AS cost_usd"
        )
        meta_join = ""

    sql = f"""
    WITH sib AS (
      SELECT s.question_id, s.horizon_m, s.score_type, s.value AS sibyl_value
      FROM scores s
      WHERE s.model_name = '{_SIBYL_MODEL_NAME}'{_tf_s} {sib_run_filter}
    ),
    std_ranked AS (
      SELECT s.question_id, s.horizon_m, s.score_type, s.model_name, s.value,
             ROW_NUMBER() OVER (
               PARTITION BY s.question_id, s.horizon_m, s.score_type
               ORDER BY CASE s.model_name
                 {" ".join(f"WHEN '{m}' THEN {i}" for i, m in enumerate(pref))}
                 ELSE 99 END
             ) AS rn
      FROM scores s
      WHERE s.model_name IN ({pref_in}) {baseline_filter}{_tf_s}
    ),
    std AS (SELECT * FROM std_ranked WHERE rn = 1)
    {meta_cte}
    SELECT
      sib.question_id, q.iso3, q.hazard_code, UPPER(q.metric) AS metric,
      CASE WHEN UPPER(q.metric) = 'EVENT_OCCURRENCE' THEN 'binary' ELSE 'spd' END AS score_family,
      sib.horizon_m, sib.score_type,
      sib.sibyl_value, std.value AS standard_value, std.model_name AS standard_model_name,
      {meta_select}
    FROM sib
    JOIN std ON std.question_id = sib.question_id
            AND std.horizon_m = sib.horizon_m
            AND std.score_type = sib.score_type
    JOIN questions q ON q.question_id = sib.question_id
    {meta_join}
    ORDER BY q.hazard_code, sib.question_id, sib.score_type, sib.horizon_m
    """
    pairs = _rows_from_cursor(_execute(con, sql, params if params else None))

    return {
        "has_sibyl": True,
        "baseline_used": baseline_used,
        "available_baselines": available_baselines,
        "pairs": pairs,
        "aggregate": _aggregate_pairs(pairs),
        "by_hazard_metric": _by_hazard_metric(pairs),
        "runs": runs,
    }


def _load_sibyl_runs(
    con, include_test: bool, sibyl_run_id: Optional[str]
) -> List[Dict[str, Any]]:
    """Coverage / cost / budget rows from sibyl_runs (empty if the table is
    absent). Used for KPIs even when no scores exist yet."""
    if not _table_exists(con, "sibyl_runs"):
        return []
    _tf = _test_filter(include_test, "")
    where = "WHERE 1=1" + _tf
    params: Dict[str, Any] = {}
    if sibyl_run_id:
        where += " AND sibyl_run_id = :srid"
        params["srid"] = sibyl_run_id
    try:
        return _rows_from_cursor(
            _execute(
                con,
                f"""
                SELECT sibyl_run_id, hs_run_id, created_at,
                       n_selected, n_forecast, n_skipped, budget_capped,
                       run_cost_usd, opus_cost_usd, brave_cost_usd, run_hard_cap_usd
                FROM sibyl_runs
                {where}
                ORDER BY created_at DESC
                """,
                params if params else None,
            )
        )
    except Exception:
        logger.debug("Failed to load sibyl_runs for comparison", exc_info=True)
        return []
