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

from pythia.tools.forecast_of_record import record_run_clause, record_run_subquery
from pythia.tools.scoring_class import has_scoring_class, scored_only_clause, scored_only_sql

from pythia.api.core import (
    _blocked_hazard_filter,
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

    f = _score_filters(con, metric=metric, track=track, include_test=include_test)
    params: dict = f["params"]
    metric_filter = f["metric_filter"]
    track_filter = f["track_filter"]
    has_track = f["has_track"]
    _tf = f["test_filter"]

    # An indicative ACE/PA month (pythia/tools/scoring_class.py) is a
    # selected sample: it stays out of every average and question count,
    # and each group says how many questions it left out
    # (n_indicative_excluded) rather than dropping them silently.
    _scored_tc = scored_only_clause(con, "s")
    if has_scoring_class(con):
        _sc = f"({scored_only_sql('s')})"
        _agg = (
            f"COUNT(*) FILTER (WHERE {_sc}) AS n_samples,\n"
            f"          COUNT(DISTINCT s.question_id) FILTER (WHERE {_sc}) AS n_questions,\n"
            f"          AVG(s.value) FILTER (WHERE {_sc}) AS avg_value,\n"
            f"          MEDIAN(s.value) FILTER (WHERE {_sc}) AS median_value,\n"
            f"          COUNT(DISTINCT s.question_id) FILTER (WHERE NOT {_sc}) AS n_indicative_excluded"
        )
    else:
        _agg = (
            "COUNT(*) AS n_samples,\n"
            "          COUNT(DISTINCT s.question_id) AS n_questions,\n"
            "          AVG(s.value) AS avg_value,\n"
            "          MEDIAN(s.value) AS median_value,\n"
            "          0 AS n_indicative_excluded"
        )

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
            WHERE 1=1 {tc_filter}{_tf}{_scored_tc}
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
                WHERE q.track IS NOT NULL {tc_filter}{_tf}{_scored_tc}
                GROUP BY q.track
            """, {k: v for k, v in params.items() if k != "track"}).fetchall()
            for t, n in tc_rows:
                if t == 1:
                    track_counts["track1"] = int(n)
                elif t == 2:
                    track_counts["track2"] = int(n)
        except Exception:
            logger.debug("Failed to compute track counts for performance")

    # Query 1 -- summary rows (all models including ensemble). One run per
    # question: its forecast of record (pythia/tools/forecast_of_record.py).
    # A same-epoch rerun (13 Oct 2026 after 1 Oct) would otherwise weigh a
    # re-asked question twice in every mean; every run still appears in the
    # per-run rows below.
    _record = record_run_clause(con, "s", include_test=include_test)

    sql_summary = f"""
        SELECT
          q.hazard_code,
          UPPER(q.metric) AS metric,
          CASE WHEN UPPER(q.metric) = 'EVENT_OCCURRENCE' THEN 'binary' ELSE 'spd' END AS score_family,
              (COALESCE(s.model_name, '') LIKE '%\\_\\_raw' ESCAPE '\\' OR COALESCE(s.model_name, '') LIKE '%\\_\\_recal' ESCAPE '\\') AS recalibration_copy,
          s.score_type,
          s.model_name,
          {_agg}
        FROM scores s
        JOIN questions q ON q.question_id = s.question_id
        WHERE 1=1 {metric_filter} {track_filter}{_tf}{_record}
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
              (COALESCE(s.model_name, '') LIKE '%\\_\\_raw' ESCAPE '\\' OR COALESCE(s.model_name, '') LIKE '%\\_\\_recal' ESCAPE '\\') AS recalibration_copy,
              s.score_type,
              s.model_name,
              {_agg}
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
              (COALESCE(s.model_name, '') LIKE '%\\_\\_raw' ESCAPE '\\' OR COALESCE(s.model_name, '') LIKE '%\\_\\_recal' ESCAPE '\\') AS recalibration_copy,
              s.score_type,
              s.model_name,
              {_agg}
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
              (COALESCE(s.model_name, '') LIKE '%\\_\\_raw' ESCAPE '\\' OR COALESCE(s.model_name, '') LIKE '%\\_\\_recal' ESCAPE '\\') AS recalibration_copy,
              s.score_type,
              s.model_name,
              {_agg}
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

    # Paired comparisons (Nov 2026): the ensemble aggregates against every
    # external reference (the headline cards), and every model against
    # climatology (the Detailed Scores skill bar). Each pair is computed on
    # the (question, horizon) cells BOTH sides scored, never on two
    # different question sets.
    reference_rows: List[Dict[str, Any]] = []
    skill_rows: List[Dict[str, Any]] = []
    try:
        base = _scored_rows_cte(con, f, include_test=include_test, with_month=False)
        if base:
            reference_rows = _paired_stats(
                con, base, f["params"],
                left_sql=_ENSEMBLE_AGG_SQL, right_sql=_REFERENCE_SQL,
            )
            skill_rows = _paired_stats(
                con, base, f["params"],
                left_sql="a.model_name NOT LIKE '\\_\\_ext\\_%' ESCAPE '\\'",
                right_sql=f"b.model_name = '{CLIMATOLOGY_MODEL}'",
            )
    except Exception:  # noqa: BLE001 - additive; never fail the page on it
        logger.debug("paired performance comparisons failed", exc_info=True)

    return {
        "summary_rows": summary_rows,
        "run_rows": run_rows,
        "track_counts": track_counts,
        "reference_rows": reference_rows,
        "skill_rows": skill_rows,
        "active_models": _active_models(),
    }


# ---------------------------------------------------------------------------
# Shared filters, the scored-rows CTE and paired comparisons (Nov 2026)
# ---------------------------------------------------------------------------

#: The reference every model's skill is measured against.
CLIMATOLOGY_MODEL = "__ext_climatology"

#: The ensemble aggregates a headline or a time series is drawn for.
_ENSEMBLE_AGG_SQL = "a.model_name LIKE 'ensemble%'"

#: External references, Sibyl's own scored variants excluded (they are
#: measurements of Sibyl, shown in its section, not benchmarks).
_REFERENCE_SQL = (
    "b.model_name LIKE '\\_\\_ext\\_%' ESCAPE '\\' "
    "AND b.model_name NOT LIKE '\\_\\_ext\\_sibyl%' ESCAPE '\\'"
)


def _score_filters(con, *, metric: Optional[str], track: Optional[int],
                   include_test: bool) -> Dict[str, Any]:
    """The filters every performance reader applies, built once.

    The metric filter, the track filter, the test filter on scores, the
    blocked-hazard exclusion, the indicative ACE/PA rule and the forecast of
    record. ``metric_filter`` carries the blocked-hazard exclusion so every
    query that applies one applies the other.
    """
    params: Dict[str, Any] = {}
    metric_filter = _blocked_hazard_filter("q")
    if metric:
        metric_filter += " AND UPPER(q.metric) = UPPER(:metric)"
        params["metric"] = metric.upper()
    track_filter = ""
    has_track = _table_has_columns(con, "questions", ["track"])
    if track and has_track:
        track_filter = " AND q.track = :track"
        params["track"] = track
    return {
        "params": params,
        "metric_filter": metric_filter,
        "track_filter": track_filter,
        "has_track": has_track,
        "test_filter": _test_filter(include_test, "s"),
        "scored": scored_only_clause(con, "s"),
        "record": record_run_clause(con, "s", include_test=include_test),
    }


def _forecast_month_sql(con, include_test: bool) -> Optional[str]:
    """SQL for the forecast month (YYYY-MM) of a score row ``s`` on question ``q``.

    The month of the forecaster run's date, as ``run_rows`` dates a run: the
    run's HS run via ``run_provenance``, else the question's own HS run. A
    run-less reference row takes its question's forecast-of-record run.
    None when ``hs_runs`` is absent (no run can be dated).
    """
    if not _table_exists(con, "hs_runs") or not _table_has_columns(con, "hs_runs", ["generated_at"]):
        return None
    run_expr = "s.run_id" if _table_has_columns(con, "scores", ["run_id"]) else "NULL"
    if _table_has_columns(con, "forecasts_ensemble", ["run_id"]):
        rec = record_run_subquery(con, "q.question_id", include_test=include_test)
        run_expr = f"COALESCE({run_expr}, {rec})"
    if _table_exists(con, "run_provenance") and _table_has_columns(
        con, "run_provenance", ["forecaster_run_id", "hs_run_id"]
    ):
        hs_expr = (
            f"COALESCE((SELECT MAX(_rp.hs_run_id) FROM run_provenance _rp "
            f"WHERE _rp.forecaster_run_id = {run_expr}), q.hs_run_id)"
        )
    else:
        hs_expr = "q.hs_run_id"
    return (
        f"(SELECT STRFTIME(MAX(_h.generated_at), '%Y-%m') FROM hs_runs _h "
        f"WHERE _h.hs_run_id = {hs_expr})"
    )


def _scored_rows_cte(con, f: Dict[str, Any], *, include_test: bool,
                     with_month: bool, month_cutoff: bool = False) -> Optional[str]:
    """``WITH sc AS (...)``: one row per scored (question, horizon, score
    type, model) after every performance filter, with its family and, when
    asked, its forecast month. None when no row can carry a month."""
    month_col = ""
    if with_month:
        month_sql = _forecast_month_sql(con, include_test)
        if month_sql is None or not _table_has_columns(con, "questions", ["hs_run_id"]):
            return None
        month_col = f", {month_sql} AS month"
    cut = (
        ", sc AS (SELECT * FROM sc_all WHERE month >= :month_cutoff)"
        if with_month and month_cutoff else ", sc AS (SELECT * FROM sc_all)"
    )
    return f"""
        WITH sc_all AS (
          SELECT s.question_id, s.horizon_m, s.score_type, s.model_name, s.value,
                 CASE WHEN UPPER(q.metric) = 'EVENT_OCCURRENCE' THEN 'binary'
                      ELSE 'spd' END AS score_family{month_col}
          FROM scores s
          JOIN questions q ON q.question_id = s.question_id
          WHERE s.value IS NOT NULL AND s.model_name IS NOT NULL
            {f["metric_filter"]} {f["track_filter"]}{f["test_filter"]}{f["scored"]}{f["record"]}
        ){cut}
    """


def _paired_stats(con, base: str, params: Dict[str, Any], *, left_sql: str,
                  right_sql: str, by_month: bool = False) -> List[Dict[str, Any]]:
    """Left model against right model on the cells both scored.

    One row per (month?, left model, right model, family, score type) with
    the mean and median of each side over the same pairs (DuckDB MEDIAN in
    SQL, never a median of medians) and the pair and question counts.
    """
    month_sel = "a.month AS month," if by_month else ""
    month_grp = "a.month," if by_month else ""
    month_where = "AND a.month IS NOT NULL" if by_month else ""
    sql = f"""
        {base}
        SELECT {month_sel}
               a.model_name AS model_name, b.model_name AS reference,
               a.score_family, a.score_type,
               COUNT(*) AS n_samples,
               COUNT(DISTINCT a.question_id) AS n_questions,
               AVG(a.value) AS model_avg, MEDIAN(a.value) AS model_median,
               AVG(b.value) AS reference_avg, MEDIAN(b.value) AS reference_median
        FROM sc a
        JOIN sc b ON b.question_id = a.question_id
                 AND b.horizon_m = a.horizon_m
                 AND b.score_type = a.score_type
        WHERE a.model_name <> b.model_name AND ({left_sql}) AND ({right_sql}) {month_where}
        GROUP BY {month_grp} a.model_name, b.model_name, a.score_family, a.score_type
        ORDER BY {month_grp} a.model_name, b.model_name, a.score_family, a.score_type
    """
    return _rows_from_cursor(_execute(con, sql, params or None))


def _active_models() -> List[Dict[str, str]]:
    """The current profile's voting members plus the Track 2 model, as the
    scores table names them (pythia.llm_profiles). Empty on any config
    failure: an unreadable profile must not take the page down."""
    try:
        from pythia.llm_profiles import active_scored_models  # noqa: PLC0415

        return active_scored_models()
    except Exception:  # noqa: BLE001
        logger.debug("active model lookup failed", exc_info=True)
        return []


def _month_cutoff(months: int, today: Optional[str] = None) -> str:
    """The first month (YYYY-MM) of a window of ``months`` ending this month."""
    import datetime as _dt  # noqa: PLC0415

    d = _dt.date.fromisoformat(today) if today else _dt.date.today()
    idx = d.year * 12 + (d.month - 1) - (int(months) - 1)
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


@router.get("/v1/performance/timeseries")
def performance_timeseries(
    metric: Optional[str] = Query(None, description="Question metric filter (PA, FATALITIES, ...)"),
    track: Optional[int] = Query(None, description="Filter by track (1 or 2)"),
    include_test: bool = Query(False),
    months: Optional[int] = Query(None, ge=1, le=240,
                                  description="Keep the last N forecast months (this month included)"),
    paired_with: Optional[str] = Query(
        None, description="Restrict the paired series to this external reference",
    ),
):
    """Scores by forecast month for the ensemble aggregates and the external
    references, plus the ensembles paired with each reference.

    ``rows``: month x score family x score type x model, unpaired, with the
    mean and the SQL median of the scored cells and the sample and question
    counts. ``paired_rows``: per month, each ensemble aggregate and a
    reference on the cells both scored (``model_avg``/``reference_avg`` and
    the medians). The month is the forecast month (``month_basis``). Every
    filter of ``/v1/performance/scores`` applies. An older DB gives empty
    lists, never an error.
    """
    empty: Dict[str, Any] = {"month_basis": "forecast_month", "rows": [], "paired_rows": []}
    con = _con()
    if not _table_exists(con, "scores") or not _table_exists(con, "questions"):
        return empty
    if paired_with is not None and not paired_with.startswith("__ext_"):
        paired_with = None  # only an external reference can be paired with
    f = _score_filters(con, metric=metric, track=track, include_test=include_test)
    params = dict(f["params"])
    try:
        if months:
            params["month_cutoff"] = _month_cutoff(months)
        base = _scored_rows_cte(con, f, include_test=include_test, with_month=True,
                                month_cutoff=bool(months))
        if base is None:
            return empty
        rows = _rows_from_cursor(_execute(con, f"""
            {base}
            SELECT a.month, a.score_family, a.score_type, a.model_name,
                   COUNT(*) AS n_samples,
                   COUNT(DISTINCT a.question_id) AS n_questions,
                   AVG(a.value) AS avg_value,
                   MEDIAN(a.value) AS median_value
            FROM sc a
            WHERE a.month IS NOT NULL
              AND (a.model_name LIKE 'ensemble%' OR a.model_name = 'track2_flash'
                   OR ({_REFERENCE_SQL.replace("b.", "a.")}))
            GROUP BY a.month, a.score_family, a.score_type, a.model_name
            ORDER BY a.month, a.score_family, a.score_type, a.model_name
        """, params or None))
        right = _REFERENCE_SQL
        if paired_with:
            right += " AND b.model_name = :paired_with"
            params["paired_with"] = paired_with
        paired_rows = _paired_stats(
            con, base, params, left_sql=_ENSEMBLE_AGG_SQL, right_sql=right, by_month=True,
        )
    except Exception:  # noqa: BLE001 - an older schema degrades to empty
        logger.debug("performance timeseries failed", exc_info=True)
        return empty
    return {"month_basis": "forecast_month", "rows": rows, "paired_rows": paired_rows}


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
        "variants": {},
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
    if has_sibyl:
        empty["variants"] = _sibyl_variant_comparison(con, include_test)
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
            "AND (s.question_id, s.run_id) IN (SELECT question_id, run_id FROM sibyl_forecasts "
            "WHERE sibyl_run_id = :srid)"
            if _table_has_columns(con, "sibyl_forecasts", ["run_id"])
            and _table_has_columns(con, "scores", ["run_id"])
            else "AND s.question_id IN (SELECT question_id FROM sibyl_forecasts "
            "WHERE sibyl_run_id = :srid)"
        )
        params["srid"] = sibyl_run_id

    # Per-question Sibyl metadata (one row per question; latest run, or the
    # requested run) for the divergence / cost / volatility visuals.
    has_sf = _table_exists(con, "sibyl_forecasts")
    if has_sf:
        _tf_sf = _test_filter(include_test, "sf")
        sf_run_filter = "AND sf.sibyl_run_id = :srid" if sibyl_run_id else ""
        _sf_run = "sf.run_id" if _table_has_columns(con, "sibyl_forecasts", ["run_id"]) else "NULL"
        meta_cte = f"""
        , sf_pick AS (
          SELECT sf.question_id,
                 sf.js_divergence_vs_standard,
                 sf.js_divergence_inter_trial,
                 sf.volatility_score,
                 sf.cost_usd,
                 {_sf_run} AS run_id,
                 ROW_NUMBER() OVER (
                   PARTITION BY sf.question_id, {_sf_run} ORDER BY sf.created_at DESC
                 ) AS rn
          FROM sibyl_forecasts sf
          WHERE 1=1 {sf_run_filter}{_tf_sf}
        )
        """
        meta_select = (
            "sf.js_divergence_vs_standard, sf.js_divergence_inter_trial, "
            "sf.volatility_score, sf.cost_usd"
        )
        meta_join = (
            "LEFT JOIN sf_pick sf ON sf.question_id = sib.question_id AND sf.rn = 1"
            + (" AND sf.run_id IS NOT DISTINCT FROM sib.run_id"
               if _sf_run != "NULL" and _table_has_columns(con, "scores", ["run_id"]) else "")
        )
    else:
        meta_cte = ""
        meta_select = (
            "NULL AS js_divergence_vs_standard, NULL AS js_divergence_inter_trial, "
            "NULL AS volatility_score, NULL AS cost_usd"
        )
        meta_join = ""

    # One run per question on BOTH sides, and the SAME run: Sibyl's latest
    # production forecast of the question, against the standard forecast of
    # that run. A question forecast in several runs has score rows for each;
    # without a run filter the Sibyl side contributed one pair per run. With
    # MAX(run_id) over every model, a same-epoch rerun that re-asked a
    # question without Sibyl (13 Oct 2026 after 1 Oct) dropped the question
    # from the comparison, and a test run made after production did the same.
    # Pairing within one run keeps the comparison fair: both forecasts were
    # made from the same evidence on the same day.
    _has_score_run = _table_has_columns(con, "scores", ["run_id"])
    _ls_test = (
        "" if include_test or not _table_has_columns(con, "scores", ["is_test"])
        else " AND NOT COALESCE(_ls.is_test, FALSE)"
    )
    _latest = (
        f" AND s.run_id = (SELECT MAX(_ls.run_id) FROM scores _ls "
        f"WHERE _ls.question_id = s.question_id AND _ls.model_name = '{_SIBYL_MODEL_NAME}' "
        f"AND _ls.run_id IS NOT NULL{_ls_test})"
        if _has_score_run and not sib_run_filter
        else ""
    )
    _run_col = "s.run_id" if _has_score_run else "NULL"
    _same_run = " AND std.run_id IS NOT DISTINCT FROM sib.run_id" if _has_score_run else ""

    # A Sibyl forecast that rested on no evidence (sibyl_forecasts.evidence_ok
    # FALSE — the July 2026 run, whose searches all failed) keeps its score
    # rows but is left out of the comparison. Judged on the latest stored ok
    # forecast for the question and forecast run, as compute_scores scored it.
    _evidence = ""
    if has_sf and _table_has_columns(con, "sibyl_forecasts", ["evidence_ok"]):
        _evidence = (
            " AND COALESCE((SELECT _ev.evidence_ok FROM sibyl_forecasts _ev "
            "WHERE _ev.question_id = s.question_id AND _ev.run_id = s.run_id "
            "AND _ev.status = 'ok' ORDER BY _ev.created_at DESC LIMIT 1), TRUE)"
        )
        meta_cte = meta_cte.replace(
            f"WHERE 1=1 {sf_run_filter}{_tf_sf}",
            f"WHERE COALESCE(sf.evidence_ok, TRUE) {sf_run_filter}{_tf_sf}",
        )

    # An indicative ACE/PA month (pythia/tools/scoring_class.py) is a
    # selected sample and is never part of the head-to-head.
    _scored = scored_only_clause(con, "s")

    sql = f"""
    WITH sib AS (
      SELECT s.question_id, s.horizon_m, s.score_type, s.value AS sibyl_value,
             {_run_col} AS run_id
      FROM scores s
      WHERE s.model_name = '{_SIBYL_MODEL_NAME}'{_tf_s} {sib_run_filter}{_latest}{_evidence}{_scored}
    ),
    std_ranked AS (
      SELECT s.question_id, s.horizon_m, s.score_type, s.model_name, s.value,
             {_run_col} AS run_id,
             ROW_NUMBER() OVER (
               PARTITION BY s.question_id, s.horizon_m, s.score_type, {_run_col}
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
            AND std.score_type = sib.score_type{_same_run}
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
        "variants": empty["variants"],
    }


#: Sibyl's comparisons beyond the standard track (Oct 2026, Part 6). Each is
#: (left series, right series); in the aggregate "sibyl_*" is the left side
#: and "standard_*" the right, so a negative mean_delta means the left won.
_SIBYL_VARIANT_PAIRS = {
    "vs_sibyl_ref": (_SIBYL_MODEL_NAME, "__ext_sibyl_ref"),
    "vs_conflictology12": (_SIBYL_MODEL_NAME, "__ext_conflictology12"),
    "raw_vs_sibyl_ref": ("__ext_sibyl_raw", "__ext_sibyl_ref"),
    "vs_raw": (_SIBYL_MODEL_NAME, "__ext_sibyl_raw"),
    # Rearranged variants of the trials already run (Oct 2026, review Part 3):
    # variant on the left, so a negative mean_delta means the variant won.
    "abc_vs_sibyl": ("__ext_sibyl_abc", _SIBYL_MODEL_NAME),
    "w25_vs_sibyl": ("__ext_sibyl_w25", _SIBYL_MODEL_NAME),
    "w50_vs_sibyl": ("__ext_sibyl_w50", _SIBYL_MODEL_NAME),
    "w75_vs_sibyl": ("__ext_sibyl_w75", _SIBYL_MODEL_NAME),
    "dw_vs_sibyl": ("__ext_sibyl_dw", _SIBYL_MODEL_NAME),
    "noguard_vs_sibyl": ("__ext_sibyl_noguard", _SIBYL_MODEL_NAME),
}
#: The rearranged-variant model names, kept out of the Detailed Scores table.
SIBYL_REARRANGED_MODELS = tuple(
    left for name, (left, _) in _SIBYL_VARIANT_PAIRS.items() if name.endswith("_vs_sibyl")
)


def _sibyl_variant_comparison(con, include_test: bool) -> Dict[str, Any]:
    """Sibyl against its own reference, the 12-month conflictology and its raw pool.

    Paired on (question, horizon, score_type), on the questions whose latest
    Sibyl forecast rested on evidence; Sibyl's side is its latest scored
    run, the references are the ``run_id IS NULL`` rows. Each comparison is
    also split by the selection pass that chose the question (floor, fill,
    control). Never raises: an older DB gives ``{}`` or empty comparisons.
    """
    if not _table_exists(con, "sibyl_forecasts"):
        return {}
    try:
        sel = (
            "sf.selection_pass" if _table_has_columns(con, "sibyl_forecasts", ["selection_pass"])
            else "CAST(NULL AS TEXT)"
        )
        ev = (
            "AND COALESCE(sf.evidence_ok, TRUE)"
            if _table_has_columns(con, "sibyl_forecasts", ["evidence_ok"]) else ""
        )
        _tf_sf = _test_filter(include_test, "sf")
        _tf_s = _test_filter(include_test, "s")
        _rearranged_sql = ", ".join(f"'{m}'" for m in SIBYL_REARRANGED_MODELS)
        rows = _execute(
            con,
            f"""
            WITH pick AS (
              SELECT sf.question_id, {sel} AS selection_pass,
                     ROW_NUMBER() OVER (PARTITION BY sf.question_id
                                        ORDER BY sf.created_at DESC) AS rn
              FROM sibyl_forecasts sf
              WHERE sf.status = 'ok' {ev}{_tf_sf}
            ),
            sib_latest AS (
              SELECT s.question_id, MAX(s.run_id) AS run_id FROM scores s
              WHERE s.model_name = '{_SIBYL_MODEL_NAME}'{_tf_s}
              GROUP BY s.question_id
            )
            SELECT s.question_id, s.horizon_m, s.score_type, s.model_name, s.value,
                   p.selection_pass, q.hazard_code, UPPER(q.metric)
            FROM scores s
            JOIN pick p ON p.question_id = s.question_id AND p.rn = 1
            JOIN questions q ON q.question_id = s.question_id
            LEFT JOIN sib_latest l ON l.question_id = s.question_id
            WHERE ((s.model_name = '{_SIBYL_MODEL_NAME}' AND s.run_id = l.run_id)
                   OR (s.model_name IN ('__ext_sibyl_ref', '__ext_sibyl_raw',
                                        '__ext_conflictology12', {_rearranged_sql})
                       AND s.run_id IS NULL)){_tf_s}{scored_only_clause(con, "s")}
            """,
        ).fetchall()
    except Exception:
        logger.debug("sibyl variant comparison failed", exc_info=True)
        return {}

    values: Dict[tuple, Dict[str, float]] = {}
    meta: Dict[str, tuple] = {}
    for qid, h, st, model, v, sp, hz, metric in rows:
        if v is None:
            continue
        values.setdefault((qid, int(h), st), {})[model] = float(v)
        meta[qid] = (sp, hz, metric)

    out: Dict[str, Any] = {}
    for name, (left, right) in _SIBYL_VARIANT_PAIRS.items():
        pairs = []
        for (qid, h, st), bym in sorted(values.items()):
            if left in bym and right in bym:
                sp, hz, metric = meta[qid]
                pairs.append({
                    "question_id": qid, "horizon_m": h, "score_type": st,
                    "hazard_code": hz, "metric": metric, "selection_pass": sp or "unknown",
                    "score_family": _family_of(metric),
                    "sibyl_value": bym[left], "standard_value": bym[right],
                })
        split: Dict[str, Any] = {}
        for sp in sorted({p["selection_pass"] for p in pairs}):
            split[sp] = _aggregate_pairs([p for p in pairs if p["selection_pass"] == sp])
        out[name] = {
            "left": left, "right": right,
            "n_questions": len({p["question_id"] for p in pairs}),
            "aggregate": _aggregate_pairs(pairs),
            "by_selection_pass": split,
        }
    # Each rearranged variant minus Sibyl with its interval, "not_yet" below
    # SIBYL_VARIANT_MIN_QUESTIONS (sibyl/score_variants.py).
    try:
        from sibyl.score_variants import variant_comparison  # noqa: PLC0415 - lazy

        paired = variant_comparison(con, include_test=include_test).get("variants") or {}
        for name, (left, _right) in _SIBYL_VARIANT_PAIRS.items():
            if left in paired:
                out[name]["paired"] = paired[left]
    except Exception:  # noqa: BLE001
        logger.debug("sibyl rearranged variant comparison failed", exc_info=True)
    return out


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
    time_capped = (
        "COALESCE(time_capped, FALSE) AS time_capped"
        if _table_has_columns(con, "sibyl_runs", ["time_capped"])
        else "FALSE AS time_capped"
    )
    try:
        return _rows_from_cursor(
            _execute(
                con,
                f"""
                SELECT sibyl_run_id, hs_run_id, created_at,
                       n_selected, n_forecast, n_skipped, budget_capped,
                       {time_capped},
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
