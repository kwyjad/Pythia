# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Diagnostics and HS-debug routes:
/v1/diagnostics/{memory,summary,resolution_rates,kpi_scopes,run_summary},
/v1/hs_runs, /v1/hs_triage/all and the token-gated /v1/debug/hs_* group.

Endpoint functions moved verbatim from pythia.api.app (July 2026
decomposition); shared helpers come from pythia.api.core.
"""

import logging
import resource
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header, Query

from pythia.api.core import (
    _DUCKDB_MEMORY_LIMIT,
    _DUCKDB_THREADS,
    _HEAVY_REQUEST_SEMAPHORE,
    _con,
    _count_distinct_active_questions,
    _execute,
    _format_year_month_label,
    _month_window,
    _parse_year_month,
    _pick_col,
    _pick_timestamp_column,
    _require_debug_token,
    _rows_from_cursor,
    _shift_ym,
    _table_columns,
    _table_exists,
    _table_has_columns,
    _test_filter,
)
from pythia.api.db_sync import get_cached_latest_hs
from resolver.query.debug_ui import (
    _get_hs_triage_llm_calls_with_debug,
    _get_hs_triage_rows_with_debug,
    _list_hs_runs_with_debug,
    get_hs_triage_all,
    get_country_run_summary,
    list_hs_runs,
)
from resolver.query.kpi_scopes import compute_countries_triaged_for_month_with_source

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/v1/diagnostics/memory")
def diagnostics_memory():
    """Return current process memory and DuckDB buffer pool usage."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    rss_bytes = rusage.ru_maxrss
    # macOS returns bytes, Linux returns KB
    import sys as _sys
    if _sys.platform == "darwin":
        rss_mb = rss_bytes / (1024 * 1024)
    else:
        rss_mb = rss_bytes / 1024
    result: Dict[str, Any] = {"rss_mb": round(rss_mb, 1)}
    try:
        con = _con()
        result["duckdb_memory"] = _rows_from_cursor(con.execute("SELECT * FROM duckdb_memory()"))
    except Exception:
        result["duckdb_memory"] = []
    result["limits"] = {
        "memory_limit": _DUCKDB_MEMORY_LIMIT,
        "threads": _DUCKDB_THREADS,
        "max_concurrent_heavy": _HEAVY_REQUEST_SEMAPHORE._value,
    }
    return result


@router.get("/v1/diagnostics/summary")
def diagnostics_summary(include_test: bool = Query(False)):
    """
    Return a high-level summary of Pythia's state:

      - question counts by status
      - number of questions with forecasts (ensemble)
      - number of questions with resolutions
      - number of questions with scores
      - latest HS run (hs_runs)
      - latest calibration as_of_month (calibration_weights)
    """
    con = _con()

    _tf = _test_filter(include_test)
    try:
        # questions_by_status is still grouped by status (including 'retired'),
        # so the count of retired questions remains visible/auditable here.
        q_counts = _rows_from_cursor(con.execute(
            f"SELECT status, COUNT(*) AS n FROM questions WHERE 1=1{_tf} GROUP BY status"
        ))
    except Exception:
        q_counts = []

    # The "with X" counts below exclude retired questions so the dashboard's
    # active-question metrics aren't inflated by unforecastable legacy rows.
    q_with_forecast = _count_distinct_active_questions(con, "forecasts_ensemble", include_test)
    q_with_resolutions = _count_distinct_active_questions(con, "resolutions", include_test)
    q_with_scores = _count_distinct_active_questions(con, "scores", include_test)

    latest_hs = get_cached_latest_hs()
    if latest_hs is None:
        hs_row = None
        if _table_exists(con, "hs_runs"):
            hs_cols = _table_columns(con, "hs_runs")
            hs_id_col = _pick_col(hs_cols, ["run_id", "hs_run_id"])
            hs_time_col = _pick_col(hs_cols, ["created_at", "finished_at", "started_at"])
            hs_meta_col = _pick_col(
                hs_cols,
                ["meta", "meta_json", "requested_countries_json", "countries_json"],
            )
            if hs_id_col and hs_time_col:
                meta_select = f"{hs_meta_col} AS meta" if hs_meta_col else "NULL AS meta"
                hs_row = con.execute(
                    f"""
                    SELECT {hs_id_col} AS run_id,
                           {hs_time_col} AS created_at,
                           {meta_select}
                    FROM hs_runs
                    ORDER BY {hs_time_col} DESC NULLS LAST
                    LIMIT 1
                    """
                ).fetchone()
        if hs_row:
            latest_hs = {
                "run_id": hs_row[0],
                "created_at": hs_row[1],
                "meta": hs_row[2],
            }

    cal_row = None
    if _table_exists(con, "calibration_weights"):
        cal_cols = _table_columns(con, "calibration_weights")
        if {"as_of_month", "created_at"}.issubset(cal_cols):
            cal_row = con.execute(
                """
                SELECT as_of_month, MAX(created_at)
                FROM calibration_weights
                GROUP BY as_of_month
                ORDER BY as_of_month DESC
                LIMIT 1
                """
            ).fetchone()
    if cal_row:
        latest_calibration = {
            "as_of_month": cal_row[0],
            "created_at": cal_row[1],
        }
    else:
        latest_calibration = None

    return {
        "questions_by_status": q_counts,
        "questions_with_forecasts": int(q_with_forecast),
        "questions_with_resolutions": int(q_with_resolutions),
        "questions_with_scores": int(q_with_scores),
        "latest_hs_run": latest_hs,
        "latest_calibration": latest_calibration,
    }


@router.get("/v1/debug/hs_runs")
def debug_hs_runs(
    limit: int = Query(50, ge=1, le=500),
    include_test: bool = Query(False),
    x_fred_debug_token: Optional[str] = Header(default=None, alias="X-Fred-Debug-Token"),
):
    _require_debug_token(x_fred_debug_token)
    con = _con()
    rows, schema_debug = _list_hs_runs_with_debug(con, limit=limit, include_test=include_test)
    logger.info("Debug hs_runs rows=%d schema=%s", len(rows), schema_debug)
    return {"rows": rows, "schema_debug": schema_debug}


@router.get("/v1/hs_runs")
def hs_runs(
    limit: int = Query(50, ge=1, le=500),
    include_test: bool = Query(False),
):
    con = _con()
    rows = list_hs_runs(con, limit=limit, include_test=include_test)
    logger.info("HS runs rows=%d", len(rows))
    return {"rows": rows}


@router.get("/v1/hs_triage/all")
def hs_triage_all(
    run_id: str = Query(...),
    iso3: Optional[str] = Query(None),
    hazard_code: Optional[str] = Query(None),
    limit: int = Query(2000, ge=1, le=5000),
    include_test: bool = Query(False),
):
    con = _con()
    rows, diagnostics = get_hs_triage_all(
        con, run_id=run_id, iso3=iso3, hazard_code=hazard_code, limit=limit,
        include_test=include_test,
    )
    logger.info("HS triage all rows=%d run_id=%s", len(rows), run_id)
    return {"rows": rows, "diagnostics": diagnostics}


@router.get("/v1/debug/hs_triage")
def debug_hs_triage(
    run_id: str = Query(...),
    iso3: Optional[str] = Query(None),
    hazard_code: Optional[str] = Query(None),
    limit: int = Query(500, ge=1, le=2000),
    include_test: bool = Query(False),
    x_fred_debug_token: Optional[str] = Header(default=None, alias="X-Fred-Debug-Token"),
):
    _require_debug_token(x_fred_debug_token)
    con = _con()
    rows, schema_debug = _get_hs_triage_rows_with_debug(
        con, run_id=run_id, iso3=iso3, hazard_code=hazard_code, limit=limit,
        include_test=include_test,
    )
    logger.info("Debug hs_triage rows=%d schema=%s", len(rows), schema_debug)
    return {"rows": rows, "schema_debug": schema_debug}


@router.get("/v1/debug/hs_triage_llm_calls")
def debug_hs_triage_llm_calls(
    run_id: str = Query(...),
    iso3: Optional[str] = Query(None),
    hazard_code: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=2000),
    preview_chars: int = Query(800, ge=50, le=5000),
    include_test: bool = Query(False),
    x_fred_debug_token: Optional[str] = Header(default=None, alias="X-Fred-Debug-Token"),
):
    _require_debug_token(x_fred_debug_token)
    con = _con()
    rows, schema_debug = _get_hs_triage_llm_calls_with_debug(
        con,
        run_id=run_id,
        iso3=iso3,
        hazard_code=hazard_code,
        limit=limit,
        preview_chars=preview_chars,
        include_test=include_test,
    )
    logger.info("Debug hs_triage_llm_calls rows=%d schema=%s", len(rows), schema_debug)
    return {"rows": rows, "schema_debug": schema_debug}


@router.get("/v1/debug/hs_country_summary")
def debug_hs_country_summary(
    run_id: str = Query(...),
    iso3: str = Query(...),
    include_test: bool = Query(False),
    x_fred_debug_token: Optional[str] = Header(default=None, alias="X-Fred-Debug-Token"),
):
    _require_debug_token(x_fred_debug_token)
    con = _con()
    row = get_country_run_summary(con, run_id=run_id, iso3=iso3, include_test=include_test)
    logger.info("Debug hs_country_summary run_id=%s iso3=%s", run_id, iso3)
    return {"row": row}


@router.get("/v1/diagnostics/resolution_rates")
def resolution_rates(
    forecaster_run_id: Optional[str] = Query(None),
    hazard_code: Optional[str] = Query(None),
    include_test: bool = Query(False),
    include_blocked: bool = Query(
        False,
        description=(
            "Include blocked hazards (DI, HW, CU, ACO) that are no longer "
            "forecasted but still have legacy questions in the DB."
        ),
    ),
):
    """Compute resolution rates by hazard and metric.

    Returns how many questions were resolved (have at least 1 resolution row)
    vs total, broken down by (hazard_code, metric). Blocked hazards (DI, HW,
    CU, ACO) are excluded by default — pass ``include_blocked=true`` to see
    them.
    """
    con = _con()
    if not _table_exists(con, "questions"):
        return {"rows": []}

    has_resolutions = _table_exists(con, "resolutions") and _table_has_columns(
        con, "resolutions", ["question_id"]
    )

    hazard_filter = ""
    params: dict = {}
    if hazard_code:
        hazard_filter = "AND UPPER(q.hazard_code) = UPPER(:hazard_code)"
        params["hazard_code"] = hazard_code.upper()

    run_filter = ""
    if forecaster_run_id and _table_has_columns(con, "questions", ["forecaster_run_id"]):
        run_filter = "AND q.forecaster_run_id = :forecaster_run_id"
        params["forecaster_run_id"] = forecaster_run_id

    # Exclude retired questions (legacy DI/HW/CU/ACO that will never resolve).
    # These were the bulk of the dashboard's permanent-0% tiles.
    retired_filter = "AND COALESCE(q.status, '') != 'retired'"

    # Also exclude blocked hazards by hazard_code. The retired-status guard
    # above only catches questions that were retired by
    # scripts/retire_blocked_hazard_questions.py — but a fresh DB or a DB
    # where that script hasn't run yet still has active rows for DI/HW/CU/ACO.
    # Block them here too so the resolution coverage panel doesn't show
    # permanently-unresolvable tiles for hazards we don't forecast anymore.
    blocked_filter = ""
    if not include_blocked:
        blocked_filter = (
            "AND UPPER(q.hazard_code) NOT IN ('DI', 'HW', 'CU', 'ACO')"
        )

    # The resolution pipeline's calendar cutoff is the previous complete
    # month (see pythia/tools/compute_resolutions.py). A question's earliest
    # horizon maps to its window_start_date month. So a question is
    # "pending_too_new" — structurally unresolvable until the calendar
    # advances — when window_start_date > the first day of the cutoff month.
    has_window_start = _table_has_columns(con, "questions", ["window_start_date"])

    # Total questions by (hazard_code, metric)
    _tf = _test_filter(include_test, "r")
    total_sql = f"""
        SELECT q.hazard_code, UPPER(q.metric) AS metric,
               COUNT(DISTINCT q.question_id) AS total_questions
        FROM questions q
        WHERE 1=1 {hazard_filter} {run_filter} {retired_filter} {blocked_filter}{_test_filter(include_test, "q")}
        GROUP BY q.hazard_code, UPPER(q.metric)
    """
    try:
        total_rows = _execute(con, total_sql, params).fetchall()
    except Exception:
        logger.exception("resolution_rates total-count query failed")
        return {"rows": []}

    pending_map: dict[tuple[str, str], int] = {}
    if has_window_start:
        # window_start_date is stored as TEXT in some DBs and DATE in others.
        # DATE_TRUNC works on both via DuckDB's TRY_CAST behaviour.
        pending_sql = f"""
            SELECT q.hazard_code, UPPER(q.metric) AS metric,
                   COUNT(DISTINCT q.question_id) AS pending
            FROM questions q
            WHERE 1=1 {hazard_filter} {run_filter} {retired_filter} {blocked_filter}{_test_filter(include_test, "q")}
              AND CAST(q.window_start_date AS DATE)
                  > DATE_TRUNC('month', CURRENT_DATE - INTERVAL 1 MONTH)
            GROUP BY q.hazard_code, UPPER(q.metric)
        """
        try:
            for hc, m, n in _execute(con, pending_sql, params).fetchall():
                pending_map[(hc, m)] = int(n)
        except Exception:
            # If the date cast fails or column is unusable, fall through with
            # an empty pending_map — the response still includes 0 pending.
            pending_map = {}

    if not has_resolutions:
        return {
            "rows": [
                {
                    "hazard_code": hc,
                    "metric": m,
                    "total_questions": int(t),
                    "resolved_questions": 0,
                    "skipped_questions": int(t),
                    "pending_too_new": pending_map.get((hc, m), 0),
                    "resolution_rate": 0.0,
                }
                for hc, m, t in total_rows
            ]
        }

    # Resolved questions (at least 1 resolution row)
    resolved_sql = f"""
        SELECT q.hazard_code, UPPER(q.metric) AS metric,
               COUNT(DISTINCT r.question_id) AS resolved_questions
        FROM resolutions r
        JOIN questions q ON q.question_id = r.question_id
        WHERE 1=1 {hazard_filter} {run_filter} {retired_filter} {blocked_filter}{_tf}
        GROUP BY q.hazard_code, UPPER(q.metric)
    """
    try:
        resolved_rows = _execute(con, resolved_sql, params).fetchall()
    except Exception:
        resolved_rows = []

    resolved_map: dict[tuple[str, str], int] = {}
    for hc, m, n in resolved_rows:
        resolved_map[(hc, m)] = int(n)

    result = []
    for hc, m, total in total_rows:
        total_int = int(total)
        resolved = resolved_map.get((hc, m), 0)
        skipped = total_int - resolved
        rate = resolved / total_int if total_int > 0 else 0.0
        result.append({
            "hazard_code": hc,
            "metric": m,
            "total_questions": total_int,
            "resolved_questions": resolved,
            "skipped_questions": skipped,
            "pending_too_new": pending_map.get((hc, m), 0),
            "resolution_rate": round(rate, 4),
        })

    return {"rows": result}


@router.get("/v1/diagnostics/kpi_scopes")
def diagnostics_kpi_scopes(
    metric_scope: str = Query("PA"),
    year_month: Optional[str] = Query(None),
    forecaster_run_id: Optional[str] = Query(None, description="Forecaster run ID to scope KPIs"),
    include_test: bool = Query(False),
):
    con = _con()
    notes: List[str] = []
    diagnostics: Dict[str, Any] = {
        "month_source": None,
        "forecast_source": None,
        "metric_scope": metric_scope,
        "selected_month": None,
        "countries_triaged_source": None,
    }

    metric_scope = metric_scope.upper()
    if metric_scope not in {"PA", "FATALITIES", "EVENT_OCCURRENCE", "PHASE3PLUS_IN_NEED"}:
        notes.append("metric_scope_unrecognized")

    questions_cols = _table_columns(con, "questions") if _table_exists(con, "questions") else set()
    has_questions = _table_has_columns(con, "questions", ["question_id"])
    has_metric = "metric" in questions_cols
    has_iso3 = "iso3" in questions_cols
    has_status = "status" in questions_cols
    has_hazard = "hazard_code" in questions_cols

    if not has_questions:
        notes.append("questions_table_missing")

    def metric_clause() -> tuple[str, List[Any]]:
        if not metric_scope or not has_metric:
            if metric_scope and not has_metric:
                notes.append("metric_scope_ignored_missing_column")
            return "", []
        return " AND UPPER(q.metric) = ?", [metric_scope]

    def status_clause(status_value: str) -> tuple[str, List[Any]]:
        if not has_status:
            notes.append("status_filter_ignored_missing_column")
            return "", []
        return " AND q.status = ?", [status_value]

    month_source_table: Optional[str] = None
    month_source_ts: Optional[str] = None
    month_source_phase: Optional[List[str]] = None
    forecast_source_table: Optional[str] = None
    forecast_source_ts: Optional[str] = None
    forecast_source_phase: Optional[List[str]] = None
    question_source_table: Optional[str] = None
    question_source_ts: Optional[str] = None

    forecast_ts = _pick_timestamp_column(
        con, "forecasts_ensemble", ["created_at", "timestamp", "started_at"]
    )
    if forecast_ts and _table_has_columns(con, "forecasts_ensemble", ["question_id", forecast_ts]):
        month_source_table = "forecasts_ensemble"
        month_source_ts = forecast_ts
        forecast_source_table = "forecasts_ensemble"
        forecast_source_ts = forecast_ts
        diagnostics["month_source"] = f"forecasts_ensemble.{forecast_ts}"
        diagnostics["forecast_source"] = f"forecasts_ensemble.{forecast_ts}"
    else:
        llm_ts = _pick_timestamp_column(
            con, "llm_calls", ["created_at", "timestamp", "started_at"]
        )
        if llm_ts and _table_has_columns(con, "llm_calls", ["question_id", llm_ts]):
            month_source_table = "llm_calls"
            month_source_ts = llm_ts
            month_source_phase = ["forecast", "research", "triage", "context"]
            diagnostics["month_source"] = f"llm_calls.{llm_ts}"
            if _table_has_columns(con, "llm_calls", ["phase"]):
                forecast_source_table = "llm_calls"
                forecast_source_ts = llm_ts
                forecast_source_phase = ["forecast"]
                diagnostics["forecast_source"] = f"llm_calls.{llm_ts}.phase"
        else:
            notes.append("month_source_unavailable")
            diagnostics["forecast_source"] = "unavailable"

    available_months: List[str] = []
    if month_source_table and month_source_ts:
        _tf_months = _test_filter(include_test)
        sql = (
            f"SELECT DISTINCT strftime({month_source_ts}, '%Y-%m') AS year_month "
            f"FROM {month_source_table} WHERE {month_source_ts} IS NOT NULL{_tf_months}"
        )
        params: List[Any] = []
        if month_source_phase:
            placeholders = ", ".join(["?"] * len(month_source_phase))
            sql += f" AND phase IN ({placeholders})"
            params.extend(month_source_phase)
        try:
            rows = con.execute(sql, params).fetchall()
            available_months = [row[0] for row in rows if row and row[0]]
        except Exception:
            notes.append("available_months_failed")

    parsed_months: List[tuple[int, int, str]] = []
    for ym in available_months:
        parsed = _parse_year_month(ym)
        if parsed:
            parsed_months.append((parsed[0], parsed[1], ym))
    parsed_months.sort(reverse=True)
    sorted_months = [entry[2] for entry in parsed_months]

    selected_month = None
    if year_month:
        parsed = _parse_year_month(year_month)
        if parsed:
            if sorted_months and year_month not in sorted_months:
                notes.append("selected_month_not_available")
                selected_month = sorted_months[0]
            else:
                selected_month = year_month
        else:
            notes.append("selected_month_invalid")
    if selected_month is None and sorted_months:
        selected_month = sorted_months[0]

    diagnostics["selected_month"] = selected_month

    available_month_rows: List[Dict[str, Any]] = []
    for index, ym in enumerate(sorted_months):
        parsed = _parse_year_month(ym)
        if not parsed:
            continue
        available_month_rows.append(
            {
                "year_month": ym,
                "label": _format_year_month_label(parsed[0], parsed[1]),
                "is_latest": index == 0,
            }
        )

    def _count(sql: str, params: List[Any], note: str) -> int:
        try:
            row = con.execute(sql, params).fetchone()
            return int(row[0]) if row else 0
        except Exception:
            notes.append(note)
            return 0

    def _fetch_hazard_counts(sql: str, params: List[Any], note: str) -> Dict[str, int]:
        try:
            rows = con.execute(sql, params).fetchall()
        except Exception:
            notes.append(note)
            return {}
        output: Dict[str, int] = {}
        for hazard, count in rows:
            if hazard is None:
                continue
            output[str(hazard)] = int(count)
        return output

    def _scope_from_question_ids(
        question_ids_sql: str,
        question_ids_params: List[Any],
        status_filter: Optional[str] = None,
        forecast_window_ym: Optional[tuple] = None,
    ) -> Dict[str, Any]:
        scope: Dict[str, Any] = {
            "questions": 0,
            "forecasts": 0,
            "countries": 0,
            "countries_total": 0,
            "countries_with_forecasts": 0,
            "resolved_questions": 0,
            "forecasts_by_hazard": {},
        }
        if not has_questions:
            return scope

        metric_sql, metric_params = metric_clause()
        status_sql, status_params = ("", [])
        if status_filter:
            status_sql, status_params = status_clause(status_filter)

        base_sql = (
            f"FROM questions q JOIN ({question_ids_sql}) src ON src.question_id = q.question_id "
            f"WHERE 1=1{metric_sql}{status_sql}"
        )
        scope["questions"] = _count(
            f"SELECT COUNT(DISTINCT q.question_id) {base_sql}",
            question_ids_params + metric_params + status_params,
            "scope_questions_failed",
        )

        if has_iso3:
            scope["countries_total"] = _count(
                f"SELECT COUNT(DISTINCT q.iso3) {base_sql}",
                question_ids_params + metric_params + status_params,
                "scope_countries_failed",
            )
        else:
            notes.append("countries_ignored_missing_column")

        if _table_has_columns(con, "resolutions", ["question_id"]):
            res_sql = (
                f"SELECT COUNT(DISTINCT r.question_id) FROM resolutions r "
                f"JOIN ({question_ids_sql}) src ON src.question_id = r.question_id "
                f"JOIN questions q ON r.question_id = q.question_id "
                f"WHERE 1=1{metric_sql}"
            )
            res_params = list(question_ids_params) + metric_params
            if forecast_window_ym and _table_has_columns(con, "resolutions", ["observed_month"]):
                res_sql += " AND r.observed_month >= ? AND r.observed_month < ?"
                res_params.extend(forecast_window_ym)
            scope["resolved_questions"] = _count(
                res_sql, res_params, "scope_resolutions_failed",
            )
        elif has_status:
            resolved_sql = f"{base_sql} AND q.status IN ('resolved', 'closed')"
            scope["resolved_questions"] = _count(
                f"SELECT COUNT(DISTINCT q.question_id) {resolved_sql}",
                question_ids_params + metric_params + status_params,
                "scope_resolved_failed",
            )
        else:
            notes.append("resolved_questions_unavailable")

        if has_hazard and forecast_source_table and forecast_source_ts:
            forecast_ids_sql = question_ids_sql
            forecast_params = list(question_ids_params)
            # Only switch to a time-window forecast query when the caller
            # actually passed time-window params (>= 2).  When called with a
            # single run_id param the original question_ids_sql already
            # correctly identifies forecasts for that run.
            if (
                (forecast_source_table != question_source_table or forecast_source_phase)
                and len(question_ids_params) >= 2
            ):
                forecast_ids_sql = (
                    f"SELECT DISTINCT question_id FROM {forecast_source_table} "
                    f"WHERE {forecast_source_ts} >= ? AND {forecast_source_ts} < ?"
                )
                forecast_params = list(question_ids_params[:2])
                if forecast_source_phase:
                    placeholders = ", ".join(["?"] * len(forecast_source_phase))
                    forecast_ids_sql += f" AND phase IN ({placeholders})"
                    forecast_params.extend(forecast_source_phase)

            forecast_base_sql = (
                f"FROM questions q JOIN ({forecast_ids_sql}) f ON f.question_id = q.question_id "
                f"WHERE 1=1{metric_sql}{status_sql}"
            )
            scope["forecasts"] = _count(
                f"SELECT COUNT(DISTINCT q.question_id) {forecast_base_sql}",
                forecast_params + metric_params + status_params,
                "scope_forecasts_failed",
            )
            if has_iso3:
                # countries_with_forecasts respects the metric scope so the
                # count is consistent with the forecasts count shown alongside.
                scope["countries_with_forecasts"] = _count(
                    f"SELECT COUNT(DISTINCT q.iso3) {forecast_base_sql}",
                    forecast_params + metric_params + status_params,
                    "scope_countries_with_forecasts_failed",
                )
            else:
                notes.append("countries_with_forecasts_ignored_missing_column")
            scope["forecasts_by_hazard"] = _fetch_hazard_counts(
                f"""
                SELECT q.hazard_code, COUNT(DISTINCT q.question_id)
                {forecast_base_sql}
                GROUP BY q.hazard_code
                ORDER BY q.hazard_code
                """,
                forecast_params + metric_params + status_params,
                "scope_forecasts_by_hazard_failed",
            )
        elif forecast_source_table and forecast_source_ts:
            forecast_ids_sql = question_ids_sql
            forecast_params = list(question_ids_params)
            if (
                (forecast_source_table != question_source_table or forecast_source_phase)
                and len(question_ids_params) >= 2
            ):
                forecast_ids_sql = (
                    f"SELECT DISTINCT question_id FROM {forecast_source_table} "
                    f"WHERE {forecast_source_ts} >= ? AND {forecast_source_ts} < ?"
                )
                forecast_params = list(question_ids_params[:2])
                if forecast_source_phase:
                    placeholders = ", ".join(["?"] * len(forecast_source_phase))
                    forecast_ids_sql += f" AND phase IN ({placeholders})"
                    forecast_params.extend(forecast_source_phase)

            forecast_base_sql = (
                f"FROM questions q JOIN ({forecast_ids_sql}) f ON f.question_id = q.question_id "
                f"WHERE 1=1{metric_sql}{status_sql}"
            )
            scope["forecasts"] = _count(
                f"SELECT COUNT(DISTINCT q.question_id) {forecast_base_sql}",
                forecast_params + metric_params + status_params,
                "scope_forecasts_failed",
            )
            if has_iso3:
                scope["countries_with_forecasts"] = _count(
                    f"SELECT COUNT(DISTINCT q.iso3) {forecast_base_sql}",
                    forecast_params + metric_params + status_params,
                    "scope_countries_with_forecasts_failed",
                )
            else:
                notes.append("countries_with_forecasts_ignored_missing_column")
        else:
            notes.append("forecasts_unavailable")

        scope["countries"] = scope["countries_with_forecasts"]
        return scope

    def _scope_from_questions(status_filter: Optional[str] = None) -> Dict[str, Any]:
        scope: Dict[str, Any] = {
            "questions": 0,
            "forecasts": 0,
            "countries": 0,
            "countries_total": 0,
            "countries_with_forecasts": 0,
            "resolved_questions": 0,
            "forecasts_by_hazard": {},
        }
        if not has_questions:
            return scope

        metric_sql, metric_params = metric_clause()
        status_sql, status_params = ("", [])
        if status_filter:
            status_sql, status_params = status_clause(status_filter)

        base_sql = f"FROM questions q WHERE 1=1{metric_sql}{status_sql}"
        scope["questions"] = _count(
            f"SELECT COUNT(DISTINCT q.question_id) {base_sql}",
            metric_params + status_params,
            "scope_questions_failed",
        )

        if has_iso3:
            scope["countries_total"] = _count(
                f"SELECT COUNT(DISTINCT q.iso3) {base_sql}",
                metric_params + status_params,
                "scope_countries_failed",
            )
        else:
            notes.append("countries_ignored_missing_column")

        if _table_has_columns(con, "resolutions", ["question_id"]):
            scope["resolved_questions"] = _count(
                f"SELECT COUNT(DISTINCT r.question_id) FROM resolutions r "
                f"JOIN questions q ON r.question_id = q.question_id "
                f"WHERE 1=1{metric_sql}",
                metric_params,
                "scope_resolutions_failed",
            )
        elif has_status:
            resolved_sql = f"{base_sql} AND q.status IN ('resolved', 'closed')"
            scope["resolved_questions"] = _count(
                f"SELECT COUNT(DISTINCT q.question_id) {resolved_sql}",
                metric_params + status_params,
                "scope_resolved_failed",
            )
        else:
            notes.append("resolved_questions_unavailable")

        forecast_base_params = metric_params + status_params
        if forecast_source_table and forecast_source_ts:
            if forecast_source_table == "forecasts_ensemble":
                forecast_join = (
                    "FROM questions q JOIN forecasts_ensemble f ON f.question_id = q.question_id "
                    f"WHERE 1=1{metric_sql}{status_sql}"
                )
                scope["forecasts"] = _count(
                    f"SELECT COUNT(DISTINCT q.question_id) {forecast_join}",
                    forecast_base_params,
                    "scope_forecasts_failed",
                )
                if has_iso3:
                    scope["countries_with_forecasts"] = _count(
                        f"SELECT COUNT(DISTINCT q.iso3) {forecast_join}",
                        forecast_base_params,
                        "scope_countries_with_forecasts_failed",
                    )
                else:
                    notes.append("countries_with_forecasts_ignored_missing_column")
                if has_hazard:
                    scope["forecasts_by_hazard"] = _fetch_hazard_counts(
                        f"""
                        SELECT q.hazard_code, COUNT(DISTINCT q.question_id)
                        {forecast_join}
                        GROUP BY q.hazard_code
                        ORDER BY q.hazard_code
                        """,
                        forecast_base_params,
                        "scope_forecasts_by_hazard_failed",
                    )
            elif forecast_source_table == "llm_calls" and forecast_source_phase:
                placeholders = ", ".join(["?"] * len(forecast_source_phase))
                forecast_join = (
                    "FROM questions q JOIN llm_calls l ON l.question_id = q.question_id "
                    f"WHERE l.phase IN ({placeholders}){metric_sql}{status_sql}"
                )
                params = list(forecast_source_phase) + forecast_base_params
                scope["forecasts"] = _count(
                    f"SELECT COUNT(DISTINCT q.question_id) {forecast_join}",
                    params,
                    "scope_forecasts_failed",
                )
                if has_iso3:
                    scope["countries_with_forecasts"] = _count(
                        f"SELECT COUNT(DISTINCT q.iso3) {forecast_join}",
                        params,
                        "scope_countries_with_forecasts_failed",
                    )
                else:
                    notes.append("countries_with_forecasts_ignored_missing_column")
                if has_hazard:
                    scope["forecasts_by_hazard"] = _fetch_hazard_counts(
                        f"""
                        SELECT q.hazard_code, COUNT(DISTINCT q.question_id)
                        {forecast_join}
                        GROUP BY q.hazard_code
                        ORDER BY q.hazard_code
                        """,
                        params,
                        "scope_forecasts_by_hazard_failed",
                    )
        else:
            notes.append("forecasts_unavailable")

        scope["countries"] = scope["countries_with_forecasts"]
        return scope

    # ── Available forecast runs for the selected month ──
    # Determined FIRST so selected_run_id can scope KPIs below.
    available_runs: List[Dict[str, Any]] = []
    selected_run_id: Optional[str] = None
    if selected_month and _table_has_columns(con, "forecasts_ensemble", ["run_id"]):
        parsed_sel = _parse_year_month(selected_month)
        fe_ts = _pick_timestamp_column(
            con, "forecasts_ensemble", ["created_at", "timestamp", "started_at"]
        )
        if parsed_sel and fe_ts:
            sel_start, sel_end = _month_window(parsed_sel[0], parsed_sel[1])
            try:
                _tf_fe = _test_filter(include_test, "fe")
                _has_is_test = _table_has_columns(con, "forecasts_ensemble", ["is_test"])
                _is_test_expr = "COALESCE(fe.is_test, FALSE)" if _has_is_test else "FALSE"
                runs_rows = con.execute(
                    f"""
                    SELECT fe.run_id,
                           MIN(fe.{fe_ts}) AS started_at,
                           COUNT(DISTINCT fe.question_id) AS n_questions,
                           BOOL_OR({_is_test_expr}) AS is_test
                    FROM forecasts_ensemble fe
                    WHERE fe.{fe_ts} >= ? AND fe.{fe_ts} < ?
                      AND fe.run_id IS NOT NULL
                      {_tf_fe}
                    GROUP BY fe.run_id
                    ORDER BY fe.run_id DESC
                    """,
                    [sel_start, sel_end],
                ).fetchall()
                for idx, rr in enumerate(runs_rows):
                    available_runs.append({
                        "run_id": rr[0],
                        "started_at": str(rr[1]) if rr[1] else None,
                        "n_questions": int(rr[2]) if rr[2] else 0,
                        "is_latest": idx == 0,
                        "is_test": bool(rr[3]) if rr[3] is not None else False,
                    })
                if available_runs:
                    selected_run_id = available_runs[0]["run_id"]
            except Exception:
                notes.append("available_runs_failed")

    # ── Scope KPIs ──
    # Use the effective run: explicit forecaster_run_id, or the auto-selected latest run.
    effective_run_id = forecaster_run_id or selected_run_id

    selected_scope: Dict[str, Any] = {
        "label": f"Selected run {selected_month}" if selected_month else "Selected run",
        "questions": 0,
        "forecasts": 0,
        "countries": 0,
        "countries_triaged": 0,
        "countries_with_forecasts": 0,
        "resolved_questions": 0,
        "forecasts_by_hazard": {},
    }

    if effective_run_id and _table_has_columns(con, "forecasts_ensemble", ["run_id", "question_id"]):
        # ── Scope KPIs to the specific forecaster run ──
        question_ids_sql = (
            "SELECT DISTINCT fe.question_id FROM forecasts_ensemble fe WHERE fe.run_id = ?"
        )
        question_ids_params: List[Any] = [effective_run_id]
        forecast_window_ym = None
        if selected_month:
            parsed = _parse_year_month(selected_month)
            if parsed:
                _fw_start = _shift_ym(parsed[0], parsed[1], 1)
                _fw_end = _shift_ym(parsed[0], parsed[1], 7)
                forecast_window_ym = (
                    f"{_fw_start[0]:04d}-{_fw_start[1]:02d}",
                    f"{_fw_end[0]:04d}-{_fw_end[1]:02d}",
                )
        selected_scope = _scope_from_question_ids(
            question_ids_sql, question_ids_params,
            forecast_window_ym=forecast_window_ym,
        )
        # Derive countries_triaged from ALL countries in the HS run(s) that
        # fed this forecaster run — including degraded triage results.
        try:
            triaged_row = con.execute(
                """
                SELECT COUNT(DISTINCT UPPER(ht.iso3))
                FROM hs_triage ht
                WHERE ht.run_id IN (
                    SELECT DISTINCT q.hs_run_id
                    FROM questions q
                    JOIN forecasts_ensemble fe ON fe.question_id = q.question_id
                    WHERE fe.run_id = ?
                      AND q.hs_run_id IS NOT NULL
                )
                AND ht.iso3 IS NOT NULL
                """,
                [effective_run_id],
            ).fetchone()
            selected_scope["countries_triaged"] = int(triaged_row[0]) if triaged_row else 0
        except Exception:
            logger.debug("countries_triaged for forecaster_run_id failed", exc_info=True)
            selected_scope["countries_triaged"] = 0
        diagnostics["countries_triaged_source"] = "forecaster_run_id"
        selected_scope.pop("countries_total", None)
    elif selected_month and month_source_table and month_source_ts:
        parsed = _parse_year_month(selected_month)
        if parsed:
            start_iso, end_iso = _month_window(parsed[0], parsed[1])
            llm_ts = _pick_timestamp_column(
                con, "llm_calls", ["created_at", "timestamp", "started_at"]
            )
            if llm_ts and _table_has_columns(con, "llm_calls", ["question_id", llm_ts]):
                question_source_table = "llm_calls"
                question_source_ts = llm_ts
            elif month_source_table and month_source_ts:
                question_source_table = month_source_table
                question_source_ts = month_source_ts

            if question_source_table and question_source_ts:
                question_ids_sql = (
                    f"SELECT DISTINCT question_id FROM {question_source_table} "
                    f"WHERE {question_source_ts} >= ? AND {question_source_ts} < ?"
                )
                question_ids_params = [start_iso, end_iso]
                _fw_start = _shift_ym(parsed[0], parsed[1], 1)
                _fw_end = _shift_ym(parsed[0], parsed[1], 7)
                forecast_window_ym = (
                    f"{_fw_start[0]:04d}-{_fw_start[1]:02d}",
                    f"{_fw_end[0]:04d}-{_fw_end[1]:02d}",
                )
                selected_scope = _scope_from_question_ids(
                    question_ids_sql, question_ids_params,
                    forecast_window_ym=forecast_window_ym,
                )
                countries_triaged, triaged_source = (
                    compute_countries_triaged_for_month_with_source(con, selected_month)
                )
                selected_scope["countries_triaged"] = countries_triaged
                selected_scope.pop("countries_total", None)
                if triaged_source:
                    diagnostics["countries_triaged_source"] = triaged_source
            else:
                notes.append("selected_run_questions_unavailable")
        else:
            notes.append("selected_month_invalid")

    # Override selected_run_id when a specific forecaster run was requested.
    if forecaster_run_id:
        selected_run_id = forecaster_run_id

    explanations = []

    return {
        "available_months": available_month_rows,
        "available_runs": available_runs,
        "selected_month": selected_month,
        "selected_run_id": selected_run_id,
        "scopes": {
            "selected_run": selected_scope,
            "total_active": {
                "label": "Total active",
                **_scope_from_questions(status_filter="active"),
            },
            "total_all": {
                "label": "Total active + inactive",
                **_scope_from_questions(),
            },
        },
        "explanations": explanations,
        "diagnostics": diagnostics,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Run summary endpoint — aggregate KPIs for the "All metrics" summary view
# ---------------------------------------------------------------------------

@router.get("/v1/diagnostics/run_summary")
def diagnostics_run_summary(
    year_month: Optional[str] = Query(None),
    forecaster_run_id: Optional[str] = Query(None),
    include_test: bool = Query(False),
):
    """Return aggregate run-level KPIs for the summary dashboard view."""
    con = _con()
    tf_q = _test_filter(include_test, "q")
    tf_ht = _test_filter(include_test, "ht")
    tf_lc = _test_filter(include_test, "lc")

    # ---- Discover run ids ------------------------------------------------
    hs_run_id: Optional[str] = None
    run_id: Optional[str] = None
    updated_at: Optional[str] = None

    # If forecaster_run_id given, use it directly
    if forecaster_run_id:
        run_id = forecaster_run_id
    else:
        # Find latest forecaster run_id from forecasts_ensemble for the month
        if _table_exists(con, "forecasts_ensemble"):
            fe_ts = _pick_timestamp_column(con, "forecasts_ensemble", ["created_at", "timestamp"])
            if fe_ts:
                sql = f"SELECT DISTINCT run_id FROM forecasts_ensemble"
                params: list = []
                if year_month:
                    sql += f" WHERE strftime({fe_ts}, '%Y-%m') = ?"
                    params.append(year_month)
                sql += f" ORDER BY run_id DESC LIMIT 1"
                try:
                    row = con.execute(sql, params).fetchone()
                    if row:
                        run_id = row[0]
                except Exception:
                    pass

    # Derive hs_run_id from questions linked to this forecaster run
    if run_id and _table_exists(con, "questions") and _table_exists(con, "forecasts_ensemble"):
        try:
            row = con.execute(
                "SELECT DISTINCT q.hs_run_id FROM questions q "
                "JOIN forecasts_ensemble fe ON fe.question_id = q.question_id "
                "WHERE fe.run_id = ? AND q.hs_run_id IS NOT NULL LIMIT 1",
                [run_id],
            ).fetchone()
            if row:
                hs_run_id = row[0]
        except Exception:
            pass

    # If still no hs_run_id, try hs_triage directly
    if not hs_run_id and _table_exists(con, "hs_triage"):
        try:
            sql = "SELECT DISTINCT run_id FROM hs_triage"
            params = []
            if year_month:
                ht_ts = _pick_timestamp_column(con, "hs_triage", ["created_at", "timestamp"])
                if ht_ts:
                    sql += f" WHERE strftime({ht_ts}, '%Y-%m') = ?"
                    params.append(year_month)
            sql += " ORDER BY run_id DESC LIMIT 1"
            row = con.execute(sql, params).fetchone()
            if row:
                hs_run_id = row[0]
        except Exception:
            pass

    # Get updated_at timestamp
    if hs_run_id and _table_exists(con, "hs_triage"):
        ht_ts = _pick_timestamp_column(con, "hs_triage", ["created_at", "timestamp"])
        if ht_ts:
            try:
                row = con.execute(
                    f"SELECT MAX({ht_ts}) FROM hs_triage WHERE run_id = ?",
                    [hs_run_id],
                ).fetchone()
                if row and row[0]:
                    updated_at = str(row[0])
            except Exception:
                pass

    # ---- Helper: safe query -----------------------------------------------
    def _q(sql: str, params: list = []) -> list:
        try:
            return con.execute(sql, params).fetchall()
        except Exception:
            return []

    def _q1(sql: str, params: list = []):
        try:
            row = con.execute(sql, params).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    # ---- Coverage funnel --------------------------------------------------
    coverage: Dict[str, Any] = {
        "countries_scanned": 0,
        "hazard_pairs_assessed": 0,
        "seasonal_screenouts": 0,
        "acled_low_activity": 0,
        "pairs_with_questions": 0,
        "total_questions": 0,
        "countries_with_forecasts": 0,
        "countries_no_questions": 0,
        "triaged_quiet": 0,
    }

    has_dq_json = False
    if hs_run_id and _table_exists(con, "hs_triage"):
        has_dq_json = "data_quality_json" in _table_columns(con, "hs_triage")

        coverage["countries_scanned"] = _q1(
            f"SELECT COUNT(DISTINCT UPPER(ht.iso3)) FROM hs_triage ht "
            f"WHERE ht.run_id = ?{tf_ht}",
            [hs_run_id],
        ) or 0

        # Seasonal screen-outs are marked via data_quality_json.status = 'seasonal_skip'
        if has_dq_json:
            coverage["seasonal_screenouts"] = _q1(
                f"SELECT COUNT(*) FROM hs_triage ht "
                f"WHERE ht.run_id = ?{tf_ht} "
                f"AND ht.data_quality_json LIKE '%seasonal_skip%'",
                [hs_run_id],
            ) or 0

            # ACLED low-activity (quiet conflict) screen-outs
            coverage["acled_low_activity"] = _q1(
                f"SELECT COUNT(*) FROM hs_triage ht "
                f"WHERE ht.run_id = ?{tf_ht} "
                f"AND ht.data_quality_json LIKE '%acled_low_activity%'",
                [hs_run_id],
            ) or 0

        total_triage_rows = _q1(
            f"SELECT COUNT(*) FROM hs_triage ht WHERE ht.run_id = ?{tf_ht}",
            [hs_run_id],
        ) or 0

        # Hazard pairs assessed = total minus seasonal skips and ACLED low-activity
        coverage["hazard_pairs_assessed"] = (
            total_triage_rows
            - coverage["seasonal_screenouts"]
            - coverage["acled_low_activity"]
        )

        # Triaged quiet: tier = 'quiet' excluding seasonal skips, ACLED low-activity, and RC-promoted
        if has_dq_json:
            coverage["triaged_quiet"] = _q1(
                f"SELECT COUNT(*) FROM hs_triage ht "
                f"WHERE ht.run_id = ?{tf_ht} "
                f"AND LOWER(ht.tier) = 'quiet' "
                f"AND ht.data_quality_json NOT LIKE '%seasonal_skip%' "
                f"AND ht.data_quality_json NOT LIKE '%acled_low_activity%' "
                f"AND ht.data_quality_json NOT LIKE '%rc_promoted%' "
                f"AND COALESCE(ht.regime_change_level, 0) = 0",
                [hs_run_id],
            ) or 0
        else:
            coverage["triaged_quiet"] = _q1(
                f"SELECT COUNT(*) FROM hs_triage ht "
                f"WHERE ht.run_id = ?{tf_ht} "
                f"AND LOWER(ht.tier) = 'quiet' "
                f"AND ht.regime_change_likelihood IS NOT NULL "
                f"AND COALESCE(ht.regime_change_level, 0) = 0",
                [hs_run_id],
            ) or 0

    # Questions coverage (from questions table linked to this run)
    q_filter = ""
    q_params: list = []
    if run_id and _table_exists(con, "forecasts_ensemble"):
        q_filter = (
            f"q.question_id IN ("
            f"SELECT DISTINCT fe.question_id FROM forecasts_ensemble fe WHERE fe.run_id = ?"
            f")"
        )
        q_params = [run_id]
    elif hs_run_id and _table_exists(con, "questions"):
        q_filter = "q.hs_run_id = ?"
        q_params = [hs_run_id]

    if q_filter and _table_exists(con, "questions"):
        coverage["total_questions"] = _q1(
            f"SELECT COUNT(DISTINCT q.question_id) FROM questions q WHERE {q_filter}{tf_q}",
            q_params,
        ) or 0

        coverage["countries_with_forecasts"] = _q1(
            f"SELECT COUNT(DISTINCT UPPER(q.iso3)) FROM questions q WHERE {q_filter}{tf_q}",
            q_params,
        ) or 0

        coverage["pairs_with_questions"] = _q1(
            f"SELECT COUNT(DISTINCT UPPER(q.iso3) || '_' || UPPER(q.hazard_code)) "
            f"FROM questions q WHERE {q_filter}{tf_q}",
            q_params,
        ) or 0

        # Countries with no questions
        if hs_run_id and _table_exists(con, "hs_triage"):
            coverage["countries_no_questions"] = _q1(
                f"SELECT COUNT(DISTINCT UPPER(ht.iso3)) FROM hs_triage ht "
                f"WHERE ht.run_id = ?{tf_ht} "
                f"AND UPPER(ht.iso3) NOT IN ("
                f"  SELECT DISTINCT UPPER(q.iso3) FROM questions q WHERE {q_filter}{tf_q}"
                f")",
                [hs_run_id] + q_params,
            ) or 0

    # ---- Metrics breakdown ------------------------------------------------
    metrics_list: list = []
    METRIC_DEFS = [
        ("FATALITIES", "Fatalities"),
        ("PA", "People affected"),
        ("EVENT_OCCURRENCE", "Event occurrence"),
        ("PHASE3PLUS_IN_NEED", "Phase 3+ population"),
    ]
    if q_filter and _table_exists(con, "questions"):
        for metric_code, metric_label in METRIC_DEFS:
            q_count = _q1(
                f"SELECT COUNT(DISTINCT q.question_id) FROM questions q "
                f"WHERE {q_filter}{tf_q} AND UPPER(q.metric) = ?",
                q_params + [metric_code],
            ) or 0
            c_count = _q1(
                f"SELECT COUNT(DISTINCT UPPER(q.iso3)) FROM questions q "
                f"WHERE {q_filter}{tf_q} AND UPPER(q.metric) = ?",
                q_params + [metric_code],
            ) or 0
            hazard_rows = _q(
                f"SELECT UPPER(q.hazard_code) AS hc, COUNT(DISTINCT q.question_id) AS n "
                f"FROM questions q WHERE {q_filter}{tf_q} AND UPPER(q.metric) = ? "
                f"GROUP BY UPPER(q.hazard_code) ORDER BY n DESC",
                q_params + [metric_code],
            )
            hazards = [{"hazard_code": str(h), "count": int(n)} for h, n in hazard_rows]
            metrics_list.append({
                "metric": metric_code,
                "label": metric_label,
                "questions": q_count,
                "countries": c_count,
                "hazards": hazards,
            })

    # ---- RC assessment ----------------------------------------------------
    rc_assessment: Dict[str, Any] = {
        "total_assessed": 0,
        "levels": {"L0": 0, "L1": 0, "L2": 0, "L3": 0},
        "l1_plus_rate": 0.0,
        "by_hazard": [],
        "countries_by_level": {"L1": 0, "L2": 0, "L3": 0},
    }

    if hs_run_id and _table_exists(con, "hs_triage"):
        # RC level counts (regime_change_level is pre-computed in the table)
        rc_rows = _q(
            f"SELECT COALESCE(ht.regime_change_level, 0) AS rc_level, COUNT(*) AS n "
            f"FROM hs_triage ht "
            f"WHERE ht.run_id = ?{tf_ht} AND ht.regime_change_likelihood IS NOT NULL "
            f"GROUP BY COALESCE(ht.regime_change_level, 0)",
            [hs_run_id],
        )
        total_assessed = 0
        for rc_level_val, cnt in rc_rows:
            cnt = int(cnt)
            total_assessed += cnt
            if rc_level_val == 0:
                rc_assessment["levels"]["L0"] = cnt
            elif rc_level_val == 1:
                rc_assessment["levels"]["L1"] = cnt
            elif rc_level_val == 2:
                rc_assessment["levels"]["L2"] = cnt
            elif rc_level_val >= 3:
                rc_assessment["levels"]["L3"] = rc_assessment["levels"].get("L3", 0) + cnt
        rc_assessment["total_assessed"] = total_assessed
        l1_plus = (
            rc_assessment["levels"]["L1"]
            + rc_assessment["levels"]["L2"]
            + rc_assessment["levels"]["L3"]
        )
        rc_assessment["l1_plus_rate"] = round(l1_plus / total_assessed, 3) if total_assessed else 0.0

        # By hazard breakdown
        hazard_rc_rows = _q(
            f"SELECT UPPER(ht.hazard_code) AS hc, "
            f"  COALESCE(ht.regime_change_level, 0) AS rc_level, COUNT(*) AS n "
            f"FROM hs_triage ht "
            f"WHERE ht.run_id = ?{tf_ht} AND ht.regime_change_likelihood IS NOT NULL "
            f"GROUP BY UPPER(ht.hazard_code), COALESCE(ht.regime_change_level, 0) "
            f"ORDER BY UPPER(ht.hazard_code)",
            [hs_run_id],
        )
        hazard_rc: Dict[str, Dict[str, int]] = {}
        for hc, rc_lev, cnt in hazard_rc_rows:
            hc = str(hc)
            if hc not in hazard_rc:
                hazard_rc[hc] = {"L0": 0, "L1": 0, "L2": 0, "L3": 0}
            key = f"L{min(int(rc_lev), 3)}"
            hazard_rc[hc][key] = hazard_rc[hc].get(key, 0) + int(cnt)
        rc_assessment["by_hazard"] = [
            {"hazard_code": hc, **levels} for hc, levels in sorted(hazard_rc.items())
        ]

        # Countries by level: distinct iso3 where max RC level across hazards = that level
        for level_val, level_key in [(1, "L1"), (2, "L2"), (3, "L3")]:
            rc_assessment["countries_by_level"][level_key] = _q1(
                f"SELECT COUNT(*) FROM ("
                f"  SELECT UPPER(ht.iso3) AS iso3 FROM hs_triage ht "
                f"  WHERE ht.run_id = ?{tf_ht} "
                f"  AND ht.regime_change_likelihood IS NOT NULL "
                f"  GROUP BY UPPER(ht.iso3) "
                f"  HAVING MAX(COALESCE(ht.regime_change_level, 0)) = ?"
                f")",
                [hs_run_id, level_val],
            ) or 0

    # ---- Track split ------------------------------------------------------
    tracks: Dict[str, Any] = {
        "track1": {"questions": 0, "countries": 0, "models": 0},
        "track2": {"questions": 0, "countries": 0},
    }

    if q_filter and _table_exists(con, "questions"):
        has_track = "track" in _table_columns(con, "questions")
        if has_track:
            for track_val, track_key in [(1, "track1"), (2, "track2")]:
                tracks[track_key]["questions"] = _q1(
                    f"SELECT COUNT(DISTINCT q.question_id) FROM questions q "
                    f"WHERE {q_filter}{tf_q} AND q.track = ?",
                    q_params + [track_val],
                ) or 0
                tracks[track_key]["countries"] = _q1(
                    f"SELECT COUNT(DISTINCT UPPER(q.iso3)) FROM questions q "
                    f"WHERE {q_filter}{tf_q} AND q.track = ?",
                    q_params + [track_val],
                ) or 0

    # Ensemble model count from forecasts_raw (Track 1 questions only)
    if run_id and _table_exists(con, "forecasts_raw"):
        model_col = "model_name" if "model_name" in _table_columns(con, "forecasts_raw") else None
        if model_col:
            # Count distinct models for Track 1 questions only (exclude track2_flash, ensemble aggregation)
            if q_filter and _table_exists(con, "questions") and "track" in _table_columns(con, "questions"):
                tracks["track1"]["models"] = _q1(
                    f"SELECT COUNT(DISTINCT fr.{model_col}) FROM forecasts_raw fr "
                    f"JOIN questions q ON fr.question_id = q.question_id "
                    f"WHERE fr.run_id = ? AND COALESCE(fr.ok, TRUE) = TRUE "
                    f"AND q.track = 1 "
                    f"AND fr.{model_col} NOT LIKE 'ensemble_%' "
                    f"AND fr.{model_col} NOT LIKE 'track2_%'",
                    [run_id],
                ) or 0
            else:
                tracks["track1"]["models"] = _q1(
                    f"SELECT COUNT(DISTINCT fr.{model_col}) FROM forecasts_raw fr "
                    f"WHERE fr.run_id = ? AND COALESCE(fr.ok, TRUE) = TRUE",
                    [run_id],
                ) or 0

    # ---- Ensemble health --------------------------------------------------
    # Expected member count comes from the configured SPD ensemble (falls
    # back to the current 5-member lineup if config is unreadable). Also
    # surface the specific Track 2 model id so the dashboard never has to
    # hardcode a generic label like "Gemini Flash".
    try:
        from pythia.llm_profiles import get_ensemble_resolved, get_role_model, split_model_ref

        expected_members = len(get_ensemble_resolved()) or 5
        tracks["track2"]["model"] = split_model_ref(get_role_model("track2_spd"))[1]
    except Exception:
        expected_members = 5
    ensemble: Dict[str, int] = {"expected": expected_members, "ok": tracks["track1"]["models"]}

    # ---- Cost breakdown ---------------------------------------------------
    cost: Dict[str, Any] = {
        "total_usd": 0.0,
        "total_tokens": 0,
        "by_phase": [],
    }

    llm_health: Dict[str, Any] = {
        "total_calls": 0,
        "errors": 0,
        "error_rate": 0.0,
    }

    if _table_exists(con, "llm_calls"):
        # Build filter for this run's llm_calls
        lc_filters = []
        lc_params: list = []
        if run_id:
            lc_filters.append("lc.run_id = ?")
            lc_params.append(run_id)
        if hs_run_id:
            if lc_filters:
                lc_filters = [f"({lc_filters[0]} OR lc.hs_run_id = ?)"]
                lc_params.append(hs_run_id)
            else:
                lc_filters.append("lc.hs_run_id = ?")
                lc_params.append(hs_run_id)

        if lc_filters:
            where = " AND ".join(lc_filters)

            # Total cost and tokens
            row = _q(
                f"SELECT COALESCE(SUM(lc.cost_usd), 0), COALESCE(SUM(lc.total_tokens), 0) "
                f"FROM llm_calls lc WHERE {where}{tf_lc}",
                lc_params,
            )
            if row:
                cost["total_usd"] = round(float(row[0][0]), 2)
                cost["total_tokens"] = int(row[0][1])

            # By phase
            PHASE_LABELS = [
                ("hs_triage", "Horizon Scan Triage and RC"),
                ("spd_v2", "Ensemble Forecasts"),
                ("binary_v2", "Binary Forecasts"),
                ("scenario_v2", "Scenarios"),
            ]
            for phase_val, phase_label in PHASE_LABELS:
                phase_cost = _q1(
                    f"SELECT COALESCE(SUM(lc.cost_usd), 0) FROM llm_calls lc "
                    f"WHERE {where}{tf_lc} AND lc.phase = ?",
                    lc_params + [phase_val],
                )
                cost["by_phase"].append({
                    "phase": phase_val,
                    "label": phase_label,
                    "cost_usd": round(float(phase_cost or 0), 2),
                })

            # LLM health
            llm_health["total_calls"] = _q1(
                f"SELECT COUNT(*) FROM llm_calls lc WHERE {where}{tf_lc}",
                lc_params,
            ) or 0

            # Error detection: check for non-empty error_text or status = 'error'
            has_error_text = "error_text" in _table_columns(con, "llm_calls")
            has_status = "status" in _table_columns(con, "llm_calls")
            error_cond = ""
            if has_error_text and has_status:
                error_cond = (
                    " AND ((lc.error_text IS NOT NULL AND lc.error_text != '')"
                    " OR lc.status = 'error')"
                )
            elif has_error_text:
                error_cond = " AND lc.error_text IS NOT NULL AND lc.error_text != ''"
            elif has_status:
                error_cond = " AND lc.status = 'error'"

            if error_cond:
                llm_health["errors"] = _q1(
                    f"SELECT COUNT(*) FROM llm_calls lc WHERE {where}{tf_lc}{error_cond}",
                    lc_params,
                ) or 0

            if llm_health["total_calls"] > 0:
                llm_health["error_rate"] = round(
                    llm_health["errors"] / llm_health["total_calls"], 3
                )

    # ---- Performance scores -------------------------------------------------
    performance: Dict[str, Any] = {
        "resolved_questions": 0,
        "total_questions": coverage.get("total_questions", 0),
        "brier": {"avg": None, "median": None},
        "log": {"avg": None, "median": None},
        "crps": {"avg": None, "median": None},
    }
    if _table_exists(con, "resolutions") and q_filter:
        performance["resolved_questions"] = _q1(
            f"SELECT COUNT(DISTINCT r.question_id) FROM resolutions r "
            f"JOIN questions q ON r.question_id = q.question_id "
            f"WHERE {q_filter}{tf_q}",
            q_params,
        ) or 0

    if _table_exists(con, "scores") and q_filter:
        for score_type in ["brier", "log", "crps"]:
            row = _q(
                f"SELECT AVG(s.value), MEDIAN(s.value) FROM scores s "
                f"JOIN questions q ON s.question_id = q.question_id "
                f"WHERE {q_filter}{tf_q} AND s.score_type = ? "
                f"AND s.model_name LIKE 'ensemble_%'",
                q_params + [score_type],
            )
            if row and row[0][0] is not None:
                performance[score_type] = {
                    "avg": round(float(row[0][0]), 4),
                    "median": round(float(row[0][1]), 4) if row[0][1] is not None else None,
                }

    # ---- Sibyl parallel track coverage -------------------------------------
    sibyl_block: Optional[Dict[str, Any]] = None
    if _table_exists(con, "sibyl_runs"):
        sibyl_run = _q(
            "SELECT sibyl_run_id, budget_capped, run_cost_usd, opus_cost_usd, "
            "brave_cost_usd, n_selected, n_forecast, n_skipped, k, aggregation "
            "FROM sibyl_runs "
            + ("WHERE hs_run_id = ? " if hs_run_id else "")
            + "ORDER BY created_at DESC LIMIT 1",
            [hs_run_id] if hs_run_id else [],
        )
        if sibyl_run:
            (s_run_id, s_capped, s_cost, s_opus, s_brave,
             s_selected, s_forecast, s_skipped, s_k, s_agg) = sibyl_run[0]
            skipped_by_cap = _q1(
                "SELECT COUNT(*) FROM sibyl_forecasts "
                "WHERE sibyl_run_id = ? AND status = 'skipped' "
                "AND skip_reason = 'run budget cap'",
                [s_run_id],
            ) or 0
            sibyl_block = {
                "sibyl_run_id": s_run_id,
                "budget_capped": bool(s_capped),
                "run_cost_usd": round(float(s_cost or 0.0), 2),
                "opus_cost_usd": round(float(s_opus or 0.0), 2),
                "brave_cost_usd": round(float(s_brave or 0.0), 2),
                "n_selected": int(s_selected or 0),
                "n_forecast": int(s_forecast or 0),
                "n_skipped": int(s_skipped or 0),
                "n_skipped_budget_cap": int(skipped_by_cap),
                "k": int(s_k or 0),
                "aggregation": s_agg,
            }

    return {
        "run_id": run_id,
        "hs_run_id": hs_run_id,
        "updated_at": updated_at,
        "coverage": coverage,
        "metrics": metrics_list,
        "rc_assessment": rc_assessment,
        "tracks": tracks,
        "ensemble": ensemble,
        "cost": cost,
        "performance": performance,
        "llm_health": llm_health,
        "sibyl": sibyl_block,
    }
