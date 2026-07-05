# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Question routes: /v1/questions and /v1/question_bundle.

Endpoint functions moved verbatim from pythia.api.app (July 2026
decomposition); shared helpers come from pythia.api.core.
"""

import logging
from typing import Any, Dict, List, Optional

import duckdb
from fastapi import APIRouter, HTTPException, Query

from pythia.api.core import (
    _HEAVY_REQUEST_SEMAPHORE,
    _apply_json_fields,
    _bucket_centroids,
    _bucket_labels,
    _con,
    _execute,
    _fetch_one,
    _json_sanitize,
    _latest_questions_view,
    _resolve_forecaster_run_id,
    _resolve_latest_questions_columns,
    _rows_from_cursor,
    _safe_json_load,
    _table_columns,
    _table_exists,
    _table_has_columns,
    _test_filter,
)
from pythia.api.models import (
    ContextBundle,
    ForecastBundle,
    HsBundle,
    LlmCallsBundle,
    QuestionBundleResponse,
)
from resolver.query.questions_index import (
    compute_questions_forecast_summary,
    compute_questions_triage_summary,
)

logger = logging.getLogger(__name__)

router = APIRouter()

def _resolve_question_row(
    con: duckdb.DuckDBPyConnection, question_id: str, hs_run_id: Optional[str]
) -> Dict[str, Any]:
    params: Dict[str, Any] = {"question_id": question_id}
    q_run_col, h_run_col, hs_timestamp_expr = _resolve_latest_questions_columns(con)
    hs_runs_exists = _table_exists(con, "hs_runs")

    join_clause = ""
    if hs_runs_exists and q_run_col and h_run_col:
        join_clause = f"LEFT JOIN hs_runs h ON q.{q_run_col} = h.{h_run_col}"

    sql = f"""
      SELECT q.*, {hs_timestamp_expr} AS hs_run_created_at
      FROM questions q
      {join_clause}
      WHERE q.question_id = :question_id
    """
    order_by_parts = [f"{hs_timestamp_expr} DESC NULLS LAST"]
    if q_run_col:
        order_by_parts.append(f"q.{q_run_col} DESC")
    sql += f" ORDER BY {', '.join(order_by_parts)} LIMIT 1"

    def fetch_row(query: str, query_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return _fetch_one(con, query, query_params)

    if hs_run_id and q_run_col:
        filtered_params = dict(params)
        filtered_params["hs_run_id"] = hs_run_id
        filtered_sql = sql.replace(
            "WHERE q.question_id = :question_id",
            f"WHERE q.question_id = :question_id AND q.{q_run_col} = :hs_run_id",
        )
        row = fetch_row(filtered_sql, filtered_params)
        if row:
            row["requested_hs_run_id"] = hs_run_id
            row["requested_hs_run_id_matched"] = True
            return row
        logger.warning(
            "Requested hs_run_id %s not found for question_id %s; falling back to latest run",
            hs_run_id,
            question_id,
        )
        row = fetch_row(sql, params)
        if row:
            row["requested_hs_run_id"] = hs_run_id
            row["requested_hs_run_id_matched"] = False
            return row
        available_run_ids: List[str] = []
        if q_run_col:
            available_rows = _rows_from_cursor(_execute(
                con,
                f"""
                SELECT DISTINCT q.{q_run_col} AS run_id
                FROM questions q
                WHERE q.question_id = :question_id
                ORDER BY q.{q_run_col} DESC
                LIMIT 10
                """,
                {"question_id": question_id},
            ))
            available_run_ids = [r["run_id"] for r in available_rows if r.get("run_id")]
        raise HTTPException(
            status_code=404,
            detail={
                "question_id": question_id,
                "requested_hs_run_id": hs_run_id,
                "q_run_col": q_run_col,
                "available_run_ids": available_run_ids,
            },
        )

    row = fetch_row(sql, params)
    if not row:
        raise HTTPException(status_code=404, detail="Question not found")
    if hs_run_id:
        row["requested_hs_run_id"] = hs_run_id
        row["requested_hs_run_id_matched"] = False
    return row


def _build_llm_calls_bundle(
    con: duckdb.DuckDBPyConnection,
    *,
    question_id: str,
    hs_run_id: Optional[str],
    forecaster_run_id: Optional[str],
    iso3: str,
    hazard_code: str,
    include_llm_calls: bool,
    include_transcripts: bool,
    limit_llm_calls: int,
    transcript_phases: Optional[List[str]] = None,
) -> LlmCallsBundle:
    if not include_llm_calls:
        return LlmCallsBundle(included=False, transcripts_included=False, rows=[], by_phase={})

    def build_debug_payload(by_phase: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        return {
            "hs_filter_requested": bool(hs_run_id and iso3 and hazard_code),
            "hs_run_id": hs_run_id,
            "iso3": iso3,
            "hazard_code": hazard_code,
            "phases": sorted(by_phase.keys()),
            "counts_by_phase": {key: len(value) for key, value in by_phase.items()},
        }

    if not _table_exists(con, "llm_calls"):
        return LlmCallsBundle(
            included=True,
            transcripts_included=False,
            rows=[],
            by_phase={},
            debug=build_debug_payload({}),
        )

    available_columns = _table_columns(con, "llm_calls")
    if not available_columns:
        return LlmCallsBundle(
            included=True,
            transcripts_included=False,
            rows=[],
            by_phase={},
            debug=build_debug_payload({}),
        )

    filters: List[str] = []
    params: Dict[str, Any] = {"limit": limit_llm_calls}
    transcripts_available = {"prompt_text", "response_text"}.issubset(available_columns)
    transcripts_included = include_transcripts and transcripts_available

    if forecaster_run_id:
        if not _table_has_columns(con, "llm_calls", ["run_id", "question_id", "phase"]):
            return LlmCallsBundle(
                included=True,
                transcripts_included=transcripts_included,
                rows=[],
                by_phase={},
                debug=build_debug_payload({}),
            )
        filters.append(
            "("
            "run_id = :forecaster_run_id "
            "AND question_id = :question_id "
            "AND phase IN ("
            "'research_v2','spd_v2','scenario_v2',"
            "'research_web_research','forecast_web_research'"
            ")"
            ")"
        )
        params["forecaster_run_id"] = forecaster_run_id
        params["question_id"] = question_id

    if hs_run_id and iso3 and hazard_code:
        if not _table_has_columns(con, "llm_calls", ["hs_run_id", "iso3", "hazard_code", "phase"]):
            return LlmCallsBundle(
                included=True,
                transcripts_included=transcripts_included,
                rows=[],
                by_phase={},
                debug=build_debug_payload({}),
            )
        filters.append(
            "("
            "hs_run_id = :hs_run_id "
            "AND UPPER(iso3) = :iso3 "
            "AND ("
            "  UPPER(hazard_code) = :hazard_code "
            "  OR UPPER(hazard_code) LIKE 'RC_' || :hazard_code || '%' "
            "  OR UPPER(hazard_code) LIKE 'GROUNDING_' || :hazard_code || '%' "
            "  OR UPPER(hazard_code) LIKE 'TRIAGE_' || :hazard_code || '%' "
            "  OR UPPER(hazard_code) LIKE 'TRIAGE_GROUNDING_' || :hazard_code || '%' "
            ") "
            "AND phase IN ('hs_triage','hs_web_research')"
            ")"
        )
        params["hs_run_id"] = hs_run_id
        params["iso3"] = iso3
        params["hazard_code"] = hazard_code

    if not filters:
        return LlmCallsBundle(
            included=True,
            transcripts_included=transcripts_included,
            rows=[],
            by_phase={},
            debug=build_debug_payload({}),
        )

    # ALWAYS exclude large TEXT columns from the main query to prevent OOM
    # on 512 MB Render instances.  Transcripts are fetched in a separate,
    # targeted query below (only for the phases the frontend actually needs).
    exclude_cols: set[str] = {"parsed_json", "prompt_text", "response_text"}
    select_columns = sorted(available_columns - exclude_cols)
    order_fields: List[str] = []
    if "timestamp" in available_columns:
        order_fields.append("timestamp DESC NULLS LAST")
    elif "created_at" in available_columns:
        order_fields.append("created_at DESC NULLS LAST")
    if "call_id" in available_columns:
        order_fields.append("call_id DESC")

    where_clause = " OR ".join(filters)
    sql = f"""
      SELECT {', '.join(select_columns)}
      FROM llm_calls
      WHERE {where_clause}
    """
    if order_fields:
        sql += " ORDER BY " + ", ".join(order_fields)
    sql += " LIMIT :limit"
    rows = _rows_from_cursor(_execute(con, sql, params))

    # Query 2: fetch transcripts ONLY for rows matching transcript_phases.
    # This keeps peak memory bounded (~4 MB for ~20 matching rows instead of
    # ~40 MB for all 200 rows).
    transcript_map: Dict[str, Dict[str, str]] = {}
    has_call_id = "call_id" in available_columns
    if transcripts_included and transcript_phases and has_call_id and transcripts_available:
        tp_conditions: List[str] = []
        tp_params = dict(params)  # copy base params (includes filters' params + limit)
        for i, tp in enumerate(transcript_phases):
            pk = f"tp_{i}"
            tp_params[pk] = tp.upper()
            tp_conditions.append(f"UPPER(phase) = :{pk}")
            tp_conditions.append(f"UPPER(hazard_code) LIKE :{pk} || '%'")
        phase_filter = " OR ".join(tp_conditions)
        transcript_sql = f"""
          SELECT call_id, prompt_text, response_text
          FROM llm_calls
          WHERE ({where_clause}) AND ({phase_filter})
          LIMIT :limit
        """
        for t_row in _rows_from_cursor(_execute(con, transcript_sql, tp_params)):
            cid = t_row.get("call_id")
            if cid:
                transcript_map[cid] = {
                    "prompt_text": t_row.get("prompt_text") or "",
                    "response_text": t_row.get("response_text") or "",
                }

    cleaned_rows: List[Dict[str, Any]] = []
    by_phase: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        parsed = _apply_json_fields(row, ["usage_json"])
        # Merge transcripts from the targeted query
        cid = parsed.get("call_id")
        if cid and cid in transcript_map:
            parsed["prompt_text"] = transcript_map[cid]["prompt_text"]
            parsed["response_text"] = transcript_map[cid]["response_text"]
        cleaned_rows.append(parsed)
        phase = parsed.get("phase")
        if phase:
            by_phase.setdefault(phase, []).append(parsed)

    return LlmCallsBundle(
        included=True,
        transcripts_included=transcripts_included,
        rows=cleaned_rows,
        by_phase=by_phase,
        debug=build_debug_payload(by_phase),
    )


@router.get("/v1/questions")
def get_questions(
    iso3: Optional[str] = Query(None),
    hazard_code: Optional[str] = Query(None),
    metric: Optional[str] = Query(None),
    target_month: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    run_id: Optional[str] = Query(None),
    latest_only: bool = Query(False),
    include_test: bool = Query(False),
):
    con = _con()
    run_col, _, _ = _resolve_latest_questions_columns(con)
    params = {}
    if iso3:
        params["iso3"] = iso3
    if hazard_code:
        params["hazard_code"] = hazard_code
    if metric:
        params["metric"] = metric
    if target_month:
        params["target_month"] = target_month
    if status:
        params["status"] = status
    if run_id and run_col:
        params["run_id"] = run_id

    if not latest_only:
        where_bits = []
        if iso3:
            where_bits.append("iso3 = :iso3")
        if hazard_code:
            where_bits.append("hazard_code = :hazard_code")
        if metric:
            where_bits.append("UPPER(metric) = UPPER(:metric)")
        if target_month:
            where_bits.append("target_month = :target_month")
        if status:
            where_bits.append("status = :status")
        if run_id and run_col:
            where_bits.append(f"{run_col} = :run_id")

        sql = "SELECT * FROM questions"
        if where_bits:
            sql += " WHERE " + " AND ".join(where_bits)
            sql += _test_filter(include_test)
        else:
            sql += " WHERE 1=1" + _test_filter(include_test)
        sql += " ORDER BY target_month, iso3, hazard_code, metric"
        if run_col:
            sql += f", {run_col}"
        rows = _rows_from_cursor(_execute(con, sql, params))
    else:
        # latest_only=True: one row per concept (iso3, hazard, metric, target_month) from latest run
        cte, _ = _latest_questions_view(
            con,
            iso3=iso3,
            hazard_code=hazard_code,
            metric=metric,
            target_month=target_month,
            status=status,
        )
        sql = cte + """
        SELECT *
        FROM latest_q
        """
        if run_id and run_col:
            sql += f" WHERE {run_col} = :run_id"
        else:
            sql += " WHERE rn = 1"
        if run_id and run_col:
            sql += " AND rn = 1"
        sql += " ORDER BY target_month, iso3, hazard_code, metric"
        rows = _rows_from_cursor(_execute(con, sql, params))

    # Enrich with forecast and triage summaries.
    triage_summary: dict[str, dict[str, Any]] = {}
    try:
        question_ids = [row["question_id"] for row in rows if row.get("question_id")]
        summary = compute_questions_forecast_summary(con, question_ids=question_ids)
        if not summary:
            logger.warning("Forecast summary empty; EIV will be null for questions page.")
        for row in rows:
            qid = row.get("question_id")
            forecast = summary.get(qid or "", {})
            row["forecast_date"] = forecast.get("forecast_date")
            row["forecast_horizon_max"] = forecast.get("horizon_max")
            row["eiv_total"] = forecast.get("eiv_total")
            row["eiv_peak"] = forecast.get("eiv_peak")
    except Exception:
        logger.exception("Failed to enrich questions with forecast summary")
    try:
        triage_summary = compute_questions_triage_summary(con, rows)
    except Exception:
        logger.exception("Failed to enrich questions with triage summary")
    for row in rows:
        qid = row.get("question_id")
        triage = triage_summary.get(qid or "", {})
        row["triage_score"] = triage.get("triage_score")
        row["triage_tier"] = triage.get("triage_tier")
        row["triage_need_full_spd"] = triage.get("triage_need_full_spd")
        row["triage_date"] = triage.get("triage_date")
        row["regime_change_likelihood"] = triage.get("regime_change_likelihood")
        row["regime_change_direction"] = triage.get("regime_change_direction")
        row["regime_change_magnitude"] = triage.get("regime_change_magnitude")
        row["regime_change_score"] = triage.get("regime_change_score")
        row["regime_change_level"] = triage.get("regime_change_level")
    return {"rows": rows}


@router.get("/v1/question_bundle")
def get_question_bundle(
    question_id: str = Query(..., description="Question identifier"),
    hs_run_id: Optional[str] = Query(None, description="Optional HS run override"),
    forecaster_run_id: Optional[str] = Query(None, description="Optional forecaster run override"),
    include_llm_calls: bool = Query(False, description="Include llm_calls rows"),
    include_transcripts: bool = Query(False, description="Include prompt/response text in llm_calls"),
    transcript_phases: Optional[str] = Query(None, description="Comma-separated phases to keep transcripts for (strips others to save memory)"),
    limit_llm_calls: int = Query(200, ge=1, le=2000, description="Max llm_calls rows to return"),
    include_test: bool = Query(False),
):
    if not _HEAVY_REQUEST_SEMAPHORE.acquire(timeout=30):
        raise HTTPException(status_code=503, detail="Server busy, try again")
    try:
        return _question_bundle_impl(
            question_id, hs_run_id, forecaster_run_id,
            include_llm_calls, include_transcripts, transcript_phases,
            limit_llm_calls, include_test,
        )
    finally:
        _HEAVY_REQUEST_SEMAPHORE.release()


def _question_bundle_impl(
    question_id, hs_run_id, forecaster_run_id,
    include_llm_calls, include_transcripts, transcript_phases,
    limit_llm_calls, include_test,
):
    con = _con()

    question_row = _resolve_question_row(con, question_id, hs_run_id)
    question = _apply_json_fields(question_row, ["scenario_ids_json", "pythia_metadata_json"])

    question_run_id = question.get("hs_run_id") or question.get("run_id")
    resolved_hs_run_id = question_run_id or hs_run_id
    iso3 = (question.get("iso3") or "").upper()
    hazard_code = (question.get("hazard_code") or "").upper()

    scenario_ids_raw = _safe_json_load(question.get("scenario_ids_json") or [])
    scenario_ids: List[str] = scenario_ids_raw if isinstance(scenario_ids_raw, list) else []

    hs_run = None
    if resolved_hs_run_id and _table_exists(con, "hs_runs"):
        hs_run = _fetch_one(
            con, "SELECT * FROM hs_runs WHERE hs_run_id = :hs_run_id", {"hs_run_id": resolved_hs_run_id}
        )

    triage = None
    if resolved_hs_run_id and iso3 and hazard_code:
        if _table_has_columns(con, "hs_triage", ["run_id", "iso3", "hazard_code"]):
            triage_row = _fetch_one(
                con,
                """
                SELECT *
                FROM hs_triage
                WHERE run_id = :hs_run_id AND iso3 = :iso3 AND hazard_code = :hazard_code
                ORDER BY created_at DESC
                LIMIT 1
                """,
                {"hs_run_id": resolved_hs_run_id, "iso3": iso3, "hazard_code": hazard_code},
            )
            if triage_row:
                triage = _apply_json_fields(triage_row, ["drivers_json", "regime_shifts_json", "data_quality_json", "regime_change_json"])

    country_report = None
    if resolved_hs_run_id and iso3:
        if _table_has_columns(con, "hs_country_reports", ["hs_run_id", "iso3"]):
            report_row = _fetch_one(
                con,
                """
                SELECT *
                FROM hs_country_reports
                WHERE hs_run_id = :hs_run_id AND iso3 = :iso3
                LIMIT 1
                """,
                {"hs_run_id": resolved_hs_run_id, "iso3": iso3},
            )
            if report_row:
                country_report = _apply_json_fields(report_row, ["sources_json"])

    scenarios: List[Dict[str, Any]] = []
    if resolved_hs_run_id and scenario_ids and _table_has_columns(con, "hs_scenarios", ["hs_run_id", "scenario_id"]):
        placeholders = ",".join(["?"] * len(scenario_ids))
        scenario_rows = [_apply_json_fields(r, ["scenario_json"]) for r in _rows_from_cursor(con.execute(
            f"""
            SELECT *
            FROM hs_scenarios
            WHERE hs_run_id = ? AND scenario_id IN ({placeholders})
            """,
            [resolved_hs_run_id, *scenario_ids],
        ))]
        order = {sid: idx for idx, sid in enumerate(scenario_ids)}
        scenarios = sorted(scenario_rows, key=lambda r: order.get(r.get("scenario_id"), len(order)))

    resolved_forecaster_run_id = _resolve_forecaster_run_id(con, question_id, forecaster_run_id)

    research = None
    ensemble_spd: List[Dict[str, Any]] = []
    raw_spd: List[Dict[str, Any]] = []
    scenario_writer: List[Dict[str, Any]] = []
    if resolved_forecaster_run_id:
        if _table_has_columns(con, "question_research", ["run_id", "question_id"]):
            # Exclude deprecated evidence JSON columns (large, unused by frontend).
            qr_cols = _table_columns(con, "question_research")
            qr_exclude = {"hs_evidence_json", "question_evidence_json", "merged_evidence_json"}
            qr_select = sorted(qr_cols - qr_exclude)
            research_row = _fetch_one(
                con,
                f"""
                SELECT {', '.join(qr_select)}
                FROM question_research
                WHERE run_id = :run_id AND question_id = :question_id
                ORDER BY created_at DESC
                LIMIT 1
                """,
                {"run_id": resolved_forecaster_run_id, "question_id": question_id},
            )
            if research_row:
                research = _apply_json_fields(research_row, ["research_json"])

        if _table_has_columns(con, "forecasts_ensemble", ["run_id", "question_id"]):
            ensemble_order_clause = "ORDER BY month_index, bucket_index, model_name"
            if _table_has_columns(con, "forecasts_ensemble", ["horizon_m", "class_bin"]):
                ensemble_order_clause = "ORDER BY horizon_m, class_bin, model_name"
            ensemble_spd = _rows_from_cursor(_execute(
                con,
                f"""
                SELECT *
                FROM forecasts_ensemble
                WHERE run_id = :run_id AND question_id = :question_id
                {ensemble_order_clause}
                """,
                {"run_id": resolved_forecaster_run_id, "question_id": question_id},
            ))

        raw_order_fields: List[str] = []
        if _table_has_columns(con, "forecasts_raw", ["horizon_m"]):
            raw_order_fields.append("horizon_m")
        elif _table_has_columns(con, "forecasts_raw", ["month_index"]):
            raw_order_fields.append("month_index")
        if _table_has_columns(con, "forecasts_raw", ["class_bin"]):
            raw_order_fields.append("class_bin")
        elif _table_has_columns(con, "forecasts_raw", ["bucket_index"]):
            raw_order_fields.append("bucket_index")
        if _table_has_columns(con, "forecasts_raw", ["model_name"]):
            raw_order_fields.append("model_name")
        raw_order_clause = ""
        if raw_order_fields:
            raw_order_clause = " ORDER BY " + ", ".join(raw_order_fields)

        if _table_has_columns(con, "forecasts_raw", ["run_id", "question_id"]):
            # Select only needed columns; exclude spd_json (large, unused by frontend).
            raw_cols = _table_columns(con, "forecasts_raw")
            raw_select = sorted(raw_cols - {"spd_json"})
            raw_spd = _rows_from_cursor(_execute(
                con,
                f"""
                SELECT {', '.join(raw_select)}
                FROM forecasts_raw
                WHERE run_id = :run_id AND question_id = :question_id
                {raw_order_clause}
                """,
                {"run_id": resolved_forecaster_run_id, "question_id": question_id},
            ))

        if _table_has_columns(con, "scenarios", ["run_id", "iso3", "hazard_code", "metric"]):
            scenario_writer = _rows_from_cursor(_execute(
                con,
                """
                SELECT *
                FROM scenarios
                WHERE run_id = :run_id AND iso3 = :iso3 AND hazard_code = :hazard_code AND metric = :metric
                ORDER BY scenario_type, bucket_label
                """,
                {
                    "run_id": resolved_forecaster_run_id,
                    "iso3": iso3,
                    "hazard_code": hazard_code,
                    "metric": question.get("metric"),
                },
            ))

    question_context = None
    if resolved_forecaster_run_id and _table_exists(con, "question_context"):
        question_context = _fetch_one(
            con,
            """
            SELECT *
            FROM question_context
            WHERE question_id = :question_id AND run_id = :run_id
            ORDER BY COALESCE(snapshot_end_month, snapshot_start_month) DESC NULLS LAST
            LIMIT 1
            """,
            {"question_id": question_id, "run_id": resolved_forecaster_run_id},
        )

    if not question_context and _table_exists(con, "question_context"):
        question_context = _fetch_one(
            con,
            """
            SELECT *
            FROM question_context
            WHERE question_id = :question_id
            ORDER BY COALESCE(snapshot_end_month, snapshot_start_month) DESC NULLS LAST
            LIMIT 1
            """,
            {"question_id": question_id},
        )

    if question_context:
        question_context = _apply_json_fields(question_context, ["pa_history_json", "context_json"])

    resolutions: List[Dict[str, Any]] = []
    if _table_exists(con, "resolutions"):
        resolutions = _rows_from_cursor(_execute(
            con,
            """
            SELECT *
            FROM resolutions
            WHERE question_id = :question_id
            ORDER BY observed_month
            """,
            {"question_id": question_id},
        ))

    scores: List[Dict[str, Any]] = []
    if _table_exists(con, "scores") and _table_has_columns(con, "scores", ["question_id", "score_type", "value"]):
        try:
            scores = _rows_from_cursor(_execute(
                con,
                """
                SELECT *
                FROM scores
                WHERE question_id = :question_id
                ORDER BY created_at DESC NULLS LAST, horizon_m ASC NULLS LAST, score_type ASC, model_name ASC NULLS LAST
                """,
                {"question_id": question_id},
            ))
        except Exception:
            logger.exception("Failed to load scores for question_bundle")

    llm_calls_bundle = _build_llm_calls_bundle(
        con,
        question_id=question_id,
        hs_run_id=resolved_hs_run_id,
        forecaster_run_id=resolved_forecaster_run_id,
        iso3=iso3,
        hazard_code=hazard_code,
        include_llm_calls=include_llm_calls,
        include_transcripts=include_transcripts,
        limit_llm_calls=limit_llm_calls,
        transcript_phases=[p.strip() for p in transcript_phases.split(",") if p.strip()] if transcript_phases else None,
    )

    metric = (question.get("metric") or "").upper()
    bucket_labels = _bucket_labels(con, metric)
    bucket_centroids = _bucket_centroids(con, metric, hazard_code, len(bucket_labels))

    response = QuestionBundleResponse(
        question=question,
        hs=HsBundle(
            hs_run=hs_run,
            triage=triage,
            scenario_ids=scenario_ids,
            scenarios=scenarios,
            country_report=country_report,
        ),
        forecast=ForecastBundle(
            forecaster_run_id=resolved_forecaster_run_id,
            research=research,
            ensemble_spd=ensemble_spd,
            raw_spd=raw_spd,
            scenario_writer=scenario_writer,
            bucket_labels=bucket_labels,
            bucket_centroids=bucket_centroids,
        ),
        context=ContextBundle(question_context=question_context, resolutions=resolutions, scores=scores),
        llm_calls=llm_calls_bundle,
    )
    return _json_sanitize(response.model_dump())
