# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import os

import duckdb, pandas as pd
import numpy as np
from fastapi import Body, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from pythia.api.auth import require_admin_token
from pythia.api.db_sync import (
    DbSyncError,
    get_cached_latest_hs,
    get_cached_manifest,
    maybe_sync_latest_db,
)
from pythia.api.models import (
    ContextBundle,
    ForecastBundle,
    HsBundle,
    LlmCallsBundle,
    QuestionBundleResponse,
)
from pythia.config import load as load_cfg
from pythia.pipeline.run import enqueue_run

app = FastAPI(title="Pythia API", version="1.0.0")
cors_origins_env = os.getenv("PYTHIA_CORS_ALLOW_ORIGINS", "*").strip()
cors_origins = (
    ["*"]
    if cors_origins_env in ("", "*")
    else [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
logger = logging.getLogger(__name__)


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: _json_sanitize(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(value) for value in obj]
    if isinstance(obj, tuple):
        return [_json_sanitize(value) for value in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (float, np.floating)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    return obj


def _rows_from_df(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    rows = df.to_dict(orient="records")
    return _json_sanitize(rows)


def _con():
    db_url = load_cfg()["app"]["db_url"].replace("duckdb:///", "")
    db_path = Path(db_url)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        maybe_sync_latest_db()
    except DbSyncError as exc:
        if not db_path.exists():
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        logger.warning("DB sync failed: %s", exc)
    if not db_path.exists():
        raise HTTPException(status_code=503, detail="DB not available yet")
    return duckdb.connect(db_url, read_only=True)


@app.on_event("startup")
def _startup_sync():
    try:
        maybe_sync_latest_db()
    except DbSyncError as exc:
        logger.warning("DB sync failed during startup: %s", exc)


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE LOWER(table_name) = LOWER(?)",
            [table],
        ).fetchone()
        return bool(row and row[0])
    except Exception:
        pass

    try:
        df = con.execute("PRAGMA show_tables").fetchdf()
    except Exception:
        return False
    if df.empty:
        return False
    first_col = df.columns[0]
    return df[first_col].astype(str).str.lower().eq(table.lower()).any()


def _table_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    try:
        df = con.execute(f"PRAGMA table_info('{table}')").fetchdf()
    except Exception:
        return set()
    if df.empty or "name" not in df.columns:
        return set()
    return set(df["name"].astype(str).str.lower().tolist())


def _pick_col(cols: set[str], candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate.lower() in cols:
            return candidate
    return None


def _table_has_columns(con: duckdb.DuckDBPyConnection, table: str, required: List[str]) -> bool:
    cols = _table_columns(con, table)
    return set(c.lower() for c in required).issubset(cols)


def _compile_named_params(sql: str, params: Dict[str, Any]) -> tuple[str, List[Any]]:
    pattern = re.compile(r"(?<!:):([A-Za-z_][A-Za-z0-9_]*)")
    args: List[Any] = []

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in params:
            raise KeyError(f"Missing SQL parameter: {name}")
        args.append(params[name])
        return "?"

    compiled = pattern.sub(_replace, sql)
    return compiled, args


def _execute(
    con: duckdb.DuckDBPyConnection, sql: str, params: Optional[Any] = None
) -> duckdb.DuckDBPyConnection:
    if not hasattr(con, "execute"):
        raise TypeError(f"_execute expected a DuckDB connection, got {type(con)}")
    if params is None:
        return con.execute(sql)
    if isinstance(params, dict):
        compiled_sql, args = _compile_named_params(sql, params)
        return con.execute(compiled_sql, args)
    return con.execute(sql, params)


def _safe_count_distinct(
    con: duckdb.DuckDBPyConnection, table: str, col_candidates: List[str]
) -> int:
    if not _table_exists(con, table):
        return 0
    cols = _table_columns(con, table)
    column = _pick_col(cols, col_candidates)
    if not column:
        return 0
    try:
        row = con.execute(f"SELECT COUNT(DISTINCT {column}) AS n FROM {table}").fetchone()
    except Exception:
        return 0
    return int(row[0]) if row else 0


def _safe_json_load(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def _apply_json_fields(row: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    out = dict(row)
    for field in fields:
        if field in out:
            out[field] = _safe_json_load(out[field])
    return out


def _fetch_one(con: duckdb.DuckDBPyConnection, sql: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    df = _execute(con, sql, params).fetchdf()
    rows = _rows_from_df(df)
    if not rows:
        return None
    return rows[0]


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
    if hs_run_id and q_run_col:
        sql += f" AND q.{q_run_col} = :hs_run_id"
        params["hs_run_id"] = hs_run_id
    order_by_parts = [f"{hs_timestamp_expr} DESC NULLS LAST"]
    if q_run_col:
        order_by_parts.append(f"q.{q_run_col} DESC")
    sql += f" ORDER BY {', '.join(order_by_parts)} LIMIT 1"

    df = _execute(con, sql, params).fetchdf()
    rows = _rows_from_df(df)
    if not rows:
        raise HTTPException(status_code=404, detail="Question not found")
    return rows[0]


def _resolve_forecaster_run_id(
    con: duckdb.DuckDBPyConnection, question_id: str, forecaster_run_id: Optional[str]
) -> Optional[str]:
    if forecaster_run_id:
        return forecaster_run_id
    if not _table_exists(con, "forecasts_ensemble"):
        return None
    df = con.execute(
        """
        SELECT run_id
        FROM forecasts_ensemble
        WHERE question_id = ?
        ORDER BY created_at DESC NULLS LAST
        LIMIT 1
        """,
        [question_id],
    ).fetchdf()
    if df.empty:
        return None
    return df["run_id"].iloc[0]


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
) -> LlmCallsBundle:
    if not include_llm_calls:
        return LlmCallsBundle(included=False, transcripts_included=False, rows=[], by_phase={})

    if not _table_exists(con, "llm_calls"):
        return LlmCallsBundle(
            included=True,
            transcripts_included=False,
            rows=[],
            by_phase={},
        )

    available_columns = _table_columns(con, "llm_calls")
    if not available_columns:
        return LlmCallsBundle(included=True, transcripts_included=False, rows=[], by_phase={})

    filters: List[str] = []
    params: Dict[str, Any] = {"limit": limit_llm_calls}
    transcripts_available = {"prompt_text", "response_text"}.issubset(available_columns)
    transcripts_included = include_transcripts and transcripts_available

    if forecaster_run_id:
        if not _table_has_columns(con, "llm_calls", ["run_id", "question_id", "phase"]):
            return LlmCallsBundle(
                included=True, transcripts_included=transcripts_included, rows=[], by_phase={}
            )
        filters.append(
            "(run_id = :forecaster_run_id AND question_id = :question_id AND phase IN ('research_v2','spd_v2','scenario_v2'))"
        )
        params["forecaster_run_id"] = forecaster_run_id
        params["question_id"] = question_id

    if hs_run_id and iso3 and hazard_code:
        if not _table_has_columns(con, "llm_calls", ["hs_run_id", "iso3", "hazard_code", "phase"]):
            return LlmCallsBundle(
                included=True, transcripts_included=transcripts_included, rows=[], by_phase={}
            )
        filters.append(
            "(hs_run_id = :hs_run_id AND iso3 = :iso3 AND hazard_code = :hazard_code AND phase = 'hs_triage')"
        )
        params["hs_run_id"] = hs_run_id
        params["iso3"] = iso3
        params["hazard_code"] = hazard_code

    if not filters:
        return LlmCallsBundle(
            included=True, transcripts_included=transcripts_included, rows=[], by_phase={}
        )

    select_columns = sorted(available_columns)
    order_fields: List[str] = []
    if "timestamp" in available_columns:
        order_fields.append("timestamp DESC NULLS LAST")
    elif "created_at" in available_columns:
        order_fields.append("created_at DESC NULLS LAST")
    if "call_id" in available_columns:
        order_fields.append("call_id DESC")

    sql = f"""
      SELECT {', '.join(select_columns)}
      FROM llm_calls
      WHERE {' OR '.join(filters)}
    """
    if order_fields:
        sql += " ORDER BY " + ", ".join(order_fields)
    sql += " LIMIT :limit"
    df = _execute(con, sql, params).fetchdf()

    cleaned_rows: List[Dict[str, Any]] = []
    by_phase: Dict[str, List[Dict[str, Any]]] = {}
    for row in _rows_from_df(df):
        parsed = _apply_json_fields(row, ["parsed_json", "usage_json"])
        if not transcripts_included:
            parsed.pop("prompt_text", None)
            parsed.pop("response_text", None)
        cleaned_rows.append(parsed)
        phase = parsed.get("phase")
        if phase:
            by_phase.setdefault(phase, []).append(parsed)

    return LlmCallsBundle(
        included=True,
        transcripts_included=transcripts_included,
        rows=cleaned_rows,
        by_phase=by_phase,
    )


def _resolve_latest_questions_columns(
    con: duckdb.DuckDBPyConnection,
) -> tuple[Optional[str], Optional[str], str]:
    q_cols = _table_columns(con, "questions")
    q_run_col = _pick_col(q_cols, ["hs_run_id", "run_id"])

    hs_runs_exists = _table_exists(con, "hs_runs")
    h_cols = _table_columns(con, "hs_runs") if hs_runs_exists else set()
    h_run_col = _pick_col(h_cols, ["hs_run_id", "run_id"]) if hs_runs_exists else None

    if hs_runs_exists and h_run_col:
        if h_run_col == "hs_run_id" and "hs_run_id" in q_cols:
            q_run_col = "hs_run_id"
        elif h_run_col == "run_id" and "run_id" in q_cols:
            q_run_col = "run_id"

    hs_created_col = "created_at" if "created_at" in h_cols else None
    hs_generated_col = "generated_at" if "generated_at" in h_cols else None
    hs_timestamp_expr = "NULL"
    if hs_runs_exists and h_run_col and q_run_col:
        if hs_created_col and hs_generated_col:
            hs_timestamp_expr = "COALESCE(h.created_at, h.generated_at)"
        elif hs_created_col:
            hs_timestamp_expr = "h.created_at"
        elif hs_generated_col:
            hs_timestamp_expr = "h.generated_at"

    return q_run_col, h_run_col, hs_timestamp_expr


def _resolve_forecasts_ensemble_columns(
    con: duckdb.DuckDBPyConnection,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Returns (horizon_col, bucket_col, prob_col)
    - horizon_col: 'horizon_m' or 'month_index'
    - bucket_col:  'class_bin' or 'bucket_index'
    - prob_col:    'probability' (required)
    """
    if not _table_exists(con, "forecasts_ensemble"):
        return None, None, None
    cols = _table_columns(con, "forecasts_ensemble")
    horizon_col = _pick_col(cols, ["horizon_m", "month_index"])
    bucket_col = _pick_col(cols, ["class_bin", "bucket_index"])
    prob_col = _pick_col(cols, ["probability"])
    return horizon_col, bucket_col, prob_col


def _latest_forecasted_target_month(
    con: duckdb.DuckDBPyConnection, metric_upper: str, horizon_col: str, horizon_m: int
) -> Optional[str]:
    if not _table_exists(con, "questions") or not _table_exists(con, "forecasts_ensemble"):
        return None
    row = _execute(
        con,
        f"""
        SELECT MAX(q.target_month) AS target_month
        FROM forecasts_ensemble fe
        JOIN questions q ON q.question_id = fe.question_id
        WHERE UPPER(q.metric) = :metric
          AND fe.{horizon_col} = :horizon_m
        """,
        {"metric": metric_upper, "horizon_m": horizon_m},
    ).fetchone()
    return row[0] if row and row[0] else None


def _latest_available_horizon(
    con: duckdb.DuckDBPyConnection, metric_upper: str, horizon_col: str, target_month: str
) -> Optional[int]:
    if not _table_exists(con, "questions") or not _table_exists(con, "forecasts_ensemble"):
        return None
    row = _execute(
        con,
        f"""
        SELECT MAX(fe.{horizon_col}) AS h
        FROM forecasts_ensemble fe
        JOIN questions q ON q.question_id = fe.question_id
        WHERE UPPER(q.metric) = :metric
          AND q.target_month = :target_month
        """,
        {"metric": metric_upper, "target_month": target_month},
    ).fetchone()
    if not row or row[0] is None:
        return None
    return int(row[0])


def _latest_questions_view(
    con: duckdb.DuckDBPyConnection,
    iso3: Optional[str] = None,
    hazard_code: Optional[str] = None,
    metric: Optional[str] = None,
    target_month: Optional[str] = None,
    status: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """
    Returns a SQL string for a 'latest questions' CTE called latest_q, parameterised
    by filters. The idea:

      - Identify question concepts: (iso3, hazard_code, metric, target_month)
      - For each concept, pick the question with the latest hs_runs.created_at
        (i.e. latest HS run).
      - Join questions q with hs_runs h to get run timestamps.

    NOTE: This helper builds only the CTE string; you still need to bind the same
    filter parameters to the main query.
    """
    q_run_col, h_run_col, hs_timestamp_expr = _resolve_latest_questions_columns(con)
    hs_runs_exists = _table_exists(con, "hs_runs")

    # We build filters into both the inner and outer query for simplicity
    where_bits = []
    if iso3:
        where_bits.append("q.iso3 = :iso3")
    if hazard_code:
        where_bits.append("q.hazard_code = :hazard_code")
    if metric:
        where_bits.append("UPPER(q.metric) = UPPER(:metric)")
    if target_month:
        where_bits.append("q.target_month = :target_month")
    if status:
        where_bits.append("q.status = :status")

    where_clause = ""
    if where_bits:
        where_clause = "WHERE " + " AND ".join(where_bits)

    join_clause = ""
    if hs_runs_exists and h_run_col and q_run_col:
        join_clause = f"LEFT JOIN hs_runs h ON q.{q_run_col} = h.{h_run_col}"

    order_bits = [f"{hs_timestamp_expr} DESC NULLS LAST"]
    if q_run_col:
        order_bits.append(f"q.{q_run_col} DESC")

    cte = f"""
    WITH latest_q AS (
      SELECT
        q.*,
        {hs_timestamp_expr} AS hs_run_created_at,
        ROW_NUMBER() OVER (
          PARTITION BY q.iso3, q.hazard_code, q.metric, q.target_month
          ORDER BY {", ".join(order_bits)}
        ) AS rn
      FROM questions q
      {join_clause}
      {where_clause}
    )
    """
    return cte, q_run_col


@app.get("/v1/health")
def health():
    return {"ok": True}


@app.get("/v1/version")
def api_version() -> Dict[str, Any]:
    try:
        manifest = maybe_sync_latest_db()
    except DbSyncError as exc:
        manifest = get_cached_manifest()
        if not manifest:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not manifest:
        raise HTTPException(status_code=503, detail="Manifest not available yet")
    return manifest


@app.post("/v1/run")
def start_run(payload: dict = Body(...), _=Depends(require_admin_token)):
    countries = payload.get("countries") or []
    run_id = enqueue_run(countries)
    return {"accepted": True, "run_id": run_id}


@app.get("/v1/ui_runs/{ui_run_id}")
def get_ui_run(ui_run_id: str):
    """
    Return status for a given ui_run_id created by /v1/run.

    Response shape:
      - found: bool
      - row: dict | None (full ui_runs row if found)
    """
    con = _con()
    df = con.execute(
        "SELECT * FROM ui_runs WHERE ui_run_id = ?",
        [ui_run_id],
    ).fetchdf()
    if df.empty:
        return {"found": False, "row": None}
    rows = _rows_from_df(df)
    row = rows[0] if rows else None
    return {"found": True, "row": row}


@app.get("/v1/questions")
def get_questions(
    iso3: Optional[str] = Query(None),
    hazard_code: Optional[str] = Query(None),
    metric: Optional[str] = Query(None),
    target_month: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    run_id: Optional[str] = Query(None),
    latest_only: bool = Query(False),
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
        sql += " ORDER BY target_month, iso3, hazard_code, metric"
        if run_col:
            sql += f", {run_col}"
        df = _execute(con, sql, params).fetchdf()
        return {"rows": _rows_from_df(df)}

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
    df = _execute(con, sql, params).fetchdf()
    return {"rows": _rows_from_df(df)}


@app.get("/v1/question_bundle")
def get_question_bundle(
    question_id: str = Query(..., description="Question identifier"),
    hs_run_id: Optional[str] = Query(None, description="Optional HS run override"),
    forecaster_run_id: Optional[str] = Query(None, description="Optional forecaster run override"),
    include_llm_calls: bool = Query(False, description="Include llm_calls rows"),
    include_transcripts: bool = Query(False, description="Include prompt/response text in llm_calls"),
    limit_llm_calls: int = Query(200, ge=1, le=2000, description="Max llm_calls rows to return"),
):
    con = _con()

    question_row = _resolve_question_row(con, question_id, hs_run_id)
    question = _apply_json_fields(question_row, ["scenario_ids_json", "pythia_metadata_json"])

    resolved_hs_run_id = hs_run_id or question.get("hs_run_id")
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
                triage = _apply_json_fields(triage_row, ["drivers_json", "regime_shifts_json", "data_quality_json"])

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
        df = con.execute(
            f"""
            SELECT *
            FROM hs_scenarios
            WHERE hs_run_id = ? AND scenario_id IN ({placeholders})
            """,
            [resolved_hs_run_id, *scenario_ids],
        ).fetchdf()
        scenario_rows = [_apply_json_fields(r, ["scenario_json"]) for r in _rows_from_df(df)]
        order = {sid: idx for idx, sid in enumerate(scenario_ids)}
        scenarios = sorted(scenario_rows, key=lambda r: order.get(r.get("scenario_id"), len(order)))

    resolved_forecaster_run_id = _resolve_forecaster_run_id(con, question_id, forecaster_run_id)

    research = None
    ensemble_spd: List[Dict[str, Any]] = []
    raw_spd: List[Dict[str, Any]] = []
    scenario_writer: List[Dict[str, Any]] = []
    if resolved_forecaster_run_id:
        if _table_has_columns(con, "question_research", ["run_id", "question_id"]):
            research_row = _fetch_one(
                con,
                """
                SELECT *
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
            ensemble_df = _execute(
                con,
                f"""
                SELECT *
                FROM forecasts_ensemble
                WHERE run_id = :run_id AND question_id = :question_id
                {ensemble_order_clause}
                """,
                {"run_id": resolved_forecaster_run_id, "question_id": question_id},
            ).fetchdf()
            ensemble_spd = _rows_from_df(ensemble_df)

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
            raw_df = _execute(
                con,
                f"""
                SELECT *
                FROM forecasts_raw
                WHERE run_id = :run_id AND question_id = :question_id
                {raw_order_clause}
                """,
                {"run_id": resolved_forecaster_run_id, "question_id": question_id},
            ).fetchdf()
            raw_spd = [_apply_json_fields(r, ["spd_json"]) for r in _rows_from_df(raw_df)]

        if _table_has_columns(con, "scenarios", ["run_id", "iso3", "hazard_code", "metric"]):
            scenario_df = _execute(
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
            ).fetchdf()
            scenario_writer = _rows_from_df(scenario_df)

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
        res_df = _execute(
            con,
            """
            SELECT *
            FROM resolutions
            WHERE question_id = :question_id
            ORDER BY observed_month
            """,
            {"question_id": question_id},
        ).fetchdf()
        resolutions = _rows_from_df(res_df)

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
    )

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
        ),
        context=ContextBundle(question_context=question_context, resolutions=resolutions),
        llm_calls=llm_calls_bundle,
    )
    return _json_sanitize(response.model_dump())


@app.get("/v1/calibration/weights")
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

    df = _execute(con, sql, params).fetchdf()

    if df.empty:
        return {"found": False, "as_of_month": as_of_month, "rows": []}

    # We return rows, plus the resolved as_of_month for convenience
    return {
        "found": True,
        "as_of_month": as_of_month,
        "rows": _rows_from_df(df),
    }


@app.get("/v1/calibration/advice")
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

    df = _execute(con, sql, params).fetchdf()

    if df.empty:
        return {"found": False, "as_of_month": as_of_month, "rows": []}

    return {
        "found": True,
        "as_of_month": as_of_month,
        "rows": _rows_from_df(df),
    }


@app.get("/v1/forecasts/ensemble")
def get_forecasts_ensemble(
    iso3: Optional[str] = Query(None),
    hazard_code: Optional[str] = Query(None),
    metric: Optional[str] = Query(None),
    target_month: Optional[str] = Query(None),
    horizon_m: Optional[int] = Query(None),
    latest_only: bool = Query(True),
):
    con = _con()
    params = {}
    if iso3:
        params["iso3"] = iso3
    if hazard_code:
        params["hazard_code"] = hazard_code
    if metric:
        params["metric"] = metric
    if target_month:
        params["target_month"] = target_month
    if horizon_m is not None:
        params["horizon_m"] = horizon_m

    if latest_only:
        cte, _ = _latest_questions_view(
            con,
            iso3=iso3,
            hazard_code=hazard_code,
            metric=metric,
            target_month=target_month,
            status=None,
        )
        sql = cte + """
        SELECT
          fe.question_id,
          q.iso3,
          q.hazard_code,
          q.metric,
          q.target_month,
          fe.horizon_m,
          fe.class_bin,
          fe.p,
          fe.aggregator,
          fe.ensemble_version
        FROM forecasts_ensemble fe
        JOIN latest_q q ON fe.question_id = q.question_id
        """
        where_bits = []
        where_bits.append("q.rn = 1")
        if horizon_m is not None:
            where_bits.append("fe.horizon_m = :horizon_m")
        if where_bits:
            sql += " WHERE " + " AND ".join(where_bits)
        sql += " ORDER BY q.iso3, q.hazard_code, q.metric, q.target_month, fe.horizon_m, fe.class_bin"
        df = _execute(con, sql, params).fetchdf()
        return {"rows": _rows_from_df(df)}

    # latest_only=False: historical view (all runs)
    sql = """
      SELECT
        fe.question_id,
        q.iso3,
        q.hazard_code,
        q.metric,
        q.target_month,
        q.run_id,
        fe.horizon_m,
        fe.class_bin,
        fe.p,
        fe.aggregator,
        fe.ensemble_version
      FROM forecasts_ensemble fe
      JOIN questions q ON fe.question_id = q.question_id
      WHERE 1=1
    """
    if iso3:
        sql += " AND q.iso3 = :iso3"
    if hazard_code:
        sql += " AND q.hazard_code = :hazard_code"
    if metric:
        sql += " AND UPPER(q.metric) = UPPER(:metric)"
    if target_month:
        sql += " AND q.target_month = :target_month"
    if horizon_m is not None:
        sql += " AND fe.horizon_m = :horizon_m"

    sql += " ORDER BY q.target_month, q.iso3, q.hazard_code, q.metric, q.run_id, fe.horizon_m, fe.class_bin"
    df = _execute(con, sql, params).fetchdf()
    return {"rows": _rows_from_df(df)}


@app.get("/v1/forecasts/history")
def get_forecasts_history(
    iso3: str = Query(...),
    hazard_code: str = Query(...),
    metric: str = Query(...),
    target_month: str = Query(...),
):
    """
    Return all historical ensemble forecasts for a given question concept
    (iso3, hazard_code, metric, target_month), grouped by HS run.

    Each row includes:
      - run_id
      - hs_run_created_at
      - horizon_m
      - class_bin
      - p
    """
    con = _con()
    params = {
        "iso3": iso3,
        "hazard_code": hazard_code,
        "metric": metric,
        "target_month": target_month,
    }

    sql = """
      SELECT
        q.run_id,
        h.created_at AS hs_run_created_at,
        fe.question_id,
        fe.horizon_m,
        fe.class_bin,
        fe.p
      FROM forecasts_ensemble fe
      JOIN questions q ON fe.question_id = q.question_id
      JOIN hs_runs h ON q.run_id = h.run_id
      WHERE q.iso3 = :iso3
        AND q.hazard_code = :hazard_code
        AND UPPER(q.metric) = UPPER(:metric)
        AND q.target_month = :target_month
      ORDER BY h.created_at, fe.horizon_m, fe.class_bin
    """
    df = _execute(con, sql, params).fetchdf()
    return {"rows": _rows_from_df(df)}


@app.get("/v1/resolutions")
def list_resolutions(iso3: str, month: str, metric: str = "PIN"):
    con = _con()
    if not _table_exists(con, "resolutions"):
        return {"rows": []}
    qsql = "SELECT question_id FROM questions WHERE iso3=? AND target_month=? AND metric=?"
    qids = [r[0] for r in con.execute(qsql, [iso3.upper(), month, metric]).fetchall()]
    if not qids:
        return {"rows": []}
    inlist = ",".join(["?"] * len(qids))
    df = con.execute(
        f"SELECT * FROM resolutions WHERE question_id IN ({inlist})",
        qids,
    ).fetchdf()
    return {"rows": _rows_from_df(df)}


@app.get("/v1/risk_index")
def get_risk_index(
    metric: str = Query("PA", description="Metric to rank on, e.g. 'PA'"),
    target_month: Optional[str] = Query(None, description="Target month 'YYYY-MM'"),
    horizon_m: int = Query(1, ge=1, le=6, description="Forecast horizon in months ahead"),
    normalize: bool = Query(True, description="If true, include per-capita ranking"),
):
    """
    Country-level risk index for a given metric/target_month/horizon.

    For each country (iso3), this sums expected value (centroid-based) across all
    questions with the given metric and target_month at the specified horizon_m.
    It then optionally normalises by population.

    Returns:
      - iso3
      - expected_value (EV of metric, summed over hazards)
      - per_capita (EV / population) if normalize=true
    """
    con = _con()
    metric_upper = metric.upper()
    horizon_col, bucket_col, prob_col = _resolve_forecasts_ensemble_columns(con)

    if (
        not _table_exists(con, "questions")
        or horizon_col is None
        or bucket_col is None
        or prob_col is None
    ):
        return {
            "metric": metric_upper,
            "target_month": target_month or None,
            "horizon_m": horizon_m,
            "normalize": normalize,
            "rows": [],
        }

    selected_month = target_month
    if not selected_month:
        selected_month = _latest_forecasted_target_month(
            con, metric_upper, horizon_col, horizon_m
        )
        if selected_month is None:
            row = _execute(
                con,
                """
                SELECT MAX(q.target_month) AS target_month
                FROM forecasts_ensemble fe
                JOIN questions q ON q.question_id = fe.question_id
                WHERE UPPER(q.metric) = :metric
                """,
                {"metric": metric_upper},
            ).fetchone()
            selected_month = row[0] if row and row[0] else None

    if selected_month is None:
        return {
            "metric": metric_upper,
            "target_month": None,
            "horizon_m": horizon_m,
            "normalize": normalize,
            "rows": [],
        }

    def _run_query(month: str, horizon_value: int) -> pd.DataFrame:
        sql = f"""
        WITH q AS (
          SELECT question_id, iso3
          FROM questions
          WHERE UPPER(metric) = :metric
            AND target_month = :target_month
        ),
        ev_per_question AS (
          SELECT
            q.iso3,
            fe.question_id,
            SUM(CAST(fe.{bucket_col} AS DOUBLE) * fe.{prob_col}) AS expected_value
          FROM forecasts_ensemble fe
          JOIN q ON q.question_id = fe.question_id
          WHERE fe.{horizon_col} = :horizon_m
          GROUP BY q.iso3, fe.question_id
        )
        SELECT
          iso3,
          SUM(expected_value) AS expected_value
        FROM ev_per_question
        GROUP BY iso3
        ORDER BY expected_value DESC NULLS LAST
        """
        return _execute(
            con,
            sql,
            {"metric": metric_upper, "target_month": month, "horizon_m": horizon_value},
        ).fetchdf()

    selected_horizon = horizon_m
    df = _run_query(selected_month, selected_horizon)
    if df.empty:
        fallback_horizon = _latest_available_horizon(
            con, metric_upper, horizon_col, selected_month
        )
        if fallback_horizon is not None and fallback_horizon != selected_horizon:
            selected_horizon = fallback_horizon
            df = _run_query(selected_month, selected_horizon)

    if df.empty and target_month:
        fallback_month = _latest_forecasted_target_month(
            con, metric_upper, horizon_col, selected_horizon
        )
        if fallback_month and fallback_month != selected_month:
            selected_month = fallback_month
            df = _run_query(selected_month, selected_horizon)

    return {
        "metric": metric_upper,
        "target_month": selected_month,
        "horizon_m": selected_horizon,
        "normalize": normalize,
        "rows": _rows_from_df(df),
    }


@app.get("/v1/rankings")
def rankings(month: str, metric: str = "PIN", normalize: bool = True):
    con = _con()
    sql = """
    WITH ev AS (
      SELECT q.iso3, fe.horizon_m,
             SUM(
               fe.p * COALESCE(
                 bc.ev,
                 CASE fe.class_bin
                   WHEN '<10k' THEN 5000
                   WHEN '10k-<50k' THEN 25000
                   WHEN '50k-<250k' THEN 120000
                   WHEN '250k-<500k' THEN 350000
                   WHEN '>=500k' THEN 700000
                 END
               )
             ) AS ev_pin
      FROM forecasts_ensemble fe
      JOIN questions q ON q.question_id=fe.question_id
      LEFT JOIN bucket_centroids bc
        ON bc.metric = q.metric
       AND bc.class_bin = fe.class_bin
      AND bc.hazard_code = q.hazard_code
      WHERE q.metric=? AND q.target_month=?
      GROUP BY 1,2
    ), pop AS (
      SELECT iso3, MAX_BY(population, year) AS population
      FROM populations GROUP BY 1
    )
    SELECT ev.iso3, ev.horizon_m,
           ev.ev_pin AS expected_value,
           CASE WHEN ? THEN ev.ev_pin/NULLIF(pop.population,0) ELSE NULL END AS per_capita
    FROM ev LEFT JOIN pop ON ev.iso3=pop.iso3
      ORDER BY (CASE WHEN ? THEN per_capita ELSE expected_value END) DESC
    """
    df = con.execute(sql, [metric, month, normalize, normalize]).fetchdf()
    return {"rows": _rows_from_df(df)}


@app.get("/v1/diagnostics/summary")
def diagnostics_summary():
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

    try:
        q_counts_df = con.execute(
            "SELECT status, COUNT(*) AS n FROM questions GROUP BY status"
        ).fetchdf()
        q_counts = _rows_from_df(q_counts_df)
    except Exception:
        q_counts = []

    q_with_forecast = _safe_count_distinct(con, "forecasts_ensemble", ["question_id"])
    q_with_resolutions = _safe_count_distinct(con, "resolutions", ["question_id"])
    q_with_scores = _safe_count_distinct(con, "scores", ["question_id"])

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


@app.get("/v1/countries")
def get_countries():
    con = _con()
    if not _table_exists(con, "questions"):
        return {"rows": []}

    if _table_exists(con, "forecasts_ensemble"):
        sql = """
          SELECT
            q.iso3,
            COUNT(DISTINCT q.question_id) AS n_questions,
            COUNT(DISTINCT fe.question_id) AS n_forecasted
          FROM questions q
          LEFT JOIN forecasts_ensemble fe ON fe.question_id = q.question_id
          GROUP BY q.iso3
          ORDER BY q.iso3
        """
    else:
        sql = """
          SELECT
            q.iso3,
            COUNT(DISTINCT q.question_id) AS n_questions,
            0 AS n_forecasted
          FROM questions q
          GROUP BY q.iso3
          ORDER BY q.iso3
        """

    df = con.execute(sql).fetchdf()
    if df.empty:
        return {"rows": []}
    return {"rows": _rows_from_df(df)}


@app.get("/v1/llm/costs")
def llm_costs(
    component: str | None = Query(None),
    model: str | None = Query(None),
    since: str | None = Query(None),
    limit: int = Query(200, ge=1, le=5000),
):
    """
    Return recent LLM call cost/usage rows from llm_calls.

    Optional filters:
      - component: "HS" | "Researcher" | "Forecaster" | etc.
      - model: model_name (exact match)
      - since: ISO timestamp (created_at >= since)
    """
    con = _con()
    sql = "SELECT * FROM llm_calls WHERE 1=1"
    params: list = []

    if component:
        sql += " AND component = ?"
        params.append(component)
    if model:
        sql += " AND model_name = ?"
        params.append(model)
    if since:
        sql += " AND created_at >= ?"
        params.append(since)

    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    df = _execute(con, sql, params).fetchdf()
    return {"rows": _rows_from_df(df)}


@app.get("/v1/llm/costs/summary")
def llm_costs_summary(
    component: Optional[str] = Query(
        None, description="Filter by component, e.g. 'HS', 'Researcher', 'Forecaster'"
    ),
    model: Optional[str] = Query(None, description="Filter by model_name"),
    llm_profile: Optional[str] = Query(None, description="Filter by llm_profile, e.g. 'test' or 'prod'"),
    hs_run_id: Optional[str] = Query(None, description="Filter by HS run id"),
    ui_run_id: Optional[str] = Query(None, description="Filter by UI run id"),
    forecaster_run_id: Optional[str] = Query(None, description="Filter by Forecaster run id"),
    since: Optional[str] = Query(
        None,
        description="Filter by created_at >= since (ISO date or datetime string). If omitted, no time filter.",
    ),
    group_by: str = Query(
        "component,model_name,llm_profile",
        description="Comma-separated list of grouping fields: any of 'component','model_name','llm_profile','hs_run_id','ui_run_id','forecaster_run_id'",
    ),
    limit: int = Query(1000, ge=1, le=5000),
):
    """
    Summarise LLM usage and cost from llm_calls.

    Example:
      - Group by component,model_name,llm_profile since a given date.
      - Inspect total cost_usd and tokens per model or per run.

    Returns aggregated metrics per group:
      - calls
      - tokens_in
      - tokens_out
      - cost_usd
    """
    con = _con()

    # Parse and validate group_by
    allowed_fields = {
        "component",
        "model_name",
        "llm_profile",
        "hs_run_id",
        "ui_run_id",
        "forecaster_run_id",
    }
    group_fields: List[str] = [f.strip() for f in group_by.split(",") if f.strip()]
    if not group_fields:
        group_fields = ["component", "model_name", "llm_profile"]

    invalid = [f for f in group_fields if f not in allowed_fields]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid group_by fields: {', '.join(invalid)}")

    # Build WHERE clause
    where_bits = ["1=1"]
    params: dict = {}

    if component:
        where_bits.append("component = :component")
        params["component"] = component
    if model:
        where_bits.append("model_name = :model_name")
        params["model_name"] = model
    if llm_profile:
        where_bits.append("llm_profile = :llm_profile")
        params["llm_profile"] = llm_profile
    if hs_run_id:
        where_bits.append("hs_run_id = :hs_run_id")
        params["hs_run_id"] = hs_run_id
    if ui_run_id:
        where_bits.append("ui_run_id = :ui_run_id")
        params["ui_run_id"] = ui_run_id
    if forecaster_run_id:
        where_bits.append("forecaster_run_id = :forecaster_run_id")
        params["forecaster_run_id"] = forecaster_run_id
    if since:
        where_bits.append("created_at >= :since")
        params["since"] = since

    where_clause = " AND ".join(where_bits)

    # Build SELECT and GROUP BY
    group_select = ", ".join(group_fields) if group_fields else ""
    group_by_clause = ""
    if group_fields:
        group_by_clause = "GROUP BY " + ", ".join(group_fields)

    select_fields = group_select + (", " if group_select else "")
    select_fields += """
        COUNT(*) AS calls,
        SUM(tokens_in) AS tokens_in,
        SUM(tokens_out) AS tokens_out,
        SUM(cost_usd) AS cost_usd
    """

    sql = f"""
      SELECT {select_fields}
      FROM llm_calls
      WHERE {where_clause}
      {group_by_clause}
      ORDER BY cost_usd DESC NULLS LAST
      LIMIT :limit
    """
    params["limit"] = limit

    df = _execute(con, sql, params).fetchdf()
    return {
        "group_by": group_fields,
        "filters": {
            "component": component,
            "model": model,
            "llm_profile": llm_profile,
            "hs_run_id": hs_run_id,
            "ui_run_id": ui_run_id,
            "forecaster_run_id": forecaster_run_id,
            "since": since,
        },
        "rows": _rows_from_df(df),
    }
