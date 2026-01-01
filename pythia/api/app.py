# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
import logging
import math
import re
from datetime import datetime
from importlib.util import find_spec
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import os

import duckdb, pandas as pd
import numpy as np
from fastapi import Body, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse

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
from resolver.query.countries_index import compute_countries_index
from resolver.query.costs import (
    COST_COLUMNS,
    build_costs_monthly,
    build_costs_runs,
    build_costs_total,
    build_latencies_runs,
    build_run_runtimes,
)
from resolver.query.downloads import build_forecast_spd_export, build_triage_export
from resolver.query.kpi_scopes import compute_countries_triaged_for_month_with_source
from resolver.query.questions_index import (
    compute_questions_forecast_summary,
    compute_questions_triage_summary,
)
from pythia.buckets import BUCKET_SPECS
from resolver.query import eiv_sql

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

_COUNTRY_NAME_BY_ISO3: dict[str, str] = {}
_POPULATION_BY_ISO3: dict[str, int] = {}


def _load_country_registry() -> None:
    global _COUNTRY_NAME_BY_ISO3
    if _COUNTRY_NAME_BY_ISO3:
        return
    try:
        repo_root = Path(__file__).resolve().parents[2]
        path = repo_root / "resolver" / "data" / "countries.csv"
        if not path.exists():
            return
        df = pd.read_csv(path, dtype=str).fillna("")
        if "iso3" not in df.columns or "country_name" not in df.columns:
            return
        _COUNTRY_NAME_BY_ISO3 = {
            str(iso3).strip().upper(): str(name).strip()
            for iso3, name in zip(df["iso3"], df["country_name"])
            if str(iso3).strip()
        }
    except Exception:
        return


def _country_name(iso3: str) -> str:
    _load_country_registry()
    return _COUNTRY_NAME_BY_ISO3.get((iso3 or "").upper(), "")


def _load_population_registry() -> None:
    global _POPULATION_BY_ISO3
    if _POPULATION_BY_ISO3:
        return
    try:
        env_path = os.getenv("PYTHIA_POPULATION_CSV_PATH")
        if env_path:
            path = Path(env_path)
        else:
            repo_root = Path(__file__).resolve().parents[2]
            path = repo_root / "resolver" / "data" / "population.csv"
        if not path.exists():
            return
        try:
            df = pd.read_csv(path, dtype=str, sep=None, engine="python").fillna("")
        except Exception:
            df = pd.read_csv(path, dtype=str, sep="\t").fillna("")
        df.columns = [str(col).strip().lower() for col in df.columns]
        if "iso3" not in df.columns or "population" not in df.columns:
            logger.warning(
                "Population registry present but could not be parsed; per-capita disabled"
            )
            return
        population_map: dict[str, int] = {}
        for iso3, population in zip(df["iso3"], df["population"]):
            iso = str(iso3).strip().upper()
            if not iso:
                continue
            pop_raw = str(population).strip().replace(",", "").replace(" ", "")
            if not pop_raw:
                continue
            try:
                pop_value = int(float(pop_raw))
            except Exception:
                continue
            if pop_value <= 0:
                continue
            population_map[iso] = pop_value
        _POPULATION_BY_ISO3 = population_map
        if population_map:
            logger.info("Loaded %d population rows from %s", len(population_map), path)
        else:
            logger.warning(
                "Population registry present but could not be parsed; per-capita disabled"
            )
    except Exception:
        logger.warning("Population registry present but could not be parsed; per-capita disabled")
        return


def _population(iso3: str) -> Optional[int]:
    _load_population_registry()
    return _POPULATION_BY_ISO3.get((iso3 or "").upper())


class _HealthAccessFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return "/v1/health" not in msg


logging.getLogger("uvicorn.access").addFilter(_HealthAccessFilter())


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


def _concat_cost_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [
        tables.get("summary"),
        tables.get("by_model"),
        tables.get("by_phase"),
    ]
    frames = [frame for frame in frames if frame is not None]
    if not frames or all(frame.empty for frame in frames):
        return pd.DataFrame(columns=COST_COLUMNS)
    return pd.concat(frames, ignore_index=True)


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


def _nonnull_count(con: duckdb.DuckDBPyConnection, table: str, col: str) -> int:
    try:
        row = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NOT NULL").fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _pick_col(cols: set[str], candidates: List[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate.lower() in cols:
            return candidate
    return None


def _pick_timestamp_column(
    con: duckdb.DuckDBPyConnection, table: str, candidates: List[str]
) -> Optional[str]:
    if not _table_exists(con, table):
        return None
    cols = _table_columns(con, table)
    return _pick_col(cols, candidates)


def _month_window(year: int, month: int) -> tuple[str, str]:
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)
    return start.isoformat(), end.isoformat()


def _parse_year_month(value: Optional[str]) -> Optional[tuple[int, int]]:
    if not value:
        return None
    match = re.match(r"^(\d{4})-(\d{2})$", value.strip())
    if not match:
        return None
    year = int(match.group(1))
    month = int(match.group(2))
    if month < 1 or month > 12:
        return None
    return year, month


def _format_year_month_label(year: int, month: int) -> str:
    return datetime(year, month, 1).strftime("%B %Y")


def _table_has_columns(con: duckdb.DuckDBPyConnection, table: str, required: List[str]) -> bool:
    cols = _table_columns(con, table)
    return set(c.lower() for c in required).issubset(cols)


def _bucket_labels(con: duckdb.DuckDBPyConnection, metric: str) -> List[str]:
    metric_upper = (metric or "").upper()
    labels: List[str] = []
    if _table_has_columns(con, "bucket_definitions", ["metric", "bucket_index", "label"]):
        rows = con.execute(
            """
            SELECT bucket_index, label
            FROM bucket_definitions
            WHERE UPPER(metric) = ?
            ORDER BY bucket_index
            """,
            [metric_upper],
        ).fetchall()
        labels = [str(label) for _, label in rows if label is not None]
    if labels:
        return labels
    specs = BUCKET_SPECS.get(metric_upper)
    if not specs:
        return []
    return [spec.label for spec in specs]


def _bucket_centroids(
    con: duckdb.DuckDBPyConnection,
    metric: str,
    hazard_code: str,
    bucket_count: int,
) -> List[float]:
    metric_upper = (metric or "").upper()
    hazard_upper = (hazard_code or "").upper()
    centroids: List[float] = []
    if _table_has_columns(con, "bucket_centroids", ["hazard_code", "metric", "bucket_index", "centroid"]):
        rows = con.execute(
            """
            SELECT bucket_index, centroid
            FROM bucket_centroids
            WHERE UPPER(metric) = ? AND UPPER(hazard_code) = ?
            ORDER BY bucket_index
            """,
            [metric_upper, hazard_upper],
        ).fetchall()
        centroids = [float(centroid) for _, centroid in rows if centroid is not None]
        if len(centroids) != bucket_count:
            rows = con.execute(
                """
                SELECT bucket_index, centroid
                FROM bucket_centroids
                WHERE UPPER(metric) = ? AND hazard_code = '*'
                ORDER BY bucket_index
                """,
                [metric_upper],
            ).fetchall()
            centroids = [float(centroid) for _, centroid in rows if centroid is not None]
    if centroids and len(centroids) == bucket_count:
        return centroids
    specs = BUCKET_SPECS.get(metric_upper)
    if not specs:
        return []
    return [float(spec.centroid) for spec in specs]


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
    order_by_parts = [f"{hs_timestamp_expr} DESC NULLS LAST"]
    if q_run_col:
        order_by_parts.append(f"q.{q_run_col} DESC")
    sql += f" ORDER BY {', '.join(order_by_parts)} LIMIT 1"

    def fetch_row(query: str, query_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        df = _execute(con, query, query_params).fetchdf()
        rows = _rows_from_df(df)
        if not rows:
            return None
        return rows[0]

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
            available_df = _execute(
                con,
                f"""
                SELECT DISTINCT q.{q_run_col} AS run_id
                FROM questions q
                WHERE q.question_id = :question_id
                ORDER BY q.{q_run_col} DESC
                LIMIT 10
                """,
                {"question_id": question_id},
            ).fetchdf()
            if not available_df.empty:
                available_run_ids = available_df["run_id"].dropna().tolist()
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
            "AND UPPER(hazard_code) = :hazard_code "
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
        debug=build_debug_payload(by_phase),
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
    Picks the *populated* column when both variants exist.
    """
    if not _table_exists(con, "forecasts_ensemble"):
        return None, None, None

    cols = _table_columns(con, "forecasts_ensemble")
    prob_col = _pick_col(cols, ["probability"])
    if not prob_col:
        return None, None, None

    horizon_candidates = [c for c in ["month_index", "horizon_m"] if c in cols]
    bucket_candidates = [c for c in ["bucket_index", "class_bin"] if c in cols]

    horizon_col = None
    if horizon_candidates:
        scored = [(c, _nonnull_count(con, "forecasts_ensemble", c)) for c in horizon_candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        horizon_col = scored[0][0] if scored[0][1] > 0 else horizon_candidates[0]

    bucket_col = None
    if bucket_candidates:
        scored = [(c, _nonnull_count(con, "forecasts_ensemble", c)) for c in bucket_candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        bucket_col = scored[0][0] if scored[0][1] > 0 else bucket_candidates[0]

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
    rows = _rows_from_df(df)
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
    return {"rows": rows}


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

    scores: List[Dict[str, Any]] = []
    if _table_exists(con, "scores") and _table_has_columns(con, "scores", ["question_id", "score_type", "value"]):
        try:
            scores_df = _execute(
                con,
                """
                SELECT *
                FROM scores
                WHERE question_id = :question_id
                ORDER BY created_at DESC NULLS LAST, horizon_m ASC NULLS LAST, score_type ASC, model_name ASC NULLS LAST
                """,
                {"question_id": question_id},
            ).fetchdf()
            scores = _rows_from_df(scores_df)
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
    hazard_code: Optional[str] = Query(None, description="Optional hazard code filter"),
    target_month: Optional[str] = Query(None, description="Target month 'YYYY-MM'"),
    horizon_m: int = Query(1, ge=1, le=6, description="Forecast horizon in months ahead"),
    normalize: bool = Query(True, description="If true, include per-capita ranking"),
    agg: str = Query("surge", description="Aggregation mode: surge (default) or burden (legacy)"),
    alpha: float = Query(0.25, ge=0, le=1, description="Surge blending weight"),
):
    con = _con()
    metric_upper = (metric or "").strip().upper() or "PA"
    hazard_code_upper = (hazard_code or "").strip().upper() or None

    if metric_upper != "PA":
        return {
            "metric": metric_upper,
            "target_month": target_month,
            "horizon_m": horizon_m,
            "normalize": normalize,
            "rows": [],
        }

    horizon_col, bucket_col, prob_col = _resolve_forecasts_ensemble_columns(con)
    if not horizon_col or not bucket_col or not prob_col:
        return {
            "metric": metric_upper,
            "target_month": target_month,
            "horizon_m": horizon_m,
            "normalize": normalize,
            "rows": [],
        }

    if not target_month:
        target_month = _latest_forecasted_target_month(
            con, metric_upper, horizon_col, horizon_m
        )
        if not target_month:
            row = _execute(
                con,
                "SELECT MAX(target_month) FROM questions WHERE UPPER(metric)=:metric",
                {"metric": metric_upper},
            ).fetchone()
            target_month = row[0] if row and row[0] else None

    if not target_month:
        return {
            "metric": metric_upper,
            "target_month": None,
            "horizon_m": 6,
            "normalize": normalize,
            "rows": [],
        }

    db_population_map: dict[str, int] = {}
    populations_available = False
    populations_table_available = _table_exists(con, "populations") and _table_has_columns(
        con, "populations", ["iso3", "population", "year"]
    )
    if populations_table_available:
        pop_df = _execute(
            con,
            """
            SELECT iso3, population
            FROM (
              SELECT iso3, population,
                     ROW_NUMBER() OVER (PARTITION BY iso3 ORDER BY year DESC) AS rn
              FROM populations
              WHERE population IS NOT NULL AND population > 0
            )
            WHERE rn = 1
            """,
        ).fetchdf()
        if not pop_df.empty:
            for iso3, population in zip(pop_df["iso3"], pop_df["population"]):
                iso = str(iso3).strip().upper()
                if not iso:
                    continue
                try:
                    pop_value = int(float(population))
                except Exception:
                    continue
                if pop_value <= 0:
                    continue
                db_population_map[iso] = pop_value
            populations_available = bool(db_population_map)
    agg_mode = "burden"
    registry_available = False
    if (normalize or agg_mode == "surge") and not populations_available:
        _load_population_registry()
        registry_available = bool(_POPULATION_BY_ISO3)
        if not registry_available:
            logger.debug("Population registry empty; per-capita values unavailable.")
    if agg is not None and str(agg).strip():
        agg_mode = str(agg).strip().lower()
    if agg_mode not in {"surge", "burden"}:
        raise HTTPException(status_code=400, detail="agg must be 'surge' or 'burden'")

    centroids_available = _table_exists(con, "bucket_centroids") and _table_has_columns(
        con, "bucket_centroids", ["metric", "hazard_code", "bucket_index", "centroid"]
    )

    pop_cte = ""
    pop_select = """
      , NULL AS population
      , NULL AS m1_pc
      , NULL AS m2_pc
      , NULL AS m3_pc
      , NULL AS m4_pc
      , NULL AS m5_pc
      , NULL AS m6_pc
      , NULL AS total_pc
    """
    pop_join = ""
    pop_join_monthly = ""
    if populations_available:
        pop_cte = """
        , pop AS (
          SELECT iso3, population
          FROM (
            SELECT iso3, population,
                   ROW_NUMBER() OVER (PARTITION BY iso3 ORDER BY year DESC) AS rn
            FROM populations
            WHERE population IS NOT NULL AND population > 0
          )
          WHERE rn = 1
        )
        """
        pop_select = """
          , pop.population
          , CASE
              WHEN :normalize AND pop.population IS NOT NULL AND pop.population != 0
                THEN p.m1 / pop.population
              ELSE NULL
            END AS m1_pc
          , CASE
              WHEN :normalize AND pop.population IS NOT NULL AND pop.population != 0
                THEN p.m2 / pop.population
              ELSE NULL
            END AS m2_pc
          , CASE
              WHEN :normalize AND pop.population IS NOT NULL AND pop.population != 0
                THEN p.m3 / pop.population
              ELSE NULL
            END AS m3_pc
          , CASE
              WHEN :normalize AND pop.population IS NOT NULL AND pop.population != 0
                THEN p.m4 / pop.population
              ELSE NULL
            END AS m4_pc
          , CASE
              WHEN :normalize AND pop.population IS NOT NULL AND pop.population != 0
                THEN p.m5 / pop.population
              ELSE NULL
            END AS m5_pc
          , CASE
              WHEN :normalize AND pop.population IS NOT NULL AND pop.population != 0
                THEN p.m6 / pop.population
              ELSE NULL
            END AS m6_pc
          , CASE
              WHEN :normalize AND pop.population IS NOT NULL AND pop.population != 0
                THEN p.total / pop.population
              ELSE NULL
            END AS total_pc
        """
        pop_join = "LEFT JOIN pop ON pop.iso3 = p.iso3"
        pop_join_monthly = "LEFT JOIN pop ON pop.iso3 = ms.iso3"

    bucket_is_label = bucket_col == "class_bin"
    centroid_join = ""
    centroid_expr = "NULL"
    if centroids_available:
        centroid_join, centroid_expr, _bucket_index_expr = eiv_sql.build_centroid_join(
            base_alias="fe",
            metric_expr=":metric",
            hazard_expr="fe.hazard_code",
            bucket_expr=f"fe.{bucket_col}",
            bucket_is_label=bucket_is_label,
        )
    else:
        bucket_index_expr = eiv_sql.bucket_index_expr(
            ":metric", f"fe.{bucket_col}", bucket_is_label=bucket_is_label
        )
        centroid_expr = eiv_sql.fallback_centroid_expr(":metric", bucket_index_expr)

    model_name_available = _table_has_columns(con, "forecasts_ensemble", ["model_name"])
    base_cte = ""
    per_row_cte = f"""
        , per_row AS (
          SELECT
            q.iso3,
            q.hazard_code,
            q.metric,
            fe.{horizon_col} AS m,
            fe.{bucket_col} AS b,
            fe.{prob_col} AS p,
            COALESCE({centroid_expr}, 0) AS centroid
          FROM forecasts_ensemble fe
          JOIN q ON q.question_id = fe.question_id
          {centroid_join}
      WHERE fe.{horizon_col} BETWEEN 1 AND 6
        AND fe.{bucket_col} IS NOT NULL
        AND fe.{prob_col} IS NOT NULL
    )
    """
    if model_name_available:
        base_cte = f"""
        , base AS (
          SELECT
            q.question_id,
            q.iso3,
            q.hazard_code,
            q.metric,
            fe.{horizon_col} AS {horizon_col},
            fe.{bucket_col} AS {bucket_col},
            fe.{prob_col} AS {prob_col},
            COALESCE(fe.model_name, '') AS model_name
          FROM forecasts_ensemble fe
          JOIN q ON q.question_id = fe.question_id
          WHERE fe.{horizon_col} BETWEEN 1 AND 6
            AND fe.{bucket_col} IS NOT NULL
            AND fe.{prob_col} IS NOT NULL
        ),
        chosen_model AS (
          SELECT
            question_id,
            CASE
              WHEN SUM(CASE WHEN model_name = 'ensemble_bayesmc_v2' THEN 1 ELSE 0 END) > 0
                THEN 'ensemble_bayesmc_v2'
              WHEN SUM(CASE WHEN model_name = 'ensemble_mean_v2' THEN 1 ELSE 0 END) > 0
                THEN 'ensemble_mean_v2'
              ELSE MIN(model_name)
            END AS chosen_model
          FROM base
          GROUP BY question_id
        ),
        filtered AS (
          SELECT base.*
          FROM base
          JOIN chosen_model USING (question_id)
          WHERE base.model_name = chosen_model.chosen_model
        )
        """
        per_row_cte = f"""
        , per_row AS (
          SELECT
            fe.iso3,
            fe.hazard_code,
            fe.metric,
            fe.{horizon_col} AS m,
            fe.{bucket_col} AS b,
            fe.{prob_col} AS p,
            COALESCE({centroid_expr}, 0) AS centroid
          FROM filtered fe
          {centroid_join}
        )
        """

    if populations_available:
        monthly_eiv_expr = """
          CASE
            WHEN :agg = 'surge' THEN
              CASE
                WHEN pop.population IS NOT NULL
                  THEN LEAST(pop.population, ms.max_eiv + :alpha * (ms.sum_eiv - ms.max_eiv))
                ELSE ms.max_eiv + :alpha * (ms.sum_eiv - ms.max_eiv)
              END
            ELSE ms.sum_eiv
          END
        """
    else:
        monthly_eiv_expr = """
          CASE
            WHEN :agg = 'surge' THEN ms.max_eiv + :alpha * (ms.sum_eiv - ms.max_eiv)
            ELSE ms.sum_eiv
          END
        """

    sql = f"""
    WITH q AS (
      SELECT question_id, iso3, hazard_code, metric, target_month
      FROM questions
      WHERE UPPER(metric) = :metric
        AND target_month = :target_month
        AND (:hazard_code IS NULL OR UPPER(hazard_code) = UPPER(:hazard_code))
    )
    {pop_cte}
    {base_cte}
    {per_row_cte},
    monthly_hazards AS (
      SELECT
        iso3,
        hazard_code,
        m,
        SUM(p * centroid) AS eiv
      FROM per_row
      GROUP BY iso3, hazard_code, m
    ),
    monthly_summary AS (
      SELECT
        iso3,
        m,
        SUM(eiv) AS sum_eiv,
        MAX(eiv) AS max_eiv
      FROM monthly_hazards
      GROUP BY iso3, m
    ),
    monthly AS (
      SELECT
        ms.iso3,
        ms.m,
        {monthly_eiv_expr} AS eiv
      FROM monthly_summary ms
      {pop_join_monthly}
    ),
    hazards AS (
      SELECT iso3, COUNT(DISTINCT hazard_code) AS n_hazards_forecasted
      FROM q
      GROUP BY iso3
    ),
    pivoted AS (
      SELECT
        iso3,
        SUM(CASE WHEN m = 1 THEN eiv ELSE 0 END) AS m1,
        SUM(CASE WHEN m = 2 THEN eiv ELSE 0 END) AS m2,
        SUM(CASE WHEN m = 3 THEN eiv ELSE 0 END) AS m3,
        SUM(CASE WHEN m = 4 THEN eiv ELSE 0 END) AS m4,
        SUM(CASE WHEN m = 5 THEN eiv ELSE 0 END) AS m5,
        SUM(CASE WHEN m = 6 THEN eiv ELSE 0 END) AS m6,
        CASE
          WHEN :agg = 'surge' THEN MAX(eiv)
          ELSE SUM(eiv)
        END AS total
      FROM monthly
      GROUP BY iso3
    )
    SELECT
      p.iso3,
      h.n_hazards_forecasted,
      p.m1, p.m2, p.m3, p.m4, p.m5, p.m6,
      p.total
      {pop_select}
    FROM pivoted p
    LEFT JOIN hazards h ON h.iso3 = p.iso3
    {pop_join}
    ORDER BY p.total DESC NULLS LAST
    """
    params = {
        "metric": metric_upper,
        "target_month": target_month,
        "hazard_code": hazard_code_upper,
        "agg": agg_mode,
        "alpha": alpha,
    }
    if populations_available:
        params["normalize"] = normalize
    df = _execute(con, sql, params).fetchdf()

    rows = _rows_from_df(df)
    for row in rows:
        row.setdefault("population", None)
        row.setdefault("m1_pc", None)
        row.setdefault("m2_pc", None)
        row.setdefault("m3_pc", None)
        row.setdefault("m4_pc", None)
        row.setdefault("m5_pc", None)
        row.setdefault("m6_pc", None)
        row.setdefault("total_pc", None)
        pop_value = row.get("population")
        if not isinstance(pop_value, (int, float)) or pop_value <= 0:
            pop_value = db_population_map.get(row.get("iso3", ""))
        if normalize and (not pop_value or pop_value <= 0):
            pop_value = _population(row.get("iso3", ""))
        if pop_value and pop_value > 0:
            row["population"] = pop_value
            if agg_mode == "surge":
                capped_months = []
                for key in ["m1", "m2", "m3", "m4", "m5", "m6"]:
                    value = row.get(key)
                    if value is None:
                        capped_months.append(None)
                        continue
                    capped_value = min(float(value), float(pop_value))
                    row[key] = capped_value
                    capped_months.append(capped_value)
                capped_values = [v for v in capped_months if v is not None]
                if capped_values:
                    row["total"] = max(capped_values)
            if normalize:
                row["m1_pc"] = (
                    row["m1"] / pop_value if row.get("m1") is not None else None
                )
                row["m2_pc"] = (
                    row["m2"] / pop_value if row.get("m2") is not None else None
                )
                row["m3_pc"] = (
                    row["m3"] / pop_value if row.get("m3") is not None else None
                )
                row["m4_pc"] = (
                    row["m4"] / pop_value if row.get("m4") is not None else None
                )
                row["m5_pc"] = (
                    row["m5"] / pop_value if row.get("m5") is not None else None
                )
                row["m6_pc"] = (
                    row["m6"] / pop_value if row.get("m6") is not None else None
                )
                row["total_pc"] = (
                    row["total"] / pop_value if row.get("total") is not None else None
                )

    # Back-compat: keep the v1 keys the frontend expects.
    for row in rows:
        # Prefer existing values if present, otherwise map from new names.
        if row.get("expected_value") is None:
            if row.get("total") is not None:
                row["expected_value"] = row["total"]
            elif row.get("eiv_total") is not None:
                row["expected_value"] = row["eiv_total"]

        if row.get("per_capita") is None:
            if row.get("total_pc") is not None:
                row["per_capita"] = row["total_pc"]
            elif row.get("eiv_total_pc") is not None:
                row["per_capita"] = row["eiv_total_pc"]

    for row in rows:
        row["country_name"] = _country_name(row.get("iso3", ""))

    return {
        "metric": metric_upper,
        "target_month": target_month,
        "horizon_m": 6,
        "normalize": normalize,
        "rows": rows,
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


@app.get("/v1/diagnostics/kpi_scopes")
def diagnostics_kpi_scopes(
    metric_scope: str = Query("PA"),
    year_month: Optional[str] = Query(None),
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
    if metric_scope not in {"PA", "FATALITIES"}:
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
        return " AND q.metric = ?", [metric_scope]

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
        sql = (
            f"SELECT DISTINCT strftime({month_source_ts}, '%Y-%m') AS year_month "
            f"FROM {month_source_table} WHERE {month_source_ts} IS NOT NULL"
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

        if has_status:
            resolved_sql = f"{base_sql} AND q.status IN ('resolved', 'closed')"
            scope["resolved_questions"] = _count(
                f"SELECT COUNT(DISTINCT q.question_id) {resolved_sql}",
                question_ids_params + metric_params + status_params,
                "scope_resolved_failed",
            )
        elif _table_has_columns(con, "resolutions", ["question_id"]):
            scope["resolved_questions"] = _count(
                f"SELECT COUNT(DISTINCT r.question_id) FROM resolutions r "
                f"JOIN ({question_ids_sql}) src ON src.question_id = r.question_id",
                question_ids_params,
                "scope_resolutions_failed",
            )
        else:
            notes.append("resolved_questions_unavailable")

        if has_hazard and forecast_source_table and forecast_source_ts:
            forecast_ids_sql = question_ids_sql
            forecast_params = list(question_ids_params)
            if forecast_source_table != question_source_table or forecast_source_phase:
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
            if forecast_source_table != question_source_table or forecast_source_phase:
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

        if has_status:
            resolved_sql = f"{base_sql} AND q.status IN ('resolved', 'closed')"
            scope["resolved_questions"] = _count(
                f"SELECT COUNT(DISTINCT q.question_id) {resolved_sql}",
                metric_params + status_params,
                "scope_resolved_failed",
            )
        elif _table_has_columns(con, "resolutions", ["question_id"]):
            scope["resolved_questions"] = _count(
                "SELECT COUNT(DISTINCT question_id) FROM resolutions",
                [],
                "scope_resolutions_failed",
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

    if selected_month and month_source_table and month_source_ts:
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
                selected_scope = _scope_from_question_ids(
                    question_ids_sql, question_ids_params
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

    explanations = [
        "Questions can exceed forecasts because runs include triaged or researched questions that did not receive forecasts.",
    ]

    return {
        "available_months": available_month_rows,
        "selected_month": selected_month,
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


@app.get("/v1/countries")
def get_countries():
    con = _con()
    try:
        rows = compute_countries_index(con)
        return {"rows": rows}
    except Exception:
        logger.exception("Failed to compute countries index, falling back.")

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


@app.get("/v1/downloads/forecasts.xlsx")
def download_forecasts_xlsx():
    if find_spec("openpyxl") is None:
        logger.warning("openpyxl missing; falling back to CSV export")
        return RedirectResponse(url="/v1/downloads/forecasts.csv", status_code=307)

    con = _con()
    try:
        df = build_forecast_spd_export(con)
    except Exception as exc:
        logger.exception("Failed to build forecast download export")
        raise HTTPException(status_code=500, detail="Failed to build forecast download export") from exc

    buffer = BytesIO()
    try:
        df.to_excel(buffer, index=False, engine="openpyxl")
    except Exception as exc:
        logger.exception("Failed to serialize forecast download export")
        raise HTTPException(status_code=500, detail="Failed to serialize forecast download export") from exc

    buffer.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="pythia_forecasts_export.xlsx"'}
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@app.get("/v1/downloads/forecasts.csv")
def download_forecasts_csv():
    con = _con()
    try:
        df = build_forecast_spd_export(con)
    except Exception as exc:
        logger.exception("Failed to build forecast download export")
        raise HTTPException(status_code=500, detail="Failed to build forecast download export") from exc

    buffer = StringIO()
    try:
        df.to_csv(buffer, index=False)
    except Exception as exc:
        logger.exception("Failed to serialize forecast download export")
        raise HTTPException(status_code=500, detail="Failed to serialize forecast download export") from exc

    buffer.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="pythia_forecasts_export.csv"'}
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


@app.get("/v1/downloads/triage.csv")
def download_triage_csv():
    con = _con()
    try:
        df = build_triage_export(con)
    except Exception as exc:
        logger.exception("Failed to build triage download export")
        raise HTTPException(status_code=500, detail="Failed to build triage download export") from exc

    logger.info(
        "Triage download export rows=%s runs=%s iso3=%s",
        len(df),
        df["Run ID"].nunique(dropna=True),
        df["ISO3"].nunique(dropna=True),
    )

    buffer = StringIO()
    try:
        df.to_csv(buffer, index=False)
    except Exception as exc:
        logger.exception("Failed to serialize triage download export")
        raise HTTPException(status_code=500, detail="Failed to serialize triage download export") from exc

    buffer.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="run_triage_results.csv"'}
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


@app.get("/v1/downloads/total_costs.csv")
def download_total_costs_csv():
    con = _con()
    try:
        tables = build_costs_total(con)
        df = _concat_cost_tables(tables)
    except Exception as exc:
        logger.exception("Failed to build total cost export")
        raise HTTPException(status_code=500, detail="Failed to build total cost export") from exc

    buffer = StringIO()
    try:
        df.to_csv(buffer, index=False)
    except Exception as exc:
        logger.exception("Failed to serialize total cost export")
        raise HTTPException(status_code=500, detail="Failed to serialize total cost export") from exc

    buffer.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="total_costs.csv"'}
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


@app.get("/v1/downloads/monthly_costs.csv")
def download_monthly_costs_csv():
    con = _con()
    try:
        tables = build_costs_monthly(con)
        df = _concat_cost_tables(tables)
    except Exception as exc:
        logger.exception("Failed to build monthly cost export")
        raise HTTPException(status_code=500, detail="Failed to build monthly cost export") from exc

    buffer = StringIO()
    try:
        df.to_csv(buffer, index=False)
    except Exception as exc:
        logger.exception("Failed to serialize monthly cost export")
        raise HTTPException(status_code=500, detail="Failed to serialize monthly cost export") from exc

    buffer.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="monthly_costs.csv"'}
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


@app.get("/v1/downloads/run_costs.csv")
def download_run_costs_csv():
    con = _con()
    try:
        tables = build_costs_runs(con)
        df = _concat_cost_tables(tables)
    except Exception as exc:
        logger.exception("Failed to build run cost export")
        raise HTTPException(status_code=500, detail="Failed to build run cost export") from exc

    buffer = StringIO()
    try:
        df.to_csv(buffer, index=False)
    except Exception as exc:
        logger.exception("Failed to serialize run cost export")
        raise HTTPException(status_code=500, detail="Failed to serialize run cost export") from exc

    buffer.seek(0)
    headers = {"Content-Disposition": 'attachment; filename="run_costs.csv"'}
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers=headers,
    )


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


@app.get("/v1/costs/total")
def costs_total():
    con = _con()
    try:
        tables = build_costs_total(con)
    except Exception as exc:
        logger.exception("Failed to build total costs")
        raise HTTPException(status_code=500, detail="Failed to build total costs") from exc

    return {"tables": {key: _rows_from_df(df) for key, df in tables.items()}}


@app.get("/v1/costs/monthly")
def costs_monthly():
    con = _con()
    try:
        tables = build_costs_monthly(con)
    except Exception as exc:
        logger.exception("Failed to build monthly costs")
        raise HTTPException(status_code=500, detail="Failed to build monthly costs") from exc

    return {"tables": {key: _rows_from_df(df) for key, df in tables.items()}}


@app.get("/v1/costs/runs")
def costs_runs():
    con = _con()
    try:
        tables = build_costs_runs(con)
    except Exception as exc:
        logger.exception("Failed to build run costs")
        raise HTTPException(status_code=500, detail="Failed to build run costs") from exc

    return {"tables": {key: _rows_from_df(df) for key, df in tables.items()}}


@app.get("/v1/costs/latencies")
def costs_latencies():
    con = _con()
    try:
        df = build_latencies_runs(con)
    except Exception as exc:
        logger.exception("Failed to build run latencies")
        raise HTTPException(status_code=500, detail="Failed to build run latencies") from exc

    return {"rows": _rows_from_df(df)}


@app.get("/v1/costs/run_runtimes")
def costs_run_runtimes():
    con = _con()
    try:
        df = build_run_runtimes(con)
    except Exception as exc:
        logger.exception("Failed to build run runtimes")
        raise HTTPException(status_code=500, detail="Failed to build run runtimes") from exc

    rows = _rows_from_df(df)
    missing_question_p50 = sum(
        1 for row in rows if row.get("question_p50_ms") is None
    )
    missing_country_p50 = sum(
        1 for row in rows if row.get("country_p50_ms") is None
    )
    logger.info(
        "costs/run_runtimes rows=%d missing_question_p50=%d missing_country_p50=%d",
        len(rows),
        missing_question_p50,
        missing_country_p50,
    )
    if not rows:
        logger.warning(
            "costs/run_runtimes empty total_rows=%s missing_elapsed_ms=%s missing_created_at=%s missing_run_id=%s",
            df.attrs.get("total_rows"),
            df.attrs.get("missing_elapsed_ms"),
            df.attrs.get("missing_created_at"),
            df.attrs.get("missing_run_id"),
        )
    return {"rows": rows}
