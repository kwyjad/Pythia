# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
import logging
import math
import re
import threading
import time
from datetime import datetime
from importlib.util import find_spec
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import os
import resource

import duckdb, pandas as pd
import numpy as np
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse

from pythia.api.auth import require_admin_token
from pythia.api import db_sync as _db_sync_mod
from pythia.api.db_sync import (
    DbSyncError,
    db_was_refreshed,
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
from pythia.db.schema import connect as db_connect, ensure_schema
from pythia.db.util import ensure_llm_calls_columns
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
from resolver.query.downloads import (
    build_ensemble_scores_export,
    build_forecast_spd_export,
    build_model_scores_export,
    build_rationale_export,
    build_triage_export,
)
from resolver.query.kpi_scopes import compute_countries_triaged_for_month_with_source
from resolver.query.questions_index import (
    compute_questions_forecast_summary,
    compute_questions_triage_summary,
)
from resolver.query.debug_ui import (
    _get_hs_triage_llm_calls_with_debug,
    _get_hs_triage_rows_with_debug,
    _list_hs_runs_with_debug,
    get_hs_triage_all,
    get_country_run_summary,
    list_hs_runs,
)
from resolver.query.resolver_ui import get_connector_last_updated, get_country_facts
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

_DUCKDB_MEMORY_LIMIT = os.getenv("PYTHIA_DUCKDB_MEMORY_LIMIT", "150MB")
_DUCKDB_THREADS = int(os.getenv("PYTHIA_DUCKDB_THREADS", "2"))
_HEAVY_REQUEST_SEMAPHORE = threading.Semaphore(int(os.getenv("PYTHIA_MAX_CONCURRENT_HEAVY", "2")))

_COUNTRY_NAME_BY_ISO3: dict[str, str] = {}
_POPULATION_BY_ISO3: dict[str, int] = {}


def _test_filter(include_test: bool, alias: str = "") -> str:
    """Return a SQL AND clause to exclude test rows unless include_test is True."""
    if include_test:
        return ""
    prefix = f"{alias}." if alias else ""
    return f" AND COALESCE({prefix}is_test, FALSE) = FALSE"


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


def _rows_from_cursor(cursor: duckdb.DuckDBPyConnection) -> List[Dict[str, Any]]:
    """Convert DuckDB cursor results to list of dicts without pandas."""
    desc = cursor.description
    if not desc:
        return []
    columns = [d[0] for d in desc]
    raw = cursor.fetchall()
    if not raw:
        return []
    return _json_sanitize([dict(zip(columns, row)) for row in raw])


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


_READ_CON: Optional[duckdb.DuckDBPyConnection] = None
_READ_CON_LOCK = threading.Lock()
_LAST_SYNC_CHECK: Optional[float] = None
_SYNC_CHECK_INTERVAL_S = 60  # how often to poll for DB refresh


def _open_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """Open a fresh DuckDB connection to the configured DB path.

    If the WAL file is stale, delete it and retry.  Observed failure modes:

    * ``CatalogException``: WAL contains a duplicate ALTER TABLE ADD COLUMN
      from a previous ``ensure_schema`` run.
    * ``InternalException``: WAL replay fails with ``Calling
      DatabaseManager::GetDefaultDatabase with no default database set``,
      typically when the WAL was produced by a different DuckDB process/build
      (e.g. downloaded alongside an artifact) and the main database has not
      yet been attached at replay time.

    The WAL is a replay log of uncommitted changes; removing it loses at most
    the last incomplete transaction, which is acceptable for a read-heavy API
    server whose authoritative data comes from the synced DB file.
    """
    db_url = load_cfg()["app"]["db_url"].replace("duckdb:///", "")
    try:
        con = duckdb.connect(db_url, read_only=False)
    except (duckdb.CatalogException, duckdb.InternalException) as exc:
        wal_path = Path(db_url + ".wal")
        if wal_path.exists():
            logger.warning(
                "DuckDB WAL replay failed (%s); removing stale WAL file %s and retrying",
                exc, wal_path,
            )
            wal_path.unlink()
            con = duckdb.connect(db_url, read_only=False)
        else:
            raise
    con.execute(f"SET memory_limit='{_DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET threads={_DUCKDB_THREADS}")
    logger.info("DuckDB connection opened (memory_limit=%s threads=%s)", _DUCKDB_MEMORY_LIMIT, _DUCKDB_THREADS)
    return con


def _ensure_read_connection() -> duckdb.DuckDBPyConnection:
    """Return a singleton DuckDB connection (created on first call).

    Uses ``read_only=False`` to match the configuration used by
    ``_startup_sync()`` (which opens a write connection for schema
    migrations).  DuckDB rejects opening the same file with a different
    ``read_only`` flag within one process, so both must agree.  All
    request handlers only issue SELECT queries via cursors, so this is safe.
    """
    global _READ_CON  # noqa: PLW0603
    if _READ_CON is not None:
        return _READ_CON
    with _READ_CON_LOCK:
        if _READ_CON is not None:
            return _READ_CON
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
        _READ_CON = _open_duckdb_connection()
        return _READ_CON


def _maybe_refresh_db() -> None:
    """Periodically check if the DB file was replaced and reconnect if so.

    ``maybe_sync_latest_db()`` downloads a new DuckDB file via atomic
    ``os.replace()``, but on Unix the existing open file descriptor still
    reads from the old inode.  This function detects that a new file was
    downloaded and reopens the connection so queries see the latest data.
    """
    global _READ_CON, _LAST_SYNC_CHECK  # noqa: PLW0603
    if _READ_CON is None:
        return

    now = time.monotonic()
    if _LAST_SYNC_CHECK is not None and now - _LAST_SYNC_CHECK < _SYNC_CHECK_INTERVAL_S:
        return
    _LAST_SYNC_CHECK = now

    try:
        maybe_sync_latest_db()
    except DbSyncError as exc:
        logger.warning("Periodic DB sync check failed: %s", exc)
        return

    if db_was_refreshed():
        with _READ_CON_LOCK:
            old_con = _READ_CON
            try:
                _READ_CON = _open_duckdb_connection()
            except Exception:
                logger.warning("Failed to reopen DuckDB after refresh; keeping old connection")
                return
            try:
                if old_con is not None:
                    old_con.close()
            except Exception:
                pass
            logger.info("DuckDB connection reopened after DB refresh")


def _con():
    _maybe_refresh_db()
    return _ensure_read_connection().cursor()


def _require_debug_token(token: Optional[str]) -> None:
    expected = os.getenv("FRED_DEBUG_TOKEN")
    if not expected:
        raise HTTPException(status_code=404, detail="Not found")
    if token != expected:
        raise HTTPException(status_code=403, detail="Forbidden")


def _try_artifact_sync() -> None:
    """Fallback: sync DB from CI artifacts when Release-based sync is unavailable.

    Opt-in via ``PYTHIA_SYNC_FROM_ARTIFACTS=1``.  Requires the ``gh`` CLI.
    """
    if os.environ.get("PYTHIA_SYNC_FROM_ARTIFACTS", "").strip() not in ("1", "true", "yes"):
        return
    try:
        from scripts.sync_db import sync  # noqa: C0415

        sync()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Artifact-based DB sync failed: %s", exc)


@app.post("/v1/admin/force_sync")
def admin_force_sync(token: Optional[str] = Query(None)):
    """Force an immediate DB sync from the GitHub release, bypassing the throttle."""
    global _READ_CON, _LAST_SYNC_CHECK  # noqa: PLW0603
    _require_debug_token(token)
    _LAST_SYNC_CHECK = None
    _db_sync_mod._LAST_SYNC_AT = None  # noqa: SLF001
    _db_sync_mod._LAST_MANIFEST = None  # noqa: SLF001
    try:
        manifest = maybe_sync_latest_db()
    except DbSyncError as exc:
        raise HTTPException(status_code=502, detail=f"Sync failed: {exc}") from exc
    # Force connection refresh if a new DB was downloaded
    if db_was_refreshed():
        with _READ_CON_LOCK:
            old_con = _READ_CON
            try:
                _READ_CON = _open_duckdb_connection()
            except Exception:
                logger.warning("Failed to reopen DuckDB after force sync")
                raise HTTPException(status_code=502, detail="DB reopen failed")
            if old_con is not None:
                try:
                    old_con.close()
                except Exception:
                    pass
        logger.info("Force sync: DB refreshed and connection reopened")
    return {"status": "ok", "manifest": manifest, "db_refreshed": True}


@app.on_event("startup")
def _startup_sync():
    try:
        maybe_sync_latest_db()
    except DbSyncError as exc:
        logger.warning("DB sync failed during startup: %s", exc)
        _try_artifact_sync()
    con = None
    try:
        con = db_connect(read_only=False)
        con.execute(f"SET memory_limit='{_DUCKDB_MEMORY_LIMIT}'")
        con.execute(f"SET threads={_DUCKDB_THREADS}")
        ensure_schema(con)
        ensure_llm_calls_columns(con)
    except Exception as exc:  # noqa: BLE001
        logger.warning("DB schema sync failed during startup: %s", exc)
    finally:
        try:
            if con is not None:
                con.close()
        except Exception:
            pass


from pythia.db.helpers import (
    table_exists as _table_exists,
    table_columns as _table_columns,
    pick_column as _pick_col,
    pick_timestamp_column as _pick_timestamp_column,
)


def _nonnull_count(con: duckdb.DuckDBPyConnection, table: str, col: str) -> int:
    try:
        row = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NOT NULL").fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return 0


def _month_window(year: int, month: int) -> tuple[str, str]:
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)
    return start.isoformat(), end.isoformat()


def _shift_ym(year: int, month: int, delta: int) -> tuple[int, int]:
    """Shift a (year, month) pair by ``delta`` months."""
    total = (year * 12 + (month - 1)) + delta
    return total // 12, (total % 12) + 1


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
    cursor = _execute(con, sql, params)
    desc = cursor.description
    if not desc:
        return None
    columns = [d[0] for d in desc]
    row = cursor.fetchone()
    if not row:
        return None
    return _json_sanitize(dict(zip(columns, row)))


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


def _resolve_forecaster_run_id(
    con: duckdb.DuckDBPyConnection, question_id: str, forecaster_run_id: Optional[str]
) -> Optional[str]:
    if forecaster_run_id:
        return forecaster_run_id
    if not _table_exists(con, "forecasts_ensemble"):
        return None
    row = con.execute(
        """
        SELECT run_id
        FROM forecasts_ensemble
        WHERE question_id = ?
        ORDER BY COALESCE(created_at, '1970-01-01'::TIMESTAMP) DESC, run_id DESC
        LIMIT 1
        """,
        [question_id],
    ).fetchone()
    if not row:
        return None
    return row[0]


def _run_filter_cte(
    con: duckdb.DuckDBPyConnection,
    forecaster_run_id: Optional[str] = None,
) -> tuple[str, str]:
    """Return (cte_sql, join_sql) to filter forecasts_ensemble to one run per question.

    If *forecaster_run_id* is given, filter to that exact run.
    Otherwise, pick the latest run per question via MAX(run_id).
    Returns ("", "") if the table lacks a run_id column (backward compat).
    """
    if not _table_has_columns(con, "forecasts_ensemble", ["run_id"]):
        return ("", "")
    if forecaster_run_id:
        cte = (
            "fc_run_filter AS (\n"
            "            SELECT question_id, run_id\n"
            "            FROM forecasts_ensemble\n"
            f"            WHERE run_id = '{forecaster_run_id}'\n"
            "            GROUP BY question_id, run_id\n"
            "        )"
        )
        join = "JOIN fc_run_filter fr ON fr.question_id = fe.question_id AND fr.run_id = fe.run_id"
    else:
        cte = (
            "fc_run_filter AS (\n"
            "            SELECT question_id, MAX(run_id) AS run_id\n"
            "            FROM forecasts_ensemble\n"
            "            WHERE run_id IS NOT NULL\n"
            "            GROUP BY question_id\n"
            "        )"
        )
        join = "JOIN fc_run_filter fr ON fr.question_id = fe.question_id AND fr.run_id = fe.run_id"
    return (cte, join)


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
    con: duckdb.DuckDBPyConnection,
    metric_upper: str,
    horizon_col: str,
    horizon_m: int,
    include_test: bool = False,
    forecaster_run_id: Optional[str] = None,
) -> Optional[str]:
    if not _table_exists(con, "questions") or not _table_exists(con, "forecasts_ensemble"):
        return None
    _tf = _test_filter(include_test, "q")
    run_clause = ""
    params: dict[str, Any] = {"metric": metric_upper, "horizon_m": horizon_m}
    if forecaster_run_id:
        run_clause = " AND fe.run_id = :run_id"
        params["run_id"] = forecaster_run_id
    row = _execute(
        con,
        f"""
        SELECT MAX(q.target_month) AS target_month
        FROM forecasts_ensemble fe
        JOIN questions q ON q.question_id = fe.question_id
        WHERE UPPER(q.metric) = :metric
          AND fe.{horizon_col} = :horizon_m
          {_tf}{run_clause}
        """,
        params,
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
    result = dict(manifest)
    # Add DB staleness diagnostics so operators can verify the API has the latest data.
    try:
        con = _con()
        row = con.execute(
            "SELECT MAX(strftime(created_at, '%Y-%m')) FROM forecasts_ensemble WHERE created_at IS NOT NULL"
        ).fetchone()
        result["latest_forecast_month"] = row[0] if row and row[0] else None
        run_row = con.execute(
            "SELECT run_id FROM forecasts_ensemble WHERE run_id IS NOT NULL ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
        result["latest_forecast_run_id"] = run_row[0] if run_row and run_row[0] else None
    except Exception:
        result["latest_forecast_month"] = None
        result["latest_forecast_run_id"] = None
    return result


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
    rows = _rows_from_cursor(con.execute(
        "SELECT * FROM ui_runs WHERE ui_run_id = ?",
        [ui_run_id],
    ))
    if not rows:
        return {"found": False, "row": None}
    return {"found": True, "row": rows[0]}


@app.get("/v1/questions")
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


@app.get("/v1/question_bundle")
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

    rows = _rows_from_cursor(_execute(con, sql, params))

    if not rows:
        return {"found": False, "as_of_month": as_of_month, "rows": []}

    # We return rows, plus the resolved as_of_month for convenience
    return {
        "found": True,
        "as_of_month": as_of_month,
        "rows": rows,
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


@app.get("/v1/performance/scores")
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

    # Track counts for KPI cards (always unfiltered by track)
    track_counts = {"track1": 0, "track2": 0}
    if has_track:
        try:
            tc_filter = metric_filter  # respect metric filter but not track filter
            tc_rows = _execute(con, f"""
                SELECT q.track, COUNT(DISTINCT s.question_id) AS n
                FROM scores s
                JOIN questions q ON q.question_id = s.question_id
                WHERE q.track IS NOT NULL {tc_filter}
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
    _tf = _test_filter(include_test, "s")
    sql_summary = f"""
        SELECT
          q.hazard_code,
          UPPER(q.metric) AS metric,
          s.score_type,
          s.model_name,
          COUNT(*) AS n_samples,
          COUNT(DISTINCT s.question_id) AS n_questions,
          AVG(s.value) AS avg_value,
          MEDIAN(s.value) AS median_value
        FROM scores s
        JOIN questions q ON q.question_id = s.question_id
        WHERE 1=1 {metric_filter} {track_filter}{_tf}
        GROUP BY q.hazard_code, UPPER(q.metric), s.score_type, s.model_name
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
            GROUP BY q.hs_run_id, q.hazard_code, UPPER(q.metric), s.score_type, s.model_name
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
              s.score_type,
              s.model_name,
              COUNT(*) AS n_samples,
              COUNT(DISTINCT s.question_id) AS n_questions,
              AVG(s.value) AS avg_value,
              MEDIAN(s.value) AS median_value
            FROM scores s
            JOIN questions q ON q.question_id = s.question_id
            WHERE 1=1 {metric_filter} {track_filter}{_tf}
            GROUP BY q.hs_run_id, q.hazard_code, UPPER(q.metric), s.score_type, s.model_name
            ORDER BY q.hs_run_id DESC, q.hazard_code, s.score_type,
                     s.model_name NULLS FIRST
        """
    run_rows = _rows_from_cursor(_execute(con, sql_runs, params))

    return {
        "summary_rows": summary_rows,
        "run_rows": run_rows,
        "track_counts": track_counts,
    }


@app.get("/v1/forecasts/ensemble")
def get_forecasts_ensemble(
    iso3: Optional[str] = Query(None),
    hazard_code: Optional[str] = Query(None),
    metric: Optional[str] = Query(None),
    target_month: Optional[str] = Query(None),
    horizon_m: Optional[int] = Query(None),
    latest_only: bool = Query(True),
    include_test: bool = Query(False),
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
        return {"rows": _rows_from_cursor(_execute(con, sql, params))}

    # latest_only=False: historical view (all runs)
    sql = f"""
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
      WHERE 1=1{_test_filter(include_test, "fe")}
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
    return {"rows": _rows_from_cursor(_execute(con, sql, params))}


@app.get("/v1/forecasts/history")
def get_forecasts_history(
    iso3: str = Query(...),
    hazard_code: str = Query(...),
    metric: str = Query(...),
    target_month: str = Query(...),
    include_test: bool = Query(False),
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

    sql = f"""
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
        {_test_filter(include_test, "fe")}
      ORDER BY h.created_at, fe.horizon_m, fe.class_bin
    """
    return {"rows": _rows_from_cursor(_execute(con, sql, params))}


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
    return {"rows": _rows_from_cursor(con.execute(
        f"SELECT * FROM resolutions WHERE question_id IN ({inlist})",
        qids,
    ))}


def _get_risk_index_binary(
    con,
    metric_upper: str,
    hazard_code_upper: Optional[str],
    target_month: Optional[str],
    horizon_m: int,
    duration_m: int,
    normalize: bool,
    forecaster_run_id: Optional[str],
    include_test: bool = False,
) -> dict:
    """Risk index for binary EVENT_OCCURRENCE questions.

    The 'risk value' is the ensemble P(event) stored in bucket_1.
    No centroid multiplication is needed.
    """
    horizon_col, bucket_col, prob_col = _resolve_forecasts_ensemble_columns(con)
    if not horizon_col or not bucket_col or not prob_col:
        return {
            "metric": metric_upper, "target_month": target_month,
            "horizon_m": horizon_m, "duration_m": duration_m,
            "normalize": normalize, "rows": [], "metric_type": "binary",
        }

    if not target_month:
        target_month = _latest_forecasted_target_month(
            con, metric_upper, horizon_col, horizon_m,
            include_test=include_test,
            forecaster_run_id=forecaster_run_id,
        )
        if not target_month:
            _tf = _test_filter(include_test)
            _run_clause = ""
            _fallback_params: dict[str, Any] = {"metric": metric_upper}
            if forecaster_run_id:
                _run_clause = " AND question_id IN (SELECT question_id FROM forecasts_ensemble WHERE run_id = :run_id)"
                _fallback_params["run_id"] = forecaster_run_id
            row = _execute(
                con,
                f"SELECT MAX(target_month) FROM questions WHERE UPPER(metric)=:metric{_tf}{_run_clause}",
                _fallback_params,
            ).fetchone()
            target_month = row[0] if row and row[0] else None

    if not target_month:
        return {
            "metric": metric_upper, "target_month": None,
            "horizon_m": 6, "duration_m": duration_m,
            "normalize": normalize, "rows": [], "metric_type": "binary",
        }

    run_cte, run_join = _run_filter_cte(con, forecaster_run_id)
    run_cte_sql = f", {run_cte}" if run_cte else ""

    model_name_available = _table_has_columns(con, "forecasts_ensemble", ["model_name"])

    # For binary questions, bucket_1 = P(yes). We detect bucket_1 via
    # bucket_index=1 or class_bin label matching.
    bucket_is_label = bucket_col == "class_bin"
    if bucket_is_label:
        bucket_filter = f"AND fe.{bucket_col} IN ('1', 'bucket_1', 'yes')"
    else:
        bucket_filter = f"AND fe.{bucket_col} = 1"

    # Model selection CTE (prefer ensemble_bayesmc_v2 > ensemble_mean_v2)
    model_cte = ""
    from_alias = "fe"
    if model_name_available:
        model_cte = f"""
        , base_b AS (
          SELECT
            q.question_id, q.iso3, q.hazard_code,
            fe.{horizon_col} AS m,
            fe.{prob_col} AS p,
            COALESCE(fe.model_name, '') AS model_name
          FROM forecasts_ensemble fe
          JOIN q ON q.question_id = fe.question_id
          {run_join}
          WHERE fe.{horizon_col} BETWEEN 1 AND 6
            AND fe.{prob_col} IS NOT NULL
            {bucket_filter}
        ),
        chosen_model_b AS (
          SELECT question_id,
            CASE
              WHEN SUM(CASE WHEN model_name = 'ensemble_bayesmc_v2' THEN 1 ELSE 0 END) > 0
                THEN 'ensemble_bayesmc_v2'
              WHEN SUM(CASE WHEN model_name = 'ensemble_mean_v2' THEN 1 ELSE 0 END) > 0
                THEN 'ensemble_mean_v2'
              ELSE MIN(model_name)
            END AS chosen_model
          FROM base_b GROUP BY question_id
        ),
        binary_probs AS (
          SELECT base_b.iso3, base_b.hazard_code, base_b.m, base_b.p
          FROM base_b
          JOIN chosen_model_b USING (question_id)
          WHERE base_b.model_name = chosen_model_b.chosen_model
        )
        """
        from_alias = "binary_probs"
    else:
        model_cte = f"""
        , binary_probs AS (
          SELECT q.iso3, q.hazard_code,
                 fe.{horizon_col} AS m,
                 fe.{prob_col} AS p
          FROM forecasts_ensemble fe
          JOIN q ON q.question_id = fe.question_id
          {run_join}
          WHERE fe.{horizon_col} BETWEEN 1 AND 6
            AND fe.{prob_col} IS NOT NULL
            {bucket_filter}
        )
        """
        from_alias = "binary_probs"

    _tf = _test_filter(include_test)
    sql = f"""
    WITH q AS (
      SELECT question_id, iso3, hazard_code, metric, target_month
      FROM questions
      WHERE UPPER(metric) = :metric
        AND target_month = :target_month
        AND (:hazard_code IS NULL OR UPPER(hazard_code) = UPPER(:hazard_code))
        {_tf}
    )
    {run_cte_sql}
    {model_cte},
    monthly AS (
      SELECT iso3, m, MAX(p) AS prob
      FROM {from_alias}
      GROUP BY iso3, m
    ),
    hazards AS (
      SELECT iso3, COUNT(DISTINCT hazard_code) AS n_hazards_forecasted
      FROM q GROUP BY iso3
    ),
    pivoted AS (
      SELECT
        iso3,
        MAX(CASE WHEN m = 1 THEN prob END) AS m1,
        MAX(CASE WHEN m = 2 THEN prob END) AS m2,
        MAX(CASE WHEN m = 3 THEN prob END) AS m3,
        MAX(CASE WHEN m = 4 THEN prob END) AS m4,
        MAX(CASE WHEN m = 5 THEN prob END) AS m5,
        MAX(CASE WHEN m = 6 THEN prob END) AS m6,
        MAX(prob) AS total
      FROM monthly
      GROUP BY iso3
    )
    SELECT
      p.iso3,
      h.n_hazards_forecasted,
      p.m1, p.m2, p.m3, p.m4, p.m5, p.m6,
      p.total,
      NULL AS population,
      p.m1 AS m1_pc, p.m2 AS m2_pc, p.m3 AS m3_pc,
      p.m4 AS m4_pc, p.m5 AS m5_pc, p.m6 AS m6_pc,
      p.total AS total_pc
    FROM pivoted p
    LEFT JOIN hazards h ON h.iso3 = p.iso3
    ORDER BY p.total DESC NULLS LAST
    """
    params = {
        "metric": metric_upper,
        "target_month": target_month,
        "hazard_code": hazard_code_upper,
    }
    rows = _rows_from_cursor(_execute(con, sql, params))

    for row in rows:
        row["country_name"] = _country_name(row.get("iso3", ""))
        row["metric_type"] = "binary"
        # For binary, per-capita IS the raw probability (already a rate)
        if row.get("total") is not None:
            row["expected_value"] = row["total"]
            row["per_capita"] = row["total"]

    return {
        "metric": metric_upper,
        "target_month": target_month,
        "horizon_m": 6,
        "duration_m": duration_m,
        "normalize": normalize,
        "forecaster_run_id": forecaster_run_id,
        "rows": rows,
        "metric_type": "binary",
    }


@app.get("/v1/risk_index")
def get_risk_index(
    metric: str = Query("PA", description="Metric to rank on, e.g. 'PA'"),
    hazard_code: Optional[str] = Query(None, description="Optional hazard code filter"),
    target_month: Optional[str] = Query(None, description="Target month 'YYYY-MM'"),
    horizon_m: int = Query(1, ge=1, le=6, description="Forecast horizon in months ahead"),
    duration_m: int = Query(6, ge=1, le=12, description="Duration in months (metadata only)"),
    normalize: bool = Query(True, description="If true, include per-capita ranking"),
    agg: str = Query("surge", description="Aggregation mode: surge (default) or burden (legacy)"),
    alpha: float = Query(0.1, ge=0, le=1, description="Surge blending weight"),
    forecaster_run_id: Optional[str] = Query(None, description="Forecaster run ID to scope results"),
    include_test: bool = Query(False),
):
    con = _con()
    metric_upper = (metric or "").strip().upper() or "PA"
    hazard_code_upper = (hazard_code or "").strip().upper() or None
    is_pa = metric_upper == "PA"
    is_fatalities = metric_upper == "FATALITIES"
    is_binary = metric_upper == "EVENT_OCCURRENCE"
    is_phase3 = metric_upper == "PHASE3PLUS_IN_NEED"

    if is_binary:
        return _get_risk_index_binary(
            con, metric_upper, hazard_code_upper, target_month,
            horizon_m, duration_m, normalize, forecaster_run_id,
            include_test=include_test,
        )

    if not (is_pa or is_fatalities or is_phase3):
        return {
            "metric": metric_upper,
            "target_month": target_month,
            "horizon_m": horizon_m,
            "duration_m": duration_m,
            "normalize": normalize,
            "rows": [],
        }

    horizon_col, bucket_col, prob_col = _resolve_forecasts_ensemble_columns(con)
    if not horizon_col or not bucket_col or not prob_col:
        return {
            "metric": metric_upper,
            "target_month": target_month,
            "horizon_m": horizon_m,
            "duration_m": duration_m,
            "normalize": normalize,
            "rows": [],
        }

    if not target_month:
        target_month = _latest_forecasted_target_month(
            con, metric_upper, horizon_col, horizon_m,
            include_test=include_test,
            forecaster_run_id=forecaster_run_id,
        )
        if not target_month:
            _tf = _test_filter(include_test)
            _run_clause = ""
            _fallback_params: dict[str, Any] = {"metric": metric_upper}
            if forecaster_run_id:
                _run_clause = " AND question_id IN (SELECT question_id FROM forecasts_ensemble WHERE run_id = :run_id)"
                _fallback_params["run_id"] = forecaster_run_id
            row = _execute(
                con,
                f"SELECT MAX(target_month) FROM questions WHERE UPPER(metric)=:metric{_tf}{_run_clause}",
                _fallback_params,
            ).fetchone()
            target_month = row[0] if row and row[0] else None

    if not target_month:
        return {
            "metric": metric_upper,
            "target_month": None,
            "horizon_m": 6,
            "duration_m": duration_m,
            "normalize": normalize,
            "rows": [],
        }

    db_population_map: dict[str, int] = {}
    populations_available = False
    populations_table_available = _table_exists(con, "populations") and _table_has_columns(
        con, "populations", ["iso3", "population", "year"]
    )
    if populations_table_available:
        for pop_row in _rows_from_cursor(_execute(
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
        )):
            iso = str(pop_row.get("iso3", "")).strip().upper()
            if not iso:
                continue
            try:
                pop_value = int(float(pop_row.get("population", 0)))
            except Exception:
                continue
            if pop_value <= 0:
                continue
            db_population_map[iso] = pop_value
        populations_available = bool(db_population_map)
    agg_mode = "surge" if (is_pa or is_phase3) else "burden"
    registry_available = False
    if (normalize or agg_mode == "surge") and not populations_available:
        _load_population_registry()
        registry_available = bool(_POPULATION_BY_ISO3)
        if not registry_available:
            logger.debug("Population registry empty; per-capita values unavailable.")
    if agg is not None and str(agg).strip():
        requested_agg = str(agg).strip().lower()
        if requested_agg not in {"surge", "burden"}:
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

    run_cte, run_join = _run_filter_cte(con, forecaster_run_id)
    run_cte_sql = f", {run_cte}" if run_cte else ""

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
          {run_join}
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
          {run_join}
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

    if is_pa:
        if populations_available:
            monthly_eiv_expr = """
              CASE
                WHEN pop.population IS NOT NULL
                  THEN LEAST(pop.population, ms.max_eiv + :alpha * (ms.sum_eiv - ms.max_eiv))
                ELSE ms.max_eiv + :alpha * (ms.sum_eiv - ms.max_eiv)
              END
            """
        else:
            monthly_eiv_expr = """
              ms.max_eiv + :alpha * (ms.sum_eiv - ms.max_eiv)
            """
    else:
        monthly_eiv_expr = "ms.sum_eiv"

    _tf = _test_filter(include_test)
    sql = f"""
    WITH q AS (
      SELECT question_id, iso3, hazard_code, metric, target_month
      FROM questions
      WHERE UPPER(metric) = :metric
        AND target_month = :target_month
        AND (:hazard_code IS NULL OR UPPER(hazard_code) = UPPER(:hazard_code))
        {_tf}
    )
    {run_cte_sql}
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
    rows = _rows_from_cursor(_execute(con, sql, params))
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
        row["metric_type"] = "spd"

    return {
        "metric": metric_upper,
        "target_month": target_month,
        "horizon_m": 6,
        "duration_m": duration_m,
        "normalize": normalize,
        "forecaster_run_id": forecaster_run_id,
        "rows": rows,
        "metric_type": "spd",
    }


@app.get("/v1/rankings")
def rankings(
    month: str,
    metric: str = "PIN",
    normalize: bool = True,
    forecaster_run_id: Optional[str] = Query(None, description="Forecaster run ID to scope results"),
    include_test: bool = Query(False),
):
    con = _con()
    run_cte, run_join = _run_filter_cte(con, forecaster_run_id)
    run_cte_sql = f"{run_cte}," if run_cte else ""
    sql = f"""
    WITH {run_cte_sql}
    ev AS (
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
      {run_join}
      LEFT JOIN bucket_centroids bc
        ON bc.metric = q.metric
       AND bc.class_bin = fe.class_bin
      AND bc.hazard_code = q.hazard_code
      WHERE q.metric=? AND q.target_month=?{_test_filter(include_test, "fe")}
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
    return {"rows": _rows_from_cursor(con.execute(sql, [metric, month, normalize, normalize]))}


@app.get("/v1/diagnostics/memory")
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


@app.get("/v1/diagnostics/summary")
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
        q_counts = _rows_from_cursor(con.execute(
            f"SELECT status, COUNT(*) AS n FROM questions WHERE 1=1{_tf} GROUP BY status"
        ))
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


@app.get("/v1/debug/hs_runs")
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


@app.get("/v1/hs_runs")
def hs_runs(
    limit: int = Query(50, ge=1, le=500),
    include_test: bool = Query(False),
):
    con = _con()
    rows = list_hs_runs(con, limit=limit, include_test=include_test)
    logger.info("HS runs rows=%d", len(rows))
    return {"rows": rows}


@app.get("/v1/hs_triage/all")
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


@app.get("/v1/debug/hs_triage")
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


@app.get("/v1/debug/hs_triage_llm_calls")
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


@app.get("/v1/debug/hs_country_summary")
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


@app.get("/v1/diagnostics/resolution_rates")
def resolution_rates(
    forecaster_run_id: Optional[str] = Query(None),
    hazard_code: Optional[str] = Query(None),
    include_test: bool = Query(False),
):
    """Compute resolution rates by hazard and metric.

    Returns how many questions were resolved (have at least 1 resolution row)
    vs total, broken down by (hazard_code, metric).
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

    # Total questions by (hazard_code, metric)
    _tf = _test_filter(include_test, "r")
    total_sql = f"""
        SELECT q.hazard_code, UPPER(q.metric) AS metric,
               COUNT(DISTINCT q.question_id) AS total_questions
        FROM questions q
        WHERE 1=1 {hazard_filter} {run_filter}{_test_filter(include_test, "q")}
        GROUP BY q.hazard_code, UPPER(q.metric)
    """
    try:
        total_rows = _execute(con, total_sql, params).fetchall()
    except Exception:
        return {"rows": []}

    if not has_resolutions:
        return {
            "rows": [
                {
                    "hazard_code": hc,
                    "metric": m,
                    "total_questions": int(t),
                    "resolved_questions": 0,
                    "skipped_questions": int(t),
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
        WHERE 1=1 {hazard_filter} {run_filter}{_tf}
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
            "resolution_rate": round(rate, 4),
        })

    return {"rows": result}


@app.get("/v1/diagnostics/kpi_scopes")
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

@app.get("/v1/diagnostics/run_summary")
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
    ensemble: Dict[str, int] = {"expected": 6, "ok": tracks["track1"]["models"]}

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
    }


@app.get("/v1/countries")
def get_countries(
    metric_scope: Optional[str] = Query(None),
    year_month: Optional[str] = Query(None),
    forecaster_run_id: Optional[str] = Query(None, description="Forecaster run ID to scope countries"),
    include_test: bool = Query(False),
):
    con = _con()
    try:
        rows = compute_countries_index(
            con, metric_scope=metric_scope, year_month=year_month,
            forecaster_run_id=forecaster_run_id,
            include_test=include_test,
        )
        return {"rows": rows}
    except Exception:
        logger.exception("Failed to compute countries index, falling back.")

    if not _table_exists(con, "questions"):
        return {"rows": []}

    metric_filter = ""
    params: list[str] = []
    if metric_scope:
        metric_filter = "WHERE UPPER(q.metric) = ?"
        params = [metric_scope.upper()]

    if _table_exists(con, "forecasts_ensemble"):
        sql = f"""
          SELECT
            q.iso3,
            COUNT(DISTINCT q.question_id) AS n_questions,
            COUNT(DISTINCT fe.question_id) AS n_forecasted
          FROM questions q
          LEFT JOIN forecasts_ensemble fe ON fe.question_id = q.question_id
          {metric_filter}
          GROUP BY q.iso3
          ORDER BY q.iso3
        """
    else:
        sql = f"""
          SELECT
            q.iso3,
            COUNT(DISTINCT q.question_id) AS n_questions,
            0 AS n_forecasted
          FROM questions q
          {metric_filter}
          GROUP BY q.iso3
          ORDER BY q.iso3
        """

    return {"rows": _rows_from_cursor(con.execute(sql, params))}


@app.get("/v1/resolver/connector_status")
def get_resolver_connector_status():
    con = _con()
    rows, diagnostics = get_connector_last_updated(con)
    summary = ", ".join(
        f"{row.get('source')}={row.get('last_updated')}" for row in rows
    )
    logger.info("Resolver connector status rows=%s updates=%s", len(rows), summary)
    return {"rows": rows, "diagnostics": diagnostics}


@app.get("/v1/resolver/country_facts")
def get_resolver_country_facts(
    iso3: str = Query(..., description="ISO3 country code"),
    limit: int = Query(5000, description="Maximum rows to return"),
):
    iso3_value = (iso3 or "").strip().upper()
    if not re.fullmatch(r"[A-Z]{3}", iso3_value or ""):
        raise HTTPException(status_code=400, detail="iso3 must be a 3-letter code")
    con = _con()
    rows, diagnostics = get_country_facts(con, iso3_value, limit=limit)
    return {"rows": rows, "iso3": iso3_value, "diagnostics": diagnostics}


# ---------------------------------------------------------------------------
# Resolver data explorer endpoints
# ---------------------------------------------------------------------------

_DB_SUMMARY_TABLES = [
    # (table_name, date_column_candidates, has_iso3)
    # Freshness columns: prefer ingestion-time over data-period columns
    ("facts_resolved", ["created_at", "as_of_date"], True),
    ("facts_deltas", ["created_at"], True),
    ("acled_monthly_fatalities", ["updated_at"], True),
    ("conflict_forecasts", ["created_at", "forecast_issue_date"], True),
    ("reliefweb_reports", ["fetched_at", "published_date"], True),
    ("acled_political_events", ["fetched_at", "event_date"], True),
    ("acaps_inform_severity", ["fetched_at", "snapshot_date"], True),
    ("acaps_risk_radar", ["fetched_at"], True),
    ("acaps_daily_monitoring", ["fetched_at", "entry_date"], True),
    ("acaps_humanitarian_access", ["fetched_at", "snapshot_date"], True),
    ("seasonal_forecasts", ["created_at", "forecast_issue_date"], True),
    ("enso_state", ["created_at", "fetch_date"], False),
    ("seasonal_tc_outlooks", ["fetched_at"], False),
    ("seasonal_tc_context_cache", ["fetched_at"], True),
    ("hdx_signals", ["fetched_at", "signal_date"], True),
    ("crisiswatch_entries", ["fetched_at"], True),
]


@app.get("/v1/resolver/db_summary")
def get_resolver_db_summary():
    con = _con()
    tables = []
    for tbl, date_candidates, has_iso3 in _DB_SUMMARY_TABLES:
        if not _table_exists(con, tbl):
            continue
        try:
            row_count = con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        except Exception:
            row_count = 0
        last_updated = None
        cols = _table_columns(con, tbl)
        for cand in date_candidates:
            if cand.lower() in cols:
                try:
                    val = con.execute(
                        f"SELECT MAX({cand}) FROM {tbl} WHERE {cand} IS NOT NULL"
                    ).fetchone()
                    if val and val[0] is not None:
                        last_updated = str(val[0])[:10]
                        break
                except Exception:
                    pass
        tables.append({
            "name": tbl,
            "row_count": row_count,
            "last_updated": last_updated,
            "has_iso3": has_iso3,
        })
    return {"tables": tables}


def _resolver_query(table: str, iso3: str | None, limit: int,
                    order_by: str = "", extra_where: str = "",
                    extra_params: list | None = None,
                    exclude_cols: set[str] | None = None) -> dict:
    """Generic helper for resolver data-explorer endpoints."""
    con = _con()
    if not _table_exists(con, table):
        return {"rows": []}
    cols = _table_columns(con, table)
    if exclude_cols:
        select_cols = ", ".join(
            c for c in sorted(cols) if c not in {e.lower() for e in exclude_cols}
        )
    else:
        select_cols = "*"
    sql = f"SELECT {select_cols} FROM {table}"
    params: list = list(extra_params or [])
    clauses: list[str] = []
    if iso3 and "iso3" in cols:
        clauses.append("iso3 = ?")
        params.append(iso3.upper())
    if extra_where:
        clauses.append(extra_where)
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    if order_by:
        sql += f" ORDER BY {order_by}"
    sql += f" LIMIT {min(limit, 5000)}"
    return {"rows": _rows_from_cursor(con.execute(sql, params))}


@app.get("/v1/resolver/facts_deltas")
def get_resolver_facts_deltas(
    iso3: str | None = Query(None), limit: int = Query(500),
):
    return _resolver_query("facts_deltas", iso3, limit, order_by="created_at DESC")


@app.get("/v1/resolver/acled_monthly_fatalities")
def get_resolver_acled_monthly_fatalities(
    iso3: str | None = Query(None), limit: int = Query(500),
):
    return _resolver_query("acled_monthly_fatalities", iso3, limit,
                           order_by="year DESC, month DESC")


@app.get("/v1/resolver/conflict_forecasts")
def get_resolver_conflict_forecasts(
    iso3: str | None = Query(None),
    source: str | None = Query(None),
    limit: int = Query(500),
):
    extra_where = ""
    extra_params: list = []
    if source:
        extra_where = "source = ?"
        extra_params.append(source)
    return _resolver_query("conflict_forecasts", iso3, limit,
                           order_by="forecast_issue_date DESC",
                           extra_where=extra_where, extra_params=extra_params)


@app.get("/v1/resolver/reliefweb_reports")
def get_resolver_reliefweb_reports(
    iso3: str | None = Query(None),
    include_body: bool = Query(False),
    limit: int = Query(100),
):
    exclude = {"body"} if not include_body else set()
    return _resolver_query("reliefweb_reports", iso3, limit,
                           order_by="date DESC", exclude_cols=exclude)


@app.get("/v1/resolver/acled_political_events")
def get_resolver_acled_political_events(
    iso3: str | None = Query(None), limit: int = Query(500),
):
    return _resolver_query("acled_political_events", iso3, limit,
                           order_by="event_date DESC")


@app.get("/v1/resolver/acaps")
def get_resolver_acaps(
    iso3: str | None = Query(None),
    dataset: str = Query("inform_severity",
                         description="inform_severity|risk_radar|daily_monitoring|humanitarian_access"),
    limit: int = Query(500),
):
    table_map = {
        "inform_severity": ("acaps_inform_severity", "snapshot_date DESC"),
        "risk_radar": ("acaps_risk_radar", "fetched_at DESC"),
        "daily_monitoring": ("acaps_daily_monitoring", "entry_date DESC"),
        "humanitarian_access": ("acaps_humanitarian_access", "snapshot_date DESC"),
    }
    tbl, order = table_map.get(dataset, ("acaps_inform_severity", "snapshot_date DESC"))
    return _resolver_query(tbl, iso3, limit, order_by=order)


@app.get("/v1/resolver/seasonal_forecasts")
def get_resolver_seasonal_forecasts(
    iso3: str | None = Query(None), limit: int = Query(500),
):
    return _resolver_query("seasonal_forecasts", iso3, limit,
                           order_by="forecast_issue_date DESC")


@app.get("/v1/resolver/hdx_signals")
def get_resolver_hdx_signals(
    iso3: str | None = Query(None), limit: int = Query(500),
):
    return _resolver_query("hdx_signals", iso3, limit,
                           order_by="signal_date DESC")


@app.get("/v1/resolver/crisiswatch")
def get_resolver_crisiswatch(
    iso3: str | None = Query(None), limit: int = Query(500),
):
    return _resolver_query("crisiswatch_entries", iso3, limit,
                           order_by="year DESC, month DESC")


@app.get("/v1/resolver/enso_state")
def get_resolver_enso_state(limit: int = Query(10)):
    return _resolver_query("enso_state", None, limit,
                           order_by="fetch_date DESC")


@app.get("/v1/resolver/seasonal_tc_outlooks")
def get_resolver_seasonal_tc_outlooks(limit: int = Query(50)):
    return _resolver_query("seasonal_tc_outlooks", None, limit,
                           order_by="fetched_at DESC")


# ---------------------------------------------------------------------------
# Source-level data explorer (accordion page)
# ---------------------------------------------------------------------------

# Best column for "when was this data last ingested" per table.
_FRESHNESS_CANDIDATES = ("fetched_at", "created_at", "stored_at",
                         "ingested_at", "fetch_date", "updated_at")

_SOURCE_REGISTRY: dict[str, dict] = {
    # --- Resolution Data (facts_resolved, filtered by publisher) ---
    "ifrc":             {"table": "facts_resolved",
                         "filter": "LOWER(publisher) IN ('ifrc', 'ifrc_go', 'ifrc_montandon')",
                         "columns": ["iso3", "hazard_code", "metric", "ym",
                                     "value", "as_of_date", "publisher", "source_id"],
                         "order": "created_at DESC"},
    "idmc":             {"table": "facts_resolved",
                         "filter": "LOWER(publisher) IN ('idmc')",
                         "columns": ["iso3", "hazard_code", "metric", "ym",
                                     "value", "as_of_date", "publisher", "source_id"],
                         "order": "created_at DESC"},
    "acled":            {"table": "facts_resolved",
                         "filter": "LOWER(publisher) IN ('acled')",
                         "columns": ["iso3", "hazard_code", "metric", "ym",
                                     "value", "as_of_date", "publisher", "source_id"],
                         "order": "created_at DESC"},
    "gdacs":            {"table": "facts_resolved",
                         "filter": "publisher = 'GDACS / JRC'",
                         "columns": ["iso3", "hazard_code", "metric", "ym",
                                     "value", "alertlevel", "as_of_date", "publisher"],
                         "order": "created_at DESC"},
    "fewsnet":          {"table": "facts_resolved",
                         "filter": "publisher = 'FEWS NET'",
                         "columns": ["iso3", "hazard_code", "metric", "ym",
                                     "value", "as_of_date", "publisher"],
                         "order": "created_at DESC"},
    "ipc_api":          {"table": "facts_resolved",
                         "filter": "publisher = 'IPC'",
                         "columns": ["iso3", "hazard_code", "metric", "ym",
                                     "value", "as_of_date", "publisher"],
                         "order": "created_at DESC"},
    "acled_fatalities":  {"table": "acled_monthly_fatalities",
                         "columns": ["iso3", "month", "fatalities", "source"],
                         "order": "month DESC"},
    # --- Conflict Forecasts ---
    "views":            {"table": "conflict_forecasts",
                         "filter": "source = 'VIEWS'",
                         "columns": ["iso3", "hazard_code", "metric",
                                     "lead_months", "value",
                                     "forecast_issue_date", "target_month"],
                         "order": "forecast_issue_date DESC"},
    "conflictforecast": {"table": "conflict_forecasts",
                         "filter": "source = 'conflictforecast_org'",
                         "columns": ["iso3", "hazard_code", "metric",
                                     "lead_months", "value",
                                     "forecast_issue_date", "target_month"],
                         "order": "forecast_issue_date DESC"},
    "acled_cast":       {"table": "conflict_forecasts",
                         "filter": "source = 'ACLED_CAST'",
                         "columns": ["iso3", "hazard_code", "metric",
                                     "lead_months", "value",
                                     "forecast_issue_date", "target_month"],
                         "order": "forecast_issue_date DESC"},
    "crisiswatch":      {"table": "crisiswatch_entries",
                         "columns": ["iso3", "year", "month", "arrow",
                                     "alert_type", "country_name"],
                         "order": "year DESC, month DESC"},
    # --- Weather and Climate ---
    "nmme":             {"table": "seasonal_forecasts",
                         "columns": ["iso3", "variable", "lead_months",
                                     "anomaly_value", "tercile_category",
                                     "forecast_issue_date"],
                         "order": "forecast_issue_date DESC"},
    "enso":             {"table": "enso_state", "has_iso3": False,
                         "columns": ["fetch_date", "enso_phase",
                                     "nino34_anomaly", "iod_phase"],
                         "order": "fetch_date DESC"},
    "seasonal_tc":      {"table": "seasonal_tc_outlooks", "has_iso3": False,
                         "columns": ["basin", "source", "forecast_season",
                                     "named_storms_forecast", "category",
                                     "fetched_at"],
                         "order": "fetched_at DESC"},
    "tc_context":       {"table": "seasonal_tc_context_cache",
                         "columns": ["iso3", "context_text"],
                         "order": "fetched_at DESC"},
    # --- Situation Reports ---
    "reliefweb":        {"table": "reliefweb_reports",
                         "columns": ["iso3", "title", "sources",
                                     "published_date", "url"],
                         "order": "published_date DESC",
                         "exclude_default": {"body_excerpt"}},
    "acaps_daily":      {"table": "acaps_daily_monitoring",
                         "columns": ["iso3", "entry_date",
                                     "latest_developments", "source"],
                         "order": "entry_date DESC"},
    "acled_political":  {"table": "acled_political_events",
                         "columns": ["iso3", "event_date", "event_type",
                                     "sub_event_type", "fatalities",
                                     "actor1", "location"],
                         "order": "event_date DESC"},
    # --- Other Alerts ---
    "hdx_signals":      {"table": "hdx_signals",
                         "columns": ["iso3", "hazard_code", "indicator",
                                     "concern_level", "indicator_value",
                                     "signal_date"],
                         "order": "signal_date DESC"},
    "acaps_risk_radar": {"table": "acaps_risk_radar",
                         "columns": ["iso3", "risk_title", "risk_level",
                                     "risk_type", "risk_trend"],
                         "order": "fetched_at DESC"},
    # --- Other ---
    "acaps_inform":     {"table": "acaps_inform_severity",
                         "columns": ["iso3", "crisis_name", "severity_score",
                                     "severity_category", "snapshot_date"],
                         "order": "snapshot_date DESC"},
    "acaps_access":     {"table": "acaps_humanitarian_access",
                         "columns": ["iso3", "access_score",
                                     "access_category", "snapshot_date"],
                         "order": "snapshot_date DESC"},
}

_SOURCE_LABELS: dict[str, str] = {
    "ifrc": "IFRC", "idmc": "IDMC", "acled": "ACLED",
    "gdacs": "GDACS", "fewsnet": "FEWS NET",
    "acled_fatalities": "ACLED Monthly Fatalities",
    "views": "VIEWS", "conflictforecast": "conflictforecast.org",
    "acled_cast": "ACLED CAST", "crisiswatch": "CrisisWatch",
    "nmme": "NMME Seasonal", "enso": "ENSO State",
    "seasonal_tc": "Seasonal TC Outlooks", "tc_context": "TC Context",
    "reliefweb": "ReliefWeb", "acaps_daily": "ACAPS Daily Monitoring",
    "acled_political": "ACLED Political Events",
    "hdx_signals": "HDX Signals", "acaps_risk_radar": "ACAPS Risk Radar",
    "acaps_inform": "ACAPS INFORM Severity",
    "acaps_access": "ACAPS Humanitarian Access",
    "ipc_api": "IPC API",
}

_SOURCE_CATEGORIES: dict[str, str] = {
    "ifrc": "resolution_data", "idmc": "resolution_data",
    "acled": "resolution_data", "gdacs": "resolution_data",
    "fewsnet": "resolution_data", "acled_fatalities": "resolution_data",
    "views": "conflict_forecasts", "conflictforecast": "conflict_forecasts",
    "acled_cast": "conflict_forecasts", "crisiswatch": "conflict_forecasts",
    "nmme": "weather_climate", "enso": "weather_climate",
    "seasonal_tc": "weather_climate", "tc_context": "weather_climate",
    "reliefweb": "situation_reports", "acaps_daily": "situation_reports",
    "acled_political": "situation_reports",
    "hdx_signals": "other_alerts", "acaps_risk_radar": "other_alerts",
    "acaps_inform": "other", "acaps_access": "other",
    "ipc_api": "resolution_data",
}


def _best_freshness_column(con, table: str) -> str | None:
    """Return the best column for 'last ingested' timestamp."""
    cols = _table_columns(con, table)
    for candidate in _FRESHNESS_CANDIDATES:
        if candidate in cols:
            return candidate
    return None


def _source_freshness(con, spec: dict) -> str | None:
    """Compute last-updated date for a source using ingestion-time columns."""
    table = spec["table"]
    if not _table_exists(con, table):
        return None
    ts_col = _best_freshness_column(con, table)
    if not ts_col:
        return None
    filt = spec.get("filter", "")
    sql = f"SELECT MAX({ts_col}) FROM {table}"
    if filt:
        sql += f" WHERE {filt}"
    try:
        val = con.execute(sql).fetchone()
        if val and val[0] is not None:
            return str(val[0])[:10]
    except Exception:
        pass
    return None


def _source_row_count(con, spec: dict, iso3: str | None = None) -> int:
    """Count rows for a source, optionally filtered by iso3."""
    table = spec["table"]
    if not _table_exists(con, table):
        return 0
    filt = spec.get("filter", "")
    clauses: list[str] = []
    if filt:
        clauses.append(filt)
    if iso3 and spec.get("has_iso3", True):
        cols = _table_columns(con, table)
        if "iso3" in cols:
            clauses.append(f"iso3 = '{iso3.upper()}'")
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    try:
        return con.execute(f"SELECT COUNT(*) FROM {table}{where}").fetchone()[0]
    except Exception:
        return 0


def _validated_columns(con, table: str, curated: list[str]) -> list[str]:
    """Return only curated columns that actually exist in the table schema."""
    actual = _table_columns(con, table)
    return [c for c in curated if c.lower() in actual]


@app.get("/v1/resolver/source_inventory")
def get_resolver_source_inventory(
    iso3: str | None = Query(None, description="ISO3 country code"),
):
    """Per-source metadata for the accordion data explorer."""
    con = _con()
    iso3_val = iso3.strip().upper() if iso3 else None
    sources = []
    for key, spec in _SOURCE_REGISTRY.items():
        has_iso3 = spec.get("has_iso3", True)
        table = spec["table"]
        exists = _table_exists(con, table)
        sources.append({
            "key": key,
            "label": _SOURCE_LABELS.get(key, key),
            "category": _SOURCE_CATEGORIES.get(key, "other"),
            "last_updated": _source_freshness(con, spec) if exists else None,
            "global_rows": _source_row_count(con, spec) if exists else 0,
            "country_rows": (
                _source_row_count(con, spec, iso3_val)
                if exists and iso3_val and has_iso3 else None
            ),
            "has_iso3": has_iso3,
        })
    return {"sources": sources}


@app.get("/v1/resolver/source_data")
def get_resolver_source_data(
    source: str = Query(..., description="Source key from registry"),
    iso3: str | None = Query(None, description="ISO3 country code"),
    limit: int = Query(500, description="Max rows", ge=1, le=5000),
    all_columns: bool = Query(False, description="SELECT * instead of curated"),
    include_body: bool = Query(False, description="Include body text (reliefweb)"),
):
    """Lazy-load rows for a single source. Called when accordion expands."""
    spec = _SOURCE_REGISTRY.get(source)
    if not spec:
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}")
    con = _con()
    table = spec["table"]
    if not _table_exists(con, table):
        return {"rows": [], "columns": []}

    actual_cols = _table_columns(con, table)

    if all_columns:
        exclude = set()
        if source == "reliefweb" and not include_body:
            exclude.add("body_excerpt")
        select_list = sorted(c for c in actual_cols if c not in exclude)
    else:
        select_list = _validated_columns(con, table, spec["columns"])
        if not select_list:
            select_list = sorted(actual_cols)

    select_sql = ", ".join(select_list)
    clauses: list[str] = []
    filt = spec.get("filter", "")
    if filt:
        clauses.append(filt)
    if iso3 and spec.get("has_iso3", True) and "iso3" in actual_cols:
        clauses.append(f"iso3 = '{iso3.strip().upper()}'")

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    order = spec.get("order", "")
    # Validate order columns exist
    if order:
        order_cols_valid = all(
            col.strip().lower().replace(" desc", "").replace(" asc", "") in actual_cols
            for col in order.split(",")
        )
        if not order_cols_valid:
            order = ""
    order_sql = f" ORDER BY {order}" if order else ""
    sql = f"SELECT {select_sql} FROM {table}{where}{order_sql} LIMIT {min(limit, 5000)}"

    try:
        rows = _rows_from_cursor(con.execute(sql))
    except Exception as exc:
        logger.warning("source_data query failed for %s: %s", source, exc)
        return {"rows": [], "columns": select_list}

    return {"rows": rows, "columns": select_list}


def _acquire_heavy() -> None:
    """Acquire the heavy-request semaphore or raise 503."""
    if not _HEAVY_REQUEST_SEMAPHORE.acquire(timeout=30):
        raise HTTPException(status_code=503, detail="Server busy, try again")


def _stream_csv(df: pd.DataFrame, filename: str):
    """Stream a DataFrame as CSV with chunked output, releasing the semaphore when done."""
    def _gen():
        try:
            yield df.iloc[:0].to_csv(index=False)
            for start in range(0, len(df), 500):
                yield df.iloc[start:start + 500].to_csv(index=False, header=False)
        finally:
            _HEAVY_REQUEST_SEMAPHORE.release()

    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    return StreamingResponse(_gen(), media_type="text/csv; charset=utf-8", headers=headers)


@app.get("/v1/downloads/forecasts.xlsx")
def download_forecasts_xlsx(include_test: bool = Query(False)):
    if find_spec("openpyxl") is None:
        logger.warning("openpyxl missing; falling back to CSV export")
        return RedirectResponse(url="/v1/downloads/forecasts.csv", status_code=307)

    _acquire_heavy()
    try:
        con = _con()
        try:
            df = build_forecast_spd_export(con, include_test=include_test)
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
    finally:
        _HEAVY_REQUEST_SEMAPHORE.release()


@app.get("/v1/downloads/forecasts.csv")
def download_forecasts_csv(include_test: bool = Query(False)):
    _acquire_heavy()
    try:
        con = _con()
        df = build_forecast_spd_export(con, include_test=include_test)
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build forecast download export")
        raise HTTPException(status_code=500, detail="Failed to build forecast download export") from exc
    return _stream_csv(df, "pythia_forecasts_export.csv")


@app.get("/v1/downloads/triage.csv")
def download_triage_csv(include_test: bool = Query(False)):
    _acquire_heavy()
    try:
        con = _con()
        df = build_triage_export(con, include_test=include_test)
        logger.info(
            "Triage download export rows=%s runs=%s iso3=%s",
            len(df),
            df["Run ID"].nunique(dropna=True),
            df["ISO3"].nunique(dropna=True),
        )
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build triage download export")
        raise HTTPException(status_code=500, detail="Failed to build triage download export") from exc
    return _stream_csv(df, "run_triage_results.csv")


@app.get("/v1/downloads/total_costs.csv")
def download_total_costs_csv(include_test: bool = Query(False)):
    _acquire_heavy()
    try:
        con = _con()
        tables = build_costs_total(con, include_test=include_test)
        df = _concat_cost_tables(tables)
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build total cost export")
        raise HTTPException(status_code=500, detail="Failed to build total cost export") from exc
    return _stream_csv(df, "total_costs.csv")


@app.get("/v1/downloads/monthly_costs.csv")
def download_monthly_costs_csv(include_test: bool = Query(False)):
    _acquire_heavy()
    try:
        con = _con()
        tables = build_costs_monthly(con, include_test=include_test)
        df = _concat_cost_tables(tables)
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build monthly cost export")
        raise HTTPException(status_code=500, detail="Failed to build monthly cost export") from exc
    return _stream_csv(df, "monthly_costs.csv")


@app.get("/v1/downloads/run_costs.csv")
def download_run_costs_csv(include_test: bool = Query(False)):
    _acquire_heavy()
    try:
        con = _con()
        tables = build_costs_runs(con, include_test=include_test)
        df = _concat_cost_tables(tables)
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build run cost export")
        raise HTTPException(status_code=500, detail="Failed to build run cost export") from exc
    return _stream_csv(df, "run_costs.csv")


@app.get("/v1/downloads/scores_ensemble_mean.csv")
def download_scores_ensemble_mean_csv(include_test: bool = Query(False)):
    _acquire_heavy()
    try:
        con = _con()
        df = build_ensemble_scores_export(con, "ensemble_mean", include_test=include_test)
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build ensemble_mean scores export")
        raise HTTPException(status_code=500, detail="Failed to build ensemble_mean scores export") from exc
    return _stream_csv(df, "scores_ensemble_mean.csv")


@app.get("/v1/downloads/scores_ensemble_bayesmc.csv")
def download_scores_ensemble_bayesmc_csv(include_test: bool = Query(False)):
    _acquire_heavy()
    try:
        con = _con()
        df = build_ensemble_scores_export(con, "ensemble_bayesmc", include_test=include_test)
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build ensemble_bayesmc scores export")
        raise HTTPException(status_code=500, detail="Failed to build ensemble_bayesmc scores export") from exc
    return _stream_csv(df, "scores_ensemble_bayesmc.csv")


@app.get("/v1/downloads/scores_model.csv")
def download_scores_model_csv(include_test: bool = Query(False)):
    _acquire_heavy()
    try:
        con = _con()
        df = build_model_scores_export(con, include_test=include_test)
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build model scores export")
        raise HTTPException(status_code=500, detail="Failed to build model scores export") from exc
    return _stream_csv(df, "scores_model.csv")


@app.get("/v1/downloads/rationales.csv")
def download_rationales_csv(
    hazard: str = Query(..., description="Hazard code filter (e.g. FL, DR, TC)"),
    model: str | None = Query(None, description="Model name filter (e.g. OpenAI, Claude, Gemini Flash)"),
    include_test: bool = Query(False),
):
    _acquire_heavy()
    try:
        con = _con()
        df = build_rationale_export(con, hazard_code=hazard, model_name=model, include_test=include_test)
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build rationale export")
        raise HTTPException(status_code=500, detail="Failed to build rationale export") from exc
    parts = ["rationales", hazard.strip().upper()]
    if model:
        safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model)
        parts.append(safe_model)
    return _stream_csv(df, "_".join(parts) + ".csv")


@app.get("/v1/llm/costs")
def llm_costs(
    component: str | None = Query(None),
    model: str | None = Query(None),
    since: str | None = Query(None),
    limit: int = Query(200, ge=1, le=5000),
    include_test: bool = Query(False),
):
    """
    Return recent LLM call cost/usage rows from llm_calls.

    Optional filters:
      - component: "HS" | "Researcher" | "Forecaster" | etc.
      - model: model_name (exact match)
      - since: ISO timestamp (created_at >= since)
    """
    con = _con()
    sql = f"SELECT * FROM llm_calls WHERE 1=1{_test_filter(include_test)}"
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

    return {"rows": _rows_from_cursor(_execute(con, sql, params))}


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
    include_test: bool = Query(False),
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
      WHERE {where_clause}{_test_filter(include_test)}
      {group_by_clause}
      ORDER BY cost_usd DESC NULLS LAST
      LIMIT :limit
    """
    params["limit"] = limit

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
        "rows": _rows_from_cursor(_execute(con, sql, params)),
    }


@app.get("/v1/costs/total")
def costs_total(track: Optional[int] = Query(None), include_test: bool = Query(False)):
    con = _con()
    try:
        tables = build_costs_total(con, track=track, include_test=include_test)
    except Exception as exc:
        logger.exception("Failed to build total costs")
        raise HTTPException(status_code=500, detail="Failed to build total costs") from exc

    return {"tables": {key: _rows_from_df(df) for key, df in tables.items()}}


@app.get("/v1/costs/monthly")
def costs_monthly(track: Optional[int] = Query(None), include_test: bool = Query(False)):
    con = _con()
    try:
        tables = build_costs_monthly(con, track=track, include_test=include_test)
    except Exception as exc:
        logger.exception("Failed to build monthly costs")
        raise HTTPException(status_code=500, detail="Failed to build monthly costs") from exc

    return {"tables": {key: _rows_from_df(df) for key, df in tables.items()}}


@app.get("/v1/costs/runs")
def costs_runs(track: Optional[int] = Query(None), include_test: bool = Query(False)):
    con = _con()
    try:
        tables = build_costs_runs(con, track=track, include_test=include_test)
    except Exception as exc:
        logger.exception("Failed to build run costs")
        raise HTTPException(status_code=500, detail="Failed to build run costs") from exc

    return {"tables": {key: _rows_from_df(df) for key, df in tables.items()}}


@app.get("/v1/costs/latencies")
def costs_latencies(track: Optional[int] = Query(None), include_test: bool = Query(False)):
    con = _con()
    try:
        df = build_latencies_runs(con, track=track, include_test=include_test)
    except Exception as exc:
        logger.exception("Failed to build run latencies")
        raise HTTPException(status_code=500, detail="Failed to build run latencies") from exc

    return {"rows": _rows_from_df(df)}


@app.get("/v1/costs/run_runtimes")
def costs_run_runtimes(track: Optional[int] = Query(None), include_test: bool = Query(False)):
    con = _con()
    try:
        df = build_run_runtimes(con, track=track, include_test=include_test)
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
