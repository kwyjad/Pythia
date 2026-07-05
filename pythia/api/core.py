# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Shared infrastructure for the Pythia API (July 2026 decomposition).

Everything here was moved verbatim from ``pythia.api.app`` so that the
route-group modules under ``pythia.api.routes`` can share it without
importing the FastAPI application module (which imports them). This module
owns the mutable connection-layer state (``_READ_CON`` and friends) — it is
the single authoritative copy; ``pythia.api.app`` forwards legacy reads and
writes of those names here via module-class properties.

The connection layer (``_ensure_read_connection`` / ``_maybe_refresh_db`` /
``_con``) encodes fixes for four documented failure modes (stale DB
connection after sync; flag-only gate unreliability; the force_sync blind
spot; the wedged temp path) — see "Known failure modes" in CLAUDE.md before
changing anything here.
"""

import json
import logging
import math
import os
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from pythia.api.db_sync import (
    DbSyncError,
    db_was_refreshed,
    maybe_sync_latest_db,
)
from pythia.buckets import BUCKET_SPECS
from pythia.config import load as load_cfg
from resolver.query.costs import COST_COLUMNS

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
_READ_CON_MTIME: Optional[float] = None  # DB file mtime when _READ_CON was opened
_LAST_SYNC_CHECK: Optional[float] = None
_SYNC_CHECK_INTERVAL_S = 60  # how often to poll for DB refresh


def _db_file_mtime() -> Optional[float]:
    """Return the mtime of the configured DuckDB file, or None if missing."""
    try:
        db_url = load_cfg()["app"]["db_url"].replace("duckdb:///", "")
        return os.path.getmtime(db_url)
    except (OSError, Exception):
        return None


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
        global _READ_CON_MTIME  # noqa: PLW0603
        _READ_CON_MTIME = _db_file_mtime()
        return _READ_CON


def _maybe_refresh_db() -> None:
    """Periodically check if the DB file was replaced and reconnect if so.

    ``maybe_sync_latest_db()`` downloads a new DuckDB file via atomic
    ``os.replace()``, but on Unix the existing open file descriptor still
    reads from the old inode.  This function detects that a new file was
    downloaded and reopens the connection so queries see the latest data.

    Reopen is triggered by EITHER of:

    * ``db_was_refreshed()`` — the consume-once flag set immediately after
      ``download_db_atomic``. Fast path for the worker that performed the
      download.
    * The DB file's on-disk mtime is newer than the mtime captured when the
      current connection was opened. Covers multi-worker deployments (each
      worker has its own ``_READ_CON`` but sees the same shared DB file)
      and any scenario where the flag was consumed by another code path
      before this worker could act on it.
    """
    global _READ_CON, _READ_CON_MTIME, _LAST_SYNC_CHECK  # noqa: PLW0603
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

    flag_refreshed = db_was_refreshed()
    current_mtime = _db_file_mtime()
    mtime_newer = (
        current_mtime is not None
        and _READ_CON_MTIME is not None
        and current_mtime > _READ_CON_MTIME
    )

    if flag_refreshed or mtime_newer:
        with _READ_CON_LOCK:
            old_con = _READ_CON
            try:
                _READ_CON = _open_duckdb_connection()
                _READ_CON_MTIME = _db_file_mtime()
            except Exception:
                logger.warning("Failed to reopen DuckDB after refresh; keeping old connection")
                return
            logger.info(
                "DuckDB connection reopened (flag=%s mtime_newer=%s)",
                flag_refreshed, mtime_newer,
            )
            try:
                if old_con is not None:
                    old_con.close()
            except Exception:
                pass


def _con():
    _maybe_refresh_db()
    return _ensure_read_connection().cursor()


def _require_debug_token(token: Optional[str]) -> None:
    expected = os.getenv("FRED_DEBUG_TOKEN")
    if not expected:
        raise HTTPException(status_code=404, detail="Not found")
    if token != expected:
        raise HTTPException(status_code=403, detail="Forbidden")


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



# ---------------------------------------------------------------------------
# Read-connection retry-once resilience (Task: audit/2026-07-s4).
#
# When the cached read connection dies mid-query (DB file replaced under us,
# connection invalidated or closed), a single reopen-and-retry rescues the
# request instead of surfacing a 500. This is deliberately narrow: ordinary
# query errors (CatalogException, BinderException, syntax errors, constraint
# violations, ...) are NEVER retried — they would fail identically on a fresh
# connection and a reopen would only mask the real bug.
# ---------------------------------------------------------------------------

_CONNECTION_RETRY_EXC_TYPES = (
    # "Connection Error: Connection has already been closed", etc.
    duckdb.ConnectionException,
    # "database has been invalidated because of a previous fatal error" —
    # by definition the connection is unusable until reopened.
    duckdb.FatalException,
)

# Connection-level failures that surface as a generic duckdb.Error or
# RuntimeError (older duckdb versions / wrapper layers) are matched by
# message. The markers are phrases that only appear in connection-level
# failures, never in ordinary query errors.
_CONNECTION_ERROR_MARKERS = (
    "database has been invalidated",
    "connection has already been closed",
    "connection error",
    "connection closed",
)


def _is_connection_level_error(exc: BaseException) -> bool:
    """Return True only for errors meaning the connection itself is unusable.

    Conservative by design: returns False for anything that looks like an
    ordinary query error, so callers never retry those.
    """
    if isinstance(exc, _CONNECTION_RETRY_EXC_TYPES):
        return True
    if isinstance(exc, (duckdb.Error, RuntimeError)):
        message = str(exc).lower()
        return any(marker in message for marker in _CONNECTION_ERROR_MARKERS)
    return False


def _reopen_read_connection() -> duckdb.DuckDBPyConnection:
    """Close the cached read connection, reopen from the same DB path.

    Returns a fresh cursor for the retry. Only called by ``_execute`` after
    ``_is_connection_level_error`` matched; ``_execute`` retries exactly once
    and never loops.
    """
    global _READ_CON, _READ_CON_MTIME  # noqa: PLW0603
    with _READ_CON_LOCK:
        old_con = _READ_CON
        _READ_CON = None
        _READ_CON_MTIME = None
        try:
            if old_con is not None:
                old_con.close()
        except Exception:
            pass
    return _ensure_read_connection().cursor()


def _execute_on(
    con: duckdb.DuckDBPyConnection, sql: str, params: Optional[Any] = None
) -> duckdb.DuckDBPyConnection:
    """Single execution attempt (the pre-retry body of ``_execute``)."""
    if params is None:
        return con.execute(sql)
    if isinstance(params, dict):
        compiled_sql, args = _compile_named_params(sql, params)
        return con.execute(compiled_sql, args)
    return con.execute(sql, params)


def _execute(
    con: duckdb.DuckDBPyConnection, sql: str, params: Optional[Any] = None
) -> duckdb.DuckDBPyConnection:
    if not hasattr(con, "execute"):
        raise TypeError(f"_execute expected a DuckDB connection, got {type(con)}")
    try:
        return _execute_on(con, sql, params)
    except Exception as exc:  # filtered immediately below; non-matching re-raise
        if not _is_connection_level_error(exc):
            raise
        logger.warning(
            "Connection-level DuckDB error during query (%s: %s); reopening "
            "cached read connection and retrying once",
            type(exc).__name__,
            exc,
        )
        retry_con = _reopen_read_connection()
        return _execute_on(retry_con, sql, params)


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


def _count_distinct_active_questions(
    con: duckdb.DuckDBPyConnection, table: str, include_test: bool
) -> int:
    """COUNT(DISTINCT question_id) in `table`, JOINed with `questions` so that
    retired (and optionally test) questions are excluded. Used by the
    /v1/diagnostics/summary endpoint."""
    if not _table_exists(con, table) or not _table_exists(con, "questions"):
        return 0
    if "question_id" not in _table_columns(con, table):
        return 0
    test_clause = "" if include_test else " AND COALESCE(q.is_test, FALSE) = FALSE"
    try:
        row = con.execute(
            f"""
            SELECT COUNT(DISTINCT t.question_id)
            FROM {table} t
            JOIN questions q ON q.question_id = t.question_id
            WHERE COALESCE(q.status, '') != 'retired'{test_clause}
            """
        ).fetchone()
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
    # NOTE(code-motion): this function historically lived in pythia.api.app,
    # and pythia/tests/test_api_input_validation.py monkeypatches
    # pythia.api.app._table_has_columns around it. Resolve the helper late
    # through that module (when loaded) so the patch seam keeps working;
    # otherwise fall back to this module's implementation.
    _app_module = sys.modules.get("pythia.api.app")
    _table_has_columns_impl = (
        getattr(_app_module, "_table_has_columns", _table_has_columns)
        if _app_module is not None
        else _table_has_columns
    )
    if not _table_has_columns_impl(con, "forecasts_ensemble", ["run_id"]):
        return ("", "")
    if forecaster_run_id:
        # The run id is interpolated into a CTE that is embedded in larger
        # queries, so it must be validated here — this is a security boundary
        # for the user-supplied forecaster_run_id query param.
        if not re.match(r"^[A-Za-z0-9_.:\-]{1,128}$", forecaster_run_id):
            raise HTTPException(status_code=400, detail="Invalid forecaster_run_id")
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


_ISO3_PATTERN = re.compile(r"^[A-Za-z]{3}$")


def _validate_iso3_param(iso3: str | None) -> str | None:
    """Normalize an iso3 query param; reject anything that is not 3 letters.

    The value is interpolated into SQL filters downstream, so this is a
    security boundary, not just input hygiene.
    """
    if iso3 is None:
        return None
    val = iso3.strip().upper()
    if not val:
        return None
    if not _ISO3_PATTERN.match(val):
        raise HTTPException(status_code=400, detail="Invalid iso3 parameter")
    return val


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
