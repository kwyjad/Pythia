"""Helpers for connecting to and writing into the DuckDB backend."""

from __future__ import annotations

import json
import os
import re
import uuid
import datetime as dt
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - import guard for optional dependency
    import duckdb
except ImportError as exc:  # pragma: no cover - guidance for operators
    raise RuntimeError(
        "DuckDB is required for database-backed resolver operations. Install 'duckdb'."
    ) from exc

from resolver.common import compute_series_semantics, dict_counts, df_schema
from resolver.db.conn_shared import get_shared_duckdb_conn

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "db" / "schema.sql"

FACTS_RESOLVED_KEY_COLUMNS = [
    "ym",
    "iso3",
    "hazard_code",
    "metric",
    "series_semantics",
]
FACTS_DELTAS_KEY_COLUMNS = [
    "ym",
    "iso3",
    "hazard_code",
    "metric",
]
FACTS_RESOLVED_KEY = FACTS_RESOLVED_KEY_COLUMNS  # Backwards compatibility
FACTS_DELTAS_KEY = FACTS_DELTAS_KEY_COLUMNS
DATE_STRING_COLUMNS: dict[str, tuple[str, ...]] = {
    "facts_resolved": ("as_of_date", "publication_date"),
    "facts_deltas": ("as_of",),
}
DEFAULT_DB_URL = os.environ.get(
    "RESOLVER_DB_URL", f"duckdb:///{ROOT / 'db' / 'resolver.duckdb'}"
)

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:  # pragma: no cover - avoid "No handler" warnings in tests
    LOGGER.addHandler(logging.NullHandler())
DEBUG_ENABLED = os.getenv("RESOLVER_DEBUG") == "1"
if DEBUG_ENABLED:
    LOGGER.setLevel(logging.DEBUG)


def _merge_enabled() -> bool:
    """Return ``True`` when DuckDB MERGE statements should be attempted."""

    flag = os.getenv("RESOLVER_DUCKDB_DISABLE_MERGE", "").strip()
    return flag not in {"1", "true", "True"}


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


_YM_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")

_SEM_MAP_STOCK = {"stock", "stock estimate", "snapshot", "inventory"}
_SEM_MAP_NEW = {"new", "delta", "deltas", "increment", "change"}


def _normalize_keys_df(frame: pd.DataFrame | None, table_name: str) -> pd.DataFrame | None:
    """Normalize key columns prior to persistence."""

    if frame is None or frame.empty:
        return frame
    normalised = frame.copy()
    if "ym" in normalised.columns:
        normalised["ym"] = normalised["ym"].astype(str).str.strip()
        bad = ~normalised["ym"].fillna("").str.match(_YM_RE)
        if bad.any():
            raise ValueError(
                f"{table_name}: invalid ym format; expected YYYY-MM, got {sorted(normalised.loc[bad, 'ym'].unique())[:3]}"
            )
    if "iso3" in normalised.columns:
        normalised["iso3"] = normalised["iso3"].astype(str).str.strip().str.upper()
    if "hazard_code" in normalised.columns:
        normalised["hazard_code"] = (
            normalised["hazard_code"].astype(str).str.strip().str.upper()
        )
    return normalised


def _canonicalize_semantics(
    frame: pd.DataFrame | None, table_name: str, default_target: str
) -> tuple[pd.DataFrame | None, dict[str, dict[str, int]]]:
    """Canonicalize ``series_semantics`` values to ``{'new', 'stock'}``."""

    if frame is None or frame.empty or "series_semantics" not in frame.columns:
        return frame, {}
    canonical = frame.copy()
    raw = canonical["series_semantics"].astype(str).fillna("").str.strip()
    lowered = raw.str.lower()
    out: list[str] = []
    before_counts = raw.value_counts(dropna=False).to_dict()
    for value in lowered:
        if value in _SEM_MAP_STOCK:
            out.append("stock")
        elif value in _SEM_MAP_NEW:
            out.append("new")
        elif value in {"", "none", "null", "nan"}:
            out.append(default_target)
        else:
            out.append(default_target)
    canonical["series_semantics"] = out
    after_counts = canonical["series_semantics"].value_counts(dropna=False).to_dict()
    if DEBUG_ENABLED and LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "canonicalized semantics (%s): %s -> %s",
            table_name,
            before_counts,
            after_counts,
        )
    return canonical, {"before": before_counts, "after": after_counts}


def _canonicalise_series_semantics(series: pd.Series) -> pd.Series:
    """Return ``series`` mapped into the canonical {"", "new", "stock"}."""

    semantics = series.where(series.notna(), "")
    semantics = semantics.astype(str).str.strip()
    lowered = semantics.str.lower()
    semantics = semantics.mask(lowered.isin({"none", "nan"}), "")
    semantics = semantics.mask(lowered.isin(_SEM_MAP_NEW), "new")
    semantics = semantics.mask(lowered.isin(_SEM_MAP_STOCK), "stock")
    lowered = semantics.astype(str).str.lower()
    semantics = semantics.mask(~lowered.isin({"", "new", "stock"}), "")
    return semantics.astype(str)


def _assert_semantics_required(frame: pd.DataFrame, table: str) -> None:
    if frame is None or frame.empty or "series_semantics" not in frame.columns:
        return
    values = (
        frame["series_semantics"].astype(str).str.strip().str.lower().unique().tolist()
    )
    if not set(values).issubset({"new", "stock"}):
        raise ValueError(
            f"{table}: series_semantics must be 'new' or 'stock', got {sorted(set(values))}"
        )


def _coerce_numeric_cols(
    frame: pd.DataFrame | None, cols: Sequence[str], table_name: str
) -> pd.DataFrame | None:
    """Coerce known numeric columns to floats, mapping placeholder strings to NULL."""

    if frame is None or frame.empty:
        return frame

    coerced = frame.copy()
    for col in cols:
        if col not in coerced.columns:
            continue
        series = coerced[col]
        if pd.api.types.is_numeric_dtype(series):
            coerced[col] = pd.to_numeric(series, errors="coerce")
            continue
        stringified = series.astype(str).str.strip()
        lowered = stringified.str.lower()
        stringified = stringified.mask(
            lowered.isin({"", "none", "null", "nan"}), np.nan
        )
        coerced[col] = pd.to_numeric(stringified, errors="coerce")
    if DEBUG_ENABLED and LOGGER.isEnabledFor(logging.DEBUG):
        present = [col for col in cols if col in coerced.columns]
        LOGGER.debug(
            "duckdb.numeric_coercion | table=%s columns=%s", table_name, present
        )
    return coerced


def _normalise_iso_date_strings(frame: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Cast ``columns`` to ISO ``YYYY-MM-DD`` strings when present in ``frame``."""

    normalised: list[str] = []
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column]
        formatted: pd.Series
        if pd.api.types.is_datetime64_any_dtype(series):
            formatted = series.dt.strftime("%Y-%m-%d")
        else:
            parsed = pd.to_datetime(series, errors="coerce")
            formatted = parsed.dt.strftime("%Y-%m-%d")
            # Preserve original strings where parsing failed but coerce missing to empty
            fallback = series.fillna("").astype(str)
            fallback = fallback.replace({"NaT": "", "<NA>": "", "nan": "", "NaN": ""})
            mask = parsed.notna()
            formatted = formatted.where(mask, fallback)
        formatted = formatted.fillna("").replace({"NaT": "", "<NA>": "", "nan": "", "NaN": ""})
        frame[column] = formatted.astype(str)
        normalised.append(column)
    return normalised


def _constraint_column_sets(
    conn: "duckdb.DuckDBPyConnection", table: str
) -> list[list[str]]:
    """Return column name sets for PK and UNIQUE constraints on ``table``."""

    constraints: list[list[str]] = []

    try:
        table_info = conn.execute(
            f"PRAGMA table_info({_quote_literal(table)})"
        ).fetchall()
    except Exception:  # pragma: no cover - defensive logging helper
        LOGGER.debug("Constraint inspection failed via PRAGMA table_info", exc_info=True)
        table_info = []

    pk_columns: list[str] = []
    for row in table_info:
        try:
            is_pk = bool(row[5])
        except IndexError:
            is_pk = False
        if is_pk:
            pk_columns.append(str(row[1]))
    if pk_columns:
        constraints.append(pk_columns)

    try:
        unique_rows = conn.execute(
            """
            SELECT LIST(column_name ORDER BY ordinal_position) AS column_names
            FROM information_schema.key_column_usage
            WHERE table_schema = current_schema()
              AND table_name = ?
              AND constraint_name IN (
                    SELECT constraint_name
                    FROM information_schema.table_constraints
                    WHERE table_schema = current_schema()
                      AND table_name = ?
                      AND constraint_type = 'UNIQUE'
              )
            GROUP BY constraint_name
            """,
            [table, table],
        ).fetchall()
    except Exception:  # pragma: no cover - diagnostics only
        LOGGER.debug("Constraint inspection failed via information_schema", exc_info=True)
        unique_rows = []

    for (columns,) in unique_rows:
        if not columns:
            continue
        constraint_columns = [str(column) for column in columns]
        if constraint_columns:
            constraints.append(constraint_columns)
    return constraints


def _has_declared_key(
    conn: "duckdb.DuckDBPyConnection", table: str, keys: Sequence[str] | None
) -> bool:
    if not keys:
        return False
    expected = [key.lower() for key in keys]
    for constraint_columns in _constraint_column_sets(conn, table):
        if [column.lower() for column in constraint_columns] == expected:
            return True
    return False


def _delete_where(
    conn: "duckdb.DuckDBPyConnection",
    table: str,
    where_sql: str,
    params: Sequence[object],
) -> int:
    """Delete rows from ``table`` matching ``where_sql`` and return the count."""

    table_ident = _quote_identifier(table)
    base_delete = f"DELETE FROM {table_ident} WHERE {where_sql}"
    try:
        rows = conn.execute(f"{base_delete} RETURNING 1", params).fetchall()
        return len(rows)
    except duckdb.Error:
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table_ident} WHERE {where_sql}", params
        ).fetchone()[0]
        conn.execute(base_delete, params)
        return int(count or 0)


def _normalise_db_url(path_or_url: str | None) -> str:
    if not path_or_url:
        return DEFAULT_DB_URL
    if path_or_url.startswith("duckdb://"):
        return path_or_url
    if path_or_url.startswith(":memory:"):
        return f"duckdb:///{path_or_url}"
    path = Path(path_or_url)
    return f"duckdb:///{path}" if not path_or_url.startswith("duckdb:") else path_or_url


def get_db(path_or_url: str | None = None) -> "duckdb.DuckDBPyConnection":
    """Return a DuckDB connection for the given path or URL."""

    url = _normalise_db_url(path_or_url or os.environ.get("RESOLVER_DB_URL"))
    conn, resolved_path = get_shared_duckdb_conn(url)
    if os.getenv("RESOLVER_DEBUG") == "1" and LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "DuckDB connection resolved: path=%s from=%s cache_disabled=%s",
            resolved_path,
            url,
            os.getenv("RESOLVER_DISABLE_CONN_CACHE") == "1",
        )
    try:
        conn.execute("PRAGMA threads=4")
        conn.execute("PRAGMA enable_progress_bar=false")
        return conn
    except duckdb.ConnectionException as exc:
        LOGGER.debug(
            "DuckDB connection unhealthy for %s (%s); forcing reopen", resolved_path, exc
        )
        conn, resolved_path = get_shared_duckdb_conn(url, force_reopen=True)
        conn.execute("PRAGMA threads=4")
        conn.execute("PRAGMA enable_progress_bar=false")
        return conn


def init_schema(
    conn: "duckdb.DuckDBPyConnection", schema_sql_path: Path | None = None
) -> None:
    """Initialise database schema if it does not already exist."""

    schema_path = schema_sql_path or SCHEMA_PATH
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema SQL not found at {schema_path}")

    expected_tables = {
        "facts_resolved",
        "facts_deltas",
        "manifests",
        "meta_runs",
        "snapshots",
    }
    existing_tables = {
        row[0]
        for row in conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name IN ('facts_resolved','facts_deltas','manifests','meta_runs','snapshots')
            """
        ).fetchall()
    }

    if expected_tables.issubset(existing_tables):
        LOGGER.debug("DuckDB schema already initialised; skipping DDL execution")
        return

    sql = schema_path.read_text(encoding="utf-8")
    for statement in [s.strip() for s in sql.split(";") if s.strip()]:
        conn.execute(statement)
    LOGGER.debug("Ensured DuckDB schema from %s", schema_path)
    if LOGGER.isEnabledFor(logging.DEBUG):
        table_details = conn.execute(
            """
            SELECT table_name, column_name, ordinal_position
            FROM information_schema.columns
            WHERE table_schema = 'main'
              AND table_name IN ('facts_resolved','facts_deltas','manifests','meta_runs','snapshots')
            ORDER BY table_name, ordinal_position
            """
        ).fetchall()
        current_table = None
        columns: list[str] = []
        for table_name, column_name, _ in table_details:
            if table_name != current_table:
                if current_table is not None:
                    LOGGER.debug(
                        "Table %s columns: %s", current_table, ", ".join(columns)
                    )
                current_table = table_name
                columns = []
            columns.append(column_name)
        if current_table is not None:
            LOGGER.debug("Table %s columns: %s", current_table, ", ".join(columns))


def upsert_dataframe(
    conn: "duckdb.DuckDBPyConnection",
    table: str,
    df: pd.DataFrame,
    keys: Sequence[str] | None = None,
) -> int:
    """Upsert rows into ``table`` using ``keys`` as the natural key.

    Preferred call signature::

        upsert_dataframe(conn, table: str, df: pd.DataFrame, keys=...)

    For backwards compatibility, the legacy order with the dataframe and
    table name swapped is also accepted::

        upsert_dataframe(conn, df: pd.DataFrame, table: str, keys=...)

    Any other argument ordering will raise ``TypeError``.
    """

    if isinstance(table, pd.DataFrame) and isinstance(df, str):
        LOGGER.debug(
            "duckdb.upsert.argorder_swapped | accepted call signature (conn, df, table, keys)"
        )
        table, df = df, table

    if not isinstance(table, str) or not isinstance(df, pd.DataFrame):
        raise TypeError(
            "upsert_dataframe expects (conn, table: str, df: pandas.DataFrame, keys=...) or "
            "(conn, df: pandas.DataFrame, table: str, keys=...)"
        )

    merge_sql = ""
    insert_sql = ""
    delete_sql = ""
    diag_join_count_sql = ""

    try:
        _source_frame = df
        _sample = (
            _source_frame.iloc[0].to_dict() if _source_frame is not None and len(_source_frame) else {}
        )
        _insert_cols_preview = (
            list(_source_frame.columns) if _source_frame is not None else None
        )
        LOGGER.debug(
            "UPSERT starting | table=%s | keys=%s | insert_columns=%s | sample_row=%s",
            table,
            keys,
            _insert_cols_preview,
            json.dumps(_sample, default=str),
        )
    except Exception as _e:  # pragma: no cover - logging helper only
        LOGGER.debug("UPSERT debug prelude failed: %s", _e)

    if df is None or df.empty:
        return 0

    import duckdb  # keep inside function to avoid module-top import churn

    frame = df.copy()
    if table == "facts_resolved":
        coerced = _coerce_numeric_cols(frame, ["value"], table)
        if coerced is not None:
            frame = coerced
    elif table == "facts_deltas":
        coerced = _coerce_numeric_cols(frame, ["value_new", "value_stock"], table)
        if coerced is not None:
            frame = coerced
    LOGGER.info("Upserting %s rows into %s", len(frame), table)
    LOGGER.debug("Incoming frame schema: %s", df_schema(frame))

    if "series_semantics_out" in frame.columns:
        semantics_out = frame["series_semantics_out"].where(
            frame["series_semantics_out"].notna(), ""
        ).astype(str)
        if "series_semantics" in frame.columns:
            semantics_current_raw = frame["series_semantics"]
            semantics_current = semantics_current_raw.astype(str)
            prefer_out = semantics_current_raw.isna() | semantics_current.str.strip().eq("")
            frame.loc[prefer_out, "series_semantics"] = semantics_out.loc[prefer_out]
        else:
            frame["series_semantics"] = semantics_out
        frame = frame.drop(columns=["series_semantics_out"])

    if "series_semantics" not in frame.columns:
        frame["series_semantics"] = ""

    frame["series_semantics"] = _canonicalise_series_semantics(
        frame["series_semantics"]
    )

    table_info = conn.execute(f"PRAGMA table_info({_quote_literal(table)})").fetchall()
    if not table_info:
        raise ValueError(f"Table '{table}' does not exist in DuckDB database")
    table_columns = [row[1] for row in table_info]
    has_declared_key = _has_declared_key(conn, table, keys) if keys else False
    if keys and not has_declared_key:
        raise ValueError(
            f"Declared upsert keys {list(keys)} for table '{table}' do not match a primary "
            "key or unique constraint."
        )

    insert_columns = [col for col in table_columns if col in frame.columns]
    dropped = [col for col in frame.columns if col not in table_columns]
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Table %s columns: %s", table, ", ".join(table_columns))
        LOGGER.debug("Insert columns: %s", ", ".join(insert_columns))
        if dropped:
            LOGGER.debug(
                "Dropping columns not present in %s: %s (expected for staging-only fields)",
                table,
                ", ".join(dropped),
            )
    if not insert_columns:
        raise ValueError(f"No matching columns to insert into '{table}'")

    frame = frame.loc[:, insert_columns].copy()

    normalised_dates = []
    if table in DATE_STRING_COLUMNS:
        normalised_dates = _normalise_iso_date_strings(
            frame, DATE_STRING_COLUMNS[table]
        )
        if normalised_dates and LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "Normalized date columns to ISO strings: %s",
                ", ".join(normalised_dates),
            )

    if keys:
        missing_keys = [k for k in keys if k not in frame.columns]
        if missing_keys:
            raise KeyError(
                f"Upsert keys {missing_keys} are missing from dataframe for table '{table}'"
            )
        for key in keys:
            frame[key] = frame[key].where(frame[key].notna(), "").astype(str).str.strip()
        before = len(frame)
        frame = frame.drop_duplicates(subset=list(keys), keep="last").reset_index(drop=True)
        if LOGGER.isEnabledFor(logging.DEBUG) and before != len(frame):
            LOGGER.debug(
                "Dropped %s duplicate rows for %s based on keys %s",
                before - len(frame),
                table,
                keys,
            )

    object_columns = frame.select_dtypes(include=["object"]).columns
    for column in object_columns:
        frame[column] = frame[column].astype(str)

    if frame.empty:
        LOGGER.debug(
            "duckdb.upsert.no_rows | table=%s | keys=%s", table, keys
        )
        return 0

    temp_name = f"tmp_{uuid.uuid4().hex}"
    conn.register(temp_name, frame)
    upsert_completed = False
    try:
        table_ident = _quote_identifier(table)
        temp_ident = _quote_identifier(temp_name)
        if keys and LOGGER.isEnabledFor(logging.DEBUG):
            diag_join_pred = " AND ".join(
                f"t.{_quote_identifier(k)} = s.{_quote_identifier(k)}" for k in keys
            )
            diag_join_count_sql = (
                "\n".join(
                    [
                        "SELECT COUNT(*) AS match_rows",
                        f"FROM {table_ident} t",
                        f"JOIN {temp_ident} s",
                        f"  ON {diag_join_pred}",
                    ]
                )
            )
            LOGGER.debug("DIAG join-count SQL:\n%s", diag_join_count_sql)
            try:
                match_rows = conn.execute(diag_join_count_sql).fetchone()[0]
            except Exception:
                match_rows = None
                LOGGER.debug("DIAG join-count execution failed", exc_info=True)
            else:
                LOGGER.debug(
                    "DIAG matched %s existing rows in %s using keys %s",
                    match_rows,
                    table,
                    "[" + ", ".join(keys) + "]",
                )

        use_legacy_path = True
        if keys and has_declared_key:
            if not _merge_enabled():
                LOGGER.debug(
                    "MERGE disabled via RESOLVER_DUCKDB_DISABLE_MERGE for table %s",
                    table,
                )
            else:
                all_columns = insert_columns
                non_key_columns = [c for c in all_columns if c not in keys]
                on_predicate = " AND ".join(
                    f"t.{_quote_identifier(k)} = s.{_quote_identifier(k)}" for k in keys
                )
                update_assignments = (
                    ", ".join(
                        f"{_quote_identifier(c)} = s.{_quote_identifier(c)}"
                        for c in non_key_columns
                    )
                    if non_key_columns
                    else ""
                )
                insert_cols_sql = ", ".join(_quote_identifier(c) for c in all_columns)
                select_cols_sql = ", ".join(
                    f"s.{_quote_identifier(c)}" for c in all_columns
                )

                merge_parts = [
                    f"MERGE INTO {table_ident} AS t",
                    f"USING {temp_ident} AS s",
                    f"ON {on_predicate}",
                ]
                if update_assignments:
                    merge_parts.append(
                        f"WHEN MATCHED THEN UPDATE SET {update_assignments}"
                    )
                merge_parts.append(
                    f"WHEN NOT MATCHED THEN INSERT ({insert_cols_sql}) VALUES ({select_cols_sql})"
                )
                merge_sql = "\n".join(merge_parts) + ";"

                LOGGER.debug("MERGE SQL:\n%s", merge_sql)
                try:
                    conn.execute(merge_sql)
                except duckdb.Error as exc:
                    use_legacy_path = True
                    LOGGER.warning(
                        "duckdb.merge_failed | table=%s | rows=%s | error=%s",
                        table,
                        len(frame),
                        exc,
                        exc_info=LOGGER.isEnabledFor(logging.DEBUG),
                    )
                else:
                    LOGGER.info("Upserted %s rows into %s via MERGE", len(frame), table)
                    use_legacy_path = False
                    upsert_completed = True

        if use_legacy_path:
            if keys and has_declared_key and merge_sql:
                LOGGER.debug(
                    "Falling back to legacy delete+insert after MERGE failure for table %s",
                    table,
                )
            if keys:
                delete_predicate = " AND ".join(
                    f"t.{_quote_identifier(k)} = s.{_quote_identifier(k)}" for k in keys
                )
                delete_sql = (
                    "\n".join(
                        [
                            f"DELETE FROM {table_ident} AS t",
                            "WHERE EXISTS (",
                            f"    SELECT 1 FROM {temp_ident} AS s",
                            f"    WHERE {delete_predicate}",
                            ")",
                        ]
                    )
                    + ";"
                )
                LOGGER.debug("DELETE SQL:\n%s", delete_sql)
                try:
                    deleted_rows = conn.execute(
                        f"{delete_sql[:-1]} RETURNING 1"
                    ).fetchall()
                    deleted_count = len(deleted_rows)
                except duckdb.Error:
                    LOGGER.debug(
                        "DELETE ... RETURNING failed; retrying without RETURNING",
                        exc_info=True,
                    )
                    count_sql = (
                        "\n".join(
                            [
                                "SELECT COUNT(*)",
                                f"FROM {table_ident} AS t",
                                f"JOIN {temp_ident} AS s",
                                f"  ON {delete_predicate}",
                            ]
                        )
                    )
                    deleted_count = conn.execute(count_sql).fetchone()[0]
                    conn.execute(delete_sql)
                else:
                    deleted_count = int(deleted_count)
                LOGGER.info(
                    "duckdb.upsert.legacy_delete | Deleted %s existing rows from %s using keys %s",
                    int(deleted_count or 0),
                    table,
                    "[" + ", ".join(keys or []) + "]",
                )

            cols_csv = ", ".join(_quote_identifier(col) for col in insert_columns)
            insert_sql = (
                f"INSERT INTO {table_ident} ({cols_csv}) SELECT {cols_csv} FROM {temp_ident}"
            )
            LOGGER.debug("INSERT SQL:\n%s", insert_sql)
            conn.execute(insert_sql)
            LOGGER.info(
                "duckdb.upsert.legacy_insert | Inserted %s rows into %s",
                len(frame),
                table,
            )
            upsert_completed = True
    except Exception as e:
        if upsert_completed:
            LOGGER.debug(
                "duckdb.upsert.completed_with_followup_error | table=%s", table, exc_info=True
            )
        else:
            try:
                LOGGER.error(
                    (
                        "DuckDB upsert failed for table %s.\n"
                        "MERGE SQL:\n%s\nDELETE SQL:\n%s\nINSERT SQL:\n%s\n"
                        "JOIN-COUNT SQL:\n%s\nschema=%s | first_row=%s"
                    ),
                    table,
                    merge_sql or "<unset>",
                    delete_sql or "<unset>",
                    insert_sql or "<unset>",
                    diag_join_count_sql or "<unset>",
                    table_columns,
                    frame.head(1).to_dict(orient="records"),
                    exc_info=True,
                )
            except Exception:  # pragma: no cover - logging failure should not mask error
                LOGGER.exception("DIAG: assembling failure log failed")
            raise e
    finally:
        conn.unregister(temp_name)

    LOGGER.info("Processed %s rows for %s", len(frame), table)
    return len(frame)


def _default_created_at(value: str | None = None) -> str:
    if value:
        return value
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = frame.copy()
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df


def write_snapshot(
    conn: "duckdb.DuckDBPyConnection",
    *,
    ym: str,
    facts_resolved: pd.DataFrame | None,
    facts_deltas: pd.DataFrame | None,
    manifests: Iterable[Mapping[str, object]] | None,
    meta: Mapping[str, object] | None,
) -> None:
    """Write a snapshot bundle transactionally into the database."""

    facts_resolved = facts_resolved.copy() if facts_resolved is not None else None
    facts_deltas = facts_deltas.copy() if facts_deltas is not None else None

    facts_resolved = _normalize_keys_df(facts_resolved, "facts_resolved")
    facts_deltas = _normalize_keys_df(facts_deltas, "facts_deltas")

    conn.execute("BEGIN TRANSACTION")
    try:
        facts_rows = 0
        deltas_rows = 0

        deleted_resolved = _delete_where(conn, "facts_resolved", "ym = ?", [ym])
        LOGGER.debug("Deleted %s facts_resolved rows for ym=%s", deleted_resolved, ym)

        if facts_resolved is not None and not facts_resolved.empty:
            facts_resolved = _ensure_columns(
                facts_resolved,
                FACTS_RESOLVED_KEY_COLUMNS + ["value"],
            )
            for key in FACTS_RESOLVED_KEY_COLUMNS:
                series = (
                    facts_resolved[key]
                    .where(facts_resolved[key].notna(), "")
                    .astype(str)
                    .str.strip()
                )
                if key == "ym":
                    series = series.replace("", ym)
                facts_resolved[key] = series
            computed_semantics = facts_resolved.apply(
                lambda row: compute_series_semantics(
                    metric=row.get("metric"), existing=row.get("series_semantics")
                ),
                axis=1,
            )
            facts_resolved["series_semantics"] = computed_semantics
            facts_resolved, _ = _canonicalize_semantics(
                facts_resolved,
                "facts_resolved",
                default_target="stock",
            )
            _assert_semantics_required(facts_resolved, "facts_resolved")
            facts_resolved = _coerce_numeric_cols(
                facts_resolved, ["value"], "facts_resolved"
            )
            facts_resolved = facts_resolved.drop_duplicates(
                subset=FACTS_RESOLVED_KEY_COLUMNS,
                keep="last",
            ).reset_index(drop=True)
            facts_rows = upsert_dataframe(
                conn,
                "facts_resolved",
                facts_resolved,
                keys=FACTS_RESOLVED_KEY_COLUMNS,
            )
            LOGGER.info("facts_resolved rows upserted: %s", facts_rows)
            LOGGER.debug(
                "facts_resolved series_semantics distribution: %s",
                dict_counts(facts_resolved["series_semantics"]),
            )

        deleted_deltas = 0
        if facts_deltas is not None and not facts_deltas.empty:
            deleted_deltas = _delete_where(conn, "facts_deltas", "ym = ?", [ym])
            LOGGER.debug("Deleted %s facts_deltas rows for ym=%s", deleted_deltas, ym)
            facts_deltas = _ensure_columns(
                facts_deltas,
                FACTS_DELTAS_KEY_COLUMNS
                + ["series_semantics", "value_new", "value_stock"],
            )
            for key in FACTS_DELTAS_KEY_COLUMNS:
                series = (
                    facts_deltas[key]
                    .where(facts_deltas[key].notna(), "")
                    .astype(str)
                    .str.strip()
                )
                if key == "ym":
                    series = series.replace("", ym)
                facts_deltas[key] = series
            if "series_semantics" not in facts_deltas.columns:
                facts_deltas["series_semantics"] = "new"
            facts_deltas, _ = _canonicalize_semantics(
                facts_deltas,
                "facts_deltas",
                default_target="new",
            )
            _assert_semantics_required(facts_deltas, "facts_deltas")
            facts_deltas = _coerce_numeric_cols(
                facts_deltas, ["value_new", "value_stock"], "facts_deltas"
            )
            facts_deltas = facts_deltas.drop_duplicates(
                subset=FACTS_DELTAS_KEY_COLUMNS,
                keep="last",
            ).reset_index(drop=True)
            deltas_rows = upsert_dataframe(
                conn,
                "facts_deltas",
                facts_deltas,
                keys=FACTS_DELTAS_KEY_COLUMNS,
            )
            LOGGER.info("facts_deltas rows upserted: %s", deltas_rows)
            LOGGER.debug(
                "facts_deltas series_semantics distribution: %s",
                dict_counts(facts_deltas["series_semantics"]),
            )
        else:
            deleted_deltas = _delete_where(conn, "facts_deltas", "ym = ?", [ym])
            if deleted_deltas:
                LOGGER.debug(
                    "Deleted %s facts_deltas rows for ym=%s (no deltas frame provided)",
                    deleted_deltas,
                    ym,
                )
        manifest_rows: list[dict] = []
        manifest_created_at = _default_created_at(meta.get("created_at_utc") if meta else None)
        meta_schema_version = str(meta.get("schema_version", "")) if meta else ""
        meta_source_id = str(meta.get("source_id", "")) if meta else ""
        meta_sha256 = str(meta.get("sha256", "")) if meta else ""
        if manifests:
            for entry in manifests:
                payload = dict(entry)
                name = str(payload.get("name") or payload.get("path") or "artifact")
                raw_path = str(payload.get("path") or "").strip()
                if not raw_path:
                    raw_path = f"{ym}/{name}"
                manifest_rows.append(
                    {
                        "path": raw_path,
                        "sha256": str(
                            payload.get("checksum")
                            or payload.get("sha256")
                            or meta_sha256
                        ),
                        "row_count": int(payload.get("rows") or payload.get("row_count") or 0),
                        "schema_version": str(
                            payload.get("schema_version")
                            or meta_schema_version
                        ),
                        "source_id": str(payload.get("source_id") or meta_source_id),
                        "created_at": manifest_created_at,
                    }
                )
        patterns = {
            f"%/{ym}/%",
            f"%\\{ym}\\%",
            f"{ym}/%",
            f"{ym}\\%",
        }
        deleted_manifests = 0
        for pattern in patterns:
            deleted_manifests += _delete_where(
                conn, "manifests", "path LIKE ?", [pattern]
            )
        if manifest_rows:
            upsert_dataframe(
                conn,
                "manifests",
                pd.DataFrame(manifest_rows),
                keys=["path"],
            )
        LOGGER.debug("Deleted %s manifest rows for ym=%s", deleted_manifests, ym)
        snapshot_payload = {
            "ym": ym,
            "created_at": _default_created_at(meta.get("created_at_utc") if meta else None),
            "git_sha": str(meta.get("source_commit_sha", "")) if meta else "",
            "export_version": str(meta.get("export_version", "")) if meta else "",
            "facts_rows": facts_rows,
            "deltas_rows": deltas_rows,
            "meta": json.dumps(dict(meta or {}), sort_keys=True),
        }
        _delete_where(conn, "snapshots", "ym = ?", [ym])
        conn.execute(
            """
            INSERT INTO snapshots
            (ym, created_at, git_sha, export_version, facts_rows, deltas_rows, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                snapshot_payload["ym"],
                snapshot_payload["created_at"],
                snapshot_payload["git_sha"],
                snapshot_payload["export_version"],
                int(snapshot_payload["facts_rows"] or 0),
                int(snapshot_payload["deltas_rows"] or 0),
                snapshot_payload["meta"],
            ],
        )
        LOGGER.debug(
            "Snapshot summary: %s",
            snapshot_payload,
        )
        LOGGER.info(
            (
                "DuckDB snapshot write complete: ym=%s facts_resolved=%s deltas=%s "
                "deleted_resolved=%s deleted_deltas=%s"
            ),
            ym,
            facts_rows,
            deltas_rows,
            deleted_resolved,
            deleted_deltas,
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
