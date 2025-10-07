"""Helpers for connecting to and writing into the DuckDB backend."""

from __future__ import annotations

import json
import os
import uuid
import datetime as dt
import logging
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

try:  # pragma: no cover - import guard for optional dependency
    import duckdb
except ImportError as exc:  # pragma: no cover - guidance for operators
    raise RuntimeError(
        "DuckDB is required for database-backed resolver operations. Install 'duckdb'."
    ) from exc

from resolver.common import (
    compute_series_semantics,
    get_logger,
    dict_counts,
    df_schema,
)

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
    "series_semantics",
]
FACTS_RESOLVED_KEY = FACTS_RESOLVED_KEY_COLUMNS  # Backwards compatibility
FACTS_DELTAS_KEY = FACTS_DELTAS_KEY_COLUMNS
DEFAULT_DB_URL = os.environ.get(
    "RESOLVER_DB_URL", f"duckdb:///{ROOT / 'db' / 'resolver.duckdb'}"
)

LOGGER = get_logger(__name__)


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


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
    if url.startswith("duckdb:///"):
        db_path = Path(url.replace("duckdb:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(database=str(db_path), read_only=False)
    elif url.startswith("duckdb://"):
        conn = duckdb.connect(url.replace("duckdb://", "", 1))
    else:
        conn = duckdb.connect(url or None)
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
        "snapshots",
    }
    existing_tables = {
        row[0]
        for row in conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_name IN ('facts_resolved','facts_deltas','manifests','snapshots')
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
              AND table_name IN ('facts_resolved','facts_deltas','manifests','snapshots')
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
    """Upsert rows into ``table`` using ``keys`` as the natural key."""

    if df is None or df.empty:
        return 0

    frame = df.copy()
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
    else:
        semantics_series = frame["series_semantics"].where(
            frame["series_semantics"].notna(), ""
        )
        semantics_series = semantics_series.astype(str).str.strip()
        semantics_series = semantics_series.mask(
            semantics_series.str.lower().isin({"none", "nan"}), ""
        )
        frame["series_semantics"] = semantics_series

    table_info = conn.execute(f"PRAGMA table_info({_quote_literal(table)})").fetchall()
    if not table_info:
        raise ValueError(f"Table '{table}' does not exist in DuckDB database")
    table_columns = [row[1] for row in table_info]

    insert_columns = [col for col in frame.columns if col in table_columns]
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

    if keys:
        missing_keys = [k for k in keys if k not in frame.columns]
        if missing_keys:
            raise KeyError(
                f"Upsert keys {missing_keys} are missing from dataframe for table '{table}'"
            )
        for key in keys:
            if key in frame.columns:
                frame[key] = frame[key].astype(str).str.strip()
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

    if "series_semantics" in frame.columns:
        semantics = frame["series_semantics"].astype(str)
        normalised = semantics.str.strip().str.lower()
        frame.loc[normalised.str.startswith("new"), "series_semantics"] = "new"
        frame.loc[normalised.str.startswith("stock"), "series_semantics"] = "stock"
        cleaned = frame["series_semantics"].astype(str).str.strip().str.lower()
        frame.loc[~cleaned.isin({"", "new", "stock"}), "series_semantics"] = ""
        frame["series_semantics"] = frame["series_semantics"].astype(str).str.strip()

    temp_name = f"tmp_{uuid.uuid4().hex}"
    conn.register(temp_name, frame)
    try:
        if keys:
            comparisons = " AND ".join(
                f"coalesce(t.{_quote_identifier(k)}, '') = coalesce(s.{_quote_identifier(k)}, '')"
                for k in keys
            )
            table_ident = _quote_identifier(table)
            temp_ident = _quote_identifier(temp_name)
            exists_sql = (
                f"SELECT COUNT(*) FROM {table_ident} AS t "
                f"WHERE EXISTS (SELECT 1 FROM {temp_ident} AS s WHERE {comparisons})"
            )
            delete_sql = (
                f"DELETE FROM {table_ident} AS t "
                f"WHERE EXISTS (SELECT 1 FROM {temp_ident} AS s WHERE {comparisons})"
            )
            delete_count = int(conn.execute(exists_sql).fetchone()[0])
            if delete_count:
                conn.execute(delete_sql)
            LOGGER.info(
                "Deleted %s existing rows from %s using keys %s",
                delete_count,
                table,
                list(keys),
            )
        cols_csv = ", ".join(_quote_identifier(col) for col in insert_columns)
        table_ident = _quote_identifier(table)
        temp_ident = _quote_identifier(temp_name)
        insert_sql = (
            f"INSERT INTO {table_ident} ({cols_csv}) SELECT {cols_csv} FROM {temp_ident}"
        )
        conn.execute(insert_sql)
    except Exception:
        LOGGER.error(
            "DuckDB upsert failed for table %s. SQL: %s | schema=%s | first_row=%s",
            table,
            insert_sql,
            table_columns,
            frame.head(1).to_dict(orient="records"),
            exc_info=True,
        )
        raise
    finally:
        conn.unregister(temp_name)
    LOGGER.info("Inserted %s rows into %s", len(frame), table)
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

    conn.execute("BEGIN")
    try:
        deleted_resolved = _delete_where(conn, "facts_resolved", "ym = ?", [ym])
        deleted_deltas = _delete_where(conn, "facts_deltas", "ym = ?", [ym])

        facts_rows = 0
        deltas_rows = 0
        if facts_resolved is not None and not facts_resolved.empty:
            facts_resolved = _ensure_columns(
                facts_resolved,
                FACTS_RESOLVED_KEY_COLUMNS + ["value"],
            )
            for key in FACTS_RESOLVED_KEY_COLUMNS:
                facts_resolved[key] = (
                    facts_resolved[key]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )
            facts_resolved["ym"] = facts_resolved["ym"].replace("", ym)
            if "value" in facts_resolved.columns:
                facts_resolved["value"] = pd.to_numeric(
                    facts_resolved["value"], errors="coerce"
                )
            facts_resolved["series_semantics"] = facts_resolved.apply(
                lambda row: compute_series_semantics(
                    metric=row.get("metric"), existing=row.get("series_semantics")
                ),
                axis=1,
            )
            facts_resolved["series_semantics"] = (
                facts_resolved["series_semantics"].fillna("").astype(str).str.strip()
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
        if facts_deltas is not None and not facts_deltas.empty:
            facts_deltas = facts_deltas.copy()
            facts_deltas = _ensure_columns(
                facts_deltas,
                FACTS_DELTAS_KEY_COLUMNS + ["value_new", "value_stock"],
            )
            for key in FACTS_DELTAS_KEY_COLUMNS:
                series = (
                    facts_deltas[key]
                    .fillna("new" if key == "series_semantics" else "")
                    .astype(str)
                    .str.strip()
                )
                if key == "series_semantics":
                    series = series.replace("", "new")
                facts_deltas[key] = series
            facts_deltas["ym"] = facts_deltas["ym"].replace("", ym)
            numeric_delta_columns = [
                col
                for col in ("value_new", "value_stock")
                if col in facts_deltas.columns
            ]
            for column in numeric_delta_columns:
                facts_deltas[column] = pd.to_numeric(
                    facts_deltas[column], errors="coerce"
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
            if "series_semantics" in facts_deltas.columns:
                LOGGER.debug(
                    "facts_deltas series_semantics distribution: %s",
                    dict_counts(facts_deltas["series_semantics"]),
                )
        manifest_rows: list[dict] = []
        if manifests:
            for entry in manifests:
                payload = dict(entry)
                name = str(payload.get("name") or payload.get("path") or "artifact")
                manifest_rows.append(
                    {
                        "ym": ym,
                        "name": name,
                        "path": str(payload.get("path", "")),
                        "rows": int(payload.get("rows", 0) or 0),
                        "checksum": str(payload.get("checksum", "")),
                        "payload": json.dumps(payload, sort_keys=True),
                    }
                )
        deleted_manifests = _delete_where(conn, "manifests", "ym = ?", [ym])
        if manifest_rows:
            upsert_dataframe(conn, "manifests", pd.DataFrame(manifest_rows))
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
