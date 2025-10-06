"""Helpers for connecting to and writing into the DuckDB backend."""

from __future__ import annotations

import json
import os
import uuid
import datetime as dt
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

try:  # pragma: no cover - import guard for optional dependency
    import duckdb
except ImportError as exc:  # pragma: no cover - guidance for operators
    raise RuntimeError(
        "DuckDB is required for database-backed resolver operations. Install 'duckdb'."
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "db" / "schema.sql"
DEFAULT_DB_URL = os.environ.get(
    "RESOLVER_DB_URL", f"duckdb:///{ROOT / 'db' / 'resolver.duckdb'}"
)


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
    sql = schema_path.read_text(encoding="utf-8")
    # DuckDB executes multiple statements separated by semicolons.
    for statement in [s.strip() for s in sql.split(";") if s.strip()]:
        conn.execute(statement)


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
    for column in frame.columns:
        if frame[column].dtype == "object":
            frame[column] = frame[column].astype(str)
    temp_name = f"tmp_{uuid.uuid4().hex}"
    conn.register(temp_name, frame)
    try:
        if keys:
            comparisons = " AND ".join(
                f"coalesce(t.{k}, '') = coalesce(s.{k}, '')" for k in keys
            )
            delete_sql = f"DELETE FROM {table} t USING {temp_name} s WHERE {comparisons}"
            conn.execute(delete_sql)
        insert_sql = f"INSERT INTO {table} BY NAME SELECT * FROM {temp_name}"
        conn.execute(insert_sql)
    finally:
        conn.unregister(temp_name)
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
        facts_rows = 0
        deltas_rows = 0
        if facts_resolved is not None and not facts_resolved.empty:
            facts_resolved = _ensure_columns(
                facts_resolved,
                [
                    "ym",
                    "iso3",
                    "hazard_code",
                    "metric",
                    "series_semantics",
                    "value",
                ],
            )
            facts_resolved["ym"] = facts_resolved["ym"].replace("", ym)
            facts_resolved["series_semantics"] = facts_resolved.get(
                "series_semantics", "stock"
            ).replace("", "stock")
            facts_rows = upsert_dataframe(
                conn,
                "facts_resolved",
                facts_resolved,
                keys=["ym", "iso3", "hazard_code", "metric", "series_semantics"],
            )
        if facts_deltas is not None and not facts_deltas.empty:
            facts_deltas = facts_deltas.copy()
            facts_deltas = _ensure_columns(
                facts_deltas,
                [
                    "ym",
                    "iso3",
                    "hazard_code",
                    "metric",
                    "value_new",
                ],
            )
            facts_deltas["ym"] = facts_deltas["ym"].replace("", ym)
            facts_deltas["series_semantics"] = facts_deltas.get(
                "series_semantics", "new"
            ).replace("", "new")
            deltas_rows = upsert_dataframe(
                conn,
                "facts_deltas",
                facts_deltas,
                keys=["ym", "iso3", "hazard_code", "metric"],
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
        if manifest_rows:
            upsert_dataframe(conn, "manifests", pd.DataFrame(manifest_rows), keys=["ym", "name"])
        snapshot_payload = {
            "ym": ym,
            "created_at": _default_created_at(meta.get("created_at_utc") if meta else None),
            "git_sha": str(meta.get("source_commit_sha", "")) if meta else "",
            "export_version": str(meta.get("export_version", "")) if meta else "",
            "facts_rows": facts_rows,
            "deltas_rows": deltas_rows,
            "meta": json.dumps(dict(meta or {}), sort_keys=True),
        }
        upsert_dataframe(conn, "snapshots", pd.DataFrame([snapshot_payload]), keys=["ym"])
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
