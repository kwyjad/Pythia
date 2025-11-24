from __future__ import annotations

import datetime as dt
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from resolver.db import duckdb_io

LOG = logging.getLogger(__name__)

SNAPSHOT_TABLE = "facts_snapshot"
SNAPSHOTS_META_TABLE = "snapshots"


@dataclass
class SnapshotResult:
    """Summary of a per-month snapshot operation."""

    ym: str
    snapshot_rows: int
    resolved_rows: int
    delta_rows: int
    acled_rows: int
    snapshot_path: Optional[Path]
    db_url: str
    created_at: dt.datetime
    run_id: str


def _get_table_columns(conn, table: str) -> set[str]:
    """Return the set of column names for a DuckDB table (best-effort)."""

    try:
        rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("Failed to inspect schema for %s: %s", table, exc)
        return set()

    cols = {str(row[1]) for row in rows if len(row) >= 2}
    LOG.debug("Detected columns for %s: %s", table, sorted(cols))
    return cols


def _ensure_tables(conn) -> None:
    """Ensure the snapshot tables exist in the DuckDB database."""

    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SNAPSHOT_TABLE} (
            ym TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            series_semantics TEXT,
            value DOUBLE,
            source TEXT,
            as_of_date DATE,
            provenance_table TEXT,
            run_id TEXT
        );
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {SNAPSHOTS_META_TABLE} (
            ym TEXT PRIMARY KEY,
            created_at TIMESTAMP,
            run_id TEXT
        );
        """
    )


def _delete_existing_for_month(conn, ym: str) -> None:
    """Remove existing snapshot rows/metadata for the given month (idempotent)."""

    conn.execute(f"DELETE FROM {SNAPSHOT_TABLE} WHERE ym = ?", [ym])
    conn.execute(f"DELETE FROM {SNAPSHOTS_META_TABLE} WHERE ym = ?", [ym])


def _insert_from_facts_tables(conn, ym: str, run_id: str) -> Tuple[int, int, int]:
    """
    Copy rows for the given ym from facts_resolved, facts_deltas, and
    acled_monthly_fatalities into the unified snapshot table.

    Returns (resolved_rows_inserted, delta_rows_inserted, acled_rows_inserted).
    """

    resolved_cols = _get_table_columns(conn, "facts_resolved")
    deltas_cols = _get_table_columns(conn, "facts_deltas")

    # --- facts_resolved mapping ---
    if "source" in resolved_cols:
        resolved_source_expr = "source"
    elif "publisher" in resolved_cols:
        resolved_source_expr = "publisher"
    elif "source_name" in resolved_cols:
        resolved_source_expr = "source_name"
    elif "source_id" in resolved_cols:
        resolved_source_expr = "source_id"
    else:
        resolved_source_expr = "''"

    if "series_semantics" in resolved_cols:
        resolved_series_expr = "series_semantics"
    else:
        resolved_series_expr = "'stock'"

    if "value" in resolved_cols:
        resolved_value_expr = "value"
    else:
        resolved_value_expr = "COALESCE(value_new, value_stock)"

    if "as_of_date" in resolved_cols:
        resolved_asof_expr = "CAST(as_of_date AS DATE)"
    elif "as_of" in resolved_cols:
        resolved_asof_expr = "CAST(as_of AS DATE)"
    else:
        resolved_asof_expr = "CAST(NULL AS DATE)"

    LOG.debug(
        "facts_resolved mapping for snapshot ym=%s: source=%s, series=%s, value=%s, as_of=%s",
        ym,
        resolved_source_expr,
        resolved_series_expr,
        resolved_value_expr,
        resolved_asof_expr,
    )

    try:
        res_cursor = conn.execute(
            f"""
            INSERT INTO {SNAPSHOT_TABLE}
            SELECT
                ym,
                iso3,
                hazard_code,
                metric,
                {resolved_series_expr} AS series_semantics,
                {resolved_value_expr} AS value,
                {resolved_source_expr} AS source,
                {resolved_asof_expr} AS as_of_date,
                'facts_resolved' AS provenance_table,
                ?
            FROM facts_resolved
            WHERE ym = ?
            """,
            [run_id, ym],
        )
        resolved_rows = conn.execute(
            f"""
            SELECT COUNT(*) FROM {SNAPSHOT_TABLE}
            WHERE ym = ? AND provenance_table = 'facts_resolved'
            """,
            [ym],
        ).fetchone()[0]
        LOG.debug(
            "Snapshot insert stats for ym=%s (facts_resolved): rowcount=%s, counted=%s",
            ym,
            res_cursor.rowcount,
            resolved_rows,
        )
    except Exception as exc:
        LOG.error("Error inserting facts_resolved for ym=%s: %s", ym, exc)
        raise

    # --- facts_deltas mapping ---
    if "source" in deltas_cols:
        delta_source_expr = "source"
    elif "source_name" in deltas_cols:
        delta_source_expr = "source_name"
    elif "source_id" in deltas_cols:
        delta_source_expr = "source_id"
    else:
        delta_source_expr = "''"

    if "series_semantics" in deltas_cols:
        delta_series_expr = "series_semantics"
    elif "semantics" in deltas_cols:
        delta_series_expr = "semantics"
    else:
        delta_series_expr = "'new'"

    if "value_new" in deltas_cols and "value_stock" in deltas_cols:
        delta_value_expr = "COALESCE(value_new, value_stock)"
    elif "value_new" in deltas_cols:
        delta_value_expr = "value_new"
    elif "value" in deltas_cols:
        delta_value_expr = "value"
    elif "value_stock" in deltas_cols:
        delta_value_expr = "value_stock"
    else:
        delta_value_expr = "0.0"

    if "as_of_date" in deltas_cols:
        delta_asof_expr = "CAST(as_of_date AS DATE)"
    elif "as_of" in deltas_cols:
        delta_asof_expr = "CAST(as_of AS DATE)"
    else:
        delta_asof_expr = "CAST(NULL AS DATE)"

    LOG.debug(
        "facts_deltas mapping for snapshot ym=%s: source=%s, series=%s, value=%s, as_of=%s",
        ym,
        delta_source_expr,
        delta_series_expr,
        delta_value_expr,
        delta_asof_expr,
    )

    try:
        delta_cursor = conn.execute(
            f"""
            INSERT INTO {SNAPSHOT_TABLE}
            SELECT
                ym,
                iso3,
                hazard_code,
                metric,
                {delta_series_expr} AS series_semantics,
                {delta_value_expr} AS value,
                {delta_source_expr} AS source,
                {delta_asof_expr} AS as_of_date,
                'facts_deltas' AS provenance_table,
                ?
            FROM facts_deltas
            WHERE ym = ?
            """,
            [run_id, ym],
        )
        delta_rows = conn.execute(
            f"""
            SELECT COUNT(*) FROM {SNAPSHOT_TABLE}
            WHERE ym = ? AND provenance_table = 'facts_deltas'
            """,
            [ym],
        ).fetchone()[0]
        LOG.debug(
            "Snapshot insert stats for ym=%s (facts_deltas): rowcount=%s, counted=%s",
            ym,
            delta_cursor.rowcount,
            delta_rows,
        )
    except Exception as exc:
        LOG.error("Error inserting facts_deltas for ym=%s: %s", ym, exc)
        raise

    acled_rows = 0
    try:
        acled_exists_row = conn.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_name = 'acled_monthly_fatalities'
            LIMIT 1
            """
        ).fetchone()
        has_acled = acled_exists_row is not None
        if has_acled:
            acled_cursor = conn.execute(
                f"""
                INSERT INTO {SNAPSHOT_TABLE}
                SELECT
                    strftime(month, '%Y-%m') AS ym,
                    iso3,
                    'conflict' AS hazard_code,
                    'fatalities_acled' AS metric,
                    'new' AS series_semantics,
                    CAST(fatalities AS DOUBLE) AS value,
                    'ACLED' AS source,
                    month::DATE AS as_of_date,
                    'acled_monthly_fatalities' AS provenance_table,
                    ?
                FROM acled_monthly_fatalities
                WHERE strftime(month, '%Y-%m') = ?
                """,
                [run_id, ym],
            )
            acled_rows = conn.execute(
                f"""
                SELECT COUNT(*) FROM {SNAPSHOT_TABLE}
                WHERE ym = ? AND provenance_table = 'acled_monthly_fatalities'
                """,
                [ym],
            ).fetchone()[0]
            LOG.debug(
                "Snapshot insert stats for ym=%s (acled_monthly_fatalities): rowcount=%s, counted=%s",
                ym,
                acled_cursor.rowcount,
                acled_rows,
            )
    except Exception as exc:
        LOG.warning(
            "Failed to include ACLED monthly fatalities in snapshot for ym=%s: %s",
            ym,
            exc,
        )
        acled_rows = 0

    return int(max(resolved_rows, 0)), int(max(delta_rows, 0)), int(max(acled_rows, 0))


def _insert_snapshot_meta(conn, ym: str, run_id: str, created_at: dt.datetime) -> None:
    """Insert/update a row in the snapshots metadata table for this month.

    This is schema-aware: it inspects the available columns on the snapshots table
    and only inserts into columns that exist (e.g., canonical schema lacks run_id).
    """

    cols = _get_table_columns(conn, SNAPSHOTS_META_TABLE)
    if not cols:
        LOG.warning(
            "Snapshot meta table '%s' has no columns; skipping metadata insert for ym=%s",
            SNAPSHOTS_META_TABLE,
            ym,
        )
        return

    conn.execute(f"DELETE FROM {SNAPSHOTS_META_TABLE} WHERE ym = ?", [ym])

    insert_columns: list[str] = []
    params: list[object] = []

    insert_columns.append("ym")
    params.append(ym)

    if "created_at" in cols:
        insert_columns.append("created_at")
        params.append(created_at)

    if "run_id" in cols:
        insert_columns.append("run_id")
        params.append(run_id)

    LOG.debug(
        "Snapshot meta insert for ym=%s: columns=%s available=%s",
        ym,
        insert_columns,
        sorted(cols),
    )

    placeholders = ", ".join("?" for _ in insert_columns)
    col_list = ", ".join(insert_columns)
    conn.execute(
        f"INSERT INTO {SNAPSHOTS_META_TABLE} ({col_list}) VALUES ({placeholders})",
        params,
    )


def build_snapshot_for_month(
    conn,
    ym: str,
    run_id: Optional[str] = None,
    snapshot_root: Path = Path("data") / "snapshots",
    write_parquet: bool = True,
) -> SnapshotResult:
    """
    Build a unified snapshot for a single month.

    - Drops any existing snapshot rows for `ym`.
    - Copies rows for `ym` from `facts_resolved`, `facts_deltas`, and, if present,
      `acled_monthly_fatalities` into `facts_snapshot`.
    - Optionally writes a Parquet file at `snapshot_root/<ym>/facts.parquet`.
    """

    created_at = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    actual_run_id = run_id or str(uuid.uuid4())

    LOG.info("Building snapshot for ym=%s (run_id=%s)", ym, actual_run_id)

    _ensure_tables(conn)
    _delete_existing_for_month(conn, ym)

    try:
        resolved_rows, delta_rows, acled_rows = _insert_from_facts_tables(
            conn, ym, actual_run_id
        )
    except Exception as exc:
        for tbl in ("facts_resolved", "facts_deltas", "acled_monthly_fatalities"):
            try:
                cols = conn.execute(f"PRAGMA table_info('{tbl}')").fetchall()
                LOG.error("Schema for %s: %s", tbl, cols)
            except Exception as exc2:
                LOG.error("Failed to inspect schema for %s: %s", tbl, exc2)
        LOG.error("Snapshot insert failed for ym=%s: %s", ym, exc)
        raise

    snapshot_rows = resolved_rows + delta_rows + acled_rows

    snapshot_path: Optional[Path] = None
    if write_parquet and snapshot_rows > 0:
        snapshot_dir = snapshot_root / ym
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / "facts.parquet"
        LOG.info("Writing snapshot parquet for ym=%s to %s", ym, snapshot_path)
        snapshot_df = conn.execute(
            f"""
            SELECT
                ym,
                iso3,
                hazard_code,
                metric,
                series_semantics,
                value,
                source,
                as_of_date,
                provenance_table,
                run_id
            FROM {SNAPSHOT_TABLE}
            WHERE ym = ?
            ORDER BY iso3, hazard_code, metric, series_semantics, as_of_date
            """,
            [ym],
        ).df()
        snapshot_df.to_parquet(snapshot_path)

    db_url = ""
    try:
        db_url = getattr(conn, "database_name", "") or ""
    except Exception:
        db_url = ""

    _insert_snapshot_meta(conn, ym, actual_run_id, created_at)

    return SnapshotResult(
        ym=ym,
        snapshot_rows=snapshot_rows,
        resolved_rows=resolved_rows,
        delta_rows=delta_rows,
        acled_rows=acled_rows,
        snapshot_path=snapshot_path,
        db_url=db_url,
        created_at=created_at,
        run_id=actual_run_id,
    )


def build_monthly_snapshot(
    conn,
    ym: str,
    run_id: Optional[str] = None,
    snapshot_root: Path = Path("data") / "snapshots",
    write_parquet: bool = True,
) -> SnapshotResult:
    """
    Backwards-compatible alias for build_snapshot_for_month.

    Used by resolver.cli.snapshot_from_db and GitHub workflows.
    """

    return build_snapshot_for_month(
        conn,
        ym=ym,
        run_id=run_id,
        snapshot_root=snapshot_root,
        write_parquet=write_parquet,
    )


def build_snapshots(
    db_url: str,
    months: Iterable[str],
    write_parquet: bool = True,
    run_id: Optional[str] = None,
) -> List[SnapshotResult]:
    """
    Build snapshots for the given list of `months` against the DuckDB database at `db_url`.
    """

    try:
        from resolver.db.duckdb_io import canonicalize_duckdb_target  # type: ignore

        db_url_canonical, _ = canonicalize_duckdb_target(db_url)
    except Exception:
        db_url_canonical = db_url

    conn = duckdb_io.get_db(db_url_canonical)
    results: List[SnapshotResult] = []
    try:
        for ym in months:
            res = build_snapshot_for_month(
                conn,
                ym=ym,
                run_id=run_id,
                snapshot_root=Path("data") / "snapshots",
                write_parquet=write_parquet,
            )
            results.append(res)
    finally:
        duckdb_io.close_db(conn)

    return results
