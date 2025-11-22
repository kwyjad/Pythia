from __future__ import annotations

import datetime as _dt
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from resolver.db import duckdb_io


@dataclass
class SnapshotResult:
    ym: str
    snapshot_rows: int
    resolved_rows: int
    delta_rows: int
    acled_rows: int
    snapshot_path: Optional[Path]
    db_url: str
    created_at: _dt.datetime
    run_id: str


SNAPSHOT_TABLE = "facts_snapshot"
SNAPSHOTS_META_TABLE = "snapshots"


def _ensure_tables(conn) -> None:
    """
    Ensure the snapshot tables exist in the connected DuckDB database.

    - facts_snapshot: unified per-month snapshot (both stock + flow).
    - snapshots: metadata about each built month (ym, created_at, run_id).
    """

    conn.execute(
        f"
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
        ""
    )
    conn.execute(
        f"
        CREATE TABLE IF NOT EXISTS {SNAPSHOTS_META_TABLE} (
            ym TEXT PRIMARY KEY,
            created_at TIMESTAMP,
            run_id TEXT
        );
        ""
    )


def _delete_existing_for_month(conn, ym: str) -> None:
    """
    Remove any existing snapshot data for this month to keep the operation idempotent.
    """

    conn.execute(f"DELETE FROM {SNAPSHOT_TABLE} WHERE ym = ?", [ym])
    conn.execute(f"DELETE FROM {SNAPSHOTS_META_TABLE} WHERE ym = ?", [ym])


def _insert_from_facts_tables(conn, ym: str, run_id: str) -> Tuple[int, int, int]:
    """
    Copy rows for the given ym from facts_resolved, facts_deltas, and acled_monthly_fatalities
    into the unified facts_snapshot table.

    Returns a tuple: (resolved_rows_inserted, delta_rows_inserted, acled_rows_inserted).
    """

    resolved_rows = conn.execute(
        f"
        INSERT INTO {SNAPSHOT_TABLE}
        SELECT
            ym,
            iso3,
            hazard_code,
            metric,
            series_semantics,
            value,
            source,
            as_of_date,
            'facts_resolved' AS provenance_table,
            ?
        FROM facts_resolved
        WHERE ym = ?
        ",
        [run_id, ym],
    ).rowcount

    delta_rows = conn.execute(
        f"
        INSERT INTO {SNAPSHOT_TABLE}
        SELECT
            ym,
            iso3,
            hazard_code,
            metric,
            series_semantics,
            value,
            source,
            as_of_date,
            'facts_deltas' AS provenance_table,
            ?
        FROM facts_deltas
        WHERE ym = ?
        ",
        [run_id, ym],
    ).rowcount

    acled_rows = 0
    try:
        conn.execute(
            """
            CREATE TEMP TABLE __acled_monthly AS
            SELECT
                strftime(month, '%Y-%m') AS ym,
                iso3,
                CAST(fatalities AS DOUBLE) AS value,
                month::DATE AS as_of_date
            FROM acled_monthly_fatalities
            WHERE strftime(month, '%Y-%m') = ?
            """,
            [ym],
        )
        acled_rows = conn.execute(
            f"
            INSERT INTO {SNAPSHOT_TABLE}
            SELECT
                ym,
                iso3,
                'conflict_violence' AS hazard_code,
                'fatalities_acled' AS metric,
                'new' AS series_semantics,
                value,
                'ACLED' AS source,
                as_of_date,
                'acled_monthly_fatalities' AS provenance_table,
                ?
            FROM __acled_monthly
            ",
            [run_id],
        ).rowcount
        conn.execute("DROP TABLE IF EXISTS __acled_monthly;")
    except Exception:
        pass

    return resolved_rows or 0, delta_rows or 0, acled_rows or 0


def _insert_snapshot_meta(conn, ym: str, run_id: str, created_at: _dt.datetime) -> None:
    conn.execute(
        f"INSERT INTO {SNAPSHOTS_META_TABLE} (ym, created_at, run_id) VALUES (?, ?, ?)",
        [ym, created_at, run_id],
    )


def build_snapshot_for_month(
    conn,
    ym: str,
    run_id: Optional[str] = None,
    snapshot_root: Path = Path("data") / "snapshots",
    write_parquet: bool = True,
) -> SnapshotResult:
    """
    Build a unified snapshot for a single month (ym) into the connected DuckDB database.

    - `conn` is a live duckdb connection (see resolver.db.duckdb_io.get_db).
    - `ym` is a 'YYYY-MM' string.
    - Existing rows for this ym in facts_snapshot/snapshots are removed (idempotent).
    - The new snapshot is written into the `facts_snapshot` table and optionally
      a per-month Parquet file at `snapshot_root/<ym>/facts.parquet`.

    Returns a SnapshotResult with counts and metadata.
    """

    created_at = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)
    run_id = run_id or str(uuid.uuid4())

    _ensure_tables(conn)
    _delete_existing_for_month(conn, ym)

    resolved_rows, delta_rows, acled_rows = _insert_from_facts_tables(conn, ym, run_id)
    total_rows = resolved_rows + delta_rows + acled_rows

    snapshot_path: Optional[Path] = None
    if write_parquet and total_rows > 0:
        snapshot_dir = snapshot_root / ym
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / "facts.parquet"
        conn.execute(
            f"
            COPY (
                SELECT *
                FROM {SNAPSHOT_TABLE}
                WHERE ym = ?
                ORDER BY iso3, hazard_code, metric, series_semantics, as_of_date
            ) TO ? (FORMAT 'parquet')
            ",
            [ym, str(snapshot_path)],
        )

    _insert_snapshot_meta(conn, ym, run_id, created_at)

    db_url = ""
    try:
        db_url = getattr(conn, "database_name", "") or ""
    except Exception:
        db_url = ""

    return SnapshotResult(
        ym=ym,
        snapshot_rows=total_rows,
        resolved_rows=resolved_rows,
        delta_rows=delta_rows,
        acled_rows=acled_rows,
        snapshot_path=snapshot_path,
        db_url=db_url,
        created_at=created_at,
        run_id=run_id,
    )


def build_snapshots(
    db_url: str,
    months: Iterable[str],
    write_parquet: bool = True,
    run_id: Optional[str] = None,
) -> List[SnapshotResult]:
    """
    Build snapshots for one or more months against the DuckDB database at `db_url`.

    Example:
        build_snapshots("duckdb:///data/resolver_backfill.duckdb", ["2025-10", "2025-11"])

    - Opens a connection via resolver.db.duckdb_io.get_db.
    - For each month, calls build_snapshot_for_month.
    - Returns a list of SnapshotResult objects (one per month).
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
            res = build_snapshot_for_month(conn, ym, run_id, write_parquet=write_parquet)
            results.append(res)
    finally:
        try:
            duckdb_io.close_db(conn)
        except Exception:
            pass
    return results
