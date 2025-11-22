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
    """Metadata about a single monthly snapshot build."""

    ym: str
    snapshot_rows: int
    resolved_rows: int
    delta_rows: int
    acled_rows: int
    snapshot_path: Optional[Path]
    db_url: str
    created_at: dt.datetime
    run_id: str


def _ensure_tables(conn) -> None:
    """
    Ensure that the snapshot tables exist in the connected DuckDB database.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS {snapshot} (
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
        """.format(snapshot=SNAPSHOT_TABLE)
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS {meta} (
            ym TEXT PRIMARY KEY,
            created_at TIMESTAMP,
            run_id TEXT
        );
        """.format(meta=SNAPSHOTS_META_TABLE)
    )


def _delete_existing_for_month(conn, ym: str) -> None:
    """
    Remove existing snapshot rows and metadata for the given month.
    This makes the operation idempotent.
    """
    conn.execute(
        "DELETE FROM {snapshot} WHERE ym = ?".format(snapshot=SNAPSHOT_TABLE), [ym]
    )
    conn.execute(
        "DELETE FROM {meta} WHERE ym = ?".format(meta=SNAPSHOTS_META_TABLE), [ym]
    )


def _insert_from_facts_tables(conn, ym: str, run_id: str) -> Tuple[int, int, int]:
    """
    Copy rows for the given ym from facts_resolved, facts_deltas, and (if present)
    acled_monthly_fatalities into the unified snapshot table.

    Returns (resolved_rows_inserted, delta_rows_inserted, acled_rows_inserted).
    """

    conn.execute(
        """
        INSERT INTO {snapshot} (ym, iso3, hazard_code, metric, series_semantics, value, source, as_of_date, provenance_table, run_id)
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
        """.format(
            snapshot=SNAPSHOT_TABLE
        ),
        [run_id, ym],
    )
    resolved_rows = conn.execute(
        "SELECT COUNT(*) FROM facts_resolved WHERE ym = ?", [ym]
    ).fetchone()[0]

    conn.execute(
        """
        INSERT INTO {snapshot} (ym, iso3, hazard_code, metric, series_semantics, value, source, as_of_date, provenance_table, run_id)
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
        """.format(
            snapshot=SNAPSHOT_TABLE
        ),
        [run_id, ym],
    )
    delta_rows = conn.execute(
        "SELECT COUNT(*) FROM facts_deltas WHERE ym = ?", [ym]
    ).fetchone()[0]

    acled_rows = 0
    try:
        conn.execute(
            """
            SELECT 1
            FROM information_schema.tables
            WHERE table_name = 'acled_monthly_fatalities'
            LIMIT 1
            """
        )
        has_acled = conn.fetchone() is not None
    except Exception:
        has_acled = False

    if has_acled:
        conn.execute(
            """
            CREATE TEMP TABLE IF NOT EXISTS __acled_monthly AS
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
        conn.execute(
            """
            INSERT INTO {snapshot} (ym, iso3, hazard_code, metric, series_semantics, value, source, as_of_date, provenance_table, run_id)
            SELECT
                ym,
                iso3,
                'conflict' AS hazard_code,
                'fatalities_acled' AS metric,
                'new' AS series_semantics,
                value,
                'ACLED' AS source,
                as_of_date,
                'acled_monthly_fatalities' AS provenance_table,
                ?
            FROM __acled_monthly
            """.format(
                snapshot=SNAPSHOT_TABLE
            ),
            [run_id],
        )
        acled_rows = conn.execute(
            "SELECT COUNT(*) FROM __acled_monthly"
        ).fetchone()[0]
        conn.execute("DROP TABLE IF EXISTS __acled_monthly")

    return int(resolved_rows), int(delta_rows), int(acled_rows)


def _insert_snapshot_meta(
    conn, ym: str, run_id: str, created_at: dt.datetime
) -> None:
    conn.execute(
        "INSERT INTO {meta} (ym, created_at, run_id) VALUES (?, ?, ?)".format(
            meta=SNAPSHOTS_META_TABLE
        ),
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

    - Drops any existing snapshot rows for this ym in {snapshot}.
    - Copies rows for ym from facts_resolved, facts_deltas, and acled_monthly_fatalities.
    - Optionally writes a Parquet file at snapshot_root/<ym>/facts.parquet.

    This function does not modify any connector logic and is safe to run repeatedly.
    """.format(
        snapshot=SNAPSHOT_TABLE
    )
    created_at = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    actual_run_id = run_id or str(uuid.uuid4())

    LOG.info("Building snapshot for ym=%s (run_id=%s)", ym, actual_run_id)

    _ensure_tables(conn)
    _delete_existing_for_month(conn, ym)
    resolved_rows, delta_rows, acled_rows = _insert_from_facts_tables(
        conn, ym, actual_run_id
    )
    snapshot_rows = resolved_rows + delta_rows + acled_rows

    snapshot_path: Optional[Path] = None
    if write_parquet and snapshot_rows > 0:
        snapshot_dir = snapshot_root / ym
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / "facts.parquet"
        LOG.info("Writing snapshot parquet for ym=%s to %s", ym, snapshot_path)
        conn.execute(
            """
            COPY (
                SELECT *
                FROM {snapshot}
                WHERE ym = ?
                ORDER BY iso3, hazard_code, metric, series_semantics, as_of_date
            ) TO ? (FORMAT 'parquet');
            """.format(
                snapshot=SNAPSHOT_TABLE
            ),
            [ym, str(snapshot_path)],
        )

    _insert_snapshot_meta(conn, ym, actual_run_id, created_at)

    db_url = ""
    try:
        db_path = getattr(conn, "database_name", None)
        if isinstance(db_path, str):
            db_url = db_path
    except Exception:
        pass

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


def build_snapshots(
    db_url: str,
    months: Iterable[str],
    write_parquet: bool = True,
    run_id: Optional[str] = None,
) -> List[SnapshotResult]:
    """
    Build snapshots for one or more months using the DuckDB database at `db_url`.

    Example:
        build_snapshots("duckdb:///data/resolver_backfill.duckdb", ["2025-10", "2025-11"])
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
        try:
            duckdb_io.close_db(conn)
        except Exception:
            pass

    return results
