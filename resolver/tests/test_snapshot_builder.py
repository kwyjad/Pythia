from __future__ import annotations

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from resolver.snapshot.builder import SnapshotResult, build_snapshot_for_month


def _setup_minimal_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute(
        """
        CREATE TABLE facts_resolved (
            ym TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            series_semantics TEXT,
            value DOUBLE,
            source TEXT,
            as_of_date DATE
        );
        """
    )
    con.execute(
        """
        CREATE TABLE facts_deltas (
            ym TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            series_semantics TEXT,
            value DOUBLE,
            source TEXT,
            as_of_date DATE
        );
        """
    )
    con.execute(
        """
        CREATE TABLE acled_monthly_fatalities (
            iso3 TEXT,
            month DATE,
            fatalities DOUBLE
        );
        """
    )
    return con


def test_build_snapshot_for_month_inserts_from_all_sources(tmp_path: Path) -> None:
    con = _setup_minimal_db()
    con.execute(
        """
        INSERT INTO facts_resolved VALUES
            ('2025-11', 'SDN', 'displacement', 'idp_displacement_stock_idmc', 'stock', 1000.0, 'IDMC', DATE '2025-11-30');
        """
    )
    con.execute(
        """
        INSERT INTO facts_deltas VALUES
            ('2025-11', 'SDN', 'displacement', 'idp_displacement_new_dtm', 'new', 50.0, 'IOM DTM', DATE '2025-11-30'),
            ('2025-11', 'SDN', 'drought', 'emdat_affected', 'new', 5000.0, 'EMDAT', DATE '2025-11-30');
        """
    )
    con.execute(
        """
        INSERT INTO acled_monthly_fatalities VALUES
            ('SDN', DATE '2025-11-01', 7.0);
        """
    )

    result: SnapshotResult = build_snapshot_for_month(
        con,
        ym="2025-11",
        run_id="test-run-123",
        snapshot_root=tmp_path,
        write_parquet=True,
    )

    assert result.snapshot_rows == 4
    assert result.resolved_rows == 1
    assert result.delta_rows == 2
    assert result.acled_rows == 1
    assert result.ym == "2025-11"
    assert result.run_id == "test-run-123"
    assert result.snapshot_path is not None
    assert result.snapshot_path.exists()

    rows = con.execute(
        "SELECT ym, iso3, hazard_code, metric, series_semantics, value, source, provenance_table "
        "FROM facts_snapshot WHERE ym = '2025-11' ORDER BY metric"
    ).fetchall()

    metrics = [r[3] for r in rows]
    assert set(metrics) == {
        "emdat_affected",
        "fatalities_acled",
        "idp_displacement_new_dtm",
        "idp_displacement_stock_idmc",
    }

    meta = con.execute("SELECT ym, run_id FROM snapshots WHERE ym = '2025-11'").fetchall()
    assert len(meta) == 1
    assert meta[0][1] == "test-run-123"


def test_build_snapshot_for_month_is_idempotent(tmp_path: Path) -> None:
    con = _setup_minimal_db()
    con.execute(
        """
        INSERT INTO facts_resolved VALUES
            ('2025-10', 'KEN', 'displacement', 'idp_displacement_stock_idmc', 'stock', 123.0, 'IDMC', DATE '2025-10-31');
        """
    )

    first = build_snapshot_for_month(con, ym="2025-10", run_id="run1", snapshot_root=tmp_path, write_parquet=False)
    assert first.snapshot_rows == 1

    con.execute("UPDATE facts_snapshot SET value = 9999.0 WHERE ym = '2025-10'")
    con.execute("UPDATE facts_resolved SET value = 200.0 WHERE ym = '2025-10'")

    second = build_snapshot_for_month(con, ym="2025-10", run_id="run2", snapshot_root=tmp_path, write_parquet=False)
    assert second.snapshot_rows == 1

    val = con.execute("SELECT value, run_id FROM facts_snapshot WHERE ym = '2025-10'").fetchone()
    assert val[0] == 200.0
    assert val[1] == "run2"
