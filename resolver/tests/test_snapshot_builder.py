from __future__ import annotations

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from resolver.snapshot.builder import build_monthly_snapshot, build_snapshot_for_month


def _setup_minimal_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    # Minimal schemas for the canonical tables we depend on
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


def _setup_canonical_db() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    # Simplified representation of the canonical schema in resolver/db/schema.sql
    con.execute(
        """
        CREATE TABLE facts_resolved (
            ym TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            hazard_label TEXT,
            hazard_class TEXT,
            metric TEXT,
            series_semantics TEXT,
            value DOUBLE,
            unit TEXT,
            as_of DATE,
            as_of_date VARCHAR,
            publication_date VARCHAR,
            publisher TEXT,
            source_id TEXT,
            source_type TEXT,
            source_url TEXT,
            doc_title TEXT,
            definition_text TEXT,
            precedence_tier TEXT,
            event_id TEXT,
            proxy_for TEXT,
            confidence TEXT
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
            value_new DOUBLE,
            value_stock DOUBLE,
            series_semantics TEXT,
            as_of VARCHAR,
            as_of_date VARCHAR,
            source_id TEXT,
            source_name TEXT,
            definition_text TEXT,
            delta_negative_clamped INTEGER,
            first_observation INTEGER,
            rebase_flag INTEGER,
            source_url TEXT
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
    # Insert one resolved (DTM-style), two deltas (IDMC/EM-DAT style), and one ACLED monthly row
    con.execute(
        """
        INSERT INTO facts_resolved VALUES
            ('2025-11', 'SDN', 'displacement', 'idp_displacement_stock_dtm', 'stock', 1000.0, 'IOM DTM', DATE '2025-11-30');
        """
    )
    con.execute(
        """
        INSERT INTO facts_deltas VALUES
            ('2025-11', 'SDN', 'displacement', 'idp_displacement_new_dtm', 'new', 50.0, 'IDMC', DATE '2025-11-30'),
            ('2025-11', 'SDN', 'drought', 'emdat_affected', 'new', 5000.0, 'EMDAT', DATE '2025-11-30');
        """
    )
    con.execute(
        """
        INSERT INTO acled_monthly_fatalities VALUES
            ('SDN', DATE '2025-11-01', 7.0);
        """
    )

    result = build_snapshot_for_month(
        con,
        ym="2025-11",
        run_id="snapshot-test-1",
        snapshot_root=tmp_path,
        write_parquet=True,
    )

    # We expect 4 snapshot rows: 1 from facts_resolved, 2 from facts_deltas, 1 from ACLED
    assert result.snapshot_rows == 4
    assert result.resolved_rows == 1
    assert result.delta_rows == 2
    assert result.acled_rows == 1
    assert result.ym == "2025-11"
    assert result.snapshot_path is not None
    assert result.snapshot_path.exists()

    rows = con.execute(
        "SELECT ym, iso3, hazard_code, metric, series_semantics, value, source, provenance_table "
        "FROM facts_snapshot WHERE ym = '2025-11' ORDER BY provenance_table, metric"
    ).fetchall()

    provenance = [r[7] for r in rows]
    assert "facts_resolved" in provenance
    assert "facts_deltas" in provenance
    assert "acled_monthly_fatalities" in provenance

    meta = con.execute(
        "SELECT ym, run_id FROM snapshots WHERE ym = '2025-11'"
    ).fetchall()
    assert len(meta) == 1
    assert meta[0][0] == "2025-11"
    assert meta[0][1] == "snapshot-test-1"


def test_build_snapshot_for_month_is_idempotent(tmp_path: Path) -> None:
    con = _setup_minimal_db()
    con.execute(
        """
        INSERT INTO facts_resolved VALUES
            ('2025-10', 'KEN', 'displacement', 'idp_displacement_stock_dtm', 'stock', 123.0, 'IOM DTM', DATE '2025-10-31');
        """
    )

    res1 = build_snapshot_for_month(
        con,
        ym="2025-10",
        run_id="run-a",
        snapshot_root=tmp_path,
        write_parquet=False,
    )
    assert res1.snapshot_rows == 1

    con.execute(
        """
        UPDATE facts_resolved
        SET value = 200.0
        WHERE ym = '2025-10' AND iso3 = 'KEN';
        """
    )
    res2 = build_snapshot_for_month(
        con,
        ym="2025-10",
        run_id="run-b",
        snapshot_root=tmp_path,
        write_parquet=False,
    )
    assert res2.snapshot_rows == 1
    row = con.execute(
        "SELECT value, run_id FROM facts_snapshot WHERE ym = '2025-10'",
    ).fetchone()
    assert row[0] == 200.0
    assert row[1] == "run-b"


def test_build_monthly_snapshot_alias(tmp_path: Path) -> None:
    con = _setup_minimal_db()
    con.execute(
        """
        INSERT INTO facts_resolved VALUES
            ('2025-12', 'UGA', 'displacement',
             'idp_displacement_stock_dtm', 'stock',
             42.0, 'IOM DTM', DATE '2025-12-31');
        """
    )

    result = build_monthly_snapshot(
        con,
        ym="2025-12",
        run_id="alias-test",
        snapshot_root=tmp_path,
        write_parquet=True,
    )

    assert result.ym == "2025-12"
    assert result.resolved_rows == 1
    assert result.delta_rows == 0
    assert result.acled_rows == 0
    assert result.snapshot_rows == 1
    assert result.snapshot_path is not None
    assert result.snapshot_path.exists()


def test_build_snapshot_with_canonical_db_schema(tmp_path: Path) -> None:
    con = _setup_canonical_db()
    con.execute(
        """
        INSERT INTO facts_resolved (
            ym, iso3, hazard_code, hazard_label, hazard_class,
            metric, series_semantics, value, unit,
            as_of, as_of_date, publication_date,
            publisher, source_id
        )
        VALUES (
            '2025-11', 'SDN', 'displacement', 'Displacement', 'displacement',
            'idp_displacement_stock_dtm', 'stock', 1000.0, 'persons',
            DATE '2025-11-30', '2025-11-30', '2025-12-01',
            'IOM DTM', 'dtm_admin0'
        );
        """
    )
    con.execute(
        """
        INSERT INTO facts_deltas (
            ym, iso3, hazard_code, metric,
            value_new, value_stock, series_semantics,
            as_of, as_of_date, source_id, source_name
        )
        VALUES (
            '2025-11', 'SDN', 'displacement', 'new_displacements',
            50.0, 1050.0, 'new',
            '2025-11-30', '2025-11-30', 'idmc', 'IDMC'
        );
        """
    )
    con.execute(
        """
        INSERT INTO acled_monthly_fatalities VALUES
            ('SDN', DATE '2025-11-01', 7.0);
        """
    )

    result = build_snapshot_for_month(
        con,
        ym="2025-11",
        run_id="canonical-test",
        snapshot_root=tmp_path,
        write_parquet=True,
    )

    assert result.snapshot_rows == 3
    assert result.resolved_rows == 1
    assert result.delta_rows == 1
    assert result.acled_rows == 1

    rows = con.execute(
        "SELECT ym, iso3, hazard_code, metric, series_semantics, value, source, provenance_table "
        "FROM facts_snapshot WHERE ym = '2025-11' ORDER BY provenance_table, metric"
    ).fetchall()
    sources = {r[6] for r in rows}
    assert "IOM DTM" in sources or "dtm_admin0" in sources
    assert "IDMC" in sources or "idmc" in sources
