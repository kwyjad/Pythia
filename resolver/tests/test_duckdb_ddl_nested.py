"""Regression tests for nested DuckDB DDL transactions."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("duckdb")
import duckdb

from resolver.db import duckdb_io


@pytest.fixture()
def conn():
    connection = duckdb.connect(":memory:")
    try:
        yield connection
    finally:
        connection.close()


def test_init_schema_outside_transaction_ok(conn) -> None:
    duckdb_io.init_schema(conn)
    # Should be idempotent outside a transaction.
    duckdb_io.init_schema(conn)
    assert conn.execute("SELECT 1").fetchone()[0] == 1


def test_init_schema_inside_open_transaction_ok(conn) -> None:
    conn.execute("BEGIN TRANSACTION")
    try:
        duckdb_io.init_schema(conn)
        duckdb_io.init_schema(conn)
    finally:
        conn.execute("ROLLBACK")
    assert conn.execute("SELECT 1").fetchone()[0] == 1


def test_write_snapshot_performs_ddl_inside_transaction(conn) -> None:
    duckdb_io.init_schema(conn)

    facts_resolved = pd.DataFrame(
        [
            {
                "ym": "2024-01",
                "iso3": "USA",
                "hazard_code": "HZ",
                "metric": "cases",
                "value": 1.0,
            }
        ]
    )
    facts_deltas = pd.DataFrame(
        [
            {
                "ym": "2024-01",
                "iso3": "USA",
                "hazard_code": "HZ",
                "metric": "cases",
                "value_new": 2.0,
                "value_stock": 3.0,
            }
        ]
    )

    duckdb_io.write_snapshot(
        conn,
        ym="2024-01",
        facts_resolved=facts_resolved,
        facts_deltas=facts_deltas,
        manifests=None,
        meta={"created_at_utc": "2024-01-01T00:00:00Z"},
    )

    assert conn.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM facts_deltas").fetchone()[0] == 1
