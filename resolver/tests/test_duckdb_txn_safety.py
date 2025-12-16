# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests ensuring DuckDB schema initialisation is transaction safe."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("duckdb")

from resolver.db import duckdb_io


_EXPECTED_TABLES = {
    "facts_resolved",
    "facts_deltas",
    "manifests",
    "meta_runs",
    "snapshots",
}


def _unique_index_count(conn, table: str, index: str) -> int:
    query = (
        "SELECT COUNT(*) FROM duckdb_indexes() "
        "WHERE table_name = ? AND index_name = ?"
    )
    return int(conn.execute(query, [table, index]).fetchone()[0])


def test_init_schema_transaction_safe(tmp_path: Path) -> None:
    db_path = tmp_path / "txn_safe.duckdb"
    conn = duckdb_io.get_db(str(db_path))
    try:
        duckdb_io.init_schema(conn)
        # A second invocation should be idempotent and avoid nested transactions.
        duckdb_io.init_schema(conn)

        tables = {
            row[0]
            for row in conn.execute("PRAGMA show_tables").fetchall()
        }
        assert _EXPECTED_TABLES.issubset(tables)

        resolved_index = _unique_index_count(
            conn, "facts_resolved", "ux_facts_resolved_series"
        )
        deltas_index = _unique_index_count(
            conn, "facts_deltas", "ux_facts_deltas_series"
        )
        assert resolved_index == 1
        assert deltas_index == 1
    finally:
        conn.close()
