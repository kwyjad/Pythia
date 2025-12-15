# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import pytest

from resolver.tests.fixtures.bootstrap_fast_exports import build_fast_exports

pytest.importorskip("duckdb")


def test_build_fast_exports_creates_artifacts(tmp_path):
    bundle = build_fast_exports()
    assert bundle.snapshots_period_dir.joinpath("facts_deltas.parquet").exists()
    assert bundle.snapshots_period_dir.joinpath("facts_resolved.parquet").exists()
    assert bundle.exports_dir.joinpath("facts_deltas.csv").exists()
    assert bundle.exports_dir.joinpath("facts_resolved.csv").exists()


def test_build_fast_exports_duckdb_tables_have_rows():
    import duckdb

    bundle = build_fast_exports()
    con = duckdb.connect(str(bundle.db_path))
    try:
        tables = {row[0] for row in con.execute("SHOW TABLES").fetchall()}
        assert "facts_resolved" in tables
        assert "facts_deltas" in tables
        resolved_count = con.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0]
        deltas_count = con.execute("SELECT COUNT(*) FROM facts_deltas").fetchone()[0]
        assert resolved_count > 0
        assert deltas_count > 0
    finally:
        con.close()
