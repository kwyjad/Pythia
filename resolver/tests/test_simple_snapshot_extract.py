from __future__ import annotations

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")


def _make_db_with_facts(tmp_path: Path, ym: str) -> Path:
    db_path = tmp_path / "resolver.duckdb"
    con = duckdb.connect(db_path.as_posix())
    con.execute(
        """
        CREATE TABLE facts_resolved (
            ym TEXT,
            iso3 TEXT,
            metric TEXT,
            series_semantics TEXT,
            value DOUBLE
        );
        """
    )
    con.execute(
        """
        CREATE TABLE facts_deltas (
            ym TEXT,
            iso3 TEXT,
            metric TEXT,
            series_semantics TEXT,
            value DOUBLE
        );
        """
    )
    con.execute(
        "INSERT INTO facts_resolved VALUES (?, 'AFG', 'affected', 'stock', 1000.0)",
        [ym],
    )
    con.execute(
        "INSERT INTO facts_deltas VALUES (?, 'AFG', 'new_displacements', 'new', 50.0)",
        [ym],
    )
    con.close()
    return db_path


def test_write_simple_snapshot_writes_parquet(tmp_path: Path) -> None:
    from resolver.cli.simple_snapshot import write_simple_snapshot  # type: ignore

    ym = "2025-10"
    db_path = _make_db_with_facts(tmp_path, ym)
    out_root = tmp_path / "snapshots" / "simple"

    write_simple_snapshot(db_path, ym, out_root, include_deltas=True)

    facts_resolved_path = out_root / ym / "facts_resolved.parquet"
    facts_deltas_path = out_root / ym / "facts_deltas.parquet"
    assert facts_resolved_path.exists()
    assert facts_deltas_path.exists()

    con = duckdb.connect()
    try:
        res_rows = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [str(facts_resolved_path)]
        ).fetchone()[0]
        delta_rows = con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [str(facts_deltas_path)]
        ).fetchone()[0]
    finally:
        con.close()

    assert res_rows == 1
    assert delta_rows == 1


def test_write_simple_snapshot_handles_missing_deltas(tmp_path: Path) -> None:
    from resolver.cli.simple_snapshot import write_simple_snapshot  # type: ignore

    ym = "2025-10"
    db_path = tmp_path / "resolver.duckdb"
    con = duckdb.connect(db_path.as_posix())
    con.execute(
        """
        CREATE TABLE facts_resolved (
            ym TEXT,
            iso3 TEXT,
            metric TEXT,
            series_semantics TEXT,
            value DOUBLE
        );
        """
    )
    con.execute(
        "INSERT INTO facts_resolved VALUES (?, 'AFG', 'affected', 'stock', 1000.0)",
        [ym],
    )
    con.close()

    out_root = tmp_path / "snapshots" / "simple"
    write_simple_snapshot(db_path, ym, out_root, include_deltas=True)

    facts_resolved_path = out_root / ym / "facts_resolved.parquet"
    facts_deltas_path = out_root / ym / "facts_deltas.parquet"

    assert facts_resolved_path.exists()
    assert not facts_deltas_path.exists()
