from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.cli import idmc_to_duckdb
from resolver.db import duckdb_io
from resolver.tests.utils.runtime_fixtures import create_idmc_runtime_fixture


def _db_url(db_path: Path) -> str:
    return f"duckdb:///{db_path}" if not str(db_path).startswith("duckdb://") else str(db_path)


def _fetch(conn, sql: str):
    return conn.execute(sql).fetchall()


def _expected_resolved_rows(csv_path: Path) -> int:
    frame = pd.read_csv(csv_path)
    deduped = frame.drop_duplicates(subset=["iso3", "as_of_date", "metric", "series_semantics"])
    return len(deduped)


@pytest.mark.duckdb
def test_cli_writes_rows_once(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    fixture = create_idmc_runtime_fixture(tmp_path / "case_write")
    staging_root = fixture.staging_dir
    db_path = tmp_path / "resolver.duckdb"
    out_dir = tmp_path / "preview"

    code = idmc_to_duckdb.run(
        [
            "--staging-dir",
            str(staging_root),
            "--db-url",
            str(db_path),
            "--out",
            str(out_dir),
            "--log-level",
            "INFO",
        ]
    )
    assert code == 0
    output = capfd.readouterr().out
    assert "✅ Wrote" in output
    assert db_path.exists()

    conn = duckdb_io.get_db(_db_url(db_path))
    try:
        count = conn.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0]
        assert count == _expected_resolved_rows(fixture.flow_csv)
        value = conn.execute(
            """
            SELECT value
            FROM facts_resolved
            WHERE iso3='COL' AND ym='2024-02' AND metric='new_displacements'
            """
        ).fetchone()[0]
        assert pytest.approx(value) == 150.0
    finally:
        conn.close()


@pytest.mark.duckdb
def test_cli_is_idempotent(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    fixture = create_idmc_runtime_fixture(tmp_path / "case_idempotent")
    staging_root = fixture.staging_dir
    db_path = tmp_path / "resolver.duckdb"
    out_dir = tmp_path / "preview"

    first = idmc_to_duckdb.run(
        [
            "--staging-dir",
            str(staging_root),
            "--db-url",
            str(db_path),
            "--out",
            str(out_dir),
            "--log-level",
            "INFO",
        ]
    )
    assert first == 0
    capfd.readouterr()
    second = idmc_to_duckdb.run(
        [
            "--staging-dir",
            str(staging_root),
            "--db-url",
            str(db_path),
            "--out",
            str(out_dir),
            "--log-level",
            "INFO",
        ]
    )
    assert second == 0
    output = capfd.readouterr().out
    assert "✅ Wrote 0 rows" in output

    conn = duckdb_io.get_db(_db_url(db_path))
    try:
        count = conn.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0]
        assert count == _expected_resolved_rows(fixture.flow_csv)
    finally:
        conn.close()


@pytest.mark.duckdb
def test_cli_warns_on_missing_stock_not_fail(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    fixture = create_idmc_runtime_fixture(
        tmp_path / "case_missing_stock",
        include_stock=False,
    )
    staging_root = fixture.staging_dir
    db_path = tmp_path / "resolver.duckdb"
    out_dir = tmp_path / "preview"

    code = idmc_to_duckdb.run(
        [
            "--staging-dir",
            str(staging_root),
            "--db-url",
            str(db_path),
            "--out",
            str(out_dir),
            "--log-level",
            "INFO",
        ]
    )
    assert code == 0
    output = capfd.readouterr().out
    assert "Warnings:" in output


@pytest.mark.duckdb
def test_cli_strict_mode_fails_on_warning(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    fixture = create_idmc_runtime_fixture(
        tmp_path / "case_strict_warning",
        include_stock=False,
    )
    staging_root = fixture.staging_dir
    db_path = tmp_path / "resolver.duckdb"
    out_dir = tmp_path / "preview"

    code = idmc_to_duckdb.run(
        [
            "--staging-dir",
            str(staging_root),
            "--db-url",
            str(db_path),
            "--out",
            str(out_dir),
            "--log-level",
            "INFO",
            "--strict",
        ]
    )
    assert code == 2
    output = capfd.readouterr().out
    assert "Warnings:" in output


@pytest.mark.duckdb
def test_db_contains_expected_columns(tmp_path: Path) -> None:
    fixture = create_idmc_runtime_fixture(tmp_path / "case_columns")
    staging_root = fixture.staging_dir
    db_path = tmp_path / "resolver.duckdb"
    out_dir = tmp_path / "preview"

    code = idmc_to_duckdb.run(
        [
            "--staging-dir",
            str(staging_root),
            "--db-url",
            str(db_path),
            "--out",
            str(out_dir),
            "--log-level",
            "INFO",
        ]
    )
    assert code == 0

    conn = duckdb_io.get_db(_db_url(db_path))
    try:
        columns = _fetch(conn, "PRAGMA table_info('facts_resolved')")
        names = [row[1] for row in columns]
        for required in ["ym", "iso3", "metric", "series_semantics", "value"]:
            assert required in names
    finally:
        conn.close()


@pytest.mark.duckdb
def test_merge_updates_on_conflict(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    fixture = create_idmc_runtime_fixture(tmp_path / "case_update")
    staging_root = fixture.staging_dir
    db_path = tmp_path / "resolver.duckdb"
    out_dir = tmp_path / "preview"

    first = idmc_to_duckdb.run(
        [
            "--staging-dir",
            str(staging_root),
            "--db-url",
            str(db_path),
            "--out",
            str(out_dir),
            "--log-level",
            "INFO",
        ]
    )
    assert first == 0
    capfd.readouterr()

    # Update flow CSV to change a value for an existing key
    flow_path = staging_root / "flow.csv"
    df = pd.read_csv(flow_path)
    mask = df["iso3"].eq("COL") & df["as_of_date"].eq("2024-02-29")
    df.loc[mask, "value"] = 175
    df.to_csv(flow_path, index=False)

    second = idmc_to_duckdb.run(
        [
            "--staging-dir",
            str(staging_root),
            "--db-url",
            str(db_path),
            "--out",
            str(out_dir),
            "--log-level",
            "INFO",
        ]
    )
    assert second == 0
    output = capfd.readouterr().out
    assert "✅ Wrote 0 rows" in output

    conn = duckdb_io.get_db(_db_url(db_path))
    try:
        value = conn.execute(
            "SELECT value FROM facts_resolved WHERE iso3='COL' AND ym='2024-02'"
        ).fetchone()[0]
        assert pytest.approx(value) == 175.0
    finally:
        conn.close()


@pytest.mark.duckdb
def test_parquet_fixture_created_or_skipped(tmp_path: Path) -> None:
    fixture = create_idmc_runtime_fixture(tmp_path / "case_parquet")
    if not fixture.parquet_available:
        pytest.skip("Parquet engine unavailable; fixture written as CSV only")

    assert fixture.parquet_path is not None
    assert fixture.parquet_path.exists()
