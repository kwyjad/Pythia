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


def _expected_delta_rows(csv_path: Path) -> int:
    frame = pd.read_csv(csv_path)
    deduped = frame.drop_duplicates(subset=["iso3", "as_of_date", "metric", "series_semantics"])
    return len(deduped)


def _expected_stock_rows(csv_path: Path | None) -> int:
    if csv_path is None:
        return 0
    frame = pd.read_csv(csv_path)
    deduped = frame.drop_duplicates(subset=["iso3", "as_of_date", "metric", "series_semantics"])
    return len(deduped)


def _write_canonical_facts(tmp_path: Path) -> Path:
    data = [
        {
            "iso3": "COL",
            "as_of_date": "2024-02-29",
            "metric": "new_displacements",
            "value": 150,
            "series_semantics": "new",
            "source": "IDMC",
        },
        {
            "iso3": "COL",
            "as_of_date": "2024-02-29",
            "metric": "idp_displacement_stock_idmc",
            "value": 250,
            "series_semantics": "stock",
            "source": "IDMC",
        },
    ]
    frame = pd.DataFrame(data)
    path = tmp_path / "facts.csv"
    frame.to_csv(path, index=False)
    return path


def _write_empty_canonical_facts(tmp_path: Path) -> Path:
    path = tmp_path / "facts_empty.csv"
    frame = pd.DataFrame(
        columns=[
            "iso3",
            "as_of_date",
            "metric",
            "value",
            "series_semantics",
            "source",
        ]
    )
    frame.to_csv(path, index=False)
    return path


@pytest.mark.duckdb
def test_cli_dry_run_with_facts_csv(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    facts_path = _write_canonical_facts(tmp_path / "facts_dry_run")
    db_path = tmp_path / "resolver_dry.duckdb"
    out_dir = tmp_path / "dry_run_out"

    code = idmc_to_duckdb.run(
        [
            "--facts-csv",
            str(facts_path),
            "--db-url",
            str(db_path),
            "--out",
            str(out_dir),
        ]
    )
    assert code == 0
    output = capfd.readouterr().out
    assert "✅ Wrote 0 rows to DuckDB (dry-run)" in output
    assert not db_path.exists()


@pytest.mark.duckdb
def test_cli_writes_rows_from_facts_csv(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    facts_path = _write_canonical_facts(tmp_path / "facts_write")
    db_path = tmp_path / "resolver_write.duckdb"
    out_dir = tmp_path / "write_out"

    code = idmc_to_duckdb.run(
        [
            "--facts-csv",
            str(facts_path),
            "--db-url",
            str(db_path),
            "--out",
            str(out_dir),
            "--write-db",
        ]
    )
    assert code == 0
    output = capfd.readouterr().out
    assert "✅ Wrote" in output and "(dry-run)" not in output

    conn = duckdb_io.get_db(_db_url(db_path))
    try:
        deltas_count = conn.execute("SELECT COUNT(*) FROM facts_deltas").fetchone()[0]
        resolved_count = conn.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0]
    finally:
        conn.close()
    assert deltas_count > 0
    assert resolved_count > 0


@pytest.mark.duckdb
def test_cli_exits_nonzero_for_empty_facts(tmp_path: Path, capfd: pytest.CaptureFixture[str]) -> None:
    facts_path = _write_empty_canonical_facts(tmp_path / "facts_empty")
    db_path = tmp_path / "resolver_empty.duckdb"
    out_dir = tmp_path / "empty_out"

    code = idmc_to_duckdb.run(
        [
            "--facts-csv",
            str(facts_path),
            "--db-url",
            str(db_path),
            "--out",
            str(out_dir),
            "--write-db",
        ]
    )
    assert code == 4
    output = capfd.readouterr().out
    assert "✅ Wrote 0 rows to DuckDB" in output


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
            "--write-db",
        ]
    )
    assert code == 0
    output = capfd.readouterr().out
    assert "✅ Wrote" in output
    assert db_path.exists()

    conn = duckdb_io.get_db(_db_url(db_path))
    try:
        deltas_count = conn.execute("SELECT COUNT(*) FROM facts_deltas").fetchone()[0]
        assert deltas_count == _expected_delta_rows(fixture.flow_csv)
        resolved_count = conn.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0]
        assert resolved_count == _expected_stock_rows(fixture.stock_csv)
        value = conn.execute(
            """
            SELECT value_new
            FROM facts_deltas
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
            "--write-db",
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
            "--write-db",
        ]
    )
    assert second == 0
    output = capfd.readouterr().out
    assert "✅ Wrote 0 rows" in output

    conn = duckdb_io.get_db(_db_url(db_path))
    try:
        deltas_count = conn.execute("SELECT COUNT(*) FROM facts_deltas").fetchone()[0]
        assert deltas_count == _expected_delta_rows(fixture.flow_csv)
        resolved_count = conn.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0]
        assert resolved_count == _expected_stock_rows(fixture.stock_csv)
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
            "--write-db",
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
            "--write-db",
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
            "--write-db",
        ]
    )
    assert code == 0

    conn = duckdb_io.get_db(_db_url(db_path))
    try:
        resolved_columns = _fetch(conn, "PRAGMA table_info('facts_resolved')")
        deltas_columns = _fetch(conn, "PRAGMA table_info('facts_deltas')")
        resolved_names = [row[1] for row in resolved_columns]
        deltas_names = [row[1] for row in deltas_columns]
        for required in ["ym", "iso3", "metric", "series_semantics", "value"]:
            assert required in resolved_names
        for required in ["ym", "iso3", "metric", "series_semantics", "value_new"]:
            assert required in deltas_names
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
            "--write-db",
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
            "--write-db",
        ]
    )
    assert second == 0
    output = capfd.readouterr().out
    assert "✅ Wrote 0 rows" in output

    conn = duckdb_io.get_db(_db_url(db_path))
    try:
        value = conn.execute(
            """
            SELECT value_new
            FROM facts_deltas
            WHERE iso3='COL' AND ym='2024-02' AND metric='new_displacements'
            """
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
