# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from pathlib import Path

import pytest

from resolver.diagnostics import odp_smoke
from resolver.db import duckdb_io
from resolver.db._duckdb_available import DUCKDB_AVAILABLE
from resolver.ingestion import odp_series


pytestmark = pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="DuckDB not installed")


def _create_sample_db(db_path: Path) -> None:
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    try:
        conn.execute(
            """
            CREATE TABLE odp_timeseries_raw (
                source_id TEXT,
                iso3 TEXT,
                origin_iso3 TEXT,
                admin_name TEXT,
                ym TEXT,
                as_of_date DATE,
                metric TEXT,
                series_semantics TEXT,
                value DOUBLE,
                unit TEXT,
                extra TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO odp_timeseries_raw
            VALUES
            ('src', 'AAA', NULL, NULL, '2023-01', DATE '2023-01-31', 'metric_one', 'stock', 10, 'persons', '{}'),
            ('src', 'BBB', NULL, NULL, '2023-02', DATE '2023-02-28', 'metric_two', 'stock', 5, 'persons', '{}')
            """
        )
    finally:
        duckdb_io.close_db(conn)


def test_build_smoke_summary_success(tmp_path: Path) -> None:
    config_path = tmp_path / "unhcr_odp.yml"
    config_path.write_text("pages: []\n", encoding="utf-8")
    normalizers_path = tmp_path / "normalizers.yml"
    normalizers_path.write_text("defaults: {}\n", encoding="utf-8")
    diag_dir = tmp_path / "diagnostics"
    artifact = diag_dir / "discovery.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("{}", encoding="utf-8")

    db_path = tmp_path / "odp.duckdb"
    _create_sample_db(db_path)

    stats = odp_series.OdpPipelineStats(
        config_pages=1,
        pages_discovered=1,
        json_links_found=2,
        json_links_matched=1,
        json_links_unmatched=1,
        raw_records_total=3,
        normalized_rows_total=2,
        normalized_rows_per_series={"src": 2},
        unmatched_labels=["Unmatched widget"],
    )

    summary = odp_smoke.build_smoke_summary(
        config_path=config_path,
        normalizers_path=normalizers_path,
        db_path=db_path,
        diagnostics_dir=diag_dir,
        rows=2,
        error=None,
        traceback_text=None,
        stats=stats,
    )

    assert "Outcome: SUCCESS" in summary
    assert "odp_timeseries_raw total rows: 2" in summary
    assert "| AAA | metric_one | 1 | 2023-01 | 2023-01 |" in summary
    assert "Config pages: 1" in summary
    assert "JSON links found: 2" in summary
    assert "Normalized rows total: 2" in summary
    assert "| src | 2 |" in summary
    assert "discovery.json" in summary

    out_path = tmp_path / "diagnostics" / "odp_smoke_summary.md"
    written = odp_smoke.write_smoke_summary(summary, out_path)
    assert written.exists()
    assert written.read_text(encoding="utf-8").startswith("# ODP smoke summary")


def test_build_smoke_summary_failure_includes_error(tmp_path: Path) -> None:
    summary = odp_smoke.build_smoke_summary(
        config_path=tmp_path / "missing_config.yml",
        normalizers_path=tmp_path / "missing_norm.yml",
        db_path=tmp_path / "no_db.duckdb",
        diagnostics_dir=tmp_path / "diag",
        rows=None,
        error=RuntimeError("ODP discovery config not found"),
        traceback_text="traceback sample",
        stats=None,
    )

    assert "Outcome: FAILED" in summary
    assert "RuntimeError" in summary
    assert "DuckDB exists: no" in summary
    assert "traceback sample" in summary


def test_build_smoke_summary_handles_empty_db(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.duckdb"
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    try:
        conn.execute(
            """
            CREATE TABLE odp_timeseries_raw (
                source_id TEXT,
                iso3 TEXT,
                origin_iso3 TEXT,
                admin_name TEXT,
                ym TEXT,
                as_of_date DATE,
                metric TEXT,
                series_semantics TEXT,
                value DOUBLE,
                unit TEXT,
                extra TEXT
            )
            """
        )
    finally:
        duckdb_io.close_db(conn)

    summary = odp_smoke.build_smoke_summary(
        config_path=tmp_path / "unhcr_odp.yml",
        normalizers_path=tmp_path / "normalizers.yml",
        db_path=db_path,
        diagnostics_dir=tmp_path / "diag",
        rows=0,
        error=None,
        traceback_text=None,
        stats=odp_series.OdpPipelineStats(normalized_rows_total=0),
    )

    assert "DuckDB exists: yes" in summary
    assert "odp_timeseries_raw total rows: 0" in summary
    assert "(no rows in odp_timeseries_raw)" in summary
