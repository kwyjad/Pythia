# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests covering the DuckDB MERGE fallback path."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

_ = pytest.importorskip("duckdb")

from resolver.db import duckdb_io


@pytest.mark.duckdb
def test_merge_fallback_no_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Ensure MERGE fallback completes without emitting error logs."""

    monkeypatch.setenv("RESOLVER_DUCKDB_DISABLE_MERGE", "1")
    caplog.set_level("INFO", logger=duckdb_io.LOGGER.name)

    db_path = tmp_path / "merge_fallback.duckdb"
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    try:
        duckdb_io.init_schema(conn)

        frame = pd.DataFrame(
            [
                dict(
                    ym="2024-01",
                    iso3="PHL",
                    hazard_code="TC",
                    metric="in_need",
                    series_semantics="stock",
                    value=100,
                ),
                dict(
                    ym="2024-01",
                    iso3="PHL",
                    hazard_code="EQ",
                    metric="affected",
                    series_semantics="",
                    value=50,
                ),
            ]
        )

        rows_inserted = duckdb_io.upsert_dataframe(
            conn,
            "facts_resolved",
            frame,
            keys=duckdb_io.FACTS_RESOLVED_KEY_COLUMNS,
        )
        assert rows_inserted.rows_written == len(frame)
        assert rows_inserted.rows_delta == len(frame)

        frame_update = frame.copy()
        frame_update.loc[frame_update["hazard_code"].eq("TC"), "value"] = 120
        rows_updated = duckdb_io.upsert_dataframe(
            conn,
            "facts_resolved",
            frame_update,
            keys=duckdb_io.FACTS_RESOLVED_KEY_COLUMNS,
        )
        assert rows_updated.rows_written == len(frame_update)

        final_count = conn.execute(
            "SELECT COUNT(*) FROM facts_resolved WHERE ym='2024-01'"
        ).fetchone()[0]
        assert final_count == len(frame)

        assert not any(record.levelname == "ERROR" for record in caplog.records)
        assert any(
            record.message.startswith("duckdb.upsert.legacy_delete")
            or record.message.startswith("duckdb.upsert.legacy_insert")
            for record in caplog.records
        )
    finally:
        conn.close()
