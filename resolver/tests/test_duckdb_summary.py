import logging
from pathlib import Path

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")

from resolver.db import duckdb_io
from scripts.ci import duckdb_summary


@pytest.mark.duckdb
def test_duckdb_summary_uses_shared_connection(tmp_path: Path, monkeypatch, capsys) -> None:
    db_path = tmp_path / "summary.duckdb"
    real_conn = duckdb.connect(str(db_path))
    try:
        real_conn.execute(
            """
            CREATE TABLE facts_resolved (
                ym TEXT,
                iso3 TEXT,
                hazard_code TEXT,
                metric TEXT,
                series_semantics TEXT,
                value DOUBLE,
                source_id TEXT
            )
            """
        )
        real_conn.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, series_semantics, value, source_id)
            VALUES ('2024-02', 'COL', 'hz', 'new_displacements', 'new', 150.0, 'IDMC')
            """
        )
        real_conn.execute(
            """
            CREATE TABLE facts_deltas (
                ym TEXT,
                iso3 TEXT,
                hazard_code TEXT,
                metric TEXT,
                value_new DOUBLE,
                value_stock DOUBLE,
                series_semantics TEXT,
                source_id TEXT
            )
            """
        )
        real_conn.execute(
            """
            INSERT INTO facts_deltas (ym, iso3, hazard_code, metric, value_new, value_stock, series_semantics, source_id)
            VALUES ('2024-02', 'COL', 'hz', 'new_displacements', 150.0, 150.0, 'new', 'IDMC')
            """
        )
    finally:
        real_conn.close()

    captured_url: dict[str, str] = {}

    def fake_get_db(url: str | None) -> duckdb.DuckDBPyConnection:
        captured_url["value"] = url or ""
        return duckdb.connect(str(db_path))

    closed_flag = {"called": False}

    def fake_close_db(conn: duckdb.DuckDBPyConnection | None) -> None:
        closed_flag["called"] = True
        if conn is not None:
            conn.close()

    monkeypatch.setattr(duckdb_io, "get_db", fake_get_db)
    monkeypatch.setattr(duckdb_io, "close_db", fake_close_db)

    rc = duckdb_summary.main(
        ["--db", str(db_path), "--tables", "facts_resolved,facts_deltas"]
    )
    assert rc == 0

    output = capsys.readouterr().out
    assert "Rows by source / metric / semantics" in output
    assert "| facts_resolved | IDMC | new_displacements | new | 1 |" in output
    assert "| facts_deltas | IDMC | new_displacements | new | 1 |" in output

    expected_url = f"duckdb:///{db_path.as_posix()}"
    assert captured_url.get("value") == expected_url
    assert closed_flag["called"] is True


@pytest.mark.duckdb
def test_write_facts_tables_logs_counts(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="resolver.db.duckdb_io")
    conn = duckdb.connect(str(tmp_path / "log-counts.duckdb"))
    try:
        resolved_df = pd.DataFrame(
            {
                "event_id": ["evt-stock"],
                "iso3": ["COL"],
                "hazard_code": ["HZ"],
                "metric": ["idps_present"],
                "series_semantics": ["stock"],
                "value": [125.0],
                "unit": ["persons"],
                "as_of_date": ["2024-01-31"],
                "publication_date": ["2024-02-01"],
                "source_id": ["dtm_admin0"],
                "ym": ["2024-01"],
            }
        )
        deltas_df = pd.DataFrame(
            {
                "event_id": ["evt-new"],
                "iso3": ["COL"],
                "hazard_code": ["HZ"],
                "metric": ["new_displacements"],
                "series_semantics": ["new"],
                "value_new": [25.0],
                "value_stock": [0.0],
                "as_of": ["2024-01-31"],
                "as_of_date": ["2024-01-31"],
                "publication_date": ["2024-02-01"],
                "source_id": ["idmc"],
                "ym": ["2024-01"],
            }
        )

        results = duckdb_io.write_facts_tables(
            conn, facts_resolved=resolved_df, facts_deltas=deltas_df
        )
    finally:
        conn.close()

    assert set(results) == {"facts_resolved", "facts_deltas"}
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "resolver.db.duckdb_io"
    ]
    assert any(
        message.startswith("duckdb.write_facts_tables.inputs")
        and "facts_resolved=1" in message
        and "facts_deltas=1" in message
        for message in messages
    )
    assert any(
        "duckdb.write_facts_tables.result | table=facts_resolved" in message
        and "rows_written=1" in message
        for message in messages
    )
    assert any(
        "duckdb.write_facts_tables.result | table=facts_deltas" in message
        and "rows_written=1" in message
        for message in messages
    )
