import pytest

from pathlib import Path

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
