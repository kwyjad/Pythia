import pytest

from resolver.db import duckdb_io
from resolver.ingestion.emdat_normalize import (
    normalize_emdat_pa,
    write_emdat_pa_to_duckdb,
)
from resolver.ingestion.emdat_stub import fetch_raw


@pytest.mark.duckdb
def test_emdat_duckdb_write_idempotent(tmp_path):
    pytest.importorskip("duckdb")

    db_path = tmp_path / "emdat.duckdb"
    db_url = f"duckdb:///{db_path.as_posix()}"
    conn = duckdb_io.get_db(db_url)
    try:
        raw = fetch_raw(2010, 2030)
        normalized = normalize_emdat_pa(
            raw, info={"timestamp": "2024-01-31T00:00:00Z"}
        )

        first = write_emdat_pa_to_duckdb(conn, normalized)
        assert first.rows_written == len(normalized)

        second = write_emdat_pa_to_duckdb(conn, normalized)
        assert second.rows_delta == 0

        row_count = conn.execute("SELECT COUNT(*) FROM emdat_pa").fetchone()[0]
        assert row_count == len(normalized)

        columns = [row[1] for row in conn.execute("PRAGMA table_info('emdat_pa')").fetchall()]
        for column in (
            "iso3",
            "ym",
            "shock_type",
            "pa",
            "as_of_date",
            "publication_date",
            "source_id",
            "disno_first",
        ):
            assert column in columns

        pa_values = [value for (value,) in conn.execute("SELECT pa FROM emdat_pa").fetchall()]
        assert all(isinstance(value, int) for value in pa_values)
        assert all(value >= 0 for value in pa_values)

        indexes = conn.execute(
            "SELECT index_name FROM duckdb_indexes() WHERE table_name = 'emdat_pa'"
        ).fetchall()
        assert any(name == "ux_emdat_pa" for (name,) in indexes)
    finally:
        duckdb_io.close_db(conn)
