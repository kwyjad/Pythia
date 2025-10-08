import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.db import duckdb_io


@pytest.fixture()
def duck_conn(tmp_path):
    db_path = tmp_path / "argorder.duckdb"
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    duckdb_io.init_schema(conn)
    try:
        yield conn
    finally:
        conn.close()


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ym": "2024-01",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "series_semantics": "stock",
                "value": 100.0,
                "as_of_date": "2024-01-10",
                "publication_date": "2024-01-11",
            }
        ]
    )


def _count_rows(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0]


def test_upsert_accepts_table_df_order(duck_conn) -> None:
    frame = _sample_frame()
    inserted = duckdb_io.upsert_dataframe(
        duck_conn,
        "facts_resolved",
        frame,
        keys=duckdb_io.FACTS_RESOLVED_KEY_COLUMNS,
    )
    assert inserted == len(frame)
    assert _count_rows(duck_conn) == len(frame)


def test_upsert_accepts_df_table_order(duck_conn) -> None:
    frame = _sample_frame()
    inserted = duckdb_io.upsert_dataframe(
        duck_conn,
        frame,
        "facts_resolved",
        keys=duckdb_io.FACTS_RESOLVED_KEY_COLUMNS,
    )
    assert inserted == len(frame)
    assert _count_rows(duck_conn) == len(frame)
