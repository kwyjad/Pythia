import pandas as pd

from resolver.db import duckdb_io


def test_init_schema_declares_keys(tmp_path):
    url = f"duckdb:///{(tmp_path / 'keys.duckdb').as_posix()}"
    conn = duckdb_io.get_db(url)
    duckdb_io.init_schema(conn)

    assert duckdb_io._has_declared_key(
        conn,
        "facts_resolved",
        duckdb_io.FACTS_RESOLVED_KEY_COLUMNS,
    )
    assert duckdb_io._has_declared_key(
        conn,
        "facts_deltas",
        duckdb_io.FACTS_DELTAS_KEY_COLUMNS,
    )


def test_upsert_heals_missing_indexes(tmp_path):
    url = f"duckdb:///{(tmp_path / 'heal.duckdb').as_posix()}"
    conn = duckdb_io.get_db(url)

    conn.execute(
        """
        CREATE TABLE facts_resolved (
            ym VARCHAR,
            iso3 VARCHAR,
            hazard_code VARCHAR,
            metric VARCHAR,
            series_semantics VARCHAR,
            value DOUBLE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE facts_deltas (
            ym VARCHAR,
            iso3 VARCHAR,
            hazard_code VARCHAR,
            metric VARCHAR,
            value_new DOUBLE,
            value_stock DOUBLE
        )
        """
    )

    resolved_frame = pd.DataFrame(
        [
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "series_semantics": "stock",
                "value": 1.0,
            }
        ]
    )
    deltas_frame = pd.DataFrame(
        [
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "value_new": 1.0,
                "value_stock": None,
            }
        ]
    )

    duckdb_io.upsert_dataframe(
        conn,
        "facts_resolved",
        resolved_frame,
        keys=duckdb_io.FACTS_RESOLVED_KEY_COLUMNS,
    )
    duckdb_io.upsert_dataframe(
        conn,
        "facts_deltas",
        deltas_frame,
        keys=duckdb_io.FACTS_DELTAS_KEY_COLUMNS,
    )

    assert duckdb_io._has_declared_key(
        conn,
        "facts_resolved",
        duckdb_io.FACTS_RESOLVED_KEY_COLUMNS,
    )
    assert duckdb_io._has_declared_key(
        conn,
        "facts_deltas",
        duckdb_io.FACTS_DELTAS_KEY_COLUMNS,
    )

    assert conn.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM facts_deltas").fetchone()[0] == 1
