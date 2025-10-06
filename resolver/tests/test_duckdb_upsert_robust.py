import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.db import duckdb_io


def test_upsert_handles_unknown_columns_and_alias(tmp_path):
    db_path = tmp_path / "test.duckdb"
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    try:
        duckdb_io.init_schema(conn)
        df = pd.DataFrame(
            [
                {
                    "ym": "2024-01",
                    "iso3": "PHL",
                    "hazard_code": "TC",
                    "metric": "in_need",
                    "value_new": 1500,
                    "value_stock": 1200,
                    "series_semantics_out": "new",
                    "as_of": "2024-01-31",
                    "source_name": "OCHA",
                    "source_url": "https://example.org",
                    "definition_text": "Pin",
                    "unknown_debug_col": "ignore-me",
                }
            ]
        )

        inserted = duckdb_io.upsert_dataframe(
            conn,
            "facts_deltas",
            df,
            keys=["ym", "iso3", "hazard_code", "metric"],
        )
        assert inserted == 1
        rows = conn.execute(
            "SELECT ym, iso3, hazard_code, metric, series_semantics, value_new FROM facts_deltas"
        ).fetchall()
        assert rows == [("2024-01", "PHL", "TC", "in_need", "new", 1500.0)]
    finally:
        conn.close()
