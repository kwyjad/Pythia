from typing import List, Tuple

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.ci.verify_duckdb_counts import _fetch_breakdown


def test_fetch_breakdown_handles_null_source_and_group_by_alias_safely() -> None:
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            """
            CREATE TABLE facts_resolved (
                source VARCHAR,
                metric VARCHAR,
                series_semantics VARCHAR
            );
            """
        )
        con.execute(
            "INSERT INTO facts_resolved VALUES "
            "(NULL, 'idp_displacement_stock_dtm', 'stock'),"
            "('IOM DTM', 'idp_displacement_stock_dtm', 'stock'),"
            "('emdat', 'affected', 'new');"
        )

        rows: List[Tuple[str, str, str, int]] = _fetch_breakdown(con)
        result = {(src, metric, semantics): count for src, metric, semantics, count in rows}

        assert result[("", "idp_displacement_stock_dtm", "stock")] == 1
        assert result[("IOM DTM", "idp_displacement_stock_dtm", "stock")] == 1
        assert result[("emdat", "affected", "new")] == 1
    finally:
        con.close()
