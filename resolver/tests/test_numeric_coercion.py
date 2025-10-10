import pandas as pd

from resolver.db import duckdb_io


def test_numeric_coercion_none_and_blanks(tmp_path):
    db_path = tmp_path / "coerce.duckdb"
    url = f"duckdb:///{db_path}"
    conn = duckdb_io.get_db(url)
    duckdb_io.init_schema(conn)

    resolved = pd.DataFrame(
        [
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "series_semantics": "stock",
                "value": "None",
            }
        ]
    )
    deltas = pd.DataFrame(
        [
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "series_semantics": "new",
                "value_new": "",
                "value_stock": "None",
            },
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "affected",
                "series_semantics": "new",
                "value_new": "  123  ",
                "value_stock": "  ",
            },
        ]
    )

    duckdb_io.write_snapshot(
        conn,
        ym="2024-02",
        facts_resolved=resolved,
        facts_deltas=deltas,
        manifests=None,
        meta={},
    )

    value = conn.execute(
        """
        SELECT value
        FROM facts_resolved
        WHERE ym='2024-02' AND iso3='PHL' AND hazard_code='TC' AND metric='in_need'
        """
    ).fetchone()[0]
    assert value is None

    rows = conn.execute(
        """
        SELECT metric, value_new, value_stock
        FROM facts_deltas
        WHERE ym='2024-02' AND iso3='PHL' AND hazard_code='TC'
        ORDER BY metric
        """
    ).fetchall()

    got = {metric: (value_new, value_stock) for metric, value_new, value_stock in rows}
    assert got["affected"][0] == 123.0
    assert got["affected"][1] is None
    assert got["in_need"][0] is None
    assert got["in_need"][1] is None
