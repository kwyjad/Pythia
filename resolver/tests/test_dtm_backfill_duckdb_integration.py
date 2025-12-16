# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import pytest
import pandas as pd

from resolver.db import duckdb_io
from resolver.tools import export_facts


pytest.importorskip("duckdb")


def _dtm_facts_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "iso3": "COD",
                "hazard_code": "displacement",
                "as_of_date": "2025-09-30",
                "ym": "2025-09",
                "metric": "idp_displacement_stock_dtm",
                "value": 629672,
                "series_semantics": "stock",
                "source": "IOM DTM",
            },
            {
                "iso3": "SDN",
                "hazard_code": "displacement",
                "as_of_date": "2025-09-30",
                "ym": "2025-09",
                "metric": "idp_displacement_stock_dtm",
                "value": 334810,
                "series_semantics": "stock",
                "source": "IOM DTM",
            },
            {
                "iso3": "COD",
                "hazard_code": "displacement",
                "as_of_date": "2025-09-30",
                "ym": "2025-09",
                "metric": "idp_displacement_new_dtm",
                "value": 0,
                "series_semantics": "new",
                "source": "IOM DTM",
            },
            {
                "iso3": "SDN",
                "hazard_code": "displacement",
                "as_of_date": "2025-09-30",
                "ym": "2025-09",
                "metric": "idp_displacement_new_dtm",
                "value": 0,
                "series_semantics": "new",
                "source": "IOM DTM",
            },
        ]
    )


def test_dtm_duckdb_roundtrip():
    facts = _dtm_facts_frame()

    resolved_for_db, deltas_for_db = export_facts.prepare_duckdb_tables(facts)
    resolved_prepared = export_facts._prepare_resolved_for_db(resolved_for_db)
    deltas_prepared = export_facts._prepare_deltas_for_db(deltas_for_db)

    conn = duckdb_io.get_db("duckdb:///:memory:")
    try:
        duckdb_io.write_facts_tables(
            conn,
            facts_resolved=resolved_prepared,
            facts_deltas=deltas_prepared,
        )

        resolved_rows = conn.execute(
            """
            SELECT metric, series_semantics, COUNT(*) AS count
            FROM facts_resolved
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).fetchall()

        deltas_rows = conn.execute(
            """
            SELECT metric, series_semantics, COUNT(*) AS count
            FROM facts_deltas
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).fetchall()
    finally:
        duckdb_io.close_db(conn)

    assert resolved_prepared is not None and len(resolved_prepared) == 2
    assert deltas_prepared is not None and len(deltas_prepared) == 2
    assert resolved_rows == [("idp_displacement_stock_dtm", "stock", 2)]
    assert deltas_rows == [("idp_displacement_new_dtm", "new", 2)]
