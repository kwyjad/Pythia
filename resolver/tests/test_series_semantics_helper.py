import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.common import compute_series_semantics
from resolver.db import duckdb_io
from resolver.query import db_reader


def test_stock_metric_overrides_blank_existing():
    assert compute_series_semantics("in_need", "") == "stock"
    assert compute_series_semantics("in_need", None) == "stock"


def test_existing_value_preserved():
    assert compute_series_semantics("affected", "incident") == "incident"


def test_blank_metric_defaults_to_empty_string():
    assert compute_series_semantics(None, None) == ""
    assert compute_series_semantics("", "  ") == ""


def test_writer_rejects_blank_semantics(tmp_path):
    db_path = tmp_path / "blank_semantics.duckdb"
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    duckdb_io.init_schema(conn)

    resolved = pd.DataFrame(
        [
            {
                "ym": "2024-01",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "custom_metric",
                "series_semantics": "",
                "value": 10,
            }
        ]
    )

    with pytest.raises(ValueError, match="series_semantics must be 'new' or 'stock'"):
        duckdb_io.write_snapshot(
            conn,
            ym="2024-01",
            facts_resolved=resolved,
            facts_deltas=None,
            manifests=None,
            meta=None,
        )


def test_reader_coalesces_series_semantics_and_series(tmp_path):
    db_path = tmp_path / "reader_semantics.duckdb"
    url = f"duckdb:///{db_path}"
    conn = duckdb_io.get_db(url)
    duckdb_io.init_schema(conn)

    conn.execute(
        """
        INSERT INTO facts_resolved (
            ym, iso3, hazard_code, metric, series_semantics, series, value, as_of_date
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "2024-01",
            "PHL",
            "TC",
            "in_need",
            "",
            "stock",
            100,
            "2024-01-31",
        ],
    )
    conn.execute(
        """
        INSERT INTO facts_deltas (
            ym, iso3, hazard_code, metric, value_new, value_stock, series_semantics, series, as_of
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "2024-01",
            "PHL",
            "TC",
            "in_need",
            100,
            100,
            "",
            "new",
            "2024-01-31",
        ],
    )

    resolved_row = db_reader.fetch_resolved_point(
        conn,
        ym="2024-01",
        iso3="PHL",
        hazard_code="TC",
        cutoff="2024-02-01",
        preferred_metric="in_need",
    )
    assert resolved_row is not None
    assert resolved_row.get("series_semantics") == "stock"

    deltas_row = db_reader.fetch_deltas_point(
        conn,
        ym="2024-01",
        iso3="PHL",
        hazard_code="TC",
        cutoff="2024-02-01",
        preferred_metric="in_need",
    )
    assert deltas_row is not None
    assert deltas_row.get("series_semantics") == "new"
