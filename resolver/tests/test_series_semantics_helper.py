# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

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


def test_writer_normalizes_keys(tmp_path):
    db_path = tmp_path / "normalize_keys.duckdb"
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    duckdb_io.init_schema(conn)

    invalid = pd.DataFrame(
        [
            {
                "ym": "2024-2",
                "iso3": "phl",
                "hazard_code": "tc",
                "metric": "custom_metric",
                "series_semantics": "stock",
                "value": 10,
            }
        ]
    )

    with pytest.raises(ValueError, match="invalid ym format"):
        duckdb_io.write_snapshot(
            conn,
            ym="2024-02",
            facts_resolved=invalid,
            facts_deltas=None,
            manifests=None,
            meta=None,
        )

    valid = invalid.assign(ym="2024-02")
    duckdb_io.write_snapshot(
        conn,
        ym="2024-02",
        facts_resolved=valid,
        facts_deltas=None,
        manifests=None,
        meta=None,
    )

    stored = conn.execute(
        "SELECT ym, iso3, hazard_code FROM facts_resolved WHERE ym = '2024-02'"
    ).fetchall()
    conn.close()

    assert stored == [("2024-02", "PHL", "TC")]


def test_writer_uppercases_codes(tmp_path):
    db_path = tmp_path / "uppercase_codes.duckdb"
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    duckdb_io.init_schema(conn)

    resolved = pd.DataFrame(
        [
            {
                "ym": "2024-03",
                "iso3": "phl",
                "hazard_code": "tc",
                "metric": "in_need",
                "series_semantics": "",
                "value": 25,
            }
        ]
    )
    deltas = pd.DataFrame(
        [
            {
                "ym": "2024-03",
                "iso3": "Phl",
                "hazard_code": "tC",
                "metric": "in_need",
                "value_new": 5,
                "value_stock": 30,
                "series_semantics": "",
            }
        ]
    )

    duckdb_io.write_snapshot(
        conn,
        ym="2024-03",
        facts_resolved=resolved,
        facts_deltas=deltas,
        manifests=None,
        meta=None,
    )

    resolved_codes = conn.execute(
        "SELECT iso3, hazard_code, series_semantics FROM facts_resolved WHERE ym = '2024-03'"
    ).fetchone()
    deltas_codes = conn.execute(
        "SELECT iso3, hazard_code, series_semantics FROM facts_deltas WHERE ym = '2024-03'"
    ).fetchone()
    conn.close()

    assert resolved_codes == ("PHL", "TC", "stock")
    assert deltas_codes == ("PHL", "TC", "new")


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
