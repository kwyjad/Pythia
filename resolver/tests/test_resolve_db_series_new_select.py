# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.db import duckdb_io
from resolver.query import selectors


def _seed_duckdb(tmp_path):
    db_path = tmp_path / "resolver_select.duckdb"
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    duckdb_io.init_schema(conn)

    resolved = pd.DataFrame(
        [
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "hazard_label": "Tropical Cyclone",
                "hazard_class": "Cyclone",
                "metric": "in_need",
                "value": 1500,
                "unit": "persons",
                "as_of_date": "2024-02-28",
                "publication_date": "2024-03-02",
                "publisher": "OCHA",
                "source_type": "situation_report",
                "source_url": "https://example.org/resolved",
                "doc_title": "SitRep",
                "definition_text": "",
                "precedence_tier": "tier1",
                "event_id": "E1",
                "proxy_for": "",
                "confidence": "medium",
                "series_semantics": "stock",
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
                "value_new": 500,
                "series_semantics": "new",
                "as_of": "2024-02-28",
                "source_id": "OCHA",
            }
        ]
    )

    duckdb_io.write_snapshot(
        conn,
        ym="2024-02",
        facts_resolved=resolved,
        facts_deltas=deltas,
        manifests=[{"name": "2024-02", "rows": 1}],
        meta={"created_at_utc": "2024-03-02T00:00:00Z"},
    )

    return conn, db_path


def test_db_series_new_select_aliases_provenance(tmp_path, monkeypatch):
    _, db_path = _seed_duckdb(tmp_path)
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")

    df, dataset_label, series_used = selectors.load_series_from_db("2024-02", "new")

    assert series_used == "new"
    assert dataset_label == "db_facts_deltas"
    assert df is not None and not df.empty

    row = df[(df["iso3"] == "PHL") & (df["hazard_code"] == "TC")].iloc[0]
    assert float(row["value"]) == 500.0
    assert row["series_semantics"] == "new"
    for field in ["source_url", "source_type", "doc_title", "definition_text"]:
        assert field in row
        assert row[field] == ""

    # Stock selection continues to work with the same seed
    stock_df, stock_label, stock_series = selectors.load_series_from_db("2024-02", "stock")
    assert stock_series == "stock"
    assert stock_label == "db_facts_resolved"
    assert stock_df is not None and not stock_df.empty
    stock_row = stock_df[(stock_df["iso3"] == "PHL") & (stock_df["hazard_code"] == "TC")].iloc[0]
    assert int(stock_row["value"]) == 1500


def test_db_series_new_missing_month_returns_none(tmp_path, monkeypatch):
    _, db_path = _seed_duckdb(tmp_path)
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")

    df, dataset_label, series_used = selectors.load_series_from_db("2024-03", "new")

    assert df is None or df.empty
    assert dataset_label == "db_facts_deltas"
    assert series_used == "new"


def test_db_series_new_returns_numeric_value(tmp_path, monkeypatch):
    _, db_path = _seed_duckdb(tmp_path)
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")

    df, dataset_label, series_used = selectors.load_series_from_db("2024-02", "new")

    assert series_used == "new"
    assert dataset_label == "db_facts_deltas"
    assert df is not None and not df.empty

    row = df[(df["iso3"] == "PHL") & (df["hazard_code"] == "TC")].iloc[0]
    assert pytest.approx(float(row["value"])) == 500.0
    assert pytest.approx(float(row["value_new"])) == 500.0
