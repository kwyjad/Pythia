"""End-to-end enforcement tests for series_semantics canonicalization."""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("duckdb")
import duckdb

from resolver.db import duckdb_io


@pytest.fixture()
def conn():
    connection = duckdb.connect(":memory:")
    try:
        yield connection
    finally:
        connection.close()


def _snapshot_meta() -> dict[str, object]:
    return {"created_at_utc": "2024-01-01T00:00:00Z"}


def test_facts_resolved_canonicalizes_to_stock_only(conn) -> None:
    facts_resolved = pd.DataFrame(
        [
            {
                "ym": "2024-01",
                "iso3": "USA",
                "hazard_code": "HZ",
                "metric": "cases",
                "value": 10.0,
                "series_semantics": "stock_estimate",
            },
            {
                "ym": "2024-01",
                "iso3": "CAN",
                "hazard_code": "HZ",
                "metric": "cases",
                "value": 5.0,
                "series_semantics": "Stock estimate",
            },
            {
                "ym": "2024-01",
                "iso3": "MEX",
                "hazard_code": "HZ",
                "metric": "cases",
                "value": 7.0,
                "series_semantics": "new",
            },
            {
                "ym": "2024-01",
                "iso3": "COL",
                "hazard_code": "HZ",
                "metric": "cases",
                "value": 4.0,
                "series_semantics": "",
            },
            {
                "ym": "2024-01",
                "iso3": "BRA",
                "hazard_code": "HZ",
                "metric": "cases",
                "value": 2.0,
                "series_semantics": None,
            },
        ]
    )

    duckdb_io.write_snapshot(
        conn,
        ym="2024-01",
        facts_resolved=facts_resolved,
        facts_deltas=None,
        manifests=None,
        meta=_snapshot_meta(),
    )

    results = conn.execute(
        "SELECT DISTINCT series_semantics FROM facts_resolved"
    ).fetchall()
    assert results == [("stock",)]


def test_facts_deltas_canonicalizes_to_new_only(conn) -> None:
    facts_deltas = pd.DataFrame(
        [
            {
                "ym": "2024-02",
                "iso3": "USA",
                "hazard_code": "HZ",
                "metric": "cases",
                "value_new": 1.0,
                "value_stock": 2.0,
                "series_semantics": "stock_estimate",
            },
            {
                "ym": "2024-02",
                "iso3": "CAN",
                "hazard_code": "HZ",
                "metric": "cases",
                "value_new": 3.0,
                "value_stock": 4.0,
                "series_semantics": "",
            },
            {
                "ym": "2024-02",
                "iso3": "MEX",
                "hazard_code": "HZ",
                "metric": "cases",
                "value_new": 5.0,
                "value_stock": 6.0,
                "series_semantics": None,
            },
        ]
    )

    duckdb_io.write_snapshot(
        conn,
        ym="2024-02",
        facts_resolved=None,
        facts_deltas=facts_deltas,
        manifests=None,
        meta=_snapshot_meta(),
    )

    results = conn.execute(
        "SELECT DISTINCT series_semantics FROM facts_deltas"
    ).fetchall()
    assert results == [("new",)]


@pytest.mark.parametrize("table_name, default", [("facts_resolved", "stock"), ("facts_deltas", "new")])
def test_upsert_dataframe_enforces_table_rules(conn, table_name, default) -> None:
    duckdb_io.init_schema(conn)
    if table_name == "facts_resolved":
        frame = pd.DataFrame(
            [
                {
                    "ym": "2024-03",
                    "iso3": "USA",
                    "hazard_code": "HZ",
                    "metric": "cases",
                    "series_semantics": "stock_estimate",
                    "value": 1.0,
                }
            ]
        )
        keys = duckdb_io.FACTS_RESOLVED_KEY_COLUMNS
    else:
        frame = pd.DataFrame(
            [
                {
                    "ym": "2024-03",
                    "iso3": "USA",
                    "hazard_code": "HZ",
                    "metric": "cases",
                    "series_semantics": "stock_estimate",
                    "value_new": 1.0,
                    "value_stock": 2.0,
                }
            ]
        )
        keys = duckdb_io.FACTS_DELTAS_KEY_COLUMNS

    canonical, _ = duckdb_io._canonicalize_semantics(frame, table_name, default)
    duckdb_io._assert_semantics_required(canonical, table_name)
    duckdb_io.upsert_dataframe(conn, table_name, canonical, keys=keys)

    stored = conn.execute(
        f"SELECT DISTINCT series_semantics FROM {table_name}"
    ).fetchall()
    expected = ("stock",) if table_name == "facts_resolved" else ("new",)
    assert stored == [expected]
