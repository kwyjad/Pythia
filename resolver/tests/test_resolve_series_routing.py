# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.db import duckdb_io
from resolver.query.selectors import resolve_point


def _setup_db(tmp_path):
    db_path = tmp_path / "resolver.duckdb"
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    duckdb_io.init_schema(conn)
    return conn, db_path


def _seed_common_rows(conn):
    resolved = pd.DataFrame(
        [
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "hazard_label": "Tropical Cyclone",
                "hazard_class": "Cyclone",
                "metric": "in_need",
                "series_semantics": "stock",
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
                "confidence": "medium",
                "proxy_for": "",
            }
        ]
    )
    duckdb_io.upsert_dataframe(
        conn,
        resolved,
        "facts_resolved",
        keys=duckdb_io.FACTS_RESOLVED_KEY,
    )

    deltas = pd.DataFrame(
        [
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "value_new": 500,
                "value_stock": 1500,
                "series_semantics": "new",
                "as_of": "2024-02-28",
                "source_id": "sr-1",
            }
        ]
    )
    duckdb_io.upsert_dataframe(
        conn,
        deltas,
        "facts_deltas",
        keys=duckdb_io.FACTS_DELTAS_KEY,
    )


def test_resolve_new_uses_deltas_and_labels_new(tmp_path, monkeypatch):
    conn, db_path = _setup_db(tmp_path)
    _seed_common_rows(conn)
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")
    monkeypatch.delenv("RESOLVER_ALLOW_SERIES_FALLBACK", raising=False)

    result = resolve_point(
        iso3="PHL",
        hazard_code="TC",
        cutoff="2024-02-28",
        series="new",
        backend="db",
    )

    assert result is not None
    assert result["series_returned"] == "new"
    assert result["value"] == 500
    assert result["source_dataset"] == "db_facts_deltas"


def test_resolve_stock_uses_resolved_and_labels_stock(tmp_path, monkeypatch):
    conn, db_path = _setup_db(tmp_path)
    _seed_common_rows(conn)
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")
    monkeypatch.delenv("RESOLVER_ALLOW_SERIES_FALLBACK", raising=False)

    result = resolve_point(
        iso3="PHL",
        hazard_code="TC",
        cutoff="2024-02-28",
        series="stock",
        backend="db",
    )

    assert result is not None
    assert result["series_returned"] == "stock"
    assert result["value"] == 1500
    assert result["source_dataset"] == "db_facts_resolved"


def test_resolve_new_no_data_no_fallback(tmp_path, monkeypatch):
    conn, db_path = _setup_db(tmp_path)
    # Seed only the resolved table so deltas are missing.
    resolved_only = pd.DataFrame(
        [
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "hazard_label": "Tropical Cyclone",
                "hazard_class": "Cyclone",
                "metric": "in_need",
                "series_semantics": "stock",
                "value": 1500,
                "unit": "persons",
                "as_of_date": "2024-02-28",
                "publication_date": "2024-03-02",
                "publisher": "OCHA",
            }
        ]
    )
    duckdb_io.upsert_dataframe(
        conn,
        resolved_only,
        "facts_resolved",
        keys=duckdb_io.FACTS_RESOLVED_KEY,
    )

    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")
    monkeypatch.delenv("RESOLVER_ALLOW_SERIES_FALLBACK", raising=False)

    result = resolve_point(
        iso3="PHL",
        hazard_code="TC",
        cutoff="2024-02-28",
        series="new",
        backend="db",
    )

    assert result is None
