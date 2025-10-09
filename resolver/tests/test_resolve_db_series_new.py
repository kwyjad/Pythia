import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.db import duckdb_io
from resolver.query import selectors


def _seed_duckdb(tmp_path):
    db_path = tmp_path / "resolver.duckdb"
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    duckdb_io.init_schema(conn)

    resolved = pd.DataFrame(
        [
            {
                "ym": "2024-01",
                "iso3": "PHL",
                "hazard_code": "TC",
                "hazard_label": "Tropical Cyclone",
                "hazard_class": "Cyclone",
                "metric": "in_need",
                "value": 1000,
                "unit": "persons",
                "as_of_date": "2024-01-31",
                "publication_date": "2024-02-02",
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
            },
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
            },
        ]
    )

    deltas = pd.DataFrame(
        [
            {
                "ym": "2024-01",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "value_new": 1000,
                "value_stock": 1000,
                "series_semantics": "new",
                "rebase_flag": 0,
                "first_observation": 1,
                "delta_negative_clamped": 0,
                "as_of": "2024-01-31",
                "source_id": "OCHA",
                "source_url": "https://example.org/resolved",
                "definition_text": "",
            },
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "value_new": 500,
                "value_stock": 1500,
                "series_semantics": "new",
                "rebase_flag": 0,
                "first_observation": 0,
                "delta_negative_clamped": 0,
                "as_of": "2024-02-28",
                "source_id": "OCHA",
                "source_url": "https://example.org/resolved",
                "definition_text": "",
            },
        ]
    )

    duckdb_io.write_snapshot(
        conn,
        ym="2024-01",
        facts_resolved=resolved[resolved["ym"] == "2024-01"],
        facts_deltas=deltas[deltas["ym"] == "2024-01"],
        manifests=[{"name": "2024-01", "rows": 1}],
        meta={"created_at_utc": "2024-02-02T00:00:00Z"},
    )
    duckdb_io.write_snapshot(
        conn,
        ym="2024-02",
        facts_resolved=resolved[resolved["ym"] == "2024-02"],
        facts_deltas=deltas[deltas["ym"] == "2024-02"],
        manifests=[{"name": "2024-02", "rows": 1}],
        meta={"created_at_utc": "2024-03-02T00:00:00Z"},
    )

    return conn, db_path


def test_db_resolve_new_returns_value_new_for_cutoff(tmp_path, monkeypatch):
    conn, db_path = _seed_duckdb(tmp_path)
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")

    result = selectors.resolve_point(
        iso3="PHL",
        hazard_code="TC",
        cutoff="2024-02-28",
        series="new",
        metric="in_need",
        backend="db",
    )

    assert result is not None
    assert result["ok"] is True
    assert result["value"] == 500
    assert result["series_returned"] == "new"
    assert result["series_semantics"] == "new"
    assert result["source_dataset"] == "db_facts_deltas"

    total_deltas = conn.execute(
        "SELECT sum(value_new) FROM facts_deltas WHERE ym IN ('2024-01','2024-02')"
    ).fetchone()[0]
    stock_value = conn.execute(
        "SELECT value FROM facts_resolved WHERE ym = '2024-02'"
    ).fetchone()[0]
    assert int(total_deltas) == int(stock_value)


def test_db_resolve_stock_returns_value_for_cutoff(tmp_path, monkeypatch):
    _, db_path = _seed_duckdb(tmp_path)
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")

    result = selectors.resolve_point(
        iso3="PHL",
        hazard_code="TC",
        cutoff="2024-02-28",
        series="stock",
        metric="in_need",
        backend="db",
    )

    assert result is not None
    assert result["ok"] is True
    assert result["value"] == 1500
    assert result["series_returned"] == "stock"
    assert result["series_semantics"] == "stock"
    assert result["source_dataset"] == "db_facts_resolved"
