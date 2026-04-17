# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import pytest
import pandas as pd

duckdb = pytest.importorskip("duckdb")

from resolver.transform.resolve_sources import resolve_sources


def test_resolve_sources_prefers_configured_priority(tmp_path):
    conn = duckdb.connect(str(tmp_path / "resolver.duckdb"))
    try:
        conn.execute(
            """
            CREATE TABLE facts_resolved (
                ym TEXT,
                iso3 TEXT,
                hazard_code TEXT,
                hazard_label TEXT,
                hazard_class TEXT,
                metric TEXT,
                series_semantics TEXT,
                value DOUBLE,
                unit TEXT,
                as_of DATE,
                as_of_date VARCHAR,
                publication_date VARCHAR,
                publisher TEXT,
                source_id TEXT,
                source_type TEXT,
                source_url TEXT,
                doc_title TEXT,
                definition_text TEXT,
                precedence_tier TEXT,
                event_id TEXT,
                proxy_for TEXT,
                confidence TEXT,
                series TEXT,
                provenance_source TEXT,
                provenance_rank INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        rows = pd.DataFrame(
            [
                {
                    "ym": "2024-01",
                    "iso3": "EXL",
                    "hazard_code": "DR",
                    "hazard_label": "Drought",
                    "hazard_class": "natural",
                    "metric": "in_need",
                    "series_semantics": "stock",
                    "value": 100.0,
                    "unit": "persons",
                    "as_of": "2024-01-15",
                    "as_of_date": "2024-01-15",
                    "publication_date": "2024-01-20",
                    "publisher": "ACLED",
                    "source_id": "ACLED",
                    "source_type": None,
                    "source_url": None,
                    "doc_title": None,
                    "definition_text": None,
                    "precedence_tier": None,
                    "event_id": "evt-1",
                    "proxy_for": None,
                    "confidence": None,
                    "series": None,
                    "provenance_source": None,
                    "provenance_rank": None,
                },
                {
                    "ym": "2024-01",
                    "iso3": "EXL",
                    "hazard_code": "DR",
                    "hazard_label": "Drought",
                    "hazard_class": "natural",
                    "metric": "in_need",
                    "series_semantics": "stock",
                    "value": 90.0,
                    "unit": "persons",
                    "as_of": "2024-01-15",
                    "as_of_date": "2024-01-15",
                    "publication_date": "2024-01-18",
                    "publisher": "IDMC",
                    "source_id": "IDMC",
                    "source_type": None,
                    "source_url": None,
                    "doc_title": None,
                    "definition_text": None,
                    "precedence_tier": None,
                    "event_id": "evt-2",
                    "proxy_for": None,
                    "confidence": None,
                    "series": None,
                    "provenance_source": None,
                    "provenance_rank": None,
                },
                {
                    "ym": "2024-02",
                    "iso3": "EXL",
                    "hazard_code": "DR",
                    "hazard_label": "Drought",
                    "hazard_class": "natural",
                    "metric": "in_need",
                    "series_semantics": "stock",
                    "value": 120.0,
                    "unit": "persons",
                    "as_of": "2024-02-15",
                    "as_of_date": "2024-02-15",
                    "publication_date": "2024-02-20",
                    "publisher": "IDMC",
                    "source_id": "IDMC",
                    "source_type": None,
                    "source_url": None,
                    "doc_title": None,
                    "definition_text": None,
                    "precedence_tier": None,
                    "event_id": "evt-3",
                    "proxy_for": None,
                    "confidence": None,
                    "series": None,
                    "provenance_source": None,
                    "provenance_rank": None,
                },
            ]
        )

        conn.register("tmp_resolved_fixture", rows)
        try:
            cols = ", ".join(rows.columns)
            conn.execute(
                f"INSERT INTO facts_resolved ({cols}) SELECT {cols} FROM tmp_resolved_fixture"
            )
        finally:
            conn.unregister("tmp_resolved_fixture")

        prioritized = resolve_sources(conn)
        assert prioritized == 2

        rows = conn.execute(
            """
            SELECT iso3, as_of_date, value, source_id, provenance_source, provenance_rank
            FROM facts_resolved
            ORDER BY as_of_date
            """
        ).df()
    finally:
        conn.close()

    assert len(rows) == 2
    first = rows.iloc[0]
    second = rows.iloc[1]

    assert first["source_id"] == "ACLED"
    assert first["provenance_source"] == "ACLED"
    assert first["provenance_rank"] == 2  # ACLED is 2nd in source_precedence
    assert first["value"] == 100

    assert second["source_id"] == "IDMC"
    assert second["provenance_source"] == "IDMC"
    assert second["provenance_rank"] > first["provenance_rank"]
    assert second["value"] == 120


def test_resolve_sources_stock_and_new_coexist(tmp_path):
    """Stock and new rows for the same iso3/hazard/date survive dedup."""
    conn = duckdb.connect(str(tmp_path / "resolver.duckdb"))
    try:
        conn.execute(
            """
            CREATE TABLE facts_resolved (
                ym TEXT,
                iso3 TEXT,
                hazard_code TEXT,
                hazard_label TEXT,
                hazard_class TEXT,
                metric TEXT,
                series_semantics TEXT,
                value DOUBLE,
                unit TEXT,
                as_of DATE,
                as_of_date VARCHAR,
                publication_date VARCHAR,
                publisher TEXT,
                source_id TEXT,
                source_type TEXT,
                source_url TEXT,
                doc_title TEXT,
                definition_text TEXT,
                precedence_tier TEXT,
                event_id TEXT,
                proxy_for TEXT,
                confidence TEXT,
                series TEXT,
                provenance_source TEXT,
                provenance_rank INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        rows = pd.DataFrame(
            [
                {
                    "ym": "2024-01",
                    "iso3": "ETH",
                    "hazard_code": "FL",
                    "hazard_label": "Flood",
                    "hazard_class": "natural",
                    "metric": "displaced",
                    "series_semantics": "stock",
                    "value": 5000.0,
                    "unit": "persons",
                    "as_of": "2024-01-15",
                    "as_of_date": "2024-01-15",
                    "publication_date": "2024-01-20",
                    "publisher": "IFRC",
                    "source_id": "IFRC",
                    "source_type": None,
                    "source_url": None,
                    "doc_title": None,
                    "definition_text": None,
                    "precedence_tier": None,
                    "event_id": "evt-ifrc-1",
                    "proxy_for": None,
                    "confidence": None,
                    "series": None,
                    "provenance_source": None,
                    "provenance_rank": None,
                },
                {
                    "ym": "2024-01",
                    "iso3": "ETH",
                    "hazard_code": "FL",
                    "hazard_label": "Flood",
                    "hazard_class": "natural",
                    "metric": "displaced",
                    "series_semantics": "new",
                    "value": 1200.0,
                    "unit": "persons",
                    "as_of": "2024-01-15",
                    "as_of_date": "2024-01-15",
                    "publication_date": "2024-01-18",
                    "publisher": "IDMC",
                    "source_id": "IDMC",
                    "source_type": None,
                    "source_url": None,
                    "doc_title": None,
                    "definition_text": None,
                    "precedence_tier": None,
                    "event_id": "evt-idmc-1",
                    "proxy_for": None,
                    "confidence": None,
                    "series": None,
                    "provenance_source": None,
                    "provenance_rank": None,
                },
            ]
        )

        conn.register("tmp_resolved_fixture", rows)
        try:
            cols = ", ".join(rows.columns)
            conn.execute(
                f"INSERT INTO facts_resolved ({cols}) SELECT {cols} FROM tmp_resolved_fixture"
            )
        finally:
            conn.unregister("tmp_resolved_fixture")

        prioritized = resolve_sources(conn)
        assert prioritized == 2, (
            f"Expected both stock and new rows to survive, got {prioritized}"
        )

        result = conn.execute(
            """
            SELECT iso3, as_of_date, value, source_id, series_semantics
            FROM facts_resolved
            ORDER BY series_semantics
            """
        ).df()
    finally:
        conn.close()

    assert len(result) == 2

    new_row = result[result["series_semantics"] == "new"].iloc[0]
    stock_row = result[result["series_semantics"] == "stock"].iloc[0]

    assert new_row["source_id"] == "IDMC"
    assert new_row["value"] == 1200.0

    assert stock_row["source_id"] == "IFRC"
    assert stock_row["value"] == 5000.0


def test_resolve_sources_scoped_preserves_out_of_scope(tmp_path):
    """Scoped dedup must leave rows from non-scoped sources alone.

    Regression test for the Resolver Update workflow crash where
    load_and_derive's unscoped ``resolve_sources`` call collapsed
    GDACS / FEWS NET / IPC API rows coexisting with Phase 1 rows.
    """

    conn = duckdb.connect(str(tmp_path / "resolver.duckdb"))
    try:
        conn.execute(
            """
            CREATE TABLE facts_resolved (
                ym TEXT,
                iso3 TEXT,
                hazard_code TEXT,
                hazard_label TEXT,
                hazard_class TEXT,
                metric TEXT,
                series_semantics TEXT,
                value DOUBLE,
                unit TEXT,
                as_of DATE,
                as_of_date VARCHAR,
                publication_date VARCHAR,
                publisher TEXT,
                source_id TEXT,
                source_type TEXT,
                source_url TEXT,
                doc_title TEXT,
                definition_text TEXT,
                precedence_tier TEXT,
                event_id TEXT,
                proxy_for TEXT,
                confidence TEXT,
                series TEXT,
                provenance_source TEXT,
                provenance_rank INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        base = {
            "hazard_label": None,
            "hazard_class": None,
            "unit": "persons",
            "as_of": None,
            "publication_date": "2024-01-20",
            "source_type": None,
            "source_url": None,
            "doc_title": None,
            "definition_text": None,
            "precedence_tier": None,
            "proxy_for": None,
            "confidence": None,
            "series": None,
            "provenance_source": None,
            "provenance_rank": None,
        }

        rows = pd.DataFrame(
            [
                # Two Phase 1 candidates for the same cell — IFRC should win.
                {
                    **base,
                    "ym": "2024-01",
                    "iso3": "SYR",
                    "hazard_code": "FL",
                    "metric": "in_need",
                    "series_semantics": "stock",
                    "value": 5000.0,
                    "as_of_date": "2024-01-15",
                    "publisher": "IFRC",
                    "source_id": "ifrc_go",
                    "event_id": "evt-ifrc",
                },
                {
                    **base,
                    "ym": "2024-01",
                    "iso3": "SYR",
                    "hazard_code": "FL",
                    "metric": "in_need",
                    "series_semantics": "stock",
                    "value": 4200.0,
                    "as_of_date": "2024-01-15",
                    "publisher": "IDMC",
                    "source_id": "idmc",
                    "event_id": "evt-idmc",
                },
                # Phase 2 GDACS row — same cell, but different source and
                # NOT in the scope passed to resolve_sources.
                {
                    **base,
                    "ym": "2024-01",
                    "iso3": "SYR",
                    "hazard_code": "FL",
                    "metric": "in_need",
                    "series_semantics": "stock",
                    "value": 6100.0,
                    "as_of_date": "2024-01-15",
                    "publisher": "GDACS / JRC",
                    "source_id": "GDACS / JRC",
                    "event_id": "evt-gdacs",
                },
                # Phase 2 FEWS NET row — different metric, different cell.
                {
                    **base,
                    "ym": "2024-01",
                    "iso3": "SYR",
                    "hazard_code": "DR",
                    "metric": "phase3plus_in_need",
                    "series_semantics": "stock",
                    "value": 7500.0,
                    "as_of_date": "2024-01-15",
                    "publisher": "FEWS NET",
                    "source_id": "FEWS NET",
                    "event_id": "evt-fewsnet",
                },
            ]
        )

        conn.register("tmp_fixture", rows)
        try:
            cols = ", ".join(rows.columns)
            conn.execute(
                f"INSERT INTO facts_resolved ({cols}) SELECT {cols} FROM tmp_fixture"
            )
        finally:
            conn.unregister("tmp_fixture")

        kept = resolve_sources(
            conn,
            sources=["acled", "idmc", "ifrc_go"],
        )

        result = conn.execute(
            "SELECT source_id, value FROM facts_resolved ORDER BY source_id"
        ).df()
    finally:
        conn.close()

    # After scoped resolution we expect 3 rows:
    #   - IFRC (winner within Phase 1 scope)
    #   - GDACS / JRC (out of scope, preserved)
    #   - FEWS NET (out of scope, preserved)
    assert kept == 1, f"Scoped dedup should write 1 Phase 1 row, got {kept}"
    assert len(result) == 3
    source_ids = set(result["source_id"])
    assert source_ids == {"ifrc_go", "GDACS / JRC", "FEWS NET"}
    ifrc_row = result[result["source_id"] == "ifrc_go"].iloc[0]
    assert ifrc_row["value"] == 5000.0
