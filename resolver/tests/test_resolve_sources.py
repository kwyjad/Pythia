from __future__ import annotations
import duckdb
import pandas as pd

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
                    "publisher": "IPC",
                    "source_id": "IPC",
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
                    "publisher": "ReliefWeb",
                    "source_id": "ReliefWeb",
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
                    "publisher": "ReliefWeb",
                    "source_id": "ReliefWeb",
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

    assert first["source_id"] == "IPC"
    assert first["provenance_source"] == "IPC"
    assert first["provenance_rank"] == 1
    assert first["value"] == 100

    assert second["source_id"] == "ReliefWeb"
    assert second["provenance_source"] == "ReliefWeb"
    assert second["provenance_rank"] > first["provenance_rank"]
    assert second["value"] == 120
