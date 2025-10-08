from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.db import duckdb_io
from resolver.tools import export_facts, freeze_snapshot


def _basic_facts_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": "E1",
                "country_name": "Philippines",
                "iso3": "PHL",
                "hazard_code": "TC",
                "hazard_label": "Cyclone",
                "hazard_class": "Cyclone",
                "metric": "in_need",
                "value": "1000",
                "unit": "persons",
                "as_of_date": "2024-01-15",
                "publication_date": "2024-01-16",
                "publisher": "OCHA",
                "source_type": "report",
                "source_url": "https://example.com/tc",
                "doc_title": "TC Update",
                "definition_text": "People in need",
                "method": "survey",
                "confidence": "",
                "revision": "",
                "ingested_at": "",
                "ym": "",
                "series_semantics": "",
            },
            {
                "event_id": "E2",
                "country_name": "Philippines",
                "iso3": "PHL",
                "hazard_code": "EQ",
                "hazard_label": "Earthquake",
                "hazard_class": "Geophysical",
                "metric": "affected",
                "value": "250",
                "unit": "persons",
                "as_of_date": "2024-01-10",
                "publication_date": "2024-01-11",
                "publisher": "OCHA",
                "source_type": "report",
                "source_url": "https://example.com/eq",
                "doc_title": "EQ Update",
                "definition_text": "Affected population",
                "method": "assessment",
                "confidence": "",
                "revision": "",
                "ingested_at": "",
                "ym": "",
                "series_semantics": "Stock estimate",
            },
        ]
    )


def test_init_schema_creates_all_tables_and_keys(tmp_path: Path) -> None:
    db_path = tmp_path / "schema.duckdb"
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    try:
        duckdb_io.init_schema(conn)

        tables = {
            row[0]
            for row in conn.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'main'
                """
            ).fetchall()
        }
        expected = {"facts_resolved", "facts_deltas", "manifests", "meta_runs", "snapshots"}
        assert expected.issubset(tables)

        assert duckdb_io._has_declared_key(
            conn, "facts_resolved", duckdb_io.FACTS_RESOLVED_KEY_COLUMNS
        )
        assert duckdb_io._has_declared_key(
            conn, "facts_deltas", duckdb_io.FACTS_DELTAS_KEY_COLUMNS
        )
        manifest_constraints = duckdb_io._constraint_column_sets(conn, "manifests")
        assert ["path"] in [[col.lower() for col in cols] for cols in manifest_constraints]
        meta_constraints = duckdb_io._constraint_column_sets(conn, "meta_runs")
        assert ["run_id"] in [[col.lower() for col in cols] for cols in meta_constraints]

        duckdb_io.init_schema(conn)
    finally:
        conn.close()


def test_dual_writes_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "resolver.duckdb"
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")

    conn = duckdb_io.get_db(str(db_path))
    duckdb_io.init_schema(conn)
    conn.close()

    facts = _basic_facts_frame()

    export_facts._maybe_write_to_db(facts_resolved=facts)

    manifest = {
        "created_at_utc": "2024-01-31T00:00:00Z",
        "source_commit_sha": "abc123",
        "rows": len(facts),
    }
    facts_out = tmp_path / "facts.parquet"

    freeze_snapshot._maybe_write_db(
        ym="2024-01",
        facts_df=facts,
        resolved_df=facts,
        deltas_df=None,
        manifest=manifest,
        facts_out=facts_out,
        deltas_out=None,
    )
    freeze_snapshot._maybe_write_db(
        ym="2024-01",
        facts_df=facts,
        resolved_df=facts,
        deltas_df=None,
        manifest=manifest,
        facts_out=facts_out,
        deltas_out=None,
    )

    conn = duckdb_io.get_db(str(db_path))
    facts_rows = conn.execute(
        "SELECT COUNT(*) FROM facts_resolved WHERE ym = '2024-01'"
    ).fetchone()[0]
    snapshots_rows = conn.execute(
        "SELECT COUNT(*) FROM snapshots WHERE ym = '2024-01'"
    ).fetchone()[0]
    conn.close()

    assert facts_rows == 2
    assert snapshots_rows == 1


def test_semantics_canonicalisation(tmp_path: Path) -> None:
    db_path = tmp_path / "semantics.duckdb"
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    duckdb_io.init_schema(conn)

    resolved = pd.DataFrame(
        [
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "series_semantics": "",
                "value": "100",
            },
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "EQ",
                "metric": "affected",
                "series_semantics": "None",
                "value": "20",
            },
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "series_semantics": "Stock estimate",
                "value": "150",
            },
        ]
    )
    deltas = pd.DataFrame(
        [
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "value_new": "100",
                "value_stock": "50",
                "series_semantics": "",
            },
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "EQ",
                "metric": "affected",
                "value_new": "25",
                "value_stock": "",
                "series_semantics": "Random text",
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

    semantics_resolved = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT series_semantics FROM facts_resolved WHERE ym = '2024-02'"
        ).fetchall()
    }
    semantics_deltas = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT series_semantics FROM facts_deltas WHERE ym = '2024-02'"
        ).fetchall()
    }
    conn.close()

    assert semantics_resolved == {"stock", ""}
    assert semantics_deltas == {"new"}


def test_init_schema_fastpath(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("DEBUG", logger=duckdb_io.LOGGER.name)
    conn = duckdb_io.get_db(f"duckdb:///{tmp_path / 'fastpath.duckdb'}")
    duckdb_io.init_schema(conn)
    caplog.clear()
    duckdb_io.init_schema(conn)
    conn.close()

    assert any("skipping DDL execution" in record.message for record in caplog.records)


def test_delete_logging(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger=duckdb_io.LOGGER.name)
    conn = duckdb_io.get_db(f"duckdb:///{tmp_path / 'delete.duckdb'}")
    duckdb_io.init_schema(conn)

    frame = pd.DataFrame(
        [
            {
                "ym": "2024-03",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "series_semantics": "stock",
                "value": 10,
            }
        ]
    )

    duckdb_io.upsert_dataframe(
        conn,
        "facts_resolved",
        frame,
        keys=duckdb_io.FACTS_RESOLVED_KEY_COLUMNS,
    )
    caplog.clear()
    frame_updated = frame.assign(value=20)
    duckdb_io.upsert_dataframe(
        conn,
        "facts_resolved",
        frame_updated,
        keys=duckdb_io.FACTS_RESOLVED_KEY_COLUMNS,
    )
    conn.close()

    legacy_message = "Deleted 1 existing rows from facts_resolved"
    merge_message = "Upserted 1 rows into facts_resolved via MERGE"
    assert any(
        (legacy_message in record.message) or (merge_message in record.message)
        for record in caplog.records
    )
