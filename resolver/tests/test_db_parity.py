from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.db import duckdb_io


@pytest.mark.usefixtures("monkeypatch")
def test_exporter_dual_writes_to_duckdb(tmp_path, monkeypatch):
    staging = tmp_path / "staging.csv"
    data = pd.DataFrame(
        [
            {
                "event_id": "E1",
                "country_name": "Philippines",
                "iso3": "PHL",
                "hazard_code": "TC",
                "hazard_label": "Tropical Cyclone",
                "hazard_class": "Cyclone",
                "metric": "in_need",
                "value": "1000",
                "unit": "persons",
                "as_of_date": "2024-01-15",
                "publication_date": "2024-01-16",
                "publisher": "OCHA",
                "source_type": "situation_report",
                "source_url": "https://example.org/report",
                "doc_title": "SitRep",
                "definition_text": "People in need",
                "method": "survey",
                "confidence": "medium",
                "revision": "1",
                "ingested_at": "2024-01-17T00:00:00Z",
            },
            {
                "event_id": "E2",
                "country_name": "Philippines",
                "iso3": "PHL",
                "hazard_code": "EQ",
                "hazard_label": "Earthquake",
                "hazard_class": "Geophysical",
                "metric": "affected",
                "value": "500",
                "unit": "persons",
                "as_of_date": "2024-01-10",
                "publication_date": "2024-01-11",
                "publisher": "OCHA",
                "source_type": "situation_report",
                "source_url": "https://example.org/report2",
                "doc_title": "SitRep 2",
                "definition_text": "People affected",
                "method": "survey",
                "confidence": "high",
                "revision": "1",
                "ingested_at": "2024-01-12T00:00:00Z",
            },
        ]
    )
    data.to_csv(staging, index=False)

    config = {
        "mapping": {c: [c] for c in data.columns},
        "constants": {},
    }
    config_path = tmp_path / "config.yml"
    config_path.write_text(json.dumps(config))

    out_dir = tmp_path / "exports"
    db_path = tmp_path / "resolver.duckdb"
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")

    module = __import__("resolver.tools.export_facts", fromlist=["main"])
    argv = [
        "export_facts",
        "--in",
        str(staging),
        "--config",
        str(config_path),
        "--out",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    module.main()

    csv_path = out_dir / "facts.csv"
    assert csv_path.exists(), "CSV export missing"

    exported = pd.read_csv(csv_path)

    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    duckdb_io.init_schema(conn)
    db_rows = conn.execute(
        "SELECT event_id, iso3, hazard_code, metric, value, unit, as_of_date, publication_date FROM facts_raw ORDER BY event_id"
    ).fetch_df()

    assert len(db_rows) == len(exported)
    joined = db_rows.merge(
        exported,
        on=["event_id", "iso3", "hazard_code", "metric", "unit", "as_of_date", "publication_date", "value"],
        how="inner",
    )
    assert len(joined) == len(exported), "DuckDB facts_raw rows diverge from CSV output"
