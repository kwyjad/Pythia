import json
import sys

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.db import duckdb_io


def test_exporter_dual_write_matches_csv(tmp_path, monkeypatch):
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
                "source_url": "https://example.org/tc",
                "doc_title": "Report TC",
                "definition_text": "People in need",
                "method": "reported",
                "confidence": "medium",
                "revision": "1",
                "ingested_at": "2024-01-16T00:00:00Z",
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
                "source_url": "https://example.org/eq",
                "doc_title": "Report EQ",
                "definition_text": "People affected",
                "method": "reported",
                "confidence": "high",
                "revision": "1",
                "ingested_at": "2024-01-11T00:00:00Z",
            },
        ]
    )
    data.to_csv(staging, index=False)

    mapping = {column: [column] for column in data.columns}
    config = {"mapping": mapping, "constants": {}}
    config_path = tmp_path / "config.yml"
    config_path.write_text(json.dumps(config))

    out_dir = tmp_path / "exports"
    out_dir.mkdir()

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

    csv_df = pd.read_csv(out_dir / "facts.csv")
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    try:
        rows = conn.execute(
            "SELECT event_id, iso3, hazard_code, metric FROM facts_resolved ORDER BY event_id"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == len(csv_df)
    assert sorted(row[2] for row in rows) == sorted(csv_df["hazard_code"].tolist())
