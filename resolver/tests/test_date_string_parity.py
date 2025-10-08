import json
import re
import sys

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.db import duckdb_io


def test_duckdb_and_csv_date_strings_align(tmp_path, monkeypatch):
    staging = tmp_path / "staging.csv"
    rows = pd.DataFrame(
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
                "as_of_date": "2024-01-31",
                "publication_date": "2024-02-02",
                "publisher": "OCHA",
                "source_type": "report",
                "source_url": "https://example.org/1",
                "doc_title": "Report",
                "definition_text": "People in need",
                "method": "survey",
                "confidence": "medium",
                "revision": "1",
                "ingested_at": "2024-02-03T00:00:00Z",
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
                "as_of_date": "2024-02-28",
                "publication_date": "2024-03-04",
                "publisher": "OCHA",
                "source_type": "report",
                "source_url": "https://example.org/2",
                "doc_title": "Report 2",
                "definition_text": "People affected",
                "method": "survey",
                "confidence": "high",
                "revision": "1",
                "ingested_at": "2024-03-05T00:00:00Z",
            },
        ]
    )
    rows.to_csv(staging, index=False)

    config = {"mapping": {column: [column] for column in rows.columns}}
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
    csv_df = pd.read_csv(csv_path)
    assert pd.api.types.is_object_dtype(csv_df["as_of_date"])
    assert pd.api.types.is_object_dtype(csv_df["publication_date"])

    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    duckdb_io.init_schema(conn)
    db_df = conn.execute(
        "SELECT event_id, as_of_date, publication_date FROM facts_resolved ORDER BY event_id"
    ).fetch_df()

    assert pd.api.types.is_object_dtype(db_df["as_of_date"])
    assert pd.api.types.is_object_dtype(db_df["publication_date"])
    iso = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    assert db_df["as_of_date"].map(lambda v: bool(iso.match(str(v)))).all()
    assert db_df["publication_date"].map(lambda v: bool(iso.match(str(v)))).all()
    csv_dates = (
        csv_df[["event_id", "as_of_date", "publication_date"]]
        .sort_values("event_id")
        .reset_index(drop=True)
    )
    db_dates = (
        db_df[["event_id", "as_of_date", "publication_date"]]
        .sort_values("event_id")
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(db_dates, csv_dates)

    conn.close()
