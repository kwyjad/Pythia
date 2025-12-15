# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.db import duckdb_io
from resolver.tools import export_facts


def _build_staging(tmp_path: Path) -> tuple[Path, Path, pd.DataFrame]:
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
    staging = tmp_path / "staging.csv"
    data.to_csv(staging, index=False)

    mapping = {column: [column] for column in data.columns}
    config = {"mapping": mapping, "constants": {}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    return staging, config_path, data


@pytest.mark.duckdb
def test_auto_write_enabled_by_env(tmp_path, monkeypatch):
    staging, config_path, frame = _build_staging(tmp_path)
    out_dir = tmp_path / "exports"
    out_dir.mkdir()

    db_path = tmp_path / "auto.duckdb"
    monkeypatch.setenv("RESOLVER_DB_URL", str(db_path))
    monkeypatch.delenv("RESOLVER_WRITE_DB", raising=False)

    result = export_facts.export_facts(
        inp=staging,
        config_path=config_path,
        out_dir=out_dir,
    )

    assert "facts_resolved" in result.db_stats
    stats = result.db_stats["facts_resolved"]
    assert int(stats.get("rows_after", 0)) == len(frame)

    conn = duckdb_io.get_db(str(db_path))
    try:
        rows = conn.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0]
    finally:
        conn.close()

    assert rows == len(frame)


@pytest.mark.duckdb
def test_auto_write_can_be_disabled(tmp_path, monkeypatch):
    staging, config_path, _ = _build_staging(tmp_path)
    out_dir = tmp_path / "exports"
    out_dir.mkdir()

    db_path = tmp_path / "skip.duckdb"
    monkeypatch.setenv("RESOLVER_DB_URL", str(db_path))

    result = export_facts.export_facts(
        inp=staging,
        config_path=config_path,
        out_dir=out_dir,
        write_db="0",
    )

    assert "facts_resolved" not in result.db_stats
    assert not db_path.exists()
