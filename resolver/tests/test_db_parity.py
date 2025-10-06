from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

pytest.importorskip("duckdb")

from resolver.db import duckdb_io
from resolver.common import compute_series_semantics


def _expected_resolved(df: pd.DataFrame) -> pd.DataFrame:
    expected = df.copy()
    expected["ym"] = ""
    if "as_of_date" in expected.columns:
        derived = pd.to_datetime(expected["as_of_date"], errors="coerce").dt.strftime("%Y-%m")
        expected.loc[expected["ym"].astype(str).str.len() == 0, "ym"] = derived.fillna("")
    if "publication_date" in expected.columns:
        mask = expected["ym"].astype(str).str.len() == 0
        fallback = pd.to_datetime(expected.loc[mask, "publication_date"], errors="coerce").dt.strftime("%Y-%m")
        expected.loc[mask, "ym"] = fallback.fillna("")
    expected["value"] = pd.to_numeric(expected.get("value", 0), errors="coerce")
    expected["series_semantics"] = expected.apply(
        lambda row: compute_series_semantics(
            metric=row.get("metric"), existing=row.get("series_semantics")
        ),
        axis=1,
    )
    subset = expected[
        [
            "event_id",
            "iso3",
            "hazard_code",
            "metric",
            "value",
            "unit",
            "as_of_date",
            "publication_date",
            "series_semantics",
            "ym",
        ]
    ].copy()
    subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
    subset["series_semantics"] = subset["series_semantics"].fillna("").astype(str)
    subset["ym"] = subset["ym"].fillna("").astype(str)
    return subset


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
        "SELECT event_id, iso3, hazard_code, metric, value, unit, as_of_date, publication_date, series_semantics, ym "
        "FROM facts_resolved ORDER BY event_id"
    ).fetch_df()

    expected_resolved = _expected_resolved(exported)
    assert len(db_rows) == len(expected_resolved)
    joined = db_rows.merge(
        expected_resolved,
        on=[
            "event_id",
            "iso3",
            "hazard_code",
            "metric",
            "unit",
            "as_of_date",
            "publication_date",
            "value",
            "series_semantics",
            "ym",
        ],
        how="inner",
    )
    assert len(joined) == len(expected_resolved), "DuckDB facts_resolved rows diverge from CSV output"

    conn.close()

    freeze = __import__("resolver.tools.freeze_snapshot", fromlist=["main"])
    monkeypatch.setattr(freeze, "run_validator", lambda _path: None)
    monkeypatch.setattr(freeze, "SNAPSHOTS", tmp_path / "snapshots")
    argv = [
        "freeze_snapshot",
        "--facts",
        str(csv_path),
        "--month",
        "2024-01",
        "--overwrite",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    freeze.main()

    snapshot_dir = tmp_path / "snapshots" / "2024-01"
    facts_parquet = snapshot_dir / "facts.parquet"
    assert facts_parquet.exists(), "Snapshot facts parquet missing"
    snap_df = pd.read_parquet(facts_parquet)
    snap_expected = _expected_resolved(snap_df)

    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    duckdb_io.init_schema(conn)
    db_snapshot = conn.execute(
        "SELECT event_id, iso3, hazard_code, metric, value, unit, as_of_date, publication_date, series_semantics, ym "
        "FROM facts_resolved WHERE ym = '2024-01' ORDER BY event_id"
    ).fetch_df()
    merged_snapshot = db_snapshot.merge(
        snap_expected,
        on=[
            "event_id",
            "iso3",
            "hazard_code",
            "metric",
            "unit",
            "as_of_date",
            "publication_date",
            "value",
            "series_semantics",
            "ym",
        ],
        how="inner",
    )
    assert len(merged_snapshot) == len(snap_expected)

    snapshots_db = conn.execute("SELECT ym, git_sha, created_at FROM snapshots").fetch_df()
    assert not snapshots_db.empty
    assert (snapshots_db["ym"] == "2024-01").any()
    conn.close()
