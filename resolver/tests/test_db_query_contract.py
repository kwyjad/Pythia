from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip(
    "duckdb",
    reason=(
        "duckdb not installed. Install via extras: `pip install .[db]` or offline: "
        "`scripts/install_db_extra_offline.(sh|ps1)`"
    ),
)
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from resolver.db import duckdb_io
from resolver.query import db_reader


@pytest.fixture()
def resolver_fixture(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    countries = pd.DataFrame([{"country_name": "Philippines", "iso3": "PHL"}])
    hazards = pd.DataFrame(
        [
            {
                "hazard_label": "Tropical Cyclone",
                "hazard_code": "TC",
                "hazard_class": "Cyclone",
            }
        ]
    )
    countries.to_csv(data_dir / "countries.csv", index=False)
    hazards.to_csv(data_dir / "shocks.csv", index=False)

    exports_dir = tmp_path / "exports"
    exports_dir.mkdir()

    resolved_df = pd.DataFrame(
        [
            {
                "ym": "2024-01",
                "iso3": "PHL",
                "hazard_code": "TC",
                "hazard_label": "Tropical Cyclone",
                "hazard_class": "Cyclone",
                "metric": "in_need",
                "value": 1500,
                "unit": "persons",
                "as_of_date": "2024-01-31",
                "publication_date": "2024-02-02",
                "publisher": "OCHA",
                "source_type": "situation_report",
                "source_url": "https://example.org/resolved",
                "doc_title": "SitRep",
                "definition_text": "People in need",
                "precedence_tier": "tier1",
                "event_id": "E1",
                "proxy_for": "",
                "confidence": "medium",
                "series_semantics": "stock",
            }
        ]
    )
    resolved_df.to_csv(exports_dir / "resolved.csv", index=False)

    deltas_df = pd.DataFrame(
        [
            {
                "ym": "2024-01",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "value_new": 1500,
                "value_stock": 1500,
                "series_semantics_out": "new",
                "rebase_flag": 0,
                "first_observation": 1,
                "delta_negative_clamped": 0,
                "as_of": "2024-01-31",
                "source_name": "OCHA",
                "source_url": "https://example.org/resolved",
                "definition_text": "People in need",
            }
        ]
    )
    deltas_df.to_csv(exports_dir / "deltas.csv", index=False)

    db_path = tmp_path / "resolver.duckdb"
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")
    monkeypatch.setenv("RESOLVER_CLI_BACKEND", "auto")
    monkeypatch.setenv("RESOLVER_API_BACKEND", "auto")

    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    duckdb_io.init_schema(conn)
    duckdb_io.write_snapshot(
        conn,
        ym="2024-01",
        facts_resolved=resolved_df,
        facts_deltas=deltas_df,
        manifests=[{"name": "resolved.csv", "rows": len(resolved_df)}],
        meta={"created_at_utc": "2024-02-02T00:00:00Z", "source_commit_sha": "deadbeef"},
    )

    import resolver.cli.resolver_cli as cli

    cli.EXPORTS = exports_dir
    cli.SNAPSHOTS = tmp_path / "snapshots"
    cli.STATE = tmp_path / "state"
    cli.DATA = data_dir
    cli.COUNTRIES_CSV = data_dir / "countries.csv"
    cli.SHOCKS_CSV = data_dir / "shocks.csv"

    api_module = importlib.reload(importlib.import_module("resolver.api.app"))

    return {
        "cli": cli,
        "api": api_module,
        "resolved_df": resolved_df,
        "deltas_df": deltas_df,
        "db_path": db_path,
        "exports_dir": exports_dir,
    }


def _run_cli(cli_module, capsys, backend: str, series: str = "new") -> dict:
    args = [
        "resolver_cli.py",
        "--iso3",
        "PHL",
        "--hazard_code",
        "TC",
        "--cutoff",
        "2024-01-31",
        "--json_only",
        "--series",
        series,
        "--backend",
        backend,
    ]
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(sys, "argv", args)
    try:
        cli_module.main()
        captured = capsys.readouterr()
    finally:
        monkeypatch.undo()
    return json.loads(captured.out.strip())


def test_cli_matches_between_backends(resolver_fixture, capsys):
    cli = resolver_fixture["cli"]
    files_output = _run_cli(cli, capsys, backend="files")
    db_output = _run_cli(cli, capsys, backend="db")
    assert files_output["value"] == db_output["value"]
    assert db_output["series_returned"] == "new"
    assert db_output["source"] == "db"


def test_api_matches_between_backends(resolver_fixture):
    api_module = resolver_fixture["api"]
    client = TestClient(api_module.app)

    files_resp = client.get(
        "/resolve",
        params={
            "iso3": "PHL",
            "hazard_code": "TC",
            "cutoff": "2024-01-31",
            "series": "new",
            "backend": "files",
        },
    )
    db_resp = client.get(
        "/resolve",
        params={
            "iso3": "PHL",
            "hazard_code": "TC",
            "cutoff": "2024-01-31",
            "series": "new",
            "backend": "db",
        },
    )

    assert files_resp.status_code == 200
    assert db_resp.status_code == 200
    assert files_resp.json()["value"] == db_resp.json()["value"]
    assert db_resp.json()["source"] == "db"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_api_batch_matches_cli(resolver_fixture, tmp_path):
    cli = resolver_fixture["cli"]
    api_module = resolver_fixture["api"]
    client = TestClient(api_module.app)

    queries = [
        {
            "iso3": "PHL",
            "hazard_code": "TC",
            "cutoff": "2024-01-31",
            "series": "new",
            "backend": "db",
        }
    ]

    resp = client.post("/resolve_batch", json=queries)
    assert resp.status_code == 200
    api_rows = resp.json()
    assert len(api_rows) == 1

    input_path = tmp_path / "queries.json"
    output_path = tmp_path / "results.jsonl"
    with input_path.open("w", encoding="utf-8") as handle:
        json.dump([{k: v for k, v in queries[0].items() if k != "backend"}], handle)

    cli.run_batch_resolve(
        input_path,
        output_path,
        backend="db",
        series=None,
        max_workers=1,
    )

    cli_rows = _read_jsonl(output_path)
    assert len(cli_rows) == 1
    assert cli_rows[0]["value"] == api_rows[0]["value"]
    assert cli_rows[0]["source"] == api_rows[0]["source"]
    assert cli_rows[0]["series_returned"] == api_rows[0]["series_returned"]


def test_batch_backend_parity(resolver_fixture, tmp_path):
    cli = resolver_fixture["cli"]
    api_module = resolver_fixture["api"]
    client = TestClient(api_module.app)

    base_query = {
        "iso3": "PHL",
        "hazard_code": "TC",
        "cutoff": "2024-01-31",
        "series": "new",
    }

    input_path = tmp_path / "batch.json"
    with input_path.open("w", encoding="utf-8") as handle:
        json.dump([base_query], handle)

    db_output = tmp_path / "db.jsonl"
    csv_output = tmp_path / "csv.jsonl"

    cli.run_batch_resolve(
        input_path,
        db_output,
        backend="db",
        series=None,
        max_workers=1,
    )
    cli.run_batch_resolve(
        input_path,
        csv_output,
        backend="csv",
        series=None,
        max_workers=1,
    )

    rows_db = _read_jsonl(db_output)
    rows_csv = _read_jsonl(csv_output)

    assert len(rows_db) == len(rows_csv) == 1
    assert rows_db[0]["value"] == rows_csv[0]["value"]
    assert rows_db[0]["series_returned"] == rows_csv[0]["series_returned"]
    assert rows_db[0]["source"] == "db"

    resp_db = client.post("/resolve_batch", json=[{**base_query, "backend": "db"}])
    resp_files = client.post("/resolve_batch", json=[{**base_query, "backend": "files"}])

    assert resp_db.status_code == 200
    assert resp_files.status_code == 200
    api_db = resp_db.json()
    api_files = resp_files.json()

    assert len(api_db) == len(api_files) == 1
    assert api_db[0]["value"] == api_files[0]["value"]
    assert api_db[0]["series_returned"] == api_files[0]["series_returned"]
    assert api_db[0]["source"] == "db"


def test_fetch_resolved_point_basic(tmp_path, monkeypatch):
    db_path = tmp_path / "resolved_basic.duckdb"
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    duckdb_io.init_schema(conn)

    resolved_df = pd.DataFrame(
        [
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
                "publication_date": "2024-03-01",
                "publisher": "OCHA",
                "source_id": "SR-001",
                "source_type": "situation_report",
                "source_url": "https://example.org/resolved",
                "doc_title": "SitRep",
                "definition_text": "People in need",
                "precedence_tier": "tier1",
                "event_id": "EVT-1",
                "proxy_for": "",
                "confidence": "medium",
                "series_semantics": "stock",
            }
        ]
    )

    duckdb_io.write_snapshot(
        conn,
        ym="2024-02",
        facts_resolved=resolved_df,
        facts_deltas=None,
        manifests=[{"name": "2024-02", "rows": 1}],
        meta={"created_at_utc": "2024-03-01T00:00:00Z"},
    )

    row = db_reader.fetch_resolved_point(
        conn,
        ym="2024-02",
        iso3="PHL",
        hazard_code="TC",
        cutoff="2024-02-28",
        preferred_metric="in_need",
    )

    assert row is not None
    assert row["metric"] == "in_need"
    assert pytest.approx(float(row["value"])) == 1500.0
    assert (row.get("series_semantics") or "stock").strip().lower() == "stock"


def test_fetch_resolved_point_respects_cutoff_and_metric_preference(tmp_path, monkeypatch):
    db_path = tmp_path / "resolved_preference.duckdb"
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")
    conn = duckdb_io.get_db(f"duckdb:///{db_path}")
    duckdb_io.init_schema(conn)

    resolved_df = pd.DataFrame(
        [
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "value": 2000,
                "as_of_date": "2024-02-26",
                "publication_date": "2024-02-27",
                "series_semantics": "stock",
            },
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "affected",
                "value": 1200,
                "as_of_date": "2024-02-20",
                "publication_date": "2024-02-21",
                "series_semantics": "stock",
            },
        ]
    )

    duckdb_io.write_snapshot(
        conn,
        ym="2024-02",
        facts_resolved=resolved_df,
        facts_deltas=None,
        manifests=[{"name": "2024-02", "rows": len(resolved_df)}],
        meta={"created_at_utc": "2024-03-06T00:00:00Z"},
    )

    # Preferred metric defaults to the requested metric when available before cutoff
    row_in_need = db_reader.fetch_resolved_point(
        conn,
        ym="2024-02",
        iso3="PHL",
        hazard_code="TC",
        cutoff="2024-02-28",
        preferred_metric="in_need",
    )
    assert row_in_need is not None
    assert row_in_need["metric"].lower() == "in_need"
    assert pytest.approx(float(row_in_need["value"])) == 2000.0

    # When preferred metric is "affected", that row should win if inside cutoff
    row_affected = db_reader.fetch_resolved_point(
        conn,
        ym="2024-02",
        iso3="PHL",
        hazard_code="TC",
        cutoff="2024-02-28",
        preferred_metric="affected",
    )
    assert row_affected is not None
    assert row_affected["metric"].lower() == "affected"

    # Tighten the cutoff so the later row is excluded
    row_cutoff = db_reader.fetch_resolved_point(
        conn,
        ym="2024-02",
        iso3="PHL",
        hazard_code="TC",
        cutoff="2024-02-21",
        preferred_metric="in_need",
    )
    assert row_cutoff is not None
    assert row_cutoff["metric"].lower() == "affected"
