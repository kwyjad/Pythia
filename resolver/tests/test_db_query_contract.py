from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

pytest.importorskip("duckdb")

from resolver.db import duckdb_io


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
