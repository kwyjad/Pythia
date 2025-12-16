# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import json
import sys

import pandas as pd
import pytest

pytest.importorskip(
    "duckdb",
    reason=(
        "duckdb not installed. Install via extras: `pip install .[db]` or offline: "
        "`scripts/install_db_extra_offline.(sh|ps1)`"
    ),
)

from resolver.db import duckdb_io


def _run_cli(cli_module, args, capsys) -> dict:
    runner = pytest.MonkeyPatch()
    runner.setattr(sys, "argv", args)
    try:
        cli_module.main()
        captured = capsys.readouterr()
    finally:
        runner.undo()
    return json.loads(captured.out.strip())


def test_cumulative_new_equals_stock_e2e(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    countries = pd.DataFrame([
        {"country_name": "Philippines", "iso3": "PHL"},
    ])
    hazards = pd.DataFrame([
        {
            "hazard_label": "Tropical Cyclone",
            "hazard_code": "TC",
            "hazard_class": "Cyclone",
        }
    ])
    countries.to_csv(data_dir / "countries.csv", index=False)
    hazards.to_csv(data_dir / "shocks.csv", index=False)

    db_path = tmp_path / "e2e.duckdb"
    db_url = f"duckdb:///{db_path}"
    monkeypatch.setenv("RESOLVER_DB_URL", db_url)
    monkeypatch.setenv("RESOLVER_CLI_BACKEND", "db")

    conn = duckdb_io.get_db(db_url)
    duckdb_io.init_schema(conn)

    january_deltas = pd.DataFrame(
        [
            {
                "ym": "2024-01",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "value_new": "100",
                "value_stock": "100",
                "series_semantics": "new",
                "as_of": "2024-01-31",
                "source_id": "delta-2024-01",
            }
        ]
    )
    february_deltas = pd.DataFrame(
        [
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "value_new": "150",
                "value_stock": "250",
                "series_semantics": "new",
                "as_of": "2024-02-28",
                "source_id": "delta-2024-02",
            }
        ]
    )
    february_stock = pd.DataFrame(
        [
            {
                "ym": "2024-02",
                "iso3": "PHL",
                "hazard_code": "TC",
                "metric": "in_need",
                "value": "250",
                "unit": "persons",
                "series_semantics": "stock",
                "as_of_date": "2024-02-28",
                "publication_date": "2024-02-28",
                "publisher": "Resolver E2E",
            }
        ]
    )

    duckdb_io.write_snapshot(
        conn,
        ym="2024-01",
        facts_resolved=None,
        facts_deltas=january_deltas,
        manifests=[{"name": "deltas.csv", "rows": len(january_deltas)}],
        meta={"created_at_utc": "2024-01-31T00:00:00Z", "source_commit_sha": "e2e"},
    )
    duckdb_io.write_snapshot(
        conn,
        ym="2024-02",
        facts_resolved=february_stock,
        facts_deltas=february_deltas,
        manifests=[{"name": "deltas.csv", "rows": len(february_deltas)}],
        meta={"created_at_utc": "2024-02-28T00:00:00Z", "source_commit_sha": "e2e"},
    )

    import resolver.cli.resolver_cli as cli

    cli.DATA = data_dir
    cli.COUNTRIES_CSV = data_dir / "countries.csv"
    cli.SHOCKS_CSV = data_dir / "shocks.csv"
    cli.EXPORTS = tmp_path / "exports"
    cli.SNAPSHOTS = tmp_path / "snapshots"
    cli.STATE = tmp_path / "state"
    for path in (cli.EXPORTS, cli.SNAPSHOTS, cli.STATE):
        path.mkdir(parents=True, exist_ok=True)

    feb_args = [
        "resolver_cli.py",
        "--iso3",
        "PHL",
        "--hazard_code",
        "TC",
        "--cutoff",
        "2024-02-28",
        "--series",
        "new",
        "--backend",
        "db",
        "--json_only",
    ]
    jan_args = feb_args.copy()
    jan_args[jan_args.index("2024-02-28")] = "2024-01-31"

    feb_new = _run_cli(cli, feb_args, capsys)
    jan_new = _run_cli(cli, jan_args, capsys)

    stock_args = feb_args.copy()
    stock_args[stock_args.index("new")] = "stock"
    stock_result = _run_cli(cli, stock_args, capsys)

    cumulative_new = int(jan_new["value"]) + int(feb_new["value"])

    assert feb_new["series_returned"] == "new"
    assert stock_result["series_returned"] == "stock"
    assert cumulative_new == int(stock_result["value"])
    assert int(stock_result["value"]) == 250

    expected_keys = {"iso3", "hazard_code", "cutoff", "value", "unit", "ok"}
    assert expected_keys.issubset(stock_result)
    assert expected_keys.issubset(feb_new)
