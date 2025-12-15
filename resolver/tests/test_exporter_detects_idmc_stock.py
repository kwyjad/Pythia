# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from pathlib import Path

import pandas as pd

from resolver.ingestion.idmc.exporter import STOCK_EXPORT_COLUMNS
from resolver.tools.export_facts import DEFAULT_CONFIG, export_facts


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_exporter_detects_idmc_stock(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    staging_dir = tmp_path / "resolver" / "staging" / "idmc"
    staging_dir.mkdir(parents=True)
    _write_csv(
        staging_dir / "flow.csv",
        [
            {
                "iso3": "AAA",
                "as_of_date": "2024-01-31",
                "metric": "new_displacements",
                "value": 1,
                "series_semantics": "new",
                "source": "Unit Test",
            }
        ],
    )
    _write_csv(
        staging_dir / "stock.csv",
        [
            {
                "iso3": "BBB",
                "as_of_date": "2023-12-31",
                "metric": "idp_displacement_stock_idmc",
                "value": 2,
                "series_semantics": "stock",
                "source": "Unit Test",
            },
            {
                "iso3": "CCC",
                "as_of_date": "2023-12-31",
                "metric": "idp_displacement_stock_idmc",
                "value": 3,
                "series_semantics": "stock",
                "source": "Unit Test",
            },
        ],
    )

    out_dir = tmp_path / "exports"
    result = export_facts(
        inp=tmp_path / "resolver" / "staging",
        config_path=DEFAULT_CONFIG,
        out_dir=out_dir,
        write_db="0",
    )

    matched = {Path(entry["path"]).name: entry for entry in result.report["matched_files"]}
    assert "flow.csv" in matched
    assert "stock.csv" in matched
    assert matched["flow.csv"]["rows_in"] == 1
    assert matched["stock.csv"]["rows_in"] == 2


def test_exporter_marks_empty_stock_as_matched(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    staging_dir = tmp_path / "resolver" / "staging" / "idmc"
    staging_dir.mkdir(parents=True)
    _write_csv(
        staging_dir / "flow.csv",
        [
            {
                "iso3": "AAA",
                "as_of_date": "2024-01-31",
                "metric": "new_displacements",
                "value": 5,
                "series_semantics": "new",
                "source": "Unit Test",
            }
        ],
    )
    pd.DataFrame(columns=STOCK_EXPORT_COLUMNS).to_csv(
        staging_dir / "stock.csv", index=False
    )

    out_dir = tmp_path / "exports"
    result = export_facts(
        inp=tmp_path / "resolver" / "staging",
        config_path=DEFAULT_CONFIG,
        out_dir=out_dir,
        write_db="0",
    )

    matched = {Path(entry["path"]).name: entry for entry in result.report["matched_files"]}
    assert matched["stock.csv"]["rows_in"] == 0
    assert "stock.csv" not in result.report.get("unmatched_files", [])
    assert all("regex_miss" not in warning for warning in result.report["warnings"])
