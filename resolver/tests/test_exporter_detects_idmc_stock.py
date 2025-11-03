from pathlib import Path

import pandas as pd

from resolver.tools.export_facts import DEFAULT_CONFIG, export_facts


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_exporter_detects_idmc_stock(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    staging_dir = tmp_path / "resolver" / "staging" / "idmc"
    staging_dir.mkdir(parents=True)
    _write_csv(staging_dir / "flow.csv", [{"iso3": "AAA", "value": 1}])
    _write_csv(
        staging_dir / "stock.csv",
        [
            {"iso3": "BBB", "value": 2},
            {"iso3": "CCC", "value": 3},
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
