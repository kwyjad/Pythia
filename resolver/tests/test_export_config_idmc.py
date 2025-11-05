from pathlib import Path

import pandas as pd

from resolver.tools.export_facts import DEFAULT_CONFIG, export_facts


def test_export_config_maps_idmc_flow(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    staging_dir = tmp_path / "resolver" / "staging" / "idmc"
    staging_dir.mkdir(parents=True)

    flow_rows = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "as_of_date": "2024-06-30",
                "metric": "new_displacements",
                "value": 123,
                "series_semantics": "new",
                "source": "idmc_idu",
            }
        ]
    )
    flow_rows.to_csv(staging_dir / "flow.csv", index=False)

    out_dir = tmp_path / "exports"
    result = export_facts(
        inp=tmp_path / "resolver" / "staging",
        config_path=DEFAULT_CONFIG,
        out_dir=out_dir,
        write_db="0",
    )

    df = result.dataframe
    assert not df.empty
    idmc_rows = df[df["metric"] == "new_displacements"].reset_index(drop=True)
    assert len(idmc_rows) == 1
    assert idmc_rows.iloc[0]["value"] == "123"

    matched = {Path(entry["path"]).name: entry for entry in result.report["matched_files"]}
    assert "flow.csv" in matched
    assert matched["flow.csv"]["rows_in"] == 1
