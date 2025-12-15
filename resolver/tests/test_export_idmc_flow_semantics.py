# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from pathlib import Path

import pandas as pd

from resolver.tools.export_facts import DEFAULT_CONFIG, export_facts


def test_idmc_flow_rows_route_to_new_semantics(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    staging_dir = tmp_path / "resolver" / "staging" / "idmc"
    staging_dir.mkdir(parents=True)

    flow_rows = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "as_of_date": "2024-06-30",
                "value": 123,
                "series_semantics": "stock",
                "source": "IDMC",
            }
        ]
    )
    flow_rows.to_csv(staging_dir / "flow.csv", index=False)

    result = export_facts(
        inp=tmp_path / "resolver" / "staging",
        config_path=DEFAULT_CONFIG,
        out_dir=tmp_path / "exports",
        write_db="0",
        only_strategy="idmc-staging",
    )

    df = result.dataframe
    assert not df.empty

    idmc_flow = df[df["metric"] == "new_displacements"].reset_index(drop=True)
    assert len(idmc_flow) == 1
    assert idmc_flow.iloc[0]["series_semantics"] == "new"

    matched = {Path(entry["path"]).name: entry for entry in result.report["matched_files"]}
    assert matched["flow.csv"]["strategy"] == "idmc-staging"
