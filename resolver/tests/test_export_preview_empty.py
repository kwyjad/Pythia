from pathlib import Path

import pandas as pd

from resolver.tools import export_facts
from resolver.tools._facts_preview import CANONICAL_PREVIEW_COLUMNS
from resolver.tools.export_facts import DEFAULT_CONFIG


def test_exporter_writes_preview_for_empty_dataframe(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    data = pd.DataFrame(
        [
            {
                "CountryISO3": "PHL",
                "ReportingDate": "2024-01-15",
                "idp_count": 0,
            }
        ]
    )

    staging = tmp_path / "dtm_displacement.csv"
    data.to_csv(staging, index=False)

    out_dir = tmp_path / "exports"
    out_dir.mkdir()

    result = export_facts.export_facts(
        inp=staging,
        config_path=DEFAULT_CONFIG,
        out_dir=out_dir,
    )

    assert result.dataframe.empty

    preview_path = Path("diagnostics/ingestion/export_preview/facts.csv")
    assert preview_path.exists()

    preview_df = pd.read_csv(preview_path)
    assert preview_df.empty
    assert list(preview_df.columns) == CANONICAL_PREVIEW_COLUMNS
