from pathlib import Path

import pandas as pd

from resolver.tools import export_facts
from resolver.tools._facts_preview import CANONICAL_PREVIEW_COLUMNS


def test_emdat_preview_header_matches_canonical(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    staging_dir = tmp_path / "resolver" / "staging"
    staging_dir.mkdir(parents=True)

    sample = pd.DataFrame(
        [
            {
                "iso3": "BGD",
                "ym": "2022-05",
                "as_of_date": "2022-05-31",
                "shock_type": "flood",
                "pa": 1234,
            }
        ]
    )
    sample.to_csv(staging_dir / "emdat_pa.csv", index=False)

    repo_root = Path(__file__).resolve().parents[2]
    export_facts.export_facts(
        inp=staging_dir,
        config_path=repo_root / "resolver" / "tools" / "export_config.yml",
        out_dir=tmp_path / "resolver" / "exports",
        write_db=False,
    )

    preview_path = Path("diagnostics/ingestion/export_preview/facts.csv")
    assert preview_path.exists()

    header = preview_path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == CANONICAL_PREVIEW_COLUMNS
