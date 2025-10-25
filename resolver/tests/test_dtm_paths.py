from pathlib import Path

import pandas as pd

from resolver.ingestion import dtm_client as dc


def test_diagnostics_base_is_repo_root():
    repo_root = Path(__file__).resolve().parents[2]
    expected = repo_root / "diagnostics" / "ingestion"
    assert dc.DIAGNOSTICS_DIR == expected


def test_static_iso3_roster_present_and_ok():
    path = Path(dc.STATIC_ISO3_PATH)
    assert path.exists(), f"Missing {path}"
    df = pd.read_csv(path)
    assert {"admin0Pcode", "admin0Name"} <= set(df.columns)
    assert len(df) > 180
