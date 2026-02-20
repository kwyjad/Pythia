# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import pandas as pd
from pathlib import Path
import pytest

from resolver.tools import freeze_snapshot

pytestmark = [
    pytest.mark.legacy_freeze,
]


def test_freeze_snapshot_skips_empty_facts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    facts_path = Path("facts.csv")
    empty_df = pd.DataFrame(
        columns=["iso3", "as_of_date", "metric", "value", "source"]
    )
    empty_df.to_csv(facts_path, index=False)

    def _fail_validator(path: Path) -> None:
        raise AssertionError("validator should not run for empty facts")

    monkeypatch.setattr(freeze_snapshot, "run_validator", _fail_validator)

    out_dir = tmp_path / "snapshots"
    result = freeze_snapshot.freeze_snapshot(
        facts=facts_path,
        month="2024-01",
        outdir=out_dir,
    )

    assert result.skipped is True
    assert "No rows" in result.skip_reason
    assert not (out_dir / "2024-01").exists()
    assert result.resolved_csv is None
    assert result.resolved_parquet is None
    assert result.manifest is None
