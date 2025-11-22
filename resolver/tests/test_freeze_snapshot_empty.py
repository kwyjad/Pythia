import pandas as pd
from pathlib import Path
import pytest

from resolver.tools import freeze_snapshot

pytestmark = [
    pytest.mark.legacy_freeze,
    pytest.mark.xfail(
        reason="Legacy freeze_snapshot pipeline is retired and replaced by DB-backed snapshot builder."
    ),
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
