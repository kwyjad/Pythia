from pathlib import Path

import pandas as pd
import pytest

from resolver.tools import freeze_snapshot

pytest.importorskip("duckdb")


@pytest.mark.duckdb
def test_snapshot_write_appends_counts_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    month = "2024-05"
    monkeypatch.chdir(tmp_path)

    data = pd.DataFrame(
        [
            {
                "iso3": "AAA",
                "hazard_code": "TC",
                "metric": "in_need",
                "value": 10,
                "series_semantics": "stock",
                "ym": month,
            },
            {
                "iso3": "BBB",
                "hazard_code": "EQ",
                "metric": "affected",
                "value": 5,
                "series_semantics": "stock",
                "ym": month,
            },
        ]
    )

    facts_path = tmp_path / "facts.csv"
    data.to_csv(facts_path, index=False)

    db_path = tmp_path / "counts.duckdb"
    db_url = f"duckdb:///{db_path.as_posix()}"

    freeze_snapshot._maybe_write_db(
        facts_path=facts_path,
        resolved_path=facts_path,
        deltas_path=None,
        manifest_path=None,
        month=month,
        db_url=db_url,
        write_db=True,
    )

    summary_path = tmp_path / "diagnostics" / "summary.md"
    assert summary_path.exists()

    summary_text = summary_path.read_text(encoding="utf-8")
    assert "### Snapshot DB write — pre/post counts" in summary_text
    assert f"ym: `{month}`" in summary_text
    assert "routing: `resolved_passthrough`" in summary_text
    assert "wrote: resolved=2, deltas=0" in summary_text
