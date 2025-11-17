from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from resolver.tools import build_forecaster_features as features


def test_build_features_falls_back_to_as_of(tmp_path: Path) -> None:
    snapshots_dir = tmp_path / "snapshots"
    month_dir = snapshots_dir / "2024-01"
    month_dir.mkdir(parents=True)

    resolved = pd.DataFrame(
        [
            {
                "ym": "2024-01",
                "iso3": "AAA",
                "hazard_code": "DR",
                "metric": "in_need",
                "as_of": "2024-01-05",
                "hazard_class": "natural",
                "precedence_tier": "tier1",
            }
        ]
    )
    deltas = pd.DataFrame(
        [
            {
                "ym": "2024-01",
                "iso3": "AAA",
                "hazard_code": "DR",
                "metric": "in_need",
                "value_new": 50,
                "as_of": "2024-01-15",
            }
        ]
    )

    resolved.to_csv(month_dir / "facts_resolved.csv", index=False)
    deltas.to_csv(month_dir / "facts_deltas.csv", index=False)
    (month_dir / "manifest.json").write_text(
        json.dumps({"created_at_utc": "2024-02-01T00:00:00Z"}),
        encoding="utf-8",
    )

    output_path = tmp_path / "resolver_features.parquet"
    frame = features.build_features(
        snapshots_dir=snapshots_dir,
        output_path=output_path,
        db_url=None,
        metrics=("in_need",),
    )

    assert not frame.empty
    assert output_path.exists()
    assert frame.iloc[0]["as_of_date"] == "2024-01-15"
    assert frame.attrs.get("as_of_date_fallback_used") is True
