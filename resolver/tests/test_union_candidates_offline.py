from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_union_candidates_offline(tmp_path: Path) -> None:
    candidates_dir = tmp_path / "candidates"
    diagnostics_dir = tmp_path / "diagnostics"
    candidates_dir.mkdir()

    idmc = pd.DataFrame(
        {
            "iso3": ["COL"],
            "as_of_date": ["2024-01-31"],
            "metric": ["internal_displacement_new"],
            "value": [1200],
            "source_system": ["IDMC"],
            "collection_type": ["curated_event"],
            "coverage": ["national"],
            "freshness_days": [5],
        }
    )
    idmc.to_csv(candidates_dir / "idmc_candidates.csv", index=False)

    dtm = pd.DataFrame(
        {
            "iso3": ["COL", "COL"],
            "as_of_date": ["2024-01-31", "2024-02-29"],
            "metric": ["displacement_influx_new", "displacement_influx_new"],
            "value": [150, 175],
            "source_system": ["DTM", "DTM"],
            "collection_type": ["flow_monitoring", "flow_monitoring"],
            "coverage": ["corridor", "corridor"],
            "freshness_days": [7, 4],
            "origin_iso3": ["VEN", "VEN"],
            "destination_iso3": ["COL", "COL"],
            "qa_rank": [2, 2],
        }
    )
    dtm.to_csv(candidates_dir / "dtm_candidates.csv", index=False)

    env = os.environ.copy()
    env.update(
        {
            "CANDIDATES_DIR": str(candidates_dir),
            "DIAGNOSTICS_DIR": str(diagnostics_dir),
        }
    )

    subprocess.check_call(
        [sys.executable, "scripts/precedence/union_candidates.py"],
        cwd=REPO_ROOT,
        env=env,
    )

    union_path = diagnostics_dir / "union_candidates.csv"
    summary_path = diagnostics_dir / "union_summary.json"

    assert union_path.exists()
    assert summary_path.exists()

    union = pd.read_csv(union_path)
    assert list(union.columns) == [
        "iso3",
        "as_of_date",
        "metric",
        "value",
        "source_system",
        "collection_type",
        "coverage",
        "freshness_days",
        "origin_iso3",
        "destination_iso3",
        "method_note",
        "series",
        "indicator",
        "indicator_kind",
        "qa_rank",
    ]
    assert len(union) == 3

    numeric_cols = {"value", "freshness_days", "qa_rank"}
    for column in numeric_cols:
        assert pd.api.types.is_numeric_dtype(union[column]), column

    parsed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert parsed_summary["total_rows"] == 3
    assert parsed_summary["by_source_system"]["DTM"] == 2
    assert parsed_summary["by_source_system"]["IDMC"] == 1
    assert parsed_summary["by_metric"]["displacement_influx_new"] == 2
