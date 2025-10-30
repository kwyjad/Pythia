from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_precedence_cli_smoke_offline(tmp_path: Path) -> None:
    candidates = pd.DataFrame(
        [
            {
                "iso3": "COL",
                "as_of_date": "2024-01-31",
                "metric": "displacement_influx_new",
                "value": 150.0,
                "source_system": "UNHCR",
                "collection_type": "registration",
                "coverage": "national",
                "freshness_days": 3,
                "indicator": "arrivals",
            },
            {
                "iso3": "COL",
                "as_of_date": "2024-01-31",
                "metric": "displacement_influx_new",
                "value": 200.0,
                "source_system": "DTM",
                "collection_type": "flow_monitoring",
                "coverage": "corridor",
                "freshness_days": 4,
                "indicator_kind": "flow",
            },
        ]
    )

    union_path = tmp_path / "union_candidates.csv"
    candidates.to_csv(union_path, index=False)

    selected_path = tmp_path / "selected.csv"

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "resolver.cli.precedence_cli",
            "--config",
            str(REPO_ROOT / "tools/precedence_config.yml"),
            "--candidates",
            str(union_path),
            "--out",
            str(selected_path),
        ],
        cwd=REPO_ROOT,
        env=os.environ.copy(),
    )

    assert selected_path.exists()

    selected = pd.read_csv(selected_path, parse_dates=["as_of_date"])
    assert len(selected) == 1
    row = selected.iloc[0]
    assert row["iso3"] == "COL"
    assert row["metric"] == "displacement_influx_new"
    assert row["source_system"] == "UNHCR"
    assert pytest.approx(row["value"], rel=1e-6) == 150.0
    assert row["semantics"] == "new"
