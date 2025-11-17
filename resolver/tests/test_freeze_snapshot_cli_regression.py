import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _pythonpath_env() -> str:
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(_repo_root())]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def test_freeze_snapshot_cli_regression(tmp_path):
    facts_path = tmp_path / "facts.csv"
    data = pd.DataFrame(
        [
            {
                "iso3": "KEN",
                "ym": "2024-01",
                "hazard_code": "CU",
                "hazard_label": "Conflict",
                "hazard_class": "conflict",
                "metric": "events",
                "value": "5",
                "as_of_date": "2024-01-15",
                "publication_date": "2024-01-20",
                "publisher": "ACLED",
                "source_type": "dataset",
            }
        ]
    )
    data.to_csv(facts_path, index=False)

    outdir = tmp_path / "snapshots"
    cmd = [
        sys.executable,
        "-m",
        "resolver.tools.freeze_snapshot",
        "--facts",
        str(facts_path),
        "--month",
        "2024-01",
        "--outdir",
        str(outdir),
        "--write-db",
        "0",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = _pythonpath_env()

    result = subprocess.run(
        cmd,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    resolved_csv = outdir / "2024-01" / "facts_resolved.csv"
    assert resolved_csv.exists()
