from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from resolver.ingestion.dtm_client import CANONICAL_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[3]
CONNECTORS_REPORT = REPO_ROOT / "diagnostics" / "connectors_report.jsonl"
STAGING_CSV = REPO_ROOT / "resolver" / "staging" / "dtm_displacement.csv"
SUMMARY_PATH = REPO_ROOT / "diagnostics" / "ingestion" / "dtm" / "summary.json"


def _clean_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("DTM_API_KEY", None)
    env.pop("DTM_SUBSCRIPTION_KEY", None)
    repo_str = str(REPO_ROOT)
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = repo_str if not pythonpath else f"{repo_str}:{pythonpath}"
    return env


def test_connectors_report_entry_is_emitted() -> None:
    CONNECTORS_REPORT.unlink(missing_ok=True)
    SUMMARY_PATH.unlink(missing_ok=True)
    STAGING_CSV.unlink(missing_ok=True)

    env = _clean_environment()
    result = subprocess.run(
        [sys.executable, "-m", "resolver.ingestion.dtm_client", "--offline-smoke"],
        check=False,
        env=env,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0
    assert CONNECTORS_REPORT.exists()

    with CONNECTORS_REPORT.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    assert len(records) == 1
    record = records[0]

    required = {"connector", "status", "reason", "rows_out", "output_path", "started_at", "ended_at", "elapsed_s"}
    assert required.issubset(record.keys())
    assert record["connector"] == "dtm"
    assert record["status"] in {"ok", "skipped", "error"}
    assert isinstance(record["rows_out"], int)
    assert record["output_path"].endswith("dtm_displacement.csv")
    assert record["started_at"].endswith("Z")
    assert record["ended_at"].endswith("Z")
    assert isinstance(record["elapsed_s"], (int, float))

    csv_header = STAGING_CSV.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert csv_header == list(CANONICAL_COLUMNS)
