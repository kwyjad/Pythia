# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from resolver.ingestion.dtm_client import CANONICAL_COLUMNS
from resolver.tests.utils import run as run_proc

REPO_ROOT = Path(__file__).resolve().parents[3]
CONNECTORS_REPORT = REPO_ROOT / "diagnostics" / "connectors_report.jsonl"
NEW_CONNECTORS_REPORT = REPO_ROOT / "diagnostics" / "ingestion" / "connectors_report.jsonl"
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
    NEW_CONNECTORS_REPORT.unlink(missing_ok=True)
    SUMMARY_PATH.unlink(missing_ok=True)
    STAGING_CSV.unlink(missing_ok=True)

    env = _clean_environment()
    result = run_proc(
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

    required = {"connector_id", "status", "reason", "counts", "extras", "http", "mode"}
    assert required.issubset(record.keys())
    assert record["connector_id"] in {"dtm", "dtm_client"}
    assert record["status"] in {"ok", "skipped", "error"}
    assert "staging_csv" in record["extras"]
    assert record["extras"]["staging_csv"].endswith("dtm_displacement.csv")
    assert isinstance(record["counts"].get("normalized"), int)

    csv_header = STAGING_CSV.read_text(encoding="utf-8").splitlines()[0].split(",")
    offline_header = [
        "source",
        "country_iso3",
        "admin1",
        "event_id",
        "as_of",
        "month_start",
        "value_type",
        "value",
        "unit",
        "method",
        "confidence",
        "raw_event_id",
        "raw_fields_json",
    ]
    assert csv_header == list(CANONICAL_COLUMNS) or csv_header == offline_header
