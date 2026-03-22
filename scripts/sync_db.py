# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Download the latest pythia-resolver-db artifact from a successful CI run.

Usage
-----
    python -m scripts.sync_db                   # latest HS run
    python -m scripts.sync_db --run 23382871054 # specific run ID

Requires the ``gh`` CLI to be installed and authenticated.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pythia.config import load as load_cfg

_ARTIFACT_NAME = "pythia-resolver-db"
_DB_FILENAME = "resolver.duckdb"

# Candidate workflows that produce the canonical DB artifact, searched in order.
_CANDIDATE_WORKFLOWS = [
    "run_horizon_scanner.yml",
    "ingest-structured-data.yml",
    "compute_scores.yml",
    "compute_calibration_pythia.yml",
]


def _find_latest_run(workflow: str | None = None) -> int | None:
    """Return the run ID of the most recent successful run with a DB artifact."""
    for wf in ([workflow] if workflow else _CANDIDATE_WORKFLOWS):
        cmd = [
            "gh", "run", "list",
            f"--workflow={wf}",
            "--status=success",
            "--limit=1",
            "--json=databaseId,displayTitle,createdAt",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            continue
        try:
            runs = json.loads(result.stdout)
        except json.JSONDecodeError:
            continue
        if runs:
            run = runs[0]
            print(
                f"Found: {run.get('displayTitle', '?')} "
                f"({run.get('createdAt', '?')}) from {wf}"
            )
            return int(run["databaseId"])
    return None


def _download_artifact(run_id: int, dest: Path) -> None:
    """Download the DB artifact from a specific run and copy to *dest*."""
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "gh", "run", "download", str(run_id),
            f"--name={_ARTIFACT_NAME}",
            "--dir", tmp,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"✗ Failed to download artifact: {result.stderr.strip()}")
            sys.exit(1)

        src = Path(tmp) / _DB_FILENAME
        if not src.exists():
            print(f"✗ Artifact downloaded but {_DB_FILENAME} not found")
            sys.exit(1)

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def _db_dest() -> Path:
    """Resolve the local DuckDB path from Pythia config."""
    cfg = load_cfg()
    db_url = cfg.get("app", {}).get("db_url", "")
    if not db_url:
        print("✗ No db_url in pythia config")
        sys.exit(1)
    return Path(db_url.replace("duckdb:///", "", 1))


def sync(run_id: int | None = None, workflow: str | None = None) -> None:
    """Download the latest (or specified) DB artifact to the local DB path."""
    dest = _db_dest()

    if run_id is None:
        print("Searching for latest successful CI run …")
        run_id = _find_latest_run(workflow)
        if run_id is None:
            print("✗ No successful run found with a DB artifact")
            sys.exit(1)

    print(f"Downloading artifact from run {run_id} …")
    _download_artifact(run_id, dest)
    print(f"✓ Synced DB → {dest}  (run {run_id})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync local DuckDB from a CI artifact."
    )
    parser.add_argument(
        "--run",
        type=int,
        default=None,
        help="Specific GitHub Actions run ID to download from.",
    )
    parser.add_argument(
        "--workflow",
        type=str,
        default=None,
        help="Limit search to a specific workflow file (e.g. run_horizon_scanner.yml).",
    )
    args = parser.parse_args()
    sync(run_id=args.run, workflow=args.workflow)


if __name__ == "__main__":
    main()
