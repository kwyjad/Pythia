#!/usr/bin/env python3
"""Lightweight smoke tests for the collect-diagnostics composite action."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECT_SH = REPO_ROOT / ".github" / "actions" / "collect-diagnostics" / "collect.sh"


def _run_collect(job_name: str, with_sample_data: bool) -> Path:
    workspace = Path(tempfile.mkdtemp(prefix="collect-test-"))
    try:
        target_script = workspace / "collect.sh"
        shutil.copy2(COLLECT_SH, target_script)
        target_script.chmod(0o755)

        if with_sample_data:
            raw_dir = workspace / "data" / "staging" / "ci-smoke" / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            (raw_dir / "sample.csv").write_text("id,value\n1,example\n", encoding="utf-8")
            logs_dir = workspace / "resolver" / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            (logs_dir / "pipeline.log").write_text("log line\n", encoding="utf-8")
            snapshot_dir = workspace / "data" / "snapshots" / "ci-smoke"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            (snapshot_dir / "facts_resolved.parquet").write_bytes(b"PAR1")

        env = os.environ.copy()
        env.update(
            {
                "INPUT_JOB_NAME": job_name,
                "GITHUB_WORKSPACE": str(workspace),
                "GITHUB_RUN_ID": "12345",
                "GITHUB_RUN_ATTEMPT": "2",
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            }
        )
        subprocess.run(["bash", str(target_script)], check=True, cwd=workspace, env=env)

        archive = workspace / "dist" / f"diagnostics-{job_name}-12345-2.zip"
        if not archive.exists():
            raise AssertionError(f"Expected archive {archive} to exist")

        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
        expected_summary = f"diagnostics/{job_name}/SUMMARY.md"
        if expected_summary not in names:
            raise AssertionError(f"Summary {expected_summary} missing from archive contents: {sorted(names)}")

        if with_sample_data:
            staging_listing = f"diagnostics/{job_name}/data/staging/listing.txt"
            if staging_listing not in names:
                raise AssertionError("Expected staging listing to be captured for sample data run")

        return archive
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def main() -> int:
    _run_collect("job-empty", with_sample_data=False)
    _run_collect("job-data", with_sample_data=True)
    print("collect.sh smoke tests completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
