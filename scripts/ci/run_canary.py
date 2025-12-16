#!/usr/bin/env python
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Execute the Codex canary test suite."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List

from resolver.ci.canary import CANARY_TESTS


def _parse_failed_tests(output: str) -> List[str]:
    failed: List[str] = []
    for line in output.splitlines():
        line = line.strip()
        if not line or "FAILED" not in line:
            continue
        if line.startswith("FAILED ") and "::" in line:
            # pytest -q uses "FAILED path::test" format
            failed.append(line.split("FAILED ", 1)[1].split(None, 1)[0])
        elif line.startswith("FAILED (failures="):
            # json style, ignore summary lines
            continue
        elif line.endswith("FAILED") and "::" in line:
            # Coverage for verbose output if pytest changes formatting
            failed.append(line.rsplit(" ", 1)[0])
    return failed


def _ensure_report_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default=os.environ.get("CANARY_REPORT", "canary_results.json"),
        help="Path to write JSON summary for downstream workflow steps.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    report_path = _ensure_report_path(args.report)

    print("[canary] Python:", platform.python_version())
    print("[canary] Selected tests:")
    for test in CANARY_TESTS:
        print("  -", test)

    cmd = [sys.executable, "-m", "pytest", "--maxfail=0", "--disable-warnings", "-q", *CANARY_TESTS]
    print("[canary] Command:", " ".join(cmd))
    start = time.time()
    timeout_seconds = int(os.environ.get("CANARY_TIMEOUT_SECONDS", "600"))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    duration = time.time() - start
    output = (result.stdout or "") + (result.stderr or "")

    # Replay captured output so it still appears in the job logs in order.
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")

    failed_tests = _parse_failed_tests(output)
    payload = {
        "tests": CANARY_TESTS,
        "failed_tests": failed_tests,
        "returncode": result.returncode,
        "duration_seconds": duration,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[canary] Duration: {duration:.2f}s")
    print(f"[canary] Report written to {report_path}")

    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
