"""Download DuckDB-related wheels for offline installation.

Run this script on a machine with internet access to refresh the offline wheel
cache committed under ``tools/offline_wheels``.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REQS = [
    "duckdb==0.10.3",
    "pytest==8.3.2",
    "httpx==0.28.1",
]


def main() -> None:
    target = Path("tools/offline_wheels").resolve()
    target.mkdir(parents=True, exist_ok=True)

    constraints = target / "constraints-db.txt"
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--dest",
        str(target),
        "--constraint",
        str(constraints),
    ] + REQS

    print(f"Downloading wheels to {target}")
    subprocess.check_call(cmd)
    print("Done.")


if __name__ == "__main__":
    main()
