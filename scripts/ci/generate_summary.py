#!/usr/bin/env python3
"""Generate a lightweight CI SUMMARY.md for artifact uploads."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _read_tail(path: Path, limit: int = 4000) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return "(file not found)"
    except Exception as exc:  # pragma: no cover - defensive
        return f"(error reading {path.name}: {exc})"
    if limit and len(data) > limit:
        return data[-limit:]
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="Path to write SUMMARY.md")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env_lines = [
        f"- OS: {os.uname().sysname if hasattr(os, 'uname') else 'unknown'}",
        f"- Python: {sys.version.split()[0]}",
    ]

    sections: list[str] = []

    junit_path = Path("pytest-junit.xml")
    if not junit_path.exists():
        junit_path = out_path.parent / "pytest-junit.xml"
    if junit_path.exists():
        sections.append("## PyTest JUnit\n```\n" + _read_tail(junit_path, 4000) + "\n```")

    db_junit = Path("db.junit.xml")
    if not db_junit.exists():
        db_junit = out_path.parent / "db.junit.xml"
    if db_junit.exists():
        sections.append("## DB JUnit\n```\n" + _read_tail(db_junit, 4000) + "\n```")

    pytest_log = Path(f".ci/pytest-{os.environ.get('RUNNER_OS', 'Linux')}.out.log")
    if pytest_log.exists():
        sections.append("## pytest stdout (tail)\n```\n" + _read_tail(pytest_log, 4000) + "\n```")

    content_lines = ["# CI Diagnostics Summary", "## Environment", *env_lines, ""]
    content_lines.extend(sections)
    if sections:
        content_lines.append("")

    out_path.write_text("\n".join(content_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
