#!/usr/bin/env python3
"""Generate a lightweight CI SUMMARY.md for artifact uploads."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from .protocol_probe import summarize_graphql_probe


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

    probe_path = Path("diagnostics/ingestion/emdat/probe.json")
    probe_lines: list[str] = []
    try:
        if probe_path.exists():
            raw = probe_path.read_text(encoding="utf-8", errors="replace")
            payload = json.loads(raw)
            if isinstance(payload, dict):
                probe_lines = summarize_graphql_probe(payload)
            else:
                probe_lines = [
                    f"- probe.json: unexpected payload type ({type(payload).__name__})"
                ]
        else:
            probe_lines = [
                f"- probe.json: missing (expected at {probe_path})"
            ]
    except Exception as exc:  # pragma: no cover - diagnostics only
        probe_lines = [f"- error reading probe.json ({exc})"]

    preview_path = Path("diagnostics/ingestion/export_preview/facts.csv")
    preview_status: list[str] = []
    try:
        resolved_preview_path = preview_path if preview_path.exists() else None
        if resolved_preview_path:
            with resolved_preview_path.open(newline="", encoding="utf-8") as fh:
                reader = csv.reader(fh)
                header = next(reader, None)
                rows = sum(1 for _ in reader)
            preview_status.append(
                f"- facts.csv: present ({rows} row{'s' if rows != 1 else ''}) @ {resolved_preview_path}"
            )
            if header:
                preview_status.append(
                    f"  - columns: {', '.join(header)}"
                )
        else:
            preview_status.append(
                f"- facts.csv: missing (expected at {preview_path})"
            )
    except Exception as exc:  # pragma: no cover - diagnostics only
        preview_status.append(
            f"- facts.csv: error reading preview ({exc})"
        )

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
    if probe_lines:
        content_lines.extend(["## EMDAT Reachability", *probe_lines, ""])
    if preview_status:
        content_lines.extend(["## Export Preview", *preview_status, ""])
    content_lines.extend(sections)
    if sections:
        content_lines.append("")

    out_path.write_text("\n".join(content_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
