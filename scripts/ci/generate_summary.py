#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Generate a lightweight CI SUMMARY.md for artifact uploads."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from scripts.ci._emdat_probe import (
    load_effective_payload,
    load_probe_payload,
    summarize_effective,
    summarize_probe,
)


def _summarize_probe_sample(payload: Mapping[str, Any] | None) -> list[str]:
    if payload is None:
        return []
    if not isinstance(payload, Mapping):
        return [
            f"- probe_sample.json: unexpected payload type ({type(payload).__name__})"
        ]

    lines: list[str] = []
    ok = bool(payload.get("ok"))
    status = payload.get("http_status")
    elapsed = payload.get("elapsed_ms")
    details: list[str] = []
    if isinstance(status, int):
        details.append(f"HTTP {status}")
    elif status:
        details.append(str(status))
    if isinstance(elapsed, (int, float)):
        details.append(f"{int(round(elapsed))} ms")
    line = f"- status: {'ok' if ok else 'fail'}"
    if details:
        line += f" ({', '.join(details)})"
    lines.append(line)

    rows = payload.get("rows")
    total = payload.get("total_available")
    lines.append(f"- rows returned: {rows}")
    if total is not None:
        lines.append(f"- total_available: {total}")

    filters = payload.get("filters")
    if isinstance(filters, Mapping):
        window = (filters.get("from"), filters.get("to"))
        lines.append(f"- window: {window[0]} → {window[1]}")

    histogram = payload.get("classif_histogram")
    if isinstance(histogram, list) and histogram:
        formatted = ", ".join(
            f"{entry.get('classif_key')}: {entry.get('count')}" for entry in histogram
        )
        lines.append(f"- classif histogram: {formatted}")

    recorded_at = payload.get("recorded_at")
    if recorded_at:
        lines.append(f"- recorded_at: {recorded_at}")

    error = payload.get("error")
    if error:
        lines.append(f"- error: {error}")

    return lines


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


def _append_preview_validator_section(lines: list[str]) -> None:
    diag_dir = Path("diagnostics/ingestion/export_preview")
    stderr_path = diag_dir / "validator_stderr.txt"
    stdout_path = diag_dir / "validator_stdout.txt"
    if not stderr_path.exists() and not stdout_path.exists():
        return

    lines.extend(["### Preview Validator Output", ""])

    if stderr_path.exists():
        stderr_tail = _read_tail(stderr_path, 1000).strip()
        lines.extend(["**stderr (tail):**", "```", stderr_tail or "<empty>", "```", ""])
    else:
        lines.extend(["**stderr:** (missing)", ""])

    if stdout_path.exists():
        stdout_tail = _read_tail(stdout_path, 1000).strip()
        lines.extend(["**stdout (tail):**", "```", stdout_tail or "<empty>", "```", ""])

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
    try:
        probe_payload = load_probe_payload(probe_path)
        probe_lines = summarize_probe(probe_payload)
    except TypeError as exc:
        probe_lines = [f"- probe.json: {exc}"]
    except Exception as exc:  # pragma: no cover - diagnostics only
        probe_lines = [f"- error reading probe.json ({exc})"]

    effective_path = Path("diagnostics/ingestion/emdat/effective.json")
    try:
        effective_payload = load_effective_payload(effective_path)
        effective_lines = summarize_effective(effective_payload)
    except TypeError as exc:
        effective_lines = [f"- effective.json: {exc}"]
    except Exception as exc:  # pragma: no cover - diagnostics only
        effective_lines = [f"- error reading effective.json ({exc})"]

    sample_path = Path("diagnostics/ingestion/emdat/probe_sample.json")
    try:
        sample_payload = None
        if sample_path.exists():
            raw = sample_path.read_text(encoding="utf-8", errors="replace")
            sample_payload = json.loads(raw)
        sample_lines = _summarize_probe_sample(sample_payload)
    except Exception as exc:  # pragma: no cover - diagnostics only
        sample_lines = [f"- error reading probe_sample.json ({exc})"]

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

        diag_dir = preview_path.parent
        diag_files = [
            (diag_dir / "validator_stdout.txt", "stdout"),
            (diag_dir / "validator_stderr.txt", "stderr"),
        ]
        for diag_path, label in diag_files:
            if not diag_path.exists():
                continue
            snippet = _read_tail(diag_path, 400).strip()
            if snippet:
                headline = snippet.splitlines()[0].strip()
                preview_status.append(
                    f"- validator {label}: {diag_path} — {headline}"
                )
            else:
                preview_status.append(
                    f"- validator {label}: {diag_path} (empty)"
                )
    except Exception as exc:  # pragma: no cover - diagnostics only
        preview_status.append(
            f"- facts.csv: error reading preview ({exc})"
        )

    junit_candidates = [
        Path("diagnostics/pytest-junit.xml"),
        Path("pytest-junit.xml"),
        out_path.parent / "pytest-junit.xml",
    ]
    for candidate in junit_candidates:
        if candidate.exists():
            sections.append(
                "## PyTest JUnit\n```\n" + _read_tail(candidate, 4000) + "\n```")
            break

    db_junit = Path("db.junit.xml")
    if not db_junit.exists():
        db_junit = out_path.parent / "db.junit.xml"
    if db_junit.exists():
        sections.append("## DB JUnit\n```\n" + _read_tail(db_junit, 4000) + "\n```")

    pytest_log = Path(f".ci/pytest-{os.environ.get('RUNNER_OS', 'Linux')}.out.log")
    if pytest_log.exists():
        sections.append("## pytest stdout (tail)\n```\n" + _read_tail(pytest_log, 4000) + "\n```")

    content_lines = ["# CI Diagnostics Summary", "## Environment", *env_lines, ""]
    if effective_lines:
        content_lines.extend(["## EMDAT Effective Mode", *effective_lines, ""])
    if probe_lines:
        content_lines.extend(["## EMDAT Probe", *probe_lines, ""])
    if sample_lines:
        content_lines.extend(["## EMDAT Probe Sample", *sample_lines, ""])
    if preview_status:
        content_lines.extend(["## Export Preview", *preview_status, ""])
    _append_preview_validator_section(content_lines)
    content_lines.extend(sections)
    if sections:
        content_lines.append("")

    out_path.write_text("\n".join(content_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
