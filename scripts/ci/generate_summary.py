#!/usr/bin/env python3
"""Generate a lightweight CI SUMMARY.md for artifact uploads."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


def _summarize_emdat_probe(payload: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(payload, Mapping):
        if payload is None:
            return ["- probe.json: missing"]
        return [f"- probe.json: unexpected payload type ({type(payload).__name__})"]

    lines: list[str] = []
    ok = bool(payload.get("ok"))
    status = payload.get("status")
    if status is None:
        status = payload.get("http_status")
    details: list[str] = []
    if isinstance(status, int):
        details.append(f"HTTP {status}")
    elif status:
        details.append(str(status))
    elapsed = payload.get("elapsed_ms")
    if isinstance(elapsed, (int, float)):
        details.append(f"{int(round(elapsed))} ms")
    elif elapsed is not None:
        details.append(f"elapsed={elapsed}")

    line = f"- status: {'ok' if ok else 'fail'}"
    if details:
        line += f" ({', '.join(details)})"
    lines.append(line)

    if payload.get("skipped"):
        lines.append("- note: probe skipped (offline mode)")

    error = payload.get("error")
    if error:
        lines.append(f"- error: {error}")

    api_version = payload.get("api_version")
    if api_version:
        lines.append(f"- api_version: {api_version}")

    version = payload.get("version")
    timestamp = payload.get("timestamp")
    info = payload.get("info")
    if isinstance(info, Mapping):
        version = version or info.get("version")
        timestamp = timestamp or info.get("timestamp")

    if version:
        lines.append(f"- dataset version: {version}")
    if timestamp:
        lines.append(f"- metadata timestamp: {timestamp}")

    recorded_at = payload.get("recorded_at")
    if recorded_at:
        lines.append(f"- recorded_at: {recorded_at}")

    total_available = payload.get("total_available")
    if total_available is not None:
        lines.append(f"- total_available: {total_available}")

    return lines


def _summarize_effective_params(payload: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(payload, Mapping):
        if payload is None:
            return ["- effective_params.json: missing"]
        return [
            f"- effective_params.json: unexpected payload type ({type(payload).__name__})"
        ]

    lines: list[str] = []
    network = payload.get("network") if isinstance(payload.get("network"), Mapping) else {}
    api = payload.get("api") if isinstance(payload.get("api"), Mapping) else {}
    filters = payload.get("filters") if isinstance(payload.get("filters"), Mapping) else {}

    requested = network.get("requested") if isinstance(network, Mapping) else None
    env_value = network.get("env_value") if isinstance(network, Mapping) else None
    if requested is not None or env_value:
        lines.append(
            f"- network: {'on' if requested else 'off'} (env={env_value or 'unset'})"
        )

    base_url = api.get("base_url") if isinstance(api, Mapping) else None
    key_present = api.get("key_present") if isinstance(api, Mapping) else None
    if base_url:
        lines.append(f"- api base: {base_url}")
    if key_present is not None:
        lines.append(f"- api key present: {bool(key_present)}")

    classif = filters.get("classif") if isinstance(filters, Mapping) else []
    iso = filters.get("iso") if isinstance(filters, Mapping) else None
    include_hist = filters.get("include_hist") if isinstance(filters, Mapping) else None
    from_year = filters.get("from") if isinstance(filters, Mapping) else None
    to_year = filters.get("to") if isinstance(filters, Mapping) else None

    if from_year is not None and to_year is not None:
        lines.append(f"- window: {from_year} → {to_year}")
    if include_hist is not None:
        lines.append(f"- include_hist: {bool(include_hist)}")

    classif_count = len(classif) if isinstance(classif, list) else 0
    if classif_count:
        lines.append(f"- classif keys: {classif_count}")
    else:
        lines.append("- classif keys: none")

    if isinstance(iso, list) and iso:
        lines.append(f"- iso filters: {len(iso)}")
    else:
        lines.append("- iso filters: none")

    graph_vars = payload.get("graphQL_vars")
    if isinstance(graph_vars, Mapping):
        lines.append(f"- graphQL vars: {json.dumps(graph_vars, sort_keys=True)}")

    recorded_at = payload.get("recorded_at")
    if recorded_at:
        lines.append(f"- recorded_at: {recorded_at}")

    return lines


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
        probe_payload = None
        if probe_path.exists():
            raw = probe_path.read_text(encoding="utf-8", errors="replace")
            probe_payload = json.loads(raw)
        probe_lines = _summarize_emdat_probe(probe_payload)
    except Exception as exc:  # pragma: no cover - diagnostics only
        probe_lines = [f"- error reading probe.json ({exc})"]

    effective_path = Path("diagnostics/ingestion/emdat/effective_params.json")
    try:
        effective_payload = None
        if effective_path.exists():
            raw = effective_path.read_text(encoding="utf-8", errors="replace")
            effective_payload = json.loads(raw)
        effective_lines = _summarize_effective_params(effective_payload)
    except Exception as exc:  # pragma: no cover - diagnostics only
        effective_lines = [f"- error reading effective_params.json ({exc})"]

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
        content_lines.extend(["## EMDAT Probe", *probe_lines, ""])
    if effective_lines:
        content_lines.extend(["## EMDAT Effective Params", *effective_lines, ""])
    if sample_lines:
        content_lines.extend(["## EMDAT Probe Sample", *sample_lines, ""])
    if preview_status:
        content_lines.extend(["## Export Preview", *preview_status, ""])
    content_lines.extend(sections)
    if sections:
        content_lines.append("")

    out_path.write_text("\n".join(content_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
