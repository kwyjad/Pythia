#!/usr/bin/env python3
"""Generate an AI friendly SUMMARY.md for CI diagnostics.

The script is intentionally defensive – nothing raised here should fail the
workflow.  Every error is caught and converted into textual notes inside the
summary file.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import xml.etree.ElementTree as ET

SUMMARY_DIR = Path(".ci/diagnostics")
SUMMARY_PATH = SUMMARY_DIR / "SUMMARY.md"

# Thresholds in bytes
LOG_TRUNCATE_THRESHOLD = int(1.5 * 1024 * 1024)
LOG_HEAD_BYTES = 256 * 1024
LOG_TAIL_BYTES = 512 * 1024

DUCKDB_SAMPLE_LIMIT = 20
DUCKDB_ROWCOUNT_LIMIT = 100_000

SECRET_PATTERNS = re.compile(r"token|secret|key|password|pwd", re.IGNORECASE)
ENV_INTERESTING_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^RESOLVER_",
        r"duckdb",
        r"database",
        r"db_",
        r"^AWS_",
    )
]


def _safe_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _gather_metadata() -> str:
    now = _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    env = os.environ
    lines = ["# CI Diagnostics Summary", ""]
    lines.append("## Run Metadata")
    lines.append("")
    fields = {
        "Commit": env.get("GITHUB_SHA", "unknown"),
        "Ref": env.get("GITHUB_REF", env.get("GITHUB_HEAD_REF", "unknown")),
        "Workflow": env.get("GITHUB_WORKFLOW", "unknown"),
        "Job": env.get("GITHUB_JOB", "unknown"),
        "Job Name": env.get("GITHUB_JOB_NAME", env.get("JOB_NAME", "unknown")),
        "Job ID": env.get("GITHUB_RUN_ID", "unknown"),
        "Run Attempt": env.get("GITHUB_RUN_ATTEMPT", "unknown"),
        "Timestamp (UTC)": now,
    }
    for key, value in fields.items():
        lines.append(f"- **{key}:** {value}")
    lines.append("")
    return "\n".join(lines)


def _gather_environment(diag_root: Path) -> str:
    lines = ["## Environment", ""]
    uname = platform.platform()
    python = sys.version.replace("\n", " ")
    lines.append(f"- **OS:** {uname}")
    lines.append(f"- **Python:** {python}")

    pip_freeze_path = diag_root / "pip-freeze.txt"
    if pip_freeze_path.exists():
        content = pip_freeze_path.read_text(encoding="utf-8", errors="replace").strip()
        if content:
            lines.append("- **pip freeze:** captured from diagnostics")
            lines.append("\n````text\n" + content + "\n````")
        else:
            lines.append("- **pip freeze:** file present but empty")
    else:
        lines.append("- **pip freeze:** not present in diagnostics")

    try:
        import duckdb  # type: ignore

        lines.append(f"- **DuckDB:** {duckdb.__version__}")
    except Exception as exc:  # pragma: no cover - optional dependency
        lines.append(f"- **DuckDB:** unavailable ({exc})")

    lines.append("")
    return "\n".join(lines)


def _redact_value(name: str, value: str) -> str:
    if not value:
        return ""
    if SECRET_PATTERNS.search(name) or SECRET_PATTERNS.search(value):
        return "***redacted***"
    if len(value) <= 8:
        return "***"
    return f"***{value[-4:]}"


def _gather_workflow_context() -> str:
    lines = ["## Workflow Context", ""]
    entries: List[Tuple[str, str]] = []
    for key, value in sorted(os.environ.items()):
        if any(p.search(key) for p in ENV_INTERESTING_PATTERNS):
            display = _redact_value(key, value)
            entries.append((key, display))
    if not entries:
        lines.append("(No relevant environment variables detected.)")
    else:
        for key, value in entries:
            lines.append(f"- `{key}` = {value}")
    lines.append("")
    return "\n".join(lines)


def _parse_pytest_results(junit_path: Path) -> Dict[str, object]:
    stats = {
        "tests": 0,
        "errors": 0,
        "failures": 0,
        "skipped": 0,
        "xfail": 0,
        "passed": 0,
        "details": [],
    }
    try:
        tree = ET.parse(junit_path)
        root = tree.getroot()
    except Exception as exc:
        stats["error"] = f"Failed to parse JUnit XML: {exc}"
        return stats

    testsuite_elements = root.findall(".//testsuite")
    if not testsuite_elements:
        testsuite_elements = [root]

    total_failures = 0
    total_errors = 0
    total_skipped = 0
    total_tests = 0
    details: List[Dict[str, str]] = []

    for suite in testsuite_elements:
        total_tests += int(suite.attrib.get("tests", 0))
        total_failures += int(suite.attrib.get("failures", 0))
        total_errors += int(suite.attrib.get("errors", 0))
        total_skipped += int(suite.attrib.get("skipped", 0))
        for case in suite.findall("testcase"):
            failure_nodes = case.findall("failure")
            error_nodes = case.findall("error")
            if not failure_nodes and not error_nodes:
                continue
            detail_lines = []
            for node in failure_nodes + error_nodes:
                message = node.attrib.get("message", "(no message)")
                node_type = node.attrib.get("type", node.tag)
                text = node.text or ""
                detail_lines.append(
                    f"### {node_type}: {message}\n\n````text\n{text.strip()}\n````"
                )
            system_out = "\n".join(
                [n.text or "" for n in case.findall("system-out")] +
                [n.text or "" for n in case.findall("system-err")]
            ).strip()
            if system_out:
                detail_lines.append("**Captured Output:**\n\n````text\n" + system_out + "\n````")

            classname = case.attrib.get("classname", "")
            name = case.attrib.get("name", "")
            nodeid = case.attrib.get("file", "")
            header = f"## Test Failure: {classname}::{name}" if classname else f"## Test Failure: {name}"
            if nodeid:
                header += f"\n\n- File: `{nodeid}`"
            details.append({
                "header": header,
                "body": "\n\n".join(detail_lines),
            })

    stats.update(
        {
            "tests": total_tests,
            "failures": total_failures,
            "errors": total_errors,
            "skipped": total_skipped,
            "passed": max(total_tests - total_failures - total_errors - total_skipped, 0),
            "details": details,
        }
    )
    return stats


def _extract_pytest_failure_names(diag_root: Path, limit: int = 10) -> List[str]:
    candidates: List[str] = []
    seen = set()
    for tail_path in sorted(diag_root.glob("pytest-*.tail.txt")):
        try:
            text = tail_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("FAILED "):
                continue
            remainder = stripped[len("FAILED "):]
            name = remainder.split(" - ", 1)[0].strip()
            if not name:
                continue
            if name not in seen:
                candidates.append(name)
                seen.add(name)
            if len(candidates) >= limit:
                return candidates
    return candidates


def _render_pytest_summary(diag_root: Path) -> str:
    lines = ["## Pytest Summary", ""]
    junit_path = diag_root / "pytest-junit.xml"
    if not junit_path.exists():
        lines.append("No pytest JUnit report found (expected `.ci/diagnostics/pytest-junit.xml`).")
        failures = _extract_pytest_failure_names(diag_root)
        if failures:
            lines.append("")
            lines.append("Potential failing tests inferred from pytest output:")
            for name in failures:
                lines.append(f"- {name}")
        lines.append("")
        return "\n".join(lines)

    stats = _parse_pytest_results(junit_path)
    if "error" in stats:
        lines.append(str(stats["error"]))
        lines.append("")
        return "\n".join(lines)

    lines.append(
        "- **Total:** {tests} | **Passed:** {passed} | **Failures:** {failures} | "
        "**Errors:** {errors} | **Skipped:** {skipped}".format(**stats)
    )

    details: Sequence[Dict[str, str]] = stats.get("details", [])  # type: ignore
    if details:
        for item in details:
            lines.append("")
            lines.append(item["header"])
            lines.append("")
            lines.append(item["body"])
            lines.append("")
    else:
        lines.append("\nAll tests passed.")

    lines.append("")
    return "\n".join(lines)


def _read_file_section(path: Path) -> str:
    size = path.stat().st_size
    header = [f"### Log: `{path}`", "", f"- Size: {size} bytes"]
    if size > LOG_TRUNCATE_THRESHOLD:
        header.append("- Note: **TRUNCATED** to keep artifact manageable.")
        header.append("")
        with path.open("rb") as fh:
            head = fh.read(LOG_HEAD_BYTES)
            fh.seek(max(size - LOG_TAIL_BYTES, 0))
            tail = fh.read()
        parts = [
            "````text",
            head.decode("utf-8", errors="replace"),
            "--- TRUNCATED ---",
            tail.decode("utf-8", errors="replace"),
            "````",
        ]
    else:
        data = path.read_text(encoding="utf-8", errors="replace")
        parts = ["````text", data, "````"]
    return "\n".join(header + parts + [""])


def _gather_logs(root: Path) -> str:
    lines = ["## Key Logs", ""]
    patterns = [
        root.glob("*.log"),
        root.glob("resolver/**/*.log"),
        root.glob("resolver/**/logs/**/*.log"),
        root.glob("resolver/ingestion/**/*.log"),
    ]
    seen: Dict[Path, None] = {}
    for pattern in patterns:
        for path in pattern:
            if path.is_file():
                seen.setdefault(path.resolve(), None)
    log_paths = sorted(seen.keys())
    if not log_paths:
        lines.append("No log files discovered.")
        lines.append("")
        return "\n".join(lines)

    for path in log_paths:
        try:
            lines.append(_read_file_section(path))
        except Exception as exc:
            lines.append(f"### Log: `{path}`\n\nFailed to read log: {exc}\n")
    return "\n".join(lines)


def _inspect_duckdb(root: Path) -> str:
    lines = ["## DuckDB Inspection", ""]
    try:
        import duckdb  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        lines.append(f"DuckDB module unavailable: {exc}")
        lines.append("")
        return "\n".join(lines)

    duckdb_paths = [
        path
        for path in root.rglob("*.duckdb")
        if path.is_file() and path.stat().st_size <= 100 * 1024 * 1024
    ]

    if not duckdb_paths:
        lines.append("No DuckDB files discovered (<=100MB).")
        lines.append("")
        return "\n".join(lines)

    for db_path in sorted(duckdb_paths):
        lines.append(f"### Database: `{db_path}`")
        try:
            conn = duckdb.connect(str(db_path), read_only=True)
        except Exception as exc:
            lines.append(f"- Unable to open database: {exc}\n")
            continue
        try:
            tables = conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema NOT IN ('information_schema', 'pg_catalog')"
            ).fetchall()
            table_names = [row[0] for row in tables]
            if not table_names:
                lines.append("- No user tables found.\n")
                conn.close()
                continue
            for table in table_names:
                lines.append(f"#### Table `{table}`")
                try:
                    count = conn.execute(
                        f"SELECT COUNT(*) FROM {table} LIMIT {DUCKDB_ROWCOUNT_LIMIT}"
                    ).fetchone()[0]
                    lines.append(f"- Approximate rows (limit {DUCKDB_ROWCOUNT_LIMIT}): {count}")
                except Exception as exc:
                    lines.append(f"- Row count unavailable: {exc}")
                try:
                    info = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
                    schema_lines = ["| cid | name | type | notnull | dflt | pk |", "| --- | --- | --- | --- | --- | --- |"]
                    for row in info:
                        schema_lines.append(
                            "| {cid} | {name} | {type} | {notnull} | {dflt_value} | {pk} |".format(
                                cid=row[0],
                                name=row[1],
                                type=row[2],
                                notnull=row[3],
                                dflt_value=row[4] if row[4] is not None else "",
                                pk=row[5],
                            )
                        )
                    lines.extend(schema_lines)
                except Exception as exc:
                    lines.append(f"- Schema inspection failed: {exc}")

                try:
                    indexes = conn.execute("PRAGMA show_indexes").fetchall()
                    relevant = [idx for idx in indexes if idx[1] == table]
                    if relevant:
                        lines.append("- Indexes:")
                        for idx in relevant:
                            lines.append(
                                f"  - {idx[0]} (unique={idx[2]}, columns={idx[3]})"
                            )
                    else:
                        lines.append("- Indexes: none")
                except Exception as exc:
                    lines.append(f"- Index inspection failed: {exc}")

                if re.search(r"snapshot|fact", table, re.IGNORECASE):
                    try:
                        sample = conn.execute(
                            f"SELECT * FROM {table} LIMIT {DUCKDB_SAMPLE_LIMIT}"
                        ).fetchdf()
                        lines.append("- Sample rows:")
                        lines.append("\n````json")
                        lines.append(
                            sample.to_json(orient="records", indent=2)  # type: ignore[attr-defined]
                            if hasattr(sample, "to_json")
                            else json.dumps(sample, indent=2)
                        )
                        lines.append("````")
                    except Exception as exc:
                        lines.append(f"- Sample query failed: {exc}")
                lines.append("")
        finally:
            try:
                conn.close()
            except Exception:
                pass
    return "\n".join(lines)


def _tail_repo_logs(root: Path) -> str:
    lines = ["## Repository Log Tail", ""]
    git_log = root / ".git" / "logs" / "HEAD"
    if git_log.exists():
        try:
            tail = git_log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            lines.append("````text")
            lines.extend(tail)
            lines.append("````")
        except Exception as exc:
            lines.append(f"Failed to read git HEAD log: {exc}")
    else:
        lines.append("Git log not available.")
    lines.append("")
    return "\n".join(lines)


def _render_command_tails(diag_root: Path) -> str:
    lines = ["## Command Tails", ""]
    tail_files = sorted(diag_root.glob("*.tail.txt"))
    if not tail_files:
        lines.append("No command tail files captured.")
        lines.append("")
        return "\n".join(lines)

    for tail_path in tail_files:
        try:
            content = tail_path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as exc:
            lines.append(f"### {tail_path.name}\n\nFailed to read tail file: {exc}\n")
            continue
        title = tail_path.name.replace(".tail.txt", "")
        lines.append(f"### {title}")
        lines.append("")
        if content:
            lines.append("````text")
            lines.append(content)
            lines.append("````")
        else:
            lines.append("(no output captured)")
        lines.append("")
    return "\n".join(lines)


def _read_json_payload(report_path: Path) -> Dict[str, object] | None:
    try:
        lines = report_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    for idx, line in enumerate(lines):
        if line.strip().startswith("{"):
            text = "\n".join(lines[idx:])
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                return None
            if isinstance(payload, dict):
                return payload
            return None
    return None


def _render_canonical_listing(diag_root: Path) -> str:
    lines = ["## Canonical Listing", ""]
    report = diag_root / "canonical-listing.txt"
    if not report.exists():
        lines.append("Canonical listing report not present.")
        lines.append("")
        return "\n".join(lines)

    payload = _read_json_payload(report)
    if not payload:
        lines.append(f"Could not parse canonical listing payload from `{report}`.")
        lines.append("")
        return "\n".join(lines)

    directory = payload.get("directory", "(unknown)")
    exists = payload.get("exists")
    total_rows = payload.get("total_rows", "n/a")
    unknown_counts = payload.get("unknown_row_counts", 0)
    files = payload.get("files", [])
    lines.append(f"- Directory: `{directory}`")
    lines.append(f"- Exists: {exists}")
    lines.append(f"- Files listed: {len(files) if isinstance(files, list) else 'n/a'}")
    lines.append(f"- Total rows: {total_rows}")
    if isinstance(unknown_counts, int) and unknown_counts:
        lines.append(f"- Files with unknown row count: {unknown_counts}")
    lines.append(f"- Detailed report: `{report}`")
    lines.append("")
    return "\n".join(lines)


def _render_duckdb_counts(diag_root: Path) -> str:
    lines = ["## DuckDB Table Counts", ""]
    report = diag_root / "duckdb-counts.txt"
    if not report.exists():
        lines.append("DuckDB counts report not present.")
        lines.append("")
        return "\n".join(lines)

    payload = _read_json_payload(report)
    if not payload:
        lines.append(f"Could not parse DuckDB counts payload from `{report}`.")
        lines.append("")
        return "\n".join(lines)

    lines.append(f"- Database: `{payload.get('database', '(unknown)')}`")
    exists = payload.get("exists")
    lines.append(f"- Exists: {exists}")
    tables = payload.get("tables", {})
    if isinstance(tables, dict) and tables:
        lines.append("")
        lines.append("| Table | Rows | Note |")
        lines.append("| --- | --- | --- |")
        for name, info in sorted(tables.items()):
            if isinstance(info, dict):
                rows = info.get("rows")
                note = info.get("error") if info.get("error") else ""
                exists_flag = info.get("exists")
            else:
                rows = "n/a"
                note = ""
                exists_flag = False
            if not exists_flag:
                note = (note + "; " if note else "") + "missing"
            lines.append(f"| `{name}` | {rows if rows is not None else 'n/a'} | {note} |")
    else:
        lines.append("- No matching `facts_` tables enumerated.")
    lines.append("")
    lines.append(f"- Detailed report: `{report}`")
    lines.append("")
    return "\n".join(lines)


def _render_exit_codes(exit_root: Path) -> str:
    lines = ["## Exit Code Breadcrumbs", ""]
    if not exit_root.exists():
        lines.append("No exit codes captured.")
        lines.append("")
        return "\n".join(lines)

    files = sorted(p for p in exit_root.iterdir() if p.is_file())
    if not files:
        lines.append("No exit code files present.")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Step | Exit | Details |")
    lines.append("| --- | --- | --- |")
    for path in files:
        try:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as exc:
            lines.append(f"| `{path.name}` | n/a | failed to read ({exc}) |")
            continue
        match = re.search(r"exit=([-0-9]+)", content)
        exit_value = match.group(1) if match else "n/a"
        details = content.replace("\n", " ")
        lines.append(f"| `{path.name}` | {exit_value} | {details} |")
    lines.append("")
    return "\n".join(lines)


def build_summary(repo_root: Path) -> str:
    diag_root = SUMMARY_DIR if SUMMARY_DIR.exists() else repo_root
    exit_root = Path(".ci/exitcodes")

    parts: List[str] = []
    parts.append(_gather_metadata())
    parts.append(_gather_environment(diag_root))
    parts.append(_gather_workflow_context())
    parts.append(_render_pytest_summary(diag_root))
    parts.append(_render_command_tails(diag_root))
    parts.append(_render_canonical_listing(diag_root))
    parts.append(_render_duckdb_counts(diag_root))
    parts.append(_render_exit_codes(exit_root))
    parts.append(_gather_logs(diag_root))
    parts.append(_inspect_duckdb(repo_root))
    parts.append(_tail_repo_logs(repo_root))
    parts.append("## Known Issues / Heuristics\n\n- Pending analysis.")
    return "\n".join(parts) + "\n"


def main() -> int:
    root = Path.cwd()
    try:
        summary = build_summary(root)
        _safe_write_text(SUMMARY_PATH, summary)
    except Exception as exc:  # pragma: no cover - defensive catch-all
        fallback = [
            "# CI Diagnostics Summary",
            "",
            "An unexpected error occurred while generating diagnostics.",
            "",
            f"````text\n{exc}\n````",
        ]
        try:
            _safe_write_text(SUMMARY_PATH, "\n".join(fallback))
        except Exception:
            # As a last resort, print to stdout so the workflow at least has
            # some context.
            print("Failed to write SUMMARY.md:", exc, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
