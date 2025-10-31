from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Tuple
import xml.etree.ElementTree as ET

FENCE_TICK = "```"
TAIL_LINES = 200
MAX_READ_BYTES = 200_000
LOG_GLOBS = ("*.log", "*.ndjson")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_text_safe(path: Path, *, max_bytes: int | None = None, default: str = "_file not found_") -> str:
    try:
        data = path.read_bytes()
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        return f"{default} ({exc})"
    if max_bytes and len(data) > max_bytes:
        data = data[-max_bytes:]
    try:
        return data.decode("utf-8", errors="replace")
    except Exception as exc:  # pragma: no cover - defensive I/O guard
        return f"{default} ({exc})"


def fence(body: str, lang: str = "") -> str:
    start = FENCE_TICK + (lang or "")
    return "\n".join((start, body, FENCE_TICK, ""))


def _junit_totals(node: ET.Element) -> Tuple[int, int, int, int]:
    tests = int(node.attrib.get("tests", 0))
    failures = int(node.attrib.get("failures", 0))
    errors = int(node.attrib.get("errors", 0))
    skipped = int(node.attrib.get("skipped", 0))
    return tests, failures, errors, skipped


def junit_summary(junit_path: Path, *, limit: int = 5) -> str:
    if not junit_path.exists():
        return "_No JUnit file found._\n"
    try:
        tree = ET.parse(str(junit_path))
    except Exception as exc:
        return f"_Failed to parse JUnit XML: {exc}_\n"

    root = tree.getroot()
    suites: Iterable[ET.Element]
    if root.tag == "testsuite":
        suites = [root]
    else:
        suites = list(root.findall("testsuite"))

    total = failures = errors = skipped = 0
    failing_cases: List[str] = []

    for suite in suites:
        t, f, e, s = _junit_totals(suite)
        total += t
        failures += f
        errors += e
        skipped += s
        for case in suite.findall("testcase"):
            failure_node = case.find("failure") or case.find("error")
            if failure_node is None:
                continue
            classname = case.attrib.get("classname", "")
            name = case.attrib.get("name", "")
            label = f"{classname}::{name}" if classname else name
            message = failure_node.attrib.get("message", "").strip()
            details = (failure_node.text or "").strip().splitlines()
            excerpt = "\n".join(details[:10]).strip()
            if excerpt:
                block = fence(excerpt, "")
            else:
                block = ""
            entry = "\n".join(filter(None, (f"- **{label}** — {message}", block)))
            failing_cases.append(entry)

    lines = [f"**Totals** — tests: {total}, failures: {failures}, errors: {errors}, skipped: {skipped}"]
    if failing_cases:
        lines.append(f"\n**Failing tests (first {limit}):**")
        lines.extend(failing_cases[:limit])
    return "\n".join(lines).rstrip() + "\n"


def tail_text(path: Path, *, max_lines: int = TAIL_LINES) -> str:
    content = read_text_safe(path, max_bytes=MAX_READ_BYTES)
    lines = content.splitlines() if content else []
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _metadata_table(pairs: Iterable[Tuple[str, str]]) -> str:
    rows = ["| Key | Value |", "| --- | --- |"]
    for key, value in pairs:
        safe_value = (value or "").replace("|", "\\|")
        rows.append(f"| `{key}` | {safe_value} |")
    return "\n".join(rows)


def _collect_run_metadata() -> List[Tuple[str, str]]:
    keys = (
        "GITHUB_REPOSITORY",
        "GITHUB_REF",
        "GITHUB_SHA",
        "GITHUB_WORKFLOW",
        "GITHUB_JOB",
        "GITHUB_RUN_ID",
        "GITHUB_RUN_ATTEMPT",
        "RUN_TS",
        "SAFE_SUFFIX",
    )
    return [(key, os.environ.get(key, "")) for key in keys]


def _collect_environment_rows() -> List[Tuple[str, str]]:
    keys = (
        "PYTHONPATH",
        "RESOLVER_API_BACKEND",
        "RESOLVER_DB_URL",
        "RESOLVER_SKIP_DTM",
        "PYTEST_ADDOPTS",
    )
    rows: List[Tuple[str, str]] = []
    for key in keys:
        rows.append((key, os.environ.get(key, "")))
    for key, value in sorted(os.environ.items()):
        if key.startswith("RESOLVER_") and all(row[0] != key for row in rows):
            rows.append((key, value))
    return rows


def _artifact_listing(root: Path) -> str:
    if not root.exists():
        return "_artifact directory missing_\n"
    entries = []
    for child in sorted(root.iterdir()):
        kind = "dir" if child.is_dir() else "file"
        entries.append((child.name, kind))
    if not entries:
        return "_artifact directory is empty_\n"
    return _metadata_table(entries)


def _log_tails(root: Path) -> List[str]:
    sections: List[str] = []
    for pattern in LOG_GLOBS:
        for candidate in sorted(root.glob(pattern)):
            tail = tail_text(candidate, max_lines=50)
            header = f"### {candidate.name} (tail)\n"
            sections.append(header + fence(tail, ""))
    return sections


def main() -> int:
    if len(sys.argv) > 2:
        print("usage: collect_diagnostics.py [ART_DIR]", file=sys.stderr)
        return 2

    if len(sys.argv) == 2:
        art_dir = Path(sys.argv[1]).resolve()
    else:
        art_dir = Path(os.environ.get("ART_DIR", "ci_artifacts")).resolve()

    art_dir.mkdir(parents=True, exist_ok=True)

    env_txt = art_dir / "env.txt"
    freeze_txt = art_dir / "pip-freeze.txt"
    pytest_junit = art_dir / "pytest-junit.xml"
    db_junit = art_dir / "db.junit.xml"
    pytest_out = art_dir / "pytest-db.out"
    pytest_exit = art_dir / "pytest-db.exit"

    sections: List[str] = []
    sections.append("# CI Diagnostics Summary\n")
    sections.append(f"Generated: {_now_utc()}")
    sections.append("\n")
    sections.append(f"Artifacts directory: `{art_dir}`\n")

    sections.append("## Run Metadata\n\n")
    sections.append(_metadata_table(_collect_run_metadata()))
    sections.append("\n\n")

    sections.append("## Environment Variables\n\n")
    sections.append(_metadata_table(_collect_environment_rows()))
    sections.append("\n\n")

    sections.append("## env.txt\n\n")
    sections.append(fence(read_text_safe(env_txt, default="env.txt missing"), "bash"))

    sections.append("## pip freeze\n\n")
    sections.append(fence(read_text_safe(freeze_txt, default="pip-freeze.txt missing"), ""))

    sections.append("## Pytest (JUnit)\n\n")
    sections.append(junit_summary(pytest_junit))

    if db_junit.exists():
        sections.append("### Database JUnit\n\n")
        sections.append(junit_summary(db_junit))

    if pytest_out.exists():
        sections.append("## pytest-db.out (tail)\n\n")
        sections.append(fence(tail_text(pytest_out), ""))

    if pytest_exit.exists():
        sections.append("## pytest-db.exit\n\n")
        sections.append(fence(read_text_safe(pytest_exit).strip(), ""))

    sections.append("## Artifact Listing\n\n")
    sections.append(_artifact_listing(art_dir))
    sections.append("\n")

    sections.extend(_log_tails(art_dir))

    summary_path = art_dir / "summary.md"
    summary_path.write_text("".join(sections).rstrip() + "\n", encoding="utf-8")

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        try:
            with open(step_summary, "a", encoding="utf-8") as handle:
                handle.write(summary_path.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover - diagnostics helper
            pass

    print(str(summary_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
