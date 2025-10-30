#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
from datetime import datetime, timezone
from typing import Iterable, List, Sequence, Tuple

if __package__ in (None, ""):
    sys.path.append(str(pathlib.Path(__file__).resolve().parent))

from junit_to_md import junit_reports_to_markdown

_ART_DIR_ENV = "ART_DIR"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _safe_tail(path: pathlib.Path, limit: int = 400) -> str:
    if not path.exists():
        return f"_not found: {path.name}_"
    content = _safe_read(path)
    if not content:
        return "(empty)"
    lines = content.splitlines()
    tail = lines[-limit:]
    return "\n".join(tail)


def _kv_table(title: str, rows: Iterable[Tuple[str, str]]) -> str:
    rendered_rows = list(rows)
    if not rendered_rows:
        return "\n".join([f"### {title}", "", "_No data available._", ""])
    header = [f"### {title}", "", "| Key | Value |", "| --- | --- |"]
    for key, value in rendered_rows:
        display_value = (value or "").replace("\n", " ").strip()
        header.append(f"| `{key}` | {display_value or '(unset)'} |")
    header.append("")
    return "\n".join(header)


def _artifact_rows(art_dir: pathlib.Path) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for entry in sorted(art_dir.glob("*")):
        name = entry.name
        if entry.is_file():
            try:
                size = entry.stat().st_size
                rows.append((name, f"{size:,} bytes"))
            except OSError:
                rows.append((name, "unavailable"))
        elif entry.is_dir():
            rows.append((name + "/", "directory"))
        else:
            rows.append((name, "other"))
    return rows


def _run_metadata() -> List[Tuple[str, str]]:
    env = os.environ
    commit = env.get("GITHUB_SHA", "")
    if commit:
        commit = commit[:12]
    timestamp = env.get("RUN_TS") or _utc_now()
    rows = [
        ("repository", env.get("GITHUB_REPOSITORY", "")),
        ("commit", commit),
        ("ref", env.get("GITHUB_REF", "")),
        ("workflow", env.get("GITHUB_WORKFLOW", "")),
        ("job", env.get("GITHUB_JOB", "")),
        ("run_id", env.get("GITHUB_RUN_ID", "")),
        ("run_attempt", env.get("GITHUB_RUN_ATTEMPT", "")),
        ("run_timestamp", timestamp),
    ]
    return rows


def _env_snapshot() -> List[Tuple[str, str]]:
    keys = [
        "ART_ROOT",
        "ART_DIR",
        "RUN_TS",
        "SAFE_SUFFIX",
        "RESOLVER_API_BACKEND",
        "RESOLVER_DB_URL",
        "PYTEST_ADDOPTS",
        "pythonLocation",
    ]
    return [(key, os.environ.get(key, "")) for key in keys]


def _collect_junit(art_dir: pathlib.Path) -> Tuple[str, int, int, int]:
    candidates: List[str] = []
    candidates.extend(str(path) for path in sorted(art_dir.glob("*.junit.xml")))
    candidates.extend(
        str(path)
        for path in sorted(art_dir.glob("*.xml"))
        if "junit" in path.name.lower()
    )
    unique: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique.append(candidate)
    return junit_reports_to_markdown(unique)


def _log_sections(art_dir: pathlib.Path) -> List[str]:
    sections: List[str] = []
    primary_logs = [
        ("pytest.out", 400),
        ("pytest-db.out", 400),
    ]
    seen = set()
    for filename, limit in primary_logs:
        path = art_dir / filename
        seen.add(path)
        sections.extend(_render_log_tail(filename, path, limit))

    for extra in sorted(art_dir.glob("*.out")):
        if extra in seen:
            continue
        sections.extend(_render_log_tail(extra.name, extra, 200))
    return sections


def _render_log_tail(title: str, path: pathlib.Path, limit: int) -> List[str]:
    tail = _safe_tail(path, limit)
    return [f"### Log tail — {title}", "", "```", tail, "```", ""]


def _collect_export_sections(art_dir: pathlib.Path) -> List[str]:
    candidates = [
        art_dir / "export-facts-summary.txt",
        art_dir / "export-facts-summary.md",
        art_dir / "db.summary.txt",
    ]
    sections: List[str] = []
    for path in candidates:
        if not path.exists():
            continue
        sections.append(f"### Export / Mapping Summary — {path.name}")
        sections.append("")
        sections.append("```")
        sections.append(_safe_tail(path, 2000))
        sections.append("```")
        sections.append("")
    return sections


def _write_pip_freeze(art_dir: pathlib.Path) -> pathlib.Path:
    output = art_dir / "pip-freeze.txt"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
        )
        freeze_output = proc.stdout or proc.stderr or ""
        output.write_text(freeze_output, encoding="utf-8")
    except Exception:
        output.write_text("(pip freeze unavailable)", encoding="utf-8")
    return output


def _git_status() -> str:
    try:
        proc = subprocess.run(
            ["git", "status", "--short"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "(git status unavailable)"
    combined = proc.stdout or proc.stderr
    combined = combined.strip()
    return combined or "(clean tree)"


def _filesystem_snapshot(limit: int = 200) -> str:
    root = pathlib.Path(".")
    try:
        entries = sorted(root.rglob("*"))
    except Exception:
        return "(filesystem snapshot unavailable)"
    names = [str(path) for path in entries[:limit]]
    return "\n".join(names) if names else "(empty)"


def main(argv: Sequence[str]) -> int:
    args = list(argv)
    art_dir: pathlib.Path
    if len(args) >= 2 and args[1]:
        art_dir = pathlib.Path(args[1]).expanduser().resolve()
    else:
        env_path = os.environ.get(_ART_DIR_ENV, "./ci_artifacts")
        art_dir = pathlib.Path(env_path).expanduser().resolve()

    art_dir.mkdir(parents=True, exist_ok=True)

    pip_path = _write_pip_freeze(art_dir)
    context = {
        "art_dir": str(art_dir),
        "pip_freeze_path": str(pip_path),
        "metadata": dict(_run_metadata()),
    }
    (art_dir / "context.json").write_text(json.dumps(context, indent=2, sort_keys=True), encoding="utf-8")

    junit_md, total, failures, errors = _collect_junit(art_dir)

    header_lines = [
        "# CI Diagnostics Summary — nightly",
        "",
        f"Generated: {_utc_now()}",
        f"Artifacts directory: {art_dir}",
        "",
    ]

    body_parts: List[str] = []
    body_parts.extend(header_lines)
    body_parts.append(_kv_table("Run Metadata", _run_metadata()))
    body_parts.append(_kv_table("Environment Snapshot", _env_snapshot()))

    junit_section_header = ["### Test Results (JUnit)", ""]
    summary_line = f"- totals: tests={total}, failures={failures}, errors={errors}"
    junit_section = "\n".join(junit_section_header + [summary_line, "", junit_md])
    body_parts.append(junit_section)

    body_parts.extend(_log_sections(art_dir))
    body_parts.extend(_collect_export_sections(art_dir))

    git_section = ["### Git Status", "", "```", _git_status(), "```", ""]
    body_parts.append("\n".join(git_section))

    fs_section = ["### Workspace Snapshot", "", "```", _filesystem_snapshot(), "```", ""]
    body_parts.append("\n".join(fs_section))

    artifact_table = _kv_table("Artifacts in ART_DIR", _artifact_rows(art_dir))
    body_parts.append(artifact_table)

    summary_path = art_dir / "summary.md"
    summary_path.write_text("\n".join(body_parts) + "\n", encoding="utf-8")

    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary_path:
        pathlib.Path(step_summary_path).write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
