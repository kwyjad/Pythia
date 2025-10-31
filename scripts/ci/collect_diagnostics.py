from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

TAIL_LINES = 200


def sh(cmd: list[str], cwd: str | None = None) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, text=True)
        return out
    except Exception as exc:
        return f"(command failed: {' '.join(cmd)}; {exc})"


def read_text(path: Path, max_bytes: int | None = None) -> str:
    try:
        data = path.read_bytes()
        if max_bytes and len(data) > max_bytes:
            data = data[-max_bytes:]
        return data.decode(errors="replace")
    except Exception as exc:
        return f"(_not found or unreadable: {path} — {exc}_)"


def tail_text(path: Path, n: int = TAIL_LINES) -> str:
    try:
        lines = path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return f"_not found: {path.name}_"


def junit_summary(junit_path: Path) -> tuple[str, dict]:
    if not junit_path.exists():
        return "_not found: db.junit.xml_", {}
    try:
        tree = ET.parse(junit_path)
        root = tree.getroot()
        totals = {"tests": 0, "failures": 0, "errors": 0}
        details = []
        for suite in root.findall("testsuite"):
            totals["tests"] += int(suite.attrib.get("tests", 0))
            totals["failures"] += int(suite.attrib.get("failures", 0))
            totals["errors"] += int(suite.attrib.get("errors", 0))
            for case in suite.findall("testcase"):
                name = f"{case.attrib.get('classname','')}.{case.attrib.get('name','')}".strip(".")
                fail = case.find("failure")
                err = case.find("error")
                if fail is not None or err is not None:
                    node = fail if fail is not None else err
                    msg = (node.attrib.get("message", "") or "").strip()
                    text = (node.text or "").strip()
                    top = "\n".join(text.splitlines()[:30])
                    details.append((name, msg, top))
        hdr = (
            f"- totals: tests={totals['tests']}, failures={totals['failures']}, errors={totals['errors']}\n"
        )
        hdr += (
            f"\n- `db.junit.xml`: tests={totals['tests']}, failures={totals['failures']}, errors={totals['errors']}\n"
        )
        if details:
            hdr += "\n**Failing tests (truncated):**\n"
            for name, msg, top in details:
                hdr += f"- **{name}** — _{msg}_\n\n```\n{top}\n```\n"
        return hdr, totals
    except Exception as exc:
        return f"(_could not parse JUnit: {exc}_)", {}


def env_snapshot() -> str:
    keys = [
        "GITHUB_REPOSITORY",
        "GITHUB_REF",
        "GITHUB_SHA",
        "GITHUB_WORKFLOW",
        "GITHUB_JOB",
        "GITHUB_RUN_ID",
        "ART_ROOT",
        "ART_DIR",
        "RUN_TS",
        "SAFE_SUFFIX",
        "RESOLVER_API_BACKEND",
        "RESOLVER_DB_URL",
        "PYTEST_ADDOPTS",
        "pythonLocation",
    ]
    extra = {
        k: v
        for k, v in os.environ.items()
        if k.startswith("RESOLVER_") or k.startswith("PYTEST_")
    }
    duckdb_ver = sh([sys.executable, "-c", "import duckdb,sys; print(duckdb.__version__)"]).strip()
    rows = []
    for k in keys:
        rows.append((k, os.environ.get(k, "")))
    rows.append(("duckdb_version", duckdb_ver))
    for k in sorted(extra):
        if k not in keys:
            rows.append((k, extra[k]))
    out = ["| Key | Value |", "| --- | --- |"]
    for k, v in rows:
        v_clean = (v or "").replace("|", "\\|")
        out.append(f"| `{k}` | {v_clean} |")
    return "\n".join(out)


def crawl_artifacts(root: Path) -> str:
    if not root.exists():
        return f"_not found: {root}_"
    lines = ["| Key | Value |", "| --- | --- |"]
    for p in sorted(root.glob("*")):
        kind = "directory" if p.is_dir() else "file"
        lines.append(f"| `{p.name}` | {kind} |")
    return "\n".join(lines)


def scrape_export_facts(pytest_out: str) -> str:
    if "tests (db backend) summary" not in pytest_out:
        return "_not found in pytest-db.out_"
    start = pytest_out.find("tests (db backend) summary")
    chunk = pytest_out[start : start + 3000]
    return f"```\n{chunk.strip()}\n```"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: collect_diagnostics.py ART_DIR", file=sys.stderr)
        return 2
    art_dir = Path(sys.argv[1]).resolve()
    art_dir.mkdir(parents=True, exist_ok=True)
    junit_path = art_dir / "db.junit.xml"
    pytest_db_out = art_dir / "pytest-db.out"

    run_meta = {
        "repository": os.environ.get("GITHUB_REPOSITORY", ""),
        "commit": os.environ.get("GITHUB_SHA", ""),
        "ref": os.environ.get("GITHUB_REF", ""),
        "workflow": os.environ.get("GITHUB_WORKFLOW", ""),
        "job": os.environ.get("GITHUB_JOB", ""),
        "run_id": os.environ.get("GITHUB_RUN_ID", ""),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        "run_timestamp": os.environ.get("RUN_TS", ""),
    }

    junit_md, _totals = junit_summary(junit_path)

    pytest_tail = tail_text(pytest_db_out)
    pytest_full = read_text(pytest_db_out, max_bytes=200_000)

    export_facts_md = scrape_export_facts(pytest_full)

    git_status = sh(["git", "status", "--porcelain=v1"])
    ws_tree = sh(["bash", "-lc", "ls -1a | head -200"])

    art_table = crawl_artifacts(art_dir)

    context_json = read_text(art_dir / "context.json", max_bytes=200_000)

    generated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    md: list[str] = []
    md.append("# CI Diagnostics Summary — nightly\n")
    md.append(f"Generated: {generated_at}")
    md.append(f"\nArtifacts directory: {art_dir}\n")

    md.append("\n### Run Metadata\n\n| Key | Value |\n| --- | --- |\n")
    for k, v in run_meta.items():
        v_clean = (v or "").replace("|", "\\|")
        md.append(f"| `{k}` | {v_clean} |\n")

    md.append("\n### Environment Snapshot\n\n")
    md.append(env_snapshot())
    md.append("\n\n### Test Results (JUnit)\n\n")
    md.append(junit_md)

    md.append("\n### “Tests (db backend) summary” (best-effort)\n\n")
    md.append(export_facts_md)

    md.append("\n\n### Log tail — pytest-db.out\n\n```\n")
    md.append(pytest_tail)
    md.append("\n```\n")

    md.append("\n### Git Status\n\n```\n")
    md.append(git_status)
    md.append("\n```\n")

    md.append("\n### Workspace Snapshot\n\n```\n")
    md.append(ws_tree)
    md.append("\n```\n")

    md.append("\n### Artifacts in ART_DIR\n\n")
    md.append(art_table)
    md.append("\n")

    md.append("\n### context.json\n\n```\n")
    md.append(context_json)
    md.append("\n```\n")

    out_path = art_dir / "summary.md"
    out_path.write_text("".join(md), encoding="utf-8")
    print(str(out_path))

    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        Path(step_summary).write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(main())
