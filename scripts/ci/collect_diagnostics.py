#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Collect CI diagnostics into a single directory and render a single summary.md.

Design:
- Never zip. Let upload-artifact compress once.
- Never crash on missing inputs. Emit placeholders instead.
- One-stop summary with:
  * Run metadata (GitHub env)
  * Python version + pip freeze
  * Selected env vars relevant to Resolver
  * Pytest JUnit digest (if present)
  * Artifact tree listing
"""

import os
import sys
import textwrap
import traceback
from datetime import datetime, timezone
from xml.etree import ElementTree as ET
from pathlib import Path
import subprocess

KEY_ENV_VARS = [
    "GITHUB_REPOSITORY", "GITHUB_REF", "GITHUB_SHA", "GITHUB_WORKFLOW",
    "GITHUB_JOB", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT",
    "RUN_TS", "SAFE_SUFFIX",
    "RESOLVER_API_BACKEND", "RESOLVER_DB_URL",
    "RESOLVER_LOG_LEVEL", "PYTEST_ADDOPTS", "PYTHONPATH",
    "RESOLVER_EXPORT_ENABLE_FLOW", "RESOLVER_SKIP_DTM",
]

def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"{path.name} missing ({repr(e)})"

def _run_cmd(cmd, cwd=None) -> str:
    try:
        out = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.STDOUT, text=True)
        return out
    except Exception as e:
        return f"Command {cmd!r} failed: {e}"

def _parse_junit(junit_path: Path) -> dict:
    if not junit_path.exists():
        return {"present": False}
    try:
        root = ET.parse(junit_path).getroot()
        # JUnit root may be <testsuite> or <testsuites>
        suites = []
        if root.tag == "testsuite":
            suites = [root]
        elif root.tag == "testsuites":
            suites = list(root.findall("testsuite"))
        total = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
        for s in suites:
            total["tests"] += int(s.attrib.get("tests", 0))
            total["failures"] += int(s.attrib.get("failures", 0))
            total["errors"] += int(s.attrib.get("errors", 0))
            total["skipped"] += int(s.attrib.get("skipped", 0))
            total["time"] += float(s.attrib.get("time", 0.0))
        return {"present": True, "totals": total}
    except Exception as e:
        return {"present": False, "error": repr(e)}

def _tree_listing(root_dir: Path) -> str:
    lines = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        rel = os.path.relpath(dirpath, root_dir)
        prefix = "." if rel == "." else rel
        lines.append(prefix + "/")
        for f in sorted(filenames):
            lines.append(f"  {f}")
    return "\n".join(lines) if lines else "_artifact directory is empty_"

def main():
    if len(sys.argv) != 2:
        print("Usage: collect_diagnostics.py <ART_DIR>", file=sys.stderr)
        sys.exit(2)

    art_dir = Path(sys.argv[1]).resolve()
    art_dir.mkdir(parents=True, exist_ok=True)

    # Attempt to generate env/pip-freeze if missing (best-effort).
    env_txt = art_dir / "env.txt"
    if not env_txt.exists():
        try:
            env_dump = "\n".join(sorted(f"{k}={v}" for k, v in os.environ.items()))
            _write_text(env_txt, env_dump)
        except Exception:
            pass

    pip_freeze = art_dir / "pip-freeze.txt"
    if not pip_freeze.exists():
        try:
            freeze_out = _run_cmd([sys.executable, "-m", "pip", "freeze"])
            _write_text(pip_freeze, freeze_out)
        except Exception:
            pass

    junit_path = art_dir / "pytest-junit.xml"

    # Build summary.md
    try:
        now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Metadata table
        rows = []
        for k in KEY_ENV_VARS:
            rows.append((k, os.environ.get(k, "")))
        meta_table = "\n".join(
            ["| Key | Value |", "| --- | --- |"] +
            [f"| `{k}` | {v if v else ' '} |" for k, v in rows]
        )

        # Selected env previews
        env_preview = _safe_read(env_txt)
        freeze_preview = _safe_read(pip_freeze)

        # Pytest digest
        junit_info = _parse_junit(junit_path)
        if junit_info.get("present"):
            totals = junit_info["totals"]
            junit_block = textwrap.dedent(f"""
            **Pytest (JUnit)**

            - tests: {totals['tests']}
            - failures: {totals['failures']}
            - errors: {totals['errors']}
            - skipped: {totals['skipped']}
            - time (s): {totals['time']}
            """).strip()
        else:
            err = junit_info.get("error", "No JUnit file found.")
            junit_block = f"**Pytest (JUnit)**\n\n_{err}_"

        # Python version
        py_ver = sys.version.replace("\n", " ")

        # Artifact tree
        tree = _tree_listing(art_dir)

        summary = textwrap.dedent(f"""
        # CI Diagnostics Summary
        Generated: {now_utc}
        Artifacts directory: `{art_dir}`

        ## Run Metadata

        {meta_table}

        ## Python
        `{py_ver}`

        ## Environment Variables (env.txt)
        ```bash
        {env_preview}
        ```

        ## pip freeze
        ```
        {freeze_preview}
        ```

        ## {junit_block}

        ## Artifact Listing

        {tree}
        """).strip()

        summary_path = art_dir / "summary.md"
        _write_text(summary_path, summary)

        step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary:
            try:
                with open(step_summary, "a", encoding="utf-8") as handle:
                    handle.write(summary + "\n")
            except Exception:
                pass

        print(f"Diagnostics written to: {art_dir}")
        sys.exit(0)
    except Exception as e:
        # Last resort: never crash the job — emit a minimal summary.
        tb = traceback.format_exc()
        fallback = "# CI Diagnostics Summary (fallback)\n\nError while writing summary:\n\n```\n" + tb + "\n```"
        summary_path = art_dir / "summary.md"
        _write_text(summary_path, fallback)
        step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary:
            try:
                with open(step_summary, "a", encoding="utf-8") as handle:
                    handle.write(fallback + "\n")
            except Exception:
                pass
        print("collect_diagnostics.py encountered an error but wrote fallback summary.md", file=sys.stderr)
        sys.exit(0)

if __name__ == "__main__":
    main()
