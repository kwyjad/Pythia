#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap
from typing import Iterable, Tuple


def sh(cmd: str) -> str:
    """Run *cmd* in the shell and return combined stdout/stderr."""
    proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if proc.returncode != 0 and not proc.stdout.strip():
        return proc.stderr.strip()
    output_parts = []
    if proc.stdout:
        output_parts.append(proc.stdout.rstrip())
    if proc.stderr:
        output_parts.append(proc.stderr.rstrip())
    return "\n".join(part for part in output_parts if part).strip()


def relpath(path: pathlib.Path, base: pathlib.Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def tail_lines(text: str, limit: int = 200) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    return "\n".join(lines[-limit:])


def load_text(path: pathlib.Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except OSError:
        return ""


def normalize_codex_sections(base: pathlib.Path, patterns: Iterable[str]) -> str:
    sections = []
    for pattern in patterns:
        for candidate in base.glob(pattern):
            if not candidate.is_file():
                continue
            try:
                raw = candidate.read_text(errors="ignore")
            except OSError:
                continue
            # Collapse any language-specific fences to plain triple-backtick and wrap once.
            body = re.sub(r"```[a-zA-Z0-9_-]*\n", "```\n", raw)
            body = body.strip()
            if body.startswith("```") and body.endswith("```"):
                inner = body[3:-3].strip()
            else:
                inner = body
            sections.append(textwrap.dedent(f"""
                ### {relpath(candidate, base)}
                ```
                {inner}
                ```
            """).strip())
    return "\n\n".join(sections) if sections else "_No codex cards found._"


def cpu_count_display() -> str:
    count = os.cpu_count()
    if count is not None:
        return str(count)
    fallback = sh("nproc || sysctl -n hw.logicalcpu || echo '?' ")
    return fallback or "?"


def mem_total_display() -> str:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 3:
                        return f"{parts[1]} {parts[2]}"
                    if len(parts) >= 2:
                        return f"{parts[1]} kB"
                    break
    except OSError:
        pass
    fallback = sh("sysctl hw.memsize || echo '?' ")
    return fallback or "?"


def runner_os_display() -> str:
    try:
        with open("/etc/os-release", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    fallback = sh("uname -sr || echo 'unknown'")
    return fallback or "unknown"


def gather_fs_snapshot() -> str:
    script = """
import itertools
import pathlib
root = pathlib.Path('.')
entries = []
for path in itertools.islice(sorted(root.rglob('*')), 200):
    entries.append(str(path))
print('\n'.join(entries))
"""
    snapshot = sh(f"python - <<'PY'\n{script}\nPY")
    return snapshot or "(no filesystem snapshot captured)"


def main(argv: Iterable[str]) -> int:
    args = list(argv)
    if len(args) != 2:
        print("usage: collect_diagnostics.py <ART_DIR>", file=sys.stderr)
        return 2

    art_dir = pathlib.Path(args[1]).resolve()
    art_dir.mkdir(parents=True, exist_ok=True)
    workspace = pathlib.Path.cwd()

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    sha = os.environ.get("GITHUB_SHA", "")[:12]
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    job = os.environ.get("GITHUB_JOB", "")
    attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "")

    runner_os = runner_os_display()
    kernel = sh("uname -a")
    pyver = sh("python -VV")
    pytest_ver = sh("pytest --version")
    duckdb_ver = sh("python - <<'P'\nimport duckdb, sys\nprint(getattr(duckdb, '__version__', 'n/a'))\nP") or "n/a"
    pip_freeze = sh("python -m pip freeze") or "(pip freeze unavailable)"
    cpu = cpu_count_display()
    mem = mem_total_display()
    df = sh("df -hT | sed -n '1,15p'")
    git_status = sh("git status --short || true")
    fs_snapshot = gather_fs_snapshot()

    pip_path = art_dir / "pip-freeze.txt"
    pip_path.write_text(pip_freeze + ("\n" if not pip_freeze.endswith("\n") else ""), encoding="utf-8")

    env_keys = [
        "RESOLVER_API_BACKEND",
        "RESOLVER_DB_URL",
        "RESOLVER_LOG_LEVEL",
        "PYTEST_ADDOPTS",
        "SAFE_SUFFIX",
        "RUN_TS",
    ]
    env_highlights = {key: os.environ.get(key, "") for key in env_keys if key in os.environ}
    env_lines = "\n".join(
        f"- **{key}:** {value or '(unset)'}" for key, value in sorted(env_highlights.items())
    )
    if not env_lines:
        env_lines = "- _(no tracked environment variables found)_"

    pytest_artifacts: Tuple[Tuple[str, str], ...] = (
        ("Nightly suite", "pytest"),
        ("DB parity suite", "pytest-db"),
    )
    pytest_sections = []
    for label, stem in pytest_artifacts:
        out_path = art_dir / f"{stem}.out"
        junit_path = art_dir / ("junit.xml" if stem == "pytest" else "db.junit.xml")
        exit_path = art_dir / f"{stem}.exit"
        log_text = load_text(out_path)
        tail = tail_lines(log_text, 200)
        exit_code = load_text(exit_path).strip() or "(missing)"
        junit_status = "present" if junit_path.exists() else "missing"
        pytest_sections.append(textwrap.dedent(f"""
            ### {label}
            * **JUnit:** `{relpath(junit_path, workspace)}` ({junit_status})
            * **Log:** `{relpath(out_path, workspace)}` ({'present' if out_path.exists() else 'missing'})
            * **Exit code:** {exit_code}

            <details>
            <summary>Last 200 lines</summary>

            ```
            {tail or '(no log output found)'}
            ```
            </details>
        """).strip())

    codex_md = normalize_codex_sections(workspace, [
        "outputs/**/*.md",
        "artifacts/**/*.md",
        "codex/**/*.md",
        "docs/codex/**/*.md",
    ])

    summary = textwrap.dedent(f"""
        # Nightly CI Summary

        **Repository:** {repo}  
        **Commit:** `{sha}` · **Job:** `{job}` · **Run:** {run_id} (attempt {attempt})

        ## Platform
        **OS:** {runner_os}  
        **Kernel:** {kernel}  
        **Python:** {pyver}  
        **Pytest:** {pytest_ver}  
        **DuckDB:** {duckdb_ver.strip()}

        ## Resources
        **CPU cores:** {cpu} · **Mem:** {mem}

        **Filesystem (top 15 from `df -hT`):**

        ```
        {df or '(df output unavailable)'}
        ```

        **Workspace snapshot (first 200 paths):**

        ```
        {fs_snapshot}
        ```

        **Git status:**

        ```
        {git_status or '(clean tree)'}
        ```

        ## Environment highlights
        {env_lines}

        ## Pytest diagnostics
        {'\n\n'.join(pytest_sections) if pytest_sections else '_No pytest outputs found._'}

        ## Pip freeze (saved to `{relpath(pip_path, workspace)}`)

        ```
        {pip_freeze}
        ```

        ## Codex cards
        {codex_md}
    """).strip()

    summary_path = art_dir / "summary.md"
    summary_path.write_text(summary + "\n", encoding="utf-8")

    # Persist a machine-readable context snapshot for debugging.
    context = {
        "repository": repo,
        "commit": sha,
        "job": job,
        "run_id": run_id,
        "attempt": attempt,
        "art_dir": str(art_dir),
        "env_highlights": env_highlights,
        "pip_freeze_path": relpath(pip_path, workspace),
    }
    (art_dir / "context.json").write_text(json.dumps(context, indent=2, sort_keys=True), encoding="utf-8")

    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary_path:
        pathlib.Path(step_summary_path).write_text(summary + "\n", encoding="utf-8")

    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
