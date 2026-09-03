# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Verbatim copies of the files most often implicated in a bad run.

The bundle is read by somebody who has the zip and not the repository, and
often weeks after the fact when ``main`` has moved. Reading a stale
CrisisWatch edition or a collapsed batch against the code that produced it
means having that code, at that commit, in the same artifact.

Beside the sources: the commits between this run's SHA and the previous
production run's, taken from ``hs_runs.git_sha`` rather than guessed. Most
regressions are introduced by a merge between two cycles, and that list is
where a reader looks first.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.debug_bundle.redaction import redact_text

SNAPSHOT_FILES: tuple[str, ...] = (
    "pythia/llm_batch.py",
    "horizon_scanner/crisiswatch.py",
    "scripts/refresh_crisiswatch.py",
    "horizon_scanner/conflict_forecasts.py",
    "forecaster/cli.py",
    "horizon_scanner/horizon_scanner.py",
    ".github/workflows/pythia_pipeline_stage.yml",
    ".github/workflows/poll_llm_batches.yml",
    ".github/workflows/run_sibyl.yml",
)


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=120)
    except Exception as exc:  # noqa: BLE001
        return 1, f"{type(exc).__name__}: {exc}"
    return proc.returncode, (proc.stdout or proc.stderr or "").strip()


def previous_production_git_sha(con, current_hs_run_id: str | None) -> tuple[str | None, str | None]:
    """(git_sha, hs_run_id) of the production run before this one.

    Production means ``is_test`` false: a test run's commit is not the
    baseline anybody's regression is measured against. Returns (None, None)
    rather than raising when the column, the table or the row is absent.
    """

    try:
        # table_info yields (cid, name, ...): the name is column 1.
        cols = {r[1] for r in con.execute("PRAGMA table_info('hs_runs')").fetchall()}
    except Exception:
        return None, None
    if "git_sha" not in cols:
        return None, None
    test_filter = "AND COALESCE(is_test, FALSE) = FALSE" if "is_test" in cols else ""
    order_col = "generated_at" if "generated_at" in cols else "hs_run_id"
    params: list[Any] = []
    before = ""
    if current_hs_run_id:
        before = "AND hs_run_id < ?"
        params.append(current_hs_run_id)
    try:
        row = con.execute(
            f"""
            SELECT git_sha, hs_run_id
            FROM hs_runs
            WHERE git_sha IS NOT NULL AND git_sha <> '' {test_filter} {before}
            ORDER BY {order_col} DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
    except Exception:
        return None, None
    if not row:
        return None, None
    return (str(row[0]) if row[0] else None), (str(row[1]) if row[1] else None)


def _commits_since(repo_root: Path, previous_sha: str | None, current_sha: str | None) -> str:
    header = [
        "# Commits between the previous PRODUCTION run and this one.",
        f"# previous run commit: {previous_sha or '(not recorded in hs_runs.git_sha)'}",
        f"# this run commit:     {current_sha or '(unknown)'}",
        "",
    ]
    if not previous_sha:
        header.append(
            "No previous production run carried a git_sha, so there is no range to "
            "diff. This is expected on a fresh database and on the first run after a "
            "reset; on any other run it means hs_runs.git_sha was not written."
        )
        return "\n".join(header) + "\n"
    rc, out = _run(["git", "cat-file", "-e", f"{previous_sha}^{{commit}}"], repo_root)
    if rc != 0:
        header.append(
            f"Commit {previous_sha} is not in this checkout — Actions checks out with "
            "fetch-depth 1, so history before the current commit is absent. Run "
            f"`git log --oneline {previous_sha}..{current_sha or 'HEAD'}` against a full clone."
        )
        return "\n".join(header) + "\n"
    rc, out = _run(
        ["git", "log", "--oneline", f"{previous_sha}..{current_sha or 'HEAD'}"], repo_root
    )
    if rc != 0:
        header.append(f"git log failed: {redact_text(out)}")
        return "\n".join(header) + "\n"
    if not out:
        header.append("(no commits — this run is on the same commit as the previous one)")
    else:
        header.append(out)
    return "\n".join(header) + "\n"


def collect(
    out_dir: Path,
    *,
    repo_root: Path,
    con=None,
    current_hs_run_id: str | None = None,
    current_sha: str | None = None,
) -> dict[str, Any]:
    """Copy the snapshot files and write the commit range. Never raises."""

    snap_dir = out_dir / "code_snapshot"
    snap_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {"files": [], "problems": []}

    for rel in SNAPSHOT_FILES:
        src = repo_root / rel
        # Flatten the path into the filename so the directory stays one
        # level deep and a reader can see at a glance what is in it.
        dest = snap_dir / rel.replace("/", "__")
        try:
            text = src.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            index["problems"].append(f"{rel}: {redact_text(str(exc))}")
            continue
        # Source files hold no credentials, but a workflow file quotes env
        # names beside values often enough to be worth the pass.
        dest.write_text(redact_text(text), encoding="utf-8")
        index["files"].append(
            {"source": rel, "file": f"code_snapshot/{dest.name}", "bytes": dest.stat().st_size}
        )

    previous_sha, previous_run = (None, None)
    if con is not None:
        previous_sha, previous_run = previous_production_git_sha(con, current_hs_run_id)
    if current_sha is None:
        rc, out = _run(["git", "rev-parse", "HEAD"], repo_root)
        current_sha = out if rc == 0 else None

    commits_path = snap_dir / "commits_since_last_production_run.txt"
    commits_path.write_text(_commits_since(repo_root, previous_sha, current_sha), encoding="utf-8")
    index["previous_production_hs_run_id"] = previous_run
    index["previous_production_git_sha"] = previous_sha
    index["current_git_sha"] = current_sha
    index["files"].append(
        {
            "source": "(generated)",
            "file": "code_snapshot/commits_since_last_production_run.txt",
            "bytes": commits_path.stat().st_size,
        }
    )
    try:
        (snap_dir / "INDEX.json").write_text(
            json.dumps(index, indent=2, default=str), encoding="utf-8"
        )
    except Exception:  # pragma: no cover - defensive
        pass
    return index
