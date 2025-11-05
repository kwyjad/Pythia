#!/usr/bin/env python3
"""Bundle resolver CI diagnostics into a tarball without failing the workflow."""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Iterable, Iterator


REPO_ROOT = Path(__file__).resolve().parents[2]
DIAG_ROOT = REPO_ROOT / ".ci" / "diagnostics"
DEFAULT_JOB = "job"
DEFAULT_LABEL = "default"
MAX_LOG_FILES = 200


def _sanitize_fragment(value: str, fallback: str) -> str:
    cleaned = value.strip().replace(" ", "_")
    safe = [ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in cleaned]
    text = "".join(safe).strip("-._")
    return text or fallback


def _iter_files(paths: Iterable[Path]) -> Iterator[Path]:
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        yield path


def _iter_rglob(root: Path, pattern: str, limit: int | None = None) -> Iterator[Path]:
    count = 0
    try:
        for path in root.rglob(pattern):
            if not path.is_file():
                continue
            yield path
            count += 1
            if limit is not None and count >= limit:
                break
    except (OSError, RuntimeError):
        return


def _copy_into_diagnostics(src: Path, base: Path) -> None:
    try:
        relative = src.relative_to(base)
        destination = DIAG_ROOT / relative
    except ValueError:
        cleaned = src.as_posix().lstrip("/").replace(":", "_")
        destination = DIAG_ROOT / "external" / cleaned
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(src, destination)
    except OSError:
        # Best-effort; skip files that disappear or are unreadable.
        return


def _write_text(filename: str, content: str) -> None:
    path = DIAG_ROOT / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _collect_env() -> None:
    lines = [f"{key}={value}" for key, value in sorted(os.environ.items())]
    _write_text("env.txt", "\n".join(lines))


def _collect_pip_freeze() -> None:
    try:
        proc = subprocess.run(
            ["python", "-m", "pip", "freeze"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except OSError:
        return
    output = proc.stdout if proc.returncode == 0 else proc.stdout + proc.stderr
    _write_text("pip-freeze.txt", output)


def _collect_workspace_artifacts() -> None:
    junit_candidates = [
        REPO_ROOT / "pytest-junit.xml",
        REPO_ROOT / "pytest.xml",
    ]
    for junit in _iter_files(junit_candidates):
        _copy_into_diagnostics(junit, REPO_ROOT)

    resolver_logs = REPO_ROOT / "resolver" / ".logs"
    if resolver_logs.exists():
        for path in _iter_rglob(resolver_logs, "**/*"):
            _copy_into_diagnostics(path, REPO_ROOT)

    for pattern in ("*.log", "*.gz"):
        for path in _iter_rglob(REPO_ROOT, pattern, limit=MAX_LOG_FILES):
            _copy_into_diagnostics(path, REPO_ROOT)

    for path in _iter_rglob(REPO_ROOT, "*.duckdb"):
        _copy_into_diagnostics(path, REPO_ROOT)


def _collect_runner_tmp() -> None:
    candidates = [Path("/tmp/pytest-of-runner")]
    for env_var in ("RUNNER_TEMP", "TMPDIR", "TEMP"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value) / "pytest-of-runner")
    seen: set[Path] = set()
    for root in candidates:
        try:
            resolved = root.resolve()
        except (OSError, RuntimeError):
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        for path in _iter_rglob(resolved, "*.duckdb"):
            _copy_into_diagnostics(path, resolved)


def _make_tarball(job: str, label: str) -> Path:
    tar_name = f"diagnostics_{job}_{label}.tar.gz"
    tar_path = REPO_ROOT / tar_name
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(DIAG_ROOT, arcname="diagnostics")
    return tar_path


def main() -> int:
    job = _sanitize_fragment(os.environ.get("JOB_NAME", DEFAULT_JOB), DEFAULT_JOB)
    label = _sanitize_fragment(os.environ.get("DUCKDB_LABEL", DEFAULT_LABEL), DEFAULT_LABEL)

    if DIAG_ROOT.exists():
        shutil.rmtree(DIAG_ROOT, ignore_errors=True)
    DIAG_ROOT.mkdir(parents=True, exist_ok=True)

    _collect_env()
    _collect_pip_freeze()
    _collect_workspace_artifacts()
    _collect_runner_tmp()

    tar_path = _make_tarball(job, label)
    collected_files = [
        str(path.relative_to(DIAG_ROOT))
        for path in DIAG_ROOT.rglob("*")
        if path.is_file()
    ]
    collected_files.sort()
    summary_lines = [
        f"Created {tar_path.name} with {len(collected_files)} files",
        *collected_files,
    ]
    _write_text("SUMMARY.txt", "\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
