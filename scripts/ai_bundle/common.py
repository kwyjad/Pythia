# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Shared plumbing for AI analysis bundle builders.

Deliberately light on dependencies (stdlib + duckdb) so builders can run in
minimal-deps workflow jobs. Defensive-schema helpers are re-exported from
``pythia.tools._db_utils`` (itself stdlib+duckdb) — do not re-implement them.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import duckdb

from pythia.tools._db_utils import (  # noqa: F401  (re-exported)
    column_exists,
    row_count,
    rollback_quietly,
    table_exists,
)

LOGGER = logging.getLogger(__name__)

BUILDER_VERSION = "1.0.0"


def resolve_db_path(db: str) -> Path:
    """Accept a duckdb:/// URL or a plain path and return a filesystem path."""
    raw = (db or "").strip()
    if raw.startswith("duckdb://"):
        raw = raw[len("duckdb://") :]
        # duckdb:///rel/path → "/rel/path" with a single leading slash meaning
        # a path relative to cwd unless it exists absolutely (mirrors how
        # resolver.db.duckdb_io treats duckdb:///data/resolver.duckdb).
        if raw.startswith("/") and not Path(raw).exists():
            candidate = Path(raw.lstrip("/"))
            if candidate.exists() or not Path(raw).parent.exists():
                return candidate
    return Path(raw)


def open_db(db: str) -> duckdb.DuckDBPyConnection:
    """Open the DB read-only; fall back to read-write if a WAL blocks it.

    The builder runs as its own process after all pipeline writers have
    closed, so a read-write fallback (which replays + checkpoints any WAL)
    is safe and preferable to failing the bundle.
    """
    path = resolve_db_path(db)
    if not path.exists():
        raise FileNotFoundError(f"DuckDB not found at {path}")
    try:
        return duckdb.connect(str(path), read_only=True)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Read-only open failed (%s); retrying read-write", exc)
        return duckdb.connect(str(path), read_only=False)


def rows_as_dicts(
    con: duckdb.DuckDBPyConnection, sql: str, params: list[Any] | None = None
) -> list[dict[str, Any]]:
    """Run a query and return rows as dicts keyed by column name."""
    cur = con.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def safe_json_loads(text: Any) -> Any:
    """Parse JSON that may be None/empty/invalid; return None on failure."""
    if not text or not isinstance(text, str):
        return None
    try:
        return json.loads(text)
    except Exception:  # noqa: BLE001
        return None


def json_default(obj: Any) -> str:
    return str(obj)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=1, default=json_default),
        encoding="utf-8",
    )


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> int:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            n += 1
    return n


def gz_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Stream rows into a gzipped JSONL file; returns the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=json_default))
            fh.write("\n")
            n += 1
    return n


def size_guard(path: Path, budget_kb: int) -> None:
    """Warn (never fail) when a digest/briefing file exceeds its size target."""
    try:
        kb = path.stat().st_size / 1024
    except OSError:
        return
    if kb > budget_kb:
        LOGGER.warning(
            "%s is %.0f KB (target %d KB) — consider trimming its inputs",
            path.name,
            kb,
            budget_kb,
        )


def write_manifest(
    out_dir: Path,
    *,
    bundle_kind: str,
    db_path: Path,
    table_counts: Mapping[str, int],
    extra: Mapping[str, Any] | None = None,
) -> None:
    manifest: dict[str, Any] = {
        "bundle_kind": bundle_kind,
        "builder_version": BUILDER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": os.getenv("GITHUB_SHA") or None,
        "workflow": os.getenv("GITHUB_WORKFLOW") or None,
        "workflow_run_id": os.getenv("GITHUB_RUN_ID") or None,
        "db_path": str(db_path),
        "table_counts": dict(table_counts),
    }
    if extra:
        manifest.update(extra)
    write_json(out_dir / "manifest.json", manifest)


def write_bundle_zip(staging_dir: Path, zip_path: Path) -> None:
    """Zip the staging directory, PRESERVING subdirectories.

    (Unlike scripts/dump_pythia_debug_bundle.py's build_flat_zip, which
    flattens — the bundle's questions/, case_studies/ and briefing/ layout is
    part of its contract with the consuming AI.)
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    skip_suffixes = {".duckdb", ".db", ".wal", ".pyc"}
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(staging_dir.rglob("*")):
            if not file.is_file():
                continue
            if file.suffix.lower() in skip_suffixes:
                continue
            zf.write(file, file.relative_to(staging_dir).as_posix())
