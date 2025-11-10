#!/usr/bin/env python3
"""Emit a Markdown DuckDB summary for GitHub Actions step output."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence


def _import_duckdb():  # pragma: no cover - import guarded in tests
    try:
        import duckdb  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        print("## DuckDB\n")
        print(f"- **Status:** duckdb import failed: {exc}")
        return None
    return duckdb


def _normalise_db_target(raw: str | None) -> tuple[str | None, str | None]:
    candidate = (raw or "").strip()
    if not candidate:
        return None, None
    if candidate.lower().startswith("duckdb://"):
        if candidate.lower().startswith("duckdb:///"):
            fs_part = candidate[len("duckdb:///") :]
            try:
                resolved = Path(fs_part).expanduser().resolve()
            except OSError:
                resolved = Path(fs_part).expanduser()
            return resolved.as_posix(), f"duckdb:///{resolved.as_posix()}"
        return candidate, candidate
    try:
        resolved_path = Path(candidate).expanduser().resolve()
    except OSError:
        resolved_path = Path(candidate).expanduser()
    return resolved_path.as_posix(), f"duckdb:///{resolved_path.as_posix()}"


def _iter_tables(raw: Sequence[str] | str | None) -> Iterable[str]:
    default = ("facts_raw", "facts_resolved", "facts_deltas")
    if raw is None:
        return default
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        items = [str(part).strip() for part in raw if str(part).strip()]
    return items or default


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None, help="DuckDB file path or duckdb:/// URL")
    parser.add_argument("--db-url", default=None, help="Alias for --db")
    parser.add_argument(
        "--tables",
        default=None,
        help="Optional comma-separated tables to summarize (default: facts_raw,facts_resolved,facts_deltas)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    db_arg = args.db or args.db_url or os.environ.get("RESOLVER_DB_URL") or os.environ.get("RESOLVER_DB_PATH")
    db_path, display_target = _normalise_db_target(db_arg)

    duckdb = _import_duckdb()
    if duckdb is None:
        return 0

    print("## DuckDB")
    print("")

    if not db_path:
        print("- **Status:** no DuckDB target provided")
        return 0

    fs_path = Path(db_path)
    if not fs_path.exists():
        shown = display_target or db_path
        print(f"- **Status:** database not found at `{shown}`")
        return 0

    try:
        conn = duckdb.connect(db_path, read_only=True)
    except Exception as exc:  # pragma: no cover - connection failure
        shown = display_target or db_path
        print(f"- **Status:** failed to open `{shown}`: {exc}")
        return 0

    try:
        print(f"- **Database:** `{display_target or db_path}`")
        rows: list[tuple[str, int]] = []
        for table in _iter_tables(args.tables):
            if not table:
                continue
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except Exception:
                continue
            rows.append((table, int(count)))

        if not rows:
            print("- **Status:** no matching tables present")
            return 0

        print("")
        print("| table | rows |")
        print("| --- | ---: |")
        for table, count in rows:
            print(f"| {table} | {count} |")
    finally:
        try:
            conn.close()
        except Exception:  # pragma: no cover - best effort
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
