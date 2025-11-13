#!/usr/bin/env python3
"""Summarize DuckDB write results for ingestion diagnostics."""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
from typing import Iterable, Sequence, Tuple

try:
    import duckdb
except Exception as exc:  # pragma: no cover - optional dependency
    print(f"duckdb import failed: {exc}")
    sys.exit(0)


MARKDOWN_HEADER = "## DuckDB write verification\n\n"
COUNTS_PATH = pathlib.Path("diagnostics/ingestion/duckdb_counts.md")
SUMMARY_PATH = pathlib.Path("diagnostics/ingestion/summary.md")


def _normalise_db_path(db_url: str) -> str:
    if db_url.startswith("duckdb:///"):
        db_url = db_url.replace("duckdb:///", "", 1)
    return os.path.abspath(os.path.expanduser(db_url))


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    result = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
        [table],
    ).fetchone()
    return bool(result)


def _fetch_breakdown(
    con: duckdb.DuckDBPyConnection,
) -> Iterable[Tuple[str, str, str, int]]:
    return con.execute(
        """
        SELECT
          COALESCE(source, '') AS source,
          COALESCE(metric, '') AS metric,
          COALESCE(series_semantics, '') AS semantics,
          COUNT(*) AS rows
        FROM facts_resolved
        GROUP BY 1, 2, 3
        ORDER BY rows DESC, source, metric, semantics
        """
    ).fetchall()


def _write_summary(
    db_path: str,
    rows_total: int,
    breakdown: Iterable[Tuple[str, str, str, int]],
) -> str:
    COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [MARKDOWN_HEADER]
    lines.append(f"- Database: {db_path}\n")
    lines.append(f"- facts_resolved rows: {rows_total}\n\n")
    breakdown = list(breakdown)
    if rows_total and breakdown:
        lines.append("| source | metric | semantics | rows |\n")
        lines.append("| --- | --- | --- | ---: |\n")
        for source, metric, semantics, count in breakdown:
            lines.append(f"| {source} | {metric} | {semantics} | {count} |\n")
    COUNTS_PATH.write_text("".join(lines), encoding="utf-8")
    return "".join(lines)


def _append_to_summary(markdown: str) -> None:
    if SUMMARY_PATH.exists():
        with SUMMARY_PATH.open("a", encoding="utf-8") as handle:
            handle.write("\n\n")
            handle.write(markdown)


def _append_to_step_summary(markdown: str) -> None:
    summary_target = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_target:
        with open(summary_target, "a", encoding="utf-8") as handle:
            handle.write(markdown)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify DuckDB facts_resolved counts and append diagnostics summaries."
    )
    parser.add_argument(
        "db",
        nargs="?",
        help="Optional DuckDB URL or path override (defaults to RESOLVER_DB_URL).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    candidate = (args.db or "").strip()
    if candidate:
        db_url = candidate
    else:
        env_url = os.environ.get("RESOLVER_DB_URL", "").strip()
        if not env_url:
            raise SystemExit("RESOLVER_DB_URL not set")
        db_url = env_url

    db_path = _normalise_db_path(db_url)

    try:
        con = duckdb.connect(db_path)
    except Exception as exc:  # pragma: no cover - missing db
        print(f"Failed to connect to DuckDB at {db_path}: {exc}")
        return 1

    try:
        if not _table_exists(con, "facts_resolved"):
            markdown = _write_summary(db_path, 0, [])
            _append_to_step_summary(markdown)
            _append_to_summary(markdown)
            return 0

        rows_total = con.execute(
            "SELECT COUNT(*) FROM facts_resolved"
        ).fetchone()[0]
        breakdown = _fetch_breakdown(con) if rows_total else []
    finally:
        con.close()

    markdown = _write_summary(db_path, rows_total, breakdown)
    _append_to_step_summary(markdown)
    _append_to_summary(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
