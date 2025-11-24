#!/usr/bin/env python3
"""Summarize DuckDB write results for ingestion diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
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
DEFAULT_TABLES: Tuple[str, ...] = ("facts_resolved",)


def _append_error_to_summary(section: str, exc: Exception, context: dict[str, object]) -> None:
    if not sys.executable:
        return

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.ci.append_error_to_summary",
                "--section",
                section,
                "--error-type",
                type(exc).__name__,
                "--message",
                str(exc),
                "--context",
                json.dumps(context, sort_keys=True),
            ],
            check=False,
        )
    except Exception:
        pass


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


def _get_table_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    """Return set of column names for the given DuckDB table (best-effort)."""
    try:
        rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    except Exception:
        return set()
    return {str(row[1]) for row in rows if len(row) >= 2}


def _fetch_breakdown(
    con: duckdb.DuckDBPyConnection,
) -> Iterable[Tuple[str, str, str, int]]:
    """Return per-table/source/metric/semantics counts across known tables."""

    def build_select_for_table(table: str, fixed_source: str | None = None) -> str | None:
        cols = _get_table_columns(con, table)
        if not cols:
            return None

        if fixed_source is not None:
            source_expr = f"'{fixed_source}'"
        elif "source" in cols:
            source_expr = "source"
        elif "source_id" in cols:
            source_expr = "source_id"
        elif "publisher" in cols:
            source_expr = "publisher"
        elif "source_name" in cols:
            source_expr = "source_name"
        elif "source_type" in cols:
            source_expr = "source_type"
        else:
            source_expr = "''"

        if "metric" in cols:
            metric_expr = "metric"
        elif "series" in cols:
            metric_expr = "series"
        else:
            metric_expr = "''"

        if "semantics" in cols:
            semantics_expr = "semantics"
        elif "series_semantics" in cols:
            semantics_expr = "series_semantics"
        else:
            semantics_expr = "''"

        return (
            f"SELECT '{table}' AS table_name, "
            f"{source_expr} AS source, "
            f"{metric_expr} AS metric, "
            f"{semantics_expr} AS semantics "
            f"FROM {table}"
        )

    selects: list[str] = []

    acled_sel = build_select_for_table("acled_monthly_fatalities", fixed_source="ACLED")
    if acled_sel:
        selects.append(acled_sel)

    for t in ("facts_resolved", "facts_deltas"):
        sel = build_select_for_table(t)
        if sel:
            selects.append(sel)

    if not selects:
        return []

    union_sql = " UNION ALL ".join(selects)
    sql = f"""
    WITH unioned AS (
        {union_sql}
    )
    SELECT
        table_name AS table,
        COALESCE(source, '') AS source,
        COALESCE(metric, '') AS metric,
        COALESCE(semantics, '') AS semantics,
        COUNT(*) AS count
    FROM unioned
    GROUP BY table_name, source, metric, semantics
    ORDER BY table_name, source, metric, semantics
    """
    return con.execute(sql).fetchall()


def _write_summary(
    db_path: str,
    rows_total: int,
    breakdown: Iterable[Tuple[str, str, str, int]],
    table_counts: Sequence[Tuple[str, int]],
    missing_tables: Sequence[str],
) -> str:
    COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [MARKDOWN_HEADER]
    lines.append(f"- Database: {db_path}\n")
    lines.append(f"- facts_resolved rows: {rows_total}\n\n")
    if table_counts:
        lines.append("| table | rows |\n")
        lines.append("| --- | ---: |\n")
        for table, count in table_counts:
            lines.append(f"| {table} | {count} |\n")
        lines.append("\n")
    if missing_tables:
        lines.append("**Missing tables**\n\n")
        for table in missing_tables:
            lines.append(f"- {table}\n")
        lines.append("\n")
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


def _main_impl(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify DuckDB facts_resolved counts and append diagnostics summaries."
    )
    parser.add_argument(
        "db",
        nargs="?",
        help="Optional DuckDB URL or path override (defaults to RESOLVER_DB_URL).",
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        default=None,
        help="Optional list of tables to verify (default: facts_resolved).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Treat missing tables as warnings instead of errors.",
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

    tables_arg = args.tables or []
    tables: list[str] = []
    seen: set[str] = set()
    for table in tables_arg:
        cleaned = str(table).strip()
        if not cleaned:
            continue
        if cleaned not in seen:
            tables.append(cleaned)
            seen.add(cleaned)
    if not tables:
        tables = list(DEFAULT_TABLES)
        seen = set(tables)
    if "facts_resolved" not in seen:
        tables.insert(0, "facts_resolved")
        seen.add("facts_resolved")

    table_counts: list[Tuple[str, int]] = []
    missing_tables: list[str] = []
    rows_total = 0
    try:
        for table in tables:
            if not _table_exists(con, table):
                missing_tables.append(table)
                continue
            try:
                count = int(
                    con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
            except Exception as exc:  # pragma: no cover - query failure
                print(f"Failed to count rows for {table}: {exc}")
                return 1
            table_counts.append((table, count))
            if table == "facts_resolved":
                rows_total = count
        breakdown = _fetch_breakdown(con) if rows_total else []
    finally:
        con.close()

    markdown = _write_summary(db_path, rows_total, breakdown, table_counts, missing_tables)
    _append_to_step_summary(markdown)
    _append_to_summary(markdown)
    if missing_tables and not args.allow_missing:
        for table in missing_tables:
            print(f"ERROR: Table '{table}' not found in {db_path}")
        return 1
    if missing_tables:
        for table in missing_tables:
            print(f"WARNING: Table '{table}' not found in {db_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _main_impl(argv)
    except Exception as exc:
        context = {
            "argv": list(argv) if argv is not None else sys.argv[1:],
            "exception_class": type(exc).__name__,
            "resolver_db_url": os.environ.get("RESOLVER_DB_URL", ""),
        }
        _append_error_to_summary("Verify DuckDB Counts — error", exc, context)
        raise


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
