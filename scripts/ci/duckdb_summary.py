#!/usr/bin/env python3
"""Emit a Markdown DuckDB summary for GitHub Actions step output."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import json

from resolver.db import duckdb_io


def _import_duckdb():  # pragma: no cover - import guarded in tests
    try:
        import duckdb  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency
        print("## DuckDB\n")
        print(f"- **Status:** duckdb import failed: {exc}")
        return None
    return duckdb


def _iter_tables(raw: Sequence[str] | str | None) -> Iterable[str]:
    default = ("facts_raw", "facts_resolved", "facts_deltas")
    if raw is None:
        return default
    if isinstance(raw, str):
        items = [part.strip() for part in raw.split(",") if part.strip()]
    else:
        items = [str(part).strip() for part in raw if str(part).strip()]
    return items or default


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_safe_identifier(name: str) -> bool:
    return bool(_IDENTIFIER_RE.match(name))


def _table_columns(conn, table: str) -> list[str]:
    try:
        cursor = conn.execute(f"SELECT * FROM {table} LIMIT 0")
    except Exception:
        return []
    description = getattr(cursor, "description", None)
    if not description:
        return []
    return [str(col[0]) for col in description if col and col[0]]


def _build_expr(columns: list[str], candidates: Sequence[str], fallback: str = "''") -> str:
    available = [name for name in candidates if name in columns]
    if not available:
        return fallback
    inner = ", ".join(available + [fallback])
    return f"COALESCE({inner})"


def _collect_breakdown(conn, tables: Iterable[str], limit: int = 20) -> list[tuple[str, str, str, str, int]]:
    rows: list[tuple[str, str, str, str, int]] = []
    for table in tables:
        if not table or not _is_safe_identifier(table):
            continue
        columns = _table_columns(conn, table)
        if not columns:
            continue
        source_expr = _build_expr(columns, ("source_id", "source"))
        metric_expr = _build_expr(columns, ("metric",))
        semantics_expr = _build_expr(columns, ("series_semantics", "semantics"))
        table_literal = table.replace("'", "''")
        query = (
            "SELECT '{table_literal}' AS table_name, {source} AS source, {metric} AS metric, "
            "{semantics} AS semantics, COUNT(*) AS count FROM {table} GROUP BY 1,2,3,4 "
            "ORDER BY 1,2,3,4"
        ).format(
            table=table,
            table_literal=table_literal,
            source=source_expr,
            metric=metric_expr,
            semantics=semantics_expr,
        )
        try:
            entries = conn.execute(query).fetchall()
        except Exception:
            continue
        for row in entries:
            try:
                table_name = str(row[0] or table)
                source = str(row[1] or "")
                metric = str(row[2] or "")
                semantics = str(row[3] or "")
                count = int(row[4])
            except Exception:
                continue
            rows.append((table_name, source, metric, semantics, count))
    rows.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    if limit > 0:
        return rows[:limit]
    return rows


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
    canonical_url: str | None = None
    db_path: str | None = None
    if db_arg:
        try:
            canonical_url, db_path = duckdb_io.normalize_duckdb_target(db_arg)
        except Exception:
            canonical_url = None
            db_path = None

    if _import_duckdb() is None:
        return 0

    print("## DuckDB Diagnostics (duckdb_summary.py)")
    print("")

    if not db_path:
        message = "no DuckDB target provided"
        print(f"- **Status:** {message}")
        _print_error_json(message)
        return 0

    fs_path = Path(db_path)
    if not fs_path.exists():
        shown = canonical_url or db_path
        message = f"database not found: {shown}"
        print(f"- **Status:** {message}")
        _print_error_json(message)
        return 0

    try:
        target = canonical_url or db_path
        conn = duckdb_io.get_db(target)
    except Exception as exc:  # pragma: no cover - connection failure
        shown = canonical_url or db_path
        message = f"failed to open {shown}: {exc}"
        print(f"- **Status:** {message}")
        _print_error_json(message)
        return 0

    try:
        target_display = canonical_url or db_path
        print(f"- **Database:** `{target_display}`")
        tables_list: list[str] = []
        try:
            tables_list = [
                str(row[0])
                for row in conn.execute("SHOW TABLES").fetchall()
                if row and row[0]
            ]
        except Exception:
            tables_list = []
        if tables_list:
            print(f"- **Existing tables:** {', '.join(sorted(tables_list))}")
        else:
            print("- **Existing tables:** (none)")
        rows: list[tuple[str, int]] = []
        for table in _iter_tables(args.tables):
            if not table:
                continue
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except Exception:
                continue
            rows.append((table, int(count)))

        print("")
        print("| table | rows |")
        print("| --- | --- |")
        if rows:
            for table, count in rows:
                print(f"| {table} | {count} |")
        else:
            print("| (none) | 0 |")

        breakdown = _collect_breakdown(conn, (table for table, _ in rows))
        if breakdown:
            print("")
            print("### Rows by source / metric / semantics")
            print("| table | source | metric | semantics | count |")
            print("| --- | --- | --- | --- | --- |")
            for table_name, source, metric, semantics, count in breakdown:
                print(f"| {table_name} | {source} | {metric} | {semantics} | {count} |")
        elif rows:
            print("")
            print("### Rows by source / metric / semantics")
            print("| table | source | metric | semantics | count |")
            print("| --- | --- | --- | --- | --- |")
            print("| (no breakdown) |  |  |  | 0 |")
    finally:
        duckdb_io.close_db(conn)
    return 0


def _print_error_json(message: str) -> None:
    print("")
    print("```json")
    print(json.dumps({"error": message}, sort_keys=True))
    print("```")


if __name__ == "__main__":
    sys.exit(main())
