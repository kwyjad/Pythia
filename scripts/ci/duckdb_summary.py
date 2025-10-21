#!/usr/bin/env python3
"""Emit lightweight DuckDB table counts for diagnostics."""

from __future__ import annotations

import os
import sys

try:
    import duckdb
except Exception as exc:  # pragma: no cover - optional dependency
    print(f"duckdb import failed: {exc}")
    sys.exit(0)


def main() -> int:
    db_path = os.environ.get("RESOLVER_DB_PATH", "data/resolver.duckdb")
    try:
        con = duckdb.connect(db_path)
    except Exception as exc:  # pragma: no cover - missing database
        print(f"no db: {exc}")
        return 0

    targets = ("facts_raw", "facts_resolved", "facts_deltas")
    for table in targets:
        try:
            count = con.sql(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"{table} {count}")
        except Exception as exc:  # pragma: no cover - missing table
            print(f"{table} unavailable {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
