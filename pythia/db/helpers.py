# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Shared DuckDB introspection helpers used across query modules."""

from __future__ import annotations

from typing import Optional


def table_exists(con, table: str) -> bool:
    """Check whether *table* exists in the connected DuckDB database."""
    try:
        row = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE LOWER(table_name) = LOWER(?)",
            [table],
        ).fetchone()
        return bool(row and row[0])
    except Exception:
        pass

    try:
        df = con.execute("PRAGMA show_tables").fetchdf()
    except Exception:
        return False
    if df.empty:
        return False
    first_col = df.columns[0]
    return df[first_col].astype(str).str.lower().eq(table.lower()).any()


def table_columns(con, table: str) -> set[str]:
    """Return the set of (lowercased) column names for *table*."""
    try:
        df = con.execute(f"PRAGMA table_info('{table}')").fetchdf()
    except Exception:
        return set()
    if df.empty or "name" not in df.columns:
        return set()
    return set(df["name"].astype(str).str.lower().tolist())


def pick_column(columns: set[str], candidates: list[str]) -> Optional[str]:
    """Return the first candidate present in *columns* (lowercased), or ``None``."""
    for candidate in candidates:
        if candidate.lower() in columns:
            return candidate.lower()
    return None


def pick_timestamp_column(
    con, table: str, candidates: list[str]
) -> Optional[str]:
    """Convenience wrapper: pick the first matching timestamp column for *table*."""
    if not table_exists(con, table):
        return None
    cols = table_columns(con, table)
    return pick_column(cols, candidates)
