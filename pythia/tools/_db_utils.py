# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Shared DuckDB helpers for the post-forecast pipeline tools.

These were previously duplicated (with drift) across compute_resolutions,
compute_scores, compute_calibration_pythia, and generate_calibration_advice.
The canonical versions here are the rollback-safe ones: DuckDB >= 1.5 leaves
a connection in an aborted-transaction state after a failed statement inside
an implicit transaction, so every except path scrubs with ROLLBACK before
returning a fallback value (see the DuckDB 1.5.2 calibration no-op entry in
CLAUDE.md's Known failure modes).
"""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

__all__ = [
    "rollback_quietly",
    "table_exists",
    "row_count",
    "column_exists",
    "add_column_if_missing",
]


def rollback_quietly(conn: Any) -> None:
    """Scrub a possibly-aborted transaction so the next statement runs.

    DuckDB >= 1.5 leaves the connection in an aborted-transaction state
    after a failed DDL inside an implicit transaction. Issue ROLLBACK so
    the next statement isn't poisoned with "Current transaction is
    aborted (please ROLLBACK)".
    """
    try:
        conn.execute("ROLLBACK")
    except Exception:
        pass


def table_exists(conn: Any, name: str) -> bool:
    try:
        conn.execute(f"PRAGMA table_info('{name}')").fetchall()
        return True
    except Exception as exc:
        LOGGER.debug("table_exists(%s) returned False: %s", name, exc)
        rollback_quietly(conn)
        return False


def row_count(conn: Any, name: str) -> int:
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0] or 0
    except Exception as exc:
        LOGGER.warning("row_count(%s) failed, returning 0: %s", name, exc)
        rollback_quietly(conn)
        return 0


def column_exists(conn: Any, table: str, column: str) -> bool:
    try:
        rows = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        return any(str(r[1]).lower() == column.lower() for r in rows)
    except Exception:
        rollback_quietly(conn)
        return False


def add_column_if_missing(conn: Any, table: str, column: str, col_type: str) -> None:
    """Idempotent ALTER TABLE ADD COLUMN.

    Checks PRAGMA table_info first and only ALTERs when the column is
    missing — the "try ALTER, swallow the error" pattern aborts the
    transaction on DuckDB >= 1.5 (see module docstring).
    """
    if column_exists(conn, table, column):
        return
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except Exception as exc:
        LOGGER.warning("Failed to add %s.%s: %s", table, column, exc)
        rollback_quietly(conn)
