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


def _is_aborted_txn_error(exc: Exception) -> bool:
    """True for DuckDB's "current transaction is aborted" state error.

    That error says nothing about the object being probed — it says the
    PREVIOUS statement failed. Answering the probe from it is how a live
    `scores` table got reported as missing.
    """

    message = str(exc).lower()
    return "transaction is aborted" in message or "please rollback" in message


def _probe(conn: Any, sql: str, on_error: Any, label: str, *, warn: bool = False) -> Any:
    """Run a read-only probe, retrying ONCE after scrubbing an aborted txn.

    Rolling back without re-running the probe still returns the fallback for
    THIS call — which is exactly the silent no-op the scrub was meant to
    prevent (compute_calibration_pythia reads `_table_exists(conn, "scores")`
    once and skips the whole run on False). The retry makes the helpers
    answer about the object rather than about the connection's state.
    """

    try:
        return conn.execute(sql).fetchall()
    except Exception as exc:
        aborted = _is_aborted_txn_error(exc)
        rollback_quietly(conn)
        if aborted:
            try:
                return conn.execute(sql).fetchall()
            except Exception as retry_exc:  # noqa: BLE001
                LOGGER.warning("%s failed after ROLLBACK retry: %s", label, retry_exc)
                rollback_quietly(conn)
                return on_error
        if warn:
            LOGGER.warning("%s failed: %s", label, exc)
        else:
            LOGGER.debug("%s failed: %s", label, exc)
        return on_error


_MISSING = object()


def table_exists(conn: Any, name: str) -> bool:
    result = _probe(conn, f"PRAGMA table_info('{name}')", _MISSING, f"table_exists({name})")
    return result is not _MISSING


def row_count(conn: Any, name: str) -> int:
    rows = _probe(
        conn, f"SELECT COUNT(*) FROM {name}", _MISSING, f"row_count({name})", warn=True
    )
    if rows is _MISSING or not rows:
        return 0
    return rows[0][0] or 0


def column_exists(conn: Any, table: str, column: str) -> bool:
    rows = _probe(conn, f"PRAGMA table_info('{table}')", _MISSING, f"column_exists({table}.{column})")
    if rows is _MISSING:
        return False
    return any(str(r[1]).lower() == column.lower() for r in rows)


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
