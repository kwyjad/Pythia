# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Diagnostic logging helpers gated by the ``RESOLVER_DIAG`` flag."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, Optional


def diag_enabled() -> bool:
    """Return ``True`` when detailed diagnostics should be emitted."""

    return os.getenv("RESOLVER_DIAG") == "1"


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured for diagnostics when enabled."""

    logger = logging.getLogger(name)
    if not diag_enabled():
        return logger
    has_diag_handler = any(getattr(handler, "_resolver_diag", False) for handler in logger.handlers)
    if not has_diag_handler:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        handler._resolver_diag = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


def log_json(logger: logging.Logger, msg: str, **payload: Any) -> None:
    """Emit ``msg`` with a JSON payload when diagnostics are enabled."""

    if not diag_enabled():
        return
    try:
        serialised = json.dumps(payload, default=str, sort_keys=True)
    except Exception:
        serialised = str(payload)
    logger.debug("%s %s", msg, serialised)


def duckdb_ex_classes(duckdb_mod):
    """Return a tuple of DuckDB exception classes tolerant of version drift."""

    base = getattr(duckdb_mod, "Error", Exception)
    catalog = getattr(duckdb_mod, "CatalogException", base)
    dependency = getattr(duckdb_mod, "DependencyException", base)
    connection = getattr(duckdb_mod, "ConnectionException", base)
    return (base, catalog, dependency, connection)


def dump_table_meta(conn, table: str) -> Dict[str, Any]:
    """Return PRAGMA metadata for ``table`` (best-effort)."""

    info: Dict[str, Any] = {"table": table, "exists": False, "columns": [], "indexes": []}
    try:
        table_info = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
        info["columns"] = table_info
        info["exists"] = len(table_info) > 0
    except Exception as exc:  # pragma: no cover - defensive diagnostics only
        info["error_table_info"] = repr(exc)
        return info

    try:
        index_rows = conn.execute(f"PRAGMA indexes('{table}')").fetchall()
    except Exception as exc:  # pragma: no cover - defensive diagnostics only
        info["error_indexes"] = repr(exc)
        return info

    for row in index_rows:
        idx_name = row[0] if row else None
        columns = []
        if idx_name:
            try:
                columns = conn.execute(f"PRAGMA index_info('{idx_name}')").fetchall()
            except Exception as exc:  # pragma: no cover - defensive diagnostics only
                columns = [("index_info_error", repr(exc))]
        info["indexes"].append({"name": idx_name, "columns": columns})
    return info


def dump_counts(
    conn,
    ym: Optional[str] = None,
    iso3: Optional[str] = None,
    hazard: Optional[str] = None,
    cutoff: Optional[str] = None,
) -> Dict[str, Any]:
    """Return total and keyed row counts for the DuckDB facts tables."""

    counts: Dict[str, Any] = {}
    try:
        counts["facts_resolved_total"] = conn.execute("SELECT COUNT(*) FROM facts_resolved").fetchone()[0]
    except Exception as exc:  # pragma: no cover - diagnostic helper
        counts["facts_resolved_total_error"] = repr(exc)
    try:
        counts["facts_deltas_total"] = conn.execute("SELECT COUNT(*) FROM facts_deltas").fetchone()[0]
    except Exception as exc:  # pragma: no cover - diagnostic helper
        counts["facts_deltas_total_error"] = repr(exc)

    if ym and iso3 and hazard:
        params = [ym, iso3, hazard]
        try:
            counts["facts_deltas_key_count"] = conn.execute(
                "SELECT COUNT(*) FROM facts_deltas WHERE ym=? AND iso3=? AND hazard_code=?",
                params,
            ).fetchone()[0]
        except Exception as exc:  # pragma: no cover - diagnostic helper
            counts["facts_deltas_key_count_error"] = repr(exc)
        try:
            counts["facts_resolved_key_count"] = conn.execute(
                "SELECT COUNT(*) FROM facts_resolved WHERE ym=? AND iso3=? AND hazard_code=?",
                params,
            ).fetchone()[0]
        except Exception as exc:  # pragma: no cover - diagnostic helper
            counts["facts_resolved_key_count_error"] = repr(exc)
        if cutoff:
            try:
                counts["facts_deltas_cutoff_count"] = conn.execute(
                    """
                    WITH a AS (
                        SELECT TRY_CAST(as_of AS DATE) AS as_of_date
                        FROM facts_deltas
                        WHERE ym=? AND iso3=? AND hazard_code=?
                    )
                    SELECT COUNT(*) FROM a
                    WHERE as_of_date IS NULL OR as_of_date <= TRY_CAST(? AS DATE)
                    """,
                    [ym, iso3, hazard, cutoff],
                ).fetchone()[0]
            except Exception as exc:  # pragma: no cover - diagnostic helper
                counts["facts_deltas_cutoff_count_error"] = repr(exc)
    return counts
