"""Utilities for determining fast-fixtures bootstrap behaviour."""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

from resolver.db._duckdb_available import DUCKDB_AVAILABLE, duckdb_unavailable_reason

FAST_FIXTURES_ENV = "RESOLVER_FAST_FIXTURES_MODE"
_VALID_MODES = {"duckdb", "noop"}


def resolve_fast_fixtures_mode() -> Tuple[str, bool, Optional[str]]:
    """Return (mode, auto_fallback, reason) for fast-fixtures bootstrap."""

    raw = os.getenv(FAST_FIXTURES_ENV, "duckdb")
    mode = raw.strip().lower() or "duckdb"
    if mode not in _VALID_MODES:
        mode = "duckdb"

    auto_fallback = False
    reason = None

    if mode == "duckdb" and not DUCKDB_AVAILABLE:
        auto_fallback = True
        mode = "noop"
        reason = duckdb_unavailable_reason() or "duckdb unavailable"

    return mode, auto_fallback, reason


def log_fast_fixture_mode(logger: logging.Logger) -> str:
    """Log the resolved fast-fixtures mode and return it."""

    mode, auto_fallback, reason = resolve_fast_fixtures_mode()
    if mode == "duckdb":
        if auto_fallback:
            logger.warning(
                "Fast fixtures: DuckDB unavailable (%s) -> falling back to noop mode",
                reason,
            )
        else:
            logger.debug("Fast fixtures: using DuckDB-backed bootstrap")
    else:
        if auto_fallback:
            logger.warning(
                "Fast fixtures: DuckDB unavailable (%s) -> noop/offline-smoke fallback",
                reason,
            )
        else:
            logger.info(
                "Fast fixtures: RESOLVER_FAST_FIXTURES_MODE=noop -> using offline-smoke fallback",
            )
    return mode


__all__ = ["FAST_FIXTURES_ENV", "resolve_fast_fixtures_mode", "log_fast_fixture_mode"]
