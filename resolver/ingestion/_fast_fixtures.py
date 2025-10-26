"""Utilities for determining fast-fixtures bootstrap behaviour."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from resolver.db.runtime_flags import (
    FAST_FIXTURES_ENV,
    resolve_fast_fixtures_mode as _resolve_fast_fixtures_mode,
)


def resolve_fast_fixtures_mode() -> Tuple[str, bool, Optional[str]]:
    """Delegate to :mod:`resolver.db.runtime_flags` for the resolved mode."""

    return _resolve_fast_fixtures_mode()


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
