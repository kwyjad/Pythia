# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Runtime flags governing DuckDB usage in Resolver."""

from __future__ import annotations

import os
from typing import Optional, Tuple

from resolver.db._duckdb_available import (
    DUCKDB_AVAILABLE,
    duckdb_unavailable_reason,
)

FAST_FIXTURES_ENV = "RESOLVER_FAST_FIXTURES_MODE"
_VALID_MODES = {"duckdb", "noop"}


def _resolve_fast_fixtures_mode() -> Tuple[str, bool, Optional[str]]:
    """Return (mode, auto_fallback, reason) for fast-fixtures bootstrap."""

    raw = os.getenv(FAST_FIXTURES_ENV, "duckdb")
    mode = raw.strip().lower() or "duckdb"
    if mode not in _VALID_MODES:
        mode = "duckdb"

    auto_fallback = False
    reason: Optional[str] = None

    if mode == "duckdb" and not DUCKDB_AVAILABLE:
        auto_fallback = True
        mode = "noop"
        reason = duckdb_unavailable_reason() or "duckdb unavailable"

    return mode, auto_fallback, reason


FAST_FIXTURES_MODE, FAST_FIXTURES_AUTO_FALLBACK, FAST_FIXTURES_REASON = (
    _resolve_fast_fixtures_mode()
)
FORCE_NOOP = FAST_FIXTURES_MODE == "noop"
USE_DUCKDB = not FORCE_NOOP and DUCKDB_AVAILABLE


def resolve_fast_fixtures_mode() -> Tuple[str, bool, Optional[str]]:
    """Public helper mirroring :func:`_resolve_fast_fixtures_mode`."""

    return _resolve_fast_fixtures_mode()


__all__ = [
    "FAST_FIXTURES_ENV",
    "FAST_FIXTURES_MODE",
    "FAST_FIXTURES_AUTO_FALLBACK",
    "FAST_FIXTURES_REASON",
    "FORCE_NOOP",
    "USE_DUCKDB",
    "resolve_fast_fixtures_mode",
]
