"""Helpers for probing optional DuckDB dependency availability."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Optional

DUCKDB_AVAILABLE: bool
_DUCKDB_MODULE: Optional[ModuleType]
_DUCKDB_IMPORT_ERROR: Optional[Exception]

try:  # pragma: no cover - minimal import guard
    _module = import_module("duckdb")
except Exception as exc:  # pragma: no cover - optional dependency missing
    DUCKDB_AVAILABLE = False
    _DUCKDB_MODULE = None
    _DUCKDB_IMPORT_ERROR = exc
else:  # pragma: no branch - import succeeded
    DUCKDB_AVAILABLE = True
    _DUCKDB_MODULE = _module
    _DUCKDB_IMPORT_ERROR = None


def get_duckdb() -> ModuleType:
    """Return the imported DuckDB module or raise a helpful error."""

    if not DUCKDB_AVAILABLE or _DUCKDB_MODULE is None:
        raise RuntimeError(
            "DuckDB is required for this operation but is not installed. "
            "Install the 'duckdb' extra or set RESOLVER_FAST_FIXTURES_MODE=noop to bypass."
        ) from _DUCKDB_IMPORT_ERROR
    return _DUCKDB_MODULE


def duckdb_unavailable_reason() -> Optional[str]:
    """Return the string form of the DuckDB import failure, if any."""

    if DUCKDB_AVAILABLE:
        return None
    if _DUCKDB_IMPORT_ERROR is None:
        return "unknown"
    return str(_DUCKDB_IMPORT_ERROR)


__all__ = ["DUCKDB_AVAILABLE", "duckdb_unavailable_reason", "get_duckdb"]
