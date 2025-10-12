from __future__ import annotations

from functools import lru_cache
from types import ModuleType

import importlib.util

import pytest


@lru_cache(maxsize=1)
def _maybe_conn_shared() -> ModuleType | None:
    """Return ``resolver.db.conn_shared`` if DuckDB is available."""

    if importlib.util.find_spec("duckdb") is None:
        return None
    from resolver.db import conn_shared as module

    return module


@pytest.fixture(scope="session")
def clear_duckdb_cache():
    """Return a helper that clears a specific DuckDB cache entry by URL."""

    module = _maybe_conn_shared()
    if module is None:
        pytest.skip("duckdb optional dependency is not installed")

    def _clear(db_url: str) -> None:
        module.clear_cached_connection(db_url)

    return _clear


@pytest.fixture(autouse=True)
def _duckdb_cache_hygiene():
    """Ensure all DuckDB caches are cleared around each test invocation."""

    module = _maybe_conn_shared()
    if module is None:
        yield
        return

    module.clear_all_cached_connections()
    yield
    module.clear_all_cached_connections()
