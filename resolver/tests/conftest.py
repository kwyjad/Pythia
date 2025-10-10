import pytest

from resolver.db import conn_shared


@pytest.fixture(scope="session")
def clear_duckdb_cache():
    """Return a helper that clears a specific DuckDB cache entry by URL."""

    def _clear(db_url: str) -> None:
        conn_shared.clear_cached_connection(db_url)

    return _clear


@pytest.fixture(autouse=True)
def _duckdb_cache_hygiene():
    """Ensure all DuckDB caches are cleared around each test invocation."""

    conn_shared.clear_all_cached_connections()
    yield
    conn_shared.clear_all_cached_connections()
