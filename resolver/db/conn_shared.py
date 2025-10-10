"""Shared DuckDB connection helpers with canonical URL handling."""

from __future__ import annotations

import logging
import os
import pathlib
from typing import Optional, Tuple
from urllib.parse import urlparse

from resolver.common import get_logger

LOGGER = get_logger(__name__)
if os.getenv("RESOLVER_DEBUG") == "1":
    LOGGER.setLevel(logging.DEBUG)

_DUCKDB_CACHE: dict[str, "duckdb.DuckDBPyConnection"] = {}


def normalize_duckdb_url(db_url: str) -> str:
    """Return a canonical DuckDB filesystem path for ``db_url``."""

    raw = (db_url or "").strip()
    if not raw:
        return ":memory:"
    if raw == ":memory:":
        return raw

    parsed = urlparse(raw)
    path_candidate: pathlib.Path
    if parsed.scheme == "duckdb":
        if parsed.netloc:
            candidate = f"{parsed.netloc}{parsed.path}" if parsed.path else parsed.netloc
        else:
            candidate = parsed.path
        if not candidate:
            return ":memory:"
        path_candidate = pathlib.Path(candidate)
    else:
        path_candidate = pathlib.Path(raw)

    if not path_candidate.is_absolute():
        path_candidate = pathlib.Path(os.getcwd()) / path_candidate

    path_candidate = path_candidate.expanduser().resolve()
    try:
        path_candidate.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Parent directories may already exist or be outside writable roots.
        pass
    return str(path_candidate)


def get_shared_duckdb_conn(db_url: Optional[str]) -> Tuple["duckdb.DuckDBPyConnection", str, bool]:
    """Return a DuckDB connection, its path, and whether it was reused."""

    import duckdb

    path = normalize_duckdb_url(db_url or "")
    conn = _DUCKDB_CACHE.get(path)
    reused = True
    if conn is None:
        conn = duckdb.connect(database=path, read_only=False)
        _DUCKDB_CACHE[path] = conn
        reused = False
    if os.getenv("RESOLVER_DEBUG") == "1" and LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug(
            "%s DuckDB connection for path: %s", "Reused" if reused else "Created", path
        )

    return conn, path, reused
