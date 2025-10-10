"""Shared DuckDB connection helpers with canonical URL handling and caching."""

from __future__ import annotations

import logging
import os
import pathlib
import re
import sys
import threading
from typing import Optional, Tuple

logger = logging.getLogger(__name__)
if os.getenv("RESOLVER_DEBUG") == "1":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

_CACHE: dict[str, "ConnectionWrapper"] = {}
_LOCK = threading.RLock()


def normalize_duckdb_url(db_url: str) -> str:
    """Return a canonical DuckDB filesystem path for ``db_url``."""

    raw = (db_url or "").strip()
    if not raw or raw == ":memory:":
        return ":memory:"
    if raw.startswith("duckdb://"):
        path = re.sub(r"^duckdb:/+", "", raw)
        if not path.startswith("/"):
            path = os.path.join(os.getcwd(), path)
    else:
        path = raw

    path_obj = pathlib.Path(path).expanduser().resolve()
    try:
        path_obj.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return str(path_obj)


class ConnectionWrapper:
    """Proxy that evicts cache entries on ``close`` and exposes health checks."""

    def __init__(self, path: str, raw: "duckdb.DuckDBPyConnection") -> None:
        self._path = path
        self._raw = raw
        self._closed = False

    def __getattr__(self, name: str):  # pragma: no cover - thin proxy
        return getattr(self._raw, name)

    def _healthcheck(self) -> bool:
        try:
            self._raw.execute("SELECT 1")
            return True
        except Exception as exc:  # pragma: no cover - diagnostic path
            logger.debug("DuckDB healthcheck failed for %s: %r", self._path, exc)
            return False

    def close(self) -> None:  # pragma: no cover - exercised via tests
        with _LOCK:
            if not self._closed:
                try:
                    self._raw.close()
                finally:
                    self._closed = True
                    if _CACHE.get(self._path) is self:
                        _CACHE.pop(self._path, None)
                        logger.debug("DuckDB cache evicted on close: %s", self._path)


def _open_new(path: str) -> ConnectionWrapper:
    import duckdb

    raw = duckdb.connect(database=path, read_only=False)
    return ConnectionWrapper(path, raw)


def get_shared_duckdb_conn(
    db_url: Optional[str], *, force_reopen: bool = False
) -> Tuple[ConnectionWrapper, str]:
    """Return a shared DuckDB connection wrapper and its canonical path."""

    path = normalize_duckdb_url(db_url or "")
    if os.getenv("RESOLVER_DISABLE_CONN_CACHE") == "1":
        wrapper = _open_new(path)
        logger.debug("DuckDB opened (cache disabled): %s", path)
        return wrapper, path

    with _LOCK:
        wrapper = _CACHE.get(path)
        if (
            force_reopen
            or wrapper is None
            or wrapper._closed
            or not wrapper._healthcheck()
        ):
            wrapper = _open_new(path)
            _CACHE[path] = wrapper
            logger.debug("DuckDB opened (force=%s): %s", force_reopen, path)
        else:
            logger.debug("DuckDB cache hit: %s", path)
        return wrapper, path


def clear_cached_connection(db_url: str) -> None:
    """Evict ``db_url`` from the shared cache."""

    path = normalize_duckdb_url(db_url)
    with _LOCK:
        wrapper = _CACHE.pop(path, None)
        if wrapper and not wrapper._closed:
            try:
                wrapper.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        logger.debug("DuckDB cache cleared: %s", path)
