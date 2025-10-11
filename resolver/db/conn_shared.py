"""Shared DuckDB connection helpers with canonical URL handling and caching."""

from __future__ import annotations

import atexit
import logging
import os
import pathlib
import re
import threading
from typing import Dict, Optional, Tuple

from resolver.diag.diagnostics import get_logger as get_diag_logger, log_json

logger = logging.getLogger(__name__)
if not logger.handlers:  # pragma: no cover - silence library default
    logger.addHandler(logging.NullHandler())
if os.getenv("RESOLVER_DEBUG") == "1":
    logger.setLevel(logging.DEBUG)

_PROCESS_CACHE: Dict[str, "ConnectionWrapper"] = {}
_LOCK = threading.RLock()
_TLS = threading.local()
_ALL_PATHS: set[str] = set()

_DIAG_LOGGER = get_diag_logger(f"{__name__}.diag")


def _cache_mode() -> str:
    mode = os.getenv("RESOLVER_CONN_CACHE_MODE", "process").strip().lower()
    if mode not in {"process", "thread"}:
        return "process"
    return mode


def _get_cache() -> Dict[str, "ConnectionWrapper"]:
    if _cache_mode() == "thread":
        cache = getattr(_TLS, "cache", None)
        if cache is None:
            cache = {}
            _TLS.cache = cache
        return cache
    return _PROCESS_CACHE


def _iter_current_caches() -> tuple[Dict[str, "ConnectionWrapper"], ...]:
    caches: list[Dict[str, "ConnectionWrapper"]] = [_PROCESS_CACHE]
    tls_cache = getattr(_TLS, "cache", None)
    if isinstance(tls_cache, dict):
        caches.append(tls_cache)
    return tuple(caches)


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
        self._last_event = "miss"

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
                    cache = _get_cache()
                    if cache.get(self._path) is self:
                        cache.pop(self._path, None)
                    logger.debug("DuckDB cache evicted on close: %s", self._path)


def _open_new(path: str) -> ConnectionWrapper:
    import duckdb

    raw = duckdb.connect(database=path, read_only=False)
    wrapper = ConnectionWrapper(path, raw)
    _ALL_PATHS.add(path)
    return wrapper


def get_shared_duckdb_conn(
    db_url: Optional[str], *, force_reopen: bool = False
) -> Tuple[ConnectionWrapper, str]:
    """Return a shared DuckDB connection wrapper and its canonical path."""

    path = normalize_duckdb_url(db_url or "")
    cache_disabled = os.getenv("RESOLVER_DISABLE_CONN_CACHE") == "1"
    if cache_disabled:
        wrapper = _open_new(path)
        wrapper._last_event = "miss"
        logger.debug("DuckDB opened (cache disabled): %s", path)
        log_json(
            _DIAG_LOGGER,
            "db_cache_event",
            cache_event="miss",
            cache_disabled=True,
            path=path,
        )
        return wrapper, path

    with _LOCK:
        cache = _get_cache()
        wrapper = cache.get(path)
        event = "hit"
        if force_reopen or wrapper is None or wrapper._closed or not wrapper._healthcheck():
            event = "reopen" if wrapper is not None else "miss"
            wrapper = _open_new(path)
            cache[path] = wrapper
            logger.debug("DuckDB opened (force=%s): %s", force_reopen, path)
        else:
            logger.debug("DuckDB cache hit: %s", path)
        wrapper._last_event = event
        log_json(
            _DIAG_LOGGER,
            "db_cache_event",
            cache_event=event,
            cache_disabled=False,
            path=path,
        )
        return wrapper, path


def clear_cached_connection(db_url: str) -> None:
    """Evict ``db_url`` from the shared cache."""

    path = normalize_duckdb_url(db_url)
    with _LOCK:
        cache = _get_cache()
        wrapper = cache.pop(path, None)
        if wrapper and not wrapper._closed:
            try:
                wrapper.close()
            except Exception:  # pragma: no cover - best effort cleanup
                pass
        logger.debug("DuckDB cache cleared: %s", path)


def clear_all_cached_connections() -> None:
    """Close and evict all cached DuckDB connections for the current context."""

    to_close: list[ConnectionWrapper] = []
    with _LOCK:
        for cache in _iter_current_caches():
            for path, wrapper in list(cache.items()):
                to_close.append(wrapper)
                cache.pop(path, None)
        _ALL_PATHS.clear()

    for wrapper in to_close:
        try:
            wrapper.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass


atexit.register(clear_all_cached_connections)

