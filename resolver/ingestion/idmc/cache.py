"""File-based cache utilities for IDMC downloads."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

__all__ = ["CacheEntry", "cache_key", "cache_get", "cache_put"]


@dataclass
class CacheEntry:
    """In-memory representation of cached payload and metadata."""

    body: bytes
    metadata: Dict[str, Any]


def cache_key(url: str, params: Dict[str, Any] | None = None) -> str:
    """Return a stable cache key for the request."""

    payload = {"url": url, "params": params or {}}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _paths(base_dir: str, key: str) -> Tuple[str, str]:
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{key}.bin"), os.path.join(base_dir, f"{key}.meta.json")


def cache_get(base_dir: str, key: str, ttl_seconds: int | None) -> CacheEntry | None:
    """Return cached bytes if present and within the TTL."""

    data_path, meta_path = _paths(base_dir, key)
    if not os.path.exists(data_path):
        return None

    if ttl_seconds is not None and ttl_seconds >= 0:
        mtime = os.path.getmtime(data_path)
        age = time.time() - mtime
        if age > ttl_seconds:
            return None

    with open(data_path, "rb") as handle:
        body = handle.read()

    metadata: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except json.JSONDecodeError:  # pragma: no cover - defensive
            metadata = {}
    return CacheEntry(body=body, metadata=metadata)


def cache_put(base_dir: str, key: str, body: bytes, metadata: Dict[str, Any] | None = None) -> CacheEntry:
    """Persist a cached payload and optional metadata."""

    data_path, meta_path = _paths(base_dir, key)
    with open(data_path, "wb") as handle:
        handle.write(body)
    meta = metadata or {}
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    return CacheEntry(body=body, metadata=meta)
