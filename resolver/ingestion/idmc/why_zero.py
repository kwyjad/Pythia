"""Helpers for persisting IDMC zero-row diagnostics."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Mapping

from .diagnostics import serialize_http_status_counts

DEFAULT_PATH = "diagnostics/ingestion/idmc/why_zero.json"


def write_why_zero(payload: Dict[str, Any], path: str = DEFAULT_PATH) -> str:
    """Persist the provided diagnostics payload to ``path`` and return it."""

    working = dict(payload)
    if "http_status_counts" in working:
        working["http_status_counts"] = serialize_http_status_counts(
            working.get("http_status_counts")
        )
    fallback_block = working.get("fallback")
    if isinstance(fallback_block, Mapping):
        normalized = dict(fallback_block)
        if "used" not in normalized and "fallback_used" in working:
            normalized["used"] = bool(working.get("fallback_used"))
        normalized.setdefault("resource_url", normalized.get("resource_url"))
        if "rows" not in normalized and isinstance(working.get("rows"), Mapping):
            rows_block = working.get("rows") or {}
            rows_value = rows_block.get("normalized") or rows_block.get("staged")
            if rows_value is not None:
                normalized["rows"] = rows_value
        working["fallback"] = normalized

    dest = pathlib.Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(working, handle, ensure_ascii=False, indent=2)
    return dest.as_posix()
