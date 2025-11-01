"""Helpers for persisting IDMC zero-row diagnostics."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

DEFAULT_PATH = "diagnostics/ingestion/idmc/why_zero.json"


def write_why_zero(payload: Dict[str, Any], path: str = DEFAULT_PATH) -> str:
    """Persist the provided diagnostics payload to ``path`` and return it."""

    dest = pathlib.Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return dest.as_posix()
