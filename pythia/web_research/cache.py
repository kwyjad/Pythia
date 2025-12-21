# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


_CACHE_PATH = Path(".cache") / "web_research_cache.json"


def _load_cache() -> Dict[str, Any]:
    try:
        if not _CACHE_PATH.exists():
            return {}
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _persist_cache(cache: Dict[str, Any]) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
    except Exception:
        return


def is_enabled() -> bool:
    return os.getenv("PYTHIA_WEB_RESEARCH_CACHE", "1") != "0"


def get(key: str) -> Dict[str, Any] | None:
    if not is_enabled():
        return None
    return _load_cache().get(key)


def set(key: str, value: Dict[str, Any]) -> None:
    if not is_enabled():
        return
    cache = _load_cache()
    cache[key] = value
    _persist_cache(cache)
