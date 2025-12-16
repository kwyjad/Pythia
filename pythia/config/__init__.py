# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


def _default_config_path() -> Path:
    env_path = os.getenv("PYTHIA_CONFIG_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "config.yaml"


@lru_cache(maxsize=1)
def load() -> Dict[str, Any]:
    """Load the application configuration from YAML."""

    path = _default_config_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        return {}
    return data
