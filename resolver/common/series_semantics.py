# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Shared helpers for deriving series semantics values."""

from __future__ import annotations

import functools
import importlib.resources as resources
from typing import Optional

import yaml


@functools.lru_cache(maxsize=1)
def _load_config() -> frozenset[str]:
    try:
        config_text = resources.files("resolver.config").joinpath("series_semantics.yml").read_text(
            encoding="utf-8"
        )
    except (FileNotFoundError, ModuleNotFoundError):
        return frozenset()
    data = yaml.safe_load(config_text) or {}
    stock = data.get("stock_metrics") or []
    return frozenset(str(item).strip().lower() for item in stock if str(item).strip())


def compute_series_semantics(metric: Optional[str], existing: Optional[str] = None) -> str:
    """Return the canonical series semantics for a record."""

    if existing is not None:
        text = str(existing).strip()
        if text and text.lower() not in {"none", "nan"}:
            return text

    metric_key = (metric or "").strip().lower()
    stock_metrics = _load_config()
    if metric_key and metric_key in stock_metrics:
        return "stock"
    return ""


__all__ = ["compute_series_semantics"]
