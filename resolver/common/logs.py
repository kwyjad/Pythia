"""Structured logging helpers shared across resolver modules."""
from __future__ import annotations
import logging
import os
from functools import lru_cache
from typing import Iterable

import pandas as pd

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _resolve_level() -> int:
    level_name = os.getenv("RESOLVER_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


@lru_cache(maxsize=None)
def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with a consistent formatter.

    The first call configures the logger and caches it so repeated invocations
    reuse the same handler without duplicating output.
    """

    logger = logging.getLogger(name)
    logger.setLevel(_resolve_level())
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def configure_root_logger(*, level: str | int) -> None:
    """Configure the root logger with the shared formatter and level."""

    resolved_level = getattr(logging, str(level).upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(resolved_level)
    if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)


def dict_counts(series: Iterable | None) -> dict[str, int]:
    """Return a stable mapping of value counts for diagnostic logging."""

    if series is None:
        return {}
    if not isinstance(series, pd.Series):
        series = pd.Series(list(series))
    if series.empty:
        return {}
    normalised = series.fillna("").astype(str)
    counts = normalised.value_counts(dropna=False, sort=False)
    return {str(index): int(count) for index, count in counts.items()}


def df_schema(frame: pd.DataFrame | None) -> dict[str, object]:
    """Return lightweight schema metadata for diagnostics."""

    if frame is None:
        return {"columns": [], "dtypes": {}, "rows": 0}
    return {
        "columns": list(frame.columns),
        "dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
        "rows": int(len(frame)),
    }
