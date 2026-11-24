# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Structured logging helpers shared across resolver modules."""
from __future__ import annotations
import logging
import os
from functools import lru_cache
from typing import TYPE_CHECKING, Iterable, Mapping

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

_KNOWN_LEVELS = frozenset(
    {"CRITICAL", "FATAL", "ERROR", "WARNING", "WARN", "INFO", "DEBUG", "NOTSET"}
)


def env_log_level(
    name: str = "LOG_LEVEL",
    default: str = "INFO",
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return the log level an environment variable names, upper-case.

    A variable set from a workflow input is set to the EMPTY STRING on a
    scheduled run, which has no inputs, and ``os.getenv(name, "INFO")``
    returns that empty string rather than the default:
    ``logging.basicConfig(level='')`` then raises ``ValueError: Unknown
    level: ''`` before the script has done anything. That is how both
    Phase 2.6 steps of the 11 Nov 2026 Resolver Update died on their first
    line and reported success. Unset, empty and unknown all read as the
    default; a non-empty unknown value also logs one WARNING, because
    somebody typed it. ``environ`` reads a mapping other than the process
    environment (a subprocess env being assembled).

    Imports nothing beyond the standard library, so a CI script with no
    scientific stack can call it.
    """

    fallback = (default or "INFO").strip().upper() or "INFO"
    if fallback not in _KNOWN_LEVELS:
        fallback = "INFO"
    raw = (os.environ if environ is None else environ).get(name)
    if raw is None:
        return fallback
    value = raw.strip().upper()
    if not value:
        return fallback
    if value not in _KNOWN_LEVELS:
        logging.getLogger(__name__).warning(
            "%s=%r is not a log level; using %s", name, raw, fallback
        )
        return fallback
    return value


def _resolve_level() -> int:
    level_name = env_log_level("RESOLVER_LOG_LEVEL")
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

    import pandas as pd

    if series is None:
        return {}
    if not isinstance(series, pd.Series):
        series = pd.Series(list(series))
    if series.empty:
        return {}
    normalised = series.fillna("").astype(str)
    counts = normalised.value_counts(dropna=False, sort=False)
    return {str(index): int(count) for index, count in counts.items()}


def df_schema(frame: "pd.DataFrame | None") -> dict[str, object]:
    """Return lightweight schema metadata for diagnostics."""

    if frame is None:
        return {"columns": [], "dtypes": {}, "rows": 0}
    return {
        "columns": list(frame.columns),
        "dtypes": {col: str(dtype) for col, dtype in frame.dtypes.items()},
        "rows": int(len(frame)),
    }
