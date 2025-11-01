"""Utilities for parsing feature-flag style environment variables."""

from __future__ import annotations

import os
from typing import Optional

TRUE_SET = {"1", "true", "t", "yes", "y", "on"}
FALSE_SET = {"0", "false", "f", "no", "n", "off", ""}


def as_bool(value: Optional[str], default: bool = False) -> bool:
    """Interpret *value* as a boolean using common textual conventions."""

    if value is None:
        return default
    text = str(value).strip().lower()
    if text in TRUE_SET:
        return True
    if text in FALSE_SET:
        return False
    return default


def getenv_bool(name: str, default: bool = False) -> bool:
    """Fetch *name* from the environment and parse it using :func:`as_bool`."""

    return as_bool(os.getenv(name), default=default)


__all__ = ["TRUE_SET", "FALSE_SET", "as_bool", "getenv_bool"]
