"""Utilities for parsing feature-flag style environment variables."""

from __future__ import annotations

import os
from typing import Optional

FALSE_SET = {"0", "false", "no", "", "off"}


def as_bool(value: Optional[str], default: bool = False) -> bool:
    """Interpret *value* as a boolean using common textual conventions.

    Values are considered false if they are "0", "false", "no", "", or "off"
    (case-insensitive). All other non-None values are considered true.
    """
    if value is None:
        return default
    return str(value).strip().lower() not in FALSE_SET


def getenv_bool(name: str, default: bool = False) -> bool:
    """Fetch *name* from the environment and parse it using :func:`as_bool`."""

    return as_bool(os.getenv(name), default=default)


__all__ = ["FALSE_SET", "as_bool", "getenv_bool"]
