"""Legacy helper exports for CI summary tooling."""
from __future__ import annotations

from typing import Optional


def _fmt_count(value: Optional[int | float]) -> str:
    """Return '-' for None or zero; otherwise stringified value."""

    if value is None:
        return "-"
    try:
        if float(value) == 0:
            return "-"
    except Exception:
        pass
    return str(value)
