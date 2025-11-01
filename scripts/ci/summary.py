"""Legacy helper exports for CI summary tooling."""
from __future__ import annotations

from typing import Optional


def _fmt_count(x: Optional[int | float]) -> str:
    """Dash for None or zero; otherwise str(x)."""

    if x is None:
        return "-"
    try:
        if float(x) == 0:
            return "-"
    except Exception:
        pass
    return str(x)
