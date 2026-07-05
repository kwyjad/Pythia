# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Month/window arithmetic helpers for the forecaster (moved verbatim from cli.py)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime, timezone
from typing import Any, Dict, Optional

from pythia.buckets import NUM_HORIZONS


def _coerce_date(val: Any) -> Optional[date]:
    if isinstance(val, date):
        return val
    if isinstance(val, str):
        try:
            return datetime.fromisoformat(val).date()
        except Exception:
            return None
    return None


def _parse_month_key(s: str) -> Optional[date]:
    """
    Parse a month key into a date anchored to the first of the month.

    Supported formats:
      - YYYY-MM
      - YYYY-MM-DD (day component is discarded)
    """
    if not isinstance(s, str):
        return None

    text = s.strip()
    if not text:
        return None

    try:
        if len(text) == 7:
            dt = datetime.strptime(text, "%Y-%m")
            return date(dt.year, dt.month, 1)
        dt = datetime.fromisoformat(text)
        return date(dt.year, dt.month, 1)
    except Exception:
        try:
            parts = text.split("-")
            if len(parts) >= 2:
                year = int(parts[0])
                month = int(parts[1])
                return date(year, month, 1)
        except Exception:
            return None
    return None


def _sanitize_month_series(
    month_to_value: Dict[str, Any],
) -> tuple[Dict[str, Any], list[str], list[str]]:
    """
    Remove entries with month keys later than the current month.

    Returns (cleaned_dict, dropped_future_months, unparseable_month_keys).
    """
    now_month = datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m")
    cleaned: Dict[str, Any] = {}
    dropped: list[str] = []
    unparseable: list[str] = []

    for key in sorted(month_to_value.keys()):
        val = month_to_value[key]
        if str(key).strip() == "":
            unparseable.append(str(key))
            continue
        parsed = _parse_month_key(str(key))
        if parsed is None:
            unparseable.append(str(key))
            continue

        month_key = parsed.strftime("%Y-%m")
        if month_key > now_month:
            dropped.append(str(key))
            continue

        cleaned[str(key)] = val

    return cleaned, dropped, unparseable


def _is_calendar_month_key(key: str) -> bool:
    try:
        import re as _re  # local import to keep global imports minimal

        return bool(_re.match(r"^\d{4}-\d{2}$", str(key).strip()))
    except Exception:
        return False


def _parse_month_offset_key(key: str) -> int | None:
    """
    Parse month offset keys like 'month_0', 'month_1', 'm0', 'm1'.
    Returns the integer offset or None if not recognized.
    """
    if not isinstance(key, str):
        return None
    k = key.strip().lower()
    if k.startswith("month_") and k[6:].isdigit():
        try:
            return int(k[6:])
        except Exception:
            return None
    if k.startswith("m") and k[1:].isdigit():
        try:
            return int(k[1:])
        except Exception:
            return None
    return None


def _add_months(ym: str, offset: int) -> str:
    """Return YYYY-MM shifted by offset months (offset can be negative)."""
    parts = str(ym or "").split("-")
    if len(parts) != 2:
        return ""
    try:
        y = int(parts[0])
        m = int(parts[1])
    except Exception:
        return ""
    total_months = (y * 12 + (m - 1)) + int(offset)
    year = total_months // 12
    month = (total_months % 12) + 1
    return f"{year:04d}-{month:02d}"


def _expected_months(anchor_month: str, n: int = NUM_HORIZONS) -> list[str]:
    """Return the n forecast-window months starting at ``anchor_month``.

    ``anchor_month`` is the FIRST window month (horizon_m=1) — see
    ``_anchor_month_for_question``. It must never be the questions-table
    ``target_month`` (the 6th window month).
    """
    if not anchor_month:
        return []
    return [_add_months(anchor_month, i) for i in range(n)]


def _first_target_month(target_months: Any) -> str | None:
    """Return the first target month string from a string or iterable, if present."""

    if isinstance(target_months, str) and target_months.strip():
        return target_months.strip()

    if isinstance(target_months, (list, tuple)):
        for month_val in target_months:
            if isinstance(month_val, str) and month_val.strip():
                return month_val.strip()

    return None


def _anchor_month_for_question(rec: Mapping[str, Any]) -> str | None:
    """Return the first forecast-window month ('YYYY-MM') for a question row.

    ``window_start_date`` is authoritative: compute_resolutions maps
    horizon_m=1 to the window_start month, so month labels in prompts and
    month-offset expansion must anchor there. ``target_month`` in the
    questions table is the 6th (last) window month, so when window_start
    is missing the anchor is target_month minus 5 months. Anchoring at
    target_month directly shifts every forecast +5 months (the bug that
    affected runs 2026-03-21 → 2026-07-01).
    """
    ws = _coerce_date(rec.get("window_start_date"))
    if ws is not None:
        return f"{ws.year:04d}-{ws.month:02d}"
    tm = _first_target_month(rec.get("target_months") or rec.get("target_month"))
    if tm:
        anchored = _add_months(tm[:7], -5)
        return anchored or None
    return None


def _month_index_for_label(label: str, anchor_month: str | None) -> int | None:
    """Map a forecast month label to its 1-based horizon index (1..6).

    Calendar labels ('YYYY-MM' / 'YYYY-MM-DD') are offset from
    ``anchor_month`` (the first window month, = resolutions horizon_m=1);
    canonical 'month_N' labels map to N directly. Returns None for labels
    that fall outside the 6-month window or cannot be parsed — positional
    enumeration must never be used instead, because a missing or off-window
    label would silently shift every subsequent month against resolutions.
    """
    s = str(label).strip()

    def _cal_index(y: int, m: int) -> int | None:
        if not anchor_month:
            return None
        try:
            ay, am = map(int, anchor_month.split("-"))
        except Exception:
            return None
        idx = (y * 12 + m) - (ay * 12 + am) + 1
        return idx if 1 <= idx <= NUM_HORIZONS else None

    if _is_calendar_month_key(s):
        y, m = map(int, s.split("-"))
        return _cal_index(y, m)

    offset = _parse_month_offset_key(s)
    if offset is not None:
        idx = offset if offset >= 1 else 1  # 'month_0' style → first month
        return idx if 1 <= idx <= NUM_HORIZONS else None

    dt = _parse_month_key(s)
    if dt is not None:
        return _cal_index(dt.year, dt.month)

    return None
