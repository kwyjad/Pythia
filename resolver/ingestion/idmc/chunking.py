"""Date window chunking helpers for IDMC."""
from __future__ import annotations

import datetime as dt
from typing import List, Tuple

__all__ = ["month_ends", "split_by_month"]


def _end_of_month(day: dt.date) -> dt.date:
    if day.month == 12:
        next_month = dt.date(day.year + 1, 1, 1)
    else:
        next_month = dt.date(day.year, day.month + 1, 1)
    return next_month - dt.timedelta(days=1)


def month_ends(start: dt.date, end: dt.date) -> List[dt.date]:
    """Return the list of month-end dates between ``start`` and ``end``."""

    if end < start:
        return []
    cursor = dt.date(start.year, start.month, 1)
    final = []
    while cursor <= end:
        final.append(_end_of_month(cursor))
        if cursor.month == 12:
            cursor = dt.date(cursor.year + 1, 1, 1)
        else:
            cursor = dt.date(cursor.year, cursor.month + 1, 1)
    return [day for day in final if start <= day <= end]


def split_by_month(start: dt.date, end: dt.date) -> List[Tuple[dt.date, dt.date]]:
    """Split a date range into inclusive month spans."""

    if end < start:
        return []
    spans: List[Tuple[dt.date, dt.date]] = []
    cursor = start
    while cursor <= end:
        month_end = _end_of_month(cursor)
        chunk_end = end if month_end > end else month_end
        spans.append((cursor, chunk_end))
        cursor = chunk_end + dt.timedelta(days=1)
    return spans
