from __future__ import annotations

import pandas as pd


def parse_dates(series: pd.Series, fmt: str | None) -> tuple[pd.Series, int]:
    """Parse ``series`` into UTC timestamps.

    Returns the parsed series and the number of parse errors encountered.
    """

    if fmt:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce", utc=True)
    else:
        parsed = pd.to_datetime(series, errors="coerce", utc=True, infer_datetime_format=True)
    return parsed, int(parsed.isna().sum())


def window_mask(series: pd.Series, start_iso: str, end_iso: str) -> pd.Series:
    """Return a boolean mask for rows whose values fall within the window."""

    start = pd.Timestamp(start_iso).tz_localize(
        "UTC", nonexistent="shift_forward", ambiguous="NaT"
    )
    end = pd.Timestamp(end_iso).tz_localize(
        "UTC", nonexistent="shift_forward", ambiguous="NaT"
    )
    return (series >= start) & (series <= end)
