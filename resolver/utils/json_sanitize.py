"""Helpers for producing JSON-serialisable payloads from pandas/numpy types."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd


def _datetime_to_iso(value: dt.datetime) -> str:
    """Return an ISO string, truncating to date when time is midnight."""

    if value.tzinfo is not None:
        value = value.astimezone(dt.timezone.utc).replace(tzinfo=None)
    if (
        value.hour == 0
        and value.minute == 0
        and value.second == 0
        and value.microsecond == 0
    ):
        return value.date().isoformat()
    return value.isoformat()


def json_default(obj: Any) -> Any:
    """JSON serializer for objects not supported by the default encoder."""

    if obj is None:
        return None
    if obj is pd.NaT:
        return None
    if isinstance(obj, pd.Timestamp):
        if pd.isna(obj):
            return None
        return _datetime_to_iso(obj.to_pydatetime())
    if isinstance(obj, dt.datetime):
        return _datetime_to_iso(obj)
    if isinstance(obj, dt.date):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        if np.isnat(obj):
            return None
        return _datetime_to_iso(pd.to_datetime(obj).to_pydatetime())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timedelta):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
