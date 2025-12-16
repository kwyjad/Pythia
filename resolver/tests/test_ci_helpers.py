# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for ``resolver.tools.ci_helpers`` used by CI workflows."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from resolver.tools.ci_helpers import (
    MonthlyWindow,
    monthly_snapshot_window,
    previous_month_istanbul,
)


@pytest.mark.parametrize(
    "moment, expected",
    [
        (dt.datetime(2024, 3, 14, 9, 30), "2024-02"),
        (dt.datetime(2024, 1, 1, 0, 1, tzinfo=ZoneInfo("Europe/Istanbul")), "2023-12"),
        (dt.datetime(2024, 4, 30, 22, 15, tzinfo=ZoneInfo("UTC")), "2024-04"),
    ],
)
def test_previous_month_istanbul(moment: dt.datetime, expected: str) -> None:
    """The helper should yield the previous month using Istanbul boundaries."""

    assert previous_month_istanbul(moment) == expected


def test_monthly_snapshot_window_bounds() -> None:
    """The monthly window helper returns ISO bounds for the prior month."""

    now = dt.datetime(2024, 2, 1, 1, 30, tzinfo=ZoneInfo("UTC"))
    window = monthly_snapshot_window(now)
    assert isinstance(window, MonthlyWindow)
    assert window.ym == "2024-01"
    assert window.start_iso == "2024-01-01"
    assert window.end_iso == "2024-01-31"
    assert window.to_env() == {
        "SNAPSHOT_TARGET_YM": "2024-01",
        "RESOLVER_PERIOD": "2024-01",
        "RESOLVER_START_ISO": "2024-01-01",
        "RESOLVER_END_ISO": "2024-01-31",
    }
