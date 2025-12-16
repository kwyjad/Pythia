# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for IDMC month chunking helpers."""
from __future__ import annotations

import datetime as dt

from resolver.ingestion.idmc.chunking import split_by_month


def test_split_by_month_handles_partial_months() -> None:
    start = dt.date(2024, 1, 15)
    end = dt.date(2024, 4, 12)
    chunks = split_by_month(start, end)
    assert chunks == [
        (dt.date(2024, 1, 15), dt.date(2024, 1, 31)),
        (dt.date(2024, 2, 1), dt.date(2024, 2, 29)),
        (dt.date(2024, 3, 1), dt.date(2024, 3, 31)),
        (dt.date(2024, 4, 1), dt.date(2024, 4, 12)),
    ]


def test_split_by_month_empty_when_inverted() -> None:
    start = dt.date(2024, 5, 1)
    end = dt.date(2024, 4, 1)
    assert split_by_month(start, end) == []
