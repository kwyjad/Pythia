# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import pytest

pytest.importorskip("duckdb")

import forecaster.cli as cli  # type: ignore


def test_truncate_tail_pack_signals_orders_and_caps() -> None:
    signals = [
        "DAMPENER | month_1-2 | DOWN | dampener-1",
        "TRIGGER | month_1-2 | UP | trigger-1",
        "BASELINE | month_1-2 | MIXED | baseline-1",
        "misc-1",
        "TRIGGER | month_1-2 | UP | trigger-2",
        "DAMPENER | month_1-2 | DOWN | dampener-2",
        "BASELINE | month_1-2 | MIXED | baseline-2",
        "misc-2",
        "TRIGGER | month_1-2 | UP | trigger-3",
        "DAMPENER | month_1-2 | DOWN | dampener-3",
        "BASELINE | month_1-2 | MIXED | baseline-3",
        "misc-3",
        "TRIGGER | month_1-2 | UP | trigger-4",
        "DAMPENER | month_1-2 | DOWN | dampener-4",
        "BASELINE | month_1-2 | MIXED | baseline-4",
        "misc-4",
        "",
    ]

    truncated = cli._truncate_tail_pack_signals(signals, 12)

    assert len(truncated) == 12
    assert all(item.strip() for item in truncated)
    assert truncated[0].startswith("TRIGGER")
    assert any(item.startswith("DAMPENER") for item in truncated)
    assert any(item.startswith("BASELINE") for item in truncated)

    trigger_index = max(idx for idx, item in enumerate(truncated) if item.startswith("TRIGGER"))
    dampener_index = min(idx for idx, item in enumerate(truncated) if item.startswith("DAMPENER"))
    baseline_index = min(idx for idx, item in enumerate(truncated) if item.startswith("BASELINE"))

    assert trigger_index < dampener_index
    assert dampener_index < baseline_index
