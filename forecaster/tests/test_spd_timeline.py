# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from datetime import date

import pytest

duckdb = pytest.importorskip("duckdb")

from forecaster import cli
from forecaster.prompts import build_time_horizon_block


def test_spd_months_match_question_window() -> None:
    window_start = date(2026, 1, 1)
    window_end = date(2026, 6, 30)

    labels = cli._build_month_labels(window_start, horizon_months=6)

    assert labels[1] == "January 2026"
    assert labels[6] == "June 2026"

    block = build_time_horizon_block(
        window_start_date=window_start,
        window_end_date=window_end,
        month_labels=labels,
        hazard_code="FL",
        metric="PA",
        resolution_source="EMDAT",
    )

    assert "`month_1` = January 2026" in block
    assert "`month_6` = June 2026" in block
