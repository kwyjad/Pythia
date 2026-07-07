# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl base-rate injection: per-hazard framing, per-calendar-month vs
annualized-mean paths, and the conflict recency path."""

from __future__ import annotations

import pytest

from sibyl.base_rates import (
    anchor_quantiles_from_summary,
    build_framing_notes,
)
from sibyl.config import QUANTILE_LEVELS

FORECAST_KEYS = ["2026-08", "2026-09", "2026-10", "2026-11", "2026-12", "2027-01"]


def _seasonal_summary(months: dict) -> dict:
    return {
        "type": "seasonal_profile",
        "source": "IFRC",
        "data_range": "2015-2026",
        "years_of_data": 11,
        "months": months,
    }


def test_seasonal_per_calendar_month_path():
    months = {
        m: {"min": 0, "max": 50_000, "mean": 8_000, "median": 3_000, "n_observations": 9}
        for m in (8, 9, 10, 11, 12, 1)
    }
    summary = _seasonal_summary(months)
    notes = " ".join(build_framing_notes(summary, FORECAST_KEYS))
    assert "per-calendar-month" in notes
    assert "ANNUALIZED" not in notes
    # Right-skew widening instruction is always present for seasonal data.
    assert "right-skewed" in notes

    anchor = anchor_quantiles_from_summary(summary, FORECAST_KEYS)
    assert anchor is not None
    assert anchor[0.5] == pytest.approx(3_000)  # median of monthly medians
    assert anchor[0.1] == pytest.approx(0)  # min of monthly mins
    assert anchor[0.9] == pytest.approx(50_000)  # max of monthly maxes
    assert anchor[0.99] > anchor[0.95] > anchor[0.9]  # extended right tail


def test_seasonal_annualized_mean_path():
    """No per-month rows for the window months -> flagged as annualized with
    a seasonal-adjustment instruction, anchored on the overall mean."""
    months = {2: {"min": 0, "max": 100, "mean": 60, "median": 40, "n_observations": 4}}
    summary = _seasonal_summary(months)
    notes = " ".join(build_framing_notes(summary, FORECAST_KEYS))
    assert "ANNUALIZED" in notes
    assert "seasonal" in notes.lower()
    assert "adjust the anchor" in notes

    anchor = anchor_quantiles_from_summary(summary, FORECAST_KEYS)
    assert anchor is not None
    assert anchor[0.5] == pytest.approx(60)  # centred on the annualized mean
    assert anchor[0.99] == pytest.approx(600)  # heavy-tail multiplier


def test_conflict_recency_path():
    summary = {
        "type": "conflict_trajectory",
        "fatalities": {"trailing_3m_avg": 120, "last_month": {"ym": "2026-06", "value": 150}},
        "displacements": {},
    }
    notes = " ".join(build_framing_notes(summary, FORECAST_KEYS))
    assert "RECENT TRAJECTORY" in notes
    assert "autocorrelated" in notes
    # Recency framing, never climatology.
    assert "per-calendar-month" not in notes

    anchor = anchor_quantiles_from_summary(summary, FORECAST_KEYS)
    assert anchor is not None
    assert anchor[0.5] == pytest.approx(120)  # centred on the 3-month average
    assert anchor[0.99] == pytest.approx(1200)


def test_conflict_path_falls_back_to_last_month():
    summary = {
        "type": "conflict_trajectory",
        "fatalities": {"trailing_3m_avg": None, "last_month": {"ym": "2026-06", "value": 30}},
    }
    anchor = anchor_quantiles_from_summary(summary, FORECAST_KEYS)
    assert anchor is not None and anchor[0.5] == pytest.approx(30)


def test_fewsnet_path_uses_recent_mean_and_peak():
    summary = {
        "type": "fewsnet_phase3",
        "recent_mean": 2_000_000,
        "recent_max": 3_500_000,
    }
    notes = " ".join(build_framing_notes(summary, FORECAST_KEYS))
    assert "Phase 3+" in notes
    assert "null months mean no assessment" in notes

    anchor = anchor_quantiles_from_summary(summary, FORECAST_KEYS)
    assert anchor is not None
    assert anchor[0.5] == pytest.approx(2_000_000)
    assert anchor[0.9] >= 3_500_000  # peak pulls the upper tail up


def test_no_base_rate_path():
    summary = {"type": "no_base_rate", "note": "nothing available"}
    notes = " ".join(build_framing_notes(summary, FORECAST_KEYS))
    assert "No historical base rate" in notes
    assert anchor_quantiles_from_summary(summary, FORECAST_KEYS) is None


def test_anchor_not_target_framing_is_always_first():
    for summary in (
        _seasonal_summary({}),
        {"type": "conflict_trajectory", "fatalities": {}},
        {"type": "no_base_rate"},
    ):
        notes = build_framing_notes(summary, FORECAST_KEYS)
        assert "never a target" in notes[0]


def test_anchor_quantiles_are_monotone():
    summary = {
        "type": "conflict_trajectory",
        "fatalities": {"trailing_3m_avg": 55},
    }
    anchor = anchor_quantiles_from_summary(summary, FORECAST_KEYS)
    values = [anchor[lv] for lv in QUANTILE_LEVELS]
    assert values == sorted(values)
