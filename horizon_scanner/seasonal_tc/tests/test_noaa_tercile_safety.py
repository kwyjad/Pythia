# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Regression tests for the NOAA CPC None-tercile crash.

Same failure class as the July-2026 TSR tercile fix: the three probability
fields are parsed by independent regexes, so a press release phrasing that
matches "above-normal" but not near/below leaves those None. Both
``to_prompt_context`` and ``_log_forecast`` used to guard only
``prob_above_normal`` and then format all three with ``:.0%`` — a partial
parse raised TypeError inside ``noaa_process``, and ``collect_noaa``
swallowed it, silently dropping the whole NOAA batch. (No network.)
"""

from __future__ import annotations

from horizon_scanner.seasonal_tc.noaa_cpc_scraper import (
    SeasonalForecast,
    _log_forecast,
)


def _partial() -> SeasonalForecast:
    return SeasonalForecast(
        basin="ATL",
        basin_full="North Atlantic",
        season_year=2026,
        prob_above_normal=0.6,
        prob_near_normal=None,
        prob_below_normal=None,
    )


def test_prompt_context_partial_terciles_does_not_raise():
    ctx = _partial().to_prompt_context()  # must not raise
    assert isinstance(ctx, str)
    # The parsed tercile still renders, without half-formatting the Nones.
    assert "60% above-normal" in ctx
    assert "near-normal" not in ctx


def test_log_forecast_partial_terciles_does_not_raise():
    _log_forecast(_partial())  # must not raise


def test_prompt_context_full_terciles_renders():
    f = SeasonalForecast(
        basin="ATL",
        basin_full="North Atlantic",
        season_year=2026,
        prob_above_normal=0.6,
        prob_near_normal=0.3,
        prob_below_normal=0.1,
    )
    ctx = f.to_prompt_context()
    assert "60% above-normal" in ctx
    assert "30% near-normal" in ctx
    assert "10% below-normal" in ctx


def test_prompt_context_no_terciles_does_not_raise():
    f = SeasonalForecast(basin="ATL", basin_full="North Atlantic", season_year=2026)
    ctx = f.to_prompt_context()
    assert "above-normal" not in ctx
