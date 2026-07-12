# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Regression tests for the seasonal-TC TSR None-tercile crash.

A July-2026 Resolver Update wrote 0 seasonal_tc rows: TSR downloaded and
parsed the current 2026 forecasts, but one PDF reported a partial tercile
set (``tercile_above`` present, ``tercile_near`` / ``tercile_below`` None).
Formatting a None with ``:.0%`` raised ``TypeError``, and because
``discover_and_extract`` only caught ``requests.RequestException`` the error
propagated and discarded every TSR forecast — so ``collect_all`` returned
empty and ``fetch_and_store_seasonal_tc`` returned False.

These tests pin the None-safety of the tercile formatting (no network).
"""

from __future__ import annotations

from horizon_scanner.seasonal_tc.tsr_seasonal_extractor import SeasonalForecast


def test_prompt_context_partial_terciles_does_not_raise():
    # above present, near/below missing — the exact shape that crashed.
    f = SeasonalForecast(
        basin="ATL",
        tercile_above=0.5,
        tercile_near=None,
        tercile_below=None,
    )
    ctx = f.to_prompt_context()  # must not raise
    assert isinstance(ctx, str)
    # A partial tercile set is omitted rather than half-rendered.
    assert "Tercile probabilities" not in ctx


def test_prompt_context_full_terciles_renders():
    f = SeasonalForecast(
        basin="ATL",
        tercile_above=0.32,
        tercile_near=0.49,
        tercile_below=0.19,
    )
    ctx = f.to_prompt_context()
    assert "Tercile probabilities" in ctx
    assert "32% above-normal" in ctx


def test_prompt_context_no_terciles_does_not_raise():
    f = SeasonalForecast(basin="NWP")
    ctx = f.to_prompt_context()
    assert isinstance(ctx, str)
    assert "Tercile probabilities" not in ctx
