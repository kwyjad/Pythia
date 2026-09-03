# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The ENSO prompt header names its sources and never renders empty ones.

The original defect: an unguarded f-string rendered "published )" when the
regex-scraped publication date was empty (July 2026 two-country run). The
header now also names the numeric source the phase was computed from, since
that is the source a reader would want to check.
"""

from __future__ import annotations

from horizon_scanner.enso.enso_module import ENSOForecast


def test_header_without_publication_date():
    ctx = ENSOForecast(current_state="Neutral").to_prompt_context()
    header = ctx.splitlines()[0]
    assert "published )" not in header
    assert header == "## ENSO State and Forecast"


def test_header_with_publication_date():
    ctx = ENSOForecast(
        current_state="Neutral", publication_date="10 July 2026"
    ).to_prompt_context()
    assert "IRI outlook published 10 July 2026" in ctx.splitlines()[0]


def test_header_names_the_numeric_source_that_decided_the_phase():
    ctx = ENSOForecast(
        current_state="El Niño",
        strength="strong",
        nino34_source="cpc_oni_ascii",
    ).to_prompt_context()
    header = ctx.splitlines()[0]
    assert "index: cpc_oni_ascii" in header
    assert "published )" not in header


def test_the_state_line_carries_the_strength_band_and_the_oni():
    """"El Niño" alone under-describes what the Pacific is doing."""

    ctx = ENSOForecast(
        current_state="El Niño",
        strength="very strong",
        oni=2.1,
        observation_date="2026-08-19",
    ).to_prompt_context()
    assert "El Niño, very strong" in ctx
    assert "+2.10" in ctx
    assert "Observed 2026-08-19" in ctx
