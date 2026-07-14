# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""ENSO prompt header must not render 'published )' when the regex-scraped
publication date is empty (July 2026 two-country run defect)."""

from __future__ import annotations

from horizon_scanner.enso.enso_module import ENSOForecast


def test_header_without_publication_date():
    ctx = ENSOForecast(current_state="Neutral").to_prompt_context()
    header = ctx.splitlines()[0]
    assert "published )" not in header
    assert header == "## ENSO State and Forecast (IRI/CPC)"


def test_header_with_publication_date():
    ctx = ENSOForecast(
        current_state="Neutral", publication_date="10 July 2026"
    ).to_prompt_context()
    assert "published 10 July 2026" in ctx.splitlines()[0]
