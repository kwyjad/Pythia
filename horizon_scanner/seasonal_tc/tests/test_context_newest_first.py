# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""Seasonal TC context: newest outlook first, no page furniture (Oct 2026).

The 1 Oct 2026 Atlantic block opened on TSR's December 2025 forecast and
reached the August update last, and the NOAA block carried the press page's
navigation ("Focus areas: Weather Climate Topics: ... Share to Twitter").
"""

from __future__ import annotations

from horizon_scanner.seasonal_tc import compose_country_context
from horizon_scanner.seasonal_tc import noaa_cpc_scraper as noaa


def _tsr(date, kind, ns):
    return {
        "source": "TSR", "basin": "ATL", "forecast_type": kind, "issue_date": date,
        "prompt_context": (
            f"## North Atlantic — 2026 Seasonal Forecast (TSR, {kind}, issued {date})\n"
            f"Forecast: {ns} named storms, 5 hurricanes."
        ),
    }


def test_newest_outlook_first_and_older_ones_in_one_line_each():
    blocks = compose_country_context(
        [_tsr("2025-12-11", "extended_range", 14), _tsr("2026-08-07", "august_update", 10),
         _tsr("2026-05-28", "pre_season", 11)],
        ["ATL"],
    )
    assert len(blocks) == 1
    text = blocks[0]
    assert text.startswith("## North Atlantic — 2026 Seasonal Forecast (TSR, august_update, issued 2026-08-07)")
    assert "Earlier TSR outlooks, superseded by the one above:" in text
    assert "- pre_season, issued 2026-05-28: Forecast: 11 named storms, 5 hurricanes." in text
    assert text.index("2026-05-28") < text.index("2025-12-11")
    assert text.count("## North Atlantic") == 1


PAGE = """NOAA predicts below-normal 2026 Atlantic hurricane season
Early preparation essential to staying safe all season
Focus areas:
Weather
Climate
Topics:
Atlantic hurricane season
Share:
Share to Twitter
May 21, 2026
A NOAA satellite view of a massive Hurricane Erin churning off the U.
NOAA's Climate Prediction Center is forecasting a range of 8 to 14 total named storms.
"""


def test_the_summary_is_one_line_of_the_page():
    f = noaa.extract_atlantic(PAGE)
    assert f.summary == "NOAA predicts below-normal 2026 Atlantic hurricane season"
    ctx = f.to_prompt_context()
    for furniture in ("Focus areas", "Share to Twitter", "Topics:", "satellite view"):
        assert furniture not in ctx
