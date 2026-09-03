# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The CrisisWatch inject must say when its edition is old.

On 2026-09-01 the June 2026 edition fed an October-to-March forecast, labelled
"(June 2026)" and nothing else, while the 45-day VIEWS flag beside it said
"stale". The formatter now carries an explicit warning for an entry three or
more editions old, and nothing for the normal one-to-two-edition lag.
"""

from __future__ import annotations

from datetime import datetime, timezone

from horizon_scanner.crisiswatch import (
    _STALE_EDITION_MONTHS,
    crisiswatch_edition_age_months,
    format_crisiswatch_for_prompt,
)


def _entry(month: str, year: int) -> dict:
    return {
        "AFG": {
            "country": "Afghanistan",
            "iso3": "AFG",
            "arrow": "deteriorated",
            "alert_type": "conflict_risk",
            "summary": "Kabul and Islamabad traded cross-border attacks.",
            "month": f"{month} {year}",
            "year": year,
        }
    }


def test_edition_age_counts_months():
    today = datetime(2026, 9, 1, tzinfo=timezone.utc)
    assert crisiswatch_edition_age_months(_entry("June", 2026)["AFG"], today=today) == 3
    assert crisiswatch_edition_age_months(_entry("August", 2026)["AFG"], today=today) == 1
    assert crisiswatch_edition_age_months(_entry("December", 2025)["AFG"], today=today) == 9
    assert crisiswatch_edition_age_months({"month": "", "year": 0}, today=today) is None
    assert crisiswatch_edition_age_months(None, today=today) is None


def test_three_month_old_edition_gets_a_warning():
    today = datetime(2026, 9, 1, tzinfo=timezone.utc)
    text = format_crisiswatch_for_prompt("AFG", _entry("June", 2026), today=today)
    assert text is not None
    assert "STALENESS WARNING" in text
    assert "June 2026 edition, 3 months old" in text
    # The original signal is still there — flagged, not dropped.
    assert "cross-border attacks" in text
    assert "CONFLICT RISK" in text


def test_normal_publication_lag_carries_no_warning():
    today = datetime(2026, 9, 1, tzinfo=timezone.utc)
    for month in ("July", "August"):
        text = format_crisiswatch_for_prompt("AFG", _entry(month, 2026), today=today)
        assert text is not None
        assert "STALENESS WARNING" not in text


def test_threshold_is_three_editions():
    assert _STALE_EDITION_MONTHS == 3
