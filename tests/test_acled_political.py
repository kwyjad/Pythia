# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the ACLED political events connector."""

from __future__ import annotations

import os

import pytest

from pythia.acled_political import (
    _apply_significance_filter,
    _safe_int,
    format_political_events_for_prompt,
    format_political_events_for_spd,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_event(
    *,
    event_type: str = "Strategic developments",
    sub_event_type: str = "Other",
    fatalities: int = 0,
    event_date: str = "2026-02-15",
    notes: str = "Test event",
    actor1: str = "",
    actor2: str = "",
    location: str = "",
    admin1: str = "",
) -> dict:
    return {
        "event_id_cnty": f"EVT-{id(event_type)}-{fatalities}",
        "event_date": event_date,
        "event_type": event_type,
        "sub_event_type": sub_event_type,
        "actor1": actor1,
        "actor2": actor2,
        "admin1": admin1,
        "location": location,
        "notes": notes,
        "fatalities": fatalities,
        "source": "Test",
    }


# ---------------------------------------------------------------------------
# Significance filtering
# ---------------------------------------------------------------------------


class TestSignificanceFilter:
    """Tests for _apply_significance_filter."""

    def test_always_includes_peace_agreement(self):
        events = [_make_event(sub_event_type="Peace agreement")]
        result = _apply_significance_filter(events)
        assert len(result) == 1

    def test_always_includes_arrests(self):
        events = [_make_event(sub_event_type="Arrests")]
        result = _apply_significance_filter(events)
        assert len(result) == 1

    def test_always_includes_change_to_group(self):
        events = [_make_event(sub_event_type="Change to group/activity")]
        result = _apply_significance_filter(events)
        assert len(result) == 1

    def test_always_includes_fatal_events(self):
        events = [_make_event(
            event_type="Riots",
            sub_event_type="Violent demonstration",
            fatalities=3,
        )]
        result = _apply_significance_filter(events)
        assert len(result) == 1

    def test_excludes_single_protest(self):
        """A single non-fatal protest should be excluded."""
        events = [_make_event(
            event_type="Protests",
            sub_event_type="Peaceful protest",
            fatalities=0,
        )]
        result = _apply_significance_filter(events)
        assert len(result) == 0

    def test_includes_protest_wave(self):
        """5+ protests in the period should all be included."""
        events = [
            _make_event(
                event_type="Protests",
                sub_event_type="Peaceful protest",
                event_date=f"2026-02-{10 + i:02d}",
            )
            for i in range(6)
        ]
        result = _apply_significance_filter(events)
        assert len(result) == 6

    def test_excludes_nonfatal_riot(self):
        """A riot with 0 fatalities should be excluded."""
        events = [_make_event(
            event_type="Riots",
            sub_event_type="Violent demonstration",
            fatalities=0,
        )]
        result = _apply_significance_filter(events)
        assert len(result) == 0

    def test_mixed_events(self):
        """Verify correct filtering of a mixed batch."""
        events = [
            _make_event(sub_event_type="Peace agreement"),
            _make_event(
                event_type="Protests",
                sub_event_type="Peaceful protest",
                fatalities=0,
            ),
            _make_event(
                event_type="Riots",
                sub_event_type="Violent demonstration",
                fatalities=5,
            ),
        ]
        result = _apply_significance_filter(events)
        # Peace agreement: yes (always-include sub-type)
        # Single protest: no (< 5 protests, 0 fatalities)
        # Fatal riot: yes (fatalities > 0)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Prompt formatter
# ---------------------------------------------------------------------------


class TestFormatForPrompt:
    """Tests for format_political_events_for_prompt."""

    def test_returns_empty_for_none(self):
        assert format_political_events_for_prompt(None) == ""

    def test_returns_empty_for_empty_list(self):
        assert format_political_events_for_prompt([]) == ""

    def test_contains_header(self):
        events = [
            {
                "event_id": "E1",
                "event_date": "2026-02-15",
                "event_type": "Strategic developments",
                "sub_event_type": "Peace agreement",
                "actor1": "Gov",
                "actor2": "Rebels",
                "admin1": "Khartoum",
                "location": "Khartoum",
                "notes_excerpt": "Peace talks concluded.",
                "fatalities": 0,
            },
        ]
        result = format_political_events_for_prompt(events, iso3="SDN")
        assert "RECENT POLITICAL EVENTS" in result
        assert "SDN" in result
        assert "Strategic developments:" in result

    def test_groups_events_by_type(self):
        events = [
            {
                "event_id": "E1",
                "event_date": "2026-02-15",
                "event_type": "Strategic developments",
                "sub_event_type": "Arrests",
                "actor1": "",
                "actor2": "",
                "admin1": "",
                "location": "",
                "notes_excerpt": "Key figure arrested.",
                "fatalities": 0,
            },
            {
                "event_id": "E2",
                "event_date": "2026-02-14",
                "event_type": "Protests",
                "sub_event_type": "Peaceful protest",
                "actor1": "",
                "actor2": "",
                "admin1": "",
                "location": "Cairo",
                "notes_excerpt": "Demonstration in Cairo.",
                "fatalities": 0,
            },
        ]
        result = format_political_events_for_prompt(events)
        assert "Strategic developments:" in result
        assert "Protests/Riots:" in result

    def test_truncates_at_15_events(self):
        events = [
            {
                "event_id": f"E{i}",
                "event_date": f"2026-02-{i + 1:02d}",
                "event_type": "Strategic developments",
                "sub_event_type": "Other",
                "actor1": "",
                "actor2": "",
                "admin1": "",
                "location": "",
                "notes_excerpt": f"Event {i}.",
                "fatalities": 0,
            }
            for i in range(20)
        ]
        result = format_political_events_for_prompt(events)
        assert "5 additional events not shown" in result


# ---------------------------------------------------------------------------
# SPD formatter
# ---------------------------------------------------------------------------


class TestFormatForSpd:
    """Tests for format_political_events_for_spd."""

    def test_returns_empty_for_none(self):
        assert format_political_events_for_spd(None) == ""

    def test_returns_empty_for_empty_list(self):
        assert format_political_events_for_spd([]) == ""

    def test_max_10_events(self):
        events = [
            {
                "event_id": f"E{i}",
                "event_date": f"2026-02-{i + 1:02d}",
                "event_type": "Strategic developments",
                "sub_event_type": "Other",
                "actor1": "",
                "actor2": "",
                "admin1": "",
                "location": "",
                "notes_excerpt": f"Event {i}.",
                "fatalities": 0,
            }
            for i in range(15)
        ]
        result = format_political_events_for_spd(events, iso3="SDN")
        assert "KEY POLITICAL EVENTS" in result
        # Header + 10 event lines
        lines = [l for l in result.strip().split("\n") if l.startswith("- ")]
        assert len(lines) == 10

    def test_includes_fatalities(self):
        events = [
            {
                "event_id": "E1",
                "event_date": "2026-02-15",
                "event_type": "Riots",
                "sub_event_type": "Violent demonstration",
                "actor1": "",
                "actor2": "",
                "admin1": "",
                "location": "",
                "notes_excerpt": "Riot event.",
                "fatalities": 12,
            },
        ]
        result = format_political_events_for_spd(events)
        assert "[12 killed]" in result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestSafeInt:
    def test_none(self):
        assert _safe_int(None) == 0

    def test_string(self):
        assert _safe_int("42") == 42

    def test_invalid(self):
        assert _safe_int("abc") == 0


# ---------------------------------------------------------------------------
# Live integration (conditional)
# ---------------------------------------------------------------------------

_has_acled_creds = bool(
    os.environ.get("ACLED_ACCESS_TOKEN")
    or os.environ.get("ACLED_USERNAME")
)


@pytest.mark.allow_network
@pytest.mark.skipif(not _has_acled_creds, reason="ACLED credentials not set")
def test_live_fetch_sdn():
    """Smoke test: fetch political events for Sudan."""
    from pythia.acled_political import fetch_acled_political_events

    events = fetch_acled_political_events("SDN", days_back=60)
    assert isinstance(events, list)
    if events:
        ev = events[0]
        assert "event_id" in ev
        assert "event_date" in ev
        assert "event_type" in ev
        assert "sub_event_type" in ev
        assert "notes_excerpt" in ev
        assert isinstance(ev["fatalities"], int)
