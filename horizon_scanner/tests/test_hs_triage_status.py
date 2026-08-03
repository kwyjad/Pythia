# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Guards the country-level triage status roll-up.

Regression cover for the 2026-08-01 production run, where
``hs_triage_coverage__*.csv`` reported ``final_status=degraded`` for 49 of 122
countries and ``HS_TRIAGE_DEGRADED_ISO3S`` named every one of them for rerun —
while all 261 triage LLM calls had succeeded and the DB held zero ``error``
rows. The roll-up filter had drifted out of sync with ``hazards_to_ground``:
it omitted the ACE low-activity exclusion, so the deliberate skip status
``acled_low_activity`` (neither "ok" nor "error") dragged the whole country to
"degraded". Exactly 49 rows carried that status.
"""

from __future__ import annotations

import pytest

from horizon_scanner.triage import (
    _TRIAGE_HAZARDS,
    _scored_hazards,
    combine_hazard_statuses,
)

ALL = ["ACE", "DR", "FL", "TC"]


class TestScoredHazards:
    def test_excludes_ace_when_acled_shows_low_activity(self):
        scored = _scored_hazards(ALL, set(ALL), set(), ace_low_activity=True)
        assert "ACE" not in scored
        assert set(scored) == {"DR", "FL", "TC"}

    def test_excludes_rc_promoted(self):
        scored = _scored_hazards(ALL, set(ALL), {"DR"}, ace_low_activity=False)
        assert "DR" not in scored

    def test_excludes_seasonally_inactive(self):
        scored = _scored_hazards(ALL, {"ACE", "DR"}, set(), ace_low_activity=False)
        assert set(scored) == {"ACE", "DR"} & set(_TRIAGE_HAZARDS)

    def test_keeps_ace_when_activity_is_normal(self):
        scored = _scored_hazards(ALL, set(ALL), set(), ace_low_activity=False)
        assert "ACE" in scored


class TestCombineHazardStatuses:
    @pytest.mark.parametrize(
        "statuses,expected",
        [
            ([], "ok"),
            (["ok"], "ok"),
            (["ok", "ok", "ok"], "ok"),
            (["error"], "failed"),
            (["error", "error"], "failed"),
            (["ok", "error"], "degraded"),
        ],
    )
    def test_roll_up(self, statuses, expected):
        assert combine_hazard_statuses(statuses) == expected


class TestSkipsAreNotFailures:
    """The production bug, expressed directly."""

    def test_ace_low_activity_country_is_ok_not_degraded(self):
        # What the run really looked like for those 49 countries: ACE skipped
        # on purpose, every scored hazard fine.
        merged = {
            "ACE": {"status": "acled_low_activity"},
            "DR": {"status": "ok"},
            "FL": {"status": "ok"},
            "TC": {"status": "ok"},
        }
        scored = _scored_hazards(ALL, set(ALL), set(), ace_low_activity=True)
        status = combine_hazard_statuses([merged[hz]["status"] for hz in scored])
        assert status == "ok", (
            "a deliberate ACE low-activity skip must not read as a triage failure"
        )

    def test_rc_promoted_hazard_is_not_a_failure(self):
        merged = {
            "ACE": {"status": "rc_promoted"},
            "DR": {"status": "ok"},
            "FL": {"status": "ok"},
            "TC": {"status": "ok"},
        }
        scored = _scored_hazards(ALL, set(ALL), {"ACE"}, ace_low_activity=False)
        assert combine_hazard_statuses([merged[hz]["status"] for hz in scored]) == "ok"

    def test_a_real_error_still_degrades_the_country(self):
        merged = {
            "ACE": {"status": "acled_low_activity"},
            "DR": {"status": "error"},
            "FL": {"status": "ok"},
            "TC": {"status": "ok"},
        }
        scored = _scored_hazards(ALL, set(ALL), set(), ace_low_activity=True)
        status = combine_hazard_statuses([merged[hz]["status"] for hz in scored])
        assert status == "degraded"

    def test_a_country_with_nothing_left_to_score_is_ok(self):
        # Every hazard skipped for a legitimate reason -> nothing failed.
        scored = _scored_hazards(ALL, {"ACE"}, set(), ace_low_activity=True)
        assert scored == []
        assert combine_hazard_statuses([]) == "ok"
