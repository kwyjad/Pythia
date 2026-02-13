# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for RC scoring calibration improvements.

Covers:
- Updated L0/L1 thresholds (likelihood >= 0.45, score >= 0.25)
- L2/L3 thresholds (unchanged)
- Distribution sanity check function
- Prompt contains calibration anchoring text
"""
from __future__ import annotations

import os

import pytest

from horizon_scanner.regime_change import (
    check_rc_distribution,
    compute_level,
    compute_score,
    should_force_full_spd,
)


# ---- Score computation (unchanged logic, sanity check) ----


class TestComputeScore:
    def test_basic(self):
        assert compute_score(0.5, 0.6) == pytest.approx(0.30)

    def test_none_likelihood(self):
        assert compute_score(None, 0.5) == 0.0

    def test_none_magnitude(self):
        assert compute_score(0.5, None) == 0.0

    def test_both_none(self):
        assert compute_score(None, None) == 0.0

    def test_clamp_high(self):
        assert compute_score(1.0, 1.0) == 1.0

    def test_clamp_low(self):
        assert compute_score(0.0, 0.0) == 0.0


# ---- Level thresholds (updated L0/L1 to 0.45/0.25) ----


class TestComputeLevel:
    """Verify the new default thresholds: L0/L1 at 0.45/0.25."""

    def test_level_0_low_likelihood(self):
        # likelihood < 0.45 => L0 regardless of score
        assert compute_level(0.30, 0.80, 0.24) == 0

    def test_level_0_low_score(self):
        # score < 0.25 => L0
        assert compute_level(0.44, 0.50, 0.22) == 0

    def test_level_0_borderline_below(self):
        # likelihood = 0.44 (just below 0.45) => L0
        assert compute_level(0.44, 0.60, 0.264) == 0

    def test_level_1_at_threshold(self):
        # likelihood = 0.45, score = 0.25 => L1
        assert compute_level(0.45, 0.56, 0.252) == 1

    def test_level_1_above_threshold(self):
        # likelihood = 0.50, score = 0.30 => L1
        assert compute_level(0.50, 0.60, 0.30) == 1

    def test_level_2_at_threshold(self):
        # likelihood >= 0.60 AND magnitude >= 0.50 => L2
        assert compute_level(0.60, 0.50, 0.30) == 2

    def test_level_2_above(self):
        assert compute_level(0.70, 0.55, 0.385) == 2

    def test_level_3_at_threshold(self):
        # likelihood >= 0.75 AND magnitude >= 0.60 => L3
        assert compute_level(0.75, 0.60, 0.45) == 3

    def test_level_3_high(self):
        assert compute_level(0.90, 0.80, 0.72) == 3

    def test_old_l1_now_l0(self):
        """Values that used to qualify for L1 (likelihood=0.35, score=0.20)
        should now be L0 with the raised thresholds."""
        # Old threshold: likelihood >= 0.35 AND score >= 0.20 => L1
        # New threshold: likelihood >= 0.45 AND score >= 0.25 => L1
        # This should now be L0
        assert compute_level(0.35, 0.60, 0.21) == 0

    def test_old_l1_borderline_now_l0(self):
        """likelihood=0.40 was above old 0.35 threshold, below new 0.45."""
        assert compute_level(0.40, 0.60, 0.24) == 0

    def test_env_override(self, monkeypatch):
        """Verify env vars still override thresholds."""
        monkeypatch.setenv("PYTHIA_HS_RC_LEVEL1_LIKELIHOOD", "0.30")
        monkeypatch.setenv("PYTHIA_HS_RC_LEVEL1_SCORE", "0.15")
        monkeypatch.setenv("PYTHIA_HS_RC_LEVEL0_LIKELIHOOD", "0.30")
        monkeypatch.setenv("PYTHIA_HS_RC_LEVEL0_SCORE", "0.15")
        # With lowered thresholds, this should be L1
        assert compute_level(0.35, 0.50, 0.175) == 1


# ---- Distribution sanity check ----


class TestCheckRcDistribution:
    def test_empty(self):
        result = check_rc_distribution([])
        assert result["total"] == 0
        assert result["warnings"] == []

    def test_healthy_distribution(self):
        """80% L0, 12% L1, 5% L2, 3% L3 => no warnings."""
        levels = (
            [0] * 80
            + [1] * 12
            + [2] * 5
            + [3] * 3
        )
        result = check_rc_distribution(levels, run_id="test_healthy")
        assert result["total"] == 100
        assert result["counts"] == {0: 80, 1: 12, 2: 5, 3: 3}
        assert result["warnings"] == []

    def test_l1_warning(self):
        """30% L1 exceeds 25% threshold."""
        levels = [0] * 70 + [1] * 30
        result = check_rc_distribution(levels, run_id="test_l1_warn")
        assert len(result["warnings"]) == 1
        assert "L1" in result["warnings"][0]

    def test_l2_warning(self):
        """20% L2 exceeds 15% threshold."""
        levels = [0] * 70 + [1] * 10 + [2] * 20
        result = check_rc_distribution(levels, run_id="test_l2_warn")
        assert any("L2" in w for w in result["warnings"])

    def test_l3_warning(self):
        """15% L3 exceeds 8% threshold."""
        levels = [0] * 70 + [1] * 10 + [2] * 5 + [3] * 15
        result = check_rc_distribution(levels, run_id="test_l3_warn")
        assert any("L3" in w for w in result["warnings"])

    def test_multiple_warnings(self):
        """Both L2 and L3 exceed thresholds."""
        levels = [0] * 50 + [1] * 10 + [2] * 20 + [3] * 20
        result = check_rc_distribution(levels, run_id="test_multi_warn")
        assert len(result["warnings"]) >= 2

    def test_fractions_computed(self):
        levels = [0] * 8 + [1] * 2
        result = check_rc_distribution(levels, run_id="test_fracs")
        assert result["fractions"][0] == pytest.approx(0.8)
        assert result["fractions"][1] == pytest.approx(0.2)

    def test_realistic_overinflated_run(self):
        """Simulate the problematic distribution from the last run:
        105 countries, ~38 L1, ~34 L2, ~28 L3 = 100 flagged.
        With 6 hazards each = 630 total (assuming 105 countries).
        Simplified: 38 L1, 34 L2, 28 L3 at country-worst level,
        but per-hazard many more would be flagged."""
        # Simulating ~720 assessments (120 countries x 6 hazards)
        # with the problematic distribution
        levels = [0] * 90 + [1] * 228 + [2] * 204 + [3] * 198
        result = check_rc_distribution(levels, run_id="test_overinflated")
        # All three thresholds should be violated
        assert len(result["warnings"]) == 3


# ---- Prompt calibration text ----


class TestPromptCalibration:
    def test_prompt_contains_calibration_section(self):
        from horizon_scanner.prompts import build_hs_triage_prompt

        prompt = build_hs_triage_prompt(
            country_name="Testland",
            iso3="TST",
            hazard_catalog={"ACE": "Armed Conflict Escalation"},
            resolver_features={},
        )
        # B1: Calibration anchoring
        assert "REGIME CHANGE CALIBRATION" in prompt
        assert "departure from the established pattern" in prompt.lower() or \
               "DEPARTURE FROM" in prompt

        # B2: Distinction from triage_score
        assert "triage_score" in prompt
        assert "overall risk level" in prompt.lower() or \
               "OVERALL risk level" in prompt

        # B3: Distribution guidance
        assert "80%" in prompt
        assert "likelihood" in prompt.lower()

    def test_prompt_default_values_guidance(self):
        from horizon_scanner.prompts import build_hs_triage_prompt

        prompt = build_hs_triage_prompt(
            country_name="Testland",
            iso3="TST",
            hazard_catalog={"ACE": "Armed Conflict Escalation"},
            resolver_features={},
        )
        # Check that the prompt guides toward low default values
        assert "0.05" in prompt  # default likelihood suggestion
        assert "Do NOT assign likelihood > 0.10 unless" in prompt
