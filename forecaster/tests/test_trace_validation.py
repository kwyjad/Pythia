# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for forecaster.trace_validation."""

from __future__ import annotations

import pytest

from forecaster.trace_validation import validate_reasoning_traces


def _make_model_spec(name: str = "test_model"):
    """Minimal stand-in for a ModelSpec with a .name attribute."""

    class _FakeSpec:
        pass

    s = _FakeSpec()
    s.name = name  # type: ignore[attr-defined]
    return s


def _perfect_raw_call() -> dict:
    """A raw_call with a perfect reasoning trace."""
    return {
        "model_spec": _make_model_spec("perfect_model"),
        "text": "",
        "usage": {},
        "error": None,
        "reasoning_trace": {
            "prior": {
                # 7 FATALITIES buckets; mode on 1-<5 (matches base rate avg 3.0)
                "spd": [0.30, 0.35, 0.15, 0.10, 0.05, 0.03, 0.02],
                "rationale": "Based on historical low fatality counts.",
            },
            "updates": [
                {
                    "signal": "Recent ACLED spike",
                    "direction": "UP",
                    "magnitude": "MODERATE",
                    "months_affected": "all",
                    "delta": [-0.10, 0.02, 0.03, 0.02, 0.01, 0.01, 0.01],
                    "post_update_spd": [0.20, 0.37, 0.18, 0.12, 0.06, 0.04, 0.03],
                },
                {
                    "signal": "Peace talks initiated",
                    "direction": "DOWN",
                    "magnitude": "SMALL",
                    "months_affected": "3-6",
                    "delta": [0.04, -0.01, -0.01, -0.01, -0.005, -0.0025, -0.0025],
                    "post_update_spd": [0.24, 0.36, 0.17, 0.11, 0.055, 0.0375, 0.0275],
                },
            ],
            "point_estimate": "~8 fatalities",
            "point_estimate_bucket": 3,
            "rc_assessment": "partially_accepted",
        },
        "human_explanation": "Test explanation.",
    }


def _base_rate_summary_fatalities() -> dict:
    """A conflict_trajectory base rate whose modal bucket is 1-<5 (index 1)."""
    return {
        "type": "conflict_trajectory",
        "fatalities": {"trailing_3m_avg": 3.0},
    }


class TestPerfectTrace:
    def test_quality_near_one(self):
        results = validate_reasoning_traces(
            raw_calls=[_perfect_raw_call()],
            base_rate_summary=_base_rate_summary_fatalities(),
            hazard_code="ACE",
            metric="FATALITIES",
        )
        assert len(results) == 1
        r = results[0]
        assert r["has_trace"] is True
        assert r["trace_quality_score"] >= 0.85


class TestBadDeltaArithmetic:
    def test_bad_deltas_lower_quality(self):
        rc = _perfect_raw_call()
        # Make delta not sum to 0
        rc["reasoning_trace"]["updates"][0]["delta"] = [0.10] * 7
        # Also break post_update_spd consistency
        results = validate_reasoning_traces(
            raw_calls=[rc],
            base_rate_summary=_base_rate_summary_fatalities(),
            hazard_code="ACE",
            metric="FATALITIES",
        )
        assert len(results) == 1
        r = results[0]
        assert r["has_trace"] is True
        assert r["trace_quality_score"] < 0.85


class TestMismatchedPrior:
    def test_prior_mode_off_by_two(self):
        rc = _perfect_raw_call()
        # Set prior mode far above the implied mode (index 1 per base rate)
        rc["reasoning_trace"]["prior"]["spd"] = [0.05, 0.05, 0.05, 0.05, 0.55, 0.15, 0.10]
        results = validate_reasoning_traces(
            raw_calls=[rc],
            base_rate_summary=_base_rate_summary_fatalities(),
            hazard_code="ACE",
            metric="FATALITIES",
        )
        assert len(results) == 1
        r = results[0]
        assert r["has_trace"] is True
        assert r["trace_quality_score"] < 0.8


class TestMissingTrace:
    def test_no_trace_zero_quality(self):
        rc = {
            "model_spec": _make_model_spec("no_trace_model"),
            "text": "",
            "usage": {},
            "error": None,
            "reasoning_trace": None,
            "human_explanation": "",
        }
        results = validate_reasoning_traces(
            raw_calls=[rc],
            base_rate_summary=_base_rate_summary_fatalities(),
            hazard_code="ACE",
            metric="FATALITIES",
        )
        assert len(results) == 1
        r = results[0]
        assert r["has_trace"] is False
        assert r["trace_quality_score"] == 0.0


class TestPartialTrace:
    def test_prior_only_still_valid(self):
        """A trace with prior but no updates should still score > 0."""
        rc = _perfect_raw_call()
        rc["reasoning_trace"]["updates"] = []
        results = validate_reasoning_traces(
            raw_calls=[rc],
            base_rate_summary=_base_rate_summary_fatalities(),
            hazard_code="ACE",
            metric="FATALITIES",
        )
        assert len(results) == 1
        r = results[0]
        assert r["has_trace"] is True
        assert r["trace_quality_score"] > 0.0
