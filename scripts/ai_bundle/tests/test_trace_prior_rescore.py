# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The scored bundle's trace prior is measured against the base rate SHOWN.

Before Oct 2026 the prior component was a constant 0.7, because the bundle
called trace_validation with no base-rate summary; 40% of every
trace_quality_score was that constant.
"""

from __future__ import annotations

from scripts.ai_bundle import build_scored_forecast_bundle as b


def _member(prior):
    return {
        "model_name": "m",
        "reasoning_trace": {"prior": {"spd": prior}},
        "trace_quality": {
            "has_trace": True,
            "prior_quality": {"score": 0.7, "detail": "no base rate to compare"},
            "delta_arithmetic": {"score": 1.0},
            "magnitude_consistency": {"score": 1.0},
            "trace_quality_score": 0.88,
        },
    }


def _record(shown, prior):
    return {"base_rate_shown": shown, "members": [_member(prior)]}


SHOWN_LV = {
    "level_volatility": {"shown": True, "spd_by_horizon": {"1": [0.7, 0.2, 0.05, 0.05, 0, 0, 0]}},
    "anchor": {"available": True, "probs": [0, 0, 0, 0, 0, 0.5, 0.5], "source": "x"},
}


def test_a_prior_far_from_the_shown_base_rate_scores_low():
    rec = _record(SHOWN_LV, [0.0, 0.0, 0.0, 0.1, 0.8, 0.1, 0.0])
    b.rescore_trace_prior(rec)
    tq = rec["members"][0]["trace_quality"]
    assert tq["prior_quality"]["score"] == 0.3
    assert tq["prior_quality"]["shown_source"] == "level_volatility_h1"
    assert tq["trace_quality_score"] == round(0.4 * 0.3 + 0.4 + 0.2, 4)


def test_a_prior_on_the_shown_mode_scores_one():
    rec = _record(SHOWN_LV, [0.6, 0.3, 0.1, 0, 0, 0, 0])
    b.rescore_trace_prior(rec)
    assert rec["members"][0]["trace_quality"]["prior_quality"]["score"] == 1.0


def test_the_anchor_is_used_when_no_level_vector_was_shown():
    shown = {"level_volatility": {"shown": False},
             "anchor": {"available": True, "probs": [0.1, 0.6, 0.1, 0.1, 0.05, 0.05], "source": "IDMC"}}
    rec = _record(shown, [0.1, 0.2, 0.5, 0.1, 0.05, 0.05])
    b.rescore_trace_prior(rec)
    pq = rec["members"][0]["trace_quality"]["prior_quality"]
    assert pq["score"] == 0.7 and pq["shown_source"] == "anchor:IDMC"


def test_no_shown_distribution_is_never_a_constant():
    rec = _record({"anchor": {"available": False}}, [0.5, 0.5, 0, 0, 0, 0])
    b.rescore_trace_prior(rec)
    tq = rec["members"][0]["trace_quality"]
    assert tq["prior_quality"]["score"] is None
    assert tq["prior_quality"]["compared"] is False
    assert tq["trace_quality_basis"] == "delta_and_magnitude_only"
    assert tq["trace_quality_score"] == 1.0
