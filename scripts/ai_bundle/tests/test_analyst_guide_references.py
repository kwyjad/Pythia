# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""The analyst guide says what the prompt showed, beside how each reference is built.

It used to call ``__ext_climatology`` "the base-rate SPD the forecaster was
shown". For ACE/FATALITIES the prompt showed a six-month trajectory and no
distribution, while climatology is built from thirty-six months.
"""

from scripts.ai_bundle.guides import build_analyst_guide


def _guide() -> str:
    return build_analyst_guide({})


def test_the_guide_no_longer_claims_climatology_is_what_the_prompt_showed():
    text = _guide()
    assert "the base-rate SPD the forecaster was shown" not in text
    assert "It is NOT, in general, what the prompt showed" in text


def test_ace_fatalities_names_both_windows():
    text = _guide()
    row = next(line for line in text.splitlines() if line.startswith("| ACE / FATALITIES |"))
    assert "6-month trajectory" in row
    assert "36 complete months" in row
    assert "prior_anchor_v1" in row


def test_every_reference_forecaster_is_described():
    text = _guide()
    for name in ("__ext_climatology", "__ext_uniform", "__ext_persistence", "__ext_level_volatility"):
        assert f"`{name}`" in text


def test_every_error_attribution_file_is_described():
    text = _guide()
    for name in (
        "headline.json", "trace_stages.csv", "trace_stages_summary.csv", "update_value.csv",
        "update_value_summary.csv", "rc_outcomes.csv", "rc_outcomes_summary.csv",
        "unasked_outcomes.csv", "experiments.csv", "skill_history.csv", "tail_outcomes.csv",
        "binary_reliability.csv", "inject_health.csv", "input_partial_month", "base_rate_shown",
    ):
        assert f"`{name}`" in text, name
    assert "CLAIMED attribution" in text
