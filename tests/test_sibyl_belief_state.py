# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl belief state: JSON parse/validation and monotone-quantile
enforcement/repair."""

from __future__ import annotations

import json

import pytest

from sibyl.belief_state import (
    BeliefStateError,
    enforce_monotone_quantiles,
    initial_belief,
    parse_step_response,
)
from sibyl.config import QUANTILE_LEVELS
from tests.sibyl_test_utils import make_search_response, make_submit_response


def test_parse_valid_submit_response():
    decision = parse_step_response(make_submit_response())
    assert decision.action == "submit"
    assert decision.belief.confidence == "medium"
    assert set(decision.belief.quantiles) == set(QUANTILE_LEVELS)
    assert decision.repaired is False


def test_parse_valid_search_response_requires_input():
    decision = parse_step_response(make_search_response("Ethiopia conflict"))
    assert decision.action == "brave_search"
    assert decision.action_input == "Ethiopia conflict"


def test_parse_tolerates_code_fences_and_prose():
    wrapped = "Here is my decision:\n```json\n" + make_submit_response() + "\n```\nDone."
    decision = parse_step_response(wrapped)
    assert decision.action == "submit"


def test_parse_rejects_bad_action():
    bad = json.loads(make_submit_response())
    bad["action"] = "google_search"
    with pytest.raises(BeliefStateError, match="invalid action"):
        parse_step_response(json.dumps(bad))


def test_parse_rejects_tool_action_without_input():
    bad = json.loads(make_search_response())
    bad["action_input"] = ""
    with pytest.raises(BeliefStateError, match="non-empty action_input"):
        parse_step_response(json.dumps(bad))


def test_parse_rejects_missing_quantile_levels():
    bad = json.loads(make_submit_response())
    del bad["belief_state"]["month_1"]["quantiles_positive"]["0.95"]
    with pytest.raises(BeliefStateError, match="missing required positive quantile levels"):
        parse_step_response(json.dumps(bad))


def test_parse_rejects_a_missing_horizon():
    bad = json.loads(make_submit_response())
    del bad["belief_state"]["month_6"]
    with pytest.raises(BeliefStateError, match="month_6"):
        parse_step_response(json.dumps(bad))


def test_parse_rejects_non_numeric_quantiles():
    bad = json.loads(make_submit_response())
    bad["belief_state"]["month_1"]["quantiles_positive"]["0.5"] = "around a hundred"
    with pytest.raises(BeliefStateError, match="non-numeric"):
        parse_step_response(json.dumps(bad))


def test_parse_rejects_a_missing_p_zero():
    bad = json.loads(make_submit_response())
    del bad["belief_state"]["month_1"]["p_zero"]
    with pytest.raises(BeliefStateError, match="p_zero"):
        parse_step_response(json.dumps(bad))


def test_parse_rejects_missing_belief_state():
    with pytest.raises(BeliefStateError, match="belief_state"):
        parse_step_response(json.dumps({"action": "submit", "action_input": ""}))


def test_parse_rejects_empty_and_json_free_responses():
    with pytest.raises(BeliefStateError):
        parse_step_response("")
    with pytest.raises(BeliefStateError):
        parse_step_response("I could not decide on an action this step.")


def test_monotone_violation_is_repaired_not_rejected():
    bad = json.loads(make_submit_response())
    bad["belief_state"]["month_1"]["quantiles_positive"] = {
        "0.05": 100, "0.25": 50, "0.5": 200, "0.75": 150, "0.95": 400,
    }
    decision = parse_step_response(json.dumps(bad))
    assert decision.repaired is True
    q = decision.belief.month_1.quantiles_positive
    assert [q[lv] for lv in (0.05, 0.25, 0.5, 0.75, 0.95)] == [100, 100, 200, 200, 400]


def test_out_of_range_values_are_clamped_and_flagged():
    bad = json.loads(make_submit_response())
    bad["belief_state"]["month_6"]["p_zero"] = 1.4
    bad["belief_state"]["month_6"]["quantiles_positive"]["0.05"] = 0
    decision = parse_step_response(json.dumps(bad))
    assert decision.repaired is True
    assert decision.belief.month_6.p_zero == 1.0
    assert decision.belief.month_6.quantiles_positive[0.05] == 1.0


def test_legacy_quantiles_read_month_1_at_the_old_levels():
    decision = parse_step_response(make_submit_response())
    q = decision.belief.quantiles
    assert set(q) == set(QUANTILE_LEVELS)
    # p_zero 0.1: the 0.1 quantile sits at (or just above) zero.
    assert q[0.1] < 1.0
    assert [q[lv] for lv in QUANTILE_LEVELS] == sorted(q[lv] for lv in QUANTILE_LEVELS)


def test_negative_quantiles_floored_at_zero():
    repaired, was_repaired = enforce_monotone_quantiles({0.1: -5.0, 0.5: 10.0, 0.9: 20.0})
    assert was_repaired is True
    assert repaired[0.1] == 0.0


def test_enforce_monotone_no_op_on_valid_input():
    q = {lv: float(i * 10) for i, lv in enumerate(QUANTILE_LEVELS)}
    repaired, was_repaired = enforce_monotone_quantiles(q)
    assert was_repaired is False
    assert repaired == q


def test_initial_belief_seeds_from_the_reference():
    ref = {1: [0.3, 0.1, 0.3, 0.2, 0.05, 0.03, 0.02], 6: [0.2, 0.1, 0.3, 0.2, 0.1, 0.05, 0.05]}
    belief = initial_belief(ref, "FATALITIES")
    assert belief.month_1.p_zero == pytest.approx(0.3)
    assert belief.month_6.p_zero == pytest.approx(0.2)
    assert belief.confidence == "low"
    assert "reference" in belief.baserate_reconciliation


def test_initial_belief_without_reference_is_a_labelled_placeholder():
    belief = initial_belief(None, "FATALITIES")
    assert "placeholder" in belief.baserate_reconciliation
    assert all(v == 1.0 for v in belief.month_1.quantiles_positive.values())
