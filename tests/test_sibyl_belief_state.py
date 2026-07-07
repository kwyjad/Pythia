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
    initial_belief_from_anchor,
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
    del bad["belief_state"]["quantiles"]["0.95"]
    with pytest.raises(BeliefStateError, match="missing required quantile levels"):
        parse_step_response(json.dumps(bad))


def test_parse_rejects_non_numeric_quantiles():
    bad = json.loads(make_submit_response())
    bad["belief_state"]["quantiles"]["0.5"] = "around a hundred"
    with pytest.raises(BeliefStateError, match="non-numeric"):
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
    bad["belief_state"]["quantiles"] = {
        "0.1": 100, "0.25": 50, "0.5": 200, "0.75": 150,
        "0.9": 500, "0.95": 400, "0.99": 1000,
    }
    decision = parse_step_response(json.dumps(bad))
    assert decision.repaired is True
    values = [decision.belief.quantiles[lv] for lv in QUANTILE_LEVELS]
    assert values == sorted(values)
    # Running-max repair: raised to at least the previous level's value.
    assert decision.belief.quantiles[0.25] == 100
    assert decision.belief.quantiles[0.95] == 500


def test_negative_quantiles_floored_at_zero():
    repaired, was_repaired = enforce_monotone_quantiles({0.1: -5.0, 0.5: 10.0, 0.9: 20.0})
    assert was_repaired is True
    assert repaired[0.1] == 0.0


def test_enforce_monotone_no_op_on_valid_input():
    q = {lv: float(i * 10) for i, lv in enumerate(QUANTILE_LEVELS)}
    repaired, was_repaired = enforce_monotone_quantiles(q)
    assert was_repaired is False
    assert repaired == q


def test_initial_belief_seeds_from_anchor():
    anchor = {0.1: 1.0, 0.25: 2.0, 0.5: 5.0, 0.75: 9.0, 0.9: 20.0, 0.95: 40.0, 0.99: 90.0}
    belief = initial_belief_from_anchor(anchor)
    assert belief.quantiles == anchor
    assert belief.confidence == "low"


def test_initial_belief_without_anchor_is_zero_knowledge():
    belief = initial_belief_from_anchor(None)
    assert all(belief.quantiles[lv] == 0.0 for lv in QUANTILE_LEVELS)
