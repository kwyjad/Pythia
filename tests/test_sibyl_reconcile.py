# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The reconciler trial, lane R (Oct 2026, review Part 4).

Routing by rule, the brief's content, de-duplication and cap, brief items
never counted toward either gate, lanes A to E unchanged byte for byte, and
no reconciler for a control.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("duckdb")

import sibyl.agent as sibyl_agent
import sibyl.config as sibyl_config
import sibyl.run as sibyl_run
from sibyl import reconcile
from sibyl.belief_state import MonthBelief, empty_plan, initial_belief
from sibyl.cost import CostTracker
from sibyl.select_questions import SELECTION_CONTROL, SibylQuestion
from sibyl.trials import RULE_DEPARTURE, RULE_DISAGREEMENT
from tests.sibyl_test_utils import (
    HS_RUN_ID,
    Q1,
    Q2,
    make_search_response,
    make_submit_response,
    stub_reference,
)
from tests.test_sibyl_lanes import FAR, LOW, _forecast_row, _lane_model, run_env  # noqa: F401

TODAY = date(2026, 10, 9)


# --- routing ------------------------------------------------------------------------

def test_disagreement_runs_r_and_e_and_records_the_reconcile_check(run_env):  # noqa: F811
    calls = []
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1,
                        model_call=_lane_model({"C": FAR, "*": LOW}, calls))
    assert sorted(calls) == ["A", "B", "C", "E", "R"]
    status, _k, trials, rule, checks, _ = _forecast_row(Q1)
    assert rule == RULE_DISAGREEMENT
    trials = json.loads(trials)
    r = next(t for t in trials if t["lane"] == "R")
    assert r["role"] == RULE_DISAGREEMENT and r["reconcile_brief_chars"] > 0
    rec = json.loads(checks)["reconcile"]
    assert rec["max_pairwise_jsd_before"] > 0.10
    assert rec["dispute"]["medians_by_lane"]["month_1"]["C"] > 1000
    # R forecast LOW, inside the range of the production medians (LOW .. FAR).
    assert rec["r_median_inside_production_range"] is True


def test_departure_keeps_lanes_d_and_e(run_env):  # noqa: F811
    calls = []
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=_lane_model({"*": FAR}, calls))
    assert sorted(calls) == ["A", "B", "C", "D", "E"]
    _status, _k, _trials, rule, checks, _ = _forecast_row(Q1)
    assert rule == RULE_DEPARTURE and "reconcile" not in json.loads(checks)


def test_a_control_never_gets_a_reconciler(run_env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(sibyl_config, "N_CONTROL", 1)
    calls = []
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=2,
                        model_call=_lane_model({"C": FAR, "*": LOW}, calls))
    _s, _k, trials, rule, checks, sel = _forecast_row(Q2)
    assert sel == SELECTION_CONTROL and rule is None
    assert [t["lane"] for t in json.loads(trials)] == ["A"]
    assert calls.count("R") == 1  # Q1's, and only Q1's


# --- the brief ----------------------------------------------------------------------

def _trial(idx, lane, median, ledger, *, reconciliation="near the reference"):
    q = {0.05: median / 5, 0.25: median / 2, 0.5: median, 0.75: median * 2, 0.95: median * 5}
    belief = {"plan": {"resolver": {"status": "done", "finding": f"lane {lane} resolver note"}},
              "baserate_reconciliation": reconciliation}
    t = sibyl_agent.TrialResult(trial_index=idx, perspective="", quantiles={0.5: median},
                                confidence="medium", lane=lane)
    t.month_beliefs = {1: MonthBelief(0.1, dict(q)), 6: MonthBelief(0.1, dict(q))}
    t.belief_trace = [sibyl_agent.TrialStepRecord(step=1, action="submit", action_input="",
                                                  tool_ok=None, belief=belief, repaired=False)]
    t.ledger = ledger
    return t


def _item(url, quote, tier=3, date="2026-09-01", direction="higher"):
    return {"url": url, "quote": quote, "tier": tier, "date": date, "kind": "measurement",
            "direction": direction}


def test_the_brief_states_each_trial_and_the_dispute():
    trials = [
        _trial(0, "A", 10, [_item("u1", "41 killed in August")]),
        _trial(1, "B", 12, [_item("u1", "41 killed in August"), _item("u2", "ceasefire signed")]),
        _trial(2, "C", 5000, [_item("u3", "offensive announced", tier=4)]),
    ]
    brief = reconcile.build_brief(trials, "FATALITIES")
    text = brief.text
    assert "Trial 0, lane A:" in text and "Trial 2, lane C:" in text
    assert "month 1: p_zero 0.10, median" in text
    assert "reconciliation with the reference: near the reference" in text
    assert "plan resolver: lane B resolver note" in text
    assert "month 1 medians by lane: A " in text and "C 4,286" in text
    assert "largest pairwise month-1 Jensen-Shannon divergence" in text
    assert text.count("41 killed in August") == 1  # de-duplicated by URL and quote
    assert '[A/B] 2026-09-01, tier 3' in text
    assert brief.n_items == 3 and brief.n_items_dropped == 0
    assert brief.dispute["max_pairwise_jsd"] > 0.1
    assert len(brief.dispute["buckets"]) == 2


def test_the_cap_drops_whole_items_and_keeps_strong_ones_first():
    weak = [_item(f"w{i}", "x" * 200, tier=5, date=None, direction="neutral") for i in range(30)]
    strong = [_item("s1", "1,200 displaced on 3 September", tier=1)]
    trials = [_trial(0, "A", 10, weak), _trial(1, "B", 12, strong), _trial(2, "C", 14, [])]
    brief = reconcile.build_brief(trials, "FATALITIES", max_chars=3000)
    assert brief.chars <= 3200
    assert "1,200 displaced on 3 September" in brief.text
    assert brief.n_items_dropped > 0
    assert f"({brief.n_items_dropped} ledger item(s) left out for length" in brief.text
    assert '"' + "x" * 200 + '"' in brief.text or brief.n_items_dropped == 30  # never cut short


def test_brief_items_count_toward_neither_gate():
    q = SibylQuestion(question_id=Q1, hs_run_id=HS_RUN_ID, iso3="ETH", hazard_code="ACE",
                      metric="FATALITIES", target_month="2027-01",
                      window_start_date=date(2026, 11, 1), wording="How many?",
                      volatility_score=0.8, triage_score=0.9)
    brief = "Their evidence:\n- [A] read https://a.example: 41 killed"
    prompts = []

    def model(prompt):
        prompts.append(prompt)
        return make_submit_response(), {"cost_usd": 0.01}, ""

    trial = sibyl_agent.run_trial(
        q, stub_reference(), as_of=TODAY, trial_index=3, run_id="sr",
        tracker=CostTracker(run_hard_cap_usd=100),
        forecast_months=["2026-11", "2026-12"] + ["2027-01"] * 4, country_name="Ethiopia",
        model_call=model, lane="R", trial_brief=brief,
    )
    assert brief in prompts[0] and "Lane R, reconcile:" in prompts[0]
    assert (trial.n_search_ok, trial.n_docs_read) == (0, 0)
    assert not trial.evidence_ok
    assert trial.reconcile_brief_chars == len(brief) and trial.dispute_points is not None


# --- lanes A to E unchanged ---------------------------------------------------------

def _old_agent():
    path = Path(__file__).resolve().parents[1] / "docs" / "prompts" / "2026-10-09-2" / "sibyl_agent.py"
    spec = importlib.util.spec_from_file_location("sibyl_agent_before_part4", path)
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = mod  # dataclasses resolve their module by name
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    # The archived copy reads resolver cards relative to itself; point it home.
    mod._CARD_DIR = sibyl_agent._CARD_DIR
    return mod


@pytest.mark.parametrize("lane", ["A", "B", "C", "D", "E"])
def test_prompts_for_lanes_a_to_e_are_byte_identical(lane):
    old = _old_agent()
    q = SibylQuestion(question_id=Q1, hs_run_id=HS_RUN_ID, iso3="ETH", hazard_code="ACE",
                      metric="FATALITIES", target_month="2027-01",
                      window_start_date=date(2026, 11, 1), wording="How many?",
                      volatility_score=0.8, triage_score=0.9)
    ref = stub_reference()
    start = initial_belief(ref.by_month, "FATALITIES")
    start.plan = empty_plan()
    kw = dict(step=2, as_of=TODAY, forecast_months=["2026-11"] * 6,
              transcript_text="=== STEP 1 ===\nx\n", country_name="Ethiopia",
              track_record="t", lessons="")
    new = sibyl_agent.build_step_prompt(q, ref, start, perspective=sibyl_agent.TRIAL_LANES[lane],
                                        **kw)
    before = old.build_step_prompt(q, ref, start, perspective=old.TRIAL_LANES[lane], **kw)
    assert new == before


def test_the_reconciler_lane_text():
    text = sibyl_agent.TRIAL_LANES["R"]
    assert text.startswith("Lane R, reconcile:")
    assert "Do not average the others and do not defer to the majority." in text
    assert sibyl_agent.LANE_IDS == ("A", "B", "C", "D", "E")  # R is outside the round robin
    assert make_search_response()  # helper import kept honest
