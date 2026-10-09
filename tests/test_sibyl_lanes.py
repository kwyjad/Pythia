# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Part 5 of the Oct 2026 Sibyl work: lanes, parallel and extra trials, the
outlier guard, and the controls.
"""

from __future__ import annotations

import json
import re
import threading

import pytest

pytest.importorskip("duckdb")

import sibyl.agent as sibyl_agent
import sibyl.config as sibyl_config
import sibyl.run as sibyl_run
from sibyl.select_questions import (
    SELECTION_CONTROL,
    SELECTION_FILL,
    SELECTION_FLOOR,
    SibylQuestion,
    draw_controls,
    floor_then_fill,
)
from sibyl.trials import (
    RULE_DEPARTURE,
    RULE_DISAGREEMENT,
    extra_trials_rule,
    max_pairwise_jsd,
    month_median,
    outlier_indices,
    run_trial_batch,
)
from tests.sibyl_test_utils import (
    HS_RUN_ID,
    Q1,
    Q2,
    disable_evidence_gate,
    make_submit_response,
    seed_db,
    stub_reference,
)

SLOTS = ("resolver", "nowcast", "drivers", "calendar", "reversion", "disconfirm")


# --- lanes ------------------------------------------------------------------------

def test_five_lanes_each_fill_every_slot():
    assert sibyl_agent.LANE_IDS == ("A", "B", "C", "D", "E")
    for lane, text in sibyl_agent.TRIAL_LANES.items():
        assert text.startswith(f"Lane {lane},")
        assert "every" in text and "slot" in text
    assert "'resolver'" in sibyl_agent.TRIAL_LANES["A"] and "'nowcast'" in sibyl_agent.TRIAL_LANES["A"]
    assert "'drivers'" in sibyl_agent.TRIAL_LANES["B"] and "'calendar'" in sibyl_agent.TRIAL_LANES["B"]
    assert "'reversion'" in sibyl_agent.TRIAL_LANES["C"] and "'disconfirm'" in sibyl_agent.TRIAL_LANES["C"]
    assert "language" in sibyl_agent.TRIAL_LANES["D"]
    assert "reference" in sibyl_agent.TRIAL_LANES["E"]


def test_lane_follows_the_trial_index():
    assert [sibyl_agent.lane_for_trial(i) for i in range(6)] == ["A", "B", "C", "D", "E", "A"]


def test_a_trial_stores_its_lane_in_perspective_and_shows_it(monkeypatch):
    disable_evidence_gate(monkeypatch)
    from datetime import date

    from sibyl.cost import CostTracker

    prompts = []

    def model(prompt):
        prompts.append(prompt)
        return make_submit_response(), {"cost_usd": 0.0}, ""

    q = SibylQuestion(
        question_id=Q1, hs_run_id=HS_RUN_ID, iso3="ETH", hazard_code="ACE",
        metric="FATALITIES", window_start_date=date(2026, 11, 1), target_month="2027-04",
        wording="How many?", volatility_score=0.5, triage_score=0.0,
    )
    sink = []
    trial = sibyl_agent.run_trial(
        q, stub_reference(), as_of=date(2026, 10, 3), trial_index=0, run_id="r",
        tracker=CostTracker(run_hard_cap_usd=10), forecast_months=["2026-11"] * 6,
        country_name="Ethiopia", model_call=model, lane="D", log_sink=sink,
    )
    assert trial.lane == "D" and trial.perspective == sibyl_agent.TRIAL_LANES["D"]
    assert trial.to_dict()["lane"] == "D" and trial.to_dict()["perspective"].startswith("Lane D,")
    assert "=== YOUR RESEARCH LANE ===\nLane D," in prompts[0]
    # A sink takes the llm_calls rows instead of the database.
    assert sink and sink[0]["call_type"] == "sibyl_trial0_step1"


# --- the extra-trial rules ------------------------------------------------------------

A = [0.6, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01]
B = [0.05, 0.05, 0.1, 0.2, 0.3, 0.2, 0.1]


def test_disagreement_fires_above_the_limit():
    rule, m = extra_trials_rule([A, A, B], A, A, jsd_limit=0.10, departure_limit=0.25)
    assert rule == RULE_DISAGREEMENT
    assert m["max_pairwise_jsd"] == pytest.approx(max_pairwise_jsd([A, A, B]))
    assert m["max_pairwise_jsd"] > 0.10


def test_departure_fires_when_the_trials_agree_far_from_the_reference():
    rule, m = extra_trials_rule([B, B, B], B, A, jsd_limit=0.10, departure_limit=0.25)
    assert rule == RULE_DEPARTURE
    assert m["max_pairwise_jsd"] == pytest.approx(0.0, abs=1e-12)
    assert m["departure_jsd"] > 0.25


def test_no_rule_when_trials_agree_near_the_reference():
    rule, m = extra_trials_rule([A, A], A, A)
    assert rule is None and m["departure_jsd"] == pytest.approx(0.0, abs=1e-12)
    # No reference: only the disagreement test can fire.
    assert extra_trials_rule([B, B], B, None)[0] is None


def test_the_limits_are_read_from_config(monkeypatch):
    monkeypatch.setattr(sibyl_config, "EXTRA_TRIALS_JSD", 10.0)
    monkeypatch.setattr(sibyl_config, "EXTRA_TRIALS_DEPARTURE_JSD", 10.0)
    assert extra_trials_rule([A, B], A, B)[0] is None


# --- the outlier guard -------------------------------------------------------------------

def test_month_median():
    q = {0.05: 2, 0.25: 6, 0.5: 15, 0.75: 60, 0.95: 400}
    assert month_median(0.6, q) == 0.0
    assert month_median(0.0, q) == pytest.approx(15.0)
    # p_zero 0.2 -> the positive curve's 0.375 quantile, between 6 and 15.
    assert 6 < month_median(0.2, q) < 15


def test_a_far_trial_is_dropped():
    assert outlier_indices([10, 12, 15, 50_000]) == [3]
    assert outlier_indices([10, 12, 0]) == []  # log10(11) - log10(1) < 1.5


def test_the_guard_never_leaves_fewer_than_two():
    assert outlier_indices([1, 100_000]) == []
    # Three spread trials: one goes (the farthest from the other two's
    # median, here the smallest), then the guard stops at two.
    assert len(outlier_indices([1, 10_000, 1_000_000_000])) == 1
    assert outlier_indices([10, 12, 50_000]) == [2]
    for meds in ([1, 10**4, 10**8], [5, 5, 10**6, 10**7]):
        assert len(meds) - len(outlier_indices(meds)) >= 2


def test_within_the_limit_nothing_is_dropped():
    assert outlier_indices([10, 100, 300]) == []


# --- running trials --------------------------------------------------------------------------

def test_batch_runs_in_threads_and_writes_only_on_the_main_thread():
    seen_threads = set()
    writes = []

    def run_one(idx, lane, sink):
        seen_threads.add(threading.current_thread().name)
        sink.append({"idx": idx, "lane": lane})
        return (idx, lane)

    def write_log(**kw):
        writes.append((threading.current_thread() is threading.main_thread(), kw["idx"]))

    out = run_trial_batch([(0, "A"), (1, "B"), (2, "C")], run_one, workers=3, write_log=write_log)
    assert out == [(0, "A"), (1, "B"), (2, "C")]
    assert writes == [(True, 0), (True, 1), (True, 2)]


def test_a_raising_trial_does_not_sink_the_batch():
    def run_one(idx, lane, sink):
        if idx == 1:
            raise RuntimeError("boom")
        return idx

    assert run_trial_batch([(0, "A"), (1, "B"), (2, "C")], run_one, workers=2) == [0, None, 2]


# --- end to end -----------------------------------------------------------------------------------

def _lane_model(beliefs: dict, calls: list):
    lock = threading.Lock()

    def model(prompt):
        lane = re.search(r"Lane ([A-ER]),", prompt).group(1)
        with lock:
            calls.append(lane)
        return make_submit_response(*beliefs.get(lane, beliefs["*"])), {"cost_usd": 0.01}, ""

    return model


LOW = ({"p_zero": 0.05, "q": {0.05: 2, 0.25: 5, 0.5: 12, 0.75: 25, 0.95: 60}}, None)
FAR = ({"p_zero": 0.0, "q": {0.05: 20000, 0.25: 40000, 0.5: 60000, 0.75: 80000, 0.95: 120000}}, None)


@pytest.fixture()
def run_env(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)
    disable_evidence_gate(monkeypatch)
    monkeypatch.setattr(sibyl_run, "build_reference", stub_reference)
    rows = []
    monkeypatch.setattr(sibyl_agent, "log_sibyl_call", lambda **kw: rows.append(
        (threading.current_thread() is threading.main_thread(), kw)))
    monkeypatch.setattr(sibyl_config, "N_CONTROL", 0)
    return rows


def _forecast_row(qid):
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        return con.execute(
            "SELECT status, k, trials_json, extra_trials_rule, trial_checks_json, selection_pass "
            "FROM sibyl_forecasts WHERE question_id = ?", [qid],
        ).fetchone()
    finally:
        con.close()


def test_trials_that_agree_with_the_reference_run_no_extras(run_env, monkeypatch):
    calls = []
    monkeypatch.setattr(sibyl_run, "extra_trials_rule", lambda *a, **k: (None, {"max_pairwise_jsd": 0.0}))
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=_lane_model({"*": LOW}, calls))
    assert sorted(calls) == ["A", "B", "C"]
    status, k, trials, rule, checks, _ = _forecast_row(Q1)
    assert status == "ok" and k == 3 and rule is None
    assert [t["lane"] for t in json.loads(trials)] == ["A", "B", "C"]
    # Every llm_calls row is written on the main thread.
    assert run_env and all(main for main, _ in run_env)


def test_disagreement_adds_lanes_r_and_e_and_the_outlier_is_left_out(run_env):
    # Since Oct 2026 (review Part 4) the reconciler, lane R, takes D's place
    # when the extra trials are called for by disagreement.
    calls = []
    model = _lane_model({"C": FAR, "*": LOW}, calls)
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=model)
    assert sorted(calls) == ["A", "B", "C", "E", "R"]
    status, k, trials, rule, checks, _ = _forecast_row(Q1)
    trials = json.loads(trials)
    assert status == "ok" and rule == RULE_DISAGREEMENT
    assert [t["role"] for t in trials] == ["production"] * 3 + [RULE_DISAGREEMENT] * 2
    dropped = [t for t in trials if t["outlier_dropped"]]
    assert [t["lane"] for t in dropped] == ["C"]
    assert k == 4  # pooled from five trials less the outlier
    checks = json.loads(checks)
    assert checks["outliers_dropped"] == [2] and checks["n_trials_run"] == 5
    assert checks["extra_trials_measures"]["max_pairwise_jsd"] > 0.10
    assert checks["reconcile"]["max_pairwise_jsd_before"] > 0.10
    assert [t["lane"] for t in trials][3:] == ["R", "E"]


def test_k_max_bounds_the_extra_trials(run_env, monkeypatch):
    monkeypatch.setattr(sibyl_config, "K_MAX", 4)
    calls = []
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=_lane_model({"C": FAR, "*": LOW}, calls))
    assert sorted(calls) == ["A", "B", "C", "R"]


def test_no_extras_start_once_the_budget_is_spent(run_env, monkeypatch):
    from sibyl.cost import CostTracker

    monkeypatch.setattr(sibyl_run, "CostTracker", lambda *a, **k: CostTracker(run_hard_cap_usd=0.02))
    calls = []
    summary = sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=_lane_model({"C": FAR, "*": LOW}, calls))
    assert sorted(calls) == ["A", "B", "C"]  # the batch already running finished
    assert summary["budget_capped"] is True


# --- controls ---------------------------------------------------------------------------------------

_HM = {"ACE": "FATALITIES", "DR": "PHASE3PLUS_IN_NEED", "FL": "PA", "TC": "PA"}


def _cq(qid, hz, vol=0.0, rc=0, run="hs_r"):
    return SibylQuestion(
        question_id=qid, hs_run_id=run, iso3=qid[:3], hazard_code=hz, metric=_HM[hz],
        window_start_date=None, target_month="", wording="", volatility_score=vol,
        triage_score=0.0, rc_level=rc,
    )


def _pool():
    out = [_cq(f"A{i:02d}", "ACE", vol=0.01 * i, rc=(1 if i % 4 == 0 else 0)) for i in range(12)]
    out += [_cq(f"D{i:02d}", "DR", vol=0.01 * i, rc=(None if i % 2 else 0)) for i in range(8)]
    out += [_cq(f"F{i:02d}", "FL", rc=0) for i in range(5)]
    out += [_cq(f"T{i:02d}", "TC", rc=0) for i in range(5)]
    return out


def test_controls_are_three_conflict_and_two_drought_with_no_rc_flag():
    ctl = draw_controls(_pool(), 5)
    assert [q.hazard_code for q in ctl] == ["ACE", "ACE", "ACE", "DR", "DR"]
    assert all(not q.rc_level for q in ctl)
    assert all(q.selection_pass == SELECTION_CONTROL and q.is_control for q in ctl)


def test_controls_skip_questions_already_chosen():
    pool = _pool()
    first = draw_controls(pool, 5)
    again = draw_controls(_pool(), 5, exclude=[q.question_id for q in first])
    assert not {q.question_id for q in first} & {q.question_id for q in again}


def test_control_draw_is_deterministic_and_blind_to_order_and_volatility():
    import random

    pool = _pool()
    shuffled = list(_pool())
    random.Random(3).shuffle(shuffled)
    a = [q.question_id for q in draw_controls(pool, 5)]
    b = [q.question_id for q in draw_controls(shuffled, 5)]
    assert a == b
    # A different run draws differently.
    other = [q.question_id for q in draw_controls([_cq(q.question_id, q.hazard_code, rc=q.rc_level, run="hs_other") for q in pool], 5)]
    assert other != a


def test_a_short_class_is_made_up_from_the_other():
    pool = [_cq(f"A{i}", "ACE") for i in range(6)] + [_cq("D0", "DR")]
    ctl = draw_controls(pool, 5)
    assert [q.hazard_code for q in ctl] == ["ACE"] * 4 + ["DR"]
    assert len(draw_controls([_cq("A0", "ACE")], 5)) == 1
    assert draw_controls(pool, 0) == []


def test_run_order_is_floor_then_controls_then_fill(tmp_path, monkeypatch):
    from sibyl import select_questions as sq

    pool = _pool()
    monkeypatch.setattr(sq, "load_candidates", lambda *a, **k: [
        _cq(q.question_id, q.hazard_code, vol=q.volatility_score, rc=q.rc_level) for q in pool])
    qs = sq.select_top_questions("hs_r", n=25, n_control=5,
                                 max_per_hazard_overrides={"FL": 3, "TC": 3})
    passes = [q.selection_pass for q in qs]
    assert passes == ([SELECTION_FLOOR] * 12 + [SELECTION_CONTROL] * 5
                      + [SELECTION_FILL] * (len(qs) - 17))
    assert sum(q.hazard_code == "FL" for q in qs if not q.is_control) == 3
    assert sum(q.hazard_code == "TC" for q in qs if not q.is_control) == 3
    assert len({q.question_id for q in qs}) == len(qs)


def test_overrides_cap_the_floor_too():
    pool = [_cq(f"F{i}", "FL", vol=0.5) for i in range(5)]
    assert len(floor_then_fill(pool, 10, min_per_hazard=3, max_per_hazard_overrides={"FL": 2})) == 2


def test_overrides_parse_from_the_environment():
    assert sibyl_config._parse_overrides("FL:3, tc:3,bad,ACE:x") == {"FL": 3, "TC": 3}
    assert sibyl_config.MAX_PER_HAZARD_OVERRIDES == {"FL": 3, "TC": 3}


def test_a_control_runs_one_trial_on_lane_a_and_no_extras(run_env, monkeypatch):
    monkeypatch.setattr(sibyl_config, "N_CONTROL", 1)
    calls = []
    # Q1 (floor) and Q2 (control); the control's lane A disagrees with nothing.
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=2, model_call=_lane_model({"C": FAR, "*": LOW}, calls))
    status, k, trials, rule, _, sel = _forecast_row(Q2)
    assert sel == SELECTION_CONTROL and status == "ok" and k == 1 and rule is None
    assert [t["lane"] for t in json.loads(trials)] == ["A"]
    assert _forecast_row(Q1)[5] == SELECTION_FLOOR
