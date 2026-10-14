# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The 2026-10-14 Sibyl defaults: two trials paired on the pack, the
regime-change selection, the cost gate and the carried-forward label."""

from __future__ import annotations

import json
from datetime import date

import pytest

import sibyl.config as sibyl_config
import sibyl.run as sibyl_run
import sibyl.select_questions as sq
from sibyl.labels import carried_forward
from sibyl.select_questions import (
    SELECTION_CONTROL,
    SELECTION_FILL,
    SELECTION_FLOOR,
    SibylQuestion,
    pool_breakdown,
    select_on_rc,
)
from tests.sibyl_test_utils import HS_RUN_ID, Q1, Q2, set_rc_level
from tests.test_sibyl_lanes import FAR, LOW, _forecast_row, _lane_model, run_env  # noqa: F401


def _cq(qid, hz, *, vol=0.5, rc=1, metric=None):
    metric = metric or {"ACE": "FATALITIES", "DR": "PHASE3PLUS_IN_NEED"}.get(hz, "PA")
    return SibylQuestion(question_id=qid, hs_run_id="hs_r", iso3=qid[:3], hazard_code=hz,
                         metric=metric, window_start_date=date(2026, 11, 1),
                         target_month="2027-04", wording="w", volatility_score=vol,
                         triage_score=0.0, rc_level=rc)


# --- the defaults ---------------------------------------------------------------------

def test_the_defaults():
    assert (sibyl_config.K, sibyl_config.K_MAX, sibyl_config.MIN_VALID_TRIALS) == (2, 2, 1)
    assert (sibyl_config.N_QUESTIONS, sibyl_config.N_CONTROL) == (15, 2)
    assert sibyl_config.FLOOR_PER_HAZARD == {"ACE": 4, "DR": 2, "FL": 1, "TC": 0}
    assert sibyl_config.CAP_PER_HAZARD == {"ACE": 7, "DR": 3, "FL": 2, "TC": 1}
    assert sibyl_config.HAZARD_MIN_RC_LEVEL == {"TC": 2}
    assert sibyl_config.CONTROL_CLASSES == (("ACE", "FATALITIES"),)
    assert sibyl_config.RUN_HARD_CAP_USD == 25.0
    assert sibyl_config.BUDGET_USD_PER_QUESTION == 2.5
    assert (sibyl_config.SUBMIT_MIN_DOCS, sibyl_config.MAX_STEPS) == (5, 12)  # owner decision
    assert not hasattr(sibyl_config, "PACK_SHARE")


# --- selection --------------------------------------------------------------------------

def _pool():
    pool = [_cq(f"A{i:02d}", "ACE", vol=0.9 - 0.05 * i, rc=(2 if i < 3 else 1)) for i in range(10)]
    pool += [_cq(f"D{i:02d}", "DR", vol=0.8 - 0.05 * i, rc=2) for i in range(5)]
    pool += [_cq(f"F{i:02d}", "FL", vol=0.7 - 0.05 * i, rc=1) for i in range(4)]
    pool += [_cq("T00", "TC", vol=0.99, rc=1), _cq("T01", "TC", vol=0.3, rc=2)]
    pool += [_cq(f"Z{i:02d}", "ACE", vol=0.95, rc=0) for i in range(4)]  # no RC flag
    return pool


def test_only_flagged_questions_are_picks_and_cyclone_needs_level_two():
    picks = select_on_rc(_pool(), 13)
    ids = {q.question_id for q in picks}
    assert not any(i.startswith("Z") for i in ids)  # RC 0 never a pick
    assert "T00" not in ids and "T01" in ids  # TC at level 1 is not eligible


def test_floors_caps_and_overflow():
    picks = select_on_rc(_pool(), 13)
    by = {}
    for q in picks:
        by[q.hazard_code] = by.get(q.hazard_code, 0) + 1
    assert by == {"ACE": 7, "DR": 3, "FL": 2, "TC": 1}
    # A thin pool: slots others cannot fill go to ACE first, then DR.
    thin = [q for q in _pool() if q.hazard_code in ("ACE", "DR")]
    picks = select_on_rc(thin, 13)
    by = {}
    for q in picks:
        by[q.hazard_code] = by.get(q.hazard_code, 0) + 1
    assert by == {"ACE": 10, "DR": 3}


def test_level_two_and_three_rank_ahead_of_level_one():
    picks = select_on_rc(_pool(), 13)
    ace = [q for q in picks if q.hazard_code == "ACE"]
    assert [q.question_id for q in ace][:3] == ["A00", "A01", "A02"]
    floor = [q for q in picks if q.selection_pass == SELECTION_FLOOR]
    assert {q.hazard_code for q in floor} == {"ACE", "DR", "FL"}  # TC floor is 0


def test_run_order_is_controls_first_and_the_pool_is_logged(monkeypatch, caplog):
    monkeypatch.setattr(sq, "load_candidates", lambda *a, **k: _pool())
    with caplog.at_level("INFO"):
        qs = sq.select_top_questions("hs_r")
    passes = [q.selection_pass for q in qs]
    assert passes[:2] == [SELECTION_CONTROL] * 2
    assert all(q.hazard_code == "ACE" and q.metric == "FATALITIES" for q in qs[:2])
    assert len(qs) == 15
    assert passes.index(SELECTION_FILL) > passes.index(SELECTION_FLOOR)
    assert "eligible pool by hazard and RC level" in caplog.text
    assert pool_breakdown(_pool())["TC"] == {"1": 1, "2": 1}


# --- trials -----------------------------------------------------------------------------

def test_a_pick_runs_two_trials_and_no_extra_when_they_agree(run_env):  # noqa: F811
    calls = []
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=_lane_model({"*": LOW}, calls))
    assert sorted(set(calls)) == ["A", "B"]
    status, k, trials, rule, _checks, _sel = _forecast_row(Q1)
    assert status == "ok" and k == 2 and rule is None
    assert [t["lane"] for t in json.loads(trials)] == ["A", "B"]


def test_disagreement_adds_only_the_reconciler(run_env):  # noqa: F811
    calls = []
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=_lane_model({"B": FAR, "*": LOW}, calls))
    assert sorted(set(calls)) == ["A", "B", "R"]
    _s, _k, trials, rule, checks, _ = _forecast_row(Q1)
    assert rule == "disagreement" and "reconcile" in json.loads(checks)


def test_departure_alone_adds_nothing_at_k_max_two(run_env):  # noqa: F811
    calls = []
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=_lane_model({"*": FAR}, calls))
    assert sorted(set(calls)) == ["A", "B"]


def test_a_control_runs_one_trial_on_the_no_pack_lane(run_env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(sibyl_config, "N_CONTROL", 1)
    set_rc_level("SOM", 0)
    calls = []
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=2, model_call=_lane_model({"*": LOW}, calls))
    _status, k, trials, _rule, _c, sel = _forecast_row(Q2)
    assert sel == SELECTION_CONTROL and k == 1
    assert [t["lane"] for t in json.loads(trials)] == [sibyl_config.NO_PACK_LANE]
    assert json.loads(trials)[0]["with_pack"] is False


# --- the cost gate -----------------------------------------------------------------------

def test_cost_gate_verdicts():
    v = sibyl_run.cost_gate_verdict
    assert v(4.0, 3, stated=1.4) == ("ok", pytest.approx(4.0 / 3 / 1.4))
    assert v(7.0, 3, stated=1.4)[0] == "warn"
    assert v(11.0, 3, stated=1.4)[0] == "stop"
    assert v(1.0, 0, stated=1.4) == (None, None)


def test_the_gate_stops_new_questions(run_env, monkeypatch):  # noqa: F811
    from sibyl.cost import CostTracker

    monkeypatch.setattr(sibyl_config, "COST_GATE_AFTER_QUESTIONS", 1)
    monkeypatch.setattr(sibyl_config, "STATED_COST_PER_QUESTION_USD", 0.005)
    monkeypatch.setattr(sibyl_run, "CostTracker",
                        lambda *a, **k: CostTracker(run_hard_cap_usd=100, budget_usd_per_question=None))
    summary = sibyl_run.run_sibyl(HS_RUN_ID, n_questions=2,
                                  model_call=_lane_model({"*": LOW}, []))
    assert summary["cost_gate_status"] == "stop" and summary["cost_gate_ratio"] > 2.5
    status, *_ = _forecast_row(Q2)
    assert status == "skipped"
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        (reason,) = con.execute("SELECT skip_reason FROM sibyl_forecasts WHERE question_id = ?",
                                [Q2]).fetchone()
        assert reason == sibyl_run.SKIP_REASON_COST_GATE
        assert con.execute("SELECT cost_gate_status FROM sibyl_runs").fetchone()[0] == "stop"
    finally:
        con.close()


def test_the_bundle_line_fails_a_stopped_run():
    from scripts.dump_pythia_debug_bundle import _sibyl_cost_check

    run = {"sibyl_run_id": "s", "n_forecast": 1, "n_selected": 2, "run_cost_usd": 9.0,
           "cost_gate_status": "stop", "cost_gate_ratio": 3.0, "stated_cost_per_question_usd": 1.4}
    assert _sibyl_cost_check(run)["status"] == "FAIL"
    assert _sibyl_cost_check(dict(run, cost_gate_status="warn"))["status"] == "WARN"
    assert _sibyl_cost_check(dict(run, cost_gate_status="ok"))["status"] == "OK"


# --- the carried-forward label -----------------------------------------------------------

def test_the_label_names_the_earlier_run_and_its_date():
    carry = carried_forward(sibyl_run_id="sibyl_1790890510860",
                            sibyl_forecast_run_id="fc_1790831584",
                            current_run_id="fc_1791866322",
                            sibyl_created_at="2026-10-01 21:48:12")
    assert carry["carried_forward"] is True and carry["sibyl_date"] == "2026-10-01"
    assert "sibyl_1790890510860" in carry["label"] and "fc_1791866322" in carry["label"]
    assert carried_forward(sibyl_run_id="s", sibyl_forecast_run_id="fc_1",
                           current_run_id="fc_1") is None
    assert carried_forward(sibyl_run_id="s", sibyl_forecast_run_id=None,
                           current_run_id="fc_1") is None


def test_the_current_run_bundle_labels_an_earlier_sibyl_forecast():
    import duckdb

    from scripts.ai_bundle.build_current_run_bundle import _sibyl_carried_forward

    con = duckdb.connect()
    con.execute("CREATE TABLE sibyl_forecasts (sibyl_run_id TEXT, run_id TEXT, question_id TEXT, "
                "status TEXT, evidence_ok BOOLEAN, created_at TIMESTAMP)")
    con.execute("INSERT INTO sibyl_forecasts VALUES ('sibyl_1790890510860', 'fc_1790831584', "
                "'MEX_FL_PA_2026-11', 'ok', TRUE, TIMESTAMP '2026-10-01 21:48:12')")
    rows = [{"question_id": "MEX_FL_PA_2026-11"}, {"question_id": "OTHER"}]
    out = _sibyl_carried_forward(con, "fc_1791866322", rows, covered=set())
    assert [r["question_id"] for r in out] == ["MEX_FL_PA_2026-11"]
    assert "2026-10-01" in rows[0]["sibyl_carried_forward_label"]
    assert "sibyl_carried_forward_label" not in rows[1]
    # The same run is not carried forward.
    assert _sibyl_carried_forward(con, "fc_1790831584", [{"question_id": "MEX_FL_PA_2026-11"}],
                                  covered=set()) == []


# --- a usage limit mid-run ----------------------------------------------------------

def test_a_usage_limit_stops_new_questions(run_env):  # noqa: F811
    from forecaster import providers
    from sibyl.agent import note_provider_refusal

    limit = ("Anthropic HTTP 400: You have reached your specified API usage limits. "
             "You will regain access on 2026-11-01 at 00:00 UTC.")
    providers.reset_terminal_providers()

    def refused(prompt):
        note_provider_refusal("anthropic", limit)
        return "", {"cost_usd": 0.0}, limit

    try:
        summary = sibyl_run.run_sibyl(HS_RUN_ID, n_questions=2, model_call=refused)
    finally:
        providers.reset_terminal_providers()
    assert summary["provider_refused"] == "usage_limit"
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        (reason,) = con.execute("SELECT skip_reason FROM sibyl_forecasts WHERE question_id = ?",
                                [Q2]).fetchone()
    finally:
        con.close()
    assert reason == sibyl_run.SKIP_REASON_PROVIDER
