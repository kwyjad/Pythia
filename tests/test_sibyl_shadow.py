# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Part 7 of the Oct 2026 Sibyl work: the GPT Sol shadow arm.

Both providers are mocked throughout. The tests pin: when the arm runs and
why not (the missing key above all); that the shadow trial runs after every
question's production trials and is skipped first near the caps; that the
shadow series replaces the Claude lane C trial and never reaches a forecast
table; the evidence gate on the shadow trial; its cost kind; and the scores
and the paired comparison read back from them.
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from typing import List

import pytest

from pythia.web_research.types import EvidencePack, EvidenceSource

import sibyl.agent as sibyl_agent
import sibyl.config as sibyl_config
import sibyl.run as sibyl_run
import sibyl.tools as sibyl_tools
from sibyl import score_variants as sv
from sibyl import shadow as sh
from tests.sibyl_test_utils import (
    HS_RUN_ID,
    Q1,
    disable_submit_gate,
    make_search_response,
    make_submit_response,
    seed_db,
    stub_reference,
)

pytestmark = pytest.mark.db

SOL_M1 = {"p_zero": 0.05, "q": {0.05: 20, 0.25: 60, 0.5: 120, 0.75: 300, 0.95: 900}}
SOL_M6 = {"p_zero": 0.05, "q": {0.05: 20, 0.25: 60, 0.5: 120, 0.75: 300, 0.95: 900}}


# --- 1. whether the arm runs ---------------------------------------------------------

def _fake(prompt):
    return "", {}, ""


@pytest.mark.parametrize("env, kwargs, expect", [
    ({"SHADOW_MODEL": "off"}, {}, "off"),
    ({}, {"backtest": True}, "backtest"),
    ({}, {"today": date(2027, 5, 1)}, "expired"),
    ({"SHADOW_UNTIL": "off"}, {}, "off"),
    ({"SHADOW_MODEL": "no_such_alias"}, {}, "unknown_model"),
    ({"SHADOW_MODEL": "anthropic:claude-opus-5-5"}, {}, "unsupported_provider"),
])
def test_the_arm_states_why_it_does_not_run(monkeypatch, env, kwargs, expect):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    for k, v in env.items():
        monkeypatch.setattr(sibyl_config, k, v)
    kwargs.setdefault("today", date(2026, 10, 3))
    kwargs.setdefault("backtest", False)
    assert sh.shadow_setup(**kwargs).status == expect


def test_the_last_month_still_runs_and_the_alias_resolves_to_sol(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = sh.shadow_setup(today=date(2027, 4, 30), backtest=False)
    assert s.status == "on" and s.on
    assert (s.provider, s.model_id) == ("openai", "gpt-6-sol")


def test_a_missing_key_skips_the_arm(monkeypatch, caplog):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with caplog.at_level("WARNING"):
        s = sh.shadow_setup(today=date(2026, 10, 3), backtest=False)
    assert s.status == "no_key" and not s.on
    assert "OPENAI_API_KEY is not set" in caplog.text


def test_a_mocked_production_model_never_sends_the_shadow_to_the_network(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = sh.shadow_setup(today=date(2026, 10, 3), backtest=False, model_call_injected=True)
    assert s.status == "no_shadow_call" and not s.on
    s = sh.shadow_setup(today=date(2026, 10, 3), backtest=False, model_call_injected=True,
                        shadow_call=_fake)
    assert s.on and s.call is _fake


def test_the_default_call_goes_to_call_openai_at_high_effort(monkeypatch):
    import forecaster.providers as providers

    seen = {}

    def fake_call_openai(prompt, model, temperature, *, reasoning_effort=None,
                         prompt_cache_key=None):
        seen.update(model=model, effort=reasoning_effort, key=prompt_cache_key)
        return SimpleNamespace(text="{}", usage={"prompt_tokens": 1000, "completion_tokens": 100},
                               error=None)

    monkeypatch.setattr(providers, "call_openai", fake_call_openai)
    text, usage, err = sh.openai_shadow_call("gpt-6-sol")("hello")
    assert seen == {"model": "gpt-6-sol", "effort": "high", "key": "pythia:sibyl:shadow"}
    assert text == "{}" and err == "" and usage["cost_usd"] > 0  # priced from model_costs.json


# --- 2. a run, both providers mocked --------------------------------------------------

def _ok_pack(query, **kwargs):
    pack = EvidencePack(query=query, backend="brave", grounded=True)
    pack.sources = [EvidenceSource(title="R", url="https://news.example.com/r",
                                   summary="Clashes reported.", date="2026-09-20")]
    pack.debug = {"usage": {"cost_usd": 0.005}, "status_code": 200}
    return pack


def _setup_run(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)
    monkeypatch.setattr(sibyl_config, "MIN_SEARCH_OK", 1)
    monkeypatch.setattr(sibyl_config, "MIN_DOCS_READ", 0)
    disable_submit_gate(monkeypatch)
    monkeypatch.setattr(sibyl_agent, "log_sibyl_call", lambda **kw: None)
    monkeypatch.setattr(sibyl_tools, "_sleep", lambda s: None)
    monkeypatch.setattr(sibyl_tools, "fetch_via_brave_search", _ok_pack)
    monkeypatch.setattr(sibyl_run, "build_reference", stub_reference)
    monkeypatch.setattr(sibyl_config, "TRIAL_WORKERS", 1)
    sibyl_tools.reset_run_state()


def _researching(tag: str, calls: List[str], m1=None, m6=None, cost=0.1):
    """A model that searches at step 1 and submits at step 2."""

    def call(prompt):
        calls.append(tag)
        searched = "=== STEP 1: YOUR RESPONSE ===" in prompt
        text = make_submit_response(m1, m6) if searched else make_search_response("eth clashes")
        return text, {"cost_usd": cost}, ""

    return call


def _db():
    from pythia.db.schema import connect

    return connect(read_only=False)


def test_a_run_builds_the_shadow_series_and_publishes_only_sibyl(tmp_path, monkeypatch):
    _setup_run(tmp_path, monkeypatch)
    calls: List[str] = []
    summary = sibyl_run.run_sibyl(
        HS_RUN_ID, n_questions=1,
        model_call=_researching("claude", calls),
        shadow_call=_researching("sol", calls, SOL_M1, SOL_M6, cost=0.2),
    )
    assert summary["n_forecast"] == 1
    assert summary["shadow_status"] == "on" and summary["n_shadow_series"] == 1
    assert summary["shadow_cost_usd"] == pytest.approx(0.4 + 0.005)  # two steps, one search
    con = _db()
    try:
        models = {r[0] for r in con.execute(
            "SELECT DISTINCT model_name FROM forecasts_raw").fetchall()}
        models |= {r[0] for r in con.execute(
            "SELECT DISTINCT model_name FROM forecasts_ensemble WHERE model_name LIKE 'sib%'"
        ).fetchall()}
        assert models == {"sibyl"}
        sj, shadow_cost, cost, trials, final = con.execute(
            "SELECT shadow_json, shadow_cost_usd, cost_usd, trials_json, final_by_month_json "
            "FROM sibyl_forecasts WHERE question_id = ?", [Q1]).fetchone()
        payload = json.loads(sj)
        lanes = {t["trial_index"]: t["lane"] for t in json.loads(trials)}
        assert payload["status"] == "ok"
        assert payload["replaced_trial_index"] == 2 and lanes[2] == "C"
        assert payload["shadow_trial_index"] == 3 and 3 not in lanes  # not in trials_json
        assert payload["trial_indices"] == [0, 1, 3]
        assert payload["trial"]["role"] == "shadow" and payload["trial"]["lane"] == "C"
        assert set(payload["final_by_month"]) == {str(m) for m in range(1, 7)}
        assert payload["final_by_month"]["1"] != json.loads(final)["1"]
        assert shadow_cost == pytest.approx(0.405)
        assert cost == pytest.approx(0.6 + 0.015)  # production only
        roles = con.execute("SELECT role, COUNT(*) FROM sibyl_evidence GROUP BY 1 ORDER BY 1"
                            ).fetchall()
        assert roles == [("production", 3), ("shadow", 1)]
        run = con.execute("SELECT shadow_status, shadow_model, n_shadow_trials, n_search_calls "
                          "FROM sibyl_runs").fetchone()
        assert run == ("on", "openai:gpt-6-sol", 1, 3)  # shadow searches are not counted
    finally:
        con.close()
    sibyl_tools.reset_run_state()


def test_shadow_trials_come_after_every_production_trial(tmp_path, monkeypatch):
    _setup_run(tmp_path, monkeypatch)
    calls: List[str] = []
    summary = sibyl_run.run_sibyl(
        HS_RUN_ID, n_questions=2,
        model_call=_researching("claude", calls),
        shadow_call=_researching("sol", calls, SOL_M1, SOL_M6),
    )
    assert summary["n_shadow_trials"] == 2
    first_sol = calls.index("sol")
    assert all(c == "claude" for c in calls[:first_sol])
    assert all(c == "sol" for c in calls[first_sol:])
    sibyl_tools.reset_run_state()


def test_near_the_budget_cap_the_shadow_is_skipped_first(tmp_path, monkeypatch):
    _setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(sibyl_config, "SHADOW_HEADROOM_USD", 1000.0)
    calls: List[str] = []
    summary = sibyl_run.run_sibyl(
        HS_RUN_ID, n_questions=2,
        model_call=_researching("claude", calls),
        shadow_call=_researching("sol", calls),
    )
    assert summary["n_forecast"] == 2  # production untouched
    assert "sol" not in calls
    assert summary["n_shadow_skipped"] == 2 and summary["n_shadow_trials"] == 0
    assert summary["config"]["shadow_skip_reasons"] == {"run budget headroom": 2}
    con = _db()
    try:
        got = {json.loads(r[0])["reason"] for r in con.execute(
            "SELECT shadow_json FROM sibyl_forecasts").fetchall()}
        assert got == {"run budget headroom"}
    finally:
        con.close()
    sibyl_tools.reset_run_state()


def test_near_the_time_cap_the_shadow_is_skipped(tmp_path, monkeypatch):
    _setup_run(tmp_path, monkeypatch)
    ticks = iter([0.0] + [600.0] * 1000)  # ten minutes in, at every later reading
    calls: List[str] = []
    summary = sibyl_run.run_sibyl(
        HS_RUN_ID, n_questions=1, max_runtime_min=25, clock=lambda: next(ticks),
        model_call=_researching("claude", calls),
        shadow_call=_researching("sol", calls),
    )
    assert summary["n_forecast"] == 1 and "sol" not in calls
    assert summary["config"]["shadow_skip_reasons"] == {"run time headroom": 1}
    sibyl_tools.reset_run_state()


def test_without_the_key_the_run_is_flagged_and_production_is_whole(tmp_path, monkeypatch):
    _setup_run(tmp_path, monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls: List[str] = []
    summary = sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1,
                                  model_call=_researching("claude", calls))
    assert summary["n_forecast"] == 1
    assert summary["shadow_status"] == "no_key" and summary["n_shadow_trials"] == 0
    con = _db()
    try:
        assert con.execute("SELECT shadow_json FROM sibyl_forecasts").fetchone()[0] is None
        assert con.execute("SELECT shadow_status FROM sibyl_runs").fetchone()[0] == "no_key"
    finally:
        con.close()
    sibyl_tools.reset_run_state()


def test_a_shadow_trial_without_evidence_gives_no_series(tmp_path, monkeypatch):
    _setup_run(tmp_path, monkeypatch)
    calls: List[str] = []

    def blind_sol(prompt):  # submits at step 1, never searches
        calls.append("sol")
        return make_submit_response(SOL_M1, SOL_M6), {"cost_usd": 0.2}, ""

    summary = sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1,
                                  model_call=_researching("claude", calls),
                                  shadow_call=blind_sol)
    assert summary["n_shadow_trials"] == 1 and summary["n_shadow_series"] == 0
    con = _db()
    try:
        payload = json.loads(con.execute("SELECT shadow_json FROM sibyl_forecasts").fetchone()[0])
        assert payload["status"] == "no_valid_shadow_trial" and "final_by_month" not in payload
    finally:
        con.close()
    sibyl_tools.reset_run_state()


def test_controls_get_no_shadow_trial():
    out = SimpleNamespace(status="ok", shadow_ctx=None)
    # A control's outcome carries no context (process_question leaves it
    # None), so the phase is handed nothing to run.
    setup = sh.ShadowSetup("on", "openai", "gpt-6-sol", _fake)
    counts = sh.run_shadow_phase(None, [o.shadow_ctx for o in [out] if o.shadow_ctx], setup,
                                 sibyl_run_id="r", tracker=None, minutes_left=lambda: 999)
    assert counts.n_trials == 0 and counts.n_skipped == 0


# --- 3. scores and the comparison -----------------------------------------------------

def _seed_scored(tmp_path, monkeypatch, n_questions: int):
    seed_db(tmp_path, monkeypatch)
    con = _db()
    from pythia.tools.score_baselines import ensure_baseline_tables

    ensure_baseline_tables(con)
    con.execute("CREATE TABLE IF NOT EXISTS resolutions (question_id TEXT, horizon_m INTEGER, "
                "value DOUBLE, is_test BOOLEAN DEFAULT FALSE)")
    good = [0.05, 0.05, 0.1, 0.6, 0.1, 0.05, 0.05]   # mass on bucket 3 (25 to <100)
    poor = [0.6, 0.2, 0.1, 0.04, 0.03, 0.02, 0.01]
    by_m = lambda v: {str(m): v for m in range(1, 7)}  # noqa: E731
    for i in range(n_questions):
        qid = f"X{i:02d}_ACE_FATALITIES_2026-08"
        con.execute(
            "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric, "
            "target_month, window_start_date, wording, status, track) VALUES "
            "(?, ?, 'ETH', 'ACE', 'FATALITIES', '2027-01', DATE '2026-08-01', 'w', 'active', 1)",
            [qid, HS_RUN_ID])
        shadow = {"status": "ok", "model": "openai:gpt-6-sol", "final_by_month": by_m(good),
                  "shadow_trial_by_month": by_m(good), "claude_lane_c_by_month": by_m(poor)}
        con.execute(
            "INSERT INTO sibyl_forecasts (sibyl_run_id, run_id, question_id, hazard_code, "
            "metric, status, raw_by_month_json, reference_json, final_by_month_json, "
            "evidence_ok, shadow_json, created_at) VALUES ('sr1', 'fc', ?, 'ACE', "
            "'FATALITIES', 'ok', ?, ?, ?, TRUE, ?, CURRENT_TIMESTAMP)",
            [qid, json.dumps({"vectors": by_m(poor)}), json.dumps({"by_month": by_m(poor)}),
             json.dumps(by_m(poor)), json.dumps(shadow)])
        con.execute("INSERT INTO resolutions (question_id, horizon_m, value) VALUES (?, 1, 50)",
                    [qid])
        # Sibyl's own score row, as compute_scores writes it under the run id.
        from sibyl.score_variants import score_vector

        for st, v in score_vector(poor, 3).items():
            con.execute("INSERT INTO scores (question_id, horizon_m, metric, score_type, "
                        "model_name, value, run_id, created_at, is_test) VALUES "
                        "(?, 1, 'FATALITIES', ?, 'sibyl', ?, 'fc', CURRENT_TIMESTAMP, FALSE)",
                        [qid, st, v])
    return con


def test_the_shadow_series_and_its_trial_are_scored(tmp_path, monkeypatch):
    con = _seed_scored(tmp_path, monkeypatch, 2)
    try:
        counters = sv.score_variants(con, as_of_month="2026-10")
        assert counters["scored_shadow"] == 2 and counters["shadow_trial_rows"] == 2 * 2 * 3
        rows = con.execute("SELECT score_type, value FROM scores WHERE model_name = ? "
                           "AND run_id IS NULL ORDER BY question_id, score_type",
                           [sv.SHADOW_MODEL_NAME]).fetchall()
        want = sv.score_vector([0.05, 0.05, 0.1, 0.6, 0.1, 0.05, 0.05], 3)
        assert rows[:3] == [(st, pytest.approx(want[st])) for st in sorted(want)]
        series = {r[0] for r in con.execute("SELECT DISTINCT series FROM sibyl_variant_scores"
                                            ).fetchall()}
        assert series == {sh.SERIES_SHADOW_TRIAL, sh.SERIES_CLAUDE_LANE_C}
        # A rescore leaves one set of rows, and a forecast that loses its
        # shadow series loses its shadow scores.
        sv.score_variants(con, as_of_month="2026-10")
        assert con.execute("SELECT COUNT(*) FROM scores WHERE model_name = ?",
                           [sv.SHADOW_MODEL_NAME]).fetchone()[0] == 6
        con.execute("UPDATE sibyl_forecasts SET shadow_json = NULL")
        sv.score_variants(con, as_of_month="2026-10")
        assert con.execute("SELECT COUNT(*) FROM scores WHERE model_name = ?",
                           [sv.SHADOW_MODEL_NAME]).fetchone()[0] == 0
    finally:
        con.close()


def test_the_comparison_says_not_yet_below_twenty_questions(tmp_path, monkeypatch):
    con = _seed_scored(tmp_path, monkeypatch, 3)
    try:
        sv.score_variants(con, as_of_month="2026-10")
        got = sh.shadow_comparison(con)
        assert got["min_questions"] == 20 and got["model"] == "openai:gpt-6-sol"
        for part in ("series", "trial"):
            assert got[part]["brier"] == {"status": "not_yet", "n_questions": 3,
                                          "min_questions": 20}
        got = sh.shadow_comparison(con, min_questions=3)
        b = got["series"]["brier"]
        assert b["status"] == "ok" and b["n_questions"] == 3
        assert b["mean_diff"] < 0 and b["lo"] <= b["mean_diff"] <= b["hi"]  # shadow better
        assert got["trial"]["log"]["mean_diff"] < 0
    finally:
        con.close()


def test_the_comparison_on_an_old_database_is_empty_not_an_error(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)
    con = _db()
    try:
        con.execute("ALTER TABLE sibyl_forecasts DROP COLUMN shadow_json")
        got = sh.shadow_comparison(con)
        assert got["series"]["brier"]["status"] == "not_yet" and got["model"] is None
    finally:
        con.close()


def test_the_advice_findings_carry_the_comparison(tmp_path, monkeypatch):
    from sibyl import advice

    con = _seed_scored(tmp_path, monkeypatch, 3)
    try:
        sv.score_variants(con, as_of_month="2026-10")
        rows = advice.generate(con, as_of_month="2026-10")
        pooled = next(r for r in rows if r["scope"] == "pooled")
        assert pooled["findings"]["shadow"]["series"]["brier"]["status"] == "not_yet"
        assert "shadow" not in pooled["advice"].lower()
        assert all("shadow" not in r["findings"] for r in rows if r["scope"] != "pooled")
    finally:
        con.close()
