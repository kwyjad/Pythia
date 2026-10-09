# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The structured-data starting pack (Oct 2026, review Part 6).

The arm and its salt, the allowlist and hazard gates, the forecast products
behind their switch, the cap, nothing in backtest, the prompts unchanged when
no pack is shown, the record a run leaves, and the comparison's thresholds.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("duckdb")

import sibyl.agent as sibyl_agent
import sibyl.config as sibyl_config
import sibyl.run as sibyl_run
from sibyl import pack as sp
from sibyl.advice import advice_arm
from sibyl.belief_state import empty_plan, initial_belief
from sibyl.select_questions import SibylQuestion
from tests.sibyl_test_utils import HS_RUN_ID, Q1, stub_reference
from tests.test_sibyl_lanes import LOW, _lane_model, run_env  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs" / "prompts" / "2026-10-09-4"


def _q(hazard="ACE", metric="FATALITIES", qid="ETH_ACE_FATALITIES_2026-11"):
    return SimpleNamespace(question_id=qid, iso3="ETH", hazard_code=hazard, metric=metric)


SD = {
    "crisiswatch": "ICG CRISISWATCH (newest edition held: 2026-09)\nEthiopia deteriorated.",
    "gdelt_conflict_indicators": "GDELT CONFLICT INDICATORS:\nT1 events rising.",
    "hdx_signals": "HDX SIGNALS:\nacled_conflict high.",
    "enso_context": "ENSO: El Nino.",
    "conflict_forecasts": "CONFLICT FORECASTS:\nVIEWS says 120.",
    "hazard_grounding": {"report_markdown": "the ensemble's grounding"},
    "adversarial_check": {"verdict": "the ensemble's check"},
    "reliefweb_reports": [{"title": "a report"}],
}


@pytest.fixture()
def share_one(monkeypatch):
    monkeypatch.setattr(sibyl_config, "PACK_SHARE", 1.0)
    monkeypatch.setattr(sibyl_config, "BACKTEST_MODE", False)


# --- the arm -----------------------------------------------------------------------

def test_the_arm_is_a_stable_hash_with_its_own_salt():
    ids = [f"Q{i:03d}_ACE_FATALITIES_2026-11" for i in range(400)]
    arms = [sp.pack_arm(q, 0.5) for q in ids]
    assert arms == [sp.pack_arm(q, 0.5) for q in ids]
    share = arms.count(sp.ARM_PACK) / len(arms)
    assert 0.4 < share < 0.6
    # Independent of the advice arm: the two splits do not coincide.
    same = sum((a == sp.ARM_PACK) == (advice_arm(q, 0.5) == "advice") for a, q in zip(arms, ids))
    assert 0.35 < same / len(ids) < 0.65
    assert sp.pack_arm("x", 0.0) == sp.ARM_NO_PACK and sp.pack_arm("x", 1.0) == sp.ARM_PACK


def test_no_pack_loads_nothing(monkeypatch):
    monkeypatch.setattr(sibyl_config, "PACK_SHARE", 0.0)
    called = []
    p = sp.build_pack(_q(), date.today(), loader=lambda *a: called.append(a) or SD)
    assert (p.arm, p.text, called) == (sp.ARM_NO_PACK, "", [])


def test_nothing_in_backtest(share_one, monkeypatch):
    assert sp.build_pack(_q(), date(2020, 1, 1), loader=lambda *a: SD).arm is None
    monkeypatch.setattr(sibyl_config, "BACKTEST_MODE", True)
    assert sp.build_pack(_q(), date.today(), loader=lambda *a: SD).text == ""


# --- the content -------------------------------------------------------------------

def test_the_allowlist_never_carries_the_ensembles_own_research(share_one):
    p = sp.build_pack(_q(), date.today(), loader=lambda *a: SD)
    assert p.arm == sp.ARM_PACK
    assert p.text.startswith("\n\n=== STRUCTURED DATA HELD BY THE PIPELINE (as of ")
    for own in ("the ensemble's grounding", "the ensemble's check", "a report"):
        assert own not in p.text
    assert "Ethiopia deteriorated" in p.text and "T1 events rising" in p.text
    # Hazard gates: ENSO is for climate hazards, not conflict.
    assert "El Nino" not in p.text
    assert p.sections == ["crisiswatch", "hdx_signals", "gdelt_conflict_indicators"]


def test_conflict_forecasts_only_behind_their_switch(share_one, monkeypatch):
    assert "VIEWS says 120" not in sp.build_pack(_q(), date.today(), loader=lambda *a: SD).text
    monkeypatch.setattr(sibyl_config, "PACK_INCLUDE_FORECASTS", True)
    assert "VIEWS says 120" in sp.build_pack(_q(), date.today(), loader=lambda *a: SD).text


def test_food_security_is_left_out_where_the_reading_shows_phase3(share_one, monkeypatch):
    import pythia.food_security as fs

    monkeypatch.setattr(fs, "format_food_security_for_spd", lambda d: "FOOD SECURITY: 1,000,000")
    sd = {"fewsnet_food_security": {"x": 1}, "enso_context": "ENSO: El Nino."}
    q = _q("DR", "PHASE3PLUS_IN_NEED", "SOM_DR_PHASE3PLUS_IN_NEED_2026-11")
    shown = sp.build_pack(q, date.today(), loader=lambda *a: sd, resolver_reading_shown=True)
    assert "FOOD SECURITY" not in shown.text and "El Nino" in shown.text
    alone = sp.build_pack(q, date.today(), loader=lambda *a: sd, resolver_reading_shown=False)
    assert "FOOD SECURITY: 1,000,000" in alone.text


def test_sections_render_with_the_spd_formatters(share_one, monkeypatch):
    import pythia.acaps as acaps

    seen = []
    monkeypatch.setattr(acaps, "format_inform_severity_for_spd",
                        lambda d: seen.append(d) or "INFORM SEVERITY: 4.1/10")
    p = sp.build_pack(_q(), date.today(), loader=lambda *a: {"inform_severity": {"s": 4.1}})
    assert seen == [{"s": 4.1}] and "INFORM SEVERITY: 4.1/10" in p.text


def test_the_cap_drops_whole_sections_from_the_end_and_names_them(share_one):
    sd = {"crisiswatch": "C" * 3000, "hdx_signals": "H" * 3000,
          "gdelt_conflict_indicators": "G" * 3000}
    p = sp.build_pack(_q(), date.today(), loader=lambda *a: sd, max_chars=7000)
    assert p.sections == ["crisiswatch", "hdx_signals"]
    assert p.dropped == ["GDELT conflict indicators"]
    assert "(Left out for length: GDELT conflict indicators.)" in p.text
    assert p.chars <= 7000 and "G" * 10 not in p.text


def test_an_empty_or_failing_load_is_pack_empty(share_one):
    assert sp.build_pack(_q(), date.today(), loader=lambda *a: {}).arm == sp.ARM_PACK_EMPTY

    def boom(*a):
        raise RuntimeError("down")

    p = sp.build_pack(_q(), date.today(), loader=boom)
    assert p.arm == sp.ARM_PACK_EMPTY and p.text == "" and "down" in (p.error or "")


# --- the prompts -------------------------------------------------------------------

def test_the_spd_prompt_builder_knows_nothing_of_the_pack():
    # The pack calls the SPD formatters; the SPD prompt never calls the pack,
    # so main-ensemble prompts cannot change with it.
    src = (ROOT / "forecaster" / "prompts.py").read_text()
    assert "sibyl" not in src.lower()


def _old_agent():
    path = ARCHIVE / "sibyl_agent.py"
    spec = importlib.util.spec_from_file_location("sibyl_agent_before_part6", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(spec.name, None)
    mod._CARD_DIR = sibyl_agent._CARD_DIR
    return mod


def _args():
    q = SibylQuestion(question_id=Q1, hs_run_id=HS_RUN_ID, iso3="ETH", hazard_code="ACE",
                      metric="FATALITIES", target_month="2027-01",
                      window_start_date=date(2026, 11, 1), wording="How many?",
                      volatility_score=0.8, triage_score=0.9)
    ref = stub_reference()
    start = initial_belief(ref.by_month, "FATALITIES")
    start.plan = empty_plan()
    kw = dict(step=2, as_of=date(2026, 10, 9), forecast_months=["2026-11"] * 6,
              transcript_text="=== STEP 1 ===\nx\n", country_name="Ethiopia",
              track_record="t", lessons="", perspective=sibyl_agent.TRIAL_LANES["B"],
              resolver_reading="\n\n=== RESOLVING SOURCE: LATEST READING (as of 2026-10-09) ===\nr")
    return q, ref, start, kw


def test_no_pack_leaves_the_sibyl_prompt_byte_identical():
    q, ref, start, kw = _args()
    assert sibyl_agent.build_step_prompt(q, ref, start, **kw) == \
        _old_agent().build_step_prompt(q, ref, start, **kw)


def test_the_pack_follows_the_reading_in_the_question_segment():
    q, ref, start, kw = _args()
    block = "\n\n=== STRUCTURED DATA HELD BY THE PIPELINE (as of 2026-10-09) ===\np"
    segs = sibyl_agent.build_step_prompt(q, ref, start, pack=block, return_segments=True, **kw)
    seg = segs[1][0]
    assert seg.index("RESOLVING SOURCE") < seg.index("STRUCTURED DATA HELD") \
        < seg.index("=== YOUR TRACK RECORD ===")


# --- a run -------------------------------------------------------------------------

def test_a_run_shows_the_pack_to_every_trial_and_records_the_arm(run_env, monkeypatch):  # noqa: F811
    monkeypatch.setattr(sibyl_config, "PACK_SHARE", 1.0)
    monkeypatch.setattr(sp, "_default_loader", lambda iso3, hz: {"hdx_signals": "HDX SIGNALS: high"})
    monkeypatch.setattr(sibyl_run, "extra_trials_rule",
                        lambda *a, **k: (None, {"max_pairwise_jsd": 0.0}))
    prompts = []
    base = _lane_model({"*": LOW}, [])

    def model(prompt):
        prompts.append(prompt)
        return base(prompt)

    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=model)
    assert prompts and all("HDX SIGNALS: high" in p for p in prompts)
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        arm, pj = con.execute("SELECT pack_arm, pack_json FROM sibyl_forecasts "
                              "WHERE question_id = ?", [Q1]).fetchone()
        assert arm == sp.ARM_PACK and json.loads(pj)["sections"] == ["hdx_signals"]
    finally:
        con.close()


# --- the comparison ----------------------------------------------------------------

def _seed(tmp_path, monkeypatch, n_per_arm, *, control=False):
    from tests.sibyl_test_utils import seed_db
    from pythia.db.schema import connect
    from pythia.tools.score_baselines import ensure_baseline_tables

    seed_db(tmp_path, monkeypatch)
    con = connect(read_only=False)
    ensure_baseline_tables(con)
    trials = json.dumps([{"n_docs_read": 5, "steps_used": 4}] * 3)
    ref = json.dumps({"by_month": {"1": [0.5, 0.5]}})
    raw = json.dumps({"vectors": {"1": [0.4, 0.6]}})
    for arm, gain in ((sp.ARM_PACK, -0.05), (sp.ARM_NO_PACK, 0.01)):
        for i in range(n_per_arm):
            qid = f"{arm}{i:02d}_ACE_FATALITIES_2026-08"
            con.execute(
                "INSERT INTO sibyl_forecasts (sibyl_run_id, run_id, question_id, hazard_code, "
                "metric, status, trials_json, reference_json, raw_by_month_json, evidence_ok, "
                "selection_pass, pack_arm, js_divergence_vs_standard, created_at) VALUES "
                "('sr', 'fc', ?, 'ACE', 'FATALITIES', 'ok', ?, ?, ?, TRUE, ?, ?, 0.1, "
                "CURRENT_TIMESTAMP)",
                [qid, trials, ref, raw, "control" if control else "fill", arm])
            for model, v in (("sibyl", 0.3 + gain + 0.001 * i), ("__ext_sibyl_ref", 0.3)):
                con.execute(
                    "INSERT INTO scores (question_id, horizon_m, metric, score_type, model_name, "
                    "value, run_id) VALUES (?, 1, 'FATALITIES', 'brier', ?, ?, ?)",
                    [qid, model, v, None if model.startswith("__ext") else "fc"])
    return con


def test_the_comparison_says_not_yet_below_its_thresholds(tmp_path, monkeypatch):
    con = _seed(tmp_path, monkeypatch, 5)
    try:
        g = sp.pack_comparison(con)["groups"]["selected"]
        assert g["arms"][sp.ARM_PACK]["immediate"] == {"status": "not_yet"}
        assert g["arms"][sp.ARM_PACK]["gain_over_reference"] == {"status": "not_yet"}
        assert g["pack_minus_no_pack"]["status"] == "not_yet"
    finally:
        con.close()


def test_the_comparison_reports_from_its_thresholds(tmp_path, monkeypatch):
    con = _seed(tmp_path, monkeypatch, 20)
    try:
        out = sp.pack_comparison(con)
        g = out["groups"]["selected"]
        imm = g["arms"][sp.ARM_PACK]["immediate"]
        assert imm["status"] == "ok" and imm["docs_per_trial"] == 5.0
        assert imm["jsd_month1_from_reference"] > 0
        d = g["pack_minus_no_pack"]
        assert d["status"] == "ok" and d["brier_diff"] < 0 and d["lo"] <= d["brier_diff"] <= d["hi"]
        assert sp.pack_comparison(con) == out  # fixed seed
        assert "control" not in out["groups"]
    finally:
        con.close()


def test_controls_are_compared_apart(tmp_path, monkeypatch):
    con = _seed(tmp_path, monkeypatch, 3, control=True)
    try:
        assert set(sp.pack_comparison(con)["groups"]) == {"control"}
    finally:
        con.close()


def test_nothing_switches_the_pack_on_by_itself():
    import inspect

    for mod in (sibyl_run, __import__("sibyl.measure", fromlist=["x"])):
        assert "pack_comparison" not in inspect.getsource(mod)
