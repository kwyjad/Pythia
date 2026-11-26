# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The RC shadow arm: stored, compared, never used.

The property that matters most is the first: a shadow that answers with
extreme likelihoods must leave production RC, levels and tracks exactly as a
run with the shadow switched off would.
"""

from __future__ import annotations

import json

import duckdb
import pytest

from horizon_scanner import rc_shadow
from horizon_scanner import regime_change_llm as rc_llm


PROD = {"likelihood": 0.05, "magnitude": 0.1, "direction": "unclear", "window": "",
        "rationale_bullets": ["quiet"], "trigger_signals": []}
EXTREME = {"likelihood": 0.95, "magnitude": 0.95, "direction": "up", "window": "month_1-2",
           "rationale_bullets": ["break imminent"], "trigger_signals": []}


@pytest.fixture()
def db(tmp_path, monkeypatch):
    url = f"duckdb:///{tmp_path}/shadow.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", url)
    rc_shadow.reset()
    yield url
    rc_shadow.reset()


def _stub(monkeypatch, prompts_seen: list) -> None:
    async def _fake_call_rc_model(prompt_text, *, run_id=None, fallback_specs=None, pass_idx=1):
        prompts_seen.append(("pass", prompt_text))
        spec = rc_llm.ModelSpec(name="flash", provider="google", model_id="gemini-3.8-flash", active=True)
        return json.dumps(PROD), {"total_tokens": 10}, "", spec

    async def _fake_call_chat_ms(ms, prompt, temperature=0.2, **kw):
        prompts_seen.append(("shadow", prompt, ms.model_id, ms.thinking))
        return json.dumps(EXTREME), {"prompt_tokens": 1000, "completion_tokens": 500}, ""

    monkeypatch.setattr(rc_llm, "_call_rc_model", _fake_call_rc_model)
    monkeypatch.setattr(rc_llm, "build_rc_prompt", lambda *a, **k: "THE RC PROMPT")
    monkeypatch.setattr(rc_llm, "log_hs_llm_call", lambda **k: None)
    import forecaster.providers as prov

    monkeypatch.setattr(prov, "call_chat_ms", _fake_call_chat_ms)


def _run_country(monkeypatch) -> dict:
    monkeypatch.setattr(rc_llm, "get_active_hazards", lambda iso3, d: {"ACE", "FL", "DR", "TC"})
    monkeypatch.setattr(rc_llm, "_run_grounding_for_hazard", lambda *a, **k: None)
    monkeypatch.setattr(rc_llm, "log_hs_hazard_tail_packs_to_db", lambda *a, **k: None)
    out = rc_llm.run_rc_for_country(
        "hs_20261213T000000", "TST", "Testland", {"ACE": "", "FL": "", "DR": "", "TC": ""},
        {}, None, [], expected_hazards=["ACE", "DR", "FL", "TC"],
    )
    rc_shadow.wait_background(60)
    return out


def _triage_rows(rc_out: dict) -> list:
    """Write hs_triage through the production writer and read it back."""

    from horizon_scanner import horizon_scanner as hs
    from pythia.db.schema import connect

    triage = {"hazards": {hz: {"triage_score": 0.6, "regime_change": rc}
                          for hz, rc in rc_out["hazards"].items()}}
    hs._write_hs_triage("hs_20261213T000000", "TST", triage)
    con = connect(read_only=False)
    try:
        return con.execute(
            """SELECT hazard_code, regime_change_likelihood, regime_change_level, track,
                      need_full_spd, regime_change_json
               FROM hs_triage WHERE run_id = 'hs_20261213T000000' ORDER BY hazard_code"""
        ).fetchall()
    finally:
        con.close()


def test_shadow_never_changes_production_rc_levels_or_tracks(monkeypatch, db) -> None:
    seen_off: list = []
    _stub(monkeypatch, seen_off)
    monkeypatch.setenv("PYTHIA_RC_SHADOW_ENABLED", "0")
    off = _run_country(monkeypatch)

    seen_on: list = []
    _stub(monkeypatch, seen_on)
    monkeypatch.setenv("PYTHIA_RC_SHADOW_ENABLED", "1")
    on = _run_country(monkeypatch)

    assert on["hazards"] == off["hazards"]
    assert on["status"] == off["status"]
    # hs_triage, levels and tracks as the production writer stores them.
    assert _triage_rows(off) == _triage_rows(on)
    assert {r[2] for r in _triage_rows(on)} == {0}  # production level stays 0
    assert not any(s[0] == "shadow" for s in seen_off)
    shadow_calls = [s for s in seen_on if s[0] == "shadow"]
    assert len(shadow_calls) == 4
    # Byte-identical prompt to pass 1, on the shadow model at medium effort.
    assert {s[1] for s in shadow_calls} == {"THE RC PROMPT"}
    assert {(s[2], s[3]) for s in shadow_calls} == {("gpt-6.1-sol", "medium")}

    con = duckdb.connect(db.replace("duckdb:///", ""))
    rows = con.execute("SELECT hazard_code, level, likelihood, status FROM hs_rc_shadow ORDER BY 1").fetchall()
    assert [r[0] for r in rows] == ["ACE", "DR", "FL", "TC"]
    assert all(r[1] == 3 and r[3] == "ok" for r in rows)
    con.close()


def test_rc_pass_count_and_merge_are_untouched(monkeypatch, db) -> None:
    monkeypatch.delenv("PYTHIA_HS_RC_PASSES", raising=False)
    assert rc_llm._rc_pass_count() == 1


def test_a_shadow_failure_never_raises(monkeypatch, db) -> None:
    async def _boom(*a, **k):
        raise RuntimeError("provider down")

    import forecaster.providers as prov

    monkeypatch.setattr(prov, "call_chat_ms", _boom)
    monkeypatch.setenv("PYTHIA_RC_SHADOW_ENABLED", "1")
    rc_shadow.run_for_hazard("P", hs_run_id="hs_x", iso3="TST", hazard_code="ACE", mode="full")
    rc_shadow.wait_background(30)
    assert rc_shadow.stats()["failures"] >= 1
    con = duckdb.connect(db.replace("duckdb:///", ""))
    assert con.execute("SELECT status FROM hs_rc_shadow").fetchone()[0] == "error"
    con.close()


def test_the_cost_cap_stops_new_calls_and_counts_them(monkeypatch, db) -> None:
    calls: list = []

    async def _fake(ms, prompt, temperature=0.2, **kw):
        calls.append(prompt)
        return json.dumps(PROD), {"prompt_tokens": 100, "completion_tokens": 100, "cost_usd": 0.4}, ""

    import forecaster.providers as prov

    monkeypatch.setattr(prov, "call_chat_ms", _fake)
    monkeypatch.setattr(rc_shadow, "estimate_cost", lambda p, m: 0.4)
    monkeypatch.setenv("PYTHIA_RC_SHADOW_MAX_USD", "1.0")
    for hz in ("ACE", "DR", "FL", "TC"):
        rc_shadow._sync_call("P", hs_run_id="hs_x", iso3="TST", hazard_code=hz, source="sync")
    assert len(calls) == 2
    assert rc_shadow.stats()["skipped_cap"] == 2
    con = duckdb.connect(db.replace("duckdb:///", ""))
    assert con.execute("SELECT COUNT(*) FROM hs_rc_shadow WHERE status = 'skipped_cap'").fetchone()[0] == 2
    con.close()


def test_off_in_test_mode_unless_set(monkeypatch) -> None:
    monkeypatch.delenv("PYTHIA_RC_SHADOW_ENABLED", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("PYTHIA_TEST_MODE", "1")
    assert not rc_shadow.enabled()
    monkeypatch.setenv("PYTHIA_TEST_MODE", "0")
    assert rc_shadow.enabled()
    monkeypatch.setenv("PYTHIA_TEST_MODE", "1")
    monkeypatch.setenv("PYTHIA_RC_SHADOW_ENABLED", "1")
    assert rc_shadow.enabled()


def test_env_overrides_the_shadow_model(monkeypatch) -> None:
    monkeypatch.setenv("PYTHIA_RC_SHADOW_MODEL", "anthropic:claude-opus-5-5")
    monkeypatch.setenv("PYTHIA_RC_SHADOW_THINKING", "high")
    spec = rc_shadow.shadow_spec()
    assert (spec.provider, spec.model_id, spec.thinking) == ("anthropic", "claude-opus-5-5", "high")


def test_the_shadow_family_is_non_blocking() -> None:
    from pythia import llm_batch

    assert rc_shadow.SHADOW_FAMILY in llm_batch.NON_BLOCKING_FAMILIES
    cid = llm_batch.make_custom_id(rc_shadow.SHADOW_FAMILY, iso3="TST", hazard_code="ACE",
                                   pass_idx=1, pipeline_id="pl_x")
    prod = llm_batch.make_custom_id("hs_rc", iso3="TST", hazard_code="ACE", pass_idx=1, pipeline_id="pl_x")
    assert cid != prod


def test_drain_and_batch_state_skip_the_shadow_family(monkeypatch, tmp_path) -> None:
    from pythia import llm_batch

    polled: list = []
    monkeypatch.setattr(llm_batch, "pending_batches", lambda con, pid: [
        {"batch_id": "b_shadow", "family": "hs_rc_shadow", "provider": "openai",
         "provider_batch_id": "x", "submitted_at": None},
    ])
    monkeypatch.setattr(llm_batch, "_poll_with_retry", lambda con, bid: polled.append(bid))
    rep = llm_batch.drain_pending(None, "pl_x", wait_min=0)
    assert polled == [] and rep["waited_sec"] == 0.0

    import scripts.ci.emit_batch_state as ebs

    src = open(ebs.__file__).read()
    assert "NON_BLOCKING_FAMILIES" in src


def test_shadow_rows_are_labelled_in_the_logging_column(monkeypatch, db) -> None:
    from horizon_scanner.llm_logging import log_hs_llm_call

    spec = rc_shadow.shadow_spec()
    log_hs_llm_call(hs_run_id="hs_x", iso3="TST", hazard_code=rc_shadow.hazard_label("ACE"),
                    model_spec=spec, prompt_text="P", response_text="{}", usage={},
                    error_text=None, call_type="rc_shadow", is_shadow=True)
    log_hs_llm_call(hs_run_id="hs_x", iso3="TST", hazard_code="rc_ACE_pass_1",
                    model_spec=spec, prompt_text="P", response_text="{}", usage={},
                    error_text=None, call_type="rc_pass_1")
    con = duckdb.connect(db.replace("duckdb:///", ""))
    rows = dict(con.execute("SELECT hazard_code, is_shadow FROM llm_calls").fetchall())
    con.close()
    assert rows == {"RC_ACE_SHADOW": True, "RC_ACE_PASS_1": False}


def test_parse_answer_uses_the_production_thresholds(monkeypatch) -> None:
    out = rc_shadow.parse_answer(json.dumps({**PROD, "likelihood": 0.4, "magnitude": 0.5}))
    assert out["status"] == "ok" and out["level"] == 2
    monkeypatch.setenv("PYTHIA_HS_RC_LEVEL2_LIKELIHOOD", "0.5")
    assert rc_shadow.parse_answer(json.dumps({**PROD, "likelihood": 0.4, "magnitude": 0.5}))["level"] == 1
    assert rc_shadow.parse_answer("not json")["status"] == "parse_error"


def test_comparison_matrix_and_lists(db) -> None:
    from scripts.debug_bundle.rc_shadow import markdown, shadow_comparison

    con = duckdb.connect(db.replace("duckdb:///", ""))
    con.execute("""CREATE TABLE hs_triage (run_id TEXT, iso3 TEXT, hazard_code TEXT,
                   regime_change_likelihood DOUBLE, regime_change_level INTEGER)""")
    con.execute("""INSERT INTO hs_triage VALUES ('hs_x','AAA','ACE',0.05,0),
                   ('hs_x','BBB','FL',0.6,3), ('hs_x','CCC','DR',0.05,0)""")
    for iso3, hz, lvl, lik in (("AAA", "ACE", 2, 0.4), ("BBB", "FL", 0, 0.05), ("CCC", "DR", 0, 0.05)):
        rc_shadow.write_row(con, hs_run_id="hs_x", iso3=iso3, hazard_code=hz, provider="openai",
                            model_id="gpt-6.1-sol", cost_usd=0.1,
                            result={"status": "ok", "level": lvl, "likelihood": lik,
                                    "rationale_bullets": ["why"]})
    rep = shadow_comparison(con, "hs_x")
    con.close()
    assert rep["agreement_matrix"]["0"]["2"] == 1
    assert rep["agreement_matrix"]["3"]["0"] == 1
    assert rep["agreement_matrix"]["0"]["0"] == 1
    assert [d["iso3"] for d in rep["would_have_forecast"]] == ["AAA"]
    assert [d["iso3"] for d in rep["production_only"]] == ["BBB"]
    assert rep["share_low_likelihood"]["production"] == pytest.approx(0.667, abs=1e-3)
    assert "Would have forecast" in markdown(rep)
