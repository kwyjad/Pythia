# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Part 1 of the Oct 2026 Sibyl work: an honest record.

No forecast without evidence is stored as valid, the July 2026 run stops
counting, the Brave breaker cannot blind a whole run, every written vector
carries a floor, a trial that already learned something survives a failed
step, and the small fixes beside them (seed label, question wording,
month-position check) hold.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pytest

from pythia.web_research.types import EvidencePack, EvidenceSource

import sibyl.agent as sibyl_agent
import sibyl.config as sibyl_config
import sibyl.run as sibyl_run
import sibyl.tools as sibyl_tools
from sibyl.belief_state import initial_belief_from_anchor
from sibyl.cost import CostTracker
from sibyl.evidence import backfill_evidence_ok, legacy_evidence_ok
from sibyl.select_questions import SibylQuestion
from sibyl.spd import apply_bucket_floor
from tests.sibyl_test_utils import (
    HS_RUN_ID,
    Q1,
    Q2,
    STANDARD_RUN_ID,
    disable_submit_gate,
    make_search_response,
    make_submit_response,
    seed_db,
    stub_base_rate,
    stub_reference,
)

pytestmark = pytest.mark.db

TODAY = date(2026, 10, 3)


def _usage(cost: float = 0.1) -> dict:
    return {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost_usd": cost}


def _ok_pack(query, **kwargs):
    pack = EvidencePack(query=query, backend="brave", grounded=True)
    pack.sources = [
        EvidenceSource(title="Report", url="https://news.example.com/r",
                       summary="Clashes reported.", date="2026-09-20"),
    ]
    pack.debug = {"usage": {"cost_usd": 0.005}, "status_code": 200}
    return pack


def _tripped_pack(query, **kwargs):
    pack = EvidencePack(query=query, backend="brave")
    pack.error = {"type": "circuit_breaker_tripped", "message": "open"}
    pack.debug = {"grounding_backend": "brave_circuit_breaker_tripped"}
    return pack


def _http_fail_pack(query, **kwargs):
    pack = EvidencePack(query=query, backend="brave")
    pack.error = {"type": "no_results", "message": "Brave Search returned no results"}
    pack.debug = {"usage": {"cost_usd": 0.005}, "status_code": 402}
    return pack


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    # These tests pin the evidence gate's mechanics at its Part 1 thresholds
    # (one search, no document). Part 3 raised the defaults and added a
    # submit gate; both are tested in tests/test_sibyl_documents.py.
    monkeypatch.setattr(sibyl_config, "MIN_SEARCH_OK", 1)
    monkeypatch.setattr(sibyl_config, "MIN_DOCS_READ", 0)
    disable_submit_gate(monkeypatch)
    monkeypatch.setattr(sibyl_agent, "log_sibyl_call", lambda **kwargs: None)
    monkeypatch.setattr(sibyl_tools, "_sleep", lambda s: None)
    sibyl_tools.reset_run_state()
    yield
    sibyl_tools.reset_run_state()


def _question() -> SibylQuestion:
    return SibylQuestion(
        question_id=Q1, hs_run_id=HS_RUN_ID, iso3="ETH", hazard_code="ACE",
        metric="FATALITIES", target_month="2027-01", window_start_date=date(2026, 8, 1),
        wording="How many?", volatility_score=0.8, triage_score=0.9,
    )


def _trial(model_call, **kw):
    return sibyl_agent.run_trial(
        _question(), stub_base_rate(), as_of=TODAY, trial_index=0, run_id="sr",
        tracker=CostTracker(run_hard_cap_usd=100), forecast_months=["2026-08"] * 6,
        country_name="Ethiopia", model_call=model_call, **kw,
    )


# --- 1. evidence counts ---------------------------------------------------------

def test_trial_counts_successful_searches_and_documents_read(monkeypatch):
    monkeypatch.setattr(sibyl_tools, "fetch_via_brave_search", _ok_pack)
    monkeypatch.setattr(
        sibyl_agent, "fetch_url",
        lambda url, as_of, **kw: sibyl_tools.ToolResult(tool="fetch_url", ok=True, text="page"),
    )
    script = iter([
        make_search_response("q1"),
        json.dumps({**json.loads(make_search_response()), "action": "fetch_url",
                    "action_input": "https://news.example.com/r"}),
        make_search_response("q2"),
        make_submit_response(),
    ])
    trial = _trial(lambda prompt: (next(script), _usage(), ""))
    assert (trial.n_search_ok, trial.n_docs_read) == (2, 1)
    assert trial.evidence_ok is True
    d = trial.to_dict()
    assert (d["n_search_ok"], d["n_docs_read"], d["evidence_ok"]) == (2, 1, True)


def test_a_trial_that_never_searched_has_no_evidence():
    trial = _trial(lambda prompt: (make_submit_response(), _usage(), ""))
    assert trial.ok is True
    assert trial.evidence_ok is False


def test_a_search_that_found_nothing_is_not_evidence(monkeypatch):
    monkeypatch.setattr(sibyl_tools, "fetch_via_brave_search", _tripped_pack)
    monkeypatch.setattr(sibyl_config, "BREAKER_MAX_RESETS", 0)
    script = iter([make_search_response(), make_submit_response()])
    trial = _trial(lambda prompt: (next(script), _usage(), ""))
    assert trial.n_search_ok == 0
    assert trial.evidence_ok is False


# --- 2. the gate, end to end ---------------------------------------------------

def _run_env(tmp_path, monkeypatch, pack_fn):
    seed_db(tmp_path, monkeypatch)
    monkeypatch.setattr(sibyl_run, "build_reference", stub_reference)
    monkeypatch.setattr(sibyl_tools, "fetch_via_brave_search", pack_fn)
    # Each trial searches once and then submits. The decision is read from the
    # trial's own prompt (its transcript holds step 1 once it has searched),
    # never from a counter shared across trials: trials run on worker threads
    # and a shared counter interleaves between them.
    def model_call(prompt):
        searched = "=== STEP 1: YOUR RESPONSE ===" in prompt
        return (make_submit_response() if searched else make_search_response()), _usage(), ""

    return model_call


def test_a_run_whose_searches_all_fail_stores_every_question_failed(tmp_path, monkeypatch):
    model_call = _run_env(tmp_path, monkeypatch, _tripped_pack)
    summary = sibyl_run.run_sibyl(HS_RUN_ID, n_questions=2, model_call=model_call)

    assert summary["n_forecast"] == 0
    assert summary["n_skipped"] == 2
    # Bounded: five resets, then the trips are counted and the run moves on.
    assert summary["n_search_failed"] == summary["n_search_calls"] == 6
    assert summary["n_breaker_trips"] >= 6

    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        rows = con.execute(
            "SELECT question_id, status, skip_reason, evidence_ok, trials_json "
            "FROM sibyl_forecasts ORDER BY question_id"
        ).fetchall()
        assert [(r[1], r[2], r[3]) for r in rows] == [("failed", "no evidence", False)] * 2
        assert all(len(json.loads(r[4])) == 3 for r in rows)  # trials kept
        for table in ("forecasts_raw", "forecasts_ensemble"):
            n = con.execute(
                f"SELECT COUNT(*) FROM {table} WHERE model_name = 'sibyl'"
            ).fetchone()[0]
            assert n == 0, table
        run = con.execute(
            "SELECT n_search_calls, n_search_failed, n_docs_read FROM sibyl_runs"
        ).fetchone()
        assert run == (6, 6, 0)
    finally:
        con.close()


def test_a_run_with_evidence_writes_a_floored_forecast(tmp_path, monkeypatch):
    model_call = _run_env(tmp_path, monkeypatch, _ok_pack)
    summary = sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=model_call)
    assert summary["n_forecast"] == 1
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        probs = [r[0] for r in con.execute(
            "SELECT probability FROM forecasts_raw WHERE model_name = 'sibyl' "
            "AND question_id = ? AND run_id = ?", [Q1, STANDARD_RUN_ID]
        ).fetchall()]
        assert probs and min(probs) >= sibyl_config.BUCKET_FLOOR - 1e-12
        (ev,) = con.execute(
            "SELECT evidence_ok FROM sibyl_forecasts WHERE question_id = ?", [Q1]
        ).fetchone()
        assert ev is True
    finally:
        con.close()


def test_one_trial_with_evidence_is_not_enough(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)
    monkeypatch.setattr(sibyl_run, "build_reference", stub_reference)
    monkeypatch.setattr(sibyl_tools, "fetch_via_brave_search", _ok_pack)
    # Trial 0 searches then submits; trials 1 and 2 submit blind.
    calls = iter([make_search_response(), make_submit_response(),
                  make_submit_response(), make_submit_response()])
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1,
                        model_call=lambda p: (next(calls), _usage(), ""))
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        (status, reason) = con.execute(
            "SELECT status, skip_reason FROM sibyl_forecasts WHERE question_id = ?", [Q1]
        ).fetchone()
        assert (status, reason) == ("failed", "no evidence")
    finally:
        con.close()


# --- 3. the breaker ----------------------------------------------------------------

def test_a_tripped_breaker_is_waited_out_reset_and_retried_once(monkeypatch):
    seq = iter([_tripped_pack, _ok_pack])
    resets = []
    monkeypatch.setattr(sibyl_tools, "fetch_via_brave_search",
                        lambda q, **kw: next(seq)(q, **kw))
    monkeypatch.setattr(sibyl_tools, "_sleep", lambda s: resets.append(s))
    result = sibyl_tools.brave_search("ethiopia clashes", TODAY, today=TODAY)
    assert result.ok is True and result.sources
    assert resets == [sibyl_config.BREAKER_COOLDOWN_SEC]
    snap = sibyl_tools.COUNTERS.snapshot()
    assert (snap["n_search_calls"], snap["n_search_failed"], snap["n_breaker_trips"]) == (1, 0, 1)


def test_resets_are_capped_per_run(monkeypatch):
    monkeypatch.setattr(sibyl_tools, "fetch_via_brave_search", _tripped_pack)
    monkeypatch.setattr(sibyl_config, "BREAKER_MAX_RESETS", 2)
    for _ in range(5):
        r = sibyl_tools.brave_search("q", TODAY, today=TODAY)
        assert r.ok is False and r.search_failed
    assert sibyl_tools.COUNTERS.snapshot()["n_breaker_resets"] == 2


def test_run_start_resets_the_shared_breaker():
    from pythia.web_research import brave_circuit_breaker as bcb

    for _ in range(3):
        bcb.get_breaker().record_failure(429, "x")
    assert bcb.is_tripped()
    sibyl_tools.reset_run_state()
    assert not bcb.is_tripped()


def test_a_failed_search_names_its_http_status(monkeypatch):
    monkeypatch.setattr(sibyl_tools, "fetch_via_brave_search", _http_fail_pack)
    r = sibyl_tools.brave_search("q", TODAY, today=TODAY)
    assert "HTTP 402" in r.text
    assert r.search_failed is True


def test_an_empty_answer_with_http_200_is_not_a_failure(monkeypatch):
    def empty(q, **kw):
        pack = EvidencePack(query=q, backend="brave")
        pack.error = {"type": "no_results"}
        pack.debug = {"status_code": 200, "usage": {"cost_usd": 0.005}}
        return pack

    monkeypatch.setattr(sibyl_tools, "fetch_via_brave_search", empty)
    r = sibyl_tools.brave_search("q", TODAY, today=TODAY)
    assert r.ok is False and r.search_failed is False


# --- 5/6. the backfill and the readers ------------------------------------------------

def _july_trials(search_ok: bool, n_trials: int = 3) -> list:
    """Trials shaped like sibyl_1784113515141: a search step, then submit."""
    return [
        {
            "trial_index": i,
            "perspective": "Base-rate-weighted perspective: ...",
            "quantiles": {"0.1": 0, "0.5": 10, "0.9": 100},
            "belief_trace": [
                {"step": 1, "action": "brave_search", "action_input": "q",
                 "tool_ok": search_ok, "belief": {}, "repaired": False},
                {"step": 2, "action": "submit", "action_input": "",
                 "tool_ok": None, "belief": {}, "repaired": False},
            ],
        }
        for i in range(n_trials)
    ]


def _insert_forecast(con, srid, qid, trials, *, evidence_ok=None, run_id=STANDARD_RUN_ID):
    con.execute(
        "INSERT INTO sibyl_forecasts (sibyl_run_id, run_id, question_id, iso3, "
        "hazard_code, metric, status, trials_json, pooled_quantiles_json, "
        "bucket_probs_json, evidence_ok, created_at, is_test) VALUES "
        "(?, ?, ?, 'ETH', 'ACE', 'FATALITIES', 'ok', ?, ?, ?, ?, CURRENT_TIMESTAMP, FALSE)",
        [srid, run_id, qid, json.dumps(trials),
         json.dumps({"0.1": 0, "0.5": 10, "0.9": 100, "0.99": 500}),
         json.dumps([0.3, 0.2, 0.2, 0.1, 0.1, 0.05, 0.05]), evidence_ok],
    )


def test_legacy_rule():
    assert legacy_evidence_ok(_july_trials(False)) is False
    assert legacy_evidence_ok(_july_trials(True, 2)) is True
    mixed = _july_trials(True, 1) + _july_trials(False, 2)
    assert legacy_evidence_ok(mixed) is False


def test_backfill_flags_the_july_run_and_is_idempotent(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        for i in range(10):
            _insert_forecast(con, "sibyl_1784113515141", f"JULY_{i}", _july_trials(False))
        _insert_forecast(con, "sibyl_later", "GOOD", _july_trials(True))
        _insert_forecast(con, "sibyl_new", "NEW", [], evidence_ok=True)  # already set
        assert backfill_evidence_ok(con) == 11
        flags = dict(con.execute(
            "SELECT question_id, evidence_ok FROM sibyl_forecasts"
        ).fetchall())
        assert all(flags[f"JULY_{i}"] is False for i in range(10))
        assert flags["GOOD"] is True and flags["NEW"] is True
        assert backfill_evidence_ok(con) == 0  # idempotent
    finally:
        con.close()


def test_advice_record_leaves_out_forecasts_without_evidence(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)
    from pythia.db.schema import connect
    from sibyl.advice import load_records

    con = connect(read_only=False)
    try:
        con.execute("INSERT INTO sibyl_runs (sibyl_run_id, is_test, created_at) "
                    "VALUES ('sibyl_1784113515141', FALSE, CURRENT_TIMESTAMP)")
        _insert_forecast(con, "sibyl_1784113515141", Q1, _july_trials(False))
        _insert_forecast(con, "sibyl_1784113515141", Q2, _july_trials(True))
        con.execute("CREATE TABLE IF NOT EXISTS resolutions (question_id TEXT, "
                    "horizon_m INTEGER, value DOUBLE)")
        for qid in (Q1, Q2):
            con.execute("INSERT INTO resolutions (question_id, horizon_m, value) "
                        "VALUES (?, 1, 12)", [qid])
        backfill_evidence_ok(con)
        assert {r.question_id for r in load_records(con)} == {Q2}
    finally:
        con.close()


def test_the_interpreter_bundle_leaves_them_out(tmp_path, monkeypatch):
    bundle = pytest.importorskip("scripts.ai_bundle.build_current_run_bundle")
    seed_db(tmp_path, monkeypatch)
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        _insert_forecast(con, "s", Q1, [], evidence_ok=False)
        _insert_forecast(con, "s", Q2, [], evidence_ok=True)
        assert bundle._sibyl_covered(con, [Q1, Q2]) == {Q2}
    finally:
        con.close()


# --- 7. floor --------------------------------------------------------------------

@pytest.mark.parametrize("probs", [
    [1.0, 0, 0, 0, 0, 0, 0],
    [0.5, 0.5, 0, 0, 0, 0],
    [0.001, 0.002, 0.3, 0.3, 0.397, 0.0],
    [1 / 7] * 7,
])
def test_floor_leaves_no_bucket_below_it(probs):
    out = apply_bucket_floor(probs, 0.005)
    assert sum(out) == pytest.approx(1.0, abs=1e-12)
    assert min(out) >= 0.005 - 1e-12
    # Order is kept among buckets that were above the floor.
    big = [i for i, p in enumerate(probs) if p > 0.01]
    assert np.argsort([out[i] for i in big]).tolist() == np.argsort([probs[i] for i in big]).tolist()


def test_floor_is_idempotent():
    once = apply_bucket_floor([0.9, 0.1, 0, 0, 0, 0])
    assert apply_bucket_floor(once) == pytest.approx(once)


# --- 8. salvage --------------------------------------------------------------------

def test_a_step_that_fails_after_a_valid_one_keeps_the_last_belief(monkeypatch):
    monkeypatch.setattr(sibyl_tools, "fetch_via_brave_search", _ok_pack)
    calls = {"n": 0}

    def model(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            return make_search_response(), _usage(), ""
        return "", _usage(), "provider error"

    trial = _trial(model)
    assert trial.ok is True
    assert trial.degraded == "model_step_failed"
    assert trial.quantiles is not None
    assert trial.to_dict()["degraded"] == "model_step_failed"


def test_a_trial_with_no_valid_step_is_discarded():
    trial = _trial(lambda p: ("", _usage(), "provider error"))
    assert trial.ok is False
    assert trial.error == "model_step_failed"


# --- 9. seed label -------------------------------------------------------------------

def test_seed_without_anchor_does_not_claim_a_base_rate():
    belief = initial_belief_from_anchor(None)
    assert "Seeded from" not in belief.baserate_reconciliation
    assert "No reference" in belief.baserate_reconciliation


# --- 10. question wording ----------------------------------------------------------------

@pytest.mark.parametrize("hz", ["FL", "TC", "DR"])
def test_pa_wording_names_the_sources_that_resolve_it(hz):
    from scripts.create_questions_from_triage import _build_question_wording

    text = _build_question_wording("PHL", hz, "PA", date(2026, 11, 1), date(2027, 4, 30))
    assert "EM-DAT" not in text
    assert "IFRC GO" in text and "IDMC" in text
    assert "not resolved" in text


# --- 11. month-position check ------------------------------------------------------------

def test_month_position_check_reads_members_only(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)
    from pythia.db.schema import connect
    from pythia.tools.generate_calibration_advice import (
        _compute_month_position_bias,
        _month_position_n_buckets,
    )

    assert _month_position_n_buckets("FATALITIES") == 7
    assert _month_position_n_buckets("EVENT_OCCURRENCE") == 5

    con = connect(read_only=False)
    try:
        def put(model, m1, m6):
            for month, vec in ((1, m1), (6, m6)):
                for b, p in enumerate(vec, start=1):
                    con.execute(
                        "INSERT INTO forecasts_raw (run_id, question_id, model_name, "
                        "month_index, bucket_index, probability) VALUES "
                        "(?, ?, ?, ?, ?, ?)", [STANDARD_RUN_ID, Q1, model, month, b, p])

        flat = [1 / 7] * 7
        # A member that does separate the horizons.
        put("gpt-6-sol", [0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.6])
        # Flat aggregates and references that must not be averaged in.
        for name in ("sibyl", "ensemble_mean_v2", "__ext_climatology"):
            put(name, flat, flat)
        out = _compute_month_position_bias(con, "ACE", "FATALITIES", n_buckets=7)
        assert out is not None
        assert out["jsd_m1_m6"] > 0.2
        assert set(out["by_month_b5"]) == {1, 6}
    finally:
        con.close()
