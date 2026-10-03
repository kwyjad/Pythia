# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Part 6 of the Oct 2026 Sibyl work: measuring what Sibyl read and did.

The variant scorers against hand-computed values, the FL/TC two-part
scores, the pool-weight fit at 19 and 20 questions, the evidence table,
the process measures and the reference weight a run reads.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest

from pythia.web_research.types import EvidencePack, EvidenceSource

import sibyl.agent as sibyl_agent
import sibyl.config as sibyl_config
import sibyl.run as sibyl_run
import sibyl.tools as sibyl_tools
from sibyl import measure, score_variants as sv
from sibyl.spd import apply_bucket_floor
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


# --- 1. scorers against hand-computed values ------------------------------------

def test_score_vector_matches_hand_values():
    s = sv.score_vector([0.5, 0.3, 0.2], 1)
    assert s["brier"] == pytest.approx(0.25 + 0.49 + 0.04)
    assert s["log"] == pytest.approx(-math.log(0.3))
    # RPS: F = [0.5, 0.8], H = [0, 1]; ((0.5)^2 + (0.2)^2) / 2
    assert s["crps"] == pytest.approx(0.145)


def test_two_part_scores_with_a_record():
    s = sv.two_part_scores([0.4, 0.3, 0.2, 0.1], True, 2)
    assert s["occurrence_brier"] == pytest.approx((0.6 - 1.0) ** 2)
    # Non-zero buckets renormalised: [0.5, 1/3, 1/6], scored on index 1.
    assert s["conditional_brier"] == pytest.approx(0.25 + (2 / 3) ** 2 + (1 / 6) ** 2)
    assert s["conditional_log"] == pytest.approx(math.log(3))
    assert s["conditional_crps"] == pytest.approx(((0.5) ** 2 + (1 / 6) ** 2) / 2)


def test_two_part_scores_without_a_record():
    s = sv.two_part_scores([0.4, 0.3, 0.2, 0.1], False, None)
    assert s == {"occurrence_brier": pytest.approx(0.36)}


def test_mix_is_floored_and_normalised():
    v = sv.mix([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.5)
    assert sum(v) == pytest.approx(1.0)
    assert min(v) >= sibyl_config.BUCKET_FLOOR - 1e-12


# --- 2. pool-weight fit ------------------------------------------------------------

def _cases(n: int, *, ref_right: bool):
    ref = [0.9, 0.05, 0.05]
    raw = [0.05, 0.05, 0.9]
    j = 0 if ref_right else 2
    return {f"q{i}": [(ref, raw, j)] for i in range(n)}


def test_fit_at_19_questions_writes_no_weight():
    fit = sv.fit_pool_weight(_cases(19, ref_right=True))
    assert fit["status"] == "too_few"
    assert fit["weight"] is None and fit["n_questions"] == 19


def test_fit_at_20_questions_is_held_by_the_prior():
    # The reference is right every time: 0.75 is best, but 20 questions
    # against a prior worth 20 at 0.5 gives 0.625, which ties and keeps 0.5.
    fit = sv.fit_pool_weight(_cases(20, ref_right=True))
    assert fit["status"] == "fitted"
    assert fit["best_weight"] == 0.75
    assert fit["fitted_weight"] == pytest.approx(0.625)
    assert fit["weight"] == 0.5
    losses = {float(k): v for k, v in fit["losses"].items()}
    assert losses[0.75] < losses[0.5] < losses[0.25]


def test_fit_moves_once_the_evidence_outweighs_the_prior():
    fit = sv.fit_pool_weight(_cases(60, ref_right=True))
    assert fit["fitted_weight"] == pytest.approx((60 * 0.75 + 20 * 0.5) / 80)
    assert fit["weight"] == 0.75
    fit = sv.fit_pool_weight(_cases(60, ref_right=False))
    assert fit["weight"] == 0.25


# --- 3. end to end into a DB -----------------------------------------------------

FL_A = "PHL_FL_PA_2026-08"
FL_B = "BGD_FL_PA_2026-08"


def _seed_forecasts(con):
    con.execute("CREATE TABLE IF NOT EXISTS resolutions (question_id TEXT, horizon_m INTEGER, "
                "value DOUBLE, is_test BOOLEAN DEFAULT FALSE)")
    raw = {"vectors": {str(m): [0.5, 0.3, 0.1, 0.05, 0.03, 0.02, 0.0] for m in range(1, 7)},
           "quantiles": {}}
    ref = {"source": "stub", "by_month": {str(m): [0.2, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05]
                                          for m in range(1, 7)}}
    final = {str(m): [0.35, 0.3, 0.15, 0.08, 0.07, 0.03, 0.02] for m in range(1, 7)}
    con.execute(
        """
        INSERT INTO sibyl_forecasts (sibyl_run_id, run_id, question_id, hazard_code, metric,
            status, raw_by_month_json, reference_json, final_by_month_json, evidence_ok,
            created_at)
        VALUES ('sr1', 'fc', ?, 'ACE', 'FATALITIES', 'ok', ?, ?, ?, TRUE, CURRENT_TIMESTAMP)
        """,
        [Q1, json.dumps(raw), json.dumps(ref), json.dumps(final)],
    )
    con.execute("INSERT INTO resolutions (question_id, horizon_m, value) VALUES (?, 1, 3)", [Q1])
    # Two flood questions, same window. B is resolved at horizon 2, which
    # puts that calendar month in reach; A has no row there -> no record.
    pa = [0.4, 0.3, 0.2, 0.05, 0.03, 0.02]
    for qid, iso3 in ((FL_A, "PHL"), (FL_B, "BGD")):
        con.execute(
            """
            INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric,
                target_month, window_start_date, wording, status, track)
            VALUES (?, ?, ?, 'FL', 'PA', '2027-01', DATE '2026-08-01', 'w', 'active', 1)
            """,
            [qid, HS_RUN_ID, iso3],
        )
    con.execute(
        """
        INSERT INTO sibyl_forecasts (sibyl_run_id, run_id, question_id, hazard_code, metric,
            status, raw_by_month_json, reference_json, final_by_month_json, evidence_ok,
            created_at)
        VALUES ('sr1', 'fc', ?, 'FL', 'PA', 'ok', ?, ?, ?, TRUE, CURRENT_TIMESTAMP)
        """,
        [FL_A, json.dumps({"vectors": {str(m): pa for m in range(1, 7)}}),
         json.dumps({"by_month": {str(m): pa for m in range(1, 7)}}),
         json.dumps({str(m): pa for m in range(1, 7)})],
    )
    con.execute("INSERT INTO resolutions (question_id, horizon_m, value) VALUES (?, 1, 20000)", [FL_A])
    con.execute("INSERT INTO resolutions (question_id, horizon_m, value) VALUES (?, 2, 500)", [FL_B])


def test_score_variants_writes_both_series_and_the_two_part_rows(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        _seed_forecasts(con)
        counters = sv.score_variants(con, as_of_month="2026-10")
        assert counters["scored_raw"] == 2 and counters["scored_ref"] == 2

        raw_f = apply_bucket_floor([0.5, 0.3, 0.1, 0.05, 0.03, 0.02, 0.0])
        rows = dict(con.execute(
            "SELECT score_type, value FROM scores WHERE model_name = '__ext_sibyl_raw' "
            "AND question_id = ? AND run_id IS NULL", [Q1]).fetchall())
        expect = sv.score_vector(raw_f, 1)  # 3 deaths -> bucket "1-<5"
        assert rows == pytest.approx(expect)
        ref_brier = con.execute(
            "SELECT value FROM scores WHERE model_name = '__ext_sibyl_ref' AND question_id = ? "
            "AND score_type = 'brier'", [Q1]).fetchone()[0]
        assert ref_brier == pytest.approx(
            sv.score_vector([0.2, 0.3, 0.2, 0.1, 0.1, 0.05, 0.05], 1)["brier"])
        audit = con.execute(
            "SELECT COUNT(*) FROM baseline_scored_forecasts WHERE model_name LIKE '__ext_sibyl_%'"
        ).fetchone()[0]
        assert audit == 4

        two = con.execute(
            "SELECT horizon_m, series, score_type, value FROM sibyl_variant_scores "
            "WHERE question_id = ? ORDER BY 1, 2, 3", [FL_A]).fetchall()
        h1 = {(s, st): v for h, s, st, v in two if h == 1}
        h2 = {(s, st): v for h, s, st, v in two if h == 2}
        assert h1[("sibyl", "occurrence_brier")] == pytest.approx(0.16)  # record: (0.6 - 1)^2
        assert ("sibyl", "conditional_log") in h1
        assert h2 == {(s, "occurrence_brier"): pytest.approx(0.36)  # no record: 0.6^2
                      for s in ("sibyl", "sibyl_raw", "sibyl_ref")}
        assert not {h for h, *_ in two} - {1, 2}  # unreached months are not scored

        w = con.execute("SELECT status, weight FROM sibyl_pool_weights").fetchall()
        assert w == [("too_few", None)]

        # Idempotent: a rerun replaces, never duplicates.
        sv.score_variants(con, as_of_month="2026-10")
        assert con.execute(
            "SELECT COUNT(*) FROM sibyl_variant_scores WHERE question_id = ?", [FL_A]
        ).fetchone()[0] == len(two)
    finally:
        con.close()


def test_score_variants_on_a_db_without_sibyl_columns(tmp_path):
    import duckdb

    con = duckdb.connect(str(tmp_path / "old.duckdb"))
    con.execute("CREATE TABLE sibyl_forecasts (question_id TEXT, status TEXT)")
    counters = sv.score_variants(con, as_of_month="2026-10")
    assert counters["scored_raw"] == 0
    assert counters["pool_weight"]["status"] == "too_few"


# --- 4. reference weight a run reads ---------------------------------------------

def test_reference_weight_reads_the_fitted_row(tmp_path, monkeypatch):
    import duckdb

    from pythia.db.schema import ensure_sibyl_measurement_tables

    con = duckdb.connect(str(tmp_path / "w.duckdb"))
    assert measure.reference_weight(con, "2026-10") == (0.5, "fixed")  # no table
    ensure_sibyl_measurement_tables(con)
    con.execute("INSERT INTO sibyl_pool_weights (as_of_month, weight, status) "
                "VALUES ('2026-09', 0.75, 'fitted'), ('2026-11', 0.25, 'fitted'), "
                "('2026-10', NULL, 'too_few')")
    assert measure.reference_weight(con, "2026-10") == (0.75, "fitted")
    monkeypatch.setattr(sibyl_config, "REFERENCE_WEIGHT_MODE", "fixed")
    assert measure.reference_weight(con, "2026-10") == (0.5, "fixed")
    monkeypatch.setattr(sibyl_config, "REFERENCE_WEIGHT_MODE", "fitted")
    monkeypatch.setattr(sibyl_config, "BACKTEST_MODE", True)
    assert measure.reference_weight(con, "2026-10") == (0.5, "backtest")


# --- 5. process measures ----------------------------------------------------------

def test_process_measures_over_outcomes():
    t1 = SimpleNamespace(resolver_status="done", n_docs_read=3,
                         ledger=[{"date": "2026-09-01", "quote": "1,200 killed"},
                                 {"date": None, "quote": "about 50"}])
    t2 = SimpleNamespace(resolver_status="failed", n_docs_read=1,
                         ledger=[{"date": "2026-09-02", "quote": "fighting continued"}])
    floor = sibyl_config.BUCKET_FLOOR
    o1 = SimpleNamespace(trials=[t1, t2], final_by_month={1: [floor, 0.5, 0.5 - floor]},
                         raw_month1=[0.2, 0.3, 0.5], reference_month1=[0.2, 0.3, 0.5])
    o2 = SimpleNamespace(trials=[], final_by_month={}, raw_month1=None, reference_month1=None)
    m = measure.process_measures([o1, o2])
    assert m["share_resolver_done"] == 0.5
    assert m["docs_per_trial"] == 2.0
    assert m["share_ledger_dated_figure"] == pytest.approx(1 / 3)
    assert m["share_forecasts_at_floor"] == 1.0
    assert m["mean_jsd_from_reference"] == pytest.approx(0.0)
    assert measure.process_measures([]) == {k: None for k in m}


# --- 6. the evidence table, end to end -------------------------------------------

def _ok_pack(query, **kwargs):
    pack = EvidencePack(query=query, backend="brave", grounded=True)
    pack.sources = [EvidenceSource(title="R", url="https://news.example.com/r",
                                   summary="Clashes reported.", date="2026-09-20")]
    pack.debug = {"usage": {"cost_usd": 0.005}, "status_code": 200}
    return pack


def test_a_run_writes_evidence_rows_and_process_measures(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)
    monkeypatch.setattr(sibyl_config, "MIN_SEARCH_OK", 1)
    monkeypatch.setattr(sibyl_config, "MIN_DOCS_READ", 0)
    disable_submit_gate(monkeypatch)
    monkeypatch.setattr(sibyl_agent, "log_sibyl_call", lambda **kw: None)
    monkeypatch.setattr(sibyl_tools, "_sleep", lambda s: None)
    monkeypatch.setattr(sibyl_tools, "fetch_via_brave_search", _ok_pack)
    monkeypatch.setattr(sibyl_run, "build_reference", stub_reference)
    sibyl_tools.reset_run_state()

    def model_call(prompt):
        searched = "=== STEP 1: YOUR RESPONSE ===" in prompt
        return (make_submit_response() if searched else make_search_response("eth clashes")), \
            {"cost_usd": 0.1}, ""

    summary = sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=model_call)
    assert summary["n_forecast"] == 1
    assert summary["n_evidence_rows"] == 3  # one search in each of three trials

    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        rows = con.execute(
            "SELECT question_id, trial_index, step, tool, target, lane, http_status, ok, "
            "length(sha256), shown_text FROM sibyl_evidence ORDER BY trial_index").fetchall()
        assert [r[1] for r in rows] == [0, 1, 2]
        assert all(r[0] == Q1 and r[2] == 1 and r[3] == "brave_search" for r in rows)
        assert all(r[4] == "eth clashes" and r[5] == "news" and r[7] for r in rows)
        assert all(r[8] == 64 and "Clashes reported" in r[9] for r in rows)
        run = con.execute(
            "SELECT docs_per_trial, share_forecasts_at_floor, mean_jsd_from_reference, "
            "reference_weight, reference_weight_source FROM sibyl_runs").fetchone()
        assert run[0] == 0.0 and run[1] is not None and run[2] is not None
        assert run[3:] == (0.5, "fixed")
    finally:
        con.close()
    sibyl_tools.reset_run_state()


def test_evidence_row_caps_the_document_and_hashes_its_full_text(monkeypatch):
    import hashlib
    from datetime import datetime

    monkeypatch.setattr(sibyl_config, "EVIDENCE_DOC_MAX_CHARS", 10)
    tr = sibyl_tools.ToolResult(tool="fetch_url", ok=True, text="extracted", status_code=200,
                                doc_text="x" * 25)
    call = SimpleNamespace(action="fetch_url", action_input="https://a.example/p", options={})
    row = sibyl_agent.evidence_row(tr, call, step=2, call_index=1,
                                   retrieved_at=datetime(2026, 10, 3))
    assert row["doc_text"] == "x" * 10 and row["doc_chars"] == 25
    assert row["sha256"] == hashlib.sha256(b"x" * 25).hexdigest()
    assert row["shown_text"] == "extracted" and row["lane"] is None
