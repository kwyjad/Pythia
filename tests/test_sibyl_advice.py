# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl's calibration advice loop (sibyl/advice.py, sibyl.calibration.load_advice).

The generator measures Sibyl's own scored record per class, writes advice
only where a finding's 90% interval excludes its calibrated value and the
class holds 20 distinct questions, and the prompt shows that advice to half
the questions. Each test pins one part of that contract.
"""

from __future__ import annotations

import json
import math
import random
from datetime import date
from typing import List

import pytest

pytest.importorskip("duckdb")

from sibyl import advice as adv
from sibyl.advice import SibylRecord, build_advice_text, build_rows, diagnose

QUANTILES = {0.1: 10.0, 0.25: 20.0, 0.5: 50.0, 0.75: 100.0, 0.9: 200.0, 0.95: 300.0, 0.99: 600.0}
LEVELS = sorted(QUANTILES)


def _draw(u: float) -> float:
    """Inverse of the forecast CDF, interpolated in log1p between the knots."""
    knots = [(0.0, 0.0)] + [(lv, QUANTILES[lv]) for lv in LEVELS] + [(1.0, 1200.0)]
    for (p0, v0), (p1, v1) in zip(knots, knots[1:]):
        if u <= p1:
            t = (u - p0) / (p1 - p0)
            return math.expm1(math.log1p(v0) + t * (math.log1p(v1) - math.log1p(v0)))
    return 1200.0


def _records(n: int, *, scale: float = 1.0, months: int = 6, seed: int = 11,
             hazard: str = "ACE", metric: str = "FATALITIES") -> List[SibylRecord]:
    """n questions whose outcomes are drawn from the forecast itself (calibrated
    when scale == 1), each with *months* resolved months."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        ys = [scale * _draw(rng.random()) for _ in range(months)]
        out.append(SibylRecord(
            question_id=f"Q{i:03d}_{hazard}_{metric}", hazard_code=hazard, metric=metric,
            quantiles=dict(QUANTILES), outcomes=ys, zero_mass=0.0,
            base_median=50.0 * scale if scale != 1.0 else None,
        ))
    return out


# --- findings -> instructions ------------------------------------------------


def test_a_known_bias_yields_the_matching_instruction():
    recs = _records(24, scale=0.2)  # outcomes run far below the forecast
    diag = diagnose(recs)
    text = build_advice_text(diag, len(recs))
    assert "fell below your 10% quantile" in text
    assert "Bring q0.1 and q0.25 down" in text
    assert "outcomes ran below your median" in text
    assert "Lower q0.5" in text
    # the line carries its evidence: a count, and the number expected
    assert "In 24 resolved questions" in text and "expected" in text
    assert len(text) <= adv.ADVICE_MAX_CHARS


def test_a_calibrated_record_yields_no_instruction():
    recs = _records(40, scale=1.0, seed=11)
    diag = diagnose(recs)
    assert not any(
        diag[k].excludes(adv.CALIBRATED[k]) for k in adv.CALIBRATED if diag[k].value is not None
    ), {k: diag[k].to_dict() for k in adv.CALIBRATED}
    assert build_advice_text(diag, len(recs)) == ""


def test_an_upper_tail_too_low_asks_to_raise_it():
    recs = _records(30, scale=4.0)
    text = build_advice_text(diagnose(recs), 30)
    assert "rose above your 90% quantile" in text and "Raise q0.9 and q0.95" in text


def test_anchor_departure_that_cost_accuracy_is_reported():
    # Base-rate median sits on the outcomes; Sibyl's median does not.
    recs = []
    for i in range(25):
        recs.append(SibylRecord(
            question_id=f"Q{i}", hazard_code="ACE", metric="FATALITIES",
            quantiles=dict(QUANTILES), outcomes=[300.0 + i, 310.0, 290.0],
            zero_mass=0.0, base_median=300.0,
        ))
    diag = diagnose(recs)
    assert diag["anchor_departure"].value == 0.0
    text = build_advice_text(diag, 25)
    assert "closer to the outcome than the base-rate median in only 0" in text


# --- gating -----------------------------------------------------------------


def test_nineteen_questions_give_no_advice_and_twenty_do():
    rows19 = build_rows(_records(19, scale=0.2), {}, as_of_month="2026-11")
    rows20 = build_rows(_records(20, scale=0.2), {}, as_of_month="2026-11")
    group19 = next(r for r in rows19 if r["scope"] == "group")
    group20 = next(r for r in rows20 if r["scope"] == "group")
    assert group19["advice"] == "" and group19["findings"]["gate"] == "19 of 20 resolved questions"
    assert group20["advice"] != ""
    # The findings row is written even when the advice is empty.
    assert group19["findings"]["diagnostics"]["below_q10"]["n_questions"] == 19


def test_pooled_row_stands_in_across_classes():
    recs = _records(12, scale=0.2, hazard="ACE", metric="FATALITIES") + _records(
        12, scale=0.2, hazard="DR", metric="PHASE3PLUS_IN_NEED", seed=5
    )
    rows = build_rows(recs, {}, as_of_month="2026-11")
    groups = [r for r in rows if r["scope"] == "group"]
    pooled = next(r for r in rows if r["scope"] == "pooled")
    assert all(r["advice"] == "" for r in groups)
    assert pooled["hazard_code"] == pooled["metric"] == "*"
    assert pooled["n_questions"] == 24 and pooled["advice"] != ""


def test_a_blocked_class_gets_no_advice_and_leaves_the_pool():
    rows = build_rows(_records(25, scale=0.2), {}, as_of_month="2026-11",
                      blocked=[("ACE", "FATALITIES")])
    group = next(r for r in rows if r["scope"] == "group")
    pooled = next(r for r in rows if r["scope"] == "pooled")
    assert group["advice"] == "" and "blocked" in group["findings"]["gate"]
    assert pooled["n_questions"] == 0 and pooled["advice"] == ""


def test_advice_text_never_names_a_model_an_ensemble_or_a_country():
    recs = _records(30, scale=0.2)
    recs += [SibylRecord(question_id=f"Z{i}", hazard_code="ACE", metric="FATALITIES",
                         quantiles=dict(QUANTILES), outcomes=[0.0, 0.0],
                         zero_mass=0.0, base_median=0.0) for i in range(10)]
    scores = {r.question_id: {"sibyl": {"brier": 0.9}, "ensemble_mean_v2": {"brier": 0.1},
                              "__ext_climatology": {"brier": 0.2}} for r in recs}
    rows = build_rows(recs, scores, as_of_month="2026-11")
    text = " ".join(r["advice"] for r in rows).lower()
    assert text  # something was said, so the check is not vacuous
    for banned in ("ensemble", "sibyl", "climatology", "claude", "opus", "gpt", "gemini",
                   "standard track", "mean_v2", "ethiopia", "somalia", " eth", " som"):
        assert banned not in text, banned
    # paired skill is measured and kept out of the text
    pooled = next(r for r in rows if r["scope"] == "pooled")
    assert pooled["findings"]["paired_skill"]["ensemble_mean_v2"]["brier"]["n_questions"] == 40


# --- the record: test rows, superseded runs, the question as the unit --------


def _seed_record_db(tmp_path, monkeypatch):
    url = f"duckdb:///{tmp_path / 'advice.duckdb'}"
    monkeypatch.setenv("PYTHIA_DB_URL", url)
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
    from pythia.db.schema import connect, ensure_schema

    ensure_schema()
    con = connect(read_only=False)
    con.execute(
        "CREATE TABLE IF NOT EXISTS resolutions (question_id TEXT, horizon_m INTEGER, "
        "observed_month TEXT, value DOUBLE, source_snapshot_ym TEXT, source_desc TEXT, "
        "created_at TIMESTAMP DEFAULT now(), is_test BOOLEAN DEFAULT FALSE)"
    )

    def q(qid, is_test=False):
        con.execute(
            "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric, "
            "target_month, window_start_date, status, is_test) VALUES "
            "(?, 'hs', 'ETH', 'ACE', 'FATALITIES', '2027-01', DATE '2026-08-01', 'active', ?)",
            [qid, is_test],
        )

    def run(srid, created, is_test=False):
        con.execute(
            "INSERT INTO sibyl_runs (sibyl_run_id, created_at, is_test) VALUES (?, ?, ?)",
            [srid, created, is_test],
        )

    def fc(srid, qid, median, is_test=False, status="ok"):
        qs = {str(k): (v * median / 50.0) for k, v in QUANTILES.items()}
        con.execute(
            "INSERT INTO sibyl_forecasts (sibyl_run_id, run_id, question_id, hazard_code, "
            "metric, status, pooled_quantiles_json, bucket_probs_json, trials_json, "
            "created_at, is_test, evidence_ok) VALUES (?, 'fc', ?, 'ACE', 'FATALITIES', ?, ?, ?, ?, "
            "CURRENT_TIMESTAMP, ?, TRUE)",
            [srid, qid, status, json.dumps(qs), json.dumps([0.0] * 7),
             json.dumps([{"perspective": "Base-rate-weighted perspective: x", "quantiles": qs}]),
             is_test],
        )

    return con, q, run, fc


def test_test_rows_and_superseded_runs_are_ignored(tmp_path, monkeypatch):
    con, q, run, fc = _seed_record_db(tmp_path, monkeypatch)
    try:
        q("A")
        q("B")
        q("C")
        q("T", is_test=True)
        run("old", "2026-08-01 06:00")
        run("new", "2026-09-01 06:00")
        run("testrun", "2026-09-15 06:00", is_test=True)
        fc("old", "A", median=5.0)
        fc("new", "A", median=500.0)          # supersedes the August forecast
        fc("testrun", "B", median=1.0, is_test=True)  # a test run only
        fc("new", "C", median=50.0)
        fc("testrun", "C", median=1.0, is_test=True)  # later, but a test run
        fc("new", "T", median=50.0)           # a test question
        for qid in ("A", "B", "C", "T"):
            for h in range(1, 7):
                con.execute(
                    "INSERT INTO resolutions (question_id, horizon_m, observed_month, value) "
                    "VALUES (?, ?, '2026-08', 40)", [qid, h],
                )
        recs = {r.question_id: r for r in adv.load_records(con, "2026-10")}
    finally:
        con.close()
    assert set(recs) == {"A", "C"}
    assert recs["A"].sibyl_run_id == "new" and recs["A"].quantiles[0.5] == pytest.approx(500.0)
    assert recs["C"].sibyl_run_id == "new"
    # six resolved months of one question are one question
    assert len(recs["A"].outcomes) == 6
    rows = build_rows(list(recs.values()), {}, as_of_month="2026-10")
    assert next(r for r in rows if r["scope"] == "group")["n_questions"] == 2


def test_generate_on_a_thin_record_writes_findings_and_no_advice(tmp_path, monkeypatch):
    con, q, run, fc = _seed_record_db(tmp_path, monkeypatch)
    try:
        for i in range(6):
            q(f"Q{i}")
        run("r", "2026-09-01 06:00")
        for i in range(6):
            fc("r", f"Q{i}", median=50.0)
            con.execute("INSERT INTO resolutions (question_id, horizon_m, observed_month, value) "
                        "VALUES (?, 1, '2026-08', 60)", [f"Q{i}"])
        rows = adv.generate(con, as_of_month="2026-10")
        stored = con.execute(
            "SELECT hazard_code, metric, scope, n_questions, advice, findings_json "
            "FROM sibyl_calibration_advice ORDER BY scope"
        ).fetchall()
    finally:
        con.close()
    assert len(rows) == 2 and all(r["advice"] == "" for r in rows)
    assert [(s[2], s[3], s[4]) for s in stored] == [("group", 6, ""), ("pooled", 6, "")]
    assert json.loads(stored[1][5])["arm_comparison"]["status"] == "not yet"


# --- loading into the prompt -------------------------------------------------


def _advice_table(con, rows):
    from pythia.db.schema import ensure_sibyl_calibration_advice_table

    ensure_sibyl_calibration_advice_table(con)
    for month, hz, m, text in rows:
        con.execute(
            "INSERT INTO sibyl_calibration_advice (as_of_month, hazard_code, metric, scope, "
            "n_questions, advice, findings_json, advice_version) VALUES (?, ?, ?, ?, 20, ?, '{}', 'v')",
            [month, hz, m, "pooled" if hz == "*" else "group", text],
        )


def test_load_advice_class_first_then_pooled_and_never_from_the_future(tmp_path, monkeypatch):
    import duckdb

    from sibyl.calibration import load_advice

    con = duckdb.connect(str(tmp_path / "a.duckdb"))
    _advice_table(con, [
        ("2026-09", "ACE", "FATALITIES", "old class advice"),
        ("2026-10", "ACE", "FATALITIES", ""),            # class re-measured, now gated
        ("2026-10", "*", "*", "pooled advice"),
        ("2026-10", "DR", "PHASE3PLUS_IN_NEED", "drought advice"),
        ("2026-12", "*", "*", "future advice"),
    ])
    got = load_advice("DR", "PHASE3PLUS_IN_NEED", date(2026, 11, 1), con=con, backtest=False)
    assert got.text == "drought advice" and got.scope == "group"
    # the newest month decides: last month's class row does not outrank it
    got = load_advice("ACE", "FATALITIES", date(2026, 11, 1), con=con, backtest=False)
    assert got.text == "pooled advice" and got.as_of_month == "2026-10"
    # a row dated after the as-of month is not loaded
    got = load_advice("ACE", "FATALITIES", "2026-09", con=con, backtest=False)
    assert got.text == "old class advice"
    assert load_advice("ACE", "FATALITIES", "2026-08", con=con, backtest=False) is None
    # backtest loads nothing
    assert load_advice("DR", "PHASE3PLUS_IN_NEED", date(2026, 11, 1), con=con, backtest=True) is None
    # a blocked class gets nothing
    monkeypatch.setenv("PYTHIA_ADVICE_BLOCK_GROUPS", "DR/PHASE3PLUS_IN_NEED")
    assert load_advice("DR", "PHASE3PLUS_IN_NEED", date(2026, 11, 1), con=con, backtest=False) is None
    con.close()


def test_load_advice_without_a_table_is_none(tmp_path):
    import duckdb

    from sibyl.calibration import load_advice

    con = duckdb.connect(str(tmp_path / "empty.duckdb"))
    assert load_advice("ACE", "FATALITIES", "2026-11", con=con, backtest=False) is None
    con.close()


def test_arm_is_stable_and_splits_the_questions():
    ids = [f"Q{i}_ACE_FATALITIES_2026-11" for i in range(400)]
    arms = [adv.advice_arm(q, 0.5) for q in ids]
    assert arms == [adv.advice_arm(q, 0.5) for q in ids]
    assert 150 < arms.count("no_advice") < 250
    assert all(adv.advice_arm(q, 0.0) == "advice" for q in ids[:20])
    assert all(adv.advice_arm(q, 1.0) == "no_advice" for q in ids[:20])
    # independent of the standard track's split, which hashes the bare id
    import hashlib

    std = ["no_advice" if int(hashlib.sha1(q.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < 0.5
           else "advice" for q in ids]
    assert std != arms


def test_arm_comparison_says_not_yet_below_ten_per_arm():
    scores = {f"Q{i}": {"brier": 0.5, "crps": 0.1} for i in range(15)}
    arms = {f"Q{i}": ("advice" if i < 9 else "no_advice") for i in range(15)}
    assert adv.arm_comparison(scores, arms)["status"] == "not yet"
    scores = {f"Q{i}": {"brier": 0.5, "crps": 0.1} for i in range(20)}
    arms = {f"Q{i}": ("advice" if i < 10 else "no_advice") for i in range(20)}
    out = adv.arm_comparison(scores, arms)
    assert out["status"] == "ok" and out["brier"]["advice"]["value"] == pytest.approx(0.5)


# --- end to end through the run ----------------------------------------------


@pytest.mark.parametrize("share, expect_arm, expect_section", [
    (0.0, "advice", True),
    (1.0, "no_advice", False),
])
def test_run_shows_advice_only_in_the_advice_arm(tmp_path, monkeypatch, share, expect_arm,
                                                  expect_section):
    import sibyl.agent as sibyl_agent
    import sibyl.run as sibyl_run
    from tests.sibyl_test_utils import (
        HS_RUN_ID, Q1, disable_evidence_gate, make_submit_response, seed_db, stub_base_rate,
    stub_reference,
    )

    seed_db(tmp_path, monkeypatch)
    disable_evidence_gate(monkeypatch)
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        _advice_table(con, [(date.today().strftime("%Y-%m"), "*", "*",
                             "- In 24 resolved questions outcomes ran above your median.")])
    finally:
        con.close()

    prompts: List[str] = []

    def fake_model_call(prompt: str):
        prompts.append(prompt)
        return make_submit_response(), {"cost_usd": 0.01}, ""

    monkeypatch.setattr(sibyl_run, "build_reference", stub_reference)
    monkeypatch.setattr(sibyl_agent, "log_sibyl_call", lambda **kwargs: None)
    monkeypatch.setattr(sibyl_run, "ADVICE_EXPERIMENT_SHARE", share)
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=fake_model_call)

    assert prompts
    assert all(("=== YOUR TRACK RECORD ===" in p) is expect_section for p in prompts)
    con = connect(read_only=False)
    try:
        arm, month, base = con.execute(
            "SELECT advice_arm, advice_as_of_month, base_rate_json FROM sibyl_forecasts "
            "WHERE question_id = ?", [Q1],
        ).fetchone()
    finally:
        con.close()
    assert arm == expect_arm
    assert (month is not None) is expect_section
    assert json.loads(base)["anchor_quantiles"] is not None


def test_run_without_any_advice_stores_no_arm(tmp_path, monkeypatch):
    import sibyl.agent as sibyl_agent
    import sibyl.run as sibyl_run
    from tests.sibyl_test_utils import (
        HS_RUN_ID, Q1, disable_evidence_gate, make_submit_response, seed_db, stub_base_rate,
    stub_reference,
    )

    seed_db(tmp_path, monkeypatch)
    disable_evidence_gate(monkeypatch)
    prompts: List[str] = []
    monkeypatch.setattr(sibyl_run, "build_reference", stub_reference)
    monkeypatch.setattr(sibyl_agent, "log_sibyl_call", lambda **kwargs: None)
    sibyl_run.run_sibyl(
        HS_RUN_ID, n_questions=1,
        model_call=lambda p: (prompts.append(p) or make_submit_response(), {"cost_usd": 0.01}, ""),
    )
    assert prompts and not any("YOUR TRACK RECORD" in p for p in prompts)
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        assert con.execute(
            "SELECT advice_arm FROM sibyl_forecasts WHERE question_id = ?", [Q1]
        ).fetchone()[0] is None
    finally:
        con.close()
