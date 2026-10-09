# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Failure types in Sibyl's post-mortems (Oct 2026, review Part 2).

Labels validated against the enum, unknown ones kept raw, the note prompt
carrying the trials' record and the code's inside/outside verdict, rates
counted per question with a share only from ten, old notes re-labelled
inside the cap, and an older database reading nothing. Model mocked.
"""

from __future__ import annotations

import json
from typing import List

import pytest

import sibyl.config as sibyl_config
from sibyl import postmortem as pm

pytestmark = pytest.mark.db


# --- validation ------------------------------------------------------------------

def test_the_enum_is_the_single_source():
    assert len(pm.FAILURE_TYPES) == 17
    assert list(pm.FAILURE_TYPES)[0] == "resolver_misread"
    assert "unforeseeable" in pm.FAILURE_TYPES and "no_fault" in pm.FAILURE_TYPES
    text = pm._failure_type_lines()
    assert all(f"- {k}:" in text for k in pm.FAILURE_TYPES)


def test_labels_are_validated_and_unknown_ones_kept_raw():
    out = pm.validate_labels({
        "failure_types": ["Double_Counted", "made_up", "tails_too_thin", "double_counted",
                          "thin_research", "no_fault"],
        "label_evidence": {"double_counted": "E3", "made_up": "x", "tails_too_thin": "q"},
    })
    assert out["status"] == "labelled"
    assert out["failure_types"] == ["double_counted", "tails_too_thin", "thin_research"]
    assert out["failure_types_raw"] == ["made_up"]
    assert out["label_evidence"] == {"double_counted": "E3", "tails_too_thin": "q"}


@pytest.mark.parametrize("parsed", [None, {}, {"failure_types": ["nonsense"]},
                                    {"failure_types": "also nonsense"}])
def test_a_note_with_no_valid_label_is_unlabelled(parsed):
    out = pm.validate_labels(parsed)
    assert out["status"] == "unlabelled" and out["failure_types"] == []


# --- the prompt's own facts -------------------------------------------------------

def test_the_code_states_whether_each_outcome_fell_inside_the_raw_range():
    ctx = {"metric": "FATALITIES",
           "raw_quantiles": {"1": {"0.05": 2, "0.95": 40}, "6": {"0.05": 1, "0.95": 10}},
           "raw_vectors": {"1": [0.1, 0.9]}, "reference": {"1": [0.5, 0.5]},
           "final": {"1": [0.3, 0.7]}}
    lines = pm.month_lines(ctx, [12, 55], [1, 6])
    assert "outcome inside" in lines[0] and "bucket 3" in lines[0]
    assert "reference [0.50, 0.50]" in lines[0] and "published [0.30, 0.70]" in lines[0]
    assert "outcome ABOVE" in lines[1]


def test_the_trials_block_drops_whole_ledger_items_and_says_so():
    trials = [{"lane": "A", "plan": {"resolver": "ACLED all types"},
               "reconciliation": "above the reference",
               "ledger": [{"id": f"E{i}", "date": "2026-09-01", "tier": 2,
                           "kind": "measurement", "quote": "x" * 80, "direction": "higher"}
                          for i in range(1, 30)]}]
    text, dropped = pm.trial_text(trials, 1200)
    assert dropped > 0 and len(text) < 1300
    assert f"({dropped} ledger item(s) left out for length)" in text
    assert "plan resolver: ACLED all types" in text and "[E1]" in text


# --- in a database ----------------------------------------------------------------

def _seed(tmp_path, monkeypatch, n, hazard="ACE", metric="FATALITIES"):
    from tests.sibyl_test_utils import HS_RUN_ID, seed_db

    seed_db(tmp_path, monkeypatch)
    from pythia.db.schema import connect

    con = connect(read_only=False)
    con.execute("CREATE TABLE IF NOT EXISTS resolutions (question_id TEXT, horizon_m INTEGER, "
                "value DOUBLE, is_test BOOLEAN DEFAULT FALSE)")
    con.execute("INSERT INTO sibyl_runs (sibyl_run_id, created_at, is_test) "
                "VALUES ('sr1', CURRENT_TIMESTAMP, FALSE)")
    q = {"0.1": 1, "0.25": 3, "0.5": 8, "0.75": 20, "0.9": 50, "0.95": 80, "0.99": 200}
    trials = [{"lane": "A", "belief_trace": [{"belief": {
        "plan": {"resolver": {"status": "done", "finding": "ACLED, all event types"}},
        "baserate_reconciliation": "held near the reference"}}],
        "ledger": [{"id": "E1", "date": "2026-07-20", "tier": 1, "kind": "measurement",
                    "quote": "41 killed in July", "direction": "higher"}]}]
    raw = {"vectors": {"1": [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05]},
           "quantiles": {"1": {"0.05": 1, "0.5": 8, "0.95": 80}}}
    for i in range(n):
        qid = f"X{i:02d}_{hazard}_{metric}_2026-08"
        con.execute(
            "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric, "
            "target_month, window_start_date, wording, status, track) VALUES "
            "(?, ?, 'ETH', ?, ?, '2027-01', DATE '2026-08-01', 'How many?', 'active', 1)",
            [qid, HS_RUN_ID, hazard, metric])
        con.execute(
            "INSERT INTO sibyl_forecasts (sibyl_run_id, run_id, question_id, hazard_code, "
            "metric, status, pooled_quantiles_json, bucket_probs_json, trials_json, "
            "raw_by_month_json, evidence_ok, created_at) VALUES ('sr1', 'fc', ?, ?, ?, 'ok', ?, "
            "'[0.1,0.2,0.3,0.2,0.1,0.05,0.05]', ?, ?, TRUE, CURRENT_TIMESTAMP)",
            [qid, hazard, metric, json.dumps(q), json.dumps(trials), json.dumps(raw)])
        con.execute("INSERT INTO resolutions VALUES (?, 1, 12, FALSE)", [qid])
    return con


def _labelling_call(calls: List[str], labels=("double_counted",)):
    def call(prompt):
        calls.append(prompt)
        if prompt.startswith("Below are post-mortem notes"):
            return json.dumps({"lessons": []}), {"cost_usd": 0.1}, ""
        body = {"failure_types": list(labels), "label_evidence": {labels[0]: "E1"}}
        if not prompt.startswith("Below is a post-mortem"):
            body.update({"what_happened": "w", "forecast_vs_outcome": "too high",
                         "what_would_have_helped": "h", "general_lesson": "g"})
        return json.dumps(body), {"cost_usd": 0.1}, ""
    return call


def test_a_new_note_carries_its_labels_and_the_prompt_its_record(tmp_path, monkeypatch):
    con = _seed(tmp_path, monkeypatch, 2)
    try:
        calls: List[str] = []
        out = pm.run(con, as_of_month="2026-10", call=_labelling_call(calls), log=None)
        assert out["notes_written"] == 2 and out["notes_relabelled"] == 0
        prompt = calls[0]
        assert "plan resolver: ACLED, all event types" in prompt
        assert "reconciliation with the reference: held near the reference" in prompt
        assert "[E1] 2026-07-20 tier 1 measurement higher: 41 killed in July" in prompt
        assert "Month 1: outcome 12 (bucket 3); raw 0.05-0.95 range 1 to 80: outcome inside" in prompt
        assert "unforeseeable" in prompt and "never a fault" in prompt
        ftj, ver, note = con.execute(
            "SELECT failure_types_json, prompt_version, note_json FROM sibyl_postmortem_notes "
            "LIMIT 1").fetchone()
        assert ver == "pm_v2"
        assert json.loads(ftj)["failure_types"] == ["double_counted"]
        assert "failure_types" not in json.loads(note)
    finally:
        con.close()


def test_old_notes_are_relabelled_oldest_first_inside_the_cap(tmp_path, monkeypatch):
    con = _seed(tmp_path, monkeypatch, 4)
    try:
        for i in range(4):
            con.execute(
                "INSERT INTO sibyl_postmortem_notes (question_id, sibyl_run_id, iso3, "
                "hazard_code, metric, note_json, model, cost_usd, created_at) VALUES "
                "(?, 'sr1', 'ETH', 'ACE', 'FATALITIES', '{\"what_happened\": \"w\"}', 'm', 0, "
                "TIMESTAMP '2026-09-01 00:00:00' + INTERVAL (?) HOUR)",
                [f"X{i:02d}_ACE_FATALITIES_2026-08", 4 - i])
        monkeypatch.setattr(sibyl_config, "POSTMORTEM_CAP_USD", 0.15)
        calls: List[str] = []
        out = pm.run(con, as_of_month="2026-10", call=_labelling_call(calls), log=None)
        # The cap allows two calls ($0.10 each, stop once $0.15 is passed).
        assert out["notes_written"] == 0 and out["notes_relabelled"] == 2
        assert all(c.startswith("Below is a post-mortem") for c in calls)
        done = [r[0] for r in con.execute(
            "SELECT question_id FROM sibyl_postmortem_notes WHERE prompt_version = 'pm_v2' "
            "ORDER BY question_id").fetchall()]
        assert done == ["X02_ACE_FATALITIES_2026-08", "X03_ACE_FATALITIES_2026-08"]  # oldest
    finally:
        con.close()


def test_rates_are_counted_per_question_with_a_share_from_ten(tmp_path, monkeypatch):
    con = _seed(tmp_path, monkeypatch, 0)
    try:
        from pythia.db.schema import ensure_sibyl_measurement_tables

        ensure_sibyl_measurement_tables(con)

        def note(qid, srid, hz, labels, created):
            con.execute(
                "INSERT INTO sibyl_postmortem_notes (question_id, sibyl_run_id, hazard_code, "
                "metric, note_json, failure_types_json, prompt_version, created_at) VALUES "
                "(?, ?, ?, 'FATALITIES', '{}', ?, 'pm_v2', TIMESTAMP '2026-09-01' + "
                "INTERVAL (?) DAY)",
                [qid, srid, hz, json.dumps(pm.validate_labels({"failure_types": labels})),
                 created])

        for i in range(10):
            note(f"A{i}", "s1", "ACE", ["double_counted"] if i < 4 else ["no_fault"], 0)
        # A second note on one question: counted once, from the newest.
        note("A0", "s2", "ACE", ["thin_research"], 5)
        note("D1", "s1", "DR", ["thin_research"], 0)
        note("D2", "s1", "DR", [], 0)
        rates = pm.failure_rates(con)
        ace = rates["classes"]["ACE/FATALITIES"]
        assert ace["n_labelled_questions"] == 10 and ace["status"] == "ok"
        assert ace["counts"]["double_counted"] == 3 and ace["counts"]["thin_research"] == 1
        assert ace["shares"]["no_fault"] == pytest.approx(0.6)
        dr = rates["classes"]["DR/FATALITIES"]
        assert dr["n_labelled_questions"] == 1 and dr["n_unlabelled_questions"] == 1
        assert dr["status"] == "not_yet" and dr["shares"]["thin_research"] is None
        assert dr["counts"]["thin_research"] == 1
        assert rates["pooled"]["n_labelled_questions"] == 11
    finally:
        con.close()


def test_the_rates_reach_the_pooled_advice_row(tmp_path, monkeypatch):
    con = _seed(tmp_path, monkeypatch, 1)
    try:
        from pythia.db.schema import ensure_sibyl_calibration_advice_table

        ensure_sibyl_calibration_advice_table(con)
        con.execute("INSERT INTO sibyl_calibration_advice (as_of_month, hazard_code, metric, "
                    "scope, n_questions, advice, findings_json) VALUES "
                    "('2026-10', '*', '*', 'pooled', 1, '', '{\"gate\": \"x\"}')")
        pm.run(con, as_of_month="2026-10", call=_labelling_call([]), log=None)
        f = json.loads(con.execute(
            "SELECT findings_json FROM sibyl_calibration_advice").fetchone()[0])
        assert f["gate"] == "x"
        assert f["failure_types"]["pooled"]["counts"]["double_counted"] == 1
    finally:
        con.close()


def test_lessons_see_the_labels_and_trials_never_do(tmp_path, monkeypatch):
    con = _seed(tmp_path, monkeypatch, 8)
    try:
        calls: List[str] = []
        pm.run(con, as_of_month="2026-10", call=_labelling_call(calls), log=None)
        lessons_prompt = next(c for c in calls if c.startswith("Below are post-mortem notes"))
        assert "failure types: double_counted" in lessons_prompt
        from types import SimpleNamespace

        q = SimpleNamespace(question_id="Z", iso3="ETH", hazard_code="ACE",
                            metric="FATALITIES")
        block = pm.lessons_block_for(con, q, None)
        assert "double_counted" not in block
    finally:
        con.close()


def test_an_older_database_reads_no_rates():
    import duckdb

    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE sibyl_postmortem_notes (question_id TEXT, note_json TEXT)")
    out = pm.failure_rates(con)
    assert out["pooled"] is None and out["classes"] == {}
