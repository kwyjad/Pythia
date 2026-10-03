# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Part 6 of the Oct 2026 Sibyl work: post-mortems and lessons.

The lessons gate (cases, countries, years, length), notes written once per
resolved question, lessons only from 8 notes and only when a note is new,
the cost cap, and the block reaching the prompt only in the track-record
arm and never in backtest. The model is mocked throughout.
"""

from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from typing import List

import pytest

import sibyl.config as sibyl_config
from sibyl import postmortem as pm

pytestmark = pytest.mark.db

TERMS = ["Ethiopia", "Somalia", "South Sudan"]


# --- 1. the lessons gate --------------------------------------------------------

def test_a_lesson_needs_three_cited_cases_from_the_notes():
    ids = ["A", "B", "C", "D"]
    assert pm.lesson_problem("Read the latest situation report.", ["A", "B"], ids,
                             country_terms=TERMS).startswith("rests on 2")
    # A case that is not one of the notes does not count.
    assert pm.lesson_problem("Read the latest situation report.", ["A", "B", "Z"], ids,
                             country_terms=TERMS).startswith("rests on 2")
    assert pm.lesson_problem("Read the latest situation report.", ["A", "B", "C"], ids,
                             country_terms=TERMS) is None


@pytest.mark.parametrize("text, why", [
    ("Fighting in Ethiopia ran above the base rate.", "names a country (Ethiopia)"),
    ("Watch the ETH ceasefire.", "names a country code (ETH)"),
    ("The 2024 floods were larger than expected.", "names a year"),
    ("Do not trust south sudan figures early.", "names a country (South Sudan)"),
])
def test_a_lesson_naming_a_country_or_a_year_is_refused(text, why):
    assert pm.lesson_problem(text, ["A", "B", "C"], ["A", "B", "C"], iso3s=["ETH"],
                             country_terms=TERMS) == why


def test_gate_keeps_the_good_lessons_and_cuts_by_whole_lesson(monkeypatch):
    ids = ["A", "B", "C"]
    proposed = [
        {"lesson": "Weigh official casualty updates above media tallies.", "cases": ids},
        {"lesson": "Somalia data lag.", "cases": ids},
        {"lesson": "x" * 80, "cases": ids},
        "not a dict",
    ]
    kept, rejected, text = pm.gate_lessons(proposed, ids, country_terms=TERMS, max_chars=90)
    assert [k["lesson"] for k in kept] == ["Weigh official casualty updates above media tallies."]
    assert {r["reason"] for r in rejected} == {"names a country (Somalia)", "over the length limit"}
    assert text == "- Weigh official casualty updates above media tallies. (seen in 3 resolved questions)"


# --- 2. notes and lessons in a DB -------------------------------------------------

def _seed(tmp_path, monkeypatch, n: int):
    from tests.sibyl_test_utils import HS_RUN_ID, seed_db

    seed_db(tmp_path, monkeypatch)
    from pythia.db.schema import connect

    con = connect(read_only=False)
    con.execute("CREATE TABLE IF NOT EXISTS resolutions (question_id TEXT, horizon_m INTEGER, "
                "value DOUBLE, is_test BOOLEAN DEFAULT FALSE)")
    con.execute("INSERT INTO sibyl_runs (sibyl_run_id, created_at, is_test) "
                "VALUES ('sr1', CURRENT_TIMESTAMP, FALSE)")
    q = {"0.1": 1, "0.25": 3, "0.5": 8, "0.75": 20, "0.9": 50, "0.95": 80, "0.99": 200}
    for i in range(n):
        qid = f"X{i:02d}_ACE_FATALITIES_2026-08"
        con.execute(
            "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric, "
            "target_month, window_start_date, wording, status, track) VALUES "
            "(?, ?, ?, 'ACE', 'FATALITIES', '2027-01', DATE '2026-08-01', 'How many?', "
            "'active', 1)", [qid, HS_RUN_ID, "ETH" if i % 2 else "SOM"])
        con.execute(
            "INSERT INTO sibyl_forecasts (sibyl_run_id, run_id, question_id, hazard_code, "
            "metric, status, pooled_quantiles_json, bucket_probs_json, trials_json, "
            "evidence_ok, created_at) VALUES ('sr1', 'fc', ?, 'ACE', 'FATALITIES', 'ok', ?, "
            "'[0.1,0.2,0.3,0.2,0.1,0.05,0.05]', '[]', TRUE, CURRENT_TIMESTAMP)",
            [qid, json.dumps(q)])
        con.execute("INSERT INTO resolutions VALUES (?, 1, 12, FALSE)", [qid])
    return con


def _note_call(calls: List[str]):
    def call(prompt):
        calls.append(prompt)
        if prompt.startswith("Below are post-mortem notes"):
            ids = [ln[1:ln.index("]")] for ln in prompt.splitlines() if ln.startswith("[")]
            return json.dumps({"lessons": [
                {"lesson": "Check whether a lull follows a ceasefire before moving the median.",
                 "cases": ids[:3]},
                {"lesson": "One case only.", "cases": ids[:1]},
            ]}), {"cost_usd": 0.10}, ""
        return json.dumps({
            "what_happened": "Deaths stayed near the reference.",
            "forecast_vs_outcome": "about right",
            "what_would_have_helped": "Nothing more.",
            "general_lesson": "The reference was a good start.",
        }), {"cost_usd": 0.10}, ""
    return call


def test_notes_are_written_once_and_lessons_wait_for_eight(tmp_path, monkeypatch):
    con = _seed(tmp_path, monkeypatch, 7)
    try:
        calls: List[str] = []
        out = pm.run(con, as_of_month="2026-10", call=_note_call(calls), log=None)
        assert out["notes_written"] == 7 and out["lessons"] == []
        assert "Outcome by window month: month 1: 12" in calls[0]
        # A rerun writes nothing new.
        out = pm.run(con, as_of_month="2026-10", call=_note_call(calls), log=None)
        assert out["notes_written"] == 0 and len(calls) == 7
    finally:
        con.close()


def test_lessons_at_eight_notes_keep_only_supported_ones(tmp_path, monkeypatch):
    con = _seed(tmp_path, monkeypatch, 8)
    try:
        calls: List[str] = []
        out = pm.run(con, as_of_month="2026-10", call=_note_call(calls), log=None)
        assert out["notes_written"] == 8
        assert out["lessons"] == [{"hazard_code": "ACE", "metric": "FATALITIES", "version": 1,
                                   "n_kept": 1, "n_rejected": 1}]
        text, rejected = con.execute(
            "SELECT lessons_text, rejected_json FROM sibyl_lessons").fetchone()
        assert text.startswith("- Check whether a lull follows a ceasefire")
        assert "rests on 1" in rejected
        # No new note -> no new version.
        out = pm.run(con, as_of_month="2026-11", call=_note_call(calls), log=None)
        assert out["lessons"] == []
        assert pm.load_lessons_text(con, "ACE", "FATALITIES", backtest=False).startswith("- Check")
        assert pm.load_lessons_text(con, "ACE", "FATALITIES", backtest=True) == ""
    finally:
        con.close()


def test_the_cap_stops_the_notes(tmp_path, monkeypatch):
    con = _seed(tmp_path, monkeypatch, 8)
    monkeypatch.setattr(sibyl_config, "POSTMORTEM_CAP_USD", 0.25)
    try:
        calls: List[str] = []
        out = pm.run(con, as_of_month="2026-10", call=_note_call(calls), log=None)
        assert out["notes_written"] == 3 and len(calls) == 3  # 0.30 spent, then stop
        assert out["lessons"] == []
    finally:
        con.close()


def test_analogues_put_the_same_country_first(tmp_path, monkeypatch):
    con = _seed(tmp_path, monkeypatch, 6)
    try:
        pm.write_notes(con, call=_note_call([]))
        q = SimpleNamespace(question_id="X01_ACE_FATALITIES_2026-08", iso3="ETH",
                            hazard_code="ACE", metric="FATALITIES")
        got = pm.load_analogues(con, q, k=4, backtest=False)
        assert len(got) == 4
        assert [a["iso3"] for a in got[:2]] == ["ETH", "ETH"]
        assert q.question_id not in {a["question_id"] for a in got}
        assert pm.load_analogues(con, q, backtest=True) == []
        block = pm.render_lessons_block("", got)
        assert block.startswith("\n\n" + pm.LESSONS_HEADING)
        assert pm.render_lessons_block("", []) == ""
    finally:
        con.close()


# --- 3. the block reaches the prompt only in the track-record arm ------------------

@pytest.mark.parametrize("share, expect_section", [(0.0, True), (1.0, False)])
def test_lessons_reach_the_prompt_only_in_the_track_record_arm(tmp_path, monkeypatch, share,
                                                               expect_section):
    import sibyl.agent as sibyl_agent
    import sibyl.run as sibyl_run
    from tests.sibyl_test_utils import (
        HS_RUN_ID, disable_evidence_gate, make_submit_response, seed_db, stub_reference,
    )

    seed_db(tmp_path, monkeypatch)
    disable_evidence_gate(monkeypatch)
    from pythia.db.schema import connect, ensure_sibyl_measurement_tables

    con = connect(read_only=False)
    try:
        ensure_sibyl_measurement_tables(con)
        con.execute(
            "INSERT INTO sibyl_lessons (hazard_code, metric, version, as_of_month, lessons_text) "
            "VALUES ('ACE', 'FATALITIES', 1, ?, '- Check the latest casualty update.')",
            [date.today().strftime("%Y-%m")])
    finally:
        con.close()

    prompts: List[str] = []

    def fake(prompt):
        prompts.append(prompt)
        return make_submit_response(), {"cost_usd": 0.01}, ""

    monkeypatch.setattr(sibyl_run, "build_reference", stub_reference)
    monkeypatch.setattr(sibyl_agent, "log_sibyl_call", lambda **kw: None)
    monkeypatch.setattr(sibyl_run, "ADVICE_EXPERIMENT_SHARE", share)
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=fake)
    assert prompts
    assert all((pm.LESSONS_HEADING in p) is expect_section for p in prompts)
    assert all("=== YOUR TRACK RECORD ===" not in p for p in prompts)  # no advice exists
