# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""The advice experiment compares something only where the advice arm got text."""

from __future__ import annotations

import duckdb

from scripts.ci import advice_arm_text as aat


def test_a_group_with_no_shared_advice_and_no_member_note_is_reported_empty(monkeypatch):
    from forecaster import prompts

    monkeypatch.setattr(prompts, "_load_calibration_advice_for_hazard",
                        lambda hz, m, model_name=None: "observations" if hz == "ACE" else "")
    monkeypatch.setattr(prompts, "load_member_calibration_advice",
                        lambda hz, m, name, question_id=None: "note" if (hz, name) == ("TC", "b") else "")
    prev = aat.preview((("ACE", "PA"), ("TC", "PA"), ("FL", "PA")), members=["a", "b"])
    verdict = {(r["hazard_code"], r["metric"]): r["advice_arm_gets_text"] for r in prev}
    assert verdict == {("ACE", "PA"): True, ("TC", "PA"): True, ("FL", "PA"): False}
    text = aat.render(prev, None, None)
    assert "| FL | PA | 0 | none | **no** |" in text
    assert "compares nothing): FL/PA." in text


def test_the_recorded_arms_of_a_run_are_counted_in_distinct_questions():
    con = duckdb.connect()
    con.execute("CREATE TABLE questions (question_id TEXT, hazard_code TEXT, metric TEXT)")
    con.execute("CREATE TABLE forecasts_raw (question_id TEXT, run_id TEXT, model_name TEXT, advice_arm TEXT)")
    con.execute("INSERT INTO questions VALUES ('q1','FL','PA'), ('q2','FL','PA')")
    con.execute(
        "INSERT INTO forecasts_raw VALUES ('q1','r','a','advice_empty'), ('q1','r','b','advice_empty'), "
        "('q2','r','a','no_advice'), ('q2','old','a','advice')"
    )
    arms = {a["arm"]: a["questions"] for a in aat.run_arms(con, "r")}
    assert arms == {"advice_empty": 1, "no_advice": 1}
