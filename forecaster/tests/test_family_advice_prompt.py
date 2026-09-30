# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""What a member is told: exact advice, carried family advice, or nothing.

``PYTHIA_ADVICE_FAMILY_CARRYOVER`` shows a new version of a model line its
predecessors' calibration findings until it has twenty questions of its own,
as observations rather than orders. ``PYTHIA_ADVICE_EXPERIMENT_SHARE`` puts a
share of questions in a no-advice arm so the advice can be measured.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

duckdb = pytest.importorskip("duckdb")

from forecaster import prompts

FINDINGS = {
    "tail_coverage": {"avg_assigned_tail": 0.09, "actual_tail_rate": 0.34},
    "bucket_calibration": [
        {"bucket_index": 5, "class_bin": "100-<500", "mean_assigned": 0.09, "actual_rate": 0.34},
        {"bucket_index": 1, "class_bin": "0", "mean_assigned": 0.20, "actual_rate": 0.21},
    ],
    "horizon_diff": {"flat": True, "jsd_m1_m6": 0.001},
    "prior_anchoring": {"worst_bucket_gap": {"bucket": 3, "gap_pp": 12.0}},
}


@pytest.fixture
def advice_db(tmp_path, monkeypatch):
    path = tmp_path / "a.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE calibration_advice (as_of_month TEXT, hazard_code TEXT, metric TEXT, "
        "model_name TEXT, advice TEXT, findings_json TEXT, advice_version TEXT, created_at TIMESTAMP)"
    )

    def add(name, findings, text="LEGACY TEXT: ACTION: Increase bucket 5."):
        con.execute(
            "INSERT INTO calibration_advice VALUES ('2026-09','ACE','FATALITIES',?,?,?,'v1',?)",
            [name, text, json.dumps(findings), datetime(2026, 9, 28)],
        )

    add("gpt-6-sol", {**FINDINGS, "n_questions": 25})
    add("claude-opus-5-5", {**FINDINGS, "n_questions": 8})
    add("family:claude", {**FINDINGS, "n_questions": 40,
                          "contributing_ids": {"claude-opus-5": 32, "claude-opus-5-5": 8}})
    add("family:gemini_flash", {**FINDINGS, "n_questions": 30,
                                "contributing_ids": {"gemini-3.5-flash": 30}})
    add("family:gpt_mini", {**FINDINGS, "n_questions": 30,
                            "contributing_ids": {"gpt-6-luna": 30}})
    con.close()
    url = f"duckdb:///{path}"
    monkeypatch.setattr(prompts, "_pythia_db_url_from_config", lambda: url)
    for var in ("PYTHIA_ADVICE_FAMILY_CARRYOVER", "PYTHIA_ADVICE_EXPERIMENT_SHARE",
                "PYTHIA_FAMILY_RECALIBRATION_MODE", "PYTHIA_PRIOR_ANCHOR_SPD",
                "PYTHIA_ADVICE_BLOCK_GROUPS", "PYTHIA_MEMBER_ADVICE"):
        monkeypatch.delenv(var, raising=False)
    prompts.reset_member_calibration_advice_cache()
    yield url
    prompts.reset_member_calibration_advice_cache()


def _advice(name, **kw):
    prompts.reset_member_calibration_advice_cache()
    return prompts.load_member_calibration_advice("ACE", "FATALITIES", name, **kw)


def test_flag_off_serves_the_stored_text_as_before(advice_db):
    assert _advice("gpt-6-sol") == "LEGACY TEXT: ACTION: Increase bucket 5."
    # No carried advice for a model with no row of its own.
    assert _advice("gpt-6-luna") == ""


def test_exact_advice_with_twenty_questions_is_rendered_as_observations(advice_db, monkeypatch):
    monkeypatch.setenv("PYTHIA_ADVICE_FAMILY_CARRYOVER", "1")
    text = _advice("gpt-6-sol")
    assert text.startswith("From your own 25 scored questions")
    assert "Top two buckets: you assigned 9% on average; observed 34%." in text
    assert "Bucket 100-<500: you assigned 9%; observed 34%." in text
    assert "Bucket 0:" not in text  # a one-point gap is not worth a line
    assert "ACTION" not in text and "Increase" not in text
    assert text.endswith(prompts.ADVICE_TENDENCY_LINE)


def test_a_new_version_gets_its_family_decayed_by_its_own_questions(advice_db, monkeypatch):
    monkeypatch.setenv("PYTHIA_ADVICE_FAMILY_CARRYOVER", "1")
    text = _advice("claude-opus-5-5")
    assert "Carried from earlier versions of your model line (claude-opus-5; 40 scored questions" in text
    # 8 own questions: carried gaps at 60%. 9% assigned, 34% observed -> 24%.
    assert "observed 24%" in text
    assert "shown at 60% of the measured gap" in text
    assert "ACTION" not in text


def test_track2_takes_the_flash_family(advice_db, monkeypatch):
    monkeypatch.setenv("PYTHIA_ADVICE_FAMILY_CARRYOVER", "1")
    assert prompts.advice_family_for("track2_flash") == "gemini_flash"
    text = _advice("track2_flash")
    assert "gemini-3.5-flash" in text and "30 scored questions" in text
    assert "shown at" not in text  # track2_flash has no questions of its own in the family


def test_a_family_the_member_alone_filled_carries_nothing(advice_db, monkeypatch):
    monkeypatch.setenv("PYTHIA_ADVICE_FAMILY_CARRYOVER", "1")
    # gpt-6-luna is the only contributor to its family row: nothing is "carried".
    assert _advice("gpt-6-luna") == ""


def test_prior_anchoring_is_dropped_where_the_prompt_hands_over_the_prior(advice_db, monkeypatch):
    monkeypatch.setenv("PYTHIA_ADVICE_FAMILY_CARRYOVER", "1")
    assert "declared priors" in _advice("gpt-6-sol")
    monkeypatch.setenv("PYTHIA_PRIOR_ANCHOR_SPD", "1")
    assert "declared priors" not in _advice("gpt-6-sol")


def test_per_bucket_numbers_are_dropped_where_recalibration_applies(advice_db, monkeypatch):
    from pythia.tools import family_recalibration as fr

    monkeypatch.setenv("PYTHIA_ADVICE_FAMILY_CARRYOVER", "1")
    monkeypatch.setenv("PYTHIA_FAMILY_RECALIBRATION_MODE", "apply")
    monkeypatch.setattr(fr, "lookup", lambda *a, **k: {"mode": "apply", "factors": {1: 1.0}})
    text = _advice("gpt-6-sol")
    assert "Bucket 100-<500" not in text and "Top two buckets" not in text
    assert "nearly identical" in text  # the horizon observation stays
    monkeypatch.setattr(fr, "lookup", lambda *a, **k: {"mode": "auto_shadow", "factors": {1: 1.0}})
    assert "Bucket 100-<500" in _advice("gpt-6-sol")


def test_the_arm_is_a_function_of_the_question_and_the_share(monkeypatch):
    monkeypatch.delenv("PYTHIA_ADVICE_EXPERIMENT_SHARE", raising=False)
    assert prompts.advice_arm("SOM_ACE_FATALITIES_2026-10") is None
    monkeypatch.setenv("PYTHIA_ADVICE_EXPERIMENT_SHARE", "0.5")
    arms = [prompts.advice_arm(f"Q{i}") for i in range(400)]
    assert arms == [prompts.advice_arm(f"Q{i}") for i in range(400)]  # deterministic
    share = arms.count("no_advice") / len(arms)
    assert 0.4 < share < 0.6
    monkeypatch.setenv("PYTHIA_ADVICE_EXPERIMENT_SHARE", "1")
    assert prompts.advice_arm("anything") == "no_advice"
    monkeypatch.setenv("PYTHIA_ADVICE_EXPERIMENT_SHARE", "junk")
    assert prompts.advice_arm("anything") is None


def test_the_no_advice_arm_gets_neither_member_nor_shared_advice(advice_db, monkeypatch):
    monkeypatch.setenv("PYTHIA_ADVICE_EXPERIMENT_SHARE", "1")
    assert _advice("gpt-6-sol", question_id="Q1") == ""
    monkeypatch.setattr(prompts, "_load_calibration_advice_for_hazard", lambda *a, **k: "SHARED NOTE")
    q = {"question_id": "SOM_ACE_FATALITIES_2026-10", "iso3": "SOM", "hazard_code": "ACE",
         "metric": "FATALITIES", "wording": "?", "window_start_date": "2026-10-01"}
    text = prompts.build_spd_prompt_v2(q, {"source": "acled"}, {}, {}, track=1)
    assert "SHARED NOTE" not in text
    monkeypatch.setenv("PYTHIA_ADVICE_EXPERIMENT_SHARE", "0")
    assert "SHARED NOTE" in prompts.build_spd_prompt_v2(q, {"source": "acled"}, {}, {}, track=1)
