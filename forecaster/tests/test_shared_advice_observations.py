# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""The shared calibration advice, as the prompt shows it (Oct 2026).

On 1 October the advice arm of the conflict-death questions read "ACTION:
Widen uncertainty for later months" and "ACTION: Start with more mass in
bucket 5" above an instruction to copy the base-rate distribution exactly.
The shared advice is now rendered from its findings as observations, drops
the prior line where the prompt hands the member its prior, drops the
month-1/month-6 line until later horizons have resolved, has no global
fallback, and a question in the advice arm that got no advice text is
stamped as such.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest

duckdb = pytest.importorskip("duckdb")

from forecaster import prompts

FINDINGS = {
    "n_questions": 16,
    "bucket_calibration": [
        {"bucket_index": 5, "class_bin": "100-<500", "mean_assigned": 0.09, "actual_rate": 0.34},
    ],
    "month_position_bias": {"flat": True, "jsd_m1_m6": 0.001},
    "prior_anchoring": {"worst_bucket_gap": {"bucket": 5, "gap_pp": -10.0}},
}
LEGACY_TEXT = "ACTION: Widen uncertainty for later months (4-6).\nACTION: Start with more mass in bucket 5."


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "a.duckdb"
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE calibration_advice (as_of_month TEXT, hazard_code TEXT, metric TEXT, "
        "model_name TEXT, advice TEXT, findings_json TEXT, advice_version TEXT, created_at TIMESTAMP)"
    )
    con.execute("CREATE TABLE questions (question_id TEXT, hazard_code TEXT, metric TEXT, is_test BOOLEAN)")
    con.execute("CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, value DOUBLE)")
    for hz, m in (("ACE", "FATALITIES"), ("*", "*")):
        con.execute(
            "INSERT INTO calibration_advice VALUES ('2026-09',?,?,'__shared__',?,?,'v1',?)",
            [hz, m, LEGACY_TEXT, json.dumps(FINDINGS), datetime(2026, 9, 28)],
        )
    con.close()
    url = f"duckdb:///{path}"
    monkeypatch.setattr(prompts, "_pythia_db_url_from_config", lambda: url)
    for var in ("PYTHIA_PRIOR_ANCHOR_SPD", "PYTHIA_ADVICE_BLOCK_GROUPS", "PYTHIA_ADVICE_VERSION",
                "PYTHIA_ADVICE_EXPERIMENT_SHARE", "PYTHIA_MEMBER_ADVICE",
                "PYTHIA_ADVICE_FAMILY_CARRYOVER", "PYTHIA_FAMILY_RECALIBRATION_MODE"):
        monkeypatch.delenv(var, raising=False)
    prompts.reset_member_calibration_advice_cache()
    return path


def _later_horizons(path, n):
    con = duckdb.connect(str(path))
    for i in range(n):
        con.execute("INSERT INTO questions VALUES (?, 'ACE', 'FATALITIES', FALSE)", [f"q{i}"])
        con.execute("INSERT INTO resolutions VALUES (?, 2, 5.0)", [f"q{i}"])
    con.close()


def test_shared_advice_is_observations_with_no_action_line(db):
    txt = prompts._load_calibration_advice_for_hazard("ACE", "FATALITIES")
    assert "ACTION" not in txt
    assert "Bucket 100-<500: you assigned 9%; observed 34%." in txt
    assert prompts.ADVICE_TENDENCY_LINE in txt


def test_prior_line_dropped_when_the_prompt_hands_over_the_prior(db, monkeypatch):
    assert "declared priors" in prompts._load_calibration_advice_for_hazard("ACE", "FATALITIES")
    monkeypatch.setenv("PYTHIA_PRIOR_ANCHOR_SPD", "1")
    assert "declared priors" not in prompts._load_calibration_advice_for_hazard("ACE", "FATALITIES")


def test_horizon_line_waits_for_resolved_later_months(db):
    assert "month-6" not in prompts._load_calibration_advice_for_hazard("ACE", "FATALITIES")
    _later_horizons(db, prompts.ADVICE_LATER_HORIZON_MIN_QUESTIONS)
    assert "month-6" in prompts._load_calibration_advice_for_hazard("ACE", "FATALITIES")


def test_no_global_fallback_for_a_group_without_its_own_row(db):
    assert prompts._load_calibration_advice_for_hazard("FL", "PA") == ""


def test_advice_arm_with_no_text_is_stamped_empty(db, monkeypatch):
    monkeypatch.setenv("PYTHIA_ADVICE_EXPERIMENT_SHARE", "0.5")
    qid = "XYZ_FL_PA_2026-05"  # hashes into the advice arm at 0.5
    assert prompts.advice_arm(qid) == "advice"
    assert prompts.effective_advice_arm(qid, "FL", "PA") == prompts.ADVICE_ARM_EMPTY
    assert prompts.effective_advice_arm(qid, "ACE", "FATALITIES") == "advice"
