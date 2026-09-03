# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The debug bundle must see a member whose call succeeded but whose output
never became a forecast, and must attribute HS time and grounding calls to
the stage that made them.

Regression cover for the 2026-09-01 bundle: three member responses failed
JSON parsing after status='ok' calls, forecasts_raw was short 105 rows, and
the health line still read "SPD Ensemble | OK | 5/5 models"; every country's
rc_elapsed_ms equalled its triage_elapsed_ms; all 797 grounding calls were
labelled hs_triage.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.dump_pythia_debug_bundle import (  # noqa: E402
    _complete_member_spds_by_question,
    _grounding_stage_for_call,
    _member_gap_summary,
    _timing_stage_for_call,
    emit_timing_breakdown_csv,
)


def _forecasts_raw(con) -> None:
    con.execute(
        """
        CREATE TABLE forecasts_raw (
            run_id TEXT, question_id TEXT, model_name TEXT,
            month_index INTEGER, bucket_index INTEGER, probability DOUBLE,
            status TEXT
        )
        """
    )


def test_complete_member_requires_every_window_month(tmp_path: Path):
    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    try:
        _forecasts_raw(con)
        rows = []
        # Model A: complete 6-month SPD. Model B: one no_forecast row (the
        # parse-failure signature). Model C: only 5 months.
        for m in range(1, 7):
            for b in range(1, 7):
                rows.append(("run", "Q1", "model-a", m, b, 1 / 6, "ok"))
        rows.append(("run", "Q1", "model-b", None, None, None, "no_forecast"))
        for m in range(1, 6):
            for b in range(1, 7):
                rows.append(("run", "Q1", "model-c", m, b, 1 / 6, "ok"))
        con.executemany("INSERT INTO forecasts_raw VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

        out = _complete_member_spds_by_question(con, "run", ["Q1", "Q2"])
        assert out == {"Q1": {"model-a"}}  # Q2 has no rows → not keyed
    finally:
        con.close()


def test_member_gap_summary_counts_track1_cells_only():
    metrics = [
        {"n_spd_models_expected": 5, "missing_model_ids_json": json.dumps(["claude-opus-5"])},
        {"n_spd_models_expected": 5, "missing_model_ids_json": json.dumps(["claude-opus-5"])},
        {"n_spd_models_expected": 5, "missing_model_ids_json": json.dumps(["gpt-5.6-luna"])},
        {"n_spd_models_expected": 5, "missing_model_ids_json": "[]"},
        {"n_spd_models_expected": 1, "missing_model_ids_json": json.dumps(["gemini-3.5-flash"])},
    ]
    gaps = _member_gap_summary(metrics)
    assert gaps["n_cells_missing"] == 3
    assert gaps["n_cells_expected"] == 20
    assert gaps["n_questions_affected"] == 3
    assert gaps["by_model"] == {"claude-opus-5": 2, "gpt-5.6-luna": 1}


def test_timing_stage_comes_from_call_type_not_phase():
    # Every HS row carries phase='hs_triage'; call_type must split them.
    assert _timing_stage_for_call("hs_triage", "rc_pass_1", "RC_ACE_PASS_1") == "rc"
    assert _timing_stage_for_call("hs_triage", "rc_grounding", "GROUNDING_ACE") == "rc"
    assert _timing_stage_for_call("hs_triage", "triage_pass_1", "ACE") == "triage"
    assert _timing_stage_for_call("hs_triage", "triage_grounding", "TRIAGE_GROUNDING_ACE") == "triage"
    assert _timing_stage_for_call("hs_triage", "adversarial_search", "ADVERSARIAL_ACE") == "adversarial"
    assert _timing_stage_for_call("hs_triage", "adversarial_synthesis", "ADVERSARIAL_SYNTH_ACE") == "adversarial"
    assert _timing_stage_for_call("spd_v2", "spd_v2", "ACE") == "spd"
    assert _timing_stage_for_call("binary_v2", "binary_v2", "FL") == "spd"
    # Legacy rows with no call_type fall back to the synthetic hazard code.
    assert _timing_stage_for_call("hs_triage", "", "RC_DR_PASS_2") == "rc"
    assert _timing_stage_for_call("hs_triage", "", "TRIAGE_GROUNDING_DR") == "triage"
    assert _timing_stage_for_call("hs_triage", "", "DR") == "triage"
    assert _timing_stage_for_call("scenario_v2", "", "ACE") is None


def test_grounding_stage_labels():
    assert _grounding_stage_for_call("rc_grounding", "GROUNDING_ACE") == "rc_grounding"
    assert _grounding_stage_for_call("triage_grounding", "TRIAGE_GROUNDING_ACE") == "triage_grounding"
    assert _grounding_stage_for_call("adversarial_search", "ADVERSARIAL_ACE") == "adversarial_search"
    assert _grounding_stage_for_call("", "TRIAGE_GROUNDING_FL") == "triage_grounding"
    assert _grounding_stage_for_call("", "GROUNDING_FL") == "rc_grounding"
    assert _grounding_stage_for_call("", "ADVERSARIAL_SYNTH_FL") == "other"


def test_timing_breakdown_splits_rc_from_triage(tmp_path: Path):
    """RC ran 10:00-10:05, triage 10:20-10:23, SPD (batched, replayed at
    11:00) — the CSV must show three different spans, not one shared pair."""
    from scripts.dump_pythia_debug_bundle import BundleData

    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    try:
        con.execute(
            """
            CREATE TABLE llm_calls (
                run_id TEXT, hs_run_id TEXT, iso3 TEXT, hazard_code TEXT,
                phase TEXT, call_type TEXT, timestamp TIMESTAMP, usage_json TEXT
            )
            """
        )
        t0 = datetime(2026, 9, 1, 10, 0, 0)
        rows = [
            (None, "hs1", "SOM", "GROUNDING_ACE", "hs_triage", "rc_grounding", t0, "{}"),
            (None, "hs1", "SOM", "RC_ACE_PASS_1", "hs_triage", "rc_pass_1", t0 + timedelta(minutes=5), "{}"),
            (None, "hs1", "SOM", "TRIAGE_GROUNDING_ACE", "hs_triage", "triage_grounding", t0 + timedelta(minutes=20), "{}"),
            (None, "hs1", "SOM", "ACE", "hs_triage", "triage_pass_1", t0 + timedelta(minutes=23), "{}"),
            ("fc1", "hs1", "SOM", "ACE", "spd_v2", "spd_v2", t0 + timedelta(minutes=60), '{"service_tier": "batch"}'),
            ("fc1", "hs1", "SOM", "ACE", "spd_v2", "spd_v2", t0 + timedelta(minutes=61), '{"service_tier": "batch"}'),
        ]
        con.executemany("INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        data = BundleData()
        data.hs_run_id = "hs1"
        data.forecaster_run_id = "fc1"
        data.out_run_id = "fc1"
        emit_timing_breakdown_csv(data, con, tmp_path)
    finally:
        con.close()

    import csv

    with open(tmp_path / "timing_breakdown__fc1.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert int(row["rc_elapsed_ms"]) == 5 * 60 * 1000
    assert int(row["triage_elapsed_ms"]) == 3 * 60 * 1000
    assert int(row["spd_elapsed_ms"]) == 60 * 1000
    assert row["rc_elapsed_ms"] != row["triage_elapsed_ms"]
    assert int(row["n_rc_calls"]) == 2 and int(row["n_triage_calls"]) == 2
    assert int(row["n_spd_batched"]) == 2
