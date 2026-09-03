# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The six defects the 2026-09-01 bundle carried in its existing outputs.

Each test states the wrong value the bundle printed, so a reader of the
diff can see what changed and a future refactor cannot quietly restore it.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.dump_pythia_debug_bundle import (  # noqa: E402
    BundleData,
    _load_counts_before,
    _question_target_month,
    _reported_triage_tier,
    emit_grounding_detail_csv,
    emit_question_metrics_csv,
    emit_timing_breakdown_csv,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# 1. target_month said 2027-03 beside a question id reading 2026-10
# ---------------------------------------------------------------------------

def test_target_month_is_the_questions_own_month_not_its_last_horizon():
    question = {
        "question_id": "SOM_ACE_PA_2026-10",
        "window_start_date": "2026-10-01",
        "target_month": "2027-03",
    }
    assert _question_target_month(question) == "2026-10"


def test_target_month_falls_back_to_the_id_when_the_window_start_is_missing():
    assert _question_target_month({"question_id": "TCD_FL_PA_2026-10"}) == "2026-10"
    assert _question_target_month({"question_id": "no-epoch-here"}) == ""


# ---------------------------------------------------------------------------
# 2. triage_tier said "quiet" for questions that were never triaged
# ---------------------------------------------------------------------------

def test_an_rc_promoted_hazard_is_not_reported_as_quiet():
    # An RC-promoted hazard skips triage entirely, so whatever the tier
    # column holds is a synthetic default. Until 2026-09-03 that default was
    # "quiet", and printing it made 105 Track-1 questions read as quiet
    # hazards that had somehow been given a full ensemble.
    assert _reported_triage_tier("quiet", {"rc_level": 2, "data_quality_json": None}) == "not_triaged"
    assert (
        _reported_triage_tier(
            "quiet", {"rc_level": 2, "data_quality_json": json.dumps({"status": "rc_promoted"})}
        )
        == "not_triaged"
    )
    assert _reported_triage_tier("rc_promoted", {}) == "not_triaged"


def test_a_genuinely_triaged_quiet_hazard_still_reads_quiet():
    assert (
        _reported_triage_tier(
            "quiet", {"rc_level": 0, "data_quality_json": json.dumps({"status": "ok"})}
        )
        == "quiet"
    )


# ---------------------------------------------------------------------------
# The two above, through the emitter
# ---------------------------------------------------------------------------

def _metrics_db(tmp_path: Path):
    con = duckdb.connect(str(tmp_path / "m.duckdb"))
    con.execute(
        """
        CREATE TABLE hs_triage (
            run_id TEXT, iso3 TEXT, hazard_code TEXT, triage_score DOUBLE,
            regime_change_likelihood DOUBLE, regime_change_level INTEGER,
            regime_change_direction TEXT, track INTEGER, tier TEXT,
            data_quality_json TEXT
        )
        """
    )
    con.execute(
        "CREATE TABLE forecasts_ensemble (run_id TEXT, question_id TEXT, status TEXT, created_at TIMESTAMP)"
    )
    con.execute(
        "CREATE TABLE llm_calls (run_id TEXT, question_id TEXT, usage_json TEXT)"
    )
    return con


def test_question_metrics_publishes_both_months_and_an_honest_tier(tmp_path: Path):
    con = _metrics_db(tmp_path)
    con.execute(
        "INSERT INTO hs_triage VALUES ('hs_1','SOM','ACE',0.0,0.6,2,'up',1,'quiet',NULL)"
    )
    con.execute("INSERT INTO forecasts_ensemble VALUES ('fc_1','SOM_ACE_PA_2026-10','ok',now())")
    data = BundleData()
    data.hs_run_id = "hs_1"
    data.forecaster_run_id = "fc_1"
    data.out_run_id = "fc_1"
    data.questions = [
        {
            "question_id": "SOM_ACE_PA_2026-10",
            "iso3": "SOM",
            "hazard_code": "ACE",
            "metric": "PA",
            "window_start_date": "2026-10-01",
            "target_month": "2027-03",
            "track": 1,
        }
    ]
    data.scenario_status_rows = [
        {"question_id": "SOM_ACE_PA_2026-10", "triage_tier": "quiet", "status": "generated"}
    ]
    emit_question_metrics_csv(data, con, tmp_path)
    (row,) = _read_csv(tmp_path / "question_metrics__fc_1.csv")
    assert row["target_month"] == "2026-10"
    assert row["horizon_end_month"] == "2027-03"
    assert row["triage_tier"] == "not_triaged"


# ---------------------------------------------------------------------------
# 4. Track 2 questions had no timing at all
# ---------------------------------------------------------------------------

def _timing_db(tmp_path: Path):
    con = duckdb.connect(str(tmp_path / "t.duckdb"))
    con.execute(
        """
        CREATE TABLE llm_calls (
            iso3 TEXT, phase TEXT, call_type TEXT, hazard_code TEXT,
            timestamp TIMESTAMP, run_id TEXT, hs_run_id TEXT, usage_json TEXT
        )
        """
    )
    # The real columns: forecasts_raw carries elapsed_ms and no created_at;
    # forecasts_ensemble carries created_at. The fallback reads both, and a
    # test that invents a created_at on forecasts_raw would pass against a
    # query production can never run.
    con.execute(
        """
        CREATE TABLE forecasts_raw (
            run_id TEXT, question_id TEXT, model_name TEXT, month_index INTEGER,
            bucket_index INTEGER, status TEXT, elapsed_ms BIGINT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE forecasts_ensemble (
            run_id TEXT, question_id TEXT, model_name TEXT, month_index INTEGER,
            bucket_index INTEGER, created_at TIMESTAMP
        )
        """
    )
    return con


def test_a_country_whose_questions_all_took_track_2_still_gets_an_spd_row(tmp_path: Path):
    con = _timing_db(tmp_path)
    t0 = datetime(2026, 9, 1, 5, 0, 0)
    # Chad was triaged (so it has HS calls) but its one question ran on the
    # Track-2 path with a batched call, which logs no per-country spd row.
    con.execute(
        "INSERT INTO llm_calls VALUES ('TCD','hs_triage','rc_pass_1','RC_ACE_PASS_1',?,NULL,'hs_1','{}')",
        [t0],
    )
    for month in range(1, 7):
        con.execute(
            "INSERT INTO forecasts_raw VALUES ('fc_1','TCD_ACE_PA_2026-10','track2_flash',?,1,'ok',4200)",
            [month],
        )
        con.execute(
            "INSERT INTO forecasts_ensemble VALUES ('fc_1','TCD_ACE_PA_2026-10','track2_flash',?,1,?)",
            [month, t0 + timedelta(seconds=30 * month)],
        )
    data = BundleData()
    data.hs_run_id = "hs_1"
    data.forecaster_run_id = "fc_1"
    data.out_run_id = "fc_1"
    data.questions = [
        {"question_id": "TCD_ACE_PA_2026-10", "iso3": "TCD", "hazard_code": "ACE", "track": 2}
    ]
    emit_timing_breakdown_csv(data, con, tmp_path)
    rows = {r["iso3"]: r for r in _read_csv(tmp_path / "timing_breakdown__fc_1.csv")}
    assert "TCD" in rows
    assert rows["TCD"]["n_questions"] == "1"
    assert int(rows["TCD"]["spd_elapsed_ms"]) > 0
    # And the row says where the number came from, because it is a write
    # span rather than a model latency — naming the column it actually came
    # from, which is on forecasts_ensemble; forecasts_raw has no created_at.
    assert "forecasts_ensemble.created_at" in rows["TCD"]["spd_timing_source"]


def test_rc_and_triage_no_longer_share_one_span(tmp_path: Path):
    con = _timing_db(tmp_path)
    t0 = datetime(2026, 9, 1, 5, 0, 0)
    con.execute(
        "INSERT INTO llm_calls VALUES ('SOM','hs_triage','rc_pass_1','RC_ACE_PASS_1',?,NULL,'hs_1','{}')",
        [t0],
    )
    con.execute(
        "INSERT INTO llm_calls VALUES ('SOM','hs_triage','rc_pass_1','RC_ACE_PASS_1',?,NULL,'hs_1','{}')",
        [t0 + timedelta(seconds=10)],
    )
    con.execute(
        "INSERT INTO llm_calls VALUES ('SOM','hs_triage','triage_pass_1','ACE',?,NULL,'hs_1','{}')",
        [t0 + timedelta(seconds=20)],
    )
    con.execute(
        "INSERT INTO llm_calls VALUES ('SOM','hs_triage','triage_pass_1','ACE',?,NULL,'hs_1','{}')",
        [t0 + timedelta(seconds=95)],
    )
    data = BundleData()
    data.hs_run_id = "hs_1"
    data.out_run_id = "hs_1"
    emit_timing_breakdown_csv(data, con, tmp_path)
    (row,) = _read_csv(tmp_path / "timing_breakdown__hs_1.csv")
    # Every HS row carries phase='hs_triage', which is why reading phase
    # gave all 122 countries rc_elapsed_ms == triage_elapsed_ms.
    assert row["rc_elapsed_ms"] == "10000"
    assert row["triage_elapsed_ms"] == "75000"


# ---------------------------------------------------------------------------
# 5. grounding_detail.phase was hs_triage on all 797 rows
# ---------------------------------------------------------------------------

def test_grounding_detail_phase_names_the_stage_and_keeps_the_raw_column(tmp_path: Path):
    con = duckdb.connect(str(tmp_path / "g.duckdb"))
    con.execute(
        """
        CREATE TABLE llm_calls (
            iso3 TEXT, hazard_code TEXT, phase TEXT, call_type TEXT, model_id TEXT,
            response_text TEXT, error_text TEXT, prompt_text TEXT,
            elapsed_ms BIGINT, timestamp TIMESTAMP, hs_run_id TEXT, run_id TEXT
        )
        """
    )
    now = datetime(2026, 9, 1, 5, 0, 0)
    for hazard, call_type in (
        ("GROUNDING_ACE", "rc_grounding"),
        ("TRIAGE_GROUNDING_ACE", "triage_grounding"),
        ("ADVERSARIAL_ACE", "adversarial_search"),
    ):
        con.execute(
            "INSERT INTO llm_calls VALUES ('SOM',?, 'hs_triage', ?, 'brave-web-search',"
            "'{\"n_sources\": 3}', NULL, 'q', 100, ?, 'hs_1', NULL)",
            [hazard, call_type, now],
        )
    data = BundleData()
    data.hs_run_id = "hs_1"
    data.out_run_id = "hs_1"
    emit_grounding_detail_csv(data, con, tmp_path)
    rows = _read_csv(tmp_path / "grounding_detail__hs_1.csv")
    assert {r["phase"] for r in rows} == {
        "rc_grounding", "triage_grounding", "adversarial_search"
    }
    # The raw value is kept so the mapping stays auditable.
    assert {r["db_phase"] for r in rows} == {"hs_triage"}
    assert all(r["phase"] == r["stage"] for r in rows)


# ---------------------------------------------------------------------------
# 6. row_counts_before was null for every table
# ---------------------------------------------------------------------------

def test_counts_before_come_from_the_signature_the_stage_already_writes(tmp_path: Path):
    path = tmp_path / "db_signature_before.json"
    path.write_text(
        json.dumps(
            {
                "required_counts": {"questions": 190, "hs_triage": 488},
                "optional_counts": {"forecasts_raw": 13000},
            }
        )
    )
    counts = _load_counts_before(str(path))
    assert counts["questions"] == 190
    assert counts["hs_triage"] == 488
    assert counts["forecasts_raw"] == 13000
    # A table the signature does not cover stays None. An absent baseline
    # must never be published as a zero, which would turn every row into a
    # fabricated gain.
    assert counts["snapshots"] is None


def test_a_missing_signature_file_leaves_the_baseline_unknown():
    counts = _load_counts_before("does/not/exist.json")
    assert set(counts.values()) == {None}
