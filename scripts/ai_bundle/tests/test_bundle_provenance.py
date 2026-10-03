# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The scored bundle says what each forecast was made WITH.

The August 2026 bundle could not say which ENSO reading, how much GDACS
history, which CrisisWatch edition or which base rate a question was
forecast on; which models at which effort made it; what series resolved it;
why calibration_weights was empty; or what it cost.
"""

from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import duckdb
import pytest

from scripts.ai_bundle import provenance as prov
from scripts.ai_bundle.build_scored_forecast_bundle import build_bundle
from scripts.ai_bundle.tests.test_scored_forecast_bundle import (  # noqa: F401
    FC_RUN,
    HS_RUN,
    QID_BIN,
    QID_SPD,
    mini_db,
)


def _path(db_url: str) -> str:
    return db_url.replace("duckdb:///", "")


@pytest.fixture
def rich_db(mini_db: str) -> str:  # noqa: F811
    con = duckdb.connect(_path(mini_db))
    con.execute("CREATE TABLE hs_runs (hs_run_id TEXT, generated_at TIMESTAMP)")
    con.execute(f"INSERT INTO hs_runs VALUES ('{HS_RUN}', TIMESTAMP '2026-01-02 04:00:00')")
    con.execute(
        "CREATE TABLE enso_state (fetch_date DATE, enso_phase TEXT, oni DOUBLE, "
        "observation_date DATE, status TEXT, age_days INTEGER, row_kind TEXT)"
    )
    con.execute(
        "INSERT INTO enso_state VALUES "
        "(DATE '2025-12-28', 'El Nino', 1.1, DATE '2025-11-01', 'fresh', 57, 'live'),"
        "(DATE '2026-02-15', 'Neutral', 0.2, DATE '2026-01-01', 'fresh', 45, 'live')"
    )
    con.execute(
        "CREATE TABLE crisiswatch_entries (iso3 TEXT, month INTEGER, year INTEGER, "
        "fetched_at TIMESTAMP)"
    )
    con.execute(
        "INSERT INTO crisiswatch_entries VALUES ('ETH', 10, 2025, TIMESTAMP '2025-11-05'),"
        "('ETH', 1, 2026, TIMESTAMP '2026-02-05'),"
        # Another country in a newer edition held before the run: ETH is
        # then "not listed in the newest edition".
        "('SOM', 12, 2025, TIMESTAMP '2026-01-01')"
    )
    con.execute(
        "CREATE TABLE forecast_deviation (run_id TEXT, question_id TEXT, "
        "model_name TEXT, baserate_source TEXT)"
    )
    con.execute(
        f"INSERT INTO forecast_deviation VALUES ('{FC_RUN}', '{QID_SPD}', "
        "'ensemble_mean_v2', 'acled_monthly_fatalities')"
    )
    con.execute("ALTER TABLE llm_calls ADD COLUMN cost_usd DOUBLE")
    con.execute("UPDATE llm_calls SET cost_usd = 0.25 WHERE question_id = ?", [QID_SPD])
    con.execute(
        "CREATE TABLE baseline_scored_forecasts (question_id TEXT, horizon_m INTEGER, "
        "model_name TEXT, metric TEXT, spd_json TEXT)"
    )
    con.execute(
        f"INSERT INTO baseline_scored_forecasts VALUES ('{QID_SPD}', 1, "
        "'__ext_climatology', 'FATALITIES', '[0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05]')"
    )
    con.close()
    return mini_db


@pytest.fixture
def extracted(rich_db: str, tmp_path: Path) -> Path:
    zip_path = build_bundle(rich_db, tmp_path / "out", months_back=0, n_case_studies=1)
    out = tmp_path / "x"
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out)
    return out


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_inject_status_reads_what_was_current_on_the_run_date(extracted: Path):
    rec = json.loads((extracted / "questions" / f"{QID_SPD}.json").read_text())
    inj = rec["inject_status"]
    assert inj["run_date"] == "2026-01-02"
    # The record fetched after the run must not answer for it.
    assert inj["enso"]["enso_phase"] == "El Nino"
    assert inj["enso"]["observation_date"] == "2025-11-01"
    assert inj["crisiswatch"]["edition"] == "2025-10"
    assert inj["crisiswatch"]["edition_age_months"] == 3
    assert inj["crisiswatch"]["newest_edition_held"] == "2025-12"
    assert inj["crisiswatch"]["coverage"] == "not_listed_in_newest"
    assert inj["base_rate"]["source"] == "acled_monthly_fatalities"
    assert inj["gdacs_history"] == {"applicable": False}


def test_binary_flood_question_reports_its_gdacs_window(extracted: Path):
    rec = json.loads((extracted / "questions" / f"{QID_BIN}.json").read_text())
    g = rec["inject_status"]["gdacs_history"]
    assert g["applicable"] is True
    assert g["available"] is False and g["reason"]


def test_lineup_resolution_series_and_cost(extracted: Path):
    rec = json.loads((extracted / "questions" / f"{QID_SPD}.json").read_text())
    assert rec["lineup"]["lineup_id"]
    assert rec["lineup"]["members"][0]["model_id"] == "model-a-id"
    assert "ALL event types" in rec["resolution_series"]
    assert rec["cost_usd"]["__total__"] == pytest.approx(0.25)
    idx = {r["question_id"]: r for r in _csv(extracted / "questions_index.csv")}
    assert idx[QID_SPD]["lineup_id"] == rec["lineup"]["lineup_id"]
    assert idx[QID_SPD]["spd_prompt_missing"] == "False"
    assert idx[QID_SPD]["enso_observation_date"] == "2025-11-01"
    manifest = json.loads((extracted / "manifest.json").read_text())
    assert manifest["lineups"][rec["lineup"]["lineup_id"]]["n_questions"] >= 1


def test_manifest_counts_resolutions_by_horizon_and_month(extracted: Path):
    manifest = json.loads((extracted / "manifest.json").read_text())
    rq = manifest["resolved_questions"]
    assert rq["by_horizon"] == {"1": 2, "2": 1}
    assert rq["by_observed_month"] == {"2026-01": 2, "2026-02": 1}


def test_empty_calibration_says_why(extracted: Path):
    rows = _csv(extracted / "calibration_status.csv")
    by = {(r["hazard_code"], r["metric"]): r for r in rows}
    ace = by[("ACE", "FATALITIES")]
    assert ace["floor"] == "20"
    assert ace["n_questions_with_member_scores"] == "1"
    # The fixture does carry ACE weights, so the status reports them.
    assert ace["has_weights"] == "True"


def test_reference_vectors_in_forecast_vs_outcome(extracted: Path):
    rows = _csv(extracted / "forecast_vs_outcome.csv")
    clim = [r for r in rows if r["model_name"] == "__ext_climatology"]
    assert len(clim) == 1
    # 40 deaths fall in bucket 4 (25 to <100).
    assert json.loads(clim[0]["probs"])[3] == pytest.approx(0.2)
    assert clim[0]["realized_bucket"] == "4"
    assert clim[0]["p_realized_bucket"] == "0.2"
    ens = [r for r in rows if r["model_name"] == "ensemble_mean_v2" and r["question_id"] == QID_SPD]
    assert len(json.loads(ens[0]["probs"])) == 7


def test_rollups_carry_cost_per_question(extracted: Path):
    rows = _csv(extracted / "rollups.csv")
    member = [r for r in rows if r["model_name"] == "model-a"]
    assert member and float(member[0]["cost_per_question_usd"]) == pytest.approx(0.25)


def test_missing_prompt_is_flagged_with_a_reason(rich_db: str, tmp_path: Path):
    con = duckdb.connect(_path(rich_db))
    con.execute("UPDATE llm_calls SET prompt_text = '' WHERE question_id = ?", [QID_SPD])
    con.close()
    zip_path = build_bundle(rich_db, tmp_path / "o2", months_back=0, n_case_studies=1)
    with zipfile.ZipFile(zip_path) as zf:
        rec = json.loads(zf.read(f"{zip_path.stem}/questions/{QID_SPD}.json")
                         if f"{zip_path.stem}/questions/{QID_SPD}.json" in zf.namelist()
                         else zf.read(f"questions/{QID_SPD}.json"))
    assert rec["spd_prompt"] is None
    assert "empty prompt_text" in rec["spd_prompt_missing_reason"]


def test_below_floor_calibration_reads_as_below_floor():
    con = duckdb.connect()
    con.execute("CREATE TABLE questions (question_id TEXT, iso3 TEXT, hazard_code TEXT, "
                "metric TEXT, target_month TEXT, is_test BOOLEAN)")
    con.execute("CREATE TABLE scores (question_id TEXT, model_name TEXT, score_type TEXT)")
    con.execute("INSERT INTO questions VALUES ('q1', 'SOM', 'DR', 'EVENT_OCCURRENCE', '2026-08', FALSE)")
    con.execute("INSERT INTO scores VALUES ('q1', 'gpt-6-sol', 'brier'), "
                "('q1', '__ext_climatology', 'brier'), ('q1', 'ensemble_mean_v2', 'brier')")
    rows = prov.calibration_status(con)
    assert rows == [{
        "hazard_code": "DR", "metric": "EVENT_OCCURRENCE",
        "n_questions_with_member_scores": 1, "floor": 20, "has_weights": False,
        "status": "below floor (1 of 20 questions)",
    }]
