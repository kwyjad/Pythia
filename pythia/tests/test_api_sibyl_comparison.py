# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""/v1/performance/sibyl_comparison — fair covered-set head-to-head.

Sibyl writes its pooled SPD under model_name='sibyl' on the same run_id /
question_id as the standard track, so compute_scores scores it head-to-head.
This endpoint restricts the comparison to the exact (question, horizon,
score_type) set Sibyl actually forecast and compares against a chosen standard
baseline. These tests pin: covered-set restriction, baseline selection
(best-available + pinned), delta sign / win rate, family separation,
include_test filtering, and the graceful has_sibyl:false empties that the
current post-DB-reset (no scores) state must return.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest

pytest.importorskip("fastapi")
duckdb = pytest.importorskip("duckdb")

from fastapi.testclient import TestClient

from pythia import config as pythia_config
import pythia.api.app as _app_mod
from pythia.api.app import app


def _make_db(db_path: Path, *, with_scores: bool) -> None:
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE questions (
          question_id TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT,
          is_test BOOLEAN DEFAULT FALSE
        );
        CREATE TABLE scores (
          question_id TEXT, horizon_m INTEGER, metric TEXT, score_type TEXT,
          model_name TEXT, value DOUBLE, run_id TEXT, is_test BOOLEAN DEFAULT FALSE
        );
        CREATE TABLE sibyl_forecasts (
          question_id TEXT, sibyl_run_id TEXT, created_at TIMESTAMP,
          js_divergence_vs_standard DOUBLE, js_divergence_inter_trial DOUBLE,
          volatility_score DOUBLE, cost_usd DOUBLE, is_test BOOLEAN DEFAULT FALSE
        );
        CREATE TABLE sibyl_runs (
          sibyl_run_id TEXT, hs_run_id TEXT, created_at TIMESTAMP,
          n_selected INTEGER, n_forecast INTEGER, n_skipped INTEGER,
          budget_capped BOOLEAN, run_cost_usd DOUBLE, opus_cost_usd DOUBLE,
          brave_cost_usd DOUBLE, run_hard_cap_usd DOUBLE, is_test BOOLEAN DEFAULT FALSE
        );
        """
    )
    # Q1 (ACE/FATALITIES) and Q2 (TC/PA) are Sibyl-covered; Q3 (FL/PA) is
    # ensemble-only and must never appear in the comparison.
    con.execute(
        """INSERT INTO questions VALUES
        ('Q1','SOM','ACE','FATALITIES',FALSE),
        ('Q2','PHL','TC','PA',FALSE),
        ('Q3','KEN','FL','PA',FALSE),
        ('QT','TST','ACE','FATALITIES',TRUE);"""
    )
    con.execute(
        """INSERT INTO sibyl_runs VALUES
        ('sr1','hs1', TIMESTAMP '2026-07-15 10:00:00', 10, 8, 2, FALSE,
         12.5, 10.0, 2.5, 40.0, FALSE);"""
    )
    con.execute(
        """INSERT INTO sibyl_forecasts VALUES
        ('Q1','sr1', TIMESTAMP '2026-07-15 10:00:00', 0.12, 0.05, 0.8, 1.5, FALSE),
        ('Q2','sr1', TIMESTAMP '2026-07-15 10:00:00', 0.30, 0.09, 0.6, 2.0, FALSE);"""
    )

    if not with_scores:
        con.close()
        return

    rows = []
    # Sibyl beats both ensembles on Q1/Q2 across 2 horizons and all 3 score types.
    for q, metric in [("Q1", "FATALITIES"), ("Q2", "PA")]:
        for hz in (1, 2):
            for st, base in [("brier", 0.40), ("log", 0.90), ("crps", 0.20)]:
                rows.append((q, hz, metric, st, "sibyl", base - 0.05, "run1", False))
                rows.append((q, hz, metric, st, "ensemble_bayesmc_v2", base, "run1", False))
                rows.append((q, hz, metric, st, "ensemble_mean_v2", base + 0.10, "run1", False))
    # Q3 ensemble-only brier row (must be excluded — no sibyl score).
    rows.append(("Q3", 1, "PA", "brier", "ensemble_bayesmc_v2", 0.30, "run1", False))
    # A test-mode sibyl+ensemble pair that must be hidden unless include_test.
    rows.append(("QT", 1, "FATALITIES", "brier", "sibyl", 0.10, "run1", True))
    rows.append(("QT", 1, "FATALITIES", "brier", "ensemble_bayesmc_v2", 0.20, "run1", True))
    con.executemany(
        "INSERT INTO scores VALUES (?,?,?,?,?,?,?,?)",
        [(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]) for r in rows],
    )
    con.close()


@pytest.fixture()
def api_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _boot(with_scores: bool) -> TestClient:
        db_path = tmp_path / f"api_{with_scores}.duckdb"
        _make_db(db_path, with_scores=with_scores)
        config_path = tmp_path / f"config_{with_scores}.yaml"
        config_path.write_text(
            f"app:\n  db_url: 'duckdb:///{db_path}'\n", encoding="utf-8"
        )
        monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(config_path))
        pythia_config.load.cache_clear()
        _app_mod._READ_CON = None
        return TestClient(app)

    try:
        yield _boot
    finally:
        _app_mod._READ_CON = None
        pythia_config.load.cache_clear()


def test_covered_set_and_best_available_baseline(api_env) -> None:
    client = api_env(True)
    resp = client.get("/v1/performance/sibyl_comparison")
    assert resp.status_code == 200
    body = resp.json()

    assert body["has_sibyl"] is True
    # Q3 is ensemble-only → never in pairs. Only Q1/Q2.
    qids = {p["question_id"] for p in body["pairs"]}
    assert qids == {"Q1", "Q2"}
    # 2 questions x 2 horizons x 3 score types = 12 pairs.
    assert len(body["pairs"]) == 12
    # Best-available prefers bayesmc.
    assert {p["standard_model_name"] for p in body["pairs"]} == {"ensemble_bayesmc_v2"}
    assert body["baseline_used"] == "best_available"
    assert body["available_baselines"] == ["ensemble_bayesmc_v2", "ensemble_mean_v2"]

    spd = body["aggregate"]["spd"]
    # Sibyl is uniformly 0.05 better → negative delta, 100% win rate.
    assert spd["brier"]["mean_delta"] == pytest.approx(-0.05, abs=1e-9)
    assert spd["brier"]["win_rate"] == pytest.approx(1.0)
    assert spd["brier"]["sibyl_wins"] == 2
    assert spd["brier"]["n_questions"] == 2
    # Never a binary family here (Sibyl pairs are all SPD).
    assert "binary" not in body["aggregate"]

    # Metadata joined from sibyl_forecasts.
    q1 = next(p for p in body["pairs"] if p["question_id"] == "Q1")
    assert q1["js_divergence_vs_standard"] == pytest.approx(0.12)
    assert q1["volatility_score"] == pytest.approx(0.8)

    # Runs populated for coverage/cost KPIs.
    assert body["runs"] and body["runs"][0]["n_forecast"] == 8


def test_pinned_baseline(api_env) -> None:
    client = api_env(True)
    resp = client.get(
        "/v1/performance/sibyl_comparison", params={"baseline": "ensemble_mean_v2"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["baseline_used"] == "ensemble_mean_v2"
    assert {p["standard_model_name"] for p in body["pairs"]} == {"ensemble_mean_v2"}
    # vs mean, Sibyl is 0.15 better on brier.
    assert body["aggregate"]["spd"]["brier"]["mean_delta"] == pytest.approx(-0.15, abs=1e-9)


def test_by_hazard_metric_rollup(api_env) -> None:
    client = api_env(True)
    body = client.get("/v1/performance/sibyl_comparison").json()
    rollup = {(r["hazard_code"], r["metric"]): r for r in body["by_hazard_metric"]}
    assert set(rollup) == {("ACE", "FATALITIES"), ("TC", "PA")}
    assert rollup[("ACE", "FATALITIES")]["delta"] == pytest.approx(-0.05, abs=1e-9)


def test_include_test_filtering(api_env) -> None:
    client = api_env(True)
    # Default excludes the test-mode QT question.
    default_body = client.get("/v1/performance/sibyl_comparison").json()
    assert "QT" not in {p["question_id"] for p in default_body["pairs"]}
    # include_test surfaces it.
    it_body = client.get(
        "/v1/performance/sibyl_comparison", params={"include_test": "true"}
    ).json()
    assert "QT" in {p["question_id"] for p in it_body["pairs"]}


def test_empty_when_no_scores(api_env) -> None:
    # Post-DB-reset reality: sibyl_runs exist but no scores yet.
    client = api_env(False)
    resp = client.get("/v1/performance/sibyl_comparison")
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_sibyl"] is False
    assert body["pairs"] == []
    assert body["aggregate"] == {}
    # Runs still surface so the UI can show an "awaiting resolutions" state.
    assert body["runs"] and body["runs"][0]["sibyl_run_id"] == "sr1"


def test_an_earlier_run_of_a_rerun_question_is_ignored(api_env, tmp_path) -> None:
    """A question forecast twice keeps only its LATEST run on both sides."""

    client = api_env(True)
    db = next(tmp_path.glob("api_True.duckdb"))
    con = duckdb.connect(str(db))
    # run0 predates run1: a far worse Sibyl score and a far better ensemble
    # one. If either side read run0, the delta or the pair count would move.
    con.executemany(
        "INSERT INTO scores VALUES (?,?,?,?,?,?,?,?)",
        [
            ("Q1", 1, "FATALITIES", "brier", "sibyl", 1.90, "run0", False),
            ("Q1", 1, "FATALITIES", "brier", "ensemble_bayesmc_v2", 0.01, "run0", False),
        ],
    )
    con.close()
    _app_mod._READ_CON = None
    body = client.get("/v1/performance/sibyl_comparison").json()
    assert len(body["pairs"]) == 12
    assert body["aggregate"]["spd"]["brier"]["mean_delta"] == pytest.approx(-0.05, abs=1e-9)


def test_a_forecast_without_evidence_is_left_out(api_env, tmp_path) -> None:
    """sibyl_forecasts.evidence_ok FALSE (the July 2026 run): the score rows
    stay in the table, and the comparison leaves the question out."""

    client = api_env(True)
    db = next(tmp_path.glob("api_True.duckdb"))
    con = duckdb.connect(str(db))
    con.execute("ALTER TABLE sibyl_forecasts ADD COLUMN run_id TEXT")
    con.execute("ALTER TABLE sibyl_forecasts ADD COLUMN status TEXT")
    con.execute("ALTER TABLE sibyl_forecasts ADD COLUMN evidence_ok BOOLEAN")
    con.execute(
        "UPDATE sibyl_forecasts SET run_id = 'run1', status = 'ok', "
        "evidence_ok = (question_id <> 'Q2')"
    )
    con.close()
    _app_mod._READ_CON = None
    body = client.get("/v1/performance/sibyl_comparison").json()
    assert {p["question_id"] for p in body["pairs"]} == {"Q1"}
    assert len(body["pairs"]) == 6


def test_variants_are_empty_on_a_db_without_the_new_columns(api_env) -> None:
    body = api_env(True).get("/v1/performance/sibyl_comparison").json()
    assert body["variants"] == {}
    assert len(body["pairs"]) == 12  # the head-to-head is untouched


def test_variants_pair_sibyl_with_its_reference_and_split_by_selection_pass(
    api_env, tmp_path
) -> None:
    """Oct 2026: Sibyl against __ext_sibyl_ref, __ext_conflictology12 and its
    raw pool, paired on (question, horizon, score_type), split by pass."""

    client = api_env(True)
    db = next(tmp_path.glob("api_True.duckdb"))
    con = duckdb.connect(str(db))
    con.execute("ALTER TABLE sibyl_forecasts ADD COLUMN run_id TEXT")
    con.execute("ALTER TABLE sibyl_forecasts ADD COLUMN status TEXT")
    con.execute("ALTER TABLE sibyl_forecasts ADD COLUMN selection_pass TEXT")
    con.execute("ALTER TABLE sibyl_forecasts ADD COLUMN evidence_ok BOOLEAN")
    con.execute(
        "UPDATE sibyl_forecasts SET run_id = 'run1', status = 'ok', evidence_ok = TRUE, "
        "selection_pass = CASE question_id WHEN 'Q1' THEN 'floor' ELSE 'control' END"
    )
    rows = []
    for q, metric in [("Q1", "FATALITIES"), ("Q2", "PA")]:
        for hz in (1, 2):
            for st, base in [("brier", 0.40), ("log", 0.90), ("crps", 0.20)]:
                rows.append((q, hz, metric, st, "__ext_sibyl_ref", base + 0.02, None, False))
                rows.append((q, hz, metric, st, "__ext_sibyl_raw", base - 0.10, None, False))
    rows.append(("Q1", 1, "FATALITIES", "brier", "__ext_conflictology12", 0.50, None, False))
    con.executemany("INSERT INTO scores VALUES (?,?,?,?,?,?,?,?)", rows)
    con.close()
    _app_mod._READ_CON = None

    v = client.get("/v1/performance/sibyl_comparison").json()["variants"]
    ref = v["vs_sibyl_ref"]
    assert ref["n_questions"] == 2
    # sibyl = base - 0.05, ref = base + 0.02: Sibyl better by 0.07.
    assert ref["aggregate"]["spd"]["brier"]["mean_delta"] == pytest.approx(-0.07)
    assert set(ref["by_selection_pass"]) == {"floor", "control"}
    assert ref["by_selection_pass"]["control"]["spd"]["brier"]["n_questions"] == 1
    c12 = v["vs_conflictology12"]
    assert c12["n_questions"] == 1
    assert c12["aggregate"]["spd"]["brier"]["mean_delta"] == pytest.approx(0.35 - 0.50)
    raw = v["raw_vs_sibyl_ref"]
    assert raw["aggregate"]["spd"]["brier"]["mean_delta"] == pytest.approx(-0.12)
    assert v["vs_raw"]["aggregate"]["spd"]["log"]["mean_delta"] == pytest.approx(0.05)
