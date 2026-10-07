# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Indicative ACE/PA months stay out of everything that learns or headlines.

``resolutions.scoring_class = 'indicative'`` (pythia/tools/scoring_class.py)
marks an ACE/PA month whose country IDMC does not report regularly: it is
forecast, published, resolved and scored, but it never reaches calibration
weights, advice, recalibration, centroids, the Sibyl comparison or a
headline skill figure. Each test below builds one scored question (QS) and
one indicative question (QI) that differ only in that column, and fails if
the indicative one leaks in.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import duckdb
import pytest

RES_DDL = """
CREATE TABLE resolutions (
  question_id TEXT, horizon_m INTEGER, observed_month TEXT, value DOUBLE,
  source_snapshot_ym TEXT, source_desc TEXT, created_at TIMESTAMP,
  is_test BOOLEAN DEFAULT FALSE, scoring_class TEXT, scoring_class_reason TEXT
)
"""
REASON = "not a regular IDMC reporter: reported in fewer than 8 of the 12 months before 2026-03"


def _base(con, questions, *, with_class: bool = True) -> None:
    con.execute("CREATE TABLE hs_runs (hs_run_id TEXT, generated_at TIMESTAMP)")
    con.execute("INSERT INTO hs_runs VALUES ('hs1', TIMESTAMP '2026-02-13')")
    con.execute(
        "CREATE TABLE questions (question_id TEXT, hs_run_id TEXT, iso3 TEXT, hazard_code TEXT, "
        "metric TEXT, target_month TEXT, window_start_date DATE, window_end_date DATE, "
        "wording TEXT, status TEXT, track INTEGER, pythia_metadata_json TEXT, "
        "is_test BOOLEAN DEFAULT FALSE)"
    )
    ddl = RES_DDL if with_class else RES_DDL.replace(
        ", scoring_class TEXT, scoring_class_reason TEXT", "")
    con.execute(ddl)
    con.execute(
        "CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, metric TEXT, score_type TEXT, "
        "model_name TEXT, value DOUBLE, run_id TEXT, is_test BOOLEAN DEFAULT FALSE)"
    )
    con.execute("CREATE TABLE forecasts_ensemble (question_id TEXT, run_id TEXT, is_test BOOLEAN DEFAULT FALSE)")
    for qid, iso3, cls in questions:
        con.execute(
            "INSERT INTO questions VALUES (?, 'hs1', ?, 'ACE', 'PA', '2026-08', DATE '2026-03-01', "
            "DATE '2026-08-31', 'w', 'active', 1, NULL, FALSE)",
            [qid, iso3],
        )
        if with_class:
            con.execute(
                "INSERT INTO resolutions VALUES (?, 1, '2026-03', 5000, '2026-03', 'idmc', "
                "TIMESTAMP '2026-10-01 00:00:00', FALSE, ?, ?)",
                [qid, cls, REASON if cls == "indicative" else None],
            )
        else:
            con.execute(
                "INSERT INTO resolutions VALUES (?, 1, '2026-03', 5000, '2026-03', 'idmc', "
                "TIMESTAMP '2026-10-01 00:00:00', FALSE)",
                [qid],
            )
        con.execute("INSERT INTO forecasts_ensemble VALUES (?, 'r1', FALSE)", [qid])
        for model, v in (("gpt-6-sol", 0.4), ("ensemble_mean_v2", 0.5), ("__ext_climatology", 0.6)):
            for st in ("brier", "crps"):
                con.execute(
                    "INSERT INTO scores VALUES (?, 1, 'PA', ?, ?, ?, ?, FALSE)",
                    [qid, st, model, v, None if model.startswith("__ext_") else "r1"],
                )


@pytest.fixture()
def con(tmp_path):
    c = duckdb.connect(str(tmp_path / "s.duckdb"))
    _base(c, [("QS", "SOM", "scored"), ("QI", "AFG", "indicative")])
    yield c
    c.close()


# --- calibration weights ----------------------------------------------------

def test_calibration_weight_samples_leave_out_indicative_months(con):
    from pythia.tools.compute_calibration_pythia import _load_samples

    samples = _load_samples(con, "2026-12")
    assert samples, "the scored question must still be sampled"
    assert {s.question_key[0] for s in samples} == {"SOM"}


def test_calibration_samples_on_a_db_without_the_column(tmp_path):
    from pythia.tools.compute_calibration_pythia import _load_samples

    c = duckdb.connect(str(tmp_path / "old.duckdb"))
    _base(c, [("QS", "SOM", None), ("QI", "AFG", None)], with_class=False)
    assert {s.question_key[0] for s in _load_samples(c, "2026-12")} == {"SOM", "AFG"}


# --- calibration advice -----------------------------------------------------

def test_advice_counts_shown_to_models_leave_out_indicative_months(con):
    from pythia.tools import generate_calibration_advice as gca

    assert gca._count_resolved(con, "ACE", "PA") == 1
    per, total = gca._family_question_counts(con, "ACE", "PA", ["gpt-6-sol"])
    assert per == {"gpt-6-sol": 1} and total == 1
    brier = gca._compute_per_model_brier(con, "ACE", "PA")
    assert brier is not None
    assert all(m.get("n_questions", 1) == 1 for m in brier.get("all_models", []))


def test_advice_pairs_are_not_discovered_from_indicative_months_alone(tmp_path):
    from pythia.tools import generate_calibration_advice as gca

    c = duckdb.connect(str(tmp_path / "ind.duckdb"))
    _base(c, [("QI", "AFG", "indicative")])
    assert gca._discover_hazard_metric_pairs(c) == []


# --- family recalibration fit -----------------------------------------------

def test_family_recalibration_fits_no_indicative_month(con):
    from pythia.tools import family_recalibration as fr

    con.execute(
        "CREATE TABLE forecasts_raw (run_id TEXT, question_id TEXT, model_name TEXT, "
        "month_index INTEGER, bucket_index INTEGER, probability DOUBLE)"
    )
    for q in ("QS", "QI"):
        for b in range(1, 7):
            con.execute("INSERT INTO forecasts_raw VALUES ('r1', ?, 'gpt-6-sol', 1, ?, ?)",
                        [q, b, 1 / 6])
    assert {r[0] for r in fr._member_rows(con)} == {"QS"}


# --- centroid EMA -----------------------------------------------------------

def test_indicative_resolutions_never_move_a_centroid(tmp_path):
    from pythia.tools.compute_bucket_centroids import update_bucket_centroids_ema
    from resolver.db import duckdb_io

    db = tmp_path / "c.duckdb"
    c = duckdb.connect(str(db))
    _base(c, [])
    c.execute(
        "CREATE TABLE bucket_centroids (hazard_code TEXT, metric TEXT, bucket_index INTEGER, "
        "centroid DOUBLE, as_of_month TEXT, schema_version TEXT)"
    )
    # 12 indicative months in bucket 2 (1-<10k) and 12 scored months in bucket 3.
    for i in range(12):
        for qid, val, cls in ((f"I{i}", 5000.0, "indicative"), (f"S{i}", 20000.0, "scored")):
            c.execute(
                "INSERT INTO questions VALUES (?, 'hs1', 'XXX', 'ACE', 'PA', '2026-08', "
                "DATE '2026-03-01', DATE '2026-08-31', 'w', 'active', 1, NULL, FALSE)", [qid])
            c.execute(
                "INSERT INTO resolutions VALUES (?, 1, '2026-03', ?, NULL, NULL, NULL, FALSE, ?, NULL)",
                [qid, val, cls])
    c.close()
    update_bucket_centroids_ema(f"duckdb:///{db}", metric="PA", as_of=date(2026, 10, 7))
    try:
        duckdb_io.close_db(duckdb_io.get_db(f"duckdb:///{db}"))
    except Exception:  # noqa: BLE001
        pass
    c = duckdb.connect(str(db))
    buckets = {r[0] for r in c.execute(
        "SELECT bucket_index FROM bucket_centroids WHERE as_of_month = '2026-10'").fetchall()}
    c.close()
    assert buckets == {3}


# --- the API: performance scores and the Sibyl comparison --------------------

@pytest.fixture()
def api(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pythia import config as pythia_config
    import pythia.api.app as app_mod

    db = tmp_path / "api.duckdb"
    c = duckdb.connect(str(db))
    _base(c, [("QS", "SOM", "scored"), ("QI", "AFG", "indicative")])
    # Sibyl and the standard track on both questions.
    for q in ("QS", "QI"):
        for st, base in (("brier", 0.4), ("crps", 0.2)):
            c.execute("INSERT INTO scores VALUES (?, 1, 'PA', ?, 'sibyl', ?, 'r1', FALSE)",
                      [q, st, base - (0.05 if q == "QS" else -0.3)])
            c.execute("INSERT INTO scores VALUES (?, 1, 'PA', ?, 'ensemble_bayesmc_v2', ?, 'r1', FALSE)",
                      [q, st, base])
    c.execute(
        "CREATE TABLE sibyl_forecasts (question_id TEXT, sibyl_run_id TEXT, run_id TEXT, "
        "created_at TIMESTAMP, status TEXT, js_divergence_vs_standard DOUBLE, "
        "js_divergence_inter_trial DOUBLE, volatility_score DOUBLE, cost_usd DOUBLE, "
        "is_test BOOLEAN DEFAULT FALSE)"
    )
    for q in ("QS", "QI"):
        c.execute("INSERT INTO sibyl_forecasts VALUES (?, 'sr1', 'r1', TIMESTAMP '2026-03-13', "
                  "'ok', 0.1, 0.05, 0.5, 1.0, FALSE)", [q])
    c.close()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"app:\n  db_url: 'duckdb:///{db}'\n", encoding="utf-8")
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(cfg))
    pythia_config.load.cache_clear()
    app_mod._READ_CON = None
    try:
        yield TestClient(app_mod.app)
    finally:
        app_mod._READ_CON = None
        pythia_config.load.cache_clear()


def test_sibyl_comparison_pairs_leave_out_indicative_months(api):
    body = api.get("/v1/performance/sibyl_comparison").json()
    assert body["has_sibyl"] is True
    assert {p["question_id"] for p in body["pairs"]} == {"QS"}
    assert body["aggregate"]["spd"]["brier"]["mean_delta"] == pytest.approx(-0.05)


def test_performance_scores_summary_counts_what_it_left_out(api):
    body = api.get("/v1/performance/scores").json()
    rows = [r for r in body["summary_rows"]
            if r["model_name"] == "gpt-6-sol" and r["score_type"] == "brier"]
    assert len(rows) == 1
    row = rows[0]
    assert row["n_questions"] == 1
    assert row["n_indicative_excluded"] == 1
    assert row["avg_value"] == pytest.approx(0.4)
    assert body["track_counts"]["total"] == 1
    assert all("n_indicative_excluded" in r for r in body["run_rows"])


# --- Sibyl's own record -----------------------------------------------------

def test_sibyl_question_scores_leave_out_indicative_months(con):
    from types import SimpleNamespace

    from sibyl.advice import load_question_scores

    for q in ("QS", "QI"):
        con.execute("INSERT INTO scores VALUES (?, 1, 'PA', 'brier', 'sibyl', 0.3, 'r1', FALSE)", [q])
    recs = [SimpleNamespace(question_id=q, forecast_run_id="r1") for q in ("QS", "QI")]
    assert set(load_question_scores(con, recs)) == {"QS"}


def test_sibyl_pool_weight_fit_skips_indicative_months(con):
    from sibyl.score_variants import indicative_keys

    assert indicative_keys(con) == {("QI", 1)}


# --- interpreter Part B -----------------------------------------------------

def test_interpreter_skill_claim_counts_only_scored_pairs():
    from interpreter import performance

    rows = [(0.3, 0.4, "scored")] * 5 + [(0.9, 0.4, "indicative")] * 3 + [(0.3, 0.4)] * 2
    claim = performance.skill_claim(rows, min_resolved=7, samples=50)
    assert claim["n_resolved"] == 7
    assert claim["n_indicative_excluded"] == 3
    assert claim["claim_allowed"] is True
    assert claim["skill"] == pytest.approx(0.25)


# --- the scored bundle ------------------------------------------------------

def test_bundle_indicative_table_and_resolution_reading(con, tmp_path):
    from scripts.ai_bundle import build_scored_forecast_bundle as b

    out = tmp_path / "out"
    out.mkdir()
    n = b._emit_indicative_questions(con, out, ["QS", "QI"])
    rows = list(csv.DictReader((out / "indicative_questions.csv").open()))
    assert n == len(rows) and rows
    assert {r["question_id"] for r in rows} == {"QI"}
    assert rows[0]["scoring_class_reason"] == REASON
    assert {(r["model_name"], r["score_type"]) for r in rows} >= {("gpt-6-sol", "brier")}
    assert rows[0]["score_value"] != ""
    # 2026-03 resolved on 2026-10-01: 184 days after month end, the d180 reading.
    assert rows[0]["resolution_reading"] == "d180"

    b._emit_scores_flat(con, out, ["QS", "QI"])
    flat = list(csv.DictReader((out / "scores_flat.csv").open()))
    by_q = {r["question_id"]: r for r in flat}
    assert by_q["QI"]["scoring_class"] == "indicative"
    assert by_q["QS"]["scoring_class"] == "scored"
    assert {r["resolution_reading"] for r in flat} == {"d180"}

    rollups = b._emit_rollups(con, out, ["QS", "QI"])
    gpt = [r for r in rollups if r["model_name"] == "gpt-6-sol" and r["score_type"] == "brier"]
    assert gpt[0]["n_questions_scored"] == 1
    assert gpt[0]["n_indicative_excluded"] == 1
    assert gpt[0]["skill_vs_climatology"] == pytest.approx(1 - 0.4 / 0.6, abs=1e-4)


def test_bundle_record_resolutions_carry_class_and_reading(con):
    from scripts.ai_bundle import build_scored_forecast_bundle as b

    con.execute("DROP TABLE forecasts_ensemble")
    q = {"question_id": "QI", "iso3": "AFG", "hazard_code": "ACE", "metric": "PA", "hs_run_id": "hs1"}
    record = b.build_question_record(con, q, include_test=False, include_sibyl_trials=False,
                                     run_id="r1")
    (res,) = record["outcome"]["resolutions"]
    assert res["scoring_class"] == "indicative"
    assert res["resolution_reading"] == "d180"
    assert res["scoring_class_reason"] == REASON


def test_headline_leaves_out_indicative_months(con):
    from scripts.ai_bundle import error_attribution as ea

    ctx = ea.build_context(con, ["QS", "QI"])
    assert ctx.indicative == {("QI", 1)}
    head = ea.build_headline(ctx)
    groups = head["groups"]
    assert groups and groups[0]["n_indicative_excluded"] == 1
    assert groups[0]["scores"]["brier"]["n_paired_questions"] == 1
    assert all(p["question_id"] == "QS" for p in ea._skill_pairs(ctx))
