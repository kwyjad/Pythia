# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""The forecast of record after a same-epoch rerun (Oct 2026).

The 13 Oct 2026 run opens epoch 2026-11, the epoch of the 1 Oct run, and
re-asks most of its questions under the same question ids. Every reader that
counts a question once must take the question's LATEST PRODUCTION run, and a
question the rerun did not ask again must keep its 1 Oct forecast. Before
this, readers disagreed: calibration counted a re-asked question once per
run, recalibration, the headline and the experiments took MAX(run_id) over
test runs too (the 2, 6 and 7 Oct test runs are all in epoch 2026-11), the
scored bundle took the run with the most score rows, the Sibyl comparison
dropped a question the rerun re-asked without Sibyl, and the Performance
summary averaged every run.

Fixture: Q1 re-asked (1 Oct production, 13 Oct production), Q2 kept
(1 Oct production, then a 7 Oct TEST run). Scores carry the run in their
value: 0.1 = 1 Oct, 0.5 = test, 0.9 = 13 Oct.
"""

from __future__ import annotations

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.db.schema import ensure_schema
from pythia.tools.forecast_of_record import record_run_clause, record_runs

OLD, TEST, NEW = "fc_1790831584", "fc_1791380544", "fc_1791860000"
VALUE = {OLD: 0.1, TEST: 0.5, NEW: 0.9}
MEMBER = "gpt-6-sol"


def _build(con) -> None:
    ensure_schema(con)
    # compute_resolutions / compute_scores create these two.
    con.execute(
        "CREATE TABLE IF NOT EXISTS resolutions (question_id TEXT, horizon_m INTEGER, "
        "observed_month TEXT, value DOUBLE, is_test BOOLEAN, scoring_class TEXT)"
    )
    con.execute(
        "CREATE TABLE IF NOT EXISTS scores (question_id TEXT, horizon_m INTEGER, metric TEXT, "
        "score_type TEXT, model_name TEXT, value DOUBLE, run_id TEXT, "
        "created_at TIMESTAMP DEFAULT now(), is_test BOOLEAN)"
    )
    for hs, at, test in [("hs_20261001T045127", "2026-10-01", False),
                         ("hs_20261007T132648", "2026-10-07", True),
                         ("hs_20261013T000500", "2026-10-13", False)]:
        con.execute("INSERT INTO hs_runs (hs_run_id, generated_at, is_test) VALUES (?, ?, ?)",
                    [hs, at, test])
    for qid, hs in [("SOM_ACE_FATALITIES_2026-11", "hs_20261013T000500"),
                    ("ETH_ACE_FATALITIES_2026-11", "hs_20261001T045127")]:
        con.execute(
            "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric, target_month, "
            "window_start_date, window_end_date, wording, status, track, is_test) "
            "VALUES (?, ?, ?, 'ACE', 'FATALITIES', '2027-04', DATE '2026-11-01', DATE '2027-04-30', "
            "'w', 'active', 1, FALSE)",
            [qid, hs, qid[:3]],
        )
        con.execute("INSERT INTO resolutions (question_id, horizon_m, observed_month, value, is_test) "
                    "VALUES (?, 1, '2026-11', 7, FALSE)", [qid])
    q1, q2 = "SOM_ACE_FATALITIES_2026-11", "ETH_ACE_FATALITIES_2026-11"
    runs = [(OLD, q1, False, "advice"), (NEW, q1, False, "no_advice"),
            (OLD, q2, False, "advice"), (TEST, q2, True, "no_advice")]
    for run, qid, test, arm in runs:
        for model in ("ensemble_mean_v2", MEMBER):
            for b in range(1, 8):
                for tbl in ("forecasts_ensemble", "forecasts_raw"):
                    if tbl == "forecasts_ensemble" and model == MEMBER:
                        continue
                    con.execute(
                        f"INSERT INTO {tbl} (run_id, question_id, model_name, month_index, bucket_index, "
                        "probability, is_test, advice_arm) VALUES (?, ?, ?, 1, ?, ?, ?, ?)",
                        [run, qid, model, b, 1.0 / 7, test, arm],
                    )
            con.execute(
                "INSERT INTO scores (question_id, horizon_m, metric, score_type, model_name, value, "
                "run_id, is_test) VALUES (?, 1, 'FATALITIES', 'brier', ?, ?, ?, ?)",
                [qid, model, VALUE[run], run, test],
            )
    # Sibyl forecast Q1 on 1 Oct only; the 13 Oct rerun re-asked Q1 without it.
    con.execute("INSERT INTO scores (question_id, horizon_m, metric, score_type, model_name, value, "
                "run_id, is_test) VALUES (?, 1, 'FATALITIES', 'brier', 'sibyl', 0.05, ?, FALSE)",
                [q1, OLD])


@pytest.fixture()
def con():
    c = duckdb.connect(":memory:")
    _build(c)
    yield c
    c.close()


Q1, Q2 = "SOM_ACE_FATALITIES_2026-11", "ETH_ACE_FATALITIES_2026-11"


def test_the_record_is_the_latest_production_run_per_question(con):
    assert record_runs(con) == {Q1: NEW, Q2: OLD}
    # With test runs admitted, the 7 Oct test run is Q2's newest.
    assert record_runs(con, include_test=True) == {Q1: NEW, Q2: TEST}
    rows = con.execute(
        "SELECT question_id, run_id FROM scores s WHERE model_name = 'ensemble_mean_v2'"
        + record_run_clause(con, "s")
    ).fetchall()
    assert sorted(rows) == sorted([(Q1, NEW), (Q2, OLD)])


def test_every_run_stays_stored(con):
    """The 1 Oct run keeps its rows: nothing here deletes, and per-run
    analyses can still select it by id."""
    assert con.execute("SELECT COUNT(DISTINCT question_id) FROM forecasts_raw WHERE run_id = ?",
                       [OLD]).fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM scores WHERE run_id = ?", [OLD]).fetchone()[0] == 5


def test_calibration_counts_a_question_once(con):
    from pythia.tools.compute_calibration_pythia import _load_samples

    samples = _load_samples(con, "2026-12")
    got = sorted((s.question_key[0], s.model_name, s.value) for s in samples if s.model_name != "sibyl")
    assert got == [("ETH", "ensemble_mean_v2", 0.1), ("ETH", MEMBER, 0.1),
                   ("SOM", "ensemble_mean_v2", 0.9), ("SOM", MEMBER, 0.9)]


def test_family_recalibration_reads_the_record_and_never_a_test_run(con):
    from pythia.tools.family_recalibration import _member_rows

    runs = {(r[0], r[10]) for r in _member_rows(con)}
    assert runs == {(Q1, NEW), (Q2, OLD)}


def test_headline_reads_the_record(con):
    from scripts.ai_bundle.error_attribution import _latest_runs

    assert _latest_runs(con) == {Q1: NEW, Q2: OLD}


def test_scored_bundle_record_reads_the_record(con):
    from scripts.ai_bundle.build_scored_forecast_bundle import _forecast_run_id_for_question

    assert _forecast_run_id_for_question(con, Q1, False) == NEW
    assert _forecast_run_id_for_question(con, Q2, False) == OLD


def test_advice_arms_count_each_question_once_from_its_record(con, tmp_path):
    """Q1's record is 13 Oct (no_advice), Q2's is 1 Oct (advice); the 7 Oct
    test run's no_advice stamp on Q2 is not an arm."""
    from scripts.ai_bundle.experiments import emit_advice_experiment

    rows = emit_advice_experiment(con, tmp_path, [Q1, Q2])
    assert len(rows) == 1
    assert (rows[0]["n_advice"], rows[0]["n_no_advice"]) == (1, 1)
    assert rows[0]["mean_brier_advice"] == pytest.approx(0.1)
    assert rows[0]["mean_brier_no_advice"] == pytest.approx(0.9)


def test_calibration_advice_reads_the_record(con):
    from pythia.tools.generate_calibration_advice import _latest_run_clause

    rows = con.execute(
        "SELECT question_id, run_id FROM scores s WHERE model_name = ?"
        + _latest_run_clause(con, "scores", "s"), [MEMBER]
    ).fetchall()
    assert sorted(rows) == sorted([(Q1, NEW), (Q2, OLD)])


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from pythia import config as pythia_config
    import pythia.api.app as app_mod

    db_path = tmp_path / "record.duckdb"
    c = duckdb.connect(str(db_path))
    _build(c)
    c.close()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"app:\n  db_url: 'duckdb:///{db_path}'\n", encoding="utf-8")
    monkeypatch.setenv("PYTHIA_CONFIG_PATH", str(cfg))
    pythia_config.load.cache_clear()
    app_mod._READ_CON = None
    try:
        yield TestClient(app_mod.app)
    finally:
        app_mod._READ_CON = None
        pythia_config.load.cache_clear()


def test_performance_summary_counts_a_question_once(client):
    body = client.get("/v1/performance/scores", params={"metric": "FATALITIES"}).json()
    row = next(r for r in body["summary_rows"]
               if r["model_name"] == "ensemble_mean_v2" and r["score_type"] == "brier")
    assert row["n_samples"] == 2
    assert row["avg_value"] == pytest.approx(0.5)  # 0.9 (Q1, 13 Oct) and 0.1 (Q2, 1 Oct)


def test_sibyl_is_compared_with_the_standard_forecast_of_its_own_run(client):
    """Sibyl forecast Q1 on 1 Oct; the 13 Oct rerun re-asked Q1 without
    Sibyl. The pair stays, against the 1 Oct standard forecast."""
    body = client.get("/v1/performance/sibyl_comparison").json()
    pairs = body["pairs"]
    assert [(p["question_id"], p["standard_value"]) for p in pairs] == [(Q1, pytest.approx(0.1))]

