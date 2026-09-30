# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Advice counted in distinct questions, carried by model family, fitted on raw forecasts."""

from __future__ import annotations

import json

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.tools import generate_calibration_advice as gca


def _db():
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE questions (question_id TEXT, hazard_code TEXT, metric TEXT, "
        "is_test BOOLEAN DEFAULT FALSE)"
    )
    con.execute(
        "CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, score_type TEXT, "
        "model_name TEXT, value DOUBLE, run_id TEXT)"
    )
    con.execute("CREATE TABLE forecasts_ensemble (question_id TEXT, run_id TEXT)")
    con.execute(
        "CREATE TABLE forecasts_raw (run_id TEXT, question_id TEXT, model_name TEXT, "
        "month_index INTEGER, bucket_index INTEGER, probability DOUBLE, reasoning_trace_json TEXT)"
    )
    con.execute("CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, value DOUBLE)")
    return con


def _score(con, q, model, v, run="fc_9"):
    con.execute("INSERT INTO scores VALUES (?, 1, 'brier', ?, ?, ?)", [q, model, v, run])


def test_nine_reruns_of_two_questions_are_two_questions():
    con = _db()
    for q in ("Q1", "Q2"):
        con.execute("INSERT INTO questions VALUES (?, 'ACE', 'FATALITIES', FALSE)", [q])
        for r in range(1, 10):
            con.execute("INSERT INTO forecasts_ensemble VALUES (?, ?)", [q, f"fc_{r}"])
            _score(con, q, "gpt-6-sol", 0.3, run=f"fc_{r}")
    out = gca._compute_per_model_brier(con, "ACE", "FATALITIES")
    (m,) = out["all_models"]
    assert m["n"] == 2 < gca.MIN_DISTINCT_QUESTIONS_PER_MODEL == 20
    # And a family built from them writes nothing either.
    assert gca._write_family_advice(con, "ACE", "FATALITIES", "2026-09", out["all_models"], {}, "v1") == 0


def test_recalibration_copies_are_never_advised():
    con = _db()
    con.execute("INSERT INTO questions VALUES ('Q1', 'ACE', 'FATALITIES', FALSE)")
    con.execute("INSERT INTO forecasts_ensemble VALUES ('Q1', 'fc_9')")
    for name in ("gpt-6-sol", "gpt-6-sol__raw", "gpt-6-sol__recal"):
        _score(con, "Q1", name, 0.3)
    names = {m["name"] for m in gca._compute_per_model_brier(con, "ACE", "FATALITIES")["all_models"]}
    assert names == {"gpt-6-sol"}


def test_family_row_pools_versions_and_records_who_contributed(monkeypatch):
    con = _db()
    for i in range(25):
        q = f"Q{i}"
        con.execute("INSERT INTO questions VALUES (?, 'ACE', 'FATALITIES', FALSE)", [q])
        con.execute("INSERT INTO forecasts_ensemble VALUES (?, 'fc_9')", [q])
        _score(con, q, "gpt-5.6-sol" if i < 20 else "gpt-6-sol", 0.3)
    written: list = []
    monkeypatch.setattr(
        gca, "_upsert_advice",
        lambda conn, month, hz, m, text, findings, model_name, advice_version=None: written.append(
            (model_name, text, findings)
        ),
    )
    all_models = gca._compute_per_model_brier(con, "ACE", "FATALITIES")["all_models"]
    n = gca._write_family_advice(con, "ACE", "FATALITIES", "2026-09", all_models, {}, "v1")
    assert n == 1
    name, _text, findings = written[0]
    assert name == "family:gpt"
    assert findings["n_questions"] == 25
    assert findings["contributing_ids"] == {"gpt-5.6-sol": 20, "gpt-6-sol": 5}
    json.dumps(findings)


def test_the_raw_forecast_is_read_where_a_correction_was_applied():
    con = _db()
    con.execute("INSERT INTO questions VALUES ('Q1', 'ACE', 'FATALITIES', FALSE)")
    con.execute("INSERT INTO forecasts_ensemble VALUES ('Q1', 'fc_9')")
    con.execute("INSERT INTO resolutions VALUES ('Q1', 1, 300)")
    # Corrected row: 90% on bucket 5. Raw row: 10% on bucket 5.
    for b in range(1, 8):
        con.execute(
            "INSERT INTO forecasts_raw VALUES ('fc_9','Q1','gpt-6-sol',1,?,?,NULL)",
            [b, 0.9 if b == 5 else 0.1 / 6],
        )
        con.execute(
            "INSERT INTO forecasts_raw VALUES ('fc_9','Q1','gpt-6-sol__raw',1,?,?,NULL)",
            [b, 0.1 if b == 5 else 0.9 / 6],
        )
    cal = gca._compute_per_model_bucket_calibration(con, "ACE", "FATALITIES", "gpt-6-sol")
    b5 = next(e for e in cal if e["bucket_index"] == 5)
    assert b5["mean_assigned"] == pytest.approx(0.1)
    assert b5["n_samples"] == 1


def test_advice_impact_waits_for_ten_questions_an_arm():
    con = _db()
    con.execute("ALTER TABLE forecasts_ensemble ADD COLUMN advice_arm TEXT")
    for i in range(24):
        q = f"Q{i}"
        arm = "advice" if i % 2 else "no_advice"
        con.execute("INSERT INTO questions VALUES (?, 'ACE', 'FATALITIES', FALSE)", [q])
        con.execute("INSERT INTO forecasts_ensemble VALUES (?, 'fc_9', ?)", [q, arm])
        _score(con, q, "ensemble_mean_v2", 0.2 if arm == "advice" else 0.4)
    out = gca.compute_advice_arm_impact(con, "ACE", "FATALITIES")
    assert out["available"] is True
    assert out["n_advice"] == 12 and out["n_no_advice"] == 12
    assert out["difference"] == pytest.approx(-0.2)
    con.execute("DELETE FROM forecasts_ensemble WHERE advice_arm = 'no_advice' AND question_id > 'Q14'")
    thin = gca.compute_advice_arm_impact(con, "ACE", "FATALITIES")
    assert thin["available"] is False and "10" in thin["reason"]
    con2 = _db()
    assert gca.compute_advice_arm_impact(con2, "ACE", "FATALITIES")["available"] is False
