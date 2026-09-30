# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""The scored bundle measures the advice arms and the recalibration effect."""

from __future__ import annotations

import csv

import duckdb
import pytest

from scripts.ai_bundle import experiments as exp


def _db(tmp_path):
    con = duckdb.connect(str(tmp_path / "e.duckdb"))
    con.execute("CREATE TABLE questions (question_id TEXT, hazard_code TEXT, metric TEXT, track INTEGER)")
    con.execute(
        "CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, metric TEXT, score_type TEXT, "
        "model_name TEXT, value DOUBLE, run_id TEXT)"
    )
    con.execute("CREATE TABLE forecasts_ensemble (question_id TEXT, run_id TEXT, advice_arm TEXT)")
    con.execute(
        "CREATE TABLE forecasts_raw (run_id TEXT, question_id TEXT, model_name TEXT, "
        "month_index INTEGER, bucket_index INTEGER, probability DOUBLE)"
    )
    con.execute("CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, value DOUBLE)")
    return con


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_advice_arms_are_compared_per_group(tmp_path):
    con = _db(tmp_path)
    qids = []
    for i in range(20):
        q = f"Q{i}"
        qids.append(q)
        arm = "advice" if i % 2 else "no_advice"
        con.execute("INSERT INTO questions VALUES (?, 'ACE', 'FATALITIES', 1)", [q])
        con.execute("INSERT INTO forecasts_ensemble VALUES (?, 'fc_1', ?)", [q, arm])
        v = 0.3 if arm == "advice" else 0.5
        con.execute("INSERT INTO scores VALUES (?,1,'FATALITIES','brier','ensemble_mean_v2',?,'fc_1')", [q, v])
    rows = exp.emit_advice_experiment(con, tmp_path, qids)
    (row,) = rows
    assert row["n_advice"] == 10 and row["n_no_advice"] == 10
    assert row["difference"] == pytest.approx(-0.2)
    assert row["ci90_low"] == pytest.approx(-0.2) and row["ci90_high"] == pytest.approx(-0.2)
    assert _read(tmp_path / "advice_experiment.csv")[0]["track"] == "1"


def test_no_arm_column_writes_a_header_only_file(tmp_path):
    con = duckdb.connect(str(tmp_path / "x.duckdb"))
    con.execute("CREATE TABLE forecasts_ensemble (question_id TEXT, run_id TEXT)")
    assert exp.emit_advice_experiment(con, tmp_path, ["Q1"]) == []
    assert (tmp_path / "advice_experiment.csv").read_text().startswith("hazard_code,")


def test_recalibration_effect_is_paired_per_member_and_for_the_mean(tmp_path):
    con = _db(tmp_path)
    qids = []
    for i in range(6):
        q = f"Q{i}"
        qids.append(q)
        con.execute("INSERT INTO questions VALUES (?, 'ACE', 'FATALITIES', 1)", [q])
        con.execute("INSERT INTO forecasts_ensemble VALUES (?, 'fc_1', NULL)", [q])
        con.execute("INSERT INTO resolutions VALUES (?, 1, 300)", [q])
        for model, v in (("gpt-6-sol", 0.3), ("gpt-6-sol__raw", 0.5),
                         ("claude-opus-5-5", 0.4), ("claude-opus-5-5__recal", 0.35)):
            con.execute("INSERT INTO scores VALUES (?,1,'FATALITIES','brier',?,?,'fc_1')", [q, model, v])
        for model, p5 in (("gpt-6-sol", 0.6), ("gpt-6-sol__raw", 0.2),
                          ("gemini-3.5-flash", 0.6), ("gemini-3.5-flash__raw", 0.2)):
            for b in range(1, 8):
                con.execute(
                    "INSERT INTO forecasts_raw VALUES ('fc_1', ?, ?, 1, ?, ?)",
                    [q, model, b, p5 if b == 5 else (1 - p5) / 6],
                )
    rows = exp.emit_recalibration_effect(con, tmp_path, qids)
    by = {(r["model"], r["comparison"]): r for r in rows}
    applied = by[("gpt-6-sol", "applied")]
    assert applied["n_paired"] == 6 and applied["difference"] == pytest.approx(-0.2)
    shadow = by[("claude-opus-5-5", "shadow")]
    assert shadow["difference"] == pytest.approx(-0.05)
    mean = by[("member_mean (unweighted)", "applied")]
    assert mean["n_paired"] == 6 and mean["mean_corrected"] < mean["mean_uncorrected"]
