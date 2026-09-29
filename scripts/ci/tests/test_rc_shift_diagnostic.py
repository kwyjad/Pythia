# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""The prior-to-posterior shift/spread diagnostic (scripts/ci/rc_shift_diagnostic.py)."""

from __future__ import annotations

import json
import math

import duckdb
import pytest

from scripts.ci import rc_shift_diagnostic as d


def test_arithmetic():
    assert d.entropy_bits([0.5, 0.5]) == pytest.approx(1.0)
    assert d.expected_index([0.0, 0.0, 1.0]) == 2
    assert d.far_mass([0.1, 0.1, 0.6, 0.1, 0.1], 2) == (0.1, 0.1)
    assert d.brier([1.0, 0.0], 0) == 0.0
    assert d.rps([0.0, 1.0, 0.0], 1) == 0.0
    assert d.rc_direction_group("UP") == "up" and d.rc_direction_group(None) == "unclear"


def test_compare_separates_shift_from_spread():
    prior = [0.1, 0.2, 0.4, 0.2, 0.1]
    shifted = [0.0, 0.1, 0.2, 0.4, 0.3]
    spread = [0.2, 0.2, 0.2, 0.2, 0.2]
    s = d.compare(prior, shifted)
    assert s["shift"] > 0.5 and s["far_below"] < 0
    w = d.compare(prior, spread)
    assert abs(w["shift"]) < 1e-9 and w["h_post"] > w["h_prior"]
    assert w["far_below"] > 0 and w["far_above"] > 0


def test_run_end_to_end(tmp_path):
    con = duckdb.connect(str(tmp_path / "d.duckdb"))
    con.execute(
        "CREATE TABLE questions (question_id TEXT, hs_run_id TEXT, iso3 TEXT, hazard_code TEXT, "
        "metric TEXT, window_start_date DATE, track INTEGER, is_test BOOLEAN)"
    )
    con.execute(
        "CREATE TABLE hs_triage (run_id TEXT, iso3 TEXT, hazard_code TEXT, "
        "regime_change_level INTEGER, regime_change_direction TEXT)"
    )
    con.execute(
        "CREATE TABLE forecasts_raw (run_id TEXT, question_id TEXT, model_name TEXT, "
        "month_index INTEGER, bucket_index INTEGER, probability DOUBLE, reasoning_trace_json TEXT)"
    )
    con.execute("CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month DATE, fatalities DOUBLE)")
    con.execute("INSERT INTO questions VALUES ('q1','hs1','SOM','ACE','FATALITIES','2026-07-01',1,FALSE)")
    con.execute("INSERT INTO hs_triage VALUES ('hs1','SOM','ACE',2,'up')")
    con.execute("INSERT INTO acled_monthly_fatalities VALUES ('SOM','2026-07-01',210)")
    prior = [0.02, 0.05, 0.10, 0.20, 0.45, 0.13, 0.05]
    post = [0.02, 0.05, 0.20, 0.20, 0.25, 0.13, 0.15]
    trace = json.dumps({"prior": {"spd": prior}})
    for i, p in enumerate(post, start=1):
        con.execute("INSERT INTO forecasts_raw VALUES ('r1','q1','m1',1,?,?,?)", [i, p, trace])
    rows = d.load_rows(con)
    assert len(rows) == 1 and rows[0]["rc"] == "1+" and rows[0]["dir"] == "up"
    d.attach_outcomes(con, rows, today_ym="2026-09")
    assert rows[0]["bucket"] == 4  # 210 deaths → 100-<500
    text = d.run(con, today_ym="2026-09")
    assert "FATALITIES: by track and RC level" in text
    assert "| T1 · 1+ | 1 (1) |" in text
    # A month not yet over has no outcome.
    d.attach_outcomes(con, rows, today_ym="2026-07")
    assert rows[0]["bucket"] is None
    assert not math.isnan(d.entropy_bits(post))
