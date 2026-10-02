# Pythia / Copyright (c) 2025 Kevin Wyjad
"""The ACLED history backfill's before/after evidence."""

from __future__ import annotations

from pathlib import Path

import duckdb

from scripts.ci import acled_history_report as rep


def _db(path: Path) -> str:
    con = duckdb.connect(str(path))
    con.execute(
        "CREATE TABLE questions (question_id TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, "
        "window_start_date DATE, is_test BOOLEAN DEFAULT FALSE)"
    )
    con.execute(
        "CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, metric TEXT, score_type TEXT, "
        "model_name TEXT, value DOUBLE, run_id TEXT)"
    )
    con.execute(
        "CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month DATE, fatalities BIGINT, "
        "source TEXT, updated_at TIMESTAMP)"
    )
    for q, iso in (("SOM_ACE_FATALITIES_2026-08", "SOM"), ("KEN_ACE_FATALITIES_2026-08", "KEN")):
        con.execute("INSERT INTO questions VALUES (?, ?, 'ACE', 'FATALITIES', DATE '2026-08-01', FALSE)", [q, iso])
        for m, v in (("__ext_climatology", 0.4), ("__ext_level_volatility", 0.3), ("ensemble_mean_v2", 0.5)):
            con.execute("INSERT INTO scores VALUES (?, 1, 'FATALITIES', 'brier', ?, ?, NULL)", [q, m, v])
    con.execute("INSERT INTO questions VALUES ('X_ACE_FATALITIES_2026-07', 'X', 'ACE', 'FATALITIES', DATE '2026-07-01', FALSE)")
    con.execute("INSERT INTO scores VALUES ('X_ACE_FATALITIES_2026-07', 1, 'FATALITIES', 'brier', '__ext_climatology', 0.9, NULL)")
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES "
        "('SOM', DATE '2018-01-01', 10, 'ACLED', TIMESTAMP '2026-10-02 03:00:00'), "
        "('KEN', DATE '2018-01-01', 2, 'ACLED', TIMESTAMP '2026-10-02 03:00:00'), "
        "('SOM', DATE '2026-09-01', 5, 'ACLED', TIMESTAMP '2026-09-28 03:00:00')"
    )
    con.close()
    return str(path)


def test_snapshot_keeps_the_epochs_references_only(tmp_path):
    db = _db(tmp_path / "a.duckdb")
    out = tmp_path / "before.csv"
    assert rep.main(["snapshot", "--db", db, "--epoch", "2026-08", "--out", str(out)]) == 0
    rows = rep.read_snapshot(out)
    assert {r["model_name"] for r in rows} == {"__ext_climatology", "__ext_level_volatility"}
    assert {r["question_id"] for r in rows} == {"SOM_ACE_FATALITIES_2026-08", "KEN_ACE_FATALITIES_2026-08"}


def test_compare_reports_both_sides_and_the_pairs(tmp_path):
    db = _db(tmp_path / "b.duckdb")
    before = tmp_path / "before.csv"
    rep.main(["snapshot", "--db", db, "--epoch", "2026-08", "--out", str(before)])
    con = duckdb.connect(db)
    con.execute("UPDATE scores SET value = 0.2 WHERE model_name = '__ext_level_volatility'")
    con.execute(
        "INSERT INTO scores SELECT question_id, 1, 'FATALITIES', 'brier', '__ext_level_transition', 0.25, NULL "
        "FROM questions WHERE question_id LIKE '%2026-08'"
    )
    con.close()
    after = tmp_path / "after.csv"
    rep.main(["snapshot", "--db", db, "--epoch", "2026-08", "--out", str(after)])
    out = tmp_path / "cmp.md"
    rep.main(["compare", "--before", str(before), "--after", str(after), "--epoch", "2026-08", "--out", str(out)])
    md = out.read_text()
    assert "| __ext_level_volatility | 0.300 (n=2) | 0.200 (n=2) |" in md
    assert "| __ext_level_transition | — | 0.250 (n=2) |" in md
    assert "paired over the 2 (question, horizon) pairs (2 questions)" in md


def test_verify_counts_partial_rows_and_live_months(tmp_path):
    db = _db(tmp_path / "c.duckdb")
    out = tmp_path / "verify.json"
    rep.main(["verify", "--db", db, "--since", "2018-01", "--out", str(out)])
    import json

    v = json.loads(out.read_text())
    assert v["partial_rows_total"] == 1
    assert v["by_year"]["2018"] == {"months": 1, "rows": 2, "fatalities": 12}
    assert "2018-01" in v["coverage_live_months"]
    assert v["months_not_live"] == ["2026-09"]
