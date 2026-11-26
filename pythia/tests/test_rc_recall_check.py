# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The RC recall check: breaks read from the resolver, scored per arm.

Unresolvable pairs stay out of every denominator; a break is a break even
with months missing; production and shadow are scored apart.
"""

from __future__ import annotations

from datetime import datetime

import duckdb
import pytest

from pythia.tools import rc_recall_check as rr


def _db(tmp_path):
    con = duckdb.connect(str(tmp_path / "r.duckdb"))
    con.execute("CREATE TABLE hs_runs (hs_run_id TEXT, generated_at TIMESTAMP, is_test BOOLEAN)")
    con.execute("""CREATE TABLE hs_triage (run_id TEXT, iso3 TEXT, hazard_code TEXT,
                   regime_change_likelihood DOUBLE, regime_change_level INTEGER, data_quality_json TEXT)""")
    con.execute("""CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month DATE, fatalities DOUBLE)""")
    con.execute("INSERT INTO hs_runs VALUES ('hs_a', '2026-01-13 00:00:00', FALSE), "
                "('hs_t', '2026-01-14 00:00:00', TRUE)")
    # Three years of quiet history (5-24 deaths -> bucket 2) for AAA, BBB, CCC.
    for iso3 in ("AAA", "BBB", "CCC"):
        for i in range(36):
            y, m = divmod(2022 * 12 + 11 + i, 12)  # 2022-12 .. 2025-11
            con.execute("INSERT INTO acled_monthly_fatalities VALUES (?, ?, 10)",
                        [iso3, f"{y:04d}-{m + 1:02d}-01"])
    # Window 2026-02 .. 2026-07. AAA breaks in March (300 deaths: bucket 4).
    # BBB stays quiet for four months. CCC's flood pair has no resolution
    # machine rows at all: unresolvable.
    for ym, v in (("2026-02", 12), ("2026-03", 300)):
        con.execute("INSERT INTO acled_monthly_fatalities VALUES ('AAA', ?, ?)", [ym + "-01", v])
    for ym in ("2026-02", "2026-03", "2026-04", "2026-05"):
        con.execute("INSERT INTO acled_monthly_fatalities VALUES ('BBB', ?, 8)", [ym + "-01"])
    con.execute("INSERT INTO acled_monthly_fatalities VALUES ('ZZZ', '2026-02-01', 1)")
    # Production: AAA quiet (missed the break), BBB flagged (false alarm),
    # CCC's flood flagged (unresolvable).
    con.execute("""INSERT INTO hs_triage VALUES
        ('hs_a','AAA','ACE',0.05,0,NULL), ('hs_a','BBB','ACE',0.6,3,NULL),
        ('hs_a','CCC','FL',0.5,2,NULL),
        ('hs_a','DDD','FL',0.05,0,'{"status": "seasonal_skip"}')""")
    from horizon_scanner import rc_shadow

    # Shadow: catches AAA, leaves BBB quiet.
    for iso3, lvl, lik in (("AAA", 2, 0.5), ("BBB", 0, 0.05)):
        rc_shadow.write_row(con, hs_run_id="hs_a", iso3=iso3, hazard_code="ACE", provider="openai",
                            model_id="gpt-6.1-sol", result={"status": "ok", "level": lvl, "likelihood": lik},
                            is_test=False)
    return con


def test_window_is_the_six_months_after_the_run_month() -> None:
    assert rr.run_window(datetime(2026, 12, 13)) == [
        "2027-01", "2027-02", "2027-03", "2027-04", "2027-05", "2027-06"]


def test_median_bucket() -> None:
    assert rr.median_bucket([0.1, 0.3, 0.2, 0.4]) == 2
    assert rr.median_bucket([0.6, 0.4]) == 0
    assert rr.median_bucket([]) is None


def test_outcomes_and_scores(tmp_path) -> None:
    con = _db(tmp_path)
    rows = rr.compute(con)
    by = {(r["hs_run_id"], r["hazard"], r["arm"]): r for r in rows}
    prod = by[("hs_a", "ACE", "production")]
    shadow = by[("hs_a", "ACE", "shadow")]
    assert prod["n_pairs"] == 2 and prod["n_resolvable"] == 2
    assert prod["n_breaks"] == 1 and prod["n_hits"] == 0 and prod["recall"] == 0.0
    assert prod["precision"] == 0.0 and prod["n_track1"] == 1
    # CCC is unresolvable and out of every denominator; DDD was never
    # assessed (seasonal skip) and is not a pair at all.
    allp = by[("hs_a", "*", "production")]
    assert allp["n_pairs"] == 3 and allp["n_unresolvable"] == 1 and allp["n_resolvable"] == 2
    assert allp["n_flagged"] == 1 and allp["n_track1"] == 2
    assert shadow["n_breaks"] == 1 and shadow["n_hits"] == 1 and shadow["recall"] == 1.0
    assert shadow["precision"] == 1.0 and shadow["n_track1"] == 1
    assert shadow["brier"] == pytest.approx(((0.5 - 1) ** 2 + 0.05 ** 2) / 2, abs=1e-4)
    # Test runs are left out.
    assert not any(r["hs_run_id"] == "hs_t" for r in rows)
    pooled = by[("*", "*", "shadow")]
    assert pooled["n_hits"] == 1
    con.close()


def test_a_missing_month_never_reads_as_no_break(tmp_path) -> None:
    con = _db(tmp_path)
    outcomes = rr.Outcomes(con)
    window = rr.run_window(datetime(2026, 1, 13))
    assert rr.pair_outcome(con, outcomes, "CCC", "FL", window)["state"] == "unresolvable"
    assert rr.pair_outcome(con, outcomes, "BBB", "ACE", window)["state"] == "no_break"
    assert rr.pair_outcome(con, outcomes, "AAA", "ACE", window)["state"] == "break"
    con.close()


def test_write_and_report(tmp_path) -> None:
    con = _db(tmp_path)
    rows = rr.compute(con)
    rr.write(con, rows)
    assert con.execute("SELECT COUNT(*) FROM rc_recall_check").fetchone()[0] == len(rows)
    md = rr.markdown(rows)
    assert "optimistic" in md and "| hs_a |" in md and "(1/1)" in md
    con.close()
