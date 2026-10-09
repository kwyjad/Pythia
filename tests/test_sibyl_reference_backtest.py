# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Mechanical backtests of Sibyl's references (Oct 2026, review Part 7).

Production timing and knowability, the candidates per class, unresolved
months left out, "too few" for flood and cyclone, the fitted drought
schedule held non-increasing and scored out of sample, fixed-seed intervals,
the outputs, and the per-month drought weights byte-identical at default.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from types import SimpleNamespace

import pytest

duckdb = pytest.importorskip("duckdb")

import sibyl.config as sibyl_config
from pythia.tools import base_rate_spd as brs
from sibyl import reference_backtest as rb
from sibyl.reference import build_reference


def _db(tmp_path):
    con = duckdb.connect(str(tmp_path / "bt.duckdb"))
    con.execute("CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month DATE, "
                "fatalities BIGINT, source TEXT, updated_at TIMESTAMP)")
    con.execute("CREATE TABLE facts_resolved (iso3 TEXT, hazard_code TEXT, metric TEXT, "
                "ym TEXT, value DOUBLE, publisher TEXT)")
    return con


def _acled(con, iso, first, values):
    for i, v in enumerate(values):
        ym = brs._add_months(first, i)
        nxt = brs._add_months(ym, 1)
        con.execute("INSERT INTO acled_monthly_fatalities VALUES (?, ?, ?, 'ACLED', ?)",
                    [iso, f"{ym}-01", int(v), datetime(int(nxt[:4]), int(nxt[5:7]), 20)])


def _facts(con, iso, hazard, metric, first, values):
    for i, v in enumerate(values):
        if v is None:
            continue
        con.execute("INSERT INTO facts_resolved VALUES (?, ?, ?, ?, ?, 'X')",
                    [iso, hazard, metric, brs._add_months(first, i), float(v)])


# --- helpers ----------------------------------------------------------------------

def test_months_between_and_the_isotonic_fit():
    assert rb.months_between("2024-11", "2025-02") == ["2024-11", "2024-12", "2025-01", "2025-02"]
    assert rb._isotonic_non_increasing([0.9, 0.5, 0.75, 0.25, 0.5, 0.0]) == \
        [0.9, 0.625, 0.625, 0.375, 0.375, 0.0]
    out = rb._isotonic_non_increasing([0.25, 0.9, 0.5, 0.5, 0.75, 1.0])
    assert all(a >= b - 1e-12 for a, b in zip(out, out[1:]))


def test_scores_are_floored_so_the_log_score_is_finite():
    s = rb.score_vector([1.0, 0, 0, 0, 0, 0, 0], 6)
    assert s["log"] < 10 and 0 <= s["rps"] <= 1


# --- conflict ---------------------------------------------------------------------

def _conflict_db(tmp_path):
    con = _db(tmp_path)
    for iso, base in (("ETH", 120), ("SOM", 30), ("MLI", 400)):
        _acled(con, iso, "2018-01", [base + (i % 7) * 5 for i in range(72)])
    return con


def test_conflict_candidates_and_the_production_reference(tmp_path):
    con = _conflict_db(tmp_path)
    rows, man = rb.run_backtest(con, start="2021-01", end="2021-06", classes=["ACE"])
    cands = {r["candidate"] for r in rows if r["class"] == "ACE"}
    assert {"conflictology12", "level_transition", "level_volatility",
            "pool_0.5", "pool_0.75", "pool_0.9"} <= cands
    prod = [r for r in rows if r["candidate"] == "pool_0.75" and r["status"] == "ok"]
    assert prod and all("diff_vs_production" not in r for r in prod)
    other = next(r for r in rows if r["candidate"] == "conflictology12"
                 and r["horizon"] == "all" and r["score_type"] == "brier")
    assert other["status"] == "ok" and other["lo"] <= other["mean"] <= other["hi"]
    assert "diff_vs_production" in other
    assert man["sections"]["ACE"]["n_country_forecasts"] == 18  # 3 countries x 6 months
    assert set(man["reproduction"]) >= {"conflictology12", "pool_0.75", "level_volatility"}


def test_a_month_with_no_record_is_unresolved(tmp_path):
    con = _conflict_db(tmp_path)
    # ACLED dark from 2021-04 on: those target months have no record anywhere.
    con.execute("DELETE FROM acled_monthly_fatalities WHERE month >= DATE '2021-04-01'")
    scores = rb.Scores()
    rb.backtest_conflict(con, ["2021-01"], scores)
    hs = {r["h"] for r in scores.rows[("ACE", None, "conflictology12")]}
    assert hs == {1, 2}  # Feb and Mar resolve; Apr..Jul do not


def test_the_backtest_is_deterministic(tmp_path):
    con = _conflict_db(tmp_path)
    a, _ = rb.run_backtest(con, start="2021-01", end="2021-03", classes=["ACE"])
    b, _ = rb.run_backtest(con, start="2021-01", end="2021-03", classes=["ACE"])
    assert a == b


def test_the_cache_is_removed_after_the_run(tmp_path):
    original = brs._acled_series_all
    rb.run_backtest(_conflict_db(tmp_path), start="2021-01", end="2021-01", classes=["ACE"])
    assert brs._acled_series_all is original


# --- drought ----------------------------------------------------------------------

def _drought_db(tmp_path):
    con = _db(tmp_path)
    for iso, base in (("SOM", 3e6), ("ETH", 2e7), ("KEN", 5e5)):
        _facts(con, iso, "DR", "phase3plus_in_need", "2019-01",
               [base * (1 + 0.3 * ((i // 4) % 3)) for i in range(84)])
    return con


def test_drought_reports_every_lag_and_every_weight(tmp_path):
    con = _drought_db(tmp_path)
    rows, man = rb.run_backtest(con, start="2022-06", end="2024-06", classes=["DR"],
                                fit_end="2023-06")
    lags = {r["lag"] for r in rows if r["class"] == "DR"}
    assert lags == {1, 2, 3}
    cands = {r["candidate"] for r in rows if r["class"] == "DR"}
    assert {f"persistence_w{w:g}" for w in rb.DR_WEIGHTS} <= cands
    assert "fitted_schedule" in cands
    for info in man["sections"]["DR"]["lags"].values():
        s = info["schedule"]
        assert all(a >= b - 1e-9 for a, b in zip(s, s[1:]))
        assert info["n_fit"] > 0 and info["n_test"] > 0


def test_a_longer_lag_sees_less(tmp_path, monkeypatch):
    con = _drought_db(tmp_path)
    seen = []
    real = brs.last_observed_value

    def spy(con_, iso, hz, metric, before):
        seen.append(before)
        return real(con_, iso, hz, metric, before)

    monkeypatch.setattr(brs, "last_observed_value", spy)
    rb.backtest_drought(con, ["2023-03"], rb.Scores(), lags=[1, 3], countries=["SOM"])
    # Forecast on 13 Mar 2023: lag 1 reads rows up to Feb, lag 3 up to Dec.
    assert seen == ["2023-03", "2023-01"]


# --- flood and cyclone ------------------------------------------------------------

def test_flood_below_a_hundred_forecasts_is_too_few(tmp_path):
    con = _db(tmp_path)
    _facts(con, "BGD", "FL", "affected", "2018-01",
           [50000 if i % 12 in (6, 7) else None for i in range(60)])
    rows, _ = rb.run_backtest(con, start="2021-01", end="2021-06", classes=["FL"])
    fl = [r for r in rows if r["class"] == "FL"]
    assert fl and all(r["status"] == "too_few" for r in fl)
    assert {r["candidate"] for r in fl} == {"per_month", "pooled", "uniform"}


def test_flood_and_cyclone_sections_carry_the_selection_caveat(tmp_path):
    fl_dir, ace_dir = tmp_path / "fl", tmp_path / "ace"
    fl_dir.mkdir()
    ace_dir.mkdir()
    con = _db(fl_dir)
    _facts(con, "BGD", "FL", "affected", "2018-01",
           [50000 if i % 12 in (6, 7) else None for i in range(60)])
    rows, man = rb.run_backtest(con, start="2021-01", end="2021-06", classes=["FL"])
    assert rb.PA_SELECTION_NOTE in rb.render_markdown(rows, man).split("## FL", 1)[1]
    rows, man = rb.run_backtest(_conflict_db(ace_dir), start="2021-01", end="2021-01",
                                classes=["ACE"])
    assert rb.PA_SELECTION_NOTE not in rb.render_markdown(rows, man)


# --- outputs ----------------------------------------------------------------------

def test_the_cli_writes_three_files(tmp_path):
    con = _conflict_db(tmp_path)
    con.close()
    out = tmp_path / "out"
    assert rb.main(["--db", str(tmp_path / "bt.duckdb"), "--out-dir", str(out),
                    "--start", "2021-01", "--end", "2021-02", "--classes", "ACE"]) == 0
    assert {p.name for p in out.iterdir()} == {"reference_backtest.md", "reference_backtest.csv",
                                               "manifest.json"}
    md = (out / "reference_backtest.md").read_text()
    assert "Reproducing the quoted conflict figures" in md and "8,371" in md
    assert json.loads((out / "manifest.json").read_text())["bootstrap"]["draws"] == 2000


def test_the_backtest_writes_nothing_to_the_db(tmp_path):
    import inspect

    src = inspect.getsource(rb)
    for verb in ("INSERT", "UPDATE ", "DELETE", "CREATE TABLE", "DROP"):
        assert verb not in src


# --- the per-month drought weights ------------------------------------------------

def _dr_ref(tmp_path, name="a"):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    con = _db(d)
    _facts(con, "SOM", "DR", "phase3plus_in_need", "2026-05", [2e6, 2.2e6, 2.5e6, 3e6])
    keys = [brs._add_months("2026-11", i) for i in range(6)]
    q = SimpleNamespace(question_id="Q", iso3="SOM", hazard_code="DR", metric="PHASE3PLUS_IN_NEED")
    return build_reference(con, q, keys, date(2026, 11, 1))


def _reload_config(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("SIBYL_DR_PERSISTENCE_WEIGHTS", raising=False)
    else:
        monkeypatch.setenv("SIBYL_DR_PERSISTENCE_WEIGHTS", value)
    monkeypatch.setattr(sibyl_config, "DR_PERSISTENCE_WEIGHTS", sibyl_config._dr_persistence_weights())


def test_default_weights_leave_the_reference_byte_identical(tmp_path, monkeypatch):
    _reload_config(monkeypatch, None)
    base = _dr_ref(tmp_path)
    assert sibyl_config.DR_PERSISTENCE_WEIGHTS == (0.5,) * 6
    _reload_config(monkeypatch, "0.5,0.5,0.5,0.5,0.5,0.5")
    same = _dr_ref(tmp_path, "b")
    assert json.dumps(same.to_dict()) == json.dumps(base.to_dict())
    assert same.prompt_text == base.prompt_text
    assert base.source.startswith("pool:persistence_0.5+")
    assert "persistence_weights" not in base.detail


@pytest.mark.parametrize("bad", ["0.5,0.5", "a,b,c,d,e,f", "0.9,0.8,0.7,0.6,0.5,1.4"])
def test_a_malformed_schedule_falls_back_to_the_single_weight(monkeypatch, bad):
    _reload_config(monkeypatch, bad)
    assert sibyl_config.DR_PERSISTENCE_WEIGHTS == (float(sibyl_config.DR_PERSISTENCE_WEIGHT),) * 6


def test_a_schedule_gives_each_month_its_own_weight(tmp_path, monkeypatch):
    _reload_config(monkeypatch, "0.9,0.8,0.7,0.6,0.5,0.4")
    ref = _dr_ref(tmp_path)
    assert ref.source.startswith("pool:persistence_schedule+")
    assert ref.detail["persistence_weights"] == [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    # More persistence, more mass on the last observed value's bucket (1M-<5M).
    assert ref.by_month[1][3] > ref.by_month[6][3]
