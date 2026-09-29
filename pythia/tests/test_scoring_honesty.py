# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What the scored run says about itself, honestly (Oct 2026).

The August 2026 scored bundle could not answer four questions a reader asks
first: would standing still have done as well (no persistence reference),
does a question forecast nine times count nine times (it did), how far off
was a wrong bucket (Brier cannot say; RPS can), and did the outcome move
after it was scored (every re-resolution overwrote the last). Each test
below pins one answer.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import duckdb
import pytest

from pythia.tools import compute_resolutions as cr
from pythia.tools.base_rate_spd import last_observed_value
from pythia.tools.score_baselines import (
    PERSISTENCE_MODEL_NAME,
    PERSISTENCE_SMOOTHING,
    persistence_spd,
    score_baselines,
)


# ---------------------------------------------------------------------------
# (a) the persistence reference
# ---------------------------------------------------------------------------


def _acled(con) -> None:
    con.execute(
        "CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month DATE, fatalities BIGINT, "
        "source TEXT, updated_at TIMESTAMP)"
    )


def test_last_observed_value_reads_the_month_before_the_window() -> None:
    con = duckdb.connect(":memory:")
    _acled(con)
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES "
        "('SOM', DATE '2026-06-01', 300, 'ACLED', now()), "
        "('SOM', DATE '2026-07-01', 412, 'ACLED', now()), "
        "('SOM', DATE '2026-08-01', 999, 'ACLED', now())"
    )
    assert last_observed_value(con, "SOM", "ACE", "FATALITIES", "2026-08") == (
        412.0, "2026-07", "acled_monthly_fatalities",
    )


def test_a_live_month_with_no_row_is_a_quiet_zero() -> None:
    con = duckdb.connect(":memory:")
    _acled(con)
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES "
        "('BEN', DATE '2026-05-01', 4, 'ACLED', now()), "
        "('NGA', DATE '2026-07-01', 800, 'ACLED', now())"
    )
    value, ym, source = last_observed_value(con, "BEN", "ACE", "FATALITIES", "2026-08")
    assert (value, ym) == (0.0, "2026-07") and source.endswith("quiet_month")


def test_phase3_persistence_takes_the_latest_stock() -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE facts_resolved (ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, value DOUBLE)"
    )
    con.execute(
        "INSERT INTO facts_resolved VALUES "
        "('2026-03', 'ETH', 'DR', 'phase3plus_in_need', 4000000), "
        "('2026-06', 'ETH', 'DR', 'phase3plus_in_need', 5200000), "
        "('2026-09', 'ETH', 'DR', 'phase3plus_in_need', 9000000)"
    )
    assert last_observed_value(con, "ETH", "DR", "PHASE3PLUS_IN_NEED", "2026-08")[:2] == (
        5200000.0, "2026-06",
    )
    assert last_observed_value(con, "ETH", "FL", "PA", "2026-08") is None


def test_persistence_vector_is_smoothed_so_log_loss_stays_finite() -> None:
    from pythia.buckets import n_buckets_for
    from pythia.tools.compute_scores import _log_score

    k = n_buckets_for("FATALITIES")
    vec = persistence_spd(412.0, "FATALITIES")
    assert len(vec) == k and sum(vec) == pytest.approx(1.0)
    assert max(vec) == pytest.approx(1 - PERSISTENCE_SMOOTHING + PERSISTENCE_SMOOTHING / k)
    assert min(vec) == pytest.approx(PERSISTENCE_SMOOTHING / k)
    for j in range(k):
        assert math.isfinite(_log_score(vec, j))


def test_score_baselines_writes_persistence_for_ace_fatalities(tmp_path: Path) -> None:
    db = str(tmp_path / "p.duckdb")
    con = duckdb.connect(db)
    _acled(con)
    for m, n in [("2026-05", 380), ("2026-06", 395), ("2026-07", 412)]:
        con.execute(
            "INSERT INTO acled_monthly_fatalities VALUES ('SOM', ?, ?, 'ACLED', now())",
            [f"{m}-01", n],
        )
    con.execute("CREATE TABLE hs_runs (hs_run_id TEXT)")
    con.execute("INSERT INTO hs_runs VALUES ('hs1')")
    con.execute(
        "CREATE TABLE questions (question_id TEXT, hs_run_id TEXT, iso3 TEXT, hazard_code TEXT, "
        "metric TEXT, window_start_date DATE, target_month TEXT, is_test BOOLEAN DEFAULT FALSE)"
    )
    con.execute(
        "INSERT INTO questions VALUES ('Q1','hs1','SOM','ACE','FATALITIES',DATE '2026-08-01','2027-01',FALSE)"
    )
    con.execute(
        "CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, value DOUBLE)"
    )
    con.execute("INSERT INTO resolutions VALUES ('Q1', 1, 450)")
    con.close()

    counters = score_baselines(db)
    assert counters["scored_persistence"] == 1

    con = duckdb.connect(db)
    rows = con.execute(
        "SELECT score_type, value FROM scores WHERE model_name = ? ORDER BY 1",
        [PERSISTENCE_MODEL_NAME],
    ).fetchall()
    audit = con.execute(
        "SELECT baserate_source FROM baseline_scored_forecasts WHERE model_name = ?",
        [PERSISTENCE_MODEL_NAME],
    ).fetchone()[0]
    con.close()
    assert [r[0] for r in rows] == ["brier", "crps", "log"]
    # 412 and 450 share the 100-<500 bucket: persistence is right and sharp.
    assert dict(rows)["brier"] < 0.05
    assert audit.startswith("persistence:acled_monthly_fatalities:2026-07=412")


# ---------------------------------------------------------------------------
# (b)-(d) the scored bundle: one run per question, RPS beside Brier, edges
# ---------------------------------------------------------------------------


def _bundle_db():
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE questions (question_id TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT)"
    )
    con.execute("INSERT INTO questions VALUES ('Q1','SOM','ACE','FATALITIES')")
    con.execute(
        "CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, metric TEXT, score_type TEXT, "
        "model_name TEXT, value DOUBLE, run_id TEXT)"
    )
    con.executemany(
        "INSERT INTO scores VALUES (?,?,?,?,?,?,?)",
        [
            ("Q1", 1, "FATALITIES", "brier", "ensemble_mean_v2", 1.50, "fc_100"),
            ("Q1", 1, "FATALITIES", "brier", "ensemble_mean_v2", 0.30, "fc_200"),
            ("Q1", 1, "FATALITIES", "crps", "ensemble_mean_v2", 0.10, "fc_200"),
            ("Q1", 1, "FATALITIES", "brier", "__ext_climatology", 0.60, None),
            ("Q1", 1, "FATALITIES", "crps", "__ext_climatology", 0.20, None),
        ],
    )
    con.execute(
        "CREATE TABLE forecasts_ensemble (question_id TEXT, run_id TEXT, model_name TEXT, "
        "month_index INTEGER, bucket_index INTEGER, probability DOUBLE, ev_value DOUBLE)"
    )
    for run, hot in (("fc_100", 2), ("fc_200", 5)):
        for b in range(1, 8):
            con.execute(
                "INSERT INTO forecasts_ensemble VALUES ('Q1', ?, 'ensemble_mean_v2', 1, ?, ?, NULL)",
                [run, b, 0.94 if b == hot else 0.01],
            )
    con.execute(
        "CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, value DOUBLE)"
    )
    # 98 is within 5% of the 100 boundary between the 25-<100 and 100-<500 buckets.
    con.execute("INSERT INTO resolutions VALUES ('Q1', 1, 98)")
    return con


def test_rollups_count_only_the_latest_run(tmp_path: Path) -> None:
    from scripts.ai_bundle.build_scored_forecast_bundle import _emit_rollups

    rows = _emit_rollups(_bundle_db(), tmp_path, ["Q1"])
    mean = {(r["model_name"], r["score_type"]): r for r in rows}
    assert mean[("ensemble_mean_v2", "brier")]["n_samples"] == 1
    assert mean[("ensemble_mean_v2", "brier")]["mean_value"] == pytest.approx(0.30)
    assert mean[("ensemble_mean_v2", "brier")]["skill_vs_climatology"] == pytest.approx(0.5)
    assert mean[("ensemble_mean_v2", "crps")]["skill_vs_climatology"] == pytest.approx(0.5)


def test_forecast_vs_outcome_uses_one_run_and_flags_a_bucket_edge(tmp_path: Path) -> None:
    import csv

    from scripts.ai_bundle.build_scored_forecast_bundle import _emit_forecast_vs_outcome

    _emit_forecast_vs_outcome(_bundle_db(), tmp_path, ["Q1"])
    rows = list(csv.DictReader((tmp_path / "forecast_vs_outcome.csv").open()))
    assert len(rows) == 1
    row = rows[0]
    # The latest run (fc_200) put its mass on bucket 5; the older run's
    # bucket 2 must not leak in.
    assert row["modal_bucket"] == "5"
    assert float(row["p_modal_bucket"]) == pytest.approx(0.94)
    assert row["bucket_edge"] == "True"
    assert float(row["nearest_boundary"]) == 100.0


def test_digest_reports_rps_beside_brier(tmp_path: Path) -> None:
    from scripts.ai_bundle.build_scored_forecast_bundle import _emit_rollups, _write_digest

    rollups = _emit_rollups(_bundle_db(), tmp_path, ["Q1"])
    _write_digest(tmp_path, [], rollups, [], {"worst": [], "best": []}, months_back=12)
    text = (tmp_path / "digest.md").read_text()
    assert "mean RPS" in text
    line = next(l for l in text.splitlines() if "| ensemble_mean_v2 |" in l)
    assert "0.3000" in line and "0.1000" in line and "+0.500" in line


# ---------------------------------------------------------------------------
# (e) resolution vintages
# ---------------------------------------------------------------------------


def test_vintages_accumulate_and_are_never_overwritten() -> None:
    con = duckdb.connect(":memory:")
    cr._ensure_vintage_table(con)
    kw = dict(question_id="Q1", horizon_m=1, observed_month="2026-08",
              source_desc=cr.ACE_FATALITIES_SERIES, is_test=False)
    assert cr.record_vintages(con, value=410, source_ts="2026-09-28 09:00:00",
                              today=date(2026, 9, 29), **kw) == ["first"]
    # 58 days after month end: within the tolerance of the 60-day milestone.
    assert cr.record_vintages(con, value=425, source_ts="2026-10-28 09:00:00",
                              today=date(2026, 10, 28), **kw) == ["d60"]
    assert cr.record_vintages(con, value=431, source_ts="2026-11-28 09:00:00",
                              today=date(2026, 11, 28), **kw) == ["d90"]
    assert cr.record_vintages(con, value=999, source_ts="2026-12-28 09:00:00",
                              today=date(2026, 12, 28), **kw) == []
    rows = con.execute(
        "SELECT vintage, value, days_after_month_end, acled_snapshot_date "
        "FROM resolution_vintages ORDER BY days_after_month_end"
    ).fetchall()
    assert [(r[0], r[1], r[2]) for r in rows] == [
        ("first", 410.0, 29), ("d60", 425.0, 58), ("d90", 431.0, 89),
    ]
    assert str(rows[0][3]) == "2026-09-28"


@pytest.mark.db
def test_compute_resolutions_stamps_the_acled_snapshot_and_a_vintage(
    tmp_path: Path, monkeypatch
) -> None:
    db = tmp_path / "v.duckdb"
    db_url = f"duckdb:///{db}"
    monkeypatch.setattr(cr, "load_cfg", lambda: {"app": {"db_url": db_url}})
    con = duckdb.connect(str(db))
    _acled(con)
    con.execute(
        "INSERT INTO acled_monthly_fatalities VALUES "
        "('SOM', DATE '2026-08-01', 412, 'ACLED', TIMESTAMP '2026-09-28 09:00:00')"
    )
    con.execute("CREATE TABLE hs_runs (hs_run_id TEXT PRIMARY KEY)")
    con.execute("INSERT INTO hs_runs VALUES ('run1')")
    con.execute(
        "CREATE TABLE questions (question_id TEXT, hs_run_id TEXT, iso3 TEXT, hazard_code TEXT, "
        "metric TEXT, target_month TEXT, window_start_date DATE, status TEXT, is_test BOOLEAN DEFAULT FALSE)"
    )
    con.execute(
        "INSERT INTO questions VALUES ('Q1','run1','SOM','ACE','FATALITIES','2027-01',DATE '2026-08-01','active',FALSE)"
    )
    con.close()
    cr.compute_resolutions(db_url=db_url, today=date(2026, 9, 29))
    con = duckdb.connect(str(db))
    res = con.execute("SELECT value, acled_snapshot_date FROM resolutions").fetchall()
    vint = con.execute("SELECT vintage, value FROM resolution_vintages").fetchall()
    con.close()
    assert [(r[0], str(r[1])) for r in res] == [(412.0, "2026-09-28")]
    assert vint == [("first", 412.0)]
