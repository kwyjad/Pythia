# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Calibration run twice in one month gives what running it once gives.

The EMA centroid update blended from the newest stored centroid and deleted
the row it replaced, so a second calibration run in November 2026 would
have applied a second EMA step and lost the October value. It now blends
from the value the month started from. The other calibration outputs
(weights, shared advice, family recalibration) delete and rewrite their
own month, and these tests hold them to giving the same answer twice.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

from pythia.tools import compute_bucket_centroids as cbc


def _centroid_db(path: Path, *, value: float = 3000.0, n: int = 12) -> str:
    con = duckdb.connect(str(path))
    try:
        con.execute(
            "CREATE TABLE questions (question_id TEXT, hs_run_id TEXT, iso3 TEXT, "
            "hazard_code TEXT, metric TEXT, is_test BOOLEAN DEFAULT FALSE)"
        )
        con.execute(
            "CREATE TABLE resolutions (question_id TEXT, horizon_m INTEGER, value DOUBLE, "
            "is_test BOOLEAN DEFAULT FALSE)"
        )
        con.execute(
            "CREATE TABLE bucket_centroids (hazard_code TEXT, metric TEXT, bucket_index INTEGER, "
            "centroid DOUBLE, as_of_month TEXT)"
        )
        con.execute(
            "CREATE UNIQUE INDEX ux_bucket_centroids ON bucket_centroids "
            "(hazard_code, metric, bucket_index)"
        )
        for i in range(n):
            con.execute(
                "INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric) "
                "VALUES (?, 'r', 'PHL', 'TC', 'PA')", [f"q{i}"],
            )
            con.execute("INSERT INTO resolutions (question_id, horizon_m, value) VALUES (?, 1, ?)",
                        [f"q{i}", value])
    finally:
        con.close()
    return f"duckdb:///{path}"


def _rows(path: Path):
    con = duckdb.connect(str(path))
    try:
        return con.execute(
            "SELECT hazard_code, metric, bucket_index, centroid, as_of_month, prior_centroid, "
            "prior_as_of_month FROM bucket_centroids WHERE hazard_code <> '*' ORDER BY 1, 2, 3"
        ).fetchall()
    finally:
        con.close()


def test_two_ema_runs_in_one_month_equal_one(tmp_path):
    once = _centroid_db(tmp_path / "once.duckdb")
    twice = _centroid_db(tmp_path / "twice.duckdb")
    cbc.update_bucket_centroids_ema(db_url=once, metric="PA", as_of=date(2026, 11, 11))
    cbc.update_bucket_centroids_ema(db_url=twice, metric="PA", as_of=date(2026, 11, 11))
    cbc.update_bucket_centroids_ema(db_url=twice, metric="PA", as_of=date(2026, 11, 28))
    assert _rows(tmp_path / "once.duckdb") == _rows(tmp_path / "twice.duckdb")


def test_a_second_run_with_new_resolutions_is_one_step_from_the_months_start(tmp_path):
    db_path = tmp_path / "c.duckdb"
    url = _centroid_db(db_path, value=3000.0)
    cbc.update_bucket_centroids_ema(db_url=url, metric="PA", as_of=date(2026, 11, 11))
    ((_, _, _, first, _, prior, _),) = _rows(db_path)
    con = duckdb.connect(str(db_path))
    con.execute("UPDATE resolutions SET value = 5000.0")
    con.close()
    cbc.update_bucket_centroids_ema(db_url=url, metric="PA", as_of=date(2026, 11, 28))
    ((_, _, _, second, month, prior2, _),) = _rows(db_path)
    rate = cbc.EMA_LEARNING_RATE
    assert prior2 == prior
    assert second == pytest.approx((1 - rate) * prior + rate * 5000.0)
    assert month == "2026-11"


def test_a_new_month_stores_the_last_months_value_as_the_prior(tmp_path):
    db_path = tmp_path / "m.duckdb"
    url = _centroid_db(db_path)
    cbc.update_bucket_centroids_ema(db_url=url, metric="PA", as_of=date(2026, 10, 11))
    ((_, _, _, october, _, _, _),) = _rows(db_path)
    cbc.update_bucket_centroids_ema(db_url=url, metric="PA", as_of=date(2026, 11, 11))
    ((_, _, _, november, month, prior, prior_month),) = _rows(db_path)
    assert (month, prior, prior_month) == ("2026-11", october, "2026-10")
    rate = cbc.EMA_LEARNING_RATE
    assert november == pytest.approx((1 - rate) * october + rate * 3000.0)


def test_the_november_backfill_fills_only_empty_priors(tmp_path):
    db_path = tmp_path / "b.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE bucket_centroids (hazard_code TEXT, metric TEXT, bucket_index INTEGER, "
        "centroid DOUBLE, as_of_month TEXT, prior_centroid DOUBLE, prior_as_of_month TEXT)"
    )
    con.executemany(
        "INSERT INTO bucket_centroids VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("FL", "PA", 6, 1_238_385.24, "2026-11", None, None),   # filled
            ("TC", "PA", 2, 5_500.0, "2026-11", 4_999.0, "2026-10"),  # already set: kept
            ("FL", "PA", 5, 300_000.0, "2026-11", None, None),       # not in the file
            ("FL", "PA", 2, 4_900.0, "2026-10", None, None),         # another month
        ],
    )
    assert cbc.backfill_november_2026_priors(con) == 1
    rows = {
        (h, b): (p, pm) for h, b, p, pm in con.execute(
            "SELECT hazard_code, bucket_index, prior_centroid, prior_as_of_month FROM bucket_centroids"
        ).fetchall()
    }
    assert rows[("FL", 6)] == (837_934.24, cbc.NOVEMBER_2026_BACKFILL_MARK)
    assert rows[("TC", 2)] == (4_999.0, "2026-10")
    assert rows[("FL", 5)] == (None, None)
    assert rows[("FL", 2)] == (None, None)
    assert cbc.backfill_november_2026_priors(con) == 0  # idempotent
    con.close()


def test_the_committed_backfill_file_is_the_november_chain(tmp_path):
    text = cbc.NOVEMBER_2026_PRIORS.read_text()
    assert "FL,PA,6,837934.24" in text
    assert sum(1 for line in text.splitlines() if line and not line.startswith(("#", "hazard"))) == 17


def test_calibration_weights_and_advice_twice_in_a_month(tmp_path, monkeypatch):
    """Weights and the shared advice rewrite their own month: run twice, same rows."""

    from pythia.tests.test_calibration_loop_integration import (
        test_calibration_loop_end_to_end as _build_and_run_once,
    )
    from pythia.tools.compute_calibration_pythia import compute_calibration_pythia

    _build_and_run_once(tmp_path, monkeypatch)
    db_path = tmp_path / "calibration_test.duckdb"

    def _snapshot():
        con = duckdb.connect(str(db_path))
        try:
            weights = con.execute(
                "SELECT as_of_month, hazard_code, metric, model_name, weight "
                "FROM calibration_weights ORDER BY 1, 2, 3, 4"
            ).fetchall()
            advice = con.execute(
                "SELECT as_of_month, hazard_code, metric, model_name, advice "
                "FROM calibration_advice ORDER BY 1, 2, 3, 4"
            ).fetchall()
        finally:
            con.close()
        return weights, advice

    first = _snapshot()
    assert first[0], "the fixture must write weights"
    compute_calibration_pythia(db_url=f"duckdb:///{db_path}", as_of=date(2025, 3, 1))
    assert _snapshot() == first


def test_family_recalibration_twice_in_a_month(tmp_path):
    from pythia.tests.test_family_recalibration import _db
    from pythia.tools import family_recalibration as fr

    con = _db(tmp_path)
    fr.fit_family_recalibration(con, "2026-11")
    query = (
        "SELECT family, hazard_code, metric, score_family, bucket_index, factor, n_questions "
        "FROM family_recalibration ORDER BY 1, 2, 3, 4, 5"
    )
    first = con.execute(query).fetchall()
    fr.fit_family_recalibration(con, "2026-11")
    assert con.execute(query).fetchall() == first
