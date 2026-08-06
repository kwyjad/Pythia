# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for pythia.tools.compute_deviation — deviation-from-base-rate rows."""

from __future__ import annotations

import duckdb
import pytest

from pythia.buckets import n_buckets_for
from pythia.tools.compute_deviation import (
    DEVIATION_MODEL_NAMES,
    binary_deviation_metrics,
    compute_deviation,
    spd_deviation_metrics,
)


class TestJsProperties:
    """Phase 1 acceptance: JS is 0 at equality and rises monotonically as the
    forecast is shifted away from the base rate."""

    BASE = [0.4, 0.3, 0.15, 0.1, 0.03, 0.02]

    def test_zero_when_forecast_equals_base_rate(self):
        out = spd_deviation_metrics([self.BASE] * 6, self.BASE, None, "PA")
        assert out["js_vs_baserate"] == pytest.approx(0.0, abs=1e-12)
        assert out["log_ev_ratio"] == pytest.approx(0.0, abs=1e-12)

    def test_monotone_rise_as_forecast_shifts_away(self):
        k = len(self.BASE)
        target = [0.0] * k
        target[-1] = 1.0  # all mass on the top bucket
        prev = -1.0
        for t in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            forecast = [
                (1.0 - t) * b + t * e for b, e in zip(self.BASE, target)
            ]
            js = spd_deviation_metrics([forecast] * 6, self.BASE, None, "PA")[
                "js_vs_baserate"
            ]
            assert js > prev or (t == 0.0 and js == pytest.approx(0.0, abs=1e-12))
            prev = js

    def test_shift_to_top_bucket_raises_ev_ratio(self):
        k = len(self.BASE)
        heavy = [0.0] * k
        heavy[-1] = 1.0
        out = spd_deviation_metrics([heavy] * 6, self.BASE, None, "PA")
        assert out["log_ev_ratio"] > 0
        assert out["eiv_nominal"] > out["eiv_baserate"]

    def test_binary_metrics(self):
        out = binary_deviation_metrics([0.1] * 6, 0.1)
        assert out["js_vs_baserate"] == pytest.approx(0.0, abs=1e-12)
        out2 = binary_deviation_metrics([0.6] * 6, 0.1)
        assert out2["js_vs_baserate"] > 0
        assert out2["log_ev_ratio"] > 0


def _seed_db(path: str) -> None:
    con = duckdb.connect(path)
    from pythia.db.schema import ensure_schema

    ensure_schema(con)
    con.execute(
        "CREATE TABLE facts_resolved (ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, value DOUBLE)"
    )
    con.execute(
        "CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month TEXT, fatalities DOUBLE)"
    )

    con.execute("INSERT INTO hs_runs (hs_run_id) VALUES ('hs_1')")
    con.execute(
        """
        INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric,
                               target_month, window_start_date, is_test)
        VALUES
          ('Q_ACE', 'hs_1', 'SOM', 'ACE', 'FATALITIES', '2027-01', DATE '2026-08-01', FALSE),
          ('Q_CU',  'hs_1', 'SOM', 'CU',  'PA',         '2027-01', DATE '2026-08-01', FALSE),
          ('Q_TC',  'hs_1', 'PHL', 'TC',  'EVENT_OCCURRENCE', '2027-01', DATE '2026-08-01', TRUE)
        """
    )

    # Base-rate history (strictly before the 2026-08 window).
    for ym, fat in [("2026-02", 0), ("2026-03", 5), ("2026-04", 30),
                    ("2026-05", 120), ("2026-06", 300), ("2026-07", 600)]:
        con.execute("INSERT INTO acled_monthly_fatalities VALUES ('SOM', ?, ?)", [ym, fat])
    for year in (2023, 2024, 2025):
        for month in range(1, 13):
            occurred = 1.0 if month in (9, 10) else 0.0
            con.execute(
                "INSERT INTO facts_resolved VALUES (?, 'PHL', 'TC', 'event_occurrence', ?)",
                [f"{year:04d}-{month:02d}", occurred],
            )

    # Aggregate forecasts, full bucket vectors for months 1..6.
    k_fat = n_buckets_for("FATALITIES")
    k_pa = n_buckets_for("PA")
    for month in range(1, 7):
        for bucket in range(1, k_fat + 1):
            p = 0.5 if bucket == k_fat else 0.5 / (k_fat - 1)
            con.execute(
                "INSERT INTO forecasts_raw (run_id, question_id, model_name, month_index, bucket_index, probability) "
                "VALUES ('fc_1', 'Q_ACE', 'ensemble_mean_v2', ?, ?, ?)",
                [month, bucket, p],
            )
        for bucket in range(1, k_pa + 1):
            con.execute(
                "INSERT INTO forecasts_raw (run_id, question_id, model_name, month_index, bucket_index, probability) "
                "VALUES ('fc_1', 'Q_CU', 'ensemble_mean_v2', ?, ?, ?)",
                [month, bucket, 1.0 / k_pa],
            )
        for bucket, p in ((1, 0.35), (2, 0.65)):
            con.execute(
                "INSERT INTO forecasts_raw (run_id, question_id, model_name, month_index, bucket_index, probability) "
                "VALUES ('fc_1', 'Q_TC', 'ensemble_mean_v2', ?, ?, ?)",
                [month, bucket, p],
            )
    con.close()


class TestComputeDeviationEndToEnd:
    def test_rows_written_only_for_anchored_questions(self, tmp_path):
        db = str(tmp_path / "dev.duckdb")
        _seed_db(db)
        n = compute_deviation(db)
        assert n == 2  # Q_ACE + Q_TC; Q_CU has no base rate and gets NO row

        con = duckdb.connect(db, read_only=True)
        try:
            rows = con.execute(
                """
                SELECT question_id, model_name, score_family, js_vs_baserate,
                       log_ev_ratio, eiv_nominal, eiv_per_100k, baserate_source,
                       is_test
                FROM forecast_deviation ORDER BY question_id
                """
            ).fetchall()
        finally:
            con.close()

        assert [r[0] for r in rows] == ["Q_ACE", "Q_TC"]
        by_qid = {r[0]: r for r in rows}

        ace = by_qid["Q_ACE"]
        assert ace[1] == "ensemble_mean_v2"
        assert ace[1] in DEVIATION_MODEL_NAMES
        assert ace[2] == "spd"
        assert ace[3] > 0  # far from base rate (heavy top bucket)
        assert ace[4] > 0  # EV above the base rate's
        assert ace[5] is not None and ace[5] > 0
        assert ace[6] is not None and ace[6] > 0  # SOM is in population.csv
        assert ace[7].startswith("acled_monthly_fatalities")
        assert ace[8] is False

        tc = by_qid["Q_TC"]
        assert tc[2] == "binary"
        assert tc[3] > 0
        assert tc[5] is None and tc[6] is None  # no impact centroid for binary
        assert tc[8] is True  # inherited from the question row

    def test_idempotent_rerun(self, tmp_path):
        db = str(tmp_path / "dev2.duckdb")
        _seed_db(db)
        assert compute_deviation(db) == 2
        assert compute_deviation(db) == 2
        con = duckdb.connect(db, read_only=True)
        try:
            count = con.execute("SELECT COUNT(*) FROM forecast_deviation").fetchone()[0]
        finally:
            con.close()
        assert count == 2

    def test_run_id_scoping(self, tmp_path):
        db = str(tmp_path / "dev3.duckdb")
        _seed_db(db)
        assert compute_deviation(db, run_id="fc_nonexistent") == 0
        assert compute_deviation(db, run_id="fc_1") == 2
