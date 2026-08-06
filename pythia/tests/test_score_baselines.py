# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for pythia.tools.score_baselines — climatology/uniform reference scores."""

from __future__ import annotations

import duckdb
import pytest

from pythia.buckets import n_buckets_for
from pythia.tools.score_baselines import (
    CLIMATOLOGY_MODEL_NAME,
    UNIFORM_MODEL_NAME,
    score_baselines,
)


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
    con.execute(
        """
        CREATE TABLE resolutions (
            question_id TEXT, horizon_m INTEGER, observed_month TEXT,
            value DOUBLE, is_test BOOLEAN DEFAULT FALSE
        )
        """
    )

    con.execute("INSERT INTO hs_runs (hs_run_id) VALUES ('hs_1')")
    con.execute(
        """
        INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric,
                               target_month, window_start_date, is_test)
        VALUES
          ('Q_ACE', 'hs_1', 'SOM', 'ACE', 'FATALITIES', '2027-01', DATE '2026-08-01', TRUE),
          ('Q_CU',  'hs_1', 'SOM', 'CU',  'PA',         '2027-01', DATE '2026-08-01', FALSE),
          ('Q_TC',  'hs_1', 'PHL', 'TC',  'EVENT_OCCURRENCE', '2027-01', DATE '2026-08-01', FALSE)
        """
    )
    con.execute(
        """
        INSERT INTO resolutions (question_id, horizon_m, observed_month, value) VALUES
          ('Q_ACE', 1, '2026-08', 40.0),
          ('Q_ACE', 2, '2026-09', 700.0),
          ('Q_CU',  1, '2026-08', 12000.0),
          ('Q_TC',  1, '2026-08', 0.0)
        """
    )

    # Climatology sources (strictly before the 2026-08 window).
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
    con.close()


class TestScoreBaselinesEndToEnd:
    def test_every_scored_pair_gets_a_companion_or_a_logged_reason(self, tmp_path, caplog):
        db = str(tmp_path / "base.duckdb")
        _seed_db(db)
        import logging

        with caplog.at_level(logging.INFO, logger="pythia.tools.score_baselines"):
            counters = score_baselines(db)

        # 4 resolved (question, horizon) pairs: uniform covers all of them.
        assert counters["scored_uniform"] == 4
        # Climatology covers Q_ACE h1+h2 and Q_TC h1; Q_CU has no base rate.
        assert counters["scored_climatology"] == 3
        assert counters["skipped_no_baserate"] == 1
        # The skip is logged with the question and the reason, per acceptance.
        assert any(
            "no climatology for Q_CU" in rec.getMessage() for rec in caplog.records
        )

        con = duckdb.connect(db, read_only=True)
        try:
            rows = con.execute(
                """
                SELECT question_id, horizon_m, model_name, score_type, value, run_id, is_test
                FROM scores ORDER BY question_id, horizon_m, model_name, score_type
                """
            ).fetchall()
        finally:
            con.close()

        by_model: dict = {}
        for qid, hm, model, stype, value, run_id, is_test in rows:
            assert run_id is None  # external-benchmark convention
            by_model.setdefault((qid, hm, model), {})[stype] = (value, is_test)

        # SPD questions get all three score types from both references.
        for hm in (1, 2):
            assert set(by_model[("Q_ACE", hm, UNIFORM_MODEL_NAME)]) == {"brier", "log", "crps"}
            assert set(by_model[("Q_ACE", hm, CLIMATOLOGY_MODEL_NAME)]) == {"brier", "log", "crps"}
        # Binary question: brier only, no log/crps.
        assert set(by_model[("Q_TC", 1, UNIFORM_MODEL_NAME)]) == {"brier"}
        assert set(by_model[("Q_TC", 1, CLIMATOLOGY_MODEL_NAME)]) == {"brier"}
        # Q_CU: uniform only — no climatology row was invented.
        assert ("Q_CU", 1, UNIFORM_MODEL_NAME) in by_model
        assert ("Q_CU", 1, CLIMATOLOGY_MODEL_NAME) not in by_model

        # Uniform SPD Brier is exactly (1-1/K)^2 + (K-1)/K^2.
        k = n_buckets_for("FATALITIES")
        expected = (1.0 - 1.0 / k) ** 2 + (k - 1) * (1.0 / k) ** 2
        assert by_model[("Q_ACE", 1, UNIFORM_MODEL_NAME)]["brier"][0] == pytest.approx(expected)

        # Binary uniform Brier: outcome 0 vs p=0.5.
        assert by_model[("Q_TC", 1, UNIFORM_MODEL_NAME)]["brier"][0] == pytest.approx(0.25)
        # Climatology binary: the seasonal rate for 2026-08 is low, so its
        # Brier against outcome 0 must beat the 50/50 guess.
        assert by_model[("Q_TC", 1, CLIMATOLOGY_MODEL_NAME)]["brier"][0] < 0.25

        # is_test inherits from the question row.
        assert by_model[("Q_ACE", 1, UNIFORM_MODEL_NAME)]["brier"][1] is True
        assert by_model[("Q_TC", 1, UNIFORM_MODEL_NAME)]["brier"][1] is False

    def test_idempotent_rerun(self, tmp_path):
        db = str(tmp_path / "base2.duckdb")
        _seed_db(db)
        score_baselines(db)
        score_baselines(db)
        con = duckdb.connect(db, read_only=True)
        try:
            n = con.execute(
                "SELECT COUNT(*) FROM scores WHERE model_name LIKE '\\_\\_ext\\_%' ESCAPE '\\'"
            ).fetchone()[0]
            audit = con.execute(
                "SELECT COUNT(*) FROM baseline_scored_forecasts"
            ).fetchone()[0]
        finally:
            con.close()
        # 2 SPD horizons x 2 models x 3 types + 1 CU horizon x 1 model x 3 types
        # + 1 binary horizon x 2 models x 1 type = 12 + 3 + 2 = 17
        assert n == 17
        # Audit: one row per (question, horizon, model) actually scored = 7.
        assert audit == 7

    def test_excluded_from_calibration_softmax_filter(self):
        # The calibration query filters model_name NOT LIKE '__ext_%' — pin
        # that both new sentinels match the exclusion pattern.
        import fnmatch

        for name in (CLIMATOLOGY_MODEL_NAME, UNIFORM_MODEL_NAME):
            assert name.startswith("__ext_")
            assert fnmatch.fnmatch(name, "__ext_*")

    def test_pre_baseline_db_is_a_noop(self, tmp_path):
        db = str(tmp_path / "empty.duckdb")
        con = duckdb.connect(db)
        con.close()
        counters = score_baselines(db)
        assert counters["scored_uniform"] == 0
        assert counters["scored_climatology"] == 0
