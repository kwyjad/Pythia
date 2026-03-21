# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for binary Brier scoring (EVENT_OCCURRENCE questions)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import duckdb
import pytest

from forecaster.scoring import binary_brier


# ---- binary_brier unit tests ----

def test_binary_brier_perfect_hit():
    """P=1.0, outcome=1 → Brier = 0.0."""
    assert abs(binary_brier(1.0, 1.0)) < 1e-9


def test_binary_brier_perfect_miss():
    """P=0.0, outcome=1 → Brier = 1.0."""
    assert abs(binary_brier(0.0, 1.0) - 1.0) < 1e-9


def test_binary_brier_p03_outcome1():
    """P=0.3, outcome=1 → Brier = 0.49."""
    assert abs(binary_brier(0.3, 1.0) - 0.49) < 1e-9


def test_binary_brier_p08_outcome1():
    """P=0.8, outcome=1 → Brier = 0.04."""
    assert abs(binary_brier(0.8, 1.0) - 0.04) < 1e-9


def test_binary_brier_p01_outcome0():
    """P=0.1, outcome=0 → Brier = 0.01."""
    assert abs(binary_brier(0.1, 0.0) - 0.01) < 1e-9


def test_binary_brier_p05_outcome0():
    """P=0.5, outcome=0 → Brier = 0.25."""
    assert abs(binary_brier(0.5, 0.0) - 0.25) < 1e-9


# ---- Integration: compute_scores with binary questions ----

def _setup_scoring_db(con):
    """Create minimal tables for binary scoring test."""
    from pythia.db.schema import ensure_schema
    ensure_schema(con)
    # hs_runs
    con.execute("""
        CREATE TABLE IF NOT EXISTS hs_runs (
            hs_run_id VARCHAR PRIMARY KEY,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            git_sha VARCHAR DEFAULT '',
            config_profile VARCHAR DEFAULT '',
            countries_json VARCHAR DEFAULT '[]'
        )
    """)
    try:
        con.execute("INSERT INTO hs_runs (hs_run_id) VALUES ('test-run')")
    except Exception:
        pass
    # resolutions
    con.execute("""
        CREATE TABLE IF NOT EXISTS resolutions (
            question_id TEXT,
            horizon_m INTEGER,
            observed_month TEXT,
            value DOUBLE,
            source_snapshot_ym TEXT,
            source_ts TIMESTAMP,
            run_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)


@pytest.mark.db
def test_binary_brier_scoring_integration(tmp_path: Path):
    """Binary questions should get Brier scores, not SPD scores."""
    db_path = tmp_path / "scoring_test.duckdb"
    db_url = f"duckdb:///{db_path}"
    con = duckdb.connect(str(db_path))

    try:
        _setup_scoring_db(con)

        # Create a binary question
        con.execute("""
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status, wording)
            VALUES ('BGD_FL_EVENT_OCCURRENCE_2026-04', 'test-run', 'BGD', 'FL', 'EVENT_OCCURRENCE',
                    '2026-09', '2026-04-01', 'active', 'test binary')
        """)

        # Create resolution: event occurred in h1 (2026-04)
        con.execute("""
            INSERT INTO resolutions (question_id, horizon_m, value, source_ts, run_id)
            VALUES ('BGD_FL_EVENT_OCCURRENCE_2026-04', 1, 1.0, CURRENT_TIMESTAMP, NULL)
        """)

        # Create forecast ensemble: P(yes) = 0.3 for h1
        # bucket_1 = P(yes) = 0.3, bucket_2 = P(no) = 0.7
        for bucket_idx, prob in [(1, 0.3), (2, 0.7), (3, 0.0), (4, 0.0), (5, 0.0)]:
            con.execute("""
                INSERT INTO forecasts_ensemble
                    (question_id, iso3, hazard_code, metric, model_name,
                     month_index, bucket_index, probability, status, created_at)
                VALUES ('BGD_FL_EVENT_OCCURRENCE_2026-04', 'BGD', 'FL', 'EVENT_OCCURRENCE',
                        'ensemble_mean_v2', 1, ?, ?, 'ok', CURRENT_TIMESTAMP)
            """, [bucket_idx, prob])
    finally:
        con.close()

    from pythia.tools.compute_scores import compute_scores
    compute_scores(db_url=db_url)

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute("""
            SELECT score_type, value FROM scores
            WHERE question_id = 'BGD_FL_EVENT_OCCURRENCE_2026-04'
              AND horizon_m = 1
              AND model_name IS NULL
        """).fetchall()
        assert len(rows) > 0, f"Expected score rows, got none"
        # Should have brier score only (not log or crps)
        score_types = {r[0] for r in rows}
        assert "brier" in score_types
        # Brier = (0.3 - 1.0)^2 = 0.49
        brier_row = [r for r in rows if r[0] == "brier"]
        assert len(brier_row) == 1
        assert abs(brier_row[0][1] - 0.49) < 1e-6
    finally:
        con.close()


@pytest.mark.db
def test_spd_questions_unaffected_by_binary_scoring(tmp_path: Path):
    """PA questions should still get brier/log/crps scores."""
    db_path = tmp_path / "scoring_spd_test.duckdb"
    db_url = f"duckdb:///{db_path}"
    con = duckdb.connect(str(db_path))

    class_bins_pa = ["<10k", "10k-<50k", "50k-<250k", "250k-<500k", ">=500k"]

    try:
        _setup_scoring_db(con)

        # Create a PA question
        con.execute("""
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status, wording)
            VALUES ('ETH_FL_PA_2026-04', 'test-run', 'ETH', 'FL', 'PA',
                    '2026-09', '2026-04-01', 'active', 'test PA')
        """)

        # Create resolution: 5000 people affected in h1 (bucket 1: <10k)
        con.execute("""
            INSERT INTO resolutions (question_id, horizon_m, value, source_ts, run_id)
            VALUES ('ETH_FL_PA_2026-04', 1, 5000.0, CURRENT_TIMESTAMP, NULL)
        """)

        # Create forecast ensemble using class_bin/p/horizon_m columns
        # (the scoring code reads class_bin + p for ensemble table)
        bins_probs = [0.6, 0.2, 0.1, 0.05, 0.05]
        for i, (cb, prob) in enumerate(zip(class_bins_pa, bins_probs)):
            con.execute("""
                INSERT INTO forecasts_ensemble
                    (question_id, iso3, hazard_code, metric, model_name,
                     month_index, bucket_index, probability, status, created_at,
                     class_bin, p, horizon_m)
                VALUES ('ETH_FL_PA_2026-04', 'ETH', 'FL', 'PA',
                        'ensemble_mean_v2', 1, ?, ?, 'ok', CURRENT_TIMESTAMP,
                        ?, ?, 1)
            """, [i + 1, prob, cb, prob])
    finally:
        con.close()

    from pythia.tools.compute_scores import compute_scores
    compute_scores(db_url=db_url)

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute("""
            SELECT score_type FROM scores
            WHERE question_id = 'ETH_FL_PA_2026-04'
              AND horizon_m = 1
              AND model_name IS NULL
        """).fetchall()
        score_types = {r[0] for r in rows}
        assert "brier" in score_types
        assert "log" in score_types
        assert "crps" in score_types
    finally:
        con.close()
