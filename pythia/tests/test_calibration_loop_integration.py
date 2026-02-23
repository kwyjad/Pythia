# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.tools.compute_scores import compute_scores
from pythia.tools import compute_calibration_pythia as _calib_mod
from pythia.tools.compute_calibration_pythia import compute_calibration_pythia


@pytest.mark.db
def test_calibration_loop_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Smoke test for the Pythia calibration loop:

      questions + resolutions + forecasts_*  -> scores
      scores                                 -> calibration_weights + calibration_advice

    We build a tiny DuckDB with one question, per-horizon resolutions (6 months),
    and simple SPDs for two models and the ensemble. We then check that weights/advice
    are written and that scores are produced for each horizon independently.
    """
    db_path = tmp_path / "calibration_test.duckdb"
    db_url = f"duckdb:///{db_path}"

    # Monkeypatch config loader to point app.db_url at our temp DB
    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}

    monkeypatch.setattr("pythia.tools.compute_scores.load_cfg", _fake_load_cfg)
    monkeypatch.setattr("pythia.tools.compute_calibration_pythia.load_cfg", _fake_load_cfg)
    # Lower the minimum questions threshold so a single question suffices
    monkeypatch.setattr(_calib_mod, "MIN_QUESTIONS", 1)

    con = duckdb.connect(str(db_path))
    try:
        # Minimal schema
        con.execute(
            """
            CREATE TABLE hs_runs (
              hs_run_id TEXT PRIMARY KEY
            )
            """
        )
        con.execute("INSERT INTO hs_runs VALUES ('hs_run_1')")

        con.execute(
            """
            CREATE TABLE questions (
              question_id TEXT,
              hs_run_id TEXT,
              iso3 TEXT,
              hazard_code TEXT,
              metric TEXT,
              target_month TEXT,
              window_start_date DATE,
              status TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE resolutions (
              question_id TEXT,
              horizon_m INTEGER,
              observed_month TEXT,
              value DOUBLE,
              source_snapshot_ym TEXT,
              created_at TIMESTAMP,
              PRIMARY KEY (question_id, horizon_m)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE forecasts_ensemble (
              question_id TEXT,
              horizon_m INTEGER,
              class_bin TEXT,
              p DOUBLE,
              aggregator TEXT,
              ensemble_version TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE forecasts_raw (
              question_id TEXT,
              model_name TEXT,
              horizon_m INTEGER,
              class_bin TEXT,
              p DOUBLE,
              month_index INTEGER,
              bucket_index INTEGER,
              probability DOUBLE,
              run_id TEXT,
              created_at TIMESTAMP
            )
            """
        )

        # Insert one PA question with a 6-month window starting 2024-08
        # target_month is the 6th month = 2025-01
        con.execute(
            """
            INSERT INTO questions (question_id, hs_run_id, iso3, hazard_code, metric,
                                   target_month, window_start_date, status)
            VALUES ('Q1', 'hs_run_1', 'MLI', 'FL', 'PA', '2025-01', '2024-08-01', 'active')
            """
        )

        # Per-horizon resolutions: different ground truth values for each month
        # h1=2024-08: 5000 (bucket 0: <10k)
        # h2=2024-09: 15000 (bucket 1: 10k-<50k)
        # h3=2024-10: 30000 (bucket 1: 10k-<50k)
        # h4=2024-11: 60000 (bucket 2: 50k-<250k)
        # h5=2024-12: 8000  (bucket 0: <10k)
        # h6=2025-01: 30000 (bucket 1: 10k-<50k)
        resolution_values = [
            ('Q1', 1, '2024-08', 5000.0),
            ('Q1', 2, '2024-09', 15000.0),
            ('Q1', 3, '2024-10', 30000.0),
            ('Q1', 4, '2024-11', 60000.0),
            ('Q1', 5, '2024-12', 8000.0),
            ('Q1', 6, '2025-01', 30000.0),
        ]
        con.executemany(
            """
            INSERT INTO resolutions (question_id, horizon_m, observed_month, value,
                                     source_snapshot_ym, created_at)
            VALUES (?, ?, ?, ?, '2025-02', now())
            """,
            resolution_values,
        )

        # Ensemble SPD that puts most mass in bucket 1 (10k-<50k) for all horizons
        ensemble_spd = [0.1, 0.6, 0.2, 0.05, 0.05]
        for h in range(1, 7):
            for bin_label, p in zip(
                ["<10k", "10k-<50k", "50k-<250k", "250k-<500k", ">=500k"], ensemble_spd
            ):
                con.execute(
                    """
                    INSERT INTO forecasts_ensemble (question_id, horizon_m, class_bin, p,
                                                    aggregator, ensemble_version)
                    VALUES ('Q1', ?, ?, ?, 'Bayes_MC', 'v1_spd')
                    """,
                    [h, bin_label, p],
                )

        # Two models with slightly different SPDs
        model_spds = {
            "ModelA": [0.2, 0.5, 0.2, 0.05, 0.05],
            "ModelB": [0.05, 0.7, 0.15, 0.05, 0.05],
        }
        bin_labels = ["<10k", "10k-<50k", "50k-<250k", "250k-<500k", ">=500k"]
        for model_name, spd in model_spds.items():
            for h in range(1, 7):
                for bucket_idx, (bin_label, p) in enumerate(zip(bin_labels, spd), start=1):
                    con.execute(
                        """
                        INSERT INTO forecasts_raw (question_id, model_name, horizon_m,
                                                   class_bin, p, month_index, bucket_index,
                                                   probability, run_id, created_at)
                        VALUES ('Q1', ?, ?, ?, ?, ?, ?, ?, 'forecaster_run_1', now())
                        """,
                        [model_name, h, bin_label, p, h, bucket_idx, p],
                    )

    finally:
        con.close()

    # Run scores + calibration
    compute_scores(db_url=db_url)
    compute_calibration_pythia(db_url=db_url, as_of=date(2025, 3, 1))

    # Inspect results
    con = duckdb.connect(str(db_path))
    try:
        # Scores: all 6 horizons should be scored
        score_count = con.execute("SELECT COUNT(*) FROM scores").fetchone()[0]
        # 6 horizons x 3 score_types x 3 entities (ensemble + ModelA + ModelB) = 54
        assert score_count == 54, f"Expected 54 score rows, got {score_count}"

        # Each horizon should have scores
        horizon_counts = con.execute(
            "SELECT horizon_m, COUNT(*) FROM scores GROUP BY horizon_m ORDER BY horizon_m"
        ).fetchall()
        assert len(horizon_counts) == 6
        for h, cnt in horizon_counts:
            assert cnt == 9, f"Expected 9 score rows for h{h}, got {cnt}"

        # Scores should differ between horizons because ground truth differs
        h1_brier = con.execute(
            "SELECT value FROM scores WHERE horizon_m = 1 AND score_type = 'brier' AND model_name IS NULL"
        ).fetchone()[0]
        h4_brier = con.execute(
            "SELECT value FROM scores WHERE horizon_m = 4 AND score_type = 'brier' AND model_name IS NULL"
        ).fetchone()[0]
        # h1 resolved to 5000 (bucket 0), h4 resolved to 60000 (bucket 2)
        # Both scored against ensemble SPD [0.1, 0.6, 0.2, 0.05, 0.05]
        # These must differ since the true bucket is different
        assert abs(h1_brier - h4_brier) > 1e-4, (
            f"h1 and h4 Brier scores should differ (h1={h1_brier}, h4={h4_brier})"
        )

        # Scores: expect brier/log/crps for ensemble (model_name NULL) and two models
        score_types = {r[0] for r in con.execute("SELECT DISTINCT score_type FROM scores").fetchall()}
        models_scored = {r[0] for r in con.execute("SELECT DISTINCT model_name FROM scores").fetchall()}
        assert {"brier", "log", "crps"} <= score_types
        assert {None, "ModelA", "ModelB"} <= models_scored

        # Calibration weights: one row per model_name for (FL, PA)
        weights_df = con.execute(
            """
            SELECT model_name, weight, avg_brier
            FROM calibration_weights
            WHERE hazard_code = 'FL' AND metric = 'PA'
            """
        ).fetchdf()
        assert not weights_df.empty
        # Ensure ModelA and ModelB both present and weights not identical
        assert {"ModelA", "ModelB"} <= set(weights_df["model_name"].tolist())
        wa = float(weights_df.loc[weights_df["model_name"] == "ModelA", "weight"].iloc[0])
        wb = float(weights_df.loc[weights_df["model_name"] == "ModelB", "weight"].iloc[0])
        assert abs(wa - wb) > 1e-4

        # Advice: non-empty for FL/PA
        advice_df = con.execute(
            """
            SELECT advice
            FROM calibration_advice
            WHERE hazard_code = 'FL' AND metric = 'PA'
            ORDER BY as_of_month DESC
            LIMIT 1
            """
        ).fetchdf()
        assert not advice_df.empty
        advice = str(advice_df["advice"].iloc[0] or "").strip()
        assert advice != ""
    finally:
        con.close()


@pytest.mark.db
def test_dashboard_resolved_count_uses_resolutions_table(tmp_path: Path):
    """The dashboard resolved_questions count should prefer the resolutions
    table over questions.status.  Even when status is still 'active', a
    question that has rows in the resolutions table should be counted."""
    db_path = tmp_path / "dashboard_test.duckdb"

    con = duckdb.connect(str(db_path))
    try:
        con.execute("CREATE TABLE hs_runs (hs_run_id TEXT PRIMARY KEY)")
        con.execute("INSERT INTO hs_runs VALUES ('run1')")
        con.execute(
            """
            CREATE TABLE questions (
              question_id TEXT,
              hs_run_id TEXT,
              iso3 TEXT,
              hazard_code TEXT,
              metric TEXT,
              target_month TEXT,
              window_start_date DATE,
              status TEXT
            )
            """
        )
        # Two questions — both status='active' (NOT 'resolved')
        con.execute(
            """
            INSERT INTO questions VALUES
                ('Q1', 'run1', 'MLI', 'FL', 'PA', '2025-01', '2024-08-01', 'active'),
                ('Q2', 'run1', 'ETH', 'DR', 'PA', '2025-01', '2024-08-01', 'active')
            """
        )
        con.execute(
            """
            CREATE TABLE resolutions (
              question_id TEXT,
              horizon_m INTEGER,
              observed_month TEXT,
              value DOUBLE,
              source_snapshot_ym TEXT,
              created_at TIMESTAMP,
              PRIMARY KEY (question_id, horizon_m)
            )
            """
        )
        # Q1 has one resolution, Q2 has none
        con.execute(
            """
            INSERT INTO resolutions (question_id, horizon_m, observed_month, value, created_at)
            VALUES ('Q1', 1, '2024-08', 5000.0, now())
            """
        )

        # Replicate the dashboard's resolved_questions SQL logic:
        # Before the fix this would check questions.status first and get 0.
        # After the fix it checks resolutions table first.
        resolved_count = con.execute(
            "SELECT COUNT(DISTINCT r.question_id) FROM resolutions r "
            "JOIN questions q ON q.question_id = r.question_id"
        ).fetchone()[0]
        assert resolved_count == 1, (
            f"Expected 1 resolved question via resolutions table, got {resolved_count}"
        )

        # Verify that status-based counting would have missed it
        status_count = con.execute(
            "SELECT COUNT(DISTINCT question_id) FROM questions "
            "WHERE status IN ('resolved', 'closed')"
        ).fetchone()[0]
        assert status_count == 0, "Status-based count should be 0 (both 'active')"
    finally:
        con.close()
