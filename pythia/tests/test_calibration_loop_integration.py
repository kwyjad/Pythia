from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.tools.compute_scores import compute_scores
from pythia.tools.compute_calibration_pythia import compute_calibration_pythia


@pytest.mark.db
def test_calibration_loop_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Smoke test for the Pythia calibration loop:

      questions + resolutions + forecasts_*  -> scores
      scores                                 -> calibration_weights + calibration_advice

    We build a tiny DuckDB with one question, one resolution, and simple SPDs
    for two models and the ensemble. We then check that weights/advice are written.
    """
    db_path = tmp_path / "calibration_test.duckdb"
    db_url = f"duckdb:///{db_path}"

    # Monkeypatch config loader to point app.db_url at our temp DB
    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}

    monkeypatch.setattr("pythia.tools.compute_scores.load", _fake_load_cfg)
    monkeypatch.setattr("pythia.tools.compute_calibration_pythia.load", _fake_load_cfg)

    con = duckdb.connect(str(db_path))
    try:
        # Minimal schema
        con.execute(
            """
            CREATE TABLE questions (
              question_id TEXT,
              iso3 TEXT,
              hazard_code TEXT,
              metric TEXT,
              target_month TEXT,
              status TEXT,
              run_id TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE resolutions (
              question_id TEXT,
              observed_month TEXT,
              value DOUBLE,
              source_snapshot_ym TEXT,
              created_at TIMESTAMP
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
              run_id TEXT,
              created_at TIMESTAMP
            )
            """
        )

        # Insert one PA question
        con.execute(
            """
            INSERT INTO questions (question_id, iso3, hazard_code, metric, target_month, status, run_id)
            VALUES ('Q1', 'MLI', 'FLOOD', 'PA', '2025-01', 'active', 'hs_run_1')
            """
        )

        # Resolution: e.g. 30k people affected (falls into 10k-<50k bucket)
        con.execute(
            """
            INSERT INTO resolutions (question_id, observed_month, value, source_snapshot_ym, created_at)
            VALUES ('Q1', '2025-01', 30000.0, '2025-02', now())
            """
        )

        # Simple SPD for ensemble: modest mass in second bucket
        ensemble_spd = [0.1, 0.6, 0.2, 0.05, 0.05]  # buckets: <10k,10k-<50k,50k-<250k,250k-<500k,>=500k
        for h in range(1, 7):
            for bin_label, p in zip(
                ["<10k", "10k-<50k", "50k-<250k", "250k-<500k", ">=500k"], ensemble_spd
            ):
                con.execute(
                    """
                    INSERT INTO forecasts_ensemble (question_id, horizon_m, class_bin, p, aggregator, ensemble_version)
                    VALUES ('Q1', ?, ?, ?, 'Bayes_MC', 'v1_spd')
                    """,
                    [h, bin_label, p],
                )

        # Two models with slightly different SPDs
        model_spds = {
            "ModelA": [0.2, 0.5, 0.2, 0.05, 0.05],  # slightly worse on the true bucket
            "ModelB": [0.05, 0.7, 0.15, 0.05, 0.05],  # better on the true bucket
        }
        for model_name, spd in model_spds.items():
            for h in range(1, 7):
                for bin_label, p in zip(
                    ["<10k", "10k-<50k", "50k-<250k", "250k-<500k", ">=500k"],
                    spd,
                ):
                    con.execute(
                        """
                        INSERT INTO forecasts_raw (question_id, model_name, horizon_m, class_bin, p, run_id, created_at)
                        VALUES ('Q1', ?, ?, ?, ?, 'forecaster_run_1', now())
                        """,
                        [model_name, h, bin_label, p],
                    )

    finally:
        con.close()

    # Run scores + calibration
    compute_scores(db_url=db_url)
    compute_calibration_pythia(db_url=db_url, as_of=date(2025, 3, 1))

    # Inspect results
    con = duckdb.connect(str(db_path))
    try:
        # Scores: expect brier/log/crps for ensemble (model_name NULL) and two models
        scores_df = con.execute("SELECT DISTINCT score_type, model_name FROM scores").fetchdf()
        score_types = set(scores_df["score_type"].tolist())
        models_scored = set(scores_df["model_name"].tolist())
        assert {"brier", "log", "crps"} <= score_types
        assert models_scored.issuperset({None, "ModelA", "ModelB"})

        # Calibration weights: one row per model_name for (FLOOD, PA)
        weights_df = con.execute(
            """
            SELECT model_name, weight, avg_brier
            FROM calibration_weights
            WHERE hazard_code = 'FLOOD' AND metric = 'PA'
            """
        ).fetchdf()
        assert not weights_df.empty
        # Ensure ModelA and ModelB both present and weights not identical
        assert {"ModelA", "ModelB"} <= set(weights_df["model_name"].tolist())
        wa = float(weights_df.loc[weights_df["model_name"] == "ModelA", "weight"].iloc[0])
        wb = float(weights_df.loc[weights_df["model_name"] == "ModelB", "weight"].iloc[0])
        assert abs(wa - wb) > 1e-4

        # Advice: non-empty for FLOOD/PA
        advice_df = con.execute(
            """
            SELECT advice
            FROM calibration_advice
            WHERE hazard_code = 'FLOOD' AND metric = 'PA'
            ORDER BY as_of_month DESC
            LIMIT 1
            """
        ).fetchdf()
        assert not advice_df.empty
        advice = str(advice_df["advice"].iloc[0] or "").strip()
        assert advice != ""
    finally:
        con.close()
