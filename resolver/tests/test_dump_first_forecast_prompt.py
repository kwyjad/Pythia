from __future__ import annotations

import pytest

# Skip cleanly if duckdb is not available in the test environment
duckdb = pytest.importorskip("duckdb")

from scripts import dump_first_forecast_prompt as dfp  # type: ignore


def _create_forecasts_ensemble_table(con):
    # Minimal schema matching pythia/db/schema.py DDL for forecasts_ensemble
    con.execute(
        """
        CREATE TABLE forecasts_ensemble (
            run_id TEXT,
            question_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            month_index INTEGER,
            bucket_index INTEGER,
            probability DOUBLE,
            ev_value DOUBLE,
            weights_profile TEXT,
            created_at TIMESTAMP,
            status TEXT,
            human_explanation TEXT
        );
        """
    )


def test_load_spd_ignores_null_month_index_and_bucket_index(tmp_path):
    db_path = tmp_path / "test_forecasts_ensemble.duckdb"
    con = duckdb.connect(str(db_path))
    try:
        _create_forecasts_ensemble_table(con)

        # Row representing a no-forecast outcome: NULL month_index/bucket_index
        con.execute(
            """
            INSERT INTO forecasts_ensemble (
                run_id, question_id, iso3, hazard_code, metric,
                month_index, bucket_index, probability, ev_value,
                weights_profile, created_at, status, human_explanation
            ) VALUES ('run1', 'q1', 'ETH', 'FL', 'PA',
                      NULL, NULL, NULL, NULL,
                      'ensemble', CURRENT_TIMESTAMP, 'no_forecast', 'no forecast available')
            """
        )

        # A proper SPD row that should be used
        con.execute(
            """
            INSERT INTO forecasts_ensemble (
                run_id, question_id, iso3, hazard_code, metric,
                month_index, bucket_index, probability, ev_value,
                weights_profile, created_at, status, human_explanation
            ) VALUES ('run1', 'q1', 'ETH', 'FL', 'PA',
                      1, 1, 0.7, 123.0,
                      'ensemble', CURRENT_TIMESTAMP, 'ok', 'ok')
            """
        )

        ensemble_probs, ensemble_ev = dfp._load_spd_from_forecasts_ensemble(
            con, "run1", "q1"
        )

        # We should see only the valid SPD row reflected in the result
        assert ensemble_probs == {1: [0.7, 0.0, 0.0, 0.0, 0.0]}
        assert ensemble_ev == {1: pytest.approx(123.0)}
    finally:
        con.close()
