from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from resolver.query.questions_index import compute_questions_forecast_summary


def test_compute_questions_forecast_summary_uses_latest_run() -> None:
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE questions (
            question_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE forecasts_ensemble (
            question_id TEXT,
            run_id TEXT,
            created_at TIMESTAMP,
            status TEXT,
            month_index INTEGER,
            bucket_index INTEGER,
            probability DOUBLE
        );
        """
    )
    conn.execute(
        """
        CREATE TABLE bucket_centroids (
            hazard_code TEXT,
            metric TEXT,
            bucket_index INTEGER,
            centroid DOUBLE
        );
        """
    )
    conn.execute(
        """
        INSERT INTO questions (question_id, iso3, hazard_code, metric)
        VALUES ('q1', 'USA', 'TC', 'PIN');
        """
    )
    conn.execute(
        """
        INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid)
        VALUES ('TC', 'PIN', 1, 10.0), ('TC', 'PIN', 2, 20.0);
        """
    )
    conn.execute(
        """
        INSERT INTO forecasts_ensemble (
            question_id,
            run_id,
            created_at,
            status,
            month_index,
            bucket_index,
            probability
        )
        VALUES
            ('q1', 'run_old', '2024-01-10 00:00:00', 'ok', 1, 1, 0.25),
            ('q1', 'run_old', '2024-01-10 00:00:00', 'ok', 2, 2, 0.75),
            ('q1', 'run_new', '2024-02-10 00:00:00', 'ok', 1, 1, 0.5),
            ('q1', 'run_new', '2024-02-10 00:00:00', 'ok', 3, 2, 0.5);
        """
    )

    summary = compute_questions_forecast_summary(conn, question_ids=["q1"])
    assert summary["q1"]["forecast_date"] == "2024-02-10"
    assert summary["q1"]["horizon_max"] == 3
    assert summary["q1"]["eiv_total"] == pytest.approx(15.0)

    conn.close()
