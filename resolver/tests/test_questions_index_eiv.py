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
            model_name TEXT,
            status TEXT,
            hazard_code TEXT,
            metric TEXT,
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
            model_name,
            status,
            hazard_code,
            metric,
            month_index,
            bucket_index,
            probability
        )
        VALUES
            ('q1', 'run_old', '2024-01-10 00:00:00', 'ensemble_mean_v2', 'ok', 'TC', 'PIN', 1, 1, 0.25),
            ('q1', 'run_old', '2024-01-10 00:00:00', 'ensemble_mean_v2', 'ok', 'TC', 'PIN', 2, 2, 0.75),
            ('q1', 'run_new', '2024-02-10 00:00:00', 'ensemble_mean_v2', 'ok', 'TC', 'PIN', 1, 1, 0.5),
            ('q1', 'run_new', '2024-02-10 00:00:00', 'ensemble_mean_v2', 'ok', 'TC', 'PIN', 3, 2, 0.5);
        """
    )

    summary = compute_questions_forecast_summary(conn, question_ids=["q1"])
    assert summary["q1"]["forecast_date"] == "2024-02-10"
    assert summary["q1"]["horizon_max"] == 3
    assert summary["q1"]["eiv_total"] == pytest.approx(15.0)

    conn.close()


def test_compute_questions_forecast_summary_uses_wildcard_centroids() -> None:
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
            model_name TEXT,
            status TEXT,
            hazard_code TEXT,
            metric TEXT,
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
        VALUES ('q2', 'USA', 'ACE', 'PA');
        """
    )
    conn.execute(
        """
        INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid)
        VALUES ('*', 'PA', 1, 5.0), ('*', 'PA', 2, 15.0);
        """
    )
    conn.execute(
        """
        INSERT INTO forecasts_ensemble (
            question_id,
            run_id,
            created_at,
            model_name,
            status,
            hazard_code,
            metric,
            month_index,
            bucket_index,
            probability
        )
        VALUES
            ('q2', 'run_new', '2024-02-10 00:00:00', 'ensemble_mean_v2', 'ok', 'ACE', 'PA', 1, 1, 0.4),
            ('q2', 'run_new', '2024-02-10 00:00:00', 'ensemble_mean_v2', 'ok', 'ACE', 'PA', 2, 2, 0.6);
        """
    )

    summary = compute_questions_forecast_summary(conn, question_ids=["q2"])
    assert summary["q2"]["forecast_date"] == "2024-02-10"
    assert summary["q2"]["horizon_max"] == 2
    assert summary["q2"]["eiv_total"] == pytest.approx(11.0)

    conn.close()


def test_compute_questions_forecast_summary_falls_back_without_centroids() -> None:
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
            model_name TEXT,
            status TEXT,
            hazard_code TEXT,
            metric TEXT,
            month_index INTEGER,
            bucket_index INTEGER,
            probability DOUBLE
        );
        """
    )
    conn.execute(
        """
        INSERT INTO questions (question_id, iso3, hazard_code, metric)
        VALUES ('q3', 'USA', 'EQ', 'PA');
        """
    )
    conn.execute(
        """
        INSERT INTO forecasts_ensemble (
            question_id,
            run_id,
            created_at,
            model_name,
            status,
            hazard_code,
            metric,
            month_index,
            bucket_index,
            probability
        )
        VALUES
            ('q3', 'run_new', '2024-03-05 00:00:00', 'ensemble_mean_v2', 'ok', 'EQ', 'PA', 1, 2, 0.5),
            ('q3', 'run_new', '2024-03-05 00:00:00', 'ensemble_mean_v2', 'ok', 'EQ', 'PA', 2, 3, 0.5);
        """
    )

    summary = compute_questions_forecast_summary(conn, question_ids=["q3"])
    assert summary["q3"]["forecast_date"] == "2024-03-05"
    assert summary["q3"]["horizon_max"] == 2
    assert summary["q3"]["eiv_total"] == pytest.approx(90000.0)

    conn.close()


def test_compute_questions_forecast_summary_prefers_bayesmc() -> None:
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
            model_name TEXT,
            status TEXT,
            hazard_code TEXT,
            metric TEXT,
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
        VALUES ('q4', 'USA', 'TC', 'PIN');
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
            model_name,
            status,
            hazard_code,
            metric,
            month_index,
            bucket_index,
            probability
        )
        VALUES
            ('q4', 'run_new', '2024-02-12 00:00:00', 'ensemble_bayesmc_v2', 'ok', 'TC', 'PIN', 1, 1, 0.1),
            ('q4', 'run_new', '2024-02-12 00:00:00', 'ensemble_bayesmc_v2', 'ok', 'TC', 'PIN', 2, 2, 0.9),
            ('q4', 'run_new', '2024-02-12 00:00:00', 'ensemble_mean_v2', 'ok', 'TC', 'PIN', 1, 1, 0.5),
            ('q4', 'run_new', '2024-02-12 00:00:00', 'ensemble_mean_v2', 'ok', 'TC', 'PIN', 2, 2, 0.5);
        """
    )

    summary = compute_questions_forecast_summary(conn, question_ids=["q4"])
    assert summary["q4"]["forecast_date"] == "2024-02-12"
    assert summary["q4"]["horizon_max"] == 2
    assert summary["q4"]["eiv_total"] == pytest.approx(19.0)

    conn.close()


def test_compute_questions_forecast_summary_falls_back_to_mean() -> None:
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
            model_name TEXT,
            status TEXT,
            hazard_code TEXT,
            metric TEXT,
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
        VALUES ('q5', 'USA', 'TC', 'PIN');
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
            model_name,
            status,
            hazard_code,
            metric,
            month_index,
            bucket_index,
            probability
        )
        VALUES
            ('q5', 'run_new', '2024-02-12 00:00:00', 'ensemble_mean_v2', 'ok', 'TC', 'PIN', 1, 1, 0.5),
            ('q5', 'run_new', '2024-02-12 00:00:00', 'ensemble_mean_v2', 'ok', 'TC', 'PIN', 2, 2, 0.5);
        """
    )

    summary = compute_questions_forecast_summary(conn, question_ids=["q5"])
    assert summary["q5"]["forecast_date"] == "2024-02-12"
    assert summary["q5"]["horizon_max"] == 2
    assert summary["q5"]["eiv_total"] == pytest.approx(15.0)

    conn.close()
