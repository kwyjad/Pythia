from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from resolver.query.countries_index import compute_countries_index


def test_compute_countries_index_last_forecasted() -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE questions (
            question_id TEXT,
            iso3 TEXT,
            hs_run_id TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE hs_runs (
            hs_run_id TEXT,
            created_at TIMESTAMP
        );
        """
    )
    con.execute(
        """
        CREATE TABLE forecasts_ensemble (
            question_id TEXT,
            created_at TIMESTAMP,
            status TEXT
        );
        """
    )
    con.execute(
        """
        INSERT INTO hs_runs (hs_run_id, created_at)
        VALUES
            ('hs_run_old', '2024-01-01 00:00:00'),
            ('hs_run_new', '2024-03-01 00:00:00');
        """
    )
    con.execute(
        """
        INSERT INTO questions (question_id, iso3, hs_run_id)
        VALUES
            ('q1', 'usa', 'hs_run_old'),
            ('q2', 'usa', 'hs_run_new');
        """
    )
    con.execute(
        """
        INSERT INTO forecasts_ensemble (question_id, created_at, status)
        VALUES
            ('q1', '2024-01-15 00:00:00', 'ok'),
            ('q2', '2024-02-15 00:00:00', 'ok');
        """
    )

    rows = compute_countries_index(con)

    assert rows == [
        {
            "iso3": "USA",
            "n_questions": 2,
            "n_forecasted": 2,
            "last_triaged": "2024-03-01",
            "last_forecasted": "2024-02-15",
            "highest_rc_level": None,
            "highest_rc_score": None,
        }
    ]
