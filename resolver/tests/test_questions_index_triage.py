from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from resolver.query.questions_index import compute_questions_triage_summary


def test_compute_questions_triage_summary_selects_latest_row() -> None:
    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE questions (
            question_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            hs_run_id TEXT,
            metric TEXT,
            target_month TEXT,
            status TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE hs_triage (
            run_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            tier TEXT,
            triage_score DOUBLE,
            need_full_spd BOOLEAN,
            created_at TIMESTAMP
        );
        """
    )
    con.execute(
        """
        INSERT INTO hs_triage (
            run_id, iso3, hazard_code, tier, triage_score, need_full_spd, created_at
        )
        VALUES
            ('run_1', 'USA', 'TC', 'watchlist', 0.41, FALSE, '2024-01-01 00:00:00'),
            ('run_1', 'USA', 'TC', 'priority', 0.78, TRUE, '2024-02-01 00:00:00');
        """
    )

    rows = [
        {
            "question_id": "q1",
            "hs_run_id": "run_1",
            "iso3": "USA",
            "hazard_code": "TC",
        },
        {
            "question_id": "q2",
            "hs_run_id": None,
            "iso3": "USA",
            "hazard_code": "TC",
        },
    ]

    summary = compute_questions_triage_summary(con, rows)

    assert summary["q1"]["triage_score"] == pytest.approx(0.78)
    assert summary["q1"]["triage_tier"] == "priority"
    assert summary["q1"]["triage_need_full_spd"] is True
    assert summary.get("q2") is None

    con.close()
