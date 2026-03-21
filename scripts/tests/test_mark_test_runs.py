# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for scripts/mark_test_runs.py."""

import os
import tempfile

import duckdb
import pytest

# Allow import from scripts/
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.mark_test_runs import mark_runs


def _create_test_db(db_path: str) -> None:
    """Create a minimal DuckDB with tables that mark_test_runs expects."""
    con = duckdb.connect(db_path)

    # questions (keyed by hs_run_id)
    con.execute("""
        CREATE TABLE questions (
            question_id VARCHAR,
            hs_run_id VARCHAR,
            is_test BOOLEAN DEFAULT FALSE
        )
    """)

    # forecasts_ensemble (keyed by run_id)
    con.execute("""
        CREATE TABLE forecasts_ensemble (
            question_id VARCHAR,
            run_id VARCHAR,
            model_name VARCHAR,
            bucket_1 DOUBLE,
            is_test BOOLEAN DEFAULT FALSE
        )
    """)

    # hs_runs (keyed by hs_run_id)
    con.execute("""
        CREATE TABLE hs_runs (
            hs_run_id VARCHAR,
            started_at TIMESTAMP,
            is_test BOOLEAN DEFAULT FALSE
        )
    """)

    # llm_calls (dual-key: run_id and hs_run_id)
    con.execute("""
        CREATE TABLE llm_calls (
            call_id VARCHAR,
            run_id VARCHAR,
            hs_run_id VARCHAR,
            model VARCHAR,
            is_test BOOLEAN DEFAULT FALSE
        )
    """)

    # resolutions (inherited via question_id)
    con.execute("""
        CREATE TABLE resolutions (
            question_id VARCHAR,
            horizon_m INTEGER,
            value DOUBLE,
            is_test BOOLEAN DEFAULT FALSE
        )
    """)

    # scores (inherited via question_id)
    con.execute("""
        CREATE TABLE scores (
            question_id VARCHAR,
            horizon_m INTEGER,
            brier DOUBLE,
            is_test BOOLEAN DEFAULT FALSE
        )
    """)

    # --- Insert fixture data ---

    # Two HS runs: hs-run-1 (will be marked), hs-run-2 (should stay untouched)
    con.execute("INSERT INTO hs_runs VALUES ('hs-run-1', '2026-01-01', FALSE)")
    con.execute("INSERT INTO hs_runs VALUES ('hs-run-2', '2026-02-01', FALSE)")

    # Questions linked to HS runs
    con.execute("INSERT INTO questions VALUES ('q1', 'hs-run-1', FALSE)")
    con.execute("INSERT INTO questions VALUES ('q2', 'hs-run-1', FALSE)")
    con.execute("INSERT INTO questions VALUES ('q3', 'hs-run-2', FALSE)")

    # Forecasts linked to forecaster run_ids
    con.execute("INSERT INTO forecasts_ensemble VALUES ('q1', 'fc-run-A', 'gpt5', 0.5, FALSE)")
    con.execute("INSERT INTO forecasts_ensemble VALUES ('q1', 'fc-run-B', 'gemini', 0.6, FALSE)")
    con.execute("INSERT INTO forecasts_ensemble VALUES ('q3', 'fc-run-C', 'gpt5', 0.3, FALSE)")

    # LLM calls linked to both run types
    con.execute("INSERT INTO llm_calls VALUES ('c1', 'fc-run-A', NULL, 'gpt5', FALSE)")
    con.execute("INSERT INTO llm_calls VALUES ('c2', NULL, 'hs-run-1', 'gemini', FALSE)")
    con.execute("INSERT INTO llm_calls VALUES ('c3', 'fc-run-C', NULL, 'gpt5', FALSE)")

    # Resolutions and scores linked to question_ids
    con.execute("INSERT INTO resolutions VALUES ('q1', 1, 100.0, FALSE)")
    con.execute("INSERT INTO resolutions VALUES ('q2', 1, 200.0, FALSE)")
    con.execute("INSERT INTO resolutions VALUES ('q3', 1, 50.0, FALSE)")

    con.execute("INSERT INTO scores VALUES ('q1', 1, 0.15, FALSE)")
    con.execute("INSERT INTO scores VALUES ('q3', 1, 0.10, FALSE)")

    con.close()


@pytest.fixture
def test_db():
    """Create a temporary DuckDB file with fixture data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.duckdb")
        _create_test_db(db_path)
        yield db_path


def test_mark_hs_run_id(test_db):
    """Marking an hs_run_id should flag hs_runs and questions rows."""
    mark_runs(f"duckdb:///{test_db}", "hs-run-1")

    con = duckdb.connect(test_db, read_only=True)

    # hs_runs: hs-run-1 marked, hs-run-2 untouched
    assert con.execute("SELECT is_test FROM hs_runs WHERE hs_run_id = 'hs-run-1'").fetchone()[0] is True
    assert con.execute("SELECT is_test FROM hs_runs WHERE hs_run_id = 'hs-run-2'").fetchone()[0] is False

    # questions: q1, q2 marked (belong to hs-run-1), q3 untouched
    assert con.execute("SELECT is_test FROM questions WHERE question_id = 'q1'").fetchone()[0] is True
    assert con.execute("SELECT is_test FROM questions WHERE question_id = 'q2'").fetchone()[0] is True
    assert con.execute("SELECT is_test FROM questions WHERE question_id = 'q3'").fetchone()[0] is False

    con.close()


def test_mark_forecaster_run_id(test_db):
    """Marking a forecaster run_id should flag forecasts_ensemble and llm_calls rows."""
    mark_runs(f"duckdb:///{test_db}", "fc-run-A")

    con = duckdb.connect(test_db, read_only=True)

    # forecasts_ensemble: fc-run-A marked, fc-run-B and fc-run-C untouched
    assert con.execute("SELECT is_test FROM forecasts_ensemble WHERE run_id = 'fc-run-A'").fetchone()[0] is True
    assert con.execute("SELECT is_test FROM forecasts_ensemble WHERE run_id = 'fc-run-B'").fetchone()[0] is False
    assert con.execute("SELECT is_test FROM forecasts_ensemble WHERE run_id = 'fc-run-C'").fetchone()[0] is False

    # llm_calls: c1 (run_id=fc-run-A) marked, c2 and c3 untouched
    assert con.execute("SELECT is_test FROM llm_calls WHERE call_id = 'c1'").fetchone()[0] is True
    assert con.execute("SELECT is_test FROM llm_calls WHERE call_id = 'c2'").fetchone()[0] is False
    assert con.execute("SELECT is_test FROM llm_calls WHERE call_id = 'c3'").fetchone()[0] is False

    con.close()


def test_inherited_tables_marked_via_question_id(test_db):
    """After marking an hs_run_id, resolutions/scores for those questions should also be marked."""
    mark_runs(f"duckdb:///{test_db}", "hs-run-1")

    con = duckdb.connect(test_db, read_only=True)

    # q1 and q2 belong to hs-run-1, so their resolutions/scores should be marked
    assert con.execute("SELECT is_test FROM resolutions WHERE question_id = 'q1'").fetchone()[0] is True
    assert con.execute("SELECT is_test FROM resolutions WHERE question_id = 'q2'").fetchone()[0] is True
    assert con.execute("SELECT is_test FROM scores WHERE question_id = 'q1'").fetchone()[0] is True

    # q3 belongs to hs-run-2, should be untouched
    assert con.execute("SELECT is_test FROM resolutions WHERE question_id = 'q3'").fetchone()[0] is False
    assert con.execute("SELECT is_test FROM scores WHERE question_id = 'q3'").fetchone()[0] is False

    con.close()


def test_mark_multiple_run_ids(test_db):
    """Comma-separated run IDs should all be processed."""
    mark_runs(f"duckdb:///{test_db}", "hs-run-1, fc-run-C")

    con = duckdb.connect(test_db, read_only=True)

    # hs-run-1 tables marked
    assert con.execute("SELECT is_test FROM hs_runs WHERE hs_run_id = 'hs-run-1'").fetchone()[0] is True
    # fc-run-C tables marked
    assert con.execute("SELECT is_test FROM forecasts_ensemble WHERE run_id = 'fc-run-C'").fetchone()[0] is True
    # hs-run-2 untouched
    assert con.execute("SELECT is_test FROM hs_runs WHERE hs_run_id = 'hs-run-2'").fetchone()[0] is False

    con.close()


def test_dual_key_llm_calls_hs_run_id(test_db):
    """llm_calls rows keyed by hs_run_id should be marked when that hs_run_id is provided."""
    mark_runs(f"duckdb:///{test_db}", "hs-run-1")

    con = duckdb.connect(test_db, read_only=True)

    # c2 has hs_run_id=hs-run-1
    assert con.execute("SELECT is_test FROM llm_calls WHERE call_id = 'c2'").fetchone()[0] is True
    # c1 and c3 have different run_ids / no hs_run_id match
    assert con.execute("SELECT is_test FROM llm_calls WHERE call_id = 'c1'").fetchone()[0] is False
    assert con.execute("SELECT is_test FROM llm_calls WHERE call_id = 'c3'").fetchone()[0] is False

    con.close()


def test_nonexistent_run_id_no_error(test_db):
    """Providing a run ID that doesn't exist should not raise an error."""
    mark_runs(f"duckdb:///{test_db}", "nonexistent-run-999")

    con = duckdb.connect(test_db, read_only=True)
    # Everything should remain FALSE
    assert con.execute("SELECT COUNT(*) FROM hs_runs WHERE is_test = TRUE").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM questions WHERE is_test = TRUE").fetchone()[0] == 0
    con.close()
