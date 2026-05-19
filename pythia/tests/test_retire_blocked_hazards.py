# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the blocked-hazard question retirement migration."""

from __future__ import annotations

import duckdb
import pytest

from scripts.retire_blocked_hazard_questions import retire_blocked_hazard_questions


@pytest.fixture
def questions_db(tmp_path):
    """Small DB with a mix of active/already-retired/blocked-hazard questions."""
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE questions (
          question_id TEXT,
          hazard_code TEXT,
          metric TEXT,
          status TEXT,
          is_test BOOLEAN DEFAULT FALSE
        )
        """
    )
    rows = [
        ("ace1", "ACE", "FATALITIES", "active"),
        ("fl1", "FL", "PA", "active"),
        ("di1", "DI", "PA", "active"),
        ("di2", "DI", "PA", "active"),
        ("hw1", "HW", "PA", "active"),
        ("cu1", "CU", "PA", "active"),
        ("aco1", "ACO", "PA", "active"),
        # Already-retired DI row should not be touched / double-counted.
        ("di_old", "DI", "PA", "retired"),
        # Lowercase hazard code should still be matched (UPPER on column).
        ("hw_lc", "hw", "PA", "active"),
    ]
    con.executemany(
        "INSERT INTO questions(question_id, hazard_code, metric, status) VALUES (?,?,?,?)",
        rows,
    )
    return con


def test_retires_only_blocked_hazards(questions_db):
    # Pin the blocked set so the test doesn't drift if HAZARD_CONFIG adds new
    # blocked codes (the real BLOCKED_HAZARDS also picks up codes flagged with
    # blocked=True in pythia/config.yaml, e.g. EC, MULTI, PHE).
    blocked = {"DI", "HW", "CU", "ACO"}
    counts = retire_blocked_hazard_questions(questions_db, blocked=blocked)
    # 2 DI + 1 HW + 1 CU + 1 ACO + 1 lowercase HW = 6 rows newly retired.
    assert counts == {"ACO": 1, "CU": 1, "DI": 2, "HW": 2}

    # Verify DB state: ACE and FL are still active; all blocked-hazard rows are
    # now retired (including the originally-already-retired one — still retired).
    rows = questions_db.execute(
        "SELECT question_id, hazard_code, status FROM questions ORDER BY question_id"
    ).fetchall()
    status_by_id = {qid: status for qid, _hz, status in rows}
    assert status_by_id["ace1"] == "active"
    assert status_by_id["fl1"] == "active"
    assert status_by_id["di1"] == "retired"
    assert status_by_id["di2"] == "retired"
    assert status_by_id["hw1"] == "retired"
    assert status_by_id["cu1"] == "retired"
    assert status_by_id["aco1"] == "retired"
    assert status_by_id["di_old"] == "retired"  # unchanged
    assert status_by_id["hw_lc"] == "retired"


def test_idempotent_second_run(questions_db):
    blocked = {"DI", "HW", "CU", "ACO"}
    retire_blocked_hazard_questions(questions_db, blocked=blocked)
    counts = retire_blocked_hazard_questions(questions_db, blocked=blocked)
    # All blocked-hazard rows already retired; second run touches nothing.
    assert all(n == 0 for n in counts.values())


def test_no_op_when_no_questions_table(tmp_path):
    db_path = tmp_path / "empty.duckdb"
    con = duckdb.connect(str(db_path))
    # No questions table at all.
    counts = retire_blocked_hazard_questions(con)
    assert counts == {}


def test_no_op_when_status_column_missing(tmp_path):
    db_path = tmp_path / "no_status.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE TABLE questions (question_id TEXT, hazard_code TEXT)")
    con.execute("INSERT INTO questions VALUES ('di1', 'DI')")
    # Without a status column, the script must no-op rather than crashing.
    counts = retire_blocked_hazard_questions(con)
    assert counts == {}
