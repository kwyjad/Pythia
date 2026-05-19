# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Regression tests for the safe-DDL helpers in the calibration scripts.

These tests pin the behaviour that was broken on DuckDB 1.5.2 in production:
a failed `ALTER TABLE ... ADD COLUMN` inside `try/except: pass` left the
connection in an aborted-transaction state, causing the subsequent
`PRAGMA table_info(...)` inside `_table_exists` to fail. The scripts then
falsely reported "scores table not found" and skipped the whole pipeline.

The fix replaces "ALTER inside broad try/except" with `_add_column_if_missing`,
which checks columns via PRAGMA first and only ALTERs when needed; and adds a
`_rollback_quietly(conn)` in the except paths of `_table_exists` /
`_row_count` / `_column_exists` so any residual aborted-state gets scrubbed.
"""

from __future__ import annotations

import tempfile

import duckdb
import pytest

from pythia.tools import compute_calibration_pythia as cc
from pythia.tools import generate_calibration_advice as gca


@pytest.fixture
def db_with_scores(tmp_path):
    """A small DB with a populated `scores` table and a `questions` table
    that already has the `is_test` column — so the migration ALTER fails."""
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE questions (question_id TEXT, hazard_code TEXT, metric TEXT, is_test BOOLEAN DEFAULT FALSE)"
    )
    con.execute(
        "CREATE TABLE scores (question_id TEXT, horizon_m INTEGER, model_name TEXT, score_type TEXT, value DOUBLE)"
    )
    con.execute(
        "INSERT INTO scores VALUES ('q1', 1, 'm', 'brier', 0.5), ('q2', 2, 'm', 'brier', 0.4)"
    )
    con.close()
    return str(db_path)


@pytest.mark.parametrize("module", [cc, gca], ids=["compute_calibration_pythia", "generate_calibration_advice"])
def test_table_exists_recovers_after_failed_alter(db_with_scores, module):
    """After a failing ALTER inside try/except, _table_exists must still return
    True for an existing table. This was False on DuckDB >= 1.5 before the fix."""
    con = duckdb.connect(db_with_scores)

    # Trigger the failing ALTER pattern that historically poisoned the txn.
    try:
        con.execute("ALTER TABLE questions ADD COLUMN is_test BOOLEAN DEFAULT FALSE")
    except Exception:
        # The production code used to do `pass` here; we replicate that to
        # confirm the helpers below recover regardless.
        pass

    assert module._table_exists(con, "scores") is True
    assert module._row_count(con, "scores") == 2
    con.close()


@pytest.mark.parametrize("module", [cc, gca], ids=["compute_calibration_pythia", "generate_calibration_advice"])
def test_add_column_if_missing_is_idempotent(db_with_scores, module):
    """The helper must not raise when the column already exists, must not
    leave the connection in a poisoned state, and must add the column when
    it is genuinely missing."""
    con = duckdb.connect(db_with_scores)

    # Existing column: no-op, no exception, connection stays usable.
    module._add_column_if_missing(con, "questions", "is_test", "BOOLEAN DEFAULT FALSE")
    assert module._table_exists(con, "scores") is True

    # Missing column: gets added.
    assert module._column_exists(con, "questions", "new_flag") is False
    module._add_column_if_missing(con, "questions", "new_flag", "TEXT")
    assert module._column_exists(con, "questions", "new_flag") is True
    con.close()


def test_compute_calibration_pythia_creates_calibration_advice_with_4col_pk(tmp_path):
    """Bug 2 regression: the CREATE TABLE in compute_calibration_pythia.py
    must use the 4-column PK including model_name (matches schema.py)."""
    db_path = tmp_path / "fresh.duckdb"
    # Connect, run only the table-setup portion of compute_calibration_pythia
    # by calling the function with no scores table — it should return early
    # after creating the tables. We can't easily call the inner code only,
    # so we read the source and assert the PK shape is right.
    import inspect
    src = inspect.getsource(cc.compute_calibration_pythia)
    assert "PRIMARY KEY (as_of_month, hazard_code, metric, model_name)" in src, (
        "calibration_advice CREATE must use the 4-column PK to match schema.py; "
        "the legacy 3-column form (as_of_month, hazard_code, metric) re-triggers "
        "the dual-constraint failure documented in CLAUDE.md."
    )
    assert "PRIMARY KEY (as_of_month, hazard_code, metric)\n" not in src, (
        "The legacy 3-column PK is still in compute_calibration_pythia.py — "
        "see CLAUDE.md Known Failure Modes."
    )
