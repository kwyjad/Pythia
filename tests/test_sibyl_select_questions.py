# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Question-selection tests, including the test-mode HS run regression.

A test-mode HS pipeline stamps every question is_test=TRUE. Sibyl chains off
HS via workflow_run (which cannot carry the upstream test_mode input), so its
selector must derive test-run status from hs_runs.is_test — filtering purely
on the PYTHIA_TEST_MODE env silently excluded ALL questions of test runs and
the gate reported 0 eligible (observed 2026-07-09).
"""

from __future__ import annotations

import pytest

pytest.importorskip("duckdb")

from tests.sibyl_test_utils import HS_RUN_ID, Q1, Q2, seed_db


def _mark_run_as_test(db_url: str) -> None:
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        con.execute(
            "UPDATE hs_runs SET is_test = TRUE WHERE hs_run_id = ?", [HS_RUN_ID]
        )
        con.execute(
            "UPDATE questions SET is_test = TRUE WHERE hs_run_id = ?", [HS_RUN_ID]
        )
    finally:
        con.close()


def test_select_top_questions_orders_by_volatility(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)
    from sibyl.select_questions import select_top_questions

    qs = select_top_questions(HS_RUN_ID)
    assert [q.question_id for q in qs] == [Q1, Q2]


def test_test_mode_run_questions_still_selected_without_env(tmp_path, monkeypatch):
    """Regression: is_test questions of a test HS run must not be filtered out."""
    db_url = seed_db(tmp_path, monkeypatch)
    _mark_run_as_test(db_url)
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)

    from sibyl.select_questions import hs_run_is_test, select_top_questions

    assert hs_run_is_test(HS_RUN_ID) is True
    qs = select_top_questions(HS_RUN_ID)
    assert [q.question_id for q in qs] == [Q1, Q2]


def test_non_test_run_still_excludes_stray_test_questions(tmp_path, monkeypatch):
    db_url = seed_db(tmp_path, monkeypatch)
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        con.execute("UPDATE questions SET is_test = TRUE WHERE question_id = ?", [Q2])
    finally:
        con.close()
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)

    from sibyl.select_questions import select_top_questions

    qs = select_top_questions(HS_RUN_ID)
    assert [q.question_id for q in qs] == [Q1]


def test_run_sibyl_enables_test_mode_for_test_runs(tmp_path, monkeypatch):
    """run_sibyl must export PYTHIA_TEST_MODE when the target run is a test run."""
    db_url = seed_db(tmp_path, monkeypatch)
    _mark_run_as_test(db_url)
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
    # Zero-question budget so run_sibyl resolves the run, sets the env, and
    # exits without any model/network calls.
    from sibyl.run import run_sibyl

    import os

    run_sibyl(HS_RUN_ID, n_questions=0)
    assert os.environ.get("PYTHIA_TEST_MODE") == "1"


def test_eligibility_breakdown_reports_pairs(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)
    from sibyl.select_questions import eligibility_breakdown

    info = eligibility_breakdown(HS_RUN_ID)
    assert info["hs_run_id"] == HS_RUN_ID
    assert info["run_is_test"] is False
    assert info["pairs"]["ACE/FATALITIES"]["active"] == 2
    assert info["pairs"]["ACE/FATALITIES"]["pair_eligible"] is True
