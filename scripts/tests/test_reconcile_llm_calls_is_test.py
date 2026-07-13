# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for scripts/reconcile_llm_calls_is_test.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from resolver.db._duckdb_available import DUCKDB_AVAILABLE

pytestmark = pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="duckdb not installed")


def _make_db(db_path: Path):
    import duckdb

    con = duckdb.connect(str(db_path))
    con.execute(
        "CREATE TABLE forecasts_ensemble (run_id TEXT, question_id TEXT, is_test BOOLEAN)"
    )
    con.execute("CREATE TABLE hs_runs (hs_run_id TEXT, is_test BOOLEAN)")
    con.execute(
        "CREATE TABLE llm_calls (run_id TEXT, hs_run_id TEXT, phase TEXT, is_test BOOLEAN)"
    )
    # One test forecaster run + its hs run.
    con.execute("INSERT INTO forecasts_ensemble VALUES ('fc_test', 'Q', TRUE)")
    con.execute("INSERT INTO hs_runs VALUES ('hs_test', TRUE)")
    # One genuine production run.
    con.execute("INSERT INTO forecasts_ensemble VALUES ('fc_prod', 'Q2', FALSE)")
    con.execute("INSERT INTO hs_runs VALUES ('hs_prod', FALSE)")
    # llm_calls: the test run's spd/binary/scenario calls were mis-stamped FALSE;
    # its hs_triage call is TRUE; the prod run's calls are correctly FALSE.
    con.executemany(
        "INSERT INTO llm_calls VALUES (?, ?, ?, ?)",
        [
            ("fc_test", None, "spd_v2", False),      # mis-stamped -> should flip
            ("fc_test", None, "binary_v2", False),   # mis-stamped -> should flip
            (None, "hs_test", "hs_triage", True),    # already correct
            (None, "hs_test", "grounding", False),   # mis-stamped via hs -> flip
            ("fc_prod", None, "spd_v2", False),      # genuine production -> stay FALSE
        ],
    )
    con.close()


def test_reconcile_flips_only_test_run_calls(tmp_path: Path):
    import duckdb

    from scripts.reconcile_llm_calls_is_test import reconcile_llm_calls_is_test

    db_path = tmp_path / "r.duckdb"
    _make_db(db_path)

    con = duckdb.connect(str(db_path))
    try:
        result = reconcile_llm_calls_is_test(con, apply=True)
        # spd_v2 + binary_v2 (by run_id) + grounding (by hs_run_id) = 3 flipped.
        assert result["total"] == 3
        # The production spd_v2 call must remain FALSE.
        prod = con.execute(
            "SELECT is_test FROM llm_calls WHERE run_id = 'fc_prod'"
        ).fetchone()[0]
        assert prod is False
        # No non-test calls remain for the test run.
        leaked = con.execute(
            "SELECT COUNT(*) FROM llm_calls "
            "WHERE COALESCE(is_test, FALSE) = FALSE "
            "AND (run_id = 'fc_test' OR hs_run_id = 'hs_test')"
        ).fetchone()[0]
        assert leaked == 0
    finally:
        con.close()


def test_reconcile_is_idempotent_and_dry_run_matches(tmp_path: Path):
    import duckdb

    from scripts.reconcile_llm_calls_is_test import reconcile_llm_calls_is_test

    db_path = tmp_path / "r2.duckdb"
    _make_db(db_path)

    con = duckdb.connect(str(db_path))
    try:
        dry = reconcile_llm_calls_is_test(con, apply=False)
        assert dry["total"] == 3  # dry-run count is accurate (single combined query)
        applied = reconcile_llm_calls_is_test(con, apply=True)
        assert applied["total"] == 3
        again = reconcile_llm_calls_is_test(con, apply=True)
        assert again["total"] == 0  # idempotent
    finally:
        con.close()
