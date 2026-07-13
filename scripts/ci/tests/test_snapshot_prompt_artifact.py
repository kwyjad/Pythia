# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Regression tests for the LLM prompt-artifact generator.

Guards the bug where ``_load_question_for_hazard`` ordered by a non-existent
``questions.created_at`` column: the resulting DuckDB BinderException was
swallowed by a bare ``except`` and returned ``None`` for every hazard, so the
SPD and Scenario sections of the ``pythia-llm-prompts`` artifact were blank
for the entire run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from resolver.db._duckdb_available import DUCKDB_AVAILABLE

pytestmark = pytest.mark.skipif(not DUCKDB_AVAILABLE, reason="duckdb not installed")


# The real ``questions`` schema (see pythia/db/schema.py) — deliberately has NO
# ``created_at`` column. Keep this list in sync with the columns the loader
# selects so a future schema change surfaces here rather than silently blanking
# the artifact.
_QUESTIONS_COLUMNS = (
    "question_id TEXT, hs_run_id TEXT, scenario_ids_json TEXT, iso3 TEXT, "
    "hazard_code TEXT, metric TEXT, target_month TEXT, window_start_date DATE, "
    "window_end_date DATE, wording TEXT, status TEXT, pythia_metadata_json TEXT, "
    "track INTEGER, is_test BOOLEAN"
)


def _make_questions_db(db_path: Path) -> None:
    import duckdb

    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"CREATE TABLE questions ({_QUESTIONS_COLUMNS})")
        # Two epochs of the same hazard/country; the newer window must win.
        con.execute(
            """
            INSERT INTO questions
                (question_id, iso3, hazard_code, metric, window_start_date,
                 target_month, status, track)
            VALUES
                ('SOM_ACE_FATALITIES_2026-07', 'SOM', 'ACE', 'FATALITIES',
                 DATE '2026-07-01', '2026-12', 'active', 2),
                ('SOM_ACE_FATALITIES_2026-08', 'SOM', 'ACE', 'FATALITIES',
                 DATE '2026-08-01', '2027-01', 'active', 2),
                ('SOM_ACE_FATALITIES_2026-06', 'SOM', 'ACE', 'FATALITIES',
                 DATE '2026-06-01', '2026-11', 'retired', 2)
            """
        )
    finally:
        con.close()


def test_load_question_for_hazard_returns_active_question(tmp_path: Path) -> None:
    """The loader must return the active question, not None.

    Reproduces the ``created_at`` regression: with the buggy ORDER BY this
    raised BinderException → None; with the fix it returns the newest active
    epoch.
    """
    import duckdb

    from scripts.ci.snapshot_prompt_artifact import _load_question_for_hazard

    db_path = tmp_path / "q.duckdb"
    _make_questions_db(db_path)

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        result = _load_question_for_hazard(con, "SOM", "ACE")
    finally:
        con.close()

    assert result is not None, "loader returned None — SPD/Scenario would be blank"
    # Newest active epoch wins; the 'retired' row is excluded.
    assert result["question_id"] == "SOM_ACE_FATALITIES_2026-08"
    assert result["metric"] == "FATALITIES"


def test_load_question_for_hazard_none_when_absent(tmp_path: Path) -> None:
    """No active question for the pair → None (quiet/blocked hazards)."""
    import duckdb

    from scripts.ci.snapshot_prompt_artifact import _load_question_for_hazard

    db_path = tmp_path / "q.duckdb"
    _make_questions_db(db_path)

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        result = _load_question_for_hazard(con, "SOM", "FL")
    finally:
        con.close()

    assert result is None


def test_connect_reopens_after_shared_connection_closed(tmp_path: Path) -> None:
    """Guards the structured-data connection fix.

    ``_load_structured_data``'s sub-loaders (ipc_phases / acaps) call
    ``duckdb_io.get_db()`` (a cache HIT returns the artifact's own shared
    connection) then ``close_db()`` it, which closes AND evicts the shared
    handle. The artifact reopens via ``_connect(db_url)`` afterwards; this test
    asserts that reopen returns a usable connection even after the cached one
    was closed+evicted (mirroring the loader behaviour).
    """
    from resolver.db import duckdb_io
    from scripts.ci.snapshot_prompt_artifact import _connect

    db_path = tmp_path / "reopen.duckdb"
    url = f"duckdb:///{db_path}"

    # Seed a table via a throwaway connection so the file exists with content.
    seed = duckdb_io.get_db(url)
    seed.execute("CREATE TABLE t AS SELECT 1 AS x")
    duckdb_io.close_db(seed)

    con = _connect(url)
    assert con.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1

    # Simulate what a structured-data sub-loader does to the SHARED connection.
    duckdb_io.close_db(con)

    # The artifact's reopen must yield a working connection.
    con2 = _connect(url)
    assert con2.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    duckdb_io.close_db(con2)
