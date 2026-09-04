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

    ``_load_structured_data``'s sub-loaders (food_security / acaps) call
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


# ---------------------------------------------------------------------------
# Hazard-loop connection survival
#
# test_connect_reopens_after_shared_connection_closed (above) proves the reopen
# MECHANISM works — which is exactly why the 2026-08-01 regression survived it.
# The bug was in the loop invariant: the artifact reopened once, inside the SPD
# block, and then carried that assumption into the next hazard. ACE rendered in
# full and DR, FL and TC each reported "No HS triage data found" against a
# database holding 122 triage rows for every one of them.
# ---------------------------------------------------------------------------

_HS_TRIAGE_COLUMNS = (
    "run_id TEXT, iso3 TEXT, hazard_code TEXT, tier TEXT, triage_score DOUBLE, "
    "need_full_spd BOOLEAN, drivers_json TEXT, regime_shifts_json TEXT, "
    "data_quality_json TEXT, scenario_stub TEXT, "
    "regime_change_likelihood DOUBLE, regime_change_magnitude DOUBLE, "
    "regime_change_score DOUBLE, regime_change_level INTEGER, "
    "regime_change_direction TEXT, regime_change_window TEXT, "
    "regime_change_json TEXT, track INTEGER, created_at TIMESTAMP, is_test BOOLEAN"
)

_ARTIFACT_HAZARDS = ("ACE", "DR", "FL", "TC")


def _make_triage_db(db_path: Path) -> None:
    import duckdb

    con = duckdb.connect(str(db_path))
    try:
        con.execute(f"CREATE TABLE hs_triage ({_HS_TRIAGE_COLUMNS})")
        # The questions table is required for the SPD and Scenario blocks to
        # run at all — without it the loop short-circuits past exactly the code
        # whose connection handling is under test.
        con.execute(f"CREATE TABLE questions ({_QUESTIONS_COLUMNS})")
        for hz in _ARTIFACT_HAZARDS:
            con.execute(
                """
                INSERT INTO hs_triage
                    (run_id, iso3, hazard_code, tier, triage_score,
                     regime_change_level, created_at, is_test)
                VALUES ('hs_test', 'SOM', ?, 'priority', 0.6, 1,
                        TIMESTAMP '2026-08-01 04:02:47', FALSE)
                """,
                [hz],
            )
            con.execute(
                """
                INSERT INTO questions
                    (question_id, hs_run_id, iso3, hazard_code, metric,
                     window_start_date, target_month, status, track)
                VALUES (?, 'hs_test', 'SOM', ?, 'PA',
                        DATE '2026-09-01', '2027-02', 'active', 1)
                """,
                [f"SOM_{hz}_PA_2026-09", hz],
            )
    finally:
        con.close()


def test_hw_is_not_in_the_artifact_hazard_list() -> None:
    """HW is in BLOCKED_HAZARDS, so it could only ever render a skip line."""
    from scripts.ci.snapshot_prompt_artifact import ACTIVE_HAZARDS

    assert "HW" not in ACTIVE_HAZARDS
    assert list(ACTIVE_HAZARDS) == list(_ARTIFACT_HAZARDS)


def test_ensure_live_keeps_a_healthy_connection(tmp_path: Path) -> None:
    from resolver.db import duckdb_io
    from scripts.ci.snapshot_prompt_artifact import _connect, _ensure_live

    url = f"duckdb:///{tmp_path / 'live.duckdb'}"
    con = _connect(url)
    try:
        assert _ensure_live(con, url) is con
    finally:
        duckdb_io.close_db(con)


def test_ensure_live_reopens_a_closed_connection(tmp_path: Path) -> None:
    from resolver.db import duckdb_io
    from scripts.ci.snapshot_prompt_artifact import _connect, _ensure_live

    url = f"duckdb:///{tmp_path / 'dead.duckdb'}"
    con = _connect(url)
    con.execute("CREATE TABLE t AS SELECT 1 AS x")
    duckdb_io.close_db(con)  # what a structured-data sub-loader does

    revived = _ensure_live(con, url)
    try:
        assert revived is not con
        assert revived.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1
    finally:
        duckdb_io.close_db(revived)


def test_every_hazard_renders_when_structured_data_closes_the_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2026-08-01 regression, reproduced end-to-end.

    The stub mirrors what ``_load_structured_data``'s sub-loaders really do:
    close AND evict the shared ``duckdb_io`` handle. Before the fix only the
    first hazard survived that.
    """
    from resolver.db import duckdb_io
    from scripts.ci import snapshot_prompt_artifact as spa

    db_path = tmp_path / "artifact.duckdb"
    _make_triage_db(db_path)
    url = f"duckdb:///{db_path}"

    closed: list[str] = []

    def _closing_stub(iso3, hazard_code, run_id, rc_level):
        closed.append(hazard_code)
        duckdb_io.close_db(duckdb_io.get_db(url))
        return None

    monkeypatch.setattr(spa, "_load_structured_data_for_artifact", _closing_stub)

    md = spa.build_artifact(url)

    for hz in _ARTIFACT_HAZARDS:
        assert f"# Hazard: {hz}" in md, f"{hz} section missing entirely"
        assert f"No HS triage data found for {hz}" not in md, (
            f"{hz} was skipped as 'no data' despite having triage rows — "
            "the shared connection did not survive the previous hazard"
        )
    assert "# Hazard: HW" not in md


def test_hazard_loop_survives_a_connection_death_after_the_spd_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loop invariant, isolated.

    The pre-fix code reopened exactly once, inside the SPD block. That heals a
    connection closed BY the structured-data load — which is why the existing
    reopen test passed while production was broken. What CI actually showed was
    a connection that was fine through the ACE SPD render and dead by the DR
    lookup, i.e. something AFTER the reopen killed it (the forecaster loaders
    leave pooled handles open on the same path). Only re-acquiring at the top
    of every iteration survives that, so this stub kills the connection in the
    scenario block — after the SPD reopen, before the next hazard.
    """
    from resolver.db import duckdb_io
    from scripts.ci import snapshot_prompt_artifact as spa

    db_path = tmp_path / "late_death.duckdb"
    _make_triage_db(db_path)
    url = f"duckdb:///{db_path}"

    monkeypatch.setattr(
        spa, "_load_structured_data_for_artifact", lambda *a, **k: None
    )

    real_scenario = spa._render_scenario_prompt

    def _killing_scenario(*args, **kwargs):
        duckdb_io.close_db(duckdb_io.get_db(url))
        return real_scenario(*args, **kwargs)

    monkeypatch.setattr(spa, "_render_scenario_prompt", _killing_scenario)

    md = spa.build_artifact(url)

    for hz in _ARTIFACT_HAZARDS:
        assert f"No HS triage data found for {hz}" not in md, (
            f"{hz} was skipped as 'no data' after the connection died in a "
            "previous hazard's scenario block"
        )
