import json

import pytest


def test_hs_hazard_tail_packs_schema_and_writer(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")

    from horizon_scanner.db_writer import log_hs_hazard_tail_packs_to_db
    from pythia.db.schema import ensure_schema
    from resolver.db import duckdb_io

    db_path = tmp_path / "tailpacks.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    ensure_schema()

    conn = duckdb_io.get_db(str(db_path))
    try:
        columns = conn.execute("PRAGMA table_info('hs_hazard_tail_packs')").fetchall()
    finally:
        duckdb_io.close_db(conn)

    assert columns

    pack = {
        "iso3": "usa",
        "hazard_code": "fl",
        "rc_level": 2,
        "rc_score": 0.72,
        "rc_direction": "up",
        "rc_window": "month_1-2",
        "query": "flood displacement early signals",
        "markdown": "## Flood signals\n- River levels rising",
        "sources": [{"url": "https://example.com/source"}],
        "grounded": True,
        "grounding_debug": {"check": "ok"},
        "structural_context": "river basin exposure",
        "recent_signals": ["signal-1", "signal-2"],
    }

    log_hs_hazard_tail_packs_to_db("hs_run_1", [pack])

    conn = duckdb_io.get_db(str(db_path))
    try:
        row = conn.execute(
            """
            SELECT iso3, hazard_code, sources_json, recent_signals_json
            FROM hs_hazard_tail_packs
            WHERE hs_run_id = ?
            """,
            ["hs_run_1"],
        ).fetchone()
    finally:
        duckdb_io.close_db(conn)

    assert row is not None
    iso3, hazard_code, sources_json, recent_signals_json = row
    assert iso3 == "USA"
    assert hazard_code == "FL"
    assert json.loads(sources_json)[0]["url"] == "https://example.com/source"
    assert "signal-1" in json.loads(recent_signals_json)


def _write_pack(tmp_path, monkeypatch, test_mode, **kwargs):
    """Write one grounding pack and return its stored is_test value."""
    pytest.importorskip("duckdb")
    from horizon_scanner.db_writer import log_hs_hazard_tail_packs_to_db
    from pythia.db.schema import ensure_schema
    from resolver.db import duckdb_io

    db_path = tmp_path / "istest.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")
    if test_mode is None:
        monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
    else:
        monkeypatch.setenv("PYTHIA_TEST_MODE", test_mode)
    ensure_schema()

    pack = {"iso3": "SOM", "hazard_code": "ACE", "query": "q", "markdown": "m", "sources": []}
    log_hs_hazard_tail_packs_to_db("hs_run_istest", [pack], **kwargs)

    conn = duckdb_io.get_db(str(db_path))
    try:
        return conn.execute(
            "SELECT is_test FROM hs_hazard_tail_packs WHERE hs_run_id = 'hs_run_istest'"
        ).fetchone()[0]
    finally:
        duckdb_io.close_db(conn)


def test_grounding_packs_inherit_test_mode_when_caller_omits_flag(tmp_path, monkeypatch):
    """The live bug: neither grounding caller passes is_test.

    With `is_test: bool = False` every test run's packs were written as
    production rows — stage_health caught 10 such rows on 2026-07-29 while
    every other stamped table was correctly TRUE.
    """
    assert _write_pack(tmp_path, monkeypatch, "1") is True


def test_grounding_packs_are_production_when_mode_is_off(tmp_path, monkeypatch):
    assert _write_pack(tmp_path, monkeypatch, None) is False


def test_explicit_is_test_argument_still_wins(tmp_path, monkeypatch):
    """Inheriting must not override a caller that stated its intent."""
    assert _write_pack(tmp_path, monkeypatch, "1", is_test=False) is False
