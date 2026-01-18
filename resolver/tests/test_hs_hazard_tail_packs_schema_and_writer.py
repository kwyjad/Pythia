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
