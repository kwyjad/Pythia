import json

import pytest


def test_llm_calls_telemetry_migration_backfill(tmp_path):
    pytest.importorskip("duckdb")
    from resolver.db import duckdb_io
    from scripts.migrate_llm_calls_telemetry import backfill_llm_calls_telemetry

    db_path = tmp_path / "llm_calls_migration.duckdb"
    db_url = f"duckdb:///{db_path}"
    conn = duckdb_io.get_db(db_url)
    try:
        conn.execute(
            """
            CREATE TABLE llm_calls (
                call_id TEXT,
                phase TEXT,
                response_text TEXT,
                error_text TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO llm_calls VALUES
              ('call_ok', 'hs_triage', '{"hazards":{"ACE":{"triage_score":0.6}}}', NULL),
              ('call_timeout', 'hs_triage', NULL, 'timeout after 60s')
            """
        )
    finally:
        duckdb_io.close_db(conn)

    summary = backfill_llm_calls_telemetry(db_url)
    assert summary["status_updated"] == 2
    assert summary["hazard_scores_filled"] == 1

    conn = duckdb_io.get_db(db_url)
    try:
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info('llm_calls')").fetchall()
        }
        assert "status" in columns
        assert "error_type" in columns
        assert "error_message" in columns
        assert "response_format" in columns
        assert "hazard_scores_json" in columns
        assert "hazard_scores_parse_ok" in columns

        ok_row = conn.execute(
            """
            SELECT status, error_type, response_format, hazard_scores_json, hazard_scores_parse_ok
            FROM llm_calls WHERE call_id = 'call_ok'
            """
        ).fetchone()
        assert ok_row[0] == "ok"
        assert ok_row[1] is None
        assert ok_row[2] == "json"
        hazard_scores = json.loads(ok_row[3])
        assert hazard_scores == {"ACE": 0.6}
        assert ok_row[4] is True

        error_row = conn.execute(
            """
            SELECT status, error_type, error_message, response_format
            FROM llm_calls WHERE call_id = 'call_timeout'
            """
        ).fetchone()
        assert error_row[0] == "error"
        assert error_row[1] == "timeout"
        assert error_row[2] == "timeout after 60s"
        assert error_row[3] == "empty"
    finally:
        duckdb_io.close_db(conn)
