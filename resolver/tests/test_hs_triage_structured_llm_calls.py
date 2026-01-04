import duckdb

from resolver.query.debug_ui import get_hs_triage_all


def _seed_tables(conn):
    conn.execute(
        """
        CREATE TABLE hs_triage (
            run_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            triage_score DOUBLE,
            tier TEXT,
            created_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE llm_calls (
            hs_run_id TEXT,
            phase TEXT,
            iso3 TEXT,
            created_at TIMESTAMP,
            response_text TEXT,
            parse_error TEXT,
            status TEXT,
            error_type TEXT,
            hazard_scores_json TEXT,
            hazard_scores_parse_ok BOOLEAN
        )
        """
    )

    conn.execute(
        """
        INSERT INTO hs_triage VALUES
          ('hs_20240303', 'USA', 'FL', 0.2, 'watch', '2024-03-03 09:00:00'),
          ('hs_20240303', 'BRA', 'FL', 0.4, 'watch', '2024-03-03 09:10:00')
        """
    )

    conn.execute(
        """
        INSERT INTO llm_calls VALUES
          ('hs_20240303', 'hs_triage', 'USA', '2024-03-03 09:30:00',
           'not json', NULL, 'ok', NULL, '{"FL":0.8}', TRUE),
          ('hs_20240303', 'hs_triage', 'BRA', '2024-03-03 09:35:00',
           'timeout', NULL, 'error', 'timeout', NULL, FALSE)
        """
    )


def test_structured_llm_calls_preferred():
    conn = duckdb.connect(":memory:")
    _seed_tables(conn)

    rows, _ = get_hs_triage_all(conn, "hs_20240303")

    by_key = {(row["iso3"], row["hazard_code"]): row for row in rows}

    usa = by_key[("USA", "FL")]
    assert usa["triage_score_1"] == 0.8
    assert usa["triage_score_2"] is None
    assert usa["call_1_status"] == "ok"
    assert usa["call_2_status"] == "no_call"

    bra = by_key[("BRA", "FL")]
    assert bra["triage_score_1"] is None
    assert bra["triage_score_2"] is None
    assert bra["call_1_status"] == "error:timeout"
    assert bra["call_2_status"] == "no_call"
    assert bra["why_null"] == "call_failures:error:timeout,no_call"
