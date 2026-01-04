import duckdb

from resolver.query.debug_ui import _extract_hazard_scores, get_hs_triage_all


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
            hazard_code TEXT,
            model_id TEXT,
            created_at TIMESTAMP,
            response_text TEXT,
            triage_score DOUBLE,
            parse_error TEXT
        )
        """
    )

    conn.execute(
        """
        INSERT INTO hs_triage VALUES
          ('hs_20240101', 'UKR', 'ACE', 0.91, 'priority', '2024-01-01 12:00:00'),
          ('hs_20240101', 'UKR', 'FL', 0.11, 'quiet', '2024-01-01 12:01:00'),
          ('hs_20240101', 'CHN', 'TC', 0.15, 'watch', '2024-01-01 12:02:00')
        """
    )

    conn.execute(
        """
        INSERT INTO llm_calls VALUES
          ('hs_20240101', 'hs_triage', 'UKR', NULL, 'gemini',
           '2024-01-01 12:10:00',
           '```json {"hazards":{"ACE":{"triage_score":0.9},"FL":{"triage_score":"20%"}}} ```',
           NULL, NULL),
          ('hs_20240101', 'hs_triage', 'UKR', NULL, 'gemini',
           '2024-01-01 12:09:00',
           'Here is the JSON: {"results":[{"code":"ACE","score":0.7},{"code":"FL","score":"30%"}]} End.',
           NULL, NULL),
          ('hs_20240101', 'hs_triage', 'CHN', NULL, 'gemini',
           '2024-01-01 12:12:00', '{bad json', NULL, 'parse failed'),
          ('hs_20240101', 'hs_triage', 'CHN', NULL, 'gemini',
           '2024-01-01 12:06:00', 'n/a', NULL, 'parse failed')
        """
    )


def test_get_hs_triage_all_scores_and_nulls():
    conn = duckdb.connect(":memory:")
    _seed_tables(conn)

    rows, diagnostics = get_hs_triage_all(conn, "hs_20240101")

    assert diagnostics["total_calls"] == 4
    assert diagnostics["parsed_scores"] == 4
    assert diagnostics["null_scores"] == 2
    assert diagnostics["calls_grouped_by_iso3"] == 2
    assert diagnostics["countries_with_two_calls"] == 2
    assert diagnostics["countries_with_one_call"] == 0
    assert diagnostics["hazard_scores_extracted"] == 4
    assert diagnostics["hazard_scores_missing"] == 2
    assert diagnostics["score_avg_from_calls"] == 2
    assert diagnostics["score_avg_from_hs_triage"] == 1

    by_key = {(row["iso3"], row["hazard_code"]): row for row in rows}

    ukr_ace = by_key[("UKR", "ACE")]
    assert ukr_ace["hazard_code"] == "ACE"
    assert ukr_ace["hazard_label"] == "Armed Conflict"
    assert ukr_ace["triage_score_1"] == 0.9
    assert ukr_ace["triage_score_2"] == 0.7
    assert ukr_ace["triage_score_avg"] == 0.8
    assert ukr_ace["call_1_status"] == "ok"
    assert ukr_ace["call_2_status"] == "ok"
    assert ukr_ace["why_null"] == ""

    ukr_fl = by_key[("UKR", "FL")]
    assert ukr_fl["hazard_code"] == "FL"
    assert ukr_fl["hazard_label"] == "Flood"
    assert ukr_fl["triage_score_1"] == 0.2
    assert ukr_fl["triage_score_2"] == 0.3
    assert ukr_fl["triage_score_avg"] == 0.25

    chn = by_key[("CHN", "TC")]
    assert chn["hazard_code"] == "TC"
    assert chn["hazard_label"] == "Tropical Cyclone"
    assert chn["triage_score_1"] is None
    assert chn["triage_score_2"] is None
    assert chn["triage_score_avg"] == 0.15
    assert chn["call_1_status"] == "parse_error"
    assert chn["call_2_status"] == "parse_error"
    assert chn["why_null"] == "call_failures:parse_error,parse_error"


def test_extract_hazard_scores_variants():
    fenced = '```json {"hazards":{"ACE":{"triage_score":0.95},"FL":{"triage_score":"85%"}}} ```'
    assert _extract_hazard_scores(fenced) == {"ACE": 0.95, "FL": 0.85}

    hazards_list = '{"hazards":[{"hazard_code":"ACE","triage_score":0.7},{"hazard_code":"FL","score":0.2}]}'
    assert _extract_hazard_scores(hazards_list) == {"ACE": 0.7, "FL": 0.2}

    embedded = 'Here is the JSON: {"results":[{"code":"ACE","score":"0.4"}]} End.'
    assert _extract_hazard_scores(embedded) == {"ACE": 0.4}


def test_get_hs_triage_all_call_statuses():
    conn = duckdb.connect(":memory:")
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
            hazard_code TEXT,
            model_id TEXT,
            created_at TIMESTAMP,
            response_text TEXT,
            triage_score DOUBLE,
            parse_error TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO hs_triage VALUES
          ('hs_20240202', 'COL', 'FL', 0.3, 'watch', '2024-02-02 10:00:00'),
          ('hs_20240202', 'COL', 'TC', 0.4, 'watch', '2024-02-02 10:01:00'),
          ('hs_20240202', 'PER', 'FL', 0.2, 'watch', '2024-02-02 10:02:00'),
          ('hs_20240202', 'ECU', 'FL', 0.5, 'watch', '2024-02-02 10:03:00')
        """
    )
    conn.execute(
        """
        INSERT INTO llm_calls VALUES
          ('hs_20240202', 'hs_triage', 'COL', NULL, 'gemini',
           '2024-02-02 11:00:00',
           '```json {"hazards":[{"hazard_code":"FL","triage_score":0.6}]} ```',
           NULL, NULL),
          ('hs_20240202', 'hs_triage', 'PER', NULL, 'gemini',
           '2024-02-02 11:05:00',
           '```json {"hazards":[{"hazard_code":"FL","triage_score":"oops"}]} ```',
           NULL, NULL),
          ('hs_20240202', 'hs_triage', 'ECU', NULL, 'gemini',
           '2024-02-02 11:06:00',
           '{"hazards":[{"hazard_code":"FL","triage_score":0.7}]}',
           NULL, 'parse failed')
        """
    )

    rows, diagnostics = get_hs_triage_all(conn, "hs_20240202")
    by_key = {(row["iso3"], row["hazard_code"]): row for row in rows}

    col_tc = by_key[("COL", "TC")]
    assert col_tc["call_1_status"] == "ok"
    assert col_tc["call_2_status"] == "no_call"
    assert col_tc["why_null"] == "hazard_missing_in_call1"

    per_fl = by_key[("PER", "FL")]
    assert per_fl["call_1_status"] == "parse_failed"
    assert per_fl["call_2_status"] == "no_call"
    assert per_fl["why_null"] == "invalid_score_value"

    ecu_fl = by_key[("ECU", "FL")]
    assert ecu_fl["call_1_status"] == "parse_error"
    assert ecu_fl["call_2_status"] == "no_call"
    assert ecu_fl["why_null"] == "call_failures:parse_error,no_call"
    assert diagnostics["rows_with_invalid_score_value"] == 1
