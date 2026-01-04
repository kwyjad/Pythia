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
           '{"hazards":[{"hazard_code":"ACE","triage_score":0.9},{"hazard_code":"FL","triage_score":0.2}]}',
           NULL, NULL),
          ('hs_20240101', 'hs_triage', 'UKR', NULL, 'gemini',
           '2024-01-01 12:09:00',
           '{"results":[{"code":"ACE","score":0.7},{"code":"FL","score":0.3}]}',
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
