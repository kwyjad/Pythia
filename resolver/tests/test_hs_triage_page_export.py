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
            triage_score DOUBLE
        )
        """
    )

    conn.execute(
        """
        INSERT INTO hs_triage VALUES
          ('hs_20240101', 'UKR', 'TC', NULL, 'watch', '2024-01-01 12:00:00'),
          ('hs_20240101', 'MLI', 'FL', NULL, 'quiet', '2024-01-01 12:01:00'),
          ('hs_20240101', 'CHN', 'TC', 0.15, 'watch', '2024-01-01 12:02:00')
        """
    )

    conn.execute(
        """
        INSERT INTO llm_calls VALUES
          ('hs_20240101', 'hs_triage', 'UKR', 'tc', 'gemini',
           '2024-01-01 12:10:00', 'no score here', NULL),
          ('hs_20240101', 'hs_triage', 'UKR', 'TC', 'gemini',
           '2024-01-01 12:09:00', 'triage_score=0.40', NULL),
          ('hs_20240101', 'hs_triage', 'MLI', 'FL', 'gemini',
           '2024-01-01 12:08:00', 'triage_score: 0.20', NULL),
          ('hs_20240101', 'hs_triage', 'MLI', 'FL', 'gemini',
           '2024-01-01 12:11:00', 'ignored', 0.60),
          ('hs_20240101', 'hs_triage', 'CHN', 'TC', 'gemini',
           '2024-01-01 12:12:00', 'no triage score', NULL),
          ('hs_20240101', 'hs_triage', 'CHN', 'TC', 'gemini',
           '2024-01-01 12:06:00', 'still nothing', NULL)
        """
    )


def test_get_hs_triage_all_scores_and_nulls():
    conn = duckdb.connect(":memory:")
    _seed_tables(conn)

    rows, diagnostics = get_hs_triage_all(conn, "hs_20240101")

    assert diagnostics["total_calls"] == 6
    assert diagnostics["parsed_scores"] == 3
    assert diagnostics["null_scores"] == 3

    by_key = {(row["iso3"], row["hazard_code"]): row for row in rows}

    ukr = by_key[("UKR", "TC")]
    assert ukr["hazard_code"] == "TC"
    assert ukr["triage_score_1"] is None
    assert ukr["triage_score_2"] == 0.4
    assert ukr["triage_score_avg"] == 0.4

    mli = by_key[("MLI", "FL")]
    assert mli["hazard_code"] == "FL"
    assert mli["triage_score_1"] == 0.6
    assert mli["triage_score_2"] == 0.2
    assert mli["triage_score_avg"] == 0.4

    chn = by_key[("CHN", "TC")]
    assert chn["hazard_code"] == "TC"
    assert chn["triage_score_1"] is None
    assert chn["triage_score_2"] is None
    assert chn["triage_score_avg"] == 0.15
