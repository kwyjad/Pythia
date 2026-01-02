import duckdb

from resolver.query.debug_ui import (
    get_country_run_summary,
    get_hs_triage_llm_calls,
    get_hs_triage_rows,
    list_hs_runs,
)


def _seed_tables(conn):
    conn.execute(
        """
        CREATE TABLE hs_triage (
            run_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            tier TEXT,
            triage_score DOUBLE,
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
            provider TEXT,
            created_at TIMESTAMP,
            response_text TEXT,
            parse_error TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE questions (
            question_id INTEGER,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            hs_run_id TEXT,
            status TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE forecasts_ensemble (
            question_id INTEGER,
            created_at TIMESTAMP
        )
        """
    )

    conn.execute(
        """
        INSERT INTO hs_triage VALUES
          ('hs_20240101', 'UKR', 'TC', 'quiet', 0.0, '2024-01-01 12:00:00'),
          ('hs_20240101', 'UKR', 'FL', 'watch', 0.5, '2024-01-01 12:01:00'),
          ('hs_20240101', 'MLI', 'FL', 'watch', 0.8, '2024-01-01 12:02:00'),
          ('hs_20240201', 'UKR', 'TC', 'watch', 0.9, '2024-02-01 09:00:00')
        """
    )

    conn.execute(
        """
        INSERT INTO llm_calls VALUES
          ('hs_20240101', 'hs_triage', 'UKR', 'TC', 'gpt-4', 'openai',
           '2024-01-01 12:00:00', 'response-ukr-tc', 'parse_failed'),
          ('hs_20240101', 'hs_triage', 'MLI', 'FL', 'gpt-4', 'openai',
           '2024-01-01 12:05:00', 'response-mli-fl', NULL)
        """
    )

    conn.execute(
        """
        INSERT INTO questions VALUES
          (1, 'UKR', 'TC', 'PA', 'hs_20240101', 'active'),
          (2, 'UKR', 'FL', 'PA', 'hs_20240101', 'active'),
          (3, 'MLI', 'FL', 'PA', 'hs_20240101', 'active')
        """
    )

    conn.execute(
        """
        INSERT INTO forecasts_ensemble VALUES
          (1, '2024-01-02 00:00:00'),
          (3, '2024-01-02 00:00:00')
        """
    )


def test_list_hs_runs_order_and_counts():
    conn = duckdb.connect(":memory:")
    _seed_tables(conn)

    rows = list_hs_runs(conn)

    assert rows[0]["run_id"] == "hs_20240201"
    assert rows[0]["countries_triaged"] == 1
    assert rows[1]["run_id"] == "hs_20240101"
    assert rows[1]["countries_triaged"] == 2


def test_get_hs_triage_rows_filters():
    conn = duckdb.connect(":memory:")
    _seed_tables(conn)

    rows = get_hs_triage_rows(conn, "hs_20240101", iso3="UKR", hazard_code="TC")

    assert len(rows) == 1
    assert rows[0]["iso3"] == "UKR"
    assert rows[0]["hazard_code"] == "TC"
    assert rows[0]["triage_score"] == 0.0


def test_get_hs_triage_llm_calls_preview_and_parse_error():
    conn = duckdb.connect(":memory:")
    _seed_tables(conn)

    rows = get_hs_triage_llm_calls(conn, "hs_20240101", preview_chars=8)

    assert rows
    assert rows[0]["response_preview"] == "response"
    assert any(row["parse_error"] for row in rows)


def test_country_run_summary_counts():
    conn = duckdb.connect(":memory:")
    _seed_tables(conn)

    summary = get_country_run_summary(conn, "hs_20240101", "UKR")

    assert summary["hazards_triaged"] == 2
    assert summary["questions_generated"] == 2
    assert summary["questions_forecasted"] == 1
    assert summary["notes"] == []
    assert "diagnostics" in summary
    assert summary["diagnostics"].get("forecasts_source") in {
        "fallback_join_via_questions",
        "forecasts_table_direct",
    }
    assert summary["diagnostics"]["forecasts_source"] == "fallback_join_via_questions"
    assert summary["diagnostics"]["forecasts_run_id_missing_fallback"] is True
