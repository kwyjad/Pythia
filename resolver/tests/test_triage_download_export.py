import duckdb

from resolver.query.downloads import build_triage_export


def test_build_triage_export_collapses_hazards():
    con = duckdb.connect()
    con.execute(
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
    con.execute(
        """
        CREATE TABLE llm_calls (
            hs_run_id TEXT,
            phase TEXT,
            model_id TEXT,
            model_name TEXT
        )
        """
    )

    con.execute(
        """
        INSERT INTO hs_triage VALUES
            ('hs_20240106T010203', 'KEN', 'DR', 'quiet', 0.4, '2024-01-05'),
            ('hs_20240106T010203', 'KEN', 'ACE', 'priority', 0.8, '2024-01-06')
        """
    )
    con.execute(
        """
        INSERT INTO llm_calls VALUES
            ('hs_20240106T010203', 'hs_triage', 'gemini-3-flash', NULL)
        """
    )

    df = build_triage_export(con)

    expected_columns = [
        "Triage Year",
        "Triage Month",
        "Triage Date",
        "Run ID",
        "Triage model",
        "ISO3",
        "Country",
        "Triage Score",
        "Triage Tier",
    ]
    assert list(df.columns) == expected_columns
    assert len(df) == 1

    row = df.iloc[0]
    assert row["Run ID"] == "hs_20240106T010203"
    assert row["ISO3"] == "KEN"
    assert row["Triage Score"] == 0.8
    assert row["Triage Tier"] == "priority"
    assert row["Triage Date"] == "2024-01-06"
    assert row["Triage Year"] == 2024
    assert row["Triage Month"] == 1
    assert row["Triage model"] == "gemini-3-flash"
    assert row["Country"] in {"Kenya", "KEN"}
