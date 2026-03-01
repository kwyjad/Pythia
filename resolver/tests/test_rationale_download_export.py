import duckdb
import pytest

from resolver.query.downloads import build_rationale_export


def _make_con():
    """Create an in-memory DuckDB with the tables needed for rationale export."""
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE questions (
            question_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            target_month TEXT,
            hs_run_id TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE forecasts_raw (
            question_id TEXT,
            run_id TEXT,
            model_name TEXT,
            month_index INTEGER,
            bucket_index INTEGER,
            probability DOUBLE,
            status TEXT,
            human_explanation TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE forecasts_ensemble (
            question_id TEXT,
            run_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            model_name TEXT,
            month_index INTEGER,
            bucket_index INTEGER,
            probability DOUBLE,
            status TEXT,
            human_explanation TEXT
        )
        """
    )
    return con


def _insert_raw_rows(
    con,
    *,
    question_id: str,
    run_id: str,
    model_name: str,
    human_explanation: str,
    n_months: int = 6,
    n_buckets: int = 5,
):
    """Insert duplicated rows across months/buckets (mirrors real storage)."""
    for m in range(1, n_months + 1):
        for b in range(1, n_buckets + 1):
            con.execute(
                """
                INSERT INTO forecasts_raw (
                    question_id, run_id, model_name, month_index, bucket_index,
                    probability, status, human_explanation
                ) VALUES (?, ?, ?, ?, ?, 0.2, 'ok', ?)
                """,
                [question_id, run_id, model_name, m, b, human_explanation],
            )


def _insert_ensemble_rows(
    con,
    *,
    question_id: str,
    run_id: str,
    iso3: str,
    hazard_code: str,
    metric: str,
    model_name: str,
    human_explanation: str,
    n_months: int = 6,
    n_buckets: int = 5,
):
    """Insert duplicated ensemble rows across months/buckets."""
    for m in range(1, n_months + 1):
        for b in range(1, n_buckets + 1):
            con.execute(
                """
                INSERT INTO forecasts_ensemble (
                    question_id, run_id, iso3, hazard_code, metric, model_name,
                    month_index, bucket_index, probability, status, human_explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0.2, 'ok', ?)
                """,
                [question_id, run_id, iso3, hazard_code, metric, model_name, m, b, human_explanation],
            )


def test_deduplication():
    """30 identical rows per (question, model) should collapse to 1."""
    con = _make_con()
    con.execute(
        "INSERT INTO questions VALUES ('q1', 'KEN', 'FL', 'PA', '2025-01', 'hs-1')"
    )
    _insert_raw_rows(
        con,
        question_id="q1",
        run_id="run-1",
        model_name="OpenAI-gpt-5.2",
        human_explanation="Base rate is 12k. Upward trend from flooding signals.",
    )

    df = build_rationale_export(con, hazard_code="FL")
    assert len(df) == 1
    assert df.iloc[0]["model_name"] == "OpenAI-gpt-5.2"
    assert "Base rate" in df.iloc[0]["human_explanation"]


def test_hazard_filter():
    """Only rows matching the requested hazard should be returned."""
    con = _make_con()
    con.execute(
        """
        INSERT INTO questions VALUES
            ('q-fl', 'KEN', 'FL', 'PA', '2025-01', 'hs-1'),
            ('q-dr', 'ETH', 'DR', 'PA', '2025-01', 'hs-2')
        """
    )
    _insert_raw_rows(
        con, question_id="q-fl", run_id="r1", model_name="OpenAI",
        human_explanation="Flood rationale.",
    )
    _insert_raw_rows(
        con, question_id="q-dr", run_id="r2", model_name="OpenAI",
        human_explanation="Drought rationale.",
    )

    df = build_rationale_export(con, hazard_code="FL")
    assert len(df) == 1
    assert df.iloc[0]["hazard_code"] == "FL"

    df_dr = build_rationale_export(con, hazard_code="DR")
    assert len(df_dr) == 1
    assert df_dr.iloc[0]["hazard_code"] == "DR"


def test_model_filter():
    """Optional model filter should restrict to matching model names."""
    con = _make_con()
    con.execute(
        "INSERT INTO questions VALUES ('q1', 'KEN', 'FL', 'PA', '2025-01', 'hs-1')"
    )
    _insert_raw_rows(
        con, question_id="q1", run_id="r1", model_name="OpenAI-gpt-5.2",
        human_explanation="OpenAI rationale.",
    )
    _insert_raw_rows(
        con, question_id="q1", run_id="r1", model_name="Claude-sonnet-4.5",
        human_explanation="Claude rationale.",
    )

    df_all = build_rationale_export(con, hazard_code="FL")
    assert len(df_all) == 2

    df_openai = build_rationale_export(con, hazard_code="FL", model_name="OpenAI")
    assert len(df_openai) == 1
    assert "OpenAI" in df_openai.iloc[0]["model_name"]

    df_claude = build_rationale_export(con, hazard_code="FL", model_name="Claude")
    assert len(df_claude) == 1
    assert "Claude" in df_claude.iloc[0]["model_name"]


def test_null_and_empty_explanation_filtered():
    """NULL and empty-string explanations should be excluded."""
    con = _make_con()
    con.execute(
        "INSERT INTO questions VALUES ('q1', 'KEN', 'FL', 'PA', '2025-01', 'hs-1')"
    )
    # Valid explanation
    _insert_raw_rows(
        con, question_id="q1", run_id="r1", model_name="OpenAI",
        human_explanation="Valid rationale.",
    )
    # NULL explanation - insert directly
    for m in range(1, 7):
        for b in range(1, 6):
            con.execute(
                """
                INSERT INTO forecasts_raw (
                    question_id, run_id, model_name, month_index, bucket_index,
                    probability, status, human_explanation
                ) VALUES ('q1', 'r1', 'NullModel', ?, ?, 0.2, 'ok', NULL)
                """,
                [m, b],
            )
    # Empty string explanation
    _insert_raw_rows(
        con, question_id="q1", run_id="r1", model_name="EmptyModel",
        human_explanation="",
    )

    df = build_rationale_export(con, hazard_code="FL")
    assert len(df) == 1
    assert df.iloc[0]["model_name"] == "OpenAI"


def test_both_tables_union():
    """Rows from both forecasts_raw and forecasts_ensemble should appear."""
    con = _make_con()
    con.execute(
        "INSERT INTO questions VALUES ('q1', 'KEN', 'FL', 'PA', '2025-01', 'hs-1')"
    )
    _insert_raw_rows(
        con, question_id="q1", run_id="r1", model_name="OpenAI-gpt-5.2",
        human_explanation="Per-model rationale.",
    )
    _insert_ensemble_rows(
        con, question_id="q1", run_id="r1", iso3="KEN", hazard_code="FL",
        metric="PA", model_name="ensemble_mean_v2",
        human_explanation="Ensemble rationale.",
    )

    df = build_rationale_export(con, hazard_code="FL")
    assert len(df) == 2
    models = set(df["model_name"])
    assert "OpenAI-gpt-5.2" in models
    assert "ensemble_mean_v2" in models


def test_missing_table_graceful():
    """If forecast tables don't exist, return empty DataFrame."""
    con = duckdb.connect()
    con.execute(
        """
        CREATE TABLE questions (
            question_id TEXT, iso3 TEXT, hazard_code TEXT,
            metric TEXT, target_month TEXT, hs_run_id TEXT
        )
        """
    )

    df = build_rationale_export(con, hazard_code="FL")
    assert len(df) == 0
    assert "human_explanation" in df.columns


def test_no_connection():
    """None connection returns empty DataFrame."""
    df = build_rationale_export(None, hazard_code="FL")
    assert len(df) == 0
    assert "human_explanation" in df.columns


def test_output_columns():
    """Output should have the expected columns in order."""
    con = _make_con()
    con.execute(
        "INSERT INTO questions VALUES ('q1', 'KEN', 'FL', 'PA', '2025-01', 'hs-1')"
    )
    _insert_raw_rows(
        con, question_id="q1", run_id="r1", model_name="Gemini Flash",
        human_explanation="Test rationale.",
    )

    df = build_rationale_export(con, hazard_code="FL")
    expected_columns = [
        "question_id",
        "run_id",
        "hs_run_id",
        "iso3",
        "country_name",
        "hazard_code",
        "metric",
        "target_month",
        "model_name",
        "human_explanation",
    ]
    assert list(df.columns) == expected_columns


def test_csv_serialization():
    """Output should be serializable to CSV without errors."""
    con = _make_con()
    con.execute(
        "INSERT INTO questions VALUES ('q1', 'KEN', 'FL', 'PA', '2025-01', 'hs-1')"
    )
    _insert_raw_rows(
        con, question_id="q1", run_id="r1", model_name="OpenAI",
        human_explanation="Rationale with, commas and \"quotes\".",
    )

    df = build_rationale_export(con, hazard_code="FL")
    csv_text = df.to_csv(index=False)
    lines = [l for l in csv_text.strip().splitlines() if l]
    assert len(lines) == 2  # header + 1 data row
