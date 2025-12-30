import duckdb
import pytest

from resolver.query.downloads import build_forecast_spd_export


def _insert_forecast_rows(
    con,
    table: str,
    *,
    question_id: str,
    run_id: str,
    model_name: str,
    month_index: int,
    probabilities: list[float],
    created_at: str,
):
    for bucket_index, probability in enumerate(probabilities, start=1):
        con.execute(
            f"""
            INSERT INTO {table} (
                question_id,
                run_id,
                model_name,
                month_index,
                bucket_index,
                probability,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                question_id,
                run_id,
                model_name,
                month_index,
                bucket_index,
                probability,
                "ok",
                created_at,
            ],
        )


def test_build_forecast_spd_export():
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
        CREATE TABLE forecasts_ensemble (
            question_id TEXT,
            run_id TEXT,
            model_name TEXT,
            month_index INTEGER,
            bucket_index INTEGER,
            probability DOUBLE,
            status TEXT,
            created_at TIMESTAMP
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
            created_at TIMESTAMP
        )
        """
    )
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
        INSERT INTO questions VALUES
            ('q-pa', 'KEN', 'DR', 'PA', '2024-01', 'hs-run-1'),
            ('q-fat', 'UGA', 'DR', 'FATALITIES', '2024-02', 'hs-run-2')
        """
    )

    _insert_forecast_rows(
        con,
        "forecasts_ensemble",
        question_id="q-pa",
        run_id="ens-old",
        model_name="ensemble_mean_v2",
        month_index=1,
        probabilities=[0.2, 0.2, 0.2, 0.2, 0.2],
        created_at="2024-01-01",
    )
    _insert_forecast_rows(
        con,
        "forecasts_ensemble",
        question_id="q-pa",
        run_id="ens-new",
        model_name="ensemble_mean_v2",
        month_index=1,
        probabilities=[0.1, 0.2, 0.3, 0.2, 0.2],
        created_at="2024-01-02",
    )
    _insert_forecast_rows(
        con,
        "forecasts_ensemble",
        question_id="q-pa",
        run_id="ens-new",
        model_name="ensemble_bayesmc_v2",
        month_index=1,
        probabilities=[0.05, 0.2, 0.35, 0.2, 0.2],
        created_at="2024-01-02",
    )
    _insert_forecast_rows(
        con,
        "forecasts_ensemble",
        question_id="q-fat",
        run_id="ens-new",
        model_name="ensemble_mean_v2",
        month_index=1,
        probabilities=[0.3, 0.25, 0.2, 0.15, 0.1],
        created_at="2024-01-03",
    )
    _insert_forecast_rows(
        con,
        "forecasts_ensemble",
        question_id="q-fat",
        run_id="ens-new",
        model_name="ensemble_bayesmc_v2",
        month_index=1,
        probabilities=[0.25, 0.25, 0.2, 0.2, 0.1],
        created_at="2024-01-03",
    )

    _insert_forecast_rows(
        con,
        "forecasts_raw",
        question_id="q-pa",
        run_id="raw-1",
        model_name="gpt-5.1",
        month_index=1,
        probabilities=[0.15, 0.2, 0.25, 0.2, 0.2],
        created_at="2024-01-04",
    )
    _insert_forecast_rows(
        con,
        "forecasts_raw",
        question_id="q-fat",
        run_id="raw-2",
        model_name="gpt-5.1",
        month_index=1,
        probabilities=[0.4, 0.2, 0.15, 0.15, 0.1],
        created_at="2024-01-04",
    )

    con.execute(
        """
        INSERT INTO hs_triage VALUES
            ('hs-run-1', 'KEN', 'DR', 'tier-old', 0.2, '2024-01-01'),
            ('hs-run-1', 'KEN', 'DR', 'tier-2', 0.4, '2024-01-05'),
            ('hs-run-2', 'UGA', 'DR', 'tier-1', 0.3, '2024-01-02')
        """
    )

    df = build_forecast_spd_export(con)

    expected_columns = [
        "ISO",
        "country_name",
        "year",
        "month",
        "forecast_month",
        "metric",
        "hazard",
        "model",
        "SPD_1",
        "SPD_2",
        "SPD_3",
        "SPD_4",
        "SPD_5",
        "EIV",
        "triage_score",
        "triage_tier",
        "hs_run_ID",
    ]
    assert list(df.columns) == expected_columns

    csv_text = df.to_csv(index=False)
    csv_lines = [line for line in csv_text.strip().splitlines() if line]
    assert csv_lines[0].split(",") == expected_columns
    assert len(csv_lines) > 1

    assert {"ensemble_mean_v2", "ensemble_bayesmc_v2", "gpt-5.1"}.issubset(
        set(df["model"])
    )

    pa_row = df[(df["ISO"] == "KEN") & (df["model"] == "ensemble_mean_v2")].iloc[0]
    assert pa_row["SPD_1"] == pytest.approx(0.1)
    assert sum(pa_row[["SPD_1", "SPD_2", "SPD_3", "SPD_4", "SPD_5"]]) == pytest.approx(1.0)

    fatal_row = df[(df["ISO"] == "UGA") & (df["model"] == "ensemble_bayesmc_v2")].iloc[0]
    assert sum(fatal_row[["SPD_1", "SPD_2", "SPD_3", "SPD_4", "SPD_5"]]) == pytest.approx(1.0)

    pa_expected = 0.1 * 0 + 0.2 * 30000 + 0.3 * 150000 + 0.2 * 375000 + 0.2 * 700000
    assert pa_row["EIV"] == pytest.approx(pa_expected)

    fatal_expected = 0.25 * 0 + 0.25 * 15 + 0.2 * 62 + 0.2 * 300 + 0.1 * 700
    assert fatal_row["EIV"] == pytest.approx(fatal_expected)

    assert pa_row["triage_score"] == pytest.approx(0.4)
    assert pa_row["triage_tier"] == "tier-2"
    assert pa_row["hs_run_ID"] == "hs-run-1"
