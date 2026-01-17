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


def test_build_forecast_spd_export_latest_wins():
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
            ('q-pa', 'KEN', 'DR', 'PA', '2024-01', 'hs-run-1')
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
    con.execute(
        """
        INSERT INTO hs_triage VALUES
            ('hs-run-1', 'KEN', 'DR', 'tier-1', 0.3, '2024-01-02')
        """
    )

    df = build_forecast_spd_export(con)
    assert not df.empty
    row = df.loc[
        (df["ISO"] == "KEN") & (df["hazard"] == "DR") & (df["model"] == "ensemble_mean_v2")
    ].iloc[0]
    assert row["SPD_1"] == pytest.approx(0.1)
    assert (row[["SPD_1", "SPD_2", "SPD_3", "SPD_4", "SPD_5"]].sum()) == pytest.approx(1.0)
