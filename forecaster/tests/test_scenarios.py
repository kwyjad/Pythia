# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import pytest

duckdb = pytest.importorskip("duckdb")

import forecaster.scenario_writer as scenario_writer


def test_safe_json_loads_scenario_handles_code_fence() -> None:
    fenced = """```json
    {"primary": {"bucket_label": "bucket_3", "probability": 0.6, "context": ["c"], "needs": {"WASH": ["w"]}, "operational_impacts": ["o"]},
     "alternative": null}
    ```"""

    obj = scenario_writer._safe_json_loads_scenario(fenced)

    assert obj["primary"]["bucket_label"] == "bucket_3"
    assert obj["primary"]["probability"] == 0.6


@pytest.mark.db
def test_run_scenarios_for_run_matches_scenarios_schema(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Ensure run_scenarios_for_run does not reference non-existent columns
    (e.g., question_id) in the scenarios table.
    """

    from pathlib import Path
    from pythia.db.schema import ensure_schema

    db_path = Path(tmp_path) / "scenarios_test.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    con = duckdb.connect(str(db_path))
    try:
        ensure_schema(con)
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS questions (
                question_id TEXT,
                hs_run_id TEXT,
                iso3 TEXT,
                hazard_code TEXT,
                metric TEXT,
                target_month TEXT,
                window_start_date DATE,
                window_end_date DATE,
                wording TEXT,
                status TEXT
            )
            """
        )
        con.execute(
            """
            INSERT INTO questions (
                question_id, hs_run_id, iso3, hazard_code, metric,
                target_month, window_start_date, window_end_date,
                wording, status
            ) VALUES (
                'Q_SCEN', 'hs_test', 'ETH', 'ACE', 'FATALITIES',
                '2025-01', DATE '2024-12-01', DATE '2025-01-31',
                'Test scenario question', 'active'
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS forecasts_ensemble (
                run_id TEXT,
                question_id TEXT,
                iso3 TEXT,
                hazard_code TEXT,
                metric TEXT,
                month_index INTEGER,
                bucket_index INTEGER,
                probability DOUBLE,
                ev_value DOUBLE,
                weights_profile TEXT,
                created_at TIMESTAMP,
                status TEXT,
                human_explanation TEXT
            )
            """
        )
        con.execute(
            """
            INSERT INTO forecasts_ensemble (
                run_id, question_id, iso3, hazard_code, metric,
                month_index, bucket_index, probability, ev_value,
                weights_profile, created_at, status, human_explanation
            ) VALUES (
                'fc_scen', 'Q_SCEN', 'ETH', 'ACE', 'FATALITIES',
                1, 3, 0.6, NULL,
                'ensemble', CURRENT_TIMESTAMP, 'ok', 'Test'
            )
            """
        )
        con.execute(
            """
            INSERT INTO hs_triage (
                run_id, iso3, hazard_code, tier, triage_score, need_full_spd, created_at
            ) VALUES (
                'hs_test', 'ETH', 'ACE', 'priority', 0.9, TRUE, CURRENT_TIMESTAMP
            )
            """
        )
    finally:
        con.close()

    def fake_connect(read_only: bool = False):
        return duckdb.connect(str(db_path))

    monkeypatch.setattr(scenario_writer, "connect", fake_connect)

    async def fake_call_chat_ms(ms, prompt, **kwargs):
        return (
            '{"primary":{"bucket_label":"bucket_3","probability":0.6,'
            '"context":["c1","c2"],"needs":{"WASH":["water"],"Health":[],"Nutrition":[],"Protection":[],"Education":[],"Shelter":[],"FoodSecurity":[]},'
            '"operational_impacts":["ops"]},"alternative":null}',
            {"total_tokens": 42, "elapsed_ms": 100},
            None,
        )

    import forecaster.providers as _providers
    monkeypatch.setattr(_providers, "call_chat_ms", fake_call_chat_ms)

    scenario_writer.run_scenarios_for_run("fc_scen")

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT scenario_type, text FROM scenarios WHERE run_id = 'fc_scen' ORDER BY scenario_type"
        ).fetchall()
    finally:
        con.close()

    assert rows
    primary_text = dict(rows).get("primary", "")
    assert "Context" in primary_text
    assert "Humanitarian Needs" in primary_text
    assert "Operational Impacts" in primary_text
