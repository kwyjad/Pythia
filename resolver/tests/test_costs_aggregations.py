# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

import duckdb
import pytest

from resolver.query.costs import (
    build_costs_monthly,
    build_costs_runs,
    build_costs_total,
    build_latencies_runs,
    build_run_runtimes,
    phase_group,
)


def _create_llm_calls(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE llm_calls (
            run_id VARCHAR,
            hs_run_id VARCHAR,
            forecaster_run_id VARCHAR,
            question_id VARCHAR,
            iso3 VARCHAR,
            phase VARCHAR,
            component VARCHAR,
            model_id VARCHAR,
            model_name VARCHAR,
            cost_usd DOUBLE,
            elapsed_ms DOUBLE,
            created_at TIMESTAMP,
            is_test BOOLEAN DEFAULT FALSE
        )
        """
    )


def _seed_llm_calls(con: duckdb.DuckDBPyConnection) -> None:
    _create_llm_calls(con)
    con.execute(
        """
        INSERT INTO llm_calls (run_id, hs_run_id, question_id, iso3, phase, model_id, model_name, cost_usd, elapsed_ms, created_at) VALUES
            ('run-1', NULL, 'q1', 'USA', 'web_search', 'gpt-4', NULL, 1.0, 100, '2024-01-15 00:00:00'),
            ('run-1', NULL, 'q1', 'USA', 'hs_summarize', 'gpt-4', NULL, 2.0, 200, '2024-01-15 00:00:00'),
            ('run-1', NULL, 'q2', 'CAN', 'research_step', 'gpt-3.5', NULL, 3.0, 300, '2024-02-10 00:00:00'),
            (NULL, 'hs-9', 'q2', 'CAN', 'spd_forecast', 'gpt-3.5', NULL, 4.0, 400, '2024-02-11 00:00:00'),
            ('run-2', NULL, 'q3', 'MEX', 'scenario_build', 'gpt-4', NULL, 5.0, 500, '2024-02-20 00:00:00'),
            ('run-2', NULL, 'q3', 'MEX', 'misc', NULL, 'custom-model', 1.0, 600, '2024-02-20 00:00:00')
        """
    )


def _summary_row(df, **filters):
    subset = df
    for key, value in filters.items():
        subset = subset[subset[key] == value]
    assert len(subset) == 1
    return subset.iloc[0]


def test_costs_total_aggregations():
    con = duckdb.connect(":memory:")
    _seed_llm_calls(con)

    tables = build_costs_total(con)
    summary = tables["summary"]
    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["total_cost_usd"] == pytest.approx(16.0)
    assert row["n_questions"] == 3
    assert row["avg_cost_per_question"] == pytest.approx(16.0 / 3.0)
    assert row["median_cost_per_question"] == pytest.approx(6.0)
    assert row["n_countries"] == 4
    assert row["avg_cost_per_country"] == pytest.approx(16.0 / 4.0)
    assert row["median_cost_per_country"] == pytest.approx(3.5)

    by_model = tables["by_model"]
    assert by_model["total_cost_usd"].sum() == pytest.approx(16.0)
    model_totals = {row["model"]: row["total_cost_usd"] for _, row in by_model.iterrows()}
    assert model_totals["gpt-4"] == pytest.approx(8.0)
    assert model_totals["gpt-3.5"] == pytest.approx(7.0)
    assert model_totals["custom-model"] == pytest.approx(1.0)
    gpt4 = _summary_row(by_model, model="gpt-4")
    gpt35 = _summary_row(by_model, model="gpt-3.5")
    custom = _summary_row(by_model, model="custom-model")
    assert gpt4["n_questions"] == 2
    assert gpt35["n_questions"] == 1
    assert gpt4["avg_cost_per_question"] == pytest.approx(4.0)
    assert gpt35["avg_cost_per_question"] == pytest.approx(7.0)
    assert custom["avg_cost_per_question"] == pytest.approx(1.0)
    assert gpt4["n_countries"] == 2
    assert gpt35["n_countries"] == 2

    by_phase = tables["by_phase"]
    assert by_phase["total_cost_usd"].sum() == pytest.approx(16.0)
    web = _summary_row(by_phase, phase="web_search")
    scenario = _summary_row(by_phase, phase="scenario")
    assert web["total_cost_usd"] == pytest.approx(1.0)
    assert scenario["total_cost_usd"] == pytest.approx(5.0)
    assert web["avg_cost_per_question"] == pytest.approx(1.0)
    assert scenario["avg_cost_per_question"] == pytest.approx(5.0)


def test_costs_monthly_and_run_aggregations():
    con = duckdb.connect(":memory:")
    _seed_llm_calls(con)

    monthly = build_costs_monthly(con)["summary"]
    jan = _summary_row(monthly, year=2024, month=1)
    feb = _summary_row(monthly, year=2024, month=2)
    assert jan["total_cost_usd"] == pytest.approx(3.0)
    assert jan["avg_cost_per_question"] == pytest.approx(3.0)
    assert jan["median_cost_per_question"] == pytest.approx(3.0)
    assert feb["total_cost_usd"] == pytest.approx(13.0)
    assert feb["avg_cost_per_question"] == pytest.approx(6.5)
    assert feb["median_cost_per_question"] == pytest.approx(6.5)

    runs = build_costs_runs(con)["summary"]
    assert runs["run_id"].is_unique
    assert runs[runs["run_id"] == "run-1"].shape[0] == 1
    run_1 = _summary_row(runs, run_id="run-1")
    run_2 = _summary_row(runs, run_id="run-2")
    run_hs = _summary_row(runs, run_id="hs-9")
    assert run_1["total_cost_usd"] == pytest.approx(6.0)
    assert run_1["avg_cost_per_question"] == pytest.approx(3.0)
    assert run_2["total_cost_usd"] == pytest.approx(6.0)
    assert run_2["avg_cost_per_question"] == pytest.approx(6.0)
    assert run_hs["total_cost_usd"] == pytest.approx(4.0)
    assert run_hs["median_cost_per_question"] == pytest.approx(4.0)


def test_latencies_run_quantiles():
    con = duckdb.connect(":memory:")
    _seed_llm_calls(con)

    latencies = build_latencies_runs(con)
    row = latencies[
        (latencies["run_id"] == "run-2")
        & (latencies["model"] == "custom-model")
        & (latencies["phase"] == "other")
    ]
    assert len(row) == 1
    row = row.iloc[0]
    assert row["n_calls"] == 1
    assert row["p50_elapsed_ms"] == pytest.approx(600.0)
    assert row["p90_elapsed_ms"] == pytest.approx(600.0)


def test_run_runtimes_aggregation():
    con = duckdb.connect(":memory:")
    _seed_llm_calls(con)

    runtimes = build_run_runtimes(con)
    for column in [
        "question_p50_ms",
        "question_p90_ms",
        "country_p50_ms",
        "country_p90_ms",
    ]:
        assert column in runtimes.columns
    assert runtimes["run_id"].is_unique
    assert len(runtimes) == 3
    assert runtimes[runtimes["run_id"] == "run-1"].shape[0] == 1
    assert runtimes[runtimes["run_id"] == "run-2"].shape[0] == 1
    assert runtimes[runtimes["run_id"] == "hs-9"].shape[0] == 1

    run_1 = _summary_row(runtimes, run_id="run-1")
    assert run_1["web_search_ms"] == pytest.approx(100.0)
    assert run_1["hs_ms"] == pytest.approx(200.0)
    assert run_1["research_ms"] == pytest.approx(300.0)
    assert run_1["total_ms"] == pytest.approx(600.0)
    assert run_1["question_p50_ms"] == pytest.approx(300.0)
    assert run_1["question_p90_ms"] == pytest.approx(300.0)
    assert run_1["country_p50_ms"] == pytest.approx(300.0)
    assert run_1["country_p90_ms"] == pytest.approx(300.0)
    assert run_1["n_questions"] == 2
    assert run_1["n_countries"] == 2

    run_2 = _summary_row(runtimes, run_id="run-2")
    assert run_2["question_p50_ms"] == pytest.approx(1100.0)
    assert run_2["question_p90_ms"] == pytest.approx(1100.0)
    assert run_2["country_p50_ms"] == pytest.approx(1100.0)
    assert run_2["country_p90_ms"] == pytest.approx(1100.0)

    run_hs = _summary_row(runtimes, run_id="hs-9")
    assert run_hs["question_p50_ms"] == pytest.approx(400.0)
    assert run_hs["question_p90_ms"] == pytest.approx(400.0)
    assert run_hs["country_p50_ms"] == pytest.approx(400.0)
    assert run_hs["country_p90_ms"] == pytest.approx(400.0)


def _seed_attribution_cases(con: duckdb.DuckDBPyConnection) -> None:
    """Rows exercising Sibyl phase, component-fallback attribution, and run linkage."""
    _create_llm_calls(con)
    con.execute(
        """
        INSERT INTO llm_calls
            (run_id, hs_run_id, forecaster_run_id, question_id, iso3, phase, component,
             model_id, model_name, cost_usd, elapsed_ms, created_at) VALUES
            -- Sibyl: one Opus row + one Brave row, both phase='sibyl'
            ('run-1', NULL, NULL, 'q1', 'USA', 'sibyl', NULL,
             'claude-opus-4-8', 'sibyl', 6.0, 700, '2024-03-01 00:00:00'),
            ('run-1', NULL, NULL, 'q1', 'USA', 'sibyl', NULL,
             'brave-web-search', 'sibyl', 2.0, 50, '2024-03-01 00:00:00'),
            -- Phase-less generic row attributed via component
            ('run-1', NULL, NULL, 'q2', 'CAN', NULL, 'prediction_markets',
             NULL, 'gpt-4', 3.0, 120, '2024-03-02 00:00:00'),
            ('run-1', NULL, NULL, 'q2', 'CAN', NULL, 'HorizonScanner',
             NULL, 'gpt-4', 1.0, 80, '2024-03-02 00:00:00'),
            -- Run linkage only via forecaster_run_id (run_id + hs_run_id NULL)
            (NULL, NULL, 'fc-1', 'q3', 'MEX', 'spd_forecast', 'Forecaster',
             'gpt-4', 'gpt-4', 4.0, 400, '2024-03-03 00:00:00')
        """
    )


def test_sibyl_and_component_attribution():
    con = duckdb.connect(":memory:")
    _seed_attribution_cases(con)

    tables = build_costs_total(con)
    by_phase = tables["by_phase"]

    # Sum across phases equals the grand total (invariant must hold).
    assert by_phase["total_cost_usd"].sum() == pytest.approx(16.0)

    phase_totals = {row["phase"]: row["total_cost_usd"] for _, row in by_phase.iterrows()}
    # Sibyl is a first-class phase: Opus (6) + Brave (2) = 8.
    assert phase_totals["sibyl"] == pytest.approx(8.0)
    # Component fallback attributes phase-less rows — no stray "other".
    assert phase_totals["prediction_markets"] == pytest.approx(3.0)
    assert phase_totals["hs"] == pytest.approx(1.0)
    assert phase_totals["forecast"] == pytest.approx(4.0)
    assert "other" not in phase_totals

    # By-model still splits Sibyl Opus vs Brave credits.
    by_model = tables["by_model"]
    model_totals = {row["model"]: row["total_cost_usd"] for _, row in by_model.iterrows()}
    assert model_totals["claude-opus-4-8"] == pytest.approx(6.0)
    assert model_totals["brave-web-search"] == pytest.approx(2.0)

    # Run linkage: the forecaster-only row is attributed to fc-1.
    runs = build_costs_runs(con)["summary"]
    fc_run = _summary_row(runs, run_id="fc-1")
    assert fc_run["total_cost_usd"] == pytest.approx(4.0)


def test_run_runtimes_itemizes_sibyl_and_pm():
    con = duckdb.connect(":memory:")
    _seed_attribution_cases(con)

    runtimes = build_run_runtimes(con)
    run_1 = _summary_row(runtimes, run_id="run-1")
    # sibyl_ms is now projected and populated (was always 0 / dropped before).
    assert run_1["sibyl_ms"] == pytest.approx(750.0)  # 700 + 50
    assert run_1["prediction_markets_ms"] == pytest.approx(120.0)
    assert run_1["hs_ms"] == pytest.approx(80.0)


def test_phase_group_canonical_mapping():
    assert phase_group("sibyl") == "sibyl"
    assert phase_group("sibyl_trial0_step1") == "sibyl"
    assert phase_group("prediction_markets") == "prediction_markets"
    assert phase_group("pm") == "prediction_markets"
    assert phase_group("HorizonScanner") == "hs"
    assert phase_group("hs_triage") == "hs"
    assert phase_group("Researcher") == "research"
    assert phase_group("web_research") == "web_search"
    assert phase_group("Forecaster") == "forecast"
    assert phase_group("spd_forecast") == "forecast"
    assert phase_group("ScenarioWriter") == "scenario"
    assert phase_group(None) == "other"
    assert phase_group("") == "other"
    assert phase_group("misc") == "other"


def test_phase_group_live_pipeline_values():
    """The exact phase/call_type strings the live loggers write must never fall to
    'other'. These are the values passed at forecaster/cli.py, scenario_writer.py,
    and sibyl/cost.py call sites — a rename there without a phase_group branch would
    silently regress the Costs page."""
    assert phase_group("spd_v2") == "forecast"
    assert phase_group("binary_v2") == "forecast"
    assert phase_group("scenario_v2") == "scenario"
    assert phase_group("forecast_web_research") == "web_search"
    assert phase_group("sibyl") == "sibyl"
    # hs loggers hardcode phase='hs_triage'
    assert phase_group("hs_triage") == "hs"
