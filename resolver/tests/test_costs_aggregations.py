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
)


def _seed_llm_calls(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE llm_calls (
            run_id VARCHAR,
            hs_run_id VARCHAR,
            question_id VARCHAR,
            iso3 VARCHAR,
            phase VARCHAR,
            model_id VARCHAR,
            model_name VARCHAR,
            cost_usd DOUBLE,
            elapsed_ms DOUBLE,
            created_at TIMESTAMP
        )
        """
    )
    con.execute(
        """
        INSERT INTO llm_calls VALUES
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
    assert row["n_countries"] == 3
    assert row["avg_cost_per_country"] == pytest.approx(16.0 / 3.0)
    assert row["median_cost_per_country"] == pytest.approx(6.0)

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
    assert gpt35["n_countries"] == 1

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
