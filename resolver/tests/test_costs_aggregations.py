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

    by_phase = tables["by_phase"]
    assert by_phase["total_cost_usd"].sum() == pytest.approx(16.0)


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
