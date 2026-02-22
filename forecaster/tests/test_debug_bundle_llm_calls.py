# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.dump_pythia_debug_bundle import build_debug_bundle_markdown


def test_build_debug_bundle_llm_call_counts_without_forecaster_run_id(tmp_path: Path) -> None:
    db_path = tmp_path / "bundle_llm.duckdb"
    con = duckdb.connect(str(db_path))

    try:
        con.execute(
            """
            CREATE TABLE questions (
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
            CREATE TABLE forecasts_ensemble (
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
                human_explanation TEXT,
                model_name TEXT
            )
            """
        )
        con.execute(
            """
            CREATE TABLE forecasts_raw (
                run_id TEXT,
                question_id TEXT,
                model_name TEXT,
                month_index INTEGER,
                bucket_index INTEGER,
                probability DOUBLE
            )
            """
        )
        con.execute(
            """
            CREATE TABLE llm_calls (
                call_type TEXT,
                phase TEXT,
                provider TEXT,
                model_id TEXT,
                model_name TEXT,
                temperature DOUBLE,
                run_id TEXT,
                question_id TEXT,
                iso3 TEXT,
                hazard_code TEXT,
                metric TEXT,
                prompt_text TEXT,
                response_text TEXT,
                error_text TEXT,
                usage_json TEXT,
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
            INSERT INTO questions (
                question_id, hs_run_id, iso3, hazard_code, metric,
                target_month, window_start_date, window_end_date, wording, status
            ) VALUES (
                'Q1', 'hs_test', 'ETH', 'ACE', 'FATALITIES',
                '2025-01', DATE '2024-12-01', DATE '2025-01-31', 'Test question', 'active'
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
                'fc_test', 'Q1', 'ETH', 'ACE', 'FATALITIES',
                1, 1, 0.5, NULL,
                'ensemble', CURRENT_TIMESTAMP, 'ok', 'Test'
            )
            """
        )
        con.execute(
            """
            INSERT INTO llm_calls (
                call_type, phase, provider, model_id, model_name, run_id, question_id, usage_json, created_at
            ) VALUES (
                'research', 'research_v2', 'google', 'gemini-test', 'gemini-test', 'fc_test', 'Q1',
                '{"total_tokens": 21}', CURRENT_TIMESTAMP
            )
            """
        )

        questions = [
            {
                "question_id": "Q1",
                "iso3": "ETH",
                "hazard_code": "ACE",
                "metric": "FATALITIES",
                "hs_run_id": "hs_test",
                "target_month": "2025-01",
                "window_start_date": "2024-12-01",
                "window_end_date": "2025-01-31",
                "wording": "Test question",
            }
        ]

        md = build_debug_bundle_markdown(con, "duckdb:///fake.duckdb", "fc_test", "hs_test", questions, [])
    finally:
        con.close()

    assert "### 1.4 LLM calls by phase/provider/model_id (this run)" in md
    assert "| research_v2 | google | gemini-test | 1 | 0 |" in md
