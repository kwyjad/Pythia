from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.dump_pythia_debug_bundle import build_debug_bundle_markdown


def test_build_debug_bundle_markdown_basic(tmp_path: Path) -> None:
    db_path = tmp_path / "bundle_test.duckdb"
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
                human_explanation TEXT
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
            INSERT INTO hs_triage (run_id, iso3, hazard_code, tier, triage_score, created_at)
            VALUES ('hs_test', 'ETH', 'ACE', 'priority', 0.9, CURRENT_TIMESTAMP)
            """
        )
        con.execute(
            """
            INSERT INTO llm_calls (
                call_type, phase, provider, model_id, temperature, run_id, question_id,
                iso3, hazard_code, metric, prompt_text, response_text, error_text,
                usage_json, created_at
            ) VALUES (
                'chat', 'research_v2', 'google', 'gemini-test', 0.3, 'fc_test', 'Q1',
                'ETH', 'ACE', 'FATALITIES', 'prompt', 'response', '', '{"total_tokens": 42}', CURRENT_TIMESTAMP
            )
            """
        )

        question_types: List[Tuple[str, str, str, str]] = [("Q1", "ETH", "ACE", "FATALITIES")]
        md = build_debug_bundle_markdown(con, "duckdb:///fake.duckdb", "fc_test", question_types)
    finally:
        con.close()

    assert "# Pythia v2 Debug Bundle — Run fc_test" in md
    assert "### 2.1 ETH / ACE / FATALITIES (question_id=Q1)" in md
    assert "#### 2.1.2 Research (Research v2)" in md
    assert "##### Prompt" in md
    assert "##### Output" in md
