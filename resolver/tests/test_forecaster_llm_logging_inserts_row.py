import asyncio

import pytest


def test_forecaster_llm_logging_inserts_row(tmp_path, monkeypatch):
    pytest.importorskip("duckdb")

    from forecaster.llm_logging import log_forecaster_llm_call
    from resolver.db import duckdb_io

    db_path = tmp_path / "resolver.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")

    asyncio.run(
        log_forecaster_llm_call(
            run_id="run_1",
            question_id="q1",
            prompt_text="prompt",
            model_name="Test Model",
            provider="openai",
            model_id="gpt-test",
            phase="spd",
            call_type="chat",
            iso3="USA",
            hazard_code="ACE",
            metric="fatalities",
            response_text="response",
            usage={
                "elapsed_ms": 10,
                "prompt_tokens": 5,
                "completion_tokens": 7,
                "total_tokens": 12,
                "cost_usd": 0.01,
            },
            error_text=None,
        )
    )

    conn = duckdb_io.get_db(str(db_path))
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM llm_calls WHERE run_id = ? AND call_type = ?",
            ["run_1", "chat"],
        ).fetchone()
    finally:
        duckdb_io.close_db(conn)

    assert int(row[0] or 0) >= 1
