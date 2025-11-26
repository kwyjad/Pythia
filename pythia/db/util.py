from __future__ import annotations

import uuid
from typing import Mapping, Any

import duckdb


def write_llm_call(
    conn: duckdb.DuckDBPyConnection,
    component: str,
    model: str,
    prompt_key: str,
    version: str,
    usage: Mapping[str, Any],
    cost: float,
    latency_ms: int,
    success: bool,
    *,
    llm_profile: str | None = None,
    hs_run_id: str | None = None,
    ui_run_id: str | None = None,
    forecaster_run_id: str | None = None,
) -> None:
    """Insert a single LLM call record into the llm_calls table."""

    tokens_in = int(usage.get("prompt_tokens", 0)) if isinstance(usage, Mapping) else 0
    tokens_out = int(usage.get("completion_tokens", 0)) if isinstance(usage, Mapping) else 0
    conn.execute(
        """
        INSERT INTO llm_calls (
            call_id,
            component,
            model_name,
            prompt_key,
            prompt_version,
            tokens_in,
            tokens_out,
            cost_usd,
            latency_ms,
            success,
            llm_profile,
            hs_run_id,
            ui_run_id,
            forecaster_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            str(uuid.uuid4()),
            component,
            model,
            prompt_key,
            version,
            tokens_in,
            tokens_out,
            float(cost or 0.0),
            int(latency_ms),
            bool(success),
            llm_profile,
            hs_run_id,
            ui_run_id,
            forecaster_run_id,
        ],
    )
