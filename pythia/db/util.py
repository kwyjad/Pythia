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
            success
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        ],
    )
