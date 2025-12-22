# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import uuid
from typing import Mapping, Any

import duckdb
import json
from datetime import datetime


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


def log_web_research_call(
    conn: duckdb.DuckDBPyConnection,
    *,
    component: str,
    phase: str,
    provider: str,
    model_name: str,
    model_id: str,
    run_id: str | None,
    question_id: str | None,
    prompt_text: str,
    response_text: str,
    parsed_json: Mapping[str, Any] | None,
    usage: Mapping[str, Any] | None,
    elapsed_ms: int,
    error_text: str | None,
    success: bool,
    hs_run_id: str | None = None,
    iso3: str | None = None,
    hazard_code: str | None = None,
    metric: str | None = None,
) -> None:
    """Best-effort logger for web research calls into llm_calls."""

    usage_dict = usage or {}
    try:
        prompt_tokens = int(usage_dict.get("prompt_tokens", 0))
    except Exception:
        prompt_tokens = 0
    try:
        completion_tokens = int(usage_dict.get("completion_tokens", 0))
    except Exception:
        completion_tokens = 0
    try:
        total_tokens = int(usage_dict.get("total_tokens", 0) or (prompt_tokens + completion_tokens))
    except Exception:
        total_tokens = prompt_tokens + completion_tokens
    try:
        cost_usd = float(usage_dict.get("cost_usd", 0.0))
    except Exception:
        cost_usd = 0.0

    usage_json = json.dumps(usage_dict or {}, ensure_ascii=False)
    conn.execute(
        """
        INSERT INTO llm_calls (
            call_id, run_id, hs_run_id, question_id, call_type, phase,
            model_name, provider, model_id, prompt_text, response_text, parsed_json,
            usage_json, elapsed_ms, prompt_tokens, completion_tokens, total_tokens,
            cost_usd, error_text, timestamp, iso3, hazard_code, metric
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            str(uuid.uuid4()),
            run_id,
            hs_run_id,
            question_id,
            "web_research",
            phase,
            model_name,
            provider,
            model_id,
            prompt_text,
            response_text,
            json.dumps(parsed_json or {}, ensure_ascii=False),
            usage_json,
            int(elapsed_ms),
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost_usd,
            error_text,
            datetime.utcnow(),
            iso3,
            hazard_code,
            metric,
        ],
    )
