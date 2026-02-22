# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import re
import uuid
from typing import Mapping, Any

import duckdb
import json
from datetime import datetime


_FENCED_JSON_PATTERN = re.compile(r"```json\s*[\s\S]*?```", re.IGNORECASE)


def ensure_llm_calls_columns(conn: duckdb.DuckDBPyConnection) -> None:
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE LOWER(table_name) = 'llm_calls'"
        ).fetchone()
        if not row or not row[0]:
            return
    except Exception:  # noqa: BLE001
        return
    columns = [
        ("status", "TEXT"),
        ("error_type", "TEXT"),
        ("error_message", "TEXT"),
        ("response_format", "TEXT"),
        ("hazard_scores_json", "TEXT"),
        ("hazard_scores_parse_ok", "BOOLEAN"),
    ]
    for name, col_type in columns:
        conn.execute(f"ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS {name} {col_type}")


def derive_status(error_text: str | None) -> str:
    if error_text and str(error_text).strip():
        return "error"
    return "ok"


def derive_error_type(error_text: str | None) -> str | None:
    if not error_text or not str(error_text).strip():
        return None
    lowered = str(error_text).lower()
    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "429" in lowered or ("rate" in lowered and "limit" in lowered):
        return "rate_limit"
    if "disabled after" in lowered or "cooldown active" in lowered:
        return "provider_disabled"
    if "parse failed" in lowered or "jsondecodeerror" in lowered:
        return "parse_error"
    return "provider_error"


def derive_error_message(error_text: str | None, max_len: int = 500) -> str | None:
    if not error_text or not str(error_text).strip():
        return None
    text = str(error_text).strip()
    if len(text) <= max_len:
        return text
    return text[:max_len]


def derive_response_format(response_text: str | None) -> str:
    if not response_text or not str(response_text).strip():
        return "empty"
    text = str(response_text).strip()
    if _FENCED_JSON_PATTERN.search(text):
        return "fenced_json"
    if text.startswith("{") or text.startswith("["):
        return "json"
    return "text"


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

    ensure_llm_calls_columns(conn)
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

    ensure_llm_calls_columns(conn)
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
    status = derive_status(error_text)
    error_type = derive_error_type(error_text)
    error_message = derive_error_message(error_text)
    response_format = derive_response_format(response_text)
    conn.execute(
        """
        INSERT INTO llm_calls (
            call_id, run_id, hs_run_id, question_id, call_type, phase,
            model_name, provider, model_id, prompt_text, response_text, parsed_json,
            usage_json, elapsed_ms, prompt_tokens, completion_tokens, total_tokens,
            cost_usd, error_text, status, error_type, error_message, response_format,
            hazard_scores_json, hazard_scores_parse_ok,
            timestamp, iso3, hazard_code, metric
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            status,
            error_type,
            error_message,
            response_format,
            None,
            None,
            datetime.utcnow(),
            iso3,
            hazard_code,
            metric,
        ],
    )
