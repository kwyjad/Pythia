# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from pythia.db.schema import connect, ensure_schema
from pythia.db.util import ensure_llm_calls_columns
from resolver.query.debug_ui import (
    _extract_hazard_scores_with_diagnostics,
    _extract_json_candidate,
    _FENCED_JSON_PATTERN,
)

LOG = logging.getLogger(__name__)


def _safe_get(d: Dict[str, Any], key: str, default: Any = 0) -> Any:
    """Safe dict getter that tolerates missing or None values."""
    try:
        v = d.get(key, default)
        return v if v is not None else default
    except Exception:  # noqa: BLE001
        return default


def log_hs_llm_call(
    *,
    hs_run_id: str,
    iso3: str,
    hazard_code: str,
    model_spec: Any,
    prompt_text: str,
    response_text: str,
    usage: Dict[str, Any],
    error_text: Optional[str],
) -> None:
    """
    Log a Horizon Scanner triage LLM call into llm_calls with phase='hs_triage'.

    - Uses the shared llm_calls schema defined in pythia.db.schema.ensure_schema.
    - Populates hs_run_id and (iso3, hazard_code); leaves run_id empty because HS
      runs are decoupled from Forecaster runs.
    - Never raises on logging failure; errors are logged as warnings.
    """

    usage_dict: Dict[str, Any] = {}
    try:
        usage_dict = dict(usage or {})
    except Exception:  # noqa: BLE001
        usage_dict = {}

    # Pull basic usage stats; if elapsed_ms is missing, leave it at 0 (HS already sets it).
    elapsed_ms = int(_safe_get(usage_dict, "elapsed_ms", 0))
    prompt_tokens = int(_safe_get(usage_dict, "prompt_tokens", 0))
    completion_tokens = int(_safe_get(usage_dict, "completion_tokens", 0))
    total_tokens = int(
        _safe_get(usage_dict, "total_tokens", prompt_tokens + completion_tokens)
    )
    cost_usd = float(_safe_get(usage_dict, "cost_usd", 0.0))

    # Bubble up error text, if any.
    error_text_local = error_text or ""
    usage_error_text = _safe_get(usage_dict, "error_text", "")
    if usage_error_text and not error_text_local:
        error_text_local = str(usage_error_text)

    # For convenience, keep HS temperature in the usage JSON (no dedicated column).
    temperature = getattr(model_spec, "temperature", None)
    if temperature is not None and "temperature" not in usage_dict:
        usage_dict["temperature"] = temperature

    iso3_up = (iso3 or "").upper().strip()
    hz_up = (hazard_code or "").upper().strip()

    if hs_run_id:
        usage_dict.setdefault("hs_run_id", hs_run_id)
    usage_dict.setdefault("iso3", iso3_up)
    if hz_up:
        usage_dict.setdefault("hazard_code", hz_up)

    try:
        usage_json = json.dumps(usage_dict, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        usage_json = "{}"

    model_name = getattr(model_spec, "name", None) or getattr(model_spec, "model_id", None)
    provider = getattr(model_spec, "provider", None)
    model_id = getattr(model_spec, "model_id", None)

    error_message = str(error_text_local).strip() if error_text_local else ""
    status = "error" if error_message else "ok"
    error_type = None
    if error_message:
        lowered = error_message.lower()
        if "timeout" in lowered:
            error_type = "timeout"
        elif "rate" in lowered and "limit" in lowered:
            error_type = "rate_limit"
        elif "parse" in lowered:
            error_type = "parse_error"
        else:
            error_type = "provider_error"

    response_format = None
    if response_text:
        if _FENCED_JSON_PATTERN.search(response_text):
            response_format = "fenced_json"
        elif _extract_json_candidate(response_text):
            response_format = "json"
        else:
            response_format = "text"

    hazard_scores: dict[str, float] = {}
    hazard_scores_parse_ok = False
    if status == "ok":
        hazard_scores, _, _, _ = _extract_hazard_scores_with_diagnostics(response_text)
        hazard_scores_parse_ok = bool(hazard_scores)

    hazard_scores_json = None
    if hazard_scores:
        try:
            hazard_scores_json = json.dumps(hazard_scores, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            hazard_scores_json = None

    call_id = f"hs_{hs_run_id}_{iso3_up}_{hz_up}_{int(time.time() * 1000)}"
    ts = datetime.utcnow()

    try:
        con = connect(read_only=False)
    except Exception as exc:  # noqa: BLE001
        LOG.warning(
            "HS triage: failed to connect to DuckDB for llm_calls logging: %s: %s",
            type(exc).__name__,
            exc,
        )
        return

    try:
        # Make sure llm_calls exists with the expected schema.
        ensure_schema(con)
        ensure_llm_calls_columns(con)

        con.execute(
            """
            INSERT INTO llm_calls (
                call_id,
                run_id,
                hs_run_id,
                question_id,
                call_type,
                phase,
                model_name,
                provider,
                model_id,
                prompt_text,
                response_text,
                parsed_json,
                usage_json,
                elapsed_ms,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost_usd,
                error_text,
                status,
                error_type,
                error_message,
                hazard_scores_json,
                hazard_scores_parse_ok,
                response_format,
                timestamp,
                iso3,
                hazard_code,
                metric
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                call_id,
                "",  # HS runs don't have a Forecaster run_id
                hs_run_id,
                None,  # No per-question ID yet at HS stage
                "chat",
                "hs_triage",
                model_name,
                provider,
                model_id,
                prompt_text,
                response_text,
                None,  # parsed_json; HS triage is currently stored as raw text
                usage_json,
                elapsed_ms,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost_usd,
                error_text_local,
                status,
                error_type,
                error_message[:500] if error_message else None,
                hazard_scores_json,
                hazard_scores_parse_ok if hazard_scores_json is not None else None,
                response_format,
                ts,
                iso3_up,
                hz_up,
                None,  # metric not defined at HS stage
            ],
        )

        # Explicit INFO log so we can see that logging actually happened in CI logs.
        LOG.info(
            "HS triage: logged LLM call to llm_calls (call_id=%s, iso3=%s, hazard=%s, tokens=%s, cost_usd=%.6f).",
            call_id,
            iso3_up,
            hz_up,
            total_tokens,
            cost_usd,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning(
            "HS triage: failed to log LLM call to llm_calls for %s/%s: %s: %s",
            iso3_up,
            hz_up,
            type(exc).__name__,
            exc,
        )
    finally:
        try:
            con.close()
        except Exception:
            pass
