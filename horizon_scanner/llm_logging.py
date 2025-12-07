from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from pythia.db.schema import connect, ensure_schema

LOG = logging.getLogger(__name__)


def _safe_get(d: Dict[str, Any], key: str, default: Any = 0) -> Any:
    """
    Safe dict getter that tolerates missing or None values.
    """
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

    usage_dict: Dict[str, Any] = dict(usage or {})

    # Preserve elapsed_ms from the provider if present; otherwise default to 0.
    elapsed_ms = int(_safe_get(usage_dict, "elapsed_ms", 0))
    prompt_tokens = int(_safe_get(usage_dict, "prompt_tokens", 0))
    completion_tokens = int(_safe_get(usage_dict, "completion_tokens", 0))
    total_tokens = int(
        _safe_get(usage_dict, "total_tokens", prompt_tokens + completion_tokens)
    )
    cost_usd = float(_safe_get(usage_dict, "cost_usd", 0.0))

    # Bubble up any error text from the provider into error_text if not already set.
    error_text_local = error_text or ""
    usage_error = _safe_get(usage_dict, "error_text", "")
    if usage_error and not error_text_local:
        error_text_local = str(usage_error)

    # For convenience in the debug bundle, keep HS temperature in the usage JSON.
    temperature = getattr(model_spec, "temperature", None)
    if temperature is not None and "temperature" not in usage_dict:
        usage_dict["temperature"] = temperature

    usage_json = json.dumps(usage_dict, ensure_ascii=False)

    iso3_up = (iso3 or "").upper()
    hz_up = (hazard_code or "").upper()

    model_name = getattr(model_spec, "name", None) or getattr(model_spec, "model_id", None)
    provider = getattr(model_spec, "provider", None)
    model_id = getattr(model_spec, "model_id", None)

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
        # Ensure llm_calls exists with the expected schema.
        ensure_schema(con)

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
                timestamp,
                iso3,
                hazard_code,
                metric
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                call_id,
                "",  # HS runs are separate from forecast run IDs
                hs_run_id,
                None,  # No per-question ID at HS triage stage
                "chat",
                "hs_triage",
                model_name,
                provider,
                model_id,
                prompt_text,
                response_text,
                None,  # parsed_json; HS triage currently stored as text
                usage_json,
                elapsed_ms,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost_usd,
                error_text_local,
                ts,
                iso3_up,
                hz_up,
                None,  # metric is not defined at HS stage
            ],
        )
        LOG.debug(
            "HS triage: logged LLM call to llm_calls (call_id=%s, iso3=%s, hazard=%s, tokens=%s).",
            call_id,
            iso3_up,
            hz_up,
            total_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning(
            "HS triage: failed to log LLM call to llm_calls: %s: %s",
            type(exc).__name__,
            exc,
        )
    finally:
        try:
            con.close()
        except Exception:
            pass
