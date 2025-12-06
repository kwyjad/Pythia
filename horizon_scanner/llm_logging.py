import json
import logging
from typing import Any, Dict, Optional

from pythia.db.schema import connect

LOG = logging.getLogger(__name__)


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
    """

    try:
        usage = dict(usage or {})
        usage.setdefault("elapsed_ms", None)
        usage_json = json.dumps(usage, ensure_ascii=False)

        con = connect(read_only=False)
        try:
            con.execute(
                """
                INSERT INTO llm_calls (
                    call_type,
                    phase,
                    provider,
                    model_id,
                    temperature,
                    run_id,
                    question_id,
                    iso3,
                    hazard_code,
                    metric,
                    prompt_text,
                    response_text,
                    error_text,
                    usage_json,
                    created_at
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP
                )
                """,
                [
                    "chat",
                    "hs_triage",
                    getattr(model_spec, "provider", None),
                    getattr(model_spec, "model_id", None),
                    getattr(model_spec, "temperature", None),
                    hs_run_id,
                    None,
                    iso3,
                    hazard_code,
                    None,
                    prompt_text,
                    response_text,
                    error_text,
                    usage_json,
                ],
            )
        finally:
            con.close()
    except Exception:
        LOG.exception(
            "Failed to log HS LLM call for hs_run_id=%s iso3=%s hazard=%s",
            hs_run_id,
            iso3,
            hazard_code,
        )
