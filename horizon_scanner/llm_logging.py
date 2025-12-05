import json
import time
from datetime import datetime
from typing import Any, Dict, Optional

from forecaster.llm_logging import _compute_costs_for_usage
from pythia.db.schema import connect, ensure_schema


def _safe_get(d: Dict[str, Any], key: str, default: Any = 0) -> Any:
    try:
        value = d.get(key, default)
        return value if value is not None else default
    except Exception:
        return default


def log_hs_llm_call(
    *,
    hs_run_id: str,
    iso3: str,
    hazard_code: str,
    model_spec: Any,
    prompt_text: str,
    response_text: str,
    usage: Optional[Dict[str, Any]] = None,
    error_text: Optional[str] = None,
    elapsed_ms: Optional[int] = None,
) -> None:
    """Log a Horizon Scanner triage LLM call into ``llm_calls`` as phase hs_triage."""

    usage = usage or {}
    prompt_tokens = int(_safe_get(usage, "prompt_tokens", 0))
    completion_tokens = int(_safe_get(usage, "completion_tokens", 0))
    total_tokens = int(
        _safe_get(usage, "total_tokens", prompt_tokens + completion_tokens)
    )
    cost_usd = float(_safe_get(usage, "cost_usd", 0.0))
    elapsed_ms_val = int(
        _safe_get(usage, "elapsed_ms", elapsed_ms if elapsed_ms is not None else 0)
    )

    input_cost_usd, output_cost_usd, total_cost_usd = _compute_costs_for_usage(
        getattr(model_spec, "provider", None), getattr(model_spec, "model_id", None), usage
    )

    if total_cost_usd <= 0.0:
        total_cost_usd = cost_usd

    call_id = (
        f"hs_{hs_run_id}_{iso3}_{hazard_code}_"
        f"{int(time.time()*1000)}_{abs(hash((prompt_text[:32],)))}"
    )
    timestamp = datetime.utcnow()

    usage_payload = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "elapsed_ms": elapsed_ms_val,
        "cost_usd": total_cost_usd,
        "input_cost_usd": round(input_cost_usd, 6),
        "output_cost_usd": round(output_cost_usd, 6),
        "total_cost_usd": round(total_cost_usd, 6),
        **{
            k: v
            for k, v in usage.items()
            if k
            not in {
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "elapsed_ms",
                "cost_usd",
                "input_cost_usd",
                "output_cost_usd",
                "total_cost_usd",
            }
        },
    }
    usage_json = json.dumps(usage_payload, ensure_ascii=False)

    try:
        con = connect(read_only=False)
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
                hs_run_id,
                hs_run_id,
                "",
                "chat",
                "hs_triage",
                getattr(model_spec, "name", None),
                getattr(model_spec, "provider", None),
                getattr(model_spec, "model_id", None),
                prompt_text,
                response_text,
                json.dumps(usage_payload or {}),
                usage_json,
                elapsed_ms_val,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                total_cost_usd,
                error_text or "",
                timestamp,
                iso3,
                hazard_code,
                None,
            ],
        )
        con.close()
    except Exception as log_exc:  # pragma: no cover - best-effort logging
        print(
            "[warn] Failed to log HS LLM call to llm_calls: "
            f"{type(log_exc).__name__}: {log_exc}"
        )
