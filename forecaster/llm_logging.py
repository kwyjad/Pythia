from __future__ import annotations

import json
import time
import inspect
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

from pythia.db.schema import connect, ensure_schema
from forecaster.providers import estimate_cost_usd

LlmCallFn = Callable[..., Tuple[str, Dict[str, Any]] | Tuple[str, Dict[str, Any], Any]]


def _safe_get(d: Dict[str, Any], key: str, default: Any = 0) -> Any:
    try:
        v = d.get(key, default)
        return v if v is not None else default
    except Exception:
        return default


async def log_forecaster_llm_call(
    *,
    call_type: str,
    run_id: str,
    question_id: str,
    model_name: str,
    provider: str,
    model_id: str,
    prompt_text: str,
    low_level_call: Optional[LlmCallFn] = None,
    low_level_kwargs: Optional[Dict[str, Any]] = None,
    hs_run_id: Optional[str] = None,
    parsed_json: Optional[Any] = None,
    response_text: Optional[str] = None,
    usage: Optional[Dict[str, Any]] = None,
    error_text: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Wrap a Forecaster LLM call and log it to the llm_calls table.

    Returns (response_text, usage_dict). Any exceptions from the LLM call are
    logged with zeroed usage so callers can continue. Logging failures do NOT
    crash forecasting. Callers may also supply precomputed `response_text` and
    `usage` (e.g., when the LLM call was already executed) by omitting
    `low_level_call`.
    """

    low_level_kwargs = low_level_kwargs or {}
    t_start = time.time()
    response: str = ""
    call_usage: Dict[str, Any] = {}
    error_text_local: str = error_text or ""

    if low_level_call is not None:
        try:
            result = low_level_call(prompt_text, **low_level_kwargs)
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, tuple) and len(result) == 3:
                response, call_usage, provider_error = result
                if provider_error and not error_text_local:
                    error_text_local = str(provider_error)
            else:
                response, call_usage = result  # type: ignore[misc]
        except BaseException as exc:  # noqa: BLE001
            elapsed_ms = int((time.time() - t_start) * 1000)
            call_usage = {
                "elapsed_ms": elapsed_ms,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
            }
            error_text_local = f"{type(exc).__name__}: {str(exc)[:400]}"
        else:
            elapsed_ms = int((time.time() - t_start) * 1000)
            try:
                call_usage = dict(call_usage or {})
            except Exception:
                call_usage = {}
            call_usage.setdefault("elapsed_ms", elapsed_ms)
    else:
        response = response_text or ""
        try:
            call_usage = dict(usage or {})
        except Exception:
            call_usage = {}

    elapsed_ms = int(
        _safe_get(call_usage, "elapsed_ms", int((time.time() - t_start) * 1000))
    )

    usage_error_text = _safe_get(call_usage, "error_text", "")
    if usage_error_text and not error_text_local:
        error_text_local = str(usage_error_text)

    prompt_tokens = int(_safe_get(call_usage, "prompt_tokens", 0))
    completion_tokens = int(_safe_get(call_usage, "completion_tokens", 0))
    total_tokens = int(_safe_get(call_usage, "total_tokens", prompt_tokens + completion_tokens))
    cost_usd = float(_safe_get(call_usage, "cost_usd", 0.0))

    if cost_usd <= 0.0:
        cost_usd = float(
            estimate_cost_usd(
                model_id,
                {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            )
        )

    usage = {
        "elapsed_ms": elapsed_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
        **{
            k: v
            for k, v in call_usage.items()
            if k
            not in {
                "elapsed_ms",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "cost_usd",
            }
        },
    }

    call_id = f"fc_{run_id}_{question_id}_{int(time.time() * 1000)}"
    ts = datetime.utcnow()

    parsed_payload = parsed_json() if callable(parsed_json) else parsed_json

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
                model_name,
                provider,
                model_id,
                prompt_text,
                response_text,
                parsed_json,
                elapsed_ms,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost_usd,
                error_text,
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                call_id,
                run_id,
                hs_run_id or "",
                question_id,
                call_type,
                model_name,
                provider,
                model_id,
                prompt_text,
                response,
                json.dumps(parsed_payload) if parsed_payload is not None else None,
                elapsed_ms,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost_usd,
                error_text_local,
                ts,
            ],
        )
        con.close()
    except Exception as log_exc:  # noqa: BLE001
        print(
            f"[warn] Forecaster failed to log LLM call to llm_calls: {type(log_exc).__name__}: {log_exc}"
        )

    return response, usage
