from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

from pythia.db.schema import connect, ensure_schema

# Type alias for a generic LLM call function:
#   fn(prompt_text, **kwargs) -> (response_text, usage_dict)
LlmCallFn = Callable[..., Tuple[str, Dict[str, Any]]]


def _safe_get(d: Dict[str, Any], key: str, default: Any = 0) -> Any:
    try:
        value = d.get(key, default)
        return value if value is not None else default
    except Exception:
        return default


def log_hs_llm_call(
    *,
    hs_run_id: str,
    call_type: str,
    model_name: str,
    provider: str,
    model_id: str,
    prompt_text: str,
    low_level_call: LlmCallFn,
    low_level_kwargs: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    question_id: Optional[str] = None,
) -> str:
    """
    Wrap a Horizon Scanner LLM call and log it into the llm_calls table.

    Parameters
    ----------
    hs_run_id : str
        The HS run id (e.g. 'hs_20251128T092046').
    call_type : str
        e.g. 'hs_generation'. Leave room for 'hs_summary' later.
    model_name : str
        Internal name used in Pythia for this model (e.g. 'gemini_flash_hs').
    provider : str
        Provider label (e.g. 'google', 'openai').
    model_id : str
        Concrete model id (e.g. 'gemini-2.5-flash-lite').
    prompt_text : str
        The full prompt to send to the LLM.
    low_level_call : callable
        A function that actually calls the LLM:
            response_text, usage_dict = low_level_call(prompt_text, **kwargs)
        usage_dict may contain keys like:
            'prompt_tokens', 'completion_tokens', 'total_tokens', 'cost_usd'
        but if not, we'll default to zeros.
    low_level_kwargs : dict, optional
        Extra kwargs for the low-level call (e.g. temperature, safety settings).
    run_id : str, optional
        Forecaster run_id if we want to correlate later (usually None for HS).
    question_id : str, optional
        If this HS call is tightly linked to a future question_id. Usually None.

    Returns
    -------
    response_text : str
        The raw response from the LLM.

    Errors
    ------
    - If the LLM call itself fails, we still log an entry with error_text set
      and then re-raise the original exception.
    - Logging failures (e.g. DB issues) do NOT prevent the HS from running;
      they are printed as warnings and swallowed.
    """
    low_level_kwargs = low_level_kwargs or {}

    t_start = time.time()
    response_text: str = ""
    usage: Dict[str, Any] = {}
    error_text: str = ""
    exc_to_raise: Optional[BaseException] = None

    try:
        response_text, usage = low_level_call(prompt_text, **low_level_kwargs)
    except BaseException as exc:  # noqa: BLE001 - re-raise after logging
        exc_to_raise = exc
        error_text = f"{type(exc).__name__}: {str(exc)[:400]}"
    elapsed_ms = int((time.time() - t_start) * 1000)

    prompt_tokens = int(_safe_get(usage, "prompt_tokens", 0))
    completion_tokens = int(_safe_get(usage, "completion_tokens", 0))
    total_tokens = int(
        _safe_get(usage, "total_tokens", prompt_tokens + completion_tokens)
    )
    cost_usd = float(_safe_get(usage, "cost_usd", 0.0))

    call_id = (
        f"hs_{hs_run_id}_{int(time.time()*1000)}_"
        f"{abs(hash((model_name, call_type, prompt_text[:32])))}"
    )
    timestamp = datetime.utcnow()

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
                run_id or "",
                hs_run_id,
                question_id or "",
                call_type,
                model_name,
                provider,
                model_id,
                prompt_text,
                response_text,
                None,
                elapsed_ms,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost_usd,
                error_text,
                timestamp,
            ],
        )
        con.close()
    except Exception as log_exc:  # pragma: no cover - best-effort logging
        print(
            "[warn] Failed to log HS LLM call to llm_calls: "
            f"{type(log_exc).__name__}: {log_exc}"
        )

    if exc_to_raise is not None:
        raise exc_to_raise

    return response_text
