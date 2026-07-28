# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""LLM cost/latency routes: /v1/llm/costs, /v1/llm/costs/summary, /v1/costs/*.

Endpoint functions moved verbatim from pythia.api.app (July 2026
decomposition); shared helpers come from pythia.api.core.
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from pythia.api.core import (
    _con,
    _execute,
    _rows_from_cursor,
    _rows_from_df,
    _test_filter,
)
from resolver.query.costs import (
    build_costs_monthly,
    build_costs_runs,
    build_costs_total,
    build_latencies_runs,
    build_run_runtimes,
)

logger = logging.getLogger(__name__)

router = APIRouter()

def _llm_calls_columns(con) -> set[str]:
    try:
        return {r[0] for r in _execute(con, "DESCRIBE llm_calls", []).fetchall()}
    except Exception:  # noqa: BLE001
        return set()


def _first_present(cols: set[str], *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


@router.get("/v1/llm/costs")
def llm_costs(
    phase: str | None = Query(None, description="Filter by phase (e.g. 'spd_v2', 'hs_triage', 'sibyl')"),
    component: str | None = Query(None, description="Legacy alias for phase"),
    model: str | None = Query(None),
    since: str | None = Query(None),
    limit: int = Query(200, ge=1, le=5000),
    include_test: bool = Query(False),
):
    """
    Return recent LLM call cost/usage rows from llm_calls.

    Column names are resolved against the live table: the canonical schema
    uses timestamp/phase/prompt_tokens (the legacy created_at/component/
    tokens_in set no longer exists — the unresolved references made this
    endpoint 500 on every production DB). Legacy DBs still work via the
    fallback candidates. The multi-KB prompt/response text columns are
    excluded from the row payload.
    """
    con = _con()
    cols = _llm_calls_columns(con)
    ts_col = _first_present(cols, "timestamp", "created_at") or "timestamp"
    phase_col = _first_present(cols, "phase", "component")

    select_cols = [
        c for c in cols
        if c not in ("prompt_text", "response_text", "parsed_json", "usage_json")
    ] or ["*"]
    sql = f"SELECT {', '.join(sorted(select_cols))} FROM llm_calls WHERE 1=1{_test_filter(include_test)}"
    params: list = []

    phase_value = phase or component
    if phase_value and phase_col:
        sql += f" AND {phase_col} = ?"
        params.append(phase_value)
    if model:
        sql += " AND model_name = ?"
        params.append(model)
    if since:
        sql += f" AND {ts_col} >= ?"
        params.append(since)

    sql += f" ORDER BY {ts_col} DESC LIMIT ?"
    params.append(limit)

    return {"rows": _rows_from_cursor(_execute(con, sql, params))}


@router.get("/v1/llm/costs/summary")
def llm_costs_summary(
    component: Optional[str] = Query(
        None, description="Filter by component, e.g. 'HS', 'Researcher', 'Forecaster'"
    ),
    model: Optional[str] = Query(None, description="Filter by model_name"),
    llm_profile: Optional[str] = Query(None, description="Filter by llm_profile, e.g. 'test' or 'prod'"),
    hs_run_id: Optional[str] = Query(None, description="Filter by HS run id"),
    ui_run_id: Optional[str] = Query(None, description="Filter by UI run id"),
    forecaster_run_id: Optional[str] = Query(None, description="Filter by Forecaster run id"),
    since: Optional[str] = Query(
        None,
        description="Filter by created_at >= since (ISO date or datetime string). If omitted, no time filter.",
    ),
    group_by: str = Query(
        "phase,model_name",
        description=(
            "Comma-separated grouping fields: 'phase' (alias 'component'),"
            "'model_name','provider','run_id','hs_run_id' — plus the legacy "
            "'llm_profile'/'ui_run_id'/'forecaster_run_id' on DBs that carry "
            "those columns"
        ),
    ),
    limit: int = Query(1000, ge=1, le=5000),
    include_test: bool = Query(False),
):
    """
    Summarise LLM usage and cost from llm_calls.

    Example:
      - Group by component,model_name,llm_profile since a given date.
      - Inspect total cost_usd and tokens per model or per run.

    Returns aggregated metrics per group:
      - calls
      - tokens_in
      - tokens_out
      - cost_usd
    """
    con = _con()
    cols = _llm_calls_columns(con)
    ts_col = _first_present(cols, "timestamp", "created_at") or "timestamp"

    # Requested field → the column that actually exists on this DB (the
    # canonical schema has phase/run_id; legacy DBs had component/
    # forecaster_run_id/llm_profile). Unknown-to-this-DB fields are
    # rejected explicitly rather than 500ing in the binder.
    field_candidates = {
        "phase": ("phase", "component"),
        "component": ("phase", "component"),
        "model_name": ("model_name",),
        "provider": ("provider",),
        "llm_profile": ("llm_profile",),
        "hs_run_id": ("hs_run_id",),
        "run_id": ("run_id",),
        "ui_run_id": ("ui_run_id",),
        "forecaster_run_id": ("forecaster_run_id", "run_id"),
    }

    def _resolve_field(name: str) -> Optional[str]:
        return _first_present(cols, *field_candidates.get(name, ()))

    group_fields: List[str] = [f.strip() for f in group_by.split(",") if f.strip()]
    if not group_fields:
        group_fields = ["phase", "model_name"]

    invalid = [f for f in group_fields if f not in field_candidates]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid group_by fields: {', '.join(invalid)}")
    resolved_group = []
    for f in group_fields:
        col = _resolve_field(f)
        if col is None:
            raise HTTPException(
                status_code=400,
                detail=f"group_by field {f!r} has no matching column on this DB",
            )
        if col not in resolved_group:
            resolved_group.append(col)

    where_bits = ["1=1"]
    params: dict = {}

    def _add_filter(name: str, value: Optional[str]) -> None:
        if not value:
            return
        col = _resolve_field(name)
        if col is None:
            raise HTTPException(
                status_code=400,
                detail=f"filter {name!r} has no matching column on this DB",
            )
        where_bits.append(f"{col} = :{name}")
        params[name] = value

    _add_filter("component", component)
    _add_filter("model_name", model)
    _add_filter("llm_profile", llm_profile)
    _add_filter("hs_run_id", hs_run_id)
    _add_filter("ui_run_id", ui_run_id)
    _add_filter("forecaster_run_id", forecaster_run_id)
    if since:
        where_bits.append(f"{ts_col} >= :since")
        params["since"] = since

    where_clause = " AND ".join(where_bits)
    group_select = ", ".join(resolved_group)
    group_by_clause = "GROUP BY " + ", ".join(resolved_group)

    select_fields = group_select + ", "
    select_fields += """
        COUNT(*) AS calls,
        SUM(prompt_tokens) AS tokens_in,
        SUM(completion_tokens) AS tokens_out,
        SUM(cost_usd) AS cost_usd
    """

    sql = f"""
      SELECT {select_fields}
      FROM llm_calls
      WHERE {where_clause}{_test_filter(include_test)}
      {group_by_clause}
      ORDER BY cost_usd DESC NULLS LAST
      LIMIT :limit
    """
    params["limit"] = limit

    return {
        "group_by": resolved_group,
        "filters": {
            "component": component,
            "model": model,
            "llm_profile": llm_profile,
            "hs_run_id": hs_run_id,
            "ui_run_id": ui_run_id,
            "forecaster_run_id": forecaster_run_id,
            "since": since,
        },
        "rows": _rows_from_cursor(_execute(con, sql, params)),
    }


@router.get("/v1/costs/total")
def costs_total(track: Optional[int] = Query(None), include_test: bool = Query(False)):
    con = _con()
    try:
        tables = build_costs_total(con, track=track, include_test=include_test)
    except Exception as exc:
        logger.exception("Failed to build total costs")
        raise HTTPException(status_code=500, detail="Failed to build total costs") from exc

    return {"tables": {key: _rows_from_df(df) for key, df in tables.items()}}


@router.get("/v1/costs/monthly")
def costs_monthly(track: Optional[int] = Query(None), include_test: bool = Query(False)):
    con = _con()
    try:
        tables = build_costs_monthly(con, track=track, include_test=include_test)
    except Exception as exc:
        logger.exception("Failed to build monthly costs")
        raise HTTPException(status_code=500, detail="Failed to build monthly costs") from exc

    return {"tables": {key: _rows_from_df(df) for key, df in tables.items()}}


@router.get("/v1/costs/runs")
def costs_runs(track: Optional[int] = Query(None), include_test: bool = Query(False)):
    con = _con()
    try:
        tables = build_costs_runs(con, track=track, include_test=include_test)
    except Exception as exc:
        logger.exception("Failed to build run costs")
        raise HTTPException(status_code=500, detail="Failed to build run costs") from exc

    return {"tables": {key: _rows_from_df(df) for key, df in tables.items()}}


@router.get("/v1/costs/latencies")
def costs_latencies(track: Optional[int] = Query(None), include_test: bool = Query(False)):
    con = _con()
    try:
        df = build_latencies_runs(con, track=track, include_test=include_test)
    except Exception as exc:
        logger.exception("Failed to build run latencies")
        raise HTTPException(status_code=500, detail="Failed to build run latencies") from exc

    return {"rows": _rows_from_df(df)}


@router.get("/v1/costs/run_runtimes")
def costs_run_runtimes(track: Optional[int] = Query(None), include_test: bool = Query(False)):
    con = _con()
    try:
        df = build_run_runtimes(con, track=track, include_test=include_test)
    except Exception as exc:
        logger.exception("Failed to build run runtimes")
        raise HTTPException(status_code=500, detail="Failed to build run runtimes") from exc

    rows = _rows_from_df(df)
    missing_question_p50 = sum(
        1 for row in rows if row.get("question_p50_ms") is None
    )
    missing_country_p50 = sum(
        1 for row in rows if row.get("country_p50_ms") is None
    )
    logger.info(
        "costs/run_runtimes rows=%d missing_question_p50=%d missing_country_p50=%d",
        len(rows),
        missing_question_p50,
        missing_country_p50,
    )
    if not rows:
        logger.warning(
            "costs/run_runtimes empty total_rows=%s missing_elapsed_ms=%s missing_created_at=%s missing_run_id=%s",
            df.attrs.get("total_rows"),
            df.attrs.get("missing_elapsed_ms"),
            df.attrs.get("missing_created_at"),
            df.attrs.get("missing_run_id"),
        )
    return {"rows": rows}
