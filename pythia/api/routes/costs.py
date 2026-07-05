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

@router.get("/v1/llm/costs")
def llm_costs(
    component: str | None = Query(None),
    model: str | None = Query(None),
    since: str | None = Query(None),
    limit: int = Query(200, ge=1, le=5000),
    include_test: bool = Query(False),
):
    """
    Return recent LLM call cost/usage rows from llm_calls.

    Optional filters:
      - component: "HS" | "Researcher" | "Forecaster" | etc.
      - model: model_name (exact match)
      - since: ISO timestamp (created_at >= since)
    """
    con = _con()
    sql = f"SELECT * FROM llm_calls WHERE 1=1{_test_filter(include_test)}"
    params: list = []

    if component:
        sql += " AND component = ?"
        params.append(component)
    if model:
        sql += " AND model_name = ?"
        params.append(model)
    if since:
        sql += " AND created_at >= ?"
        params.append(since)

    sql += " ORDER BY created_at DESC LIMIT ?"
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
        "component,model_name,llm_profile",
        description="Comma-separated list of grouping fields: any of 'component','model_name','llm_profile','hs_run_id','ui_run_id','forecaster_run_id'",
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

    # Parse and validate group_by
    allowed_fields = {
        "component",
        "model_name",
        "llm_profile",
        "hs_run_id",
        "ui_run_id",
        "forecaster_run_id",
    }
    group_fields: List[str] = [f.strip() for f in group_by.split(",") if f.strip()]
    if not group_fields:
        group_fields = ["component", "model_name", "llm_profile"]

    invalid = [f for f in group_fields if f not in allowed_fields]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid group_by fields: {', '.join(invalid)}")

    # Build WHERE clause
    where_bits = ["1=1"]
    params: dict = {}

    if component:
        where_bits.append("component = :component")
        params["component"] = component
    if model:
        where_bits.append("model_name = :model_name")
        params["model_name"] = model
    if llm_profile:
        where_bits.append("llm_profile = :llm_profile")
        params["llm_profile"] = llm_profile
    if hs_run_id:
        where_bits.append("hs_run_id = :hs_run_id")
        params["hs_run_id"] = hs_run_id
    if ui_run_id:
        where_bits.append("ui_run_id = :ui_run_id")
        params["ui_run_id"] = ui_run_id
    if forecaster_run_id:
        where_bits.append("forecaster_run_id = :forecaster_run_id")
        params["forecaster_run_id"] = forecaster_run_id
    if since:
        where_bits.append("created_at >= :since")
        params["since"] = since

    where_clause = " AND ".join(where_bits)

    # Build SELECT and GROUP BY
    group_select = ", ".join(group_fields) if group_fields else ""
    group_by_clause = ""
    if group_fields:
        group_by_clause = "GROUP BY " + ", ".join(group_fields)

    select_fields = group_select + (", " if group_select else "")
    select_fields += """
        COUNT(*) AS calls,
        SUM(tokens_in) AS tokens_in,
        SUM(tokens_out) AS tokens_out,
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
        "group_by": group_fields,
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
