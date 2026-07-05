# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""CSV/XLSX download routes (/v1/downloads/*).

Endpoint functions moved verbatim from pythia.api.app (July 2026
decomposition); shared helpers come from pythia.api.core.
"""

import logging
import re
from importlib.util import find_spec
from io import BytesIO

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse, StreamingResponse

from pythia.api.core import (
    _HEAVY_REQUEST_SEMAPHORE,
    _acquire_heavy,
    _con,
    _concat_cost_tables,
    _stream_csv,
)
from resolver.query.costs import (
    build_costs_monthly,
    build_costs_runs,
    build_costs_total,
)
from resolver.query.downloads import (
    build_ensemble_scores_export,
    build_forecast_spd_export,
    build_model_scores_export,
    build_rationale_export,
    build_triage_export,
)

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/v1/downloads/forecasts.xlsx")
def download_forecasts_xlsx(include_test: bool = Query(False)):
    if find_spec("openpyxl") is None:
        logger.warning("openpyxl missing; falling back to CSV export")
        return RedirectResponse(url="/v1/downloads/forecasts.csv", status_code=307)

    _acquire_heavy()
    try:
        con = _con()
        try:
            df = build_forecast_spd_export(con, include_test=include_test)
        except Exception as exc:
            logger.exception("Failed to build forecast download export")
            raise HTTPException(status_code=500, detail="Failed to build forecast download export") from exc

        buffer = BytesIO()
        try:
            df.to_excel(buffer, index=False, engine="openpyxl")
        except Exception as exc:
            logger.exception("Failed to serialize forecast download export")
            raise HTTPException(status_code=500, detail="Failed to serialize forecast download export") from exc

        buffer.seek(0)
        headers = {"Content-Disposition": 'attachment; filename="pythia_forecasts_export.xlsx"'}
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers,
        )
    finally:
        _HEAVY_REQUEST_SEMAPHORE.release()


@router.get("/v1/downloads/forecasts.csv")
def download_forecasts_csv(include_test: bool = Query(False)):
    _acquire_heavy()
    try:
        con = _con()
        df = build_forecast_spd_export(con, include_test=include_test)
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build forecast download export")
        raise HTTPException(status_code=500, detail="Failed to build forecast download export") from exc
    return _stream_csv(df, "pythia_forecasts_export.csv")


@router.get("/v1/downloads/triage.csv")
def download_triage_csv(include_test: bool = Query(False)):
    _acquire_heavy()
    try:
        con = _con()
        df = build_triage_export(con, include_test=include_test)
        logger.info(
            "Triage download export rows=%s runs=%s iso3=%s",
            len(df),
            df["Run ID"].nunique(dropna=True),
            df["ISO3"].nunique(dropna=True),
        )
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build triage download export")
        raise HTTPException(status_code=500, detail="Failed to build triage download export") from exc
    return _stream_csv(df, "run_triage_results.csv")


@router.get("/v1/downloads/total_costs.csv")
def download_total_costs_csv(include_test: bool = Query(False)):
    _acquire_heavy()
    try:
        con = _con()
        tables = build_costs_total(con, include_test=include_test)
        df = _concat_cost_tables(tables)
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build total cost export")
        raise HTTPException(status_code=500, detail="Failed to build total cost export") from exc
    return _stream_csv(df, "total_costs.csv")


@router.get("/v1/downloads/monthly_costs.csv")
def download_monthly_costs_csv(include_test: bool = Query(False)):
    _acquire_heavy()
    try:
        con = _con()
        tables = build_costs_monthly(con, include_test=include_test)
        df = _concat_cost_tables(tables)
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build monthly cost export")
        raise HTTPException(status_code=500, detail="Failed to build monthly cost export") from exc
    return _stream_csv(df, "monthly_costs.csv")


@router.get("/v1/downloads/run_costs.csv")
def download_run_costs_csv(include_test: bool = Query(False)):
    _acquire_heavy()
    try:
        con = _con()
        tables = build_costs_runs(con, include_test=include_test)
        df = _concat_cost_tables(tables)
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build run cost export")
        raise HTTPException(status_code=500, detail="Failed to build run cost export") from exc
    return _stream_csv(df, "run_costs.csv")


@router.get("/v1/downloads/scores_ensemble_mean.csv")
def download_scores_ensemble_mean_csv(include_test: bool = Query(False)):
    _acquire_heavy()
    try:
        con = _con()
        df = build_ensemble_scores_export(con, "ensemble_mean", include_test=include_test)
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build ensemble_mean scores export")
        raise HTTPException(status_code=500, detail="Failed to build ensemble_mean scores export") from exc
    return _stream_csv(df, "scores_ensemble_mean.csv")


@router.get("/v1/downloads/scores_ensemble_bayesmc.csv")
def download_scores_ensemble_bayesmc_csv(include_test: bool = Query(False)):
    _acquire_heavy()
    try:
        con = _con()
        df = build_ensemble_scores_export(con, "ensemble_bayesmc", include_test=include_test)
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build ensemble_bayesmc scores export")
        raise HTTPException(status_code=500, detail="Failed to build ensemble_bayesmc scores export") from exc
    return _stream_csv(df, "scores_ensemble_bayesmc.csv")


@router.get("/v1/downloads/scores_model.csv")
def download_scores_model_csv(include_test: bool = Query(False)):
    _acquire_heavy()
    try:
        con = _con()
        df = build_model_scores_export(con, include_test=include_test)
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build model scores export")
        raise HTTPException(status_code=500, detail="Failed to build model scores export") from exc
    return _stream_csv(df, "scores_model.csv")


@router.get("/v1/downloads/rationales.csv")
def download_rationales_csv(
    hazard: str = Query(..., description="Hazard code filter (e.g. FL, DR, TC)"),
    model: str | None = Query(None, description="Model name filter (e.g. OpenAI, Claude, Gemini Flash)"),
    include_test: bool = Query(False),
):
    _acquire_heavy()
    try:
        con = _con()
        df = build_rationale_export(con, hazard_code=hazard, model_name=model, include_test=include_test)
    except Exception as exc:
        _HEAVY_REQUEST_SEMAPHORE.release()
        logger.exception("Failed to build rationale export")
        raise HTTPException(status_code=500, detail="Failed to build rationale export") from exc
    parts = ["rationales", hazard.strip().upper()]
    if model:
        safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model)
        parts.append(safe_model)
    return _stream_csv(df, "_".join(parts) + ".csv")
