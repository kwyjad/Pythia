# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""CSV/XLSX download routes (/v1/downloads/*).

Endpoint functions moved verbatim from pythia.api.app (July 2026
decomposition); shared helpers come from pythia.api.core.

All endpoints serve via ``_serve_export``: the DataFrame is built under
the heavy semaphore, spilled to a temp file on the ephemeral disk, freed,
and streamed from disk — peak memory lasts for the build only, not the
whole client download.
"""

import logging
import re
from importlib.util import find_spec

from fastapi import APIRouter, Query
from fastapi.responses import RedirectResponse

from pythia.api.core import (
    _con,
    _concat_cost_tables,
    _serve_export,
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

    return _serve_export(
        lambda: build_forecast_spd_export(_con(), include_test=include_test),
        "pythia_forecasts_export.xlsx",
        fmt="xlsx",
        build_error_detail="Failed to build forecast download export",
        serialize_error_detail="Failed to serialize forecast download export",
        cache_slug="forecasts",
        cache_params={"include_test": include_test},
    )


@router.get("/v1/downloads/forecasts.csv")
def download_forecasts_csv(include_test: bool = Query(False)):
    return _serve_export(
        lambda: build_forecast_spd_export(_con(), include_test=include_test),
        "pythia_forecasts_export.csv",
        build_error_detail="Failed to build forecast download export",
        cache_slug="forecasts",
        cache_params={"include_test": include_test},
    )


@router.get("/v1/downloads/triage.csv")
def download_triage_csv(include_test: bool = Query(False)):
    def _build():
        df = build_triage_export(_con(), include_test=include_test)
        logger.info(
            "Triage download export rows=%s runs=%s iso3=%s",
            len(df),
            df["Run ID"].nunique(dropna=True),
            df["ISO3"].nunique(dropna=True),
        )
        return df

    return _serve_export(
        _build,
        "run_triage_results.csv",
        build_error_detail="Failed to build triage download export",
        cache_slug="triage",
        cache_params={"include_test": include_test},
    )


@router.get("/v1/downloads/total_costs.csv")
def download_total_costs_csv(include_test: bool = Query(False)):
    return _serve_export(
        lambda: _concat_cost_tables(build_costs_total(_con(), include_test=include_test)),
        "total_costs.csv",
        build_error_detail="Failed to build total cost export",
        cache_slug="total_costs",
        cache_params={"include_test": include_test},
    )


@router.get("/v1/downloads/monthly_costs.csv")
def download_monthly_costs_csv(include_test: bool = Query(False)):
    return _serve_export(
        lambda: _concat_cost_tables(build_costs_monthly(_con(), include_test=include_test)),
        "monthly_costs.csv",
        build_error_detail="Failed to build monthly cost export",
        cache_slug="monthly_costs",
        cache_params={"include_test": include_test},
    )


@router.get("/v1/downloads/run_costs.csv")
def download_run_costs_csv(include_test: bool = Query(False)):
    return _serve_export(
        lambda: _concat_cost_tables(build_costs_runs(_con(), include_test=include_test)),
        "run_costs.csv",
        build_error_detail="Failed to build run cost export",
        cache_slug="run_costs",
        cache_params={"include_test": include_test},
    )


@router.get("/v1/downloads/scores_ensemble_mean.csv")
def download_scores_ensemble_mean_csv(include_test: bool = Query(False)):
    return _serve_export(
        lambda: build_ensemble_scores_export(_con(), "ensemble_mean", include_test=include_test),
        "scores_ensemble_mean.csv",
        build_error_detail="Failed to build ensemble_mean scores export",
        cache_slug="scores_ensemble_mean",
        cache_params={"include_test": include_test},
    )


@router.get("/v1/downloads/scores_ensemble_bayesmc.csv")
def download_scores_ensemble_bayesmc_csv(include_test: bool = Query(False)):
    return _serve_export(
        lambda: build_ensemble_scores_export(_con(), "ensemble_bayesmc", include_test=include_test),
        "scores_ensemble_bayesmc.csv",
        build_error_detail="Failed to build ensemble_bayesmc scores export",
        cache_slug="scores_ensemble_bayesmc",
        cache_params={"include_test": include_test},
    )


@router.get("/v1/downloads/scores_model.csv")
def download_scores_model_csv(include_test: bool = Query(False)):
    return _serve_export(
        lambda: build_model_scores_export(_con(), include_test=include_test),
        "scores_model.csv",
        build_error_detail="Failed to build model scores export",
        cache_slug="scores_model",
        cache_params={"include_test": include_test},
    )


@router.get("/v1/downloads/rationales.csv")
def download_rationales_csv(
    hazard: str = Query(..., description="Hazard code filter (e.g. FL, DR, TC)"),
    model: str | None = Query(None, description="Model name filter (e.g. OpenAI, Claude, Gemini Flash)"),
    include_test: bool = Query(False),
):
    parts = ["rationales", hazard.strip().upper()]
    if model:
        safe_model = re.sub(r"[^a-zA-Z0-9_-]", "_", model)
        parts.append(safe_model)
    return _serve_export(
        lambda: build_rationale_export(_con(), hazard_code=hazard, model_name=model, include_test=include_test),
        "_".join(parts) + ".csv",
        build_error_detail="Failed to build rationale export",
        cache_slug="rationales",
        cache_params={"hazard": hazard, "model": model, "include_test": include_test},
    )
