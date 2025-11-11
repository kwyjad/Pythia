#!/usr/bin/env python3
"""FastAPI wrapper for the resolver."""

import logging
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from resolver.api.batch_models import ResolveQuery, ResolveResponseRow
from resolver.cli.resolver_cli import (
    load_registries,
    resolve_country,
    resolve_hazard,
)
from resolver.query.selectors import (
    normalize_backend,
    resolve_point,
    resolve_db_url,
    ym_from_cutoff,
)

app = FastAPI(title="Resolver API", version="0.1.0")
LOGGER = logging.getLogger(__name__)
DEFAULT_BACKEND = normalize_backend(
    os.environ.get("RESOLVER_API_BACKEND"), default="files"
)

# Allow localhost by default (tweak as you like)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _normalise_backend_choice(value: Optional[str]) -> str:
    """Return a validated backend identifier or raise HTTP 422."""

    if value is None:
        return DEFAULT_BACKEND

    normalized = normalize_backend(value, default="")
    if not normalized:
        raise HTTPException(
            status_code=422, detail="Invalid backend; choose from files, db, or auto."
        )

    return normalized


def _resolve_query(
    *,
    iso3_code: str,
    hazard_code: str,
    cutoff: str,
    series: str,
    backend_choice: str,
    country_name: str,
    hazard_label: str,
    hazard_class: str,
) -> Optional[dict]:
    """Execute a single resolution request using the selectors module."""

    LOGGER.debug(
        "resolve_api request iso3=%s hazard=%s cutoff=%s series=%s backend=%s",
        iso3_code,
        hazard_code,
        cutoff,
        series,
        backend_choice,
    )

    db_url_override = None
    if backend_choice in {"db", "auto"}:
        db_url_override = resolve_db_url()

    result = resolve_point(
        iso3=iso3_code,
        hazard_code=hazard_code,
        cutoff=cutoff,
        series=series,
        metric="in_need",
        backend=backend_choice,
        db_url=db_url_override,
    )

    if not result:
        LOGGER.debug("resolve_api result=not_found iso3=%s hazard=%s", iso3_code, hazard_code)
        return None

    result.setdefault("country_name", country_name)
    result.setdefault("hazard_label", hazard_label)
    result.setdefault("hazard_class", hazard_class)
    result.setdefault("cutoff", cutoff)
    result.setdefault("series_requested", series)

    LOGGER.debug(
        "resolve_api result=found iso3=%s hazard=%s backend=%s series=%s",
        iso3_code,
        hazard_code,
        backend_choice,
        result.get("series_returned", series),
    )

    return result


@app.get("/health")
def health() -> dict:
    """Simple health endpoint for monitoring."""
    return {"ok": True, "service": "resolver", "version": "0.1.0"}


@app.get("/resolve")
def resolve(
    cutoff: str = Query(..., description="Cut-off date YYYY-MM-DD (23:59 Europe/Istanbul)"),
    country: Optional[str] = Query(None, description="Country name (alternative to iso3)"),
    iso3: Optional[str] = Query(None, description="ISO3 code (alternative to country)"),
    hazard: Optional[str] = Query(None, description="Hazard label (alternative to hazard_code)"),
    hazard_code: Optional[str] = Query(None, description="Hazard code (alternative to hazard)"),
    series: str = Query("stock", description="Series to return: stock totals or new monthly deltas."),
    backend: Optional[str] = Query(None, description="Override data backend: files, db, or auto."),
) -> dict:
    """Resolve the latest figure for the requested country, hazard, and cutoff."""
    try:
        countries, shocks = load_registries()
        country_name, iso3_code = resolve_country(countries, country, iso3)
        hazard_label, hz_code, hz_class = resolve_hazard(shocks, hazard, hazard_code)
    except SystemExit as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        ym = ym_from_cutoff(cutoff)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    backend_choice = _normalise_backend_choice(backend)

    result = _resolve_query(
        iso3_code=iso3_code,
        hazard_code=hz_code,
        cutoff=cutoff,
        series=series,
        backend_choice=backend_choice,
        country_name=country_name,
        hazard_label=hazard_label,
        hazard_class=hz_class,
    )

    if not result:
        raise HTTPException(status_code=404, detail="not found")

    result.setdefault("ym", ym)
    result.setdefault("ok", True)
    return result


@app.post("/resolve_batch", response_model=List[ResolveResponseRow])
def resolve_batch(queries: List[ResolveQuery]) -> List[ResolveResponseRow]:
    """Resolve multiple queries in a single request."""

    if not queries:
        return []

    try:
        countries, shocks = load_registries()
    except SystemExit as exc:  # pragma: no cover - registries missing is an operator error
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    responses: List[ResolveResponseRow] = []

    for query in queries:
        try:
            country_name, iso3_code = resolve_country(countries, query.country, query.iso3)
            hazard_label, hz_code, hz_class = resolve_hazard(
                shocks, query.hazard, query.hazard_code
            )
        except SystemExit:
            continue

        backend_choice = normalize_backend(query.backend, default=DEFAULT_BACKEND)
        result = _resolve_query(
            iso3_code=iso3_code,
            hazard_code=hz_code,
            cutoff=query.cutoff,
            series=query.series,
            backend_choice=backend_choice,
            country_name=country_name,
            hazard_label=hazard_label,
            hazard_class=hz_class,
        )

        if not result:
            continue

        responses.append(ResolveResponseRow.parse_obj(result))

    return responses
