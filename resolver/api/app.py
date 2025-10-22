#!/usr/bin/env python3
"""
FastAPI wrapper for the resolver.

Endpoints:
  GET /health
  GET /resolve?iso3=PHL&hazard_code=TC&cutoff=2025-09-30
  # or names:
  GET /resolve?country=Philippines&hazard=Tropical%20Cyclone&cutoff=2025-09-30
"""

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
    VALID_BACKENDS,
    normalize_backend,
    resolve_point,
    ym_from_cutoff,
)
from resolver.io import files_locator

app = FastAPI(title="Resolver API", version="0.1.0")
DEFAULT_BACKEND = normalize_backend(os.environ.get("RESOLVER_API_BACKEND"), default="files")

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

    if backend is not None:
        backend_clean = backend.strip().lower()
        if backend_clean not in VALID_BACKENDS:
            raise HTTPException(
                status_code=422,
                detail="Invalid backend; choose from files, db, or auto.",
            )
        backend_choice = backend_clean
    else:
        backend_choice = DEFAULT_BACKEND

    result = resolve_point(
        iso3=iso3_code,
        hazard_code=hz_code,
        cutoff=cutoff,
        series=series,
        metric="in_need",
        backend=backend_choice,
    )

    if not result:
        extra_hint = ""
        if backend_choice in {"files", "csv"}:
            try:
                files_root = files_locator.discover_files_root(
                    os.environ.get("RESOLVER_SNAPSHOTS_DIR")
                )
            except FileNotFoundError as exc:
                extra_hint = f" Files backend root missing: {exc}."
            else:
                table = "facts_deltas" if series == "new" else "facts_resolved"
                df_hint = files_locator.load_table(files_root, table)
                if df_hint.empty:
                    locator_reason = df_hint.attrs.get("locator_reason", "no rows located")
                    extra_hint = (
                        f" Files root {files_root} table {table}: {locator_reason}."
                    )
                else:
                    extra_hint = (
                        f" Files root {files_root} table {table}: rows located but none matched"
                        " iso3/hazard/cutoff."
                    )
        raise HTTPException(
            status_code=404,
            detail=(
                "No data found for "
                f"iso3={iso3_code}, hazard={hz_code}, series={series} at cutoff {cutoff} "
                f"(backend {backend_choice}).{extra_hint}"
            ),
        )

    row_series = (
        str(result.get("series_returned", series)).strip().lower() or series
    )
    value = result.get("value", "")
    try:
        value = int(float(value))
    except Exception:
        pass

    return {
        "ok": True,
        "iso3": iso3_code,
        "country_name": country_name,
        "hazard_code": hz_code,
        "hazard_label": hazard_label,
        "hazard_class": hz_class,
        "cutoff": cutoff,
        "metric": result.get("metric", ""),
        "unit": result.get("unit", "persons"),
        "value": value,
        "as_of_date": result.get("as_of_date", ""),
        "publication_date": result.get("publication_date", ""),
        "publisher": result.get("publisher", ""),
        "source_type": result.get("source_type", ""),
        "source_url": result.get("source_url", ""),
        "doc_title": result.get("doc_title", ""),
        "definition_text": result.get("definition_text", ""),
        "precedence_tier": result.get("precedence_tier", ""),
        "event_id": result.get("event_id", ""),
        "confidence": result.get("confidence", ""),
        "proxy_for": result.get("proxy_for", ""),
        "source": result.get("source", ""),
        "source_dataset": result.get("source_dataset", ""),
        "series_requested": result.get("series_requested", series),
        "series_returned": row_series,
        "ym": result.get("ym", ym),
        "source_id": result.get("source_id", ""),
    }


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

        backend_choice = query.backend or DEFAULT_BACKEND

        result = resolve_point(
            iso3=iso3_code,
            hazard_code=hz_code,
            cutoff=query.cutoff,
            series=query.series,
            metric="in_need",
            backend=backend_choice,
        )

        if not result:
            continue

        result.setdefault("country_name", country_name)
        result.setdefault("hazard_label", hazard_label)
        result.setdefault("hazard_class", hz_class)
        result.setdefault("cutoff", query.cutoff)
        result.setdefault("series_requested", query.series)

        responses.append(ResolveResponseRow.parse_obj(result))

    return responses
