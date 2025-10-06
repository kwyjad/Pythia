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
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from resolver.cli.resolver_cli import (
    current_ym_utc,
    load_registries,
    load_series_for_month,
    normalize_backend,
    resolve_country,
    resolve_hazard,
    select_row,
    VALID_BACKENDS,
    ym_from_cutoff,
)

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
    allow_methods=["GET"],
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

    current_month = ym == current_ym_utc()
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

    df, source_dataset, series_used = load_series_for_month(
        ym, current_month, series, backend=backend_choice
    )

    if df is None or df.empty:
        if series.lower() == "new":
            df, source_dataset, series_used = load_series_for_month(
                ym, current_month, "stock", backend=backend_choice
            )
        if df is None or df.empty:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No data found for "
                    f"{ym}. Expected snapshots/{ym}/facts.parquet, exports data, or DuckDB tables."
                ),
            )

    row = select_row(df, iso3_code, hz_code, cutoff)
    if not row:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No eligible record for iso3={iso3_code}, hazard={hz_code} at cutoff {cutoff}."
            ),
        )

    row_data = dict(row)
    value = row_data.get("value", "")
    try:
        value = int(float(value))
    except Exception:
        pass

    if source_dataset.startswith("db_"):
        source_bucket = "db"
    elif source_dataset in {"snapshot", "snapshot_deltas"}:
        source_bucket = "snapshot"
    elif source_dataset == "monthly_deltas":
        source_bucket = "state"
    else:
        source_bucket = "exports"

    return {
        "ok": True,
        "iso3": iso3_code,
        "country_name": country_name,
        "hazard_code": hz_code,
        "hazard_label": hazard_label,
        "hazard_class": hz_class,
        "cutoff": cutoff,
        "metric": row_data.get("metric", ""),
        "unit": row_data.get("unit", "persons"),
        "value": value,
        "as_of_date": row_data.get("as_of_date", ""),
        "publication_date": row_data.get("publication_date", ""),
        "publisher": row_data.get("publisher", ""),
        "source_type": row_data.get("source_type", ""),
        "source_url": row_data.get("source_url", ""),
        "doc_title": row_data.get("doc_title", ""),
        "definition_text": row_data.get("definition_text", ""),
        "precedence_tier": row_data.get("precedence_tier", ""),
        "event_id": row_data.get("event_id", ""),
        "confidence": row_data.get("confidence", ""),
        "proxy_for": row_data.get("proxy_for", ""),
        "source": source_bucket,
        "source_dataset": source_dataset,
        "series_requested": series,
        "series_returned": row_data.get("series_semantics", series_used),
    }
