"""Pydantic models for the batch resolve API."""
from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, Field, root_validator, validator

from resolver.query.selectors import VALID_BACKENDS, normalize_backend


class ResolveQuery(BaseModel):
    """Request model for a batch resolve query."""

    cutoff: str = Field(..., description="Cut-off date YYYY-MM-DD (23:59 Europe/Istanbul)")
    country: Optional[str] = Field(
        None,
        description="Country name (alternative to iso3)",
    )
    iso3: Optional[str] = Field(
        None,
        description="ISO3 code (alternative to country)",
    )
    hazard: Optional[str] = Field(
        None,
        description="Hazard label (alternative to hazard_code)",
    )
    hazard_code: Optional[str] = Field(
        None,
        description="Hazard code (alternative to hazard)",
    )
    series: str = Field(
        "stock",
        description="Series to return: stock totals or new monthly deltas.",
    )
    backend: Optional[str] = Field(
        None,
        description="Optional backend override: files/csv, db, or auto.",
    )

    @validator("series", pre=True, always=True)
    def _normalise_series(cls, value: Optional[str]) -> str:
        if not value:
            return "stock"
        candidate = str(value).strip().lower()
        return candidate if candidate in {"new", "stock"} else "stock"

    @validator("backend", pre=True, always=True)
    def _normalise_backend(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        backend = str(value).strip().lower()
        if backend == "csv":
            backend = "files"
        if backend not in VALID_BACKENDS:
            # normalize_backend will fall back to default; keep explicit rejection
            raise ValueError("backend must be one of files, db, or auto")
        return normalize_backend(backend, default="files")

    @root_validator
    def _check_identifiers(cls, values: dict) -> dict:
        country, iso3 = values.get("country"), values.get("iso3")
        hazard, hazard_code = values.get("hazard"), values.get("hazard_code")
        if not (country or iso3):
            raise ValueError("provide either country or iso3 for each query")
        if not (hazard or hazard_code):
            raise ValueError("provide either hazard or hazard_code for each query")
        return values


class ResolveResponseRow(BaseModel):
    """Standard response payload for a resolved batch row."""

    ok: bool = True
    iso3: str
    hazard_code: str
    cutoff: str
    value: Optional[Union[int, float, str]] = None
    country_name: Optional[str] = ""
    hazard_label: Optional[str] = ""
    hazard_class: Optional[str] = ""
    metric: Optional[str] = ""
    unit: Optional[str] = "persons"
    as_of_date: Optional[str] = ""
    publication_date: Optional[str] = ""
    publisher: Optional[str] = ""
    source_type: Optional[str] = ""
    source_url: Optional[str] = ""
    doc_title: Optional[str] = ""
    definition_text: Optional[str] = ""
    precedence_tier: Optional[str] = ""
    event_id: Optional[str] = ""
    confidence: Optional[str] = ""
    proxy_for: Optional[str] = ""
    source: Optional[str] = ""
    source_dataset: Optional[str] = ""
    source_id: Optional[str] = ""
    series_requested: Optional[str] = ""
    series_returned: Optional[str] = ""
    series_semantics: Optional[str] = None
    ym: Optional[str] = ""
    fallback_used: Optional[bool] = None

    class Config:
        orm_mode = True
        allow_population_by_field_name = True

