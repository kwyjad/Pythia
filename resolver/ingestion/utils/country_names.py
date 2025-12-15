# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Helpers for resolving DTM country names from ISO3 codes."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from pandas import DataFrame

LOG = logging.getLogger("resolver.ingestion.dtm.country_names")

_COUNTRY_CACHE: Optional[Dict[str, str]] = None


def _build_country_cache(frame: DataFrame) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if frame is None or frame.empty:
        return mapping
    for _, row in frame.iterrows():
        iso3 = str(row.get("ISO3") or row.get("Iso3") or "").strip().upper()
        name = str(row.get("CountryName") or row.get("Country") or "").strip()
        if iso3 and name:
            mapping.setdefault(iso3, name)
    return mapping


def resolve_accept_names(client, iso3_list: Sequence[str]) -> List[str]:
    """Resolve ISO3 codes to the names accepted by the DTM API."""

    values = [str(item).strip() for item in iso3_list if str(item).strip()]
    if not values:
        return []

    global _COUNTRY_CACHE
    if _COUNTRY_CACHE is None:
        try:
            frame = client.get_countries({})  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("Failed to load DTM countries for resolution: %s", exc)
            _COUNTRY_CACHE = {}
        else:
            _COUNTRY_CACHE = _build_country_cache(frame)

    cache = _COUNTRY_CACHE or {}
    resolved: List[str] = []
    for value in values:
        lookup = value.upper()
        name = cache.get(lookup)
        if name:
            resolved.append(name)
        else:
            resolved.append(value)
            if cache:
                LOG.warning("No DTM country name found for ISO3=%s; using provided value", lookup)
    return resolved

__all__ = ["resolve_accept_names"]
