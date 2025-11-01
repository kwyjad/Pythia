"""Helpers for persisting IDMC zero-row diagnostics."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Iterable, Mapping, MutableMapping

DEFAULT_PATH = "diagnostics/ingestion/idmc/why_zero.json"


def build_payload(
    *,
    token_present: bool,
    countries: Iterable[str] | None = None,
    window: Mapping[str, Any] | None = None,
    filters: Mapping[str, Any] | None = None,
    network_attempted: bool | None = None,
    requests_attempted: int | None = None,
    config_source: str | None = None,
    config_path_used: str | None = None,
    loader_warnings: Iterable[str] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Construct a normalised payload for the why-zero diagnostic."""

    sample_list: list[str] = []
    if countries:
        for item in countries:
            text = str(item).strip()
            if text:
                sample_list.append(text)
    warnings: list[str] = []
    if loader_warnings:
        for entry in loader_warnings:
            text = str(entry).strip()
            if text:
                warnings.append(text)
    payload: MutableMapping[str, Any] = {
        "token_present": bool(token_present),
        "countries_count": len(sample_list),
        "countries_sample": sample_list[:5],
        "window": dict(window or {}),
        "filters": dict(filters or {}),
        "network_attempted": bool(network_attempted) if network_attempted is not None else None,
        "requests_attempted": requests_attempted,
        "config_source": config_source,
        "config_path_used": config_path_used,
        "loader_warnings": warnings,
    }
    if extras:
        payload["extras"] = dict(extras)
    return dict(payload)


def write_why_zero(payload: Dict[str, Any], path: str = DEFAULT_PATH) -> str:
    """Persist the provided diagnostics payload to ``path`` and return it."""

    dest = pathlib.Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return dest.as_posix()


__all__ = ["build_payload", "write_why_zero"]
