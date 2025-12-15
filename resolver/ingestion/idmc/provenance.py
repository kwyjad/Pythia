# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Provenance helpers for the IDMC connector."""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional

REDACT_KEYS = {"authorization", "cookie", "x-api-key", "proxy-authorization"}


def redact_secrets(obj: Any) -> Any:
    """Recursively redact sensitive values from mappings and sequences."""

    if isinstance(obj, dict):
        redacted: Dict[Any, Any] = {}
        for key, value in obj.items():
            lowered = str(key).lower()
            if (
                lowered in REDACT_KEYS
                or "token" in lowered
                or "secret" in lowered
                or "password" in lowered
            ):
                redacted[key] = "***REDACTED***"
            else:
                redacted[key] = redact_secrets(value)
        return redacted
    if isinstance(obj, list):
        return [redact_secrets(item) for item in obj]
    if isinstance(obj, tuple):  # pragma: no cover - defensive
        return tuple(redact_secrets(item) for item in obj)
    return obj


def write_json(path: str, payload: Dict[str, Any]) -> str:
    """Write a JSON payload to ``path`` and return the path."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def build_provenance(
    *,
    run_meta: Dict[str, Any],
    reachability: Dict[str, Any],
    http_rollup: Dict[str, Any],
    cache_info: Dict[str, Any],
    normalize_stats: Dict[str, Any],
    export_info: Optional[Dict[str, Any]] = None,
    attribution: Optional[Dict[str, Any]] = None,
    notes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a provenance manifest payload for an IDMC run."""

    now = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )
    manifest = {
        "schema_version": 1,
        "generated_at_utc": now,
        "source_system": "IDMC",
        "series": "IDU",
        "attribution": attribution
        or {
            "source_name": "Internal Displacement Monitoring Centre (IDMC) — IDU",
            "terms_url": "",
            "citation": "IDMC Internal Displacement Update (IDU). Accessed via Resolver connector.",
            "license_note": "Subject to IDMC database terms; check rights before redistribution.",
            "method_note": "Curated event-based internal displacement flows; ~180-day rolling window.",
        },
        "run": redact_secrets(run_meta),
        "reachability": redact_secrets(reachability or {}),
        "http": redact_secrets(http_rollup or {}),
        "cache": cache_info or {},
        "normalize": normalize_stats or {},
        "export": export_info or {},
        "notes": notes or {},
    }
    return manifest


__all__ = ["build_provenance", "redact_secrets", "write_json"]
