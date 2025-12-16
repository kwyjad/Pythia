# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Structured ingestion diagnostics schema definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional

import json


HttpStats = Dict[str, Optional[int]]
CountStats = Dict[str, int]
CoverageWindow = Dict[str, Optional[str]]
SampleStats = Dict[str, List[List[Any]]]
Extras = Dict[str, Any]


@dataclass
class ConnectorRunResult:
    """Serializable summary for a single connector execution."""

    connector_id: str
    mode: Literal["real", "stub"]
    status: Literal["ok", "skipped", "error"]
    reason: Optional[str]
    started_at_utc: str
    duration_ms: int
    http: HttpStats = field(default_factory=dict)
    counts: CountStats = field(default_factory=dict)
    coverage: CoverageWindow = field(default_factory=dict)
    samples: SampleStats = field(default_factory=dict)
    extras: Extras = field(default_factory=dict)


def _normalise_samples(samples: Mapping[str, Iterable[Iterable[Any]]]) -> SampleStats:
    normalised: SampleStats = {}
    for key, values in samples.items():
        rows: List[List[Any]] = []
        if isinstance(values, Iterable):
            for item in values:
                if isinstance(item, Mapping):
                    rows.append([item.get("key"), item.get("value")])
                    continue
                if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                    rows.append(list(item))
                else:
                    rows.append([item])
        normalised[key] = rows
    return normalised


def as_dict(result: ConnectorRunResult) -> Dict[str, Any]:
    """Return the dataclass as a JSON-serialisable dictionary."""

    payload: Dict[str, Any] = {
        "connector_id": result.connector_id,
        "mode": result.mode,
        "status": result.status,
        "reason": result.reason,
        "started_at_utc": result.started_at_utc,
        "duration_ms": result.duration_ms,
        "http": dict(result.http),
        "counts": dict(result.counts),
        "coverage": dict(result.coverage),
        "samples": _normalise_samples(result.samples),
        "extras": dict(result.extras),
    }
    return payload


def to_jsonl(result: ConnectorRunResult) -> str:
    """Serialise ``ConnectorRunResult`` to a JSONL line."""

    return json.dumps(as_dict(result), sort_keys=True)
