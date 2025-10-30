"""Diagnostics helpers for the IDMC connector."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Mapping


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def diagnostics_dir() -> str:
    """Return the diagnostics directory for IDMC runs."""

    return _ensure_dir(os.path.join("diagnostics", "ingestion", "idmc"))


def connectors_log_path() -> str:
    """Return the shared connectors log path, creating parent directories."""

    return os.path.join(_ensure_dir(os.path.join("diagnostics", "ingestion")), "connectors.jsonl")


def tick() -> float:
    """Return the current monotonic timestamp for timing spans."""

    return time.perf_counter()


def to_ms(seconds: float | None) -> int:
    """Convert a span expressed in seconds to whole milliseconds."""

    if seconds is None:
        return 0
    try:
        return int(round(float(seconds) * 1000))
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 0


def timings_block(**named_spans_ms: float | int) -> Dict[str, int]:
    """Normalise timing spans into a diagnostics-friendly mapping."""

    return {name: int(round(value)) for name, value in named_spans_ms.items()}


def zero_rows_rescue(selectors: Mapping[str, Any], notes: str) -> Dict[str, Any]:
    """Construct a zero-rows diagnostics payload."""

    return {"selectors": dict(selectors), "notes": notes}


def debug_block(**fields: Any) -> Dict[str, Any]:
    """Return a filtered debug payload omitting ``None`` values."""

    return {key: value for key, value in fields.items() if value is not None}


def performance_block(
    *,
    requests: int,
    wire_bytes: int,
    body_bytes: int,
    duration_s: float,
    rows: int,
) -> Dict[str, Any]:
    """Build a normalised performance diagnostics payload."""

    seconds = max(float(duration_s), 0.001)
    throughput = int(round((wire_bytes / 1024.0) / seconds)) if wire_bytes else 0
    records = int(round(rows / seconds)) if rows else 0
    return {
        "requests": int(requests),
        "wire_bytes": int(wire_bytes),
        "body_bytes": int(body_bytes),
        "duration_s": round(seconds, 3),
        "throughput_kibps": throughput,
        "records_per_sec": records,
    }


def rate_limit_block(
    *,
    req_per_sec: float,
    max_concurrency: int,
    retries: int,
    retry_after_events: int,
    retry_after_wait_s: float,
    rate_limit_wait_s: float,
    planned_wait_s: float,
) -> Dict[str, Any]:
    """Describe rate limiting and throttling behaviour for diagnostics."""

    return {
        "req_per_sec": float(req_per_sec),
        "max_concurrency": int(max_concurrency),
        "retries": int(retries),
        "retry_after_events": int(retry_after_events),
        "retry_after_wait_s": round(float(retry_after_wait_s), 3),
        "rate_limit_wait_s": round(float(rate_limit_wait_s), 3),
        "planned_wait_s": round(float(planned_wait_s), 3),
    }


def chunks_block(enabled: bool, entries: Any, *, count: int) -> Dict[str, Any]:
    """Return chunking diagnostics including per-chunk stats."""

    return {
        "enabled": bool(enabled),
        "count": int(count),
        "by_month": list(entries),
    }


def write_connectors_line(payload: Dict[str, Any]) -> None:
    """Append a diagnostics line for the connector run."""

    line = {"connector": "idmc", "ts": int(time.time())}
    line.update(payload)
    with open(connectors_log_path(), "a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def write_sample_preview(name: str, csv_head: str) -> str:
    """Persist a sample preview CSV and return its path."""

    path = os.path.join(diagnostics_dir(), f"{name}_preview.csv")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(csv_head)
    return path


def write_drop_reasons(payload: Mapping[str, int]) -> str:
    """Persist drop-reasons histogram for debugging."""

    path = os.path.join(diagnostics_dir(), "drop_reasons.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def write_unmapped_hazards_preview(frame) -> str | None:
    """Persist a preview CSV for unmapped hazards, if any."""

    if frame is None or getattr(frame, "empty", True):
        return None

    preferred_columns = [
        "iso3",
        "as_of_date",
        "metric",
        "value",
        "displacement_type",
        "hazard_category",
        "hazard_type",
        "hazard_subtype",
        "violence_type",
        "conflict_type",
        "notes",
    ]
    available = [column for column in preferred_columns if column in frame.columns]
    if not available:
        available = list(frame.columns)
    subset = frame.loc[:, available].head(5)

    path = os.path.join(diagnostics_dir(), "unmapped_hazards_preview.csv")
    subset.to_csv(path, index=False)
    return path
