"""Helpers for emitting ingestion diagnostics reports."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, Optional

from .diagnostics_schema import ConnectorRunResult, to_jsonl

SENSITIVE_KEYWORDS = ("token", "key", "password", "authorization")
REDACT_TOKEN = "***"


def _coerce_int(value: Any) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def start_run(connector_id: str, mode: Literal["real", "stub"]) -> Dict[str, Any]:
    """Return a mutable diagnostics context for a connector run."""

    started_at = dt.datetime.now(dt.timezone.utc)
    context: Dict[str, Any] = {
        "connector_id": connector_id,
        "mode": mode,
        "started_at": started_at,
        "timer": time.perf_counter(),
        "http": {
            "2xx": 0,
            "4xx": 0,
            "5xx": 0,
            "retries": 0,
            "rate_limit_remaining": None,
            "last_status": None,
        },
        "counts": {"fetched": 0, "normalized": 0, "written": 0},
        "coverage": {
            "ym_min": None,
            "ym_max": None,
            "as_of_min": None,
            "as_of_max": None,
        },
        "samples": {"top_iso3": [], "top_hazard": []},
        "extras": {},
    }
    return context


def _merge_dict(base: Dict[str, Any], updates: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not updates:
        return dict(base)
    merged = dict(base)
    for key, value in updates.items():
        merged[key] = value
    return merged


def _canonical_status(status: str) -> str:
    text = (status or "").strip().lower()
    if text.startswith("ok"):
        return "ok"
    if text == "warning":
        return "ok"
    if text in {"skipped", "error"}:
        return text
    if not text:
        return "skipped"
    return text


def safe_redact_env(mapping: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a deep copy with sensitive keys replaced by ``***``."""

    def _redact_value(key: str, value: Any) -> Any:
        lowered = key.lower()
        if any(token in lowered for token in SENSITIVE_KEYWORDS):
            return REDACT_TOKEN
        if isinstance(value, Mapping):
            return {sub_key: _redact_value(sub_key, sub_value) for sub_key, sub_value in value.items()}
        if isinstance(value, list):
            redacted_list = []
            for item in value:
                if isinstance(item, Mapping):
                    redacted_list.append({sub_key: _redact_value(sub_key, sub_value) for sub_key, sub_value in item.items()})
                else:
                    redacted_list.append(item)
            return redacted_list
        return value

    return {key: _redact_value(key, value) for key, value in mapping.items()}


def finalize_run(
    context: Mapping[str, Any],
    status: str,
    *,
    reason: Optional[str] = None,
    http: Optional[Mapping[str, Any]] = None,
    counts: Optional[Mapping[str, Any]] = None,
    coverage: Optional[Mapping[str, Any]] = None,
    samples: Optional[Mapping[str, Iterable[Iterable[Any]]]] = None,
    extras: Optional[Mapping[str, Any]] = None,
) -> ConnectorRunResult:
    """Freeze a connector run context into a ``ConnectorRunResult``."""

    started_at = context.get("started_at")
    if isinstance(started_at, dt.datetime):
        start_dt = started_at.astimezone(dt.timezone.utc)
    else:
        start_dt = dt.datetime.now(dt.timezone.utc)
    started_at_utc = start_dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")

    timer = context.get("timer")
    duration_ms = 0
    if isinstance(timer, (int, float)):
        elapsed = time.perf_counter() - float(timer)
        duration_ms = max(0, int(elapsed * 1000))

    base_http = context.get("http") if isinstance(context.get("http"), Mapping) else {}
    http_stats = _merge_dict(dict(base_http), http)

    base_counts = context.get("counts") if isinstance(context.get("counts"), Mapping) else {}
    count_stats = _merge_dict(dict(base_counts), counts)

    base_coverage = context.get("coverage") if isinstance(context.get("coverage"), Mapping) else {}
    coverage_stats = _merge_dict(dict(base_coverage), coverage)

    base_samples = context.get("samples") if isinstance(context.get("samples"), Mapping) else {}
    sample_stats = _merge_dict(dict(base_samples), samples or {})

    base_extras = context.get("extras") if isinstance(context.get("extras"), Mapping) else {}
    merged_extras = _merge_dict(dict(base_extras), extras)
    redacted_extras = safe_redact_env(merged_extras)

    canonical_status = _canonical_status(status)
    canonical_reason = reason.strip() if isinstance(reason, str) and reason.strip() else None

    return ConnectorRunResult(
        connector_id=str(context.get("connector_id")),
        mode=str(context.get("mode") or "real"),
        status=canonical_status,  # type: ignore[arg-type]
        reason=canonical_reason,
        started_at_utc=started_at_utc,
        duration_ms=duration_ms,
        http={
            str(key): (_coerce_int(value) if value is not None else None)
            for key, value in http_stats.items()
        },
        counts={
            str(key): _coerce_int(value)
            for key, value in count_stats.items()
            if value is not None
        },
        coverage={str(key): value for key, value in coverage_stats.items()},
        samples={str(key): list(value) for key, value in sample_stats.items()},
        extras=redacted_extras,
    )


def append_jsonl(path: str | Path, result: ConnectorRunResult) -> None:
    """Append a connector result to a JSONL diagnostics file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(to_jsonl(result))
        handle.write("\n")
