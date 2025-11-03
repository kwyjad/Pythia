"""Client implementation for the IDMC connector."""
from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlencode

import pandas as pd

from resolver.ingestion.utils.country_utils import load_all_iso3, resolve_countries

from .cache import cache_get, cache_key, cache_put
from .diagnostics import (
    chunks_block as build_chunks_block,
    performance_block as build_performance_block,
    rate_limit_block as build_rate_limit_block,
)
from .chunking import split_by_month
from .config import IdmcConfig
from .http import HttpRequestError, http_get
from .rate_limit import TokenBucket

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "fixtures")
RAW_DIAG_DIR = os.path.join("diagnostics", "ingestion", "idmc", "raw")

_CACHE_LOCKS: Dict[str, threading.Lock] = {}
_CACHE_LOCKS_LOCK = threading.Lock()

LOGGER = logging.getLogger(__name__)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _cache_lock(key: str) -> threading.Lock:
    with _CACHE_LOCKS_LOCK:
        lock = _CACHE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _CACHE_LOCKS[key] = lock
        return lock


def fetch_offline(skip_network: bool = True) -> Dict[str, pd.DataFrame]:
    """Load miniature CSV fixtures for the legacy connector outputs."""

    monthly = pd.read_csv(os.path.join(FIXTURES_DIR, "sample_monthly.csv"))
    annual = pd.read_csv(os.path.join(FIXTURES_DIR, "sample_annual.csv"))
    return {"monthly_flow": monthly, "stock": annual}


def _read_json_fixture(name: str) -> Dict[str, Any]:
    with open(os.path.join(FIXTURES_DIR, name), "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_raw_snapshot(cache_key_value: str, body: bytes) -> str:
    _ensure_dir(RAW_DIAG_DIR)
    path = os.path.join(RAW_DIAG_DIR, f"{cache_key_value}.json")
    with open(path, "wb") as handle:
        handle.write(body)
    return path


def _normalise_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "data" in payload and isinstance(payload["data"], list):
        return payload["data"]
    if "features" in payload and isinstance(payload["features"], list):
        return [feature.get("properties", {}) for feature in payload["features"]]
    if isinstance(payload, list):
        return payload
    return []


def _filter_countries(frame: pd.DataFrame, only_countries: Iterable[str]) -> pd.DataFrame:
    countries = [country.strip().upper() for country in only_countries if country.strip()]
    if not countries:
        return frame
    columns = [col for col in ("iso3", "ISO3", "Country ISO3", "CountryISO3") if col in frame.columns]
    if not columns:
        return frame
    iso_column = columns[0]
    mask = frame[iso_column].astype(str).str.upper().isin(countries)
    return frame.loc[mask]


def _filter_range(frame: pd.DataFrame, start: Optional[date], end: Optional[date]) -> pd.DataFrame:
    if start is None and end is None:
        return frame
    if "displacement_date" not in frame.columns:
        return frame
    try:
        dates = pd.to_datetime(frame["displacement_date"], errors="coerce")
    except Exception:  # pragma: no cover - defensive
        return frame
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= dates.dt.date >= start
    if end is not None:
        mask &= dates.dt.date <= end
    return frame.loc[mask]


def _percentile(values: List[int], pct: float) -> int:
    """Return the percentile for a sorted list of ints."""

    if not values:
        return 0
    if len(values) == 1:
        return values[0]
    rank = pct / 100.0 * (len(values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return values[int(rank)]
    lower_val = values[lower]
    upper_val = values[upper]
    fraction = rank - lower
    return int(round(lower_val + (upper_val - lower_val) * fraction))


def _should_skip_sleep() -> bool:
    raw = os.getenv("IDMC_TEST_NO_SLEEP", "0").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _rate_sleep(duration: float) -> None:
    if duration <= 0:
        return
    if _should_skip_sleep():
        return
    time.sleep(duration)


def _window_from_days(window_days: Optional[int]) -> Tuple[Optional[date], Optional[date]]:
    if window_days is None:
        return None, None
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=max(window_days, 0))
    return start, end


def _chunk_label(start: Optional[date], end: Optional[date]) -> str:
    if start and end:
        if start.year == end.year and start.month == end.month:
            return start.strftime("%Y-%m")
        return f"{start.isoformat()}_{end.isoformat()}"
    if end:
        return f"up_to_{end.isoformat()}"
    if start:
        return f"from_{start.isoformat()}"
    return "full"


def _combine_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    combined = pd.concat(frames, ignore_index=True)
    return combined.reset_index(drop=True)


def _latency_block(attempts_ms: List[int]) -> Dict[str, int]:
    if not attempts_ms:
        return {"p50": 0, "p95": 0, "max": 0}
    sorted_latencies = sorted(attempts_ms)
    return {
        "p50": _percentile(sorted_latencies, 50),
        "p95": _percentile(sorted_latencies, 95),
        "max": max(sorted_latencies),
    }


def fetch_idu_json(
    cfg: IdmcConfig,
    *,
    base_url: Optional[str] = None,
    cache_ttl: Optional[int] = None,
    window_start: Optional[date] = None,
    window_end: Optional[date] = None,
    only_countries: Iterable[str] | None = None,
    skip_network: bool = False,
    rate_limiter: TokenBucket | None = None,
    max_bytes: Optional[int] = None,
    chunk_start: Optional[date] = None,
    chunk_end: Optional[date] = None,
    chunk_label: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch the IDU flat JSON payload for a specific window."""

    base = (base_url or cfg.api.base_url).rstrip("/")
    endpoint = cfg.api.endpoints.get("idus_json", "/data/idus_view_flat")
    query: Dict[str, str] = {}
    if chunk_start:
        query["start"] = chunk_start.isoformat()
    if chunk_end:
        query["end"] = chunk_end.isoformat()
    url = f"{base}{endpoint}"
    if query:
        url = f"{url}?{urlencode(query)}"
    params = {"chunk": chunk_label or "full", **query}
    key = cache_key(url, params=params)
    cache_dir = cfg.cache.dir
    ttl_seconds = cache_ttl if cache_ttl is not None else cfg.cache.ttl_seconds
    cache_path = os.path.join(cache_dir, f"{key}.bin")
    cache_stats: Dict[str, Any] = {
        "dir": cache_dir,
        "key": key,
        "path": cache_path,
        "ttl_seconds": ttl_seconds,
        "hit": False,
        "hits": 0,
        "misses": 0,
    }
    http_info: Dict[str, Any] = {
        "requests": 0,
        "retries": 0,
        "status_last": None,
        "duration_s": 0.0,
        "backoff_s": 0.0,
        "wire_bytes": 0,
        "body_bytes": 0,
        "retry_after_events": 0,
        "retry_after_s": [],
        "rate_limit_wait_s": [],
        "planned_sleep_s": [],
        "attempt_durations_ms": [],
        "latency_ms": {"p50": 0, "p95": 0, "max": 0},
    }
    LOGGER.debug(
        "IDMC request planned: chunk=%s url=%s params=%s",
        chunk_label or "full",
        url,
        query,
    )

    diagnostics: Dict[str, Any] = {
        "mode": "fixture",
        "url": url,
        "cache": cache_stats,
        "http": http_info,
        "filters": {},
        "raw_path": None,
        "chunk": {"start": chunk_start.isoformat() if chunk_start else None, "end": chunk_end.isoformat() if chunk_end else None},
    }

    use_cache_only = skip_network or cfg.cache.force_cache_only
    cache_entry = cache_get(cache_dir, key, None if use_cache_only else ttl_seconds)
    body: bytes | None = None
    body_path: Optional[str] = None
    if cache_entry is not None:
        body = cache_entry.body
        cache_stats["hit"] = True
        cache_stats["hits"] = 1
        cache_stats.update(cache_entry.metadata)
        diagnostics["mode"] = "cache"
        LOGGER.debug("IDMC cache hit for %s", url)
    elif use_cache_only:
        payload = _read_json_fixture("idus_view_flat.json")
        rows = _normalise_rows(payload)
        frame = pd.DataFrame(rows)
        frame = _filter_range(frame, chunk_start or window_start, chunk_end or window_end)
        frame = _filter_countries(frame, only_countries or [])
        filters = {
            "window_start": (chunk_start or window_start).isoformat() if (chunk_start or window_start) else None,
            "window_end": (chunk_end or window_end).isoformat() if (chunk_end or window_end) else None,
            "countries": sorted({c.strip().upper() for c in only_countries or [] if c.strip()}),
            "rows_before": len(rows),
            "rows_after": len(frame),
        }
        diagnostics["filters"] = filters
        diagnostics["mode"] = "fixture"
        diagnostics["reason"] = "cache-miss-cache-only"
        LOGGER.debug(
            "IDMC cache-only fallback: chunk=%s rows=%d filtered=%d",
            chunk_label or "full",
            len(rows),
            len(frame),
        )
        return frame.reset_index(drop=True), diagnostics

    if body is None and not use_cache_only:
        stream_tmp = f"{cache_path}.partial"
        try:
            status, headers, payload_body, http_diag = http_get(
                url,
                timeout=10.0,
                retries=2,
                backoff_s=0.5,
                rate_limiter=rate_limiter,
                max_bytes=max_bytes,
                stream_path=stream_tmp,
            )
            http_info.update(
                {
                    "requests": http_diag.get("attempts", 1),
                    "retries": http_diag.get("retries", 0),
                    "status_last": status,
                    "duration_s": http_diag.get("duration_s", 0.0),
                    "backoff_s": http_diag.get("backoff_s", 0.0),
                    "wire_bytes": http_diag.get("wire_bytes", 0),
                    "body_bytes": http_diag.get("body_bytes", 0),
                    "retry_after_events": len(http_diag.get("retry_after_s", []) or []),
                    "retry_after_s": http_diag.get("retry_after_s", []),
                    "rate_limit_wait_s": http_diag.get("rate_limit_wait_s", []),
                    "planned_sleep_s": http_diag.get("planned_sleep_s", []),
                    "attempt_durations_ms": [int(round(value * 1000)) for value in http_diag.get("attempt_durations_s", [])],
                }
            )
            http_info["latency_ms"] = _latency_block(http_info.get("attempt_durations_ms", []))
            diagnostics["mode"] = "online"
            metadata = {
                "status": status,
                "headers": headers,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            cache_stats["misses"] = 1
            cache_stats.update(metadata)
            with _cache_lock(key):
                if http_diag.get("streamed_to"):
                    final_path = cache_path
                    if os.path.exists(stream_tmp):
                        os.replace(stream_tmp, final_path)
                    cache_put(cache_dir, key, final_path, metadata)
                    body_path = final_path
                else:
                    payload_body = payload_body or b""
                    cache_entry = cache_put(cache_dir, key, payload_body, metadata)
                    body = cache_entry.body or payload_body
            LOGGER.debug(
                "IDMC HTTP fetch complete: status=%s wire_bytes=%s body_bytes=%s",
                status,
                http_diag.get("wire_bytes"),
                http_diag.get("body_bytes"),
            )
        except HttpRequestError as exc:
            http_info.update(
                {
                    "requests": exc.diagnostics.get("attempts", 0),
                    "retries": exc.diagnostics.get("retries", 0),
                    "status_last": exc.diagnostics.get("status"),
                    "duration_s": exc.diagnostics.get("duration_s", 0.0),
                    "backoff_s": exc.diagnostics.get("backoff_s", 0.0),
                    "wire_bytes": exc.diagnostics.get("wire_bytes", 0),
                    "body_bytes": exc.diagnostics.get("body_bytes", 0),
                    "retry_after_events": len(exc.diagnostics.get("retry_after_s", []) or []),
                    "retry_after_s": exc.diagnostics.get("retry_after_s", []),
                    "rate_limit_wait_s": exc.diagnostics.get("rate_limit_wait_s", []),
                    "planned_sleep_s": exc.diagnostics.get("planned_sleep_s", []),
                    "attempt_durations_ms": [
                        int(round(value * 1000))
                        for value in exc.diagnostics.get("attempt_durations_s", [])
                    ],
                }
            )
            http_info["latency_ms"] = _latency_block(http_info.get("attempt_durations_ms", []))
            diagnostics["mode"] = "fixture"
            diagnostics["reason"] = "http-error"
            diagnostics["error"] = exc.diagnostics
            payload = _read_json_fixture("idus_view_flat.json")
            rows = _normalise_rows(payload)
            frame = pd.DataFrame(rows)
            frame = _filter_range(frame, chunk_start or window_start, chunk_end or window_end)
            frame = _filter_countries(frame, only_countries or [])
            filters = {
                "window_start": (chunk_start or window_start).isoformat() if (chunk_start or window_start) else None,
                "window_end": (chunk_end or window_end).isoformat() if (chunk_end or window_end) else None,
                "countries": sorted({c.strip().upper() for c in only_countries or [] if c.strip()}),
                "rows_before": len(rows),
                "rows_after": len(frame),
            }
            diagnostics["filters"] = filters
            if not cache_stats["hit"]:
                cache_stats["misses"] = 1
            return frame.reset_index(drop=True), diagnostics

    if body is None and body_path is None:
        cache_stats.setdefault("misses", 0)
        if not cache_stats["hit"]:
            cache_stats["misses"] = 1
        payload = _read_json_fixture("idus_view_flat.json")
        rows = _normalise_rows(payload)
        frame = pd.DataFrame(rows)
        frame = _filter_range(frame, chunk_start or window_start, chunk_end or window_end)
        frame = _filter_countries(frame, only_countries or [])
        filters = {
            "window_start": (chunk_start or window_start).isoformat() if (chunk_start or window_start) else None,
            "window_end": (chunk_end or window_end).isoformat() if (chunk_end or window_end) else None,
            "countries": sorted({c.strip().upper() for c in only_countries or [] if c.strip()}),
            "rows_before": len(rows),
            "rows_after": len(frame),
        }
        diagnostics["filters"] = filters
        diagnostics.setdefault("reason", "cache-miss")
        LOGGER.debug(
            "IDMC fixture fallback: chunk=%s rows=%d filtered=%d",
            chunk_label or "full",
            len(rows),
            len(frame),
        )
        return frame.reset_index(drop=True), diagnostics

    payload: Dict[str, Any]
    if body is not None:
        payload = json.loads(body.decode("utf-8"))
        diagnostics["raw_path"] = _write_raw_snapshot(key, body)
    else:
        body_path = body_path or cache_path
        with open(body_path, "rb") as handle:
            raw_bytes = handle.read()
        payload = json.loads(raw_bytes.decode("utf-8"))
        diagnostics["raw_path"] = _write_raw_snapshot(key, raw_bytes)

    rows = _normalise_rows(payload)
    frame = pd.DataFrame(rows)
    frame = _filter_range(frame, chunk_start or window_start, chunk_end or window_end)
    frame = _filter_countries(frame, only_countries or [])
    filters = {
        "window_start": (chunk_start or window_start).isoformat() if (chunk_start or window_start) else None,
        "window_end": (chunk_end or window_end).isoformat() if (chunk_end or window_end) else None,
        "countries": sorted({c.strip().upper() for c in only_countries or [] if c.strip()}),
        "rows_before": len(rows),
        "rows_after": len(frame),
    }
    diagnostics["filters"] = filters
    LOGGER.debug(
        "IDMC parsed payload: chunk=%s rows_before=%d rows_after=%d",
        chunk_label or "full",
        len(rows),
        len(frame),
    )
    return frame.reset_index(drop=True), diagnostics


def fetch(
    cfg: IdmcConfig,
    *,
    skip_network: bool = False,
    soft_timeouts: bool = True,  # noqa: ARG001 - future compatibility
    window_days: Optional[int] = 30,
    only_countries: Iterable[str] | None = None,
    base_url: Optional[str] = None,
    cache_ttl: Optional[int] = None,
    rate_per_sec: Optional[float] = None,
    max_concurrency: int = 1,
    max_bytes: Optional[int] = None,
    chunk_by_month: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Return payloads for downstream normalization and diagnostics."""

    if rate_per_sec is not None:
        rate = rate_per_sec
    else:
        raw_rate = os.getenv("IDMC_REQ_PER_SEC", "0.5")
        try:
            rate = float(raw_rate)
        except ValueError:  # pragma: no cover - defensive
            rate = 0.5
    limiter = TokenBucket(rate, sleep_fn=_rate_sleep) if rate and rate > 0 else None

    window_start, window_end = _window_from_days(window_days)
    LOGGER.debug(
        "Resolved IDMC window: start=%s end=%s window_days=%s",
        window_start,
        window_end,
        window_days,
    )
    if max_bytes is None:
        raw_max_bytes = os.getenv("IDMC_MAX_BYTES")
        if raw_max_bytes is not None:
            try:
                max_bytes = int(raw_max_bytes)
            except ValueError:  # pragma: no cover - defensive
                max_bytes = 10 * 1024 * 1024
        if max_bytes is None:
            max_bytes = 10 * 1024 * 1024
    if max_bytes is not None and max_bytes <= 0:
        max_bytes = None
    chunk_ranges: List[Tuple[Optional[date], Optional[date]]] = []
    if chunk_by_month and window_start and window_end:
        chunk_ranges = split_by_month(window_start, window_end)
    if not chunk_ranges:
        chunk_ranges = [(window_start, window_end)]

    config_countries = resolve_countries(cfg.api.countries)
    selected_raw = list(
        dict.fromkeys(
            [
                str(value).strip().upper()
                for value in (only_countries or cfg.api.countries)
                if str(value).strip()
            ]
        )
    )
    selected_countries = (
        resolve_countries(list(only_countries))
        if only_countries is not None
        else config_countries
    )
    master_set = set(load_all_iso3())
    invalid = [code for code in selected_raw if code not in master_set]
    LOGGER.info(
        "IDMC country scope: %d codes (invalid=%d) sample=%s",
        len(selected_countries),
        len(invalid),
        ", ".join(selected_countries[:10]),
    )
    countries = list(selected_countries)

    jobs: List[Tuple[int, Optional[date], Optional[date]]] = [
        (index, start, end)
        for index, (start, end) in enumerate(chunk_ranges)
    ]

    LOGGER.debug(
        "Planned %d chunk(s) for IDMC fetch (chunk_by_month=%s)",
        len(jobs),
        chunk_by_month,
    )

    frames: Dict[int, pd.DataFrame] = {}
    chunk_diags: Dict[int, Dict[str, Any]] = {}

    def _run_chunk(index: int, start: Optional[date], end: Optional[date]) -> Tuple[int, pd.DataFrame, Dict[str, Any]]:
        label = _chunk_label(start, end)
        frame, diag = fetch_idu_json(
            cfg,
            base_url=base_url,
            cache_ttl=cache_ttl,
            window_start=window_start,
            window_end=window_end,
            only_countries=countries,
            skip_network=skip_network,
            rate_limiter=limiter,
            max_bytes=max_bytes,
            chunk_start=start,
            chunk_end=end,
            chunk_label=label,
        )
        LOGGER.debug(
            "Fetched chunk %s: rows=%d mode=%s status=%s",
            label,
            len(frame),
            diag.get("mode"),
            (diag.get("http") or {}).get("status_last"),
        )
        return index, frame, diag

    if max_concurrency and max_concurrency > 1 and len(jobs) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            future_to_index = {
                executor.submit(_run_chunk, index, start, end): index
                for index, start, end in jobs
            }
            for future in concurrent.futures.as_completed(future_to_index):
                idx, frame, diag = future.result()
                frames[idx] = frame
                chunk_diags[idx] = diag
    else:
        for index, start, end in jobs:
            idx, frame, diag = _run_chunk(index, start, end)
            frames[idx] = frame
            chunk_diags[idx] = diag

    ordered_frames = [frames[index] for index in sorted(frames)]
    combined = _combine_frames(ordered_frames)

    total_rows = int(len(combined))
    total_requests = 0
    total_retries = 0
    total_wire_bytes = 0
    total_body_bytes = 0
    total_retry_after_events = 0
    total_retry_after_wait = 0.0
    total_rate_wait = 0.0
    total_duration_s = 0.0
    total_planned_wait = 0.0
    attempt_latencies: List[int] = []
    cache_hits = 0
    cache_misses = 0
    raw_path = None
    chunk_entries: List[Dict[str, Any]] = []
    last_status = None

    for index in sorted(chunk_diags):
        diag = chunk_diags[index]
        http_block = diag.get("http", {})
        cache_block = diag.get("cache", {})
        total_requests += int(http_block.get("requests", 0) or 0)
        total_retries += int(http_block.get("retries", 0) or 0)
        total_wire_bytes += int(http_block.get("wire_bytes", 0) or 0)
        total_body_bytes += int(http_block.get("body_bytes", 0) or 0)
        retry_events = int(http_block.get("retry_after_events", 0) or 0)
        total_retry_after_events += retry_events
        retry_waits = http_block.get("retry_after_s", []) or []
        total_retry_after_wait += float(sum(retry_waits))
        rate_waits = http_block.get("rate_limit_wait_s", []) or []
        total_rate_wait += float(sum(rate_waits))
        planned = http_block.get("planned_sleep_s", []) or []
        total_planned_wait += float(sum(planned))
        duration = float(http_block.get("duration_s", 0.0) or 0.0)
        total_duration_s += duration
        attempt_latencies.extend(http_block.get("attempt_durations_ms", []) or [])
        cache_hits += int(cache_block.get("hits", 0) or 0)
        cache_misses += int(cache_block.get("misses", 0) or 0)
        if diag.get("raw_path"):
            raw_path = diag["raw_path"]
        status = http_block.get("status_last")
        if status is not None:
            last_status = status
        chunk = diag.get("chunk", {})
        chunk_start = chunk.get("start")
        chunk_end = chunk.get("end")
        label = _chunk_label(
            datetime.fromisoformat(chunk_start).date() if chunk_start else None,
            datetime.fromisoformat(chunk_end).date() if chunk_end else None,
        )
        chunk_entries.append(
            {
                "month": label,
                "start": chunk_start,
                "end": chunk_end,
                "rows": int(len(frames[index])),
                "fetch_ms": int(round(duration * 1000)),
                "wire_bytes": int(http_block.get("wire_bytes", 0) or 0),
                "body_bytes": int(http_block.get("body_bytes", 0) or 0),
            }
        )

    total_seconds = max(total_duration_s, 0.0)
    if total_seconds <= 0 and total_requests > 0:
        # fall back to attempts if duration unavailable
        total_seconds = sum(value / 1000.0 for value in attempt_latencies)
    total_seconds = max(total_seconds, 0.001)
    http_summary = {
        "requests": total_requests,
        "retries": total_retries,
        "status_last": last_status,
        "latency_ms": _latency_block(attempt_latencies),
        "cache": {"hits": cache_hits, "misses": cache_misses},
        "wire_bytes": total_wire_bytes,
        "body_bytes": total_body_bytes,
        "retry_after_events": total_retry_after_events,
    }

    data: Dict[str, pd.DataFrame] = {"monthly_flow": combined}

    performance = build_performance_block(
        requests=total_requests,
        wire_bytes=total_wire_bytes,
        body_bytes=total_body_bytes,
        duration_s=total_seconds,
        rows=total_rows,
    )
    rate_limit_info = build_rate_limit_block(
        req_per_sec=float(rate) if rate else 0.0,
        max_concurrency=int(max_concurrency or 1),
        retries=total_retries,
        retry_after_events=total_retry_after_events,
        retry_after_wait_s=total_retry_after_wait,
        rate_limit_wait_s=total_rate_wait,
        planned_wait_s=total_planned_wait,
    )
    chunks_info = build_chunks_block(
        chunk_by_month and len(chunk_ranges) > 1,
        chunk_entries,
        count=len(chunk_ranges),
    )

    diagnostics: Dict[str, Any] = {
        "mode": "online" if any(diag.get("mode") == "online" for diag in chunk_diags.values()) else ("cache" if any(diag.get("mode") == "cache" for diag in chunk_diags.values()) else "fixture"),
        "http": http_summary,
        "cache": {
            "dir": cfg.cache.dir,
            "hits": cache_hits,
            "misses": cache_misses,
        },
        "filters": {
            "window_start": window_start.isoformat() if window_start else None,
            "window_end": window_end.isoformat() if window_end else None,
            "countries": sorted({c.strip().upper() for c in countries if c.strip()}),
        },
        "raw_path": raw_path,
        "performance": performance,
        "rate_limit": rate_limit_info,
        "chunks": chunks_info,
    }

    return data, diagnostics
