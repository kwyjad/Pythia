"""Client implementation for the IDMC connector."""
from __future__ import annotations

import concurrent.futures
import io
import json
import logging
import math
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple
from urllib.parse import urlencode
import urllib.request

import pandas as pd

from resolver.ingestion.utils.country_utils import (
    read_countries_override_from_env,
    resolve_countries,
)

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

DEFAULT_POSTGREST_COLUMNS = (
    "iso3",
    "figure",
    "displacement_start_date",
    "displacement_end_date",
    "displacement_date",
    "displacement_type",
)
IDU_POSTGREST_LIMIT = 10000
ISO3_BATCH_SIZE = 20
SCHEMA_PROBE_QUERY = (("select", "*"), ("limit", "1"))

_CACHE_LOCKS: Dict[str, threading.Lock] = {}
_CACHE_LOCKS_LOCK = threading.Lock()

LOGGER = logging.getLogger(__name__)

NetworkMode = Literal["live", "cache_only", "fixture"]
NETWORK_MODES: Tuple[NetworkMode, ...] = ("live", "cache_only", "fixture")


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


def _filter_range(
    frame: pd.DataFrame,
    start: Optional[date],
    end: Optional[date],
    *,
    column: str = "displacement_date",
) -> pd.DataFrame:
    if start is None and end is None:
        return frame
    target_column = column if column in frame.columns else None
    if target_column is None and column != "displacement_date":
        target_column = "displacement_date" if "displacement_date" in frame.columns else None
    if target_column is None:
        return frame
    try:
        dates = pd.to_datetime(frame[target_column], errors="coerce")
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


def _batch_iso3(codes: Iterable[str], *, batch_size: int = ISO3_BATCH_SIZE) -> List[List[str]]:
    """Return upper-cased ISO3 batches for safe URL construction."""

    cleaned = [code.strip().upper() for code in codes if code and code.strip()]
    if not cleaned:
        return []
    batches: List[List[str]] = []
    for index in range(0, len(cleaned), max(batch_size, 1)):
        batches.append(cleaned[index : index + batch_size])
    return batches


def _probe_idu_schema(
    cfg: IdmcConfig,
    *,
    base_url: Optional[str] = None,
    network_mode: NetworkMode = "live",
    rate_limiter: TokenBucket | None = None,
) -> Tuple[str, List[str], Dict[str, Any]]:
    """Inspect the IDU view to determine available columns and the date field."""

    default_date_column = "displacement_date"
    default_columns = list(DEFAULT_POSTGREST_COLUMNS)
    diagnostics: Dict[str, Any] = {
        "url": None,
        "status": None,
        "columns": [],
        "error": None,
        "skipped": False,
    }

    if network_mode != "live":
        diagnostics["skipped"] = True
        diagnostics["date_column"] = default_date_column
        diagnostics["columns"] = default_columns
        return default_date_column, default_columns, diagnostics

    base = (base_url or cfg.api.base_url).rstrip("/")
    endpoint = cfg.api.endpoints.get("idus_json", "/data/idus_view_flat")
    query = urlencode(SCHEMA_PROBE_QUERY, safe="*.,()")
    url = f"{base}{endpoint}?{query}"
    diagnostics["url"] = url

    try:
        status, headers, body, http_diag = http_get(
            url,
            timeout=10.0,
            retries=1,
            backoff_s=0.1,
            rate_limiter=rate_limiter,
            headers={"Accept": "application/json"},
        )
    except HttpRequestError as exc:
        diagnostics["error"] = exc.message
        diagnostics["http_error"] = exc.diagnostics
        diagnostics["date_column"] = default_date_column
        diagnostics["columns"] = default_columns
        LOGGER.warning(
            "idmc: schema probe failed with HttpRequestError; defaulting to %s",
            default_date_column,
        )
        return default_date_column, default_columns, diagnostics
    except Exception as exc:  # pragma: no cover - defensive
        diagnostics["error"] = str(exc)
        diagnostics["date_column"] = default_date_column
        diagnostics["columns"] = default_columns
        LOGGER.warning(
            "idmc: schema probe encountered %s; defaulting to %s",
            type(exc).__name__,
            default_date_column,
        )
        return default_date_column, default_columns, diagnostics

    diagnostics["status"] = status
    diagnostics["http"] = http_diag
    diagnostics["status_bucket"] = _http_status_bucket(status)

    columns: Set[str] = set()
    payload: Any = None
    if body:
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            diagnostics["error"] = f"json:{type(exc).__name__}"
        else:
            for row in _normalise_rows(payload):
                if isinstance(row, dict):
                    columns.update(str(key) for key in row.keys())

    diagnostics["columns"] = sorted(columns)

    if not columns:
        diagnostics.setdefault("note", "no-columns")
        diagnostics["date_column"] = default_date_column
        return default_date_column, default_columns, diagnostics

    date_column = default_date_column
    for candidate in ("displacement_date", "event_date", "date"):
        if candidate in columns:
            date_column = candidate
            break
    else:
        if {"displacement_start_date", "displacement_end_date"}.issubset(columns):
            date_column = "displacement_end_date"

    select_columns = [
        column for column in DEFAULT_POSTGREST_COLUMNS if column in columns
    ]
    for candidate in ("event_date", "date"):
        if candidate in columns and candidate not in select_columns:
            select_columns.append(candidate)
    if date_column not in select_columns:
        select_columns.append(date_column)
    if (
        date_column == "displacement_end_date"
        and "displacement_start_date" in columns
        and "displacement_start_date" not in select_columns
    ):
        select_columns.append("displacement_start_date")

    diagnostics["date_column"] = date_column
    return date_column, select_columns or default_columns, diagnostics


def _postgrest_filters(
    *,
    chunk_start: Optional[date],
    chunk_end: Optional[date],
    window_start: Optional[date],
    window_end: Optional[date],
    iso_batch: Optional[List[str]],
    offset: int,
    limit: int,
    date_column: Optional[str],
    select_columns: Iterable[str],
) -> List[Tuple[str, str]]:
    """Build PostgREST query parameters for the IDU endpoint."""

    params: List[Tuple[str, str]] = [
        ("select", ",".join(select_columns)),
        ("limit", str(limit)),
    ]
    if offset:
        params.append(("offset", str(offset)))

    start_bound = chunk_start or window_start
    end_bound = chunk_end or window_end
    if date_column:
        if start_bound:
            params.append((date_column, f"gte.{start_bound.isoformat()}"))
        if end_bound:
            params.append((date_column, f"lte.{end_bound.isoformat()}"))
    if iso_batch:
        joined = ",".join(sorted({code.strip().upper() for code in iso_batch if code.strip()}))
        if joined:
            params.append(("iso3", f"in.({joined})"))
    return params


def _http_status_bucket(status: Optional[int]) -> Optional[str]:
    if status is None:
        return None
    if 200 <= status < 300:
        return "2xx"
    if 400 <= status < 500:
        return "4xx"
    if 500 <= status < 600:
        return "5xx"
    return "other"


def _hdx_fallback_enabled() -> bool:
    raw = os.getenv("IDMC_ALLOW_HDX_FALLBACK", "0").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _hdx_dataset_slug() -> str:
    return os.getenv(
        "IDMC_HDX_DATASET",
        "internal-displacement-monitoring-centre-idmc-idu",
    )


def _hdx_base_url() -> str:
    return os.getenv("HDX_BASE", "https://data.humdata.org").rstrip("/")


def _hdx_package_show_url(dataset: str) -> str:
    base = _hdx_base_url()
    return f"{base}/api/3/action/package_show?id={dataset}"


def _read_csv_from_bytes(payload: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(payload)
    return pd.read_csv(buffer)


def _hdx_download_resource(url: str, *, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "pythia-idmc/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def _hdx_fetch_latest_csv() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dataset = _hdx_dataset_slug()
    package_url = _hdx_package_show_url(dataset)
    diagnostics: Dict[str, Any] = {
        "dataset": dataset,
        "package_url": package_url,
        "resource_url": None,
    }
    try:
        with urllib.request.urlopen(package_url, timeout=30.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # pragma: no cover - network dependent
        diagnostics["error"] = f"package_show_failed:{type(exc).__name__}"
        raise

    if not payload.get("success"):
        diagnostics["error"] = "package_show_unsuccessful"
        raise RuntimeError("HDX package_show unsuccessful")

    result = payload.get("result") or {}
    resources = result.get("resources") or []
    csv_candidates = [
        resource
        for resource in resources
        if str(resource.get("format", "")).lower() == "csv"
        and "idus" in str(resource.get("name", "")).lower()
    ]
    if not csv_candidates:
        diagnostics["error"] = "no_csv_resource"
        raise RuntimeError("HDX IDU CSV resource not found")

    resource = max(
        csv_candidates,
        key=lambda item: item.get("last_modified") or item.get("created") or "",
    )
    resource_url = resource.get("url")
    diagnostics["resource_url"] = resource_url
    if not resource_url:
        diagnostics["error"] = "resource_missing_url"
        raise RuntimeError("HDX CSV resource missing URL")

    payload_bytes = _hdx_download_resource(resource_url)
    frame = _read_csv_from_bytes(payload_bytes)
    return frame, diagnostics


def _hdx_filter(
    frame: pd.DataFrame,
    *,
    iso_batches: List[List[str]],
    chunk_start: Optional[date],
    chunk_end: Optional[date],
    window_start: Optional[date],
    window_end: Optional[date],
) -> pd.DataFrame:
    working = frame.copy()
    if iso_batches:
        allowed = {code for batch in iso_batches for code in batch}
        iso_columns = [
            column
            for column in ("iso3", "ISO3", "Country ISO3", "CountryISO3")
            if column in working.columns
        ]
        if iso_columns:
            iso_column = iso_columns[0]
            working = working[
                working[iso_column].astype(str).str.upper().isin(allowed)
            ]

    start_bound = chunk_start or window_start
    end_bound = chunk_end or window_end
    if start_bound or end_bound:
        for column in [
            "displacement_end_date",
            "displacement_start_date",
            "displacement_date",
        ]:
            if column not in working.columns:
                continue
            dates = pd.to_datetime(working[column], errors="coerce")
            if start_bound:
                working = working[dates >= pd.Timestamp(start_bound)]
            if end_bound:
                working = working[dates <= pd.Timestamp(end_bound)]
            break
    return working.reset_index(drop=True)


def fetch_idu_json(
    cfg: IdmcConfig,
    *,
    base_url: Optional[str] = None,
    cache_ttl: Optional[int] = None,
    window_start: Optional[date] = None,
    window_end: Optional[date] = None,
    only_countries: Iterable[str] | None = None,
    network_mode: NetworkMode = "live",
    rate_limiter: TokenBucket | None = None,
    max_bytes: Optional[int] = None,
    chunk_start: Optional[date] = None,
    chunk_end: Optional[date] = None,
    chunk_label: Optional[str] = None,
    date_column: Optional[str] = None,
    select_columns: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch the IDU flat JSON payload for a specific window."""

    base = (base_url or cfg.api.base_url).rstrip("/")
    endpoint = cfg.api.endpoints.get("idus_json", "/data/idus_view_flat")
    base_params: Dict[str, str] = {
        "chunk": chunk_label or "full",
        "window_start": (chunk_start or window_start).isoformat()
        if (chunk_start or window_start)
        else None,
        "window_end": (chunk_end or window_end).isoformat()
        if (chunk_end or window_end)
        else None,
    }
    cache_dir = cfg.cache.dir
    ttl_seconds = cache_ttl if cache_ttl is not None else cfg.cache.ttl_seconds
    cache_stats: Dict[str, Any] = {
        "dir": cache_dir,
        "ttl_seconds": ttl_seconds,
        "hit": False,
        "hits": 0,
        "misses": 0,
        "entries": [],
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
        "status_counts": {"2xx": 0, "4xx": 0, "5xx": 0, "other": 0},
        "exception_counts": {},
    }
    LOGGER.debug(
        "IDMC request planned: chunk=%s window_start=%s window_end=%s",
        chunk_label or "full",
        base_params.get("window_start"),
        base_params.get("window_end"),
    )

    diagnostics: Dict[str, Any] = {
        "mode": "fixture",
        "url": None,
        "cache": cache_stats,
        "http": http_info,
        "filters": {},
        "raw_path": None,
        "requests": [],
        "chunk": {
            "start": chunk_start.isoformat() if chunk_start else None,
            "end": chunk_end.isoformat() if chunk_end else None,
        },
        "network_mode": network_mode,
        "fallback": None,
    }

    select_list: List[str] = []
    for column in list(select_columns or DEFAULT_POSTGREST_COLUMNS):
        if isinstance(column, str) and column and column not in select_list:
            select_list.append(column)
    if not select_list:
        select_list = list(DEFAULT_POSTGREST_COLUMNS)
    date_filter_column = date_column or "displacement_date"
    diagnostics["date_column"] = date_filter_column
    diagnostics["select_columns"] = list(select_list)

    def _build_frame(rows: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        frame = pd.DataFrame(rows)
        frame = _filter_range(
            frame,
            chunk_start or window_start,
            chunk_end or window_end,
            column=date_filter_column,
        )
        frame = _filter_countries(frame, only_countries or [])
        filters = {
            "window_start": (chunk_start or window_start).isoformat()
            if (chunk_start or window_start)
            else None,
            "window_end": (chunk_end or window_end).isoformat()
            if (chunk_end or window_end)
            else None,
            "countries": sorted(
                {c.strip().upper() for c in (only_countries or []) if c and c.strip()}
            ),
            "rows_before": len(rows),
            "rows_after": len(frame),
        }
        diagnostics["filters"] = filters
        return frame.reset_index(drop=True), filters

    if network_mode == "fixture":
        payload = _read_json_fixture("idus_view_flat.json")
        rows = _normalise_rows(payload)
        frame, filters = _build_frame(rows)
        diagnostics["mode"] = "fixture"
        diagnostics["reason"] = "network-mode-fixture"
        LOGGER.debug(
            "IDMC fixture mode: chunk=%s rows=%d filtered=%d",
            chunk_label or "full",
            filters.get("rows_before", 0),
            filters.get("rows_after", 0),
        )
        return frame, diagnostics

    use_cache_only = network_mode == "cache_only"
    iso_batches = _batch_iso3(only_countries or [])
    rows: List[Dict[str, Any]] = []
    raw_paths: List[str] = []
    request_index = 0
    mode = "cache" if use_cache_only else "online"
    fetch_errors: List[Dict[str, Any]] = []
    chunk_error = False

    def _record_exception(name: str) -> None:
        bucket = diagnostics["http"].setdefault("exception_counts", {})
        bucket[name] = bucket.get(name, 0) + 1

    try:
        for iso_batch in iso_batches or [None]:
            offset = 0
            while True:
                params = [
                    (key, value)
                    for key, value in base_params.items()
                    if value is not None
                ]
                params.extend(
                    _postgrest_filters(
                        chunk_start=chunk_start,
                        chunk_end=chunk_end,
                        window_start=window_start,
                        window_end=window_end,
                        iso_batch=iso_batch,
                        offset=offset,
                        limit=IDU_POSTGREST_LIMIT,
                        date_column=date_filter_column,
                        select_columns=select_list,
                    )
                )
                url = f"{base}{endpoint}?{urlencode(params, safe='.,()')}"
                diagnostics["url"] = url
                cache_params = {
                    **{k: v for k, v in base_params.items() if v is not None},
                    "iso_batch": ",".join(iso_batch) if iso_batch else "*",
                    "offset": offset,
                    "limit": IDU_POSTGREST_LIMIT,
                }
                key = cache_key(url, params=cache_params)
                cache_path = os.path.join(cache_dir, f"{key}.bin")
                cache_entry = cache_get(
                    cache_dir,
                    key,
                    None if use_cache_only else ttl_seconds,
                )
                payload_body: Optional[bytes] = None
                body_path: Optional[str] = None
                status: Optional[int] = None
                headers: Dict[str, str] = {}
                http_diag: Dict[str, Any] = {}
                cache_record = {
                    "key": key,
                    "path": cache_path,
                    "offset": offset,
                    "iso_batch": cache_params["iso_batch"],
                }
                if cache_entry is not None:
                    cache_stats["hits"] += 1
                    cache_stats["hit"] = True
                    cache_record.update(cache_entry.metadata)
                    payload_body = cache_entry.body
                    mode = "cache"
                    diagnostics["requests"].append(
                        {
                            "url": url,
                            "status": cache_entry.metadata.get("status"),
                            "cache": "hit",
                        }
                    )
                elif use_cache_only:
                    cache_stats["misses"] += 1
                    cache_stats["hit"] = False
                    diagnostics["requests"].append(
                        {
                            "url": url,
                            "status": None,
                            "cache": "miss",
                        }
                    )
                    break
                else:
                    cache_stats["misses"] += 1
                    request_index += 1
                    try:
                        LOGGER.debug("IDMC GET %s", url)
                        status, headers, payload_body, http_diag = http_get(
                            url,
                            timeout=10.0,
                            retries=2,
                            backoff_s=0.5,
                            rate_limiter=rate_limiter,
                            max_bytes=max_bytes,
                            stream_path=f"{cache_path}.partial",
                            headers={"Accept": "application/json"},
                        )
                        bucket = _http_status_bucket(status)
                        if bucket:
                            http_info["status_counts"].setdefault(bucket, 0)
                            http_info["status_counts"][bucket] += 1
                        diagnostics["requests"].append(
                            {
                                "url": url,
                                "status": status,
                                "elapsed_ms": int(
                                    round(
                                        (http_diag.get("duration_s") or 0.0) * 1000
                                    )
                                ),
                                "cache": "miss",
                            }
                        )
                        metadata = {
                            "status": status,
                            "headers": headers,
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                        }
                        cache_record.update(metadata)
                        with _cache_lock(key):
                            if http_diag.get("streamed_to"):
                                final_path = cache_path
                                tmp_path = f"{cache_path}.partial"
                                if os.path.exists(tmp_path):
                                    os.replace(tmp_path, final_path)
                                cache_put(cache_dir, key, final_path, metadata)
                                body_path = final_path
                            else:
                                payload_body = payload_body or b""
                                cache_entry = cache_put(
                                    cache_dir, key, payload_body, metadata
                                )
                                payload_body = cache_entry.body or payload_body
                    except HttpRequestError as exc:
                        diagnostics["mode"] = "online"
                        diagnostics["error"] = exc.diagnostics
                        _record_exception(exc.__class__.__name__)
                        error_diag = exc.diagnostics or {}
                        status_error = error_diag.get("status")
                        if status_error is not None:
                            bucket = _http_status_bucket(status_error)
                            if bucket:
                                http_info["status_counts"][bucket] += 1
                            http_info["status_last"] = status_error
                        attempts_error = int(error_diag.get("attempts", 1) or 1)
                        http_info["requests"] += attempts_error
                        http_info["retries"] += int(error_diag.get("retries", 0) or 0)
                        http_info["duration_s"] += float(
                            error_diag.get("duration_s", 0.0) or 0.0
                        )
                        http_info["backoff_s"] += float(
                            error_diag.get("backoff_s", 0.0) or 0.0
                        )
                        http_info["wire_bytes"] += int(
                            error_diag.get("wire_bytes", 0) or 0
                        )
                        http_info["body_bytes"] += int(
                            error_diag.get("body_bytes", 0) or 0
                        )
                        retry_after_err = error_diag.get("retry_after_s", []) or []
                        http_info["retry_after_events"] += len(retry_after_err)
                        http_info["retry_after_s"].extend(retry_after_err)
                        http_info["rate_limit_wait_s"].extend(
                            error_diag.get("rate_limit_wait_s", []) or []
                        )
                        http_info["planned_sleep_s"].extend(
                            error_diag.get("planned_sleep_s", []) or []
                        )
                        attempt_times = error_diag.get("attempt_durations_s", []) or []
                        http_info["attempt_durations_ms"].extend(
                            [int(round(value * 1000)) for value in attempt_times]
                        )
                        cache_record["error"] = exc.__class__.__name__
                        cache_record["status"] = status_error
                        cache_stats["entries"].append(cache_record)
                        diagnostics["requests"].append(
                            {
                                "url": url,
                                "status": status_error,
                                "cache": "miss",
                                "error": exc.__class__.__name__,
                            }
                        )
                        exceptions_list = error_diag.get("exceptions")
                        snippet = None
                        if isinstance(exceptions_list, list) and exceptions_list:
                            last_exc = exceptions_list[-1]
                            if isinstance(last_exc, dict):
                                snippet = (
                                    last_exc.get("message")
                                    or last_exc.get("reason")
                                )
                        fetch_errors.append(
                            {
                                "chunk": chunk_label or "full",
                                "url": url,
                                "iso_batch": cache_params["iso_batch"],
                                "exception": exc.__class__.__name__,
                                "status": status_error,
                                "message": snippet,
                            }
                        )
                        chunk_error = True
                        break
                    except Exception as exc:  # pragma: no cover - defensive
                        _record_exception(type(exc).__name__)
                        if not _hdx_fallback_enabled():
                            raise
                        diagnostics["fallback"] = {
                            "type": "hdx",
                            "reason": f"exception:{type(exc).__name__}",
                        }
                        LOGGER.info(
                            "idmc: HDX fallback enabled and used (%s)",
                            type(exc).__name__,
                        )
                        raise

                cache_stats["entries"].append(cache_record)
                if cache_entry is None and use_cache_only:
                    break

                if payload_body is None and cache_entry is None and not body_path:
                    break

                if payload_body is None and body_path:
                    with open(body_path, "rb") as handle:
                        payload_body = handle.read()

                if payload_body is None:
                    break

                payload = json.loads(payload_body.decode("utf-8"))
                raw_paths.append(_write_raw_snapshot(key, payload_body))
                chunk_rows = _normalise_rows(payload)
                rows.extend(chunk_rows)

                http_info["requests"] += int(http_diag.get("attempts", 1) or 1)
                http_info["retries"] += int(http_diag.get("retries", 0) or 0)
                http_info["status_last"] = status or http_info["status_last"]
                http_info["duration_s"] += float(http_diag.get("duration_s", 0.0) or 0.0)
                http_info["backoff_s"] += float(http_diag.get("backoff_s", 0.0) or 0.0)
                http_info["wire_bytes"] += int(http_diag.get("wire_bytes", 0) or 0)
                http_info["body_bytes"] += int(http_diag.get("body_bytes", 0) or 0)
                retry_after = http_diag.get("retry_after_s", []) or []
                http_info["retry_after_events"] += len(retry_after)
                http_info["retry_after_s"].extend(retry_after)
                http_info["rate_limit_wait_s"].extend(
                    http_diag.get("rate_limit_wait_s", []) or []
                )
                http_info["planned_sleep_s"].extend(
                    http_diag.get("planned_sleep_s", []) or []
                )
                attempts = http_diag.get("attempt_durations_s", []) or []
                http_info["attempt_durations_ms"].extend(
                    [int(round(value * 1000)) for value in attempts]
                )

                if len(chunk_rows) < IDU_POSTGREST_LIMIT:
                    break
                offset += IDU_POSTGREST_LIMIT

            if chunk_error:
                break

        diagnostics["mode"] = mode
    except Exception:
        if not _hdx_fallback_enabled():
            raise
        try:
            fallback_frame, fallback_diag = _hdx_fetch_latest_csv()
        except Exception as fallback_exc:  # pragma: no cover - network dependent
            diagnostics["fallback"] = {
                "type": "hdx",
                "error": str(fallback_exc),
            }
            raise
        diagnostics["fallback"] = {
            "type": "hdx",
            **fallback_diag,
        }
        iso_batches = iso_batches or []
        filtered = _hdx_filter(
            fallback_frame,
            iso_batches=iso_batches,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            window_start=window_start,
            window_end=window_end,
        )
        diagnostics["mode"] = "fallback"
        LOGGER.info(
            "idmc: HDX fallback enabled and used (rows=%d)",
            len(filtered),
        )
        frame, filters = _build_frame(filtered.to_dict("records"))
        diagnostics["filters"] = filters
        diagnostics["raw_path"] = None
        http_info["latency_ms"] = _latency_block(http_info.get("attempt_durations_ms", []))
        diagnostics["http"] = http_info
        diagnostics["cache"] = cache_stats
        return frame, diagnostics

    diagnostics["raw_path"] = raw_paths[-1] if raw_paths else None
    frame, filters = _build_frame(rows)
    diagnostics["filters"] = filters
    diagnostics["fetch_errors"] = fetch_errors
    diagnostics["chunk_error"] = bool(chunk_error)
    http_info["latency_ms"] = _latency_block(http_info.get("attempt_durations_ms", []))
    diagnostics["http"] = http_info
    diagnostics["cache"] = cache_stats
    LOGGER.debug(
        "IDMC parsed payload: chunk=%s rows_before=%d rows_after=%d",
        chunk_label or "full",
        filters.get("rows_before", 0),
        filters.get("rows_after", 0),
    )
    return frame, diagnostics


def fetch(
    cfg: IdmcConfig,
    *,
    network_mode: NetworkMode = "live",
    soft_timeouts: bool = True,  # noqa: ARG001 - future compatibility
    window_start: Optional[date] = None,
    window_end: Optional[date] = None,
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

    if network_mode not in NETWORK_MODES:
        raise ValueError(f"Unsupported IDMC network mode: {network_mode}")

    LOGGER.info("IDMC network mode: %s", network_mode)
    if network_mode != "live":
        LOGGER.warning(
            "Running in %s – no network calls will be made; results may be empty unless cache/fixtures exist.",
            network_mode,
        )

    if rate_per_sec is not None:
        rate = rate_per_sec
    else:
        raw_rate = os.getenv("IDMC_REQ_PER_SEC", "0.5")
        try:
            rate = float(raw_rate)
        except ValueError:  # pragma: no cover - defensive
            rate = 0.5
    limiter = None
    if network_mode == "live" and rate and rate > 0:
        limiter = TokenBucket(rate, sleep_fn=_rate_sleep)

    resolved_start = window_start
    resolved_end = window_end
    if resolved_start and resolved_end and resolved_start > resolved_end:
        LOGGER.debug(
            "Provided window start %s is after end %s; swapping",
            resolved_start,
            resolved_end,
        )
        resolved_start, resolved_end = resolved_end, resolved_start
    if resolved_start is None and resolved_end is None:
        resolved_start, resolved_end = _window_from_days(window_days)
    window_start = resolved_start
    window_end = resolved_end
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

    override_raw = read_countries_override_from_env()
    config_countries = resolve_countries(cfg.api.countries, override_raw)

    selected_countries: List[str] = list(config_countries)
    if only_countries is not None:
        override_list = list(only_countries)
        override_resolved = resolve_countries(override_list)
        if override_resolved:
            selected_countries = override_resolved
        else:
            LOGGER.warning(
                "Provided only_countries filter resolved to zero valid ISO3 codes; retaining %d configured countries",
                len(selected_countries),
            )

    LOGGER.info(
        "IDMC country scope: %d codes (sample=%s)",
        len(selected_countries),
        ", ".join(selected_countries[:10]),
    )
    countries = list(selected_countries)

    if window_start is None and window_end is None and window_days is None:
        LOGGER.warning(
            "IDMC fetch invoked without a date window; returning empty payload",
        )
        empty_frame = pd.DataFrame(
            columns=[
                "iso3",
                "as_of_date",
                "metric",
                "value",
                "series_semantics",
                "source",
            ]
        )
        countries_scope = sorted({c.strip().upper() for c in countries if c.strip()})
        diagnostics = {
            "mode": "offline",
            "http": {
                "requests": 0,
                "retries": 0,
                "status_last": None,
                "latency_ms": {"p50": 0, "p95": 0, "max": 0},
                "wire_bytes": 0,
                "body_bytes": 0,
                "retry_after_events": 0,
            },
            "cache": {"dir": cfg.cache.dir, "hits": 0, "misses": 0},
            "filters": {
                "window_start": None,
                "window_end": None,
                "countries": countries_scope,
                "rows_before": 0,
                "rows_after": 0,
            },
            "performance": build_performance_block(
                requests=0,
                wire_bytes=0,
                body_bytes=0,
                duration_s=0.0,
                rows=0,
            ),
            "rate_limit": build_rate_limit_block(
                req_per_sec=float(rate) if rate else 0.0,
                max_concurrency=int(max_concurrency or 1),
                retries=0,
                retry_after_events=0,
                retry_after_wait_s=0.0,
                rate_limit_wait_s=0.0,
                planned_wait_s=0.0,
            ),
            "chunks": build_chunks_block(False, [], count=0),
            "network_mode": network_mode,
            "http_status_counts": {"2xx": 0, "4xx": 0, "5xx": 0}
            if network_mode == "live"
            else None,
            "raw_path": None,
            "requests_planned": 0,
            "requests_executed": 0,
            "window": {"start": None, "end": None, "window_days": None},
        }
        return {"monthly_flow": empty_frame}, diagnostics

    schema_date_column, select_columns, schema_probe_diag = _probe_idu_schema(
        cfg,
        base_url=base_url,
        network_mode=network_mode,
        rate_limiter=limiter,
    )
    LOGGER.info("Using IDU date column: %s", schema_date_column)

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
            network_mode=network_mode,
            rate_limiter=limiter,
            max_bytes=max_bytes,
            chunk_start=start,
            chunk_end=end,
            chunk_label=label,
            date_column=schema_date_column,
            select_columns=select_columns,
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
    status_counts = {"2xx": 0, "4xx": 0, "5xx": 0, "other": 0}
    rows_before_total = 0
    attempt_latencies: List[int] = []
    cache_hits = 0
    cache_misses = 0
    raw_path = None
    chunk_entries: List[Dict[str, Any]] = []
    last_status = None
    exceptions_by_type: Dict[str, int] = {}
    total_chunk_errors = 0
    all_fetch_errors: List[Dict[str, Any]] = []

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
        status_counts_block = http_block.get("status_counts") or {}
        for bucket in ("2xx", "4xx", "5xx", "other"):
            status_counts[bucket] += int(status_counts_block.get(bucket, 0) or 0)
        status = http_block.get("status_last")
        if status is not None:
            last_status = status
        exceptions_block = http_block.get("exception_counts") or {}
        for name, value in exceptions_block.items():
            try:
                count = int(value)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                count = 0
            exceptions_by_type[name] = exceptions_by_type.get(name, 0) + count
        if diag.get("chunk_error"):
            total_chunk_errors += 1
        fetch_diag_entries = diag.get("fetch_errors") or []
        if isinstance(fetch_diag_entries, list):
            all_fetch_errors.extend(fetch_diag_entries)
        chunk = diag.get("chunk", {})
        chunk_start = chunk.get("start")
        chunk_end = chunk.get("end")
        label = _chunk_label(
            datetime.fromisoformat(chunk_start).date() if chunk_start else None,
            datetime.fromisoformat(chunk_end).date() if chunk_end else None,
        )
        filters_block = diag.get("filters") or {}
        rows_before_total += int(filters_block.get("rows_before", 0) or 0)
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
        "status_counts": status_counts,
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
        "mode": "online"
        if any(diag.get("mode") == "online" for diag in chunk_diags.values())
        else (
            "cache"
            if any(diag.get("mode") == "cache" for diag in chunk_diags.values())
            else "fixture"
        ),
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
            "rows_before": rows_before_total,
            "rows_after": total_rows,
        },
        "raw_path": raw_path,
        "performance": performance,
        "rate_limit": rate_limit_info,
        "chunks": chunks_info,
        "network_mode": network_mode,
        "http_status_counts": status_counts if network_mode == "live" else None,
        "date_column": schema_date_column,
        "select_columns": list(select_columns),
        "schema_probe": schema_probe_diag,
        "chunk_errors": total_chunk_errors,
        "fetch_errors": all_fetch_errors,
        "exceptions_by_type": exceptions_by_type,
        "window": {
            "start": window_start.isoformat() if window_start else None,
            "end": window_end.isoformat() if window_end else None,
            "window_days": window_days,
        },
        "requests_planned": len(jobs),
        "requests_executed": total_requests,
    }

    return data, diagnostics
