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

from resolver.ingestion._shared.feature_flags import getenv_bool
from resolver.ingestion.utils.country_utils import (
    read_countries_override_from_env,
    resolve_countries,
)

from .cache import cache_get, cache_key, cache_put
from .diagnostics import (
    chunks_block as build_chunks_block,
    performance_block as build_performance_block,
    rate_limit_block as build_rate_limit_block,
    serialize_http_status_counts,
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
ISO3_BATCH_SIZE = 25
SCHEMA_PROBE_QUERY = (("select", "*"), ("limit", "1"))

DEFAULT_CONNECT_TIMEOUT_S = 5.0
DEFAULT_READ_TIMEOUT_S = 25.0

_CACHE_LOCKS: Dict[str, threading.Lock] = {}
_CACHE_LOCKS_LOCK = threading.Lock()

LOGGER = logging.getLogger(__name__)

NetworkMode = Literal["live", "cache_only", "fixture"]
NETWORK_MODES: Tuple[NetworkMode, ...] = ("live", "cache_only", "fixture")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _trim_url(url: str, limit: int = 160) -> str:
    if len(url) <= limit:
        return url
    return url[: max(limit - 3, 1)] + "..."


def _env_timeout(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw.strip())
    except (TypeError, ValueError):  # pragma: no cover - defensive
        LOGGER.warning("Invalid %s=%r; using default %.1f", name, raw, default)
        return default
    if value < 0:
        LOGGER.warning("Negative %s=%r; using default %.1f", name, raw, default)
        return default
    return float(value)


def _http_timeouts() -> Tuple[float, float]:
    connect = _env_timeout("IDMC_HTTP_CONNECT_TIMEOUT_S", DEFAULT_CONNECT_TIMEOUT_S)
    read = _env_timeout("IDMC_HTTP_READ_TIMEOUT_S", DEFAULT_READ_TIMEOUT_S)
    if read <= 0:
        read = DEFAULT_READ_TIMEOUT_S
    return connect, read


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


def should_return_empty(
    window_start: Optional[date], window_end: Optional[date], window_days: Optional[int]
) -> bool:
    """Return ``True`` when the caller did not request any time window."""

    return window_start is None and window_end is None and window_days is None


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


def _http_status_bucket(status: Optional[int | str]) -> Optional[str]:
    if isinstance(status, str):
        if status == "timeout":
            return "timeout"
        return "other"
    if status is None:
        return None
    if 200 <= status < 300:
        return "2xx"
    if 400 <= status < 500:
        return "4xx"
    if 500 <= status < 600:
        return "5xx"
    return "other"


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
    allow_fallback: bool = False,
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
        "status_counts": {"2xx": 0, "4xx": 0, "5xx": 0, "other": 0, "timeout": 0},
        "exception_counts": {},
        "requests_ok_2xx": 0,
        "requests_4xx": 0,
        "requests_5xx": 0,
        "requests_other": 0,
        "timeouts": 0,
        "other_exceptions": 0,
        "last_error": None,
        "last_error_url": None,
        "last_exception": None,
    }
    LOGGER.debug(
        "IDMC request planned: chunk=%s window_start=%s window_end=%s",
        chunk_label or "full",
        base_params.get("window_start"),
        base_params.get("window_end"),
    )

    diagnostics: Dict[str, Any] = {
        "mode": network_mode,
        "url": None,
        "cache": cache_stats,
        "http": http_info,
        "filters": {},
        "raw_path": None,
        "requests": [],
        "attempts": [],
        "chunk": {
            "start": chunk_start.isoformat() if chunk_start else None,
            "end": chunk_end.isoformat() if chunk_end else None,
        },
        "network_mode": network_mode,
        "fallback": None,
        "fallback_allowed": allow_fallback,
        "fallback_used": False,
        "http_attempt_summary": {
            "planned": 0,
            "ok_2xx": 0,
            "status_4xx": 0,
            "status_5xx": 0,
            "status_other": 0,
            "timeouts": 0,
            "other_exceptions": 0,
        },
    }

    connect_timeout_s, read_timeout_s = _http_timeouts()
    diagnostics["http_timeouts"] = {
        "connect_s": connect_timeout_s,
        "read_s": read_timeout_s,
    }
    zero_row_reasons: Dict[str, str] = {}

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
                cached_entry = cache_entry
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
                refresh_live_chunk = (
                    network_mode == "live"
                    and isinstance(chunk_label, str)
                    and chunk_label
                    and chunk_label != "full"
                )
                if cache_entry is not None and not refresh_live_chunk:
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
                    diagnostics["http_attempt_summary"]["planned"] += 1
                    try:
                        LOGGER.debug("IDMC GET %s", url)
                        status, headers, payload_body, http_diag = http_get(
                            url,
                            timeout=(connect_timeout_s, read_timeout_s),
                            retries=2,
                            backoff_s=0.5,
                            rate_limiter=rate_limiter,
                            max_bytes=max_bytes,
                            stream_path=f"{cache_path}.partial",
                        )
                        bucket = _http_status_bucket(status)
                        if bucket:
                            http_info["status_counts"].setdefault(bucket, 0)
                            http_info["status_counts"][bucket] += 1
                        http_info["status_last"] = status
                        summary_counters = diagnostics["http_attempt_summary"]
                        if 200 <= status < 300:
                            http_info["requests_ok_2xx"] += 1
                            summary_counters["ok_2xx"] += 1
                        elif 400 <= status < 500:
                            http_info["requests_4xx"] += 1
                            summary_counters["status_4xx"] += 1
                        elif 500 <= status < 600:
                            http_info["requests_5xx"] += 1
                            summary_counters["status_5xx"] += 1
                        else:
                            http_info["requests_other"] += 1
                            summary_counters["status_other"] += 1
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
                                "via": "http_get",
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
                        status_display: int | str | None = status_error
                        if error_diag.get("timeout"):
                            status_display = "timeout"
                        bucket = _http_status_bucket(status_display)
                        if bucket:
                            http_info["status_counts"].setdefault(bucket, 0)
                            http_info["status_counts"][bucket] += 1
                        http_info["status_last"] = status_display
                        summary_counters = diagnostics["http_attempt_summary"]
                        if status_display == "timeout":
                            http_info["timeouts"] += 1
                            summary_counters["timeouts"] += 1
                        elif isinstance(status_display, int):
                            if 200 <= status_display < 300:
                                http_info["requests_ok_2xx"] += 1
                                summary_counters["ok_2xx"] += 1
                            elif 400 <= status_display < 500:
                                http_info["requests_4xx"] += 1
                                summary_counters["status_4xx"] += 1
                            elif 500 <= status_display < 600:
                                http_info["requests_5xx"] += 1
                                summary_counters["status_5xx"] += 1
                            else:
                                http_info["requests_other"] += 1
                                summary_counters["status_other"] += 1
                        else:
                            http_info["requests_other"] += 1
                            summary_counters["status_other"] += 1
                        if not error_diag.get("timeout") and status_error is None:
                            http_info["other_exceptions"] += 1
                            summary_counters["other_exceptions"] += 1
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
                        cache_record["status"] = status_display
                        cache_stats["entries"].append(cache_record)
                        diagnostics["requests"].append(
                            {
                                "url": url,
                                "status": status_display,
                                "cache": "miss",
                                "error": exc.__class__.__name__,
                                "via": "http_get",
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
                        chunk_name = chunk_label or "full"
                        status_display: int | str | None = status_error
                        if error_diag.get("timeout"):
                            status_display = "timeout"
                        attempt_entry = {
                            "chunk": chunk_name,
                            "via": "http_get",
                            "status": status_display,
                            "error": exc.__class__.__name__,
                        }
                        if status_display == "timeout":
                            attempt_entry["zero_rows_reason"] = "timeout"
                            zero_row_reasons[chunk_name] = "timeout"
                        diagnostics["attempts"].append(attempt_entry)
                        fetch_errors.append(
                            {
                                "chunk": chunk_name,
                                "url": url,
                                "iso_batch": cache_params["iso_batch"],
                                "exception": exc.__class__.__name__,
                                "status": status_display,
                                "message": snippet,
                            }
                        )
                        http_info["last_error"] = {
                            "type": exc.__class__.__name__,
                            "message": snippet or str(exc),
                        }
                        http_info["last_error_url"] = url
                        http_info["last_exception"] = exc.__class__.__name__
                        LOGGER.debug(
                            "idmc: chunk=%s via=http_get status=%s error=%s url=%s",
                            chunk_name,
                            status_display,
                            exc.__class__.__name__,
                            _trim_url(url),
                        )
                        if allow_fallback:
                            fallback_diag_entry: Dict[str, Any] = {
                                "type": "hdx",
                                "reason": "http_error",
                                "status": status_display,
                                "chunk": chunk_name,
                                "used": False,
                            }
                            try:
                                fallback_frame, fallback_diag = _hdx_fetch_latest_csv()
                                fallback_diag_entry.update(fallback_diag)
                                filtered = _hdx_filter(
                                    fallback_frame,
                                    iso_batches=iso_batches or [],
                                    chunk_start=chunk_start,
                                    chunk_end=chunk_end,
                                    window_start=window_start,
                                    window_end=window_end,
                                )
                                fallback_rows = filtered.to_dict("records")
                                rows.extend(fallback_rows)
                                fallback_diag_entry["rows"] = len(fallback_rows)
                                fallback_diag_entry["used"] = True
                                diagnostics["fallback_used"] = True
                                fallback_attempt = {
                                    "chunk": chunk_name,
                                    "via": "hdx_fallback",
                                    "rows": len(fallback_rows),
                                }
                                if len(fallback_rows) == 0:
                                    fallback_attempt["zero_rows_reason"] = (
                                        "timeout_fallback_empty"
                                    )
                                    zero_row_reasons[chunk_name] = "timeout_fallback_empty"
                                else:
                                    zero_row_reasons.pop(chunk_name, None)
                                diagnostics["attempts"].append(fallback_attempt)
                                LOGGER.debug(
                                    "idmc: chunk=%s via=hdx_fallback rows=%d",
                                    chunk_name,
                                    len(fallback_rows),
                                )
                            except Exception as fallback_exc:  # pragma: no cover - network dependent
                                fallback_diag_entry["error"] = str(fallback_exc)
                                LOGGER.warning(
                                    "idmc: HDX fallback failed for chunk %s (%s)",
                                    chunk_name,
                                    type(fallback_exc).__name__,
                                )
                            diagnostics["fallback"] = fallback_diag_entry
                        elif cached_entry is not None and cached_entry.body:
                            payload_body = cached_entry.body
                            cache_record.update(cached_entry.metadata)
                            mode = "cache"
                            diagnostics["requests"].append(
                                {
                                    "url": url,
                                    "status": cached_entry.metadata.get("status"),
                                    "cache": "hit",
                                    "via": "cache_refresh",
                                }
                            )
                            payload = json.loads(payload_body.decode("utf-8"))
                            raw_paths.append(_write_raw_snapshot(key, payload_body))
                            chunk_rows = _normalise_rows(payload)
                            rows.extend(chunk_rows)
                            diagnostics["attempts"].append(
                                {
                                    "chunk": chunk_name,
                                    "via": "cache_refresh",
                                    "rows": len(chunk_rows),
                                }
                            )
                            LOGGER.debug(
                                "idmc: chunk=%s served from cache after error rows=%d",
                                chunk_name,
                                len(chunk_rows),
                            )
                            break
                        chunk_error = True
                        break
                    except Exception as exc:  # pragma: no cover - defensive
                        _record_exception(type(exc).__name__)
                        if not allow_fallback:
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

                diagnostics["attempts"].append(
                    {
                        "chunk": chunk_label or "full",
                        "via": "http_get",
                        "status": status,
                        "rows": len(chunk_rows),
                    }
                )
                LOGGER.debug(
                    "idmc: chunk=%s via=http_get status=%s rows=%d url=%s",
                    chunk_label or "full",
                    status,
                    len(chunk_rows),
                    _trim_url(url),
                )

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
        if not allow_fallback:
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
            "used": True,
            **fallback_diag,
        }
        diagnostics["fallback_used"] = True
        iso_batches = iso_batches or []
        filtered = _hdx_filter(
            fallback_frame,
            iso_batches=iso_batches,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            window_start=window_start,
            window_end=window_end,
        )
        fallback_rows = filtered.to_dict("records")
        diagnostics["fallback"]["rows"] = len(fallback_rows)
        fallback_attempt = {
            "chunk": chunk_label or "full",
            "via": "hdx_fallback",
            "rows": len(fallback_rows),
        }
        if len(fallback_rows) == 0:
            fallback_attempt["zero_rows_reason"] = "timeout_fallback_empty"
            zero_row_reasons[chunk_label or "full"] = "timeout_fallback_empty"
        else:
            zero_row_reasons.pop(chunk_label or "full", None)
        diagnostics["attempts"].append(fallback_attempt)
        diagnostics["mode"] = "fallback"
        LOGGER.info(
            "idmc: HDX fallback enabled and used (rows=%d)",
            len(fallback_rows),
        )
        frame, filters = _build_frame(fallback_rows)
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
    if zero_row_reasons:
        diagnostics["zero_rows_reasons"] = dict(zero_row_reasons)
        if not filters.get("rows_after"):
            first_reason = next(iter(zero_row_reasons.values()))
            diagnostics["filters"]["zero_rows_reason"] = first_reason
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
    allow_hdx_fallback: Optional[bool] = None,
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

    fallback_allowed = (
        allow_hdx_fallback
        if allow_hdx_fallback is not None
        else getenv_bool("IDMC_ALLOW_HDX_FALLBACK", default=False)
    )

    if should_return_empty(window_start, window_end, window_days):
        LOGGER.warning(
            "IDMC fetch invoked without a date window; returning empty payload",
        )
        connect_timeout_s, read_timeout_s = _http_timeouts()
        http_attempt_summary = {
            "planned": 0,
            "ok_2xx": 0,
            "status_4xx": 0,
            "status_5xx": 0,
            "status_other": 0,
            "timeouts": 0,
            "other_exceptions": 0,
        }
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
            "http_status_counts": serialize_http_status_counts(None)
            if network_mode == "live"
            else None,
            "http_status_counts_extended": {"2xx": 0, "4xx": 0, "5xx": 0, "other": 0}
            if network_mode == "live"
            else None,
            "raw_path": None,
            "requests_planned": 0,
            "requests_executed": 0,
            "window": {"start": None, "end": None, "window_days": None},
            "fallback_allowed": fallback_allowed,
            "http_attempt_summary": http_attempt_summary,
            "http_timeouts": {"connect_s": connect_timeout_s, "read_s": read_timeout_s},
            "attempts": [],
            "fallback": None,
            "fallback_used": False,
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
            allow_fallback=fallback_allowed,
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
    status_counts = {"2xx": 0, "4xx": 0, "5xx": 0, "other": 0, "timeout": 0}
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
    zero_rows_reasons_all: Dict[str, str] = {}
    aggregated_attempts: List[Dict[str, Any]] = []
    http_attempt_summary_totals = {
        "planned": 0,
        "ok_2xx": 0,
        "status_4xx": 0,
        "status_5xx": 0,
        "status_other": 0,
        "timeouts": 0,
        "other_exceptions": 0,
    }
    http_timeouts_cfg: Optional[Dict[str, Any]] = None
    fallback_used_total = False
    fallback_rows_total = 0
    fallback_details: List[Dict[str, Any]] = []
    fallback_primary: Optional[Dict[str, Any]] = None
    total_timeouts = 0
    total_ok_2xx = 0
    total_4xx = 0
    total_5xx = 0
    total_other_responses = 0
    total_other_exceptions = 0
    last_error_info: Optional[Dict[str, Any]] = None
    last_error_url: Optional[str] = None
    last_exception: Optional[str] = None

    for index in sorted(chunk_diags):
        diag = chunk_diags[index]
        chunk = diag.get("chunk", {}) or {}
        chunk_start = chunk.get("start")
        chunk_end = chunk.get("end")
        label = _chunk_label(
            datetime.fromisoformat(chunk_start).date() if chunk_start else None,
            datetime.fromisoformat(chunk_end).date() if chunk_end else None,
        )

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
        for bucket in ("2xx", "4xx", "5xx", "other", "timeout"):
            try:
                status_counts[bucket] += int(status_counts_block.get(bucket, 0) or 0)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
        status = http_block.get("status_last")
        if status is not None:
            last_status = status
        try:
            total_timeouts += int(http_block.get("timeouts", 0) or 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            total_timeouts += 0
        try:
            total_ok_2xx += int(http_block.get("requests_ok_2xx", 0) or 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            total_ok_2xx += 0
        try:
            total_4xx += int(http_block.get("requests_4xx", 0) or 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            total_4xx += 0
        try:
            total_5xx += int(http_block.get("requests_5xx", 0) or 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            total_5xx += 0
        try:
            total_other_responses += int(http_block.get("requests_other", 0) or 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            total_other_responses += 0
        try:
            total_other_exceptions += int(http_block.get("other_exceptions", 0) or 0)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            total_other_exceptions += 0
        if http_block.get("last_error"):
            last_error_info = http_block.get("last_error")
        if http_block.get("last_error_url"):
            last_error_url = http_block.get("last_error_url")
        if http_block.get("last_exception"):
            last_exception = http_block.get("last_exception")
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
        summary_block = diag.get("http_attempt_summary") or {}
        for key in http_attempt_summary_totals:
            try:
                http_attempt_summary_totals[key] += int(summary_block.get(key, 0) or 0)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
        if http_timeouts_cfg is None:
            timeouts_block = diag.get("http_timeouts")
            if isinstance(timeouts_block, dict) and timeouts_block:
                http_timeouts_cfg = dict(timeouts_block)
        attempts_block = diag.get("attempts") or []
        for attempt in attempts_block:
            if isinstance(attempt, dict):
                attempt_entry = dict(attempt)
                attempt_entry.setdefault("chunk", label)
                aggregated_attempts.append(attempt_entry)
        fallback_block = diag.get("fallback")
        if isinstance(fallback_block, dict) and fallback_block:
            fallback_entry = dict(fallback_block)
            fallback_entry.setdefault("chunk", label)
            fallback_details.append(fallback_entry)
            try:
                fallback_rows_total += int(fallback_entry.get("rows", 0) or 0)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                fallback_rows_total += 0
            if fallback_entry.get("used"):
                fallback_used_total = True
                fallback_primary = fallback_entry
            elif fallback_primary is None:
                fallback_primary = fallback_entry
        if diag.get("fallback_used"):
            fallback_used_total = True
        filters_block = diag.get("filters") or {}
        rows_before_total += int(filters_block.get("rows_before", 0) or 0)
        zero_reason_value = filters_block.get("zero_rows_reason")
        if zero_reason_value and label:
            zero_rows_reasons_all.setdefault(label, str(zero_reason_value))
        diag_zero_rows = diag.get("zero_rows_reasons")
        if isinstance(diag_zero_rows, dict):
            for key, value in diag_zero_rows.items():
                if value is None:
                    continue
                zero_rows_reasons_all[str(key)] = str(value)
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
    status_counts_serialized = serialize_http_status_counts(status_counts)
    http_summary = {
        "requests": total_requests,
        "retries": total_retries,
        "status_last": last_status,
        "latency_ms": _latency_block(attempt_latencies),
        "cache": {"hits": cache_hits, "misses": cache_misses},
        "wire_bytes": total_wire_bytes,
        "body_bytes": total_body_bytes,
        "retry_after_events": total_retry_after_events,
        "status_counts": status_counts_serialized,
        "status_counts_extended": dict(status_counts),
        "timeouts": total_timeouts,
        "requests_ok_2xx": total_ok_2xx,
        "requests_4xx": total_4xx,
        "requests_5xx": total_5xx,
        "requests_other": total_other_responses,
        "other_exceptions": total_other_exceptions,
    }
    if exceptions_by_type:
        http_summary["exception_counts"] = dict(exceptions_by_type)
    if last_error_info:
        http_summary["last_error"] = last_error_info
    if last_error_url:
        http_summary["last_error_url"] = last_error_url
    if last_exception:
        http_summary["last_exception"] = last_exception

    data: Dict[str, pd.DataFrame] = {"monthly_flow": combined}

    if http_timeouts_cfg is None:
        connect_timeout_s, read_timeout_s = _http_timeouts()
        http_timeouts_cfg = {"connect_s": connect_timeout_s, "read_s": read_timeout_s}

    fallback_summary: Optional[Dict[str, Any]] = None
    if fallback_details:
        fallback_summary = {}
        if fallback_primary:
            for key, value in fallback_primary.items():
                if key in {"rows", "used"}:
                    continue
                fallback_summary.setdefault(key, value)
        fallback_summary["used"] = bool(fallback_used_total)
        fallback_summary["rows"] = int(fallback_rows_total)
        fallback_summary["details"] = fallback_details
    elif fallback_used_total:
        fallback_summary = {"used": True, "rows": int(fallback_rows_total)}

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
        "http_status_counts": status_counts_serialized if network_mode == "live" else None,
        "http_status_counts_extended":
            dict(status_counts) if network_mode == "live" else None,
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
        "requests_planned": int(http_attempt_summary_totals.get("planned", 0) or len(jobs)),
        "requests_executed": total_requests,
        "fallback_allowed": fallback_allowed,
        "http_attempt_summary": http_attempt_summary_totals,
        "attempts": aggregated_attempts,
        "fallback": fallback_summary,
        "fallback_used": bool(fallback_used_total),
        "http_timeouts": http_timeouts_cfg,
    }
    if zero_rows_reasons_all:
        diagnostics["zero_rows_reasons"] = dict(zero_rows_reasons_all)
        filters_block = diagnostics.get("filters") or {}
        if not filters_block.get("zero_rows_reason") and int(filters_block.get("rows_after", 0) or 0) == 0:
            first_reason = next(iter(zero_rows_reasons_all.values()), None)
            if first_reason is not None:
                filters_block = dict(filters_block)
                filters_block["zero_rows_reason"] = first_reason
                diagnostics["filters"] = filters_block

    return data, diagnostics
