"""Client implementation for the IDMC connector."""
from __future__ import annotations

import calendar
import concurrent.futures
import io
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Set, Tuple
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlparse,
    urlsplit,
    urlunparse,
)
import urllib.error
import urllib.request

import requests

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
from .probe import ProbeOptions, probe_reachability, summarize_probe_outcome
from .export import FLOW_EXPORT_COLUMNS, FLOW_METRIC, FLOW_SERIES_SEMANTICS
from .http import HttpRequestError, http_get
from .normalize import ensure_iso3_column
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

HDX_DEFAULT_RESOURCE_ID = "1ace9c2a-7daf-4563-ac15-f2aa5071cd40"
HDX_DEFAULT_GID = "123456789"

HELIX_DEFAULT_BASE = "https://helix-tools-api.idmcdb.org"
HELIX_BASE = (
    os.getenv("IDMC_HELIX_BASE_URL", HELIX_DEFAULT_BASE).strip().rstrip("/")
    or HELIX_DEFAULT_BASE
)
HELIX_DISPLACEMENTS_PATH = "/external-api/gidd/displacements"
HELIX_LAST180_PATH = "/external-api/idus/last-180-days/"

DEFAULT_CONNECT_TIMEOUT_S = 5.0
DEFAULT_READ_TIMEOUT_S = 25.0

HDX_PREAGG_COLUMN = "__hdx_preaggregated__"

_CACHE_LOCKS: Dict[str, threading.Lock] = {}
_CACHE_LOCKS_LOCK = threading.Lock()

LOGGER = logging.getLogger(__name__)

NetworkMode = Literal["live", "helix", "cache_only", "fixture"]
NETWORK_MODES: Tuple[NetworkMode, ...] = ("live", "helix", "cache_only", "fixture")

NETWORK_ERROR_KINDS = {
    "connect_timeout",
    "read_timeout",
    "timeout",
    "socket_timeout",
    "ssl_error",
    "dns_error",
    "proxy_error",
    "connection_error",
    "conn_refused",
    "conn_reset",
    "network_unreachable",
    "host_down",
    "os_error",
}

if os.getenv("IDMC_PROXY"):
    proxy_value = os.getenv("IDMC_PROXY")
    os.environ.setdefault("HTTPS_PROXY", proxy_value)
    os.environ.setdefault("HTTP_PROXY", proxy_value)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _resolve_cache_dir(
    cfg: IdmcConfig,
    *,
    env_key: str = "IDMC_CACHE_DIR",
    default: str = os.path.join(".cache", "idmc"),
) -> Path:
    """Return a usable cache directory path for IDMC fetches."""

    cache_config = getattr(cfg, "cache", None)
    cache_dir: object = None
    if cache_config is not None:
        candidate = getattr(cache_config, "dir", None)
        if candidate is not None:
            cache_dir = candidate
        elif isinstance(cache_config, (str, os.PathLike, Path)):
            cache_dir = cache_config
    if cache_dir is None:
        env_value = os.getenv(env_key)
        if env_value:
            cache_dir = env_value
    if cache_dir is None:
        cache_dir = default
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _trim_url(url: str, limit: int = 160) -> str:
    if len(url) <= limit:
        return url
    return url[: max(limit - 3, 1)] + "..."


def _request_path(url: str) -> str:
    parts = urlsplit(url)
    query = parts.query.strip()
    if query:
        return f"{parts.path}?{query}"
    return parts.path


def _env_timeout(name: str, default: float, *, aliases: Iterable[str] = ()) -> float:
    for candidate in (name, *aliases):
        raw = os.getenv(candidate)
        if raw is None:
            continue
        try:
            value = float(str(raw).strip())
        except (TypeError, ValueError):  # pragma: no cover - defensive
            LOGGER.warning("Invalid %s=%r; using default %.1f", candidate, raw, default)
            return default
        if value < 0:
            LOGGER.warning("Negative %s=%r; using default %.1f", candidate, raw, default)
            return default
        return float(value)
    return default


def _http_timeouts() -> Tuple[float, float]:
    connect = _env_timeout(
        "IDMC_HTTP_CONNECT_TIMEOUT_S",
        DEFAULT_CONNECT_TIMEOUT_S,
        aliases=(
            "IDMC_HTTP_TIMEOUT_CONNECT",
            "IDMC_HTTP_CONNECT_TIMEOUT",
            "IDMC_HTTP_TIMEOUT_CONNECT_S",
            "IDMC_CONNECT_TIMEOUT",
        ),
    )
    read = _env_timeout(
        "IDMC_HTTP_READ_TIMEOUT_S",
        DEFAULT_READ_TIMEOUT_S,
        aliases=(
            "IDMC_HTTP_TIMEOUT_READ",
            "IDMC_HTTP_READ_TIMEOUT",
            "IDMC_HTTP_TIMEOUT_READ_S",
            "IDMC_READ_TIMEOUT",
        ),
    )
    if read <= 0:
        read = DEFAULT_READ_TIMEOUT_S
    return connect, read


def _has_col(df: Any, name: str) -> bool:
    if not hasattr(df, "columns"):
        return False
    columns = getattr(df, "columns", [])
    if name not in columns:
        return False
    return not getattr(df, "empty", True)


def _apply_iso3_filter(
    frame: Optional[pd.DataFrame],
    countries: Iterable[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    allowed = {
        code.strip().upper()
        for code in countries
        if isinstance(code, str) and code.strip()
    }
    scope = sorted(allowed)
    note: Dict[str, Any] = {"fallback_filter_applied": False}
    if scope:
        note["scope"] = scope

    if frame is None:
        note["reason"] = "empty_frame"
        return pd.DataFrame(), note

    working = ensure_iso3_column(frame)
    if getattr(working, "empty", True):
        note["reason"] = "empty_frame"
        return working.copy() if hasattr(working, "copy") else working, note

    working = working.copy()
    if not scope:
        note["reason"] = "no_scope"
        return working, note

    if not _has_col(working, "iso3"):
        note["reason"] = "no_iso3_or_empty"
        return working, note

    iso_series = working["iso3"].astype(str).str.strip().str.upper()
    working["iso3"] = iso_series
    mask = iso_series.isin(scope)
    rows_before = int(mask.size)
    filtered = working.loc[mask]
    note.update(
        fallback_filter_applied=True,
        reason="applied",
        rows_before=rows_before,
        rows_after=int(filtered.shape[0]),
    )
    return filtered, note


def _resolve_http_user_agent() -> str:
    for name in ("IDMC_USER_AGENT", "RELIEFWEB_USER_AGENT", "RELIEFWEB_APPNAME"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return "Pythia-IDMC/1.0 (+contact)"


def _build_http_headers(extra: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": _resolve_http_user_agent(),
    }
    if extra:
        headers.update({str(key): str(value) for key, value in extra.items()})
    return headers


def _http_verify() -> bool | str:
    raw = os.getenv("IDMC_HTTP_VERIFY")
    if raw is None:
        return True
    candidate = raw.strip()
    if not candidate:
        return True
    lowered = candidate.lower()
    if lowered in {"0", "false", "no", "off"}:
        return False
    if lowered in {"1", "true", "yes", "on"}:
        return True
    return candidate


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

    if network_mode == "helix":
        diagnostics["skipped"] = True
        diagnostics["reason"] = "helix-mode"
        diagnostics["date_column"] = default_date_column
        diagnostics["columns"] = default_columns
        return default_date_column, default_columns, diagnostics

    if network_mode not in {"live", "helix"}:
        diagnostics["skipped"] = True
        diagnostics["date_column"] = default_date_column
        diagnostics["columns"] = default_columns
        return default_date_column, default_columns, diagnostics

    base = (base_url or cfg.api.base_url).rstrip("/")
    endpoint = cfg.api.endpoints.get("idus_json", "/data/idus_view_flat")
    query = urlencode(SCHEMA_PROBE_QUERY, safe="*.,()")
    url = f"{base}{endpoint}?{query}"
    diagnostics["url"] = url
    connect_timeout_s, read_timeout_s = _http_timeouts()
    schema_timeout = (
        max(connect_timeout_s, 0.1),
        max(min(read_timeout_s, 10.0), max(connect_timeout_s, 0.1)),
    )
    diagnostics["timeout_s"] = {"connect": schema_timeout[0], "read": schema_timeout[1]}
    verify_setting = _http_verify()
    diagnostics["verify"] = verify_setting if isinstance(verify_setting, str) else bool(verify_setting)
    request_headers = _build_http_headers()
    diagnostics["headers"] = dict(request_headers)

    try:
        status, headers, body, http_diag = http_get(
            url,
            timeout=schema_timeout,
            retries=1,
            backoff_s=0.1,
            rate_limiter=rate_limiter,
            headers=request_headers,
            verify=verify_setting,
        )
    except HttpRequestError as exc:
        diagnostics["error"] = exc.message
        diagnostics["http_error"] = exc.diagnostics
        diagnostics["error_kind"] = exc.kind
        diagnostics["date_column"] = default_date_column
        diagnostics["columns"] = default_columns
        LOGGER.warning(
            "idmc: schema probe failed with HttpRequestError kind=%s; defaulting to %s",
            exc.kind,
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
        "preliminary-internal-displacement-updates",
    )


def _ensure_date(value: Optional[date | datetime]) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    return value


def _month_start(value: date) -> date:
    return value.replace(day=1)


def _month_end(value: date) -> date:
    last_day = calendar.monthrange(value.year, value.month)[1]
    return value.replace(day=last_day)


def _hdx_resource_id() -> Optional[str]:
    for name in ("IDMC_HDX_RESOURCE_ID", "IDMC_HDX_RESOURCE"):
        raw = os.getenv(name)
        if raw:
            value = str(raw).strip()
            if value:
                return value
    return HDX_DEFAULT_RESOURCE_ID


def _hdx_base_url() -> str:
    return os.getenv("HDX_BASE", "https://data.humdata.org").rstrip("/")


def _hdx_package_show_url(dataset: str) -> str:
    base = _hdx_base_url()
    return f"{base}/api/3/action/package_show?id={dataset}"


def _hdx_pick_displacement_csv(
    package_id: str,
    *,
    base_url: str | None = None,
    opener: Any = urllib.request,
    timeout: float = 30.0,
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Return the most appropriate CSV resource for displacement data."""

    diagnostics: Dict[str, Any] = {
        "package_id": package_id,
    }
    if not package_id:
        diagnostics["error"] = "missing_package_id"
        return None, diagnostics

    base_env = os.getenv("IDMC_HDX_BASE_URL") or os.getenv("HDX_BASE")
    base = (base_url or base_env or "https://data.humdata.org").rstrip("/")
    diagnostics["base_url"] = base
    query = urlencode({"id": package_id})
    package_url = f"{base}/api/3/action/package_show?{query}"
    diagnostics["package_url"] = package_url

    headers = {
        "Accept": "application/json",
        "User-Agent": _resolve_http_user_agent(),
    }
    request = urllib.request.Request(package_url, headers=headers)
    try:
        with opener.urlopen(request, timeout=timeout) as response:
            diagnostics["package_status_code"] = getattr(
                response, "status", None
            ) or getattr(response, "code", None)
            payload = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # pragma: no cover - network dependent
        diagnostics["package_status_code"] = getattr(exc, "code", None)
        diagnostics["error"] = f"package_http_error:{exc.code}"
        return None, diagnostics
    except urllib.error.URLError as exc:  # pragma: no cover - network dependent
        diagnostics["error"] = f"package_error:{exc.__class__.__name__}"
        diagnostics["exception"] = exc.__class__.__name__
        return None, diagnostics

    try:
        info = json.loads(payload)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        diagnostics["error"] = f"package_decode_error:{exc.__class__.__name__}"
        diagnostics["exception"] = exc.__class__.__name__
        return None, diagnostics

    if not info.get("success"):
        diagnostics["error"] = "package_show_unsuccessful"
        return None, diagnostics

    result = info.get("result") or {}
    resources = result.get("resources") or []
    diagnostics["resource_candidates"] = len(resources)

    chosen: Optional[Mapping[str, Any]] = None
    keyword_selection = False
    for resource in resources:
        if not isinstance(resource, Mapping):
            continue
        fmt = str(resource.get("format") or "").strip().lower()
        if fmt != "csv":
            continue
        text = " ".join(
            [str(resource.get("name") or ""), str(resource.get("description") or "")]
        ).lower()
        if "displacement" in text or "disaggreg" in text:
            chosen = resource
            keyword_selection = True
            break

    if chosen is None:
        for resource in resources:
            if not isinstance(resource, Mapping):
                continue
            fmt = str(resource.get("format") or "").strip().lower()
            if fmt != "csv":
                continue
            chosen = resource
            break

    if chosen is None:
        diagnostics["error"] = "no_csv_resource"
        diagnostics.setdefault("zero_rows_reason", "fallback_no_valid_resource")
        return None, diagnostics

    diagnostics["resource_selection"] = (
        "keyword_match" if keyword_selection else "first_csv"
    )
    diagnostics["resource_id"] = chosen.get("id")
    diagnostics["resource_name"] = chosen.get("name")
    diagnostics["resource_format"] = chosen.get("format")

    url = (
        str(chosen.get("url") or "").strip()
        or str(chosen.get("download_url") or "").strip()
    )
    diagnostics["resource_url"] = url or None
    size_candidate = chosen.get("size")
    if size_candidate is not None:
        diagnostics["resource_content_length"] = size_candidate
    return (url or None), diagnostics


def _read_csv_from_bytes(payload: bytes) -> pd.DataFrame:
    buffer = io.BytesIO(payload)
    return pd.read_csv(buffer)


def _hdx_download_resource(url: str, *, timeout: float = 30.0) -> Tuple[bytes, Dict[str, Any]]:
    headers = {"User-Agent": "Pythia-IDMC/1.0", "Accept": "text/csv"}
    response = requests.get(
        url,
        headers=headers,
        timeout=timeout,
        allow_redirects=True,
    )
    diagnostics: Dict[str, Any] = {
        "status_code": response.status_code,
    }
    counts = diagnostics.setdefault("http_status_counts", {"2xx": 0, "4xx": 0, "5xx": 0})
    status_code = response.status_code
    if 200 <= status_code < 300:
        counts["2xx"] += 1
    elif 400 <= status_code < 500:
        counts["4xx"] += 1
    elif status_code >= 500:
        counts["5xx"] += 1
    response.raise_for_status()
    payload = response.content
    diagnostics["bytes"] = len(payload)
    diagnostics["content_length"] = response.headers.get("Content-Length")
    return payload, diagnostics


def _hdx_resolve_gid_default() -> str:
    raw = os.getenv("IDMC_HDX_GID")
    if raw is None:
        return HDX_DEFAULT_GID
    cleaned = str(raw).strip()
    return cleaned or HDX_DEFAULT_GID


def _hdx_fetch_once() -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    dataset = _hdx_dataset_slug()
    package_url = _hdx_package_show_url(dataset)
    diagnostics: Dict[str, Any] = {
        "dataset": dataset,
        "package_url": package_url,
    }
    counts = diagnostics.setdefault("http_status_counts", {"2xx": 0, "4xx": 0, "5xx": 0})
    headers = {"Accept": "application/json", "User-Agent": "Pythia-IDMC/1.0"}
    try:
        response = requests.get(package_url, headers=headers, timeout=30.0)
        diagnostics["package_status_code"] = response.status_code
        status_code = response.status_code
        if 200 <= status_code < 300:
            counts["2xx"] += 1
        elif 400 <= status_code < 500:
            counts["4xx"] += 1
        elif status_code >= 500:
            counts["5xx"] += 1
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        diagnostics["error"] = f"package_show_failed:{exc.__class__.__name__}"
        diagnostics["exception"] = exc.__class__.__name__
        diagnostics["package_status_code"] = getattr(exc.response, "status_code", None)
        status_value = diagnostics.get("package_status_code")
        if isinstance(status_value, int):
            if 200 <= status_value < 300:
                counts["2xx"] += 1
            elif 400 <= status_value < 500:
                counts["4xx"] += 1
            elif status_value >= 500:
                counts["5xx"] += 1
        return pd.DataFrame(), diagnostics

    if not payload.get("success"):
        diagnostics["error"] = "package_show_unsuccessful"
        return None, diagnostics

    result = payload.get("result") or {}
    resources = result.get("resources") or []
    resource_target = _hdx_resource_id()
    diagnostics["resource_target"] = resource_target
    csv_candidates: List[Dict[str, Any]] = []
    for resource in resources:
        if not isinstance(resource, Mapping):
            continue
        fmt = str(resource.get("format", "")).lower()
        if fmt not in {"csv", "text/csv"}:
            continue
        identifier = str(resource.get("id", "")).strip()
        url = str(resource.get("url", "")).strip()
        name = str(resource.get("name", "")).strip()
        description = str(resource.get("description", "")).strip()
        stamp = str(resource.get("last_modified") or resource.get("created") or "")
        lowered_name = name.lower()
        lowered_description = description.lower()
        keyword_match = any(
            keyword in lowered_name or keyword in lowered_description
            for keyword in ("displacement", "disaggreg")
        )
        csv_candidates.append(
            {
                "id": identifier,
                "url": url,
                "name": name,
                "description": description,
                "timestamp": stamp,
                "preferred": "idus_view_flat" in name.lower()
                or "idus_view_flat" in url.lower(),
                "keyword": keyword_match,
            }
        )
    diagnostics["resource_candidates"] = len(csv_candidates)

    if not csv_candidates:
        diagnostics["error"] = "no_csv_resource"
        diagnostics["zero_rows_reason"] = "fallback_no_valid_resource"
        return None, diagnostics

    selected: Optional[Dict[str, Any]] = None
    if resource_target:
        for candidate in csv_candidates:
            if candidate.get("id") == resource_target:
                selected = dict(candidate)
                diagnostics["resource_selection"] = "configured_id"
                break
        diagnostics["resource_configured_id"] = resource_target

    if selected is None:
        keyword_matches = [c for c in csv_candidates if c.get("keyword")]
        if keyword_matches:
            selected = dict(keyword_matches[0])
            diagnostics["resource_selection"] = "keyword"

    if selected is None:
        preferred = [c for c in csv_candidates if c.get("preferred")]
        if preferred:
            selected = dict(preferred[-1])
            diagnostics["resource_selection"] = "preferred"

    if selected is None:
        selected = dict(csv_candidates[0])
        diagnostics["resource_selection"] = "fallback"

    diagnostics["resource_id"] = selected.get("id")
    diagnostics["resource_name"] = selected.get("name")
    diagnostics["resource_description"] = selected.get("description") or None
    diagnostics["resource_preferred"] = bool(selected.get("preferred"))

    resource_url = str(selected.get("url") or "").strip()
    if not resource_url:
        diagnostics["error"] = "resource_missing_url"
        diagnostics["zero_rows_reason"] = "fallback_no_valid_resource"
        diagnostics.setdefault("resource_errors", []).append(
            {"id": selected.get("id"), "error": "missing_url"}
        )
        return None, diagnostics

    parsed = urlparse(resource_url)
    if "docs.google.com" in parsed.netloc:
        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        gid_original: Optional[str] = None
        gid_updated: Optional[str] = None
        new_pairs: List[Tuple[str, str]] = []
        has_gid = False
        for key, value in query_pairs:
            if key == "gid":
                has_gid = True
                gid_original = value
                if value in {"", "0", "0.0"}:
                    gid_updated = _hdx_resolve_gid_default()
                    new_pairs.append((key, gid_updated))
                else:
                    new_pairs.append((key, value))
            else:
                new_pairs.append((key, value))
        if not has_gid:
            gid_updated = _hdx_resolve_gid_default()
            new_pairs.append(("gid", gid_updated))
        if gid_original is not None:
            diagnostics["resource_gid_original"] = gid_original
        if gid_updated is not None:
            diagnostics["resource_gid_used"] = gid_updated
        resource_url = urlunparse(
            parsed._replace(query=urlencode(new_pairs, doseq=True))
        )

    diagnostics["resource_url"] = resource_url

    try:
        payload_bytes, payload_diag = _hdx_download_resource(resource_url)
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        diagnostics["error"] = f"resource_http_error:{exc.__class__.__name__}"
        diagnostics["exception"] = exc.__class__.__name__
        diagnostics.setdefault("resource_errors", []).append(
            {"id": selected.get("id"), "error": diagnostics["error"]}
        )
        diagnostics["resource_status_code"] = getattr(
            getattr(exc, "response", None), "status_code", None
        )
        diagnostics["zero_rows_reason"] = "fallback_http_error"
        return None, diagnostics

    diagnostics["resource_status_code"] = payload_diag.get("status_code")
    diagnostics["resource_bytes"] = payload_diag.get("bytes")
    diagnostics["resource_content_length"] = payload_diag.get("content_length")

    try:
        frame = _read_csv_from_bytes(payload_bytes)
    except Exception as exc:  # pragma: no cover - defensive
        diagnostics["error"] = f"csv_parse_error:{exc.__class__.__name__}"
        diagnostics.setdefault("resource_errors", []).append(
            {"id": selected.get("id"), "error": diagnostics["error"]}
        )
        diagnostics["zero_rows_reason"] = "hdx_empty_or_bad_header"
        return None, diagnostics

    diagnostics["resource_rows"] = int(frame.shape[0])
    diagnostics["resource_columns"] = list(frame.columns)
    diagnostics["source"] = "hdx"

    if frame.empty:
        diagnostics["zero_rows_reason"] = "hdx_empty_or_bad_header"
        diagnostics.setdefault("resource_errors", []).append(
            {"id": selected.get("id"), "error": "empty_csv"}
        )
        return None, diagnostics

    lowered_columns = {str(column).strip().lower() for column in frame.columns}
    has_iso3 = "iso3" in lowered_columns
    has_value = any(
        candidate in lowered_columns
        for candidate in {"figure", "new_displacements", "new displacements"}
    )
    if not has_iso3 or not has_value:
        diagnostics["zero_rows_reason"] = "hdx_empty_or_bad_header"
        missing: List[str] = []
        if not has_iso3:
            missing.append("iso3")
        if not has_value:
            missing.append("figure/new_displacements")
        diagnostics.setdefault("resource_errors", []).append(
            {"id": selected.get("id"), "error": f"missing:{','.join(missing)}"}
        )
        return None, diagnostics

    return frame, diagnostics


def _hdx_fetch_latest_csv() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frame, diagnostics = _hdx_fetch_once()
    if frame is None:
        return pd.DataFrame(), diagnostics
    return frame, diagnostics


def _fetch_hdx_displacements(
    *,
    package_id: Optional[str],
    base_url: Optional[str],
    start_date: Optional[date],
    end_date: Optional[date],
    iso3_list: Iterable[str],
    opener: Any = urllib.request,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Download and aggregate the HDX displacement CSV."""

    diagnostics: Dict[str, Any] = {
        "source": "hdx",
        "source_tag": "idmc_hdx",
    }
    status_counts = diagnostics.setdefault("http_status_counts", {"2xx": 0, "4xx": 0, "5xx": 0})
    canonical_columns = list(FLOW_EXPORT_COLUMNS) + [HDX_PREAGG_COLUMN]
    empty = pd.DataFrame(columns=canonical_columns)
    if not package_id:
        diagnostics["error"] = "missing_hdx_package_id"
        diagnostics["zero_rows_reason"] = "hdx_missing_package_id"
        return empty, diagnostics

    resource_url, resource_diag = _hdx_pick_displacement_csv(
        package_id,
        base_url=base_url,
        opener=opener,
    )
    diagnostics.update(resource_diag)
    if not resource_url:
        diagnostics.setdefault("zero_rows_reason", "hdx_resource_not_found")
        return empty, diagnostics

    try:
        payload_bytes, payload_diag = _hdx_download_resource(resource_url)
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        diagnostics["error"] = f"hdx_download_error:{exc.__class__.__name__}"
        diagnostics["exception"] = exc.__class__.__name__
        diagnostics.setdefault("zero_rows_reason", "hdx_download_error")
        status_value = getattr(exc.response, "status_code", None)
        if isinstance(status_value, int):
            if 200 <= status_value < 300:
                status_counts["2xx"] += 1
            elif 400 <= status_value < 500:
                status_counts["4xx"] += 1
            elif status_value >= 500:
                status_counts["5xx"] += 1
        return empty, diagnostics

    diagnostics["resource_status_code"] = payload_diag.get("status_code")
    diagnostics["resource_bytes"] = payload_diag.get("bytes")
    diagnostics["resource_content_length"] = payload_diag.get("content_length")
    payload_counts = payload_diag.get("http_status_counts")
    if isinstance(payload_counts, Mapping):
        for bucket in ("2xx", "4xx", "5xx"):
            try:
                status_counts[bucket] += int(payload_counts.get(bucket, 0) or 0)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue

    try:
        raw = _read_csv_from_bytes(payload_bytes)
    except Exception as exc:  # pragma: no cover - defensive
        diagnostics["error"] = f"csv_parse_error:{exc.__class__.__name__}"
        diagnostics.setdefault("zero_rows_reason", "hdx_empty_or_bad_header")
        return empty, diagnostics

    diagnostics["resource_rows"] = int(raw.shape[0])
    diagnostics["resource_columns"] = [str(column) for column in raw.columns]

    iso_scope = [
        code.strip().upper()
        for code in iso3_list
        if isinstance(code, str) and code.strip()
    ]
    iso_batches: List[List[str]] = [iso_scope] if iso_scope else []
    columns_lower = {str(column).strip().lower(): column for column in raw.columns}
    iso_candidates = ("iso3", "countryiso3", "country iso3")
    date_candidates = (
        "displacement_date",
        "event_date",
        "start_date",
        "date",
        "month",
    )
    value_candidates = ("new_displacements", "new displacements", "figure")

    iso_column = next((columns_lower.get(name) for name in iso_candidates if name in columns_lower), None)
    date_column = next((columns_lower.get(name) for name in date_candidates if name in columns_lower), None)
    value_column = None
    for name in value_candidates:
        if name in columns_lower:
            value_column = columns_lower[name]
            break
    if value_column is None and "figure" in columns_lower:
        value_column = columns_lower["figure"]

    if iso_column is None or date_column is None or value_column is None:
        diagnostics.setdefault("fallback_reason", "missing_required_columns")
        diagnostics.setdefault("zero_rows_reason", "hdx_missing_required_columns")
        return empty, diagnostics

    working = raw.rename(
        columns={
            iso_column: "iso3",
            date_column: "event_date",
            value_column: "value",
        }
    )
    subset = working.loc[:, ["iso3", "event_date", "value"]].copy()
    subset["iso3"] = subset["iso3"].astype(str).str.strip().str.upper()
    subset["event_date"] = pd.to_datetime(subset["event_date"], errors="coerce")
    subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
    subset = subset.dropna(subset=["iso3", "event_date", "value"])
    subset = subset.loc[subset["value"] >= 0]

    if start_date is not None:
        subset = subset.loc[subset["event_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        subset = subset.loc[subset["event_date"] <= pd.Timestamp(end_date)]
    if iso_batches:
        allowed = {code for batch in iso_batches for code in batch}
        subset = subset.loc[subset["iso3"].isin(allowed)]

    if subset.empty:
        diagnostics.setdefault("zero_rows_reason", "hdx_filtered_empty")
        return empty, diagnostics

    subset["as_of_date"] = subset["event_date"].dt.to_period("M").dt.to_timestamp("M")
    aggregated = (
        subset.groupby(["iso3", "as_of_date"], as_index=False)["value"].sum()
    )
    if aggregated.empty:
        diagnostics.setdefault("zero_rows_reason", "hdx_aggregation_empty")
        return empty, diagnostics

    aggregated["metric"] = FLOW_METRIC
    aggregated["series_semantics"] = FLOW_SERIES_SEMANTICS
    aggregated["source"] = "idmc_hdx"
    aggregated["value"] = aggregated["value"].round().astype(pd.Int64Dtype())
    aggregated[HDX_PREAGG_COLUMN] = True
    aggregated = aggregated.loc[:, canonical_columns]

    diagnostics["rows"] = int(aggregated.shape[0])
    diagnostics["resource_url"] = resource_url
    diagnostics["used"] = True
    return aggregated, diagnostics


def _helix_client_id() -> Optional[str]:
    raw = os.getenv("IDMC_HELIX_CLIENT_ID")
    if raw is None:
        return None
    cleaned = str(raw).strip()
    return cleaned or None


def _fetch_helix_idus_last180(
    client_id: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {
        "status": None,
        "bytes": 0,
        "last_request_path": None,
        "http_status_counts": serialize_http_status_counts(None),
    }

    resolved_client = (client_id or "").strip()
    if not resolved_client:
        diagnostics["error"] = "missing_client_id"
        diagnostics["zero_rows_reason"] = "helix_last180_missing_client_id"
        return pd.DataFrame(columns=["iso3", "event_date", "value"]), diagnostics

    url = (
        f"{HELIX_BASE}{HELIX_LAST180_PATH}?client_id={resolved_client}&format=json"
    )
    path = _request_path(url)
    if resolved_client:
        path = path.replace(resolved_client, "REDACTED")
    diagnostics["last_request_path"] = path

    headers = _build_http_headers({"Accept": "application/json"})
    connect_timeout_s, read_timeout_s = _http_timeouts()
    verify_setting = _http_verify()

    try:
        status, _response_headers, body, http_diag = http_get(
            url,
            headers=headers,
            timeout=(connect_timeout_s, read_timeout_s),
            retries=0,
            verify=verify_setting,
        )
    except HttpRequestError as exc:
        error_diag = exc.diagnostics or {}
        if not isinstance(error_diag, Mapping):
            error_diag = {"details": error_diag}
        diagnostics.update({
            "status": error_diag.get("status"),
            "error": error_diag or exc.message,
        })
        diagnostics.setdefault("zero_rows_reason", "helix_last180_http_error")
        bucket = _http_status_bucket(diagnostics.get("status"))
        diagnostics["http_status_counts"] = serialize_http_status_counts(
            {bucket: 1} if bucket else None
        )
        return pd.DataFrame(columns=["iso3", "event_date", "value"]), diagnostics

    diagnostics["status"] = status
    diagnostics["bytes"] = len(body or b"")
    diagnostics["url"] = url.replace(resolved_client, "REDACTED")
    if http_diag:
        diagnostics["http"] = dict(http_diag)
    bucket = _http_status_bucket(status)
    diagnostics["http_status_counts"] = serialize_http_status_counts(
        {bucket: 1} if bucket else None
    )

    try:
        payload = json.loads(body.decode("utf-8")) if body else []
    except (TypeError, ValueError):
        diagnostics["error"] = "invalid_json"
        diagnostics.setdefault("zero_rows_reason", "helix_last180_invalid_json")
        return pd.DataFrame(columns=["iso3", "event_date", "value"]), diagnostics

    if not isinstance(payload, list):
        diagnostics.setdefault("zero_rows_reason", "helix_last180_not_list")
        return pd.DataFrame(columns=["iso3", "event_date", "value"]), diagnostics

    records: List[Dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, Mapping):
            continue
        iso_value = entry.get("iso3") or entry.get("ISO3")
        date_value = (
            entry.get("displacement_date")
            or entry.get("event_date")
            or entry.get("start_date")
            or entry.get("date")
        )
        value = entry.get("new_displacements")
        if value is None:
            value = entry.get("figure")
        records.append({
            "iso3": iso_value,
            "event_date": date_value,
            "value": value,
        })

    frame = pd.DataFrame(records, columns=["iso3", "event_date", "value"])
    return frame, diagnostics


def _fetch_helix_last180(
    helix_client_id: Optional[str],
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    iso3_list: Iterable[str] = (),
    rate_limiter: TokenBucket | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch displacement figures from the Helix GIDD endpoint."""

    diagnostics: Dict[str, Any] = {
        "source": "helix_idu",
        "url": f"{HELIX_BASE}{HELIX_DISPLACEMENTS_PATH}",
        "status": None,
        "bytes": 0,
    }

    client_id = (helix_client_id or "").strip()
    if not client_id:
        diagnostics["error"] = "missing_client_id"
        diagnostics["zero_rows_reason"] = "helix_missing_client_id"
        status_counts_empty = {"2xx": 0, "4xx": 0, "5xx": 0, "other": 0, "timeout": 0}
        diagnostics["status_counts"] = dict(status_counts_empty)
        diagnostics["http_status_counts"] = serialize_http_status_counts(status_counts_empty)
        diagnostics["http_status_counts_extended"] = dict(status_counts_empty)
        diagnostics["requests_planned"] = 0
        diagnostics["requests_executed"] = 0
        return pd.DataFrame(), diagnostics

    start_value = _ensure_date(start_date)
    end_value = _ensure_date(end_date)
    start_param = _month_start(start_value) if start_value else None
    end_param = _month_end(end_value) if end_value else None
    if start_param is None and end_param is not None:
        start_param = _month_start(end_param)
    if end_param is None and start_param is not None:
        end_param = _month_end(start_param)

    iso_values = [
        code.strip().upper()
        for code in iso3_list
        if isinstance(code, str) and code.strip()
    ]
    iso_joined = ",".join(sorted(dict.fromkeys(iso_values))) if iso_values else ""

    release_env = (
        os.getenv("IDMC_HELIX_VERSION", "").strip()
        or os.getenv("IDMC_HELIX_ENV", "").strip()
        or "RELEASE"
    )

    params: Dict[str, str] = {
        "limit": "10000",
        "release_environment": release_env,
    }
    if start_param:
        params["start"] = start_param.strftime("%Y-%m-%d")
    if end_param:
        params["end"] = end_param.strftime("%Y-%m-%d")
    if iso_joined:
        params["iso3"] = iso_joined
    if client_id:
        params["client_id"] = client_id

    query = urlencode({key: value for key, value in params.items() if value})
    url = f"{HELIX_BASE}{HELIX_DISPLACEMENTS_PATH}?{query}" if query else f"{HELIX_BASE}{HELIX_DISPLACEMENTS_PATH}"
    diagnostics["url"] = url.replace(client_id, "REDACTED")
    path_value = _request_path(url)
    if client_id:
        path_value = path_value.replace(client_id, "REDACTED")
    diagnostics["path"] = path_value
    diagnostics["last_request_path"] = path_value
    diagnostics["release_environment"] = release_env
    diagnostics["iso_scope"] = iso_values
    diagnostics["requests_planned"] = 1
    diagnostics["requests_executed"] = 0

    status_counts: Dict[str, int] = {"2xx": 0, "4xx": 0, "5xx": 0, "other": 0, "timeout": 0}

    def _record_status(bucket: Optional[str]) -> None:
        if not bucket:
            return
        if bucket not in status_counts:
            status_counts[bucket] = 0
        status_counts[bucket] += 1

    headers = _build_http_headers({"Accept": "application/json"})
    connect_timeout_s, read_timeout_s = _http_timeouts()
    verify_setting = _http_verify()

    try:
        status, response_headers, body, http_diag = http_get(
            url,
            headers=headers,
            timeout=(connect_timeout_s, read_timeout_s),
            retries=1,
            rate_limiter=rate_limiter,
            verify=verify_setting,
        )
    except HttpRequestError as exc:
        error_diag = exc.diagnostics or {}
        if not isinstance(error_diag, Mapping):
            error_diag = {"details": error_diag}
        diagnostics["error"] = error_diag or exc.message
        status_value = error_diag.get("status")
        if error_diag.get("timeout"):
            status_value = "timeout"
        diagnostics["status"] = status_value
        bucket = _http_status_bucket(status_value)
        diagnostics["status_bucket"] = bucket
        _record_status(bucket)
        diagnostics["status_counts"] = dict(status_counts)
        diagnostics["http_status_counts"] = serialize_http_status_counts(status_counts)
        diagnostics["http_status_counts_extended"] = dict(status_counts)
        diagnostics["http"] = error_diag
        diagnostics["requests_executed"] = int(error_diag.get("attempts", 1) or 1)
        diagnostics["exception"] = exc.__class__.__name__
        diagnostics["zero_rows_reason"] = "helix_http_error"
        return pd.DataFrame(), diagnostics
    except Exception as exc:  # pragma: no cover - defensive
        diagnostics["error"] = str(exc)
        diagnostics["exception"] = exc.__class__.__name__
        diagnostics["zero_rows_reason"] = "helix_http_error"
        diagnostics["status_counts"] = dict(status_counts)
        diagnostics["http_status_counts"] = serialize_http_status_counts(status_counts)
        diagnostics["http_status_counts_extended"] = dict(status_counts)
        return pd.DataFrame(), diagnostics

    diagnostics["status"] = status
    bucket = _http_status_bucket(status)
    diagnostics["status_bucket"] = bucket
    _record_status(bucket)
    diagnostics["status_counts"] = dict(status_counts)
    diagnostics["http_status_counts"] = serialize_http_status_counts(status_counts)
    diagnostics["requests_executed"] = int(http_diag.get("attempts", 1) or 1)
    diagnostics["headers"] = response_headers
    diagnostics["http"] = http_diag
    diagnostics["bytes"] = len(body or b"") if isinstance(body, (bytes, bytearray)) else 0
    diagnostics["wire_bytes"] = int(http_diag.get("wire_bytes", 0) or 0)
    diagnostics["body_bytes"] = int(http_diag.get("body_bytes", 0) or diagnostics["bytes"])
    diagnostics["duration_s"] = float(http_diag.get("duration_s", 0.0) or 0.0)

    if not isinstance(status, int) or not (200 <= status < 300):
        diagnostics.setdefault("zero_rows_reason", "helix_http_error")
        return pd.DataFrame(), diagnostics

    payload = body
    if isinstance(body, bytes):
        try:
            payload = body.decode("utf-8")
        except Exception:  # pragma: no cover - defensive
            payload = body.decode("utf-8", errors="replace")

    try:
        parsed = json.loads(payload) if isinstance(payload, str) else payload
    except Exception as exc:  # pragma: no cover - defensive
        diagnostics["error"] = f"json:{exc.__class__.__name__}"
        diagnostics["exception"] = exc.__class__.__name__
        diagnostics["zero_rows_reason"] = "helix_parse_error"
        return pd.DataFrame(), diagnostics

    items: Iterable[Mapping[str, Any]]
    if isinstance(parsed, dict):
        candidate = parsed.get("results")
        if isinstance(candidate, list):
            items = candidate
        else:
            items = []
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = []

    rows: List[Dict[str, Any]] = []

    def _coalesce(record: Mapping[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in record and record[key] not in (None, ""):
                return record[key]
        return None

    for raw in items:
        if not isinstance(raw, Mapping):
            continue
        iso3_raw = _coalesce(
            raw,
            "iso3",
            "ISO3",
            "country_iso3",
            "CountryISO3",
            "geo_iso3",
        )
        iso3_value = str(iso3_raw).strip().upper() if iso3_raw else None

        date_value = _coalesce(
            raw,
            "displacement_date",
            "event_date",
            "date",
            "displacement_end_date",
            "as_of_date",
            "month",
        )
        if date_value is None:
            year = _coalesce(raw, "year", "Year")
            month = _coalesce(raw, "month_number", "Month")
            try:
                if year is not None and month is not None:
                    month_int = int(str(month))
                    year_int = int(str(year))
                    day = 1
                    candidate_date = datetime(year_int, month_int, day)
                    date_value = candidate_date.strftime("%Y-%m-%d")
            except Exception:  # pragma: no cover - defensive
                date_value = None

        figure_value = _coalesce(
            raw,
            "new_displacements",
            "figure",
            "value",
            "total_displacements",
        )

        rows.append(
            {
                "iso3": iso3_value,
                "displacement_date": date_value,
                "displacement_start_date": _coalesce(
                    raw,
                    "displacement_start_date",
                    "start_date",
                    "start",
                ),
                "displacement_end_date": _coalesce(
                    raw,
                    "displacement_end_date",
                    "end_date",
                    "end",
                ),
                "figure": figure_value,
                "raw": raw,
            }
        )

    frame = pd.DataFrame(rows)
    diagnostics["raw_rows"] = int(frame.shape[0])
    if frame.empty:
        diagnostics.setdefault("zero_rows_reason", "helix_empty")

    return frame, diagnostics


def _helix_base_url() -> str:
    base = os.getenv("IDMC_HELIX_BASE", HELIX_DEFAULT_BASE)
    return str(base).strip().rstrip("/") or HELIX_DEFAULT_BASE


def _helix_fetch_csv(
    *,
    start_date: Optional[date],
    end_date: Optional[date],
    iso3_list: Iterable[str],
    timeout: float = 60.0,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    del timeout  # compatibility with previous signature

    frame, diagnostics = _fetch_helix_last180(
        _helix_client_id(),
        start_date=start_date,
        end_date=end_date,
        iso3_list=iso3_list,
    )
    diagnostics = dict(diagnostics)
    diagnostics.setdefault("source", "helix")
    diagnostics.setdefault("source_tag", "idmc_gidd")

    filtered, filter_note = _apply_iso3_filter(frame, iso3_list)
    diagnostics["fallback_filter"] = filter_note
    if filtered.empty and filter_note.get("reason") == "empty_frame":
        return filtered, diagnostics

    start_bound = start_date
    end_bound = end_date
    if start_bound and end_bound and start_bound > end_bound:
        start_bound, end_bound = end_bound, start_bound

    if start_bound or end_bound:
        dates = pd.to_datetime(filtered["displacement_date"], errors="coerce")
        if start_bound:
            filtered = filtered.loc[dates >= pd.Timestamp(start_bound)]
        if end_bound:
            filtered = filtered.loc[dates <= pd.Timestamp(end_bound)]

    diagnostics["rows"] = int(filtered.shape[0])
    diagnostics["columns"] = list(filtered.columns)
    if filtered.empty and diagnostics.get("zero_rows_reason") is None:
        diagnostics["zero_rows_reason"] = "helix_empty"

    return filtered.reset_index(drop=True), diagnostics

def _normalise_column_key(name: object) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(name or "").strip().lower())
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _resolve_column(frame: pd.DataFrame, options: Iterable[str]) -> Optional[str]:
    if frame.empty:
        return None
    normalised = {
        _normalise_column_key(column): column for column in frame.columns
    }
    for option in options:
        key = _normalise_column_key(option)
        if key in normalised:
            return normalised[key]
    return None


def _hdx_filter_rows(
    frame: pd.DataFrame,
    *,
    iso_batches: List[List[str]],
    chunk_start: Optional[date],
    chunk_end: Optional[date],
    window_start: Optional[date],
    window_end: Optional[date],
) -> pd.DataFrame:
    working = frame.copy()
    if working.empty:
        return working

    iso_column = _resolve_column(
        working, ["iso3", "iso_3", "country_iso3", "countryiso3", "geo_iso3"]
    )
    if iso_batches and iso_column:
        allowed = {
            code.strip().upper()
            for batch in iso_batches
            for code in batch
            if code and code.strip()
        }
        if allowed:
            iso_values = working[iso_column].astype(str).str.strip().str.upper()
            working = working.loc[iso_values.isin(allowed)]

    start_bound = chunk_start or window_start
    end_bound = chunk_end or window_end
    if start_bound or end_bound:
        date_column = _resolve_column(
            working,
            [
                "displacement_end_date",
                "displacement_start_date",
                "displacement_date",
                "event_date",
                "report_date",
                "date",
            ],
        )
        if date_column:
            dates = pd.to_datetime(working[date_column], errors="coerce")
            if start_bound:
                working = working.loc[dates >= pd.Timestamp(start_bound)]
            if end_bound:
                working = working.loc[dates <= pd.Timestamp(end_bound)]
    return working.reset_index(drop=True)


def _hdx_prepare_monthly_flow(
    frame: pd.DataFrame,
    *,
    iso_batches: List[List[str]],
    chunk_start: Optional[date],
    chunk_end: Optional[date],
    window_start: Optional[date],
    window_end: Optional[date],
    source_tag: str,
) -> pd.DataFrame:
    empty = pd.DataFrame(columns=list(FLOW_EXPORT_COLUMNS))
    filtered = _hdx_filter_rows(
        frame,
        iso_batches=iso_batches,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        window_start=window_start,
        window_end=window_end,
    )
    if filtered.empty:
        return empty

    iso_column = _resolve_column(
        filtered, ["iso3", "iso_3", "country_iso3", "countryiso3", "geo_iso3"]
    )
    value_column = _resolve_column(
        filtered,
        [
            "figure",
            "new_displacements",
            "new displacement",
            "new_displacement",
            "new_displacements_(idps)",
            "new_displacements_idps",
        ],
    )
    date_column = _resolve_column(
        filtered,
        [
            "displacement_end_date",
            "displacement_start_date",
            "displacement_date",
            "event_date",
            "report_date",
            "date",
        ],
    )
    if not iso_column or not value_column or not date_column:
        return empty

    iso_series = filtered[iso_column].astype(str).str.strip().str.upper()
    value_series = pd.to_numeric(filtered[value_column], errors="coerce")
    date_series = pd.to_datetime(filtered[date_column], errors="coerce")

    mask = iso_series.notna() & date_series.notna() & value_series.notna()
    mask &= value_series >= 0
    if not mask.any():
        return empty

    iso_series = iso_series.loc[mask]
    value_series = value_series.loc[mask]
    date_series = date_series.loc[mask]

    month_end = (
        date_series.dt.to_period("M")
        .dt.to_timestamp(how="end")
        .dt.tz_localize(None)
        .dt.floor("D")
    )

    aggregated = (
        pd.DataFrame(
            {
                "iso3": iso_series.values,
                "as_of_date": month_end.values,
                "metric": FLOW_METRIC,
                "value": value_series.values,
                "series_semantics": FLOW_SERIES_SEMANTICS,
                "source": source_tag,
            }
        )
        .groupby(["iso3", "as_of_date", "metric", "series_semantics", "source"], as_index=False)[
            "value"
        ]
        .sum()
    )

    if aggregated.empty:
        return empty

    aggregated["value"] = aggregated["value"].astype(pd.Int64Dtype())
    aggregated[HDX_PREAGG_COLUMN] = True
    return aggregated.loc[:, list(FLOW_EXPORT_COLUMNS) + [HDX_PREAGG_COLUMN]]


def _normalise_fallback_monthly_flow(
    frame: pd.DataFrame,
    *,
    source_tag: str,
    chunk_start: Optional[date],
    chunk_end: Optional[date],
    window_start: Optional[date],
    window_end: Optional[date],
    countries: Iterable[str] | None,
) -> pd.DataFrame:
    canonical_columns = list(FLOW_EXPORT_COLUMNS) + [HDX_PREAGG_COLUMN]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=canonical_columns)

    working = ensure_iso3_column(frame).copy()
    start_bound = chunk_start or window_start
    end_bound = chunk_end or window_end
    allowed_countries = {
        code.strip().upper()
        for code in (countries or [])
        if isinstance(code, str) and code.strip()
    }

    if all(column in working.columns for column in FLOW_EXPORT_COLUMNS):
        result = working.loc[:, FLOW_EXPORT_COLUMNS].copy()
        result["iso3"] = result["iso3"].astype(str).str.strip().str.upper()
        result["as_of_date"] = pd.to_datetime(result["as_of_date"], errors="coerce")
        result["value"] = pd.to_numeric(result["value"], errors="coerce")
        if start_bound is not None:
            result = result.loc[result["as_of_date"] >= pd.Timestamp(start_bound)]
        if end_bound is not None:
            result = result.loc[result["as_of_date"] <= pd.Timestamp(end_bound)]
        if allowed_countries and _has_col(result, "iso3"):
            result = result.loc[result["iso3"].isin(allowed_countries)]
        result = result.dropna(subset=["iso3", "as_of_date", "value"])
        result = result.loc[result["value"] >= 0]
        if result.empty:
            return pd.DataFrame(columns=canonical_columns)
        result["metric"] = FLOW_METRIC
        result["series_semantics"] = FLOW_SERIES_SEMANTICS
        result["source"] = source_tag
        result["value"] = result["value"].round().astype(pd.Int64Dtype())
        if HDX_PREAGG_COLUMN not in result.columns:
            result[HDX_PREAGG_COLUMN] = True
        else:
            result[HDX_PREAGG_COLUMN] = result[HDX_PREAGG_COLUMN].astype(bool)
        return result.loc[:, canonical_columns]

    columns_lower = {str(column).strip().lower(): column for column in working.columns}
    iso_candidates = (
        "iso3",
        "iso_3",
        "countryiso3",
        "country iso3",
        "country_iso3",
        "geo_iso3",
    )
    date_candidates = (
        "displacement_date",
        "event_date",
        "displacement_end_date",
        "displacement_start_date",
        "start_date",
        "end_date",
        "date",
    )
    value_candidates = (
        "new_displacements",
        "new displacements",
        "figure",
        "value",
    )

    iso_col = next((columns_lower.get(name) for name in iso_candidates if name in columns_lower), None)
    date_col = next((columns_lower.get(name) for name in date_candidates if name in columns_lower), None)
    value_col = next((columns_lower.get(name) for name in value_candidates if name in columns_lower), None)

    if iso_col is None or date_col is None or value_col is None:
        return pd.DataFrame(columns=canonical_columns)

    renamed = working.rename(columns={iso_col: "iso3", date_col: "event_date", value_col: "value"})
    subset = renamed.loc[:, ["iso3", "event_date", "value"]].copy()
    subset["iso3"] = subset["iso3"].astype(str).str.strip().str.upper()
    subset["event_date"] = pd.to_datetime(subset["event_date"], errors="coerce")
    subset["value"] = pd.to_numeric(subset["value"], errors="coerce")
    subset = subset.dropna(subset=["iso3", "event_date", "value"])
    subset = subset.loc[subset["value"] >= 0]
    if start_bound is not None:
        subset = subset.loc[subset["event_date"] >= pd.Timestamp(start_bound)]
    if end_bound is not None:
        subset = subset.loc[subset["event_date"] <= pd.Timestamp(end_bound)]
    if allowed_countries and _has_col(subset, "iso3"):
        subset = subset.loc[subset["iso3"].isin(allowed_countries)]
    if subset.empty:
        return pd.DataFrame(columns=canonical_columns)

    subset["as_of_date"] = subset["event_date"].dt.to_period("M").dt.to_timestamp("M")
    aggregated = (
        subset.groupby(["iso3", "as_of_date"], as_index=False)["value"].sum()
    )
    aggregated["metric"] = FLOW_METRIC
    aggregated["series_semantics"] = FLOW_SERIES_SEMANTICS
    aggregated["source"] = source_tag
    aggregated["value"] = aggregated["value"].round().astype(pd.Int64Dtype())
    aggregated[HDX_PREAGG_COLUMN] = True
    return aggregated.loc[:, canonical_columns]


def _normalise_helix_last180_monthly(
    frame: pd.DataFrame,
    *,
    window_start: Optional[date],
    window_end: Optional[date],
    countries: Iterable[str] | None,
) -> pd.DataFrame:
    canonical_columns = list(FLOW_EXPORT_COLUMNS) + [HDX_PREAGG_COLUMN]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=canonical_columns)

    working = ensure_iso3_column(frame).copy()
    if "iso3" not in working.columns:
        return pd.DataFrame(columns=canonical_columns)

    working["iso3"] = working["iso3"].astype(str).str.strip().str.upper()
    working["event_date"] = pd.to_datetime(working["event_date"], errors="coerce")
    working["value"] = pd.to_numeric(working["value"], errors="coerce")

    allowed_countries = {
        code.strip().upper()
        for code in (countries or [])
        if isinstance(code, str) and code.strip()
    }

    subset = working.dropna(subset=["iso3", "event_date", "value"])
    subset = subset.loc[subset["value"] >= 0]
    if window_start is not None:
        subset = subset.loc[subset["event_date"] >= pd.Timestamp(window_start)]
    if window_end is not None:
        subset = subset.loc[subset["event_date"] <= pd.Timestamp(window_end)]
    if allowed_countries and _has_col(subset, "iso3"):
        subset = subset.loc[subset["iso3"].isin(allowed_countries)]
    if subset.empty:
        return pd.DataFrame(columns=canonical_columns)

    subset["as_of_date"] = (
        subset["event_date"].dt.to_period("M").dt.to_timestamp(how="end").dt.floor("D")
    )

    aggregated = subset.groupby(["iso3", "as_of_date"], as_index=False)["value"].sum()
    aggregated["metric"] = FLOW_METRIC
    aggregated["series_semantics"] = FLOW_SERIES_SEMANTICS
    aggregated["source"] = "idmc_idu"
    aggregated["value"] = aggregated["value"].round().astype(pd.Int64Dtype())
    aggregated[HDX_PREAGG_COLUMN] = False
    return aggregated.loc[:, canonical_columns]


def _hdx_filter(
    frame: pd.DataFrame,
    *,
    iso_batches: List[List[str]],
    chunk_start: Optional[date],
    chunk_end: Optional[date],
    window_start: Optional[date],
    window_end: Optional[date],
    source_tag: str,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=list(FLOW_EXPORT_COLUMNS))

    has_canonical_columns = all(column in frame.columns for column in FLOW_EXPORT_COLUMNS)
    if HDX_PREAGG_COLUMN in frame.columns or has_canonical_columns:
        working = frame.copy()
        if "iso3" in working.columns and iso_batches:
            allowed = {
                code.strip().upper()
                for batch in iso_batches
                for code in batch
                if isinstance(code, str) and code.strip()
            }
            if allowed:
                iso_series = working["iso3"].astype(str).str.strip().str.upper()
                working = working.loc[iso_series.isin(allowed)]

        start_bound = chunk_start or window_start
        end_bound = chunk_end or window_end
        if start_bound or end_bound:
            if "as_of_date" in working.columns:
                dates = pd.to_datetime(working["as_of_date"], errors="coerce")
                if start_bound:
                    working = working.loc[dates >= pd.Timestamp(start_bound)]
                if end_bound:
                    working = working.loc[dates <= pd.Timestamp(end_bound)]

        ordered = list(FLOW_EXPORT_COLUMNS)
        extras: List[str] = []
        if HDX_PREAGG_COLUMN in working.columns:
            extras.append(HDX_PREAGG_COLUMN)
        result = working.loc[:, [col for col in ordered + extras if col in working.columns]]
        return result.reset_index(drop=True)

    return _hdx_prepare_monthly_flow(
        frame,
        iso_batches=iso_batches,
        chunk_start=chunk_start,
        chunk_end=chunk_end,
        window_start=window_start,
        window_end=window_end,
        source_tag=source_tag,
    )


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
    fallback_loader: Optional[Callable[[], Tuple[pd.DataFrame, Dict[str, Any]]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch the IDU flat JSON payload for a specific window."""

    base = (base_url or cfg.api.base_url).rstrip("/")
    endpoint = cfg.api.endpoints.get("idus_json", "/data/idus_view_flat")
    window_start_iso = (chunk_start or window_start).isoformat() if (chunk_start or window_start) else None
    window_end_iso = (chunk_end or window_end).isoformat() if (chunk_end or window_end) else None
    base_params: Dict[str, str] = {
        "chunk": chunk_label or "full",
        "window_start": window_start_iso,
        "window_end": window_end_iso,
    }
    cache_dir_path = _resolve_cache_dir(cfg)
    cache_dir = cache_dir_path.as_posix()
    cache_cfg = getattr(cfg, "cache", None)
    ttl_seconds = (
        cache_ttl
        if cache_ttl is not None
        else getattr(cache_cfg, "ttl_seconds", 0)
    )
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
        "last_request_path": None,
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
    verify_setting = _http_verify()
    diagnostics["http_verify"] = (
        verify_setting if isinstance(verify_setting, str) else bool(verify_setting)
    )
    http_info["verify"] = diagnostics["http_verify"]
    zero_row_reasons: Dict[str, str] = {}

    fallback_frame_cached: Optional[pd.DataFrame] = None
    fallback_diag_cached: Optional[Dict[str, Any]] = None
    fallback_source_cached: Optional[str] = None
    helix_client_value = _helix_client_id()
    helix_enabled_local = bool(allow_fallback and helix_client_value)
    if os.getenv("IDMC_ALLOW_HELIX_FALLBACK") is not None:
        helix_enabled_local = bool(
            helix_enabled_local
            and getenv_bool("IDMC_ALLOW_HELIX_FALLBACK", default=False)
        )

    def _fetch_hdx_fallback() -> Tuple[pd.DataFrame, Dict[str, Any]]:
        nonlocal fallback_frame_cached, fallback_diag_cached, fallback_source_cached
        if fallback_frame_cached is not None and fallback_diag_cached is not None:
            diag_copy = dict(fallback_diag_cached)
            if fallback_source_cached:
                diag_copy.setdefault("source_tag", fallback_source_cached)
            return fallback_frame_cached, diag_copy

        if fallback_loader is not None:
            frame, diag = fallback_loader()
            diag_copy = dict(diag)
            source_tag_raw = str(
                diag_copy.get("source_tag") or diag_copy.get("source") or "idmc_idu"
            )
            normalized_source = (
                "idmc_gidd"
                if source_tag_raw.lower() in {"helix", "idmc_gidd"}
                else "idmc_idu"
            )
            diag_copy.setdefault("source_tag", normalized_source)
            fallback_frame_cached = frame
            fallback_diag_cached = dict(diag_copy)
            fallback_source_cached = normalized_source
            return frame, dict(diag_copy)

        hdx_frame, hdx_diag = _hdx_fetch_once()
        diag_copy: Dict[str, Any]
        source_tag = "idmc_idu"
        if hdx_frame is not None and not hdx_frame.empty:
            diag_copy = dict(hdx_diag)
            diag_copy.setdefault("source_tag", source_tag)
            fallback_frame_cached = hdx_frame
            fallback_diag_cached = dict(diag_copy)
            fallback_source_cached = source_tag
            return hdx_frame, dict(diag_copy)

        diag_copy = dict(hdx_diag)
        diag_copy.setdefault("source_tag", source_tag)
        diag_copy.setdefault("zero_rows_reason", "hdx_empty_or_bad_header")
        frame_candidate = hdx_frame if hdx_frame is not None else pd.DataFrame()

        helix_iso_scope: Iterable[str] = only_countries or cfg.api.countries or []
        if helix_enabled_local and helix_client_value:
            try:
                helix_frame, helix_diag = _helix_fetch_csv(
                    start_date=window_start,
                    end_date=window_end,
                    iso3_list=helix_iso_scope,
                )
            except Exception as exc:  # pragma: no cover - defensive
                diag_copy.setdefault("helix_error", str(exc))
                if "zero_rows_reason" not in diag_copy:
                    diag_copy["zero_rows_reason"] = "helix_exception"
            else:
                helix_diag = dict(helix_diag)
                helix_diag.setdefault("source_tag", "idmc_gidd")
                helix_diag.setdefault("hdx_attempt", diag_copy)
                if helix_frame is not None and not helix_frame.empty:
                    helix_diag.pop("zero_rows_reason", None)
                fallback_frame_cached = helix_frame
                fallback_diag_cached = dict(helix_diag)
                fallback_source_cached = "idmc_gidd"
                return helix_frame, dict(helix_diag)

        fallback_frame_cached = frame_candidate
        fallback_diag_cached = dict(diag_copy)
        fallback_source_cached = source_tag
        diag_result = dict(diag_copy)
        diag_result.setdefault("source_tag", source_tag)
        return frame_candidate, diag_result

    def _merge_fallback_metadata(entry: Dict[str, Any], diag: Mapping[str, Any]) -> None:
        metadata_keys = (
            "dataset",
            "package_url",
            "package_status_code",
            "resource_id",
            "resource_name",
            "resource_selection",
            "resource_url",
            "resource_status_code",
            "resource_bytes",
            "resource_content_length",
            "min_bytes",
            "resource_rows",
            "resource_columns",
            "resource_preferred",
            "zero_rows_reason",
            "resource_errors",
            "request_url",
            "status_code",
            "bytes",
            "content_length",
            "source",
            "source_tag",
            "hdx_attempt",
            "helix_error",
            "fallback_filter",
        )
        for key in metadata_keys:
            value = diag.get(key) if isinstance(diag, Mapping) else None
            if value is not None and key not in entry:
                entry[key] = value

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
            "chunk_label": chunk_label or "full",
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
                request_params = _postgrest_filters(
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
                base_url_no_query = f"{base}{endpoint}"
                url = base_url_no_query
                if request_params:
                    url = f"{url}?{urlencode(request_params, safe='.,()')}"
                diagnostics["url"] = url
                diagnostics["last_request_path"] = _request_path(url)
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
                if cache_entry is None and use_cache_only:
                    legacy_params = {"chunk": cache_params.get("chunk", "full")}
                    legacy_key = cache_key(base_url_no_query, params=legacy_params)
                    legacy_entry = cache_get(cache_dir, legacy_key, None)
                    if legacy_entry is not None:
                        cache_entry = legacy_entry
                        cached_entry = legacy_entry
                        key = legacy_key
                        cache_record["key"] = key
                        cache_record.setdefault("legacy_key", legacy_key)
                        cache_record.setdefault("legacy_params", legacy_params)
                refresh_live_chunk = (
                    network_mode in {"live", "helix"}
                    and isinstance(chunk_label, str)
                    and chunk_label
                    and chunk_label != "full"
                )
                cache_hit = cache_entry is not None and not refresh_live_chunk
                request_params_serialized = [
                    {"key": key, "value": value} for key, value in request_params
                ]
                diagnostics["postgrest_params"] = request_params_serialized
                if cache_hit:
                    cache_stats["hits"] += 1
                    cache_stats["hit"] = True
                    cache_record.update(cache_entry.metadata)
                    payload_body = cache_entry.body
                    http_diag = {"attempts": 0, "retries": 0}
                    mode = "cache"
                    diagnostics["requests"].append(
                        {
                            "url": url,
                            "status": cache_entry.metadata.get("status"),
                            "cache": "hit",
                            "params": request_params_serialized,
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
                            "params": request_params_serialized,
                        }
                    )
                    break
                else:
                    cache_stats["misses"] += 1
                    request_index += 1
                    diagnostics["http_attempt_summary"]["planned"] += 1
                    try:
                        LOGGER.debug("IDMC GET %s", url)
                        status, response_headers, payload_body, http_diag = http_get(
                            url,
                            timeout=(connect_timeout_s, read_timeout_s),
                            retries=1,
                            backoff_s=0.5,
                            rate_limiter=rate_limiter,
                            max_bytes=max_bytes,
                            stream_path=f"{cache_path}.partial",
                            headers=_build_http_headers(),
                            verify=verify_setting,
                        )
                        summary_counters = diagnostics["http_attempt_summary"]
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
                                "params": request_params_serialized,
                            }
                        )
                        metadata = {
                            "status": status,
                            "headers": response_headers,
                            "fetched_at": datetime.now(timezone.utc).isoformat(),
                        }
                        cache_record.update(metadata)
                        if not isinstance(status, int) or not (200 <= status < 300):
                            error_diag = dict(http_diag)
                            error_diag.setdefault("status", status)
                            raise HttpRequestError(
                                message=f"HTTP {status}",
                                diagnostics=error_diag,
                                kind="http_error",
                            )
                        bucket = _http_status_bucket(status)
                        if bucket:
                            http_info["status_counts"].setdefault(bucket, 0)
                            http_info["status_counts"][bucket] += 1
                        http_info["status_last"] = status
                        http_info["requests_ok_2xx"] += 1
                        summary_counters["ok_2xx"] += 1
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
                        error_kind = getattr(exc, "kind", None) or exc.__class__.__name__
                        bucket = _http_status_bucket(status_display)
                        if bucket:
                            http_info["status_counts"].setdefault(bucket, 0)
                            http_info["status_counts"][bucket] += 1
                        summary_counters = diagnostics["http_attempt_summary"]
                        is_timeout_kind = error_kind in {
                            "connect_timeout",
                            "read_timeout",
                            "timeout",
                            "socket_timeout",
                        }
                        if is_timeout_kind or status_display == "timeout":
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
                            if error_kind in NETWORK_ERROR_KINDS:
                                http_info["other_exceptions"] += 1
                                summary_counters["other_exceptions"] += 1
                        last_status_value: int | str | None = status_display
                        if error_kind in NETWORK_ERROR_KINDS:
                            last_status_value = None
                        http_info["status_last"] = last_status_value
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
                        cache_record["error"] = error_kind
                        cache_record["status"] = status_display
                        cache_stats["entries"].append(cache_record)
                        diagnostics["requests"].append(
                            {
                                "url": url,
                                "status": status_display,
                                "cache": "miss",
                                "error": exc.__class__.__name__,
                                "error_kind": error_kind,
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
                            "error_kind": error_kind,
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
                                "kind": error_kind,
                                "status": status_display,
                                "message": snippet,
                            }
                        )
                        http_info["last_error"] = {
                            "type": exc.__class__.__name__,
                            "kind": error_kind,
                            "message": snippet or str(exc),
                        }
                        http_info["last_error_url"] = url
                        http_info["last_exception"] = exc.__class__.__name__
                        http_info["last_exception_kind"] = error_kind
                        exception_counts = http_info.setdefault("exception_counts", {})
                        exception_counts[error_kind] = exception_counts.get(error_kind, 0) + 1
                        LOGGER.debug(
                            "idmc: chunk=%s via=http_get status=%s error=%s kind=%s url=%s",
                            chunk_name,
                            status_display,
                            exc.__class__.__name__,
                            error_kind,
                            _trim_url(url),
                        )
                        should_use_fallback = False
                        if allow_fallback:
                            is_2xx_status = (
                                isinstance(status_display, int)
                                and 200 <= status_display < 300
                            )
                            if is_timeout_kind or status_display == "timeout":
                                should_use_fallback = True
                            elif not is_2xx_status:
                                should_use_fallback = True
                        if chunk_name:
                            reason_value: str
                            if error_kind:
                                reason_value = str(error_kind)
                            elif status_display is not None:
                                reason_value = str(status_display)
                            else:
                                reason_value = "http_error"
                            zero_row_reasons[chunk_name] = reason_value
                        if should_use_fallback:
                            fallback_diag_entry: Dict[str, Any] = {
                                "type": "hdx",
                                "reason": "http_error",
                                "status": status_display,
                                "chunk": chunk_name,
                                "used": False,
                            }
                            if error_kind:
                                zero_row_reasons[chunk_name] = str(error_kind)
                            elif status_display and chunk_name:
                                zero_row_reasons[chunk_name] = str(status_display)
                            else:
                                zero_row_reasons[chunk_name] = "http_error"
                            try:
                                fallback_frame, fallback_diag = _fetch_hdx_fallback()
                                _merge_fallback_metadata(fallback_diag_entry, fallback_diag)
                                source_candidate = (
                                    fallback_diag_entry.get("source_tag")
                                    or fallback_diag.get("source_tag")
                                    or fallback_diag.get("source")
                                )
                                fallback_source_tag_inner = str(source_candidate or "idmc_idu")
                                if fallback_source_tag_inner.lower() == "helix":
                                    fallback_source_tag_inner = "idmc_gidd"
                                fallback_diag_entry["source_tag"] = fallback_source_tag_inner
                                fallback_diag_entry["type"] = (
                                    "helix" if fallback_source_tag_inner == "idmc_gidd" else "hdx"
                                )
                                fallback_scope = (
                                    only_countries if only_countries is not None else countries
                                )
                                _, filter_note = _apply_iso3_filter(
                                    fallback_frame,
                                    fallback_scope or [],
                                )
                                fallback_diag_entry.setdefault(
                                    "fallback_filter", filter_note
                                )
                                normalized_fallback = _normalise_fallback_monthly_flow(
                                    fallback_frame,
                                    source_tag=fallback_source_tag_inner,
                                    chunk_start=chunk_start,
                                    chunk_end=chunk_end,
                                    window_start=window_start,
                                    window_end=window_end,
                                    countries=fallback_scope,
                                )
                                fallback_rows = normalized_fallback.to_dict("records")
                                rows.extend(fallback_rows)
                                fallback_diag_entry["rows"] = len(fallback_rows)
                                fallback_diag_entry["used"] = True
                                diagnostics["fallback_used"] = True
                                fallback_attempt = {
                                    "chunk": chunk_name,
                                    "via": "helix_fallback"
                                    if fallback_source_tag_inner == "idmc_gidd"
                                    else "hdx_fallback",
                                    "rows": len(fallback_rows),
                                }
                                if len(fallback_rows) == 0:
                                    fallback_zero_reason = str(
                                        fallback_diag_entry.get("zero_rows_reason") or ""
                                    )
                                    if fallback_zero_reason:
                                        fallback_attempt["zero_rows_reason"] = (
                                            fallback_zero_reason
                                        )
                                        zero_row_reasons[chunk_name] = fallback_zero_reason
                                    else:
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
                                zero_row_reasons[chunk_name] = "fallback_http_error"
                                LOGGER.warning(
                                    "idmc: HDX fallback failed for chunk %s (%s)",
                                    chunk_name,
                                    type(fallback_exc).__name__,
                                )
                            diagnostics["fallback"] = fallback_diag_entry
                        else:
                            if allow_fallback:
                                diagnostics["fallback"] = {
                                    "type": "hdx",
                                    "reason": "http_error_not_triggered",
                                    "status": status_display,
                                    "chunk": chunk_name,
                                    "used": False,
                                }
                            if cached_entry is not None and cached_entry.body:
                                payload_body = cached_entry.body
                                cache_record.update(cached_entry.metadata)
                                mode = "cache"
                                diagnostics["requests"].append(
                                    {
                                        "url": url,
                                        "status": cached_entry.metadata.get("status"),
                                        "cache": "hit",
                                        "via": "cache_refresh",
                                        "params": request_params_serialized,
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
                        "via": "cache" if cache_hit else "http_get",
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

                if not cache_hit:
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
    except Exception as exc:
        if not allow_fallback:
            raise
        try:
            fallback_frame, fallback_diag = _fetch_hdx_fallback()
        except Exception as fallback_exc:  # pragma: no cover - network dependent
            diagnostics["fallback"] = {
                "type": "hdx",
                "error": str(fallback_exc),
            }
            zero_row_reasons[chunk_label or "full"] = "fallback_http_error"
            raise
        raw_source = fallback_diag.get("source_tag") or fallback_diag.get("source")
        fallback_source_tag = str(raw_source or "idmc_idu")
        if fallback_source_tag.lower() == "helix":
            fallback_source_tag = "idmc_gidd"
        fallback_entry: Dict[str, Any] = {
            "type": "hdx",
            "used": True,
            "reason": f"exception:{type(exc).__name__}",
            "source_tag": fallback_source_tag,
        }
        _merge_fallback_metadata(fallback_entry, fallback_diag)
        merged_source = str(fallback_entry.get("source_tag") or fallback_source_tag)
        if merged_source.lower() == "helix":
            merged_source = "idmc_gidd"
        fallback_entry["source_tag"] = merged_source
        fallback_entry["type"] = "helix" if merged_source == "idmc_gidd" else "hdx"
        fallback_source_tag = merged_source
        fallback_zero_reason = str(fallback_diag.get("zero_rows_reason") or "")
        if fallback_zero_reason:
            fallback_entry["zero_rows_reason"] = fallback_zero_reason
        diagnostics["fallback"] = fallback_entry
        diagnostics["fallback_used"] = True
        iso_batches = iso_batches or []
        fallback_scope = only_countries if only_countries is not None else countries
        _, filter_note = _apply_iso3_filter(
            fallback_frame,
            fallback_scope or [],
        )
        fallback_entry.setdefault("fallback_filter", filter_note)
        normalized_fallback = _normalise_fallback_monthly_flow(
            fallback_frame,
            source_tag=fallback_source_tag,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            window_start=window_start,
            window_end=window_end,
            countries=fallback_scope,
        )
        fallback_rows = normalized_fallback.to_dict("records")
        diagnostics["fallback"]["rows"] = len(fallback_rows)
        fallback_entry["rows"] = len(fallback_rows)
        fallback_attempt = {
            "chunk": chunk_label or "full",
            "via": "helix_fallback"
            if fallback_source_tag == "idmc_gidd"
            else "hdx_fallback",
            "rows": len(fallback_rows),
            "source_tag": fallback_source_tag,
        }
        if len(fallback_rows) == 0:
            if fallback_zero_reason:
                fallback_attempt["zero_rows_reason"] = fallback_zero_reason
                zero_row_reasons[chunk_label or "full"] = fallback_zero_reason
            else:
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
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
    window_days: Optional[int] = 30,
    only_countries: Iterable[str] | None = None,
    base_url: Optional[str] = None,
    cache_ttl: Optional[int] = None,
    rate_per_sec: Optional[float] = None,
    max_concurrency: int = 1,
    max_bytes: Optional[int] = None,
    chunk_by_month: bool = False,
    allow_hdx_fallback: Optional[bool] = None,
    helix_client_id: Optional[str] = None,  # noqa: ARG001 - reserved for helix mode
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Return payloads for downstream normalization and diagnostics."""

    if network_mode not in NETWORK_MODES:
        raise ValueError(f"Unsupported IDMC network mode: {network_mode}")

    requested_network_mode = network_mode
    endpoint_outcomes: Dict[str, Dict[str, Any]] = {}

    LOGGER.info("IDMC network mode: %s", network_mode)
    if network_mode not in {"live", "helix"}:
        LOGGER.warning(
            "Running in %s – no network calls will be made; results may be empty unless cache/fixtures exist.",
            network_mode,
        )

    cache_dir_text = _resolve_cache_dir(cfg).as_posix()

    start_alias = start_date if start_date is not None else start
    end_alias = end_date if end_date is not None else end
    if window_start is None:
        window_start = start_alias
    if window_end is None:
        window_end = end_alias

    helix_client_id = helix_client_id or _helix_client_id()
    helix_last180_diag: Optional[Dict[str, Any]] = None

    if rate_per_sec is not None:
        rate = rate_per_sec
    else:
        raw_rate = os.getenv("IDMC_REQ_PER_SEC", "0.5")
        try:
            rate = float(raw_rate)
        except ValueError:  # pragma: no cover - defensive
            rate = 0.5
    limiter = None
    if network_mode in {"live", "helix"} and rate and rate > 0:
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
    base_candidate = (base_url or cfg.api.base_url).rstrip("/")
    alt_config_value = getattr(cfg.api, "alternate_base_url", None)
    alt_candidate: Optional[str] = None
    if isinstance(alt_config_value, str):
        cleaned_alt = alt_config_value.strip()
        if cleaned_alt:
            alt_candidate = cleaned_alt.rstrip("/")
    if alt_candidate and alt_candidate == base_candidate:
        alt_candidate = None

    helix_failover_active = False
    endpoint_outcomes.setdefault(
        "helix",
        {
            "base_url": f"{HELIX_BASE}{HELIX_LAST180_PATH}",
            "status": "unused",
        },
    )

    if network_mode == "live":
        try:
            primary_probe = probe_reachability(ProbeOptions(base_url=base_candidate))
        except Exception as exc:  # pragma: no cover - defensive network
            primary_probe = {
                "base_url": base_candidate,
                "dns": {"ok": False, "error": str(exc), "elapsed_ms": 0},
            }
        primary_summary = summarize_probe_outcome(primary_probe)
        primary_summary.setdefault("base_url", base_candidate)
        endpoint_outcomes["primary"] = primary_summary

        should_try_alt = (
            primary_summary.get("status") == "fail"
            and str(primary_summary.get("stage")) in {"dns", "tls"}
        )
        if should_try_alt:
            if alt_candidate:
                try:
                    alternate_probe = probe_reachability(
                        ProbeOptions(base_url=alt_candidate)
                    )
                except Exception as exc:  # pragma: no cover - defensive network
                    alternate_probe = {
                        "base_url": alt_candidate,
                        "dns": {"ok": False, "error": str(exc), "elapsed_ms": 0},
                    }
                alternate_summary = summarize_probe_outcome(alternate_probe)
                alternate_summary.setdefault("base_url", alt_candidate)
                endpoint_outcomes["alternate"] = alternate_summary
                if alternate_summary.get("status") in {"ok", "warn"}:
                    LOGGER.warning(
                        "IDMC primary base unreachable (stage=%s); using alternate base %s",
                        primary_summary.get("stage"),
                        alt_candidate,
                    )
                    base_candidate = alt_candidate
                elif (
                    alternate_summary.get("status") == "fail"
                    and str(alternate_summary.get("stage")) in {"dns", "tls"}
                ):
                    helix_failover_active = True
                else:
                    helix_failover_active = False
            else:
                endpoint_outcomes["alternate"] = {
                    "status": "skipped",
                    "reason": "not_configured",
                    "base_url": None,
                }
                helix_failover_active = True
        else:
            if alt_candidate:
                endpoint_outcomes.setdefault(
                    "alternate",
                    {
                        "status": "skipped",
                        "reason": "not_attempted",
                        "base_url": alt_candidate,
                    },
                )
    else:
        endpoint_outcomes.setdefault(
            "primary",
            {
                "status": "skipped",
                "reason": f"network_mode_{network_mode}",
                "base_url": base_candidate,
            },
        )
        if alt_candidate:
            endpoint_outcomes.setdefault(
                "alternate",
                {
                    "status": "skipped",
                    "reason": f"network_mode_{network_mode}",
                    "base_url": alt_candidate,
                },
            )

    if helix_failover_active and network_mode != "helix" and helix_client_id:
        LOGGER.warning(
            "IDMC primary and alternate bases unreachable; falling back to HELIX last-180-days",
        )
        network_mode = "helix"
        endpoint_outcomes["helix"].update({"status": "pending", "reason": "failover"})

    base_url = base_candidate

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

    hdx_cfg = getattr(cfg, "hdx", None)
    hdx_package_id: Optional[str] = None
    hdx_base_url: Optional[str] = None
    if hdx_cfg is not None:
        hdx_package_id = getattr(hdx_cfg, "package_id", None)
        hdx_base_url = getattr(hdx_cfg, "base_url", None)

    env_package = os.getenv("IDMC_HDX_PACKAGE_ID")
    if env_package:
        env_package_clean = env_package.strip()
        if env_package_clean:
            hdx_package_id = env_package_clean

    env_base = os.getenv("IDMC_HDX_BASE_URL")
    if env_base:
        env_base_clean = env_base.strip()
        if env_base_clean:
            hdx_base_url = env_base_clean

    cli_allow = bool(allow_hdx_fallback) if allow_hdx_fallback is not None else False
    env_allow = getenv_bool("IDMC_ALLOW_HDX_FALLBACK", default=False)
    fallback_allowed = cli_allow or env_allow

    if fallback_allowed and not hdx_package_id:
        LOGGER.error(
            "HDX fallback requested but no package id configured (set idmc.hdx.package_id or IDMC_HDX_PACKAGE_ID)"
        )

    fallback_lock = threading.Lock()
    fallback_frame_cache_fetch: Optional[pd.DataFrame] = None
    fallback_diag_cache_fetch: Optional[Dict[str, Any]] = None
    fallback_source_cache_fetch: Optional[str] = None
    fallback_latest_frame_cache: Optional[pd.DataFrame] = None
    fallback_latest_diag_cache: Optional[Dict[str, Any]] = None
    helix_flag_raw = os.getenv("IDMC_USE_HELIX_IF_IDU_UNREACHABLE")
    if helix_flag_raw is None:
        helix_enabled = bool(_helix_client_id())
    else:
        helix_enabled = getenv_bool("IDMC_USE_HELIX_IF_IDU_UNREACHABLE", default=False)

    def _resolve_fallback_frame() -> Tuple[pd.DataFrame, Dict[str, Any], str]:
        nonlocal fallback_latest_frame_cache, fallback_latest_diag_cache

        if fallback_latest_frame_cache is None or fallback_latest_diag_cache is None:
            latest_frame, latest_diag = _hdx_fetch_latest_csv()
            if latest_frame is None:
                latest_frame = pd.DataFrame()
            fallback_latest_frame_cache = latest_frame
            fallback_latest_diag_cache = dict(latest_diag or {})

        latest_frame = fallback_latest_frame_cache
        latest_diag = dict(fallback_latest_diag_cache or {})
        latest_diag.setdefault("source", latest_diag.get("source") or "hdx")
        latest_source_tag = str(
            latest_diag.get("source_tag")
            or latest_diag.get("source")
            or "idmc_idu"
        )
        latest_diag.setdefault("source_tag", latest_source_tag)

        if latest_frame is not None and not latest_frame.empty:
            return latest_frame, latest_diag, latest_source_tag

        hdx_frame, hdx_diag = _fetch_hdx_displacements(
            package_id=hdx_package_id,
            base_url=hdx_base_url,
            start_date=window_start,
            end_date=window_end,
            iso3_list=countries,
        )
        diag_copy = dict(hdx_diag)
        diag_copy.setdefault("source_tag", "idmc_idu")
        diag_copy.setdefault("source", "hdx")
        if latest_diag:
            diag_copy.setdefault("hdx_attempt", latest_diag)

        if hdx_frame is not None and not hdx_frame.empty:
            diag_copy.pop("zero_rows_reason", None)
            source_tag_value = str(diag_copy.get("source_tag") or "idmc_idu")
            return hdx_frame, diag_copy, source_tag_value

        if "zero_rows_reason" not in diag_copy:
            diag_copy["zero_rows_reason"] = "hdx_empty_or_bad_header"

        if helix_enabled and _helix_client_id():
            try:
                helix_frame, helix_diag = _helix_fetch_csv(
                    start_date=window_start,
                    end_date=window_end,
                    iso3_list=countries,
                )
            except Exception as exc:  # pragma: no cover - defensive
                diag_copy.setdefault("helix_error", str(exc))
                if "zero_rows_reason" not in diag_copy:
                    diag_copy["zero_rows_reason"] = "helix_exception"
            else:
                helix_diag = dict(helix_diag)
                helix_diag.setdefault("source", "helix")
                helix_diag.setdefault("source_tag", "idmc_gidd")
                if latest_diag:
                    helix_diag.setdefault("hdx_attempt", latest_diag)
                else:
                    helix_diag.setdefault("hdx_attempt", diag_copy)
                if helix_frame is not None and not helix_frame.empty:
                    helix_diag.pop("zero_rows_reason", None)
                return helix_frame, helix_diag, "idmc_gidd"

        if hdx_frame is None:
            return pd.DataFrame(), diag_copy, "idmc_idu"
        return hdx_frame, diag_copy, "idmc_idu"

    def _load_fallback_cached() -> Tuple[pd.DataFrame, Dict[str, Any]]:
        nonlocal fallback_frame_cache_fetch, fallback_diag_cache_fetch, fallback_source_cache_fetch
        nonlocal fallback_latest_frame_cache, fallback_latest_diag_cache
        with fallback_lock:
            if fallback_frame_cache_fetch is None or fallback_diag_cache_fetch is None:
                frame, diag, source_tag = _resolve_fallback_frame()
                fallback_frame_cache_fetch = frame
                fallback_diag_cache_fetch = dict(diag)
                fallback_source_cache_fetch = source_tag
            frame = fallback_frame_cache_fetch
            diag_copy = dict(fallback_diag_cache_fetch)
            if fallback_source_cache_fetch:
                diag_copy.setdefault("source_tag", fallback_source_cache_fetch)
        return frame, diag_copy

    if should_return_empty(window_start, window_end, window_days) and network_mode in {"live", "helix"}:
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
        cache_dir_text = _resolve_cache_dir(cfg).as_posix()
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
            "cache": {"dir": cache_dir_text, "hits": 0, "misses": 0},
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
            if network_mode in {"live", "helix"}
            else None,
            "http_status_counts_extended": {"2xx": 0, "4xx": 0, "5xx": 0, "other": 0}
            if network_mode in {"live", "helix"}
            else None,
            "raw_path": None,
            "last_request_path": None,
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
    helix_summary: Optional[Dict[str, Any]] = None
    helix_zero_reason_global: Optional[str] = None

    if network_mode == "helix":
        connect_timeout_s, read_timeout_s = _http_timeouts()
        helix_frame_full, helix_diag = _fetch_helix_last180(
            helix_client_id,
            start_date=window_start,
            end_date=window_end,
            iso3_list=countries,
            rate_limiter=limiter,
        )
        helix_summary = dict(helix_diag or {})
        helix_entry = endpoint_outcomes.setdefault(
            "helix",
            {"base_url": f"{HELIX_BASE}{HELIX_LAST180_PATH}"},
        )
        status_value = helix_summary.get("status")
        if isinstance(status_value, int) and 200 <= status_value < 300:
            helix_entry.update({"status": "used", "status_code": status_value})
        else:
            helix_entry.update({"status": "fail", "status_code": status_value})
        rows_value = helix_summary.get("raw_rows")
        if rows_value is None:
            rows_value = helix_frame_full.shape[0]
        try:
            helix_entry["rows"] = int(rows_value)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            helix_entry["rows"] = int(helix_frame_full.shape[0])
        if helix_failover_active:
            helix_entry.setdefault("reason", "failover")
        else:
            helix_entry.setdefault("reason", f"network_mode_{requested_network_mode}")
        helix_zero_reason: Optional[str] = None
        status_value = helix_summary.get("status")
        if not isinstance(status_value, int) or not (200 <= status_value < 300):
            helix_zero_reason = "helix_http_error"
        elif int(helix_summary.get("raw_rows", 0) or 0) == 0:
            helix_zero_reason = "helix_http_error"

        allowed_iso = [
            code.strip().upper()
            for code in countries
            if isinstance(code, str) and code.strip()
        ]
        helix_frame_full, filter_note = _apply_iso3_filter(helix_frame_full, allowed_iso)
        helix_summary["fallback_filter"] = filter_note

        if "displacement_date" not in helix_frame_full.columns:
            helix_frame_full["displacement_date"] = None

        helix_timestamps = pd.to_datetime(
            helix_frame_full["displacement_date"], errors="coerce"
        )
        helix_frame_full = helix_frame_full.assign(_helix_ts=helix_timestamps)
        total_rows_pre_filter = int(helix_frame_full.shape[0])

        for index, (start, end) in enumerate(chunk_ranges):
            label = _chunk_label(start, end)
            mask = pd.Series(True, index=helix_frame_full.index)
            if start:
                mask &= helix_frame_full["_helix_ts"] >= pd.Timestamp(start)
            if end:
                mask &= helix_frame_full["_helix_ts"] <= pd.Timestamp(end)
            chunk_frame = (
                helix_frame_full.loc[mask]
                .drop(columns=["_helix_ts"], errors="ignore")
                .reset_index(drop=True)
            )

            bucket = _http_status_bucket(status_value)
            http_counts = {"2xx": 0, "4xx": 0, "5xx": 0, "other": 0, "timeout": 0}
            if bucket and index == 0:
                http_counts[bucket] = 1
            request_count = 1 if index == 0 and bucket else 0
            attempt_ms: List[int] = []
            duration_s = float(helix_summary.get("duration_s", 0.0) or 0.0)
            if index == 0 and duration_s > 0:
                attempt_ms.append(int(round(duration_s * 1000)))
            http_block = {
                "requests": request_count,
                "retries": 0,
                "status_last": status_value,
                "duration_s": duration_s if index == 0 else 0.0,
                "backoff_s": 0.0,
                "wire_bytes": int(helix_summary.get("wire_bytes", 0) or 0)
                if index == 0
                else 0,
                "body_bytes": int(helix_summary.get("body_bytes", 0) or 0)
                if index == 0
                else 0,
                "retry_after_events": 0,
                "retry_after_s": [],
                "rate_limit_wait_s": [],
                "planned_sleep_s": [],
                "attempt_durations_ms": attempt_ms,
                "latency_ms": _latency_block(attempt_ms),
                "status_counts": http_counts,
                "requests_ok_2xx": request_count if bucket == "2xx" else 0,
                "requests_4xx": request_count if bucket == "4xx" else 0,
                "requests_5xx": request_count if bucket == "5xx" else 0,
                "requests_other": request_count if bucket == "other" else 0,
                "timeouts": request_count if bucket == "timeout" else 0,
                "other_exceptions": 0,
            }
            filters_block = {
                "window_start": (start or window_start).isoformat()
                if (start or window_start)
                else None,
                "window_end": (end or window_end).isoformat()
                if (end or window_end)
                else None,
                "countries": sorted(set(allowed_iso)),
                "rows_before": total_rows_pre_filter,
                "rows_after": int(chunk_frame.shape[0]),
                "chunk_label": label or "full",
            }
            if chunk_frame.empty and helix_zero_reason:
                filters_block["zero_rows_reason"] = helix_zero_reason

            chunk_diag = {
                "mode": "online",
                "network_mode": "helix",
                "chunk": {
                    "start": start.isoformat() if start else None,
                    "end": end.isoformat() if end else None,
                },
                "http": http_block,
                "cache": {"dir": cache_dir_text, "hits": 0, "misses": 0},
                "filters": filters_block,
                "requests": [],
                "attempts": [],
                "http_attempt_summary": {
                    "planned": request_count,
                    "ok_2xx": request_count if bucket == "2xx" else 0,
                    "status_4xx": request_count if bucket == "4xx" else 0,
                    "status_5xx": request_count if bucket == "5xx" else 0,
                    "status_other": request_count if bucket == "other" else 0,
                    "timeouts": request_count if bucket == "timeout" else 0,
                    "other_exceptions": 0,
                },
                "helix": dict(helix_summary),
                "http_timeouts": {
                    "connect_s": connect_timeout_s,
                    "read_s": read_timeout_s,
                },
                "fallback": None,
                "fallback_used": False,
                "raw_path": None,
                "via": "helix",
                "last_request_path": HELIX_DISPLACEMENTS_PATH,
            }
            if helix_zero_reason and chunk_frame.empty:
                chunk_diag.setdefault("zero_rows_reasons", {"full": helix_zero_reason})

            frames[index] = chunk_frame
            chunk_diags[index] = chunk_diag

        helix_zero_reason_global = helix_zero_reason
    else:
        def _run_chunk(
            index: int, start: Optional[date], end: Optional[date]
        ) -> Tuple[int, pd.DataFrame, Dict[str, Any]]:
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
                fallback_loader=_load_fallback_cached if fallback_allowed else None,
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
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_concurrency
            ) as executor:
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

    monthly_frames: List[pd.DataFrame] = []
    if not combined.empty:
        monthly_frames.append(combined)

    helix_last180_frame: Optional[pd.DataFrame] = None
    helix_last180_fallback_summary: Optional[Dict[str, Any]] = None
    helix_last180_rows = 0
    if network_mode == "live" and helix_client_id:
        helix_last180_frame, helix_last180_diag = _fetch_helix_idus_last180(helix_client_id)
        status_value = helix_last180_diag.get("status") if helix_last180_diag else None
        helix_success = isinstance(status_value, int) and 200 <= status_value < 300
        if helix_success and helix_last180_frame is not None and not helix_last180_frame.empty:
            helix_normalized = _normalise_helix_last180_monthly(
                helix_last180_frame,
                window_start=window_start,
                window_end=window_end,
                countries=countries,
            )
            if not helix_normalized.empty:
                monthly_frames.append(helix_normalized)
            helix_last180_rows = int(helix_last180_frame.shape[0])
        elif fallback_allowed:
            hdx_frame, hdx_diag = _fetch_hdx_displacements(
                package_id=hdx_package_id,
                base_url=hdx_base_url,
                start_date=window_start,
                end_date=window_end,
                iso3_list=countries,
            )
            hdx_rows = int(hdx_frame.shape[0]) if isinstance(hdx_frame, pd.DataFrame) else 0
            if hdx_frame is not None and not hdx_frame.empty:
                monthly_frames.append(hdx_frame)
            summary = {
                "used": bool(hdx_rows),
                "rows": hdx_rows,
                "type": "hdx",
            }
            helix_last180_rows = int(hdx_rows)
            for key in (
                "resource_url",
                "dataset",
                "package_url",
                "package_status_code",
                "resource_status_code",
                "resource_bytes",
                "resource_content_length",
                "resource_id",
                "resource_selection",
                "fallback_reason",
                "zero_rows_reason",
            ):
                value = hdx_diag.get(key)
                if value is not None:
                    summary[key] = value
            helix_last180_fallback_summary = summary

    if monthly_frames:
        combined = _combine_frames(monthly_frames)
    else:
        combined = pd.DataFrame(columns=list(FLOW_EXPORT_COLUMNS) + [HDX_PREAGG_COLUMN])

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
    last_exception_kind: Optional[str] = None
    verify_value: Optional[object] = None
    last_request_path: Optional[str] = None

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
        path_value = diag.get("last_request_path")
        if path_value:
            last_request_path = path_value
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
        if http_block.get("last_exception_kind"):
            last_exception_kind = http_block.get("last_exception_kind")
        if verify_value is None and http_block.get("verify") is not None:
            verify_value = http_block.get("verify")
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
    if last_exception_kind:
        http_summary["last_exception_kind"] = last_exception_kind
    if verify_value is not None:
        http_summary["verify"] = verify_value

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

    if helix_last180_fallback_summary:
        helix_entry = dict(helix_last180_fallback_summary)
        if helix_entry.get("used"):
            fallback_used_total = True
        if fallback_summary:
            details_list = list(fallback_summary.get("details", []))
            details_list.append(helix_entry)
            fallback_summary["details"] = details_list
            try:
                fallback_summary["rows"] = int(fallback_summary.get("rows", 0) or 0) + int(
                    helix_entry.get("rows", 0) or 0
                )
            except (TypeError, ValueError):
                fallback_summary["rows"] = helix_entry.get("rows")
            if "resource_url" not in fallback_summary and helix_entry.get("resource_url"):
                fallback_summary["resource_url"] = helix_entry.get("resource_url")
        else:
            fallback_summary = helix_entry
    if fallback_summary and fallback_diag_cache_fetch:
        for meta_key in (
            "dataset",
            "package_url",
            "package_status_code",
            "resource_url",
            "resource_status_code",
            "resource_bytes",
            "resource_content_length",
            "request_url",
            "status_code",
            "bytes",
            "content_length",
            "source_tag",
            "source",
            "helix_error",
            "hdx_attempt",
        ):
            meta_value = fallback_diag_cache_fetch.get(meta_key)
            if meta_value is not None and meta_key not in fallback_summary:
                fallback_summary[meta_key] = meta_value

    raw_rows_total = 0
    for frame in ordered_frames:
        if isinstance(frame, pd.DataFrame):
            raw_rows_total += int(frame.shape[0])
    if helix_last180_rows:
        raw_rows_total += int(helix_last180_rows)
    if fallback_rows_total:
        try:
            raw_rows_total += int(fallback_rows_total)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            raw_rows_total += 0
    rows_normalized = int(total_rows)
    rows_written = rows_normalized

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
            "dir": cache_dir_text,
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
        "last_request_path": last_request_path,
        "performance": performance,
        "rate_limit": rate_limit_info,
        "chunks": chunks_info,
        "network_mode": network_mode,
        "requested_network_mode": requested_network_mode,
        "helix_failover": helix_failover_active,
        "http_status_counts": status_counts_serialized
        if network_mode in {"live", "helix"}
        else None,
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
        "endpoint_outcomes": dict(endpoint_outcomes),
    }
    diagnostics["rows_fetched"] = max(raw_rows_total, 0)
    diagnostics["rows_normalized"] = rows_normalized
    diagnostics["rows_written"] = rows_written
    if helix_last180_diag is not None and helix_last180_fallback_summary:
        helix_last180_diag = dict(helix_last180_diag)
        helix_last180_diag.setdefault("fallback", helix_last180_fallback_summary)
    if helix_last180_diag is not None:
        diagnostics["helix_last180"] = dict(helix_last180_diag)
    if network_mode == "helix":
        helix_block = dict(helix_summary or {})
        helix_block.setdefault("raw_rows", int((helix_summary or {}).get("raw_rows", 0) or 0))
        helix_block.setdefault("status_counts", status_counts_serialized)
        diagnostics["helix"] = helix_block
        if helix_zero_reason_global and total_rows == 0:
            diagnostics.setdefault("zero_rows_reason", helix_zero_reason_global)
    if network_mode in {"live", "helix"}:
        diagnostics["http_extended"] = {
            "status_counts": dict(status_counts),
            "timeouts": total_timeouts,
            "requests_ok_2xx": total_ok_2xx,
            "requests_4xx": total_4xx,
            "requests_5xx": total_5xx,
            "requests_other": total_other_responses,
            "other_exceptions": total_other_exceptions,
            "exceptions_by_type": dict(exceptions_by_type),
            "requests_planned": int(
                http_attempt_summary_totals.get("planned", 0) or len(jobs)
            ),
            "requests_executed": total_requests,
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


class IdmcClient:
    """Lightweight wrapper to prepare configuration for IDMC fetches."""

    def __init__(self, *, helix_client_id: Optional[str] = None) -> None:
        env_value = os.getenv("IDMC_HELIX_CLIENT_ID")
        env_cleaned = env_value.strip() if env_value is not None else None
        self.helix_client_id = helix_client_id or (env_cleaned or None)
        from .cli import FetchMetrics  # local import to avoid cycle in type-checking

        self.metrics: FetchMetrics = FetchMetrics()

    def fetch(
        self,
        cfg: IdmcConfig,
        *,
        network_mode: NetworkMode = "live",
        soft_timeouts: bool = True,
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
        data, diagnostics = fetch(
            cfg,
            network_mode=network_mode,
            soft_timeouts=soft_timeouts,
            window_start=window_start,
            window_end=window_end,
            window_days=window_days,
            only_countries=only_countries,
            base_url=base_url,
            cache_ttl=cache_ttl,
            rate_per_sec=rate_per_sec,
            max_concurrency=max_concurrency,
            max_bytes=max_bytes,
            chunk_by_month=chunk_by_month,
            allow_hdx_fallback=allow_hdx_fallback,
            helix_client_id=self.helix_client_id,
        )
        try:
            from .cli import FetchMetrics  # local import avoids circular typing

            rows_fetched = int(diagnostics.get("rows_fetched", 0) or 0)
            if rows_fetched == 0:
                rows_fetched = sum(
                    int(frame.shape[0])
                    for frame in data.values()
                    if isinstance(frame, pd.DataFrame)
                )
            rows_normalized = int(diagnostics.get("rows_normalized", rows_fetched) or 0)
            rows_written = int(diagnostics.get("rows_written", rows_normalized) or 0)
            staged = diagnostics.get("rows_staged") or {}
            staged_counts = {
                str(name): int(value or 0) for name, value in staged.items() if value is not None
            }
            self.metrics = FetchMetrics(
                fetched=rows_fetched,
                normalized=rows_normalized,
                written=rows_written,
                staged=staged_counts,
            )
        except Exception:  # pragma: no cover - defensive metrics update
            pass
        return data, diagnostics
