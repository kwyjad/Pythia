"""Client implementation for the IDMC connector."""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .cache import cache_get, cache_key, cache_put
from .config import IdmcConfig
from .http import HttpRequestError, http_get

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "fixtures")
RAW_DIAG_DIR = os.path.join("diagnostics", "ingestion", "idmc", "raw")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


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


def _filter_window(frame: pd.DataFrame, window_days: Optional[int]) -> pd.DataFrame:
    if window_days is None:
        return frame
    if "displacement_date" not in frame.columns:
        return frame
    try:
        dates = pd.to_datetime(frame["displacement_date"], errors="coerce")
    except Exception:  # pragma: no cover - defensive
        return frame
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=max(window_days, 0))
    mask = (dates.dt.date >= start) & (dates.dt.date <= end)
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


def fetch_idu_json(
    cfg: IdmcConfig,
    *,
    base_url: Optional[str] = None,
    cache_ttl: Optional[int] = None,
    window_days: Optional[int] = None,
    only_countries: Iterable[str] | None = None,
    skip_network: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch the IDU flat JSON payload, honouring cache and filters."""

    base = (base_url or cfg.api.base_url).rstrip("/")
    endpoint = cfg.api.endpoints.get("idus_json", "/data/idus_view_flat")
    url = f"{base}{endpoint}"
    key = cache_key(url, params=None)
    cache_dir = cfg.cache.dir
    ttl_seconds = cache_ttl if cache_ttl is not None else cfg.cache.ttl_seconds
    cache_path = os.path.join(cache_dir, f"{key}.bin")

    http_info: Dict[str, Any] = {
        "requests": 0,
        "retries": 0,
        "status_last": None,
        "duration_s": 0.0,
        "backoff_s": 0.0,
        "latency_ms": {"p50": 0, "p95": 0, "max": 0},
        "attempt_durations_ms": [],
    }
    cache_stats: Dict[str, Any] = {
        "dir": cache_dir,
        "key": key,
        "path": cache_path,
        "ttl_seconds": ttl_seconds,
        "hit": False,
        "hits": 0,
        "misses": 0,
    }
    diagnostics: Dict[str, Any] = {
        "mode": "fixture",
        "url": url,
        "cache": cache_stats,
        "http": http_info,
        "filters": {},
        "raw_path": None,
    }

    use_cache_only = skip_network or cfg.cache.force_cache_only
    cache_entry = cache_get(cache_dir, key, None if use_cache_only else ttl_seconds)
    body: bytes | None = None
    if cache_entry is not None:
        body = cache_entry.body
        diagnostics["mode"] = "cache"
        cache_stats["hit"] = True
        cache_stats["hits"] = 1
        cache_stats.update(cache_entry.metadata)
    elif use_cache_only:
        payload = _read_json_fixture("idus_view_flat.json")
        rows = _normalise_rows(payload)
        frame = pd.DataFrame(rows)
        frame = _filter_window(frame, window_days)
        frame = _filter_countries(frame, only_countries or [])
        diagnostics["mode"] = "fixture"
        diagnostics["reason"] = "cache-miss-cache-only"
        filters = {
            "window_days": window_days,
            "countries": sorted({c.strip().upper() for c in only_countries or [] if c.strip()}),
            "rows_before": len(rows),
            "rows_after": len(frame),
        }
        diagnostics["filters"] = filters
        return frame.reset_index(drop=True), diagnostics

    if body is None and not use_cache_only:
        try:
            status, headers, body, http_diag = http_get(url, timeout=10.0, retries=2, backoff_s=0.5)
            http_info.update(
                {
                    "requests": http_diag.get("attempts", 1),
                    "retries": http_diag.get("retries", 0),
                    "status_last": status,
                    "duration_s": http_diag.get("duration_s", 0.0),
                    "backoff_s": http_diag.get("backoff_s", 0.0),
                    "exceptions": http_diag.get("exceptions", []),
                    "attempt_durations_ms": [
                        int(round(value * 1000))
                        for value in http_diag.get("attempt_durations_s", [])
                    ],
                }
            )
            attempt_latencies = http_info.get("attempt_durations_ms", [])
            if attempt_latencies:
                sorted_latencies = sorted(attempt_latencies)
                http_info["latency_ms"] = {
                    "p50": _percentile(sorted_latencies, 50),
                    "p95": _percentile(sorted_latencies, 95),
                    "max": max(sorted_latencies),
                }
            diagnostics["mode"] = "online"
            metadata = {
                "status": status,
                "headers": headers,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            }
            cache_entry = cache_put(cache_dir, key, body, metadata)
            cache_stats["misses"] = 1
            cache_stats.update(cache_entry.metadata)
        except HttpRequestError as exc:
            http_info.update(
                {
                    "requests": exc.diagnostics.get("attempts", 0),
                    "retries": exc.diagnostics.get("retries", 0),
                    "status_last": exc.diagnostics.get("status"),
                    "duration_s": exc.diagnostics.get("duration_s", 0.0),
                    "backoff_s": exc.diagnostics.get("backoff_s", 0.0),
                    "exceptions": exc.diagnostics.get("exceptions", []),
                    "attempt_durations_ms": [
                        int(round(value * 1000))
                        for value in exc.diagnostics.get("attempt_durations_s", [])
                    ],
                }
            )
            attempt_latencies = http_info.get("attempt_durations_ms", [])
            if attempt_latencies:
                sorted_latencies = sorted(attempt_latencies)
                http_info["latency_ms"] = {
                    "p50": _percentile(sorted_latencies, 50),
                    "p95": _percentile(sorted_latencies, 95),
                    "max": max(sorted_latencies),
                }
            diagnostics["mode"] = "fixture"
            diagnostics["reason"] = "http-error"
            diagnostics["error"] = exc.diagnostics
            payload = _read_json_fixture("idus_view_flat.json")
            rows = _normalise_rows(payload)
            frame = pd.DataFrame(rows)
            frame = _filter_window(frame, window_days)
            frame = _filter_countries(frame, only_countries or [])
            filters = {
                "window_days": window_days,
                "countries": sorted({c.strip().upper() for c in only_countries or [] if c.strip()}),
                "rows_before": len(rows),
                "rows_after": len(frame),
            }
            diagnostics["filters"] = filters
            if not cache_stats["hit"]:
                cache_stats["misses"] = 1
            return frame.reset_index(drop=True), diagnostics

    if body is None:
        payload = _read_json_fixture("idus_view_flat.json")
        rows = _normalise_rows(payload)
        frame = pd.DataFrame(rows)
        frame = _filter_window(frame, window_days)
        frame = _filter_countries(frame, only_countries or [])
        diagnostics["mode"] = diagnostics.get("mode", "fixture")
        diagnostics["reason"] = diagnostics.get("reason", "cache-miss")
        filters = {
            "window_days": window_days,
            "countries": sorted({c.strip().upper() for c in only_countries or [] if c.strip()}),
            "rows_before": len(rows),
            "rows_after": len(frame),
        }
        diagnostics["filters"] = filters
        if not cache_stats["hit"]:
            cache_stats["misses"] = 1
        return frame.reset_index(drop=True), diagnostics

    payload = json.loads(body.decode("utf-8"))
    rows = _normalise_rows(payload)
    frame = pd.DataFrame(rows)
    frame = _filter_window(frame, window_days)
    frame = _filter_countries(frame, only_countries or [])
    filters = {
        "window_days": window_days,
        "countries": sorted({c.strip().upper() for c in only_countries or [] if c.strip()}),
        "rows_before": len(rows),
        "rows_after": len(frame),
    }
    diagnostics["filters"] = filters
    diagnostics["raw_path"] = _write_raw_snapshot(key, body)
    if not cache_stats["hit"]:
        cache_stats["misses"] = 1
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
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """Return payloads for downstream normalization and diagnostics."""

    data: Dict[str, pd.DataFrame] = {}

    countries = list(only_countries or cfg.api.countries)
    idu_frame, idu_diag = fetch_idu_json(
        cfg,
        base_url=base_url,
        cache_ttl=cache_ttl,
        window_days=window_days,
        only_countries=countries,
        skip_network=skip_network,
    )

    data["monthly_flow"] = idu_frame

    http_diag = idu_diag.get("http", {})
    cache_diag = idu_diag.get("cache", {})
    latency_block = http_diag.get("latency_ms") or {}
    http_block = {
        "requests": int(http_diag.get("requests", 0) or 0),
        "retries": int(http_diag.get("retries", 0) or 0),
        "status_last": http_diag.get("status_last"),
        "latency_ms": {
            "p50": int((latency_block.get("p50") or 0)),
            "p95": int((latency_block.get("p95") or 0)),
            "max": int((latency_block.get("max") or 0)),
        },
        "cache": {
            "hits": int(cache_diag.get("hits", 0) or 0),
            "misses": int(cache_diag.get("misses", 0) or 0),
        },
    }
    diagnostics: Dict[str, Any] = {
        "mode": idu_diag.get("mode", "offline"),
        "http": http_block,
        "cache": cache_diag,
        "filters": idu_diag.get("filters", {}),
        "raw_path": idu_diag.get("raw_path"),
    }
    return data, diagnostics
