#!/usr/bin/env python3
"""DTM connector that fetches displacement data exclusively through the official API."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import os
import pathlib
import random
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import pandas as pd
import yaml
import requests
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from resolver.ingestion._manifest import ensure_manifest_for_csv
from resolver.ingestion._shared.run_io import count_csv_rows, write_json
from resolver.ingestion.diagnostics_emitter import (
    append_jsonl as diagnostics_append_jsonl,
    finalize_run as diagnostics_finalize_run,
    start_run as diagnostics_start_run,
)
from resolver.ingestion.dtm_auth import get_dtm_api_key
from resolver.ingestion.utils import ensure_headers, flow_from_stock, month_start, stable_digest, to_iso3
from resolver.ingestion.utils.io import resolve_ingestion_window, resolve_output_path
from resolver.scripts.ingestion._dtm_debug_utils import (
    dump_json as diagnostics_dump_json,
    timing as diagnostics_timing,
    write_sample_csv as diagnostics_write_sample_csv,
)

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
STAGING = ROOT / "staging"
CONFIG_PATH = ROOT / "ingestion" / "config" / "dtm.yml"
DEFAULT_OUTPUT = ROOT / "staging" / "dtm_displacement.csv"
OUT_PATH = resolve_output_path(DEFAULT_OUTPUT)
OUT_DIR = OUT_PATH.parent
OUTPUT_PATH = OUT_PATH
META_PATH = OUT_PATH.with_suffix(OUT_PATH.suffix + ".meta.json")
HTTP_TRACE_PATH = OUT_DIR / "dtm_http.ndjson"
DIAGNOSTICS_DIR = REPO_ROOT / "diagnostics" / "ingestion"
DTM_DIAGNOSTICS_DIR = DIAGNOSTICS_DIR / "dtm"
DIAGNOSTICS_RAW_DIR = DIAGNOSTICS_DIR / "raw"
DIAGNOSTICS_METRICS_DIR = DIAGNOSTICS_DIR / "metrics"
DIAGNOSTICS_SAMPLES_DIR = DIAGNOSTICS_DIR / "samples"
DIAGNOSTICS_LOG_DIR = DIAGNOSTICS_DIR / "logs"
CONNECTORS_REPORT = DIAGNOSTICS_DIR / "connectors_report.jsonl"
RUN_DETAILS_PATH = DIAGNOSTICS_DIR / "dtm_run.json"
API_REQUEST_PATH = DIAGNOSTICS_DIR / "dtm_api_request.json"
API_SAMPLE_PATH = DIAGNOSTICS_DIR / "dtm_api_sample.json"
API_RESPONSE_SAMPLE_PATH = DIAGNOSTICS_DIR / "dtm_api_response_sample.json"
DISCOVERY_SNAPSHOT_PATH = DTM_DIAGNOSTICS_DIR / "discovery_countries.csv"
DISCOVERY_FAIL_PATH = DTM_DIAGNOSTICS_DIR / "discovery_fail.json"
DTM_HTTP_LOG_PATH = DTM_DIAGNOSTICS_DIR / "dtm_http.ndjson"
DISCOVERY_RAW_JSON_PATH = DIAGNOSTICS_RAW_DIR / "dtm_countries.json"
PER_COUNTRY_METRICS_PATH = DIAGNOSTICS_METRICS_DIR / "dtm_per_country.jsonl"
SAMPLE_ROWS_PATH = DIAGNOSTICS_SAMPLES_DIR / "dtm_sample.csv"
DTM_CLIENT_LOG_PATH = DIAGNOSTICS_LOG_DIR / "dtm_client.log"
STATIC_DATA_DIR = pathlib.Path(__file__).resolve().parent / "static"
STATIC_ISO3_PATH = STATIC_DATA_DIR / "iso3_master.csv"
METRICS_SUMMARY_PATH = DIAGNOSTICS_METRICS_DIR / "metrics.json"
SAMPLE_ADMIN0_PATH = DIAGNOSTICS_SAMPLES_DIR / "sample_admin0.csv"

CANONICAL_HEADERS = [
    "source",
    "country_iso3",
    "admin1",
    "event_id",
    "as_of",
    "month_start",
    "value_type",
    "value",
    "unit",
    "method",
    "confidence",
    "raw_event_id",
    "raw_fields_json",
]

SERIES_INCIDENT = "incident"
SERIES_CUMULATIVE = "cumulative"

__all__ = [
    "CANONICAL_HEADERS",
    "SERIES_INCIDENT",
    "SERIES_CUMULATIVE",
    "compute_monthly_deltas",
    "rollup_subnational",
    "infer_hazard",
    "load_config",
    "load_registries",
    "build_rows",
    "main",
]


@dataclass(frozen=True)
class Hazard:
    code: str
    label: str
    hclass: str


@dataclass
class DiscoveryResult:
    countries: List[str]
    frame: pd.DataFrame
    stage_used: Optional[str]
    report: Dict[str, Any]


MULTI_HAZARD = Hazard("multi", "Multi-shock Displacement/Needs", "all")
UNKNOWN_HAZARD = Hazard("UNK", "Unknown / Unspecified", "all")

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
DTM_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
for directory in (
    DIAGNOSTICS_RAW_DIR,
    DIAGNOSTICS_METRICS_DIR,
    DIAGNOSTICS_SAMPLES_DIR,
    DIAGNOSTICS_LOG_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

_LEGACY_DIAGNOSTICS_DIR = ROOT / "diagnostics" / "ingestion"


def _mirror_legacy_diagnostics() -> None:
    try:
        _LEGACY_DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
        for path in (RUN_DETAILS_PATH, CONNECTORS_REPORT):
            if path.exists():
                target = _LEGACY_DIAGNOSTICS_DIR / path.name
                target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        LOG.debug(
            "Compat mirror to resolver/diagnostics failed (non-fatal)",
            exc_info=True,
        )

LOG = logging.getLogger("resolver.ingestion.dtm")

_FILE_LOGGING_INITIALIZED = False

OFFLINE = False

COLUMNS = CANONICAL_HEADERS

DEFAULT_CAUSE = "unknown"
HTTP_COUNT_KEYS = ("2xx", "4xx", "5xx", "timeout", "error")
ROW_COUNT_KEYS = ("admin0", "admin1", "admin2", "total")

ADMIN_METHODS = {
    "admin0": "get_idp_admin0",
    "admin1": "get_idp_admin1",
    "admin2": "get_idp_admin2",
}


def _is_no_country_match_error(err: BaseException) -> bool:
    """Return ``True`` when *err* carries the discovery soft-skip signature."""

    try:
        message = str(err).lower()
        return "no country found matching your query" in message
    except Exception:
        return False


DISCOVERY_ERROR_LOG: List[Dict[str, Any]] = []
_ADMIN0_SAMPLE_LIMIT = 5
_ADMIN0_SAMPLE_WRITTEN = 0
RETRYABLE_EXCEPTIONS = (
    requests.exceptions.Timeout,
    requests.exceptions.ConnectionError,
)


def _ensure_out_dir_exists(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_header_only_csv(path: Path, headers: Sequence[str]) -> None:
    _ensure_out_dir_exists(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers))
        handle.write("\n")


def _write_http_trace_placeholder(path: Path, *, offline: bool) -> None:
    payload = {
        "offline": bool(offline),
        "ts": datetime.now(timezone.utc).isoformat(),
        "note": "placeholder trace for tests",
    }
    try:
        _ensure_out_dir_exists(path.parent)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload))
            handle.write("\n")
    except Exception:  # pragma: no cover - diagnostics helper
        LOG.debug("dtm: unable to persist HTTP trace placeholder", exc_info=True)


def ensure_zero_row_outputs(*, offline: bool) -> None:
    ensure_header_only()
    trace_path = Path(HTTP_TRACE_PATH)
    _write_http_trace_placeholder(trace_path, offline=offline)


def _append_connectors_report(
    *,
    mode: str,
    status: str,
    rows: int,
    reason: Optional[str] = None,
    extras: Optional[Mapping[str, Any]] = None,
    http: Optional[Mapping[str, Any]] = None,
    counts: Optional[Mapping[str, Any]] = None,
) -> None:
    http_payload: Dict[str, Any] = {key: 0 for key in HTTP_COUNT_KEYS}
    http_payload["retries"] = 0
    http_payload["last_status"] = None
    http_payload["rate_limit_remaining"] = None
    http_payload.update(http or {})

    counts_payload: Dict[str, Any] = {"fetched": 0, "normalized": rows, "written": rows}
    counts_payload.update(counts or {})

    extras_payload: Dict[str, Any] = {
        "mode": mode,
        "rows_total": rows,
        "status_raw": status,
        "exit_code": 0,
    }
    extras_payload.update(extras or {})

    _write_connector_report(
        status=status,
        reason=reason or f"{mode}: {status}",
        extras=extras_payload,
        http=http_payload,
        counts=counts_payload,
    )


class ConfigDict(dict):
    """Dictionary subclass that retains the source path for logging."""

    _source_path: Optional[str] = None


def _preflight_dependencies() -> Tuple[Dict[str, Any], bool]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "sys_path_entries": len(sys.path),
        "sys_path_sample": [str(entry) for entry in sys.path[:3]],
        "packages": [],
        "missing": [],
    }

    required = ("dtmapi",)
    for name in required:
        try:
            module = importlib.import_module(name)
        except Exception:
            info["missing"].append(name)
            LOG.debug("dtm: dependency %s import failed", name, exc_info=True)
        else:
            version = getattr(module, "__version__", "unknown")
            info["packages"].append({"name": name, "version": str(version)})

    return info, not info["missing"]


def _log_dependency_snapshot(info: Mapping[str, Any]) -> None:
    LOG.info(
        "Python runtime: %s (executable=%s)",
        info.get("python", "unknown"),
        info.get("executable", ""),
    )
    LOG.debug(
        "sys.path entries=%s first=%s",
        info.get("sys_path_entries"),
        info.get("sys_path_sample"),
    )
    packages = info.get("packages") if isinstance(info.get("packages"), list) else []
    if packages:
        LOG.info(
            "Dependency versions: %s",
            ", ".join(f"{pkg.get('name')}={pkg.get('version')}" for pkg in packages),
        )
    missing = info.get("missing") if isinstance(info.get("missing"), list) else []
    if missing:
        LOG.error("Missing dependencies: %s", ", ".join(map(str, missing)))
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "dtmapi"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:  # pragma: no cover - diagnostics only
        LOG.debug("dtm: unable to collect pip metadata for dtmapi", exc_info=True)
    else:
        snippet = (result.stdout or result.stderr or "").strip()
        if snippet:
            for line in snippet.splitlines()[:10]:
                LOG.debug("pip show dtmapi: %s", line.strip())


def _package_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
    except Exception:
        return "missing"
    return str(getattr(module, "__version__", "unknown"))


def _dtm_sdk_version() -> str:
    try:
        from dtmapi import DTMApi

        return str(getattr(DTMApi, "__version__", "unknown"))
    except Exception:  # pragma: no cover - diagnostics only
        try:
            module = importlib.import_module("dtmapi")
        except Exception:
            return "unknown"
        return str(getattr(module, "__version__", "unknown"))


def _persist_discovery_payload(payload: Mapping[str, Any]) -> None:
    try:
        DISCOVERY_FAIL_PATH.parent.mkdir(parents=True, exist_ok=True)
        DISCOVERY_FAIL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:  # pragma: no cover - diagnostics only
        LOG.debug("Unable to persist discovery failure diagnostics", exc_info=True)


def _clear_discovery_error_log() -> None:
    DISCOVERY_ERROR_LOG.clear()


def _write_discovery_failure(
    reason: str,
    *,
    message: str,
    hint: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> None:
    entry: Dict[str, Any] = {
        "timestamp": time.time(),
        "reason": reason,
        "message": message,
    }
    if hint:
        entry["hint"] = hint
    if extra:
        entry["extra"] = dict(extra)

    DISCOVERY_ERROR_LOG.append(entry)
    baseline = {
        "timestamp": time.time(),
        "stages": [],
        "errors": list(DISCOVERY_ERROR_LOG),
        "attempts": {},
        "latency_ms": {},
        "used_stage": None,
        "reason": reason,
    }
    _persist_discovery_payload(baseline)


def _write_discovery_report(report: Mapping[str, Any]) -> None:
    payload = {
        "timestamp": time.time(),
        "stages": list(report.get("stages", [])),
        "errors": list(DISCOVERY_ERROR_LOG) + list(report.get("errors", [])),
        "attempts": dict(report.get("attempts", {})),
        "latency_ms": dict(report.get("latency_ms", {})),
        "used_stage": report.get("used_stage"),
    }
    if "reason" in report:
        payload["reason"] = report["reason"]
    elif DISCOVERY_ERROR_LOG:
        payload["reason"] = DISCOVERY_ERROR_LOG[-1].get("reason")
    _persist_discovery_payload(payload)


def _append_metrics(entry: Mapping[str, Any]) -> None:
    try:
        PER_COUNTRY_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PER_COUNTRY_METRICS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str))
            handle.write("\n")
    except Exception:  # pragma: no cover - diagnostics only
        LOG.debug("Unable to append metrics entry", exc_info=True)


def _init_metrics_summary() -> Dict[str, Any]:
    return {
        "countries_attempted": 0,
        "countries_ok": 0,
        "countries_skipped_no_match": 0,
        "countries_failed_other": 0,
        "rows_fetched": 0,
        "duration_sec": 0.0,
        "stage_used": None,
    }


def _write_metrics_summary_file(summary: Mapping[str, Any]) -> None:
    try:
        METRICS_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        METRICS_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    except Exception:  # pragma: no cover - diagnostics only
        LOG.debug("Unable to persist metrics summary", exc_info=True)


def _reset_admin0_sample_counter() -> None:
    global _ADMIN0_SAMPLE_WRITTEN
    _ADMIN0_SAMPLE_WRITTEN = 0


def _ensure_sample_headers() -> None:
    SAMPLE_ADMIN0_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLE_ADMIN0_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["operation", "admin0Name", "admin0Pcode", "reportingDate", "idp_count"])
    _reset_admin0_sample_counter()


def _ensure_diagnostics_scaffolding() -> None:
    for directory in (
        DIAGNOSTICS_DIR,
        DTM_DIAGNOSTICS_DIR,
        DIAGNOSTICS_RAW_DIR,
        DIAGNOSTICS_METRICS_DIR,
        DIAGNOSTICS_SAMPLES_DIR,
        DIAGNOSTICS_LOG_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    _clear_discovery_error_log()
    _ensure_sample_headers()
    _write_metrics_summary_file(_init_metrics_summary())
    _write_discovery_report({})
    try:
        DTM_HTTP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        DTM_HTTP_LOG_PATH.write_text("", encoding="utf-8")
    except Exception:  # pragma: no cover - diagnostics only
        LOG.debug("Unable to reset DTM HTTP log", exc_info=True)


def _record_admin0_sample(df: pd.DataFrame, *, operation: Optional[str] = None) -> None:
    global _ADMIN0_SAMPLE_WRITTEN
    if df is None or df.empty:
        return
    limit = _ADMIN0_SAMPLE_LIMIT - _ADMIN0_SAMPLE_WRITTEN
    if limit <= 0:
        return
    rows: List[List[str]] = []
    for record in df.head(limit).to_dict("records"):
        op_value = operation or record.get("Operation") or record.get("operation") or ""
        name = (
            record.get("admin0Name")
            or record.get("Admin0Name")
            or record.get("CountryName")
            or record.get("country")
            or ""
        )
        code = (
            record.get("admin0Pcode")
            or record.get("Admin0Pcode")
            or record.get("CountryPcode")
            or record.get("CountryISO3")
            or record.get("ISO3")
            or record.get("iso3")
            or ""
        )
        iso = to_iso3(code, {}) if code else to_iso3(name, {})
        iso_value = (iso or str(code or "")).upper()
        report_date = (
            record.get("ReportingDate")
            or record.get("reportingDate")
            or record.get("ReportDate")
            or record.get("Date")
            or ""
        )
        idp_value = (
            record.get("idp_count")
            or record.get("TotalIDPs")
            or record.get("IDPTotal")
            or record.get("totalIdp")
            or record.get("Total")
            or ""
        )
        rows.append([str(op_value or ""), str(name or ""), iso_value, str(report_date or ""), str(idp_value or "")])
    if not rows:
        return
    SAMPLE_ADMIN0_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLE_ADMIN0_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    _ADMIN0_SAMPLE_WRITTEN += len(rows)


def _dtm_http_get(
    path: str,
    key: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
    timeout: Tuple[float, float],
) -> Any:
    base_url = "https://dtmapi.iom.int"
    url = f"{base_url}{path}"
    headers = {"Ocp-Apim-Subscription-Key": key}
    started = time.perf_counter()
    entry: Dict[str, Any] = {"ts": time.time(), "url": url, "ok": False, "nonce": round(random.random(), 6)}
    if params:
        entry["params"] = dict(params)
    response: Optional[requests.Response] = None
    try:
        if OFFLINE:
            entry["offline"] = True
            entry["ok"] = True
            entry["status"] = None
            LOG.debug("dtm: offline skip for HTTP GET %s", url)
            return []
        response = requests.get(url, headers=headers, params=params, timeout=timeout)
        entry["status"] = response.status_code
        response.raise_for_status()
        payload = response.json()
        entry["ok"] = True
        return payload
    except requests.exceptions.Timeout as exc:
        entry["error"] = str(exc)
        entry["timeout"] = True
        raise
    except Exception as exc:
        entry["error"] = str(exc)
        status = getattr(response, "status_code", None)
        if status is not None:
            entry["status"] = status
        raise
    finally:
        entry["elapsed_ms"] = int(max(0.0, (time.perf_counter() - started) * 1000))
        try:
            DTM_HTTP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with DTM_HTTP_LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, default=str))
                handle.write("\n")
        except Exception:  # pragma: no cover - diagnostics only
            LOG.debug("Unable to append DTM HTTP log entry", exc_info=True)


def _get_country_list_via_http(
    path: str,
    key: str,
    params: Optional[Mapping[str, Any]] = None,
    *,
    connect_timeout: float = 5.0,
    read_timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 1.5,
) -> pd.DataFrame:

    @retry(
        stop=stop_after_attempt(max(1, int(retries))),
        wait=wait_exponential_jitter(max(0.0, float(backoff))),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        reraise=True,
    )
    def _call(
        path: str,
        key: str,
        params: Optional[Mapping[str, Any]],
        connect_timeout: float,
        read_timeout: float,
    ) -> pd.DataFrame:
        payload = _dtm_http_get(
            path,
            key,
            params=params,
            timeout=(connect_timeout, read_timeout),
        )
        if isinstance(payload, pd.DataFrame):
            return payload
        return pd.DataFrame(payload or [])

    try:
        frame = _call(path, key, params, connect_timeout, read_timeout)
        attempts = int(_call.retry.statistics.get("attempt_number", 1))
    except RetryError as exc:
        attempts = exc.last_attempt.attempt_number if exc.last_attempt else int(retries)
        _get_country_list_via_http.last_attempts = int(attempts)
        raise
    _get_country_list_via_http.last_attempts = int(attempts)
    return frame


_get_country_list_via_http.last_attempts = 0  # type: ignore[attr-defined]


def _setup_file_logging() -> None:
    global _FILE_LOGGING_INITIALIZED
    if _FILE_LOGGING_INITIALIZED:
        return
    try:
        DTM_CLIENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(DTM_CLIENT_LOG_PATH, mode="w", encoding="utf-8")
    except Exception:  # pragma: no cover - diagnostics only
        LOG.debug("Unable to initialize DTM client file handler", exc_info=True)
        return
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(handler)
    _FILE_LOGGING_INITIALIZED = True


def _normalize_discovery_frame(raw: Any) -> pd.DataFrame:
    frame = raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(raw or [])
    if frame.empty:
        return pd.DataFrame(columns=["admin0Name", "admin0Pcode"])

    rename_map: Dict[str, str] = {}
    for column in frame.columns:
        key = str(column)
        lowered = key.strip().lower()
        if lowered in {
            "countryname",
            "admin0name",
            "operationname",
            "country",
            "admin0",
        }:
            rename_map[key] = "admin0Name"
        elif lowered in {
            "admin0pcode",
            "countrypcode",
            "countryiso3",
            "iso3",
            "operationiso3",
            "operationadmin0pcode",
        }:
            rename_map[key] = "admin0Pcode"
    if rename_map:
        frame = frame.rename(columns=rename_map)

    if "admin0Name" not in frame.columns:
        for candidate in ("CountryName", "OperationName", "country", "Country"):
            if candidate in frame.columns:
                frame["admin0Name"] = frame[candidate]
                break
    if "admin0Name" not in frame.columns:
        frame["admin0Name"] = ""

    if "admin0Pcode" not in frame.columns:
        for candidate in (
            "CountryISO3",
            "CountryPcode",
            "ISO3",
            "iso3",
            "Admin0Pcode",
            "OperationISO3",
        ):
            if candidate in frame.columns:
                frame["admin0Pcode"] = frame[candidate]
                break
    if "admin0Pcode" not in frame.columns:
        frame["admin0Pcode"] = ""

    frame["admin0Name"] = frame["admin0Name"].astype(str).str.strip()
    codes: List[str] = []
    for name, code in zip(frame["admin0Name"], frame["admin0Pcode"].astype(str)):
        iso = str(code or "").strip().upper()
        if len(iso) != 3 or not iso.isalpha():
            iso = (to_iso3(name, {}) or iso).upper()
        codes.append(iso)
    frame["admin0Pcode"] = codes
    frame = frame[(frame["admin0Name"] != "") & (frame["admin0Pcode"] != "")]
    frame = frame.drop_duplicates(subset=["admin0Name", "admin0Pcode"]).reset_index(drop=True)
    return frame[["admin0Name", "admin0Pcode"]]


def _load_static_iso3(path: Path | None = None) -> pd.DataFrame:
    roster_path = Path(path or STATIC_ISO3_PATH)
    if not roster_path.exists():
        LOG.warning("Static ISO3 roster missing at %s", roster_path)
        raise FileNotFoundError(f"Static ISO3 roster missing at {roster_path}")
    try:
        frame = pd.read_csv(
            roster_path,
            dtype=str,
            keep_default_na=False,
            engine="python",
            quoting=csv.QUOTE_MINIMAL,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.error("Failed to load static ISO3 roster from %s: %s", roster_path, exc)
        raise
    required = {"admin0Pcode", "admin0Name"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Static ISO3 roster missing columns {sorted(missing)}")
    frame["admin0Pcode"] = frame["admin0Pcode"].astype(str).str.strip()
    frame["admin0Name"] = frame["admin0Name"].astype(str).str.strip()
    frame = frame[(frame["admin0Pcode"] != "") & (frame["admin0Name"] != "")]
    if len(frame) < 180:
        raise ValueError(f"Static ISO3 roster unexpectedly small: {len(frame)} rows")
    return frame[["admin0Pcode", "admin0Name"]]


def _perform_discovery(
    cfg: Mapping[str, Any],
    *,
    metrics: Optional[MutableMapping[str, Any]] = None,
    api_key: Optional[str] = None,
) -> DiscoveryResult:
    _ensure_diagnostics_scaffolding()

    api_cfg = cfg.get("api", {}) if isinstance(cfg.get("api"), Mapping) else {}
    requested_countries_raw = api_cfg.get("countries")
    if isinstance(requested_countries_raw, (list, tuple)):
        requested_countries = [
            str(country).strip()
            for country in requested_countries_raw
            if str(country).strip()
        ]
    else:
        requested_countries = []
    if requested_countries:
        LOG.info(
            "Config requested countries=%s -> using explicit list (discovery bypassed)",
            requested_countries,
        )
        alias_map = cfg.get("country_aliases") or {}
        normalized_records = []
        for country in requested_countries:
            iso_candidate = to_iso3(country, alias_map) or country
            iso_value = str(iso_candidate or "").strip().upper()
            if len(iso_value) != 3 or not iso_value.isalpha():
                letters_only = "".join(ch for ch in country.upper() if ch.isalpha())
                iso_value = letters_only[:3]
                if len(iso_value) != 3:
                    iso_value = iso_value.ljust(3, "X")
            normalized_records.append(
                {"admin0Name": country, "admin0Pcode": iso_value or None}
            )
        discovered = pd.DataFrame(normalized_records, columns=["admin0Name", "admin0Pcode"])
        discovered = _normalize_discovery_frame(discovered)
        stage_entry = {
            "stage": "explicit_list",
            "status": "ok" if not discovered.empty else "empty",
            "rows": int(discovered.shape[0]),
            "attempts": 1,
            "latency_ms": 0,
        }
        report = {
            "stages": [stage_entry],
            "errors": []
            if not discovered.empty
            else [{"stage": "explicit_list", "message": "empty_result"}],
            "attempts": {"explicit_list": 1},
            "latency_ms": {"explicit_list": 0},
            "used_stage": "explicit_list",
        }
        _write_discovery_report(report)
        try:
            diagnostics_dump_json(discovered.to_dict(orient="records"), DISCOVERY_RAW_JSON_PATH)
        except Exception:  # pragma: no cover - diagnostics only
            LOG.debug("Unable to persist discovery JSON snapshot", exc_info=True)
        try:
            DISCOVERY_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            discovered.to_csv(DISCOVERY_SNAPSHOT_PATH, index=False)
        except Exception:  # pragma: no cover - diagnostics only
            LOG.debug("Unable to persist discovery snapshot", exc_info=True)
        if metrics is not None:
            metrics["countries_attempted"] = len(requested_countries)
            metrics["stage_used"] = "explicit_list"
            _write_metrics_summary_file(metrics)
        return DiscoveryResult(
            countries=requested_countries,
            frame=discovered,
            stage_used="explicit_list",
            report=report,
        )
    timeouts = api_cfg.get("timeouts", {}) if isinstance(api_cfg.get("timeouts"), Mapping) else {}
    retries_cfg = api_cfg.get("retries", {}) if isinstance(api_cfg.get("retries"), Mapping) else {}
    connect_timeout = float(timeouts.get("connect_seconds", 5))
    read_timeout = float(timeouts.get("read_seconds", 30))
    retry_attempts = int(retries_cfg.get("attempts", 3))
    backoff_seconds = float(retries_cfg.get("backoff_seconds", 1.5))
    discovery_order = api_cfg.get("discovery_order") or ["countries", "operations", "static_iso3"]
    if not isinstance(discovery_order, Iterable):
        discovery_order = ["countries", "operations", "static_iso3"]
    order = [str(stage).strip().lower() for stage in discovery_order if str(stage).strip()]
    if "static_iso3" not in order:
        order.append("static_iso3")

    key = (api_key or get_dtm_api_key() or "").strip()
    if not key and not OFFLINE:
        raise RuntimeError("Missing DTM_API_KEY environment variable.")

    stage_entries: List[Dict[str, Any]] = []
    stage_errors: List[Dict[str, Any]] = []
    attempts_map: Dict[str, int] = {}
    latency_map: Dict[str, int] = {}
    discovered = pd.DataFrame(columns=["admin0Name", "admin0Pcode"])
    used_stage: Optional[str] = None

    for stage in order:
        started = time.perf_counter()
        stage_name = stage
        try:
            if stage == "countries":
                raw = _get_country_list_via_http(
                    "/v3/displacement/country-list",
                    key,
                    connect_timeout=connect_timeout,
                    read_timeout=read_timeout,
                    retries=retry_attempts,
                    backoff=backoff_seconds,
                )
                attempts = int(getattr(_get_country_list_via_http, "last_attempts", 1))
            elif stage == "operations":
                raw = _get_country_list_via_http(
                    "/v3/displacement/operations-list",
                    key,
                    connect_timeout=connect_timeout,
                    read_timeout=read_timeout,
                    retries=retry_attempts,
                    backoff=backoff_seconds,
                )
                attempts = int(getattr(_get_country_list_via_http, "last_attempts", 1))
            elif stage == "static_iso3":
                raw = _load_static_iso3()
                attempts = 1
            else:
                LOG.debug("Skipping unknown discovery stage %s", stage)
                continue
            frame = _normalize_discovery_frame(raw)
            latency_ms = int(max(0.0, (time.perf_counter() - started) * 1000))
            attempts_map[stage_name] = int(attempts)
            latency_map[stage_name] = latency_ms
            status = "ok" if not frame.empty else "empty"
            stage_entry = {
                "stage": stage_name,
                "status": status,
                "rows": int(frame.shape[0]),
                "attempts": int(attempts),
                "latency_ms": latency_ms,
            }
            stage_entries.append(stage_entry)
            if frame.empty and stage != "static_iso3":
                stage_errors.append({"stage": stage_name, "message": "empty_result"})
                continue
            if frame.empty and stage == "static_iso3":
                stage_errors.append({"stage": stage_name, "message": "static_roster_empty"})
            discovered = frame
            used_stage = stage_name
            break
        except RetryError as exc:
            attempts = exc.last_attempt.attempt_number if exc.last_attempt else retry_attempts
            latency_ms = int(max(0.0, (time.perf_counter() - started) * 1000))
            stage_entries.append(
                {
                    "stage": stage_name,
                    "status": "error",
                    "rows": 0,
                    "attempts": int(attempts),
                    "latency_ms": latency_ms,
                }
            )
            attempts_map[stage_name] = int(attempts)
            latency_map[stage_name] = latency_ms
            error_message = str(exc.last_attempt.exception()) if exc.last_attempt else str(exc)
            stage_errors.append({"stage": stage_name, "message": error_message})
            continue
        except Exception as exc:
            latency_ms = int(max(0.0, (time.perf_counter() - started) * 1000))
            attempts = int(getattr(_get_country_list_via_http, "last_attempts", 1))
            stage_entries.append(
                {
                    "stage": stage_name,
                    "status": "error",
                    "rows": 0,
                    "attempts": attempts,
                    "latency_ms": latency_ms,
                }
            )
            attempts_map[stage_name] = attempts
            latency_map[stage_name] = latency_ms
            stage_errors.append({"stage": stage_name, "message": str(exc)})
            continue

    if used_stage is None and stage_entries:
        used_stage = stage_entries[-1]["stage"]

    countries = sorted(
        {
            str(name).strip()
            for name in discovered.get("admin0Name", pd.Series(dtype=str)).dropna().astype(str)
            if str(name).strip()
        }
    )

    report = {
        "stages": stage_entries,
        "errors": stage_errors,
        "attempts": attempts_map,
        "latency_ms": latency_map,
        "used_stage": used_stage,
    }
    _write_discovery_report(report)

    try:
        diagnostics_dump_json(discovered.to_dict(orient="records"), DISCOVERY_RAW_JSON_PATH)
    except Exception:  # pragma: no cover - diagnostics only
        LOG.debug("Unable to persist discovery JSON snapshot", exc_info=True)
    try:
        DISCOVERY_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        discovered.to_csv(DISCOVERY_SNAPSHOT_PATH, index=False)
    except Exception:  # pragma: no cover - diagnostics only
        LOG.debug("Unable to persist discovery snapshot", exc_info=True)

    LOG.info(
        "Discovery stages complete | used_stage=%s countries=%d", used_stage, len(countries)
    )

    if metrics is not None:
        metrics["countries_attempted"] = len(countries)
        metrics["stage_used"] = used_stage
        _write_metrics_summary_file(metrics)

    return DiscoveryResult(countries=countries, frame=discovered, stage_used=used_stage, report=report)


def _dump_head(df: pd.DataFrame, level: str, country: str, limit: int = 100) -> None:
    out_dir = DIAGNOSTICS_DIR / "raw" / "dtm"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_country = country.replace("/", "-").replace(" ", "_")
    path = out_dir / f"{level}.{safe_country}.head.csv"
    df.head(limit).to_csv(path, index=False)
    LOG.info("raw head written: %s (rows=%d)", path, min(limit, len(df)))


def _write_level_sample(df: pd.DataFrame, level: str, limit: int = 50) -> None:
    sample_path = DTM_DIAGNOSTICS_DIR / f"sample_{level}.csv"
    if sample_path.exists():
        return
    try:
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        df.head(limit).to_csv(sample_path, index=False)
        LOG.info("Sample rows written to %s", sample_path)
    except Exception:  # pragma: no cover - diagnostics only
        LOG.debug("Unable to persist sample rows", exc_info=True)


def _append_request_log(entry: Mapping[str, Any]) -> None:
    try:
        DTM_HTTP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with DTM_HTTP_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str))
            handle.write("\n")
    except Exception:  # pragma: no cover - diagnostics only
        LOG.debug("Unable to append DTM request log entry", exc_info=True)


def _resolve_idp_value_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    for alias in aliases:
        if alias in df.columns:
            return alias

    pattern = re.compile(r"(?i)^(?:num|total).*(idp)")
    candidates = [column for column in df.columns if pattern.search(column)]
    for column in candidates:
        if str(df[column].dtype).startswith(("int", "float")):
            LOG.warning("fallback idp_count column selected by regex: %s", column)
            return column
    return candidates[0] if candidates else None


def _write_connector_report(
    *,
    status: str,
    reason: str,
    extras: Optional[Mapping[str, Any]] = None,
    http: Optional[Mapping[str, Any]] = None,
    counts: Optional[Mapping[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "connector_id": "dtm_client",
        "mode": "real",
        "status": status,
        "reason": reason,
        "http": dict(http or {}),
        "counts": dict(counts or {}),
        "extras": dict(extras or {}),
    }
    payload.setdefault("http", {})
    http_payload = payload["http"]
    for key in ("2xx", "4xx", "5xx", "retries", "rate_limit_remaining", "last_status"):
        http_payload.setdefault(key, 0 if key != "last_status" else None)
    count_payload = payload["counts"]
    for key in ("fetched", "normalized", "written"):
        count_payload.setdefault(key, 0)
    extras_payload = payload["extras"]
    extras_payload.setdefault("rows_total", 0)
    extras_payload.setdefault("status_raw", status)
    if "exit_code" not in extras_payload:
        extras_payload["exit_code"] = 1 if status == "error" else 0
    CONNECTORS_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with CONNECTORS_REPORT.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str))
        handle.write("\n")
    _mirror_legacy_diagnostics()


def _append_summary_stub_if_needed(message: str) -> None:
    summary_path = DIAGNOSTICS_DIR / "summary.md"
    if summary_path.exists():
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_content = (
        "# Connector Diagnostics\n\n"
        f"* dtm_client: {message} *\n"
    )
    summary_path.write_text(summary_content, encoding="utf-8")


def _extract_status_code(exc: Exception) -> Optional[int]:
    message = str(exc)
    for token in re.findall(r"(\d{3})", message):
        try:
            code = int(token)
        except ValueError:
            continue
        if 400 <= code < 600:
            return code
    return None

class DTMHttpError(RuntimeError):
    """Error raised when the DTM API returns an HTTP error status."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code


class DTMUnauthorizedError(DTMHttpError):
    """Error raised when authentication with the DTM API fails."""

    def __init__(self, status_code: int, message: str):
        super().__init__(status_code, message)


def _auth_probe(api_key: str, *, offline: bool = False) -> None:
    key = (api_key or "").strip()
    if key:
        return
    if offline:
        LOG.debug("dtm: offline auth probe skipped (no API key present)")
        return
    raise DTMUnauthorizedError(401, "Missing DTM_API_KEY or DTM_SUBSCRIPTION_KEY")


class DTMZeroRowsError(RuntimeError):
    """Raised when the API returns zero rows for the requested window."""

    def __init__(self, window_label: str, reason: str):
        super().__init__(f"DTM: All requests returned 0 rows in window {window_label}")
        self.window_label = window_label
        self.zero_rows_reason = reason


class DTMApiClient:
    """Minimal wrapper around the official dtmapi client used by the connector."""

    def __init__(self, config: Mapping[str, Any], *, subscription_key: Optional[str] = None):
        try:
            from dtmapi import DTMApi
        except ImportError as exc:  # pragma: no cover - defensive guard
            LOG.error(
                "Failed to import dtmapi package. Install with: pip install dtmapi>=0.1.5",
            )
            raise RuntimeError(
                "dtmapi package not installed. Run: pip install dtmapi>=0.1.5",
            ) from exc

        raw_key = (subscription_key or "").strip()
        env_key = get_dtm_api_key()
        api_key = raw_key or (env_key or "")
        if not api_key and not OFFLINE:
            raise RuntimeError("Missing DTM_API_KEY environment variable.")

        _setup_file_logging()
        if OFFLINE:
            self.client = type("OfflineDTM", (), {})()
            LOG.info(
                "DTM SDK offline stub initialized | python=%s | offline=%s",
                sys.version.split()[0],
                True,
            )
        else:
            masked_suffix = f"...{api_key[-4:]}" if api_key else "<missing>"
            self.client = DTMApi(subscription_key=api_key)
            LOG.info(
                "DTM SDK initialized | python=%s | dtmapi=%s | base_url=%s | key_suffix=%s",
                sys.version.split()[0],
                getattr(DTMApi, "__version__", "unknown"),
                getattr(self.client, "base_url", getattr(self.client, "_base_url", "unknown")),
                masked_suffix,
            )
        self.config = config
        api_cfg = config.get("api", {})
        self.rate_limit_delay = float(api_cfg.get("rate_limit_delay", 1.0))
        self.timeout = int(api_cfg.get("timeout", 60))
        self._http_counts: Dict[str, int] = {}

    def _record_success(self, http_counts: Optional[MutableMapping[str, int]], status: int) -> None:
        if http_counts is None:
            return
        http_counts["last_status"] = status
        bucket = "2xx" if 200 <= status < 300 else "4xx" if 400 <= status < 500 else "5xx"
        http_counts[bucket] = http_counts.get(bucket, 0) + 1

    def _record_failure(
        self,
        exc: Exception,
        http_counts: Optional[MutableMapping[str, int]],
    ) -> None:
        message = str(exc).lower()
        key = "error"
        if "timeout" in message:
            key = "timeout"
        elif "404" in message or "not found" in message:
            key = "4xx"
        elif "401" in message or "403" in message:
            key = "4xx"
        elif "500" in message or "server" in message:
            key = "5xx"
        if http_counts is not None:
            http_counts[key] = http_counts.get(key, 0) + 1
        status = _extract_status_code(exc)
        if status in {401, 403}:
            raise DTMUnauthorizedError(status, str(exc)) from exc
        if status:
            raise DTMHttpError(status, str(exc)) from exc
        raise DTMHttpError(0, str(exc)) from exc

    def get_countries(self, http_counts: Optional[MutableMapping[str, int]] = None) -> pd.DataFrame:
        if OFFLINE:
            LOG.debug("dtm: offline skip for get_all_countries")
            return pd.DataFrame()
        try:
            df = self.client.get_all_countries()
            self._record_success(http_counts, 200)
            return df
        except Exception as exc:
            LOG.error("Failed to fetch countries: %s", exc)
            self._record_failure(exc, http_counts)
            raise

    def get_idp_admin0(
        self,
        *,
        country: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        http_counts: Optional[MutableMapping[str, int]] = None,
    ) -> pd.DataFrame:
        if OFFLINE:
            LOG.debug("dtm: offline skip for get_idp_admin0 country=%s", country)
            return pd.DataFrame()
        try:
            frame = self.client.get_idp_admin0_data(
                CountryName=country,
                FromReportingDate=from_date,
                ToReportingDate=to_date,
            )
            self._record_success(http_counts, 200)
            if self.rate_limit_delay:
                time.sleep(self.rate_limit_delay)
            return frame
        except ValueError as exc:
            if _is_no_country_match_error(exc):
                if http_counts is not None:
                    http_counts["skip_no_match"] = http_counts.get("skip_no_match", 0) + 1
                self._http_counts["skip_no_match"] = self._http_counts.get("skip_no_match", 0) + 1
                LOG.warning("Skipping unsupported country (no match): %s", country or "<unspecified>")
                return None
            LOG.error("Admin0 request failed: %s", exc)
            self._record_failure(exc, http_counts)
            raise
        except Exception as exc:
            LOG.error("Admin0 request failed: %s", exc)
            self._record_failure(exc, http_counts)
            raise

    def get_idp_admin1(
        self,
        *,
        country: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        http_counts: Optional[MutableMapping[str, int]] = None,
    ) -> pd.DataFrame:
        if OFFLINE:
            LOG.debug("dtm: offline skip for get_idp_admin1 country=%s", country)
            return pd.DataFrame()
        try:
            frame = self.client.get_idp_admin1_data(
                CountryName=country,
                FromReportingDate=from_date,
                ToReportingDate=to_date,
            )
            self._record_success(http_counts, 200)
            if self.rate_limit_delay:
                time.sleep(self.rate_limit_delay)
            return frame
        except ValueError as exc:
            if _is_no_country_match_error(exc):
                if http_counts is not None:
                    http_counts["skip_no_match"] = http_counts.get("skip_no_match", 0) + 1
                self._http_counts["skip_no_match"] = self._http_counts.get("skip_no_match", 0) + 1
                LOG.warning("Skipping unsupported country (no match): %s", country or "<unspecified>")
                return None
            LOG.error("Admin1 request failed: %s", exc)
            self._record_failure(exc, http_counts)
            raise
        except Exception as exc:
            LOG.error("Admin1 request failed: %s", exc)
            self._record_failure(exc, http_counts)
            raise

    def get_idp_admin2(
        self,
        *,
        country: Optional[str] = None,
        operation: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        http_counts: Optional[MutableMapping[str, int]] = None,
    ) -> pd.DataFrame:
        if OFFLINE:
            LOG.debug("dtm: offline skip for get_idp_admin2 country=%s operation=%s", country, operation)
            return pd.DataFrame()
        try:
            params = {
                "CountryName": country,
                "FromReportingDate": from_date,
                "ToReportingDate": to_date,
            }
            if operation:
                params["Operation"] = operation
            frame = self.client.get_idp_admin2_data(**params)
            self._record_success(http_counts, 200)
            if self.rate_limit_delay:
                time.sleep(self.rate_limit_delay)
            return frame
        except ValueError as exc:
            if _is_no_country_match_error(exc):
                if http_counts is not None:
                    http_counts["skip_no_match"] = http_counts.get("skip_no_match", 0) + 1
                self._http_counts["skip_no_match"] = self._http_counts.get("skip_no_match", 0) + 1
                LOG.warning("Skipping unsupported country (no match): %s", country or "<unspecified>")
                return None
            LOG.error("Admin2 request failed: %s", exc)
            self._record_failure(exc, http_counts)
            raise
        except Exception as exc:
            LOG.error("Admin2 request failed: %s", exc)
            self._record_failure(exc, http_counts)
            raise


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "y", "yes", "on"}


def _ensure_http_counts(counts: Optional[MutableMapping[str, int]]) -> MutableMapping[str, int]:
    base: MutableMapping[str, int] = counts or {}
    for key in HTTP_COUNT_KEYS:
        base.setdefault(key, 0)
    base.setdefault("retries", 0)
    base.setdefault("skip_no_match", 0)
    base.setdefault("last_status", None)
    return base


def _empty_row_counts() -> Dict[str, int]:
    return {key: 0 for key in ROW_COUNT_KEYS}


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_iso_date_or_none(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    if len(text) >= 7 and text[4] == "-":
        return f"{text[:7]}-01"
    if len(text) >= 10 and text[4] in "./" and text[7] in "./":
        return f"{text[:4]}-{text[5:7]}-{text[8:10]}"
    return None


def _coerce_numeric(value: Any) -> float:
    parsed = _parse_float(value)
    return float(parsed) if parsed is not None else 0.0


def _normalize_series_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {SERIES_CUMULATIVE, SERIES_INCIDENT}:
        return text
    return SERIES_INCIDENT


def rollup_subnational(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate subnational rows to country totals before delta logic.

    The helper keeps legacy semantics used by notebooks and tests:

    * rows are grouped by (iso3, hazard_code, metric, as_of_date, series_type, source_id)
    * admin-level identifiers (admin1/admin2) are dropped from the output
    * numeric values are summed, non-numeric fields keep their first occurrence
    """

    aggregated: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("iso3"),
            row.get("hazard_code"),
            row.get("metric"),
            row.get("as_of_date"),
            _normalize_series_type(row.get("series_type")),
            row.get("source_id"),
        )
        current = aggregated.get(key)
        if current is None:
            current = dict(row)
            current.pop("admin1", None)
            current.pop("admin2", None)
            current["series_type"] = key[4]
            current["value"] = _coerce_numeric(row.get("value"))
            aggregated[key] = current
        else:
            current["value"] = _coerce_numeric(current.get("value")) + _coerce_numeric(
                row.get("value")
            )
    return list(aggregated.values())


def compute_monthly_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    value_field: str = "value",
    date_field: str = "as_of_date",
    group_fields: Sequence[str] = ("iso3", "hazard_code", "metric", "source_id"),
    allow_first_month: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Convert cumulative series to month-over-month deltas.

    Incident (already-monthly) rows are passed through unchanged. Cumulative
    series drop the first month unless `allow_first_month` (or the
    `DTM_ALLOW_FIRST_MONTH` env flag) is enabled. Negative deltas are clipped to
    zero to avoid spurious drops.
    """

    if allow_first_month is None:
        allow_first_month = _env_bool("DTM_ALLOW_FIRST_MONTH", False)

    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in group_fields)
        grouped[key].append(dict(row))

    output: List[Dict[str, Any]] = []
    for _, bucket in grouped.items():
        bucket.sort(key=lambda item: str(item.get(date_field, "")))
        series_type = _normalize_series_type(bucket[0].get("series_type"))
        if series_type == SERIES_INCIDENT:
            for item in bucket:
                clone = dict(item)
                parsed = _parse_float(clone.get(value_field))
                if parsed is not None:
                    clone[value_field] = parsed
                output.append(clone)
            continue

        previous_value: Optional[float] = None
        for idx, item in enumerate(bucket):
            current_value = _parse_float(item.get(value_field))
            if current_value is None:
                continue
            if previous_value is None:
                previous_value = current_value
                if allow_first_month:
                    clone = dict(item)
                    clone[value_field] = current_value
                    clone["series_type"] = SERIES_INCIDENT
                    output.append(clone)
                continue
            delta = max(current_value - previous_value, 0.0)
            previous_value = current_value
            if idx == 0 and not allow_first_month:
                continue
            clone = dict(item)
            clone[value_field] = delta
            clone["series_type"] = SERIES_INCIDENT
            output.append(clone)

    return output


def infer_hazard(
    texts: Iterable[str],
    *,
    shocks: Optional[pd.DataFrame],
    keywords_cfg: Mapping[str, Sequence[str]] | None = None,
    default_key: Optional[str] = "displacement_influx",
) -> Hazard:
    """Map free-form text snippets to a canonical hazard definition."""

    combined = " ".join(str(text or "") for text in texts).lower()
    keywords_cfg = keywords_cfg or {}
    matched_keys: List[str] = []
    for key, keywords in keywords_cfg.items():
        for keyword in keywords or []:
            needle = str(keyword or "").strip().lower()
            if needle and needle in combined:
                matched_keys.append(key)
                break

    if not matched_keys and default_key:
        matched_keys.append(default_key)

    # Deduplicate while preserving order
    seen = set()
    ordered_matches: List[str] = []
    for key in matched_keys:
        lowered = key.lower()
        if lowered not in seen:
            seen.add(lowered)
            ordered_matches.append(key)

    if not ordered_matches:
        return UNKNOWN_HAZARD
    if len(ordered_matches) > 1:
        return MULTI_HAZARD

    selected_key = ordered_matches[0]
    if shocks is not None and not shocks.empty:
        lowered_key = selected_key.lower()
        mask = shocks.get("key", pd.Series(dtype=str)).str.lower() == lowered_key
        if mask.any():
            record = shocks.loc[mask].iloc[0]
        else:
            mask = shocks.get("code", pd.Series(dtype=str)).str.lower() == lowered_key
            record = shocks.loc[mask].iloc[0] if mask.any() else None
        if record is not None:
            code = str(record.get("code") or record.get("key") or selected_key).strip() or "UNK"
            label = str(record.get("name") or selected_key).strip() or code
            hclass = str(record.get("hclass") or "all").strip() or "all"
            return Hazard(code=code, label=label, hclass=hclass)

    label = selected_key.replace("_", " ").title()
    code = selected_key.upper()[:3] or "UNK"
    return Hazard(code=code, label=label, hclass="all")


def _is_candidate_newer(existing_iso: str, candidate_iso: str) -> bool:
    if not candidate_iso:
        return False
    if not existing_iso:
        return True
    return candidate_iso > existing_iso


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        cfg = ConfigDict()
        cfg._source_path = str(CONFIG_PATH)
        return cfg
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    cfg = ConfigDict(data if isinstance(data, dict) else {})
    cfg._source_path = str(CONFIG_PATH)
    return cfg


def load_registries() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return minimal registry stubs for tests and CLI helpers."""

    countries = pd.DataFrame(
        [
            {"iso3": "AAA", "name": "Country A"},
            {"iso3": "BBB", "name": "Country B"},
            {"iso3": "CCC", "name": "Country C"},
        ]
    )
    shocks = pd.DataFrame(
        [
            {
                "code": "DI",
                "key": "displacement_influx",
                "name": "Displacement influx",
                "hclass": "all",
            },
            {"code": "FL", "key": "flood", "name": "Flood", "hclass": "hydro"},
            {"code": "DR", "key": "drought", "name": "Drought", "hclass": "climate"},
            {
                "code": "multi",
                "key": "multi",
                "name": "Multi-shock Displacement/Needs",
                "hclass": "all",
            },
        ]
    )
    return countries, shocks


def ensure_header_only() -> None:
    ensure_headers(OUT_PATH, COLUMNS)
    ensure_manifest_for_csv(OUT_PATH, schema_version="dtm_displacement.v1", source_id="dtm")


def _previous_month(month_start_dt: datetime) -> datetime:
    year = month_start_dt.year
    month = month_start_dt.month
    if month == 1:
        year -= 1
        month = 12
    else:
        month -= 1
    return month_start_dt.replace(year=year, month=month)


def _generate_offline_rows(reference: Optional[datetime] = None) -> List[List[Any]]:
    now = reference or datetime.now(timezone.utc)
    as_of_iso = now.isoformat()
    current_month = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    previous_month = _previous_month(current_month)

    samples = [
        {"iso3": "ETH", "admin1": "Amhara", "month": previous_month, "value": 120},
        {"iso3": "ETH", "admin1": "Amhara", "month": current_month, "value": 95},
        {"iso3": "PHL", "admin1": "Western Visayas", "month": previous_month, "value": 80},
        {"iso3": "PHL", "admin1": "Western Visayas", "month": current_month, "value": 105},
    ]

    rows: List[List[Any]] = []
    country_totals: Dict[Tuple[str, str], int] = defaultdict(int)

    for entry in samples:
        iso3 = entry["iso3"]
        admin1 = entry["admin1"]
        month_value: datetime = entry["month"]
        month_iso = month_value.date().isoformat()
        value = int(entry["value"])
        country_totals[(iso3, month_iso)] += value
        event_key = (iso3, admin1, month_iso, "offline-smoke")
        event_id = f"{iso3}-offline-{stable_digest(event_key)}"
        raw_event_id = f"offline::{iso3}::{admin1 or 'country'}::{month_iso}"
        payload = [
            "dtm",
            iso3,
            admin1,
            event_id,
            as_of_iso,
            month_iso,
            "new_displaced",
            value,
            "people",
            "dtm_stock_to_flow",
            DEFAULT_CAUSE,
            raw_event_id,
            json.dumps(
                {
                    "mode": "offline-smoke",
                    "iso3": iso3,
                    "admin1": admin1,
                    "month_start": month_iso,
                    "value": value,
                },
                ensure_ascii=False,
            ),
        ]
        rows.append(payload)

    for (iso3, month_iso), total in country_totals.items():
        if total <= 0:
            continue
        event_key = (iso3, "", month_iso, "offline-smoke-country")
        event_id = f"{iso3}-offline-{stable_digest(event_key)}"
        raw_event_id = f"offline::{iso3}::country::{month_iso}"
        payload = [
            "dtm",
            iso3,
            "",
            event_id,
            as_of_iso,
            month_iso,
            "new_displaced",
            int(total),
            "people",
            "dtm_stock_to_flow",
            DEFAULT_CAUSE,
            raw_event_id,
            json.dumps(
                {
                    "mode": "offline-smoke",
                    "iso3": iso3,
                    "aggregation": "country",
                    "month_start": month_iso,
                    "value": int(total),
                },
                ensure_ascii=False,
            ),
        ]
        rows.append(payload)

    rows.sort(key=lambda row: (row[1], row[2], row[5], row[3]))
    return rows


def _run_offline_smoke(
    *,
    no_date_filter: bool,
    strict_empty: bool,
    args: argparse.Namespace,
    record_connector: bool = True,
) -> Tuple[int, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    LOG.info("offline-smoke: skip network -> rows=0 offline=True")
    ensure_zero_row_outputs(offline=True)

    timings_ms: Dict[str, int] = {"write": 0}
    deps_payload = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "dtmapi": "skipped-offline",
        "pandas": _package_version("pandas"),
        "requests": _package_version("requests"),
    }

    http_payload: Dict[str, Any] = {key: 0 for key in HTTP_COUNT_KEYS}
    http_payload["retries"] = 0
    http_payload["last_status"] = None
    http_payload["rate_limit_remaining"] = None

    counts_payload = {"fetched": 0, "normalized": 0, "written": 0}
    reason = "offline_smoke"
    extras_payload: Dict[str, Any] = {
        "mode": "offline-smoke",
        "rows_total": 0,
        "status_raw": "ok",
        "exit_code": 0,
        "strict_empty": strict_empty,
        "no_date_filter": no_date_filter,
        "offline_smoke": True,
        "timings_ms": dict(timings_ms),
        "offline": True,
    }

    _write_meta(
        0,
        None,
        None,
        deps=deps_payload,
        effective_params={},
        http_counters=http_payload,
        timings_ms=timings_ms,
        diagnostics={"mode": "offline-smoke", "offline": True},
    )

    run_payload = {
        "window": {"start": None, "end": None},
        "countries": {},
        "http": dict(http_payload),
        "paging": {"pages": 0, "page_size": None, "total_received": 0},
        "rows": {
            "fetched": 0,
            "normalized": 0,
            "written": 0,
            "kept": 0,
            "dropped": 0,
        },
        "totals": {"rows_written": 0},
        "status": "ok",
        "reason": reason,
        "outputs": {"csv": str(OUT_PATH), "meta": str(META_PATH)},
        "extras": {
            "mode": "offline-smoke",
            "timings_ms": dict(timings_ms),
            "deps": deps_payload,
            "rows_total": 0,
            "offline": True,
        },
        "args": vars(args),
    }

    write_json(RUN_DETAILS_PATH, run_payload)
    _mirror_legacy_diagnostics()
    if record_connector:
        _append_connectors_report(
            mode="offline_smoke",
            status="ok",
            rows=0,
            reason=reason,
            extras=extras_payload,
            http=http_payload,
            counts=counts_payload,
        )

    return 0, reason, extras_payload, http_payload, counts_payload


def _finalize_skip_run(
    *,
    reason: str,
    strict_empty: bool,
    no_date_filter: bool,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    LOG.info("dtm: RESOLVER_SKIP_DTM detected; emitting header-only CSV and exiting")

    http_payload: Dict[str, Any] = {key: 0 for key in HTTP_COUNT_KEYS}
    http_payload["retries"] = 0
    http_payload["last_status"] = None
    http_payload["rate_limit_remaining"] = None

    counts_payload: Dict[str, Any] = {"fetched": 0, "normalized": 0, "written": 0}

    extras_payload: Dict[str, Any] = {
        "mode": "skip",
        "rows_total": 0,
        "status_raw": "skipped",
        "exit_code": 0,
        "strict_empty": strict_empty,
        "no_date_filter": no_date_filter,
        "offline_smoke": False,
    }

    deps_payload = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "dtmapi": "skipped",
        "pandas": _package_version("pandas"),
        "requests": _package_version("requests"),
    }

    _write_meta(
        0,
        None,
        None,
        deps=deps_payload,
        effective_params={},
        http_counters=http_payload,
        timings_ms={},
        diagnostics={"mode": "skip", "reason": reason},
    )

    run_payload = {
        "window": {"start": None, "end": None},
        "countries": {},
        "http": dict(http_payload),
        "paging": {"pages": 0, "page_size": None, "total_received": 0},
        "rows": {
            "fetched": 0,
            "normalized": 0,
            "written": 0,
            "kept": 0,
            "dropped": 0,
        },
        "totals": {"rows_written": 0},
        "status": "skipped",
        "reason": reason,
        "outputs": {"csv": str(OUT_PATH), "meta": str(META_PATH)},
        "extras": {
            "mode": "skip",
            "strict_empty": strict_empty,
            "no_date_filter": no_date_filter,
            "offline_smoke": False,
        },
        "args": vars(args),
    }

    write_json(RUN_DETAILS_PATH, run_payload)
    _mirror_legacy_diagnostics()
    return extras_payload, http_payload, counts_payload


def write_rows(rows: Sequence[Sequence[Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    ensure_manifest_for_csv(OUT_PATH, schema_version="dtm_displacement.v1", source_id="dtm")



def _iter_level_pages(
    client: DTMApiClient,
    level: str,
    *,
    country: Optional[str],
    operation: Optional[str],
    from_date: Optional[str],
    to_date: Optional[str],
    http_counts: MutableMapping[str, int],
):
    method_name = ADMIN_METHODS.get(level)
    fetcher = getattr(client, method_name) if method_name else None
    if fetcher is None:
        raise ValueError(f"Unsupported admin level: {level}")
    kwargs = {
        "country": country,
        "from_date": from_date,
        "to_date": to_date,
        "http_counts": http_counts,
    }
    if level == "admin2" and operation:
        kwargs["operation"] = operation
    frame = fetcher(**kwargs)
    safe_kwargs = {key: value for key, value in kwargs.items() if key != "http_counts"}
    LOG.debug("dtm: fetched %s with params=%s", level, safe_kwargs)
    if frame is not None and not frame.empty:
        LOG.debug(
            "dtm: first page stats level=%s rows=%s cols=%s",
            level,
            int(frame.shape[0]),
            list(frame.columns)[:6],
        )
        yield frame


def _fetch_level_pages_with_logging(
    client: DTMApiClient,
    level: str,
    *,
    country: Optional[str],
    operation: Optional[str],
    from_date: Optional[str],
    to_date: Optional[str],
    http_counts: MutableMapping[str, int],
) -> Tuple[List[pd.DataFrame], int]:
    context = {
        "level": level,
        "country": country,
        "operation": operation,
        "from": from_date,
        "to": to_date,
    }
    started = time.perf_counter()
    try:
        pages = list(
            _iter_level_pages(
                client,
                level,
                country=country,
                operation=operation,
                from_date=from_date,
                to_date=to_date,
                http_counts=http_counts,
            )
        )
    except Exception as exc:
        elapsed_ms = max(0, int((time.perf_counter() - started) * 1000))
        context.update({"status": "error", "error": str(exc), "elapsed_ms": elapsed_ms})
        _append_request_log(context)
        raise

    elapsed_ms = max(0, int((time.perf_counter() - started) * 1000))
    rows = sum(int(page.shape[0]) for page in pages if hasattr(page, "shape"))
    context.update({"status": "ok" if rows > 0 else "empty", "rows": rows, "elapsed_ms": elapsed_ms})
    _append_request_log(context)
    country_label = country or "all countries"
    op_suffix = f" / operation={operation}" if operation else ""
    LOG.info("Fetched %d rows for %s (%s%s)", rows, country_label, level, op_suffix)
    return pages, rows


def _resolve_admin_levels(cfg: Mapping[str, Any]) -> List[str]:
    api_cfg = cfg.get("api", {})
    configured = api_cfg.get("admin_levels") or cfg.get("admin_levels")
    if not configured:
        return ["admin1", "admin0"]
    levels = []
    for item in configured:
        text = str(item).strip().lower()
        if text in {"admin0", "admin1", "admin2"}:
            levels.append(text)
    return levels or ["admin1", "admin0"]


def _normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    items = []
    for item in value:
        text = str(item).strip()
        if text:
            items.append(text)
    return items


def _fetch_api_data(
    cfg: Mapping[str, Any],
    *,
    no_date_filter: bool,
    window_start: Optional[str],
    window_end: Optional[str],
    http_counts: MutableMapping[str, int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if "api" not in cfg:
        raise ValueError(
            "Config error: DTM is API-only; provide an 'api:' block in resolver/ingestion/config/dtm.yml",
        )

    http_counter = _ensure_http_counts(http_counts)
    summary: Dict[str, Any] = {
        "row_counts": _empty_row_counts(),
        "http_counts": {key: 0 for key in HTTP_COUNT_KEYS},
        "paging": {"pages": 0, "page_size": None, "total_received": 0},
        "rows": {"fetched": 0, "normalized": 0, "written": 0, "kept": 0, "dropped": 0},
        "countries": {"requested": [], "resolved": []},
    }

    run_started = time.perf_counter()
    metrics_summary = _init_metrics_summary()
    _ensure_diagnostics_scaffolding()
    summary_extras = summary.setdefault("extras", {})
    per_country_counts: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    summary_extras["per_country_counts"] = per_country_counts
    summary_extras["failures"] = failures

    try:
        PER_COUNTRY_METRICS_PATH.unlink(missing_ok=True)  # type: ignore[arg-type]
    except TypeError:  # pragma: no cover - Python < 3.8 compatibility
        if PER_COUNTRY_METRICS_PATH.exists():
            PER_COUNTRY_METRICS_PATH.unlink()
    except Exception:  # pragma: no cover - diagnostics only
        LOG.debug("Unable to reset metrics file", exc_info=True)

    try:
        client = DTMApiClient(cfg)
    except RuntimeError as exc:
        message = str(exc)
        _write_discovery_failure(
            "missing_key",
            message=message or "DTM_API_KEY missing",
            hint="Set DTM_API_KEY to the IOM DTM subscription key before running the connector.",
            extra={"sdk_version": _dtm_sdk_version()},
        )
        metrics_summary["duration_sec"] = round(max(0.0, time.perf_counter() - run_started), 3)
        _write_metrics_summary_file(metrics_summary)
        raise ValueError(message) from exc

    admin_levels = _resolve_admin_levels(cfg)
    api_cfg = cfg.get("api", {})
    requested_countries = _normalize_list(api_cfg.get("countries"))
    requested_operations = _normalize_list(api_cfg.get("operations"))

    summary["countries"]["requested"] = requested_countries
    if requested_countries:
        LOG.info(
            "Config requested countries=%s but discovery mode is enforced; ignoring list",
            requested_countries,
        )

    target = client.client if hasattr(client, "client") else client
    discovery_result = _perform_discovery(cfg, metrics=metrics_summary)
    resolved_countries = discovery_result.countries
    discovery_source = discovery_result.stage_used or "none"
    metrics_summary["stage_used"] = discovery_source

    if not resolved_countries:
        _write_discovery_failure(
            "empty_discovery",
            message="Discovery returned no valid selectors",
            hint="Verify DTM_API_KEY validity. Empty results may indicate an expired or invalid key.",
            extra={
                "sdk_version": _dtm_sdk_version(),
                "api_base": getattr(target, "base_url", getattr(target, "_base_url", "unknown")),
                "discovery_source": discovery_source,
            },
        )
        LOG.error("Discovery yielded no valid selectors; aborting connector")
        raise RuntimeError("DTM: 0 countries discovered, aborting")

    discovered_count = len(resolved_countries)
    LOG.info("Final discovery list contains %d selectors", discovered_count)
    summary["countries"]["resolved"] = resolved_countries

    country_mode = "ALL"

    operations = requested_operations if requested_operations else [None]

    LOG.info("Fetching data for %d countries across %s", discovered_count, admin_levels)

    discovery_info = {
        "total_countries": discovered_count,
        "sdk_version": _dtm_sdk_version(),
        "api_base": getattr(target, "base_url", getattr(target, "_base_url", "unknown")),
        "discovery_file": str(DISCOVERY_SNAPSHOT_PATH),
        "source": discovery_source,
        "stages": discovery_result.report.get("stages", []),
    }
    summary_extras["discovery"] = discovery_info
    diagnostics_payload = summary_extras.setdefault("diagnostics", {})
    diagnostics_payload["http_trace"] = str(DTM_HTTP_LOG_PATH)
    diagnostics_payload["raw_countries"] = str(DISCOVERY_RAW_JSON_PATH)
    diagnostics_payload["metrics"] = str(PER_COUNTRY_METRICS_PATH)
    diagnostics_payload["sample"] = str(SAMPLE_ROWS_PATH)
    diagnostics_payload["log"] = str(DTM_CLIENT_LOG_PATH)
    diagnostics_payload["discovery_fail"] = str(DISCOVERY_FAIL_PATH)
    diagnostics_payload["discovery_used_stage"] = discovery_source
    diagnostics_payload["no_data_combos"] = 0

    api_base_url = discovery_info["api_base"]
    LOG.info(
        "diagnostics: api_base_url=%s resolved_country_count=%d sample=%s",
        api_base_url,
        discovered_count,
        resolved_countries[:5],
    )

    from_date = window_start if not no_date_filter else None
    to_date = window_end if not no_date_filter else None

    try:
        DTM_HTTP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        DTM_HTTP_LOG_PATH.write_text("", encoding="utf-8")
    except Exception:  # pragma: no cover - diagnostics only
        LOG.debug("Unable to reset DTM requests log", exc_info=True)

    request_payload = {
        "admin_levels": admin_levels,
        "countries": None if resolved_countries == [None] else resolved_countries,
        "operations": None if operations == [None] else operations,
        "window_start": from_date,
        "window_end": to_date,
        "country_mode": country_mode,
    }
    write_json(API_REQUEST_PATH, {**request_payload, "api_key": "***"})

    summary["api"] = {
        "endpoint": str(api_cfg.get("endpoint", "dtmapi")),
        "requested_window": {"start": from_date, "end": to_date},
        "query_params": request_payload,
    }
    LOG.info(
        "DTM endpoint=%s window=%s→%s",
        summary["api"]["endpoint"],
        from_date or "-",
        to_date or "-",
    )
    LOG.info(
        "DTM request params: admin_levels=%s countries=%s operations=%s",
        admin_levels,
        request_payload.get("countries") or "all",
        request_payload.get("operations") or "all",
    )
    LOG.info(
        "diagnostics: resolved_admin_levels=%s date_window=%s..%s",
        admin_levels,
        from_date or "-",
        to_date or "-",
    )
    request_params_clean = {
        "admin_levels": list(admin_levels),
        "countries": request_payload.get("countries"),
        "operations": request_payload.get("operations"),
        "window": {"start": from_date, "end": to_date},
    }
    LOG.info("diagnostics: request_params_clean=%s", request_params_clean)
    diagnostics_payload.update(
        {
            "api_base_url": api_base_url,
            "resolved_country_count": discovered_count,
            "resolved_admin_levels": list(admin_levels),
            "date_window": {"start": from_date, "end": to_date},
            "request_params_clean": request_params_clean,
        }
    )
    effective_params = {
        "resource": summary["api"]["endpoint"],
        "from": from_date,
        "to": to_date,
        "admin_levels": admin_levels,
        "operations": None if operations == [None] else operations,
        "country_mode": country_mode,
        "discovered_countries_count": discovered_count,
        "countries": [] if resolved_countries == [None] else resolved_countries,
        "countries_requested": requested_countries,
        "no_date_filter": bool(no_date_filter),
        "per_page": None,
        "max_pages": None,
    }
    summary.setdefault("extras", {})["effective_params"] = effective_params

    base_idp_aliases = ["TotalIDPs", "IDPTotal", "numPresentIdpInd"]
    LOG.info(
        "effective: dates %s → %s; admin_levels=%s; country_mode=%s; discovered_countries=%d; idp_aliases=%s",
        from_date or "-",
        to_date or "-",
        admin_levels,
        country_mode,
        discovered_count,
        base_idp_aliases,
    )

    field_mapping = cfg.get("field_mapping", {})
    field_aliases = cfg.get("field_aliases", {})
    country_column = field_mapping.get("country_column", "CountryName")
    admin1_column = field_mapping.get("admin1_column", "Admin1Name")
    admin2_column = field_mapping.get("admin2_column", "Admin2Name")
    date_column = field_mapping.get("date_column", "ReportingDate")
    idp_candidates = list(field_aliases.get("idp_count", base_idp_aliases))
    for alias in (field_mapping.get("idp_column"), *base_idp_aliases):
        if alias and alias not in idp_candidates:
            idp_candidates.append(alias)

    aliases = cfg.get("country_aliases") or {}
    measure = str(cfg.get("output", {}).get("measure", "stock")).strip().lower()
    cause_map = {
        str(k).strip().lower(): str(v)
        for k, v in (cfg.get("cause_map") or {}).items()
    }

    all_records: List[Dict[str, Any]] = []
    head_written_levels: Set[str] = set()
    country_rollup: Dict[Optional[str], Dict[str, Any]] = {}
    unsupported_countries: Set[Optional[str]] = set()

    effective_params_ref = summary.get("extras", {}).get("effective_params")
    if isinstance(effective_params_ref, MutableMapping):
        effective_params_ref["idp_aliases"] = list(idp_candidates)

    no_data_combos = 0

    for level in admin_levels:
        for country_name in resolved_countries:
            status_entry = country_rollup.setdefault(
                country_name,
                {"rows": 0, "errors": False, "skip_no_match": False},
            )
            if country_name in unsupported_countries or status_entry.get("skip_no_match"):
                continue
            for operation in (operations if level == "admin2" else [None]):
                combo_count = 0
                country_label = country_name or "all countries"
                op_suffix = f", operation={operation}" if operation else ""
                LOG.info(
                    "planning fetch: level=%s country=%s from=%s to=%s%s",
                    level,
                    country_label,
                    from_date or "-",
                    to_date or "-",
                    op_suffix,
                )
                failed = False
                skip_before = http_counter.get("skip_no_match", 0)
                with diagnostics_timing(f"{country_label}|{level}") as timing_result:
                    try:
                        pages, _ = _fetch_level_pages_with_logging(
                            client,
                            level,
                            country=country_name,
                            operation=operation,
                            from_date=from_date,
                            to_date=to_date,
                            http_counts=http_counter,
                        )
                    except DTMHttpError as exc:
                        status = getattr(exc, "status_code", None)
                        if status in {401, 403} or isinstance(exc, DTMUnauthorizedError):
                            _write_discovery_failure(
                                "invalid_key",
                                message=str(exc),
                                hint="Verify DTM_API_KEY validity. Invalid or expired keys cause 401/403 responses.",
                                extra={
                                    "sdk_version": _dtm_sdk_version(),
                                    "api_base": getattr(
                                        target,
                                        "base_url",
                                        getattr(target, "_base_url", "unknown"),
                                    ),
                                },
                            )
                        raise
                    except Exception as exc:
                        LOG.error(
                            "Fetch failed for %s (%s%s): %s",
                            country_label,
                            level,
                            op_suffix,
                            exc,
                            exc_info=True,
                        )
                        failures.append(
                            {
                                "country": country_label,
                                "level": level,
                                "operation": operation,
                                "error": type(exc).__name__,
                                "message": str(exc),
                            }
                        )
                        status_entry["errors"] = True
                        failed = True
                        pages = []
                skip_after = http_counter.get("skip_no_match", 0)
                if skip_after > skip_before:
                    if not status_entry.get("skip_no_match"):
                        metrics_summary["countries_skipped_no_match"] += 1
                    status_entry["skip_no_match"] = True
                    unsupported_countries.add(country_name)
                    LOG.info(
                        "Skipping unsupported country after no-match response: %s",
                        country_label,
                    )
                    break
                if failed:
                    continue

                for page in pages:
                    summary["paging"]["pages"] += 1
                    summary["paging"]["total_received"] += int(page.shape[0])
                    if page.shape[0]:
                        size = int(page.shape[0])
                        current = summary["paging"]["page_size"] or 0
                        summary["paging"]["page_size"] = max(current, size)
                    if page.empty:
                        continue
                    LOG.debug(
                        "all columns (n=%d): %s",
                        len(page.columns),
                        list(page.columns),
                    )
                    idp_column = _resolve_idp_value_column(page, idp_candidates)
                    LOG.info("resolved value column: idp_count_col=%r", idp_column)
                    if not idp_column:
                        LOG.warning(
                            "no value column matched; aliases=%s; rows=%d",
                            idp_candidates,
                            len(page),
                        )
                        status_entry["errors"] = True
                        failures.append(
                            {
                                "country": country_label,
                                "level": level,
                                "operation": operation,
                                "error": "MissingValueColumn",
                                "message": "aliases=%s columns=%s"
                                % (idp_candidates, list(page.columns)),
                            }
                        )
                        continue
                    if level not in head_written_levels:
                        try:
                            _dump_head(page, level, country_label)
                        except Exception:  # pragma: no cover - diagnostics only
                            LOG.debug(
                                "failed to write raw head for level=%s country=%s",
                                level,
                                country_label,
                                exc_info=True,
                            )
                        else:
                            head_written_levels.add(level)
                    try:
                        _write_level_sample(page, level)
                    except Exception:  # pragma: no cover - diagnostics only
                        LOG.debug("failed to write sample rows", exc_info=True)
                    page = page.rename(columns={idp_column: "idp_count"})
                    if level == "admin0":
                        try:
                            _record_admin0_sample(page, operation=operation)
                        except Exception:  # pragma: no cover - diagnostics only
                            LOG.debug("Unable to update admin0 sample", exc_info=True)

                    per_admin: Dict[Tuple[str, str], Dict[datetime, float]] = defaultdict(dict)
                    per_admin_asof: Dict[Tuple[str, str], Dict[datetime, str]] = defaultdict(dict)
                    causes: Dict[Tuple[str, str], str] = {}

                    for _, row in page.iterrows():
                        iso = to_iso3(row.get(country_column), aliases)
                        if not iso:
                            continue
                        bucket = month_start(row.get(date_column))
                        if not bucket:
                            continue
                        admin1 = ""
                        if admin1_column in page.columns:
                            admin1 = str(row.get(admin1_column) or "").strip()
                        if level == "admin2" and admin2_column in page.columns:
                            admin2 = str(row.get(admin2_column) or "").strip()
                            if admin2:
                                admin1 = f"{admin1}/{admin2}" if admin1 else admin2
                        value = _parse_float(row.get("idp_count"))
                        if value is None or value <= 0:
                            continue
                        per_admin[(iso, admin1)][bucket] = float(value)
                        as_of_value = _parse_iso_date_or_none(row.get(date_column))
                        if not as_of_value:
                            as_of_value = datetime.now(timezone.utc).date().isoformat()
                        existing_asof = per_admin_asof[(iso, admin1)].get(bucket)
                        if not existing_asof or _is_candidate_newer(existing_asof, as_of_value):
                            per_admin_asof[(iso, admin1)][bucket] = as_of_value
                        cause_key = str(row.get("Cause", "")).strip().lower()
                        causes[(iso, admin1)] = cause_map.get(cause_key, DEFAULT_CAUSE)

                    for key, series in per_admin.items():
                        iso, admin1 = key
                        if measure == "stock":
                            flows = flow_from_stock(series)
                        else:
                            flows = {month_start(date): float(val) for date, val in series.items() if month_start(date)}
                        for bucket, value in flows.items():
                            if value is None or value <= 0:
                                continue
                            record_as_of = (
                                per_admin_asof.get(key, {}).get(bucket)
                                or datetime.now(timezone.utc).date().isoformat()
                            )
                            all_records.append(
                                {
                                    "iso3": iso,
                                    "admin1": admin1,
                                    "month_start": bucket,
                                    "value": float(value),
                                    "series_type": SERIES_INCIDENT,
                                    "cause": causes.get(key, DEFAULT_CAUSE),
                                    "measure": measure,
                                    "source_id": f"dtm_api::{level}",
                                    "as_of": record_as_of,
                                }
                            )
                            combo_count += 1
                            summary["row_counts"][level] += 1
                            summary["row_counts"]["total"] += 1
                            summary["rows"]["fetched"] += 1

                if failed:
                    continue

                range_label = f"{from_date or '-'}->{to_date or '-'}"
                per_country_counts.append(
                    {
                        "country": country_label,
                        "level": level,
                        "operation": operation,
                        "rows": combo_count,
                        "window": range_label,
                    }
                )
                if combo_count == 0:
                    no_data_combos += 1
                elapsed_ms = timing_result.elapsed_ms or 0
                _append_metrics(
                    {
                        "country": country_label,
                        "level": level,
                        "operation": operation,
                        "rows": combo_count,
                        "elapsed_ms": elapsed_ms,
                        "window": {"start": from_date, "end": to_date},
                    }
                )
                LOG.info(
                    "country=%s level=%s rows=%s total_so_far=%s elapsed_ms=%s",
                    country_label if not operation else f"{country_label} (operation={operation})",
                    level,
                    combo_count,
                    summary["rows"]["fetched"],
                    elapsed_ms,
                )
                status_entry["rows"] += combo_count
            if status_entry.get("skip_no_match"):
                continue

    def _update_country_metrics() -> None:
        attempted = len(country_rollup)
        if not attempted:
            attempted = len(resolved_countries)
        metrics_summary["countries_attempted"] = attempted
        metrics_summary["countries_skipped_no_match"] = sum(
            1 for entry in country_rollup.values() if entry.get("skip_no_match")
        )
        metrics_summary["countries_failed_other"] = sum(
            1
            for entry in country_rollup.values()
            if entry.get("errors")
            and not entry.get("skip_no_match")
            and entry.get("rows", 0) == 0
        )
        metrics_summary["countries_ok"] = sum(
            1
            for entry in country_rollup.values()
            if not entry.get("skip_no_match")
            and (entry.get("rows", 0) > 0 or not entry.get("errors"))
        )

    _update_country_metrics()

    skip_count = int(http_counter.get("skip_no_match", 0))
    fallback_skip = int(getattr(client, "_http_counts", {}).get("skip_no_match", 0))
    if fallback_skip > skip_count:
        http_counter["skip_no_match"] = fallback_skip
        summary.setdefault("http_counts", {})["skip_no_match"] = fallback_skip
        if metrics_summary.get("countries_skipped_no_match", 0) < fallback_skip:
            metrics_summary["countries_skipped_no_match"] = fallback_skip

    diagnostics_payload["no_data_combos"] = no_data_combos
    summary.setdefault("http_counts", {})["skip_no_match"] = int(http_counter.get("skip_no_match", 0))

    sample_files = [str(path) for path in sorted(DTM_DIAGNOSTICS_DIR.glob("sample_*.csv"))]
    if sample_files:
        diagnostics_payload["samples"] = sample_files

    total_rows = int(summary.get("row_counts", {}).get("total", 0))
    if total_rows == 0:
        zero_reason = "api_empty_response"
        if metrics_summary.get("countries_skipped_no_match") or http_counter.get("skip_no_match"):
            zero_reason = "no_country_match"
        elif not resolved_countries:
            zero_reason = "empty_country_list"
        elif failures:
            zero_reason = "invalid_indicator"
        elif no_data_combos:
            zero_reason = "filter_excluded_all"

        diagnostics_payload["zero_rows_reason"] = zero_reason
        summary_extras["zero_rows_reason"] = zero_reason
        summary.setdefault("rows", {}).setdefault("fetched", 0)
        summary["rows"].setdefault("normalized", 0)
        summary["rows"]["kept"] = 0
        summary["rows"]["written"] = 0
        summary.setdefault("row_counts", _empty_row_counts())
        summary_extras.setdefault("rows_written", 0)
        summary_extras.setdefault("rows_total", 0)
        summary_extras.setdefault("status_raw", "ok-empty")
        summary["reason"] = summary.get("reason") or "header-only; kept=0"

        window_label = f"{from_date or '-'}..{to_date or '-'}"
        _update_country_metrics()
        metrics_summary["rows_fetched"] = int(total_rows)
        metrics_summary["duration_sec"] = round(max(0.0, time.perf_counter() - run_started), 3)
        _write_metrics_summary_file(metrics_summary)

        skip_only = (
            metrics_summary.get("countries_attempted", 0) > 0
            and metrics_summary.get("countries_skipped_no_match", 0)
            >= metrics_summary.get("countries_attempted", 0)
            and metrics_summary.get("countries_failed_other", 0) == 0
        )
        if skip_only:
            LOG.info(
                "dtm: all %d countries skipped due to unsupported selectors",
                metrics_summary.get("countries_skipped_no_match", 0),
            )
        else:
            LOG.warning(
                "dtm: zero rows produced for window %s zero_rows_reason=%s",
                window_label,
                zero_reason,
            )
        return all_records, summary

    effective_params = summary.get("extras", {}).get("effective_params")
    if isinstance(effective_params, MutableMapping):
        effective_params["per_page"] = summary.get("paging", {}).get("page_size")
        effective_params["max_pages"] = summary.get("paging", {}).get("pages")

    _update_country_metrics()
    metrics_summary["rows_fetched"] = int(summary.get("rows", {}).get("fetched", 0))
    metrics_summary["duration_sec"] = round(max(0.0, time.perf_counter() - run_started), 3)
    _write_metrics_summary_file(metrics_summary)

    return all_records, summary


def _finalize_records(
    records: Sequence[Mapping[str, Any]],
    *,
    summary: MutableMapping[str, Any],
    write_sample: bool,
) -> List[List[Any]]:
    if not records:
        if write_sample:
            try:
                write_json(API_SAMPLE_PATH, [])
            except Exception:
                LOG.debug("dtm: unable to persist empty API sample", exc_info=True)
        return []

    run_date = datetime.now(timezone.utc).date().isoformat()
    method = (
        "dtm_stock_to_flow"
        if any(str(rec.get("measure", "")).lower() == "stock" for rec in records)
        else "dtm_flow"
    )

    dedup: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    country_totals: Dict[Tuple[str, str, str], float] = defaultdict(float)
    country_asof: Dict[Tuple[str, str, str], str] = {}

    for rec in records:
        iso3 = rec["iso3"]
        admin1 = rec.get("admin1") or ""
        month = rec.get("month_start", rec.get("month"))
        if hasattr(month, "date"):
            month = month.date()
        month_iso = month.isoformat() if hasattr(month, "isoformat") else str(month)
        value = float(rec.get("value", 0.0))
        record_as_of = rec.get("as_of") or run_date
        key = (iso3, admin1, month_iso, rec["source_id"])
        event_id = f"{iso3}-displacement-{month.strftime('%Y%m') if hasattr(month, 'strftime') else month_iso.replace('-', '')}-{stable_digest(key)}"
        payload = {
            "source": "dtm",
            "country_iso3": iso3,
            "admin1": admin1,
            "event_id": event_id,
            "as_of": record_as_of,
            "month_start": month_iso,
            "value_type": "new_displaced",
            "value": int(round(value)),
            "unit": "people",
            "method": method,
            "confidence": rec.get("cause", DEFAULT_CAUSE),
            "raw_event_id": f"{rec['source_id']}::{admin1 or 'country'}::{month.strftime('%Y%m') if hasattr(month, 'strftime') else month_iso.replace('-', '')}",
            "raw_fields_json": json.dumps(
                {
                    "source_id": rec["source_id"],
                    "admin1": admin1,
                    "month": month_iso,
                    "value": value,
                    "cause": rec.get("cause", DEFAULT_CAUSE),
                },
                ensure_ascii=False,
            ),
        }
        existing = dedup.get(key)
        if existing and not _is_candidate_newer(existing["as_of"], record_as_of):
            continue
        dedup[key] = payload
        summary.setdefault("rows", {}).setdefault("normalized", 0)
        summary["rows"]["normalized"] += 1
        summary["rows"]["kept"] = summary["rows"].get("kept", 0) + 1
        country_key = (iso3, month_iso, rec["source_id"])
        country_totals[country_key] += value
        current_asof = country_asof.get(country_key)
        if not current_asof or _is_candidate_newer(current_asof, record_as_of):
            country_asof[country_key] = record_as_of

    rows = list(dedup.values())
    for (iso3, month_iso, source_id), total in country_totals.items():
        if total <= 0:
            continue
        month_dt = datetime.strptime(month_iso, "%Y-%m-%d").date()
        event_id = f"{iso3}-displacement-{month_dt.strftime('%Y%m')}-{stable_digest([iso3, month_iso, source_id])}"
        rows.append(
            {
                "source": "dtm",
                "country_iso3": iso3,
                "admin1": "",
                "event_id": event_id,
                "as_of": country_asof.get((iso3, month_iso, source_id), run_date),
                "month_start": month_iso,
                "value_type": "new_displaced",
                "value": int(round(total)),
                "unit": "people",
                "method": method,
                "confidence": DEFAULT_CAUSE,
                "raw_event_id": f"{source_id}::country::{month_dt.strftime('%Y%m')}",
                "raw_fields_json": json.dumps(
                    {"source_id": source_id, "aggregation": "country", "total_value": total},
                    ensure_ascii=False,
                ),
            }
        )
        summary.setdefault("rows", {}).setdefault("normalized", 0)
        summary["rows"]["normalized"] += 1
        summary["rows"]["kept"] = summary["rows"].get("kept", 0) + 1

    formatted = [
        [
            rec["source"],
            rec["country_iso3"],
            rec.get("admin1", ""),
            rec["event_id"],
            rec["as_of"],
            rec["month_start"],
            rec["value_type"],
            rec["value"],
            rec["unit"],
            rec["method"],
            rec["confidence"],
            rec["raw_event_id"],
            rec["raw_fields_json"],
        ]
        for rec in rows
    ]

    formatted.sort(key=lambda row: (row[1], row[2], row[5], row[3]))

    if write_sample:
        try:
            write_json(API_SAMPLE_PATH, formatted[:100])
            write_json(API_RESPONSE_SAMPLE_PATH, formatted[:100])
            diagnostics_write_sample_csv(pd.DataFrame(formatted, columns=COLUMNS), SAMPLE_ROWS_PATH)
        except Exception:
            LOG.debug("dtm: unable to persist API sample", exc_info=True)
        else:
            summary.setdefault("extras", {})["api_sample_path"] = str(API_SAMPLE_PATH)

    summary.setdefault("rows", {}).setdefault("written", 0)
    summary["rows"]["written"] = len(formatted)
    return formatted


def build_rows(
    cfg: Mapping[str, Any],
    *,
    no_date_filter: bool,
    window_start: Optional[str],
    window_end: Optional[str],
    http_counts: Optional[MutableMapping[str, int]] = None,
    write_sample: bool = False,
) -> Tuple[List[List[Any]], Dict[str, Any]]:
    http_stats = _ensure_http_counts(http_counts)
    fetch_started = time.perf_counter()
    records, summary = _fetch_api_data(
        cfg,
        no_date_filter=no_date_filter,
        window_start=window_start,
        window_end=window_end,
        http_counts=http_stats,
    )
    fetch_elapsed = time.perf_counter() - fetch_started
    normalize_started = time.perf_counter()
    rows = _finalize_records(records, summary=summary, write_sample=write_sample)
    normalize_elapsed = time.perf_counter() - normalize_started
    timings = summary.setdefault("timings_ms", {})
    timings["fetch_total"] = max(0, int(fetch_elapsed * 1000))
    timings["normalize"] = max(0, int(normalize_elapsed * 1000))
    summary.setdefault("rows", {}).setdefault("written", 0)
    summary["rows"]["written"] = len(rows)
    summary.setdefault("http_counts", {})
    for key in HTTP_COUNT_KEYS:
        summary["http_counts"][key] = int(http_stats.get(key, 0))
    summary["http_counts"]["retries"] = int(http_stats.get("retries", 0))
    summary["http_counts"]["skip_no_match"] = int(http_stats.get("skip_no_match", 0))
    summary["http_counts"]["last_status"] = http_stats.get("last_status")
    return rows, summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-empty",
        action="store_true",
        help="Exit with code 2 when the connector writes zero rows.",
    )
    parser.add_argument(
        "--no-date-filter",
        action="store_true",
        help="Disable the ingestion window filter when pulling from the API.",
    )
    parser.add_argument(
        "--offline-smoke",
        action="store_true",
        help="Skip API calls and exit successfully for offline smoke tests.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and force file logging for troubleshooting.",
    )
    return parser.parse_args(list(argv or []))


def _write_meta(
    rows: int,
    window_start: Optional[str],
    window_end: Optional[str],
    *,
    deps: Mapping[str, Any],
    effective_params: Mapping[str, Any],
    http_counters: Mapping[str, Any],
    timings_ms: Mapping[str, Any],
    per_country_counts: Sequence[Mapping[str, Any]] = (),
    failures: Sequence[Mapping[str, Any]] = (),
    discovery: Optional[Mapping[str, Any]] = None,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {"row_count": rows}
    if window_start:
        payload["backfill_start"] = window_start
    if window_end:
        payload["backfill_end"] = window_end
    payload["deps"] = dict(deps)
    payload["effective_params"] = dict(effective_params)
    payload["http_counters"] = dict(http_counters)
    payload["timings_ms"] = {key: int(value) for key, value in timings_ms.items()}
    payload["per_country_counts"] = list(per_country_counts)
    payload["failures"] = list(failures)
    if discovery is not None:
        payload["discovery"] = dict(discovery)
    if diagnostics is not None:
        payload["diagnostics"] = dict(diagnostics)
    write_json(META_PATH, payload)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    log_level_name = str(os.getenv("LOG_LEVEL") or "INFO").upper()
    if args.debug:
        log_level_name = "DEBUG"
    logging.basicConfig(
        level=getattr(logging, log_level_name, logging.INFO),
        format="[%(levelname)s] %(message)s",
    )
    LOG.setLevel(getattr(logging, log_level_name, logging.INFO))
    _setup_file_logging()

    try:
        DTM_HTTP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        DTM_HTTP_LOG_PATH.touch(exist_ok=True)
    except Exception:  # pragma: no cover - diagnostics helper
        LOG.debug("dtm: unable to initialise diagnostics HTTP trace", exc_info=True)

    global OUT_DIR, OUTPUT_PATH, META_PATH, HTTP_TRACE_PATH, OFFLINE
    OUT_DIR = Path(OUT_PATH).parent
    OUTPUT_PATH = OUT_PATH
    META_PATH = OUT_PATH.with_suffix(OUT_PATH.suffix + ".meta.json")
    HTTP_TRACE_PATH = OUT_DIR / "dtm_http.ndjson"
    LOG.debug("dtm: canonical headers=%s", CANONICAL_HEADERS)
    LOG.debug("dtm: outputs -> csv=%s meta=%s http_trace=%s", OUT_PATH, META_PATH, HTTP_TRACE_PATH)

    strict_empty = (
        args.strict_empty
        or _env_bool("DTM_STRICT_EMPTY", False)
        or _env_bool("RESOLVER_STRICT_EMPTY", False)
    )
    no_date_filter = args.no_date_filter or _env_bool("DTM_NO_DATE_FILTER", False)
    offline_smoke = args.offline_smoke or _env_bool("DTM_OFFLINE_SMOKE", False)
    skip_requested = bool(os.getenv("RESOLVER_SKIP_DTM"))
    offline_mode = skip_requested or bool(offline_smoke)
    previous_offline = OFFLINE
    OFFLINE = offline_mode
    if skip_requested:
        LOG.info("Skip mode: no real network calls; writing header & trace placeholder")
    if OFFLINE:
        LOG.debug("dtm: offline gating enabled (skip=%s offline_smoke=%s)", skip_requested, offline_smoke)

    if offline_smoke:
        LOG.info("offline-smoke: header-only output (no API key required)")
        _ensure_out_dir_exists(OUT_DIR)
        (
            rows_written,
            reason,
            extras_payload,
            http_payload,
            counts_payload,
        ) = _run_offline_smoke(
            no_date_filter=no_date_filter,
            strict_empty=strict_empty,
            args=args,
            record_connector=False,
        )
        _append_connectors_report(
            mode="offline_smoke",
            status="ok",
            rows=rows_written,
            reason=reason,
            extras=extras_payload,
            http=http_payload,
            counts=counts_payload,
        )
        OFFLINE = previous_offline
        return 0

    preflight_started = time.perf_counter()
    dep_info, have_dtmapi = _preflight_dependencies()
    timings_ms: Dict[str, int] = {"preflight": max(0, int((time.perf_counter() - preflight_started) * 1000))}
    _log_dependency_snapshot(dep_info)
    deps_payload = {
        "python": dep_info.get("python", sys.version.split()[0]),
        "executable": dep_info.get("executable", sys.executable),
        "dtmapi": _package_version("dtmapi"),
        "pandas": _package_version("pandas"),
        "requests": _package_version("requests"),
    }
    LOG.info(
        "env: python=%s exe=%s",
        dep_info.get("python", sys.version.split()[0]),
        dep_info.get("executable", sys.executable),
    )
    LOG.info(
        "deps: dtmapi=%s pandas=%s requests=%s",
        deps_payload["dtmapi"],
        deps_payload["pandas"],
        deps_payload["requests"],
    )

    api_key_configured = bool((os.getenv("DTM_API_KEY") or os.getenv("DTM_SUBSCRIPTION_KEY") or "").strip())
    base_http: Dict[str, Any] = {key: 0 for key in HTTP_COUNT_KEYS}
    base_http["rate_limit_remaining"] = None
    base_http["retries"] = 0
    base_http["last_status"] = None
    base_extras: Dict[str, Any] = {
        "api_key_configured": api_key_configured,
        "deps": deps_payload,
        "strict_empty": strict_empty,
        "no_date_filter": no_date_filter,
        "offline_smoke": offline_smoke,
        "offline": OFFLINE,
        "timings_ms": dict(timings_ms),
    }

    if not have_dtmapi:
        reason = "dependency-missing: dtmapi (install with: pip install 'dtmapi>=0.1.5')"
        counts_payload = {"fetched": 0, "normalized": 0, "written": 0}
        extras_payload = dict(base_extras)
        extras_payload.update({"rows_total": 0, "exit_code": 1, "status_raw": "error"})
        extras_payload["timings_ms"] = dict(timings_ms)
        _write_connector_report(
            status="error",
            reason=reason,
            extras=extras_payload,
            http=dict(base_http),
            counts=counts_payload,
        )
        _append_summary_stub_if_needed(reason)
        ensure_zero_row_outputs(offline=OFFLINE)
        run_payload = {
            "window": {"start": None, "end": None},
            "countries": {},
            "http": dict(base_http),
            "paging": {"pages": 0, "page_size": None, "total_received": 0},
            "rows": {"fetched": 0, "normalized": 0, "written": 0, "kept": 0, "dropped": 0},
            "totals": {"rows_written": 0},
            "status": "error",
            "reason": reason,
            "outputs": {"csv": str(OUT_PATH), "meta": str(META_PATH)},
            "extras": {
                "deps": deps_payload,
                "timings_ms": dict(timings_ms),
                "strict_empty": strict_empty,
                "no_date_filter": no_date_filter,
            },
            "args": vars(args),
        }
        write_json(RUN_DETAILS_PATH, run_payload)
        _mirror_legacy_diagnostics()
        LOG.error(reason)
        OFFLINE = previous_offline
        return 1

    raw_auth_key = (os.getenv("DTM_API_KEY") or os.getenv("DTM_SUBSCRIPTION_KEY") or "").strip()
    try:
        _auth_probe(raw_auth_key, offline=OFFLINE)
    except DTMUnauthorizedError as exc:
        reason = f"auth: {exc}"
        counts_payload = {"fetched": 0, "normalized": 0, "written": 0}
        extras_payload = dict(base_extras)
        extras_payload.update(
            {
                "rows_total": 0,
                "exit_code": 1,
                "status_raw": "error",
                "auth_error": "unauthorized",
            }
        )
        extras_payload["timings_ms"] = dict(timings_ms)
        _write_connector_report(
            status="error",
            reason=reason,
            extras=extras_payload,
            http=dict(base_http),
            counts=counts_payload,
        )
        ensure_zero_row_outputs(offline=OFFLINE)
        run_payload = {
            "window": {"start": None, "end": None},
            "countries": {},
            "http": dict(base_http),
            "paging": {"pages": 0, "page_size": None, "total_received": 0},
            "rows": {"fetched": 0, "normalized": 0, "written": 0, "kept": 0, "dropped": 0},
            "totals": {"rows_written": 0},
            "status": "error",
            "reason": reason,
            "outputs": {"csv": str(OUT_PATH), "meta": str(META_PATH)},
            "extras": {
                "deps": deps_payload,
                "timings_ms": dict(timings_ms),
                "strict_empty": strict_empty,
                "no_date_filter": no_date_filter,
                "offline": OFFLINE,
                "auth_error": "unauthorized",
            },
            "args": vars(args),
        }
        write_json(RUN_DETAILS_PATH, run_payload)
        _mirror_legacy_diagnostics()
        LOG.error("dtm: auth probe failed: %s", exc)
        OFFLINE = previous_offline
        return 1
    except Exception as exc:
        reason = f"auth: {exc}" if str(exc) else "auth: error"
        counts_payload = {"fetched": 0, "normalized": 0, "written": 0}
        extras_payload = dict(base_extras)
        extras_payload.update(
            {
                "rows_total": 0,
                "exit_code": 1,
                "status_raw": "error",
                "auth_error": "exception",
            }
        )
        extras_payload["timings_ms"] = dict(timings_ms)
        _write_connector_report(
            status="error",
            reason=reason,
            extras=extras_payload,
            http=dict(base_http),
            counts=counts_payload,
        )
        ensure_zero_row_outputs(offline=OFFLINE)
        run_payload = {
            "window": {"start": None, "end": None},
            "countries": {},
            "http": dict(base_http),
            "paging": {"pages": 0, "page_size": None, "total_received": 0},
            "rows": {"fetched": 0, "normalized": 0, "written": 0, "kept": 0, "dropped": 0},
            "totals": {"rows_written": 0},
            "status": "error",
            "reason": reason,
            "outputs": {"csv": str(OUT_PATH), "meta": str(META_PATH)},
            "extras": {
                "deps": deps_payload,
                "timings_ms": dict(timings_ms),
                "strict_empty": strict_empty,
                "no_date_filter": no_date_filter,
                "offline": OFFLINE,
                "auth_error": "exception",
            },
            "args": vars(args),
        }
        write_json(RUN_DETAILS_PATH, run_payload)
        _mirror_legacy_diagnostics()
        LOG.error("dtm: auth probe error: %s", exc)
        OFFLINE = previous_offline
        return 1

    diagnostics_ctx = diagnostics_start_run("dtm_client", "real")
    http_stats: MutableMapping[str, int] = _ensure_http_counts({})
    extras: Dict[str, Any] = dict(base_extras)
    extras["mode"] = "skip" if skip_requested else extras.get("mode", "real")
    extras["skip_requested"] = skip_requested

    status = "ok"
    reason: Optional[str] = None
    exit_code = 0

    window_start_dt, window_end_dt = resolve_ingestion_window()
    window_start_iso = window_start_dt.isoformat() if window_start_dt else None
    window_end_iso = window_end_dt.isoformat() if window_end_dt else None

    rows_written = 0
    summary: Dict[str, Any] = {}

    try:
        cfg = load_config()
        LOG.info(
            "config_loaded_from=%s",
            getattr(cfg, "_source_path", "<unknown>"),
        )
        if not cfg.get("enabled", True):
            status = "skipped"
            reason = "disabled via config"
            ensure_zero_row_outputs(offline=OFFLINE)
        elif "api" not in cfg:
            raise ValueError(
                "Config error: DTM is API-only; provide 'api:' in resolver/ingestion/config/dtm.yml",
            )
        else:
            rows, summary = build_rows(
                cfg,
                no_date_filter=no_date_filter,
                window_start=window_start_iso,
                window_end=window_end_iso,
                http_counts=http_stats,
                write_sample=True,
            )
            rows_written = len(rows)
            write_started = time.perf_counter()
            if rows:
                write_rows(rows)
            else:
                ensure_zero_row_outputs(offline=OFFLINE)
            timings_ms["write"] = max(0, int((time.perf_counter() - write_started) * 1000))
    except ValueError as exc:
        LOG.error("dtm: %s", exc)
        status = "error"
        reason = str(exc)
        exit_code = 1
        ensure_zero_row_outputs(offline=OFFLINE)
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.exception("dtm: unexpected failure")
        status = "error"
        message = str(exc) or f"exception: {type(exc).__name__}"
        reason = message
        extras["exception"] = {"type": type(exc).__name__, "message": message}
        extras["traceback"] = traceback.format_exc()
        exit_code = 1
        ensure_zero_row_outputs(offline=OFFLINE)

    if skip_requested:
        extras["skip_reason"] = "RESOLVER_SKIP_DTM"
        if "zero_rows_reason" not in extras:
            extras["zero_rows_reason"] = summary.get("extras", {}).get("zero_rows_reason") or "skip_requested"
        zero_reason = extras.get("zero_rows_reason")
        if zero_reason:
            summary.setdefault("extras", {}).setdefault("zero_rows_reason", zero_reason)

    summary_timings = summary.get("timings_ms") if isinstance(summary, Mapping) else {}
    if isinstance(summary_timings, Mapping):
        for key, value in summary_timings.items():
            timings_ms[key] = int(value)
    for key in ("preflight", "fetch_total", "normalize", "write"):
        timings_ms.setdefault(key, 0)

    extras.update(
        {
            "rows_total": rows_written,
            "window_start": window_start_iso,
            "window_end": window_end_iso,
            "strict_empty": strict_empty,
            "no_date_filter": no_date_filter,
            "exit_code": exit_code,
            "status_raw": status,
            "timings_ms": dict(timings_ms),
        }
    )
    extras["api_attempted"] = bool(http_stats.get("last_status"))

    http_payload = {key: int(http_stats.get(key, 0)) for key in HTTP_COUNT_KEYS}
    http_payload["retries"] = int(http_stats.get("retries", 0))
    http_payload["last_status"] = http_stats.get("last_status")
    http_payload["rate_limit_remaining"] = http_stats.get("rate_limit_remaining")

    rows_summary = summary.get("rows", {}) if isinstance(summary, Mapping) else {}
    summary_extras = dict(summary.get("extras", {})) if isinstance(summary, Mapping) else {}
    summary_extras["timings_ms"] = dict(timings_ms)
    summary_extras.setdefault("deps", deps_payload)
    totals = {
        "rows_fetched": rows_summary.get("fetched", 0),
        "rows_normalized": rows_summary.get("normalized", rows_written),
        "rows_written": rows_written,
        "kept": rows_summary.get("kept", rows_written),
        "dropped": rows_summary.get("dropped", 0),
        "parse_errors": rows_summary.get("parse_errors", 0),
    }

    kept_count = int(totals.get("kept") or 0)
    zero_rows_reason = (
        extras.get("zero_rows_reason")
        or summary_extras.get("zero_rows_reason")
        or None
    )
    zero_rows = status != "error" and kept_count == 0
    if status != "error":
        if zero_rows:
            ensure_zero_row_outputs(offline=OFFLINE)
            totals["rows_written"] = 0
            totals["kept"] = 0
            extras["rows_total"] = 0
            extras["rows_written"] = 0
            summary_extras["rows_written"] = 0
            if zero_rows_reason:
                extras.setdefault("zero_rows_reason", zero_rows_reason)
                diagnostics_payload = extras.get("diagnostics")
                if isinstance(diagnostics_payload, Mapping):
                    diagnostics_map = dict(diagnostics_payload)
                else:
                    diagnostics_map = {}
                diagnostics_map["zero_rows_reason"] = zero_rows_reason
                extras["diagnostics"] = diagnostics_map
                summary_extras.setdefault("zero_rows_reason", zero_rows_reason)
            reason = summary.get("reason") or reason or "header-only; kept=0"
            status = "ok-empty"
        else:
            extras["rows_total"] = kept_count
            extras["rows_written"] = rows_written
            summary_extras["rows_written"] = rows_written
            if status == "ok" and not reason:
                reason = f"wrote {kept_count} rows"
    else:
        extras["rows_total"] = kept_count
        extras["rows_written"] = rows_written
        summary_extras["rows_written"] = rows_written

    if strict_empty and zero_rows and exit_code == 0:
        exit_code = 2
        LOG.error("dtm: strict-empty enabled; exiting with code 2 for zero rows")

    extras["status_raw"] = status
    extras["exit_code"] = exit_code
    summary_extras.setdefault("status_raw", status)
    summary_extras["exit_code"] = exit_code

    effective_params: Mapping[str, Any]
    raw_effective = summary_extras.get("effective_params")
    if isinstance(raw_effective, Mapping):
        effective_params = dict(raw_effective)
    else:
        effective_params = {}

    per_country_counts_payload = list(summary_extras.get("per_country_counts", []))
    failures_payload = list(summary_extras.get("failures", []))
    summary_extras["per_country_counts"] = per_country_counts_payload
    summary_extras["failures"] = failures_payload

    _write_meta(
        rows_written,
        window_start_iso,
        window_end_iso,
        deps=deps_payload,
        effective_params=effective_params,
        http_counters=http_payload,
        timings_ms=timings_ms,
        per_country_counts=per_country_counts_payload,
        failures=failures_payload,
        discovery=summary_extras.get("discovery"),
        diagnostics=summary_extras.get("diagnostics"),
    )

    run_payload = {
        "window": {"start": window_start_iso, "end": window_end_iso},
        "countries": summary.get("countries", {}),
        "http": http_payload,
        "paging": summary.get("paging", {"pages": 0, "page_size": None, "total_received": 0}),
        "rows": {
            "fetched": rows_summary.get("fetched", 0),
            "normalized": rows_summary.get("normalized", rows_written),
            "written": rows_written,
            "kept": rows_summary.get("kept", rows_written),
            "dropped": rows_summary.get("dropped", 0),
        },
        "totals": totals,
        "status": status,
        "reason": reason,
        "outputs": {"csv": str(OUT_PATH), "meta": str(META_PATH)},
        "extras": summary_extras,
        "args": vars(args),
    }
    if extras.get("zero_rows_reason") and "zero_rows_reason" not in run_payload["extras"]:
        run_payload["extras"]["zero_rows_reason"] = extras["zero_rows_reason"]
    if rows_written == 0 and "api_sample_path" not in run_payload["extras"] and API_SAMPLE_PATH.exists():
        run_payload["extras"]["api_sample_path"] = str(API_SAMPLE_PATH)

    write_json(RUN_DETAILS_PATH, run_payload)
    _mirror_legacy_diagnostics()
    extras["run_details_path"] = str(RUN_DETAILS_PATH)
    extras["meta_path"] = str(META_PATH)
    if API_SAMPLE_PATH.exists():
        extras["api_sample_path"] = str(API_SAMPLE_PATH)
    extras.setdefault("effective_params", effective_params)
    extras.setdefault("per_country_counts", per_country_counts_payload)
    extras.setdefault("failures", failures_payload)
    diagnostics_result = diagnostics_finalize_run(
        diagnostics_ctx,
        status=status,
        reason=reason or "",
        http=http_payload,
        counts={"written": rows_written},
        extras=extras,
    )
    diagnostics_append_jsonl(CONNECTORS_REPORT, diagnostics_result)
    _mirror_legacy_diagnostics()

    OFFLINE = previous_offline
    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))

