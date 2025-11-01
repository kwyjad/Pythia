#!/usr/bin/env python3
"""DTM connector that fetches displacement data exclusively through the official API."""

from __future__ import annotations

import argparse
import csv
import importlib
import hashlib
import json
import logging
import os
import pathlib
import random
import re
import socket
import ssl
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import pandas as pd
import yaml
import requests
import urllib3

try:  # pragma: no cover - defensive import guard
    from urllib3.exceptions import ConnectTimeoutError as Urllib3ConnectTimeoutError
    from urllib3.exceptions import ReadTimeoutError as Urllib3ReadTimeoutError
except Exception:  # pragma: no cover - defensive import guard
    Urllib3ConnectTimeoutError = None
    Urllib3ReadTimeoutError = None
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from resolver.ingestion._manifest import ensure_manifest_for_csv
from resolver.ingestion._shared.feature_flags import getenv_bool
from resolver.ingestion._shared.run_io import count_csv_rows, write_json
from resolver.ingestion.diagnostics_emitter import (
    append_jsonl as diagnostics_append_jsonl,
    finalize_run as diagnostics_finalize_run,
    start_run as diagnostics_start_run,
)
from resolver.ingestion.dtm.normalize import normalize_admin0
from resolver.ingestion.dtm_auth import build_discovery_header_variants, get_dtm_api_key
from resolver.ingestion.utils import ensure_headers, flow_from_stock, month_start, stable_digest
from resolver.ingestion.utils.iso_normalize import resolve_iso3 as resolve_iso3_fields, to_iso3
from resolver.ingestion.utils.io import resolve_ingestion_window, resolve_output_path
from resolver.scripts.ingestion._dtm_debug_utils import (
    dump_json as diagnostics_dump_json,
    timing as diagnostics_timing,
    write_sample_csv as diagnostics_write_sample_csv,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = _REPO_ROOT
RESOLVER_ROOT = REPO_ROOT / "resolver"
LEGACY_CONFIG_PATH = (RESOLVER_ROOT / "ingestion" / "config" / "dtm.yml").resolve()
REPO_CONFIG_PATH = (RESOLVER_ROOT / "config" / "dtm.yml").resolve()
SERIES_SEMANTICS_PATH = (RESOLVER_ROOT / "config" / "series_semantics.yml").resolve()


def _resolve_effective_window() -> Tuple[Optional[str], Optional[str], bool]:
    """Return the ISO window bounds, applying CI overrides when provided."""

    start_dt, end_dt = resolve_ingestion_window()
    start_iso = start_dt.isoformat() if start_dt else None
    end_iso = end_dt.isoformat() if end_dt else None
    start_override = (os.getenv("DTM_WINDOW_START") or "").strip()
    end_override = (os.getenv("DTM_WINDOW_END") or "").strip()
    override_applied = bool(start_override and end_override)
    if override_applied:
        start_iso = start_override
        end_iso = end_override
    return start_iso, end_iso, override_applied


def _resolve_config_path() -> Path:
    repo_root = Path(REPO_ROOT)
    resolver_root = repo_root / "resolver"
    env_value = os.getenv("DTM_CONFIG_PATH", "").strip()
    if env_value:
        candidate = pathlib.Path(env_value).expanduser()
        if not candidate.is_absolute():
            candidate = (repo_root / candidate).resolve()
        else:
            candidate = candidate.resolve()
        return candidate

    repo_cfg = (resolver_root / "config" / "dtm.yml").resolve()
    if repo_cfg.exists():
        return repo_cfg

    ingestion_cfg = (resolver_root / "ingestion" / "config" / "dtm.yml").resolve()
    return ingestion_cfg


CONFIG_PATH = _resolve_config_path()

# Diagnostics (repo-root)
DIAGNOSTICS_ROOT = REPO_ROOT / "diagnostics" / "ingestion"
DIAGNOSTICS_DIR = DIAGNOSTICS_ROOT  # Back-compat alias expected by older tests
DTM_DIAGNOSTICS_DIR = DIAGNOSTICS_ROOT / "dtm"

# Per-subdir directories (monkeypatchable in tests)
DTM_RAW_DIR = DTM_DIAGNOSTICS_DIR / "raw"
DTM_METRICS_DIR = DTM_DIAGNOSTICS_DIR / "metrics"
DTM_SAMPLES_DIR = DTM_DIAGNOSTICS_DIR / "samples"
DTM_LOG_DIR = DTM_DIAGNOSTICS_DIR / "logs"
DIAGNOSTICS_RAW_DIR = DIAGNOSTICS_ROOT / "raw"
DIAGNOSTICS_METRICS_DIR = DIAGNOSTICS_ROOT / "metrics"
DIAGNOSTICS_SAMPLES_DIR = DIAGNOSTICS_ROOT / "samples"
DIAGNOSTICS_LOG_DIR = DIAGNOSTICS_ROOT / "logs"

# Standard filenames under those dirs
CONNECTORS_REPORT = DIAGNOSTICS_ROOT / "connectors_report.jsonl"
LEGACY_CONNECTORS_REPORT = REPO_ROOT / "diagnostics" / "connectors_report.jsonl"
_DEFAULT_RUN_DETAILS_PATH = DTM_DIAGNOSTICS_DIR / "dtm_run.json"
RUN_DETAILS_PATH = _DEFAULT_RUN_DETAILS_PATH
DTM_HTTP_LOG_PATH = DTM_DIAGNOSTICS_DIR / "dtm_http.ndjson"
HTTP_TRACE_PATH = DTM_HTTP_LOG_PATH
DISCOVERY_SNAPSHOT_PATH = DTM_DIAGNOSTICS_DIR / "discovery_countries.csv"
DISCOVERY_FAIL_PATH = DTM_DIAGNOSTICS_DIR / "discovery_fail.json"
DISCOVERY_RAW_JSON_PATH = DTM_RAW_DIR / "dtm_countries.json"
PER_COUNTRY_METRICS_PATH = DTM_METRICS_DIR / "dtm_per_country.jsonl"
SAMPLE_ROWS_PATH = DTM_DIAGNOSTICS_DIR / "dtm_sample.csv"
DTM_CLIENT_LOG_PATH = DTM_LOG_DIR / "dtm_client.log"
API_REQUEST_PATH = DTM_DIAGNOSTICS_DIR / "dtm_api_request.json"
API_SAMPLE_PATH = DTM_DIAGNOSTICS_DIR / "dtm_api_sample.json"
API_RESPONSE_SAMPLE_PATH = DTM_DIAGNOSTICS_DIR / "dtm_api_response_sample.json"
RESCUE_PROBE_PATH = DTM_DIAGNOSTICS_DIR / "rescue_probe.json"
METRICS_SUMMARY_PATH = DTM_METRICS_DIR / "metrics.json"
SAMPLE_ADMIN0_PATH = DTM_SAMPLES_DIR / "admin0_head.csv"

# Staging outputs (repo-root)
OUT_DIR = RESOLVER_ROOT / "staging"
OUT_PATH = OUT_DIR / "dtm_displacement.csv"
OUTPUT_PATH = OUT_PATH
DEFAULT_OUTPUT = OUT_PATH
META_PATH = OUT_PATH.with_suffix(OUT_PATH.suffix + ".meta.json")
RESCUE_HEADERS = ["CountryISO3", "ReportingDate", "idp_count"]

INGESTION_SUMMARY_PATH = DTM_DIAGNOSTICS_DIR / "summary.json"

STATIC_MINIMAL_FALLBACK: List[Tuple[str, str]] = [
    ("South Sudan", "SSD"),
    ("Nigeria", "NGA"),
    ("Somalia", "SOM"),
    ("Ethiopia", "ETH"),
    ("Sudan", "SDN"),
    ("DR Congo", "COD"),
    ("Yemen", "YEM"),
]
STATIC_DATA_DIR = pathlib.Path(__file__).resolve().parent / "static"
STATIC_ISO3_PATH = STATIC_DATA_DIR / "iso3_master.csv"

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

_NORMALIZE_DROP_KEYS = [
    "no_iso3",
    "no_value_col",
    "bad_iso",
    "unknown_country",
    "missing_value",
    "non_positive_value",
    "non_integer_value",
    "missing_as_of",
    "future_as_of",
    "invalid_semantics",
    "date_parse_failed",
    "date_out_of_window",
    "no_country_match",
    "other",
]


def _zero_drop_map() -> Dict[str, int]:
    return {key: 0 for key in _NORMALIZE_DROP_KEYS}


def _resolve_run_details_override() -> Optional[Path]:
    override = os.environ.get("RUN_DETAILS_PATH") or os.environ.get(
        "RESOLVER_RUN_DETAILS_PATH"
    )
    if not override:
        return None
    candidate = Path(override).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def _refresh_run_details_path() -> Path:
    override = _resolve_run_details_override()
    if override is not None:
        path = override
    else:
        path = globals().get("RUN_DETAILS_PATH", _DEFAULT_RUN_DETAILS_PATH)
    resolved = Path(path)
    globals()["RUN_DETAILS_PATH"] = resolved
    return resolved


def _ensure_normalize_block(extras: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    if not isinstance(extras, MutableMapping):
        extras = dict(extras) if isinstance(extras, Mapping) else {}
    block_source = extras.get("normalize")
    block: Dict[str, Any]
    if isinstance(block_source, Mapping):
        block = dict(block_source)
    else:
        block = {}

    def _coerce_int(value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default

    rows_in = block.get("rows_in", block.get("rows_fetched", 0))
    rows_out = block.get("rows_out", block.get("rows_written", 0))
    block["rows_in"] = _coerce_int(rows_in, 0)
    block["rows_out"] = _coerce_int(rows_out, 0)
    block.setdefault(
        "rows_dropped",
        max(block["rows_in"] - block["rows_out"], 0),
    )
    block["rows_dropped"] = _coerce_int(block.get("rows_dropped", 0), block["rows_in"] - block["rows_out"])

    drop_raw = block.get("drop_reasons")
    drop_counts = _zero_drop_map()
    if isinstance(drop_raw, Mapping):
        for key, value in drop_raw.items():
            if key in drop_counts:
                drop_counts[key] = _coerce_int(value, 0)
            else:
                try:
                    drop_counts[str(key)] = _coerce_int(value, 0)
                except Exception:
                    drop_counts[str(key)] = value
    block["drop_reasons"] = drop_counts

    extras["normalize"] = block
    return block


def _prepare_run_details_payload(
    payload: Mapping[str, Any], *, run_details_path: Path
) -> Dict[str, Any]:
    base = dict(payload)
    extras_source = base.get("extras")
    if isinstance(extras_source, MutableMapping):
        extras: Dict[str, Any] = dict(extras_source)
    elif isinstance(extras_source, Mapping):
        extras = dict(extras_source)
    else:
        extras = {}
    base["extras"] = extras

    artifacts_source = extras.get("artifacts")
    if isinstance(artifacts_source, MutableMapping):
        artifacts: Dict[str, Any] = dict(artifacts_source)
    elif isinstance(artifacts_source, Mapping):
        artifacts = dict(artifacts_source)
    else:
        artifacts = {}
    artifacts.setdefault("run_json", str(run_details_path))
    extras["artifacts"] = artifacts

    _ensure_normalize_block(extras)
    return base


def _write_run_details(payload: Mapping[str, Any]) -> None:
    override = _resolve_run_details_override()
    if override is not None:
        globals()["RUN_DETAILS_PATH"] = override
        target = override
    else:
        target = Path(RUN_DETAILS_PATH)
    prepared = _prepare_run_details_payload(payload, run_details_path=target)
    write_json(target, prepared)

_DATE_CANDIDATES: list[str] = [
    "ReportingDate",
    "reportingDate",
    "reporting_date",
    "ReportDate",
    "Date",
    "date",
    "asOfDate",
    "PublicationDate",
    "pub_date",
    "FromReportingDate",
]
DATE_CANDIDATES: Tuple[str, ...] = tuple(_DATE_CANDIDATES)

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

for directory in (
    DIAGNOSTICS_ROOT,
    DTM_DIAGNOSTICS_DIR,
    DTM_RAW_DIR,
    DTM_METRICS_DIR,
    DTM_SAMPLES_DIR,
    DTM_LOG_DIR,
    DIAGNOSTICS_RAW_DIR,
    DIAGNOSTICS_METRICS_DIR,
    DIAGNOSTICS_SAMPLES_DIR,
    DIAGNOSTICS_LOG_DIR,
    OUT_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)


def _pick_reporting_date_column(
    df: pd.DataFrame, preferred: Sequence[str] | None = None
) -> Optional[str]:
    """Return the header that most likely represents the reporting date column."""

    if df is None or not hasattr(df, "columns"):
        return None

    try:
        columns_iterable = list(df.columns)
    except Exception:  # pragma: no cover - defensive guard
        columns_iterable = []

    def _normalize(token: Any) -> str:
        text = str(token).strip().lower()
        if not text:
            return ""
        return text.replace("_", "")

    normalized_lookup: Dict[str, str] = {}
    for column in columns_iterable:
        text = str(column).strip()
        if not text:
            continue
        key = _normalize(text)
        if key and key not in normalized_lookup:
            normalized_lookup[key] = column

    candidates: List[str] = []
    if preferred:
        for item in preferred:
            if item is None:
                continue
            candidate = str(item).strip()
            if candidate:
                candidates.append(candidate)
    candidates.extend(DATE_CANDIDATES)

    for candidate in candidates:
        key = _normalize(candidate)
        if key and key in normalized_lookup:
            return normalized_lookup[key]

    for column in columns_iterable:
        key = _normalize(column)
        if "report" in key and "date" in key and "start" not in key and "end" not in key:
            return column

    return None


def _coerce_int(value: Any) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _normalise_date_for_stats(value: Any) -> Optional[date]:
    if value in (None, "", "NaT"):
        return None
    try:
        parsed = pd.to_datetime(value)
    except Exception:
        return None
    if getattr(parsed, "tzinfo", None):
        try:
            parsed = parsed.tz_convert(None)
        except AttributeError:
            try:
                parsed = parsed.tz_localize(None)
            except Exception:
                pass
    if hasattr(parsed, "to_pydatetime"):
        parsed = parsed.to_pydatetime()
    if isinstance(parsed, datetime):
        return parsed.date()
    if isinstance(parsed, date):
        return parsed
    return None


def _merge_date_filter_stats(
    target: MutableMapping[str, Any], diag: Mapping[str, Any]
) -> MutableMapping[str, Any]:
    if not isinstance(target, MutableMapping):
        target = {}  # type: ignore[assignment]
    window_start = diag.get("window_start")
    window_end = diag.get("window_end")
    if window_start and not target.get("window_start"):
        target["window_start"] = window_start
    if window_end and not target.get("window_end"):
        target["window_end"] = window_end
    target["inclusive"] = bool(diag.get("inclusive", target.get("inclusive", True)))
    if not target.get("date_column_used") and diag.get("date_column_used"):
        target["date_column_used"] = diag.get("date_column_used")
    for key in ("parsed_ok", "parsed_total", "inside", "outside", "parse_failed"):
        target[key] = _coerce_int(target.get(key)) + _coerce_int(diag.get(key))
    if diag.get("skipped"):
        target["skipped"] = True
    else:
        target.setdefault("skipped", False)
    diag_min = _normalise_date_for_stats(diag.get("min_date"))
    diag_max = _normalise_date_for_stats(diag.get("max_date"))
    existing_min = _normalise_date_for_stats(target.get("min_date"))
    existing_max = _normalise_date_for_stats(target.get("max_date"))
    if diag_min:
        if not existing_min or diag_min < existing_min:
            target["min_date"] = diag_min.isoformat()
    elif "min_date" not in target:
        target["min_date"] = None
    if diag_max:
        if not existing_max or diag_max > existing_max:
            target["max_date"] = diag_max.isoformat()
    elif "max_date" not in target:
        target["max_date"] = None
    sample_path = diag.get("sample_path")
    if sample_path and not target.get("sample_path"):
        target["sample_path"] = sample_path
    drop_counts = diag.get("drop_counts")
    if isinstance(drop_counts, Mapping):
        aggregate = target.setdefault("drop_counts", {})
        if isinstance(aggregate, MutableMapping):
            for reason, count in drop_counts.items():
                aggregate[reason] = _coerce_int(aggregate.get(reason)) + _coerce_int(count)
    return target


def _first_parsable_date_column(
    df: pd.DataFrame, candidates: Iterable[str]
) -> Tuple[Optional[str], Optional[pd.Series], Dict[str, Any]]:
    tried: list[str] = []
    for col in candidates:
        if col not in getattr(df, "columns", []):
            continue
        tried.append(col)
        try:
            series = pd.to_datetime(df[col], errors="coerce", utc=True)
        except Exception:
            continue
        if series.notna().any():
            output = series
            if hasattr(series, "dt") and getattr(series.dt, "tz", None) is not None:
                try:
                    output = series.dt.tz_convert("UTC")
                except Exception:
                    output = series
            bad_tokens: list[str] = []
            try:
                raw = df[col].astype("string", copy=False)
                parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
                invalid_mask = parsed.isna() & raw.notna()
                if invalid_mask.any():
                    for token in raw[invalid_mask].unique().tolist():
                        token_str = str(token)
                        if token_str and token_str not in bad_tokens:
                            bad_tokens.append(token_str)
                        if len(bad_tokens) >= 5:
                            break
            except Exception:
                bad_tokens = []
            normalized = output.dt.date.astype("string") if hasattr(output, "dt") else output
            return col, normalized, {
                "attempted": tried,
                "used": col,
                "bad_samples": bad_tokens,
            }
    bad: list[str] = []
    for col in tried:
        try:
            raw = df[col].astype("string", copy=False)
            parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        except Exception:
            continue
        invalid = raw[parsed.isna()].dropna().unique().tolist()
        for token in invalid:
            token_str = str(token)
            if token_str and token_str not in bad:
                bad.append(token_str)
            if len(bad) >= 5:
                break
        if len(bad) >= 5:
            break
    return None, None, {"attempted": tried, "used": None, "bad_samples": bad}


def _apply_date_window(
    df: pd.DataFrame,
    start_iso: Optional[str],
    end_iso: Optional[str],
    *,
    disable: bool,
    diag_out_dir: Path,
    extras: MutableMapping[str, Any],
    preferred_columns: Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    diag: Dict[str, Any] = {
        "window_start": start_iso,
        "window_end": end_iso,
        "inclusive": True,
        "parsed_total": int(len(df)),
        "parsed_ok": 0,
        "inside": 0,
        "outside": 0,
        "parse_failed": 0,
        "min_date": None,
        "max_date": None,
        "skipped": bool(disable),
        "sample_path": None,
        "drop_counts": {},
    }
    total_rows = diag["parsed_total"]
    used_col = _pick_reporting_date_column(df, preferred=preferred_columns)
    diag["date_column_used"] = used_col
    if total_rows == 0:
        diag["parsed_ok"] = 0
        existing = extras.get("date_filter")
        if not isinstance(existing, MutableMapping):
            existing = {}
            extras["date_filter"] = existing
        merged = _merge_date_filter_stats(existing, diag)
        extras["date_filter"] = merged
        LOG.info(
            "date_filter: column=%s parsed=%s/%s inside=%s outside=%s min=%s max=%s skipped=%s",
            merged.get("date_column_used"),
            merged.get("parsed_ok"),
            merged.get("parsed_total"),
            merged.get("inside"),
            merged.get("outside"),
            merged.get("min_date"),
            merged.get("max_date"),
            merged.get("skipped"),
        )
        return df, diag
    if disable or not used_col or not start_iso or not end_iso:
        diag["parsed_ok"] = 0 if total_rows == 0 else total_rows
        diag["inside"] = total_rows
        existing = extras.get("date_filter")
        if not isinstance(existing, MutableMapping):
            existing = {}
            extras["date_filter"] = existing
        merged = _merge_date_filter_stats(existing, diag)
        extras["date_filter"] = merged
        LOG.info(
            "date_filter: column=%s parsed=%s/%s inside=%s outside=%s min=%s max=%s skipped=%s",
            merged.get("date_column_used"),
            merged.get("parsed_ok"),
            merged.get("parsed_total"),
            merged.get("inside"),
            merged.get("outside"),
            merged.get("min_date"),
            merged.get("max_date"),
            merged.get("skipped"),
        )
        return df, diag
    series = pd.to_datetime(df[used_col], errors="coerce", utc=False)
    valid_mask = series.notna()
    diag["parsed_ok"] = int(valid_mask.sum())
    try:
        start_date = pd.to_datetime(start_iso).date() if start_iso else None
        end_date = pd.to_datetime(end_iso).date() if end_iso else None
    except Exception:
        start_date = None
        end_date = None
    if not start_date or not end_date:
        diag["inside"] = diag["parsed_ok"]
        diag["parse_failed"] = total_rows - diag["parsed_ok"]
        diag["outside"] = total_rows - diag["inside"]
        existing = extras.get("date_filter")
        if not isinstance(existing, MutableMapping):
            existing = {}
            extras["date_filter"] = existing
        merged = _merge_date_filter_stats(existing, diag)
        extras["date_filter"] = merged
        LOG.info(
            "date_filter: column=%s parsed=%s/%s inside=%s outside=%s min=%s max=%s skipped=%s",
            merged.get("date_column_used"),
            merged.get("parsed_ok"),
            merged.get("parsed_total"),
            merged.get("inside"),
            merged.get("outside"),
            merged.get("min_date"),
            merged.get("max_date"),
            merged.get("skipped"),
        )
        return df, diag
    date_series = series.dt.date
    range_mask = (date_series >= start_date) & (date_series <= end_date)
    mask = valid_mask & range_mask
    parse_failed_mask = ~valid_mask
    outside_mask = valid_mask & ~range_mask
    inside_count = int(mask.sum())
    parse_failed_count = int(parse_failed_mask.sum())
    outside_total = int(outside_mask.sum() + parse_failed_count)
    diag.update(
        {
            "inside": inside_count,
            "outside": outside_total,
            "parse_failed": parse_failed_count,
            "skipped": bool(diag.get("skipped")),
        }
    )
    valid_dates = [
        value
        for value in date_series[valid_mask].tolist()
        if isinstance(value, date)
    ]
    if valid_dates:
        diag["min_date"] = min(valid_dates).isoformat()
        diag["max_date"] = max(valid_dates).isoformat()
    drop_counts: Dict[str, int] = {}
    if outside_mask.any():
        drop_counts["date_out_of_window"] = int(outside_mask.sum())
    if parse_failed_count:
        drop_counts["date_parse_failed"] = parse_failed_count
    diag["drop_counts"] = drop_counts
    kept = df[mask].copy()
    dropped = df[~mask].copy()
    if not dropped.empty:
        drop_reason = pd.Series("date_out_of_window", index=df.index)
        drop_reason.loc[parse_failed_mask.index[parse_failed_mask]] = "date_parse_failed"
        drop_dates_map = {
            idx: value.isoformat() if isinstance(value, date) else None
            for idx, value in zip(date_series.index, date_series.tolist())
        }
        dropped.loc[:, "_drop_reason"] = drop_reason.loc[dropped.index].tolist()
        dropped.loc[:, "_drop_date"] = [drop_dates_map.get(idx) for idx in dropped.index]
        diag_out_dir.mkdir(parents=True, exist_ok=True)
        sample_path = diag_out_dir / "normalize_drops.csv"
        existing_sample: Optional[pd.DataFrame]
        if sample_path.exists():
            try:
                existing_sample = pd.read_csv(sample_path)
            except Exception:
                existing_sample = None
        else:
            existing_sample = None
        if existing_sample is not None and not existing_sample.empty:
            combined = pd.concat([existing_sample, dropped.head(200)], ignore_index=True)
            combined = combined.head(200)
        else:
            combined = dropped.head(200)
        try:
            combined.to_csv(sample_path, index=False)
        except Exception:
            LOG.debug("dtm: unable to write normalize_drops.csv", exc_info=True)
        else:
            diag["sample_path"] = str(sample_path)
    existing = extras.get("date_filter")
    if not isinstance(existing, MutableMapping):
        existing = {}
        extras["date_filter"] = existing
    merged = _merge_date_filter_stats(existing, diag)
    extras["date_filter"] = merged
    LOG.info(
        "date_filter: column=%s parsed=%s/%s inside=%s outside=%s min=%s max=%s skipped=%s",
        merged.get("date_column_used"),
        merged.get("parsed_ok"),
        merged.get("parsed_total"),
        merged.get("inside"),
        merged.get("outside"),
        merged.get("min_date"),
        merged.get("max_date"),
        merged.get("skipped"),
    )
    return kept, diag

_LEGACY_DIAGNOSTICS_DIR = _REPO_ROOT / "legacy_diagnostics"


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


def _get_dtm_key_and_header(*, warn_on_missing: bool = True) -> Tuple[str, Optional[str]]:
    """Resolve the preferred DTM API key and header name from the environment."""

    header_name = (os.getenv("DTM_API_HEADER_NAME") or "Ocp-Apim-Subscription-Key").strip() or (
        "Ocp-Apim-Subscription-Key"
    )

    def _clean(value: Optional[str]) -> str:
        return value.strip() if value else ""

    key = _clean(os.getenv("DTM_API_KEY"))
    if key:
        LOG.debug("dtm: API key sourced from DTM_API_KEY; header=%s", header_name)
        return header_name, key

    subscription = _clean(os.getenv("DTM_SUBSCRIPTION_KEY"))
    if subscription:
        LOG.debug("dtm: API key sourced from DTM_SUBSCRIPTION_KEY; header=%s", header_name)
        return header_name, subscription

    primary = _clean(os.getenv("DTM_API_PRIMARY_KEY"))
    secondary = _clean(os.getenv("DTM_API_SECONDARY_KEY"))
    legacy = primary or secondary
    if legacy:
        LOG.debug("dtm: API key sourced from legacy primary/secondary; header=%s", header_name)
        return header_name, legacy

    helper_key = _clean(get_dtm_api_key())
    if helper_key:
        LOG.debug("dtm: API key sourced from dtm_auth helper; header=%s", header_name)
        return header_name, helper_key

    if warn_on_missing:
        LOG.warning("dtm: No DTM API key found in environment.")
    return header_name, None

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


def _call_admin_level(client: Any, level: str, **kwargs: Any) -> Tuple[Any, str]:
    filtered_kwargs = {key: value for key, value in kwargs.items() if value is not None}
    candidates: List[str] = []
    primary = ADMIN_METHODS.get(level)
    if primary:
        candidates.append(primary)
    if level == "admin0":
        candidates.extend(["get_admin0", "fetch_admin0"])
    elif level == "admin1":
        candidates.extend(["get_admin1", "fetch_admin1"])
    elif level == "admin2":
        candidates.extend(["get_admin2", "fetch_admin2"])
    candidates.extend(
        name
        for name in (
            f"get_idp_{level}",
            f"get_{level}",
            f"fetch_{level}",
            level,
        )
        if name not in candidates
    )

    for name in candidates:
        potential_targets = [client]
        inner = getattr(client, "client", None)
        if inner is not None and inner is not client:
            potential_targets.append(inner)
        for target_obj in potential_targets:
            fn = getattr(target_obj, name, None)
            if not callable(fn):
                continue
            payloads = [filtered_kwargs]
            if filtered_kwargs is not kwargs:
                payloads.append(kwargs)
            for payload in payloads:
                try:
                    result = fn(**payload)
                except TypeError:
                    continue
                setattr(target_obj, "_dtm_last_method", name)
                if target_obj is not client:
                    try:
                        setattr(client, "_dtm_last_method", name)
                    except Exception:
                        pass
                return result, name
    raise AttributeError(f"No callable for admin level '{level}' found on client")


def _is_no_country_match_error(err: BaseException) -> bool:
    """Return ``True`` when *err* carries the discovery soft-skip signature."""

    try:
        message = str(err).lower()
        return "no country found matching your query" in message
    except Exception:
        return False


def _country_kwargs(country: Optional[str]) -> Dict[str, str]:
    """Return selector kwargs honoring the fast-test contract."""

    if not country:
        return {}
    text = str(country).strip()
    if not text:
        return {}
    if len(text) == 3 and text.isalpha() and text.upper() == text:
        return {"CountryISO3": text}
    return {"CountryName": text}


def _country_filter(country: Optional[str]) -> Dict[str, str]:
    """Return DTM SDK selector kwargs using Admin0Pcode or CountryName."""

    if not country:
        return {}
    text = str(country).strip()
    if not text:
        return {}
    if len(text) == 3 and text.isalpha() and text.upper() == text:
        return {"Admin0Pcode": text}
    return {"CountryName": text}


def _sdk_call_with_arg_shim(
    client: Any,
    method_name: str,
    *,
    country_filter: Optional[Mapping[str, str]] = None,
    **kwargs: Any,
) -> Any:
    """Invoke *method_name* on *client*, remapping ISO3 selector keys if needed."""

    method = getattr(client, method_name)
    try:
        return method(**kwargs)
    except TypeError as exc:
        message = str(exc).lower()
        if "CountryISO3" in kwargs and "unexpected keyword" in message:
            remapped = dict(kwargs)
            iso_value = remapped.pop("CountryISO3", None)
            shim_kwargs: Dict[str, str] = {}
            if isinstance(country_filter, Mapping):
                shim_kwargs.update({str(k): str(v) for k, v in country_filter.items() if str(k)})
            if not shim_kwargs and iso_value:
                shim_kwargs = _country_filter(str(iso_value))
            if shim_kwargs:
                remapped.update(shim_kwargs)
            else:
                remapped["Admin0Pcode"] = str(iso_value) if iso_value else ""
            return method(**remapped)
        raise


def _countries_mode_from_stage(stage: Optional[str]) -> str:
    if not stage:
        return "discovered"
    lowered = str(stage).strip().lower()
    if "explicit" in lowered:
        return "explicit_config"
    if "static" in lowered or "iso3" in lowered or "minimal" in lowered:
        return "static_iso3_minimal"
    return "discovered"


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


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _sanitize_mapping(value)
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, (str, int, float)) or value is None:
        return value
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)


def _sanitize_mapping(mapping: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if not isinstance(mapping, Mapping):
        return result
    for key, value in mapping.items():
        result[key] = _sanitize_value(value)
    return result


def _select_summary_extras(extras: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    if not isinstance(extras, Mapping):
        return snapshot
    keys = (
        "staging_csv",
        "staging_meta",
        "mode",
        "rows_total",
        "status_raw",
        "exit_code",
        "strict_empty",
        "no_date_filter",
        "offline_smoke",
        "skip_requested",
        "zero_rows_reason",
        "auth_missing",
    )
    for key in keys:
        if key in extras:
            snapshot[key] = extras[key]
    window_info = extras.get("window") if isinstance(extras, Mapping) else None
    if isinstance(window_info, Mapping):
        snapshot["window"] = dict(window_info)
    return snapshot


def _write_ingestion_summary(
    *,
    mode: str,
    status: str,
    reason: str,
    rows_out: int,
    output_path: str,
    extras: Optional[Mapping[str, Any]] = None,
    http: Optional[Mapping[str, Any]] = None,
    counts: Optional[Mapping[str, Any]] = None,
) -> None:
    payload = {
        "mode": mode,
        "status": status,
        "reason": reason or "",
        "rows_out": int(rows_out or 0),
        "output_path": output_path,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "http": _sanitize_mapping(http),
        "counts": _sanitize_mapping(counts),
        "extras": _sanitize_mapping(extras),
    }
    try:
        write_json(INGESTION_SUMMARY_PATH, payload)
    except Exception:  # pragma: no cover - diagnostics helper
        LOG.debug("dtm: unable to write ingestion summary", exc_info=True)


def _emit_summary_from_report(
    *,
    mode: str,
    status: str,
    reason: str,
    rows: int,
    extras: Optional[Mapping[str, Any]],
    http: Optional[Mapping[str, Any]],
    counts: Optional[Mapping[str, Any]],
) -> None:
    extras_snapshot = _select_summary_extras(extras)
    counts_payload = _sanitize_mapping(counts)
    rows_out = counts_payload.get("written")
    if isinstance(rows_out, (int, float)):
        computed_rows = int(rows_out)
    else:
        computed_rows = int(extras_snapshot.get("rows_total") or rows or 0)
    output_path = str(extras_snapshot.get("staging_csv") or OUT_PATH)
    _write_ingestion_summary(
        mode=mode,
        status=status,
        reason=reason or "",
        rows_out=computed_rows,
        output_path=output_path,
        extras=extras_snapshot,
        http=http,
        counts=counts_payload,
    )


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
        mode=mode,
        status=status,
        reason=reason or f"{mode}: {status}",
        extras=extras_payload,
        http=http_payload,
        counts=counts_payload,
    )


def _complete_soft_timeout_rescue(
    *,
    args: argparse.Namespace,
    base_extras: Mapping[str, Any],
    base_http: Mapping[str, Any],
    deps: Mapping[str, Any],
    timings_ms: Mapping[str, int],
    diagnostics_ctx: Any,
    details: Mapping[str, Any],
    offline: bool,
    zero_reason: str = "connect_timeout",
) -> int:
    timings_payload = dict(timings_ms)
    http_payload = dict(base_http)
    http_payload["timeout"] = int(http_payload.get("timeout", 0) or 0) + 1
    http_payload["last_status"] = http_payload.get("last_status")
    http_payload["retries"] = int(http_payload.get("retries", 0) or 0)
    http_payload.setdefault("rate_limit_remaining", None)

    try:
        RESCUE_PROBE_PATH.parent.mkdir(parents=True, exist_ok=True)
        write_json(RESCUE_PROBE_PATH, details)
    except Exception:  # pragma: no cover - diagnostics helper
        LOG.debug("dtm: unable to persist TLS preflight details", exc_info=True)

    _ensure_out_dir_exists(OUT_DIR)
    _write_header_only_csv(OUT_PATH, RESCUE_HEADERS)
    ensure_manifest_for_csv(OUT_PATH, schema_version="dtm_displacement.v1", source_id="dtm")
    _write_http_trace_placeholder(Path(HTTP_TRACE_PATH), offline=offline)

    window_start_iso, window_end_iso, window_override_env = _resolve_effective_window()

    config_countries: List[str] = []
    discovery_struct: Optional[Dict[str, Any]] = None
    try:
        cfg = load_config()
    except Exception:  # pragma: no cover - defensive guard
        LOG.debug("dtm: failed to load config during rescue path", exc_info=True)
        cfg = None  # type: ignore[assignment]
    if isinstance(cfg, Mapping):
        api_cfg = cfg.get("api") if isinstance(cfg.get("api"), Mapping) else {}
        if isinstance(api_cfg, Mapping):
            config_countries = _normalize_list(api_cfg.get("countries"))
        config_section = dict(base_extras.get("config", {}))
        if config_countries:
            config_section["countries_mode"] = "explicit_config"
            config_section["countries_count"] = len(config_countries)
            config_section["countries_preview"] = config_countries[:5]
            config_section["selected_iso3_preview"] = config_countries[:10]
            config_section["config_countries_count"] = len(config_countries)
            config_parse = config_section.get("config_parse")
            if isinstance(config_parse, Mapping):
                updated_parse = dict(config_parse)
                updated_parse["countries"] = list(config_countries)
                updated_parse["countries_count"] = len(config_countries)
                config_section["config_parse"] = updated_parse
            config_keys = config_section.get("config_keys_found")
            if isinstance(config_keys, Mapping):
                updated_keys = dict(config_keys)
                updated_keys["countries"] = True
                config_section["config_keys_found"] = updated_keys
            discovery_struct = {
                "used_stage": "explicit_config",
                "configured_labels": list(config_countries),
                "unresolved_labels": [],
                "total_countries": len(config_countries),
            }
        base_extras_config = dict(base_extras.get("config", {}))
        base_extras_config.update(config_section)
    else:
        base_extras_config = dict(base_extras.get("config", {}))

    effective_params: Dict[str, Any] = {"countries": list(config_countries)}
    if window_start_iso or window_end_iso:
        effective_params["window"] = {
            "start": window_start_iso,
            "end": window_end_iso,
            "override_env": window_override_env,
        }

    extras_payload = dict(base_extras)
    extras_payload["config"] = base_extras_config
    extras_payload.update(
        {
            "mode": extras_payload.get("mode", "real"),
            "rows_total": 0,
            "rows_written": 0,
            "exit_code": 0,
            "status_raw": "ok",
            "zero_rows_reason": zero_reason,
            "soft_timeouts": True,
            "soft_timeout_applied": True,
            "rescue_probe": dict(details),
            "timings_ms": dict(timings_payload),
        }
    )
    extras_payload["effective_params"] = effective_params
    extras_payload["window"] = {
        "start_iso": window_start_iso,
        "end_iso": window_end_iso,
        "override_env": window_override_env,
    }
    extras_payload["dtm"] = {"base_url": "unknown"}
    if discovery_struct:
        extras_payload["discovery"] = discovery_struct
    else:
        extras_payload.setdefault("discovery", {"used_stage": None})
    extras_payload["http"] = {
        "count_2xx": 0,
        "count_4xx": 0,
        "count_5xx": 0,
        "retries": 0,
        "timeouts": int(http_payload.get("timeout", 0) or 0),
        "last_status": None,
        "endpoints_top": [],
    }
    extras_payload["artifacts"] = {
        "run_json": str(RUN_DETAILS_PATH),
        "http_trace": str(DTM_HTTP_LOG_PATH),
        "samples": str(SAMPLE_ADMIN0_PATH),
    }
    extras_payload["staging_csv"] = str(OUT_PATH)
    extras_payload["staging_meta"] = str(META_PATH)
    extras_payload["diagnostics_dir"] = str(DTM_DIAGNOSTICS_DIR)

    _write_meta(
        0,
        window_start_iso,
        window_end_iso,
        deps=deps,
        effective_params=effective_params,
        http_counters=http_payload,
        timings_ms=dict(timings_payload),
        discovery=discovery_struct,
        diagnostics=None,
        status="ok",
        reason=zero_reason,
        zero_rows_reason=zero_reason,
    )

    configured_labels = list(config_countries)
    run_payload = {
        "window": {
            "start": window_start_iso,
            "end": window_end_iso,
            "override_env": window_override_env,
        },
        "countries": {"requested": configured_labels, "resolved": configured_labels},
        "http": dict(http_payload),
        "paging": {"pages": 0, "page_size": None, "total_received": 0},
        "rows": {"fetched": 0, "normalized": 0, "written": 0, "kept": 0, "dropped": 0},
        "totals": {
            "rows_fetched": 0,
            "rows_normalized": 0,
            "rows_written": 0,
            "kept": 0,
            "dropped": 0,
            "parse_errors": 0,
        },
        "status": "ok",
        "reason": zero_reason,
        "outputs": {"csv": str(OUT_PATH), "meta": str(META_PATH)},
        "extras": dict(extras_payload),
        "args": vars(args),
    }
    try:
        _write_run_details(run_payload)
    except Exception:  # pragma: no cover - diagnostics helper
        LOG.debug("dtm: unable to persist run payload during rescue", exc_info=True)

    diagnostics_result = diagnostics_finalize_run(
        diagnostics_ctx,
        status="ok",
        reason=zero_reason,
        http=dict(http_payload),
        counts={"written": 0},
        extras=extras_payload,
    )
    diagnostics_append_jsonl(CONNECTORS_REPORT, diagnostics_result)
    _append_connectors_report(
        mode="real",
        status="ok-empty",
        rows=0,
        reason=zero_reason,
        extras=extras_payload,
        http=http_payload,
        counts={"written": 0},
    )
    _mirror_legacy_diagnostics()
    return 0


def _complete_fake_success(
    *,
    args: argparse.Namespace,
    base_extras: Mapping[str, Any],
    base_http: Mapping[str, Any],
    deps: Mapping[str, Any],
    timings_ms: Mapping[str, int],
    diagnostics_ctx: Any,
    window_start_iso: Optional[str],
    window_end_iso: Optional[str],
    configured_labels: Sequence[str],
    configured_iso3: Sequence[str],
    admin_levels: Sequence[str],
    mode: str,
    reason: str,
    offline: bool,
    diagnostics_details: Optional[Mapping[str, Any]] = None,
    window_override_env: Optional[bool] = None,
) -> int:
    from resolver.ingestion.dtm_fakes import write_fake_admin0_staging

    fake_started = time.perf_counter()
    iso_filter = list(configured_iso3)
    rows = write_fake_admin0_staging(
        out_csv=OUT_PATH,
        out_meta=META_PATH,
        window_start=window_start_iso,
        window_end=window_end_iso,
        iso3_whitelist=iso_filter or None,
        reason=reason,
        diagnostics=dict(diagnostics_details) if diagnostics_details else None,
    )
    ensure_manifest_for_csv(OUT_PATH, schema_version="dtm_displacement.v1", source_id="dtm")
    _write_http_trace_placeholder(Path(HTTP_TRACE_PATH), offline=offline)

    elapsed_ms = max(0, int((time.perf_counter() - fake_started) * 1000))
    timings_payload = dict(timings_ms)
    timings_payload["fake_write"] = elapsed_ms

    http_payload = dict(base_http)
    http_payload.setdefault("retries", 0)
    http_payload.setdefault("timeout", 0)
    http_payload.setdefault("rate_limit_remaining", None)
    http_payload["last_status"] = http_payload.get("last_status")
    http_payload["fake_runs"] = int(http_payload.get("fake_runs", 0) or 0) + 1

    extras_payload = dict(base_extras)
    if window_override_env is None:
        _, _, window_override_env = _resolve_effective_window()

    extras_payload.update(
        {
            "rows_total": rows,
            "rows_written": rows,
            "status_raw": "ok",
            "exit_code": 0,
            "window_start": window_start_iso,
            "window_end": window_end_iso,
            "window_override_env": window_override_env,
            "fake_data_used": True,
            "fake_reason": reason,
            "fake_mode": mode,
            "fake_rows_written": rows,
            "timings_ms": dict(timings_payload),
        }
    )
    extras_payload.setdefault("zero_rows_reason", None)
    extras_payload.setdefault("soft_timeouts", base_extras.get("soft_timeouts"))
    extras_payload["force_fake"] = bool(extras_payload.get("force_fake")) or mode == "force_fake"
    extras_payload["fake_on_timeout"] = bool(base_extras.get("fake_on_timeout"))
    extras_payload["http"] = dict(http_payload)
    extras_payload["deps"] = dict(deps)

    config_section_raw = extras_payload.get("config")
    if isinstance(config_section_raw, Mapping):
        config_section = dict(config_section_raw)
    else:
        config_section = {}
    if configured_labels:
        config_section.setdefault("countries_preview", list(configured_labels)[:5])
        config_section.setdefault("config_countries_count", len(configured_labels))
        config_section.setdefault("countries_mode", "explicit_config")
        if configured_iso3:
            config_section.setdefault("selected_iso3_preview", list(configured_iso3)[:10])
        parse_section_raw = config_section.get("config_parse")
        if isinstance(parse_section_raw, Mapping):
            parse_section = dict(parse_section_raw)
        else:
            parse_section = {}
        parse_section.setdefault("countries", list(configured_labels))
        if admin_levels:
            parse_section.setdefault("admin_levels", list(admin_levels))
        config_section["config_parse"] = parse_section
    if admin_levels:
        config_section.setdefault("admin_levels", list(admin_levels))
    extras_payload["config"] = config_section

    discovery_struct: Optional[Dict[str, Any]] = None
    if configured_labels:
        discovery_struct = {
            "used_stage": "explicit_config",
            "configured_labels": list(configured_labels),
            "unresolved_labels": [],
            "total_countries": len(configured_labels),
        }
        if iso_filter:
            discovery_struct["resolved_iso3"] = list(iso_filter)
        extras_payload["discovery"] = discovery_struct

    effective_params = {
        "countries": list(iso_filter or configured_labels),
        "admin_levels": list(admin_levels),
        "window": {
            "start": window_start_iso,
            "end": window_end_iso,
            "override_env": window_override_env,
        },
        "mode": mode,
        "fake_data_used": True,
    }
    extras_payload["effective_params"] = effective_params

    diagnostics_payload: Dict[str, Any]
    raw_diagnostics = extras_payload.get("diagnostics")
    if isinstance(raw_diagnostics, Mapping):
        diagnostics_payload = dict(raw_diagnostics)
    else:
        diagnostics_payload = {}
    if diagnostics_details:
        diagnostics_payload.setdefault("fake", {}).update(dict(diagnostics_details))
    diagnostics_payload.setdefault("fake_data_used", True)
    diagnostics_payload.setdefault("fake_reason", reason)
    extras_payload["diagnostics"] = diagnostics_payload

    try:
        meta_payload = json.loads(META_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        meta_payload = {}
    meta_payload.update(
        {
            "connector": "dtm_client",
            "level": "admin0",
            "fake_data": True,
            "fake_reason": reason,
            "window": {
                "start": window_start_iso,
                "end": window_end_iso,
                "override_env": window_override_env,
            },
            "iso3_filter": list(iso_filter),
            "http": dict(http_payload),
            "deps": dict(deps),
            "effective_params": effective_params,
            "written_rows": rows,
        }
    )
    if configured_labels:
        meta_payload.setdefault("configured_labels", list(configured_labels))
    if discovery_struct:
        meta_payload["discovery"] = discovery_struct
    meta_diagnostics: Dict[str, Any] = {}
    if isinstance(meta_payload.get("diagnostics"), Mapping):
        meta_diagnostics.update(meta_payload["diagnostics"])
    if diagnostics_details:
        meta_diagnostics.setdefault("fake", {}).update(dict(diagnostics_details))
    if meta_diagnostics:
        meta_payload["diagnostics"] = meta_diagnostics
    META_PATH.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    totals = {
        "rows_fetched": rows,
        "rows_normalized": rows,
        "rows_written": rows,
        "kept": rows,
        "dropped": 0,
        "parse_errors": 0,
    }
    run_payload = {
        "window": {
            "start": window_start_iso,
            "end": window_end_iso,
            "override_env": window_override_env,
        },
        "countries": {
            "requested": list(configured_labels),
            "resolved": list(iso_filter or configured_labels),
        },
        "http": dict(http_payload),
        "paging": {"pages": 0, "page_size": None, "total_received": rows},
        "rows": {
            "fetched": rows,
            "normalized": rows,
            "written": rows,
            "kept": rows,
            "dropped": 0,
        },
        "totals": totals,
        "status": "ok",
        "reason": reason,
        "outputs": {"csv": str(OUT_PATH), "meta": str(META_PATH)},
        "extras": dict(extras_payload),
        "args": vars(args),
    }
    _write_run_details(run_payload)

    diagnostics_result = diagnostics_finalize_run(
        diagnostics_ctx,
        status="ok",
        reason=reason,
        http=http_payload,
        counts={"written": rows},
        extras=extras_payload,
    )
    diagnostics_append_jsonl(CONNECTORS_REPORT, diagnostics_result)
    _append_connectors_report(
        mode="real",
        status="ok",
        rows=rows,
        reason=f"fake_data:{reason}",
        extras=extras_payload,
        http=http_payload,
        counts={"fetched": rows, "normalized": rows, "written": rows},
    )
    _mirror_legacy_diagnostics()
    return 0


class ConfigDict(dict):
    """Dictionary subclass that retains the source metadata for logging."""

    _source_path: Optional[str] = None
    _source_exists: bool = False
    _source_sha256: Optional[str] = None


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


def _preflight_tls(host: str = "dtmapi.iom.int", port: int = 443, timeout: float = 5.0) -> Tuple[bool, Dict[str, Any]]:
    details: Dict[str, Any] = {
        "host": host,
        "port": port,
        "timeout_sec": float(timeout),
    }
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            context = ssl.create_default_context()
            with context.wrap_socket(sock, server_hostname=host):
                pass
    except Exception as exc:
        details["ok"] = False
        details["error"] = str(exc) or type(exc).__name__
        details["error_type"] = type(exc).__name__
        details["elapsed_ms"] = max(0, int((time.perf_counter() - started) * 1000))
        return False, details
    details["ok"] = True
    details["elapsed_ms"] = max(0, int((time.perf_counter() - started) * 1000))
    return True, details


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
        serializable = dict(payload)
        errors = serializable.get("errors")
        if isinstance(errors, list):
            serializable["errors"] = errors[:3]
        DISCOVERY_FAIL_PATH.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
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
        writer.writerow(
            [
                "Operation",
                "admin0Name",
                "admin0Pcode",
                "CountryISO3",
                "ReportingDate",
                "idp_count",
            ]
        )
    _reset_admin0_sample_counter()


def _ensure_diagnostics_scaffolding() -> None:
    for directory in (
        DTM_DIAGNOSTICS_DIR,
        DTM_RAW_DIR,
        DTM_METRICS_DIR,
        DTM_SAMPLES_DIR,
        DTM_LOG_DIR,
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
        code_value = str(code or "").upper()
        iso_value = (iso or code_value).upper()
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
        rows.append(
            [
                str(op_value or ""),
                str(name or ""),
                code_value,
                iso_value,
                str(report_date or ""),
                str(idp_value or ""),
            ]
        )
    if not rows:
        return
    SAMPLE_ADMIN0_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SAMPLE_ADMIN0_PATH.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)
    _ADMIN0_SAMPLE_WRITTEN += len(rows)


def _sanitize_error_snippet(body: str, key: str) -> str:
    token = (key or "").strip()
    text = body[:1024]
    if token and token in text:
        text = text.replace(token, "***")
    return text


def _dtm_http_get(
    path: str,
    key: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
    timeout: Tuple[float, float],
    headers_override: Optional[Mapping[str, str]] = None,
    capture_error_body: bool = False,
) -> Any:
    base_url = "https://dtmapi.iom.int"
    url = f"{base_url}{path}"
    if headers_override:
        headers = dict(headers_override)
    else:
        header_name, env_key = _get_dtm_key_and_header(warn_on_missing=False)
        effective_key = key or (env_key or "")
        headers = {header_name: effective_key} if effective_key else {}
    started = time.perf_counter()
    entry: Dict[str, Any] = {"ts": time.time(), "url": url, "ok": False, "nonce": round(random.random(), 6)}
    if params:
        entry["params"] = dict(params)
    entry["header_variant"] = ",".join(sorted(headers.keys()))
    response: Optional[requests.Response] = None
    _dtm_http_get.last_status = None  # type: ignore[attr-defined]
    _dtm_http_get.last_error_payload = None  # type: ignore[attr-defined]
    _dtm_http_get.last_headers = dict(headers)  # type: ignore[attr-defined]
    try:
        if OFFLINE:
            entry["offline"] = True
            entry["ok"] = True
            entry["status"] = None
            LOG.debug("dtm: offline skip for HTTP GET %s", url)
            return []
        response = requests.get(url, headers=headers, params=params, timeout=timeout)
        entry["status"] = response.status_code
        _dtm_http_get.last_status = response.status_code  # type: ignore[attr-defined]
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
            _dtm_http_get.last_status = status  # type: ignore[attr-defined]
            if capture_error_body and status in {401, 403} and response is not None:
                try:
                    body_text = response.text
                except Exception:
                    body_text = ""
                if body_text:
                    _dtm_http_get.last_error_payload = _sanitize_error_snippet(body_text, key)  # type: ignore[attr-defined]
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


_dtm_http_get.last_status = None  # type: ignore[attr-defined]
_dtm_http_get.last_error_payload = None  # type: ignore[attr-defined]
_dtm_http_get.last_headers = {}  # type: ignore[attr-defined]


def _get_country_list_via_http(
    path: str,
    key: str,
    params: Optional[Mapping[str, Any]] = None,
    *,
    connect_timeout: float = 5.0,
    read_timeout: float = 30.0,
    retries: int = 3,
    backoff: float = 1.5,
    headers_variants: Optional[Sequence[Mapping[str, str]]] = None,
    capture_error_body: bool = False,
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
        headers: Optional[Mapping[str, str]],
    ) -> pd.DataFrame:
        payload = _dtm_http_get(
            path,
            key,
            params=params,
            timeout=(connect_timeout, read_timeout),
            headers_override=headers,
            capture_error_body=capture_error_body,
        )
        if isinstance(payload, pd.DataFrame):
            return payload
        return pd.DataFrame(payload or [])

    variants: Sequence[Optional[Mapping[str, str]]] = list(headers_variants or [None])
    last_exc: Optional[Exception] = None
    for index, header_variant in enumerate(variants):
        try:
            frame = _call(path, key, params, connect_timeout, read_timeout, header_variant)
            attempts = int(_call.retry.statistics.get("attempt_number", 1))
            _get_country_list_via_http.last_attempts = int(attempts)
            _get_country_list_via_http.last_status = getattr(_dtm_http_get, "last_status", None)
            if capture_error_body:
                _get_country_list_via_http.last_error_payload = getattr(_dtm_http_get, "last_error_payload", None)
            return frame
        except RetryError as exc:
            attempts = exc.last_attempt.attempt_number if exc.last_attempt else int(retries)
            _get_country_list_via_http.last_attempts = int(attempts)
            _get_country_list_via_http.last_status = getattr(_dtm_http_get, "last_status", None)
            if capture_error_body:
                _get_country_list_via_http.last_error_payload = getattr(_dtm_http_get, "last_error_payload", None)
            last_exc = exc
            status = getattr(_dtm_http_get, "last_status", None)
            if status in {401, 403} and index + 1 < len(variants):
                continue
            raise
        except Exception as exc:
            attempts = int(getattr(_call.retry, "statistics", {}).get("attempt_number", 1))
            _get_country_list_via_http.last_attempts = int(attempts)
            _get_country_list_via_http.last_status = getattr(_dtm_http_get, "last_status", None)
            if capture_error_body:
                _get_country_list_via_http.last_error_payload = getattr(_dtm_http_get, "last_error_payload", None)
            status = getattr(_dtm_http_get, "last_status", None)
            if status in {401, 403} and index + 1 < len(variants):
                last_exc = exc
                continue
            raise
    if last_exc:
        raise last_exc
    raise RuntimeError("DTM discovery HTTP fallback exhausted all header variants")


_get_country_list_via_http.last_attempts = 0  # type: ignore[attr-defined]
_get_country_list_via_http.last_status = None  # type: ignore[attr-defined]
_get_country_list_via_http.last_error_payload = None  # type: ignore[attr-defined]


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
    client: Optional[Any] = None,
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
        normalized_records: List[Dict[str, Any]] = []
        snapshot_records: List[Dict[str, Any]] = []
        resolved_selectors: List[str] = []
        unresolved_labels: List[str] = []
        ordered_selectors: List[str] = []
        seen_selectors: Set[str] = set()
        for country in requested_countries:
            iso_candidate = to_iso3(country, alias_map)
            selector = (iso_candidate or str(country)).strip()
            if iso_candidate:
                selector = iso_candidate.strip().upper()
            if not selector:
                selector = str(country).strip()
            if not iso_candidate:
                unresolved_labels.append(str(country))
            normalized_records.append(
                {
                    "admin0Name": str(country),
                    "admin0Pcode": str(selector),
                }
            )
            snapshot_records.append(
                {
                    "country_label": str(country),
                    "selector": str(selector),
                    "resolved_iso3": str(iso_candidate or ""),
                    "source": "explicit_config",
                }
            )
            resolved_selectors.append(str(selector))
            normalized_key = selector.upper() if selector and selector.isalpha() and len(selector) == 3 else selector
            if normalized_key not in seen_selectors and selector:
                ordered_selectors.append(str(selector))
                seen_selectors.add(normalized_key)
        discovered = pd.DataFrame(normalized_records, columns=["admin0Name", "admin0Pcode"])
        stage_entry = {
            "stage": "explicit_config",
            "status": "ok" if not discovered.empty else "empty",
            "rows": int(discovered.shape[0]),
            "attempts": 1,
            "latency_ms": 0,
            "http_code": None,
        }
        report = {
            "stages": [stage_entry],
            "errors": []
            if not discovered.empty
            else [{"stage": "explicit_config", "message": "empty_result"}],
            "attempts": {"explicit_config": 1},
            "latency_ms": {"explicit_config": 0},
            "used_stage": "explicit_config",
            "configured_labels": list(requested_countries),
            "resolved": list(ordered_selectors or resolved_selectors),
            "unresolved_labels": list(unresolved_labels),
        }
        _write_discovery_report(report)
        try:
            diagnostics_dump_json(discovered.to_dict(orient="records"), DISCOVERY_RAW_JSON_PATH)
        except Exception:  # pragma: no cover - diagnostics only
            LOG.debug("Unable to persist discovery JSON snapshot", exc_info=True)
        try:
            DISCOVERY_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
            snapshot_frame = pd.DataFrame(snapshot_records)
            if snapshot_frame.empty:
                snapshot_frame = pd.DataFrame(
                    [
                        {
                            "country_label": "",
                            "selector": "",
                            "resolved_iso3": "",
                            "source": "explicit_config",
                        }
                    ]
                ).iloc[0:0]
            snapshot_frame.to_csv(DISCOVERY_SNAPSHOT_PATH, index=False)
        except Exception:  # pragma: no cover - diagnostics only
            LOG.debug("Unable to persist discovery snapshot", exc_info=True)
        resolved_list = ordered_selectors or resolved_selectors
        if metrics is not None:
            metrics["countries_attempted"] = len(resolved_list)
            metrics["stage_used"] = "explicit_config"
            _write_metrics_summary_file(metrics)
        return DiscoveryResult(
            countries=resolved_list,
            frame=discovered,
            stage_used="explicit_config",
            report=report,
        )
    timeouts = api_cfg.get("timeouts", {}) if isinstance(api_cfg.get("timeouts"), Mapping) else {}
    retries_cfg = api_cfg.get("retries", {}) if isinstance(api_cfg.get("retries"), Mapping) else {}
    connect_timeout = float(timeouts.get("connect_seconds", 5))
    read_timeout = float(timeouts.get("read_seconds", 30))
    retry_attempts = int(retries_cfg.get("attempts", 3))
    backoff_seconds = float(retries_cfg.get("backoff_seconds", 1.5))

    header_name, configured_key = _get_dtm_key_and_header(warn_on_missing=not OFFLINE)
    key = (api_key or get_dtm_api_key() or configured_key or "").strip()
    if not key and not OFFLINE:
        raise RuntimeError("Missing DTM_API_KEY environment variable.")

    header_variants = build_discovery_header_variants(key)
    if not header_variants:
        header_variants = [{header_name: key}] if key else []

    stage_entries: List[Dict[str, Any]] = []
    stage_errors: List[Dict[str, Any]] = []
    attempts_map: Dict[str, int] = {}
    latency_map: Dict[str, int] = {}
    discovered = pd.DataFrame(columns=["admin0Name", "admin0Pcode"])
    snapshot_override: Optional[pd.DataFrame] = None
    used_stage: Optional[str] = None
    reason: Optional[str] = None
    first_fail_captured = False

    sdk_client = client

    def _header_label(headers: Optional[Mapping[str, str]]) -> str:
        if not headers:
            return "default"
        lowered = {str(key).strip().lower() for key in headers.keys()}
        if "x-api-key" in lowered:
            return "x_api"
        if "ocp-apim-subscription-key" in lowered:
            return "ocp"
        return "_".join(sorted(lowered))

    def _capture_http_failure(stage_name: str, path: str, status: Optional[int]) -> None:
        nonlocal first_fail_captured
        if first_fail_captured:
            return
        if status not in {401, 403}:
            return
        payload = getattr(_get_country_list_via_http, "last_error_payload", None)
        snippet = None
        if isinstance(payload, str):
            snippet = payload
        elif payload is not None:
            try:
                snippet = json.dumps(payload)
            except Exception:
                snippet = str(payload)
        _write_discovery_failure(
            f"http_{status}",
            message=f"{path} returned {status}",
            extra={"stage": stage_name, "status": status, "path": path, "body": snippet},
        )
        first_fail_captured = True

    if not requested_countries and sdk_client is not None:
        stage_name = "sdk"
        started = time.perf_counter()
        http_counts: Dict[str, int] = {}
        try:
            raw_frame = sdk_client.get_countries(http_counts=http_counts)
            latency_ms = int(max(0.0, (time.perf_counter() - started) * 1000))
            frame = _normalize_discovery_frame(raw_frame)
            attempts_map[stage_name] = 1
            latency_map[stage_name] = latency_ms
            http_code = http_counts.get("last_status")
            status = "ok" if not frame.empty else "empty"
            stage_entry = {
                "stage": stage_name,
                "status": status,
                "rows": int(frame.shape[0]),
                "attempts": 1,
                "latency_ms": latency_ms,
                "http_code": http_code,
            }
            stage_entries.append(stage_entry)
            if frame.empty:
                stage_errors.append({"stage": stage_name, "message": "empty_result"})
            else:
                discovered = frame
                used_stage = stage_name
        except DTMUnauthorizedError as exc:
            latency_ms = int(max(0.0, (time.perf_counter() - started) * 1000))
            http_code = getattr(exc, "status", getattr(exc, "status_code", None))
            attempts_map[stage_name] = 1
            latency_map[stage_name] = latency_ms
            stage_entries.append(
                {
                    "stage": stage_name,
                    "status": "error",
                    "rows": 0,
                    "attempts": 1,
                    "latency_ms": latency_ms,
                    "http_code": http_code,
                }
            )
            stage_errors.append({"stage": stage_name, "message": str(exc)})
        except Exception as exc:
            latency_ms = int(max(0.0, (time.perf_counter() - started) * 1000))
            attempts_map[stage_name] = 1
            latency_map[stage_name] = latency_ms
            stage_entries.append(
                {
                    "stage": stage_name,
                    "status": "error",
                    "rows": 0,
                    "attempts": 1,
                    "latency_ms": latency_ms,
                    "http_code": http_counts.get("last_status"),
                }
            )
            stage_errors.append({"stage": stage_name, "message": str(exc)})

    if not requested_countries and used_stage is None:
        http_paths = [
            ("http_country", "/v3/displacement/country-list"),
            ("http_operations", "/v3/displacement/operations-list"),
        ]
        for base_stage, path_suffix in http_paths:
            if used_stage:
                break
            for headers in header_variants or [None]:
                started = time.perf_counter()
                header_label = _header_label(headers)
                stage_name = f"{base_stage}_{header_label}" if header_label else base_stage
                try:
                    raw = _get_country_list_via_http(
                        path_suffix,
                        key,
                        connect_timeout=connect_timeout,
                        read_timeout=read_timeout,
                        retries=retry_attempts,
                        backoff=backoff_seconds,
                        headers_variants=[headers] if headers else None,
                        capture_error_body=True,
                    )
                    latency_ms = int(max(0.0, (time.perf_counter() - started) * 1000))
                    attempts = int(getattr(_get_country_list_via_http, "last_attempts", 1))
                    status_code = getattr(_get_country_list_via_http, "last_status", None)
                    frame = _normalize_discovery_frame(raw)
                    attempts_map[stage_name] = attempts
                    latency_map[stage_name] = latency_ms
                    status_text = "ok" if not frame.empty else "empty"
                    stage_entry = {
                        "stage": stage_name,
                        "status": status_text,
                        "rows": int(frame.shape[0]),
                        "attempts": attempts,
                        "latency_ms": latency_ms,
                        "http_code": status_code,
                    }
                    stage_entries.append(stage_entry)
                    if status_code in {401, 403}:
                        _capture_http_failure(stage_name, path_suffix, status_code)
                        stage_errors.append({"stage": stage_name, "message": f"http_{status_code}"})
                        continue
                    if frame.empty:
                        stage_errors.append({"stage": stage_name, "message": "empty_result"})
                        continue
                    discovered = frame
                    used_stage = stage_name
                    break
                except Exception as exc:
                    latency_ms = int(max(0.0, (time.perf_counter() - started) * 1000))
                    attempts = int(getattr(_get_country_list_via_http, "last_attempts", 1)) or 1
                    status_code = getattr(_get_country_list_via_http, "last_status", None)
                    attempts_map[stage_name] = attempts
                    latency_map[stage_name] = latency_ms
                    stage_entries.append(
                        {
                            "stage": stage_name,
                            "status": "error",
                            "rows": 0,
                            "attempts": attempts,
                            "latency_ms": latency_ms,
                            "http_code": status_code,
                        }
                    )
                    stage_errors.append({"stage": stage_name, "message": str(exc)})
                    _capture_http_failure(stage_name, path_suffix, status_code)
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

    if not countries:
        fallback_records = [
            {"admin0Name": name, "admin0Pcode": code}
            for name, code in STATIC_MINIMAL_FALLBACK
        ]
        discovered = pd.DataFrame(fallback_records, columns=["admin0Name", "admin0Pcode"])
        countries = [record["admin0Name"] for record in fallback_records]
        used_stage = "static_iso3_minimal"
        reason = "static_iso3_minimal_fallback"
        snapshot_override = pd.DataFrame(
            [
                {
                    "country_label": name,
                    "selector": code,
                    "resolved_iso3": code,
                    "source": "static_iso3_minimal",
                }
                for name, code in STATIC_MINIMAL_FALLBACK
            ]
        )
        stage_entries.append(
            {
                "stage": used_stage,
                "status": "fallback",
                "rows": int(discovered.shape[0]),
                "attempts": 1,
                "latency_ms": 0,
                "http_code": None,
            }
        )
        attempts_map[used_stage] = 1
        latency_map[used_stage] = 0
        stage_errors.append({"stage": used_stage, "message": "http_discovery_failed"})
        if not first_fail_captured:
            _write_discovery_failure(
                "http_discovery_failed",
                message="Discovery failed via SDK and HTTP; using minimal static ISO3 allowlist",
                extra={"fallback": [name for name, _ in STATIC_MINIMAL_FALLBACK]},
            )
            first_fail_captured = True

    report = {
        "stages": stage_entries,
        "errors": stage_errors,
        "attempts": attempts_map,
        "latency_ms": latency_map,
        "used_stage": used_stage,
        "reason": reason,
    }
    _write_discovery_report(report)

    try:
        diagnostics_dump_json(discovered.to_dict(orient="records"), DISCOVERY_RAW_JSON_PATH)
    except Exception:  # pragma: no cover - diagnostics only
        LOG.debug("Unable to persist discovery JSON snapshot", exc_info=True)
    try:
        DISCOVERY_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        snapshot_frame = snapshot_override if snapshot_override is not None else discovered
        snapshot_frame.to_csv(DISCOVERY_SNAPSHOT_PATH, index=False)
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
    out_dir = DTM_RAW_DIR
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
    mode: str = "real",
    status: str,
    reason: str,
    extras: Optional[Mapping[str, Any]] = None,
    http: Optional[Mapping[str, Any]] = None,
    counts: Optional[Mapping[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "connector_id": "dtm_client",
        "mode": mode,
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
    try:
        LEGACY_CONNECTORS_REPORT.parent.mkdir(parents=True, exist_ok=True)
        with LEGACY_CONNECTORS_REPORT.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=str))
            handle.write("\n")
    except Exception:  # pragma: no cover - diagnostics helper
        LOG.debug("dtm: unable to append legacy connectors report", exc_info=True)
    _emit_summary_from_report(
        mode=mode,
        status=status,
        reason=reason,
        rows=int(payload["counts"].get("written", 0) or 0),
        extras=extras_payload,
        http=http_payload,
        counts=payload["counts"],
    )
    _mirror_legacy_diagnostics()


def _append_summary_line(message: str) -> None:
    summary_path = DIAGNOSTICS_DIR / "summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    header = "# Connector Diagnostics\n\n"
    if summary_path.exists():
        existing = summary_path.read_text(encoding="utf-8")
    else:
        summary_path.write_text(header, encoding="utf-8")
        existing = header
    formatted = f"* dtm_client: {message} *\n"
    if formatted.strip() in {line.strip() for line in existing.splitlines()}:
        return
    with summary_path.open("a", encoding="utf-8") as handle:
        if not existing.endswith("\n"):
            handle.write("\n")
        handle.write(formatted)


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
        self._last_param_used: Optional[str] = None

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

    def _mark_skip_no_match(
        self,
        country: Optional[str],
        http_counts: Optional[MutableMapping[str, int]],
    ) -> None:
        if http_counts is not None:
            http_counts["skip_no_match"] = http_counts.get("skip_no_match", 0) + 1
        self._http_counts["skip_no_match"] = self._http_counts.get("skip_no_match", 0) + 1
        LOG.warning("Skipping unsupported country (no match): %s", country or "<unspecified>")

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

        trimmed = str(country).strip() if country else ""
        params = {"FromReportingDate": from_date, "ToReportingDate": to_date}
        attempts: List[Tuple[str, Dict[str, Any]]] = []

        base_kwargs = _country_kwargs(country)
        iso_candidate = base_kwargs.get("CountryISO3")
        if not iso_candidate and trimmed:
            iso_candidate = to_iso3(trimmed) or (trimmed.upper() if len(trimmed) == 3 else "")
            iso_candidate = (iso_candidate or "").strip().upper()[:3] or None

        if iso_candidate:
            attempts.append(("iso3", {"CountryISO3": iso_candidate}))
        if trimmed:
            attempts.append(("name", {"CountryName": trimmed}))
        if not attempts:
            attempts.append(("auto", dict(base_kwargs)))

        last_param_used: Optional[str] = None
        seen_no_match = False

        for idx, (mode, payload) in enumerate(attempts):
            if not payload:
                continue
            request_kwargs = dict(payload)
            request_kwargs.update(params)
            filter_target = payload.get("CountryISO3") or payload.get("CountryName") or trimmed or None
            try:
                frame = _sdk_call_with_arg_shim(
                    self.client,
                    "get_idp_admin0_data",
                    country_filter=_country_filter(filter_target),
                    **request_kwargs,
                )
            except ValueError as exc:
                if _is_no_country_match_error(exc):
                    seen_no_match = True
                    if idx + 1 < len(attempts):
                        if http_counts is not None:
                            http_counts["retries"] = http_counts.get("retries", 0) + 1
                        self._http_counts["retries"] = self._http_counts.get("retries", 0) + 1
                    continue
                LOG.error("Admin0 request failed: %s", exc)
                self._record_failure(exc, http_counts)
                raise
            except Exception as exc:
                LOG.error("Admin0 request failed: %s", exc)
                self._record_failure(exc, http_counts)
                raise

            self._record_success(http_counts, 200)
            if self.rate_limit_delay:
                time.sleep(self.rate_limit_delay)
            last_param_used = "iso3" if "CountryISO3" in payload else "name" if "CountryName" in payload else mode
            self._last_param_used = last_param_used
            return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)

        if seen_no_match:
            self._last_param_used = last_param_used or (attempts[-1][0] if attempts else None)
            self._mark_skip_no_match(country, http_counts)
            return pd.DataFrame()

        self._last_param_used = last_param_used
        return pd.DataFrame()

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
        trimmed = str(country).strip() if country else ""
        self._last_param_used = None
        base_country = _country_kwargs(country)
        attempts: List[Tuple[Dict[str, Any], Optional[str]]] = []
        base_kwargs: Dict[str, Any] = {**base_country}
        base_kwargs["FromReportingDate"] = from_date
        base_kwargs["ToReportingDate"] = to_date
        attempts.append(
            (
                base_kwargs,
                "iso3" if "CountryISO3" in base_country else "name" if base_country else None,
            )
        )
        if trimmed and base_country:
            if "CountryISO3" in base_country:
                fallback_country = {"CountryName": trimmed or base_country["CountryISO3"]}
                fallback_param = "name"
            else:
                iso_guess = to_iso3(trimmed) or (trimmed.upper() if trimmed else "")
                iso_guess = (iso_guess or "").strip().upper()[:3]
                if iso_guess:
                    fallback_country = {"CountryISO3": iso_guess}
                    fallback_param = "iso3"
                else:
                    fallback_country = {}
                    fallback_param = None
            if fallback_country:
                fallback_kwargs = {**fallback_country}
                fallback_kwargs["FromReportingDate"] = from_date
                fallback_kwargs["ToReportingDate"] = to_date
                if fallback_kwargs != base_kwargs:
                    attempts.append((fallback_kwargs, fallback_param))

        last_no_match = False
        last_param: Optional[str] = None
        for idx, (payload, param_used) in enumerate(attempts):
            last_param = param_used
            filter_target: Optional[str]
            if "CountryISO3" in payload:
                filter_target = payload.get("CountryISO3")
            elif "CountryName" in payload:
                filter_target = payload.get("CountryName")
            else:
                filter_target = country
            filter_kwargs = _country_filter(filter_target)
            try:
                frame = _sdk_call_with_arg_shim(
                    self.client,
                    "get_idp_admin1_data",
                    country_filter=filter_kwargs,
                    **payload,
                )
            except ValueError as exc:
                if _is_no_country_match_error(exc):
                    last_no_match = True
                    if idx + 1 < len(attempts):
                        if http_counts is not None:
                            http_counts["retries"] = http_counts.get("retries", 0) + 1
                        self._http_counts["retries"] = self._http_counts.get("retries", 0) + 1
                    continue
                LOG.error("Admin1 request failed: %s", exc)
                self._record_failure(exc, http_counts)
                raise
            except Exception as exc:
                LOG.error("Admin1 request failed: %s", exc)
                self._record_failure(exc, http_counts)
                raise
            else:
                self._record_success(http_counts, 200)
                if self.rate_limit_delay:
                    time.sleep(self.rate_limit_delay)
                self._last_param_used = param_used
                return frame

        if last_no_match:
            self._mark_skip_no_match(country, http_counts)
            self._last_param_used = last_param
            return None
        return None

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
        trimmed = str(country).strip() if country else ""
        self._last_param_used = None
        base_country = _country_kwargs(country)
        attempts: List[Tuple[Dict[str, Any], Optional[str]]] = []
        base_kwargs: Dict[str, Any] = {**base_country}
        base_kwargs["FromReportingDate"] = from_date
        base_kwargs["ToReportingDate"] = to_date
        if operation:
            base_kwargs["Operation"] = operation
        attempts.append(
            (
                base_kwargs,
                "iso3" if "CountryISO3" in base_country else "name" if base_country else None,
            )
        )
        if trimmed and base_country:
            if "CountryISO3" in base_country:
                fallback_country = {"CountryName": trimmed or base_country["CountryISO3"]}
                fallback_param = "name"
            else:
                iso_guess = to_iso3(trimmed) or (trimmed.upper() if trimmed else "")
                iso_guess = (iso_guess or "").strip().upper()[:3]
                if iso_guess:
                    fallback_country = {"CountryISO3": iso_guess}
                    fallback_param = "iso3"
                else:
                    fallback_country = {}
                    fallback_param = None
            if fallback_country:
                fallback_kwargs: Dict[str, Any] = {**fallback_country}
                fallback_kwargs["FromReportingDate"] = from_date
                fallback_kwargs["ToReportingDate"] = to_date
                if operation:
                    fallback_kwargs["Operation"] = operation
                if fallback_kwargs != base_kwargs:
                    attempts.append((fallback_kwargs, fallback_param))

        last_no_match = False
        last_param: Optional[str] = None
        for idx, (payload, param_used) in enumerate(attempts):
            last_param = param_used
            filter_target: Optional[str]
            if "CountryISO3" in payload:
                filter_target = payload.get("CountryISO3")
            elif "CountryName" in payload:
                filter_target = payload.get("CountryName")
            else:
                filter_target = country
            filter_kwargs = _country_filter(filter_target)
            try:
                frame = _sdk_call_with_arg_shim(
                    self.client,
                    "get_idp_admin2_data",
                    country_filter=filter_kwargs,
                    **payload,
                )
            except ValueError as exc:
                if _is_no_country_match_error(exc):
                    last_no_match = True
                    if idx + 1 < len(attempts):
                        if http_counts is not None:
                            http_counts["retries"] = http_counts.get("retries", 0) + 1
                        self._http_counts["retries"] = self._http_counts.get("retries", 0) + 1
                    continue
                LOG.error("Admin2 request failed: %s", exc)
                self._record_failure(exc, http_counts)
                raise
            except Exception as exc:
                LOG.error("Admin2 request failed: %s", exc)
                self._record_failure(exc, http_counts)
                raise
            else:
                self._record_success(http_counts, 200)
                if self.rate_limit_delay:
                    time.sleep(self.rate_limit_delay)
                self._last_param_used = param_used
                return frame

        if last_no_match:
            self._mark_skip_no_match(country, http_counts)
            self._last_param_used = last_param
            return None
        return None


def _env_bool(name: str, default: bool) -> bool:
    return getenv_bool(name, default=default)


def bool_from_env_or_flag(name: str, flag_value: bool, *, default: bool = False) -> bool:
    if flag_value:
        return True
    return _env_bool(name, default)


def _ensure_http_counts(counts: Optional[MutableMapping[str, int]]) -> MutableMapping[str, int]:
    base: MutableMapping[str, int] = counts or {}
    for key in HTTP_COUNT_KEYS:
        base.setdefault(key, 0)
    base.setdefault("retries", 0)
    base.setdefault("skip_no_match", 0)
    base.setdefault("last_status", None)
    return base


def _is_connect_timeout_error(exc: BaseException) -> bool:
    visited: Set[int] = set()
    current: Optional[BaseException] = exc
    while isinstance(current, BaseException) and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, requests.exceptions.ConnectTimeout):
            return True
        if isinstance(current, requests.exceptions.ReadTimeout):
            return True
        if Urllib3ConnectTimeoutError is not None and isinstance(current, Urllib3ConnectTimeoutError):
            return True
        if Urllib3ReadTimeoutError is not None and isinstance(current, Urllib3ReadTimeoutError):
            return True
        if isinstance(current, requests.exceptions.Timeout):
            message = str(current).lower()
            if "timed out" in message or "timeout" in message:
                return True
        name = type(current).__name__.lower()
        if "connecttimeout" in name:
            return True
        message = str(current).lower()
        if "connect timeout" in message or "timed out while connecting" in message or "read timeout" in message:
            return True
        next_exc = getattr(current, "__cause__", None)
        if not isinstance(next_exc, BaseException):
            next_exc = getattr(current, "__context__", None)
        current = next_exc
    return False


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


def _maybe_override_from_env(cfg: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    """Apply optional environment overrides for countries and admin levels."""

    api_cfg = cfg.setdefault("api", {})  # type: ignore[arg-type]

    countries_env = os.getenv("DTM_COUNTRIES", "").strip()
    if countries_env:
        overrides = [country.strip() for country in countries_env.split(",") if country.strip()]
        if overrides:
            LOG.debug("dtm: overriding countries via env (DTM_COUNTRIES)")
            api_cfg["countries"] = overrides

    levels_env = os.getenv("DTM_ADMIN_LEVELS", "").strip()
    if levels_env:
        overrides = [level.strip() for level in levels_env.split(",") if level.strip()]
        if overrides:
            LOG.debug("dtm: overriding admin levels via env (DTM_ADMIN_LEVELS)")
            api_cfg["admin_levels"] = overrides

    return cfg


def _normalize_keyword_mapping(raw: Mapping[str, Any]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for key, value in raw.items():
        key_text = str(key)
        items: List[str] = []
        if isinstance(value, str):
            tokens = [value]
        elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
            tokens = list(value)
        else:
            tokens = []
        for token in tokens:
            token_text = str(token).strip()
            if token_text:
                items.append(token_text)
        if items:
            normalized[key_text] = items
    return normalized


def _load_series_semantics_keywords() -> Dict[str, List[str]]:
    try:
        with SERIES_SEMANTICS_PATH.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError:
        LOG.debug("dtm: series_semantics.yml not present; using fallback keywords")
        data = {}
    except Exception:  # pragma: no cover - diagnostics helper
        LOG.debug("dtm: failed to load series_semantics.yml", exc_info=True)
        data = {}

    keywords_raw = data.get("shock_keywords") if isinstance(data, Mapping) else {}
    if isinstance(keywords_raw, Mapping):
        keywords = _normalize_keyword_mapping(keywords_raw)
        if keywords:
            return keywords

    # Minimal safe defaults required by tests
    return {"flood": ["flood"], "drought": ["drought"]}


def load_config() -> Dict[str, Any]:
    path = _resolve_config_path()
    exists = path.exists()
    data: Dict[str, Any] = {}
    sha_prefix: Optional[str] = None
    if exists:
        raw_bytes = path.read_bytes()
        sha_prefix = hashlib.sha256(raw_bytes).hexdigest()[:12]
        text = raw_bytes.decode("utf-8")
        parsed = yaml.safe_load(text) or {}
        if isinstance(parsed, dict):
            data = parsed
    semantics_keywords = _load_series_semantics_keywords()
    existing_keywords = data.get("shock_keywords") if isinstance(data, dict) else {}
    merged_keywords = dict(semantics_keywords)
    if isinstance(existing_keywords, Mapping):
        merged_keywords.update(_normalize_keyword_mapping(existing_keywords))
    if not merged_keywords:
        merged_keywords = {"flood": ["flood"], "drought": ["drought"]}
    data["shock_keywords"] = merged_keywords

    api_cfg = data.get("api") if isinstance(data.get("api"), Mapping) else {}
    countries: List[str] = []
    admin_levels: List[str] = []
    if isinstance(api_cfg, Mapping):
        raw_countries = api_cfg.get("countries")
        if isinstance(raw_countries, str):
            raw_items: Iterable[Any] = [raw_countries]
        elif isinstance(raw_countries, Iterable):
            raw_items = raw_countries
        else:
            raw_items = []
        for item in raw_items:
            text = str(item).strip()
            if text:
                countries.append(text)

        raw_levels = api_cfg.get("admin_levels")
        if isinstance(raw_levels, str):
            level_items: Iterable[Any] = [raw_levels]
        elif isinstance(raw_levels, Iterable):
            level_items = raw_levels
        else:
            level_items = []
        seen_levels: Set[str] = set()
        for level in level_items:
            token = str(level).strip().lower()
            if token in {"admin0", "admin1", "admin2"} and token not in seen_levels:
                admin_levels.append(token)
                seen_levels.add(token)

    config_keys_found = {"countries": bool(countries), "admin_levels": bool(admin_levels)}
    config_parse = {
        "countries": list(countries),
        "countries_count": len(countries),
        "countries_preview": list(countries[:5]),
        "countries_mode": "explicit_config" if countries else "discovered",
        "admin_levels": list(admin_levels),
        "config_keys_found": dict(config_keys_found),
    }

    cfg = ConfigDict(data)
    cfg._source_path = str(path.resolve())
    cfg._source_exists = bool(exists)
    cfg._source_sha256 = sha_prefix
    cfg._config_parse = config_parse
    return _maybe_override_from_env(cfg)


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
    extras_payload["staging_csv"] = str(OUT_PATH)
    extras_payload["staging_meta"] = str(META_PATH)

    _write_meta(
        0,
        None,
        None,
        deps=deps_payload,
        effective_params={},
        http_counters=http_payload,
        timings_ms=timings_ms,
        diagnostics={"mode": "offline-smoke", "offline": True},
        status="ok",
        reason=reason,
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
            "staging_csv": str(OUT_PATH),
            "staging_meta": str(META_PATH),
        },
        "args": vars(args),
    }

    _write_run_details(run_payload)
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
    mode: str = "skip",
    log_message: Optional[str] = None,
    skip_flags: Optional[Mapping[str, bool]] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if log_message:
        LOG.info(log_message)
    else:
        LOG.info("dtm: RESOLVER_SKIP_DTM detected; emitting header-only CSV and exiting")

    http_payload: Dict[str, Any] = {key: 0 for key in HTTP_COUNT_KEYS}
    http_payload["retries"] = 0
    http_payload["last_status"] = None
    http_payload["rate_limit_remaining"] = None

    counts_payload: Dict[str, Any] = {"fetched": 0, "normalized": 0, "written": 0}

    exit_code = 2 if strict_empty else 0

    extras_payload: Dict[str, Any] = {
        "mode": mode,
        "rows_total": 0,
        "status_raw": "skipped",
        "exit_code": exit_code,
        "strict_empty": strict_empty,
        "no_date_filter": no_date_filter,
        "offline_smoke": False,
    }
    if isinstance(skip_flags, Mapping):
        extras_payload["skip_flags"] = {key: bool(value) for key, value in skip_flags.items()}
    extras_payload["staging_csv"] = str(OUT_PATH)
    extras_payload["staging_meta"] = str(META_PATH)

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
        diagnostics={"mode": mode, "reason": reason},
        status="skipped",
        reason=reason,
        zero_rows_reason=reason,
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
            "mode": mode,
            "strict_empty": strict_empty,
            "no_date_filter": no_date_filter,
            "offline_smoke": False,
            "exit_code": exit_code,
            "staging_csv": str(OUT_PATH),
            "staging_meta": str(META_PATH),
        },
        "args": vars(args),
    }
    if isinstance(skip_flags, Mapping):
        run_payload["extras"]["skip_flags"] = {
            key: bool(value) for key, value in skip_flags.items()
        }

    _write_run_details(run_payload)
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
    kwargs: Dict[str, Any] = {
        "country": country,
        "from_date": from_date,
        "to_date": to_date,
        "http_counts": http_counts,
    }
    if operation is not None:
        kwargs["operation"] = operation
    frame, _ = _call_admin_level(client, level, **kwargs)
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
) -> Tuple[List[pd.DataFrame], int, Optional[str]]:
    method_name = ADMIN_METHODS.get(level, level)
    context = {
        "level": level,
        "country": country,
        "operation": operation,
        "from": from_date,
        "to": to_date,
        "method": "GET",
        "path": f"/{method_name}",
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
        context.update(
            {
                "status": "error",
                "error": str(exc),
                "elapsed_ms": elapsed_ms,
                "status_code": http_counts.get("last_status"),
            }
        )
        _append_request_log(context)
        raise

    elapsed_ms = max(0, int((time.perf_counter() - started) * 1000))
    method_used = getattr(client, "_dtm_last_method", method_name)
    context["path"] = f"/{method_used}"
    rows = sum(int(page.shape[0]) for page in pages if hasattr(page, "shape"))
    size_bytes = 0
    for page in pages:
        if hasattr(page, "memory_usage"):
            try:
                size_bytes += int(page.memory_usage(deep=True).sum())
            except Exception:
                size_bytes += 0
    param_mode = getattr(client, "_last_param_used", None)
    if param_mode == "iso3":
        context["param"] = "Admin0Pcode"
    elif param_mode == "name":
        context["param"] = "CountryName"
    elif param_mode:
        context["param"] = str(param_mode)
    context.update(
        {
            "status": "ok" if rows > 0 else "empty",
            "rows": rows,
            "elapsed_ms": elapsed_ms,
            "status_code": http_counts.get("last_status"),
            "size_bytes": size_bytes,
        }
    )
    _append_request_log(context)
    country_label = country or "all countries"
    op_suffix = f" / operation={operation}" if operation else ""
    LOG.info("Fetched %d rows for %s (%s%s)", rows, country_label, level, op_suffix)
    return pages, rows, param_mode


def _resolve_admin_levels(cfg: Mapping[str, Any]) -> List[str]:
    api_cfg = cfg.get("api", {})
    configured_raw = api_cfg.get("admin_levels")
    if configured_raw is None:
        configured_raw = cfg.get("admin_levels")
    configured = _normalize_list(configured_raw)
    if not configured:
        return ["admin0"]
    levels = []
    seen: Set[str] = set()
    for item in configured:
        text = str(item).strip().lower()
        if text in {"admin0", "admin1", "admin2"} and text not in seen:
            levels.append(text)
            seen.add(text)
    return levels or ["admin0"]


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


def _resolve_configured_iso3(labels: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    codes: List[str] = []
    for label in labels:
        iso_candidate = to_iso3(label)
        if not iso_candidate:
            continue
        iso = iso_candidate.strip().upper()
        if len(iso) != 3 or iso in seen:
            continue
        seen.add(iso)
        codes.append(iso)
    return codes


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
    attempted_date_columns: list[str] = []
    chosen_date_column: Optional[str] = None
    bad_date_samples: list[str] = []

    def _record_date_diag(diag: Mapping[str, Any]) -> None:
        nonlocal chosen_date_column
        if not isinstance(diag, Mapping):
            return
        attempted = diag.get("attempted")
        if isinstance(attempted, (list, tuple)):
            for candidate in attempted:
                candidate_str = str(candidate)
                if candidate_str and candidate_str not in attempted_date_columns:
                    attempted_date_columns.append(candidate_str)
        used_value = diag.get("used")
        if used_value and not chosen_date_column:
            chosen_date_column = str(used_value)
        bad = diag.get("bad_samples")
        if isinstance(bad, (list, tuple)):
            for token in bad:
                token_str = str(token)
                if not token_str:
                    continue
                if token_str not in bad_date_samples:
                    bad_date_samples.append(token_str)
                if len(bad_date_samples) >= 5:
                    break

    per_country_counts: List[Dict[str, Any]] = []
    per_country_diagnostics: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    summary_extras["per_country_counts"] = per_country_counts
    summary_extras["per_country"] = per_country_diagnostics
    summary_extras["failures"] = failures
    drop_reasons_counter = {key: 0 for key in _NORMALIZE_DROP_KEYS}
    drop_reasons_counter.setdefault("no_country_match", 0)
    summary_extras["drop_reasons_counter"] = drop_reasons_counter
    value_column_usage: Dict[str, int] = {}
    summary_extras["value_column_usage"] = value_column_usage
    filter_start = os.getenv("RESOLVER_START_ISO") or window_start
    filter_end = os.getenv("RESOLVER_END_ISO") or window_end
    date_filter_defaults = summary_extras.setdefault(
        "date_filter",
        {
            "window_start": filter_start,
            "window_end": filter_end,
            "inclusive": True,
            "parsed_ok": 0,
            "parsed_total": 0,
            "inside": 0,
            "outside": 0,
            "parse_failed": 0,
            "min_date": None,
            "max_date": None,
            "skipped": bool(no_date_filter),
        },
    )
    date_filter_defaults.setdefault("date_column_used", None)

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
        LOG.info("Config requested countries=%s", requested_countries)
        discovery_placeholder = summary_extras.setdefault("discovery", {})
        if not isinstance(discovery_placeholder, Mapping):
            discovery_placeholder = {}
            summary_extras["discovery"] = discovery_placeholder
        discovery_placeholder = dict(discovery_placeholder)
        discovery_placeholder.setdefault("configured_labels", list(requested_countries))
        discovery_placeholder.setdefault("total_countries", len(requested_countries))
        discovery_placeholder.setdefault("used_stage", "explicit_config")
        discovery_placeholder.setdefault("source", "explicit_config")
        discovery_placeholder.setdefault("unresolved_labels", [])
        summary_extras["discovery"] = discovery_placeholder

    target = client.client if hasattr(client, "client") else client
    discovery_result = _perform_discovery(cfg, metrics=metrics_summary, client=client)
    resolved_countries = discovery_result.countries
    discovery_source = discovery_result.stage_used or "none"
    metrics_summary["stage_used"] = discovery_source
    explicit_unresolved = []
    report_payload = discovery_result.report or {}
    if isinstance(report_payload, Mapping):
        unresolved_candidates = report_payload.get("unresolved_labels", [])
        if isinstance(unresolved_candidates, (list, tuple)):
            explicit_unresolved = [str(item) for item in unresolved_candidates if str(item).strip()]

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

    selected_preview = [str(item) for item in resolved_countries if str(item).strip()][:10]
    summary_extras["selected_iso3_preview"] = selected_preview

    country_mode = _countries_mode_from_stage(discovery_source)

    operations = requested_operations if requested_operations else [None]

    LOG.info("Fetching data for %d countries across %s", discovered_count, admin_levels)

    discovery_info = {
        "total_countries": discovered_count,
        "sdk_version": _dtm_sdk_version(),
        "api_base": getattr(target, "base_url", getattr(target, "_base_url", "unknown")),
        "discovery_file": str(DISCOVERY_SNAPSHOT_PATH),
        "source": discovery_source,
        "stages": discovery_result.report.get("stages", []),
        "report": discovery_result.report,
        "configured_countries": list(requested_countries),
        "resolved": list(resolved_countries),
        "unresolved_labels": explicit_unresolved,
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
    if explicit_unresolved:
        diagnostics_payload["unresolved_countries"] = explicit_unresolved
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
    for alias in idp_candidates:
        alias_str = str(alias)
        if not alias_str:
            continue
        value_column_usage.setdefault(alias_str, 0)

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
    level_rollup: Dict[str, Dict[str, int]] = defaultdict(lambda: {"calls": 0, "rows": 0, "elapsed_ms": 0})

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
                failure_reason: Optional[str] = None
                skip_before = http_counter.get("skip_no_match", 0)
                pages: List[pd.DataFrame] = []
                raw_rows = 0
                param_mode: Optional[str] = None
                with diagnostics_timing(f"{country_label}|{level}") as timing_result:
                    try:
                        pages, raw_rows, param_mode = _fetch_level_pages_with_logging(
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
                        failure_reason = type(exc).__name__
                        pages = []
                if param_mode is None:
                    param_mode = getattr(client, "_last_param_used", None)
                per_country_entry: Dict[str, Any] = {
                    "country": country_label,
                    "level": level,
                    "operation": operation,
                    "param": param_mode,
                    "pages": len(pages),
                    "rows": int(raw_rows),
                }
                skip_after = http_counter.get("skip_no_match", 0)
                if skip_after > skip_before:
                    if not status_entry.get("skip_no_match"):
                        metrics_summary["countries_skipped_no_match"] += 1
                    drop_reasons_counter["no_country_match"] += 1
                    status_entry["skip_no_match"] = True
                    unsupported_countries.add(country_name)
                    LOG.info(
                        "Skipping unsupported country after no-match response: %s",
                        country_label,
                    )
                    per_country_entry["skipped_no_match"] = True
                    per_country_entry["reason"] = "no_country_match"
                    per_country_diagnostics.append(per_country_entry)
                    break
                if failed:
                    per_country_entry["skipped_no_match"] = False
                    if failure_reason:
                        per_country_entry["reason"] = failure_reason
                    per_country_diagnostics.append(per_country_entry)
                    continue

                for page in pages:
                    summary["paging"]["pages"] += 1
                    summary["paging"]["total_received"] += int(page.shape[0])
                    if page.shape[0]:
                        size = int(page.shape[0])
                        current = summary["paging"]["page_size"] or 0
                        summary["paging"]["page_size"] = max(current, size)
                    raw_page = page.copy() if isinstance(page, pd.DataFrame) else page
                    if hasattr(page, "drop_duplicates"):
                        sort_cols = [
                            col
                            for col in [
                                "CountryISO3",
                                "ReportingDate",
                                "revision",
                                "ingested_at",
                            ]
                            if col in page.columns
                        ]
                        if sort_cols:
                            page = page.sort_values(
                                by=sort_cols,
                                ascending=[True] * len(sort_cols),
                                na_position="last",
                                kind="mergesort",
                            )
                        dedupe_candidates = [
                            col
                            for col in [
                                "CountryISO3",
                                "ReportingDate",
                                "idp_count",
                                "value",
                                "value_type",
                            ]
                            if col in page.columns
                        ]
                        if dedupe_candidates:
                            subset_cols = [
                                col for col in dedupe_candidates if col != "ReportingDate"
                            ]
                            if subset_cols and "ReportingDate" in dedupe_candidates:
                                subset_cols.append("ReportingDate")
                            if subset_cols:
                                before_dedupe = int(page.shape[0])
                                page = page.drop_duplicates(subset=subset_cols, keep="last")
                                page = page.reset_index(drop=True)
                                if int(page.shape[0]) != before_dedupe:
                                    LOG.debug(
                                        "dtm: deduped admin0 rows before normalization (before=%d after=%d keys=%s)",
                                        before_dedupe,
                                        int(page.shape[0]),
                                        subset_cols,
                                    )
                    if page.empty:
                        continue
                    LOG.debug(
                        "all columns (n=%d): %s",
                        len(page.columns),
                        list(page.columns),
                    )
                    original_rows = int(page.shape[0])
                    if level == "admin0":
                        original_columns = list(page.columns)

                        candidate_counts: Dict[str, int] = {}
                        for alias in idp_candidates:
                            alias_str = str(alias)
                            if not alias_str:
                                continue
                            if alias_str in page.columns:
                                numeric_series = pd.to_numeric(page[alias_str], errors="coerce")
                                count = int(numeric_series.notna().sum())
                            else:
                                count = 0
                            candidate_counts[alias_str] = count
                            if alias_str not in value_column_usage:
                                value_column_usage[alias_str] = 0
                            if count:
                                value_column_usage[alias_str] += count

                        date_col, parsed_dates, date_diag = _first_parsable_date_column(
                            page, _DATE_CANDIDATES
                        )
                        _record_date_diag(date_diag)
                        if isinstance(page, pd.DataFrame):
                            working_page = page.copy()
                        else:
                            working_page = pd.DataFrame(page)
                        if date_col and date_col in working_page.columns and date_col != "ReportingDate":
                            working_page = working_page.rename(columns={date_col: "ReportingDate"})
                        if parsed_dates is not None:
                            working_page["ReportingDate"] = parsed_dates
                        elif "ReportingDate" not in working_page.columns:
                            working_page["ReportingDate"] = pd.Series(
                                [pd.NA] * len(working_page), index=working_page.index, dtype="string"
                            )
                        page = working_page

                        def _iso_lookup(series: pd.Series) -> Tuple[Optional[str], Optional[str]]:
                            return resolve_iso3_fields(
                                series,
                                aliases=aliases,
                                name_keys=(country_column,),
                            )

                        normalized_result = normalize_admin0(
                            page,
                            idp_aliases=idp_candidates,
                            start_iso=None if no_date_filter else filter_start,
                            end_iso=None if no_date_filter else filter_end,
                            iso3_lookup=_iso_lookup,
                        )
                        local_drop = normalized_result.get("counters", {}).get("drop_reasons", {})
                        for reason, count in local_drop.items():
                            drop_reasons_counter[reason] = drop_reasons_counter.get(reason, 0) + int(count)
                        chosen_entries = normalized_result.get("counters", {}).get(
                            "chosen_value_columns", []
                        )
                        chosen_column = chosen_entries[0]["column"] if chosen_entries else None
                        chosen_count = int(chosen_entries[0]["count"]) if chosen_entries else 0
                        LOG.info("resolved value column: idp_count_col=%r", chosen_column)
                        if chosen_column:
                            value_column_usage.setdefault(chosen_column, 0)
                            if chosen_column not in candidate_counts:
                                value_column_usage[chosen_column] += chosen_count
                        normalized_page = normalized_result.get("df", pd.DataFrame())
                        summary.setdefault("rows", {}).setdefault("dropped", 0)
                        summary["rows"]["dropped"] += max(
                            0, original_rows - int(normalized_page.shape[0])
                        )
                        if normalized_result.get("zero_rows_reason") == "invalid_indicator":
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
                                    % (idp_candidates, original_columns),
                                }
                            )
                            continue
                        if normalized_page.empty:
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
                        try:
                            _record_admin0_sample(normalized_page, operation=operation)
                        except Exception:  # pragma: no cover - diagnostics only
                            LOG.debug("Unable to update admin0 sample", exc_info=True)
                        page = normalized_page
                        current_date_column = "ReportingDate"
                    else:
                        idp_column = _resolve_idp_value_column(page, idp_candidates)
                        LOG.info("resolved value column: idp_count_col=%r", idp_column)
                        if not idp_column:
                            LOG.warning(
                                "no value column matched; aliases=%s; rows=%d",
                                idp_candidates,
                                len(page),
                            )
                            drop_reasons_counter["no_value_col"] += int(page.shape[0])
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
                        raw_count = 0
                        if isinstance(raw_page, pd.DataFrame) and idp_column in raw_page.columns:
                            raw_count = int(raw_page[idp_column].notna().sum())
                        value_column_usage[idp_column] = value_column_usage.get(idp_column, 0) + raw_count
                        LOG.debug(
                                "DTM diag value-column raw count: %s=%s",
                                idp_column,
                                raw_count,
                            )
                        date_col, parsed_dates, date_diag = _first_parsable_date_column(
                            page, _DATE_CANDIDATES
                        )
                        _record_date_diag(date_diag)
                        if isinstance(page, pd.DataFrame):
                            working_page = page.copy()
                        else:
                            working_page = pd.DataFrame(page)
                        if date_col and date_col in working_page.columns and date_col != "ReportingDate":
                            working_page = working_page.rename(columns={date_col: "ReportingDate"})
                        if parsed_dates is not None:
                            working_page["ReportingDate"] = parsed_dates
                        elif "ReportingDate" not in working_page.columns:
                            working_page["ReportingDate"] = pd.Series(
                                [pd.NA] * len(working_page), index=working_page.index, dtype="string"
                            )
                        page = working_page
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
                        iso_source = None
                        for candidate in ("CountryISO3", "Admin0Pcode", "admin0Pcode", "iso3"):
                            if candidate in page.columns:
                                iso_source = candidate
                                break
                        if iso_source and iso_source != "CountryISO3":
                            page = page.rename(columns={iso_source: "CountryISO3"})
                        if "CountryISO3" in page.columns:
                            page.loc[:, "CountryISO3"] = (
                                page["CountryISO3"].astype("string", copy=False).str.strip().str.upper()
                            )
                        current_date_column = _pick_reporting_date_column(
                            page, preferred=(date_column,)
                        )
                        page, date_diag = _apply_date_window(
                            page,
                            filter_start,
                            filter_end,
                            disable=bool(no_date_filter),
                            diag_out_dir=DTM_DIAGNOSTICS_DIR,
                            extras=summary_extras,
                            preferred_columns=(date_column,),
                        )
                        drop_counts_diag = date_diag.get("drop_counts")
                        if isinstance(drop_counts_diag, Mapping):
                            for reason, count in drop_counts_diag.items():
                                drop_reasons_counter[reason] = drop_reasons_counter.get(reason, 0) + int(count)
                        summary.setdefault("rows", {}).setdefault("dropped", 0)
                        summary["rows"]["dropped"] += int(date_diag.get("outside", 0))
                        if page.empty:
                            continue
                        current_date_column = (
                            date_diag.get("date_column_used")
                            or current_date_column
                            or date_column
                        )

                    per_admin: Dict[Tuple[str, str], Dict[datetime, float]] = defaultdict(dict)
                    per_admin_asof: Dict[Tuple[str, str], Dict[datetime, str]] = defaultdict(dict)
                    causes: Dict[Tuple[str, str], str] = {}

                    for _, row in page.iterrows():
                        iso, iso_reason = resolve_iso3_fields(
                            row,
                            aliases=aliases,
                            name_keys=(country_column,),
                        )
                        if not iso:
                            drop_reasons_counter["no_iso3"] += 1
                            if iso_reason:
                                LOG.debug(
                                    "dtm: unable to resolve ISO3 (reason=%s) for %r",
                                    iso_reason,
                                    row.get(country_column),
                                )
                            continue
                        bucket = month_start(row.get(current_date_column))
                        if not bucket:
                            drop_reasons_counter["date_parse_failed"] += 1
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
                            drop_reasons_counter["other"] += 1
                            continue
                        per_admin[(iso, admin1)][bucket] = float(value)
                        as_of_value = _parse_iso_date_or_none(row.get(current_date_column))
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
                per_country_entry["skipped_no_match"] = False
                per_country_entry.setdefault("reason", "")
                per_country_diagnostics.append(per_country_entry)
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
                bucket = level_rollup[level]
                bucket["calls"] += 1
                bucket["rows"] += int(combo_count)
                bucket["elapsed_ms"] += int(elapsed_ms)
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

        rescue = _run_rescue_probe(
            client,
            aliases=aliases,
            idp_candidates=idp_candidates,
            countries=["Nigeria", "South Sudan"],
        )
        if rescue:
            summary_extras["rescue_probe"] = rescue
            diagnostics_payload = summary_extras.setdefault("diagnostics", {})
            diagnostics_payload["rescue_probe"] = rescue

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

    summary_extras["normalize_attempted_date_columns"] = list(attempted_date_columns)
    summary_extras["normalize_date_col_used"] = chosen_date_column
    summary_extras["normalize_bad_date_samples"] = bad_date_samples[:5]

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


def _run_rescue_probe(
    client: DTMApiClient,
    *,
    aliases: Mapping[str, str],
    idp_candidates: Sequence[str],
    countries: Sequence[str],
) -> Optional[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for country in countries:
        probe_counts = _ensure_http_counts({})
        try:
            frame = client.get_idp_admin0(
                country=country,
                from_date=None,
                to_date=None,
                http_counts=probe_counts,
            )
        except Exception as exc:  # pragma: no cover - network contingency
            results.append(
                {
                    "country": country,
                    "window": "no_date_filter",
                    "rows": 0,
                    "error": str(exc),
                }
            )
            continue
        if frame is None or frame.empty:
            results.append({"country": country, "window": "no_date_filter", "rows": 0})
            continue
        idp_column = _resolve_idp_value_column(frame, idp_candidates)
        rows_count = 0
        if idp_column:
            renamed = frame.rename(columns={idp_column: "idp_count"})
            normalized: Set[Tuple[str, str, str]] = set()
            for _, row in renamed.iterrows():
                iso, _ = resolve_iso3_fields(row, aliases=aliases, name_keys=("CountryName",))
                if not iso:
                    continue
                bucket = month_start(row.get("ReportingDate"))
                if not bucket:
                    continue
                value = _parse_float(row.get("idp_count"))
                if value is None or value <= 0:
                    continue
                normalized.add((str(iso).upper(), bucket, "dtm_api::admin0"))
            rows_count = len(normalized)
        results.append({"country": country, "window": "no_date_filter", "rows": rows_count})
    if not results:
        return None
    payload = {"timestamp": time.time(), "tried": results}
    write_json(RESCUE_PROBE_PATH, payload)
    return {"tried": results, "path": str(RESCUE_PROBE_PATH)}


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
        "--soft-timeouts",
        action="store_true",
        help="Treat connect timeouts as ok-empty instead of hard failures.",
    )
    parser.add_argument(
        "--force-fake",
        action="store_true",
        help="Always write clearly marked fake admin0 rows and exit successfully.",
    )
    parser.add_argument(
        "--fake-on-timeout",
        action="store_true",
        help="If a timeout occurs, write fake admin0 rows instead of failing.",
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
    status: Optional[str] = None,
    reason: Optional[str] = None,
    zero_rows_reason: Optional[str] = None,
    http_summary: Optional[Mapping[str, Any]] = None,
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
    http_payload: Dict[str, Any] = {key: int(http_counters.get(key, 0) or 0) for key in HTTP_COUNT_KEYS}
    http_payload["retries"] = int(http_counters.get("retries", 0) or 0)
    http_payload["last_status"] = http_counters.get("last_status")
    if "rate_limit_remaining" in http_counters:
        http_payload["rate_limit_remaining"] = http_counters.get("rate_limit_remaining")
    if isinstance(http_summary, Mapping):
        for key, value in http_summary.items():
            if value is not None:
                http_payload[key] = value
    payload["http"] = http_payload
    if discovery is not None:
        payload["discovery"] = dict(discovery)
    if diagnostics is not None:
        payload["diagnostics"] = dict(diagnostics)
    if status is not None:
        payload["status"] = status
    if reason is not None:
        payload["reason"] = reason
    if zero_rows_reason:
        payload["zero_rows_reason"] = zero_rows_reason
    write_json(META_PATH, payload)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _refresh_run_details_path()
    soft_timeouts = bool_from_env_or_flag(
        "DTM_SOFT_TIMEOUTS", getattr(args, "soft_timeouts", False)
    )
    force_fake = bool_from_env_or_flag(
        "DTM_FORCE_FAKE", getattr(args, "force_fake", False)
    )
    if not force_fake:
        force_fake = _env_bool("RESOLVER_FORCE_FAKE", False)
    allow_offline = _env_bool("RESOLVER_ALLOW_OFFLINE", False) or _env_bool(
        "DTM_ALLOW_OFFLINE", False
    )
    fake_on_timeout = bool_from_env_or_flag(
        "DTM_FAKE_ON_TIMEOUT", getattr(args, "fake_on_timeout", False)
    )
    global OUT_DIR, OUTPUT_PATH, META_PATH, OFFLINE, OUT_PATH
    log_level_name = str(os.getenv("LOG_LEVEL") or "INFO").upper()
    if args.debug:
        log_level_name = "DEBUG"
    logging.basicConfig(
        level=getattr(logging, log_level_name, logging.INFO),
        format="[%(levelname)s] %(message)s",
    )
    LOG.setLevel(getattr(logging, log_level_name, logging.INFO))
    _setup_file_logging()
    OUT_PATH = resolve_output_path(DEFAULT_OUTPUT)
    OUT_DIR = Path(OUT_PATH).parent
    OUTPUT_PATH = OUT_PATH
    META_PATH = OUT_PATH.with_suffix(OUT_PATH.suffix + ".meta.json")
    LOG.debug(
        "dtm: path constants -> diagnostics=%s dtm_diagnostics=%s staging=%s output=%s",
        DIAGNOSTICS_DIR,
        DTM_DIAGNOSTICS_DIR,
        OUT_DIR,
        OUT_PATH,
    )

    try:
        DTM_HTTP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        DTM_HTTP_LOG_PATH.touch(exist_ok=True)
    except Exception:  # pragma: no cover - diagnostics helper
        LOG.debug("dtm: unable to initialise diagnostics HTTP trace", exc_info=True)
    LOG.debug("dtm: canonical headers=%s", CANONICAL_HEADERS)
    LOG.debug(
        "dtm: outputs -> csv=%s meta=%s http_trace=%s",
        OUT_PATH,
        META_PATH,
        HTTP_TRACE_PATH,
    )

    strict_empty = (
        args.strict_empty
        or _env_bool("DTM_STRICT_EMPTY", False)
        or _env_bool("RESOLVER_STRICT_EMPTY", False)
    )
    no_date_filter = args.no_date_filter or _env_bool("DTM_NO_DATE_FILTER", False)
    offline_smoke = args.offline_smoke or _env_bool("DTM_OFFLINE_SMOKE", False)
    skip_env_requested = getenv_bool("RESOLVER_SKIP_DTM", default=False)
    force_run_override = getenv_bool("DTM_FORCE_RUN", default=False)
    skip_requested = skip_env_requested and not force_run_override
    skip_flags = {
        "RESOLVER_SKIP_DTM": skip_env_requested,
        "DTM_FORCE_RUN": force_run_override,
    }
    if skip_env_requested and force_run_override:
        LOG.info("dtm: DTM_FORCE_RUN override active; proceeding despite RESOLVER_SKIP_DTM")
    offline_mode = skip_requested or bool(offline_smoke)
    previous_offline = OFFLINE
    OFFLINE = offline_mode
    if skip_requested:
        LOG.info("Skip mode: no real network calls; writing header & trace placeholder")
    if OFFLINE:
        LOG.debug(
            "dtm: offline gating enabled (skip=%s offline_smoke=%s)",
            skip_requested,
            offline_smoke,
        )

    if skip_requested:
        skip_reason = "disabled via RESOLVER_SKIP_DTM"
        ensure_zero_row_outputs(offline=True)
        extras_payload, http_payload, counts_payload = _finalize_skip_run(
            reason=skip_reason,
            strict_empty=strict_empty,
            no_date_filter=no_date_filter,
            args=args,
            skip_flags=skip_flags,
        )
        window_start_iso, window_end_iso, window_override_env = _resolve_effective_window()
        config_candidate = _resolve_config_path()
        config_path_text = str(config_candidate.resolve())
        config_exists_initial = config_candidate.exists()
        try:
            config_sha = (
                hashlib.sha256(config_candidate.read_bytes()).hexdigest()[:12]
                if config_exists_initial
                else "n/a"
            )
        except Exception:  # pragma: no cover - diagnostics helper
            config_sha = "n/a"
        config_section: Dict[str, Any] = {
            "config_path_used": config_path_text,
            "config_exists": bool(config_exists_initial),
            "config_sha256": config_sha,
            "countries_mode": None,
            "countries_count": 0,
            "countries_preview": [],
            "admin_levels": [],
            "selected_iso3_preview": [],
            "no_date_filter": 1 if no_date_filter else 0,
            "config_parse": {"countries": [], "admin_levels": []},
            "config_keys_found": {"countries": False, "admin_levels": False},
            "config_countries_count": 0,
        }
        try:
            effective_cfg = load_config()
        except Exception:  # pragma: no cover - defensive guard
            LOG.debug("dtm: failed to load config during skip path", exc_info=True)
            effective_cfg = {}

        api_payload: Dict[str, Any] = {}
        if isinstance(effective_cfg, Mapping):
            api_cfg = effective_cfg.get("api") if isinstance(effective_cfg.get("api"), Mapping) else {}
            if isinstance(api_cfg, Mapping):
                countries = _normalize_list(api_cfg.get("countries"))
                admin_levels = _normalize_list(api_cfg.get("admin_levels"))
            else:
                countries = []
                admin_levels = []
            api_payload = {
                "countries": list(countries),
                "admin_levels": list(admin_levels),
            }
            if countries:
                config_section.setdefault("countries_mode", "explicit_config")
                config_section["countries_count"] = len(countries)
                config_section["config_countries_count"] = len(countries)
                config_section["countries_preview"] = countries[:5]
                config_section["selected_iso3_preview"] = countries[:10]
            if admin_levels:
                config_section["admin_levels"] = list(admin_levels)
            config_parse = config_section.get("config_parse")
            if isinstance(config_parse, Mapping):
                updated_parse = dict(config_parse)
                if countries:
                    updated_parse["countries"] = list(countries)
                    updated_parse["countries_count"] = len(countries)
                if admin_levels:
                    updated_parse["admin_levels"] = list(admin_levels)
                config_section["config_parse"] = updated_parse
            config_keys = config_section.get("config_keys_found")
            if isinstance(config_keys, Mapping):
                updated_keys = dict(config_keys)
                if countries:
                    updated_keys["countries"] = True
                if admin_levels:
                    updated_keys["admin_levels"] = True
                config_section["config_keys_found"] = updated_keys
        config_section["api"] = api_payload
        config_section["window"] = {
            "start": window_start_iso,
            "end": window_end_iso,
            "override_env": window_override_env,
        }

        exit_code = 2 if strict_empty else 0
        extras_payload["exit_code"] = exit_code
        extras_payload["config"] = config_section
        extras_payload.setdefault(
            "window",
            {
                "start": window_start_iso,
                "end": window_end_iso,
                "override_env": window_override_env,
            },
        )
        extras_for_reports = dict(extras_payload)
        extras_for_reports.update(
            {
                "skip_requested": True,
                "staging_csv": str(OUT_PATH),
                "staging_meta": str(META_PATH),
                "config": config_section,
            }
        )
        extras_for_reports.setdefault(
            "window",
            {
                "start": window_start_iso,
                "end": window_end_iso,
                "override_env": window_override_env,
            },
        )

        try:
            diagnostics_ctx_raw = diagnostics_start_run("dtm_client", "real")
        except Exception:  # pragma: no cover - diagnostics helper
            LOG.debug("dtm: diagnostics_start_run failed during skip", exc_info=True)
            diagnostics_ctx_raw = None

        if isinstance(diagnostics_ctx_raw, MutableMapping):
            diagnostics_ctx: MutableMapping[str, Any] = diagnostics_ctx_raw
        else:
            diagnostics_ctx = {
                "connector_id": "dtm_client",
                "mode": "real",
                "started_at": datetime.now(timezone.utc),
                "timer": time.perf_counter(),
                "http": {},
                "counts": {},
                "coverage": {},
                "samples": {},
                "extras": {},
            }

        existing_extras = diagnostics_ctx.get("extras") if isinstance(diagnostics_ctx, MutableMapping) else None
        ctx_extras: Dict[str, Any]
        if isinstance(existing_extras, Mapping):
            ctx_extras = dict(existing_extras)
        else:
            ctx_extras = {}
        ctx_extras.setdefault("mode", "skip")
        ctx_extras["skip_requested"] = True
        if isinstance(diagnostics_ctx, MutableMapping):
            diagnostics_ctx["extras"] = ctx_extras

        diagnostics_result = diagnostics_finalize_run(
            diagnostics_ctx,
            status="skipped",
            reason=skip_reason,
            http=http_payload,
            counts=counts_payload,
            extras=extras_for_reports,
        )
        diagnostics_append_jsonl(CONNECTORS_REPORT, diagnostics_result)
        _write_connector_report(
            mode="skip",
            status="skipped",
            reason=skip_reason,
            extras=extras_for_reports,
            http=http_payload,
            counts=counts_payload,
        )
        OFFLINE = previous_offline
        return exit_code

    _, env_key = _get_dtm_key_and_header(warn_on_missing=not OFFLINE)
    env_auth_key = (env_key or "").strip()

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

    if not have_dtmapi and not allow_offline and not force_fake:
        reason = "dependency-missing: dtmapi"
        run_payload = {
            "event": "connector_summary",
            "connector": "dtm",
            "status": "error",
            "reason": reason,
            "attempts": 1,
            "duration_ms": 0,
            "notes": "dtmapi not importable",
            "extras": {"deps": dict(dep_info)},
        }
        counts_payload = {"fetched": 0, "normalized": 0, "written": 0}
        extras_payload = {"deps": dict(dep_info), "exit_code": 1, "rows_total": 0}
        _write_run_details(run_payload)
        _write_connector_report(
            status="error",
            reason=reason,
            extras=extras_payload,
            http={},
            counts=counts_payload,
        )
        _append_summary_line(reason)
        ensure_zero_row_outputs(offline=OFFLINE)
        LOG.error(reason)
        OFFLINE = previous_offline
        return 1

    api_key_configured = bool(env_auth_key)
    base_http: Dict[str, Any] = {key: 0 for key in HTTP_COUNT_KEYS}
    base_http["rate_limit_remaining"] = None
    base_http["retries"] = 0
    base_http["last_status"] = None

    config_candidate = _resolve_config_path()
    config_path_text = str(config_candidate.resolve())
    config_exists_initial = config_candidate.exists()
    try:
        config_sha_initial = (
            hashlib.sha256(config_candidate.read_bytes()).hexdigest()[:12]
            if config_exists_initial
            else None
        )
    except Exception:  # pragma: no cover - diagnostics helper
        config_sha_initial = None
    base_extras: Dict[str, Any] = {
        "api_key_configured": api_key_configured,
        "deps": deps_payload,
        "strict_empty": strict_empty,
        "no_date_filter": no_date_filter,
        "offline_smoke": offline_smoke,
        "offline": OFFLINE,
        "timings_ms": dict(timings_ms),
        "soft_timeouts": bool(soft_timeouts),
        "force_fake": bool(force_fake),
        "fake_on_timeout": bool(fake_on_timeout),
        "skip_flags": dict(skip_flags),
    }
    base_extras["config"] = {
        "config_path_used": config_path_text,
        "config_exists": bool(config_exists_initial),
        "config_sha256": config_sha_initial or "n/a",
        "countries_mode": None,
        "countries_count": 0,
        "countries_preview": [],
        "admin_levels": [],
        "selected_iso3_preview": [],
        "no_date_filter": 1 if no_date_filter else 0,
        "config_parse": {"countries": [], "admin_levels": []},
        "config_keys_found": {"countries": False, "admin_levels": False},
        "config_countries_count": 0,
    }

    if not offline_smoke and have_dtmapi and not env_auth_key:
        ensure_zero_row_outputs(offline=True)
        extras_payload, http_payload, counts_payload = _finalize_skip_run(
            reason="auth_missing",
            strict_empty=strict_empty,
            no_date_filter=no_date_filter,
            args=args,
            mode="auth_missing",
            log_message="dtm: missing DTM API credentials; emitting header-only CSV",
            skip_flags=skip_flags,
        )
        extras_payload["auth_missing"] = True
        extras_for_reports = dict(extras_payload)
        extras_for_reports["auth_missing"] = True
        _append_connectors_report(
            mode="auth_missing",
            status="skipped",
            rows=0,
            reason="auth_missing",
            extras=extras_for_reports,
            http=http_payload,
            counts=counts_payload,
        )
        OFFLINE = previous_offline
        return 0

    diagnostics_ctx = diagnostics_start_run("dtm_client", "real")

    tls_ok = True
    tls_details: Dict[str, Any] = {}
    if soft_timeouts and not force_fake:
        try:
            tls_ok, tls_details = _preflight_tls()
        except Exception as exc:  # pragma: no cover - defensive guard
            tls_ok = False
            tls_details = {
                "ok": False,
                "error": str(exc) or type(exc).__name__,
                "error_type": type(exc).__name__,
            }
        if not tls_ok:
            LOG.warning("dtm: TLS preflight failed; applying soft-timeout rescue", exc_info=False)
            tls_details.setdefault("ok", False)
            tls_details.setdefault("stage", "preflight_tls")
            timings_ms = dict(timings_ms)
            timings_ms["preflight_tls"] = int(tls_details.get("elapsed_ms") or 0)
            base_extras["timings_ms"] = dict(timings_ms)
            exit_code = _complete_soft_timeout_rescue(
                args=args,
                base_extras=base_extras,
                base_http=base_http,
                deps=deps_payload,
                timings_ms=timings_ms,
                diagnostics_ctx=diagnostics_ctx,
                details=tls_details,
                offline=OFFLINE,
            )
            OFFLINE = previous_offline
            return exit_code

    _, raw_key = _get_dtm_key_and_header(warn_on_missing=False)
    raw_auth_key = (raw_key or "").strip()
    if not force_fake:
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
                    "skip_flags": dict(skip_flags),
                },
                "args": vars(args),
            }
            _write_run_details(run_payload)
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
                    "skip_flags": dict(skip_flags),
                },
                "args": vars(args),
            }
            _write_run_details(run_payload)
            _mirror_legacy_diagnostics()
            LOG.error("dtm: auth probe error: %s", exc)
            OFFLINE = previous_offline
            return 1
    http_stats: MutableMapping[str, int] = _ensure_http_counts({})
    extras: Dict[str, Any] = dict(base_extras)
    extras["mode"] = "skip" if skip_requested else extras.get("mode", "real")
    extras["skip_requested"] = skip_requested

    status = "ok"
    reason: Optional[str] = None
    exit_code = 0

    window_start_iso, window_end_iso, window_override_env = _resolve_effective_window()

    rows_written = 0
    summary: Dict[str, Any] = {}
    config_source_path = str(_resolve_config_path())
    config_exists_flag = False
    config_sha_prefix: Optional[str] = None
    configured_labels: List[str] = []
    configured_iso3: List[str] = []
    configured_admin_levels: List[str] = []

    try:
        cfg = load_config()
        LOG.info(
            "config_loaded_from=%s",
            getattr(cfg, "_source_path", "<unknown>"),
        )
        config_source_path = getattr(cfg, "_source_path", config_source_path)
        config_exists_flag = bool(getattr(cfg, "_source_exists", config_exists_flag))
        config_sha_prefix = getattr(cfg, "_source_sha256", config_sha_prefix)
        config_extras = base_extras.setdefault("config", {})
        config_extras.update(
            {
                "config_path_used": str(config_source_path),
                "config_exists": bool(config_exists_flag),
                "config_sha256": config_sha_prefix or "n/a",
            }
        )
        api_section = cfg.get("api") if isinstance(cfg.get("api"), Mapping) else {}
        configured_labels = _normalize_list(api_section.get("countries"))
        configured_iso3 = _resolve_configured_iso3(configured_labels)
        configured_admin_levels = _resolve_admin_levels(cfg)
        config_parse = getattr(cfg, "_config_parse", {})
        if isinstance(config_parse, Mapping):
            parsed_countries = [
                str(item)
                for item in config_parse.get("countries", [])
                if str(item).strip()
            ]
            parsed_admin_levels = [
                str(item)
                for item in config_parse.get("admin_levels", [])
                if str(item).strip()
            ]
            config_extras["config_parse"] = {
                "countries": parsed_countries,
                "admin_levels": parsed_admin_levels,
            }
            config_extras["config_keys_found"] = {
                "countries": bool(parsed_countries),
                "admin_levels": bool(parsed_admin_levels),
            }
            config_extras["config_countries_count"] = int(
                config_parse.get("countries_count", len(parsed_countries))
            )
            if parsed_countries and not config_extras.get("countries_preview"):
                config_extras["countries_preview"] = parsed_countries[:5]
            if parsed_admin_levels and not config_extras.get("admin_levels"):
                config_extras["admin_levels"] = parsed_admin_levels
            if (
                config_parse.get("countries_mode")
                and not config_extras.get("countries_mode")
            ):
                config_extras["countries_mode"] = str(config_parse.get("countries_mode"))
        if configured_labels and not config_extras.get("countries_preview"):
            config_extras["countries_preview"] = configured_labels[:5]
        if configured_iso3 and not config_extras.get("selected_iso3_preview"):
            config_extras["selected_iso3_preview"] = configured_iso3[:10]
        if configured_admin_levels and not config_extras.get("admin_levels"):
            config_extras["admin_levels"] = list(configured_admin_levels)
        if force_fake:
            LOG.warning("dtm: force-fake enabled; writing bundled staging rows and skipping API")
            diagnostics_details = {
                "mode": "force_fake",
                "configured_labels": list(configured_labels),
                "configured_iso3": list(configured_iso3),
            }
            exit_code = _complete_fake_success(
                args=args,
                base_extras=base_extras,
                base_http=base_http,
                deps=deps_payload,
                timings_ms=timings_ms,
                diagnostics_ctx=diagnostics_ctx,
                window_start_iso=window_start_iso,
                window_end_iso=window_end_iso,
                configured_labels=configured_labels,
                configured_iso3=configured_iso3,
                admin_levels=configured_admin_levels,
                mode="force_fake",
                reason="force_fake",
                offline=OFFLINE,
                diagnostics_details=diagnostics_details,
                window_override_env=window_override_env,
            )
            OFFLINE = previous_offline
            return exit_code
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
    except requests.exceptions.RequestException as exc:
        timeout_like = _is_connect_timeout_error(exc)
        if fake_on_timeout and timeout_like:
            LOG.error("dtm: timeout detected; writing fake rows due to fake-on-timeout", exc_info=True)
            details = {
                "mode": "fake_on_timeout",
                "exception": type(exc).__name__,
                "message": str(exc),
            }
            exit_code = _complete_fake_success(
                args=args,
                base_extras=base_extras,
                base_http=http_stats,
                deps=deps_payload,
                timings_ms=timings_ms,
                diagnostics_ctx=diagnostics_ctx,
                window_start_iso=window_start_iso,
                window_end_iso=window_end_iso,
                configured_labels=configured_labels,
                configured_iso3=configured_iso3,
                admin_levels=configured_admin_levels,
                mode="fake_on_timeout",
                reason="network_timeout",
                offline=OFFLINE,
                diagnostics_details=details,
                window_override_env=window_override_env,
            )
            OFFLINE = previous_offline
            return exit_code
        if soft_timeouts and timeout_like:
            LOG.error("dtm: timeout detected (%s); treating as ok-empty", exc)
            details = {
                "ok": False,
                "error": str(exc) or type(exc).__name__,
                "error_type": type(exc).__name__,
                "stage": "fetch",
            }
            timings_snapshot = dict(timings_ms)
            base_extras["timings_ms"] = dict(timings_snapshot)
            exit_code = _complete_soft_timeout_rescue(
                args=args,
                base_extras=base_extras,
                base_http=http_stats,
                deps=deps_payload,
                timings_ms=timings_snapshot,
                diagnostics_ctx=diagnostics_ctx,
                details=details,
                offline=OFFLINE,
            )
            OFFLINE = previous_offline
            return exit_code
        LOG.error("dtm: HTTP failure detected (%s)", exc)
        if timeout_like:
            http_stats["timeout"] = int(http_stats.get("timeout", 0) or 0) + 1
        else:
            http_stats["error"] = int(http_stats.get("error", 0) or 0) + 1
        status = "error"
        reason = str(exc) or ("timeout" if timeout_like else "http_error")
        exit_code = 1
        extras.setdefault(
            "exception",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        ensure_zero_row_outputs(offline=OFFLINE)
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
            "window_override_env": window_override_env,
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

    if strict_empty and zero_rows and exit_code == 0 and zero_rows_reason != "timeout":
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

    discovery_info_raw = summary_extras.get("discovery")
    discovery_info = dict(discovery_info_raw) if isinstance(discovery_info_raw, Mapping) else {}
    discovery_report = discovery_info.get("report") if isinstance(discovery_info.get("report"), Mapping) else {}
    stages_source = []
    if isinstance(discovery_report, Mapping):
        stages_source = discovery_report.get("stages") or []
    if not stages_source and isinstance(discovery_info.get("stages"), list):
        stages_source = discovery_info.get("stages") or []
    stage_attempts = discovery_report.get("attempts", {}) if isinstance(discovery_report, Mapping) else {}
    stage_latencies = discovery_report.get("latency_ms", {}) if isinstance(discovery_report, Mapping) else {}
    formatted_stages: List[Dict[str, Any]] = []
    for stage in stages_source:
        if not isinstance(stage, Mapping):
            continue
        name = stage.get("stage") or stage.get("name")
        entry = {
            "name": name,
            "status": stage.get("status"),
            "http_code": stage.get("http_status") or stage.get("code"),
            "attempts": stage.get("attempts"),
            "latency_ms": stage.get("latency_ms"),
        }
        if entry["attempts"] is None and isinstance(stage_attempts, Mapping) and name in stage_attempts:
            entry["attempts"] = stage_attempts.get(name)
        if entry["latency_ms"] is None and isinstance(stage_latencies, Mapping) and name in stage_latencies:
            entry["latency_ms"] = stage_latencies.get(name)
        formatted_stages.append(entry)
    used_stage = discovery_report.get("used_stage") if isinstance(discovery_report, Mapping) else None
    if not used_stage:
        used_stage = discovery_info.get("source")
    configured_labels = []
    if isinstance(discovery_report, Mapping):
        raw_configured = discovery_report.get("configured_labels")
        if isinstance(raw_configured, (list, tuple)):
            configured_labels = [str(item) for item in raw_configured]
    unresolved_labels = []
    if isinstance(discovery_report, Mapping):
        raw_unresolved = discovery_report.get("unresolved_labels")
        if isinstance(raw_unresolved, (list, tuple)):
            unresolved_labels = [str(item) for item in raw_unresolved]
    discovery_extras = {
        "stages": formatted_stages,
        "used_stage": used_stage,
        "reason": discovery_report.get("reason") if isinstance(discovery_report, Mapping) else discovery_info.get("reason"),
        "snapshot_path": str(DISCOVERY_SNAPSHOT_PATH),
        "first_fail_path": str(DISCOVERY_FAIL_PATH),
        "total_countries": discovery_info.get("total_countries"),
        "source": discovery_info.get("source"),
        "configured_labels": configured_labels,
        "unresolved_labels": unresolved_labels,
    }
    summary_extras["discovery"] = discovery_extras

    drop_counts_raw = summary_extras.get("drop_reasons_counter", {})
    drop_counts: Dict[str, int] = {}
    if isinstance(drop_counts_raw, Mapping):
        for key, value in drop_counts_raw.items():
            try:
                drop_counts[str(key)] = int(value)
            except Exception:
                continue
    for key in _NORMALIZE_DROP_KEYS:
        drop_counts.setdefault(key, 0)
    value_usage_raw = summary_extras.get("value_column_usage", {})
    value_usage = (
        {str(key): int(value) for key, value in value_usage_raw.items()}
        if isinstance(value_usage_raw, Mapping)
        else {}
    )
    chosen_value_columns = [
        {"column": column, "count": count}
        for column, count in sorted(value_usage.items(), key=lambda item: (-item[1], item[0]))
        if count > 0
    ]
    summary_extras.pop("drop_reasons_counter", None)
    summary_extras.pop("value_column_usage", None)

    skip_flags_ref = base_extras.get("skip_flags") if isinstance(base_extras, Mapping) else None
    if isinstance(skip_flags_ref, Mapping):
        summary_extras.setdefault("skip_flags", dict(skip_flags_ref))

    resolved_countries = []
    countries_info = summary.get("countries") if isinstance(summary, Mapping) else {}
    if isinstance(countries_info, Mapping):
        resolved_raw = countries_info.get("resolved", [])
        if isinstance(resolved_raw, list):
            resolved_countries = [item for item in resolved_raw if item]

    summary_extras["dtm"] = {
        "sdk_version": discovery_info.get("sdk_version") or _dtm_sdk_version(),
        "base_url": discovery_info.get("api_base"),
        "python_version": deps_payload.get("python"),
    }

    effective_admin_levels = effective_params.get("admin_levels") if isinstance(effective_params, Mapping) else []
    if isinstance(effective_admin_levels, (list, tuple)):
        admin_levels_list = list(effective_admin_levels)
    else:
        admin_levels_list = list(effective_params.get("admin_levels", []))
    countries_mode = _countries_mode_from_stage(used_stage)
    config_extras = base_extras.setdefault("config", {})
    selected_preview_list = summary_extras.get("selected_iso3_preview")
    if isinstance(selected_preview_list, Iterable) and not isinstance(selected_preview_list, (str, bytes)):
        selected_preview_values = [str(item) for item in selected_preview_list if str(item)]
    else:
        selected_preview_values = []
    if not selected_preview_values:
        selected_preview_values = [str(item) for item in resolved_countries if str(item)]
        summary_extras["selected_iso3_preview"] = selected_preview_values[:10]
    if not selected_preview_values:
        api_section = cfg.get("api") if isinstance(cfg, Mapping) else {}
        if isinstance(api_section, Mapping):
            raw_config_countries = api_section.get("countries")
            if isinstance(raw_config_countries, Iterable) and not isinstance(raw_config_countries, (str, bytes)):
                selected_preview_values = [
                    str(item)
                    for item in raw_config_countries
                    if str(item)
                ]
        if selected_preview_values and not summary_extras.get("selected_iso3_preview"):
            summary_extras["selected_iso3_preview"] = selected_preview_values[:10]
    config_extras.update(
        {
            "config_path_used": str(config_source_path),
            "config_exists": bool(config_exists_flag),
            "config_sha256": config_sha_prefix or "n/a",
            "admin_levels": admin_levels_list,
            "countries_mode": countries_mode,
            "countries_count": len(resolved_countries),
            "countries_preview": [str(item) for item in resolved_countries[:5]],
            "selected_iso3_preview": selected_preview_values[:10],
            "no_date_filter": 1 if no_date_filter else 0,
        }
    )
    summary_extras["config"] = dict(config_extras)

    summary_extras["window"] = {
        "start_iso": window_start_iso,
        "end_iso": window_end_iso,
        "override_env": window_override_env,
    }

    summary_extras["http"] = {
        "count_2xx": int(http_payload.get("2xx", 0)),
        "count_4xx": int(http_payload.get("4xx", 0)),
        "count_5xx": int(http_payload.get("5xx", 0)),
        "retries": int(http_payload.get("retries", 0)),
        "timeouts": int(http_payload.get("timeout", 0)),
        "last_status": http_payload.get("last_status"),
        "endpoints_top": [],
    }

    paging_info = summary.get("paging", {}) if isinstance(summary, Mapping) else {}
    fetch_extras_source = {}
    if isinstance(summary, Mapping):
        extras_ref = summary.get("extras")
        if isinstance(extras_ref, Mapping):
            fetch_extras_source = extras_ref.get("fetch", {}) if isinstance(extras_ref.get("fetch"), Mapping) else {}
    level_rollup_summary = []
    if isinstance(fetch_extras_source, Mapping):
        levels_value = fetch_extras_source.get("levels")
        if isinstance(levels_value, list):
            level_rollup_summary = [dict(entry) for entry in levels_value if isinstance(entry, Mapping)]
    summary_extras["fetch"] = {
        "pages": int(paging_info.get("pages", 0)) if isinstance(paging_info, Mapping) else 0,
        "max_page_size": paging_info.get("page_size") if isinstance(paging_info, Mapping) else None,
        "total_received": int(paging_info.get("total_received", 0)) if isinstance(paging_info, Mapping) else 0,
        "levels": level_rollup_summary,
    }

    attempted_dates = summary_extras.pop("normalize_attempted_date_columns", [])
    date_used = summary_extras.pop("normalize_date_col_used", None)
    bad_samples = summary_extras.pop("normalize_bad_date_samples", [])
    normalize_block: Dict[str, Any] = {
        "rows_fetched": int(rows_summary.get("fetched", 0)),
        "rows_normalized": int(rows_summary.get("normalized", rows_written)),
        "rows_written": rows_written,
        "raw_rows": int(rows_summary.get("fetched", 0)),
        "drop_reasons": drop_counts,
        "chosen_value_columns": chosen_value_columns,
    }
    normalize_block["config"] = {
        "countries_mode": countries_mode,
        "admin_levels": admin_levels_list,
    }
    if attempted_dates:
        normalize_block["attempted_date_columns"] = list(attempted_dates)
    if date_used:
        normalize_block["date_col_used"] = date_used
    if bad_samples:
        normalize_block["bad_date_samples"] = list(bad_samples)
    summary_extras["normalize"] = normalize_block

    rescue_info = summary_extras.get("rescue_probe")
    if isinstance(rescue_info, Mapping):
        rescue_info.setdefault("path", str(RESCUE_PROBE_PATH))

    summary_extras["artifacts"] = {
        "run_json": str(RUN_DETAILS_PATH),
        "http_trace": str(DTM_HTTP_LOG_PATH),
        "samples": str(SAMPLE_ADMIN0_PATH),
    }
    date_filter_info = summary_extras.get("date_filter")
    if isinstance(date_filter_info, Mapping):
        sample_path = date_filter_info.get("sample_path")
        if sample_path:
            summary_extras["artifacts"]["normalize_drops"] = str(sample_path)
    summary_extras["staging_csv"] = str(OUT_PATH)
    summary_extras["staging_meta"] = str(META_PATH)
    summary_extras["diagnostics_dir"] = str(DTM_DIAGNOSTICS_DIR)

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
        status=status,
        reason=reason,
        zero_rows_reason=zero_rows_reason,
        http_summary=summary_extras.get("http"),
    )

    run_payload = {
        "window": {
            "start": window_start_iso,
            "end": window_end_iso,
            "override_env": window_override_env,
        },
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

    _write_run_details(run_payload)
    _mirror_legacy_diagnostics()
    extras["run_details_path"] = str(RUN_DETAILS_PATH)
    extras["meta_path"] = str(META_PATH)
    if API_SAMPLE_PATH.exists():
        extras["api_sample_path"] = str(API_SAMPLE_PATH)
    extras["dtm"] = summary_extras.get("dtm")
    extras["config"] = summary_extras.get("config")
    extras["window"] = summary_extras.get("window")
    extras["discovery"] = summary_extras.get("discovery")
    extras["http"] = summary_extras.get("http")
    extras["fetch"] = summary_extras.get("fetch")
    extras["normalize"] = summary_extras.get("normalize")
    if summary_extras.get("rescue_probe"):
        extras["rescue_probe"] = summary_extras.get("rescue_probe")
    extras["artifacts"] = summary_extras.get("artifacts")
    extras["staging_csv"] = str(OUT_PATH)
    extras["staging_meta"] = str(META_PATH)
    extras["diagnostics_dir"] = str(DTM_DIAGNOSTICS_DIR)
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

