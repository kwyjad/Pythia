#!/usr/bin/env python3
"""DTM connector that fetches displacement data exclusively through the official API."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import yaml

from resolver.ingestion._manifest import ensure_manifest_for_csv
from resolver.ingestion._shared.run_io import count_csv_rows, write_json
from resolver.ingestion.diagnostics_emitter import (
    append_jsonl as diagnostics_append_jsonl,
    finalize_run as diagnostics_finalize_run,
    start_run as diagnostics_start_run,
)
from resolver.ingestion.dtm_auth import check_api_key_configured, get_dtm_api_key
from resolver.ingestion.utils import ensure_headers, flow_from_stock, month_start, stable_digest, to_iso3
from resolver.ingestion.utils.country_names import resolve_accept_names
from resolver.ingestion.utils.io import resolve_ingestion_window, resolve_output_path

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / "staging"
CONFIG_PATH = ROOT / "ingestion" / "config" / "dtm.yml"
DEFAULT_OUTPUT = ROOT / "staging" / "dtm_displacement.csv"
OUT_PATH = resolve_output_path(DEFAULT_OUTPUT)
OUT_DIR = OUT_PATH.parent
OUTPUT_PATH = OUT_PATH
META_PATH = OUT_PATH.with_suffix(OUT_PATH.suffix + ".meta.json")
DIAGNOSTICS_DIR = ROOT / "diagnostics" / "ingestion"
CONNECTORS_REPORT = DIAGNOSTICS_DIR / "connectors_report.jsonl"
RUN_DETAILS_PATH = DIAGNOSTICS_DIR / "dtm_run.json"
API_REQUEST_PATH = DIAGNOSTICS_DIR / "dtm_api_request.json"
API_SAMPLE_PATH = DIAGNOSTICS_DIR / "dtm_api_sample.json"
API_RESPONSE_SAMPLE_PATH = DIAGNOSTICS_DIR / "dtm_api_response_sample.json"

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


MULTI_HAZARD = Hazard("multi", "Multi-shock Displacement/Needs", "all")
UNKNOWN_HAZARD = Hazard("UNK", "Unknown / Unspecified", "all")

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

LOG = logging.getLogger("resolver.ingestion.dtm")

COLUMNS = CANONICAL_HEADERS

DEFAULT_CAUSE = "unknown"
HTTP_COUNT_KEYS = ("2xx", "4xx", "5xx", "timeout", "error")
ROW_COUNT_KEYS = ("admin0", "admin1", "admin2", "total")

ADMIN_METHODS = {
    "admin0": "get_idp_admin0",
    "admin1": "get_idp_admin1",
    "admin2": "get_idp_admin2",
}


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


def _discover_all_countries(
    api: Any, *, http_counts: Optional[MutableMapping[str, int]] = None
) -> List[str]:
    """Return a sorted list of country names discovered via the DTM SDK."""

    frame: Any = None

    getter = getattr(api, "get_all_countries", None)
    if callable(getter):
        frame = getter()
    else:
        getter = getattr(api, "get_countries", None)
        if callable(getter):
            try:
                if http_counts is not None:
                    frame = getter(http_counts=http_counts)
                else:
                    frame = getter()
            except TypeError:
                frame = getter()
        else:
            raise AttributeError("Provided API client lacks country discovery methods")

    if frame is None:
        return []

    if isinstance(frame, pd.DataFrame):
        data_frame = frame
    else:
        try:
            data_frame = pd.DataFrame(frame)
        except Exception:
            return []

    if data_frame.empty:
        return []

    names = set()
    for _, row in data_frame.iterrows():
        raw_value = row.get("CountryName", "")
        if pd.isna(raw_value):
            continue
        text = str(raw_value).strip()
        if not text:
            continue
        names.add(text)
    return sorted(names)


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

        api_key = subscription_key.strip() if subscription_key else None
        if not api_key:
            api_key = get_dtm_api_key()
        if not api_key:
            raise ValueError("DTM API key not configured")

        self.client = DTMApi(subscription_key=api_key)
        self.config = config
        api_cfg = config.get("api", {})
        self.rate_limit_delay = float(api_cfg.get("rate_limit_delay", 1.0))
        self.timeout = int(api_cfg.get("timeout", 60))

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
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


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

    summary_extras = summary.setdefault("extras", {})
    per_country_counts: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    summary_extras["per_country_counts"] = per_country_counts
    summary_extras["failures"] = failures

    primary_key = os.getenv("DTM_API_PRIMARY_KEY") or os.getenv("DTM_API_KEY")
    secondary_key = os.getenv("DTM_API_SECONDARY_KEY") or None

    client = DTMApiClient(cfg, subscription_key=primary_key)

    admin_levels = _resolve_admin_levels(cfg)
    api_cfg = cfg.get("api", {})
    requested_countries = _normalize_list(api_cfg.get("countries"))
    requested_operations = _normalize_list(api_cfg.get("operations"))

    summary["countries"]["requested"] = requested_countries

    try:
        resolved_countries = resolve_accept_names(client, requested_countries)
    except Exception as exc:
        LOG.warning("Failed to resolve country names: %s", exc)
        resolved_countries = requested_countries[:] if requested_countries else []

    discovery_mode = not requested_countries
    if not resolved_countries:
        target = client.client if hasattr(client, "client") else client
        if discovery_mode:
            LOG.info("No countries configured; discovering via DTM catalog")
        try:
            resolved_countries = _discover_all_countries(target, http_counts=http_counter)
        except Exception as exc:
            LOG.error("Failed to auto-discover countries: %s", exc)
            resolved_countries = []
        if not resolved_countries:
            try:
                countries_df = client.get_countries(http_counter)
            except Exception:
                countries_df = pd.DataFrame()
            if not countries_df.empty:
                resolved_countries = sorted(
                    {
                        str(row.get("CountryName", "")).strip()
                        for _, row in countries_df.iterrows()
                        if str(row.get("CountryName", "")).strip()
                    }
                )
        if resolved_countries:
            LOG.info("Discovered %d countries from DTM catalog", len(resolved_countries))
    summary["countries"]["resolved"] = resolved_countries

    if not resolved_countries:
        resolved_countries = [None]

    operations = requested_operations if requested_operations else [None]

    from_date = window_start if not no_date_filter else None
    to_date = window_end if not no_date_filter else None

    request_payload = {
        "admin_levels": admin_levels,
        "countries": None if resolved_countries == [None] else resolved_countries,
        "operations": None if operations == [None] else operations,
        "window_start": from_date,
        "window_end": to_date,
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
    effective_params = {
        "resource": summary["api"]["endpoint"],
        "admin_levels": admin_levels,
        "countries_requested": requested_countries,
        "countries_resolved": [] if resolved_countries == [None] else resolved_countries,
        "operations": None if operations == [None] else operations,
        "window_start": from_date,
        "window_end": to_date,
        "no_date_filter": bool(no_date_filter),
        "per_page": None,
        "max_pages": None,
        "country_mode": "ALL" if discovery_mode else "LIST",
        "countries_count": 0 if resolved_countries == [None] else len(resolved_countries),
    }
    summary.setdefault("extras", {})["effective_params"] = effective_params

    LOG.info(
        "effective: dates %s → %s; admin_levels=%s; country_mode=%s",
        from_date or "-",
        to_date or "-",
        admin_levels,
        effective_params["country_mode"],
    )

    field_mapping = cfg.get("field_mapping", {})
    field_aliases = cfg.get("field_aliases", {})
    country_column = field_mapping.get("country_column", "CountryName")
    admin1_column = field_mapping.get("admin1_column", "Admin1Name")
    admin2_column = field_mapping.get("admin2_column", "Admin2Name")
    date_column = field_mapping.get("date_column", "ReportingDate")
    idp_candidates = list(field_aliases.get("idp_count", ["TotalIDPs", "IDPTotal"]))
    if field_mapping.get("idp_column"):
        idp_candidates.insert(0, field_mapping["idp_column"])

    aliases = cfg.get("country_aliases") or {}
    measure = str(cfg.get("output", {}).get("measure", "stock")).strip().lower()
    cause_map = {
        str(k).strip().lower(): str(v)
        for k, v in (cfg.get("cause_map") or {}).items()
    }

    all_records: List[Dict[str, Any]] = []
    used_secondary = False

    for level in admin_levels:
        for country_name in resolved_countries:
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
                try:
                    pages = list(
                        _iter_level_pages(
                            client,
                            level,
                            country=country_name,
                            operation=operation,
                            from_date=from_date,
                            to_date=to_date,
                            http_counts=http_counter,
                        )
                    )
                except DTMHttpError as exc:
                    if (
                        isinstance(exc, DTMUnauthorizedError)
                        or exc.status_code in {401, 403}
                    ) and secondary_key and not used_secondary:
                        http_counter["retries"] += 1
                        LOG.warning("Retrying DTM API request with secondary key")
                        client = DTMApiClient(cfg, subscription_key=secondary_key)
                        used_secondary = True
                        try:
                            pages = list(
                                _iter_level_pages(
                                    client,
                                    level,
                                    country=country_name,
                                    operation=operation,
                                    from_date=from_date,
                                    to_date=to_date,
                                    http_counts=http_counter,
                                )
                            )
                        except Exception as retry_exc:
                            LOG.error(
                                "Failed after retry for country=%s level=%s: %s",
                                country_label,
                                level,
                                retry_exc,
                                exc_info=True,
                            )
                            failures.append(
                                {
                                    "country": country_label,
                                    "level": level,
                                    "operation": operation,
                                    "error": type(retry_exc).__name__,
                                    "message": str(retry_exc),
                                }
                            )
                            continue
                    else:
                        raise
                except Exception as exc:
                    LOG.error(
                        "country fetch failed: country=%s level=%s%s error=%s",
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
                    idp_column = next((col for col in idp_candidates if col in page.columns), None)
                    if not idp_column:
                        LOG.warning("No IDP value column present for %s", level)
                        continue

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
                        value = _parse_float(row.get(idp_column))
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
                                    "month": bucket,
                                    "value": float(value),
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
                LOG.info(
                    "country=%s level=%s rows=%s total_so_far=%s",
                    country_label if not operation else f"{country_label} (operation={operation})",
                    level,
                    combo_count,
                    summary["rows"]["fetched"],
                )

    effective_params = summary.get("extras", {}).get("effective_params")
    if isinstance(effective_params, MutableMapping):
        effective_params["per_page"] = summary.get("paging", {}).get("page_size")
        effective_params["max_pages"] = summary.get("paging", {}).get("pages")

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
        month = rec["month"]
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
    summary["http_counts"]["last_status"] = http_stats.get("last_status")
    return rows, summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-empty",
        action="store_true",
        help="Exit with code 3 when the connector writes zero rows.",
    )
    parser.add_argument(
        "--no-date-filter",
        action="store_true",
        help="Disable the ingestion window filter when pulling from the API.",
    )
    return parser.parse_args(argv)


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
    write_json(META_PATH, payload)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv or ())
    log_level_name = str(os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level_name, logging.INFO), format="[%(levelname)s] %(message)s")
    LOG.setLevel(getattr(logging, log_level_name, logging.INFO))

    strict_empty = args.strict_empty or _env_bool("DTM_STRICT_EMPTY", False)
    no_date_filter = args.no_date_filter or _env_bool("DTM_NO_DATE_FILTER", False)

    preflight_started = time.perf_counter()
    dep_info, have_dtmapi = _preflight_dependencies()
    timings_ms: Dict[str, int] = {"preflight": max(0, int((time.perf_counter() - preflight_started) * 1000))}
    _log_dependency_snapshot(dep_info)
    LOG.info(
        "env: python=%s exe=%s",
        dep_info.get("python", sys.version.split()[0]),
        dep_info.get("executable", sys.executable),
    )
    LOG.info(
        "deps: dtmapi=%s pandas=%s requests=%s",
        _package_version("dtmapi"),
        _package_version("pandas"),
        _package_version("requests"),
    )

    api_key_configured = check_api_key_configured()
    base_http: Dict[str, Any] = {key: 0 for key in HTTP_COUNT_KEYS}
    base_http["rate_limit_remaining"] = None
    base_http["retries"] = 0
    base_http["last_status"] = None
    base_extras: Dict[str, Any] = {
        "api_key_configured": api_key_configured,
        "deps": dep_info,
        "strict_empty": strict_empty,
        "no_date_filter": no_date_filter,
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
        ensure_header_only()
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
                "deps": dep_info,
                "timings_ms": dict(timings_ms),
                "strict_empty": strict_empty,
                "no_date_filter": no_date_filter,
            },
            "args": vars(args),
        }
        write_json(RUN_DETAILS_PATH, run_payload)
        LOG.error(reason)
        return 1

    diagnostics_ctx = diagnostics_start_run("dtm_client", "real")
    http_stats: MutableMapping[str, int] = _ensure_http_counts({})
    extras: Dict[str, Any] = dict(base_extras)

    status = "ok"
    reason: Optional[str] = None
    exit_code = 0

    window_start_dt, window_end_dt = resolve_ingestion_window()
    window_start_iso = window_start_dt.isoformat() if window_start_dt else None
    window_end_iso = window_end_dt.isoformat() if window_end_dt else None

    rows_written = 0
    summary: Dict[str, Any] = {}

    try:
        if os.getenv("RESOLVER_SKIP_DTM"):
            status = "skipped"
            reason = "disabled via RESOLVER_SKIP_DTM"
            ensure_header_only()
        else:
            cfg = load_config()
            if not cfg.get("enabled", True):
                status = "skipped"
                reason = "disabled via config"
                ensure_header_only()
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
                    ensure_header_only()
                timings_ms["write"] = max(0, int((time.perf_counter() - write_started) * 1000))
    except ValueError as exc:
        LOG.error("dtm: %s", exc)
        status = "error"
        reason = str(exc)
        exit_code = 2
        ensure_header_only()
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.exception("dtm: unexpected failure")
        status = "error"
        reason = f"exception: {type(exc).__name__}"
        extras["exception"] = {"type": type(exc).__name__, "message": str(exc)}
        extras["traceback"] = traceback.format_exc()
        exit_code = 1
        ensure_header_only()

    if status == "ok":
        if rows_written == 0:
            status = "ok-empty"
            reason = "header-only (0 rows)"
            extras["rows_total"] = 0
        else:
            reason = f"wrote {rows_written} rows"

    if status == "ok-empty" and strict_empty and exit_code == 0:
        exit_code = 3
        LOG.error("dtm: strict-empty enabled; failing due to zero rows")

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
    summary_extras.setdefault("deps", dep_info)
    totals = {
        "rows_fetched": rows_summary.get("fetched", 0),
        "rows_normalized": rows_summary.get("normalized", rows_written),
        "rows_written": rows_written,
        "kept": rows_summary.get("kept", rows_written),
        "dropped": rows_summary.get("dropped", 0),
        "parse_errors": rows_summary.get("parse_errors", 0),
    }

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
        deps=dep_info,
        effective_params=effective_params,
        http_counters=http_payload,
        timings_ms=timings_ms,
        per_country_counts=per_country_counts_payload,
        failures=failures_payload,
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
    if rows_written == 0 and "api_sample_path" not in run_payload["extras"] and API_SAMPLE_PATH.exists():
        run_payload["extras"]["api_sample_path"] = str(API_SAMPLE_PATH)

    write_json(RUN_DETAILS_PATH, run_payload)
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

    return exit_code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
