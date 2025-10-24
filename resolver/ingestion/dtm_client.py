#!/usr/bin/env python3
"""DTM connector that fetches displacement data exclusively through the official API."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
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
from resolver.ingestion.utils import (
    ensure_headers,
    flow_from_stock,
    month_start,
    stable_digest,
    to_iso3,
)
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

DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)

LOG = logging.getLogger("resolver.ingestion.dtm")

COLUMNS = [
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

DEFAULT_CAUSE = "unknown"
HTTP_COUNT_KEYS = ("2xx", "4xx", "5xx", "timeout", "error")
ROW_COUNT_KEYS = ("admin0", "admin1", "admin2", "total")


def _extract_status_code(exc: Exception) -> Optional[int]:
    message = str(exc)
    for token in re.findall(r"(\d{3})", message):
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
            frame = self.client.get_idp_admin2_data(
                CountryName=country,
                Operation=operation,
                FromReportingDate=from_date,
                ToReportingDate=to_date,
            )
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
    fetchers = {
        "admin0": client.get_idp_admin0,
        "admin1": client.get_idp_admin1,
        "admin2": client.get_idp_admin2,
    }
    fetcher = fetchers.get(level)
    if fetcher is None:
        raise ValueError(f"Unsupported admin level: {level}")
    kwargs = {
        "country": country,
        "from_date": from_date,
        "to_date": to_date,
        "http_counts": http_counts,
    }
    if level == "admin2":
        kwargs["operation"] = operation
    frame = fetcher(**kwargs)
    if frame is not None and not frame.empty:
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
        level_counts = 0
        for country_name in resolved_countries:
            for operation in (operations if level == "admin2" else [None]):
                combo_count = 0
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
                    else:
                        raise

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
                            level_counts += 1
                            summary["row_counts"][level] += 1
                            summary["row_counts"]["total"] += 1
                            summary["rows"]["fetched"] += 1

                range_label = f"{from_date or '-'}->{to_date or '-'}"
                country_label = country_name or "all countries"
                if level == "admin2" and operation:
                    country_label = f"{country_label} (operation={operation})"
                LOG.info(
                    "Fetched %s rows (%s) for %s %s",
                    f"{combo_count:,}",
                    level,
                    country_label,
                    range_label,
                )

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
    records, summary = _fetch_api_data(
        cfg,
        no_date_filter=no_date_filter,
        window_start=window_start,
        window_end=window_end,
        http_counts=http_stats,
    )
    rows = _finalize_records(records, summary=summary, write_sample=write_sample)
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


def _write_meta(rows: int, window_start: Optional[str], window_end: Optional[str]) -> None:
    payload: Dict[str, Any] = {"row_count": rows}
    if window_start:
        payload["backfill_start"] = window_start
    if window_end:
        payload["backfill_end"] = window_end
    write_json(META_PATH, payload)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv or ())
    log_level_name = str(os.getenv("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(level=getattr(logging, log_level_name, logging.INFO), format="[%(levelname)s] %(message)s")
    LOG.setLevel(getattr(logging, log_level_name, logging.INFO))

    api_key_configured = check_api_key_configured()
    diagnostics_ctx = diagnostics_start_run("dtm_client", "real")
    http_stats: MutableMapping[str, int] = _ensure_http_counts({})
    extras: Dict[str, Any] = {"api_key_configured": api_key_configured}

    status = "ok"
    reason: Optional[str] = None
    exit_code = 0

    strict_empty = args.strict_empty or _env_bool("DTM_STRICT_EMPTY", False)
    no_date_filter = args.no_date_filter or _env_bool("DTM_NO_DATE_FILTER", False)

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
                if rows:
                    write_rows(rows)
                else:
                    ensure_header_only()
                _write_meta(rows_written, window_start_iso, window_end_iso)
    except ValueError as exc:
        LOG.error("dtm: %s", exc)
        status = "error"
        reason = str(exc)
        exit_code = 2
        ensure_header_only()
    except Exception as exc:  # pragma: no cover - defensive guard
        LOG.exception("dtm: unexpected failure")
        status = "error"
        reason = str(exc)
        exit_code = 1
        ensure_header_only()

    if status == "ok":
        if rows_written == 0:
            status = "ok-empty"
            reason = "no rows returned"
        else:
            reason = f"wrote {rows_written} rows"

    if status == "ok-empty" and strict_empty and exit_code == 0:
        exit_code = 3
        LOG.error("dtm: strict-empty enabled; failing due to zero rows")

    extras.update(
        {
            "rows_total": rows_written,
            "window_start": window_start_iso,
            "window_end": window_end_iso,
            "strict_empty": strict_empty,
            "no_date_filter": no_date_filter,
            "exit_code": exit_code,
            "status_raw": status,
        }
    )

    http_payload = {key: int(http_stats.get(key, 0)) for key in HTTP_COUNT_KEYS}
    http_payload["retries"] = int(http_stats.get("retries", 0))
    http_payload["last_status"] = http_stats.get("last_status")

    rows_summary = summary.get("rows", {}) if isinstance(summary, Mapping) else {}

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
        "status": status,
        "reason": reason,
        "outputs": {"csv": str(OUT_PATH), "meta": str(META_PATH)},
        "extras": summary.get("extras", {}),
    }
    if rows_written == 0 and "api_sample_path" not in run_payload["extras"] and API_SAMPLE_PATH.exists():
        run_payload["extras"]["api_sample_path"] = str(API_SAMPLE_PATH)

    write_json(RUN_DETAILS_PATH, run_payload)
    extras["run_details_path"] = str(RUN_DETAILS_PATH)
    extras["meta_path"] = str(META_PATH)
    if API_SAMPLE_PATH.exists():
        extras["api_sample_path"] = str(API_SAMPLE_PATH)
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
