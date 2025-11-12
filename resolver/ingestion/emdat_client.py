"""Compatibility facade for the EM-DAT connector used in Resolver tests.

This module focuses on the subset of behaviour exercised by the unit tests:

* Reading per-event CSV inputs (optionally with HXL headers).
* Allocating totals across calendar months using either proportional or
  start-of-period policies.
* Mapping hazard keywords to Resolver hazard codes.
* Writing the canonical connector CSV even when the connector is skipped.

The implementation keeps logging and error handling lightweight while
remaining side-effect free on import so tests can monkeypatch configuration
paths easily.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from resolver.ingestion import diagnostics_emitter
from resolver.ingestion._manifest import ensure_manifest_for_csv
from resolver.ingestion._shared.config_loader import _load_yaml
from resolver.ingestion._shared.feature_flags import getenv_bool
from resolver.ingestion._shared.run_io import _ensure_parent, write_json, write_text
from resolver.ingestion.emdat_query import (
    EMDAT_METADATA_QUERY,
    EMDAT_PA_QUERY,
    apply_limit_override,
)
from resolver.ingestion.utils import (
    linear_split,
    map_hazard,
    month_start,
    parse_date,
    stable_digest,
    to_iso3,
)
from resolver.ingestion.utils.io import resolve_output_path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
STAGING = ROOT / "staging"
CONFIG_PATH = ROOT / "ingestion" / "config" / "emdat.yml"

COUNTRIES = DATA / "countries.csv"

DEFAULT_OUTPUT = ROOT / "staging" / "emdat_pa.csv"
OUT_PATH = resolve_output_path(DEFAULT_OUTPUT)
OUT_DIR = OUT_PATH.parent

SCHEMA_PATH = Path(__file__).with_name("schemas") / "emdat_pa.schema.yml"
EMDAT_DIAGNOSTICS_DIR = Path("diagnostics/ingestion/emdat")
EMDAT_PROBE_PATH = EMDAT_DIAGNOSTICS_DIR / "probe.json"
EMDAT_EFFECTIVE_PARAMS_PATH = EMDAT_DIAGNOSTICS_DIR / "effective.json"
EMDAT_PROBE_SAMPLE_PATH = EMDAT_DIAGNOSTICS_DIR / "probe_sample.json"
EMDAT_NORMALIZE_DEBUG_PATH = EMDAT_DIAGNOSTICS_DIR / "normalize_debug.json"

EMDAT_REACHABILITY_QUERY = """
query ProbeEmdat {
  api_version
  public_emdat {
    info {
      timestamp
      version
    }
  }
}
"""

EMDAT_PROBE_SAMPLE_LIMIT = 20


@lru_cache(maxsize=1)
def _emdat_schema() -> Dict[str, Any]:
    schema = _load_yaml(SCHEMA_PATH)
    if not schema:
        raise RuntimeError(f"Unexpected or empty EMDAT schema at {SCHEMA_PATH}")
    return schema


def _emdat_normalized_headers() -> List[str]:
    schema = _emdat_schema()
    columns = schema.get("columns", [])
    headers: List[str] = []
    for column in columns:
        name: Optional[str]
        if isinstance(column, Mapping):
            raw_name = column.get("name")
            name = str(raw_name) if raw_name is not None else None
        else:
            name = str(column) if column is not None else None
        if name:
            headers.append(name)
    if not headers or "iso3" not in headers or "as_of_date" not in headers:
        raise RuntimeError(f"Unexpected or empty EMDAT schema at {SCHEMA_PATH}")
    return headers


def _emdat_schema_version() -> str:
    schema = _emdat_schema()
    version = schema.get("version")
    if version is None:
        return "1"
    return str(version)


def _write_emdat_header_only_csv(out_path: Path | str) -> None:
    headers = CANONICAL_HEADERS
    target = Path(out_path)
    _ensure_parent(target)
    csv_header = ",".join(headers) + "\n"
    write_text(target, csv_header, encoding="utf-8")
    ensure_manifest_for_csv(target, schema_version=_emdat_schema_version(), source_id="emdat")

LOG = logging.getLogger("resolver.ingestion.emdat")

CANONICAL_HEADERS = _emdat_normalized_headers()


def _current_timestamp() -> str:
    stamp = datetime.now(timezone.utc)
    text = stamp.isoformat()
    if text.endswith("+00:00"):
        return text[:-6] + "Z"
    return text


def probe_emdat(
    api_key: str,
    base_url: str,
    timeout_s: float = 5.0,
    *,
    session: requests.Session | None = None,
) -> Dict[str, Any]:
    """Return reachability details for the EM-DAT GraphQL endpoint."""

    result: Dict[str, Any] = {
        "ok": False,
        "status": None,
        "latency_ms": None,
        "elapsed_ms": None,
        "api_version": None,
        "table_version": None,
        "metadata_timestamp": None,
        "error": None,
        "requests": {"total": 0, "2xx": 0, "4xx": 0, "5xx": 0},
    }

    if not api_key:
        result["error"] = "missing API key"
        return result

    headers = {"Content-Type": "application/json", "Authorization": api_key}
    payload = {"query": EMDAT_REACHABILITY_QUERY}

    sender = session.post if session is not None else requests.post  # type: ignore[union-attr]
    started = time.perf_counter()
    try:
        response = sender(base_url, json=payload, headers=headers, timeout=timeout_s)
    except requests.RequestException as exc:
        result["latency_ms"] = result["elapsed_ms"] = int(
            round((time.perf_counter() - started) * 1000)
        )
        result["error"] = str(exc)
        return result
    except Exception as exc:  # pragma: no cover - defensive guard
        result["latency_ms"] = result["elapsed_ms"] = int(
            round((time.perf_counter() - started) * 1000)
        )
        result["error"] = str(exc)
        return result

    elapsed_ms = int(round((time.perf_counter() - started) * 1000))
    status = getattr(response, "status_code", None)
    if isinstance(status, int):
        result["status"] = status
        if 200 <= status < 300:
            result["requests"]["2xx"] += 1
        elif 400 <= status < 500:
            result["requests"]["4xx"] += 1
        elif 500 <= status < 600:
            result["requests"]["5xx"] += 1
        result["requests"]["total"] += 1
    result["latency_ms"] = result["elapsed_ms"] = elapsed_ms

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        result["error"] = str(exc)
        return result

    try:
        parsed = response.json()
    except ValueError as exc:
        result["error"] = f"invalid JSON: {exc}"
        return result

    if not isinstance(parsed, Mapping):
        result["error"] = f"unexpected payload type: {type(parsed).__name__}"
        return result

    data = parsed.get("data")
    if not isinstance(data, Mapping):
        data = {}
    api_version = data.get("api_version")
    public = data.get("public_emdat")
    if not isinstance(public, Mapping):
        public = {}
    info = public.get("info")
    if not isinstance(info, Mapping):
        info = {}

    if api_version is not None:
        result["api_version"] = str(api_version)
    timestamp = info.get("timestamp")
    if timestamp is not None:
        result["metadata_timestamp"] = str(timestamp)
    version = info.get("version")
    if version is not None:
        result["table_version"] = str(version)

    result["ok"] = True
    return result


_HAZARD_INFO: Dict[str, Tuple[str, str, str]] = {
    "flood": ("FL", "Flood", "natural"),
    "drought": ("DR", "Drought", "natural"),
    "tropical_cyclone": ("TC", "Tropical Cyclone", "natural"),
    "storm": ("TC", "Tropical Cyclone", "natural"),
    "earthquake": ("EQ", "Earthquake", "natural"),
    "conflict": ("CF", "Conflict", "conflict"),
    "volcano": ("VO", "Volcanic Eruption", "natural"),
    "wildfire": ("WF", "Wildfire", "natural"),
    "landslide": ("LS", "Landslide", "natural"),
    "phe": ("PHE", "Public Health Emergency", "health"),
    "other": ("OT", "Other", "other"),
}


def _env_bool(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def ensure_header_only() -> None:
    _write_emdat_header_only_csv(OUT_PATH)


def _normalise_key(value: Any) -> str:
    return "".join(ch for ch in str(value or "").lower() if ch.isalnum())


@dataclass
class SourceFrame:
    df: pd.DataFrame
    column_map: Dict[str, str]
    hxl_map: Dict[str, str] = field(default_factory=dict)


def _prepare_frame(df: pd.DataFrame) -> SourceFrame:
    df = df.fillna("")
    hxl_map: Dict[str, str] = {}
    if not df.empty:
        for idx in list(df.index[:2]):
            row = df.loc[idx]
            if all(str(value).strip().startswith("#") for value in row.values if str(value).strip()):
                for column, tag in zip(df.columns, row.values):
                    tag_norm = _normalise_key(tag)
                    if tag_norm:
                        hxl_map[tag_norm] = column
                df = df.drop(index=idx)
        df = df.reset_index(drop=True)
    column_map: Dict[str, str] = {}
    for column in df.columns:
        norm = _normalise_key(column)
        if norm and norm not in column_map:
            column_map[norm] = column
    return SourceFrame(df=df, column_map=column_map, hxl_map=hxl_map)


def _find_column(frame: SourceFrame, keys: Sequence[str], prefer_hxl: bool) -> Optional[str]:
    for key in keys:
        norm = _normalise_key(key)
        if not norm:
            continue
        if prefer_hxl:
            column = frame.hxl_map.get(norm)
            if column:
                return column
        column = frame.column_map.get(norm)
        if column:
            return column
        if not prefer_hxl:
            column = frame.hxl_map.get(norm)
            if column:
                return column
    return None


def _best_of(
    record: MutableMapping[str, Any],
    frame: SourceFrame,
    keys: Sequence[str],
    *,
    prefer_hxl: bool,
) -> Any:
    if not keys:
        return None
    column = _find_column(frame, keys, prefer_hxl)
    if column:
        return record.get(column)
    return None


def _parse_people(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace(",", "")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if not pd.notna(number) or number <= 0:
        return None
    return float(number)


def _parse_people_int(value: Any) -> Optional[int]:
    """Best-effort integer parsing for population-style metrics."""

    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    cleaned = text.replace(",", "")
    try:
        return int(cleaned)
    except (TypeError, ValueError):
        try:
            return int(float(cleaned))
        except (TypeError, ValueError):
            return None


def _normalise_date_value(value: Any) -> str:
    parsed = parse_date(value)
    if parsed:
        return parsed.isoformat()
    return ""


def _allocation(
    total: float,
    start: date,
    end: Optional[date],
    *,
    policy: str,
) -> Dict[str, float]:
    if total is None or total <= 0 or start is None:
        return {}
    if end and end < start:
        end = start
    if policy == "start":
        bucket = month_start(start) or start.replace(day=1)
        return {f"{bucket.year:04d}-{bucket.month:02d}": float(total)}
    allocations = linear_split(total, start, end)
    buckets: Dict[str, float] = {}
    for bucket, amount in allocations:
        if amount is None:
            continue
        month = month_start(bucket) or bucket.replace(day=1)
        key = f"{month.year:04d}-{month.month:02d}"
        buckets[key] = buckets.get(key, 0.0) + float(amount)
    if not buckets:
        month = month_start(start) or start.replace(day=1)
        buckets[f"{month.year:04d}-{month.month:02d}"] = float(total)
    return buckets


def _hazard_tuple(key: str) -> Tuple[str, str, str]:
    normalised = str(key or "other").strip().lower()
    if normalised in _HAZARD_INFO:
        return _HAZARD_INFO[normalised]
    return _HAZARD_INFO["other"]


def _resolve_hazard(
    type_value: Any,
    subtype_value: Any,
    *,
    shock_map: Mapping[str, Sequence[str]],
    default_key: str,
) -> str:
    combined = " ".join(str(part or "") for part in (type_value, subtype_value)).strip()
    lowered = combined.lower()
    for hazard_key, keywords in shock_map.items():
        for keyword in keywords or []:
            if keyword and str(keyword).strip().lower() in lowered:
                return str(hazard_key).strip().lower()
    canonical = (
        map_hazard(type_value)
        or map_hazard(subtype_value)
        or map_hazard(combined)
    )
    if canonical:
        return canonical
    fallback = str(default_key or "other").strip().lower()
    if fallback:
        return fallback
    return "other"


def _country_names() -> Dict[str, str]:
    if not hasattr(_country_names, "_cache"):
        df = pd.read_csv(COUNTRIES, dtype=str).fillna("")
        _country_names._cache = {row.iso3: row.country_name for row in df.itertuples(index=False)}  # type: ignore[attr-defined]
    return getattr(_country_names, "_cache")  # type: ignore[attr-defined]


def _load_source_frame(source: Mapping[str, Any], prefer_hxl: bool) -> SourceFrame:
    if "data" in source:
        df = pd.DataFrame(source.get("data", []))
    else:
        kind = str(source.get("kind") or "csv").lower()
        url = source.get("url")
        if not url:
            return SourceFrame(df=pd.DataFrame(), column_map={})
        if kind == "xlsx":
            df = pd.read_excel(url, dtype=str)
        else:
            df = pd.read_csv(url, dtype=str)
    return _prepare_frame(df)


def _extract_metric_values(
    record: MutableMapping[str, Any],
    frame: SourceFrame,
    source: Mapping[str, Any],
    *,
    prefer_hxl: bool,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    total_value = _parse_people_int(
        _best_of(record, frame, source.get("total_affected_keys") or [], prefer_hxl=prefer_hxl)
    )
    affected_value = _parse_people_int(
        _best_of(record, frame, source.get("affected_keys") or [], prefer_hxl=prefer_hxl)
    )
    injured_value = _parse_people_int(
        _best_of(record, frame, source.get("injured_keys") or [], prefer_hxl=prefer_hxl)
    )
    homeless_value = _parse_people_int(
        _best_of(record, frame, source.get("homeless_keys") or [], prefer_hxl=prefer_hxl)
    )

    if total_value is None:
        if any(value is not None for value in (affected_value, injured_value, homeless_value)):
            total_value = (affected_value or 0) + (injured_value or 0) + (homeless_value or 0)

    if total_value is None:
        total_value = 0

    if total_value > 0:
        metrics["total_affected"] = float(total_value)

    extra_metrics = {
        "injured": injured_value,
        "homeless": homeless_value,
    }
    for metric, value in extra_metrics.items():
        if value is None or value <= 0:
            continue
        metrics[metric] = metrics.get(metric, 0.0) + float(value)

    return metrics


def collect_rows(cfg: Mapping[str, Any]) -> List[Dict[str, Any]]:
    sources = cfg.get("sources") or []
    if not isinstance(sources, Iterable):
        return []
    sources_list = [source for source in sources if isinstance(source, Mapping)]
    if not sources_list:
        return []

    prefer_hxl = bool(cfg.get("prefer_hxl", True))
    policy = str(os.getenv("EMDAT_ALLOC_POLICY") or cfg.get("allocation_policy") or "prorata").strip().lower()
    if policy not in {"prorata", "start"}:
        policy = "prorata"

    shock_map_cfg = cfg.get("shock_map") or {}
    shock_map: Dict[str, Sequence[str]] = {}
    if isinstance(shock_map_cfg, Mapping):
        for key, values in shock_map_cfg.items():
            if not key:
                continue
            if isinstance(values, str):
                value_list = [values]
            else:
                value_list = [str(v) for v in values or []]
            shock_map[str(key).strip().lower()] = value_list

    default_hazard = str(cfg.get("default_hazard", "other")).strip().lower() or "other"
    country_aliases = cfg.get("country_aliases") or {}
    if not isinstance(country_aliases, Mapping):
        country_aliases = {}

    country_names = _country_names()
    ingested_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    aggregates: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    raw_ids_by_key: Dict[Tuple[str, str, str, str], set[str]] = defaultdict(set)
    seen_records: set[Tuple[Any, ...]] = set()

    for source in sources_list:
        frame = _load_source_frame(source, prefer_hxl)
        if frame.df.empty:
            continue

        country_keys = source.get("country_keys") or ["iso3", "iso", "country"]
        start_keys = source.get("start_date_keys") or ["start", "start_date", "startdate"]
        end_keys = source.get("end_date_keys") or ["end", "end_date", "enddate"]
        type_keys = source.get("type_keys") or ["disastertype", "type"]
        subtype_keys = source.get("subtype_keys") or ["disastersubtype", "subtype"]
        id_keys = source.get("id_keys") or ["disno", "disasterno", "eventid"]
        title_keys = source.get("title_keys") or ["event name", "title"]
        source_url_keys = source.get("source_url_keys") or ["source_url", "source", "url"]
        publication_keys = source.get("publication_keys") or ["entry date", "publication_date"]

        publisher = str(source.get("publisher") or "CRED/EM-DAT")
        source_type = str(source.get("source_type") or "other")

        for record in frame.df.to_dict(orient="records"):
            iso_value = _best_of(record, frame, country_keys, prefer_hxl=prefer_hxl)
            iso3 = to_iso3(iso_value, country_aliases)
            if not iso3:
                continue

            start_raw = _best_of(record, frame, start_keys, prefer_hxl=prefer_hxl)
            end_raw = _best_of(record, frame, end_keys, prefer_hxl=prefer_hxl)
            start_date = parse_date(start_raw)
            end_date = parse_date(end_raw) or start_date
            if start_date is None:
                continue
            if end_date and end_date < start_date:
                end_date = start_date

            hazard_type = _best_of(record, frame, type_keys, prefer_hxl=prefer_hxl)
            hazard_subtype = _best_of(record, frame, subtype_keys, prefer_hxl=prefer_hxl)
            hazard_key = _resolve_hazard(
                hazard_type,
                hazard_subtype,
                shock_map=shock_map,
                default_key=default_hazard,
            )
            hazard_code, hazard_label, hazard_class = _hazard_tuple(hazard_key)

            raw_id = str(_best_of(record, frame, id_keys, prefer_hxl=prefer_hxl) or "").strip()
            if not raw_id:
                raw_id = stable_digest([
                    iso3,
                    hazard_code,
                    start_date.isoformat(),
                    end_date.isoformat() if end_date else "",
                ])

            metric_values = _extract_metric_values(record, frame, source, prefer_hxl=prefer_hxl)
            if not metric_values:
                continue

            record_key = (
                raw_id,
                iso3,
                hazard_key,
                start_date.isoformat(),
                end_date.isoformat() if end_date else "",
                tuple(sorted((metric, round(value, 6)) for metric, value in metric_values.items())),
            )
            if record_key in seen_records:
                continue
            seen_records.add(record_key)

            doc_title = str(
                _best_of(record, frame, title_keys, prefer_hxl=prefer_hxl)
                or source.get("name")
                or "EM-DAT event"
            )
            source_url = str(
                _best_of(record, frame, source_url_keys, prefer_hxl=prefer_hxl)
                or source.get("url")
                or ""
            )
            publication_date = _normalise_date_value(
                _best_of(record, frame, publication_keys, prefer_hxl=prefer_hxl)
            )

            for metric, value in metric_values.items():
                allocations = _allocation(value, start_date, end_date, policy=policy)
                if not allocations:
                    continue
                definition_text = (
                    f"EM-DAT reported {metric.replace('_', ' ')} persons ({policy} allocation)."
                )
                method = f"EM-DAT {policy} allocation"
                for month, amount in allocations.items():
                    if amount is None or amount <= 0:
                        continue
                    key = (iso3, hazard_code, metric, month)
                    entry = aggregates.get(key)
                    if not entry:
                        entry = {
                            "country_name": country_names.get(iso3, iso3),
                            "hazard_code": hazard_code,
                            "hazard_label": hazard_label,
                            "hazard_class": hazard_class,
                            "metric": metric,
                            "series_semantics": "new",
                            "semantics": "new",
                            "unit": "persons",
                            "publisher": publisher,
                            "source_type": source_type,
                            "source_url": source_url,
                            "doc_title": doc_title,
                            "definition_text": definition_text,
                            "method": method,
                            "publication_date": publication_date,
                            "value": 0.0,
                        }
                        aggregates[key] = entry
                    else:
                        if not entry["source_url"] and source_url:
                            entry["source_url"] = source_url
                        if not entry["doc_title"] and doc_title:
                            entry["doc_title"] = doc_title
                        if not entry["publication_date"] and publication_date:
                            entry["publication_date"] = publication_date
                    entry["value"] += float(amount)
                    raw_ids_by_key[key].add(raw_id)

    rows: List[Dict[str, Any]] = []
    for key, entry in sorted(aggregates.items()):
        iso3, hazard_code, metric, month = key
        value = entry.get("value", 0.0)
        if value is None or value <= 0:
            continue
        value_int = int(round(float(value)))
        digest = stable_digest([
            iso3,
            hazard_code,
            metric,
            month,
            ",".join(sorted(raw_ids_by_key.get(key, {""}))),
        ])
        event_id = f"{iso3}-EMDAT-{hazard_code}-{metric}-{month.replace('-', '')}-{digest}"
        rows.append(
            {
                "event_id": event_id,
                "country_name": entry.get("country_name", ""),
                "iso3": iso3,
                "ym": month,
                "hazard_code": hazard_code,
                "hazard_label": entry.get("hazard_label", ""),
                "hazard_class": entry.get("hazard_class", ""),
                "metric": metric,
                "series_semantics": entry.get("series_semantics", "new"),
                "semantics": entry.get("semantics", entry.get("series_semantics", "new")),
                "value": value_int,
                "unit": entry.get("unit", "persons"),
                "as_of_date": month,
                "source_id": "emdat",
                "publication_date": entry.get("publication_date", ""),
                "publisher": entry.get("publisher", ""),
                "source_type": entry.get("source_type", ""),
                "source_url": entry.get("source_url", ""),
                "doc_title": entry.get("doc_title", ""),
                "definition_text": entry.get("definition_text", ""),
                "method": entry.get("method", ""),
                "confidence": "",
                "revision": 0,
                "ingested_at": ingested_at,
            }
        )

    rows.sort(key=lambda row: (row.get("iso3", ""), row.get("as_of_date", ""), row.get("metric", "")))
    return rows


def _write_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        ensure_header_only()
        return
    df = pd.DataFrame(rows)
    for column in CANONICAL_HEADERS:
        if column not in df.columns:
            df[column] = pd.NA
    df = df[CANONICAL_HEADERS]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    ensure_manifest_for_csv(
        OUT_PATH,
        schema_version=_emdat_schema_version(),
        source_id="emdat",
    )


def _write_facts_frame(frame: pd.DataFrame | None) -> int:
    if frame is None or frame.empty:
        ensure_header_only()
        return 0
    prepared = frame.copy()
    for column in CANONICAL_HEADERS:
        if column not in prepared.columns:
            prepared[column] = pd.NA
    prepared = prepared[CANONICAL_HEADERS]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(OUT_PATH, index=False)
    ensure_manifest_for_csv(
        OUT_PATH,
        schema_version=_emdat_schema_version(),
        source_id="emdat",
    )
    return len(prepared)


API_LOG = logging.getLogger("resolver.ingestion.emdat.api")

EMDAT_API_URL = "https://api.emdat.be/v1"
EMDAT_API_KEY_ENV = "EMDAT_API_KEY"
EMDAT_NETWORK_ENV = "EMDAT_NETWORK"
EMDAT_SOURCE_ENV = "EMDAT_SOURCE"
REQUEST_TIMEOUT = (5, 30)
RETRY_STATUS_FORCELIST = (429, 500, 502, 503, 504)

FETCH_COLUMNS: Sequence[str] = (
    "disno",
    "classif_key",
    "type",
    "subtype",
    "iso",
    "country",
    "start_year",
    "start_month",
    "start_day",
    "end_year",
    "end_month",
    "end_day",
    "total_affected",
    "entry_date",
    "last_update",
)

NUMERIC_COLUMNS: Sequence[str] = (
    "start_year",
    "start_month",
    "start_day",
    "end_year",
    "end_month",
    "end_day",
    "total_affected",
)

DEFAULT_CLASSIF_KEYS: Sequence[str] = (
    "nat-cli-dro-dro",
    "nat-met-sto-tro",
    "nat-hyd-flo-riv",
    "nat-hyd-flo-fla",
)


class OfflineRequested(RuntimeError):
    """Raised when a live EM-DAT request was explicitly disabled."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _build_retry_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        status=2,
        backoff_factor=0.3,
        status_forcelist=RETRY_STATUS_FORCELIST,
        allowed_methods=("POST",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _normalise_iso(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    normalised: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            normalised.append(text.upper())
    return normalised


def _normalise_text(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    normalised: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            normalised.append(text)
    return normalised


def _iso_from_disno(disno: Any) -> str:
    text = str(disno or "").strip().upper()
    if not text or "-" not in text:
        return ""
    suffix = text.rsplit("-", 1)[-1]
    if len(suffix) == 3 and suffix.isalpha():
        return suffix
    return ""


def _coerce_int(value: Any, default: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return coerced


def _default_config() -> Dict[str, Any]:
    cfg = load_config()
    return cfg if isinstance(cfg, dict) else {}


def _default_classif(cfg: Mapping[str, Any]) -> list[str]:
    classif = _normalise_text(cfg.get("classif_keys"))
    return classif or list(DEFAULT_CLASSIF_KEYS)


def _default_iso(cfg: Mapping[str, Any]) -> list[str]:
    iso_values = cfg.get("iso")
    if isinstance(iso_values, (list, tuple)):
        return _normalise_iso(iso_values)
    return []


def _default_year_bounds(cfg: Mapping[str, Any]) -> tuple[int, int]:
    current_year = date.today().year
    default_from = _coerce_int(cfg.get("default_from_year"), current_year)
    default_to = _coerce_int(cfg.get("default_to_year"), current_year)
    if default_from > default_to:
        default_from = default_to
    return default_from, default_to


def _build_effective_params(
    *,
    network_requested: bool,
    api_key_present: bool,
    cfg: Mapping[str, Any],
    source_mode: str,
    network_env: str | None = None,
    source_override: str | None = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Return diagnostics payload and filter metadata for the current run."""

    from_year, to_year = _default_year_bounds(cfg)
    include_hist = bool(cfg.get("include_hist", False))
    classif_filter = _default_classif(cfg)
    iso_filter = sorted({value for value in _default_iso(cfg) if value})

    filters: Dict[str, Any] = {
        "from": from_year,
        "to": to_year,
        "classif": list(classif_filter),
        "include_hist": include_hist,
    }
    if iso_filter:
        filters["iso"] = iso_filter

    params = {
        "recorded_at": _current_timestamp(),
        "source_type": source_mode,
        "source_override": source_override or "",
        "network": bool(network_requested),
        "network_env": (network_env or ""),
        "api_key_present": bool(api_key_present),
        "default_from_year": from_year,
        "default_to_year": to_year,
        "include_hist": include_hist,
        "classif_count": len(classif_filter),
        "classif_keys": list(classif_filter),
        "iso_filter_applied": bool(iso_filter),
        "iso_count": len(iso_filter),
        "filters": filters,
    }
    if iso_filter:
        params["iso_values"] = iso_filter

    meta = {
        "from": from_year,
        "to": to_year,
        "include_hist": include_hist,
        "classif": list(classif_filter),
        "iso": list(iso_filter),
    }
    return params, meta


def _apply_include_hist_flag(query: str, include_hist: bool) -> str:
    if not include_hist:
        return query
    return query.replace("include_hist: false", "include_hist: true")


def _build_pa_payload(meta: Mapping[str, Any], *, limit: Optional[int] = None) -> Dict[str, Any]:
    include_hist = bool(meta.get("include_hist", False))
    query = apply_limit_override(
        _apply_include_hist_flag(EMDAT_PA_QUERY, include_hist),
        limit=limit,
    )

    variables: Dict[str, Any] = {
        "from": _coerce_int(meta.get("from"), meta.get("from", 0)),
        "to": _coerce_int(meta.get("to"), meta.get("to", 0)),
        "classif": [str(value) for value in meta.get("classif", []) if str(value)],
    }

    iso_values = [str(value).upper() for value in meta.get("iso", []) if str(value).strip()]
    if iso_values:
        variables["iso"] = iso_values

    return {"query": query, "variables": variables}


def _summarize_classif_histogram(frame: pd.DataFrame) -> list[Dict[str, Any]]:
    if frame.empty or "classif_key" not in frame.columns:
        return []
    histogram: list[Dict[str, Any]] = []
    counts = (
        frame["classif_key"].fillna("").astype(str).value_counts()
    )
    for key in sorted(counts.index):
        histogram.append({"classif_key": key, "count": int(counts[key])})
    return histogram


def _run_widened_probe(
    client: "EmdatClient",
    *,
    meta: Mapping[str, Any],
) -> Dict[str, Any]:
    to_year = _coerce_int(meta.get("to"), meta.get("to", 0))
    from_year = max(to_year - 1, 0)
    diag_meta = {
        "from": from_year,
        "to": to_year,
        "classif": list(meta.get("classif", [])),
        "iso": [],
        "include_hist": meta.get("include_hist", False),
    }

    payload = _build_pa_payload(diag_meta, limit=EMDAT_PROBE_SAMPLE_LIMIT)

    try:
        status_code, parsed, elapsed_ms = client._post_with_status(payload)
    except Exception as exc:  # pragma: no cover - defensive network errors
        return {
            "ok": False,
            "error": str(exc),
            "filters": diag_meta,
            "recorded_at": _current_timestamp(),
        }

    frame = client._frame_from_payload(parsed)
    data = parsed.get("data") if isinstance(parsed, Mapping) else {}
    if not isinstance(data, Mapping):
        data = {}
    public = data.get("public_emdat")
    if not isinstance(public, Mapping):
        public = {}
    info = public.get("info")
    if not isinstance(info, Mapping):
        info = {}

    result = {
        "ok": True,
        "http_status": status_code,
        "elapsed_ms": elapsed_ms,
        "rows": int(len(frame)),
        "total_available": public.get("total_available"),
        "info": {"version": info.get("version"), "timestamp": info.get("timestamp")},
        "filters": diag_meta,
        "classif_histogram": _summarize_classif_histogram(frame),
        "recorded_at": _current_timestamp(),
    }
    return result


class EmdatClient:
    """Thin GraphQL client for the EM-DAT public API."""

    def __init__(
        self,
        *,
        network: bool | None = None,
        api_key: Optional[str] = None,
        session: Optional[requests.Session] = None,
        base_url: str = EMDAT_API_URL,
        timeout: tuple[int, int] = REQUEST_TIMEOUT,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._network = (
            network
            if network is not None
            else getenv_bool(EMDAT_NETWORK_ENV, default=False)
        )
        self._api_key = (api_key or os.getenv(EMDAT_API_KEY_ENV, "")).strip()
        self.base_url = base_url
        self.timeout = timeout
        self.session = session or _build_retry_session()
        self.log = logger or API_LOG

    @property
    def headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = self._api_key
        return headers

    def _ensure_online(self) -> None:
        if not self._network:
            raise OfflineRequested("network access disabled; pass --network to enable")
        if not self._api_key:
            raise OfflineRequested(f"missing {EMDAT_API_KEY_ENV} environment variable")

    def _post_with_status(
        self, payload: Dict[str, Any]
    ) -> tuple[Optional[int], Dict[str, Any], float]:
        self._ensure_online()
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.log.debug("emdat.request.start|payload_bytes=%s", len(encoded))
        start = time.perf_counter()
        try:
            response = self.session.post(
                self.base_url,
                data=encoded,
                headers=self.headers,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:  # pragma: no cover - defensive
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.log.debug("emdat.request.error|elapsed_ms=%.2f|error=%s", elapsed_ms, exc)
            raise
        elapsed_ms = (time.perf_counter() - start) * 1000
        status_code = getattr(response, "status_code", None)
        self.log.debug(
            "emdat.request.done|status=%s|elapsed_ms=%.2f",
            status_code if status_code is not None else "?",
            elapsed_ms,
        )
        response.raise_for_status()
        try:
            parsed = response.json()
        except ValueError as exc:
            raise RuntimeError("failed to decode EM-DAT response as JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("unexpected EM-DAT response payload type")
        if parsed.get("errors"):
            raise RuntimeError(f"EM-DAT returned errors: {parsed['errors']}")
        return status_code if isinstance(status_code, int) else None, parsed, elapsed_ms

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        _, parsed, _ = self._post_with_status(payload)
        return parsed

    def probe(
        self,
        *,
        from_year: Optional[int] = None,
        to_year: Optional[int] = None,
        iso: Sequence[str] | None = None,
        classif: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        cfg = _default_config()
        default_from, default_to = _default_year_bounds(cfg)
        probe_to = _coerce_int(to_year, default_to)
        probe_from = _coerce_int(from_year, probe_to)
        classif_filter = _normalise_text(classif) or _default_classif(cfg)
        iso_filter = _normalise_iso(iso) or _default_iso(cfg)

        include_hist = bool(cfg.get("include_hist", False))
        query = _apply_include_hist_flag(EMDAT_METADATA_QUERY, include_hist=include_hist)
        variables: Dict[str, Any] = {
            "from": probe_from,
            "to": probe_to,
            "classif": classif_filter,
        }
        if iso_filter:
            variables["iso"] = iso_filter
        payload = {"query": query, "variables": variables}

        filters_payload = {
            "iso": list(iso_filter) if iso_filter else [],
            "from": probe_from,
            "to": probe_to,
            "classif": list(classif_filter),
            "include_hist": include_hist,
        }
        recorded_at = _current_timestamp()

        try:
            status_code, parsed, elapsed_ms = self._post_with_status(payload)
        except OfflineRequested as offline:
            reason = str(offline)
            self.log.info("emdat.probe.fail|reason=%s", reason)
            result = {
                "ok": False,
                "status": "skipped",
                "latency_ms": None,
                "elapsed_ms": None,
                "api_version": None,
                "table_version": None,
                "metadata_timestamp": None,
                "error": reason,
                "requests": {"total": 0, "2xx": 0, "4xx": 0, "5xx": 0},
                "total_available": None,
                "recorded_at": recorded_at,
                "ts": recorded_at,
                "filters": filters_payload,
                "network": bool(self._network),
                "api_key_present": bool(self._api_key),
                "source_type": "api",
            }
            write_json(EMDAT_PROBE_PATH, result)
            return result
        except Exception as exc:  # pragma: no cover - defensive logging
            reason = str(exc)
            self.log.info("emdat.probe.fail|reason=%s", reason)
            result = {
                "ok": False,
                "status": "error",
                "latency_ms": None,
                "elapsed_ms": None,
                "api_version": None,
                "table_version": None,
                "metadata_timestamp": None,
                "error": reason,
                "requests": {"total": 0, "2xx": 0, "4xx": 0, "5xx": 0},
                "total_available": None,
                "recorded_at": recorded_at,
                "ts": recorded_at,
                "filters": filters_payload,
                "network": bool(self._network),
                "api_key_present": bool(self._api_key),
                "source_type": "api",
            }
            write_json(EMDAT_PROBE_PATH, result)
            return result

        data = parsed.get("data") or {}
        api_version = data.get("api_version")
        public = data.get("public_emdat") or {}
        info = public.get("info") or {}
        total_available = public.get("total_available")
        requests_summary = {"total": 0, "2xx": 0, "4xx": 0, "5xx": 0}
        if isinstance(status_code, int):
            requests_summary["total"] = 1
            if 200 <= status_code < 300:
                requests_summary["2xx"] = 1
            elif 400 <= status_code < 500:
                requests_summary["4xx"] = 1
            elif 500 <= status_code < 600:
                requests_summary["5xx"] = 1
        result = {
            "ok": True,
            "status": status_code,
            "latency_ms": int(round(elapsed_ms)),
            "elapsed_ms": int(round(elapsed_ms)),
            "api_version": api_version,
            "table_version": info.get("version"),
            "metadata_timestamp": info.get("timestamp"),
            "error": None,
            "requests": requests_summary,
            "total_available": total_available,
            "recorded_at": recorded_at,
            "ts": recorded_at,
            "filters": filters_payload,
            "network": bool(self._network),
            "api_key_present": bool(self._api_key),
            "source_type": "api",
        }
        write_json(EMDAT_PROBE_PATH, result)
        self.log.info(
            "emdat.probe.ok|api_version=%s|status=%s",
            api_version,
            result["status"],
        )
        return result

    def fetch_raw(
        self,
        from_year: int,
        to_year: int,
        *,
        iso: Sequence[str] | None = None,
        classif: Sequence[str] | None = None,
        include_hist: bool = False,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        cfg = _default_config()
        classif_filter = _normalise_text(classif) or _default_classif(cfg)
        iso_filter = _normalise_iso(iso) or _default_iso(cfg)

        query = apply_limit_override(
            _apply_include_hist_flag(EMDAT_PA_QUERY, include_hist or cfg.get("include_hist", False)),
            limit=limit,
        )
        variables: Dict[str, Any] = {
            "from": _coerce_int(from_year, from_year),
            "to": _coerce_int(to_year, to_year),
            "classif": classif_filter,
        }
        if iso_filter:
            variables["iso"] = iso_filter
        payload = {"query": query, "variables": variables}

        self.log.debug(
            "emdat.fetch.start|from=%s|to=%s|iso=%s|classif=%s|limit=%s",
            payload["variables"]["from"],
            payload["variables"]["to"],
            len(iso_filter),
            len(classif_filter),
            limit if limit is not None else -1,
        )
        parsed = self._post(payload)
        frame = self._frame_from_payload(parsed)
        self.log.debug("emdat.fetch.finish|rows=%s", len(frame))
        return frame

    def _frame_from_payload(self, payload: Mapping[str, Any]) -> pd.DataFrame:
        data = payload.get("data") or {}
        public = data.get("public_emdat") or {}
        records = public.get("data") or []
        if not isinstance(records, list):
            records = []
        frame = pd.DataFrame(records)
        if frame.empty:
            return pd.DataFrame(columns=list(FETCH_COLUMNS))

        for column in FETCH_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.NA
        frame = frame[list(FETCH_COLUMNS)]

        frame["iso"] = frame["iso"].fillna("").astype(str)
        missing_iso = frame["iso"].str.strip() == ""
        for idx in frame.index[missing_iso]:
            derived = _iso_from_disno(frame.at[idx, "disno"])
            if derived:
                frame.at[idx, "iso"] = derived
                self.log.debug(
                    "emdat.fetch.iso_from_disno|disno=%s|iso=%s",
                    frame.at[idx, "disno"],
                    derived,
                )
        frame["iso"] = frame["iso"].str.strip().str.upper()

        for column in NUMERIC_COLUMNS:
            if column in frame.columns:
                series = pd.to_numeric(frame[column], errors="coerce")
                if column == "total_affected":
                    frame[column] = series
                else:
                    frame[column] = series.astype("Int64")

        return frame.reset_index(drop=True)


def main(argv: List[str] | None = None) -> bool:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if _env_bool("RESOLVER_SKIP_EMDAT"):
        LOG.info("emdat: skipped via RESOLVER_SKIP_EMDAT")
        ensure_header_only()
        return False

    cfg = load_config()
    http_tally: Dict[str, int] = {"total": 0, "2xx": 0, "4xx": 0, "5xx": 0}
    raw_row_count: Optional[int] = None
    normalized_row_count: Optional[int] = None
    facts_frame: Optional[pd.DataFrame] = None
    written_row_count: Optional[int] = None

    def _record_http_status(status: Optional[int]) -> None:
        if status is None:
            return
        if 200 <= status < 300:
            http_tally["2xx"] += 1
        elif 400 <= status < 500:
            http_tally["4xx"] += 1
        elif 500 <= status < 600:
            http_tally["5xx"] += 1
        http_tally["total"] += 1

    def _merge_http_counts(summary: Mapping[str, Any] | None) -> None:
        if not isinstance(summary, Mapping):
            return
        for bucket in ("total", "2xx", "4xx", "5xx"):
            value = summary.get(bucket)
            if isinstance(value, int):
                http_tally[bucket] += value

    def _http_payload() -> Dict[str, int]:
        return {bucket: http_tally[bucket] for bucket in ("total", "2xx", "4xx", "5xx")}

    def _counts_payload(written: Optional[int]) -> Dict[str, int]:
        fetched = raw_row_count if raw_row_count is not None else 0
        normalized = normalized_row_count if normalized_row_count is not None else 0
        written_count = written if written is not None else 0
        return {
            "fetched": int(fetched),
            "normalized": int(normalized),
            "written": int(written_count),
        }

    network_env_raw = os.getenv(EMDAT_NETWORK_ENV, "")
    network_on = getenv_bool(EMDAT_NETWORK_ENV, False)
    api_key = (os.getenv(EMDAT_API_KEY_ENV, "") or "").strip()
    force_source = (os.getenv(EMDAT_SOURCE_ENV, "") or "").strip().lower()

    source_cfg = cfg.get("source") if isinstance(cfg.get("source"), Mapping) else {}
    src_type_raw = source_cfg.get("type") if isinstance(source_cfg, Mapping) else None
    src_type = str(src_type_raw or "").strip().lower()
    if not src_type:
        sources_cfg = cfg.get("sources")
        if sources_cfg:
            src_type = "file"

    if force_source == "api":
        if network_on and api_key:
            src_type = "api"
        else:
            LOG.info(
                "emdat.mode.override_skipped|force=%s|network=%s|key=%s",
                force_source,
                1 if network_on else 0,
                1 if api_key else 0,
            )

    if (
        src_type == "file"
        and isinstance(source_cfg, Mapping)
        and not str(source_cfg.get("path", "")).strip()
        and network_on
        and api_key
    ):
        LOG.warning(
            "emdat.mode|yaml=FILE but no path and network+key present; promoting to API mode"
        )
        src_type = "api"

    mode = "file" if src_type == "file" else "api"

    LOG.debug(
        "emdat.effective",
        extra={
            "source_type": mode,
            "network": bool(network_on),
            "api_key_present": bool(api_key),
        },
    )

    effective_params, filter_meta = _build_effective_params(
        network_requested=network_on,
        api_key_present=bool(api_key),
        cfg=cfg,
        source_mode=mode,
        network_env=network_env_raw,
        source_override=force_source or None,
    )
    write_json(EMDAT_EFFECTIVE_PARAMS_PATH, effective_params)
    LOG.info(
        "emdat.effective|source=%s|network=%s|key_present=%s|from=%s|to=%s|classif_count=%s|iso_count=%s",
        mode,
        1 if network_on else 0,
        1 if api_key else 0,
        filter_meta["from"],
        filter_meta["to"],
        len(filter_meta["classif"]),
        len([value for value in filter_meta["iso"] if value]),
    )

    if mode != "file" and not network_on:
        probe_payload = {
            "ok": False,
            "status": "skipped",
            "latency_ms": None,
            "elapsed_ms": None,
            "api_version": None,
            "error": "offline mode (EMDAT_NETWORK != '1')",
            "table_version": None,
            "metadata_timestamp": None,
            "requests": {"total": 0, "2xx": 0, "4xx": 0, "5xx": 0},
            "network": bool(network_on),
            "api_key_present": bool(api_key),
            "source_type": mode,
            "ts": _current_timestamp(),
            "recorded_at": _current_timestamp(),
        }
        write_json(EMDAT_PROBE_PATH, probe_payload)
        LOG.info(
            "emdat: EMDAT_NETWORK != '1' → offline mode; emitting schema header only"
        )
        _write_emdat_header_only_csv(OUT_PATH)
        diagnostics_ctx = diagnostics_emitter.start_run("emdat_client", mode="offline")
        diagnostics_emitter.finalize_run(
            diagnostics_ctx,
            status="ok",
            reason="offline-no-data",
            http=_http_payload(),
            counts=_counts_payload(0),
        )
        return True

    enabled_flag = cfg.get("enabled")
    if enabled_flag is None:
        enabled = bool(cfg.get("sources"))
    else:
        enabled = bool(enabled_flag)
    if not enabled:
        LOG.info("emdat: disabled via config; writing header only")
        ensure_header_only()
        return False

    preflight_failed = False
    preflight_reason: Optional[str] = None
    client: EmdatClient | None = None

    if mode == "api":
        if not api_key:
            preflight_failed = True
            preflight_reason = "missing_key"
            probe_payload = {
                "ok": False,
                "status": "missing-key",
                "latency_ms": None,
                "elapsed_ms": None,
                "api_version": None,
                "error": "missing API key",
                "table_version": None,
                "metadata_timestamp": None,
                "requests": {"total": 0, "2xx": 0, "4xx": 0, "5xx": 0},
                "network": bool(network_on),
                "api_key_present": False,
                "source_type": mode,
                "ts": _current_timestamp(),
                "recorded_at": _current_timestamp(),
            }
            write_json(EMDAT_PROBE_PATH, probe_payload)
            _merge_http_counts(probe_payload.get("requests"))
            LOG.info("emdat.offline|reason=missing_key")
        else:
            client = EmdatClient(network=True, api_key=api_key)
            reachability = probe_emdat(
                api_key,
                client.base_url,
                timeout_s=REQUEST_TIMEOUT[0],
                session=client.session,
            )
            probe_payload = {
                "ok": reachability["ok"],
                "status": reachability.get("status")
                if reachability.get("status") is not None
                else ("error" if reachability.get("error") else None),
                "latency_ms": reachability.get("latency_ms"),
                "elapsed_ms": reachability.get("elapsed_ms"),
                "api_version": reachability["api_version"],
                "error": reachability["error"],
                "table_version": reachability["table_version"],
                "metadata_timestamp": reachability["metadata_timestamp"],
                "requests": reachability.get(
                    "requests", {"total": 0, "2xx": 0, "4xx": 0, "5xx": 0}
                ),
                "network": bool(network_on),
                "api_key_present": bool(api_key),
                "source_type": mode,
                "ts": _current_timestamp(),
                "recorded_at": _current_timestamp(),
            }
            write_json(EMDAT_PROBE_PATH, probe_payload)
            _merge_http_counts(probe_payload.get("requests"))
            if reachability["ok"]:
                LOG.info(
                    "emdat.probe.ok|status=%s|ms=%s|api_version=%s|table=%s",
                    probe_payload["status"],
                    probe_payload["latency_ms"],
                    probe_payload["api_version"],
                    probe_payload["table_version"],
                )
            else:
                LOG.info(
                    "emdat.probe.fail|status=%s|ms=%s|api_version=%s|table=%s|error=%s",
                    probe_payload["status"],
                    probe_payload["latency_ms"],
                    probe_payload["api_version"],
                    probe_payload["table_version"],
                    probe_payload.get("error"),
                )
            try:
                status_code, parsed, elapsed_ms = client._post_with_status(
                    _build_pa_payload(filter_meta)
                )
                _record_http_status(status_code)
            except OfflineRequested as exc:
                LOG.info("emdat.offline|reason=%s", exc)
                preflight_failed = True
                preflight_reason = str(exc) or "offline-requested"
            except Exception as exc:  # pragma: no cover - defensive logging
                LOG.info("emdat.fetch.error|error=%s", exc)
                preflight_failed = True
                preflight_reason = "fetch-error"
            else:
                frame = client._frame_from_payload(parsed)
                row_count = len(frame)
                raw_row_count = row_count
                info_payload: Mapping[str, Any] | None = None
                if isinstance(parsed, Mapping):
                    data = parsed.get("data")
                    if isinstance(data, Mapping):
                        public = data.get("public_emdat")
                        if isinstance(public, Mapping):
                            potential_info = public.get("info")
                            if isinstance(potential_info, Mapping):
                                info_payload = potential_info
                if row_count:
                    from resolver.ingestion.emdat_normalize import normalize_emdat_pa

                    try:
                        normalized_frame = normalize_emdat_pa(
                            frame,
                            info=info_payload,
                        )
                    except Exception:  # pragma: no cover - defensive normalization
                        normalized_row_count = None
                    else:
                        facts_frame = normalized_frame
                        stats = normalized_frame.attrs.get("normalize_stats")
                        if isinstance(stats, Mapping) and "kept_rows" in stats:
                            try:
                                normalized_row_count = int(stats.get("kept_rows") or 0)
                            except (TypeError, ValueError):
                                normalized_row_count = int(len(normalized_frame))
                        else:
                            normalized_row_count = int(len(normalized_frame))
                else:
                    normalized_row_count = 0
                if row_count == 0:
                    LOG.info(
                        "emdat.fetch.ok.zero|from=%s|to=%s|iso=%s",
                        filter_meta["from"],
                        filter_meta["to"],
                        len([value for value in filter_meta.get("iso", []) if value]),
                    )
                    if status_code is not None and 200 <= status_code < 300:
                        probe_sample = _run_widened_probe(client, meta=filter_meta)
                        _record_http_status(probe_sample.get("http_status"))
                        write_json(EMDAT_PROBE_SAMPLE_PATH, probe_sample)
                else:
                    LOG.info(
                        "emdat.fetch.ok|rows=%s|status=%s|elapsed_ms=%.2f",
                        row_count,
                        status_code,
                        elapsed_ms,
                    )

    diagnostics_mode = "live" if (mode == "file" or network_on) else "offline"
    diagnostics_ctx = diagnostics_emitter.start_run("emdat_client", mode=diagnostics_mode)
    if preflight_failed:
        ensure_header_only()
        status = "ok" if preflight_reason == "missing_key" else "error"
        diagnostics_emitter.finalize_run(
            diagnostics_ctx,
            status=status,
            reason=preflight_reason or "preflight-failed",
            http=_http_payload(),
            counts=_counts_payload(0),
        )
        return status == "ok"

    if facts_frame is not None and not facts_frame.empty:
        written_row_count = _write_facts_frame(facts_frame)
        LOG.info("emdat: wrote %s rows", written_row_count)
        diagnostics_emitter.finalize_run(
            diagnostics_ctx,
            status="ok",
            reason=None,
            http=_http_payload(),
            counts=_counts_payload(written_row_count),
        )
        return True

    try:
        rows = collect_rows(cfg)
    except Exception as exc:  # pragma: no cover - defensive logging for tests
        LOG.info("emdat: failed to collect rows: %s", exc)
        ensure_header_only()
        diagnostics_emitter.finalize_run(
            diagnostics_ctx,
            status="error",
            reason="collect-failed",
            http=_http_payload(),
            counts=_counts_payload(0),
        )
        return False

    if raw_row_count is None:
        raw_row_count = len(rows)
    if normalized_row_count is None:
        normalized_row_count = len(rows)

    if not rows:
        LOG.info("emdat: no rows collected; writing header only")
        ensure_header_only()
        empty_ok = getenv_bool("EMPTY_POLICY", True)
        if not empty_ok:
            raw_policy = os.getenv("EMPTY_POLICY", "")
            empty_ok = raw_policy.strip().lower() in {"1", "true", "yes"}
        status = "ok" if empty_ok else "no-data"
        diagnostics_emitter.finalize_run(
            diagnostics_ctx,
            status=status,
            reason="no-data",
            http=_http_payload(),
            counts=_counts_payload(0),
        )
        return bool(empty_ok)

    _write_rows(rows)
    written_row_count = len(rows)
    LOG.info("emdat: wrote %s rows", written_row_count)
    diagnostics_emitter.finalize_run(
        diagnostics_ctx,
        status="ok",
        reason=None,
        http=_http_payload(),
        counts=_counts_payload(written_row_count),
    )
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
