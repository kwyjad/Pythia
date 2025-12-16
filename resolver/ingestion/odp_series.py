# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Normalization helpers for UNHCR ODP JSON widgets."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
import calendar
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Sequence

import pandas as pd
import requests
import yaml

from resolver.common.logs import dict_counts, df_schema, get_logger
from resolver.ingestion import odp_discovery
from resolver.ingestion.utils.month_bucket import month_start

LOGGER = get_logger(__name__)
_CANONICAL_COLUMNS = [
    "source_id",
    "iso3",
    "origin_iso3",
    "admin_name",
    "ym",
    "as_of_date",
    "metric",
    "series_semantics",
    "value",
    "unit",
    "extra",
]
_DEDUP_KEYS = ["source_id", "iso3", "origin_iso3", "admin_name", "ym", "metric", "value"]
_DEFAULT_NORMALIZER_PATH = Path(__file__).resolve().parent / "config" / "odp_normalizers.yml"


@dataclass
class NormalizerSpec:
    """Configuration for converting a widget payload into canonical rows."""

    id: str
    label_regex: re.Pattern[str]
    metric: str
    series_semantics: str
    frequency: str
    value_field: str
    date_field: str
    date_format: str | None
    iso3_field: str | None
    origin_iso3_field: str | None
    admin_name_field: str | None
    required_fields: list[str]
    unit: str


@dataclass
class OdpPipelineStats:
    """Structured counters for the ODP discovery/normalization pipeline."""

    # Config-level
    config_pages: int = 0

    # Discovery-level
    pages_discovered: int = 0
    json_links_found: int = 0

    # Normalization-level
    json_links_matched: int = 0
    json_links_unmatched: int = 0
    raw_records_total: int = 0
    normalized_rows_total: int = 0
    normalized_rows_per_series: Dict[str, int] = field(default_factory=dict)

    # Debugging aids
    unmatched_labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _month_end(value: date) -> date:
    """Return the last day of the month for ``value``."""

    last_day = calendar.monthrange(value.year, value.month)[1]
    return value.replace(day=last_day)


def _ensure_frame_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in _CANONICAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[_CANONICAL_COLUMNS]


def _parse_date(text: Any, *, fmt: str | None) -> date | None:
    if text in (None, ""):
        return None
    if isinstance(text, date) and not isinstance(text, datetime):
        return text
    if isinstance(text, datetime):
        return text.date()
    text_str = str(text).strip()
    if not text_str:
        return None
    try:
        if fmt:
            return datetime.strptime(text_str, fmt).date()
        parsed = datetime.fromisoformat(text_str.replace("Z", "+00:00"))
        return parsed.date()
    except ValueError:
        parsed = month_start(text_str)
        return parsed


def iter_records(payload: Any) -> Iterable[Mapping[str, Any]]:
    """Yield dictionary-like records from a best-effort payload."""

    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, Mapping):
                yield entry
        return
    if isinstance(payload, Mapping):
        features = payload.get("features")
        if isinstance(features, Sequence):
            for feature in features:
                attrs = None
                if isinstance(feature, Mapping):
                    attrs = feature.get("attributes")
                    if not isinstance(attrs, Mapping):
                        attrs = feature.get("properties")
                if isinstance(attrs, Mapping):
                    yield attrs
            return
        records = payload.get("records")
        if isinstance(records, Sequence):
            for record in records:
                if isinstance(record, Mapping):
                    yield record
            return
        if payload:
            yield payload
        return
    LOGGER.debug("ODP payload not iterable", extra={"type": type(payload).__name__})
    return


def load_normalizer_config(path: str | Path | None = None) -> list[NormalizerSpec]:
    """Read and parse the normalizer YAML config."""

    config_path = Path(path) if path else _DEFAULT_NORMALIZER_PATH
    if not config_path.exists():
        LOGGER.warning("ODP normalizer config missing", extra={"path": str(config_path)})
        return []
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    defaults: MutableMapping[str, Any] = raw.get("defaults", {}) or {}
    specs_cfg: Sequence[Mapping[str, Any]] = raw.get("series", []) or []
    specs: list[NormalizerSpec] = []
    for entry in specs_cfg:
        regex_text = entry.get("label_regex")
        if not regex_text:
            continue
        try:
            pattern = re.compile(regex_text, re.IGNORECASE)
        except re.error:
            LOGGER.warning("Invalid ODP label regex", extra={"pattern": regex_text})
            continue
        semantics_value = entry.get("series_semantics") or defaults.get("series_semantics") or "stock"
        frequency_value = entry.get("frequency") or defaults.get("frequency") or "monthly"
        spec = NormalizerSpec(
            id=str(entry.get("id") or entry.get("metric") or spec_id_label(entry)),
            label_regex=pattern,
            metric=str(entry.get("metric") or defaults.get("metric") or spec_id_label(entry)),
            series_semantics=str(semantics_value).strip().lower() or "stock",
            frequency=str(frequency_value).strip().lower() or "monthly",
            value_field=str(entry.get("value_field") or defaults.get("value_field") or "value"),
            date_field=str(entry.get("date_field") or defaults.get("date_field") or "date"),
            date_format=entry.get("date_format") or defaults.get("date_format"),
            iso3_field=entry.get("iso3_field") or defaults.get("iso3_field"),
            origin_iso3_field=entry.get("origin_iso3_field") or defaults.get("origin_iso3_field"),
            admin_name_field=entry.get("admin_name_field") or defaults.get("admin_name_field"),
            required_fields=list(entry.get("required_fields") or defaults.get("required_fields") or []),
            unit=str(entry.get("unit") or defaults.get("unit") or "persons"),
        )
        specs.append(spec)
    return specs


def spec_id_label(entry: Mapping[str, Any]) -> str:
    label = entry.get("label_regex") or entry.get("metric") or "unknown"
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(label)).lower().strip("_") or "series"


def match_spec(
    label: str,
    sample_record: Mapping[str, Any] | None,
    specs: Sequence[NormalizerSpec],
) -> NormalizerSpec | None:
    if not label or sample_record is None:
        return None
    for spec in specs:
        if not spec.label_regex.search(label or ""):
            continue
        if any(field not in sample_record for field in spec.required_fields):
            continue
        return spec
    return None


def _record_value(record: Mapping[str, Any], field: str) -> float | None:
    value = record.get(field)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _string_or_none(record: Mapping[str, Any], field: str | None) -> str | None:
    if not field:
        return None
    value = record.get(field)
    if value is None:
        return None
    value_str = str(value).strip()
    return value_str or None


def normalize_endpoint(
    link: odp_discovery.DiscoveredLink,
    payload: Any,
    specs: Sequence[NormalizerSpec],
    *,
    page_url: str | None = None,
    today: date | None = None,
    stats: OdpPipelineStats | None = None,
) -> pd.DataFrame:
    del today  # reserved for future use
    records = list(iter_records(payload))
    if stats is not None:
        stats.raw_records_total += len(records)
    if not records:
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)
    label = getattr(link, "text", None) or getattr(link, "label", "") or ""
    spec = match_spec(label, records[0], specs)
    if not spec:
        LOGGER.debug("ODP endpoint skipped (no spec match)", extra={"label": label, "url": link.href})
        if stats is not None:
            stats.json_links_unmatched += 1
            if label:
                stats.unmatched_labels.append(label)
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)
    if stats is not None:
        stats.json_links_matched += 1
    rows: list[dict[str, Any]] = []
    for record in records:
        value = _record_value(record, spec.value_field)
        if value is None:
            continue
        date_value = _parse_date(record.get(spec.date_field), fmt=spec.date_format)
        if date_value is None:
            continue
        ym_bucket = f"{date_value.year:04d}-{date_value.month:02d}"
        month_end = _month_end(date_value)
        iso3 = (_string_or_none(record, spec.iso3_field) or "").upper() or None
        origin_iso3 = (_string_or_none(record, spec.origin_iso3_field) or "").upper() or None
        admin_name = _string_or_none(record, spec.admin_name_field)
        extra_payload = {
            "page_url": page_url,
            "json_url": link.href,
            "widget_label": label,
        }
        if "id" in record:
            extra_payload["record_id"] = record.get("id")
        rows.append(
            {
                "source_id": spec.id,
                "iso3": iso3,
                "origin_iso3": origin_iso3,
                "admin_name": admin_name,
                "ym": ym_bucket,
                "as_of_date": month_end,
                "metric": spec.metric,
                "series_semantics": spec.series_semantics,
                "value": value,
                "unit": spec.unit,
                "extra": json.dumps(extra_payload, sort_keys=True, default=str),
            }
        )
    frame = pd.DataFrame(rows, columns=_CANONICAL_COLUMNS) if rows else pd.DataFrame(columns=_CANONICAL_COLUMNS)
    return frame


def normalize_all(
    discoveries: Sequence[odp_discovery.PageDiscovery],
    fetch_json: Callable[[str], Any] | None,
    specs: Sequence[NormalizerSpec],
    *,
    today: date | None = None,
    stats: OdpPipelineStats | None = None,
) -> pd.DataFrame:
    fetch_fn = fetch_json or _default_fetch_json
    frames: list[pd.DataFrame] = []
    total_links = 0
    matched_links = 0
    for discovery in discoveries:
        for link in discovery.links:
            total_links += 1
            if stats is not None:
                stats.json_links_found += 1
            try:
                payload = fetch_fn(link.href)
            except Exception as exc:  # pragma: no cover - logging and defensive guard
                LOGGER.warning(
                    "ODP fetch failed",
                    extra={"url": link.href, "error": str(exc), "page": discovery.page_url},
                )
                if stats is not None:
                    stats.notes.append(f"fetch_failed:{link.href}")
                continue
            frame = normalize_endpoint(
                link,
                payload,
                specs,
                page_url=discovery.page_url,
                today=today,
                stats=stats,
            )
            if frame.empty:
                continue
            matched_links += 1
            frames.append(frame)
    if stats is not None:
        stats.pages_discovered = len(discoveries)
        stats.json_links_matched = max(stats.json_links_matched, matched_links)
        stats.json_links_unmatched = max(stats.json_links_unmatched, stats.json_links_found - stats.json_links_matched)
    if not frames:
        LOGGER.info(
            "ODP normalization produced no rows",
            extra={"pages": len(discoveries), "links": total_links, "matched_links": matched_links},
        )
        if stats is not None:
            stats.normalized_rows_total = 0
            stats.normalized_rows_per_series = {}
        return pd.DataFrame(columns=_CANONICAL_COLUMNS)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=_DEDUP_KEYS).reset_index(drop=True)
    combined = _ensure_frame_columns(combined)
    if stats is not None:
        stats.normalized_rows_total = int(len(combined))
        counts = combined["source_id"].fillna("").value_counts().sort_index()
        stats.normalized_rows_per_series = {k: int(v) for k, v in counts.items()}
    LOGGER.info(
        "ODP normalization summary",
        extra={
            "pages": len(discoveries),
            "links": total_links,
            "matched_links": matched_links,
            "rows": len(combined),
            "sources": dict_counts(combined["source_id"]),
            "metrics": dict_counts(combined["metric"]),
            "schema": df_schema(combined),
        },
    )
    return combined


def _default_fetch_json(url: str, timeout: float = 15.0) -> Any:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def build_odp_frame(
    *,
    config_path: str | Path,
    normalizers_path: str | Path | None = None,
    fetch_html: Callable[[str], str] | None = None,
    fetch_json: Callable[[str], Any] | None = None,
    today: date | None = None,
    stats: OdpPipelineStats | None = None,
) -> pd.DataFrame:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"ODP discovery config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    pages_cfg = config.get("pages") or []
    if stats is not None:
        stats.config_pages = len(pages_cfg)
    discoveries = odp_discovery.discover_pages(config, fetch_html=fetch_html)
    specs = load_normalizer_config(normalizers_path)
    return normalize_all(discoveries, fetch_json, specs, today=today, stats=stats)


__all__ = [
    "NormalizerSpec",
    "OdpPipelineStats",
    "build_odp_frame",
    "iter_records",
    "load_normalizer_config",
    "match_spec",
    "normalize_all",
    "normalize_endpoint",
]
