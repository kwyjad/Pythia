"""Normalization helpers for the IDMC connector."""
from __future__ import annotations

import calendar
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import pandas as pd

from resolver.ingestion.utils.iso_normalize import to_iso3

SOURCE_NAME = "IDMC"


def _coerce_month_end(values: pd.Series) -> pd.Series:
    """Coerce various date formats to ISO month-end dates."""

    def to_month_end(value):
        if pd.isna(value):
            return pd.NaT
        text = str(value).strip()
        if not text:
            return pd.NaT
        try:
            if len(text) == 4 and text.isdigit():
                dt = datetime(int(text), 12, 31)
            elif len(text) == 7 and "-" in text:
                year, month = text.split("-")
                last = calendar.monthrange(int(year), int(month))[1]
                dt = datetime(int(year), int(month), last)
            else:
                dt_value = pd.to_datetime(text, errors="coerce")
                if pd.isna(dt_value):
                    return pd.NaT
                last = calendar.monthrange(dt_value.year, dt_value.month)[1]
                dt = datetime(dt_value.year, dt_value.month, last)
            return dt.date().isoformat()
        except Exception:  # pragma: no cover - defensive guard
            return pd.NaT

    return values.apply(to_month_end)


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _first_non_null(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    available: List[str] = [col for col in columns if col in df.columns]
    if not available:
        return pd.Series([pd.NA] * len(df))
    subset = df[available].copy()
    for column in available:
        series = subset[column]
        mask = series.isna()
        if series.dtype == object:
            mask = mask | series.astype(str).str.strip().eq("")
        subset[column] = series.where(~mask, pd.NA)
    return subset.bfill(axis=1).iloc[:, 0]


def _normalize_iso(series: pd.Series) -> pd.Series:
    def clean(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        iso = to_iso3(str(value))
        return iso if iso else pd.NA

    return series.apply(clean)


def _normalize_monthly_flow(
    raw: pd.DataFrame,
    field_aliases: Dict[str, List[str]],
    date_window: Dict[str, str | None],
    *,
    map_hazards: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    drops = {
        "date_parse_failed": 0,
        "no_iso3": 0,
        "no_value_col": 0,
        "date_out_of_window": 0,
        "negative_value": 0,
        "dup_event": 0,
    }

    iso_candidates = _dedupe_preserve_order(field_aliases.get("iso3", ["iso3", "ISO3"]))
    date_candidates = _dedupe_preserve_order(
        [
            "displacement_date",
            "displacement_start_date",
            "displacement_end_date",
        ]
        + field_aliases.get("date", [])
    )
    value_candidates = _dedupe_preserve_order(["figure"] + field_aliases.get("value_flow", []))

    iso_series = _normalize_iso(_first_non_null(raw, iso_candidates))
    date_series = _first_non_null(raw, date_candidates)
    value_series = pd.to_numeric(_first_non_null(raw, value_candidates), errors="coerce")

    normalized = pd.DataFrame(
        {
            "iso3": iso_series,
            "as_of_date": _coerce_month_end(date_series),
            "metric": "idp_displacement_new_idmc",
            "value": value_series,
            "series_semantics": "new",
            "source": SOURCE_NAME,
        }
    )

    if map_hazards:
        def _raw_or_na(column: str) -> pd.Series:
            if column in raw.columns:
                return raw[column]
            return pd.Series([pd.NA] * len(raw), index=raw.index)

        hazard_columns = {
            "displacement_type": _raw_or_na("displacement_type"),
            "hazard_category": _raw_or_na("hazard_category"),
            "hazard_subcategory": _raw_or_na("hazard_subcategory"),
            "hazard_type": _raw_or_na("hazard_type"),
            "hazard_subtype": _raw_or_na("hazard_subtype"),
            "violence_type": _raw_or_na("violence_type"),
            "conflict_type": _raw_or_na("conflict_type"),
            "notes": _raw_or_na("notes"),
            "event_details": _raw_or_na("event_details"),
        }
        for column, series in hazard_columns.items():
            normalized[column] = series

    def _clean_iso(value: object) -> object:
        if pd.isna(value):
            return pd.NA
        text = str(value).strip().upper()
        return text or pd.NA

    normalized["iso3"] = normalized["iso3"].apply(_clean_iso)

    before = len(normalized)
    normalized = normalized.dropna(subset=["iso3"])
    drops["no_iso3"] += before - len(normalized)

    before = len(normalized)
    normalized = normalized.dropna(subset=["value"])
    drops["no_value_col"] += before - len(normalized)

    before = len(normalized)
    normalized = normalized[normalized["value"] >= 0]
    drops["negative_value"] += before - len(normalized)

    before = len(normalized)
    normalized = normalized.dropna(subset=["as_of_date"])
    drops["date_parse_failed"] += before - len(normalized)

    start = (date_window or {}).get("start")
    end = (date_window or {}).get("end")
    if start:
        before = len(normalized)
        normalized = normalized[normalized["as_of_date"] >= start]
        drops["date_out_of_window"] += before - len(normalized)
    if end:
        before = len(normalized)
        normalized = normalized[normalized["as_of_date"] <= end]
        drops["date_out_of_window"] += before - len(normalized)

    before = len(normalized)
    normalized = (
        normalized.sort_values("value", ascending=False)
        .drop_duplicates(["iso3", "as_of_date", "metric"], keep="first")
        .reset_index(drop=True)
    )
    drops["dup_event"] += before - len(normalized)

    return normalized, drops


def normalize_all(
    by_series: Dict[str, pd.DataFrame],
    field_aliases: Dict[str, List[str]],
    date_window: Dict[str, str | None],
    selected_series: Iterable[str] | None = None,
    *,
    map_hazards: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    tidy_frames = []
    aggregate_drops = {
        "date_parse_failed": 0,
        "no_iso3": 0,
        "no_value_col": 0,
        "date_out_of_window": 0,
        "negative_value": 0,
        "dup_event": 0,
    }

    series_filter = {
        (series or "").strip().lower() for series in (selected_series or ["flow"])
    }

    if "flow" in series_filter and "monthly_flow" in by_series:
        frame, drops = _normalize_monthly_flow(
            by_series["monthly_flow"],
            field_aliases,
            date_window,
            map_hazards=map_hazards,
        )
        tidy_frames.append(frame)
        for key, value in drops.items():
            aggregate_drops[key] = aggregate_drops.get(key, 0) + value

    if not tidy_frames:
        return (
            pd.DataFrame(
                columns=[
                    "iso3",
                    "as_of_date",
                    "metric",
                    "value",
                    "series_semantics",
                    "source",
                ]
            ),
            aggregate_drops,
        )

    return pd.concat(tidy_frames, ignore_index=True), aggregate_drops


def maybe_map_hazards(
    frame: pd.DataFrame, enabled: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Optionally apply hazard mapping to a normalized frame."""

    if not enabled:
        return frame, pd.DataFrame()

    from .hazards import apply_hazard_mapping

    hazard_context_columns = [
        "displacement_type",
        "hazard_category",
        "hazard_subcategory",
        "hazard_type",
        "hazard_subtype",
        "violence_type",
        "conflict_type",
        "notes",
        "event_details",
    ]

    working = frame.copy()
    for column in hazard_context_columns:
        if column not in working.columns:
            working[column] = pd.NA

    mapped = apply_hazard_mapping(working)
    hazard_series = mapped.get(
        "hazard_code", pd.Series([pd.NA] * len(mapped), index=mapped.index)
    )
    unmapped_mask = hazard_series.isna()
    context_columns = [
        column
        for column in (
            ["iso3", "as_of_date", "metric", "value"] + hazard_context_columns
        )
        if column in mapped.columns
    ]
    unmapped = mapped.loc[unmapped_mask, context_columns].copy()

    drop_columns = [
        column for column in hazard_context_columns if column in mapped.columns
    ]
    cleaned = mapped.drop(columns=drop_columns)
    return cleaned, unmapped
