# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Normalization helpers for the IDMC connector."""
from __future__ import annotations

import calendar
import logging
from datetime import datetime
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from resolver.ingestion.utils.iso_normalize import to_iso3

SOURCE_NAME = "idmc_idu"
FLOW_METRIC_NAME = "new_displacements"
FLOW_SERIES_SEMANTICS = "new"
HDX_PREAGG_COLUMN = "__hdx_preaggregated__"
LOGGER = logging.getLogger(__name__)


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


def ensure_iso3_column(
    frame: pd.DataFrame,
    *,
    variants: Sequence[str] = (
        "iso3",
        "ISO3",
        "CountryISO3",
        "country_iso3",
        "countryiso3",
        "country iso3",
    ),
) -> pd.DataFrame:
    """Return a frame with any known ISO3 alias renamed to ``iso3``."""

    if frame is None:
        return frame

    if "iso3" in frame.columns:
        return frame

    for candidate in variants:
        if candidate in frame.columns and candidate != "iso3":
            return frame.rename(columns={candidate: "iso3"})

    if frame.empty:
        working = frame.copy()
        working["iso3"] = pd.Series(dtype="string")
        return working

    return frame


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
    if raw.empty:
        LOGGER.debug("normalize_flow: empty input")
        empty = pd.DataFrame(
            {
                "iso3": pd.Series(dtype="string"),
                "as_of_date": pd.Series(dtype="datetime64[ns]"),
                "metric": pd.Series(dtype="string"),
                "value": pd.Series(dtype="Int64"),
                "series_semantics": pd.Series(dtype="string"),
                "source": pd.Series(dtype="string"),
            }
        )
        return empty, {
            "date_parse_failed": 0,
            "no_iso3": 0,
            "no_value_col": 0,
            "date_out_of_window": 0,
            "negative_value": 0,
            "dup_event": 0,
        }

    drops = {
        "date_parse_failed": 0,
        "no_iso3": 0,
        "no_value_col": 0,
        "date_out_of_window": 0,
        "negative_value": 0,
        "dup_event": 0,
    }

    direct_columns = {
        "iso3",
        "as_of_date",
        "metric",
        "value",
        "series_semantics",
        "source",
    }
    if direct_columns.issubset(raw.columns) and HDX_PREAGG_COLUMN in raw.columns:
        working = raw.loc[
            :, [col for col in raw.columns if col in direct_columns]
        ]
        working = working.copy()
        working["iso3"] = working["iso3"].astype(str).str.strip().str.upper()
        working = working.dropna(subset=["iso3"])

        working["as_of_date"] = pd.to_datetime(
            working["as_of_date"], errors="coerce"
        )
        try:
            working["as_of_date"] = working["as_of_date"].dt.tz_localize(None)
        except TypeError:
            pass
        working = working.dropna(subset=["as_of_date"])

        start_raw = (date_window or {}).get("start")
        end_raw = (date_window or {}).get("end")
        start_dt = pd.to_datetime(start_raw, errors="coerce")
        end_dt = pd.to_datetime(end_raw, errors="coerce")
        if not pd.isna(start_dt):
            working = working[working["as_of_date"] >= start_dt]
        if not pd.isna(end_dt):
            working = working[working["as_of_date"] <= end_dt]

        working["value"] = pd.to_numeric(working["value"], errors="coerce")
        working = working.dropna(subset=["value"])
        working = working[working["value"] >= 0]
        if working.empty:
            return (
                pd.DataFrame(columns=list(direct_columns)),
                drops,
            )

        working["value"] = working["value"].astype(pd.Int64Dtype())
        working["series_semantics"] = (
            working["series_semantics"].fillna(FLOW_SERIES_SEMANTICS)
            .astype(str)
            .str.strip()
        )
        working.loc[
            working["series_semantics"] == "", "series_semantics"
        ] = FLOW_SERIES_SEMANTICS
        working["series_semantics"] = FLOW_SERIES_SEMANTICS
        working["source"] = working["source"].fillna(SOURCE_NAME).astype(str).str.strip()
        working.loc[working["source"] == "", "source"] = SOURCE_NAME
        working["metric"] = FLOW_METRIC_NAME

        aggregated = (
            working.groupby(
                ["iso3", "as_of_date", "metric", "series_semantics", "source"],
                as_index=False,
            )["value"].sum()
        )
        aggregated["value"] = aggregated["value"].astype(pd.Int64Dtype())
        aggregated["as_of_date"] = pd.to_datetime(
            aggregated["as_of_date"], errors="coerce"
        )
        try:
            aggregated["as_of_date"] = aggregated["as_of_date"].dt.tz_localize(None)
        except TypeError:
            pass
        aggregated["as_of_date"] = (
            aggregated["as_of_date"].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
        )
        aggregated = aggregated.loc[:, [
            "iso3",
            "as_of_date",
            "metric",
            "value",
            "series_semantics",
            "source",
        ]]  # ensure canonical order
        return aggregated.reset_index(drop=True), drops

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
    source_candidates = _dedupe_preserve_order(["idmc_source", "source"])

    iso_series = _normalize_iso(_first_non_null(raw, iso_candidates))
    date_series = _first_non_null(raw, date_candidates)
    value_series = pd.to_numeric(_first_non_null(raw, value_candidates), errors="coerce")
    value_series = value_series.astype(pd.Int64Dtype())
    source_series = _first_non_null(raw, source_candidates)
    if source_series.empty:
        source_series = pd.Series([pd.NA] * len(raw))

    event_column = next((column for column in date_candidates if column in raw.columns), None)
    if event_column is not None:
        raw[event_column] = pd.to_datetime(raw[event_column], errors="coerce")

    def _safe_datetime(series_name: str) -> pd.Series:
        series = raw.get(series_name)
        if series is None:
            return pd.Series([pd.NaT] * len(raw), index=raw.index)
        return pd.to_datetime(series, errors="coerce")

    end_dates = _safe_datetime("displacement_end_date")
    start_dates = _safe_datetime("displacement_start_date")
    event_dates = pd.to_datetime(date_series, errors="coerce")

    month_candidates = end_dates.combine_first(start_dates).combine_first(event_dates)
    month_periods = month_candidates.dt.to_period("M")
    month_end = month_periods.dt.to_timestamp(how="end").dt.normalize()

    normalized = pd.DataFrame(
        {
            "iso3": iso_series,
            "as_of_date": month_end,
            "metric": FLOW_METRIC_NAME,
            "value": value_series,
            "series_semantics": "new",
            "source": source_series,
            "source_override": source_series,
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

    if normalized.empty:
        return normalized.reset_index(drop=True), drops

    normalized["as_of_date"] = pd.to_datetime(
        normalized["as_of_date"], errors="coerce", utc=False
    )

    if not normalized["as_of_date"].notna().any():
        rows = int(len(normalized))
        drops["date_parse_failed"] += rows
        if rows:
            LOGGER.debug("normalize_flow: all dates NaT after coercion")
        return normalized.head(0).reset_index(drop=True), drops

    start_raw = (date_window or {}).get("start")
    end_raw = (date_window or {}).get("end")
    start_dt = pd.to_datetime(start_raw, errors="coerce")
    end_dt = pd.to_datetime(end_raw, errors="coerce")

    if not pd.isna(start_dt):
        before = len(normalized)
        normalized = normalized[normalized["as_of_date"] >= start_dt]
        drops["date_out_of_window"] += before - len(normalized)
    if not pd.isna(end_dt):
        before = len(normalized)
        normalized = normalized[normalized["as_of_date"] <= end_dt]
        drops["date_out_of_window"] += before - len(normalized)

    before = len(normalized)
    normalized = normalized.dropna(subset=["as_of_date"])
    drops["date_parse_failed"] += before - len(normalized)

    aggregation_mapping = {"value": "sum"}
    if "source_override" in normalized.columns:
        aggregation_mapping["source_override"] = "first"

    aggregated = (
        normalized.groupby(["iso3", "as_of_date", "metric"], as_index=False)
        .agg(aggregation_mapping)
        .sort_values(["iso3", "as_of_date"])
        .reset_index(drop=True)
    )

    aggregated["series_semantics"] = FLOW_SERIES_SEMANTICS
    if "source_override" in aggregated.columns:
        source_values = (
            aggregated.pop("source_override")
            .fillna("")
            .astype(str)
            .str.strip()
        )
        aggregated["source"] = source_values.where(source_values != "", SOURCE_NAME)
    else:
        aggregated["source"] = SOURCE_NAME

    if not aggregated.empty:
        aggregated["value"] = aggregated["value"].astype(pd.Int64Dtype())
    if "as_of_date" in aggregated.columns:
        aggregated["as_of_date"] = pd.to_datetime(
            aggregated["as_of_date"], errors="coerce"
        )
        try:
            aggregated["as_of_date"] = aggregated["as_of_date"].dt.tz_localize(None)
        except TypeError:
            pass
        aggregated["as_of_date"] = (
            aggregated["as_of_date"].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
        )
        LOGGER.debug("normalize_flow: as_of_date dtype=%s", aggregated["as_of_date"].dtype)

    column_order = ["iso3", "as_of_date", "metric", "value", "series_semantics", "source"]
    for column in column_order:
        if column not in aggregated.columns:
            if column == "value":
                aggregated[column] = pd.Series(dtype=pd.Int64Dtype())
            elif column == "as_of_date":
                aggregated[column] = pd.Series(dtype="datetime64[ns]")
            else:
                aggregated[column] = pd.Series(dtype="string")
    aggregated = aggregated.loc[:, column_order]

    if aggregated.empty:
        aggregated = aggregated.astype(
            {
                "iso3": "string",
                "as_of_date": "datetime64[ns]",
                "metric": "string",
                "value": pd.Int64Dtype(),
                "series_semantics": "string",
                "source": "string",
            }
        )

    return aggregated, drops


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
                {
                    "iso3": pd.Series(dtype="string"),
                    "as_of_date": pd.Series(dtype="datetime64[ns]"),
                    "metric": pd.Series(dtype="string"),
                    "value": pd.Series(dtype=pd.Int64Dtype()),
                    "series_semantics": pd.Series(dtype="string"),
                    "source": pd.Series(dtype="string"),
                }
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
