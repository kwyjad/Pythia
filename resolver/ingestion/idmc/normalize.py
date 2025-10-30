"""Normalization helpers for the IDMC connector skeleton."""
from __future__ import annotations

import calendar
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd

SOURCE_NAME = "IDMC"


def _coerce_month_end(values: pd.Series) -> pd.Series:
    """Coerce various date formats to ISO month-end dates."""

    def to_month_end(value):
        if pd.isna(value):
            return pd.NaT
        text = str(value)
        try:
            if len(text) == 4:
                dt = datetime(int(text), 12, 31)
            elif len(text) == 7 and "-" in text:
                year, month = text.split("-")
                last = calendar.monthrange(int(year), int(month))[1]
                dt = datetime(int(year), int(month), last)
            else:
                dt = pd.to_datetime(text, errors="coerce")
                if pd.isna(dt):
                    return pd.NaT
                last = calendar.monthrange(dt.year, dt.month)[1]
                dt = datetime(dt.year, dt.month, last)
            return dt.date().isoformat()
        except Exception:  # pragma: no cover - defensive guard
            return pd.NaT

    return values.apply(to_month_end)


def _choose_first(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    for column in columns:
        if column in df.columns:
            return df[column]
    return pd.Series([pd.NA] * len(df))


def normalize_series(
    raw: pd.DataFrame,
    series: str,
    aliases: Dict[str, list[str]],
    date_window: Dict[str, str | None],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    drops = {
        "date_parse_failed": 0,
        "no_iso3": 0,
        "no_value_col": 0,
        "date_out_of_window": 0,
    }
    iso3 = _choose_first(raw, aliases.get("iso3", ["iso3"]))
    date = _choose_first(raw, aliases.get("date", ["date"]))
    if series == "flow":
        value = _choose_first(raw, aliases.get("value_flow", ["new_displacements"]))
        metric = "idp_displacement_new_idmc"
        semantics = "new"
    else:
        value = _choose_first(raw, aliases.get("value_stock", ["idps"]))
        metric = "idp_displacement_stock_idmc"
        semantics = "stock"

    normalized = pd.DataFrame(
        {
            "iso3": iso3,
            "as_of_date": _coerce_month_end(date),
            "metric": metric,
            "value": pd.to_numeric(value, errors="coerce"),
            "series_semantics": semantics,
            "source": SOURCE_NAME,
        }
    )

    before = len(normalized)
    normalized = normalized.dropna(subset=["iso3"])
    drops["no_iso3"] += before - len(normalized)
    before = len(normalized)

    normalized = normalized.dropna(subset=["value"])
    drops["no_value_col"] += before - len(normalized)
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

    normalized = (
        normalized.sort_values("value")
        .drop_duplicates(["iso3", "as_of_date", "metric"], keep="last")
        .reset_index(drop=True)
    )
    return normalized, drops


def normalize_all(
    by_series: Dict[str, pd.DataFrame],
    field_aliases: Dict[str, list[str]],
    date_window: Dict[str, str | None],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    tidy_frames = []
    aggregate_drops = {
        "date_parse_failed": 0,
        "no_iso3": 0,
        "no_value_col": 0,
        "date_out_of_window": 0,
    }

    if "monthly_flow" in by_series:
        frame, drops = normalize_series(
            by_series["monthly_flow"], "flow", field_aliases, date_window
        )
        tidy_frames.append(frame)
        for key, value in drops.items():
            aggregate_drops[key] = aggregate_drops.get(key, 0) + value

    if "stock" in by_series:
        frame, drops = normalize_series(
            by_series["stock"], "stock", field_aliases, date_window
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
