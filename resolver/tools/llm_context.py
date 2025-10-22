"""Helpers for building the Forecaster LLM context bundle."""

from __future__ import annotations

import json
import logging
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence

import pandas as pd

try:  # pragma: no cover - optional dependency during fast tests
    from resolver.query import selectors  # type: ignore
except Exception:  # pragma: no cover - selectors unavailable in some environments
    selectors = None  # type: ignore

log = logging.getLogger(__name__)

CONTEXT_COLUMNS: list[str] = [
    "iso3",
    "hazard_code",
    "ym",
    "metric",
    "unit",
    "value",
    "series",
]
# Backwards compatibility for older imports/tests that referenced TARGET_COLUMNS
TARGET_COLUMNS = CONTEXT_COLUMNS
PERSON_TOKENS = ("person", "people", "individual")


def _empty_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical schema."""

    return pd.DataFrame(columns=CONTEXT_COLUMNS)


def _normalize_iso(value: object) -> str:
    return str(value or "").strip().upper()


def _normalize_metric(value: object) -> str:
    return str(value or "").strip().lower()


def _normalize_unit(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return "persons"
    return text


def _is_person_unit(unit: str) -> bool:
    lowered = unit.lower()
    return any(token in lowered for token in PERSON_TOKENS)


def _round_person_value(value: float | int) -> int:
    try:
        return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError, TypeError):  # pragma: no cover - defensive guard
        return int(round(float(value)))


def _column(frame: pd.DataFrame, name: str, default: object = "") -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _prepare_month_frame(source: pd.DataFrame | None, ym: str, series: str) -> pd.DataFrame:
    if source is None or source.empty:
        return _empty_frame()

    frame = source.copy()

    if "value" not in frame.columns or frame["value"].isna().all():
        if "value_new" in frame.columns:
            frame["value"] = frame["value_new"]

    value_series = _column(frame, "value", default=None)
    frame["value"] = pd.to_numeric(value_series, errors="coerce")
    frame = frame[frame["value"].notna()].copy()
    if frame.empty:
        return _empty_frame()

    frame["iso3"] = _column(frame, "iso3").map(_normalize_iso)
    frame["hazard_code"] = _column(frame, "hazard_code").map(_normalize_iso)
    frame["metric"] = _column(frame, "metric").map(_normalize_metric)
    frame["unit"] = _column(frame, "unit", default="persons").map(_normalize_unit)

    mask = (frame["iso3"] != "") & (frame["hazard_code"] != "") & (frame["metric"] != "")
    frame = frame[mask].copy()
    if frame.empty:
        return _empty_frame()

    person_mask = frame["unit"].map(_is_person_unit)
    if person_mask.any():
        frame.loc[person_mask, "value"] = frame.loc[person_mask, "value"].map(_round_person_value)

    frame["series"] = series
    frame["ym"] = ym

    return frame[CONTEXT_COLUMNS]


def build_context_frame(
    months: Sequence[str] | None = None,
    *,
    backend: str = "db",
    requested_series: str = "new",
    is_current_month: bool = False,
) -> pd.DataFrame:
    """Return the canonical LLM context DataFrame for the requested months."""

    if not months:
        return _empty_frame()

    frames: list[pd.DataFrame] = []

    for ym in months:
        month_df = None
        if selectors is not None:
            try:
                result = selectors.load_series_for_month(
                    ym=ym,
                    requested_series=requested_series,
                    backend=backend,
                    is_current_month=is_current_month,
                )
            except Exception as exc:  # pragma: no cover - defensive logging in CI
                log.warning("load_series_for_month failed for ym=%s: %s", ym, exc)
                result = None
            if isinstance(result, tuple):
                month_df = result[0]
            else:
                month_df = result

        prepared = _prepare_month_frame(month_df, ym, requested_series)
        if not prepared.empty:
            frames.append(prepared)

    if not frames:
        return _empty_frame()

    frame = pd.concat(frames, ignore_index=True)
    frame = frame[CONTEXT_COLUMNS]
    frame = frame.sort_values(["iso3", "hazard_code", "ym", "metric"]).reset_index(drop=True)
    return frame


def _frame_to_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        value: object = row.value
        if _is_person_unit(str(row.unit)):
            value = _round_person_value(float(row.value))
        records.append(
            {
                "iso3": row.iso3,
                "hazard_code": row.hazard_code,
                "ym": row.ym,
                "metric": row.metric,
                "unit": row.unit,
                "value": value,
                "series": row.series,
            }
        )
    return records


def write_context_bundle(
    months: Sequence[str],
    outdir: str | Path,
    *,
    backend: str = "db",
    requested_series: str = "new",
    frame: pd.DataFrame | None = None,
    jsonl_name: str = "facts_last12.jsonl",
    parquet_name: str = "facts_last12.parquet",
) -> dict[str, str]:
    """Build and persist the JSONL/Parquet bundle for the requested months."""

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bundle_frame = frame if frame is not None else build_context_frame(
        months=months,
        backend=backend,
        requested_series=requested_series,
    )

    bundle_frame = bundle_frame[CONTEXT_COLUMNS]

    jsonl_path = outdir / jsonl_name
    parquet_path = outdir / parquet_name

    records = _frame_to_records(bundle_frame)

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Use pandas for Parquet to maintain schema guarantees (supports empty frames)
    bundle_frame.to_parquet(parquet_path, index=False)

    log.info(
        "LLM context bundle written",
        extra={
            "jsonl": str(jsonl_path),
            "parquet": str(parquet_path),
            "rows": len(bundle_frame),
        },
    )

    return {"jsonl": str(jsonl_path), "parquet": str(parquet_path)}
