#!/usr/bin/env python3
"""Build a compact LLM context bundle from Resolver monthly deltas."""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd

from resolver.common import dict_counts, get_logger
from resolver.query import selectors

try:  # pragma: no cover - Python <3.9 fallback
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

LOGGER = get_logger(__name__)
DEFAULT_METRICS: tuple[str, ...] = ("in_need", "affected")
DEFAULT_UNIT = "persons"
DEFAULT_BASENAME = "facts_last12"


@dataclass(frozen=True)
class ContextBundle:
    jsonl: Path
    parquet: Path


def _current_ym() -> str:
    if ZoneInfo is None:
        now = dt.datetime.utcnow()
    else:
        now = dt.datetime.now(ZoneInfo("Europe/Istanbul"))
    return f"{now.year:04d}-{now.month:02d}"


def recent_months(count: int, *, reference: str | None = None) -> list[str]:
    """Return a list of year-month strings covering ``count`` months."""

    if count <= 0:
        return []
    if reference:
        try:
            ref_year, ref_month = map(int, reference.split("-"))
            dt.date(ref_year, ref_month, 1)
        except Exception as exc:  # pragma: no cover - defensive parsing
            raise ValueError("reference must be YYYY-MM") from exc
    else:
        ref = _current_ym()
        ref_year, ref_month = map(int, ref.split("-"))

    months: list[str] = []
    year = ref_year
    month = ref_month
    for _ in range(count):
        months.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return list(reversed(months))


def _filter_metrics(frame: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    if not metrics:
        return frame
    allowed = {m.strip().lower() for m in metrics if m}
    if not allowed:
        return frame
    frame = frame.copy()
    frame["metric"] = frame.get("metric", "").astype(str).str.lower()
    return frame[frame["metric"].isin(allowed)]


def _normalise_units(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    return values.replace("", DEFAULT_UNIT)


def _prepare_month_frame(
    df: pd.DataFrame,
    *,
    ym: str,
    metrics: Sequence[str],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["iso3", "hazard_code", "ym", "metric", "unit", "value", "series"])

    frame = df.copy()
    if "value" not in frame.columns or frame["value"].isna().all():
        if "value_new" in frame.columns:
            frame["value"] = frame["value_new"]
    frame["value"] = pd.to_numeric(frame.get("value"), errors="coerce")
    frame = frame[frame["value"].notna()]

    for column in ("iso3", "hazard_code"):
        frame[column] = frame.get(column, "").astype(str).str.strip().str.upper()
    frame["metric"] = frame.get("metric", "").astype(str).str.lower()
    frame["unit"] = _normalise_units(frame.get("unit", ""))
    frame = _filter_metrics(frame, metrics)

    if frame.empty:
        return pd.DataFrame(columns=["iso3", "hazard_code", "ym", "metric", "unit", "value", "series"])

    frame["ym"] = ym
    grouped = (
        frame.groupby(["iso3", "hazard_code", "ym", "metric", "unit"], as_index=False)["value"].sum()
    )
    grouped["series"] = "new"
    grouped = grouped[["iso3", "hazard_code", "ym", "metric", "unit", "value", "series"]]
    return grouped


def build_context_frame(
    *,
    months: Sequence[str],
    metrics: Sequence[str] = DEFAULT_METRICS,
    backend: str = "auto",
) -> pd.DataFrame:
    """Collect monthly "new" series rows and aggregate to context rows."""

    records: list[pd.DataFrame] = []
    metrics = tuple(metrics)

    for ym in months:
        is_current = ym == selectors.current_ym_istanbul()
        df, dataset_label, series = selectors.load_series_for_month(
            ym,
            is_current,
            "new",
            backend=backend,
        )
        if df is None or df.empty:
            LOGGER.info("context month skipped", extra={"event": "context_month", "ym": ym, "rows": 0})
            continue
        month_frame = _prepare_month_frame(df, ym=ym, metrics=metrics)
        if month_frame.empty:
            LOGGER.info(
                "context month produced no rows after filtering",
                extra={"event": "context_month", "ym": ym, "rows": 0, "dataset": dataset_label, "series": series},
            )
            continue
        LOGGER.info(
            "context month prepared",
            extra={
                "event": "context_month",
                "ym": ym,
                "rows": int(len(month_frame)),
                "dataset": dataset_label,
                "series": series,
                "metrics": dict_counts(month_frame["metric"]),
            },
        )
        records.append(month_frame)

    if not records:
        return pd.DataFrame(columns=["iso3", "hazard_code", "ym", "metric", "unit", "value", "series"])

    frame = pd.concat(records, ignore_index=True)
    frame = frame.sort_values(["ym", "iso3", "hazard_code", "metric"]).reset_index(drop=True)
    return frame


def _log_summary(frame: pd.DataFrame) -> None:
    if frame.empty:
        LOGGER.info("context bundle empty", extra={"event": "context_summary", "rows": 0})
        return
    month_counts = frame.groupby("ym")["value"].count().to_dict()
    top_countries = (
        frame.groupby("iso3")["value"].sum().sort_values(ascending=False).head(10)
    )
    LOGGER.info(
        "context bundle summary",
        extra={
            "event": "context_summary",
            "rows": int(len(frame)),
            "months": month_counts,
            "top_countries": {iso: float(val) for iso, val in top_countries.items()},
        },
    )


def write_context_bundle(
    frame: pd.DataFrame,
    outdir: Path,
    *,
    basename: str = DEFAULT_BASENAME,
) -> ContextBundle:
    outdir.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    frame["value"] = pd.to_numeric(frame.get("value"), errors="coerce")
    frame = frame.fillna({"series": "new"})

    jsonl_path = outdir / f"{basename}.jsonl"
    parquet_path = outdir / f"{basename}.parquet"

    json_payload = frame.to_json(orient="records", lines=True)
    jsonl_path.write_text(json_payload, encoding="utf-8")
    frame.to_parquet(parquet_path, index=False)

    LOGGER.info(
        "context bundle written",
        extra={
            "event": "context_write",
            "jsonl": str(jsonl_path),
            "parquet": str(parquet_path),
            "rows": int(len(frame)),
        },
    )
    return ContextBundle(jsonl=jsonl_path, parquet=parquet_path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Resolver LLM context bundle")
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Number of recent months to include (default: 12)",
    )
    parser.add_argument(
        "--outdir",
        default="context",
        help="Output directory for generated files",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=sorted(selectors.VALID_BACKENDS),
        help="Preferred selector backend (default: auto)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    months = recent_months(args.months)
    frame = build_context_frame(months=months, backend=args.backend)
    _log_summary(frame)
    write_context_bundle(frame, Path(args.outdir))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
