#!/usr/bin/env python3
"""Build a compact context bundle for the Forecaster LLM."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from resolver.common import get_logger
from resolver.query import selectors

try:  # pragma: no cover - Python <3.9 fallback
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    from backports.zoneinfo import ZoneInfo  # type: ignore


LOGGER = get_logger(__name__)
ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")
TARGET_COLUMNS = ["ym", "iso3", "hazard_code", "metric", "unit", "value", "series"]
PERSON_TOKENS = ("person", "people", "individual")


@dataclass(frozen=True)
class ContextBundle:
    """Paths to the emitted bundle files."""

    jsonl: Path
    parquet: Path


def _round_person_value(value: float) -> int:
    """Round a numeric value using half-up semantics for person units."""

    try:
        return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    except (InvalidOperation, ValueError):  # pragma: no cover - defensive fallback
        return int(round(float(value)))


def _normalise_iso(code: object) -> str:
    return str(code or "").strip().upper()


def _normalise_metric(metric: object) -> str:
    return str(metric or "").strip().lower()


def _normalise_unit(unit: object) -> str:
    text = str(unit or "").strip()
    if not text:
        return "persons"
    return text


def _is_person_unit(unit: str) -> bool:
    lowered = unit.lower()
    return any(token in lowered for token in PERSON_TOKENS)


def _prepare_month_frame(df: pd.DataFrame, ym: str) -> pd.DataFrame:
    """Trim and normalise Resolver monthly rows for context output."""

    if df is None or df.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    frame = df.copy()

    if "value" not in frame.columns or frame["value"].isna().all():
        frame["value"] = frame.get("value_new")

    frame["value"] = pd.to_numeric(frame.get("value"), errors="coerce")
    frame = frame[frame["value"].notna()].copy()

    frame["iso3"] = frame.get("iso3", "").map(_normalise_iso)
    frame["hazard_code"] = frame.get("hazard_code", "").map(_normalise_iso)
    frame["metric"] = frame.get("metric", "").map(_normalise_metric)
    frame["unit"] = frame.get("unit", "").map(_normalise_unit)

    # Drop rows that still lack the required keys after normalisation.
    required_mask = (
        (frame["iso3"] != "")
        & (frame["hazard_code"] != "")
        & (frame["metric"] != "")
    )
    frame = frame[required_mask].copy()
    if frame.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)

    person_mask = frame["unit"].map(_is_person_unit)
    frame.loc[person_mask, "value"] = frame.loc[person_mask, "value"].map(
        _round_person_value
    )

    frame["ym"] = ym
    frame["series"] = "new"

    frame = frame[["ym", "iso3", "hazard_code", "metric", "unit", "value", "series"]]
    frame = frame.sort_values(["iso3", "hazard_code", "metric"])
    return frame.reset_index(drop=True)


def last_finalized_month(reference: dt.datetime | None = None) -> str:
    """Return the last fully finalized Resolver month (Europe/Istanbul)."""

    if reference is None:
        now = dt.datetime.now(ISTANBUL_TZ)
    else:
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=ISTANBUL_TZ)
        else:
            reference = reference.astimezone(ISTANBUL_TZ)
        now = reference

    year = now.year
    month = now.month - 1
    if month == 0:
        month = 12
        year -= 1
    return f"{year:04d}-{month:02d}"


def target_months(count: int, *, reference: dt.datetime | None = None) -> list[str]:
    """Compute the trailing ``count`` months ending with the finalized month."""

    if count <= 0:
        return []

    end = last_finalized_month(reference)
    year, month = map(int, end.split("-"))

    months: list[str] = []
    for _ in range(count):
        months.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return list(reversed(months))


def build_context_frame(
    months: Sequence[str],
    *,
    backend: str = "db",
    loader: Callable[[str], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Collect monthly "new" series rows for the requested months."""

    records: list[pd.DataFrame] = []
    month_counts: dict[str, int] = {}

    if loader is None:

        def loader(ym: str) -> pd.DataFrame:  # type: ignore[no-redef]
            df, _dataset, _series = selectors.load_series_for_month(
                ym,
                is_current_month=False,
                requested_series="new",
                backend=backend,
            )
            if df is None:
                return pd.DataFrame()
            return df

    for ym in months:
        raw = loader(ym)
        month_frame = _prepare_month_frame(raw, ym)
        month_counts[ym] = int(len(month_frame))
        if month_frame.empty:
            continue
        records.append(month_frame)

    if not records:
        empty = pd.DataFrame(columns=TARGET_COLUMNS)
        return empty, month_counts

    frame = pd.concat(records, ignore_index=True)
    frame = frame[TARGET_COLUMNS]
    frame = frame.sort_values(["ym", "iso3", "hazard_code", "metric"]).reset_index(drop=True)
    return frame, month_counts


def _print_summary(frame: pd.DataFrame, month_counts: dict[str, int]) -> None:
    """Print counts per month and per (iso3, hazard_code) pair."""

    print("Context month counts:")
    if not month_counts:
        print("  (no months processed)")
    else:
        for ym in sorted(month_counts):
            print(f"  {ym}: {month_counts[ym]}")

    if frame.empty:
        print("No context rows generated.")
        return

    print("Context rows by country/hazard:")
    pair_counts = (
        frame.groupby(["iso3", "hazard_code"])["value"].count().sort_values(ascending=False)
    )
    for (iso3, hazard_code), count in pair_counts.items():
        print(f"  {iso3}/{hazard_code}: {int(count)}")


def _ensure_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Convert the DataFrame to JSON-serialisable records with rounded values."""

    records: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        value = float(row.value)
        if _is_person_unit(row.unit):
            value = _round_person_value(value)
        record = {
            "ym": row.ym,
            "iso3": row.iso3,
            "hazard_code": row.hazard_code,
            "metric": row.metric,
            "unit": row.unit,
            "value": value,
            "series": row.series,
        }
        records.append(record)
    return records


def write_context_bundle(frame: pd.DataFrame, outdir: Path) -> ContextBundle:
    """Write the context bundle to JSONL and Parquet outputs."""

    outdir.mkdir(parents=True, exist_ok=True)
    records = _ensure_records(frame)
    bundle_df = pd.DataFrame.from_records(records, columns=TARGET_COLUMNS)

    jsonl_path = outdir / "facts_last12.jsonl"
    parquet_path = outdir / "facts_last12.parquet"

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    bundle_df.to_parquet(parquet_path, index=False)

    LOGGER.info(
        "context bundle written",
        extra={"jsonl": str(jsonl_path), "parquet": str(parquet_path), "rows": len(bundle_df)},
    )
    return ContextBundle(jsonl=jsonl_path, parquet=parquet_path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Resolver LLM context bundle")
    parser.add_argument(
        "--months",
        type=int,
        default=12,
        help="Number of trailing months to include (default: 12)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("context"),
        help="Directory for generated context files (default: ./context)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    months = target_months(args.months)
    frame, month_counts = build_context_frame(months)
    _print_summary(frame, month_counts)

    if frame.empty:
        print("No context rows were generated; aborting with non-zero exit code.")
        return 1

    write_context_bundle(frame, args.outdir)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
