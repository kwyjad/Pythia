#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Build monthly new PIN/PA deltas from resolved totals."""

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

REQUIRED_COLUMNS = [
    "ym",
    "iso3",
    "hazard_code",
    "metric",
    "value",
    "as_of",
    "source_name",
    "source_url",
]

SEMANTIC_COL = "series_semantics"
DEFAULT_SEMANTIC = "stock"

OUTPUT_BASE_COLUMNS = [
    "ym",
    "iso3",
    "hazard_code",
    "metric",
    "value_new",
    "value_stock",
    "series_semantics_out",
    "rebase_flag",
    "first_observation",
    "delta_negative_clamped",
    "as_of",
    "source_name",
    "source_url",
]

YM_REGEX = re.compile(r"^\d{4}-\d{2}$")


def _append_cli_error_to_summary(section: str, exc: Exception, context: Dict[str, object]) -> None:
    """Best-effort append of CLI errors to the ingestion summary."""

    if not sys.executable:
        return

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.ci.append_error_to_summary",
                "--section",
                section,
                "--error-type",
                type(exc).__name__,
                "--message",
                str(exc),
                "--context",
                json.dumps(context, sort_keys=True),
            ],
            check=False,
        )
    except Exception:
        # Never mask the original CLI failure.
        pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate monthly new deltas from resolved totals.")
    parser.add_argument("--resolved", required=True, help="Path to resolved.csv")
    parser.add_argument("--out", required=True, help="Path to write deltas CSV")
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=None,
        help="Optional number of trailing months to keep (e.g., 24)",
    )
    return parser.parse_args(argv)


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"Resolved input missing required columns: {missing}")


def parse_periods(ym_series: pd.Series) -> pd.PeriodIndex:
    invalid = ym_series[~ym_series.astype(str).str.match(YM_REGEX)]
    if not invalid.empty:
        bad_values = ", ".join(sorted(set(invalid.astype(str))))
        raise SystemExit(f"Found non YYYY-MM ym values: {bad_values}")
    try:
        return pd.PeriodIndex(ym_series.astype(str), freq="M")
    except Exception as exc:  # pragma: no cover - defensive
        raise SystemExit(f"Unable to parse ym column: {exc}") from exc


def clamp_non_negative(value: float) -> float:
    return value if value >= 0 else 0.0


def process_group(group: pd.DataFrame) -> List[dict]:
    records: List[dict] = []
    prev_stock: float | None = None
    for _, row in group.iterrows():
        semantics = str(row.get(SEMANTIC_COL, DEFAULT_SEMANTIC) or DEFAULT_SEMANTIC).strip().lower()
        if semantics != "new":
            semantics = DEFAULT_SEMANTIC

        value = row["value"]
        if pd.isna(value):
            raise SystemExit("Encountered NaN value in resolved totals; cannot compute deltas.")
        value = float(value)

        base_record = {
            "ym": row["ym"],
            "iso3": row["iso3"],
            "hazard_code": row["hazard_code"],
            "metric": row["metric"],
            "as_of": row["as_of"],
            "source_name": row["source_name"],
            "source_url": row["source_url"],
            "series_semantics_out": "new",
            "rebase_flag": 0,
            "first_observation": 0,
            "delta_negative_clamped": 0,
        }

        if semantics == "new":
            base_record["value_new"] = clamp_non_negative(value)
            base_record["value_stock"] = math.nan
        else:
            # stock series
            base_record["value_stock"] = value
            update_prev = True
            if prev_stock is None:
                base_record["value_new"] = 0.0
                base_record["first_observation"] = 1
            else:
                raw_delta = value - prev_stock
                if raw_delta >= 0:
                    base_record["value_new"] = raw_delta
                else:
                    ratio = abs(raw_delta) / max(prev_stock, 1.0)
                    if ratio > 0.5:
                        base_record["value_new"] = 0.0
                        base_record["rebase_flag"] = 1
                    else:
                        base_record["value_new"] = 0.0
                        base_record["delta_negative_clamped"] = 1
                        update_prev = False
            if update_prev:
                prev_stock = value

        records.append(base_record)
    return records


def _main_impl(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    resolved_path = Path(args.resolved)
    out_path = Path(args.out)

    if not resolved_path.exists():
        raise SystemExit(f"Resolved file not found: {resolved_path}")

    df = pd.read_csv(resolved_path)
    validate_columns(df)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if df["value"].isna().any():
        raise SystemExit("Resolved input contains non-numeric values in 'value' column.")

    df["_ym_period"] = parse_periods(df["ym"])

    if args.lookback_months:
        if args.lookback_months <= 0:
            raise SystemExit("--lookback-months must be positive if provided.")
        max_period = df["_ym_period"].max()
        min_period = max_period - (args.lookback_months - 1)
        df = df[df["_ym_period"] >= min_period]
        if df.empty:
            raise SystemExit("Lookback window removed all rows; nothing to write.")

    group_keys = ["iso3", "hazard_code", "metric"]
    output_records: List[dict] = []
    for _, group in df.sort_values(["iso3", "hazard_code", "metric", "_ym_period"]).groupby(group_keys, sort=False):
        group = group.sort_values("_ym_period")
        output_records.extend(process_group(group))

    if not output_records:
        raise SystemExit("No rows produced for deltas; check input file.")

    output_df = pd.DataFrame(output_records)

    # Include optional provenance columns if present.
    if "definition_text" in df.columns:
        optional_records = []
        for _, group in df.sort_values(["iso3", "hazard_code", "metric", "_ym_period"]).groupby(group_keys, sort=False):
            group = group.sort_values("_ym_period")
            optional_records.extend(group["definition_text"].tolist())
        output_df["definition_text"] = optional_records

    output_df["value_new"] = output_df["value_new"].astype(float)
    output_df["value_new"] = output_df["value_new"].clip(lower=0)

    # Ensure canonical column order (with optional definition_text at end if present)
    columns: List[str] = OUTPUT_BASE_COLUMNS.copy()
    if "definition_text" in output_df.columns:
        columns.append("definition_text")

    output_df = output_df[columns]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_path, index=False)

    print(f"✅ Wrote {len(output_df)} monthly deltas to {out_path}")


def main(argv: Sequence[str] | None = None) -> None:
    try:
        _main_impl(argv)
    except Exception as exc:
        context: Dict[str, object] = {
            "argv": list(argv) if argv is not None else sys.argv[1:],
            "exception_class": type(exc).__name__,
        }
        _append_cli_error_to_summary("Make Deltas — CLI error", exc, context)
        raise


if __name__ == "__main__":
    main()
