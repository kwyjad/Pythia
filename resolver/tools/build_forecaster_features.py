#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Build Resolver feature set for Forecaster calibration loops."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd

try:
    from resolver.db import duckdb_io
except Exception:  # pragma: no cover - duckdb optional in some environments
    duckdb_io = None  # type: ignore[assignment]


TOOLS_DIR = Path(__file__).resolve().parent
RESOLVER_ROOT = TOOLS_DIR.parent
REPO_ROOT = RESOLVER_ROOT.parent
DEFAULT_SNAPSHOTS = RESOLVER_ROOT / "snapshots"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "resolver_features.parquet"
DEFAULT_METRICS = ("in_need", "affected", "displaced")
SPIKE_ZSCORE_THRESHOLD = 3.0


class FeatureBuildError(RuntimeError):
    """Raised when feature building fails."""


@dataclass
class FeatureInputs:
    resolved: pd.DataFrame
    deltas: pd.DataFrame
    generated_at: dt.datetime


def _parse_metrics(metrics: Iterable[str] | None) -> Tuple[str, ...]:
    values: list[str] = []
    for metric in metrics or []:
        metric = str(metric).strip()
        if not metric:
            continue
        values.extend([m.strip() for m in metric.split(",") if m.strip()])
    return tuple(dict.fromkeys(values)) or DEFAULT_METRICS


def _latest_snapshot_dir(base: Path) -> Optional[Path]:
    if not base.exists():
        return None
    candidates = [
        p for p in base.iterdir() if p.is_dir() and p.name[:4].isdigit() and "-" in p.name
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def _load_snapshot_tables(base: Path) -> FeatureInputs:
    snap_dir = _latest_snapshot_dir(base)
    if snap_dir is None:
        raise FeatureBuildError(f"No snapshot directories found under {base}")

    resolved_path = snap_dir / "facts_resolved.parquet"
    if not resolved_path.exists():
        resolved_path = snap_dir / "facts_resolved.csv"
    if not resolved_path.exists():
        raise FeatureBuildError(f"Snapshot {snap_dir} missing facts_resolved parquet/CSV")

    deltas_path = snap_dir / "facts_deltas.parquet"
    if not deltas_path.exists():
        deltas_path = snap_dir / "facts_deltas.csv"
    if not deltas_path.exists():
        raise FeatureBuildError(f"Snapshot {snap_dir} missing facts_deltas parquet/CSV")

    manifest_path = snap_dir / "manifest.json"
    generated_at = dt.datetime.utcnow()
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            created = payload.get("created_at_utc")
            if created:
                generated_at = dt.datetime.fromisoformat(created.replace("Z", "+00:00"))
        except Exception:
            pass

    resolved_df = pd.read_parquet(resolved_path) if resolved_path.suffix == ".parquet" else pd.read_csv(resolved_path)
    deltas_df = pd.read_parquet(deltas_path) if deltas_path.suffix == ".parquet" else pd.read_csv(deltas_path)
    return FeatureInputs(resolved=resolved_df, deltas=deltas_df, generated_at=generated_at)


def _load_duckdb_tables(db_url: Optional[str]) -> Optional[FeatureInputs]:
    if duckdb_io is None:
        return None
    try:
        conn = duckdb_io.get_db(db_url or os.environ.get("RESOLVER_DB_URL"))
    except Exception:
        return None
    try:
        deltas = conn.execute(
            """
            SELECT ym, iso3, hazard_code, metric, value_new, as_of, source_name, source_url
            FROM facts_deltas
            """
        ).fetch_df()
        resolved = conn.execute(
            """
            SELECT ym, iso3, hazard_code, metric, as_of_date, hazard_class, precedence_tier
            FROM facts_resolved
            """
        ).fetch_df()
    except Exception:
        return None
    if deltas.empty or resolved.empty:
        return None
    return FeatureInputs(resolved=resolved, deltas=deltas, generated_at=dt.datetime.utcnow())


def load_feature_inputs(
    *, snapshots_dir: Path = DEFAULT_SNAPSHOTS, db_url: Optional[str] = None
) -> FeatureInputs:
    inputs = _load_duckdb_tables(db_url)
    if inputs is not None:
        return inputs
    return _load_snapshot_tables(snapshots_dir)


def _normalise_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    present = {col: mapping[col] for col in mapping if col in df.columns}
    renamed = df.rename(columns=present)
    for required, target in mapping.items():
        if target not in renamed.columns:
            renamed[target] = df.get(required)
    return renamed


def _prepare_frames(resolved: pd.DataFrame, deltas: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    resolved = resolved.copy()
    deltas = deltas.copy()

    resolved.columns = [str(c) for c in resolved.columns]
    deltas.columns = [str(c) for c in deltas.columns]

    resolved = _normalise_columns(
        resolved,
        {
            "ym": "ym",
            "iso3": "iso3",
            "hazard_code": "hazard_code",
            "metric": "metric",
            "as_of_date": "as_of_date",
            "hazard_class": "hazard_class",
            "precedence_tier": "precedence_tier",
        },
    )
    deltas = _normalise_columns(
        deltas,
        {
            "ym": "ym",
            "iso3": "iso3",
            "hazard_code": "hazard_code",
            "metric": "metric",
            "value_new": "value_new",
            "as_of": "as_of",
        },
    )

    return resolved, deltas


def _coerce_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(0.0)


def _rolling_sum(group: pd.Series, window: int) -> pd.Series:
    return group.rolling(window=window, min_periods=1).sum()


def compute_feature_frame(
    *,
    resolved: pd.DataFrame,
    deltas: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    generated_at: Optional[dt.datetime] = None,
) -> pd.DataFrame:
    if resolved.empty or deltas.empty:
        raise FeatureBuildError("facts_resolved or facts_deltas inputs are empty; cannot build features")

    resolved, deltas = _prepare_frames(resolved, deltas)

    metric_scope = {m.strip() for m in (metrics or DEFAULT_METRICS)}
    metric_scope = {m for m in metric_scope if m}
    if not metric_scope:
        metric_scope = set(DEFAULT_METRICS)

    deltas = deltas[deltas["metric"].isin(metric_scope)].copy()
    if deltas.empty:
        raise FeatureBuildError("No facts_deltas rows remain after metric filtering")

    resolved = resolved[resolved["metric"].isin(metric_scope)].copy()

    deltas["ym"] = deltas["ym"].astype(str)
    resolved["ym"] = resolved["ym"].astype(str)

    try:
        deltas["ym_period"] = pd.PeriodIndex(deltas["ym"], freq="M")
    except Exception as exc:
        raise FeatureBuildError(f"Unable to parse ym periods in facts_deltas: {exc}") from exc

    deltas["value_new"] = _coerce_numeric(deltas["value_new"])

    key_cols = ["ym", "iso3", "hazard_code", "metric"]
    merged = deltas.merge(resolved[key_cols + ["as_of_date", "hazard_class", "precedence_tier"]], on=key_cols, how="left")

    merged["as_of_date"] = merged["as_of"].fillna(merged["as_of_date"])
    merged["as_of_date"] = pd.to_datetime(merged["as_of_date"], errors="coerce")

    generated_at = generated_at or dt.datetime.utcnow()
    generated_timestamp = pd.Timestamp(generated_at)
    if generated_timestamp.tzinfo is not None:
        generated_utc = generated_timestamp.tz_convert("UTC")
    else:
        generated_utc = generated_timestamp.tz_localize("UTC")
    generated_naive = generated_utc.tz_localize(None)
    merged["as_of_recency_days"] = (generated_naive - merged["as_of_date"]).dt.days
    merged["as_of_recency_days"] = (
        pd.to_numeric(merged["as_of_recency_days"], errors="coerce").round().astype("Int64")
    )

    merged["delta_m1"] = merged["value_new"].astype(float)
    merged["ym_order"] = merged["ym_period"].astype("int64")
    merged["source_tier"] = merged["precedence_tier"].fillna("")

    grouped = merged.sort_values(["iso3", "hazard_code", "metric", "ym_period"]).groupby(["iso3", "hazard_code", "metric"], sort=False)
    merged["delta_m3_sum"] = grouped["delta_m1"].transform(lambda s: _rolling_sum(s, 3))
    merged["delta_m6_sum"] = grouped["delta_m1"].transform(lambda s: _rolling_sum(s, 6))
    merged["delta_m12_sum"] = grouped["delta_m1"].transform(lambda s: _rolling_sum(s, 12))

    month_gaps = grouped["ym_order"].diff().fillna(1)
    month_gap_values = pd.to_numeric(month_gaps, errors="coerce").fillna(1)
    merged["missing_month_flag"] = month_gap_values > 1

    rolling_mean = grouped["delta_m1"].transform(lambda s: s.rolling(window=6, min_periods=2).mean())
    rolling_std = grouped["delta_m1"].transform(lambda s: s.rolling(window=6, min_periods=2).std(ddof=0))
    merged["delta_zscore_6m"] = (merged["delta_m1"] - rolling_mean) / rolling_std
    merged.loc[rolling_std.isna() | (rolling_std == 0), "delta_zscore_6m"] = float("nan")
    merged["sudden_spike_flag"] = merged["delta_zscore_6m"].abs() >= SPIKE_ZSCORE_THRESHOLD

    merged["hazard_class"] = merged["hazard_class"].fillna("")
    merged["as_of_date"] = merged["as_of_date"].dt.strftime("%Y-%m-%d")

    merged["generated_at_utc"] = generated_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    output = merged[
        [
            "iso3",
            "hazard_code",
            "metric",
            "ym",
            "delta_m1",
            "delta_m3_sum",
            "delta_m6_sum",
            "delta_m12_sum",
            "delta_zscore_6m",
            "sudden_spike_flag",
            "missing_month_flag",
            "as_of_date",
            "as_of_recency_days",
            "source_tier",
            "hazard_class",
            "generated_at_utc",
        ]
    ].copy()

    output.rename(columns={"iso3": "country_iso3"}, inplace=True)
    output.sort_values(["country_iso3", "hazard_code", "metric", "ym"], inplace=True)
    output.reset_index(drop=True, inplace=True)

    return output


def write_feature_artifacts(features: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    csv_path = output_path.with_suffix(".csv")
    features.to_csv(csv_path, index=False)


def build_features(
    *,
    snapshots_dir: Path = DEFAULT_SNAPSHOTS,
    output_path: Path = DEFAULT_OUTPUT,
    db_url: Optional[str] = None,
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    inputs = load_feature_inputs(snapshots_dir=snapshots_dir, db_url=db_url)
    frame = compute_feature_frame(
        resolved=inputs.resolved,
        deltas=inputs.deltas,
        metrics=metrics,
        generated_at=inputs.generated_at,
    )
    write_feature_artifacts(frame, output_path)
    return frame


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots-dir", default=str(DEFAULT_SNAPSHOTS), help="Resolver snapshots directory")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to resolver_features.parquet")
    parser.add_argument("--db-url", default=None, help="Optional DuckDB URL to read facts_resolved/facts_deltas")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics to include (defaults to in_need,affected,displaced)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    metrics = _parse_metrics([args.metrics])
    try:
        frame = build_features(
            snapshots_dir=Path(args.snapshots_dir),
            output_path=Path(args.output),
            db_url=args.db_url,
            metrics=metrics,
        )
    except FeatureBuildError as exc:
        print(f"Feature build failed: {exc}", file=sys.stderr)
        return 2
    print(f"✅ Wrote {len(frame)} resolver features to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
