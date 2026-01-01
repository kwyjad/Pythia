# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from pythia.buckets import BUCKET_SPECS


def bucket_index_expr(metric_expr: str, bucket_expr: str, *, bucket_is_label: bool) -> str:
    if not bucket_is_label:
        return bucket_expr

    parts = []
    bucket_text = f"LOWER(CAST({bucket_expr} AS VARCHAR))"
    for metric, specs in BUCKET_SPECS.items():
        metric_cases = "\n".join(
            f"WHEN {bucket_text} = '{spec.label.lower()}' THEN {int(spec.idx)}"
            for spec in specs
        )
        parts.append(
            f"WHEN UPPER({metric_expr}) = '{metric}' THEN CASE {metric_cases} ELSE NULL END"
        )
    return "CASE\n  " + "\n  ".join(parts) + "\n  ELSE NULL\nEND"


def fallback_centroid_expr(metric_expr: str, bucket_index_expr: str) -> str:
    parts = []
    for metric, specs in BUCKET_SPECS.items():
        metric_cases = "\n".join(
            f"WHEN {bucket_index_expr} = {int(spec.idx)} THEN {float(spec.centroid)}"
            for spec in specs
        )
        parts.append(
            f"WHEN UPPER({metric_expr}) = '{metric}' THEN CASE {metric_cases} ELSE NULL END"
        )
    return "CASE\n  " + "\n  ".join(parts) + "\n  ELSE NULL\nEND"


def build_centroid_join(
    *,
    base_alias: str,
    metric_expr: str,
    hazard_expr: str,
    bucket_expr: str,
    bucket_is_label: bool,
    bc_bucket_col: str = "bucket_index",
    bc_centroid_col: str = "centroid",
) -> tuple[str, str, str]:
    bucket_index = bucket_index_expr(metric_expr, bucket_expr, bucket_is_label=bucket_is_label)
    fallback_expr = fallback_centroid_expr(metric_expr, bucket_index)
    join_sql = (
        "LEFT JOIN bucket_centroids bc_exact ON "
        f"UPPER(bc_exact.hazard_code) = UPPER({hazard_expr}) "
        f"AND UPPER(bc_exact.metric) = UPPER({metric_expr}) "
        f"AND bc_exact.{bc_bucket_col} = {bucket_index} "
        "LEFT JOIN bucket_centroids bc_any ON "
        "UPPER(bc_any.hazard_code) = '*' "
        f"AND UPPER(bc_any.metric) = UPPER({metric_expr}) "
        f"AND bc_any.{bc_bucket_col} = {bucket_index}"
    )
    centroid_expr = (
        f"COALESCE(bc_exact.{bc_centroid_col}, bc_any.{bc_centroid_col}, {fallback_expr})"
    )
    return join_sql, centroid_expr, bucket_index


def centroid_from_defaults(metric: str, bucket_index: int) -> float | None:
    specs = BUCKET_SPECS.get(metric.upper())
    if not specs:
        return None
    for spec in specs:
        if int(spec.idx) == int(bucket_index):
            return float(spec.centroid)
    return None
