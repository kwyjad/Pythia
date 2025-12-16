# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""
Config-driven precedence selection for monthly admin0 series.

- Pure library; no side-effects.
- Consumes:
    - pandas.DataFrame of candidate rows across sources
      required cols: iso3, as_of_date (datetime64), metric, value (float),
                     source_system, collection_type, coverage, freshness_days,
                     origin_iso3, destination_iso3, method_note,
                     plus optional discriminator keys (indicator_kind, series, indicator).
    - precedence config (dict), loaded from YAML.

- Produces:
    - DataFrame with one row per (iso3, ym, output_metric) following precedence.
    - For conflict_onset1_pa assembles components per config (sum).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

COVERAGE_RANKS_FALLBACK = {"national": 3, "corridor": 2, "site": 1, "unknown": 0, None: 0}


@dataclass
class PrecedenceConfig:
    raw: Dict[str, Any]

    @property
    def coverage_ranks(self) -> Dict[str, int]:
        return self.raw.get("coverage_ranks", COVERAGE_RANKS_FALLBACK)

    def shock_specs(self) -> Dict[str, Dict[str, Any]]:
        return self.raw.get("shocks", {})

    def default_tie_breakers(self) -> List[Dict[str, str]]:
        return self.raw.get("defaults", {}).get("tie_breakers", [])

    def metadata_defaults(self) -> Dict[str, Any]:
        return self.raw.get("defaults", {}).get("metadata_defaults", {})


def _to_ym(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.tz_localize(None).dt.to_period("M").astype(str)


def _ensure_columns(df: pd.DataFrame, defaults: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for column, default_value in defaults.items():
        if column not in out.columns:
            out[column] = default_value
    return out


def _coverage_rank(series: pd.Series, ranks: Dict[str, int]) -> pd.Series:
    return series.map(lambda value: ranks.get(value, ranks.get("unknown", 0)))


def _apply_transform(subset: pd.DataFrame, transform_spec: Optional[Dict[str, Any]]) -> pd.DataFrame:
    if not transform_spec:
        return subset

    transform_type = transform_spec.get("type", "passthrough")

    if transform_type == "passthrough":
        return subset

    if transform_type == "delta_from_stock":
        group_keys = transform_spec.get("group_keys", ["iso3"])
        guards = transform_spec.get("guards", {})
        working = subset.sort_values(group_keys + ["as_of_date"])
        working["value"] = working.groupby(group_keys)["value"].diff()

        if guards.get("suppress_negative"):
            working.loc[working["value"] < 0, "value"] = np.nan

        working = working.dropna(subset=["value"])

        flag_name = guards.get("add_flag")
        if flag_name:
            working[flag_name] = 1

        return working

    raise ValueError(f"Unsupported transform type: {transform_type}")


def _select_component(df: pd.DataFrame, spec: Dict[str, Any], cfg: PrecedenceConfig) -> pd.DataFrame:
    candidates = spec.get("candidates", [])
    if not candidates:
        return df.iloc[0:0].copy()

    tie_breakers = spec.get("tie_breakers_override", cfg.default_tie_breakers())
    ranks = cfg.coverage_ranks

    pools: List[pd.DataFrame] = []
    for candidate_spec in candidates:
        match_filters = candidate_spec.get("match", {})
        subset = df
        for key, expected in match_filters.items():
            if key not in subset.columns:
                subset = subset.iloc[0:0]
                break
            subset = subset[subset[key] == expected]
        if subset.empty:
            continue

        subset = subset.copy()
        subset["_candidate_source"] = str(match_filters)
        transformed = _apply_transform(subset, candidate_spec.get("transform"))
        if transformed.empty:
            continue
        pools.append(transformed)

    if not pools:
        return df.iloc[0:0].copy()

    pool = pd.concat(pools, ignore_index=True)
    pool["ym"] = _to_ym(pool["as_of_date"])

    if "coverage" in pool.columns and "coverage_rank" not in pool.columns:
        pool["coverage_rank"] = _coverage_rank(pool["coverage"], ranks)

    for tie_breaker in tie_breakers:
        key = tie_breaker["key"]
        if key not in pool.columns:
            pool[key] = np.nan if tie_breaker.get("order", "desc") == "desc" else np.inf

    sort_columns = ["iso3", "ym"] + [tb["key"] for tb in tie_breakers]
    ascending = [True, True] + [tb.get("order", "desc") == "asc" for tb in tie_breakers]

    ordered = pool.sort_values(by=sort_columns, ascending=ascending, kind="mergesort")
    selected = ordered.drop_duplicates(subset=["iso3", "ym"], keep="first").reset_index(drop=True)

    return selected


def _prepare_component_details(component_df: pd.DataFrame, metadata_fields: Iterable[str]) -> pd.DataFrame:
    detail_columns = ["iso3", "ym", "value", "_component"] + list(metadata_fields)
    working = component_df.copy()
    for column in detail_columns:
        if column not in working.columns:
            working[column] = None
    return working[detail_columns]


def apply_precedence(candidates: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    cfg = PrecedenceConfig(config)
    metadata_defaults = cfg.metadata_defaults()
    df = _ensure_columns(candidates, metadata_defaults)
    df["ym"] = _to_ym(df["as_of_date"])

    metadata_fields = list(metadata_defaults.keys())
    results: List[pd.DataFrame] = []
    metric_cache: Dict[str, pd.DataFrame] = {}

    for shock_name, shock_spec in cfg.shock_specs().items():
        output_metric = shock_spec.get("output_metric", shock_name)
        semantics = shock_spec.get("semantics", "new")
        assemble = shock_spec.get("assemble", "single")
        components = shock_spec.get("components", {})

        component_frames: Dict[str, pd.DataFrame] = {}
        component_details: List[pd.DataFrame] = []

        for component_name, component_spec in components.items():
            if "mirror_from" in component_spec:
                reference_metric = component_spec["mirror_from"]
                referenced = metric_cache.get(reference_metric)
                if referenced is None or referenced.empty:
                    component_frames[component_name] = referenced if referenced is not None else df.iloc[0:0].copy()
                    continue

                mirrored = referenced.copy()
                credit_key = component_spec.get("credit_to")
                if credit_key:
                    if credit_key not in mirrored.columns:
                        mirrored = mirrored.iloc[0:0]
                    else:
                        mirrored = mirrored.dropna(subset=[credit_key]).copy()
                        mirrored["iso3"] = mirrored[credit_key]
                mirrored["_component"] = component_name
                component_frames[component_name] = mirrored
                component_details.append(_prepare_component_details(mirrored.copy(), metadata_fields))
                continue

            selected = _select_component(df, component_spec, cfg)
            if selected.empty:
                component_frames[component_name] = selected
                continue

            selected = selected.copy()
            selected["_component"] = component_name
            component_frames[component_name] = selected
            component_details.append(_prepare_component_details(selected.copy(), metadata_fields))

        if assemble == "sum_components":
            value_frames = [frame[["iso3", "ym", "value"]] for frame in component_frames.values() if not frame.empty]
            if not value_frames:
                metric_cache[output_metric] = pd.DataFrame(columns=["iso3", "ym", "value"])
                continue

            summed = (
                pd.concat(value_frames, ignore_index=True)
                .groupby(["iso3", "ym"], as_index=False)["value"].sum()
            )
            summed["metric"] = output_metric
            summed["semantics"] = semantics

            for field in metadata_fields:
                summed[field] = None
            if component_details:
                detail_df = pd.concat(component_details, ignore_index=True)
                detail_map = {
                    (iso3, ym): group.drop(columns=["iso3", "ym"]).to_dict("records")
                    for (iso3, ym), group in detail_df.groupby(["iso3", "ym"])
                }
                summed["component_sources"] = [
                    detail_map.get((row.iso3, row.ym), [])
                    for row in summed.itertuples(index=False)
                ]
            else:
                summed["component_sources"] = [[] for _ in range(len(summed))]

            results.append(summed)
            metric_cache[output_metric] = summed
            continue

        # assemble == single (default)
        main = component_frames.get("main")
        if main is None or main.empty:
            metric_cache[output_metric] = main if main is not None else pd.DataFrame()
            continue

        picked = main.copy()
        picked["metric"] = output_metric
        picked["semantics"] = semantics
        missing_metadata = [field for field in metadata_fields if field not in picked.columns]
        for field in missing_metadata:
            picked[field] = None

        results.append(picked)
        metric_cache[output_metric] = picked

    if not results:
        return df.iloc[0:0].copy()

    final = pd.concat(results, ignore_index=True, sort=False)
    final["as_of_date"] = pd.PeriodIndex(final["ym"], freq="M").to_timestamp("M")

    ordered_columns = ["iso3", "as_of_date", "ym", "metric", "value", "semantics"] + metadata_fields
    if "component_sources" in final.columns:
        ordered_columns.append("component_sources")
    final = final[ordered_columns].sort_values(["metric", "iso3", "as_of_date"]).reset_index(drop=True)

    return final
