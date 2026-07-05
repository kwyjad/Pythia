#!/usr/bin/env python3
# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""
precedence_engine.py — resolve one authoritative fact per
(country_iso3, hazard_type, month, metric) from multi-source candidates.

The single entry point is :func:`resolve_facts_frame`, called by
``resolver/tools/run_pipeline.py`` with the config loaded from
``resolver/tools/precedence_config.yml`` (tiers + tiebreak + defaults).

History: this module used to also contain a standalone CLI
(``--facts ... --cutoff ...``) implementing a different, pre-connector-era
rule set (one metric per country/hazard via ``metric_preference``,
publication-lag cutoffs, ``source_mapping`` tier matching). That path had
no remaining code callers and consumed config keys the pipeline engine
ignored — a documented source of confusion (July 2026 audit) — and was
removed. One engine, one config schema. (The separate
``tools/precedence_engine.py`` at the repo root is a different library used
by the IDMC ingestion CLI, with its own config.)
"""

import datetime as dt
import os
import sys
from typing import List, Dict, Any, Optional
from zoneinfo import ZoneInfo

try:
    import pandas as pd
except ImportError:
    print("Please 'pip install pandas pyarrow python-dateutil' to run the engine.", file=sys.stderr)
    sys.exit(2)

try:
    from dateutil import parser as date_parser
except ImportError:
    print("Please 'pip install python-dateutil' to run the engine.", file=sys.stderr)
    sys.exit(2)

ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")

def _parse_as_of(value: str) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = date_parser.isoparse(str(value))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=ISTANBUL_TZ)
    return parsed.astimezone(ISTANBUL_TZ)


def _tier_mappings(cfg: Dict[str, Any]) -> tuple[Dict[str, int], Dict[int, str]]:
    tiers_cfg = cfg.get("tiers", [])
    source_to_tier: Dict[str, int] = {}
    tier_index_to_name: Dict[int, str] = {}
    for idx, tier in enumerate(tiers_cfg):
        tier_name = str(tier.get("name", f"Tier {idx}"))
        tier_index_to_name[idx] = tier_name
        for source in tier.get("sources", []) or []:
            source_to_tier[str(source).lower()] = idx
    return source_to_tier, tier_index_to_name


def _coerce_numeric(value: Any) -> Optional[float | int]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not pd.isna(value):
            parsed = float(value)
        else:
            parsed = float(str(value))
        if pd.isna(parsed):
            return None
        if parsed.is_integer():
            return int(parsed)
        return parsed
    except (TypeError, ValueError):
        return None


def resolve_facts_frame(
    df: pd.DataFrame,
    config: Dict[str, Any],
    overrides: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Resolve one fact per (country_iso3, hazard_type, month, metric).

    This helper mirrors the precedence policy logic with deterministic tie
    breaks so it can be exercised directly in unit tests without going through
    the CLI entrypoint. The public CLI behaviour remains unchanged.
    """

    if df is None:
        raise ValueError("df must be a pandas DataFrame")

    required = ["country_iso3", "hazard_type", "month", "metric", "value", "as_of", "source"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"resolve_facts_frame requires columns {missing}")

    work = df.copy()
    for col in required + ["run_id"]:
        if col in work.columns:
            work[col] = work[col].astype(str).fillna("")

    work["value_num"] = pd.to_numeric(work["value"], errors="coerce")
    work["as_of_dt"] = work["as_of"].apply(_parse_as_of)
    work["source_norm"] = work["source"].str.lower()

    source_to_tier, tier_index_to_name = _tier_mappings(config)
    default_tier = len(tier_index_to_name)
    work["tier_index"] = work["source_norm"].map(source_to_tier).fillna(default_tier)

    # Default tiebreaks when the config omits the key: recency then non-null
    # value. Without this, within-tier winners fell through to alphabetical
    # source order, so a stale or NULL-value row could beat a fresh one.
    tiebreak = config.get("tiebreak") or ["as_of_desc", "value_nonnull"]
    metric_overrides = config.get("metrics", {})
    defaults = config.get("defaults", {})
    prefer_full_row = bool(defaults.get("prefer_full_row_coverage", False))
    require_nonnegative = bool(defaults.get("require_nonnegative", False))
    debug_enabled = os.getenv("RESOLVER_PRECEDENCE_DEBUG", "0") == "1"

    result_rows: List[Dict[str, Any]] = []

    group_cols = ["country_iso3", "hazard_type", "month", "metric"]
    for key, group in work.groupby(group_cols, dropna=False, sort=False):
        candidates = group.copy()

        if prefer_full_row:
            nonnull = candidates[candidates["value_num"].notna()]
            if not nonnull.empty:
                candidates = nonnull

        if require_nonnegative:
            valid = candidates[(candidates["value_num"].isna()) | (candidates["value_num"] >= 0)]
            if not valid.empty:
                candidates = valid

        metric_cfg = metric_overrides.get(key[3], {}) if metric_overrides else {}
        prefer_sources = [str(s).lower() for s in metric_cfg.get("prefer_sources", [])]
        if prefer_sources:
            preferred = candidates[candidates["source_norm"].isin(prefer_sources)]
            if not preferred.empty:
                candidates = preferred

        best_tier = candidates["tier_index"].min()
        candidates = candidates[candidates["tier_index"] == best_tier].copy()

        if candidates.empty:
            continue

        tie_path: List[str] = []
        for rule in tiebreak:
            if len(candidates) <= 1:
                break
            if rule == "as_of_desc":
                candidates["_as_of_dt"] = candidates["as_of_dt"]
                max_as_of = candidates["_as_of_dt"].max()
                candidates = candidates[candidates["_as_of_dt"] == max_as_of]
                candidates = candidates.drop(columns=["_as_of_dt"], errors="ignore")
                tie_path.append(rule)
            elif rule == "value_nonnull":
                candidates["_nonnull"] = candidates["value_num"].notna()
                if candidates["_nonnull"].any():
                    candidates = candidates[candidates["_nonnull"]]
                    tie_path.append(rule)
                candidates = candidates.drop(columns=["_nonnull"], errors="ignore")
            elif rule == "source_alpha":
                source_min = candidates["source_norm"].fillna("").min()
                candidates = candidates[candidates["source_norm"] == source_min]
                tie_path.append(rule)

        chosen = candidates.sort_values(
            by=["source_norm", "run_id"], ascending=[True, True], kind="mergesort"
        ).iloc[0]

        tier_idx = int(chosen.get("tier_index", default_tier))
        tier_name = tier_index_to_name.get(tier_idx, "Unlisted")
        value_num = chosen.get("value_num")
        resolved_value = _coerce_numeric(value_num)

        record = {
            "country_iso3": key[0],
            "hazard_type": key[1],
            "month": key[2],
            "metric": key[3],
            "value": resolved_value,
            "selected_source": chosen.get("source", ""),
            "selected_as_of": str(chosen.get("as_of", "")),
            "selected_run_id": chosen.get("run_id", ""),
            "selected_tier": tier_name,
        }

        # Carry through metadata fields from the chosen row so they
        # survive into facts_resolved when called from run_pipeline.
        _PASSTHROUGH_FIELDS = (
            "publisher", "source_type", "source_url", "confidence",
            "definition_text", "doc_title", "hazard_label", "hazard_class",
            "unit", "series_semantics", "publication_date", "event_id",
            "proxy_for", "alertlevel",
        )
        for _field in _PASSTHROUGH_FIELDS:
            if _field in chosen.index:
                record[_field] = chosen[_field]

        # Map selected_* to DB column names for the run_pipeline path.
        record["as_of_date"] = record["selected_as_of"]
        record["precedence_tier"] = record["selected_tier"]
        record["provenance_source"] = record["selected_source"]

        if debug_enabled:
            record["tie_break_path"] = ">".join(tie_path)

        result_rows.append(record)

    resolved = pd.DataFrame(result_rows)

    if overrides is not None and not overrides.empty:
        overrides_work = overrides.copy()
        for col in ["country_iso3", "hazard_type", "month", "metric"]:
            overrides_work[col] = overrides_work[col].astype(str)
        overrides_work["override_value"] = overrides_work["override_value"].apply(_coerce_numeric)
        override_map = {
            (row.country_iso3, row.hazard_type, row.month, row.metric): row
            for row in overrides_work.itertuples(index=False)
        }

        for idx, res_row in resolved.iterrows():
            key = (
                res_row["country_iso3"],
                res_row["hazard_type"],
                res_row["month"],
                res_row["metric"],
            )
            override = override_map.get(key)
            if override and override.override_value is not None:
                resolved.at[idx, "value"] = override.override_value
                resolved.at[idx, "selected_source"] = "review_override"
                resolved.at[idx, "selected_as_of"] = ""
                resolved.at[idx, "selected_run_id"] = "override"
                resolved.at[idx, "selected_tier"] = "Override"
                resolved.at[idx, "override_note"] = getattr(override, "note", "")

    if not resolved.empty:
        resolved = resolved.sort_values(group_cols, kind="mergesort").reset_index(drop=True)

    return resolved
