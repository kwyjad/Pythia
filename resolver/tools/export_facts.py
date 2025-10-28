#!/usr/bin/env python3
"""
export_facts.py — normalize arbitrary staging inputs into resolver 'facts' outputs.

Examples:
  # Export from a folder of CSVs using the default mapping config:
  python resolver/tools/export_facts.py --in resolver/staging --out resolver/exports

  # Export from a single file with explicit config:
  python resolver/tools/export_facts.py \
      --in resolver/staging/sample_source.csv \
      --config resolver/tools/export_config.yml \
      --out resolver/exports

What it does:
  - Loads one or many staging files (.csv, .parquet, .json lines)
  - Applies a column mapping + constant fills from export_config.yml
  - Coerces datatypes and enums to match the Data Dictionary
  - Writes 'facts.csv' and 'facts.parquet' to the /resolver/exports/ folder
  - Prints how many rows written and where

After exporting:
  - Validate:   python resolver/tools/validate_facts.py --facts resolver/exports/facts.csv
  - Freeze:     python resolver/tools/freeze_snapshot.py --facts resolver/exports/facts.csv --month YYYY-MM
"""

import argparse
import datetime as dt
import json
import logging
import numbers
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

try:
    import pandas as pd
except ImportError:
    print("Please 'pip install pandas pyarrow pyyaml' to run the exporter.", file=sys.stderr)
    sys.exit(2)

_DTM_ALIAS_MAP = {
    "CountryISO3": ["CountryISO3", "country_iso3", "iso3"],
    "ReportingDate": [
        "ReportingDate",
        "as_of",
        "as_of_date",
        "reporting_date",
        "date",
        "month_start",
    ],
    "idp_count": ["idp_count", "numPresentIdpInd", "IDPTotal", "TotalIDPs", "value"],
}


def _first_present(
    df: "pd.DataFrame", candidates: Iterable[str]
) -> tuple["pd.Series", Optional[str]]:
    for candidate in candidates:
        if candidate in df.columns:
            return df[candidate], candidate
    return pd.Series([pd.NA] * len(df), index=df.index), None


def _normalize_dtm_admin0(
    df: "pd.DataFrame",
) -> tuple["pd.DataFrame", Dict[str, Any]]:
    iso_series, iso_source = _first_present(df, _DTM_ALIAS_MAP["CountryISO3"])
    iso = iso_series.astype(str).str.strip().str.upper()

    raw_date, date_source = _first_present(df, _DTM_ALIAS_MAP["ReportingDate"])
    parsed_dates = pd.to_datetime(raw_date, errors="coerce", utc=False)
    month_end = parsed_dates + pd.offsets.MonthEnd(0)
    formatted_dates = month_end.dt.strftime("%Y-%m-%d")

    raw_value, value_source = _first_present(df, _DTM_ALIAS_MAP["idp_count"])
    numeric_values = pd.to_numeric(raw_value, errors="coerce")

    valid_iso = iso.str.len() == 3
    valid_dates = parsed_dates.notna()
    valid_values = numeric_values.notna()
    keep_mask = valid_iso & valid_dates & valid_values
    positive_mask = numeric_values > 0
    filtered_mask = keep_mask & positive_mask

    filtered = pd.DataFrame(
        {
            "iso3": iso.where(filtered_mask).dropna(),
            "as_of_date": formatted_dates.where(filtered_mask).dropna(),
            "value": numeric_values.where(filtered_mask).dropna(),
        }
    ).reset_index(drop=True)

    metadata: Dict[str, Any] = {
        "rows_in": int(len(df)),
        "rows_after_filters": int(len(filtered)),
        "filters_applied": ["keep_if_not_null", "keep_if_positive"],
        "drop_histogram": {
            "keep_if_not_null": int((~keep_mask).sum()),
            "keep_if_positive": int((keep_mask & ~positive_mask).sum()),
        },
        "sources": {
            "iso3": iso_source,
            "as_of_date": date_source,
            "value": value_source,
        },
    }

    return filtered, metadata


def _map_dtm_displacement_admin0(
    frame: "pd.DataFrame",
) -> tuple["pd.DataFrame", Dict[str, Any]]:
    filtered, metadata = _normalize_dtm_admin0(frame)
    if filtered.empty:
        empty = pd.DataFrame(
            columns=[
                "iso3",
                "as_of_date",
                "ym",
                "metric",
                "value",
                "semantics",
                "series_semantics",
                "source",
            ]
        )
        metadata.update({
            "rows_after_aggregate": 0,
            "dedupe_keys": ["iso3", "as_of_date", "metric"],
            "aggregate_funcs": {"value": "max"},
        })
        return empty, metadata

    working = filtered.copy()
    working["metric"] = "idp_displacement_stock_dtm"
    working["semantics"] = "stock"
    working["series_semantics"] = "stock"
    working["source"] = "IOM DTM"

    aggregated = (
        working.groupby(["iso3", "as_of_date", "metric"], as_index=False, sort=False)[
            "value"
        ]
        .max()
        .reset_index(drop=True)
    )
    aggregated["value"] = aggregated["value"].map(_format_numeric_string)
    aggregated["semantics"] = "stock"
    aggregated["series_semantics"] = "stock"
    aggregated["source"] = "IOM DTM"
    aggregated["ym"] = pd.to_datetime(
        aggregated["as_of_date"], errors="coerce", utc=False
    ).dt.strftime("%Y-%m")

    columns_order = [
        "iso3",
        "as_of_date",
        "ym",
        "metric",
        "value",
        "semantics",
        "series_semantics",
        "source",
    ]
    aggregated = aggregated[columns_order]

    metadata.update(
        {
            "rows_after_aggregate": int(len(aggregated)),
            "dedupe_keys": ["iso3", "as_of_date", "metric"],
            "aggregate_funcs": {"value": "max"},
        }
    )

    return aggregated, metadata

try:  # Optional dependency for DB dual-write
    from resolver.db import duckdb_io
except Exception:  # pragma: no cover - allow exporter without duckdb installed
    duckdb_io = None

from resolver.common import (
    compute_series_semantics,
    get_logger,
    dict_counts,
    df_schema,
)
from resolver.helpers.series_semantics import normalize_series_semantics

try:
    import yaml
except ImportError:
    print("Please 'pip install pyyaml' to run the exporter.", file=sys.stderr)
    sys.exit(2)

ROOT = Path(__file__).resolve().parents[1]     # .../resolver
TOOLS = ROOT / "tools"
EXPORTS = ROOT / "exports"
DEFAULT_CONFIG = TOOLS / "export_config.yml"

REQUIRED = [
    "event_id",
    "country_name",
    "iso3",
    "hazard_code",
    "hazard_label",
    "hazard_class",
    "metric",
    "value",
    "unit",
    "as_of_date",
    "publication_date",
    "publisher",
    "source_type",
    "source_url",
    "doc_title",
    "definition_text",
    "method",
    "confidence",
    "revision",
    "ingested_at",
]

CANONICAL_CORE = {"iso3", "metric", "value", "as_of_date"}


def _normalize_export_df(df: "pd.DataFrame") -> "pd.DataFrame":
    if df is None or df.empty:
        return df

    required = ["iso3", "as_of_date", "metric", "value"]
    for col in required:
        if col not in df.columns:
            df[col] = None

    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()
    df["as_of_date"] = pd.to_datetime(df["as_of_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["metric"] = df["metric"].astype(str).str.strip()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["iso3", "as_of_date", "metric", "value"])
    df = df[df["iso3"].str.len() == 3]
    df = df[df["value"] > 0]

    return df


def _format_numeric_string(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, numbers.Number):
        float_val = float(value)
        if float_val.is_integer():
            return str(int(float_val))
        return str(float_val)
    return str(value)


def _map_dtm_admin0_fallback(frame: "pd.DataFrame") -> tuple["pd.DataFrame", Dict[str, Any]]:
    cols = {"CountryISO3", "ReportingDate", "idp_count"}
    metadata = {
        "rows_after_filters": 0,
        "rows_after_aggregate": 0,
        "rows_after_dedupe": 0,
        "filters_applied": ["keep_if_not_null", "keep_if_positive"],
        "drop_histogram": {"keep_if_not_null": 0, "keep_if_positive": 0},
        "aggregate_keys": ["iso3", "as_of_date", "metric"],
        "aggregate_funcs": {"value": "max"},
        "aggregate_rows_before": 0,
    }

    empty_df = pd.DataFrame(
        columns=["iso3", "as_of_date", "metric", "value", "series_semantics", "semantics", "source", "ym"]
    )

    if frame is None or frame.empty or not cols.issubset(set(frame.columns)):
        return empty_df, metadata

    iso_series = frame["CountryISO3"].astype(str).str.strip().str.upper()
    parsed_dates = pd.to_datetime(frame["ReportingDate"], errors="coerce")
    value_series = pd.to_numeric(frame["idp_count"], errors="coerce")

    valid_iso = iso_series.str.len() == 3
    valid_dates = parsed_dates.notna()
    valid_values = value_series.notna()
    not_null_mask = valid_iso & valid_dates & valid_values
    positive_mask = value_series > 0
    filters_mask = not_null_mask & positive_mask

    metadata["drop_histogram"]["keep_if_not_null"] = int((~not_null_mask).sum())
    metadata["drop_histogram"]["keep_if_positive"] = int((not_null_mask & ~positive_mask).sum())

    if not filters_mask.any():
        return empty_df, metadata

    filtered = pd.DataFrame(
        {
            "iso3": iso_series[filters_mask],
            "as_of_date": parsed_dates[filters_mask],
            "metric": "idps_stock",
            "value": value_series[filters_mask],
        }
    )
    filtered["series_semantics"] = "stock"
    filtered["semantics"] = "stock"
    filtered["source"] = "IOM DTM"

    normalized = _normalize_export_df(filtered)
    metadata["rows_after_filters"] = int(len(normalized))
    metadata["aggregate_rows_before"] = metadata["rows_after_filters"]

    if normalized.empty:
        return empty_df, metadata

    aggregated = (
        normalized.groupby(["iso3", "as_of_date", "metric"], as_index=False, sort=False)
        .agg({"value": "max"})
        .reset_index(drop=True)
    )
    aggregated["series_semantics"] = "stock"
    aggregated["semantics"] = "stock"
    aggregated["source"] = "IOM DTM"
    aggregated["as_of_date"] = pd.to_datetime(aggregated["as_of_date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    aggregated["ym"] = pd.to_datetime(aggregated["as_of_date"], errors="coerce").dt.strftime("%Y-%m")
    aggregated["value"] = aggregated["value"].apply(_format_numeric_string)

    for column in REQUIRED:
        if column not in aggregated.columns:
            aggregated[column] = ""

    metadata["rows_after_aggregate"] = int(len(aggregated))
    metadata["rows_after_dedupe"] = int(len(aggregated))

    ordered_columns = [
        "iso3",
        "as_of_date",
        "ym",
        "metric",
        "value",
        "semantics",
        "series_semantics",
        "source",
    ]
    existing = [col for col in ordered_columns if col in aggregated.columns]
    remaining = [col for col in aggregated.columns if col not in existing]
    aggregated = aggregated[existing + remaining]

    return aggregated.reset_index(drop=True), metadata

def _load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _read_one(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".csv", ".tsv"]:
        return pd.read_csv(path, dtype=str).fillna("")
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in [".json", ".jsonl"]:
        try:
            frame = pd.read_json(path, lines=True, dtype_backend="pyarrow")
        except ValueError:
            try:
                frame = pd.read_json(path, lines=False, dtype_backend="pyarrow")
            except ValueError:
                return pd.DataFrame()
        if frame is None:
            return pd.DataFrame()
        if isinstance(frame, pd.Series):
            frame = frame.to_frame().T
        return frame.astype(str).fillna("")
    raise SystemExit(f"Unsupported input extension: {ext}")

def _collect_inputs(inp: Path) -> tuple[List[Path], List[Path]]:
    skipped_meta: List[Path] = []
    if inp.is_dir():
        files: List[Path] = []
        for p in inp.rglob("*"):
            if p.name.endswith(".meta.json"):
                skipped_meta.append(p)
                continue
            if p.suffix.lower() in (".csv", ".tsv", ".parquet", ".json", ".jsonl"):
                files.append(p)
        return files, skipped_meta
    if inp.name.endswith(".meta.json"):
        return [], [inp]
    return [inp], []


def _relativize_path(path: Path) -> str:
    candidates = [ROOT.parent, Path.cwd()]
    for base in candidates:
        try:
            return path.resolve().relative_to(base.resolve()).as_posix()
        except ValueError:
            continue
    return path.resolve().as_posix()


@dataclass
class SourceApplication:
    name: str
    path: Path
    rows_in: int
    rows_mapped: int = 0
    rows_after_filters: int = 0
    rows_after_aggregate: int = 0
    rows_after_dedupe: int = 0
    strategy: str = ""
    warnings: List[str] = field(default_factory=list)
    filters_applied: List[str] = field(default_factory=list)
    drop_histogram: Dict[str, int] = field(default_factory=dict)
    dedupe_keys: List[str] = field(default_factory=list)
    dedupe_keep: str = "last"
    aggregate_keys: List[str] = field(default_factory=list)
    aggregate_funcs: Dict[str, str] = field(default_factory=dict)
    mapping_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def rows_out(self) -> int:
        return self.rows_after_dedupe

    def as_report_entry(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "source": self.name,
            "strategy": self.strategy,
            "rows_in": self.rows_in,
            "rows_mapped": self.rows_mapped,
            "rows_after_filters": self.rows_after_filters,
            "rows_after_aggregate": self.rows_after_aggregate or self.rows_after_filters,
            "rows_after_dedupe": self.rows_after_dedupe,
            "filters": list(dict.fromkeys(self.filters_applied)),
            "dedupe": {
                "keys": self.dedupe_keys,
                "keep": self.dedupe_keep,
            }
            if self.dedupe_keys
            else None,
            "aggregate": {
                "keys": self.aggregate_keys,
                "funcs": self.aggregate_funcs,
                "rows_before": self.rows_after_filters,
                "rows_after": self.rows_after_aggregate or self.rows_after_filters,
            }
            if self.aggregate_keys or self.aggregate_funcs
            else None,
            "drop_histogram": {
                key: int(value) for key, value in (self.drop_histogram or {}).items()
            },
            "warnings": list(self.warnings or []),
            "mapping": self.mapping_details,
        }


def _ensure_iterable(value: Any) -> Sequence[Any]:
    if isinstance(value, (list, tuple)):
        return value
    if value is None:
        return []
    return [value]


def _coerce_string_series(series: "pd.Series", *, preserve_index: bool = True) -> "pd.Series":
    if not isinstance(series, pd.Series):
        values = list(series) if isinstance(series, Iterable) and not isinstance(series, (str, bytes)) else [series]
        idx = None
        if preserve_index and isinstance(series, pd.Series):
            idx = series.index
        return pd.Series(values, index=idx, dtype="object").astype(str).fillna("")
    return series.astype(str).fillna("")


def _apply_op(series: "pd.Series", op: str) -> "pd.Series":
    op_normalized = (op or "").strip().lower()
    if op_normalized in {"uppercase", "upper"}:
        return series.astype(str).str.upper()
    if op_normalized in {"lowercase", "lower"}:
        return series.astype(str).str.lower()
    if op_normalized in {"trim", "strip"}:
        return series.astype(str).str.strip()
    if op_normalized == "to_date":
        parsed = pd.to_datetime(series, errors="coerce", utc=False)
        formatted = parsed.dt.strftime("%Y-%m-%d")
        fallback = series.astype(str).replace({"NaT": "", "nan": "", "NaN": ""})
        return formatted.where(parsed.notna(), fallback).fillna("")
    if op_normalized == "to_ym":
        parsed = pd.to_datetime(series, errors="coerce", utc=False)
        formatted = parsed.dt.strftime("%Y-%m")
        fallback = series.astype(str).str.slice(0, 7)
        return formatted.where(parsed.notna(), fallback).fillna("")
    if op_normalized in {"to_number", "number"}:
        numeric = pd.to_numeric(series, errors="coerce")
        def _format_number(value: Any) -> str:
            if pd.isna(value):
                return ""
            if float(value).is_integer():
                return str(int(value))
            return str(value)
        return numeric.map(_format_number).fillna("")
    return series.astype(str).fillna("")


def _resolve_mapping_series(
    frame: "pd.DataFrame",
    context: Dict[str, "pd.Series"],
    mapping: Mapping[str, Any],
    *,
    length: int,
) -> tuple["pd.Series", Dict[str, Any]]:
    if "const" in mapping:
        value = mapping.get("const", "")
        series = pd.Series([value] * length, index=frame.index, dtype="object").astype(str)
        return series, {"const": str(value)}

    chosen_series: Optional["pd.Series"] = None
    chosen_key: Optional[str] = None
    from_context = False

    for candidate in _ensure_iterable(mapping.get("from_any")):
        candidate_key = str(candidate)
        if candidate_key in context:
            chosen_series = context[candidate_key]
            chosen_key = candidate_key
            from_context = True
            break
        if candidate_key in frame.columns:
            chosen_series = frame[candidate_key]
            chosen_key = candidate_key
            break

    if chosen_series is None:
        source_key = mapping.get("from")
        if source_key is not None:
            source_key = str(source_key)
            if source_key in context:
                chosen_series = context[source_key]
                chosen_key = source_key
                from_context = True
            elif source_key in frame.columns:
                chosen_series = frame[source_key]
                chosen_key = source_key

    if chosen_series is None:
        series = pd.Series([""] * length, index=frame.index, dtype="object")
    else:
        series = chosen_series

    series = _coerce_string_series(series)
    ops_applied = [str(op) for op in _ensure_iterable(mapping.get("ops")) if str(op)]
    for op in ops_applied:
        series = _apply_op(series, op)

    info: Dict[str, Any] = {}
    if chosen_key is not None:
        info["source"] = chosen_key
        if from_context:
            info["from_context"] = True
    else:
        info["source"] = None
        info["missing"] = True
    if ops_applied:
        info["ops"] = ops_applied

    return series, info


def _apply_filters(
    frame: "pd.DataFrame", filters: Sequence[Mapping[str, Any]]
) -> tuple["pd.DataFrame", Dict[str, int], List[str]]:
    filtered = frame
    histogram: Counter[str] = Counter()
    applied: List[str] = []
    for rule in filters or []:
        if not isinstance(rule, Mapping):
            continue
        rule_name = next(iter(rule.keys()), "") if rule else ""
        before = len(filtered)
        if "keep_if_not_null" in rule:
            columns = [str(col) for col in _ensure_iterable(rule.get("keep_if_not_null"))]
            if not columns:
                continue
            mask = pd.Series(True, index=filtered.index)
            for column in columns:
                if column not in filtered.columns:
                    mask &= False
                    continue
                values = filtered[column].astype(str)
                mask &= values.str.strip().ne("")
            filtered = filtered.loc[mask]
        elif "keep_if_positive" in rule:
            columns = [str(col) for col in _ensure_iterable(rule.get("keep_if_positive"))]
            if not columns:
                continue
            mask = pd.Series(True, index=filtered.index)
            for column in columns:
                if column not in filtered.columns:
                    mask &= False
                    continue
                numeric = pd.to_numeric(filtered[column], errors="coerce")
                mask &= numeric > 0
            filtered = filtered.loc[mask]
        else:
            continue
        applied.append(rule_name)
        histogram[rule_name] += max(before - len(filtered), 0)
    return filtered, {key: int(value) for key, value in histogram.items()}, applied


def _apply_aggregate(
    frame: "pd.DataFrame", aggregate_cfg: Mapping[str, Any] | None
) -> tuple["pd.DataFrame", List[str], Dict[str, str], int]:
    if not isinstance(aggregate_cfg, Mapping):
        return frame, [], {}, 0

    keys = [str(col) for col in _ensure_iterable(aggregate_cfg.get("keys")) if str(col)]
    funcs_cfg = aggregate_cfg.get("funcs")
    configured_funcs: Dict[str, str] = {}
    if isinstance(funcs_cfg, Mapping):
        for column, func in funcs_cfg.items():
            column_name = str(column)
            func_name = str(func)
            if column_name:
                configured_funcs[column_name] = func_name

    if not keys:
        return frame, [], configured_funcs, 0

    agg_map: Dict[str, Any] = {}
    working = frame
    numeric_funcs = {"max", "min", "sum", "mean", "median"}
    for column, func in configured_funcs.items():
        if column in frame.columns:
            agg_map[column] = func
            func_normalized = str(func).strip().lower()
            if func_normalized in numeric_funcs:
                converted = pd.to_numeric(frame[column], errors="coerce")
                if not converted.isna().all():
                    if working is frame:
                        working = frame.copy()
                    working[column] = converted

    other_columns = [
        column
        for column in frame.columns
        if column not in keys and column not in agg_map
    ]
    for column in other_columns:
        agg_map[column] = "first"

    grouped = working.groupby(keys, dropna=False, sort=False)
    aggregated = grouped.agg(agg_map)
    if isinstance(aggregated.columns, pd.MultiIndex):
        aggregated.columns = [
            "_".join(str(part) for part in col if str(part)) for col in aggregated.columns
        ]
    aggregated = aggregated.reset_index()

    column_order = [column for column in frame.columns if column in aggregated.columns]
    if column_order:
        aggregated = aggregated[column_order]

    reduction = max(len(frame) - len(aggregated), 0)
    return aggregated, keys, configured_funcs, reduction


def _apply_dedupe(
    frame: "pd.DataFrame", dedupe_cfg: Mapping[str, Any] | None
) -> tuple["pd.DataFrame", List[str], str, int]:
    if not isinstance(dedupe_cfg, Mapping):
        return frame, [], "last", 0
    keys = [str(col) for col in _ensure_iterable(dedupe_cfg.get("keys")) if str(col)]
    if not keys:
        return frame, [], "last", 0
    keep = str(dedupe_cfg.get("keep", "last")).strip().lower() or "last"
    keep_arg = keep if keep in {"first", "last"} else "last"
    before = len(frame)
    deduped = frame.drop_duplicates(subset=keys, keep=keep_arg)
    removed = max(before - len(deduped), 0)
    return deduped, keys, keep_arg, removed


def _source_matches(
    path: Path, frame: "pd.DataFrame", source_cfg: Mapping[str, Any]
) -> tuple[bool, Dict[str, Any]]:
    match_cfg = source_cfg.get("match") if isinstance(source_cfg, Mapping) else None
    if not isinstance(match_cfg, Mapping):
        return True, {"regex_miss": False, "regex_checked": False, "missing_columns": []}
    reasons: Dict[str, Any] = {
        "regex_miss": False,
        "regex_checked": False,
        "missing_columns": [],
        "missing_required_any": [],
    }
    filename_regex = match_cfg.get("filename_regex")
    if filename_regex:
        try:
            pattern = str(filename_regex)
            target = path.as_posix()
            reasons["regex_checked"] = True
            if not re.search(pattern, target) and not re.search(pattern, path.name):
                reasons["regex_miss"] = True
                return False, reasons
        except re.error as exc:
            reasons["regex_checked"] = True
            reasons["regex_miss"] = True
            reasons["regex_error"] = str(exc)
            return False, reasons
    required = match_cfg.get("required_columns")
    if required:
        required_columns = {str(column) for column in _ensure_iterable(required)}
        if not required_columns.issubset(set(frame.columns)):
            reasons["missing_columns"] = sorted(
                required_columns.difference(set(frame.columns))
            )
            return False, reasons
    required_any = match_cfg.get("required_any")
    if isinstance(required_any, Mapping):
        available = {str(column) for column in frame.columns}
        missing_groups: List[str] = []
        for aliases in required_any.values():
            alias_group = [str(alias) for alias in _ensure_iterable(aliases) if str(alias)]
            if alias_group and not any(alias in available for alias in alias_group):
                missing_groups.append(
                    "/".join(alias_group)
                    if len(alias_group) > 1
                    else (alias_group[0] if alias_group else "")
                )
        if missing_groups:
            reasons["missing_required_any"] = [group for group in missing_groups if group]
            return False, reasons
    return True, reasons


def _find_source_for_file(
    path: Path,
    frame: "pd.DataFrame",
    sources_cfg: Sequence[Mapping[str, Any]],
) -> tuple[Optional[Mapping[str, Any]], List[Dict[str, Any]]]:
    attempts: List[Dict[str, Any]] = []
    for source in sources_cfg:
        if not isinstance(source, Mapping):
            continue
        matched, reasons = _source_matches(path, frame, source)
        attempt = {
            "name": str(source.get("name") or path.name),
            "matched": matched,
            "reasons": reasons,
        }
        attempts.append(attempt)
        if matched:
            return source, attempts
    return None, attempts


def _summarize_match_attempts(attempts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    regex_checked = False
    regex_success = False
    missing_columns: set[str] = set()
    missing_required_any: set[str] = set()
    for attempt in attempts:
        raw_reasons = attempt.get("reasons")
        reasons = raw_reasons if isinstance(raw_reasons, Mapping) else {}
        if reasons.get("regex_checked"):
            regex_checked = True
            if not reasons.get("regex_miss"):
                regex_success = True
        for column in reasons.get("missing_columns") or []:
            missing_columns.add(str(column))
        for group in reasons.get("missing_required_any") or []:
            missing_required_any.add(str(group))
    if regex_checked:
        summary["regex_miss"] = not regex_success
    else:
        summary["regex_miss"] = False
    if missing_columns:
        summary["missing_columns"] = sorted(missing_columns)
    if missing_required_any:
        summary["missing_required_any"] = sorted(missing_required_any)
    return summary


def _auto_detect_dtm_source(frame: "pd.DataFrame") -> Optional[Dict[str, Any]]:
    required = {"CountryISO3", "ReportingDate", "idp_count"}
    if not required.issubset(set(frame.columns)):
        return None
    return {
        "name": "dtm_displacement_admin0_auto",
        "map": {
            "iso3": {"from": "CountryISO3", "ops": ["trim", "uppercase"]},
            "as_of_date": {"from": "ReportingDate", "ops": ["to_date"]},
            "ym": {"from": "as_of_date", "ops": ["to_ym"]},
            "metric": {"const": "idps"},
            "value": {"from": "idp_count", "ops": ["to_number"]},
            "semantics": {"const": "stock"},
            "source": {"const": "IOM DTM"},
        },
        "filters": [
            {"keep_if_not_null": ["CountryISO3", "ReportingDate", "idp_count"]},
            {"keep_if_positive": ["value"]},
        ],
        "dedupe": {"keys": ["iso3", "as_of_date", "metric"], "keep": "last"},
    }


def _canonical_passthrough(frame: "pd.DataFrame") -> "pd.DataFrame":
    coerced = frame.copy()
    for column in coerced.columns:
        coerced[column] = _coerce_string_series(coerced[column])
    for column in REQUIRED:
        if column not in coerced.columns:
            coerced[column] = ""
    return coerced


def _has_canonical_columns(frame: "pd.DataFrame") -> bool:
    columns = set(frame.columns)
    if CANONICAL_CORE.issubset(columns):
        return True
    return set(REQUIRED).issubset(columns)


def _apply_source(
    *,
    path: Path,
    frame: "pd.DataFrame",
    source_cfg: Mapping[str, Any],
) -> tuple["pd.DataFrame", SourceApplication]:
    name = str(source_cfg.get("name") or path.name)
    map_cfg = source_cfg.get("map") if isinstance(source_cfg, Mapping) else None
    if not isinstance(map_cfg, Mapping):
        mapped = _canonical_passthrough(frame)
        detail = SourceApplication(
            name=name,
            path=path,
            rows_in=len(frame),
            rows_mapped=len(mapped),
            rows_after_filters=len(mapped),
            rows_after_dedupe=len(mapped),
            strategy="passthrough",
        )
        return mapped, detail

    context: Dict[str, "pd.Series"] = {}
    out = pd.DataFrame(index=frame.index)
    mapping_details: Dict[str, Dict[str, Any]] = {}
    for target, instructions in map_cfg.items():
        if not isinstance(instructions, Mapping):
            continue
        series, mapping_info = _resolve_mapping_series(
            frame, context, instructions, length=len(frame)
        )
        out[target] = series
        context[target] = series
        if isinstance(instructions, Mapping):
            candidates = [
                str(value)
                for value in _ensure_iterable(instructions.get("from_any"))
                if str(value)
            ]
            if candidates:
                mapping_info.setdefault("candidates", candidates)
            if instructions.get("from") is not None:
                mapping_info.setdefault("candidate", str(instructions.get("from")))
        mapping_details[target] = mapping_info

    rows_mapped = len(out)
    filtered, drop_histogram, filters_applied = _apply_filters(
        out, source_cfg.get("filters", [])
    )
    rows_after_filters = len(filtered)
    aggregated, aggregate_keys, aggregate_funcs, _ = _apply_aggregate(
        filtered, source_cfg.get("aggregate")
    )
    rows_after_aggregate = len(aggregated)
    deduped, dedupe_keys, dedupe_keep, _ = _apply_dedupe(
        aggregated, source_cfg.get("dedupe")
    )
    out = deduped

    for column in out.columns:
        out[column] = _coerce_string_series(out[column])

    for column in REQUIRED:
        if column not in out.columns:
            out[column] = ""

    warnings: List[str] = []
    if len(out) == 0 and len(frame) > 0:
        warnings.append(f"{name}: mapping yielded 0 rows (from {len(frame)} input rows)")

    applied = SourceApplication(
        name=name,
        path=path,
        rows_in=len(frame),
        rows_mapped=rows_mapped,
        rows_after_filters=rows_after_filters,
        rows_after_aggregate=rows_after_aggregate,
        rows_after_dedupe=len(out),
        strategy=str(source_cfg.get("name") or "config"),
        warnings=warnings,
        filters_applied=filters_applied,
        drop_histogram=drop_histogram,
        dedupe_keys=dedupe_keys,
        dedupe_keep=dedupe_keep,
        aggregate_keys=aggregate_keys,
        aggregate_funcs=aggregate_funcs,
        mapping_details=mapping_details,
    )
    return out.reset_index(drop=True), applied

def _apply_mapping(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    cfg structure:
      mapping:
        event_id: [ "event_id", "evt_id" ]     # take first existing
        country_name: [ "country", "country_name" ]
        ...
      constants:
        unit: "persons"
        method: "api"
        revision: 1
      transforms:
        metric:
          # map source values to canonical
          affected: ["affected","people_affected","pa"]
          in_need: ["pin","people_in_need","in_need"]
      dates:
        as_of_date: ["as_of","date_as_of"]
        publication_date: ["pub_date","published"]
    """
    mapping = cfg.get("mapping", {})
    constants = cfg.get("constants", {})
    transforms = cfg.get("transforms", {})
    dates = cfg.get("dates", {})

    out = pd.DataFrame()

    # Column mapping: take the first available source column
    for target, sources in mapping.items():
        val = None
        for s in sources:
            if s in df.columns:
                val = df[s]
                break
        if val is None:
            out[target] = ""  # will fill from constants/defaults later
        else:
            out[target] = val.fillna("").astype(str)

    # Date mapping (same approach)
    for target, sources in dates.items():
        if target not in out.columns:
            out[target] = ""
        if not out[target].any():
            for s in sources:
                if s in df.columns:
                    out[target] = df[s].astype(str).fillna("")
                    break

    # Constants/defaults
    for k, v in constants.items():
        if k not in out.columns or (out[k] == "").all():
            out[k] = str(v)

    # Metric transforms
    if "metric" in transforms:
        look = {}
        for canonical, alts in transforms["metric"].items():
            for a in alts:
                look[str(a).lower()] = canonical
        if "metric" in out.columns:
            out["metric"] = out["metric"].astype(str).str.lower().map(lambda x: look.get(x, x))

    # Fill missing required columns with empty string (we’ll validate later)
    for col in REQUIRED:
        if col not in out.columns:
            out[col] = ""

    # Coerce some types
    # value should be numeric string; keep as string for consistency; validator enforces numeric
    # ensure revision and other ints are stringified
    for c in out.columns:
        out[c] = out[c].astype(str).fillna("")

    # Add ingested_at if empty
    if "ingested_at" in out.columns:
        mask = out["ingested_at"].str.len() == 0
        if mask.any():
            ts = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            out.loc[mask, "ingested_at"] = ts

    return out

RESOLVED_DB_NUMERIC = {"value"}

DELTAS_DB_NUMERIC = {
    "value_new",
    "value_stock",
    "rebase_flag",
    "first_observation",
    "delta_negative_clamped",
}

DELTAS_DB_DEFAULTS = {"series_semantics": "new"}

LOGGER = get_logger(__name__)
def _to_month(series: "pd.Series") -> "pd.Series":
    dates = pd.to_datetime(series, errors="coerce")
    formatted = dates.dt.strftime("%Y-%m")
    return formatted.fillna("")


def _warn_on_non_canonical_semantics(label: str, frame: "pd.DataFrame | None") -> None:
    if frame is None or frame.empty or "series_semantics" not in frame.columns:
        return
    values = (
        frame["series_semantics"]
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )
    invalid = sorted(
        value for value in values if value not in {"", "new", "stock"}
    )
    if invalid:
        LOGGER.warning(
            "DuckDB write: %s frame has non-canonical series_semantics values: %s",
            label,
            invalid,
        )


def _apply_series_semantics(frame: "pd.DataFrame") -> "pd.DataFrame":
    if frame is None:
        return frame
    if frame.empty:
        if "series_semantics" not in frame.columns:
            frame = frame.copy()
            frame["series_semantics"] = []
        return frame
    if "series_semantics" not in frame.columns:
        frame["series_semantics"] = ""
    frame["series_semantics"] = frame.apply(
        lambda row: compute_series_semantics(
            metric=row.get("metric"), existing=row.get("series_semantics")
        ),
        axis=1,
    )
    return frame


def _isoformat_date_strings(series: "pd.Series") -> "pd.Series":
    parsed = pd.to_datetime(series, errors="coerce")
    iso = parsed.dt.strftime("%Y-%m-%d")
    fallback = series.fillna("").astype(str)
    fallback = fallback.replace({"NaT": "", "<NA>": "", "nan": "", "NaN": ""})
    return iso.where(parsed.notna(), fallback).fillna("").astype(str)


def _prepare_resolved_for_db(df: "pd.DataFrame | None") -> "pd.DataFrame | None":
    if df is None or df.empty:
        return None

    frame = df.copy()

    for column in ("as_of_date", "publication_date"):
        if column in frame.columns:
            frame[column] = _isoformat_date_strings(frame[column])

    if "ym" not in frame.columns:
        frame["ym"] = ""
    frame["ym"] = frame["ym"].fillna("").astype(str)
    mask = frame["ym"].str.strip() == ""
    if mask.any() and "as_of_date" in frame.columns:
        frame.loc[mask, "ym"] = frame.loc[mask, "as_of_date"].astype(str).str.slice(0, 7)
        mask = frame["ym"].str.strip() == ""
    if mask.any() and "publication_date" in frame.columns:
        frame.loc[mask, "ym"] = frame.loc[mask, "publication_date"].astype(str).str.slice(0, 7)

    frame = _apply_series_semantics(frame)
    frame = normalize_series_semantics(frame)
    frame["series_semantics"] = frame["series_semantics"].fillna("").astype(str)

    for column in RESOLVED_DB_NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _prepare_deltas_for_db(df: "pd.DataFrame | None") -> "pd.DataFrame | None":
    if df is None or df.empty:
        return None

    frame = df.copy()

    if "series_semantics_out" in frame.columns:
        semantics_out = frame["series_semantics_out"].where(
            frame["series_semantics_out"].notna(), ""
        ).astype(str)
        if "series_semantics" in frame.columns:
            semantics_current_raw = frame["series_semantics"]
            semantics_current = semantics_current_raw.astype(str)
            prefer_out = semantics_current_raw.isna() | semantics_current.str.strip().eq("")
            frame.loc[prefer_out, "series_semantics"] = semantics_out.loc[prefer_out]
        else:
            frame["series_semantics"] = semantics_out
        frame = frame.drop(columns=["series_semantics_out"])

    if "as_of" in frame.columns:
        frame["as_of"] = _isoformat_date_strings(frame["as_of"])

    if "ym" not in frame.columns:
        frame["ym"] = ""
    frame["ym"] = frame["ym"].fillna("").astype(str)
    mask = frame["ym"].str.len() == 0
    if mask.any() and "as_of" in frame.columns:
        frame.loc[mask, "ym"] = _to_month(frame.loc[mask, "as_of"])

    if "series_semantics" not in frame.columns:
        frame["series_semantics"] = DELTAS_DB_DEFAULTS["series_semantics"]
    semantics = frame["series_semantics"].fillna("").astype(str)
    frame["series_semantics"] = semantics.mask(
        semantics.str.strip() == "",
        DELTAS_DB_DEFAULTS["series_semantics"],
    )
    frame = normalize_series_semantics(frame)

    for column in DELTAS_DB_NUMERIC:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def _parse_write_db_flag(value: str | bool | None) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped == "":
            return None
        if stripped in {"1", "true", "True"}:
            return True
        if stripped in {"0", "false", "False"}:
            return False
    try:
        return bool(int(value))
    except Exception:
        return None


def _maybe_write_to_db(
    *,
    facts_resolved: "Optional[pd.DataFrame]" = None,
    facts_deltas: "Optional[pd.DataFrame]" = None,
    db_url: Optional[str] = None,
    write_db: Optional[bool] = None,
) -> None:
    """Write exported facts into DuckDB when enabled via environment or flag."""

    if duckdb_io is None:
        return
    env_url = os.environ.get("RESOLVER_DB_URL", "").strip()
    if db_url is not None:
        db_url = db_url.strip()
    else:
        db_url = env_url
    if write_db is None:
        write_db = bool(db_url)
    if not write_db or not db_url:
        LOGGER.debug("DuckDB write skipped: disabled or missing RESOLVER_DB_URL")
        return

    resolved_prepared = _prepare_resolved_for_db(facts_resolved)
    deltas_prepared = _prepare_deltas_for_db(facts_deltas)
    if resolved_prepared is None and deltas_prepared is None:
        LOGGER.debug("DuckDB write skipped: no prepared frames to persist")
        return

    conn = None
    try:
        LOGGER.info("Writing exports to DuckDB at %s", db_url)
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "Resolved frame schema: %s",
                df_schema(resolved_prepared),
            )
            LOGGER.debug(
                "Resolved series_semantics distribution: %s",
                dict_counts(
                    resolved_prepared["series_semantics"]
                    if resolved_prepared is not None
                    else []
                ),
            )
            LOGGER.debug(
                "Deltas frame schema: %s",
                df_schema(deltas_prepared),
            )
        _warn_on_non_canonical_semantics("facts_resolved", resolved_prepared)
        _warn_on_non_canonical_semantics("facts_deltas", deltas_prepared)
        conn = duckdb_io.get_db(db_url)
        duckdb_io.init_schema(conn)
        if resolved_prepared is not None and not resolved_prepared.empty:
            written_resolved = duckdb_io.upsert_dataframe(
                conn,
                "facts_resolved",
                resolved_prepared,
                keys=duckdb_io.FACTS_RESOLVED_KEY_COLUMNS,
            )
            LOGGER.info("DuckDB facts_resolved rows written: %s", written_resolved)
        if deltas_prepared is not None and not deltas_prepared.empty:
            written_deltas = duckdb_io.upsert_dataframe(
                conn,
                "facts_deltas",
                deltas_prepared,
                keys=duckdb_io.FACTS_DELTAS_KEY_COLUMNS,
            )
            LOGGER.info("DuckDB facts_deltas rows written: %s", written_deltas)
        LOGGER.info("DuckDB write complete")
    except Exception as exc:  # pragma: no cover - non fatal for exporter
        LOGGER.error("DuckDB write skipped: %s", exc, exc_info=True)
        print(f"Warning: DuckDB write skipped ({exc}).", file=sys.stderr)
    finally:
        # Shared DuckDB connections are cached per URL; leave them open so
        # subsequent writers/readers reuse the same wrapper instance.
        pass


@dataclass
class ExportResult:
    rows: int
    csv_path: Path
    parquet_path: Optional[Path]
    dataframe: "pd.DataFrame"
    warnings: List[str] = field(default_factory=list)
    sources: List[SourceApplication] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)


class ExportError(RuntimeError):
    pass


def _render_markdown_report(report: Mapping[str, Any]) -> str:
    matched_files: List[Mapping[str, Any]] = list(report.get("matched_files") or [])
    matched_sources = sorted(
        {
            str(entry.get("source"))
            for entry in matched_files
            if entry.get("source")
        }
    )
    inputs_scanned = int(report.get("inputs_scanned", 0) or 0)
    rows_exported = int(report.get("rows_exported", 0) or 0)
    filters_applied: List[str] = [
        str(name)
        for name in (report.get("filters_applied") or [])
        if str(name)
    ]
    dedupe_keys: List[str] = [
        str(name)
        for name in (report.get("dedupe_keys") or [])
        if str(name)
    ]
    dedupe_keep: List[str] = [
        str(name)
        for name in (report.get("dedupe_keep") or [])
        if str(name)
    ]
    drop_histogram: Mapping[str, Any] = report.get("dropped_by_filter") or {}
    unmatched_files: List[str] = [
        str(path) for path in (report.get("unmatched_files") or [])
    ]
    warnings_list: List[str] = [
        str(msg) for msg in (report.get("warnings") or []) if str(msg)
    ]
    monthly_summary: List[Mapping[str, Any]] = list(report.get("monthly_summary") or [])

    lines: List[str] = ["## Export Facts"]
    lines.append(f"- **Inputs scanned:** {inputs_scanned}")
    if matched_files:
        if matched_sources:
            lines.append(
                f"- **Matched:** {len(matched_files)} file(s) → sources: {', '.join(matched_sources)}"
            )
        else:
            lines.append(f"- **Matched:** {len(matched_files)} file(s)")
    else:
        lines.append("- **Matched:** 0 file(s)")
    lines.append(f"- **Rows exported:** {rows_exported}")
    if dedupe_keys:
        keep_display = ", ".join(sorted(set(dedupe_keep))) if dedupe_keep else "last"
        lines.append(
            f"- **Dedupe keys:** {', '.join(dedupe_keys)} (keep={keep_display})"
        )
    else:
        lines.append("- **Dedupe keys:** (none)")
    if filters_applied:
        lines.append(f"- **Filters:** {', '.join(filters_applied)}")
    else:
        lines.append("- **Filters:** (none)")
    if drop_histogram:
        drop_parts = [f"{key}: {value}" for key, value in drop_histogram.items()]
        lines.append(f"- **Rows dropped by filters:** {', '.join(drop_parts)}")
    if unmatched_files:
        preview = unmatched_files[:5]
        suffix = " …" if len(unmatched_files) > 5 else ""
        lines.append(f"- **Unmatched files:** {', '.join(preview)}{suffix}")
    else:
        lines.append("- **Unmatched files:** (none)")
    if warnings_list:
        lines.append(f"- **Warnings:** {', '.join(warnings_list[:5])}")

    if matched_files:
        lines.append("")
        lines.append("### Matched files")
        lines.append("| File | Source | Rows in | After filters | After agg | After dedupe | Strategy |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
        for entry in matched_files:
            file_name = Path(str(entry.get("path", ""))).name or str(entry.get("path", ""))
            source_name = str(entry.get("source") or "")
            rows_in = entry.get("rows_in") or 0
            rows_filtered = entry.get("rows_after_filters") or 0
            rows_aggregate = entry.get("rows_after_aggregate") or rows_filtered
            rows_dedupe = entry.get("rows_after_dedupe") or 0
            strategy = str(entry.get("strategy") or "")
            lines.append(
                f"| {file_name} | {source_name} | {rows_in} | {rows_filtered} | {rows_aggregate} | {rows_dedupe} | {strategy} |"
            )

        lines.append("")
        lines.append("#### Mapping & aggregation")
        for entry in matched_files:
            file_name = Path(str(entry.get("path", ""))).name or str(entry.get("path", ""))
            source_name = str(entry.get("source") or "")
            lines.append(f"- **{file_name}** → {source_name or '(source not named)'}")
            mapping_details = entry.get("mapping") or {}
            mapping_parts: List[str] = []
            for target, info in mapping_details.items():
                if not isinstance(info, Mapping):
                    continue
                if "const" in info:
                    mapping_parts.append(f"{target} ← const({info['const']})")
                    continue
                source_label = info.get("source")
                if not source_label:
                    source_label = "∅"
                if info.get("from_context"):
                    source_label = f"{source_label} (mapped)"
                mapping_str = f"{target} ← {source_label}"
                ops = info.get("ops") or []
                if ops:
                    mapping_str += f" (ops: {', '.join(ops)})"
                if info.get("missing"):
                    mapping_str += " [missing]"
                mapping_parts.append(mapping_str)
            if mapping_parts:
                lines.append(f"  - Mapping: {', '.join(mapping_parts)}")
            aggregate_info = entry.get("aggregate") or {}
            aggregate_keys = aggregate_info.get("keys") or []
            aggregate_funcs = aggregate_info.get("funcs") or {}
            if aggregate_keys or aggregate_funcs:
                agg_parts: List[str] = []
                if aggregate_keys:
                    agg_parts.append(f"keys={', '.join(aggregate_keys)}")
                if aggregate_funcs:
                    func_parts = [f"{col}:{func}" for col, func in aggregate_funcs.items()]
                    agg_parts.append(f"funcs={', '.join(func_parts)}")
                rows_before = aggregate_info.get("rows_before")
                rows_after = aggregate_info.get("rows_after")
                if rows_before is not None and rows_after is not None:
                    agg_parts.append(f"rows {rows_before}→{rows_after}")
                lines.append(f"  - Aggregate: {'; '.join(agg_parts)}")
            dedupe_info = entry.get("dedupe") or {}
            dedupe_keys_entry = dedupe_info.get("keys") or []
            dedupe_keep_entry = dedupe_info.get("keep") or "last"
            if dedupe_keys_entry:
                lines.append(
                    f"  - Dedupe: keys={', '.join(dedupe_keys_entry)} (keep={dedupe_keep_entry})"
                )
            else:
                lines.append("  - Dedupe: (none)")

    if monthly_summary:
        lines.append("")
        lines.append("### Monthly summary")
        lines.append("| Month | Rows |")
        lines.append("| --- | ---: |")
        for entry in monthly_summary:
            month = str(entry.get("month") or entry.get("ym") or "")
            rows = int(entry.get("rows") or 0)
            lines.append(f"| {month} | {rows} |")

    preview_rows: List[Mapping[str, Any]] = list(report.get("preview") or [])
    if preview_rows:
        lines.append("")
        lines.append("### Preview (first 5 rows)")
        columns = ["iso3", "as_of_date", "metric", "value"]
        lines.append("| iso3 | as_of_date | metric | value |")
        lines.append("| --- | --- | --- | --- |")
        for row in preview_rows:
            values = [str(row.get(col, "")) for col in columns]
            lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines).rstrip() + "\n"
def export_facts(
    *,
    inp: Path,
    config_path: Path = DEFAULT_CONFIG,
    out_dir: Path = EXPORTS,
    write_db: str | bool | None = None,
    db_url: Optional[str] = None,
    report_json_path: Optional[Path] = None,
    report_md_path: Optional[Path] = None,
    append_summary_path: Optional[Path] = None,
) -> ExportResult:
    if not config_path.exists():
        raise ExportError(f"Config not found: {config_path}")

    files, skipped_meta = _collect_inputs(inp)

    warnings: List[str] = []
    source_details: List[SourceApplication] = []
    debug_records: List[Dict[str, Any]] = []

    for meta_path in skipped_meta:
        message = f"Skipped metadata file {meta_path.name}"
        warnings.append(message)
        source_details.append(
            SourceApplication(
                name=meta_path.name,
                path=meta_path,
                rows_in=0,
                strategy="meta-skip",
                warnings=[message],
            )
        )

    if not files:
        raise ExportError(f"No supported files found under {inp}")

    cfg = _load_config(config_path)

    use_sources = isinstance(cfg, Mapping) and isinstance(cfg.get("sources"), Iterable)

    unmatched_paths: List[Path] = []

    if use_sources:
        mapped_frames: List[pd.DataFrame] = []
        sources_cfg = [source for source in cfg.get("sources", []) if isinstance(source, Mapping)]
        raw_frames: List[tuple[Path, "pd.DataFrame"]] = []
        dtm_frame: Optional["pd.DataFrame"] = None
        dtm_detail: Optional[SourceApplication] = None
        dtm_yaml_rows: Optional[int] = None
        dtm_debug_entry: Optional[Dict[str, Any]] = None
        for file_path in files:
            try:
                frame = _read_one(file_path)
            except Exception as exc:
                warning = f"Failed to read {file_path.name}: {exc}"
                warnings.append(warning)
                debug_records.append(
                    {
                        "file": _relativize_path(file_path),
                        "matched": False,
                        "used_mapping": None,
                        "columns": [],
                        "reasons": {"read_error": str(exc)},
                    }
                )
                source_details.append(
                    SourceApplication(
                        name=file_path.name,
                        path=file_path,
                        rows_in=0,
                        strategy="read-failed",
                        warnings=[warning],
                    )
                )
                unmatched_paths.append(file_path)
                continue

            if frame.empty:
                warning = f"{file_path.name}: no rows parsed (empty or invalid file)"
                warnings.append(warning)
                debug_records.append(
                    {
                        "file": _relativize_path(file_path),
                        "matched": False,
                        "used_mapping": None,
                        "columns": [],
                        "reasons": {"empty_input": True},
                    }
                )
                source_details.append(
                    SourceApplication(
                        name=file_path.name,
                        path=file_path,
                        rows_in=0,
                        strategy="empty-input",
                        warnings=[warning],
                    )
                )
                unmatched_paths.append(file_path)
                continue

            raw_frames.append((file_path, frame))
            relative_path = _relativize_path(file_path)
            columns = [str(col) for col in frame.columns]
            debug_entry: Dict[str, Any] = {
                "file": relative_path,
                "columns": columns,
            }

            if file_path.name.lower() == "dtm_displacement.csv":
                mapped, meta = _map_dtm_displacement_admin0(frame)
                filters_applied = list(meta.get("filters_applied", []))
                drop_hist = {
                    str(key): int(value)
                    for key, value in (meta.get("drop_histogram") or {}).items()
                }
                aggregate_keys = list(meta.get("dedupe_keys", []))
                aggregate_funcs = {
                    str(column): str(func)
                    for column, func in (meta.get("aggregate_funcs") or {}).items()
                }
                rows_after_filters = int(meta.get("rows_after_filters", len(mapped)))
                rows_after_aggregate = int(meta.get("rows_after_aggregate", len(mapped)))

                detail = SourceApplication(
                    name="dtm_displacement_admin0",
                    path=file_path,
                    rows_in=len(frame),
                    rows_mapped=rows_after_filters,
                    rows_after_filters=rows_after_filters,
                    rows_after_aggregate=rows_after_aggregate,
                    rows_after_dedupe=len(mapped),
                    strategy="dtm-admin0-alias",
                    filters_applied=filters_applied,
                    drop_histogram=drop_hist,
                    aggregate_keys=aggregate_keys,
                    aggregate_funcs=aggregate_funcs,
                )
                detail.dedupe_keys = aggregate_keys
                detail.dedupe_keep = "max"

                mapping_details: Dict[str, Dict[str, Any]] = {}
                sources = meta.get("sources") or {}
                iso_source = sources.get("iso3")
                if iso_source:
                    mapping_details["iso3"] = {"source": iso_source}
                date_source = sources.get("as_of_date")
                if date_source:
                    mapping_details["as_of_date"] = {
                        "source": date_source,
                        "ops": ["to_month_end"],
                    }
                value_source = sources.get("value")
                if value_source:
                    mapping_details["value"] = {
                        "source": value_source,
                        "ops": ["to_number"],
                    }
                mapping_details["metric"] = {"const": "idp_displacement_stock_dtm"}
                mapping_details["semantics"] = {"const": "stock"}
                mapping_details["series_semantics"] = {"const": "stock"}
                mapping_details["source"] = {"const": "IOM DTM"}
                detail.mapping_details = mapping_details

                if len(frame) > 0 and mapped.empty:
                    warning_msg = (
                        "dtm_displacement.csv: DTM mapper produced 0 rows after normalization"
                    )
                    detail.warnings.append(warning_msg)
                    warnings.append(warning_msg)

                source_details.append(detail)
                if not mapped.empty:
                    mapped_frames.append(mapped)

                debug_entry.update(
                    {
                        "matched": True,
                        "used_mapping": "dtm_displacement_admin0_alias",
                        "strategy": "dtm-admin0-alias",
                        "filters": filters_applied,
                        "drop_histogram": drop_hist,
                    }
                )
                if not mapped.empty:
                    debug_entry["rows_out"] = int(len(mapped))
                    debug_entry["dedupe"] = {
                        "keys": aggregate_keys,
                        "keep": detail.dedupe_keep,
                    }
                    debug_entry["aggregate"] = {
                        "keys": aggregate_keys,
                        "funcs": aggregate_funcs,
                    }
                debug_records.append(debug_entry)
                continue

            matched_cfg, attempts = _find_source_for_file(file_path, frame, sources_cfg)
            strategy = "config"
            attempt_summary = _summarize_match_attempts(attempts)
            if matched_cfg is None:
                auto_cfg = _auto_detect_dtm_source(frame)
                if auto_cfg is not None:
                    matched_cfg = auto_cfg
                    strategy = "auto-dtm"
                elif _has_canonical_columns(frame):
                    mapped = _canonical_passthrough(frame).reset_index(drop=True)
                    detail = SourceApplication(
                        name=file_path.name,
                        path=file_path,
                        rows_in=len(frame),
                        rows_mapped=len(mapped),
                        rows_after_filters=len(mapped),
                        rows_after_dedupe=len(mapped),
                        strategy="canonical-passthrough",
                    )
                    source_details.append(detail)
                    debug_entry.update(
                        {
                            "matched": True,
                            "used_mapping": "canonical-passthrough",
                            "strategy": "canonical-passthrough",
                        }
                    )
                    debug_records.append(debug_entry)
                    if not mapped.empty:
                        mapped_frames.append(mapped)
                    continue
                else:
                    warning = f"No export mapping matched {file_path.name}; file skipped"
                    warnings.append(warning)
                    unmatched_paths.append(file_path)
                    source_details.append(
                        SourceApplication(
                            name=file_path.name,
                            path=file_path,
                            rows_in=len(frame),
                            strategy="unmapped",
                            warnings=[warning],
                        )
                    )
                    reasons = attempt_summary or {}
                    if "regex_miss" not in reasons:
                        reasons["regex_miss"] = False
                    debug_entry.update(
                        {
                            "matched": False,
                            "used_mapping": None,
                            "reasons": reasons,
                        }
                    )
                    debug_records.append(debug_entry)
                    continue

            mapped, detail = _apply_source(path=file_path, frame=frame, source_cfg=matched_cfg)
            detail.strategy = strategy
            if detail.warnings:
                warnings.extend(detail.warnings)
            source_details.append(detail)
            if file_path.name == "dtm_displacement.csv":
                dtm_yaml_rows = len(mapped)
                dtm_detail = detail
            if not mapped.empty:
                mapped_frames.append(mapped)
            mapping_name = str(matched_cfg.get("name") or file_path.name)
            debug_entry.update(
                {
                    "matched": True,
                    "used_mapping": mapping_name,
                    "strategy": strategy,
                }
            )
            if detail.dedupe_keys:
                debug_entry["dedupe"] = {
                    "keys": detail.dedupe_keys,
                    "keep": detail.dedupe_keep,
                }
            debug_records.append(debug_entry)
            if file_path.name == "dtm_displacement.csv":
                dtm_debug_entry = debug_entry

        if dtm_yaml_rows == 0 and dtm_frame is not None:
            fallback, fallback_meta = _map_dtm_admin0_fallback(dtm_frame)
            if fallback is not None and not fallback.empty:
                if "series_semantics" not in fallback.columns:
                    fallback["series_semantics"] = "stock"
                mapped_frames.append(fallback)
                warning_msg = (
                    "dtm_displacement.csv: YAML mapping yielded 0 rows; applied admin0 fallback"
                )
                warnings.append(warning_msg)
                LOGGER.warning(
                    "Export mapping warning: YAML mapping yielded 0 rows; applied dtm_admin0 fallback: %s rows",
                    len(fallback),
                )
                if dtm_detail is not None:
                    dtm_detail.rows_mapped = fallback_meta.get("rows_after_filters", len(fallback))
                    dtm_detail.rows_after_filters = fallback_meta.get("rows_after_filters", len(fallback))
                    dtm_detail.rows_after_aggregate = fallback_meta.get("rows_after_aggregate", len(fallback))
                    dtm_detail.rows_after_dedupe = fallback_meta.get("rows_after_dedupe", len(fallback))
                    dtm_detail.dedupe_keys = ["iso3", "as_of_date", "metric"]
                    dtm_detail.dedupe_keep = "last"
                    dtm_detail.aggregate_keys = fallback_meta.get("aggregate_keys", [])
                    dtm_detail.aggregate_funcs = fallback_meta.get("aggregate_funcs", {})
                    dtm_detail.drop_histogram = fallback_meta.get("drop_histogram", {})
                    dtm_detail.filters_applied = fallback_meta.get("filters_applied", [])
                    metric_details = dtm_detail.mapping_details.get("metric")
                    if isinstance(metric_details, dict):
                        metric_details["const"] = "idps_stock"
                    detail_warning = (
                        "YAML mapping yielded 0 rows; applied dtm_admin0 fallback"
                    )
                    if detail_warning not in dtm_detail.warnings:
                        dtm_detail.warnings.append(detail_warning)
                if dtm_debug_entry is not None:
                    dtm_debug_entry["fallback"] = {
                        "applied": True,
                        "rows": int(len(fallback)),
                        "rows_after_filters": int(
                            fallback_meta.get("rows_after_filters", len(fallback))
                        ),
                        "rows_after_aggregate": int(
                            fallback_meta.get("rows_after_aggregate", len(fallback))
                        ),
                    }

        if not mapped_frames and raw_frames:
            staging_frames = [frame for _, frame in raw_frames]
            staging = (
                pd.concat(staging_frames, ignore_index=True)
                if len(staging_frames) > 1
                else staging_frames[0]
            )
            fallback = _apply_mapping(staging, cfg)
            fallback_raw_len = len(fallback)
            if not fallback.empty:
                canonical_columns = [col for col in CANONICAL_CORE if col in fallback.columns]
                if canonical_columns:
                    non_empty = fallback[canonical_columns].apply(
                        lambda col: col.astype(str).str.strip().ne(""), axis=0
                    )
                    valid_mask = non_empty.any(axis=1)
                    fallback = fallback.loc[valid_mask]
            detail = SourceApplication(
                name="legacy-config",
                path=raw_frames[0][0],
                rows_in=len(staging),
                rows_mapped=fallback_raw_len,
                rows_after_filters=len(fallback),
                rows_after_dedupe=len(fallback),
                strategy="legacy-fallback",
            )
            source_details.append(detail)
            fallback_warning = (
                "Legacy fallback applied: "
                f"{len(staging)} staging rows → {len(fallback)} exported rows"
            )
            warnings.append(fallback_warning)
            if not fallback.empty:
                mapped_frames.append(fallback)

        if mapped_frames:
            facts = pd.concat(mapped_frames, ignore_index=True)
        else:
            facts = pd.DataFrame(columns=REQUIRED)
    else:
        frames: List[pd.DataFrame] = []
        successful_paths: List[Path] = []
        for file_path in files:
            try:
                frame = _read_one(file_path)
            except Exception as exc:
                warning = f"Failed to read {file_path.name}: {exc}"
                warnings.append(warning)
                source_details.append(
                    SourceApplication(
                        name=file_path.name,
                        path=file_path,
                        rows_in=0,
                        strategy="read-failed",
                        warnings=[warning],
                    )
                )
                continue
            if frame.empty:
                warning = f"{file_path.name}: no rows parsed (empty or invalid file)"
                warnings.append(warning)
                source_details.append(
                    SourceApplication(
                        name=file_path.name,
                        path=file_path,
                        rows_in=0,
                        strategy="empty-input",
                        warnings=[warning],
                    )
                )
                continue
            frames.append(frame)
            successful_paths.append(file_path)

        if not frames:
            raise ExportError("No rows produced after reading staging inputs.")

        staging = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
        facts = _apply_mapping(staging, cfg)
        source_details.append(
            SourceApplication(
                name="legacy-config",
                path=successful_paths[0],
                rows_in=len(staging),
                strategy="legacy",
                rows_mapped=len(facts),
                rows_after_filters=len(facts),
                rows_after_dedupe=len(facts),
            )
        )

    facts = _apply_series_semantics(facts)
    for col in ["as_of_date", "publication_date"]:
        if col in facts.columns:
            parsed = pd.to_datetime(facts[col], errors="coerce")
            iso = parsed.dt.strftime("%Y-%m-%d")
            fallback = facts[col].fillna("").astype(str)
            fallback = fallback.replace({"NaT": "", "<NA>": "", "nan": "", "NaN": ""})
            facts[col] = iso.where(parsed.notna(), fallback).fillna("")
    LOGGER.info("Prepared %s exported rows", len(facts))
    LOGGER.debug("Export dataframe schema: %s", df_schema(facts))
    LOGGER.info(
        "series_semantics distribution: %s",
        dict_counts(facts.get("series_semantics", [])),
    )
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.debug("Sample rows: %s", facts.head(5).to_dict(orient="records"))
    for warning in warnings:
        LOGGER.warning("Export mapping warning: %s", warning)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "facts.csv"
    pq_path = out_dir / "facts.parquet"
    facts.to_csv(csv_path, index=False)

    parquet_written: Optional[Path] = None
    try:
        facts.to_parquet(pq_path, index=False)
        parquet_written = pq_path
    except Exception as exc:
        print(f"Warning: could not write Parquet ({exc}). CSV written.", file=sys.stderr)

    _maybe_write_to_db(
        facts_resolved=facts,
        db_url=db_url,
        write_db=_parse_write_db_flag(write_db),
    )

    result_rows = len(facts)

    monthly_summary: List[Dict[str, Any]] = []
    if isinstance(facts, pd.DataFrame) and not facts.empty:
        if "as_of_date" in facts.columns:
            months = facts["as_of_date"].astype(str).str.slice(0, 7)
            valid = months.str.strip().ne("")
            month_counts = (
                months[valid]
                .value_counts()
                .sort_index()
            )
            for month, count in month_counts.items():
                monthly_summary.append({"month": str(month), "rows": int(count)})

    filters_ordered: List[str] = []
    dedupe_keys_ordered: List[str] = []
    dedupe_keep_values: set[str] = set()
    drop_hist_total: Counter[str] = Counter()
    matched_entries: List[Dict[str, Any]] = []
    unmatched_set: set[str] = {str(path) for path in unmatched_paths}

    for detail in source_details:
        include_detail = detail.rows_mapped > 0 or detail.rows_after_dedupe > 0
        if detail.strategy == "legacy-fallback" and detail.rows_after_dedupe == 0:
            include_detail = False
        if include_detail:
            entry = detail.as_report_entry()
            matched_entries.append(entry)
            for name in entry.get("filters") or []:
                if name and name not in filters_ordered:
                    filters_ordered.append(str(name))
            dedupe_info = entry.get("dedupe") or {}
            for key in dedupe_info.get("keys") or []:
                if key and key not in dedupe_keys_ordered:
                    dedupe_keys_ordered.append(str(key))
            keep_value = dedupe_info.get("keep")
            if keep_value:
                dedupe_keep_values.add(str(keep_value))
            drop_hist_total.update(detail.drop_histogram or {})
        elif detail.strategy in {"unmapped", "read-failed", "empty-input", "legacy-fallback"}:
            unmatched_set.add(str(detail.path))

    unmatched_set.update(str(path) for path in unmatched_paths)
    unmatched_files = sorted(unmatched_set)
    dropped_by_filter = {
        key: int(drop_hist_total.get(key, 0))
        for key in filters_ordered
    }

    report = {
        "inputs_scanned": len(files),
        "matched_files": matched_entries,
        "unmatched_files": unmatched_files,
        "rows_exported": result_rows,
        "filters_applied": filters_ordered,
        "dedupe_keys": dedupe_keys_ordered,
        "dedupe_keep": sorted(dedupe_keep_values) if dedupe_keep_values else [],
        "dropped_by_filter": dropped_by_filter,
        "preview": facts.head(5).to_dict(orient="records"),
        "warnings": warnings,
        "meta_files_skipped": [str(path) for path in skipped_meta],
        "monthly_summary": monthly_summary,
    }

    debug_dir = Path("diagnostics/ingestion/export_preview")
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / "mapping_debug.jsonl"
        with open(debug_path, "w", encoding="utf-8") as fh:
            for record in debug_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        LOGGER.warning("Failed to write mapping debug file: %s", exc)

    report_json_path = report_json_path or (out_dir / "export_report.json")
    report_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    markdown_block = _render_markdown_report(report)
    report_md_path = report_md_path or (out_dir / "export_report.md")
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_md_path, "w", encoding="utf-8") as fh:
        fh.write(markdown_block)

    if append_summary_path is not None:
        append_summary_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            needs_leading_newline = append_summary_path.exists() and append_summary_path.stat().st_size > 0
        except OSError:
            needs_leading_newline = False
        with open(append_summary_path, "a", encoding="utf-8") as fh:
            if needs_leading_newline:
                fh.write("\n")
            fh.write(markdown_block)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as fh:
                fh.write(markdown_block)
        except OSError:
            LOGGER.debug("Could not write GitHub summary", exc_info=True)

    return ExportResult(
        rows=result_rows,
        csv_path=csv_path,
        parquet_path=parquet_written,
        dataframe=facts,
        warnings=warnings,
        sources=source_details,
        report=report,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to staging file or directory")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to export_config.yml")
    ap.add_argument("--out", default=str(EXPORTS), help="Output directory (will create if needed)")
    ap.add_argument(
        "--write-db",
        default=None,
        choices=["0", "1"],
        help="Set to 1 or 0 to force-enable or disable DuckDB dual-write (defaults to auto)",
    )
    ap.add_argument(
        "--db-url",
        default=None,
        help="Optional DuckDB URL override (defaults to RESOLVER_DB_URL)",
    )
    ap.add_argument(
        "--report-json",
        default=None,
        help="Optional path for export_report.json (defaults to <out>/export_report.json)",
    )
    ap.add_argument(
        "--report-md",
        default=None,
        help="Optional path for export_report.md (defaults to <out>/export_report.md)",
    )
    ap.add_argument(
        "--append-summary",
        default=None,
        help="Append the Export Facts markdown block to this file",
    )
    args = ap.parse_args()

    try:
        result = export_facts(
            inp=Path(args.inp),
            config_path=Path(args.config),
            out_dir=Path(args.out),
            write_db=args.write_db,
            db_url=args.db_url,
            report_json_path=Path(args.report_json) if args.report_json else None,
            report_md_path=Path(args.report_md) if args.report_md else None,
            append_summary_path=Path(args.append_summary) if args.append_summary else None,
        )
    except ExportError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    print(f"✅ Exported {result.rows} rows")
    if result.parquet_path and result.parquet_path.exists():
        print(f" - parquet: {result.parquet_path}")
    print(f" - csv (diagnostic): {result.csv_path}")

if __name__ == "__main__":
    main()
