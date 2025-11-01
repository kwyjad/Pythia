"""Render ingestion connector diagnostics into a comprehensive summary."""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import platform
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from scripts.ci._summarizer_utils import (
    gather_log_files,
    gather_meta_json_files,
    reason_histogram,
    safe_load_json,
    safe_load_jsonl,
    status_histogram,
    top_value_counts_from_csv,
)

__all__ = [
    "load_report",
    "build_markdown",
    "render_summary_md",
    "render_dtm_deep_dive",
    "_render_dtm_deep_dive",
    "main",
    "SUMMARY_TITLE",
]

SUMMARY_PATH = Path("summary.md")
DEFAULT_REPORT_PATH = Path("diagnostics") / "ingestion" / "connectors_report.jsonl"
DEFAULT_DIAG_DIR = Path("diagnostics") / "ingestion"
DEFAULT_STAGING_DIR = Path("resolver") / "staging"
SUMMARY_TITLE = "# Ingestion Superreport"
LEGACY_TITLE = "# Connector Diagnostics"
EM_DASH = "—"


_LAST_STUB_REASON_ALIAS: str | None = None


def load_report(path: os.PathLike[str] | str) -> List[Dict[str, Any]]:
    """Load a JSONL report, tolerating absent or malformed entries."""

    global _LAST_STUB_REASON_ALIAS

    target = Path(path)
    raw_entries: List[Any] = []
    if target.exists():
        raw_entries = safe_load_jsonl(target)
    if not raw_entries:
        target.parent.mkdir(parents=True, exist_ok=True)
        stub = {
            "client": "summarizer",
            "mode": "diagnostics",
            "status": "error",
            "reason": "missing or empty report",
            "extras": {"stub": "missing-report"},
            "counts": {"written": 0},
        }
        target.write_text(json.dumps(stub) + "\n", encoding="utf-8")
        raw_entries = [stub]
        _LAST_STUB_REASON_ALIAS = "missing-report"
    else:
        _LAST_STUB_REASON_ALIAS = None

    entries: List[Dict[str, Any]] = []
    for entry in raw_entries:
        if not isinstance(entry, Mapping):
            continue
        normalized = dict(entry)
        if (
            normalized.get("client") == "summarizer"
            and normalized.get("mode") == "diagnostics"
            and normalized.get("status") == "error"
            and normalized.get("reason")
            in {"missing or empty report", "missing-report"}
        ):
            # Preserve stub line on disk but do not include it in rendered diagnostics.
            if normalized.get("extras", {}).get("stub") == "missing-report":
                _LAST_STUB_REASON_ALIAS = "missing-report"
            continue
        extras = normalized.get("extras")
        if isinstance(extras, Mapping):
            normalized["extras"] = _redact_extras(extras)
        _apply_run_details_override(normalized, base_dir=target.parent)
        entries.append(normalized)
    return entries


def _display_reason(reason: Any) -> str:
    if reason is None:
        return "unknown"
    text = str(reason).strip()
    if not text:
        return "unknown"
    if text.lower() == "missing or empty report":
        return "missing-report"
    return text


def _fmt_count(value: Optional[int | float]) -> str:
    """Back-compat count formatter shared with the legacy summary helpers."""

    if value is None:
        return EM_DASH
    try:
        if float(value) == 0:
            return EM_DASH
    except Exception:
        return str(value)
    return str(value)


def _redact_extras(payload: Mapping[str, Any]) -> Dict[str, Any]:
    sensitive_tokens = {"bearer", "authorization", "token", "secret", "password"}

    def _redact(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: _redact(val) for key, val in value.items()}
        if isinstance(value, list):
            return [_redact(item) for item in value]
        if isinstance(value, str):
            lowered = value.lower()
            if any(token in lowered for token in sensitive_tokens):
                return "***"
        return value

    return {key: _redact(val) for key, val in payload.items()}


def _apply_run_details_override(entry: Dict[str, Any], *, base_dir: Path) -> None:
    extras = entry.get("extras")
    if not isinstance(extras, Mapping):
        return
    details_path_value = extras.get("run_details_path")
    if not details_path_value:
        return
    details_path = _safe_path(details_path_value)
    if not details_path:
        return
    if not details_path.is_absolute():
        details_path = (base_dir / details_path).resolve()
    details_payload = safe_load_json(details_path)
    if not isinstance(details_payload, Mapping):
        return

    counts = entry.setdefault("counts", {})
    extras_dict = dict(extras)
    override_applied = False
    if isinstance(counts, Mapping):
        rows_section = details_payload.get("rows") or details_payload.get("counts")
        if isinstance(rows_section, Mapping):
            for key in ("fetched", "normalized", "written"):
                if key in rows_section and rows_section.get(key) is not None:
                    override_applied = True
                    try:
                        counts[key] = int(rows_section[key])
                    except (TypeError, ValueError):
                        counts[key] = rows_section[key]
        entry["counts"] = dict(counts)

    totals = details_payload.get("totals")
    if isinstance(totals, Mapping) and totals.get("rows_written") is not None:
        override_applied = True
        extras_dict["rows_written"] = totals.get("rows_written")
        extras_dict["run_totals"] = dict(totals)

    if override_applied:
        extras_dict["counts_override_source"] = "run.json"
    entry["extras"] = extras_dict


def build_markdown(
    entries: Sequence[Mapping[str, Any]] | None,
    diagnostics_root: os.PathLike[str] | str = DEFAULT_DIAG_DIR,
    staging_root: os.PathLike[str] | str = DEFAULT_STAGING_DIR,
) -> str:
    """Render the superreport markdown while preserving legacy sections."""

    diagnostics_dir = Path(diagnostics_root)
    staging_dir = Path(staging_root)
    normalized_entries = [entry for entry in entries or [] if isinstance(entry, Mapping)]
    records, staging_snapshot = _prepare_records(normalized_entries, diagnostics_dir, staging_dir)

    parts: List[str] = [LEGACY_TITLE, "", SUMMARY_TITLE, ""]
    parts.extend(_render_run_overview(records, normalized_entries))
    parts.extend(
        _render_legacy_sections(normalized_entries, diagnostics_dir, staging_dir, staging_snapshot)
    )
    parts.extend(_render_connector_matrix(records))
    parts.extend(_render_connector_details(records, staging_snapshot))
    parts.extend(_render_export_snapshot(staging_snapshot))
    parts.extend(_render_anomalies(records))
    parts.extend(_render_next_actions(records))
    return "\n".join(parts)


def _safe_path(value: Any) -> Path | None:
    if isinstance(value, str) and value.strip():
        return Path(value)
    return None


def _safe_read_csv_rows(path: Path, limit: int = 5) -> List[Mapping[str, Any]]:
    rows: List[Mapping[str, Any]] = []
    if not path:
        return rows
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for idx, row in enumerate(reader):
                if idx >= limit:
                    break
                rows.append({key: value for key, value in row.items()})
    except (OSError, ValueError, csv.Error):
        return []
    return rows


def _summarize_http_trace(path: Path) -> Dict[str, Any]:
    entries = safe_load_jsonl(path) if path else []
    if not entries:
        return {}
    latencies: List[float] = []
    paths: Counter[str] = Counter()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        endpoint = str(entry.get("path") or entry.get("endpoint") or "unknown")
        paths[endpoint] += 1
        latency = entry.get("elapsed_ms") or entry.get("latency_ms")
        try:
            latencies.append(float(latency))
        except (TypeError, ValueError):
            continue

    latencies.sort()

    def _percentile(percent: float) -> float | None:
        if not latencies:
            return None
        if len(latencies) == 1:
            return latencies[0]
        idx = max(0, min(len(latencies) - 1, int(round(percent * (len(latencies) - 1)))))
        return latencies[idx]

    summary: Dict[str, Any] = {
        "count": len(entries),
        "paths": paths.most_common(5),
        "latency_avg_ms": mean(latencies) if latencies else None,
        "latency_p50_ms": _percentile(0.5),
        "latency_p95_ms": _percentile(0.95),
        "latency_max_ms": max(latencies) if latencies else None,
    }
    return summary

def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _prepare_records(
    entries: Sequence[Mapping[str, Any]],
    diagnostics_dir: Path,
    staging_dir: Path,
) -> tuple[List[ConnectorRecord], Mapping[str, Any]]:
    records = [_normalise_connector(entry) for entry in entries]
    for record in records:
        _parse_connector_context(record, diagnostics_dir)

    staging_snapshot = _collect_staging_snapshot(staging_dir)
    for record in records:
        staging_info = staging_snapshot.get(record.name, {}) if isinstance(staging_snapshot, Mapping) else {}
        if isinstance(staging_info, Mapping):
            record.exported_flow = any(
                "flow" in name and meta.get("rows", 0)
                for name, meta in staging_info.items()
                if isinstance(meta, Mapping)
            )
            record.exported_stock = any(
                "stock" in name and meta.get("rows", 0)
                for name, meta in staging_info.items()
                if isinstance(meta, Mapping)
            )

    return records, staging_snapshot


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    if not rows:
        return ["(no data)"]
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join(["---"] * len(headers)) + " |"
    output = [header_line, divider]
    for row in rows:
        output.append("| " + " | ".join(str(cell) if cell not in (None, "") else "—" for cell in row) + " |")
    return output


def _format_bool(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return "yes" if value else "no"
    if isinstance(value, str):
        return "yes" if value.strip().lower() in {"1", "true", "on", "y", "yes"} else "no"
    return "yes" if value else "no"


def _render_legacy_sections(
    entries: Sequence[Mapping[str, Any]],
    diagnostics_dir: Path,
    staging_dir: Path,
    staging_snapshot: Mapping[str, Any],
) -> List[str]:
    lines: List[str] = []

    staging_lines = _legacy_staging_readiness_section(staging_dir)
    if staging_lines:
        lines.extend(staging_lines)
        lines.append("")

    sample_lines = _legacy_samples_section(diagnostics_dir)
    if sample_lines:
        lines.extend(sample_lines)
        lines.append("")

    config_lines = _legacy_config_section(entries)
    if config_lines:
        lines.extend(config_lines)
        lines.append("")

    selector_lines = _legacy_selector_section(entries, diagnostics_dir)
    if selector_lines:
        lines.extend(selector_lines)
        lines.append("")

    dtm_lines = _legacy_dtm_sections(entries)
    if dtm_lines:
        lines.extend(dtm_lines)
        lines.append("")

    reachability_lines = _legacy_dtm_reachability_section(diagnostics_dir)
    if reachability_lines:
        lines.extend(reachability_lines)
        lines.append("")

    zero_lines = _legacy_zero_row_section(entries, staging_snapshot)
    if zero_lines:
        lines.extend(zero_lines)
        lines.append("")

    log_lines = _legacy_logs_section(entries, diagnostics_dir)
    if log_lines:
        lines.extend(log_lines)
        lines.append("")

    return lines


def _legacy_staging_readiness_section(staging_dir: Path) -> List[str]:
    resolver_files = []
    if staging_dir.exists():
        resolver_files = sorted(p.name for p in staging_dir.glob("*.csv"))

    data_dir = Path("data") / "staging"
    data_files: List[str] = []
    if data_dir.exists():
        data_files = sorted(p.name for p in data_dir.glob("*.csv"))

    if not resolver_files and not data_files:
        return []

    lines = ["## Staging readiness", ""]
    lines.append(f"- resolver/staging files: {resolver_files or '[]'}")
    lines.append(f"- data/staging files: {data_files or '[]'}")
    if data_files and not resolver_files:
        lines.append(
            "- Hint: artifacts were written to `data/staging`; exporters expect `resolver/staging` (set RESOLVER_OUTPUT_DIR=resolver/staging)."
        )
    return lines


def _legacy_samples_section(diagnostics_dir: Path) -> List[str]:
    sample = diagnostics_dir / "dtm" / "samples" / "admin0_head.csv"
    if not sample.exists():
        return []
    lines = ["## Source sample: quick checks", "", "- `dtm/admin0_head.csv` rows present"]
    iso_counts = top_value_counts_from_csv(sample, "CountryISO3")
    if iso_counts:
        iso_text = ", ".join(f"{code} ({count})" for code, count in iso_counts)
        lines.append(f"- CountryISO3 top 5: {iso_text}")
    admin_counts = top_value_counts_from_csv(sample, "admin0Name")
    if admin_counts:
        admin_text = ", ".join(f"{name} ({count})" for name, count in admin_counts)
        lines.append(f"- admin0Name top 5: {admin_text}")
    return lines


def _legacy_config_section(entries: Sequence[Mapping[str, Any]]) -> List[str]:
    for entry in entries:
        extras = entry.get("extras") if isinstance(entry, Mapping) else None
        config = extras.get("config") if isinstance(extras, Mapping) else None
        if not isinstance(config, Mapping):
            continue
        path = config.get("config_path_used") or extras.get("config_path_used") or "unknown"
        source = config.get("config_source_label") or extras.get("config_source") or "unknown"
        warnings = config.get("config_warnings")
        lines = ["## Config used", ""]
        lines.append(f"- Config source: {source}")
        lines.append(f"- Config: {path}")
        if isinstance(warnings, Sequence) and warnings:
            lines.append("- warnings:")
            for warning in warnings:
                lines.append(f"  - {warning}")

        preview = config.get("countries_preview")
        if isinstance(preview, Sequence) and not isinstance(preview, (str, bytes)):
            preview_text = ", ".join(str(item) for item in list(preview)[:5] if str(item)) or "—"
            lines.append(f"- Countries preview: {preview_text}")

        config_parse = config.get("config_parse") if isinstance(config.get("config_parse"), Mapping) else {}
        config_keys_found = config.get("config_keys_found") if isinstance(config.get("config_keys_found"), Mapping) else {}
        parse_countries = []
        raw_countries = config_parse.get("countries") if isinstance(config_parse, Mapping) else None
        if isinstance(raw_countries, Sequence) and not isinstance(raw_countries, (str, bytes)):
            parse_countries = [str(item) for item in raw_countries if str(item)]
        parsed_admin_levels = []
        raw_levels = config_parse.get("admin_levels") if isinstance(config_parse, Mapping) else None
        if isinstance(raw_levels, Sequence) and not isinstance(raw_levels, (str, bytes)):
            parsed_admin_levels = [str(item) for item in raw_levels if str(item)]

        if parse_countries:
            lines.append(f"- **Countries parse:** api.countries: found ({len(parse_countries)})")
        else:
            lines.append("- **Countries parse:** api.countries: missing or empty")

        if parsed_admin_levels:
            levels_repr = ", ".join(parsed_admin_levels)
            lines.append(f"- **Admin levels parse:** api.admin_levels: found ([{levels_repr}])")
        else:
            lines.append("- **Admin levels parse:** api.admin_levels: missing or empty")

        if isinstance(config_keys_found, Mapping):
            countries_found = bool(config_keys_found.get("countries"))
            admin_found = bool(config_keys_found.get("admin_levels"))
            lines.append(
                "- **Config keys found:** countries={c}, admin_levels={a}".format(
                    c=str(countries_found).lower(),
                    a=str(admin_found).lower(),
                )
            )
            countries_mode = str(config.get("countries_mode", "discovered")).strip().lower()
            if countries_found and countries_mode == "discovered":
                lines.append(
                    "- ⚠ config had api.countries but selector list not applied (check loader/version)."
                )
        return lines
    return []


def _legacy_selector_section(
    entries: Sequence[Mapping[str, Any]], diagnostics_dir: Path
) -> List[str]:
    top_selectors: List[str] = []
    lines = ["## Selector effectiveness", ""]
    snippets: List[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        extras = entry.get("extras") if isinstance(entry.get("extras"), Mapping) else None
        if not isinstance(extras, Mapping):
            continue
        selector_payload = extras.get("selector")
        if not selector_payload:
            continue
        connector_id = entry.get("connector_id") or entry.get("connector") or "unknown"
        if isinstance(selector_payload, Mapping):
            top_payload = selector_payload.get("top_by_rows")
            if isinstance(top_payload, Sequence):
                for item in top_payload:
                    selector_name: str | None = None
                    rows_val: str | None = None
                    if isinstance(item, Mapping):
                        selector_name = str(
                            item.get("selector")
                            or item.get("name")
                            or item.get("id")
                            or item.get("value")
                            or ""
                        ).strip() or None
                        rows_val = item.get("rows") or item.get("row_count")
                    elif isinstance(item, Sequence) and len(item) >= 2:
                        selector_name = str(item[0]).strip() or None
                        rows_val = item[1]
                    if not selector_name:
                        continue
                    summary_val = rows_val
                    try:
                        if summary_val is not None:
                            summary_val = int(summary_val)
                    except (TypeError, ValueError):
                        summary_val = rows_val
                    if summary_val is None:
                        top_selectors.append(selector_name)
                    else:
                        top_selectors.append(f"{selector_name} ({summary_val})")
            coverage = selector_payload.get("coverage")
            matched = selector_payload.get("matched")
            snippets.append(
                f"- {connector_id}: selector diagnostics present (coverage={coverage}, matched={matched})"
            )
        else:
            snippets.append(f"- {connector_id}: selector diagnostics present")

    selector_files = [
        path
        for path in diagnostics_dir.rglob("selector*.json")
        if path.is_file()
    ]
    if selector_files:
        for path in selector_files:
            relative = path.relative_to(diagnostics_dir)
            snippets.append(f"- file: `{relative}`")

    if top_selectors:
        top_text = ", ".join(top_selectors[:5])
    else:
        top_text = "n/a"
    lines.append(f"- Top selectors by rows: {top_text}")
    if snippets:
        lines.extend([""])
        lines.extend(snippets)
    else:
        lines.append("- No selector diagnostics found; cannot compare coverage.")
    return lines


def _legacy_zero_row_section(
    entries: Sequence[Mapping[str, Any]], staging_snapshot: Mapping[str, Any]
) -> List[str]:
    zero_entries: List[Mapping[str, Any]] = []
    for entry in entries:
        counts = entry.get("counts") if isinstance(entry, Mapping) else None
        extras = entry.get("extras") if isinstance(entry, Mapping) else None
        written = None
        if isinstance(counts, Mapping):
            written = counts.get("written")
        if written is None and isinstance(extras, Mapping):
            written = extras.get("rows_written")
        if written is None:
            continue
        try:
            if float(written) == 0:
                zero_entries.append(entry)
        except Exception:
            continue

    if zero_entries:
        histogram = reason_histogram(zero_entries)
        if histogram:
            primary_reason = _display_reason(histogram.most_common(1)[0][0])
            top_reasons = ", ".join(
                f"{_display_reason(reason)} ({count})"
                for reason, count in histogram.most_common(5)
            )
        else:
            primary_reason = "unknown"
            top_reasons = "—"
        return [
            "## Zero-row root cause",
            "",
            "- One or more connectors produced zero rows.",
            f"- Primary reason: {primary_reason}",
            f"- Top drop reasons: {top_reasons}",
        ]

    for connector_meta in staging_snapshot.values():
        if not isinstance(connector_meta, Mapping):
            continue
        for meta in connector_meta.values():
            if not isinstance(meta, Mapping):
                continue
            rows = meta.get("rows")
            try:
                if rows is not None and int(rows) == 0:
                    return [
                        "## Zero-row root cause",
                        "",
                        "- One or more connectors produced zero rows.",
                        "- Primary reason: unknown",
                        "- Top drop reasons: —",
                    ]
            except (TypeError, ValueError):
                continue
    return []


def _legacy_dtm_sections(entries: Sequence[Mapping[str, Any]]) -> List[str]:
    target: Mapping[str, Any] | None = None
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        connector_id = str(entry.get("connector_id") or "")
        extras = entry.get("extras") if isinstance(entry.get("extras"), Mapping) else {}
        if connector_id == "dtm_client" or (
            isinstance(extras, Mapping)
            and (extras.get("date_filter") or extras.get("per_country"))
        ):
            target = entry
            break
    if not target:
        return []

    lines: List[str] = []
    date_filter_lines = _render_dtm_date_filter_section(target)
    if date_filter_lines:
        lines.extend(date_filter_lines)
        lines.append("")
    per_country_lines = _render_dtm_per_country_section(target)
    if per_country_lines:
        lines.extend(per_country_lines)
        lines.append("")
    return lines


def _render_dtm_date_filter_section(entry: Mapping[str, Any]) -> List[str]:
    extras = entry.get("extras") if isinstance(entry.get("extras"), Mapping) else {}
    date_filter = extras.get("date_filter") if isinstance(extras, Mapping) else None
    if not isinstance(date_filter, Mapping):
        return []

    parsed_ok = _coerce_int(date_filter.get("parsed_ok"))
    parsed_total = _coerce_int(date_filter.get("parsed_total"))
    percentage = (
        f"{round((parsed_ok / parsed_total) * 100):d}%" if parsed_total else "0%"
    )
    inside = _coerce_int(date_filter.get("inside"))
    outside = _coerce_int(date_filter.get("outside"))
    parse_failed = _coerce_int(date_filter.get("parse_failed"))
    min_date = date_filter.get("min_date") or EM_DASH
    max_date = date_filter.get("max_date") or EM_DASH
    window_start = date_filter.get("window_start") or EM_DASH
    window_end = date_filter.get("window_end") or EM_DASH
    inclusive_text = "inclusive" if date_filter.get("inclusive", True) else "exclusive"
    skipped = _format_bool(date_filter.get("skipped"))
    column_used = str(date_filter.get("date_column_used") or EM_DASH)
    drop_counts = date_filter.get("drop_counts") if isinstance(date_filter.get("drop_counts"), Mapping) else {}
    artifacts = extras.get("artifacts") if isinstance(extras.get("artifacts"), Mapping) else {}
    sample_path = artifacts.get("normalize_drops") if isinstance(artifacts, Mapping) else None

    lines = ["## DTM Date Filter", ""]
    lines.append(f"- **Date column:** `{column_used}`")
    lines.append(f"- **Parsed:** {parsed_ok}/{parsed_total} ({percentage})")
    lines.append(f"- **Min/Max date:** {min_date} → {max_date}")
    lines.append(f"- **Inside window:** {inside}")
    lines.append(f"- **Outside window:** {outside}")
    if parse_failed:
        lines.append(f"- **Parse failures:** {parse_failed}")
    lines.append(f"- **Window:** {window_start} → {window_end} ({inclusive_text})")
    lines.append(f"- **Skipped filter:** {skipped}")
    if isinstance(drop_counts, Mapping) and drop_counts:
        formatted = ", ".join(
            f"{reason} ({_coerce_int(count)})" for reason, count in sorted(drop_counts.items())
        )
        if formatted:
            lines.append(f"- **Drop reasons:** {formatted}")
    if isinstance(sample_path, str) and sample_path:
        lines.append(f"- **Drop sample:** `{sample_path}`")
    return lines


def _render_dtm_per_country_section(entry: Mapping[str, Any]) -> List[str]:
    extras = entry.get("extras") if isinstance(entry.get("extras"), Mapping) else {}
    per_country = extras.get("per_country") if isinstance(extras, Mapping) else None
    if not isinstance(per_country, list) or not per_country:
        return []

    header = ["country", "level", "param", "pages", "rows", "skipped_no_match", "reason"]
    rows: List[List[str]] = []
    for item in per_country:
        if not isinstance(item, Mapping):
            continue
        country = str(item.get("country") or EM_DASH)
        level = str(item.get("level") or EM_DASH)
        param = str(item.get("param") or EM_DASH)
        if param.lower() == "iso3":
            param = "CountryISO3"
        elif param.lower() == "name":
            param = "CountryName"
        pages = str(_coerce_int(item.get("pages")))
        rows_count = str(_coerce_int(item.get("rows")))
        skipped = str(item.get("skipped_no_match") or "")
        reason = str(item.get("reason") or "")
        rows.append([country, level, param, pages, rows_count, skipped, reason])

    if not rows:
        return []

    lines = ["## DTM per-country results", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _legacy_dtm_reachability_section(diagnostics_dir: Path) -> List[str]:
    reachability_path = diagnostics_dir / "dtm" / "reachability.json"
    payload = safe_load_json(reachability_path)
    if not isinstance(payload, (Mapping, list)):
        return []

    lines = ["## DTM Reachability", ""]
    summary_lines: List[str] = []
    if isinstance(payload, Mapping):
        overall = payload.get("overall_status") or payload.get("status") or payload.get("ok")
        if overall is not None:
            summary_lines.append(f"- Overall: {overall}")
        checks = payload.get("checks") or payload.get("endpoints") or payload.get("results")
        if isinstance(checks, Mapping):
            for name, info in sorted(checks.items()):
                if not isinstance(info, Mapping):
                    continue
                status = info.get("status") or info.get("ok")
                latency = info.get("latency_ms") or info.get("latency")
                host = info.get("host") or info.get("ip")
                snippet = f"- {name}: {status}"
                if host:
                    snippet += f" host={host}"
                if latency is not None:
                    snippet += f" latency={latency}ms"
                summary_lines.append(snippet)
        elif isinstance(checks, list):
            for info in checks[:5]:
                if not isinstance(info, Mapping):
                    continue
                name = info.get("name") or info.get("endpoint") or info.get("stage") or "check"
                status = info.get("status") or info.get("ok")
                latency = info.get("latency_ms") or info.get("latency")
                snippet = f"- {name}: {status}"
                if latency is not None:
                    snippet += f" latency={latency}ms"
                summary_lines.append(snippet)
        if not summary_lines:
            summary_lines.append("- Reachability diagnostics present.")
    else:
        summary_lines.append("- Reachability diagnostics present.")

    lines.extend(summary_lines)
    return lines


def _legacy_logs_section(
    entries: Sequence[Mapping[str, Any]], diagnostics_dir: Path
) -> List[str]:
    logs = gather_log_files(diagnostics_dir)
    meta_files = gather_meta_json_files(diagnostics_dir)

    logs_by_name = {path.stem: path for path in logs}
    meta_info: Dict[str, Dict[str, Any]] = {}
    for meta_path in meta_files:
        connector_name = meta_path.parent.name or meta_path.stem
        bucket = meta_info.setdefault(connector_name, {"rows": 0, "paths": []})
        bucket["paths"].append(meta_path)
        payload = safe_load_json(meta_path)
        rows_value = None
        if isinstance(payload, Mapping):
            rows_value = payload.get("rows_written")
            if rows_value is None and isinstance(payload.get("meta"), Mapping):
                rows_value = payload["meta"].get("rows_written")
        try:
            if rows_value is not None:
                bucket["rows"] += int(rows_value)
        except (TypeError, ValueError):
            continue

    lines = ["## Logs", ""]
    lines.append("| Connector | Status | Reason | Logs | Meta rows | Meta |")
    lines.append("| --- | --- | --- | --- | ---: | --- |")

    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("connector_id") or entry.get("connector") or "unknown")
        seen.add(name)
        extras = entry.get("extras") if isinstance(entry.get("extras"), Mapping) else {}
        status = extras.get("status_raw") or entry.get("status") or EM_DASH
        reason_value = entry.get("reason")
        if reason_value is None and isinstance(extras, Mapping):
            reason_value = extras.get("reason")
        reason_text = _display_reason(reason_value) if reason_value else EM_DASH
        log_path = logs_by_name.get(name)
        if log_path is not None:
            try:
                logs_cell = log_path.relative_to(diagnostics_dir).as_posix()
            except ValueError:
                logs_cell = log_path.as_posix()
        else:
            logs_cell = EM_DASH
        meta_bucket = meta_info.get(name, {})
        meta_rows = meta_bucket.get("rows") or 0
        meta_paths = meta_bucket.get("paths") or []
        if meta_paths:
            try:
                meta_cell = ", ".join(
                    path.relative_to(diagnostics_dir).as_posix() if path.is_relative_to(diagnostics_dir) else path.as_posix()
                    for path in meta_paths[:2]
                )
            except AttributeError:
                converted: List[str] = []
                for path in meta_paths[:2]:
                    try:
                        converted.append(path.relative_to(diagnostics_dir).as_posix())
                    except ValueError:
                        converted.append(path.as_posix())
                meta_cell = ", ".join(converted)
        else:
            meta_cell = EM_DASH
        lines.append(
            "| {name} | {status} | {reason} | {logs_cell} | {rows} | {meta_cell} |".format(
                name=name,
                status=status,
                reason=reason_text,
                logs_cell=logs_cell,
                rows=_fmt_count(meta_rows),
                meta_cell=meta_cell,
            )
        )

    for name, path in logs_by_name.items():
        if name in seen:
            continue
        try:
            logs_cell = path.relative_to(diagnostics_dir).as_posix()
        except ValueError:
            logs_cell = path.as_posix()
        meta_bucket = meta_info.get(name, {})
        meta_rows = meta_bucket.get("rows") or 0
        meta_paths = meta_bucket.get("paths") or []
        if meta_paths:
            try:
                meta_cell = ", ".join(
                    sub.relative_to(diagnostics_dir).as_posix()
                    if sub.is_relative_to(diagnostics_dir)
                    else sub.as_posix()
                    for sub in meta_paths[:2]
                )
            except AttributeError:
                fallback: List[str] = []
                for sub in meta_paths[:2]:
                    try:
                        fallback.append(sub.relative_to(diagnostics_dir).as_posix())
                    except ValueError:
                        fallback.append(sub.as_posix())
                meta_cell = ", ".join(fallback)
        else:
            meta_cell = EM_DASH
        lines.append(
            "| {name} | {status} | {reason} | {logs_cell} | {rows} | {meta_cell} |".format(
                name=name,
                status=EM_DASH,
                reason=EM_DASH,
                logs_cell=logs_cell,
                rows=_fmt_count(meta_rows),
                meta_cell=meta_cell,
            )
        )

    if not logs and not meta_files:
        lines.append("| — | — | — | — | — | — |")

    lines.append("")
    if logs:
        for file in logs:
            try:
                rel = file.relative_to(diagnostics_dir)
            except ValueError:
                rel = file
            lines.append(f"- {rel.as_posix()}")
    else:
        lines.append("- No log files discovered.")
    if meta_files:
        lines.append("")
        lines.append("- Meta diagnostics present")
    elif logs:
        lines.append("")
        lines.append("- No meta diagnostics found.")
    return lines


@dataclass
class ConnectorRecord:
    name: str
    status: str
    mode: str = "real"
    reason: str | None = None
    started_at: str | None = None
    duration_ms: int = 0
    counts: Dict[str, int] = field(default_factory=dict)
    http: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
    coverage: Dict[str, Any] = field(default_factory=dict)
    samples: Dict[str, Any] = field(default_factory=dict)
    reachability: Dict[str, Any] = field(default_factory=dict)
    config_source: str | None = None
    config_path: str | None = None
    countries_count: int | None = None
    countries_sample: List[str] = field(default_factory=list)
    window_start: str | None = None
    window_end: str | None = None
    exported_flow: bool | None = None
    exported_stock: bool | None = None
    why_zero: Mapping[str, Any] | None = None
    error: Mapping[str, Any] | None = None
    drop_histogram: Mapping[str, int] = field(default_factory=dict)
    loader_warnings: List[str] = field(default_factory=list)
    detail_blocks: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def rows_fetched(self) -> int:
        return _coerce_int(self.counts.get("fetched"))

    @property
    def rows_normalized(self) -> int:
        return _coerce_int(self.counts.get("normalized"))

    @property
    def rows_written(self) -> int:
        return _coerce_int(self.counts.get("written"))


def _normalise_connector(entry: Mapping[str, Any]) -> ConnectorRecord:
    name = str(entry.get("connector_id") or entry.get("name") or "unknown")
    status = str(entry.get("status") or entry.get("status_raw") or "unknown").lower()
    mode = str(entry.get("mode") or entry.get("extras", {}).get("mode") or "real")
    extras = dict(entry.get("extras") or {})
    counts = dict(entry.get("counts") or {})
    http = dict(entry.get("http") or {})
    coverage = dict(entry.get("coverage") or {})
    samples = dict(entry.get("samples") or {})
    started_at = entry.get("started_at_utc") or extras.get("started_at_utc")
    duration_ms = _coerce_int(entry.get("duration_ms") or extras.get("duration_ms"))
    reason = entry.get("reason") or extras.get("reason")
    meta = dict(entry.get("meta") or {})
    record = ConnectorRecord(
        name=name,
        status=status,
        mode=mode,
        reason=str(reason) if reason else None,
        started_at=str(started_at) if started_at else None,
        duration_ms=duration_ms,
        counts={key: _coerce_int(value) for key, value in counts.items()},
        http=http,
        extras=extras,
        coverage=coverage,
        samples=samples,
        meta=meta,
    )
    return record


def _load_optional(path: Path) -> Mapping[str, Any]:
    payload = safe_load_json(path)
    return payload if payload else {}


def _parse_connector_context(record: ConnectorRecord, base_dir: Path) -> None:
    connector_dir = base_dir / record.name
    reachability_path = connector_dir / "reachability.json"
    manifest_path = connector_dir / "manifest.json"
    normalize_path = connector_dir / "normalize.json"
    drop_path = connector_dir / "drop_reasons.json"
    why_zero_path = connector_dir / "why_zero.json"
    error_path = connector_dir / "error.json"
    http_summary_path = connector_dir / "http_summary.json"

    record.reachability = dict(_load_optional(reachability_path))
    record.detail_blocks["manifest"] = _load_optional(manifest_path)

    normalize_payload = _load_optional(normalize_path)
    if normalize_payload:
        drop_hist = normalize_payload.get("drop_reasons")
        if isinstance(drop_hist, Mapping):
            record.drop_histogram = {str(k): _coerce_int(v) for k, v in drop_hist.items()}
        chosen = normalize_payload.get("chosen_columns")
        if isinstance(chosen, Mapping):
            record.detail_blocks["chosen_columns"] = chosen

    drop_payload = _load_optional(drop_path)
    if drop_payload and not record.drop_histogram:
        if isinstance(drop_payload, Mapping):
            record.drop_histogram = {str(k): _coerce_int(v) for k, v in drop_payload.items()}

    why_payload = _load_optional(why_zero_path)
    if why_payload:
        record.why_zero = why_payload
        warnings = why_payload.get("loader_warnings")
        if isinstance(warnings, list):
            record.loader_warnings = [str(item) for item in warnings if str(item)]
        record.config_source = why_payload.get("config_source") or record.config_source
        path = why_payload.get("config_path_used")
        if isinstance(path, str):
            record.config_path = path
        record.countries_count = why_payload.get("countries_count")
        sample = why_payload.get("countries_sample")
        if isinstance(sample, list):
            record.countries_sample = [str(item) for item in sample if str(item)]

    error_payload = _load_optional(error_path)
    if error_payload:
        record.error = error_payload

    http_summary = _load_optional(http_summary_path)
    if http_summary:
        record.detail_blocks["http_summary"] = http_summary

    config_block = record.extras.get("config") if isinstance(record.extras, Mapping) else {}
    if isinstance(config_block, Mapping):
        if not record.config_source:
            record.config_source = str(config_block.get("config_source_label") or config_block.get("config_source") or "") or None
        if not record.config_path:
            path = config_block.get("config_path_used") or config_block.get("config_path")
            if isinstance(path, str):
                record.config_path = path
        warnings = config_block.get("config_warnings")
        if isinstance(warnings, Sequence) and not isinstance(warnings, (str, bytes)):
            record.loader_warnings.extend(str(item) for item in warnings if str(item))
    sample_block = record.samples.get("top_iso3") if isinstance(record.samples, Mapping) else None
    if isinstance(sample_block, list) and not record.countries_sample:
        record.countries_sample = [
            str(item[0]) if isinstance(item, (list, tuple)) and item else str(item)
            for item in sample_block[:5]
        ]
    coverage = record.coverage
    record.window_start = str(coverage.get("ym_min") or coverage.get("as_of_min") or "") or None
    record.window_end = str(coverage.get("ym_max") or coverage.get("as_of_max") or "") or None


def _gather_run_stats(records: Sequence[ConnectorRecord]) -> Dict[str, Any]:
    earliest: _dt.datetime | None = None
    latest: _dt.datetime | None = None
    for record in records:
        if record.started_at:
            try:
                start = _dt.datetime.fromisoformat(record.started_at.replace("Z", "+00:00"))
            except ValueError:
                start = None
        else:
            start = None
        if start:
            duration = _dt.timedelta(milliseconds=record.duration_ms)
            end = start + duration
            if earliest is None or start < earliest:
                earliest = start
            if latest is None or end > latest:
                latest = end
    return {
        "start": earliest.isoformat() if earliest else None,
        "end": latest.isoformat() if latest else None,
        "duration": str(latest - earliest) if earliest and latest else None,
    }


def _git_snapshot() -> Dict[str, Any]:
    def _run(*cmd: str) -> str | None:
        try:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except OSError:
            return None
        output = result.stdout.strip()
        return output or None

    sha = _run("git", "rev-parse", "HEAD")
    branch = _run("git", "rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(_run("git", "status", "--porcelain"))
    return {"sha": sha, "branch": branch, "dirty": dirty}


def _version_snapshot() -> Dict[str, Any]:
    snapshot = {
        "python": sys.version.split()[0],
        "pip": None,
        "duckdb": None,
        "pandas": None,
        "platform": platform.platform(),
        "glibc": None,
        "timezone": os.environ.get("TZ"),
        "locale": os.environ.get("LC_ALL") or os.environ.get("LANG"),
    }
    try:
        import pip  # type: ignore

        snapshot["pip"] = getattr(pip, "__version__", None)
    except Exception:
        snapshot["pip"] = None
    try:
        import duckdb  # type: ignore

        snapshot["duckdb"] = getattr(duckdb, "__version__", None)
    except Exception:
        pass
    try:
        import pandas  # type: ignore

        snapshot["pandas"] = getattr(pandas, "__version__", None)
    except Exception:
        pass
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        version = getattr(libc, "gnu_get_libc_version", None)
        if callable(version):
            snapshot["glibc"] = version().decode("utf-8") if hasattr(version(), "decode") else version()
    except Exception:
        snapshot["glibc"] = None
    return snapshot


def _env_flags() -> Dict[str, Any]:
    flags = {key: value for key, value in os.environ.items() if key.startswith("RESOLVER_")}
    for key in ("SUMMARY_VERBOSE", "SUMMARY_LOG_TAIL_KB", "ONLY_CONNECTOR", "EMPTY_POLICY"):
        value = os.environ.get(key)
        if value is not None:
            flags[key] = value
    return dict(sorted(flags.items()))


def _secret_presence() -> Dict[str, bool]:
    secrets = {}
    for key in sorted(os.environ.keys()):
        if any(token in key for token in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")):
            secrets[key] = bool(os.environ.get(key, "").strip())
    return secrets


def _date_windows(records: Sequence[ConnectorRecord]) -> Dict[str, Any]:
    global_window = {
        "start": os.environ.get("RESOLVER_GLOBAL_WINDOW_START"),
        "end": os.environ.get("RESOLVER_GLOBAL_WINDOW_END"),
    }
    per_connector: Dict[str, Dict[str, Any]] = {}
    for record in records:
        if record.window_start or record.window_end:
            per_connector[record.name] = {
                "start": record.window_start,
                "end": record.window_end,
            }
    return {"global": global_window, "per_connector": per_connector}


def _resource_snapshot() -> Dict[str, Any]:
    disk = shutil.disk_usage(".")
    info: Dict[str, Any] = {
        "disk_free_mb": int(disk.free / (1024 * 1024)),
        "disk_total_mb": int(disk.total / (1024 * 1024)),
    }
    try:
        import psutil  # type: ignore

        mem = psutil.virtual_memory()
        info["mem_total_mb"] = int(mem.total / (1024 * 1024))
        info["mem_available_mb"] = int(mem.available / (1024 * 1024))
    except Exception:
        pass
    return info


def _collect_staging_snapshot(staging_dir: Path) -> Dict[str, Any]:
    snapshot: Dict[str, Dict[str, Any]] = {}
    if not staging_dir.exists():
        return snapshot
    for connector_dir in staging_dir.iterdir():
        if not connector_dir.is_dir():
            continue
        metrics: Dict[str, Any] = {}
        for file in connector_dir.iterdir():
            if not file.is_file() or file.suffix.lower() not in {".csv", ".tsv", ".json", ".parquet", ".jsonl"}:
                continue
            try:
                size = file.stat().st_size
            except OSError:
                size = 0
            rows = _count_rows(file)
            metrics[file.name] = {"path": file.as_posix(), "size": size, "rows": rows}
        snapshot[connector_dir.name] = metrics
    return snapshot


def _count_rows(path: Path) -> int:
    if path.suffix.lower() not in {".csv", ".tsv"}:
        return 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            next(handle, None)
            return sum(1 for _ in handle)
    except OSError:
        return 0


def _format_status_counts(entries: Sequence[Mapping[str, Any]]) -> str:
    counter = status_histogram(entries)
    if not counter:
        return "none"
    return ", ".join(f"{key}={counter[key]}" for key in sorted(counter))


def _format_reason_histogram(entries: Sequence[Mapping[str, Any]]) -> str:
    counter = reason_histogram(entries)
    if counter:
        parts = [f"{_display_reason(reason)}={counter[reason]}" for reason in sorted(counter)]
        return ", ".join(parts)
    if _LAST_STUB_REASON_ALIAS:
        return f"{_LAST_STUB_REASON_ALIAS}=1"
    return EM_DASH


def _rows_totals(records: Sequence[ConnectorRecord]) -> tuple[int, int, int, bool]:
    fetched = sum(record.rows_fetched for record in records)
    normalized = sum(record.rows_normalized for record in records)
    written = sum(record.rows_written for record in records)
    override = any(
        isinstance(record.extras, Mapping)
        and record.extras.get("counts_override_source") == "run.json"
        for record in records
    )
    return fetched, normalized, written, override


def _render_run_overview(
    records: Sequence[ConnectorRecord],
    entries: Sequence[Mapping[str, Any]],
) -> List[str]:
    lines = ["## Run Overview", ""]
    lines.append(f"* **Connectors:** {len(records)}")
    lines.append(
        f"* **Status counts:** {_format_status_counts(entries)}"
    )
    reason_hist = _format_reason_histogram(entries)
    if reason_hist != EM_DASH:
        lines.append(f"* **Reason histogram:** {reason_hist}")
    fetched, normalized, written, override = _rows_totals(records)
    footnote = " (from run.json)" if override else ""
    lines.append(f"* **Rows fetched:** {fetched}{footnote}")
    lines.append(f"* **Rows normalized:** {normalized}{footnote}")
    lines.append(f"* **Rows written:** {written}{footnote}")
    lines.append("")
    git_info = _git_snapshot()
    version_info = _version_snapshot()
    run_stats = _gather_run_stats(records)
    flags = _env_flags()
    secrets = _secret_presence()
    windows = _date_windows(records)
    resources = _resource_snapshot()

    lines.append("### Git & Timing")
    lines.append(f"- Commit: `{git_info.get('sha') or 'unknown'}` (branch: {git_info.get('branch') or 'unknown'})")
    lines.append(f"- Dirty workspace: {_format_bool(git_info.get('dirty'))}")
    lines.append(f"- Start: {run_stats.get('start') or 'n/a'}")
    lines.append(f"- End: {run_stats.get('end') or 'n/a'}")
    lines.append(f"- Duration: {run_stats.get('duration') or 'n/a'}")
    lines.append("")

    lines.append("### Host & Versions")
    lines.append(f"- Python: {version_info.get('python')}")
    lines.append(f"- pip: {version_info.get('pip') or 'unknown'}")
    lines.append(f"- duckdb: {version_info.get('duckdb') or 'unknown'}")
    lines.append(f"- pandas: {version_info.get('pandas') or 'unknown'}")
    lines.append(f"- Platform: {version_info.get('platform')}")
    lines.append(f"- glibc: {version_info.get('glibc') or 'unknown'}")
    lines.append(f"- Locale: {version_info.get('locale') or 'unknown'}")
    lines.append(f"- Timezone: {version_info.get('timezone') or 'unknown'}")
    lines.append("")

    lines.append("### Global Flags")
    if flags:
        for key, value in flags.items():
            lines.append(f"- `{key}` = `{value}`")
    else:
        lines.append("- (no RESOLVER_* flags detected)")
    lines.append("")

    lines.append("### Secrets Presence")
    if secrets:
        for key, present in secrets.items():
            lines.append(f"- {key}: {_format_bool(present)}")
    else:
        lines.append("- (no matching secrets detected)")
    lines.append("")

    lines.append("### Date Windows")
    global_window = windows["global"]
    lines.append(f"- Global: {global_window.get('start') or '—'} → {global_window.get('end') or '—'}")
    per_connector = windows["per_connector"]
    if per_connector:
        for connector, window in sorted(per_connector.items()):
            lines.append(f"- {connector}: {window.get('start') or '—'} → {window.get('end') or '—'}")
    lines.append("")

    lines.append("### Resources")
    lines.append(f"- Disk free: {resources.get('disk_free_mb')} MB")
    lines.append(f"- Disk total: {resources.get('disk_total_mb')} MB")
    if "mem_total_mb" in resources:
        lines.append(f"- Memory: {resources.get('mem_available_mb')} MB free / {resources.get('mem_total_mb')} MB total")
    lines.append("")

    module = sys.modules.get(__name__)
    exported: List[str] = []
    for name in ("render_summary_md", "render_dtm_deep_dive", "_render_dtm_deep_dive"):
        if module and hasattr(module, name):
            exported.append(name)
    lines.append(
        f"**Summarizer API (exported):** {', '.join(exported) if exported else '(none)'}"
    )
    lines.append("")
    return lines


def _render_connector_matrix(records: Sequence[ConnectorRecord]) -> List[str]:
    lines = ["## Connector Diagnostics Matrix", ""]
    header = (
        "| id | mode | status | reason | last_http | http 2xx/4xx/5xx (retries) | "
        "rows f/n/w | rows_written | dropped | parse_errors | ym_min/ym_max | asof_min/asof_max | rows_method | "
        "top_iso3 | top_hazard | meta.status | meta.rows | meta.path |"
    )
    divider = (
        "| --- | --- | --- | --- | --- | --- | --- | ---:| ---:| ---:| --- | --- | --- | --- | ---:| --- |"
    )
    lines.append(header)
    lines.append(divider)

    for record in sorted(records, key=lambda item: item.name):
        http = record.http or {}
        retries = _coerce_int(http.get("retries"))
        http_counts = (
            f"{_coerce_int(http.get('2xx'))}/{_coerce_int(http.get('4xx'))}/{_coerce_int(http.get('5xx'))}"
        )
        http_text = f"{http_counts} ({retries})"
        last_http = http.get("last_status")
        last_http_text = str(last_http) if last_http not in (None, "") else EM_DASH

        extras_mapping = record.extras if isinstance(record.extras, Mapping) else {}
        rows_written_value = extras_mapping.get("rows_written") if isinstance(extras_mapping, Mapping) else None
        if rows_written_value is None:
            rows_written_value = record.rows_written
        rows_written_text = _fmt_count(rows_written_value)

        run_totals = extras_mapping.get("run_totals") if isinstance(extras_mapping, Mapping) else {}
        dropped_value = run_totals.get("dropped") if isinstance(run_totals, Mapping) else None
        parse_errors_value = run_totals.get("parse_errors") if isinstance(run_totals, Mapping) else None
        dropped_text = _fmt_count(dropped_value)
        parse_errors_text = _fmt_count(parse_errors_value)

        ym_min = record.coverage.get("ym_min") if isinstance(record.coverage, Mapping) else None
        ym_max = record.coverage.get("ym_max") if isinstance(record.coverage, Mapping) else None
        if ym_min or ym_max:
            ym_min_text = str(ym_min or EM_DASH)
            ym_max_text = str(ym_max or EM_DASH)
            ym_text = f"{ym_min_text}/{ym_max_text}"
        else:
            ym_text = EM_DASH

        asof_min = record.coverage.get("as_of_min") if isinstance(record.coverage, Mapping) else None
        asof_max = record.coverage.get("as_of_max") if isinstance(record.coverage, Mapping) else None
        if asof_min or asof_max:
            asof_min_text = str(asof_min or EM_DASH)
            asof_max_text = str(asof_max or EM_DASH)
            asof_text = f"{asof_min_text}/{asof_max_text}"
        else:
            asof_text = EM_DASH

        rows_method = extras_mapping.get("rows_method") if isinstance(extras_mapping, Mapping) else None
        rows_method_text = str(rows_method) if rows_method else EM_DASH

        def _format_samples(values: Any) -> str:
            if not isinstance(values, (list, tuple)):
                return EM_DASH
            items = []
            for name, count in values:
                try:
                    label = str(name)
                    qty = int(count)
                except (TypeError, ValueError):
                    continue
                if not label:
                    continue
                items.append(f"`{label}` ({qty})")
            return ", ".join(items) if items else EM_DASH

        top_iso3 = _format_samples(record.samples.get("top_iso3"))
        top_hazard = _format_samples(record.samples.get("top_hazard"))

        meta_status = record.meta.get("status") if isinstance(record.meta, Mapping) else None
        meta_rows_value = None
        if isinstance(record.meta, Mapping):
            for key in ("rows", "row_count", "rows_written"):
                if record.meta.get(key) is not None:
                    meta_rows_value = record.meta.get(key)
                    break
        meta_rows_text = _fmt_count(meta_rows_value)
        meta_status_text = str(meta_status) if meta_status else EM_DASH

        meta_path = extras_mapping.get("meta_path") if isinstance(extras_mapping, Mapping) else None
        meta_path_text = str(meta_path) if meta_path else EM_DASH

        reason_text = _display_reason(record.reason) if record.reason else EM_DASH
        status_raw = extras_mapping.get("status_raw") if isinstance(extras_mapping, Mapping) else None
        status_text = str(status_raw or record.status or EM_DASH)

        mode = record.mode or "real"

        rows_line = f"{record.rows_fetched}/{record.rows_normalized}/{record.rows_written}"

        line = "| {id} | {mode} | {status} | {reason} | {last} | {http} | {rows} | {written} | {dropped} | {parse_errors} | {ym} | {asof} | {method} | {iso3} | {hazard} | {meta_status} | {meta_rows} | {meta_path} |".format(
            id=record.name,
            mode=mode,
            status=status_text,
            reason=reason_text,
            last=last_http_text,
            http=http_text,
            rows=rows_line,
            written=rows_written_text,
            dropped=dropped_text,
            parse_errors=parse_errors_text,
            ym=ym_text,
            asof=asof_text,
            method=rows_method_text,
            iso3=top_iso3,
            hazard=top_hazard,
            meta_status=meta_status_text,
            meta_rows=meta_rows_text,
            meta_path=meta_path_text,
        )
        lines.append(line)

    lines.append("")
    return lines


def _render_connector_details(records: Sequence[ConnectorRecord], staging: Mapping[str, Any]) -> List[str]:
    lines: List[str] = ["## Connector Deep Dives", ""]
    for record in sorted(records, key=lambda item: item.name):
        lines.append("<details>")
        lines.append(f"<summary>{record.name}: status={record.status}</summary>")
        lines.append("")

        lines.append("### Config & Inputs")
        lines.append(f"- Config source: {record.config_source or 'unknown'}")
        lines.append(f"- Config path: {record.config_path or 'unknown'}")
        if record.loader_warnings:
            lines.append(f"- Loader warnings: {'; '.join(record.loader_warnings)}")
        if record.countries_sample:
            sample_text = ", ".join(record.countries_sample)
            lines.append(f"- Countries ({record.countries_count or len(record.countries_sample)}): {sample_text}")
        else:
            lines.append("- Countries: n/a")
        lines.append(f"- Window: {record.window_start or '—'} → {record.window_end or '—'}")
        series = record.extras.get("series")
        if series:
            lines.append(f"- Series requested: {series}")
        flags = record.extras.get("flags")
        if flags:
            lines.append(f"- Flags: {flags}")
        if isinstance(record.extras, Mapping):
            extras_pairs: List[str] = []
            for key, value in sorted(record.extras.items()):
                if key in {"series", "flags"}:
                    continue
                if value in (None, "", [], {}, ()):  # skip empties
                    continue
                if isinstance(value, (dict, list, tuple)):
                    continue
                extras_pairs.append(f"{key}={value}")
            if extras_pairs:
                lines.append(f"- Extras: {', '.join(extras_pairs)}")
        lines.append("")

        lines.append("### Reachability & HTTP")
        reach = record.reachability or {}
        lines.append(f"- DNS → IP: {reach.get('dns_ip') or 'n/a'}")
        lines.append(f"- TCP latency: {reach.get('tcp_ms') or 'n/a'} ms")
        lines.append(f"- TLS ok: {reach.get('tls_ok') if 'tls_ok' in reach else 'n/a'}")
        http = record.http or {}
        lines.append(
            "- HTTP summary: requests={req} 2xx={s} 4xx={c} 5xx={e} retries={r} timeouts={t}".format(
                req=_coerce_int(http.get("requests") or http.get("total")),
                s=_coerce_int(http.get("2xx")),
                c=_coerce_int(http.get("4xx")),
                e=_coerce_int(http.get("5xx")),
                r=_coerce_int(http.get("retries")),
                t=_coerce_int(http.get("timeouts")),
            )
        )
        http_summary = record.detail_blocks.get("http_summary") or {}
        endpoints = http_summary.get("endpoints")
        if isinstance(endpoints, list) and endpoints:
            lines.append("- Endpoints:")
            for endpoint in endpoints[:5]:
                if not isinstance(endpoint, Mapping):
                    continue
                lines.append(
                    "  - {path}: {count} req (p95={p95}ms max={max}ms)".format(
                        path=endpoint.get("path", "unknown"),
                        count=endpoint.get("count", "?"),
                        p95=endpoint.get("p95_ms", "?"),
                        max=endpoint.get("max_ms", "?"),
                    )
                )
        lines.append("")

        lines.append("### Normalization & Filters")
        drop_hist = record.drop_histogram or {}
        if drop_hist:
            for reason, count in sorted(drop_hist.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"- Drop {reason}: {count}")
        else:
            lines.append("- Drop reasons: none recorded")
        lines.append(f"- Rows fetched/normalized/written: {record.rows_fetched}/{record.rows_normalized}/{record.rows_written}")
        lines.append("")

        lines.append("### Outputs")
        staging_info = staging.get(record.name, {}) if isinstance(staging, Mapping) else {}
        if staging_info:
            for file_name, meta in staging_info.items():
                lines.append(
                    f"- {file_name}: rows={meta.get('rows', 0)} size={meta.get('size', 0)}"
                )
        else:
            lines.append("- No staging outputs found")
        lines.append("")

        lines.append("### Why-Zero or Error")
        if record.why_zero:
            lines.append(f"- Why-zero payload: {json.dumps(record.why_zero, indent=2)[:400]}")
        if record.error:
            lines.append(f"- Error exit code: {record.error.get('exit_code')}")
            stderr_tail = record.error.get("stderr_tail") or record.error.get("log_tail")
            if stderr_tail:
                lines.append("- Log tail:")
                snippet = str(stderr_tail)
                for line in snippet.splitlines()[:10]:
                    lines.append(f"  {line}")
        if not record.why_zero and not record.error:
            lines.append("- No why-zero or error diagnostics captured")
        lines.append("")

        dtm_section = render_dtm_deep_dive(record)
        if dtm_section:
            lines.extend(dtm_section)
            lines.append("")

        lines.append("</details>")
        lines.append("")
    return lines


def render_dtm_deep_dive(record: Mapping[str, Any] | ConnectorRecord) -> List[str]:
    extras: Mapping[str, Any] | None
    if isinstance(record, ConnectorRecord):
        extras = record.extras if isinstance(record.extras, Mapping) else None
    elif isinstance(record, Mapping):
        maybe_extras = record.get("extras")
        extras = maybe_extras if isinstance(maybe_extras, Mapping) else None
    else:
        extras = None

    if not extras:
        return []

    dtm_meta = extras.get("dtm")
    if not isinstance(dtm_meta, Mapping):
        return []

    lines: List[str] = ["## DTM Deep Dive", ""]

    config = extras.get("config") if isinstance(extras.get("config"), Mapping) else {}
    window = extras.get("window") if isinstance(extras.get("window"), Mapping) else {}

    lines.append("### Overview")
    lines.append(f"- SDK version: {dtm_meta.get('sdk_version') or 'unknown'}")
    lines.append(f"- Base URL: {dtm_meta.get('base_url') or 'unknown'}")
    lines.append(f"- Python: {dtm_meta.get('python_version') or 'unknown'}")
    admin_levels = config.get("admin_levels") if isinstance(config, Mapping) else None
    if isinstance(admin_levels, Sequence) and not isinstance(admin_levels, (str, bytes)):
        admin_text = ", ".join(str(level) for level in admin_levels) or "(none)"
    else:
        admin_text = "(none)"
    lines.append(f"- Admin levels: {admin_text}")
    countries_mode = config.get("countries_mode") if isinstance(config, Mapping) else None
    countries_count = config.get("countries_count") if isinstance(config, Mapping) else None
    lines.append(
        f"- Countries mode: {countries_mode or 'unknown'} (count={countries_count if countries_count is not None else 'n/a'})"
    )
    lines.append(
        "- Window: {start} → {end}".format(
            start=window.get("start_iso") or window.get("start") or "—",
            end=window.get("end_iso") or window.get("end") or "—",
        )
    )
    lines.append("")

    lines.append("### Discovery")
    discovery = extras.get("discovery") if isinstance(extras.get("discovery"), Mapping) else {}
    stages = discovery.get("stages") if isinstance(discovery, Mapping) else None
    if isinstance(stages, Sequence) and stages:
        for stage in stages:
            if not isinstance(stage, Mapping):
                continue
            lines.append(
                "- {name}: status={status} http={http} attempts={attempts} latency={latency}ms".format(
                    name=stage.get("name", "stage"),
                    status=stage.get("status", "unknown"),
                    http=stage.get("http_code", "n/a"),
                    attempts=stage.get("attempts", "n/a"),
                    latency=stage.get("latency_ms", "n/a"),
                )
            )
    else:
        lines.append("_No discovery stages recorded._")
    reason = discovery.get("reason") if isinstance(discovery, Mapping) else None
    if reason:
        lines.append(f"- Reason: {reason}")
    fail_path = _safe_path(discovery.get("first_fail_path") if isinstance(discovery, Mapping) else None)
    if fail_path:
        fail_payload = safe_load_json(fail_path)
    else:
        fail_payload = None
    errors = []
    if isinstance(fail_payload, Mapping):
        maybe_errors = fail_payload.get("errors")
        if isinstance(maybe_errors, Sequence):
            for item in maybe_errors:
                if isinstance(item, Mapping):
                    message = item.get("message") or item.get("detail")
                    http_code = item.get("http_code") or item.get("code")
                    if message or http_code:
                        errors.append((http_code, message))
    if errors:
        lines.append("- Last failure snapshot:")
        for http_code, message in errors:
            detail = " ".join(str(part) for part in (http_code, message) if part)
            lines.append(f"  - {detail}")
    lines.append("")

    lines.append("### HTTP Roll-up")
    http_stats = extras.get("http") if isinstance(extras.get("http"), Mapping) else {}
    lines.append(
        "- Responses 2xx={two} 4xx={four} 5xx={five} retries={retries} timeouts={timeouts} last={last}".format(
            two=_coerce_int(http_stats.get("count_2xx")),
            four=_coerce_int(http_stats.get("count_4xx")),
            five=_coerce_int(http_stats.get("count_5xx")),
            retries=_coerce_int(http_stats.get("retries")),
            timeouts=_coerce_int(http_stats.get("timeouts")),
            last=http_stats.get("last_status", "n/a"),
        )
    )
    artifacts = extras.get("artifacts") if isinstance(extras.get("artifacts"), Mapping) else {}
    http_trace_summary = _summarize_http_trace(_safe_path(artifacts.get("http_trace")))
    if http_trace_summary:
        lines.append(
            "- Trace latencies: p50={p50}ms p95={p95}ms max={max}ms (n={count})".format(
                p50=int(http_trace_summary.get("latency_p50_ms") or 0),
                p95=int(http_trace_summary.get("latency_p95_ms") or 0),
                max=int(http_trace_summary.get("latency_max_ms") or 0),
                count=http_trace_summary.get("count", 0),
            )
        )
        if http_trace_summary.get("paths"):
            lines.append("- Top endpoints:")
            for endpoint, count in http_trace_summary["paths"]:
                lines.append(f"  - {endpoint}: {count} hits")
    else:
        lines.append("- HTTP trace not available.")
    lines.append("")

    lines.append("### Normalization Snapshot")
    normalize = extras.get("normalize") if isinstance(extras.get("normalize"), Mapping) else {}
    if normalize:
        lines.append(
            "- Rows fetched={fetched} normalized={normalized} written={written}".format(
                fetched=_coerce_int(normalize.get("rows_fetched")),
                normalized=_coerce_int(normalize.get("rows_normalized")),
                written=_coerce_int(normalize.get("rows_written")),
            )
        )
        drop_reasons = normalize.get("drop_reasons") if isinstance(normalize.get("drop_reasons"), Mapping) else {}
        if drop_reasons:
            lines.append("- Drop reasons:")
            for reason_name, count in sorted(drop_reasons.items(), key=lambda item: (-_coerce_int(item[1]), item[0])):
                lines.append(f"  - {reason_name}: {_coerce_int(count)}")
    else:
        lines.append("- No normalization metrics recorded.")
    rescue = extras.get("rescue_probe") if isinstance(extras.get("rescue_probe"), Mapping) else {}
    attempts = rescue.get("tried") if isinstance(rescue, Mapping) else None
    if isinstance(attempts, Sequence) and attempts:
        lines.append("- Rescue probes:")
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                continue
            country = attempt.get("country", "unknown")
            window_text = attempt.get("window", "unknown")
            rows = attempt.get("rows", "n/a")
            error_msg = attempt.get("error")
            detail = f"  - {country} window={window_text} rows={rows}"
            if error_msg:
                detail += f" error={error_msg}"
            lines.append(detail)
    lines.append("")

    lines.append("### Sample rows")
    sample_rows = _safe_read_csv_rows(_safe_path(artifacts.get("samples")))
    if sample_rows:
        headers = list(sample_rows[0].keys()) if sample_rows else []
        table_rows = [[row.get(header, "") for header in headers] for row in sample_rows]
        lines.extend(_format_table(headers, table_rows))
    else:
        lines.append("_No sample rows captured._")
    lines.append("")

    lines.append("### Actionable next steps")
    actions: List[str] = []
    if _coerce_int(http_stats.get("count_4xx")):
        actions.append("- Investigate HTTP 4xx responses (possible auth/config issues).")
    if _coerce_int(normalize.get("rows_written")) == 0:
        actions.append("- Review normalization outputs; zero rows written.")
    if not sample_rows:
        actions.append("- Capture sample rows for manual validation.")
    if not actions:
        actions.append("- No immediate blockers detected; monitor next run.")
    lines.extend(actions)
    lines.append("")

    return lines


def _render_dtm_deep_dive(*args: Any, **kwargs: Any) -> List[str]:
    """Backwards-compatible shim for legacy imports in tests."""

    return render_dtm_deep_dive(*args, **kwargs)


def _render_export_snapshot(staging_snapshot: Mapping[str, Any]) -> List[str]:
    lines = ["## Export & Snapshot", ""]
    total_rows_flow = 0
    total_rows_stock = 0
    if staging_snapshot:
        for connector, files in staging_snapshot.items():
            lines.append(f"- {connector} staging:")
            for file_name, meta in files.items():
                path = meta.get("path")
                rows = meta.get("rows", 0)
                lines.append(f"  - {file_name}: rows={rows} path={path}")
                if "flow" in file_name:
                    total_rows_flow += rows
                if "stock" in file_name:
                    total_rows_stock += rows
    else:
        lines.append("- No staging artifacts found")
    lines.append("")
    lines.append(f"- Rows written (flow): {total_rows_flow}")
    lines.append(f"- Rows written (stock): {total_rows_stock}")
    lines.append("")
    return lines


def _render_anomalies(records: Sequence[ConnectorRecord]) -> List[str]:
    lines = ["## Anomaly & Trend Checks", ""]
    status_counts = Counter(record.status for record in records)
    lines.append("### Failure Budget")
    lines.append(
        f"- ok={status_counts.get('ok', 0)} error={status_counts.get('error', 0)} zero={status_counts.get('zero', 0)} skipped={status_counts.get('skipped', 0)}"
    )
    if status_counts.get("error"):
        lines.append("- Recommendation: investigate errors; soft-fail considered if persistent.")
    else:
        lines.append("- No errors recorded.")
    lines.append("")

    lines.append("### Spike Guard")
    lines.append("- Baseline data unavailable in offline mode; skip z-score analysis.")
    lines.append("")

    lines.append("### Coverage Change")
    lines.append("- Previous summary not found; cannot compare coverage.")
    lines.append("")
    return lines


def _render_next_actions(records: Sequence[ConnectorRecord]) -> List[str]:
    lines = ["## Next Actions", ""]
    actions: List[str] = []
    for record in records:
        if record.error:
            actions.append(f"- Investigate `{record.name}` failure (exit {record.error.get('exit_code')}).")
        elif record.why_zero:
            actions.append(f"- Review `{record.name}` why-zero details; confirm window and token configuration.")
    if not actions:
        actions.append("- No critical follow-up detected; monitor next scheduled run.")
    lines.extend(actions)
    lines.append("")
    return lines


def render_summary_md(
    report_path: Path = DEFAULT_REPORT_PATH,
    diagnostics_dir: Path = DEFAULT_DIAG_DIR,
    staging_dir: Path = DEFAULT_STAGING_DIR,
) -> str:
    entries = load_report(report_path)
    return build_markdown(
        entries,
        diagnostics_root=diagnostics_dir,
        staging_root=staging_dir,
    )


def _write_summary(output_path: Path, content: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--diagnostics", default=str(DEFAULT_DIAG_DIR))
    parser.add_argument("--staging", default=str(DEFAULT_STAGING_DIR))
    parser.add_argument("--output", default=str(SUMMARY_PATH))
    parser.add_argument("--out", dest="output")
    parser.add_argument("--github-step-summary", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv if argv is not None else [])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = Path(args.report)
    diagnostics_dir = Path(args.diagnostics)
    staging_dir = Path(args.staging)
    output_path = Path(args.output)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    entries = load_report(report_path)

    content = build_markdown(
        entries,
        diagnostics_root=diagnostics_dir,
        staging_root=staging_dir,
    )
    _write_summary(output_path, content)
    if getattr(args, "github_step_summary", False):
        summary_path = os.getenv("GITHUB_STEP_SUMMARY")
        if summary_path:
            Path(summary_path).write_text(content, encoding="utf-8")
    print(f"Summary written to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main(sys.argv[1:]))
