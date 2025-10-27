"""Render ingestion connector diagnostics into Markdown summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import certifi

REASON_HISTOGRAM_LIMIT = 5

DEFAULT_HTTP_KEYS = ("2xx", "4xx", "5xx", "retries", "rate_limit_remaining", "last_status")
SUMMARY_TITLE = "# Connector Diagnostics"
MISSING_REPORT_SUMMARY = (
    "# Ingestion Diagnostics\n\n"
    "**No connectors report was produced.**  \n"
    "This usually means the ingestion step failed early (e.g., setup or backfill window).  \n"
    "Check earlier steps in the job log.\n"
)

STAGING_EXTENSIONS = {".csv", ".tsv", ".parquet", ".json", ".jsonl"}


def _ensure_dict(data: Any) -> Dict[str, Any]:
    return dict(data) if isinstance(data, Mapping) else {}


def _safe_load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        data = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, Mapping) else None


def _coerce_int(value: Any) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _fmt_count(value: Any) -> str:
    try:
        if value in (None, ""):
            return "—"
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    return "—" if number == 0 else str(number)


def _format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} B"


def _collect_staging_inventory(path: Path) -> Dict[str, Any]:
    files: List[Tuple[str, int]] = []
    total_size = 0
    count = 0
    try:
        if path.exists():
            for entry in path.rglob("*"):
                if not entry.is_file():
                    continue
                if entry.suffix.lower() not in STAGING_EXTENSIONS:
                    continue
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                files.append((entry.relative_to(path).as_posix(), size))
                total_size += size
                count += 1
    except OSError:
        pass
    files.sort(key=lambda item: (-item[1], item[0]))
    return {
        "exists": path.exists(),
        "count": count,
        "total_size": total_size,
        "files": files[:5],
    }


def _format_meta_cell(
    status_raw: Any,
    extras: Mapping[str, Any] | None,
    meta_json: Mapping[str, Any] | None,
) -> str:
    """Render the Meta cell value applying ok-empty and missing-count rules."""
    try:
        extras_map: Mapping[str, Any] = extras or {}
        raw_status = status_raw or extras_map.get("status_raw")
        if isinstance(raw_status, str) and raw_status.strip().lower() == "ok-empty":
            return "—"
        rows_written = extras_map.get("rows_written")
        if rows_written is not None and _coerce_int(rows_written) == 0:
            return "—"
        row_count = (meta_json or {}).get("row_count")
        return _fmt_count(row_count)
    except Exception:
        return "—"


def _normalise_samples(values: Any) -> List[Tuple[str, int]]:
    results: List[Tuple[str, int]] = []
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        return results
    for item in values:
        if isinstance(item, (list, tuple)) and item:
            label = str(item[0]) if len(item) >= 1 else ""
            count = _coerce_int(item[1]) if len(item) >= 2 else 0
            if label:
                results.append((label, count))
        elif isinstance(item, Mapping):
            label = str(item.get("key") or item.get("label") or "")
            count = _coerce_int(item.get("value"))
            if label:
                results.append((label, count))
    return results


def _clean_reason(reason: Any) -> str | None:
    if reason is None:
        return None
    text = str(reason).strip()
    if not text or text == "-":
        return None
    return text


def _normalise_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
    http_raw = _ensure_dict(entry.get("http"))
    http: Dict[str, Any] = {key: http_raw.get(key) for key in DEFAULT_HTTP_KEYS}
    counts_raw = _ensure_dict(entry.get("counts"))
    counts = {
        "fetched": _coerce_int(counts_raw.get("fetched")),
        "normalized": _coerce_int(counts_raw.get("normalized")),
        "written": _coerce_int(counts_raw.get("written")),
    }
    coverage_raw = _ensure_dict(entry.get("coverage"))
    coverage = {
        "ym_min": coverage_raw.get("ym_min") or None,
        "ym_max": coverage_raw.get("ym_max") or None,
        "as_of_min": coverage_raw.get("as_of_min") or None,
        "as_of_max": coverage_raw.get("as_of_max") or None,
    }
    samples_raw = _ensure_dict(entry.get("samples"))
    samples = {
        "top_iso3": _normalise_samples(samples_raw.get("top_iso3")),
        "top_hazard": _normalise_samples(samples_raw.get("top_hazard")),
    }
    extras = _ensure_dict(entry.get("extras"))
    status_raw = str(
        entry.get("status_raw")
        or extras.get("status_raw")
        or entry.get("status")
        or "skipped"
    )
    exit_code = _coerce_int(entry.get("exit_code") if entry.get("exit_code") is not None else extras.get("exit_code"))
    status = str(entry.get("status") or status_raw or "skipped")
    if status_raw.lower() == "error" or (exit_code not in (None, 0)):
        status = "error"

    return {
        "connector_id": str(entry.get("connector_id") or entry.get("name") or "unknown"),
        "mode": str(entry.get("mode") or "real"),
        "status": status,
        "status_raw": status_raw,
        "exit_code": exit_code,
        "reason": _clean_reason(entry.get("reason")),
        "started_at_utc": str(entry.get("started_at_utc") or ""),
        "duration_ms": _coerce_int(entry.get("duration_ms")),
        "http": http,
        "counts": counts,
        "coverage": coverage,
        "samples": samples,
        "extras": extras,
    }


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(data) if isinstance(data, Mapping) else {}


def load_report(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise RuntimeError(f"Unable to parse JSON on line {line_number}: {exc}") from exc
            if isinstance(payload, Mapping):
                entries.append(_normalise_entry(payload))
    return entries


def deduplicate_entries(
    entries: Sequence[Mapping[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    grouped: Dict[Tuple[str, str], List[Tuple[int, Dict[str, Any]]]] = {}
    for index, entry in enumerate(entries):
        connector_id = str(entry.get("connector_id") or "unknown")
        mode = str(entry.get("mode") or "real")
        key = (connector_id, mode)
        grouped.setdefault(key, []).append((index, dict(entry)))

    deduped: List[Tuple[int, Dict[str, Any]]] = []
    duplicates: Dict[str, int] = {}
    for key, records in grouped.items():
        if len(records) == 1:
            deduped.append(records[0])
            continue
        records.sort(key=lambda item: ((item[1].get("started_at_utc") or ""), item[0]))
        chosen = records[-1]
        deduped.append(chosen)
        connector_id, _mode = key
        duplicates[connector_id] = duplicates.get(connector_id, 0) + len(records) - 1

    deduped.sort(key=lambda item: item[0])
    return [entry for _, entry in deduped], duplicates


def _format_status_counts(entries: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(str(entry.get("status") or "") for entry in entries)
    if not counts:
        return "none"
    parts = [f"{status}={count}" for status, count in sorted(counts.items()) if status]
    return ", ".join(parts) if parts else "none"


def _format_reason_histogram(entries: Sequence[Mapping[str, Any]]) -> str:
    counter: Counter[str] = Counter()
    for entry in entries:
        reason = entry.get("reason")
        if not reason:
            continue
        cleaned = str(reason).strip()
        if cleaned:
            counter[cleaned] += 1
    if not counter:
        return "—"
    limited = list(counter.items())
    limited.sort(key=lambda item: item[0])
    parts = [f"{reason}={count}" for reason, count in limited[:REASON_HISTOGRAM_LIMIT]]
    return ", ".join(parts)


def _truncate_text(value: Any, *, limit: int = 2048) -> str:
    if not isinstance(value, str):
        return str(value)
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _format_duration(duration_ms: int) -> str:
    if duration_ms <= 0:
        return "—"
    if duration_ms < 1000:
        return f"{duration_ms} ms"
    total_seconds = duration_ms / 1000
    if total_seconds < 60:
        return f"{total_seconds:.1f}s"
    minutes, seconds = divmod(int(total_seconds), 60)
    if minutes < 60:
        return f"{minutes}m{seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{seconds:02d}s"


def _format_rows(counts: Mapping[str, int]) -> str:
    fetched = counts.get("fetched") or 0
    normalized = counts.get("normalized") or 0
    written = counts.get("written") or 0
    return f"{fetched}/{normalized}/{written}"


def _format_http(http: Mapping[str, Any]) -> str:
    success = http.get("2xx") or 0
    client = http.get("4xx") or 0
    server = http.get("5xx") or 0
    retries = http.get("retries") or 0
    return f"{success}/{client}/{server} ({retries})"


def _format_coverage(min_value: Any, max_value: Any) -> str:
    start = str(min_value).strip() if min_value else ""
    end = str(max_value).strip() if max_value else ""
    if start and end:
        if start == end:
            return start
        return f"{start} → {end}"
    if start:
        return start
    if end:
        return end
    return "—"


def _format_reason(reason: str | None) -> str:
    return reason if reason else "—"


def _format_sample_list(label: str, samples: Sequence[Tuple[str, int]]) -> str | None:
    if not samples:
        return None
    items = [f"`{name}` ({count})" for name, count in samples if name]
    if not items:
        return None
    return f"- **{label}:** {', '.join(items)}"


def _format_optional_int(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return str(int(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"


def _format_extras(extras: Mapping[str, Any]) -> str | None:
    if not extras:
        return None
    items: List[str] = []
    for key in sorted(extras.keys()):
        value = extras.get(key)
        if value in (None, ""):
            continue
        items.append(f"`{key}={value}`")
    if not items:
        return None
    return f"- **Extras:** {', '.join(items)}"


def _render_details(entry: Mapping[str, Any]) -> str:
    connector = entry.get("connector_id", "unknown")
    lines = ["<details>", f"<summary>{connector} diagnostics</summary>", ""]
    bullets: List[str] = []

    # Show API key configuration status for connectors that need it
    extras = _ensure_dict(entry.get("extras", {}))
    if "api_key_configured" in extras:
        api_configured = extras["api_key_configured"]
        if api_configured:
            bullets.append("- **API Key:** ✓ Configured")
        else:
            bullets.append("- **API Key:** ✗ Not configured")
            bullets.append("- **Action Required:** Set `DTM_API_KEY` in GitHub secrets to fetch live data")

    # Show mode (api, file, header-only) if available
    mode_info = extras.get("mode")
    if mode_info:
        bullets.append(f"- **Mode:** {mode_info}")

    top_iso3 = _format_sample_list("Top ISO3", entry.get("samples", {}).get("top_iso3", []))
    if top_iso3:
        bullets.append(top_iso3)
    top_hazard = _format_sample_list("Top hazard", entry.get("samples", {}).get("top_hazard", []))
    if top_hazard:
        bullets.append(top_hazard)
    http = entry.get("http", {})
    rate_limit = http.get("rate_limit_remaining")
    if rate_limit not in (None, ""):
        bullets.append(f"- **Rate limit remaining:** {rate_limit}")
    last_status = http.get("last_status")
    if last_status not in (None, ""):
        bullets.append(f"- **Last status:** {last_status}")

    # Show additional extras, but filter out the ones we already displayed
    filtered_extras = {
        k: v for k, v in extras.items()
        if k not in ("api_key_configured", "mode")
    }
    extras_line = _format_extras(filtered_extras)
    if extras_line:
        bullets.append(extras_line)

    if not bullets:
        bullets.append("- _No additional diagnostics recorded._")
    lines.extend(bullets)
    lines.append("")
    lines.append("</details>")
    return "\n".join(lines)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[int(rank)])
    frac = rank - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * frac)


def _load_ndjson(path: Path) -> List[Mapping[str, Any]]:
    entries: List[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, Mapping):
                    entries.append(payload)
    except OSError:
        return []
    return entries


def _aggregate_http_endpoints(trace_path: Path) -> List[Dict[str, Any]]:
    entries = _load_ndjson(trace_path)
    buckets: Dict[str, List[float]] = {}
    for entry in entries:
        path_value = str(entry.get("path") or entry.get("endpoint") or "").strip()
        if not path_value:
            continue
        latency = entry.get("elapsed_ms") or entry.get("latency_ms")
        parsed = _coerce_float(latency)
        if parsed is None:
            continue
        buckets.setdefault(path_value, []).append(parsed)
    results: List[Dict[str, Any]] = []
    for endpoint, latencies in buckets.items():
        if not latencies:
            continue
        ordered = sorted(latencies)
        results.append(
            {
                "path": endpoint,
                "count": len(ordered),
                "p50_ms": int(round(_percentile(ordered, 50))),
                "p95_ms": int(round(_percentile(ordered, 95))),
                "max_ms": int(round(max(ordered))),
            }
        )
    results.sort(key=lambda item: (-item["count"], item["path"]))
    return results[:5]


def _read_sample_rows(path: Path, limit: int = 10) -> Tuple[List[str], List[List[str]]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            headers = next(reader, [])
            rows = []
            for idx, row in enumerate(reader):
                if idx >= limit:
                    break
                rows.append([str(cell) for cell in row])
    except OSError:
        return [], []
    return [str(header) for header in headers], rows


def _format_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    if not headers or not rows:
        return []
    normalized_rows = [["" if cell is None else str(cell) for cell in row] for row in rows]
    lines = ["| " + " | ".join(str(header) for header in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in normalized_rows:
        lines.append("| " + " | ".join(row[: len(headers)]) + " |")
    return lines


def _load_discovery_errors(path: Path, limit: int = 2) -> List[str]:
    payload = _safe_load_json(path)
    if not payload:
        return []
    errors = payload.get("errors")
    if not isinstance(errors, list):
        return []
    snippets: List[str] = []
    for entry in errors[:limit]:
        if isinstance(entry, Mapping):
            try:
                snippet = json.dumps(entry, ensure_ascii=False)
            except (TypeError, ValueError):
                snippet = str(entry)
        else:
            snippet = str(entry)
        snippets.append(snippet)
    return snippets


def _render_dtm_deep_dive(entry: Mapping[str, Any]) -> List[str]:
    extras = _ensure_dict(entry.get("extras"))
    dtm = _ensure_dict(extras.get("dtm"))
    config = _ensure_dict(extras.get("config"))
    window = _ensure_dict(extras.get("window"))
    discovery = _ensure_dict(extras.get("discovery"))
    http_summary = _ensure_dict(extras.get("http"))
    fetch = _ensure_dict(extras.get("fetch"))
    normalize = _ensure_dict(extras.get("normalize"))
    rescue = extras.get("rescue_probe")
    rescue_entries = rescue.get("tried", []) if isinstance(rescue, Mapping) else []
    artifacts = _ensure_dict(extras.get("artifacts"))

    lines: List[str] = ["## DTM Deep Dive", ""]

    lines.append("### Auth & SDK")
    sdk_line = f"- **SDK Version:** `{dtm.get('sdk_version', 'unknown')}`"
    base_line = f"- **Base URL:** `{dtm.get('base_url', 'unknown')}`"
    python_line = f"- **Python:** `{dtm.get('python_version', 'unknown')}`"
    lines.extend([sdk_line, base_line, python_line, ""])

    lines.append("### Discovery")
    stage_rows: List[List[str]] = []
    for stage in discovery.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        stage_rows.append(
            [
                str(stage.get("name") or stage.get("stage") or "—"),
                str(stage.get("status") or "—"),
                str(stage.get("http_code") or "—"),
                str(stage.get("attempts") or "—"),
                str(stage.get("latency_ms") or "—"),
            ]
        )
    if stage_rows:
        lines.extend(_format_markdown_table(["Stage", "Status", "HTTP", "Attempts", "Latency (ms)"], stage_rows))
    else:
        lines.append("_No discovery stages recorded._")
    used_stage = discovery.get("used_stage") or "—"
    reason = discovery.get("reason")
    reason_text = f" (reason: {reason})" if reason else ""
    lines.append(f"- **Used stage:** `{used_stage}`{reason_text}")
    error_path = discovery.get("first_fail_path")
    if isinstance(error_path, str) and error_path:
        snippets = _load_discovery_errors(Path(error_path))
        if snippets:
            lines.append("- **Discovery errors:**")
            for snippet in snippets:
                lines.append(f"  - `{snippet}`")
    lines.append("")

    lines.append("### Effective Config")
    countries_mode = config.get("countries_mode", "discovered")
    lines.extend(
        [
            f"- **Config path:** `{config.get('config_path_used', 'unknown')}`",
            f"- **Config exists:** {'yes' if config.get('config_exists') else 'no'}",
            f"- **Config sha:** `{config.get('config_sha256', 'n/a')}`",
            f"- **Admin levels:** {', '.join(config.get('admin_levels', []) or ['—'])}",
            f"- **Countries mode:** `{countries_mode}` ({config.get('countries_count', 0)} selectors)",
            f"- **Countries preview:** {', '.join(str(item) for item in config.get('countries_preview', []) or ['—'])}",
            f"- **No date filter:** {config.get('no_date_filter', 0)}",
            f"- **Window:** {window.get('start_iso') or '—'} → {window.get('end_iso') or '—'}",
        ]
    )
    lines.append("")

    lines.append("### HTTP Roll-up")
    lines.extend(
        [
            f"- **2xx/4xx/5xx:** {http_summary.get('count_2xx', 0)}/"
            f"{http_summary.get('count_4xx', 0)}/{http_summary.get('count_5xx', 0)}",
            f"- **Retries:** {http_summary.get('retries', 0)}",
            f"- **Timeouts:** {http_summary.get('timeouts', 0)}",
            f"- **Last status:** {http_summary.get('last_status', '—')}",
        ]
    )
    http_trace_path = artifacts.get("http_trace")
    if isinstance(http_trace_path, str) and http_trace_path:
        top_endpoints = _aggregate_http_endpoints(Path(http_trace_path))
    else:
        top_endpoints = []
    if not top_endpoints:
        top_endpoints = [item for item in http_summary.get("endpoints_top", []) if isinstance(item, Mapping)]
    if top_endpoints:
        endpoint_rows = [
            [
                str(item.get("path", "—")),
                str(item.get("count", 0)),
                str(item.get("p50_ms", "—")),
                str(item.get("p95_ms", "—")),
                str(item.get("max_ms", "—")),
            ]
            for item in top_endpoints
        ]
        lines.extend(_format_markdown_table(["Path", "Count", "p50 (ms)", "p95 (ms)", "Max (ms)"], endpoint_rows))
    lines.append("")

    lines.append("### Fetch Metrics")
    lines.extend(
        [
            f"- **Pages:** {fetch.get('pages', 0)}",
            f"- **Max page size:** {fetch.get('max_page_size') or '—'}",
            f"- **Total rows received:** {fetch.get('total_received', 0)}",
        ]
    )
    lines.append("")

    lines.append("### Normalization")
    lines.extend(
        [
            f"- **Rows fetched/normalized/written:** {normalize.get('rows_fetched', 0)}/"
            f"{normalize.get('rows_normalized', 0)}/{normalize.get('rows_written', 0)}",
        ]
    )
    drop_reasons = normalize.get("drop_reasons")
    if isinstance(drop_reasons, Mapping) and drop_reasons:
        drop_rows = [[str(reason), str(drop_reasons.get(reason, 0))] for reason in sorted(drop_reasons.keys())]
        lines.extend(_format_markdown_table(["Reason", "Count"], drop_rows))
    chosen_columns = normalize.get("chosen_value_columns")
    if isinstance(chosen_columns, list) and chosen_columns:
        chosen_rows = [
            [str(item.get("column", "—")), str(item.get("count", 0))]
            for item in chosen_columns
            if isinstance(item, Mapping)
        ]
        if chosen_rows:
            lines.extend(_format_markdown_table(["Value column", "Rows"], chosen_rows))
    lines.append("")

    samples_path = artifacts.get("samples")
    if isinstance(samples_path, str) and samples_path:
        headers, sample_rows = _read_sample_rows(Path(samples_path))
        if sample_rows:
            lines.append("### Sample rows")
            lines.extend(_format_markdown_table(headers, sample_rows))
            lines.append("")

    if rescue_entries:
        lines.append("### Zero-rows rescue")
        rescue_rows = [
            [str(item.get("country", "—")), str(item.get("window", "—")), str(item.get("rows", 0)), str(item.get("error", ""))]
            for item in rescue_entries
            if isinstance(item, Mapping)
        ]
        if rescue_rows:
            lines.extend(_format_markdown_table(["Country", "Window", "Rows", "Error"], rescue_rows))
        lines.append("")

    lines.append("### Actionable next steps")
    actions: List[str] = []
    if any(str(stage.get("http_code")) in {"401", "403"} for stage in discovery.get("stages", []) if isinstance(stage, Mapping)):
        actions.append("- Verify DTM API key permissions or request access for discovery endpoints (HTTP 401/403).")
    drop_counts = normalize.get("drop_reasons") if isinstance(normalize.get("drop_reasons"), Mapping) else {}
    if drop_counts and drop_counts.get("no_country_match"):
        actions.append("- Review country aliases or explicit selectors; discovery skipped some countries.")
    if drop_counts and drop_counts.get("no_iso3"):
        actions.append("- Update ISO3 mapping/aliases; many rows lacked a resolvable ISO3 code.")
    reason_text_lower = str(discovery.get("reason") or "").lower()
    used_stage_text = str(discovery.get("used_stage") or "").lower()
    if "static_iso3_minimal" in used_stage_text or "minimal" in reason_text_lower:
        actions.append("- **Discovery fallback active:** Provide explicit `api.countries` or request discovery access to avoid the minimal static allowlist.")
    if normalize.get("rows_written", 0) == 0:
        actions.append("- Connector wrote zero rows; inspect window and rescue probe results for active countries.")
    if not actions:
        actions.append("- _No immediate blockers detected._")
    lines.extend(actions)
    lines.append("")
    return lines


def _format_yes_no(value: Any) -> str:
    truthy = {"1", "true", "y", "yes", "on"}
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return "yes" if value else "no"
    if isinstance(value, str):
        return "yes" if value.strip().lower() in truthy else "no"
    return "yes" if value else "no"


def _render_config_section(entry: Mapping[str, Any]) -> List[str]:
    extras = _ensure_dict(entry.get("extras"))
    config = _ensure_dict(extras.get("config"))
    if not config:
        return []
    admin_levels = config.get("admin_levels")
    if isinstance(admin_levels, Iterable) and not isinstance(admin_levels, (str, bytes)):
        admin_text = ", ".join(str(item) for item in admin_levels if str(item)) or "—"
    else:
        admin_text = "—"
    preview_values = config.get("countries_preview")
    if isinstance(preview_values, Iterable) and not isinstance(preview_values, (str, bytes)):
        preview_text = ", ".join(str(item) for item in list(preview_values)[:5] if str(item)) or "—"
    else:
        preview_text = "—"
    lines = [
        "## Config used",
        "",
        f"- **Path:** `{config.get('config_path_used', 'unknown')}`",
        f"- **Exists:** {_format_yes_no(config.get('config_exists'))}",
        f"- **SHA256:** `{config.get('config_sha256', 'n/a')}`",
        f"- **Countries mode:** `{config.get('countries_mode', 'discovered')}`",
        f"- **Countries count:** {config.get('countries_count', 0)}",
        f"- **Countries preview:** {preview_text}",
        f"- **Admin levels:** {admin_text}",
        f"- **No date filter:** {_format_yes_no(config.get('no_date_filter'))}",
    ]
    lines.append("")
    return lines


def _format_staging_line(label: str, stats: Mapping[str, Any]) -> str:
    if not stats.get("exists"):
        return f"- **{label}:** not present"
    count = _coerce_int(stats.get("count"))
    size_text = _format_bytes(int(stats.get("total_size", 0))) if count else "0 B"
    if count == 0:
        return f"- **{label}:** empty (0 files)"
    files = stats.get("files") if isinstance(stats.get("files"), list) else []
    if files:
        preview = ", ".join(
            f"{name} ({_format_bytes(size)})" for name, size in files if isinstance(name, str)
        )
        if preview:
            return f"- **{label}:** {count} files ({size_text}) — top: {preview}"
    return f"- **{label}:** {count} files ({size_text})"


def _render_staging_readiness(entry: Mapping[str, Any]) -> List[str]:
    resolver_path = Path("resolver/staging")
    legacy_path = Path("data/staging")
    resolver_stats = _collect_staging_inventory(resolver_path)
    legacy_stats = _collect_staging_inventory(legacy_path)

    lines = ["## Staging readiness", ""]
    lines.append(_format_staging_line("resolver/staging", resolver_stats))
    lines.append(_format_staging_line("data/staging", legacy_stats))

    resolver_count = _coerce_int(resolver_stats.get("count"))
    legacy_count = _coerce_int(legacy_stats.get("count"))
    if resolver_count == 0 and legacy_count > 0:
        lines.append("")
        lines.append(
            "**Export reads `resolver/staging` but ingest wrote to `data/staging`. "
            "Fix: set `RESOLVER_OUTPUT_DIR=resolver/staging` in the ingest job.**"
        )
    elif resolver_count == 0 and legacy_count == 0:
        lines.append("")
        lines.append("**No staging inputs found; derive/export will have nothing to process.**")
    lines.append("")
    return lines


def _render_zero_row_root_cause(entry: Mapping[str, Any]) -> List[str]:
    extras = _ensure_dict(entry.get("extras"))
    counts = _ensure_dict(entry.get("counts"))
    rows_written = _coerce_int(counts.get("written")) or _coerce_int(extras.get("rows_written"))
    if rows_written:
        return []

    reason = extras.get("zero_rows_reason") or extras.get("status_raw") or entry.get("status_raw")
    normalize = _ensure_dict(extras.get("normalize"))
    drop_reasons = _ensure_dict(normalize.get("drop_reasons"))
    sorted_reasons = sorted(
        ((str(key), _coerce_int(value)) for key, value in drop_reasons.items()),
        key=lambda item: (-item[1], item[0]),
    )
    top_reasons = [f"{label} ({count})" for label, count in sorted_reasons if count][:3]

    per_country = extras.get("per_country_counts")
    total_selectors = 0
    selectors_with_rows = 0
    if isinstance(per_country, list):
        for item in per_country:
            if not isinstance(item, Mapping):
                continue
            total_selectors += 1
            if _coerce_int(item.get("rows")) > 0:
                selectors_with_rows += 1

    discovery = _ensure_dict(extras.get("discovery"))
    fetch = _ensure_dict(extras.get("fetch"))

    bullets: List[str] = []
    if reason:
        bullets.append(f"- **Primary reason:** {reason}")
    if top_reasons:
        bullets.append(f"- **Top drop reasons:** {', '.join(top_reasons)}")
    if total_selectors:
        bullets.append(
            f"- **Selectors with rows:** {selectors_with_rows}/{total_selectors}"
        )
    discovery_stage = discovery.get("used_stage") if discovery else None
    discovery_reason = discovery.get("reason") if discovery else None
    report = discovery.get("report") if isinstance(discovery, Mapping) else None
    if isinstance(report, Mapping):
        discovery_stage = discovery_stage or report.get("used_stage")
        discovery_reason = discovery_reason or report.get("reason")
    if discovery_stage or discovery_reason:
        stage_text = discovery_stage or "—"
        if discovery_reason:
            bullets.append(f"- **Discovery:** stage={stage_text} reason={discovery_reason}")
        else:
            bullets.append(f"- **Discovery stage:** {stage_text}")
    total_received = _coerce_int(fetch.get("total_received")) if fetch else 0
    pages = _coerce_int(fetch.get("pages")) if fetch else 0
    if total_received or pages:
        bullets.append(
            f"- **Fetch totals:** rows_received={total_received} pages={pages}"
        )

    if not bullets:
        return []

    lines = ["## Zero-row root cause", ""]
    lines.extend(bullets)
    lines.append("")
    return lines


def _render_selector_effectiveness(entry: Mapping[str, Any]) -> List[str]:
    extras = _ensure_dict(entry.get("extras"))
    counts_raw = extras.get("per_country_counts")
    if not isinstance(counts_raw, list) or not counts_raw:
        return []
    normalized: List[Dict[str, Any]] = []
    for item in counts_raw:
        if not isinstance(item, Mapping):
            continue
        normalized.append(
            {
                "country": str(item.get("country") or item.get("selector") or item.get("iso3") or "unknown"),
                "rows": _coerce_int(item.get("rows")),
                "level": str(item.get("level") or "—"),
                "operation": str(item.get("operation") or "—"),
            }
        )
    if not normalized:
        return []
    total = len(normalized)
    with_rows = sum(1 for entry in normalized if entry["rows"] > 0)
    hit_rate = f"{with_rows}/{total}"
    top = sorted(normalized, key=lambda item: (-item["rows"], item["country"]))[:5]
    lines = ["## Selector effectiveness", ""]
    lines.append(f"- **Selectors attempted:** {total}")
    lines.append(f"- **Selectors with rows:** {with_rows}")
    lines.append(f"- **Hit rate:** {hit_rate}")
    if top:
        lines.append("- **Top selectors by rows:**")
        for entry in top:
            operation = entry["operation"] if entry["operation"] not in {"", "—"} else "all"
            lines.append(
                f"  - `{entry['country']}` level={entry['level']} rows={entry['rows']} (operation={operation})"
            )
    lines.append("")
    return lines


def _extract_common_name(subject: Any) -> str:
    if isinstance(subject, (list, tuple)):
        for group in subject:
            if isinstance(group, (list, tuple)):
                for key, value in group:
                    if str(key).lower() in {"commonname", "cn"}:
                        return str(value)
    return ""


def _render_reachability_section(reachability: Mapping[str, Any]) -> List[str]:
    lines = ["## DTM Reachability", ""]
    if not reachability:
        lines.append("- **diagnostics/ingestion/dtm/reachability.json:** not present")
        lines.append("")
        return lines
    target = reachability.get("target_host", "dtmapi.iom.int")
    port = reachability.get("target_port", 443)
    lines.append(f"- **Target:** `{target}:{port}`")
    captured_start = reachability.get("generated_at")
    captured_end = reachability.get("completed_at")
    if captured_start or captured_end:
        lines.append(f"- **Captured:** {captured_start or '—'} → {captured_end or '—'}")

    dns = _ensure_dict(reachability.get("dns"))
    dns_records = []
    for entry in dns.get("records", []):
        if isinstance(entry, Mapping):
            address = entry.get("address")
            family = entry.get("family")
            if address:
                dns_records.append(f"{address}{f' ({family})' if family else ''}")
    if dns.get("error"):
        dns_line = f"- **DNS:** error={dns.get('error')}"
    else:
        dns_line = f"- **DNS:** {', '.join(dns_records) if dns_records else '—'}"
    if dns.get("elapsed_ms") is not None:
        dns_line += f" ({dns.get('elapsed_ms')}ms)"
    lines.append(dns_line)

    tcp = _ensure_dict(reachability.get("tcp"))
    if tcp.get("ok"):
        peer = tcp.get("peer")
        if isinstance(peer, (list, tuple)):
            peer_text = ":".join(str(part) for part in peer)
        else:
            peer_text = str(peer or "")
        tcp_line = "- **TCP:** ok"
        if tcp.get("elapsed_ms") is not None:
            tcp_line += f" in {tcp.get('elapsed_ms')}ms"
        if peer_text:
            tcp_line += f" (peer={peer_text})"
    else:
        tcp_line = "- **TCP:** failed"
        if tcp.get("error"):
            tcp_line += f" ({tcp.get('error')})"
        if tcp.get("elapsed_ms") is not None:
            tcp_line += f" after {tcp.get('elapsed_ms')}ms"
    lines.append(tcp_line)

    tls = _ensure_dict(reachability.get("tls"))
    if tls.get("ok"):
        tls_line = "- **TLS:** ok"
        if tls.get("elapsed_ms") is not None:
            tls_line += f" in {tls.get('elapsed_ms')}ms"
        subject_cn = _extract_common_name(tls.get("subject"))
        issuer_cn = _extract_common_name(tls.get("issuer"))
        details: List[str] = []
        if subject_cn:
            details.append(f"CN={subject_cn}")
        if issuer_cn:
            details.append(f"issuer={issuer_cn}")
        if tls.get("not_after"):
            details.append(f"expires={tls.get('not_after')}")
        if details:
            tls_line += f" ({', '.join(details)})"
    else:
        tls_line = "- **TLS:** failed"
        if tls.get("error"):
            tls_line += f" ({tls.get('error')})"
    lines.append(tls_line)

    curl = _ensure_dict(reachability.get("curl_head"))
    if curl.get("error"):
        curl_line = f"- **HTTP HEAD (curl):** error={curl.get('error')}"
    else:
        status_line = curl.get("status_line") or "n/a"
        curl_line = f"- **HTTP HEAD (curl):** {status_line}"
    if curl.get("exit_code") is not None:
        curl_line += f" [exit={curl.get('exit_code')}]"
    if curl.get("stderr"):
        curl_line += f" (stderr={_truncate_text(curl.get('stderr'))})"
    lines.append(curl_line)

    egress = _ensure_dict(reachability.get("egress"))
    egress_parts: List[str] = []
    for key, value in egress.items():
        info = _ensure_dict(value)
        if info.get("error"):
            display = f"error={info.get('error')}"
        else:
            text = info.get("text") or info.get("status_code") or "n/a"
            display = str(text)
        egress_parts.append(f"{key}={display}")
    if egress_parts:
        lines.append(f"- **Egress IPs:** {', '.join(egress_parts)}")

    lines.append(f"- **CA bundle:** {reachability.get('ca_bundle', certifi.where())}")
    lines.append(f"- **Python/requests:** {reachability.get('python_version', 'unknown')} / {reachability.get('requests_version', 'unknown')}")
    lines.append("")
    return lines


def _build_table(entries: Sequence[Mapping[str, Any]]) -> List[str]:
    headers = [
        "Connector",
        "Mode",
        "Status",
        "Reason",
        "Duration",
        "2xx/4xx/5xx (retries)",
        "Rows (f/n/w)",
        "Kept",
        "Dropped",
        "ParseErrs",
        "Coverage (ym)",
        "Coverage (as_of)",
        "Logs",
        "Meta rows",
        "Meta",
    ]
    logs_dir = Path("diagnostics/ingestion/logs")
    rows: List[List[str]] = []
    dtm_run_path: Path | None = None
    dtm_run_data: Dict[str, Any] | None = None
    for entry in entries:
        coverage = entry.get("coverage", {})
        connector_id = str(entry.get("connector_id"))
        log_path = logs_dir / f"{connector_id}.log"
        extras = _ensure_dict(entry.get("extras"))
        counts_map = _ensure_dict(entry.get("counts"))
        rows_written_extra = extras.get("rows_written")
        if rows_written_extra is not None:
            rows_written_value = _coerce_int(rows_written_extra)
        else:
            rows_written_value = _coerce_int(counts_map.get("written"))
        config_issues_path = extras.get("config_issues_path")
        log_cell = str(log_path) if log_path.exists() else "—"
        if connector_id == "dtm_client" and config_issues_path:
            config_path_text = str(config_issues_path)
            log_cell = (
                f"{log_cell} / {config_path_text}"
                if log_cell != "—"
                else config_path_text
            )
        meta_path_raw = extras.get("meta_path")
        meta_cell = "—"
        meta_payload: Mapping[str, Any] | None = None
        if meta_path_raw:
            meta_path = Path(str(meta_path_raw))
            if meta_path.exists():
                meta_cell = str(meta_path)
                loaded = _safe_load_json(meta_path)
                if isinstance(loaded, Mapping):
                    meta_payload = loaded
        reason_text = entry.get("reason")
        status_text = str(entry.get("status"))
        kept_cell = "—"
        dropped_cell = "—"
        parse_cell = "—"
        if connector_id == "dtm_client":
            status_raw = str(extras.get("status_raw") or status_text)
            if status_raw == "ok-empty" or rows_written_value == 0:
                status_text = "ok-empty"
                if not reason_text:
                    reason_text = "header-only (0 rows)"
                kept_cell = "—"
                dropped_cell = "—"
                parse_cell = "—"
            elif status_raw:
                status_text = status_raw
            run_details_raw = extras.get("run_details_path")
            candidate_path = (
                Path(str(run_details_raw))
                if run_details_raw
                else Path("diagnostics/ingestion/dtm_run.json")
            )
            if dtm_run_data is None or candidate_path != dtm_run_path:
                dtm_run_path = candidate_path
                dtm_run_data = _load_json(candidate_path)
            totals = _ensure_dict(dtm_run_data.get("totals")) if dtm_run_data else {}
            kept_value = totals.get("kept")
            dropped_value = totals.get("dropped")
            parse_value = totals.get("parse_errors")
            if status_text != "ok-empty":
                if kept_value is not None:
                    kept_cell = _format_optional_int(kept_value)
                elif totals.get("rows_after") is not None:
                    kept_cell = _format_optional_int(totals.get("rows_after"))
                elif totals.get("rows_written") is not None:
                    kept_cell = _format_optional_int(totals.get("rows_written"))
                if dropped_value is not None:
                    dropped_cell = _format_optional_int(dropped_value)
                if parse_value is not None:
                    parse_cell = _format_optional_int(parse_value)
        if (
            connector_id == "dtm_client"
            and isinstance(reason_text, str)
            and "missing id_or_path" in reason_text
        ):
            invalid_count = extras.get("invalid_sources")
            status = str(entry.get("status") or "").strip() or "skipped"
            if invalid_count:
                reason_text = f"{status}: missing id_or_path ({invalid_count})"
            else:
                reason_text = f"{status}: missing id_or_path"
        reason_cell = _format_reason(reason_text)
        status_raw_normalized = (
            str(extras.get("status_raw") or entry.get("status_raw") or status_text)
            .strip()
            .lower()
        )
        meta_rows_cell = _format_meta_cell(status_raw_normalized, extras, meta_payload)
        rows.append(
            [
                connector_id,
                str(entry.get("mode")),
                status_text,
                reason_cell,
                _format_duration(entry.get("duration_ms", 0)),
                _format_http(entry.get("http", {})),
                _format_rows(entry.get("counts", {})),
                kept_cell,
                dropped_cell,
                parse_cell,
                _format_coverage(coverage.get("ym_min"), coverage.get("ym_max")),
                _format_coverage(coverage.get("as_of_min"), coverage.get("as_of_max")),
                log_cell,
                meta_rows_cell,
                meta_cell,
            ]
        )
    if not rows:
        rows.append(["—"] * len(headers))
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_markdown(
    entries: Sequence[Mapping[str, Any]],
    *,
    dedupe_notes: Mapping[str, int] | None = None,
    reachability: Mapping[str, Any] | None = None,
) -> str:
    sorted_entries = sorted(entries, key=lambda item: str(item.get("connector_id", "")))
    total_fetched = sum(entry.get("counts", {}).get("fetched", 0) for entry in sorted_entries)
    total_written = sum(entry.get("counts", {}).get("written", 0) for entry in sorted_entries)
    dtm_entry = next((entry for entry in sorted_entries if entry.get("connector_id") == "dtm_client"), None)

    lines = [SUMMARY_TITLE, "", "## Run Summary", ""]
    lines.append(f"* **Connectors:** {len(sorted_entries)}")
    lines.append(f"* **Status counts:** {_format_status_counts(sorted_entries)}")
    lines.append(f"* **Reason histogram:** {_format_reason_histogram(sorted_entries)}")
    lines.append(f"* **Rows fetched:** {total_fetched}")
    lines.append(f"* **Rows written:** {total_written}")
    lines.append("")

    if dtm_entry:
        config_section = _render_config_section(dtm_entry)
        if config_section:
            lines.extend(config_section)
        staging_section = _render_staging_readiness(dtm_entry)
        if staging_section:
            lines.extend(staging_section)
        zero_rows_section = _render_zero_row_root_cause(dtm_entry)
        if zero_rows_section:
            lines.extend(zero_rows_section)
        selector_section = _render_selector_effectiveness(dtm_entry)
        if selector_section:
            lines.extend(selector_section)

    reachability_section = _render_reachability_section(reachability or {})
    if reachability_section:
        lines.extend(reachability_section)

    lines.append("## Per-Connector Table")
    lines.append("")
    lines.extend(_build_table(sorted_entries))
    lines.append("")
    if dedupe_notes:
        for connector_id in sorted(dedupe_notes):
            count = dedupe_notes[connector_id]
            if count:
                lines.append(f"(deduplicated {count} duplicate entries for {connector_id})")
        if any(count for count in dedupe_notes.values()):
            lines.append("")
    for entry in sorted_entries:
        lines.append(_render_details(entry))
        lines.append("")
    if dtm_entry:
        lines.append("")
        lines.extend(_render_dtm_deep_dive(dtm_entry))
    return "\n".join(lines).strip() + "\n"


def write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(content)


def append_to_summary(content: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    try:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(content)
            if not content.endswith("\n"):
                handle.write("\n")
    except OSError:  # pragma: no cover - environment specific
        pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Path to connectors_report.jsonl")
    parser.add_argument("--out", required=True, help="Path for the rendered summary.md")
    parser.add_argument(
        "--github-step-summary",
        action="store_true",
        help="When set, also append the Markdown to $GITHUB_STEP_SUMMARY",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_path = Path(args.report)
    out_path = Path(args.out)
    if not report_path.exists():
        stub = {
            "connector_id": "unknown",
            "mode": "real",
            "status": "error",
            "status_raw": "error",
            "reason": "missing-report",
            "http": {},
            "counts": {},
            "extras": {
                "hint": "connector did not start or preflight failed",
                "exit_code": 1,
                "status_raw": "error",
            },
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(stub))
            handle.write("\n")
    load_error: str | None = None
    try:
        entries = load_report(report_path)
    except Exception as exc:
        load_error = str(exc)
        stub_entry = {
            "connector_id": "summarizer",
            "mode": "real",
            "status": "error",
            "status_raw": "error",
            "reason": f"summarizer-error: {load_error}",
            "http": {},
            "counts": {},
            "extras": {"exit_code": 1, "status_raw": "error", "hint": "summarizer encountered an error"},
        }
        entries = [stub_entry]
    deduped_entries, dedupe_notes = deduplicate_entries(entries)
    reachability_path = Path("diagnostics/ingestion/dtm/reachability.json")
    reachability_payload = _safe_load_json(reachability_path) or {}
    markdown = build_markdown(deduped_entries, dedupe_notes=dedupe_notes, reachability=reachability_payload)
    write_markdown(out_path, markdown)
    if args.github_step_summary:
        append_to_summary(markdown)
    return 0 if load_error is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
