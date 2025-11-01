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
from typing import Any, Dict, Iterable, List, Mapping, Sequence

__all__ = [
    "render_summary_md",
    "render_dtm_deep_dive",
    "_render_dtm_deep_dive",
]

SUMMARY_PATH = Path("summary.md")
DEFAULT_REPORT_PATH = Path("diagnostics") / "ingestion" / "connectors_report.jsonl"
DEFAULT_DIAG_DIR = Path("diagnostics") / "ingestion"
DEFAULT_STAGING_DIR = Path("resolver") / "staging"
SUMMARY_TITLE = "# Ingestion Superreport"


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
    entries = _safe_load_jsonl(path) if path else []
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

def _safe_json_load(path: Path) -> Mapping[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, Mapping) else None


def _safe_load_jsonl(path: Path) -> List[Mapping[str, Any]]:
    entries: List[Mapping[str, Any]] = []
    if not path.exists():
        return entries
    try:
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                text = raw.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, Mapping):
                    entries.append(payload)
    except OSError:
        return []
    return entries


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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


@dataclass
class ConnectorRecord:
    name: str
    status: str
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
    extras = dict(entry.get("extras") or {})
    counts = dict(entry.get("counts") or {})
    http = dict(entry.get("http") or {})
    coverage = dict(entry.get("coverage") or {})
    samples = dict(entry.get("samples") or {})
    started_at = entry.get("started_at_utc") or extras.get("started_at_utc")
    duration_ms = _coerce_int(entry.get("duration_ms") or extras.get("duration_ms"))
    reason = entry.get("reason") or extras.get("reason")
    record = ConnectorRecord(
        name=name,
        status=status,
        reason=str(reason) if reason else None,
        started_at=str(started_at) if started_at else None,
        duration_ms=duration_ms,
        counts={key: _coerce_int(value) for key, value in counts.items()},
        http=http,
        extras=extras,
        coverage=coverage,
        samples=samples,
    )
    return record


def _load_optional(path: Path) -> Mapping[str, Any]:
    payload = _safe_json_load(path)
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


def _render_run_overview(records: Sequence[ConnectorRecord]) -> List[str]:
    lines = ["## Run Overview", ""]
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
    headers = [
        "connector",
        "status",
        "rows_fetched",
        "rows_normalized",
        "rows_written",
        "http.requests",
        "2xx/4xx/5xx",
        "retries",
        "timeouts",
        "latency_ms_p95",
        "reachability",
        "config_source",
        "config_path",
        "countries",
        "window_start",
        "window_end",
        "exported(flow)",
        "exported(stock)",
        "why_zero?",
        "error?",
    ]
    rows: List[List[str]] = []
    for record in sorted(records, key=lambda item: item.name):
        http = record.http or {}
        requests = _coerce_int(http.get("requests") or http.get("total") or http.get("2xx"))
        retries = _coerce_int(http.get("retries"))
        timeouts = _coerce_int(http.get("timeouts"))
        latency_p95 = http.get("latency_ms_p95") or http.get("p95_ms")
        reachability = record.reachability or {}
        reach_text = "; ".join(
            filter(
                None,
                [
                    reachability.get("dns_ip"),
                    f"tcp={reachability.get('tcp_ms')}ms" if reachability.get("tcp_ms") else None,
                    f"tls={reachability.get('tls_ok')}" if "tls_ok" in reachability else None,
                ],
            )
        ) or "—"
        config_path = record.config_path or record.extras.get("config_path")
        countries_text = "" if record.countries_count is None else str(record.countries_count)
        rows.append(
            [
                record.name,
                record.status,
                str(record.rows_fetched or "0"),
                str(record.rows_normalized or "0"),
                str(record.rows_written or "0"),
                str(requests or "0"),
                "/".join(
                    str(_coerce_int(http.get(key))) for key in ("2xx", "4xx", "5xx")
                ),
                str(retries or "0"),
                str(timeouts or "0"),
                str(latency_p95 or "—"),
                reach_text,
                record.config_source or "—",
                config_path or "—",
                countries_text or "—",
                record.window_start or "—",
                record.window_end or "—",
                _format_bool(record.exported_flow) if record.exported_flow is not None else "no",
                _format_bool(record.exported_stock) if record.exported_stock is not None else "no",
                _format_bool(bool(record.why_zero)) if record.why_zero is not None else "no",
                _format_bool(bool(record.error)) if record.error is not None else "no",
            ]
        )
    lines.extend(_format_table(headers, rows))
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
        fail_payload = _safe_json_load(fail_path)
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
    entries = _safe_load_jsonl(report_path)
    records = [_normalise_connector(entry) for entry in entries]
    for record in records:
        _parse_connector_context(record, diagnostics_dir)
    staging_snapshot = _collect_staging_snapshot(staging_dir)
    for record in records:
        staging_info = staging_snapshot.get(record.name, {}) if isinstance(staging_snapshot, Mapping) else {}
        if isinstance(staging_info, Mapping):
            record.exported_flow = any(
                "flow" in name and meta.get("rows", 0) for name, meta in staging_info.items()
                if isinstance(meta, Mapping)
            )
            record.exported_stock = any(
                "stock" in name and meta.get("rows", 0) for name, meta in staging_info.items()
                if isinstance(meta, Mapping)
            )

    parts: List[str] = [SUMMARY_TITLE, ""]
    parts.extend(_render_run_overview(records))
    parts.extend(_render_connector_matrix(records))
    parts.extend(_render_connector_details(records, staging_snapshot))
    parts.extend(_render_export_snapshot(staging_snapshot))
    parts.extend(_render_anomalies(records))
    parts.extend(_render_next_actions(records))
    return "\n".join(parts)


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

    content = render_summary_md(report_path, diagnostics_dir, staging_dir)
    _write_summary(output_path, content)
    print(f"Summary written to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main(sys.argv[1:]))
