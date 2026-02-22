# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

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

from resolver.db import duckdb_io

REPO_ROOT = Path(__file__).resolve().parents[2]
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
EXPORT_PREVIEW_COLUMNS = ["iso3", "as_of_date", "ym", "metric", "value", "semantics", "source"]
IDMC_WHY_ZERO_PATH = Path("diagnostics/ingestion/idmc/why_zero.json")
ACLED_ZERO_ROWS_PATH = Path("diagnostics/ingestion/acled/zero_rows.json")
ACLED_RUN_INFO_PATH = Path("diagnostics/ingestion/acled_client/acled_client_run.json")
ACLED_HTTP_DIAG_PATH = Path("diagnostics/ingestion/acled/http_diag.json")
DUCKDB_SUMMARY_PATH = Path("diagnostics/ingestion/duckdb_summary.md")
EXPORT_DB_DIAG_PATH = Path("diagnostics/ingestion/export_facts_db.json")
FREEZE_DB_DIAG_PATH = Path("diagnostics/ingestion/freeze_db.json")


def _normalise_duckdb_path(raw: str) -> str | None:
    candidate = raw.strip()
    if not candidate:
        return None
    if candidate.startswith("duckdb:///"):
        candidate = candidate.replace("duckdb:///", "", 1)
    try:
        return str(Path(candidate).expanduser().resolve())
    except Exception:
        try:
            return os.path.abspath(os.path.expanduser(candidate))
        except Exception:
            return candidate


def _resolve_duckdb_target() -> Tuple[str | None, str | None]:
    env_url = (os.environ.get("RESOLVER_DB_URL") or "").strip()
    env_path = (os.environ.get("RESOLVER_DB_PATH") or "").strip()
    canonical_url: str | None = None
    canonical_path: str | None = None

    if env_url:
        canonical_url = env_url
        try:
            from resolver.db.conn_shared import canonicalize_duckdb_target

            resolved_path, resolved_url = canonicalize_duckdb_target(env_url)
            canonical_url = resolved_url or env_url
            canonical_path = resolved_path or None
        except Exception:
            pass
        if canonical_path is None and canonical_url and canonical_url.startswith("duckdb:///"):
            canonical_path = _normalise_duckdb_path(canonical_url)
    elif env_path:
        canonical_path = _normalise_duckdb_path(env_path)
        if canonical_path:
            canonical_url = f"duckdb:///{Path(canonical_path).as_posix()}"

    return canonical_url, canonical_path


def _detect_window_bounds() -> Dict[str, str | None]:
    start = (os.environ.get("RESOLVER_START_ISO") or os.environ.get("BACKFILL_START_ISO") or "").strip() or None
    end = (os.environ.get("RESOLVER_END_ISO") or os.environ.get("BACKFILL_END_ISO") or "").strip() or None
    return {"start": start, "end": end}


def _find_recent_writer_log() -> str | None:
    base = Path("logs/ingestion")
    if not base.exists():
        return None
    candidates: List[Path] = []
    try:
        for candidate in base.rglob("*.log"):
            if candidate.is_file():
                candidates.append(candidate)
    except OSError:
        return None
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime if path.exists() else 0.0)
    return latest.as_posix()


def _collect_duckdb_breakdown(
    db_path: str,
    tables: Sequence[str],
    window: Mapping[str, str | None],
) -> Tuple[Dict[str, Any], str | None]:
    try:
        conn = duckdb_io.get_db(db_path)
    except Exception as exc:  # pragma: no cover - defensive
        return {}, f"duckdb connect failed: {exc}"

    breakdown: Dict[str, Any] = {}
    error: str | None = None
    start = window.get("start") if isinstance(window, Mapping) else None
    end = window.get("end") if isinstance(window, Mapping) else None
    try:
        for table in tables:
            try:
                exists = conn.execute(
                    "SELECT 1 FROM information_schema.tables WHERE table_name = ?",
                    [table],
                ).fetchone()
            except Exception as exc:  # pragma: no cover - defensive
                error = f"table probe failed for {table}: {exc}"
                continue
            if not exists:
                continue
            params: List[str] = []
            predicates: List[str] = []
            if start:
                predicates.append("COALESCE(as_of_date, as_of) >= ?")
                params.append(start)
            if end:
                predicates.append("COALESCE(as_of_date, as_of) <= ?")
                params.append(end)
            where_clause = ""
            if predicates:
                where_clause = " WHERE " + " AND ".join(predicates)
            try:
                rows = conn.execute(
                    """
                    SELECT
                      COALESCE(source_id, '') AS source_id,
                      COALESCE(metric, '') AS metric,
                      COALESCE(series_semantics, COALESCE(semantics, '')) AS semantics,
                      COUNT(*) AS rows
                    FROM {table}
                    {where}
                    GROUP BY 1, 2, 3
                    ORDER BY rows DESC, source_id, metric, semantics
                    """.format(table=table, where=where_clause),
                    params,
                ).fetchall()
            except Exception as exc:  # pragma: no cover - defensive
                error = f"query failed for {table}: {exc}"
                continue
            breakdown[table] = {
                "rows": [
                    {
                        "source_id": str(row[0] or ""),
                        "metric": str(row[1] or ""),
                        "semantics": str(row[2] or ""),
                        "rows": int(row[3]),
                    }
                    for row in rows
                ],
            }
    finally:
        duckdb_io.close_db(conn)
    return breakdown, error


def _relativize_path(raw: Any) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        path = Path(raw)
    except (TypeError, ValueError):
        return raw
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        try:
            return path.resolve().as_posix()
        except Exception:
            return str(path)


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


def _safe_read_markdown(path: Path) -> List[str]:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return []
    lines = [line.rstrip() for line in text.splitlines()]
    return [line for line in lines if line is not None]


def _load_acled_http_diag() -> Dict[str, Any] | None:
    if not ACLED_HTTP_DIAG_PATH.exists():
        return None
    try:
        return json.loads(ACLED_HTTP_DIAG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _candidate_run_paths(raw_entry: Mapping[str, Any], connector_id: str) -> List[Path]:
    extras = _ensure_dict(raw_entry.get("extras"))
    candidates: List[Path] = []
    direct_path = extras.get("run_details_path") or extras.get("run_path")
    if isinstance(direct_path, str) and direct_path.strip():
        candidates.append(Path(direct_path.strip()))

    slug = connector_id.replace(" ", "_") if connector_id else ""
    alt_slug = slug[:-7] if slug.endswith("_client") else slug
    labels = [label for label in {slug, alt_slug} if label]

    diagnostics_dir = extras.get("diagnostics_dir")
    if isinstance(diagnostics_dir, str) and diagnostics_dir.strip():
        diag_base = Path(diagnostics_dir.strip())
        for label in labels:
            candidates.append(diag_base / f"{label}_run.json")
        candidates.append(diag_base / "run.json")

    for label in labels:
        candidates.append(Path("diagnostics/ingestion") / label / f"{label}_run.json")

    deduped: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.as_posix()
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def _extract_run_counts(payload: Mapping[str, Any]) -> Dict[str, int]:
    rows_block = _ensure_dict(payload.get("rows"))
    totals_block = _ensure_dict(payload.get("totals"))
    counts = {
        "fetched": payload.get("rows_fetched"),
        "normalized": payload.get("rows_normalized"),
        "written": payload.get("rows_written"),
    }
    if counts["fetched"] in (None, ""):
        counts["fetched"] = rows_block.get("fetched")
    if counts["normalized"] in (None, ""):
        counts["normalized"] = rows_block.get("normalized")
    if counts["written"] in (None, ""):
        counts["written"] = rows_block.get("written")
    if counts["written"] in (None, ""):
        counts["written"] = totals_block.get("rows_written")
    staged_value: Any = payload.get("rows_staged_total")
    if staged_value in (None, ""):
        staged_value = rows_block.get("staged_total")
    if staged_value in (None, ""):
        staged_value = rows_block.get("staged")
    counts["staged"] = _sum_staging_counts(staged_value)
    return {key: _coerce_int(value) for key, value in counts.items()}


def _maybe_backfill_counts(
    entry: Dict[str, Any], export_summary: Mapping[str, Any] | None = None
) -> None:
    counts = entry.setdefault("counts", {})
    if any(counts.get(key, 0) > 0 for key in ("fetched", "normalized", "written")):
        return

    extras = dict(_ensure_dict(entry.get("extras")))
    summary_counts: Dict[str, int] = {}
    summary_path_raw = extras.get("summary_json") or extras.get("summary_path")
    if isinstance(summary_path_raw, str) and summary_path_raw.strip():
        summary_path = Path(summary_path_raw)
        if not summary_path.is_absolute():
            summary_path = (Path.cwd() / summary_path).resolve()
        summary_payload = _safe_load_json(summary_path)
        if summary_payload:
            summary_counts = {
                key: _coerce_int(value)
                for key, value in _ensure_dict(summary_payload.get("counts")).items()
            }

    staging_total = _sum_staging_counts(extras.get("staging_counts"))
    rows_written_extra = _coerce_int(extras.get("rows_written"))
    rows_normalized_extra = _coerce_int(extras.get("rows_normalized"))
    rows_fetched_extra = _coerce_int(extras.get("rows_fetched"))

    export_report_rows = 0
    if export_summary:
        report_block = _ensure_dict(export_summary.get("report"))
        export_report_rows = _coerce_int(report_block.get("rows_exported"))

    candidate_counts: Dict[str, int] = {}
    source: str | None = None
    if any(summary_counts.get(key, 0) > 0 for key in ("written", "normalized", "fetched")):
        candidate_counts = summary_counts
        source = "summary_json"
    elif any(
        value > 0
        for value in (rows_written_extra, rows_normalized_extra, rows_fetched_extra, staging_total)
    ):
        written_value = max(rows_written_extra, staging_total)
        if written_value <= 0:
            written_value = max(rows_normalized_extra, rows_fetched_extra)
        normalized_value = rows_normalized_extra if rows_normalized_extra > 0 else max(written_value, staging_total)
        fetched_value = rows_fetched_extra if rows_fetched_extra > 0 else max(normalized_value, written_value)
        candidate_counts = {
            "written": written_value,
            "normalized": normalized_value,
            "fetched": fetched_value,
        }
        if staging_total > 0:
            candidate_counts["staged"] = staging_total
        source = "extras"
    elif export_report_rows > 0:
        candidate_counts = {
            "written": export_report_rows,
            "normalized": export_report_rows,
            "fetched": export_report_rows,
        }
        source = "export_report"
    else:
        return

    applied = False
    for key, value in candidate_counts.items():
        coerced = _coerce_int(value)
        if coerced <= 0:
            continue
        if counts.get(key, 0) <= 0:
            counts[key] = coerced
            applied = True

    if applied:
        if source:
            extras["counts_fallback"] = source
        entry["counts"] = counts
        entry["extras"] = extras


def _override_counts_from_run_json(
    normalized_entry: Dict[str, Any], raw_entry: Mapping[str, Any]
) -> bool:
    counts = dict(_ensure_dict(normalized_entry.get("counts")))
    connector_id = str(normalized_entry.get("connector_id") or "")
    if counts and all(counts.get(key, 0) > 0 for key in ("fetched", "normalized", "written")):
        return False

    candidate_paths = _candidate_run_paths(raw_entry, connector_id)
    seen: set[Path] = set()
    for candidate in candidate_paths:
        path = candidate
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        payload = _safe_load_json(path)
        if not payload:
            continue
        run_counts = _extract_run_counts(payload)
        fetched_override = run_counts.get("fetched", 0)
        normalized_override = run_counts.get("normalized", 0)
        written_override = run_counts.get("written", 0)
        if fetched_override == 0:
            fetched_override = normalized_override
        if fetched_override == 0:
            fetched_override = written_override
        if normalized_override == 0 and fetched_override > 0:
            normalized_override = fetched_override
        applied = False
        if fetched_override > 0 and fetched_override > counts.get("fetched", 0):
            counts["fetched"] = fetched_override
            applied = True
        if normalized_override > 0 and normalized_override > counts.get("normalized", 0):
            counts["normalized"] = normalized_override
            applied = True
        if written_override > 0 and written_override > counts.get("written", 0):
            counts["written"] = written_override
            applied = True
        if applied:
            extras = dict(normalized_entry.get("extras") or {})
            extras["counts_override_source"] = "run.json"
            extras["counts_override_path"] = path.as_posix()
            normalized_entry["extras"] = extras
            normalized_entry["counts"] = counts
            return True
    return False


def _load_mapping_debug(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, Mapping):
                    records.append(dict(parsed))
    except OSError:
        return []
    return records


def _coerce_int(value: Any) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _sum_staging_counts(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(_coerce_int(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return sum(_coerce_int(item) for item in value)
    return _coerce_int(value)


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


def _collect_export_summary(
    staging_dir: Path,
    config_path: Path,
    preview_dir: Path,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "rows": 0,
        "headers": [],
        "preview": [],
        "warnings": [],
        "sources": [],
        "error": None,
        "report": {},
        "report_path": None,
        "rows_exported": 0,
        "db_stats": {},
    }
    # export_facts was removed in PR #610; normalization is now handled by
    # resolver.transform.normalize + resolver.tools.load_and_derive.
    return summary


def _render_duckdb_section(duckdb_info: Mapping[str, Any] | None) -> List[str]:
    info = dict(duckdb_info or {})
    table_stats = info.get("table_stats") or {}
    error = info.get("error")
    note = info.get("note")
    if not table_stats and not error and not note:
        return []

    lines = ["## DuckDB", ""]
    if note:
        lines.append(f"- **Status:** {note}")
        return lines
    if error:
        lines.append(f"- **Status:** {error}")
    else:
        db_target = info.get("db_path") or info.get("db_url")
        if db_target:
            lines.append(f"- **Database:** `{db_target}`")
        window = info.get("window") or {}
        window_start = (window.get("start") if isinstance(window, Mapping) else None) or "∅"
        window_end = (window.get("end") if isinstance(window, Mapping) else None) or "∅"
        if window_start != "∅" or window_end != "∅":
            lines.append(f"- **Window:** {window_start} → {window_end}")
        tables_line: List[str] = []
        rows_written_total = 0

        for table in sorted(table_stats):
            stats = table_stats.get(table) or {}
            try:
                rows_written = int(stats.get("rows_written", 0) or 0)
            except Exception:
                rows_written = 0
            try:
                rows_delta = int(stats.get("rows_delta", 0) or 0)
            except Exception:
                rows_delta = 0
            total_after = stats.get("rows_after")
            if total_after in (None, ""):
                total_after = stats.get("rows_before")
            try:
                total_after_int = int(total_after or 0)
            except Exception:
                total_after_int = 0
            tables_line.append(
                f"{table} (written={rows_written}, Δ={rows_delta}, total={total_after_int})"
            )
            rows_written_total += rows_written

        if tables_line:
            lines.append(f"- **Tables updated:** {', '.join(tables_line)}")
            lines.append(f"- **Rows written:** {rows_written_total}")

        log_path = info.get("log_path")
        if isinstance(log_path, str) and log_path.strip():
            lines.append(f"- **Writer logs:** `{log_path}`")

    breakdown = info.get("breakdown") or {}
    if breakdown:
        for table in sorted(breakdown):
            table_rows = breakdown.get(table) or {}
            entries = table_rows.get("rows") if isinstance(table_rows, Mapping) else None
            if not entries:
                continue
            lines.append("")
            lines.append(f"### {table} rows by source/metric")
            lines.append("")
            lines.append("| source_id | metric | semantics | rows |")
            lines.append("| --- | --- | --- | ---: |")
            for row in entries:
                if not isinstance(row, Mapping):
                    continue
                source = str(row.get("source_id", ""))
                metric = str(row.get("metric", ""))
                semantics = str(row.get("semantics", ""))
                try:
                    count = int(row.get("rows", 0) or 0)
                except Exception:
                    count = 0
                lines.append(f"| {source} | {metric} | {semantics} | {count} |")

    lines.append("")
    return lines


def _render_downstream_db_summary(
    export_payload: Mapping[str, Any] | None,
    freeze_payload: Mapping[str, Any] | None,
) -> List[str]:
    export_info = dict(export_payload or {})
    freeze_info = dict(freeze_payload or {})

    if not export_info and not freeze_info:
        return []

    lines = ["### DuckDB — downstream writes (summary)", ""]

    if export_info:
        lines.append("- **Export Facts:**")
        lines.extend(
            _format_downstream_block(
                export_info,
                keys=(
                    ("db_url", "db_url"),
                    ("facts_resolved_rows", "facts_resolved rows"),
                    ("facts_deltas_rows", "facts_deltas rows"),
                    ("facts_resolved_semantics", "facts_resolved semantics"),
                    ("facts_deltas_semantics", "facts_deltas semantics"),
                ),
            )
        )

    if freeze_info:
        if export_info:
            lines.append("")
        month_label = freeze_info.get("month")
        header = "- **Freeze Snapshot:**" if not month_label else f"- **Freeze Snapshot ({month_label}):**"
        lines.append(header)
        lines.extend(
            _format_downstream_block(
                freeze_info,
                keys=(
                    ("db_url", "db_url"),
                    ("facts_resolved_rows", "facts_resolved rows"),
                    ("facts_deltas_rows", "facts_deltas rows"),
                    ("facts_resolved_semantics", "facts_resolved semantics"),
                    ("facts_deltas_semantics", "facts_deltas semantics"),
                ),
            )
        )

    lines.append("")
    return lines


def _format_downstream_block(
    payload: Mapping[str, Any],
    *,
    keys: Sequence[tuple[str, str]],
) -> List[str]:
    formatted: List[str] = []
    for key, label in keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if isinstance(value, (dict, list)):
            try:
                rendered = json.dumps(value, sort_keys=True)
            except Exception:
                rendered = str(value)
        else:
            try:
                rendered = str(_coerce_int(value)) if isinstance(value, (int, float)) else str(value)
            except Exception:
                rendered = str(value)
        formatted.append(f"  - {label}: {rendered}")
    if not formatted:
        formatted.append("  - (no details recorded)")
    return formatted


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
                normalised = _normalise_entry(payload)
                _override_counts_from_run_json(normalised, payload)
                entries.append(normalised)
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


def _render_acled_http_section(entries: Sequence[Mapping[str, Any]]) -> List[str]:
    acled_entry = next((entry for entry in entries if entry.get("connector_id") == "acled_client"), None)
    if not acled_entry:
        return []

    counts = _ensure_dict(acled_entry.get("counts"))
    http_block = _ensure_dict(acled_entry.get("http"))
    acled_http_diag = _load_acled_http_diag()
    run_info = _safe_load_json(ACLED_RUN_INFO_PATH) or {}
    zero_info = _safe_load_json(ACLED_ZERO_ROWS_PATH) or {}

    rows_fetched = _coerce_int(counts.get("fetched"))
    rows_normalized = _coerce_int(counts.get("normalized"))
    rows_written = _coerce_int(counts.get("written"))

    http_status = run_info.get("http_status")
    if http_status in (None, ""):
        http_status = http_block.get("last_status")
    if acled_http_diag and acled_http_diag.get("status") not in (None, ""):
        http_status = acled_http_diag.get("status")

    base_url = run_info.get("base_url") or zero_info.get("base_url")
    window = _ensure_dict(run_info.get("window")) or _ensure_dict(zero_info)
    params_keys = run_info.get("params_keys") or zero_info.get("params_keys") or []

    lines = ["## ACLED HTTP diagnostics", ""]
    lines.append(f"- **Rows fetched:** {rows_fetched}")
    lines.append(f"- **Rows normalized:** {rows_normalized}")
    lines.append(f"- **Rows written:** {rows_written}")

    if window:
        start = window.get("start") or "—"
        end = window.get("end") or window.get("window_end") or "—"
        lines.append(f"- **Window:** {start} → {end}")

    if http_status not in (None, ""):
        lines.append(f"- **Last HTTP status:** {http_status}")

    if base_url:
        lines.append(f"- **Base URL:** `{base_url}`")

    if acled_http_diag and acled_http_diag.get("url"):
        lines.append(f"- **Last API URL:** `{acled_http_diag.get('url')}`")

    if params_keys:
        if isinstance(params_keys, (list, tuple, set)):
            keys = sorted({str(key) for key in params_keys if str(key).strip()})
            if keys:
                lines.append(f"- **Query params:** {', '.join(keys)}")

    zero_reason = zero_info.get("reason") or zero_info.get("zero_rows_reason")
    if zero_reason:
        start = zero_info.get("start") or window.get("start") if window else None
        end = zero_info.get("end") or window.get("end") if window else None
        window_display = " → ".join(part for part in [start or "—", end or "—"])
        lines.append(f"- **Zero rows reason:** {zero_reason} (window {window_display})")

    lines.append("")
    return lines


def _render_idmc_why_zero(payload: Mapping[str, Any]) -> List[str]:
    token_present = bool(payload.get("token_present"))
    countries_count = _coerce_int(payload.get("countries_count"))
    sample_raw = payload.get("countries_sample")
    if isinstance(sample_raw, Sequence) and not isinstance(sample_raw, (str, bytes)):
        sample_list = [str(item) for item in list(sample_raw) if str(item).strip()][:5]
    else:
        sample_list = []
    sample_display = ", ".join(sample_list) if sample_list else "—"
    if sample_list and countries_count > len(sample_list):
        sample_display = sample_display + "…"
    window_block = _ensure_dict(payload.get("window"))
    window_start = window_block.get("start")
    window_end = window_block.get("end")
    start_display = str(window_start).strip() if window_start not in (None, "") else "—"
    end_display = str(window_end).strip() if window_end not in (None, "") else "—"
    if start_display == end_display:
        window_display = start_display
    else:
        window_display = f"{start_display} → {end_display}"
    filters_block = _ensure_dict(payload.get("filters"))
    dropped_window = _coerce_int(filters_block.get("date_out_of_window"))
    dropped_iso3 = _coerce_int(filters_block.get("no_iso3"))
    dropped_value = _coerce_int(filters_block.get("no_value_col"))
    network_attempted = bool(payload.get("network_attempted"))
    config_source = str(payload.get("config_source") or "unknown")
    config_path = _relativize_path(payload.get("config_path_used")) or "—"
    warnings_raw = payload.get("loader_warnings") or []
    if isinstance(warnings_raw, Sequence) and not isinstance(warnings_raw, (str, bytes)):
        warnings_list = [str(item) for item in warnings_raw if str(item).strip()]
    else:
        warnings_list = []
    lines = ["## IDMC — Why zero?", ""]
    lines.append(f"- Token present: {str(token_present).lower()}")
    lines.append(f"- Countries: {countries_count} (sample: {sample_display})")
    lines.append(f"- Window: {window_display}")
    lines.append(
        "- Filters dropped: "
        f"date_out_of_window={dropped_window}, no_iso3={dropped_iso3}, no_value_col={dropped_value}"
    )
    requests_attempted = payload.get("requests_attempted")
    requests_display = _coerce_int(requests_attempted)
    lines.append(
        f"- Network attempted: {str(network_attempted).lower()}"
        f" (requests={requests_display})"
    )
    zero_reason = payload.get("zero_rows_reason")
    rows_block = _ensure_dict(payload.get("rows"))
    normalized_rows = _coerce_int(rows_block.get("normalized"))
    if zero_reason and normalized_rows == 0:
        lines.append(f"- Zero rows reason: {zero_reason}")
    lines.append(f"- Config: {config_source} — {config_path}")
    if warnings_list:
        lines.append(f"- Loader warnings: {'; '.join(warnings_list)}")
    return lines


def _extract_idmc_fallback(
    manifest: Mapping[str, Any], why_zero: Mapping[str, Any] | None = None
) -> Mapping[str, Any]:
    notes_block = _ensure_dict(manifest.get("notes"))
    zero_rows_block = _ensure_dict(notes_block.get("zero_rows"))
    candidates = [
        notes_block.get("fallback"),
        zero_rows_block.get("fallback"),
        _ensure_dict(why_zero or {}).get("fallback"),
    ]
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _infer_helix_endpoint(manifest: Mapping[str, Any]) -> str | None:
    run_block = _ensure_dict(manifest.get("run"))
    notes_block = _ensure_dict(manifest.get("notes"))
    http_block = _ensure_dict(manifest.get("http"))
    helix_block = _ensure_dict(notes_block.get("helix"))
    helix_endpoint = (
        run_block.get("helix_endpoint")
        or helix_block.get("helix_endpoint")
        or http_block.get("helix_endpoint")
    )
    if helix_endpoint:
        return str(helix_endpoint)
    endpoints = _ensure_dict(run_block.get("endpoints"))
    for _, url in endpoints.items():
        url_text = str(url)
        lowered = url_text.lower()
        if "last-180" in lowered or "idus" in lowered or "helix" in lowered:
            return "idus_last180"
        if "gidd" in lowered:
            return "gidd_displacements"
    return None


def _render_idmc_manifest_section(
    manifest: Mapping[str, Any], *, why_zero: Mapping[str, Any] | None = None
) -> List[str]:
    lines = ["## IDMC Run Diagnostics", ""]
    if not manifest:
        lines.append("- **diagnostics/ingestion/idmc/manifest.json:** not present")
        lines.append("")
        return lines

    run_block = _ensure_dict(manifest.get("run"))
    http_block = _ensure_dict(manifest.get("http"))
    normalize_block = _ensure_dict(manifest.get("normalize"))
    notes_block = _ensure_dict(manifest.get("notes"))
    helix_endpoint = _infer_helix_endpoint(manifest)
    network_mode = run_block.get("network_mode") or run_block.get("mode") or "unknown"
    rows_fetched = _coerce_int(normalize_block.get("rows_fetched"))
    rows_normalized = _coerce_int(normalize_block.get("rows_normalized"))
    rows_written = _coerce_int(
        normalize_block.get("rows_written") or normalize_block.get("rows_staged")
    )
    zero_rows_block = _ensure_dict(notes_block.get("zero_rows"))
    zero_reason = (
        zero_rows_block.get("notes")
        or zero_rows_block.get("zero_rows_reason")
        or _ensure_dict(why_zero or {}).get("zero_rows_reason")
    )

    lines.append(f"- **Network mode:** {network_mode}")
    if helix_endpoint:
        lines.append(f"- **HELIX endpoint:** {helix_endpoint}")
    lines.append(
        f"- **Rows (fetched/normalized/written):** {rows_fetched}/{rows_normalized}/{rows_written}"
    )

    requests_count = _coerce_int(http_block.get("requests"))
    retries = _coerce_int(http_block.get("retries"))
    last_status = http_block.get("status_last") or http_block.get("last_status")
    if any(value is not None for value in (requests_count, retries, last_status)):
        lines.append(
            "- **HTTP:** "
            f"requests={requests_count}, retries={retries}, last_status={last_status or 'n/a'}"
        )
        if http_block:
            lines.append(f"- **Status counts:** {_format_http(http_block)}")

    fallback_block = _extract_idmc_fallback(manifest, why_zero)
    if fallback_block:
        used = fallback_block.get("used")
        parts: List[str] = []
        if used is not None:
            parts.append(f"used={'yes' if used else 'no'}")
        reason_text = fallback_block.get("reason")
        status_text = fallback_block.get("status") or fallback_block.get("status_code")
        rows_value = fallback_block.get("rows")
        resource_url = fallback_block.get("resource_url") or fallback_block.get("package_url")
        if reason_text:
            parts.append(f"reason={reason_text}")
        if status_text is not None:
            parts.append(f"status={status_text}")
        if rows_value is not None:
            parts.append(f"rows={rows_value}")
        if resource_url:
            parts.append(f"resource={resource_url}")
        if parts:
            lines.append(f"- **Fallback:** {', '.join(parts)}")

    if zero_reason:
        lines.append(f"- **Zero-rows reason:** {zero_reason}")

    lines.append("")
    return lines


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
    status_counts = {}
    raw_counts = http.get("status_counts") or http
    if isinstance(raw_counts, Mapping):
        status_counts = {key: raw_counts.get(key) for key in ("2xx", "4xx", "5xx")}
    success = _coerce_int(status_counts.get("2xx"))
    client = _coerce_int(status_counts.get("4xx"))
    server = _coerce_int(status_counts.get("5xx"))
    retries = _coerce_int(http.get("retries"))
    summary = f"{success}/{client}/{server} ({retries})"
    if success == 0 and client == 0 and server == 0:
        last_kind = http.get("last_exception_kind") or http.get("last_exception")
        if last_kind:
            summary += f"; neterr={last_kind}"
    return summary


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
    config_block = _ensure_dict(extras.get("config"))
    config_source = config_block.get("config_source_label") or extras.get("config_source")
    if config_source:
        bullets.append(f"- **Config source:** {config_source}")
    config_warnings = config_block.get("config_warnings")
    if isinstance(config_warnings, Sequence) and not isinstance(config_warnings, (str, bytes)):
        joined_warnings = "; ".join(str(item) for item in config_warnings if str(item))
        if joined_warnings:
            bullets.append(f"- **Config warnings:** {joined_warnings}")

    config_path_used = (
        config_block.get("config_path_used")
        or extras.get("config_path_used")
        or extras.get("config_path")
    )
    countries_mode = (
        config_block.get("countries_mode")
        or _ensure_dict(config_block.get("config_parse")).get("countries_mode")
    )
    rel_config_path = _relativize_path(config_path_used)
    if rel_config_path:
        line = f"- Config: {rel_config_path}"
        if countries_mode:
            line += f" (countries_mode={countries_mode})"
        bullets.append(line)

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
    attempted_dates = normalize.get("attempted_date_columns")
    if isinstance(attempted_dates, (list, tuple)) and attempted_dates:
        lines.append("Attempted date columns: " + ", ".join(str(col) for col in attempted_dates))
    used_date = normalize.get("date_col_used")
    if used_date:
        lines.append(f"Chosen date column: {used_date}")
    bad_samples = normalize.get("bad_date_samples")
    if isinstance(bad_samples, (list, tuple)) and bad_samples:
        lines.append("Bad date samples: " + ", ".join(str(sample) for sample in bad_samples[:5]))
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
    config_parse = _ensure_dict(config.get("config_parse"))
    config_keys_found = _ensure_dict(config.get("config_keys_found"))
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
    selected_values = config.get("selected_iso3_preview")
    if isinstance(selected_values, Iterable) and not isinstance(selected_values, (str, bytes)):
        selected_text = ", ".join(str(item) for item in list(selected_values)[:10] if str(item)) or "—"
    else:
        selected_text = "—"
    parse_countries = []
    raw_countries = config_parse.get("countries")
    if isinstance(raw_countries, Iterable) and not isinstance(raw_countries, (str, bytes)):
        parse_countries = [str(item) for item in raw_countries if str(item)]
    parsed_admin_levels = []
    raw_levels = config_parse.get("admin_levels")
    if isinstance(raw_levels, Iterable) and not isinstance(raw_levels, (str, bytes)):
        parsed_admin_levels = [str(item) for item in raw_levels if str(item)]
    if parse_countries:
        countries_parse_line = f"- **Countries parse:** api.countries: found ({len(parse_countries)})"
    else:
        countries_parse_line = "- **Countries parse:** api.countries: missing or empty"
    if parsed_admin_levels:
        levels_repr = ", ".join(parsed_admin_levels)
        admin_parse_line = f"- **Admin levels parse:** api.admin_levels: found ([{levels_repr}])"
    else:
        admin_parse_line = "- **Admin levels parse:** api.admin_levels: missing or empty"
    keys_line = "- **Config keys found:** countries={0}, admin_levels={1}".format(
        str(bool(config_keys_found.get("countries"))).lower(),
        str(bool(config_keys_found.get("admin_levels"))).lower(),
    )
    countries_mode = str(config.get("countries_mode", "discovered")).strip().lower()
    warn_unapplied = bool(config_keys_found.get("countries")) and countries_mode == "discovered"
    source_label = config.get("config_source_label") or config.get("config_source")
    raw_warnings = config.get("config_warnings")
    warnings_list: List[str] = []
    if isinstance(raw_warnings, Sequence) and not isinstance(raw_warnings, (str, bytes)):
        warnings_list = [str(item) for item in raw_warnings if str(item)]

    lines = ["## Config used", ""]
    if source_label:
        lines.append(f"- **Source:** {source_label}")
    if warnings_list:
        lines.append(f"- **Warnings:** {'; '.join(warnings_list)}")
    lines.extend(
        [
            f"- **Path:** `{config.get('config_path_used', 'unknown')}`",
            f"- **Exists:** {_format_yes_no(config.get('config_exists'))}",
            f"- **SHA256:** `{config.get('config_sha256', 'n/a')}`",
            f"- **Countries mode:** `{config.get('countries_mode', 'discovered')}`",
            f"- **Countries count:** {config.get('countries_count', 0)}",
            f"- **Countries preview:** {preview_text}",
            f"- **Selected ISO3 preview:** {selected_text}",
            f"- **Admin levels:** {admin_text}",
            f"- **No date filter:** {_format_yes_no(config.get('no_date_filter'))}",
            countries_parse_line,
            admin_parse_line,
            keys_line,
        ]
    )
    if warn_unapplied:
        lines.append("- ⚠ config had api.countries but selector list not applied (check loader/version).")
    lines.append("")
    return lines


def _render_source_sample_quick_checks() -> List[str]:
    sample_path = Path("diagnostics/ingestion/dtm/samples/admin0_head.csv")
    lines = ["## Source sample: quick checks", ""]
    if not sample_path.exists():
        lines.append("- **admin0_head.csv:** not present")
        lines.append("")
        return lines

    iso_counter: Counter[str] = Counter()
    name_counter: Counter[str] = Counter()
    has_iso_column = False
    has_admin_column = False
    try:
        with sample_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [str(field).strip() for field in (reader.fieldnames or [])]
            has_iso_column = "CountryISO3" in fieldnames
            has_admin_column = "admin0Name" in fieldnames
            for row in reader:
                if not isinstance(row, Mapping):
                    continue
                if has_iso_column:
                    iso_value = str(row.get("CountryISO3") or "").strip()
                    if iso_value:
                        iso_counter[iso_value] += 1
                if has_admin_column:
                    name_value = str(row.get("admin0Name") or "").strip()
                    if name_value:
                        name_counter[name_value] += 1
    except Exception:
        lines.append("- **admin0_head.csv:** unable to read (see artifact)")
        lines.append("")
        return lines

    if not has_iso_column:
        lines.append("- **CountryISO3 top 5:** column not present")
    elif iso_counter:
        top_iso = ", ".join(f"{code} ({count})" for code, count in iso_counter.most_common(5))
        lines.append(f"- **CountryISO3 top 5:** {top_iso}")
    else:
        lines.append("- **CountryISO3 top 5:** —")

    if not has_admin_column:
        lines.append("- **admin0Name top 5:** column not present")
    elif name_counter:
        top_names = ", ".join(f"{name} ({count})" for name, count in name_counter.most_common(5))
        lines.append(f"- **admin0Name top 5:** {top_names}")
    else:
        lines.append("- **admin0Name top 5:** —")

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


def _render_dtm_date_filter(entry: Mapping[str, Any]) -> List[str]:
    extras = _ensure_dict(entry.get("extras"))
    date_filter = _ensure_dict(extras.get("date_filter"))
    if not date_filter:
        return []

    parsed_ok = _coerce_int(date_filter.get("parsed_ok"))
    parsed_total = _coerce_int(date_filter.get("parsed_total"))
    percentage = "—"
    if parsed_total:
        percentage = f"{(parsed_ok / parsed_total) * 100:.1f}%"
    min_date = date_filter.get("min_date") or "—"
    max_date = date_filter.get("max_date") or "—"
    inside = _coerce_int(date_filter.get("inside"))
    outside = _coerce_int(date_filter.get("outside"))
    parse_failed = _coerce_int(date_filter.get("parse_failed"))
    window_start = date_filter.get("window_start") or "—"
    window_end = date_filter.get("window_end") or "—"
    inclusive_text = "inclusive" if date_filter.get("inclusive", True) else "exclusive"
    skipped = _format_yes_no(date_filter.get("skipped"))
    column_used = str(date_filter.get("date_column_used") or "—")
    drop_counts = _ensure_dict(date_filter.get("drop_counts"))
    artifacts = _ensure_dict(extras.get("artifacts"))
    sample_path = artifacts.get("normalize_drops")

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
    if drop_counts:
        formatted = ", ".join(
            f"{reason} ({_coerce_int(count)})" for reason, count in sorted(drop_counts.items())
        )
        if formatted:
            lines.append(f"- **Drop reasons:** {formatted}")
    if isinstance(sample_path, str) and sample_path:
        lines.append(f"- **Drop sample:** `{sample_path}`")
    lines.append("")
    return lines


def _render_dtm_per_country(entry: Mapping[str, Any]) -> List[str]:
    extras = _ensure_dict(entry.get("extras"))
    per_country_raw = extras.get("per_country")
    if not isinstance(per_country_raw, list) or not per_country_raw:
        return []

    rows: List[List[str]] = []
    for item in per_country_raw:
        if not isinstance(item, Mapping):
            continue
        country = str(item.get("country") or "—")
        level = str(item.get("level") or "—")
        param = str(item.get("param") or "")
        if param.lower() == "iso3":
            param = "CountryISO3"
        elif param.lower() == "name":
            param = "CountryName"
        pages = _coerce_int(item.get("pages"))
        rows_count = _coerce_int(item.get("rows"))
        skipped = bool(item.get("skipped_no_match"))
        reason = str(item.get("reason") or "")
        skipped_text = reason if skipped and reason else ("yes" if skipped else "")
        rows.append(
            [
                country,
                level,
                param or "—",
                str(pages),
                str(rows_count),
                skipped_text,
            ]
        )
    if not rows:
        return []

    header = ["Country", "Level", "Param", "Pages", "Rows", "Skipped"]
    lines = ["## DTM per-country results", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
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


def _render_idmc_reachability_section(reachability: Mapping[str, Any]) -> List[str]:
    lines = ["## IDMC Reachability", ""]
    if not reachability:
        lines.append("- **diagnostics/ingestion/idmc/probe.json:** not present")
        lines.append("")
        return lines

    target = _ensure_dict(reachability.get("target"))
    host = target.get("host", "backend.idmcdb.org")
    port = target.get("port", 443)
    lines.append(f"- **Target:** `{host}:{port}`")
    captured_start = reachability.get("generated_at") or reachability.get("captured_at")
    captured_end = reachability.get("completed_at")
    if captured_start or captured_end:
        lines.append(f"- **Captured:** {captured_start or '—'} → {captured_end or '—'}")

    dns = _ensure_dict(reachability.get("dns"))
    dns_records: List[str] = []
    for entry in dns.get("records", []):
        if isinstance(entry, Mapping):
            address = entry.get("address")
            family = entry.get("family")
            if address:
                dns_records.append(
                    f"{address}{f' ({family})' if family else ''}"
                )
    if dns.get("error"):
        dns_line = f"- **DNS:** error={dns.get('error')}"
    else:
        dns_line = f"- **DNS:** {', '.join(dns_records) if dns_records else '—'}"
    if dns.get("elapsed_ms") is not None:
        dns_line += f" ({dns.get('elapsed_ms')}ms)"
    lines.append(dns_line)

    tcp = _ensure_dict(reachability.get("tcp"))
    if tcp:
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
            tcp_line = f"- **TCP:** error={tcp.get('error', 'unknown')}"
            if tcp.get("elapsed_ms") is not None:
                tcp_line += f" after {tcp.get('elapsed_ms')}ms"
        lines.append(tcp_line)

    tls = _ensure_dict(reachability.get("tls"))
    if tls:
        if tls.get("ok"):
            tls_line = "- **TLS:** ok"
            version = tls.get("version")
            cipher = tls.get("cipher")
            if version:
                tls_line += f" version={version}"
            if cipher:
                tls_line += f" cipher={cipher}"
            if tls.get("elapsed_ms") is not None:
                tls_line += f" ({tls.get('elapsed_ms')}ms)"
        else:
            tls_line = f"- **TLS:** error={tls.get('error', 'unknown')}"
            if tls.get("elapsed_ms") is not None:
                tls_line += f" after {tls.get('elapsed_ms')}ms"
        lines.append(tls_line)

    http = _ensure_dict(reachability.get("http"))
    if http:
        if http.get("status_code") is not None:
            http_line = f"- **HTTP GET:** status={http.get('status_code')}"
            if http.get("elapsed_ms") is not None:
                http_line += f" ({http.get('elapsed_ms')}ms)"
        else:
            error_text = http.get("exception") or http.get("error") or "unknown"
            http_line = f"- **HTTP GET:** error={error_text}"
            if http.get("elapsed_ms") is not None:
                http_line += f" after {http.get('elapsed_ms')}ms"
        lines.append(http_line)

    egress = reachability.get("egress") or {}
    egress_ip = None
    if isinstance(egress, Mapping):
        for key in ("ifconfig.me", "ipify"):
            entry = _ensure_dict(egress.get(key))
            if entry.get("text"):
                egress_ip = entry.get("text")
                break
    if egress_ip:
        lines.append(f"- **Egress IP:** {egress_ip}")

    ca_bundle = reachability.get("ca_bundle") or tls.get("ca_bundle")
    if ca_bundle:
        lines.append(f"- **CA bundle:** {ca_bundle}")
    verify = http.get("verify") if isinstance(http, Mapping) else None
    if verify is not None:
        lines.append(f"- **Verify TLS:** {verify}")
    lines.append("")
    return lines


def _render_hdx_reachability_section(reachability: Mapping[str, Any]) -> List[str]:
    lines = ["## HDX Reachability", ""]
    if not reachability:
        lines.append("- **diagnostics/ingestion/idmc/hdx_probe.json:** not present")
        lines.append("")
        return lines

    dataset = reachability.get("dataset") or reachability.get("target", {}).get(
        "dataset"
    )
    if dataset:
        lines.append(f"- **Dataset:** `{dataset}`")

    package_status = reachability.get("package_status_code")
    if package_status is not None:
        lines.append(f"- **package_show:** status={package_status}")

    resource_id = reachability.get("resource_id") or reachability.get("target", {}).get(
        "resource_id"
    )
    if resource_id:
        lines.append(f"- **Resource id:** `{resource_id}`")

    resource_status = reachability.get("resource_status_code")
    resource_url = reachability.get("resource_url")
    if resource_status is not None or resource_url:
        status_text = (
            f"status={resource_status}" if resource_status is not None else "status=unknown"
        )
        if resource_url:
            status_text += f" url={resource_url}"
        lines.append(f"- **Resource:** {status_text}")

    threshold = reachability.get("bytes_threshold") or 0
    resource_bytes = reachability.get("resource_bytes")
    if resource_bytes is not None:
        bytes_ok = reachability.get("bytes_ok")
        suffix = " (ok)" if bytes_ok else " (below minimum)"
        if threshold:
            lines.append(
                f"- **Bytes downloaded:** {resource_bytes} / min {threshold}{suffix}"
            )
        else:
            lines.append(f"- **Bytes downloaded:** {resource_bytes}{suffix}")

    header_line = reachability.get("header_line")
    if header_line:
        lines.append(f"- **Header sample:** `{header_line}`")
        header_checks = []
        if reachability.get("header_has_iso3"):
            header_checks.append("iso3")
        if reachability.get("header_has_value"):
            header_checks.append("figure/new_displacements")
        if header_checks:
            lines.append(f"- **Header contains:** {', '.join(header_checks)}")

    error_text = reachability.get("error")
    if error_text:
        lines.append(f"- **Error:** {error_text}")

    resource_error = reachability.get("resource_error")
    if resource_error and resource_error != error_text:
        lines.append(f"- **Resource error:** {resource_error}")

    lines.append("")
    return lines


def _build_table(
    entries: Sequence[Mapping[str, Any]], *, idmc_manifest: Mapping[str, Any] | None = None
) -> List[str]:
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
        if connector_id == "idmc" and idmc_manifest:
            manifest_normalize = _ensure_dict(idmc_manifest.get("normalize"))
            manifest_rows_written = manifest_normalize.get("rows_written") or manifest_normalize.get(
                "rows_staged"
            )
            counts_map = dict(counts_map)
            for source_key, dest_key in (
                ("rows_fetched", "fetched"),
                ("rows_normalized", "normalized"),
            ):
                if manifest_normalize.get(source_key) is not None:
                    counts_map[dest_key] = _coerce_int(manifest_normalize.get(source_key))
            if manifest_rows_written is not None:
                counts_map["written"] = _coerce_int(manifest_rows_written)
            manifest_http = _ensure_dict(idmc_manifest.get("http"))
            if manifest_http:
                entry["http"] = manifest_http
            entry["counts"] = counts_map

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
                _format_rows(counts_map),
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
    idmc_reachability: Mapping[str, Any] | None = None,
    hdx_reachability: Mapping[str, Any] | None = None,
    export_summary: Mapping[str, Any] | None = None,
    mapping_debug: Sequence[Mapping[str, Any]] | None = None,
    idmc_manifest: Mapping[str, Any] | None = None,
) -> str:
    sorted_entries = sorted(entries, key=lambda item: str(item.get("connector_id", "")))
    export_info = export_summary or {}
    idmc_manifest_data = _ensure_dict(idmc_manifest or {})
    for entry in sorted_entries:
        if entry.get("connector_id") == "idmc" and idmc_manifest_data:
            manifest_normalize = _ensure_dict(idmc_manifest_data.get("normalize"))
            manifest_rows_written = manifest_normalize.get("rows_written") or manifest_normalize.get(
                "rows_staged"
            )
            counts_overlay = _ensure_dict(entry.get("counts"))
            counts_overlay = dict(counts_overlay)
            for source_key, dest_key in (
                ("rows_fetched", "fetched"),
                ("rows_normalized", "normalized"),
            ):
                if manifest_normalize.get(source_key) is not None:
                    counts_overlay[dest_key] = _coerce_int(manifest_normalize.get(source_key))
            if manifest_rows_written is not None:
                counts_overlay["written"] = _coerce_int(manifest_rows_written)
            entry["counts"] = counts_overlay
            manifest_http = _ensure_dict(idmc_manifest_data.get("http"))
            if manifest_http:
                entry["http"] = manifest_http
    for entry in sorted_entries:
        _maybe_backfill_counts(entry, export_info)
    total_fetched = sum(entry.get("counts", {}).get("fetched", 0) for entry in sorted_entries)
    total_normalized = sum(entry.get("counts", {}).get("normalized", 0) for entry in sorted_entries)
    total_written = sum(entry.get("counts", {}).get("written", 0) for entry in sorted_entries)
    dtm_entry = next((entry for entry in sorted_entries if entry.get("connector_id") == "dtm_client"), None)
    mapping_debug_records = list(mapping_debug or [])

    override_note = any(
        _ensure_dict(entry.get("extras")).get("counts_override_source") == "run.json"
        for entry in sorted_entries
    )
    footnote = " (from run.json)" if override_note else ""

    lines = [SUMMARY_TITLE, "", "## Run Summary", ""]
    lines.append(f"* **Connectors:** {len(sorted_entries)}")
    lines.append(f"* **Status counts:** {_format_status_counts(sorted_entries)}")
    lines.append(f"* **Reason histogram:** {_format_reason_histogram(sorted_entries)}")
    lines.append(f"* **Rows fetched:** {total_fetched}{footnote}")
    lines.append(f"* **Rows normalized:** {total_normalized}{footnote}")
    lines.append(f"* **Rows written:** {total_written}{footnote}")
    lines.append("")

    export_error = export_info.get("error")
    if export_info or export_error or mapping_debug_records:
        lines.append("## Export Facts")
        lines.append("")
        if export_error:
            lines.append(f"- **Status:** {export_error}")
        elif export_info:
            rows_written = export_info.get("rows", 0)
            csv_path = export_info.get("csv_path")
            lines.append(f"- **facts.csv rows:** {rows_written}")
            if csv_path:
                lines.append(f"- **facts.csv path:** `{csv_path}`")
            preview_headers = export_info.get("headers") or []
            preview_rows = export_info.get("preview") or []
            table_lines = _format_markdown_table(preview_headers, preview_rows)
            if table_lines:
                lines.append("")
                lines.extend(table_lines)
            warnings_list = [str(w) for w in export_info.get("warnings", []) if str(w)]
            if warnings_list:
                lines.append("")
                lines.append("**Mapping warnings:**")
                for warning in warnings_list:
                    lines.append(f"- {warning}")
            sources_details = export_info.get("sources") or []
            if sources_details:
                lines.append("")
                lines.append("**Source mappings:**")
                for detail in sources_details:
                    name = detail.get("name", "unknown")
                    strategy = detail.get("strategy", "")
                    rows_in = detail.get("rows_in", 0)
                    rows_out = detail.get("rows_out", 0)
                    summary_line = f"- `{name}` ({strategy or 'config'}): in={rows_in}, out={rows_out}"
                    warnings_detail = detail.get("warnings") or []
                    if warnings_detail:
                        joined = "; ".join(str(w) for w in warnings_detail if str(w))
                        if joined:
                            summary_line += f" — warnings: {joined}"
                    lines.append(summary_line)
        else:
            lines.append("- **Status:** export summary unavailable")

    acled_section = _render_acled_http_section(sorted_entries)
    if acled_section:
        lines.append("")
        lines.extend(acled_section)

    duckdb_lines = _render_duckdb_section(_ensure_dict(export_summary).get("duckdb"))
    if duckdb_lines:
        lines.append("")
        lines.extend(duckdb_lines)

    downstream_summary_lines = _render_downstream_db_summary(
        _safe_load_json(EXPORT_DB_DIAG_PATH),
        _safe_load_json(FREEZE_DB_DIAG_PATH),
    )
    if downstream_summary_lines:
        lines.append("")
        lines.extend(downstream_summary_lines)

    duckdb_md_lines = _safe_read_markdown(DUCKDB_SUMMARY_PATH)
    if duckdb_md_lines:
        lines.append("")
        lines.append("## DuckDB (from derive-freeze)")
        lines.append("")
        lines.extend(duckdb_md_lines)

    if mapping_debug_records:
        lines.append("")
        lines.append("### Export mapping debug")
        lines.append("")
        acled_note: str | None = None
        for record in mapping_debug_records:
            if not isinstance(record, Mapping):
                continue
            file_path = str(record.get("file") or "")
            if not file_path.endswith("acled.csv"):
                continue
            if record.get("matched"):
                continue
            reasons = record.get("reasons")
            if isinstance(reasons, Mapping) and reasons.get("regex_miss"):
                acled_note = (
                    "- **ACLED export note:** `resolver/staging/acled.csv` exists but no export mapping rule "
                    "matched (`regex_miss`). ACLED data is currently **not** contributing to `facts.csv`."
                )
                break
        if acled_note:
            lines.append(acled_note)
            lines.append("")
        limit = min(10, len(mapping_debug_records))
        for record in mapping_debug_records[:limit]:
            record_map = dict(record)
            file_path = str(record_map.get("file") or "?")
            matched = bool(record_map.get("matched"))
            used_mapping = record_map.get("used_mapping")
            lines.append(f"- `{file_path}`")
            if matched:
                if used_mapping:
                    lines.append(f"  - Matched: yes (`{used_mapping}`)")
                else:
                    lines.append("  - Matched: yes")
            else:
                lines.append("  - Matched: no")
                reasons = record_map.get("reasons")
                reason_parts: List[str] = []
                if isinstance(reasons, Mapping):
                    if reasons.get("regex_miss"):
                        reason_parts.append("regex_miss")
                    missing_cols = reasons.get("missing_columns") or []
                    if missing_cols:
                        cols_joined = ", ".join(str(col) for col in missing_cols)
                        reason_parts.append(f"missing_columns=[{cols_joined}]")
                    missing_any = reasons.get("missing_required_any") or []
                    if missing_any:
                        any_joined = ", ".join(str(group) for group in missing_any)
                        reason_parts.append(f"missing_required_any=[{any_joined}]")
                    extra_keys = [
                        key
                        for key in reasons
                        if key not in {"regex_miss", "missing_columns", "missing_required_any"}
                    ]
                    for key in extra_keys:
                        value = reasons.get(key)
                        reason_parts.append(f"{key}={value}")
                if reason_parts:
                    lines.append(f"  - Reason: {', '.join(reason_parts)}")
                else:
                    lines.append("  - Reason: unknown")
                dedupe_info = record_map.get("dedupe")
                if isinstance(dedupe_info, Mapping):
                    keys = dedupe_info.get("keys") or []
                    keep_value = dedupe_info.get("keep")
                    key_list = [str(key) for key in keys]
                    if key_list or keep_value:
                        keys_display = ", ".join(key_list) if key_list else "∅"
                        keep_display = str(keep_value) if keep_value else "last"
                        lines.append(f"  - Dedupe: keys={keys_display} (keep={keep_display})")
                raw_columns = record_map.get("columns")
                if isinstance(raw_columns, Sequence) and not isinstance(raw_columns, (str, bytes)):
                    columns_list = [str(col) for col in list(raw_columns)]
                elif raw_columns not in (None, ""):
                    columns_list = [str(raw_columns)]
                else:
                    columns_list = []
                first_cols = columns_list[:6]
                columns_display = ", ".join(first_cols) if first_cols else "∅"
                lines.append(f"  - Columns: {columns_display}")
            if len(mapping_debug_records) > limit:
                lines.append(f"(showing {limit} of {len(mapping_debug_records)} files)")
        lines.append("")

    idmc_why_zero = _safe_load_json(IDMC_WHY_ZERO_PATH)
    if idmc_why_zero:
        lines.append("")
        lines.extend(_render_idmc_why_zero(idmc_why_zero))
        lines.append("")

    if idmc_manifest_data or idmc_manifest is not None:
        lines.append("")
        lines.extend(
            _render_idmc_manifest_section(
                idmc_manifest_data,
                why_zero=idmc_why_zero if isinstance(idmc_why_zero, Mapping) else None,
            )
        )

    if dtm_entry:
        config_section = _render_config_section(dtm_entry)
        if config_section:
            lines.extend(config_section)
        date_filter_section = _render_dtm_date_filter(dtm_entry)
        if date_filter_section:
            lines.extend(date_filter_section)
        staging_section = _render_staging_readiness(dtm_entry)
        if staging_section:
            lines.extend(staging_section)
        per_country_section = _render_dtm_per_country(dtm_entry)
        if per_country_section:
            lines.extend(per_country_section)
        sample_section = _render_source_sample_quick_checks()
        if sample_section:
            lines.extend(sample_section)
        zero_rows_section = _render_zero_row_root_cause(dtm_entry)
        if zero_rows_section:
            lines.extend(zero_rows_section)
        selector_section = _render_selector_effectiveness(dtm_entry)
        if selector_section:
            lines.extend(selector_section)

    reachability_section = _render_reachability_section(reachability or {})
    if reachability_section:
        lines.extend(reachability_section)

    idmc_section = _render_idmc_reachability_section(idmc_reachability or {})
    if idmc_section:
        lines.extend(idmc_section)

    hdx_section = _render_hdx_reachability_section(hdx_reachability or {})
    if hdx_section:
        lines.extend(hdx_section)

    lines.append("## Per-Connector Table")
    lines.append("")
    lines.extend(_build_table(sorted_entries, idmc_manifest=idmc_manifest_data))
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
    export_summary = _collect_export_summary(
        Path("resolver/staging"),
        Path("resolver/tools/export_config.yml"),
        Path("diagnostics/ingestion/export_preview"),
    )
    mapping_debug_records = _load_mapping_debug(
        Path("diagnostics/ingestion/export_preview/mapping_debug.jsonl")
    )
    reachability_path = Path("diagnostics/ingestion/dtm/reachability.json")
    reachability_payload = _safe_load_json(reachability_path) or {}
    idmc_reachability_path = Path("diagnostics/ingestion/idmc/probe.json")
    idmc_reachability_payload = _safe_load_json(idmc_reachability_path) or {}
    hdx_reachability_path = Path("diagnostics/ingestion/idmc/hdx_probe.json")
    hdx_reachability_payload = _safe_load_json(hdx_reachability_path) or {}
    idmc_manifest_path = Path("diagnostics/ingestion/idmc/manifest.json")
    idmc_manifest_payload = _safe_load_json(idmc_manifest_path) or {}
    markdown = build_markdown(
        deduped_entries,
        dedupe_notes=dedupe_notes,
        reachability=reachability_payload,
        idmc_reachability=idmc_reachability_payload,
        hdx_reachability=hdx_reachability_payload,
        export_summary=export_summary,
        mapping_debug=mapping_debug_records,
        idmc_manifest=idmc_manifest_payload,
    )
    write_markdown(out_path, markdown)
    if args.github_step_summary:
        append_to_summary(markdown)
    return 0 if load_error is None else 1


def render_summary_md(
    entries: Sequence[Mapping[str, Any]],
    *,
    dedupe_notes: Mapping[str, int] | None = None,
    reachability: Mapping[str, Any] | None = None,
    idmc_reachability: Mapping[str, Any] | None = None,
    hdx_reachability: Mapping[str, Any] | None = None,
    export_summary: Mapping[str, Any] | None = None,
    mapping_debug: Sequence[Mapping[str, Any]] | None = None,
    idmc_manifest: Mapping[str, Any] | None = None,
) -> str:
    """Compatibility alias for callers expecting ``render_summary_md``."""

    return build_markdown(
        entries,
        dedupe_notes=dedupe_notes,
        reachability=reachability,
        idmc_reachability=idmc_reachability,
        hdx_reachability=hdx_reachability,
        export_summary=export_summary,
        mapping_debug=mapping_debug,
        idmc_manifest=idmc_manifest,
    )


if __name__ == "__main__":
    raise SystemExit(main())
