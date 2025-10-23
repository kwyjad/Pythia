"""Render ingestion connector diagnostics into Markdown summaries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

DEFAULT_HTTP_KEYS = ("2xx", "4xx", "5xx", "retries", "rate_limit_remaining", "last_status")
SUMMARY_TITLE = "# Connector Diagnostics"
MISSING_REPORT_SUMMARY = (
    "# Ingestion Diagnostics\n\n"
    "**No connectors report was produced.**  \n"
    "This usually means the ingestion step failed early (e.g., setup or backfill window).  \n"
    "Check earlier steps in the job log.\n"
)


def _ensure_dict(data: Any) -> Dict[str, Any]:
    return dict(data) if isinstance(data, Mapping) else {}


def _coerce_int(value: Any) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


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

    return {
        "connector_id": str(entry.get("connector_id") or entry.get("name") or "unknown"),
        "mode": str(entry.get("mode") or "real"),
        "status": str(entry.get("status") or "skipped"),
        "reason": _clean_reason(entry.get("reason")),
        "started_at_utc": str(entry.get("started_at_utc") or ""),
        "duration_ms": _coerce_int(entry.get("duration_ms")),
        "http": http,
        "counts": counts,
        "coverage": coverage,
        "samples": samples,
        "extras": extras,
    }


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


def _format_status_counts(entries: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(str(entry.get("status") or "") for entry in entries)
    if not counts:
        return "none"
    parts = [f"{status}={count}" for status, count in sorted(counts.items()) if status]
    return ", ".join(parts) if parts else "none"


def _format_reason_counts(entries: Sequence[Mapping[str, Any]]) -> str:
    counter: Counter[str] = Counter()
    for entry in entries:
        reason = entry.get("reason")
        if reason:
            counter[str(reason)] += 1
    if not counter:
        return "none"
    parts = [f"{reason}={count}" for reason, count in sorted(counter.items())]
    return ", ".join(parts)


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
    extras_line = _format_extras(entry.get("extras", {}))
    if extras_line:
        bullets.append(extras_line)
    if not bullets:
        bullets.append("- _No additional diagnostics recorded._")
    lines.extend(bullets)
    lines.append("")
    lines.append("</details>")
    return "\n".join(lines)


def _build_table(entries: Sequence[Mapping[str, Any]]) -> List[str]:
    headers = [
        "Connector",
        "Mode",
        "Status",
        "Reason",
        "Duration",
        "2xx/4xx/5xx (retries)",
        "Rows (f/n/w)",
        "Coverage (ym)",
        "Coverage (as_of)",
        "Logs",
    ]
    logs_dir = Path("diagnostics/ingestion/logs")
    rows: List[List[str]] = []
    for entry in entries:
        coverage = entry.get("coverage", {})
        connector_id = str(entry.get("connector_id"))
        log_path = logs_dir / f"{connector_id}.log"
        log_cell = str(log_path) if log_path.exists() else "—"
        rows.append(
            [
                connector_id,
                str(entry.get("mode")),
                str(entry.get("status")),
                _format_reason(entry.get("reason")),
                _format_duration(entry.get("duration_ms", 0)),
                _format_http(entry.get("http", {})),
                _format_rows(entry.get("counts", {})),
                _format_coverage(coverage.get("ym_min"), coverage.get("ym_max")),
                _format_coverage(coverage.get("as_of_min"), coverage.get("as_of_max")),
                log_cell,
            ]
        )
    if not rows:
        rows.append(["—"] * len(headers))
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_markdown(entries: Sequence[Mapping[str, Any]]) -> str:
    sorted_entries = sorted(entries, key=lambda item: str(item.get("connector_id", "")))
    total_fetched = sum(entry.get("counts", {}).get("fetched", 0) for entry in sorted_entries)
    total_written = sum(entry.get("counts", {}).get("written", 0) for entry in sorted_entries)

    lines = [SUMMARY_TITLE, "", "## Run Summary", ""]
    lines.append(f"* **Connectors:** {len(sorted_entries)}")
    lines.append(f"* **Status counts:** {_format_status_counts(sorted_entries)}")
    lines.append(f"* **Reason histogram:** {_format_reason_counts(sorted_entries)}")
    lines.append(f"* **Rows fetched:** {total_fetched}")
    lines.append(f"* **Rows written:** {total_written}")
    lines.append("")
    lines.append("## Per-Connector Table")
    lines.append("")
    lines.extend(_build_table(sorted_entries))
    lines.append("")
    for entry in sorted_entries:
        lines.append(_render_details(entry))
        lines.append("")
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
        write_markdown(out_path, MISSING_REPORT_SUMMARY)
        if args.github_step_summary:
            append_to_summary(MISSING_REPORT_SUMMARY)
        return 0
    try:
        entries = load_report(report_path)
    except Exception as exc:
        print(f"summarize_connectors: {exc}", file=sys.stderr)
        return 1
    markdown = build_markdown(entries)
    write_markdown(out_path, markdown)
    if args.github_step_summary:
        append_to_summary(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
