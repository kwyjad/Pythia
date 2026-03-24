# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Consolidated multi-phase pipeline summary with CrisisWatch inject check.

Reads ``connectors_report.jsonl`` (populated by all ingestion phases),
groups entries by phase, and produces a Markdown summary for the
GitHub Step Summary.

Usage:
    python -m pythia.tools.summarize_all_phases --report diagnostics/ingestion/connectors_report.jsonl
    python -m pythia.tools.summarize_all_phases --report path/to/report.jsonl --db data/resolver.duckdb
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Sequence

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Phase mapping — assigns each connector_id to a phase
# ---------------------------------------------------------------------------

PHASE_MAP: dict[str, tuple[int, str]] = {
    # Phase 1 — Ground truth (resolver ingestion framework)
    "acled_client": (1, "Ground Truth"),
    "ifrc_go_client": (1, "Ground Truth"),
    "idmc": (1, "Ground Truth"),
    # Phase 2 — Resolution sources (facts_resolved via run_pipeline)
    "gdacs_population_exposed": (2, "Resolution Sources"),
    "fewsnet_ipc_population": (2, "Resolution Sources"),
    # Phase 3 — Structured data (Pythia tables)
    "acaps_inform_severity": (3, "Structured Data"),
    "acaps_risk_radar": (3, "Structured Data"),
    "acaps_daily_monitoring": (3, "Structured Data"),
    "acaps_humanitarian_access": (3, "Structured Data"),
    "views_forecasts": (3, "Structured Data"),
    "conflictforecast_forecasts": (3, "Structured Data"),
    "acledcast_forecasts": (3, "Structured Data"),
    "ipc_phases": (3, "Structured Data"),
    "reliefweb_reports": (3, "Structured Data"),
    "acled_political_events": (3, "Structured Data"),
    "nmme_seasonal_forecasts": (3, "Structured Data"),
    # Phase 4 — Context sources
    "enso": (4, "Context Sources"),
    "seasonal_tc": (4, "Context Sources"),
    "hdx_signals": (4, "Context Sources"),
    "crisiswatch": (4, "Context Sources"),
}

_STATUS_EMOJI = {
    "ok": "\u2705",
    "error": "\u274c",
    "skipped": "\u23ed\ufe0f",
}


def _fmt_duration(duration_ms: int) -> str:
    if not duration_ms:
        return "\u2014"
    return f"{duration_ms / 1000:.1f}s"


def _status_icon(status: str) -> str:
    return _STATUS_EMOJI.get(status, status)


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------


def _load_jsonl(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    entries = []
    for line in p.read_text(encoding="utf-8").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            LOG.warning("Skipping invalid JSONL line: %s", line[:120])
    return entries


# ---------------------------------------------------------------------------
# CrisisWatch inject check
# ---------------------------------------------------------------------------


def _check_crisiswatch_inject(db_path: str | None = None) -> dict:
    """Check CrisisWatch inject status in DuckDB.

    Returns a dict with:
      - table_exists: bool
      - row_count: int
      - latest_date: str | None  (ISO date of most recent entry)
      - freshness: str  ("current", "stale", "empty", "missing")
    """
    if db_path is None:
        db_path = (
            os.environ.get("PYTHIA_DB_URL")
            or os.environ.get("BACKFILL_DB_PATH")
            or "data/resolver.duckdb"
        )
    # Strip duckdb:/// prefix if present
    if db_path.startswith("duckdb:///"):
        db_path = db_path[len("duckdb:///"):]

    result: dict[str, Any] = {
        "table_exists": False,
        "row_count": 0,
        "latest_date": None,
        "freshness": "missing",
    }

    try:
        import duckdb
    except ImportError:
        LOG.warning("duckdb not installed; CrisisWatch check skipped")
        result["freshness"] = "unknown"
        return result

    try:
        con = duckdb.connect(db_path, read_only=True)
    except Exception as exc:
        LOG.warning("Cannot open DuckDB at %s: %s", db_path, exc)
        result["freshness"] = "unknown"
        return result

    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name = 'crisiswatch_entries'"
            ).fetchall()
        ]
        if not tables:
            return result

        result["table_exists"] = True
        row = con.execute(
            "SELECT COUNT(*) AS cnt, MAX(year * 100 + month) AS latest_ym FROM crisiswatch_entries"
        ).fetchone()

        if row is None or row[0] == 0:
            result["freshness"] = "empty"
            return result

        result["row_count"] = row[0]
        latest_ym = row[1]
        if latest_ym:
            latest_year = latest_ym // 100
            latest_month = latest_ym % 100
            result["latest_date"] = f"{latest_year}-{latest_month:02d}-01"

            today = date.today()
            months_old = (today.year - latest_year) * 12 + (today.month - latest_month)
            if months_old <= 2:
                result["freshness"] = "current"
                result["days_old"] = months_old * 30  # approximate
            else:
                result["freshness"] = "stale"
                result["days_old"] = months_old * 30
        else:
            result["freshness"] = "empty"
    except Exception as exc:
        LOG.warning("CrisisWatch check query failed: %s", exc)
        result["freshness"] = "unknown"
    finally:
        try:
            con.close()
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Phase summary builder
# ---------------------------------------------------------------------------


def build_phase_summary(
    entries: list[dict],
    crisiswatch_status: dict | None = None,
) -> str:
    """Build a Markdown summary grouped by phase."""
    if not entries:
        return "\u26a0\ufe0f No connector diagnostics found. Check DIAGNOSTICS_REPORT_PATH env var.\n"

    # Group by phase
    by_phase: dict[int, list[dict]] = {}
    other: list[dict] = []
    for e in entries:
        cid = e.get("connector_id", "")
        phase_info = PHASE_MAP.get(cid)
        if phase_info:
            phase_num = phase_info[0]
            by_phase.setdefault(phase_num, []).append(e)
        else:
            other.append(e)

    # Overall status line
    total = len(entries)
    ok_count = sum(1 for e in entries if e.get("status") == "ok")
    error_count = sum(1 for e in entries if e.get("status") == "error")
    skipped_count = sum(1 for e in entries if e.get("status") == "skipped")
    lines = [
        f"**Overall: {ok_count}/{total} sources OK"
        + (f", {error_count} error" if error_count else "")
        + (f", {skipped_count} skipped" if skipped_count else "")
        + "**\n",
    ]

    phase_names = {
        1: "Phase 1: Ground Truth",
        2: "Phase 2: Resolution Sources",
        3: "Phase 3: Structured Data",
        4: "Phase 4: Context Sources",
    }

    for phase_num in sorted(by_phase.keys()):
        phase_entries = by_phase[phase_num]
        label = phase_names.get(phase_num, f"Phase {phase_num}")
        lines.append(f"\n### {label}\n")

        if phase_num == 1:
            lines.append("| Connector | Status | Reason | Rows (f/n/w) | Duration |")
            lines.append("|-----------|--------|--------|-------------|----------|")
            for e in phase_entries:
                cid = e.get("connector_id", "?")
                st = _status_icon(e.get("status", "?"))
                reason = e.get("reason") or "\u2014"
                counts = e.get("counts", {})
                fetched = counts.get("fetched", "\u2014")
                normalized = counts.get("normalized", "\u2014")
                written = counts.get("written", "\u2014")
                dur = _fmt_duration(e.get("duration_ms", 0))
                lines.append(f"| {cid} | {st} | {reason} | {fetched}/{normalized}/{written} | {dur} |")

        elif phase_num == 2:
            lines.append("| Source | Status | Facts | Resolved | Deltas | Duration |")
            lines.append("|--------|--------|-------|----------|--------|----------|")
            for e in phase_entries:
                cid = e.get("connector_id", "?")
                st = _status_icon(e.get("status", "?"))
                extras = e.get("extras", {})
                facts = extras.get("total_facts", "\u2014")
                resolved = extras.get("resolved_rows", "\u2014")
                deltas = extras.get("delta_rows", "\u2014")
                dur = _fmt_duration(e.get("duration_ms", 0))
                lines.append(f"| {cid} | {st} | {facts} | {resolved} | {deltas} | {dur} |")

        elif phase_num == 3:
            lines.append("| Source | Status | Countries Fetched | Written | Failed | Duration |")
            lines.append("|--------|--------|------------------|---------|--------|----------|")
            for e in phase_entries:
                cid = e.get("connector_id", "?")
                st = _status_icon(e.get("status", "?"))
                counts = e.get("counts", {})
                fetched = counts.get("fetched", "\u2014")
                written = counts.get("written", "\u2014")
                failed = counts.get("failed", "\u2014")
                dur = _fmt_duration(e.get("duration_ms", 0))
                lines.append(f"| {cid} | {st} | {fetched} | {written} | {failed} | {dur} |")

        elif phase_num == 4:
            lines.append("| Source | Status | Rows | Duration |")
            lines.append("|--------|--------|------|----------|")
            for e in phase_entries:
                cid = e.get("connector_id", "?")
                st = _status_icon(e.get("status", "?"))
                counts = e.get("counts", {})
                written = counts.get("written", "\u2014")
                dur = _fmt_duration(e.get("duration_ms", 0))
                lines.append(f"| {cid} | {st} | {written} | {dur} |")

    # Other (unknown phase)
    if other:
        lines.append("\n### Other\n")
        lines.append("| Source | Status | Rows Written | Duration |")
        lines.append("|--------|--------|-------------|----------|")
        for e in other:
            cid = e.get("connector_id", "?")
            st = _status_icon(e.get("status", "?"))
            counts = e.get("counts", {})
            written = counts.get("written", "\u2014")
            dur = _fmt_duration(e.get("duration_ms", 0))
            lines.append(f"| {cid} | {st} | {written} | {dur} |")

    # CrisisWatch inject status
    lines.append("\n### ICG CrisisWatch Inject Status\n")
    if crisiswatch_status:
        freshness = crisiswatch_status.get("freshness", "unknown")
        if freshness == "missing":
            lines.append("- **Table**: crisiswatch_entries \u2014 \u274c NOT FOUND")
            lines.append("- **Action needed**: Run CrisisWatch local ingest (launchd job) or check `horizon_scanner/data/crisiswatch_latest.json`")
        elif freshness == "empty":
            lines.append("- **Table**: crisiswatch_entries \u2014 \u26a0\ufe0f EMPTY")
            lines.append("- **Action needed**: Run CrisisWatch ingest")
        elif freshness == "unknown":
            lines.append("- CrisisWatch status: unknown (DuckDB not available)")
        else:
            row_count = crisiswatch_status.get("row_count", 0)
            latest = crisiswatch_status.get("latest_date", "?")
            days_old = crisiswatch_status.get("days_old", "?")
            icon = "\u2705" if freshness == "current" else "\u26a0\ufe0f"
            lines.append(f"- **Table**: crisiswatch_entries")
            lines.append(f"- **Rows**: {row_count}")
            lines.append(f"- **Latest entry**: {latest}")
            lines.append(f"- **Freshness**: {icon} {freshness} ({days_old} days old)")
    else:
        lines.append("- CrisisWatch status: unknown (check not run)")

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Produce a consolidated multi-phase pipeline summary",
    )
    parser.add_argument(
        "--report",
        default="diagnostics/ingestion/connectors_report.jsonl",
        help="Path to the connectors_report.jsonl file",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to resolver.duckdb for CrisisWatch check (auto-detected if not set)",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "github-step-summary"],
        default="markdown",
        help="Output format",
    )
    args = parser.parse_args(argv)

    entries = _load_jsonl(args.report)
    cw_status = _check_crisiswatch_inject(args.db)
    summary = build_phase_summary(entries, crisiswatch_status=cw_status)

    print(summary)


if __name__ == "__main__":
    main()
