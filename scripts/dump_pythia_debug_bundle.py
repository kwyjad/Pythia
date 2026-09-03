# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import argparse
import csv
import gzip
import io
import hashlib
import json
import logging
import math
import os
import statistics as _statistics
import re
import zipfile
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple

import duckdb
import pandas as pd
from forecaster.providers import SPD_ENSEMBLE, estimate_cost_usd, parse_ensemble_specs
from horizon_scanner.llm_logging import (
    GROUNDING_BREAKER_MODEL_ID,
    GROUNDING_FAILED_MODEL_ID,
    GROUNDING_UNAVAILABLE_MODEL_ID,
)
from pythia.db import schema as pythia_schema
from resolver.db import duckdb_io
from scripts.ci.llm_latency_summary import render_latency_markdown
from scripts.debug_bundle import (
    anomalies as anomalies_mod,
    batch_lifecycle,
    code_snapshot,
    connector_freshness,
    env_config,
    manifest as manifest_mod,
    model_completeness,
    prompt_cache,
    prompt_prefixes,
    provider_objects,
    retry_report,
    run_comparison,
    workflow_logs,
)

try:
    from forecaster.prompts import _bucket_labels_for_question  # type: ignore
except ImportError:  # pragma: no cover - optional helper
    _bucket_labels_for_question = None

LOG = logging.getLogger(__name__)


def _fetch_llm_rows(
    con: duckdb.DuckDBPyConnection,
    query: str,
    params: list[Any],
) -> list[dict[str, Any]]:
    cur = con.execute(query, params)
    rows = cur.fetchall()
    desc = cur.description or []
    col_names = [d[0] for d in desc]
    return [dict(zip(col_names, row)) for row in rows]


def _llm_calls_columns(con: duckdb.DuckDBPyConnection) -> Set[str]:
    try:
        rows = con.execute("PRAGMA table_info('llm_calls')").fetchall()
    except Exception:
        return set()

    cols: Set[str] = set()
    for row in rows:
        if len(row) > 1 and row[1]:
            cols.add(str(row[1]))
    return cols


def _build_in_clause(values: list[str]) -> tuple[str, list[str]]:
    cleaned = [str(v) for v in values if str(v)]
    if not cleaned:
        return "", []
    placeholders = ", ".join(["?"] * len(cleaned))
    return f"({placeholders})", cleaned


KEY_TABLES = [
    "facts_resolved",
    "facts_deltas",
    "snapshots",
    "hs_triage",
    "questions",
    "forecasts_raw",
    "forecasts_ensemble",
]
EXPECTED_HS_HAZARDS = ["ACE", "DR", "FL", "TC"]


def _table_count(con: duckdb.DuckDBPyConnection, table: str) -> int | None:
    try:
        row = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else None
    except Exception:
        return None


def _row_counts(con: duckdb.DuckDBPyConnection, tables: list[str] | None = None) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for tbl in tables or KEY_TABLES:
        counts[tbl] = _table_count(con, tbl)
    return counts


def _file_stats(db_path: str) -> dict[str, Any]:
    stats: dict[str, Any] = {"sha256": None, "size_bytes": None, "path": db_path}
    try:
        p = Path(db_path)
        if p.exists() and p.is_file():
            stats["size_bytes"] = p.stat().st_size
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            stats["sha256"] = h.hexdigest()
    except Exception:
        return stats
    return stats


def _load_counts_before(path: str | None) -> dict[str, int | None]:
    """Row counts as they were before this phase wrote anything.

    They were hardcoded to None, so the provenance table's "before" column
    was empty on every run and its delta column was empty with it — the one
    view that says what this phase actually added said nothing at all. The
    counts already exist: every stage of pythia_pipeline_stage.yml writes a
    db_signature BEFORE doing any work. This reads that file.

    Missing or unreadable means None, exactly as before: an absent baseline
    must never be published as a zero, which would turn every row into a
    fabricated gain.
    """

    counts: dict[str, int | None] = {tbl: None for tbl in KEY_TABLES}
    if not path:
        return counts
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return counts
    merged: dict[str, Any] = {}
    for key in ("required_counts", "optional_counts"):
        section = payload.get(key)
        if isinstance(section, dict):
            merged.update(section)
    for tbl in KEY_TABLES:
        value = merged.get(tbl)
        if value is not None:
            try:
                counts[tbl] = int(value)
            except Exception:
                counts[tbl] = None
    return counts


def _record_run_provenance(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str | None,
    forecaster_run_id: str | None,
    hs_run_id: str | None,
    artifact_run_id: str | None,
    artifact_workflow: str | None,
    artifact_name: str | None,
    db_stats: dict[str, Any],
    counts_before: dict[str, int | None],
    counts_after: dict[str, int | None],
) -> dict[str, Any]:
    pythia_schema.ensure_schema(con)
    con.execute(
        "DELETE FROM run_provenance WHERE forecaster_run_id = ? AND hs_run_id = ?;",
        [forecaster_run_id, hs_run_id],
    )
    con.execute(
        """
        INSERT INTO run_provenance (
            run_id, hs_run_id, forecaster_run_id, artifact_run_id, artifact_workflow, artifact_name,
            db_sha256, db_size_bytes,
            facts_resolved_before, facts_resolved_after,
            facts_deltas_before, facts_deltas_after,
            snapshots_before, snapshots_after,
            hs_triage_before, hs_triage_after,
            questions_before, questions_after,
            forecasts_raw_before, forecasts_raw_after,
            forecasts_ensemble_before, forecasts_ensemble_after
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        [
            run_id,
            hs_run_id,
            forecaster_run_id,
            artifact_run_id,
            artifact_workflow,
            artifact_name,
            db_stats.get("sha256"),
            db_stats.get("size_bytes"),
            counts_before.get("facts_resolved"),
            counts_after.get("facts_resolved"),
            counts_before.get("facts_deltas"),
            counts_after.get("facts_deltas"),
            counts_before.get("snapshots"),
            counts_after.get("snapshots"),
            counts_before.get("hs_triage"),
            counts_after.get("hs_triage"),
            counts_before.get("questions"),
            counts_after.get("questions"),
            counts_before.get("forecasts_raw"),
            counts_after.get("forecasts_raw"),
            counts_before.get("forecasts_ensemble"),
            counts_after.get("forecasts_ensemble"),
        ],
    )
    return {
        "run_id": run_id,
        "hs_run_id": hs_run_id,
        "forecaster_run_id": forecaster_run_id,
        "artifact_run_id": artifact_run_id,
        "artifact_workflow": artifact_workflow,
        "artifact_name": artifact_name,
        "db_sha256": db_stats.get("sha256"),
        "db_size_bytes": db_stats.get("size_bytes"),
    }


def _provenance_markdown(
    provenance_entry: dict[str, Any],
    counts_before: dict[str, int | None],
    counts_after: dict[str, int | None],
    db_stats: dict[str, Any],
) -> List[str]:
    def _signature_section(path: str, label: str) -> list[str]:
        p = Path(path)
        if not p.exists():
            return [f"- {label}: missing ({path})"]
        try:
            content = p.read_text(encoding="utf-8")
        except Exception as exc:  # pragma: no cover - defensive
            return [f"- {label}: unable to read {path}: {exc}"]
        return [f"- {label}: {path}", "```json", content, "```"]

    lines: List[str] = []
    lines.append("### DB provenance and rowcounts")
    lines.append("")
    lines.append(f"- Artifact run id: `{provenance_entry.get('artifact_run_id') or '(unknown)'}`")
    lines.append(f"- Artifact workflow: `{provenance_entry.get('artifact_workflow') or '(unknown)'}`")
    lines.append(f"- Artifact name: `{provenance_entry.get('artifact_name') or '(unknown)'}`")
    lines.append(f"- DB path: `{db_stats.get('path')}`")
    lines.append(f"- DB size (bytes): `{provenance_entry.get('db_size_bytes')}`")
    lines.append(f"- DB sha256: `{provenance_entry.get('db_sha256')}`")
    lines.append("")
    lines.append("| table | before | after |")
    lines.append("| ----- | ------ | ----- |")
    for tbl in KEY_TABLES:
        before = counts_before.get(tbl)
        after = counts_after.get(tbl)
        lines.append(f"| {tbl} | {before if before is not None else 'n/a'} | {after if after is not None else 'n/a'} |")
    lines.append("")
    lines.append("#### Signature files")
    lines.extend(_signature_section("diagnostics/db_signature_before.json", "db_signature_before"))
    lines.extend(_signature_section("diagnostics/db_signature_after.json", "db_signature_after"))
    lines.append("")
    return lines


def _load_self_search_stats(
    con: duckdb.DuckDBPyConnection,
    predicate: str | None,
    params: list[Any],
    phase: str = "forecast_web_research",
) -> dict[str, int]:
    sql = "SELECT response_text FROM llm_calls WHERE phase = ?"
    query_params: list[Any] = [phase]
    if predicate:
        sql += f" AND ({predicate})"
        query_params.extend(params)
    rows = con.execute(sql, query_params).fetchall()
    requests = len(rows)
    sources = 0
    for row in rows:
        resp = None
        if isinstance(row, tuple):
            resp = row[0]
        elif isinstance(row, list):
            resp = row[0] if row else None
        elif isinstance(row, dict):
            resp = row.get("response_text")
        try:
            data = json.loads(resp or "{}")
            srcs = data.get("sources") or []
            if isinstance(srcs, list):
                sources += len(srcs)
        except Exception:
            continue
    return {"requests": requests, "sources": sources}


def _load_web_research_summary(
    con: duckdb.DuckDBPyConnection,
    phase: str,
    forecaster_run_id: str | None,
    hs_run_id: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    predicate_parts: list[str] = ["phase = ?"]
    params: list[Any] = []
    params.append(phase)
    scope_parts: list[str] = []
    if forecaster_run_id:
        scope_parts.append("run_id = ?")
        params.append(forecaster_run_id)
    if hs_run_id:
        scope_parts.append("hs_run_id = ?")
        params.append(hs_run_id)
    if scope_parts:
        predicate_parts.append(f"({' OR '.join(scope_parts)})")
    predicate = " AND ".join(predicate_parts)

    try:
        rows = _fetch_llm_rows(
            con,
            f"""
            SELECT provider, model_id, response_text, error_text
            FROM llm_calls
            WHERE {predicate}
            """,
            params,
        )
    except Exception:
        return [], []

    counts: dict[tuple[str, str], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for row in rows:
        provider = str(row.get("provider") or "")
        model_id = str(row.get("model_id") or "")
        key = (provider, model_id)
        entry = counts.setdefault(
            key,
            {"provider": provider, "model_id": model_id, "n_calls": 0, "n_errors": 0, "n_verified_sources": 0},
        )
        entry["n_calls"] += 1

        error_text = (row.get("error_text") or "").strip()
        response_text = row.get("response_text") or "{}"
        error_code = ""
        error_message = ""
        try:
            payload = json.loads(response_text)
        except Exception:
            payload = {}

        sources = payload.get("sources") if isinstance(payload, dict) else None
        if isinstance(sources, list):
            entry["n_verified_sources"] += len(sources)

        error_obj = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error_obj, dict):
            error_code = str(error_obj.get("code") or "")
            error_message = str(error_obj.get("message") or "")

        if error_text or error_code or error_message:
            entry["n_errors"] += 1
            failures.append(
                {
                    "provider": provider,
                    "model_id": model_id,
                    "error_code": error_code,
                    "error_message": error_text or error_message,
                }
            )

    summary_rows = sorted(
        counts.values(),
        key=lambda r: (str(r.get("provider") or ""), str(r.get("model_id") or "")),
    )
    failure_rows = sorted(
        failures,
        key=lambda r: (
            str(r.get("provider") or ""),
            str(r.get("model_id") or ""),
            str(r.get("error_code") or ""),
            str(r.get("error_message") or ""),
        ),
    )
    return summary_rows, failure_rows


def _load_grounding_subsystem_stats(
    con: duckdb.DuckDBPyConnection,
    hazard_code_filter: str,
    forecaster_run_id: str | None,
    hs_run_id: str | None,
) -> list[dict[str, Any]]:
    """Load grounding call stats from llm_calls for RC or Triage grounding.

    *hazard_code_filter* is a SQL LIKE pattern applied (case-insensitive) to the
    ``hazard_code`` column, e.g. ``'GROUNDING_%'`` for RC grounding or
    ``'TRIAGE_GROUNDING_%'`` for triage grounding.

    Returns a list of summary dicts with keys:
        provider, model_id, n_calls, n_errors, n_verified_sources
    """
    predicate_parts: list[str] = [
        "phase = 'hs_triage'",
        "UPPER(hazard_code) LIKE ?",
    ]
    params: list[Any] = [hazard_code_filter]

    # RC grounding rows must exclude triage grounding rows when using 'GROUNDING_%'
    if hazard_code_filter == "GROUNDING_%":
        predicate_parts.append("UPPER(hazard_code) NOT LIKE 'TRIAGE_GROUNDING_%'")
    # Adversarial SEARCH stats must exclude the synthesis LLM rows
    # (ADVERSARIAL_SYNTH_%): those are generation calls with no sources and
    # would read as failed/empty grounding (false WARN, July 2026 run 2).
    if hazard_code_filter == "ADVERSARIAL_%":
        predicate_parts.append("UPPER(hazard_code) NOT LIKE 'ADVERSARIAL_SYNTH_%'")

    scope_parts: list[str] = []
    if forecaster_run_id:
        scope_parts.append("run_id = ?")
        params.append(forecaster_run_id)
    if hs_run_id:
        scope_parts.append("hs_run_id = ?")
        params.append(hs_run_id)
    if scope_parts:
        predicate_parts.append(f"({' OR '.join(scope_parts)})")
    predicate = " AND ".join(predicate_parts)

    try:
        rows = _fetch_llm_rows(
            con,
            f"""
            SELECT provider, model_id, response_text, error_text
            FROM llm_calls
            WHERE {predicate}
            """,
            params,
        )
    except Exception:
        return []

    counts: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        provider = str(row.get("provider") or "")
        model_id = str(row.get("model_id") or "")
        key = (provider, model_id)
        entry = counts.setdefault(
            key,
            {"provider": provider, "model_id": model_id, "n_calls": 0, "n_errors": 0, "n_verified_sources": 0},
        )
        entry["n_calls"] += 1

        error_text = (row.get("error_text") or "").strip()
        response_text = row.get("response_text") or "{}"
        try:
            payload = json.loads(response_text)
        except Exception:
            payload = {}

        sources = payload.get("sources") if isinstance(payload, dict) else None
        n_sources = 0
        if isinstance(sources, list):
            # Old format: full sources list
            n_sources = len(sources)
        elif isinstance(payload, dict) and isinstance(payload.get("n_sources"), (int, float)):
            # Compact format: pre-computed n_sources count
            n_sources = int(payload["n_sources"])
        elif isinstance(payload, dict) and isinstance(payload.get("source_urls"), list):
            # Compact format variant: source_urls list
            n_sources = len(payload["source_urls"])
        elif response_text and "Sources:" in response_text:
            # Markdown evidence pack — count URL lines
            after_sources = response_text.split("Sources:")[-1]
            n_sources = sum(1 for line in after_sources.splitlines() if line.strip().startswith("- http"))
        entry["n_verified_sources"] += n_sources

        # Count as error if error_text present OR response had 0 sources
        if error_text or n_sources == 0:
            entry["n_errors"] += 1

    return sorted(
        counts.values(),
        key=lambda r: (str(r.get("provider") or ""), str(r.get("model_id") or "")),
    )


def _compute_source_stats(counts: list[int]) -> dict[str, float]:
    """Return min/max/avg/median from a list of per-call source counts."""
    if not counts:
        return {"min": 0, "max": 0, "avg": 0.0, "median": 0}
    return {
        "min": min(counts),
        "max": max(counts),
        "avg": sum(counts) / len(counts),
        "median": _statistics.median(counts),
    }


def _load_grounding_call_stats(
    con: duckdb.DuckDBPyConnection,
    phase: str,
    hazard_code_filter: str | None,
    forecaster_run_id: str | None,
    hs_run_id: str | None,
) -> dict[str, Any]:
    """Load per-call grounding stats.  Returns {n_calls, n_errors, source_counts}.

    *phase* is the ``phase`` column value (``'hs_triage'`` or ``'hs_web_research'``).
    *hazard_code_filter* is an optional SQL LIKE pattern for ``hazard_code``.
    """
    predicate_parts: list[str] = [f"phase = ?"]
    params: list[Any] = [phase]

    if hazard_code_filter:
        predicate_parts.append("UPPER(hazard_code) LIKE ?")
        params.append(hazard_code_filter)
        # Exclude triage grounding when matching generic 'GROUNDING_%'
        if hazard_code_filter == "GROUNDING_%":
            predicate_parts.append("UPPER(hazard_code) NOT LIKE 'TRIAGE_GROUNDING_%'")
        # Exclude synthesis generation rows from adversarial SEARCH stats.
        if hazard_code_filter == "ADVERSARIAL_%":
            predicate_parts.append("UPPER(hazard_code) NOT LIKE 'ADVERSARIAL_SYNTH_%'")

    scope_parts: list[str] = []
    if forecaster_run_id:
        scope_parts.append("run_id = ?")
        params.append(forecaster_run_id)
    if hs_run_id:
        scope_parts.append("hs_run_id = ?")
        params.append(hs_run_id)
    if scope_parts:
        predicate_parts.append(f"({' OR '.join(scope_parts)})")
    predicate = " AND ".join(predicate_parts)

    try:
        rows = _fetch_llm_rows(
            con,
            f"SELECT response_text, error_text FROM llm_calls WHERE {predicate}",
            params,
        )
    except Exception:
        return {"n_calls": 0, "n_errors": 0, "source_counts": []}

    n_calls = 0
    n_errors = 0
    source_counts: list[int] = []

    for row in rows:
        n_calls += 1
        error_text = (row.get("error_text") or "").strip()
        response_text = row.get("response_text") or "{}"
        try:
            payload = json.loads(response_text)
        except Exception:
            payload = {}

        sources = payload.get("sources") if isinstance(payload, dict) else None
        n_sources = 0
        if isinstance(sources, list):
            # Old format: full sources list
            n_sources = len(sources)
        elif isinstance(payload, dict) and isinstance(payload.get("n_sources"), (int, float)):
            # Compact format: pre-computed n_sources count
            n_sources = int(payload["n_sources"])
        elif isinstance(payload, dict) and isinstance(payload.get("source_urls"), list):
            # Compact format variant: source_urls list
            n_sources = len(payload["source_urls"])
        elif response_text and "Sources:" in response_text:
            # Markdown evidence pack — count URL lines
            after_sources = response_text.split("Sources:")[-1]
            n_sources = sum(1 for line in after_sources.splitlines() if line.strip().startswith("- http"))
        source_counts.append(n_sources)

        if error_text or n_sources == 0:
            n_errors += 1

    return {"n_calls": n_calls, "n_errors": n_errors, "source_counts": source_counts}


def _load_adversarial_synth_stats(
    con: duckdb.DuckDBPyConnection,
    forecaster_run_id: str | None,
    hs_run_id: str | None,
) -> dict[str, Any]:
    """Stats for the adversarial synthesis LLM calls (ADVERSARIAL_SYNTH_%).

    These are generation calls (devil's-advocate synthesis), not searches —
    reported separately so they never pollute the grounding source stats.
    Returns {n_calls, n_errors, cost_usd}.
    """
    predicate_parts = [
        "phase = 'hs_triage'",
        "UPPER(hazard_code) LIKE 'ADVERSARIAL_SYNTH_%'",
    ]
    params: list[Any] = []
    scope_parts: list[str] = []
    if forecaster_run_id:
        scope_parts.append("run_id = ?")
        params.append(forecaster_run_id)
    if hs_run_id:
        scope_parts.append("hs_run_id = ?")
        params.append(hs_run_id)
    if scope_parts:
        predicate_parts.append(f"({' OR '.join(scope_parts)})")
    predicate = " AND ".join(predicate_parts)

    try:
        row = con.execute(
            f"""
            SELECT COUNT(*),
                   COALESCE(SUM(CASE WHEN COALESCE(error_text, '') <> '' THEN 1 ELSE 0 END), 0),
                   COALESCE(SUM(COALESCE(cost_usd, 0)), 0)
            FROM llm_calls
            WHERE {predicate}
            """,
            params,
        ).fetchone()
    except Exception:
        return {"n_calls": 0, "n_errors": 0, "cost_usd": 0.0}
    if not row:
        return {"n_calls": 0, "n_errors": 0, "cost_usd": 0.0}
    return {
        "n_calls": int(row[0] or 0),
        "n_errors": int(row[1] or 0),
        "cost_usd": float(row[2] or 0.0),
    }


def _extract_urls(sources: Any, limit: int = 5) -> list[str]:
    urls: list[str] = []
    if not sources:
        return urls
    for src in sources:
        url = None
        if isinstance(src, dict):
            url = src.get("url")
        else:
            url = src
        url = (url or "").strip()
        if url.startswith("http") and url not in urls:
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def _format_attempted_backends(attempts: Any) -> str:
    if not attempts:
        return "(none)"
    formatted: list[str] = []
    if isinstance(attempts, list):
        for item in attempts:
            if isinstance(item, dict):
                backend = (item.get("backend") or "(unknown)").strip()
                grounded = bool(item.get("grounded") or item.get("n_sources"))
                error_obj = item.get("error")
                error_type = ""
                if isinstance(error_obj, dict):
                    error_type = str(error_obj.get("type") or "")
                formatted.append(f"{backend} → {grounded} → {error_type or 'ok'}")
            else:
                formatted.append(str(item))
    else:
        formatted.append(str(attempts))
    return "; ".join(formatted)


def _load_web_research_summaries(
    con: duckdb.DuckDBPyConnection,
    hs_run_id: str | None,
    forecaster_run_id: str | None,
    iso3s: list[str],
    question_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hs_rows: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []

    if hs_run_id and iso3s:
        clause, params = _build_in_clause([iso3.upper() for iso3 in iso3s])
        if clause:
            try:
                hs_raw = con.execute(
                    f"""
                    SELECT iso3, grounded, sources_json, grounding_debug_json
                    FROM hs_country_reports
                    WHERE hs_run_id = ?
                      AND upper(iso3) IN {clause}
                    """,
                    [hs_run_id, *params],
                ).fetchall()
                for iso3, grounded, sources_json, debug_json in hs_raw:
                    try:
                        sources = json.loads(sources_json or "[]")
                    except Exception:
                        sources = []
                    try:
                        dbg = json.loads(debug_json or "{}")
                    except Exception:
                        dbg = {}
                    hs_rows.append(
                        {
                            "iso3": iso3,
                            "grounded": bool(grounded) or bool(sources),
                            "n_verified": len(sources) if isinstance(sources, list) else 0,
                            "n_unverified": 0,
                            "top_verified_urls": _extract_urls(sources),
                            "top_unverified_urls": [],
                            "groundingSupports_count": dbg.get("groundingSupports_count", 0),
                            "groundingChunks_count": dbg.get("groundingChunks_count", 0),
                            "n_sources_after": dbg.get("n_sources_after"),
                            "n_signals_after": dbg.get("n_signals_after"),
                            "attempted_models": dbg.get("attempted_models") or [],
                            "attempted_backends": _format_attempted_backends(dbg.get("attempted_backends")),
                            "selected_backend": dbg.get("selected_backend") or "",
                            "used_attempt": dbg.get("used_attempt"),
                            "last_errors": dbg.get("last_errors") or [],
                            "reason_code": dbg.get("reason_code"),
                        }
                    )
            except Exception:
                hs_rows = []

    if forecaster_run_id and question_ids:
        clause, params = _build_in_clause(question_ids)
        if clause:
            try:
                q_raw = con.execute(
                    f"""
                    SELECT question_id, question_evidence_json
                    FROM question_research
                    WHERE run_id = ?
                      AND question_id IN {clause}
                    """,
                    [forecaster_run_id, *params],
                ).fetchall()
                for qid, pack_json in q_raw:
                    try:
                        pack = json.loads(pack_json or "{}")
                    except Exception:
                        pack = {}
                    sources = pack.get("sources") or []
                    unverified = pack.get("unverified_sources") or []
                    dbg = pack.get("debug") or {}
                    verified_urls = _extract_urls(sources)
                    unverified_urls = _extract_urls(unverified)
                    q_rows.append(
                        {
                            "question_id": qid,
                            "grounded": bool(pack.get("grounded")) or bool(sources),
                            "n_verified": len(sources) if isinstance(sources, list) else 0,
                            "n_unverified": len(unverified) if isinstance(unverified, list) else 0,
                            "top_verified_urls": verified_urls,
                            "top_unverified_urls": unverified_urls,
                            "groundingSupports_count": dbg.get("groundingSupports_count", 0),
                            "groundingChunks_count": dbg.get("groundingChunks_count", 0),
                            "attempted_models": dbg.get("attempted_models") or dbg.get("model_id") or [],
                            "attempted_backends": _format_attempted_backends(dbg.get("attempted_backends")),
                            "selected_backend": dbg.get("selected_backend") or "",
                            "used_attempt": dbg.get("used_attempt"),
                            "last_errors": dbg.get("last_errors") or [],
                        }
                    )
            except Exception:
                q_rows = []

    return hs_rows, q_rows


def _web_research_accounting(
    con: duckdb.DuckDBPyConnection, forecaster_run_id: str | None, hs_run_id: str | None
) -> dict[str, int]:
    predicate_parts: list[str] = [
        "phase IN ('hs_web_research', 'research_web_research', 'forecast_web_research')"
    ]
    params: list[Any] = []
    scope_parts: list[str] = []
    if forecaster_run_id:
        scope_parts.append("run_id = ?")
        params.append(forecaster_run_id)
    if hs_run_id:
        scope_parts.append("hs_run_id = ?")
        params.append(hs_run_id)
    if scope_parts:
        predicate_parts.append(f"({' OR '.join(scope_parts)})")
    predicate = " AND ".join(predicate_parts)

    try:
        rows = _fetch_llm_rows(
            con,
            f"""
            SELECT model_id, usage_json, error_text
            FROM llm_calls
            WHERE {predicate}
            """,
            params,
        )
    except Exception:
        return {"n_calls": 0, "missing_model_id": 0, "missing_usage": 0, "missing_cost_usd": 0}

    counts = {"n_calls": 0, "missing_model_id": 0, "missing_usage": 0, "missing_cost_usd": 0}
    for row in rows:
        counts["n_calls"] += 1
        model_id = (row.get("model_id") or "").strip()
        if not model_id:
            counts["missing_model_id"] += 1
        usage_raw = row.get("usage_json") or "{}"
        try:
            usage_obj = json.loads(usage_raw)
        except Exception:
            usage_obj = {}
        if not usage_obj:
            counts["missing_usage"] += 1
        if usage_obj:
            total_tokens = int(usage_obj.get("total_tokens") or 0)
            cost_usd = usage_obj.get("cost_usd")
            if cost_usd is None:
                cost_usd = usage_obj.get("total_cost_usd")
            try:
                cost_val = float(cost_usd or 0.0)
            except Exception:
                cost_val = 0.0
            if total_tokens > 0 and cost_val == 0.0:
                counts["missing_cost_usd"] += 1
    return counts


def _load_spd_llm_model_ids(con: duckdb.DuckDBPyConnection, run_id: str) -> list[str]:
    try:
        rows = con.execute(
            """
            SELECT DISTINCT model_id
            FROM llm_calls
            WHERE run_id = ?
              AND call_type IN ('spd_v2', 'binary_v2')
              AND model_id IS NOT NULL
              AND model_id <> ''
            ORDER BY model_id
            """,
            [run_id],
        ).fetchall()
    except Exception:
        return []
    return [str(row[0]) for row in rows if row and row[0]]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = int(math.ceil((pct / 100.0) * len(ordered))) - 1
    idx = max(0, min(len(ordered) - 1, rank))
    return float(ordered[idx])


def _expected_ensemble_entries() -> tuple[list[dict[str, str]], str]:
    """Config-declared SPD ensemble as [{'name','model_id'}] plus a source label.

    Derived from pythia.llm_profiles.get_ensemble_resolved(), NOT
    providers.SPD_ENSEMBLE — the latter's `active` flags depend on which API
    keys the *report* process happens to see (a bundle job with only
    ANTHROPIC_API_KEY reported "Expected ensemble size: 1" while 5 models ran).
    """
    spec_override = os.getenv("PYTHIA_SPD_ENSEMBLE_SPECS", "").strip()
    if spec_override:
        specs = parse_ensemble_specs(spec_override)
        return (
            [
                {"name": (s.name or s.model_id), "model_id": s.model_id}
                for s in specs
                if getattr(s, "model_id", "")
            ],
            "PYTHIA_SPD_ENSEMBLE_SPECS",
        )
    try:
        from pythia.llm_profiles import get_ensemble_resolved

        entries = [
            {"name": str(e.get("model_id") or ""), "model_id": str(e.get("model_id") or "")}
            for e in get_ensemble_resolved()
            if e.get("model_id")
        ]
        if entries:
            return entries, "config ensemble via llm_profiles"
    except Exception:
        pass
    return (
        [
            {"name": (s.name or s.model_id), "model_id": s.model_id}
            for s in SPD_ENSEMBLE
            if getattr(s, "model_id", "")
        ],
        "SPD_ENSEMBLE",
    )


def _expected_spd_model_ids() -> list[str]:
    entries, _src = _expected_ensemble_entries()
    return sorted({e["model_id"] for e in entries if e.get("model_id")})


def _compute_question_run_metrics(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    questions: list[dict[str, Any]],
) -> None:
    if not questions:
        return

    pythia_schema.ensure_schema(con)
    llm_columns = _llm_calls_columns(con)
    question_ids = sorted({str(q.get("question_id")) for q in questions if q.get("question_id")})
    if not question_ids:
        return

    in_clause, in_params = _build_in_clause(question_ids)
    if not in_clause:
        return

    base_rows: dict[str, dict[str, Any]] = {}
    for q in questions:
        qid = q.get("question_id")
        if not qid:
            continue
        base_rows[str(qid)] = {
            "run_id": run_id,
            "question_id": str(qid),
            "iso3": q.get("iso3"),
            "hazard_code": q.get("hazard_code"),
            "metric": q.get("metric"),
            "started_at_utc": None,
            "finished_at_utc": None,
            "wall_ms": None,
            "compute_ms": None,
            "queue_ms": None,
            "cost_usd": None,
            "n_spd_models_expected": None,
            "n_spd_models_ok": None,
            "missing_model_ids_json": None,
            "phase_max_ms_json": None,
            "phase_cost_usd_json": None,
        }

    run_params = [run_id, *in_params]

    if "timestamp" in llm_columns:
        wall_rows = _fetch_llm_rows(
            con,
            f"""
            SELECT
                question_id,
                MIN(timestamp) AS started_at_utc,
                MAX(timestamp) AS finished_at_utc,
                CAST(
                    date_diff('millisecond', MIN(timestamp), MAX(timestamp))
                    AS BIGINT
                ) AS wall_ms
            FROM llm_calls
            WHERE run_id = ?
              AND question_id IN {in_clause}
              AND timestamp IS NOT NULL
            GROUP BY question_id
            """,
            run_params,
        )
        for row in wall_rows:
            qid = str(row.get("question_id") or "")
            if qid not in base_rows:
                continue
            base_rows[qid]["started_at_utc"] = row.get("started_at_utc")
            base_rows[qid]["finished_at_utc"] = row.get("finished_at_utc")
            base_rows[qid]["wall_ms"] = row.get("wall_ms")

    phase_types = ["research_v2", "research_web_research", "spd_v2", "binary_v2", "scenario_v2"]
    phase_max_by_qid: dict[str, dict[str, int]] = {}
    if "elapsed_ms" in llm_columns and "call_type" in llm_columns:
        phase_in_clause, phase_params = _build_in_clause(phase_types)
        if phase_in_clause:
            phase_rows = _fetch_llm_rows(
                con,
                f"""
                SELECT
                    question_id,
                    call_type,
                    MAX(elapsed_ms) AS max_elapsed_ms
                FROM llm_calls
                WHERE run_id = ?
                  AND question_id IN {in_clause}
                  AND call_type IN {phase_in_clause}
                  AND elapsed_ms IS NOT NULL
                GROUP BY question_id, call_type
                """,
                [run_id, *in_params, *phase_params],
            )
            for row in phase_rows:
                qid = str(row.get("question_id") or "")
                call_type = str(row.get("call_type") or "")
                if not qid or not call_type:
                    continue
                entry = phase_max_by_qid.setdefault(qid, {})
                entry[call_type] = int(row.get("max_elapsed_ms") or 0)

    cost_by_qid: dict[str, float] = {}
    cost_by_phase_by_qid: dict[str, dict[str, float]] = {}
    if "cost_usd" in llm_columns:
        cost_rows = _fetch_llm_rows(
            con,
            f"""
            SELECT question_id, SUM(cost_usd) AS cost_usd
            FROM llm_calls
            WHERE run_id = ?
              AND question_id IN {in_clause}
              AND cost_usd IS NOT NULL
            GROUP BY question_id
            """,
            run_params,
        )
        for row in cost_rows:
            qid = str(row.get("question_id") or "")
            if not qid:
                continue
            cost_by_qid[qid] = float(row.get("cost_usd") or 0.0)

        if "call_type" in llm_columns:
            phase_in_clause, phase_params = _build_in_clause(phase_types)
            if phase_in_clause:
                phase_cost_rows = _fetch_llm_rows(
                    con,
                    f"""
                    SELECT question_id, call_type, SUM(cost_usd) AS cost_usd
                    FROM llm_calls
                    WHERE run_id = ?
                      AND question_id IN {in_clause}
                      AND call_type IN {phase_in_clause}
                      AND cost_usd IS NOT NULL
                    GROUP BY question_id, call_type
                    """,
                    [run_id, *in_params, *phase_params],
                )
                for row in phase_cost_rows:
                    qid = str(row.get("question_id") or "")
                    call_type = str(row.get("call_type") or "")
                    if not qid or not call_type:
                        continue
                    entry = cost_by_phase_by_qid.setdefault(qid, {})
                    entry[call_type] = float(row.get("cost_usd") or 0.0)

    expected_model_ids = _expected_spd_model_ids()
    track_by_qid: dict[str, int | None] = {
        str(q.get("question_id")): (
            int(q["track"]) if q.get("track") is not None else None
        )
        for q in questions if q.get("question_id")
    }
    track2_model = os.getenv("PYTHIA_TRACK2_MODEL_ID", "gemini-3.5-flash")
    present_by_qid: dict[str, set[str]] = {}
    if expected_model_ids and "call_type" in llm_columns and "model_id" in llm_columns:
        error_filter = ""
        if "error_text" in llm_columns:
            error_filter = "AND (error_text IS NULL OR error_text = '')"
        spd_rows = _fetch_llm_rows(
            con,
            f"""
            SELECT question_id, model_id
            FROM llm_calls
            WHERE run_id = ?
              AND question_id IN {in_clause}
              AND call_type IN ('spd_v2', 'binary_v2')
              AND model_id IS NOT NULL
              AND model_id <> ''
              {error_filter}
            """,
            run_params,
        )
        for row in spd_rows:
            qid = str(row.get("question_id") or "")
            model_id = str(row.get("model_id") or "")
            if not qid or not model_id:
                continue
            present_by_qid.setdefault(qid, set()).add(model_id)

        # llm_calls says the CALL succeeded; forecasts_raw says whether the
        # output became a forecast. A member whose response failed to parse
        # leaves an ok call and a single no_forecast row, and until
        # 2026-09-01 that still counted as present — three such members made
        # every health view report 5/5. Where forecasts_raw carries rows for
        # a question, a model counts only with a complete ok SPD.
        complete_by_qid = _complete_member_spds_by_question(con, run_id, list(base_rows))
        if complete_by_qid:
            for qid, present in list(present_by_qid.items()):
                if qid in complete_by_qid:
                    present_by_qid[qid] = {m for m in present if m in complete_by_qid[qid]}

    # Track 2 makes ONE model call per question, so its llm_calls span is a
    # single instant and its phase_max map can be empty — which left
    # wall_ms, compute_ms and queue_ms null on all 85 Track-2 questions of
    # the 2026-09-01 run. Where llm_calls cannot date a question, the rows
    # it WROTE can: forecasts_raw carries a created_at per member row.
    write_span_by_qid = _forecast_write_spans(con, run_id, list(base_rows))

    rows_to_upsert: list[dict[str, Any]] = []
    for qid, row in sorted(base_rows.items()):
        phase_max = phase_max_by_qid.get(qid, {})
        compute_ms = (
            sum(int(val or 0) for val in phase_max.values()) if phase_max else None
        )
        span = write_span_by_qid.get(qid)
        if span:
            if row.get("started_at_utc") is None:
                row["started_at_utc"] = span.get("started_at_utc")
            if row.get("finished_at_utc") is None:
                row["finished_at_utc"] = span.get("finished_at_utc")
            if row.get("wall_ms") is None:
                row["wall_ms"] = span.get("wall_ms")
            if compute_ms is None:
                # The member call duration the writer recorded, falling back
                # to the write span. Either is a measurement, and a
                # measurement beats a blank.
                compute_ms = span.get("compute_ms")
                if compute_ms is None:
                    compute_ms = span.get("wall_ms")
        wall_ms = row.get("wall_ms")
        queue_ms = None
        if wall_ms is not None and compute_ms is not None:
            try:
                queue_ms = max(int(wall_ms) - int(compute_ms), 0)
            except Exception:
                queue_ms = None
        phase_costs = cost_by_phase_by_qid.get(qid, {})

        row["compute_ms"] = compute_ms
        row["queue_ms"] = queue_ms
        if qid in cost_by_qid:
            row["cost_usd"] = cost_by_qid[qid]

        if expected_model_ids:
            present = present_by_qid.get(qid, set())
            q_track = track_by_qid.get(qid)
            if q_track == 2:
                t2_expected = {track2_model}
                missing = sorted(t2_expected - present)
                row["n_spd_models_expected"] = len(t2_expected)
                row["n_spd_models_ok"] = len(t2_expected & present)
            else:
                missing = sorted(set(expected_model_ids) - present)
                row["n_spd_models_expected"] = len(expected_model_ids)
                row["n_spd_models_ok"] = len(set(expected_model_ids) & present)
            row["missing_model_ids_json"] = json.dumps(missing)

        if phase_max:
            row["phase_max_ms_json"] = json.dumps(phase_max, sort_keys=True)
        if phase_costs:
            row["phase_cost_usd_json"] = json.dumps(phase_costs, sort_keys=True)

        rows_to_upsert.append(row)

    if not rows_to_upsert:
        return

    frame = pd.DataFrame(rows_to_upsert)
    duckdb_io.upsert_dataframe(con, "question_run_metrics", frame, keys=["run_id", "question_id"])


def _forecast_write_spans(
    con: duckdb.DuckDBPyConnection, run_id: str, question_ids: list[str]
) -> dict[str, dict[str, Any]]:
    """{question_id: {started_at_utc, finished_at_utc, wall_ms, compute_ms}}.

    The fallback timing source for questions ``llm_calls`` cannot date. A
    Track-2 question is one model call, and a batched call is replayed with
    elapsed_ms=0 at a single collect-stage timestamp — which is why all 85
    Track-2 questions of the 2026-09-01 run had null timings.

    Two different sources, because they answer two different questions:
    ``forecasts_ensemble.created_at`` dates when the rows were WRITTEN
    (a wall span), and ``forecasts_raw.elapsed_ms`` is the per-member call
    duration the writer recorded (a compute figure). Neither is invented:
    both are columns the pipeline already fills.

    Never raises — a DB without either column returns {}.
    """

    if not question_ids:
        return {}
    in_clause, params = _build_in_clause(question_ids)
    if not in_clause:
        return {}

    out: dict[str, dict[str, Any]] = {}
    try:
        rows = _fetch_llm_rows(
            con,
            f"""
            SELECT
                question_id,
                MIN(created_at) AS started_at_utc,
                MAX(created_at) AS finished_at_utc,
                CAST(date_diff('millisecond', MIN(created_at), MAX(created_at)) AS BIGINT)
                    AS wall_ms
            FROM forecasts_ensemble
            WHERE run_id = ? AND question_id IN {in_clause}
              AND created_at IS NOT NULL
            GROUP BY question_id
            """,
            [run_id, *params],
        )
    except Exception:
        rows = []
    for row in rows:
        qid = str(row.get("question_id") or "")
        if qid:
            out[qid] = {
                "started_at_utc": row.get("started_at_utc"),
                "finished_at_utc": row.get("finished_at_utc"),
                "wall_ms": row.get("wall_ms"),
            }

    try:
        # elapsed_ms repeats across a member's bucket rows, so the slowest
        # member is MAX, not SUM.
        compute_rows = _fetch_llm_rows(
            con,
            f"""
            SELECT question_id, MAX(elapsed_ms) AS compute_ms
            FROM forecasts_raw
            WHERE run_id = ? AND question_id IN {in_clause}
              AND elapsed_ms IS NOT NULL
            GROUP BY question_id
            """,
            [run_id, *params],
        )
    except Exception:
        compute_rows = []
    for row in compute_rows:
        qid = str(row.get("question_id") or "")
        if not qid:
            continue
        entry = out.setdefault(qid, {"started_at_utc": None, "finished_at_utc": None, "wall_ms": None})
        entry["compute_ms"] = row.get("compute_ms")
    return out


def _complete_member_spds_by_question(
    con: duckdb.DuckDBPyConnection, run_id: str, question_ids: list[str]
) -> dict[str, set[str]]:
    """{question_id: {model_name with an ok SPD covering every window month}}.

    Only questions that have ANY forecasts_raw row are keyed, so a DB that
    never wrote member rows (older runs, tests) is left to the llm_calls
    answer. Never raises — a missing column returns {}.
    """

    if not question_ids:
        return {}
    try:
        from pythia.buckets import NUM_HORIZONS as _n_horizons  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        _n_horizons = 6
    in_clause, params = _build_in_clause(question_ids)
    try:
        rows = _fetch_llm_rows(
            con,
            f"""
            SELECT
                question_id,
                model_name,
                COUNT(DISTINCT CASE WHEN status = 'ok' THEN month_index END) AS n_months_ok
            FROM forecasts_raw
            WHERE run_id = ? AND question_id IN {in_clause}
            GROUP BY question_id, model_name
            """,
            [run_id, *params],
        )
    except Exception:
        return {}
    out: dict[str, set[str]] = {}
    for row in rows:
        qid = str(row.get("question_id") or "")
        if not qid:
            continue
        bucket = out.setdefault(qid, set())
        if int(row.get("n_months_ok") or 0) >= _n_horizons:
            bucket.add(str(row.get("model_name") or ""))
    return out


def _load_question_run_metrics(
    con: duckdb.DuckDBPyConnection, run_id: str
) -> list[dict[str, Any]]:
    try:
        rows = _fetch_llm_rows(
            con,
            """
            SELECT
                question_id,
                iso3,
                hazard_code,
                metric,
                wall_ms,
                compute_ms,
                queue_ms,
                cost_usd,
                missing_model_ids_json,
                n_spd_models_expected,
                n_spd_models_ok,
                phase_max_ms_json,
                phase_cost_usd_json
            FROM question_run_metrics
            WHERE run_id = ?
            """,
            [run_id],
        )
    except Exception:
        return []
    return rows


def _web_research_markdown(
    hs_rows: list[dict[str, Any]],
    question_rows: list[dict[str, Any]],
    *,
    web_research_enabled: bool,
    retriever_enabled: bool,
    self_search_call_total: int,
    hs_web_research_rows: list[dict[str, Any]] | None = None,
    hs_web_research_failures: list[dict[str, Any]] | None = None,
    research_web_research_rows: list[dict[str, Any]] | None = None,
    research_web_research_failures: list[dict[str, Any]] | None = None,
    self_search_rows: list[dict[str, Any]] | None = None,
    self_search_failures: list[dict[str, Any]] | None = None,
    accounting: dict[str, Any] | None = None,
) -> List[str]:
    lines: List[str] = []
    lines.append("### Web research evidence")
    lines.append("")
    if not web_research_enabled:
        lines.append("_Web research disabled via PYTHIA_WEB_RESEARCH_ENABLED._")
        lines.append("")
    if accounting is not None:
        lines.append("- Web research accounting:")
        lines.append(f"  - n_calls: `{accounting.get('n_calls', 0)}`")
        lines.append(f"  - missing_model_id: `{accounting.get('missing_model_id', 0)}`")
        lines.append(f"  - missing_usage: `{accounting.get('missing_usage', 0)}`")
        lines.append(f"  - missing_cost_usd: `{accounting.get('missing_cost_usd', 0)}`")
    lines.append("")
    lines.append("#### HS web research (hs_web_research) summary")
    lines.append("")
    lines.append("| provider | model_id | n_calls | n_verified_sources | n_errors |")
    lines.append("| -------- | -------- | ------- | ------------------ | -------- |")
    if hs_web_research_rows:
        for row in hs_web_research_rows:
            lines.append(
                f"| {row.get('provider')} | {row.get('model_id')} | {row.get('n_calls')} | "
                f"{row.get('n_verified_sources')} | {row.get('n_errors')} |"
            )
    else:
        lines.append("| (none) | (none) | 0 | 0 | 0 |")
    lines.append("")
    lines.append("#### HS web research failures (hs_web_research)")
    lines.append("")
    lines.append("| provider | model_id | error_code | error_message |")
    lines.append("| -------- | -------- | ---------- | ------------- |")
    if hs_web_research_failures:
        for row in hs_web_research_failures:
            lines.append(
                f"| {row.get('provider')} | {row.get('model_id')} | {row.get('error_code') or ''} | "
                f"{row.get('error_message') or ''} |"
            )
    else:
        lines.append("| (none) | (none) | (none) | (none) |")
    lines.append("")
    lines.append("#### Research web research (research_web_research) summary")
    lines.append("")
    lines.append("| provider | model_id | n_calls | n_verified_sources | n_errors |")
    lines.append("| -------- | -------- | ------- | ------------------ | -------- |")
    if research_web_research_rows:
        for row in research_web_research_rows:
            lines.append(
                f"| {row.get('provider')} | {row.get('model_id')} | {row.get('n_calls')} | "
                f"{row.get('n_verified_sources')} | {row.get('n_errors')} |"
            )
    else:
        lines.append("| (none) | (none) | 0 | 0 | 0 |")
    lines.append("")
    lines.append("#### Research web research failures (research_web_research)")
    lines.append("")
    lines.append("| provider | model_id | error_code | error_message |")
    lines.append("| -------- | -------- | ---------- | ------------- |")
    if research_web_research_failures:
        for row in research_web_research_failures:
            lines.append(
                f"| {row.get('provider')} | {row.get('model_id')} | {row.get('error_code') or ''} | "
                f"{row.get('error_message') or ''} |"
            )
    else:
        lines.append("| (none) | (none) | (none) | (none) |")
    lines.append("")
    if retriever_enabled and self_search_call_total == 0:
        lines.append("forecast_web_research: 0 (shared retriever in use)")
        lines.append("")
    elif self_search_rows or self_search_failures:
        lines.append("#### Forecast self-search (forecast_web_research) summary")
        lines.append("")
        lines.append("| provider | model_id | n_calls | n_verified_sources | n_errors |")
        lines.append("| -------- | -------- | ------- | ------------------ | -------- |")
        if self_search_rows:
            for row in self_search_rows:
                lines.append(
                    f"| {row.get('provider')} | {row.get('model_id')} | {row.get('n_calls')} | "
                    f"{row.get('n_verified_sources')} | {row.get('n_errors')} |"
                )
        else:
            lines.append("| (none) | (none) | 0 | 0 | 0 |")
        lines.append("")
        lines.append("#### Forecast self-search failures (forecast_web_research)")
        lines.append("")
        lines.append("| provider | model_id | error_code | error_message |")
        lines.append("| -------- | -------- | ---------- | ------------- |")
        if self_search_failures:
            for row in self_search_failures:
                lines.append(
                    f"| {row.get('provider')} | {row.get('model_id')} | {row.get('error_code') or ''} | "
                    f"{row.get('error_message') or ''} |"
                )
        else:
            lines.append("| (none) | (none) | (none) | (none) |")
        lines.append("")
    else:
        lines.append("#### Forecast self-search (forecast_web_research)")
        lines.append("")
        lines.append("_No forecast_web_research calls recorded for this run._")
        lines.append("")
    lines.append("#### HS country evidence packs")
    lines.append("")
    lines.append("| iso3 | grounded | n_verified | n_unverified | groundingSupports | groundingChunks | n_sources_after | n_signals_after | reason_code | attempted_models | attempted_backends | selected_backend | used_attempt | top_verified_urls | top_unverified_urls | last_errors |")
    lines.append("| ---- | -------- | ---------- | ------------- | ----------------- | --------------- | --------------- | --------------- | ----------- | ---------------- | ------------------ | ---------------- | ------------ | ----------------- | ------------------- | ----------- |")
    if hs_rows:
        for row in sorted(hs_rows, key=lambda r: r.get("iso3") or ""):
            lines.append(
                f"| {row.get('iso3')} | {row.get('grounded')} | {row.get('n_verified')} | {row.get('n_unverified')} | "
                f"{row.get('groundingSupports_count')} | {row.get('groundingChunks_count')} | "
                f"{row.get('n_sources_after') if row.get('n_sources_after') is not None else ''} | "
                f"{row.get('n_signals_after') if row.get('n_signals_after') is not None else ''} | "
                f"{row.get('reason_code') or ''} | "
                f"{', '.join(row.get('attempted_models') or [])} | {row.get('attempted_backends') or '(none)'} | "
                f"{row.get('selected_backend')} | {row.get('used_attempt') or ''} | "
                f"{', '.join(row.get('top_verified_urls') or [])} | {', '.join(row.get('top_unverified_urls') or [])} | "
                f"{'; '.join(row.get('last_errors') or [])} |"
            )
    else:
        lines.append(
            "| (none) | False | 0 | 0 | 0 | 0 | (none) | (none) | (none) | (none) | (none) | (none) | (none) | (none) | (none) | (none) |"
        )
    lines.append("")

    lines.append("#### HS country packs with 0 verified sources")
    lines.append("")
    lines.append("| iso3 | reason_code |")
    lines.append("| ---- | ----------- |")
    zero_verified = []
    if hs_rows:
        for row in sorted(hs_rows, key=lambda r: r.get("iso3") or ""):
            if int(row.get("n_verified") or 0) == 0:
                zero_verified.append(row)
    if zero_verified:
        for row in zero_verified:
            reason_code = row.get("reason_code") or ""
            lines.append(f"| {row.get('iso3')} | {reason_code} |")
    else:
        lines.append("| (none) | (none) |")
    lines.append("")

    lines.append("#### Question evidence packs")
    lines.append("")
    lines.append("| question_id | grounded | n_verified | n_unverified | groundingSupports | groundingChunks | attempted_models | attempted_backends | selected_backend | used_attempt | top_verified_urls | top_unverified_urls | last_errors |")
    lines.append("| ----------- | -------- | ---------- | ------------- | ----------------- | --------------- | ---------------- | ------------------ | ---------------- | ------------ | ----------------- | ------------------- | ----------- |")
    if question_rows:
        for row in sorted(question_rows, key=lambda r: r.get("question_id") or ""):
            lines.append(
                f"| {row.get('question_id')} | {row.get('grounded')} | {row.get('n_verified')} | {row.get('n_unverified')} | "
                f"{row.get('groundingSupports_count')} | {row.get('groundingChunks_count')} | "
                f"{', '.join(row.get('attempted_models') or [])} | {row.get('attempted_backends') or '(none)'} | "
                f"{row.get('selected_backend')} | {row.get('used_attempt') or ''} | "
                f"{', '.join(row.get('top_verified_urls') or [])} | {', '.join(row.get('top_unverified_urls') or [])} | "
                f"{'; '.join(row.get('last_errors') or [])} |"
            )
    else:
        lines.append("| (none) | False | 0 | 0 | 0 | 0 | (none) | (none) | (none) | (none) | (none) | (none) | (none) |")
    lines.append("")
    return lines


def _forecast_llm_filter(
    columns: Set[str], run_id: str, question_ids: list[str]
) -> tuple[str | None, list[Any], str]:
    if "meta_run_id" in columns:
        return "meta_run_id = ?", [run_id], "meta_run_id"
    if "forecaster_run_id" in columns:
        return "forecaster_run_id = ?", [run_id], "forecaster_run_id"
    if "run_id" in columns:
        return "run_id = ?", [run_id], "run_id"
    if "question_id" in columns and question_ids:
        clause, params = _build_in_clause(question_ids)
        if clause:
            return f"question_id IN {clause}", params, "question_id list"
    return None, [], "none"


def _hs_llm_filter(columns: Set[str], hs_run_id: str | None, iso3s: list[str]) -> tuple[str | None, list[Any], str]:
    if "hs_run_id" in columns and hs_run_id:
        return "hs_run_id = ?", [hs_run_id], "hs_run_id"
    if "iso3" in columns and iso3s:
        clause, params = _build_in_clause(iso3s)
        if clause:
            return f"iso3 IN {clause}", params, "iso3 list"
    if "phase" in columns:
        return "phase = 'hs_triage'", [], "phase_only"
    return None, [], "none"


def _combined_llm_filter(
    columns: Set[str],
    run_id: str,
    hs_run_id: str | None,
    question_ids: list[str],
    iso3s: list[str],
) -> tuple[str | None, list[Any], str]:
    forecast_pred, forecast_params, forecast_strategy = _forecast_llm_filter(columns, run_id, question_ids)
    hs_pred, hs_params, hs_strategy = _hs_llm_filter(columns, hs_run_id, iso3s)

    clauses: list[str] = []
    params: list[Any] = []
    strategy_parts: list[str] = []

    if forecast_pred:
        clauses.append(f"({forecast_pred})")
        params.extend(forecast_params)
        strategy_parts.append(f"forecast:{forecast_strategy}")

    if hs_pred:
        if "phase" not in hs_pred.lower():
            clauses.append(f"(phase = 'hs_triage' AND {hs_pred})")
        else:
            clauses.append(f"({hs_pred})")
        params.extend(hs_params)
        strategy_parts.append(f"hs:{hs_strategy}")

    if not clauses:
        return None, [], "none"

    return " OR ".join(clauses), params, "; ".join(strategy_parts)


def _hs_runs_columns(con: duckdb.DuckDBPyConnection) -> Set[str]:
    try:
        rows = con.execute("PRAGMA table_info('hs_runs')").fetchall()
    except Exception:
        return set()

    cols: Set[str] = set()
    for row in rows:
        if len(row) > 1 and row[1]:
            cols.add(str(row[1]))
    return cols


def _build_usage_json_from_row(row: dict[str, Any]) -> str:
    # If there's already a usage_json column, prefer it
    usage_raw = row.get("usage_json")
    if isinstance(usage_raw, str) and usage_raw.strip():
        return usage_raw

    # Otherwise synthesize a usage dict from known numeric columns if they exist
    usage: dict[str, Any] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "elapsed_ms",
        "cost_usd",
        "cost",
        "input_cost_usd",
        "output_cost_usd",
        "total_cost_usd",
    ):
        if key in row and row[key] is not None:
            usage[key] = row[key]

    return json.dumps(usage) if usage else "{}"


def _load_bucket_centroids_for_question(
    con: duckdb.DuckDBPyConnection,
    hazard_code: str,
    metric: str,
    bucket_labels: List[str],
) -> List[float]:
    """
    Return centroids (aligned to bucket_labels) for a hazard/metric.
    Prefer DB bucket_centroids; fallback to default PA/FATALITIES values.
    """

    hz = (hazard_code or "").upper()
    m = (metric or "").upper()

    try:
        rows = con.execute(
            """
            SELECT bucket_index, centroid
            FROM bucket_centroids
            WHERE hazard_code = ? AND metric = ?
            ORDER BY bucket_index
            """,
            [hz, m],
        ).fetchall()
    except Exception:
        rows = []

    if rows:
        centroids: List[float] = [0.0] * len(bucket_labels)
        for idx, centroid in rows:
            # bucket_index may be 0- or 1-based; normalise to 0-based
            i = int(idx) - 1 if int(idx) > 0 else int(idx)
            if 0 <= i < len(centroids):
                centroids[i] = float(centroid or 0.0)
        return centroids

    if m == "FATALITIES":
        return [0.0, 15.0, 62.0, 300.0, 700.0]
    return [0.0, 30_000.0, 150_000.0, 375_000.0, 700_000.0]


def _get_bucket_labels_for_question(question: Dict[str, Any]) -> List[str]:
    explicit = question.get("bucket_labels") or question.get("class_bins")
    if isinstance(explicit, list) and len(explicit) > 0:
        return [str(x) for x in explicit]

    if _bucket_labels_for_question is not None:
        return list(_bucket_labels_for_question(question))

    from pythia.buckets import labels_for

    metric = (question.get("metric") or "").upper()
    labels = labels_for(metric)
    return list(labels) if labels else labels_for("PA")


def _load_forecasts_raw_counts(
    con: duckdb.DuckDBPyConnection, run_id: str
) -> list[dict[str, Any]]:
    """Per-model forecasts_raw rows, plus the two counts that made the
    2026-09-01 shortfall visible: how many questions the model actually
    answered with an ok SPD, and how many no_forecast rows it left behind.
    Falls back to the bare row count on DBs without a status column."""

    try:
        return _fetch_llm_rows(
            con,
            """
            SELECT
                model_name,
                COUNT(*) AS n_rows,
                COUNT(DISTINCT CASE WHEN status = 'ok' THEN question_id END) AS n_questions_ok,
                COUNT(*) FILTER (WHERE status = 'no_forecast') AS n_no_forecast
            FROM forecasts_raw
            WHERE run_id = ?
            GROUP BY 1
            ORDER BY n_rows DESC, model_name
            """,
            [run_id],
        )
    except Exception:
        return _fetch_llm_rows(
            con,
            """
            SELECT model_name, COUNT(*) AS n_rows
            FROM forecasts_raw
            WHERE run_id = ?
            GROUP BY 1
            ORDER BY n_rows DESC, model_name
            """,
            [run_id],
        )


def _load_forecasts_ensemble_counts(
    con: duckdb.DuckDBPyConnection, run_id: str
) -> list[dict[str, Any]]:
    return _fetch_llm_rows(
        con,
        """
        SELECT model_name, COUNT(*) AS n_rows
        FROM forecasts_ensemble
        WHERE run_id = ?
        GROUP BY 1
        ORDER BY n_rows DESC, model_name
        """,
        [run_id],
    )


def _load_hs_run_metadata(
    con: duckdb.DuckDBPyConnection, hs_run_id: str | None
) -> dict[str, Any] | None:
    if not hs_run_id:
        return None

    cols = _hs_runs_columns(con)
    if not cols:
        return None

    select_cols = [
        col
        for col in (
            "hs_run_id",
            "generated_at",
            "git_sha",
            "config_profile",
            "countries_json",
            "requested_countries_json",
            "skipped_entries_json",
        )
        if col in cols
    ]

    if not select_cols:
        return None

    row = con.execute(
        f"""
        SELECT {", ".join(select_cols)}
        FROM hs_runs
        WHERE hs_run_id = ?
        ORDER BY COALESCE(generated_at, CURRENT_TIMESTAMP) DESC
        LIMIT 1
        """,
        [hs_run_id],
    ).fetchone()

    if not row:
        return None

    data = dict(zip(select_cols, row))
    parsed: dict[str, Any] = {
        "hs_run_id": data.get("hs_run_id"),
        "generated_at": data.get("generated_at"),
        "git_sha": data.get("git_sha"),
        "config_profile": data.get("config_profile"),
        "countries": [],
        "requested_countries": [],
        "skipped_entries": [],
    }

    for key, target in [
        ("countries_json", "countries"),
        ("requested_countries_json", "requested_countries"),
        ("skipped_entries_json", "skipped_entries"),
    ]:
        raw_val = data.get(key)
        if raw_val is None or key not in data:
            continue
        try:
            parsed[target] = json.loads(raw_val)
        except Exception:
            continue

    return parsed


def _load_hs_triage_summary(
    con: duckdb.DuckDBPyConnection, hs_run_id: str | None
) -> tuple[list[dict[str, Any]], int]:
    if not hs_run_id:
        return [], 0

    try:
        rows = _fetch_llm_rows(
            con,
            """
            SELECT iso3, COUNT(*) AS n_hazards, LIST(DISTINCT hazard_code) AS hazards
            FROM hs_triage
            WHERE run_id = ?
            GROUP BY iso3
            ORDER BY iso3
            """,
            [hs_run_id],
        )
    except Exception:
        rows = []

    total = sum(int(row.get("n_hazards") or 0) for row in rows)
    # Normalize hazards list to sorted comma-separated strings for deterministic output
    for row in rows:
        hazards = row.get("hazards") or []
        if isinstance(hazards, str):
            try:
                hazards = json.loads(hazards)
            except Exception:
                hazards = [hazards]
        hazards_list = sorted({str(h).upper() for h in hazards if h})
        row["hazards_sorted"] = hazards_list
    return rows, total


def _load_missing_hs_triage_combos(
    con: duckdb.DuckDBPyConnection,
    hs_run_id: str | None,
    iso3s: list[str],
) -> list[dict[str, str]]:
    if not hs_run_id or not iso3s:
        return []

    try:
        rows = con.execute(
            """
            SELECT iso3, hazard_code
            FROM hs_triage
            WHERE run_id = ?
            """,
            [hs_run_id],
        ).fetchall()
    except Exception:
        return []

    present: dict[str, set[str]] = {}
    for iso3, hazard_code in rows:
        iso3_up = str(iso3 or "").upper()
        hz_up = str(hazard_code or "").upper()
        if not iso3_up or not hz_up:
            continue
        present.setdefault(iso3_up, set()).add(hz_up)

    missing: list[dict[str, str]] = []
    for iso3 in sorted({str(code).upper() for code in iso3s if code}):
        for hazard_code in EXPECTED_HS_HAZARDS:
            if hazard_code not in present.get(iso3, set()):
                missing.append({"iso3": iso3, "hazard_code": hazard_code})
    return missing


def _load_llm_call_counts(
    con: duckdb.DuckDBPyConnection,
    predicate: str | None,
    params: list[Any],
) -> list[dict[str, Any]]:
    if not predicate:
        return []

    return _fetch_llm_rows(
        con,
        f"""
        SELECT
            COALESCE(phase, '') AS phase,
            COALESCE(provider, '') AS provider,
            COALESCE(model_id, '') AS model_id,
            COUNT(*) AS n_calls,
            SUM(
                CASE
                    WHEN error_text IS NOT NULL AND error_text <> '' THEN 1
                    ELSE 0
                END
            ) AS n_errors
        FROM llm_calls
        WHERE {predicate}
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """,
        params,
    )


def _load_batch_health(
    con: duckdb.DuckDBPyConnection,
    hs_run_id: str | None,
    forecaster_run_id: str | None,
    predicate: str | None,
    params: list[Any],
) -> dict[str, Any]:
    """Whether this run realised the provider batch discount.

    The bundle contained ZERO references to llm_batches or service_tier, so for
    the 2026-07-29 reference run it reported "LLM Calls: OK, 80 calls, 0 errors"
    for a run that had lost the discount on two thirds of its spend — and the
    bundle is the artifact a human reads first. Mirrors the queries in
    scripts/ci/batch_economics.py; everything is guarded, because a diagnostics
    gap must never become a bundle failure.
    """
    out: dict[str, Any] = {"available": False}
    ids = [i for i in (hs_run_id, forecaster_run_id) if i]
    if not ids:
        return out

    try:
        placeholders = ",".join("?" for _ in ids)
        batches = _fetch_llm_rows(
            con,
            f"""
            SELECT batch_id, COALESCE(provider,'') AS provider,
                   COALESCE(family,'') AS family, COALESCE(status,'') AS status,
                   COALESCE(n_requests,0) AS n_requests,
                   COALESCE(n_succeeded,0) AS n_succeeded,
                   COALESCE(error_text,'') AS error_text
            FROM llm_batches
            WHERE hs_run_id IN ({placeholders}) OR run_id IN ({placeholders})
            """,
            ids + ids,
        )
    except Exception:
        return out
    if not batches:
        return out

    out["available"] = True
    out["n_batches"] = len(batches)
    # A batch that yielded nothing is WHY spend lost the discount; naming it
    # turns an unexplained percentage into a cause.
    out["empty_batches"] = [
        {"batch_id": b["batch_id"], "provider": b["provider"],
         "family": b["family"], "error_text": b["error_text"]}
        for b in batches
        if int(b["n_requests"] or 0) and not int(b["n_succeeded"] or 0)
    ]

    batch_ids = [b["batch_id"] for b in batches]
    try:
        ph = ",".join("?" for _ in batch_ids)
        status_rows = _fetch_llm_rows(
            con,
            f"SELECT COALESCE(status,'') AS status, COUNT(*) AS n "
            f"FROM llm_batch_requests WHERE batch_id IN ({ph}) GROUP BY 1",
            batch_ids,
        )
        out["by_status"] = {r["status"]: int(r["n"]) for r in status_rows}
    except Exception:
        out["by_status"] = {}

    terminal = sum(
        out["by_status"].get(s, 0)
        for s in ("succeeded", "fallback_sync", "errored", "expired")
    )
    fallback = out["by_status"].get("fallback_sync", 0)
    out["n_terminal"] = terminal
    out["fallback_pct"] = round(100.0 * fallback / terminal, 1) if terminal else 0.0

    # Cost is the number that matters: on the reference run the fallback was 32%
    # of requests but two thirds of spend, because the batched and unbatched
    # members are not equally expensive.
    if predicate:
        try:
            cost_rows = _fetch_llm_rows(
                con,
                f"""
                SELECT
                    SUM(CASE WHEN json_extract_string(usage_json, '$.service_tier') = 'batch'
                             THEN COALESCE(cost_usd,0) ELSE 0 END) AS batch_cost,
                    SUM(COALESCE(cost_usd,0)) AS total_cost
                FROM llm_calls WHERE {predicate}
                """,
                params,
            )
            if cost_rows:
                bc = float(cost_rows[0].get("batch_cost") or 0.0)
                tc = float(cost_rows[0].get("total_cost") or 0.0)
                out["batch_cost_usd"] = round(bc, 4)
                out["total_cost_usd"] = round(tc, 4)
                # NOT the same as "lost discount": search-API spend (Brave)
                # and the synchronous phases (scenarios, adversarial
                # synthesis) are never batch-eligible, so a healthy run still
                # shows a non-zero share here. batch_economics.py owns the
                # real lost-discount figure, which counts only providers this
                # pipeline actually created a batch for.
                out["non_batch_tier_pct_of_spend"] = (
                    round(100.0 * (tc - bc) / tc, 1) if tc else 0.0
                )
        except Exception:
            pass
    return out


def _load_llm_error_summary(
    con: duckdb.DuckDBPyConnection,
    predicate: str | None,
    params: list[Any],
) -> list[dict[str, Any]]:
    if not predicate:
        return []

    return _fetch_llm_rows(
        con,
        f"""
        SELECT
            COALESCE(phase, '') AS phase,
            COALESCE(provider, '') AS provider,
            COALESCE(model_id, '') AS model_id,
            COUNT(*) AS n_errors
        FROM llm_calls
        WHERE {predicate}
          AND (
            error_text IS NOT NULL AND error_text <> ''
          )
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """,
        params,
    )


def _ensemble_participation_summary(
    forecasts_raw_counts: list[dict[str, Any]],
    spd_model_ids: list[str],
) -> list[str]:
    present = {row.get("model_name"): int(row.get("n_rows", 0) or 0) for row in forecasts_raw_counts}
    by_name = {row.get("model_name"): row for row in forecasts_raw_counts}
    present_spd_models = {mid for mid in spd_model_ids if mid}
    lines: list[str] = []
    if present:
        for model_name in sorted(present):
            row = by_name.get(model_name) or {}
            extra = ""
            if row.get("n_questions_ok") is not None:
                extra = f" ({int(row.get('n_questions_ok') or 0)} questions with an ok SPD"
                n_nf = int(row.get("n_no_forecast") or 0)
                extra += f", {n_nf} no_forecast" if n_nf else ""
                extra += ")"
            lines.append(f"- Model `{model_name}` wrote {present[model_name]} forecasts_raw rows{extra}.")
        # A member that answered fewer questions than its peers is the
        # silent-partial-ensemble signature (2026-09-01: Opus 69/71,
        # gpt-5.6-luna 70/71 while the health line said 5/5).
        member_cov = {
            name: int((by_name.get(name) or {}).get("n_questions_ok") or 0)
            for name in present
            if name in present_spd_models and (by_name.get(name) or {}).get("n_questions_ok") is not None
        }
        if member_cov:
            top = max(member_cov.values())
            short = {n: c for n, c in member_cov.items() if c < top}
            if short:
                lines.append(
                    "- **Partial ensemble**: "
                    + ", ".join(f"`{n}` answered {c}/{top} questions" for n, c in sorted(short.items()))
                    + " — those questions were aggregated from fewer members."
                )
    else:
        lines.append("- No forecasts_raw rows found for this run.")

    expected_entries, expected_source = _expected_ensemble_entries()
    expected_names = {e["name"] for e in expected_entries if e.get("name")}
    expected_model_ids = {e["model_id"] for e in expected_entries if e.get("model_id")}
    missing_by_model_id = sorted(expected_model_ids - present_spd_models)
    missing_by_name = sorted(expected_names - set(present.keys()))

    if expected_entries:
        lines.append(
            f"- Expected ensemble size: {len(expected_entries)} ({expected_source})."
        )
        if missing_by_model_id:
            for model_id in missing_by_model_id:
                lines.append(f"  - Missing spd_v2 llm_calls rows for model_id `{model_id}`.")
        else:
            lines.append("- All expected ensemble model_ids appear in spd_v2 llm_calls.")
        if missing_by_name:
            for name in missing_by_name:
                lines.append(f"  - Missing forecasts_raw rows for model `{name}`.")
        else:
            lines.append("- All expected ensemble models produced forecasts_raw rows.")
    else:
        lines.append("- SPD ensemble is empty or not configured; skipping expected-model check.")
    return lines


def _load_ensemble_spd_for_question(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    question_id: str,
    centroids: list[float],
) -> Dict[int, Dict[str, Any]]:
    """
    Return {month_index: {"probs": [p1..p5], "ev_value": ev or None}}.

    Falls back to computing EV from bucket probabilities and provided centroids
    when the database does not contain an ev_value.
    """

    rows = con.execute(
        """
        SELECT month_index, bucket_index, probability, ev_value
        FROM forecasts_ensemble
        WHERE run_id = ? AND question_id = ?
          AND month_index IS NOT NULL
          AND bucket_index IS NOT NULL
        ORDER BY month_index, bucket_index
        """,
        [run_id, question_id],
    ).fetchall()

    bucket_count = len(centroids) if centroids else 5
    by_month: Dict[int, Dict[str, Any]] = {}
    for month_idx, bucket_idx, prob, ev_value in rows:
        if month_idx is None or bucket_idx is None:
            continue
        m = int(month_idx)
        b = int(bucket_idx)
        entry = by_month.setdefault(m, {"probs": [0.0] * bucket_count, "ev_value": None})
        if 1 <= b <= bucket_count:
            entry["probs"][b - 1] = float(prob or 0.0)
        elif 0 <= b < bucket_count:
            entry["probs"][b] = float(prob or 0.0)

        if ev_value is not None:
            entry["ev_value"] = float(ev_value)
    for entry in by_month.values():
        if entry.get("ev_value") is not None:
            continue
        probs = entry.get("probs") or []
        n = min(len(probs), len(centroids))
        ev_calc = 0.0
        for i in range(n):
            ev_calc += float(probs[i]) * float(centroids[i])
        entry["ev_value"] = ev_calc
    return by_month


def _load_triage_entry(
    con: duckdb.DuckDBPyConnection,
    hs_run_id: str | None,
    iso3: str,
    hazard_code: str,
    cache: dict[tuple[str, str, str], dict[str, Any] | None] | None = None,
) -> dict[str, Any] | None:
    key = (hs_run_id or "", (iso3 or "").upper(), (hazard_code or "").upper())
    if cache is not None and key in cache:
        return cache[key]

    if not hs_run_id:
        if cache is not None:
            cache[key] = None
        return None

    try:
        rows = _fetch_llm_rows(
            con,
            """
            SELECT *
            FROM hs_triage
            WHERE run_id = ? AND iso3 = ? AND hazard_code = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            [hs_run_id, iso3, hazard_code],
        )
    except duckdb.CatalogException:
        rows = _fetch_llm_rows(
            con,
            """
            SELECT *
            FROM hs_triage
            WHERE run_id = ? AND iso3 = ? AND hazard_code = ?
            ORDER BY rowid DESC
            LIMIT 1
            """,
            [hs_run_id, iso3, hazard_code],
        )
    entry = rows[0] if rows else None
    if cache is not None:
        cache[key] = entry
    return entry


def _scenario_expected(
    hazard_code: str | None,
    metric: str | None,
    hs_entry: dict[str, Any] | None,
    *,
    track: int | None = None,
) -> tuple[bool, str]:
    tier = str((hs_entry or {}).get("tier") or "").lower()

    # Track 2 questions intentionally skip scenarios (single Gemini Flash model).
    if track == 2:
        return False, "track_2"

    # Track 1 questions always receive scenarios, even with quiet tier (RC-promoted).
    if track == 1:
        return True, ""

    # Non-priority (e.g. quiet) tier questions are skipped by design.
    if tier and tier != "priority":
        return False, "quiet_tier"

    # Priority tier — scenario is expected.
    return True, ""


def _load_scenario_row_count(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    iso3: str,
    hazard_code: str,
    metric: str,
) -> int:
    """Rows actually persisted to the scenarios table (DB truth, not attempts)."""
    try:
        row = con.execute(
            """
            SELECT COUNT(*)
            FROM scenarios
            WHERE run_id = ?
              AND upper(iso3) = upper(?)
              AND upper(hazard_code) = upper(?)
              AND upper(metric) = upper(?)
            """,
            [run_id, iso3, hazard_code, metric],
        ).fetchone()
        return int(row[0] or 0) if row else 0
    except Exception:
        return 0


def _load_scenario_call_count(
    con: duckdb.DuckDBPyConnection, run_id: str, question_id: str
) -> int:
    row = con.execute(
        """
        SELECT COUNT(*)
        FROM llm_calls
        WHERE run_id = ?
          AND question_id = ?
          AND (
              LOWER(COALESCE(call_type, '')) LIKE 'scenario%'
           OR LOWER(COALESCE(phase, '')) LIKE 'scenario%'
          )
        """,
        [run_id, question_id],
    ).fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dump unified Pythia v2 debug bundle (HS, Research, SPD, Scenario) for one run.",
    )
    parser.add_argument(
        "--db",
        default=os.getenv("PYTHIA_DB_URL", "duckdb:///data/resolver.duckdb"),
        help="DuckDB URL (e.g. duckdb:///data/resolver.duckdb)",
    )
    parser.add_argument(
        "--hs-run-id",
        required=True,
        help="HS run id to include in the bundle (e.g. hs_20240101T000000).",
    )
    parser.add_argument(
        "--forecaster-run-id",
        default=None,
        help="Optional Forecaster run id (e.g. fc_17648...). If omitted, build a triage-only bundle.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Deprecated alias for --forecaster-run-id.",
    )
    parser.add_argument(
        "--output-dir",
        default="debug",
        help="Directory to write the markdown bundle into (default: debug/).",
    )
    parser.add_argument(
        "--artifact-run-id",
        default=os.getenv("CANONICAL_DB_RUN_ID", ""),
        help="Source Actions run id for the DB artifact (for provenance).",
    )
    parser.add_argument(
        "--artifact-workflow",
        default=os.getenv("CANONICAL_DB_WORKFLOW", ""),
        help="Workflow name that produced the DB artifact (for provenance).",
    )
    parser.add_argument(
        "--artifact-name",
        default=os.getenv("CANONICAL_DB_ARTIFACT_NAME", ""),
        help="Artifact name used to download the DB (for provenance).",
    )
    parser.add_argument(
        "--counts-before-json",
        default="diagnostics/db_signature_before.json",
        help=(
            "Row counts captured BEFORE this phase wrote anything, as written by "
            "scripts.ci.db_signature at the start of the stage. Without it the "
            "provenance table's before column is null for every table and the "
            "deltas cannot be computed."
        ),
    )
    parser.add_argument(
        "--pipeline-id",
        default=os.getenv("PIPELINE_ID", ""),
        help="Batch pipeline id (pl_...), used to scope batch state and find this cycle's stage runs.",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        default=False,
        help="Generate legacy monolithic markdown bundle instead of split artifacts.",
    )
    return parser.parse_args()


def _resolve_db_path(db_url: str) -> str:
    if db_url.startswith("duckdb:///"):
        return db_url[len("duckdb:///") :]
    return db_url


def _select_run_id(con: duckdb.DuckDBPyConnection, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    row = con.execute(
        """
        SELECT run_id
        FROM forecasts_ensemble
        WHERE run_id LIKE 'fc_%'
        ORDER BY COALESCE(created_at, CURRENT_TIMESTAMP) DESC, run_id DESC
        LIMIT 1
        """,
    ).fetchone()
    return row[0] if row and row[0] else None


def _forecast_run_exists(con: duckdb.DuckDBPyConnection, run_id: str) -> bool:
    row = con.execute(
        """
        SELECT COUNT(*) FROM forecasts_ensemble WHERE run_id = ?
        """,
        [run_id],
    ).fetchone()
    return bool(row and row[0])


def _load_questions_for_run(con: duckdb.DuckDBPyConnection, run_id: str) -> list[dict[str, Any]]:
    rows = con.execute(
        """
        SELECT DISTINCT
            q.question_id,
            q.hs_run_id,
            q.iso3,
            q.hazard_code,
            q.metric,
            q.target_month,
            q.window_start_date,
            q.window_end_date,
            q.wording,
            q.track
        FROM questions q
        JOIN forecasts_ensemble fe
          ON fe.question_id = q.question_id
         AND fe.run_id = ?
        WHERE q.status = 'active'
        ORDER BY q.iso3, q.hazard_code, q.metric, q.question_id
        """,
        [run_id],
    ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        (
            qid,
            hs_run_id,
            iso3,
            hazard_code,
            metric,
            target_month,
            window_start_date,
            window_end_date,
            wording,
            track,
        ) = row
        out.append(
            {
                "question_id": qid,
                "hs_run_id": hs_run_id,
                "iso3": iso3,
                "hazard_code": hazard_code,
                "metric": metric,
                "target_month": target_month,
                "window_start_date": window_start_date,
                "window_end_date": window_end_date,
                "wording": wording,
                "track": track,
            }
        )
    return out


def _load_llm_calls_for_question(
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    question_id: str,
    iso3: str,
    hazard_code: str,
    hs_run_id: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Return phase -> call dict for this question/run.

    This function is robust to schema differences in llm_calls: it uses
    SELECT * and only reads columns that exist.
    """
    calls: Dict[str, Dict[str, Any]] = {}

    rows = _fetch_llm_rows(
        con,
        """
        SELECT *
        FROM llm_calls
        WHERE run_id = ?
          AND question_id = ?
          AND phase IN ('research_v2', 'spd_v2', 'scenario_v2')
        ORDER BY COALESCE(timestamp, CURRENT_TIMESTAMP) DESC
        """,
        [run_id, question_id],
    )

    for row in rows:
        phase = row.get("phase")
        if not phase or phase in calls:
            continue

        usage_json = _build_usage_json_from_row(row)
        calls[phase] = {
            "call_type": row.get("call_type"),
            "phase": phase,
            "provider": row.get("provider"),
            "model_id": row.get("model_id") or row.get("model"),
            "temperature": row.get("temperature"),
            "run_id": row.get("run_id"),
            "question_id": row.get("question_id"),
            "iso3": row.get("iso3"),
            "hazard_code": row.get("hazard_code"),
            "metric": row.get("metric"),
            "prompt_text": row.get("prompt_text") or "",
            "response_text": row.get("response_text") or "",
            "error_text": row.get("error_text") or "",
            "usage_json": usage_json,
        }

    hs_rows: list[dict[str, Any]] = []

    iso3_up = (iso3 or "").upper()
    hazard_up = (hazard_code or "").upper()

    hs_queries: list[tuple[str, list[Any]]] = []
    if hs_run_id:
        hs_queries.append(
            (
                """
                SELECT *
                FROM llm_calls
                WHERE phase = 'hs_triage'
                  AND hs_run_id = ?
                  AND iso3 = ?
                  AND hazard_code = ?
                ORDER BY COALESCE(timestamp, CURRENT_TIMESTAMP) DESC
                LIMIT 1
                """,
                [hs_run_id, iso3_up, hazard_up],
            )
        )
        hs_queries.append(
            (
                """
                SELECT *
                FROM llm_calls
                WHERE phase = 'hs_triage'
                  AND hs_run_id = ?
                  AND iso3 = ?
                ORDER BY COALESCE(timestamp, CURRENT_TIMESTAMP) DESC
                LIMIT 1
                """,
                [hs_run_id, iso3_up],
            )
        )
        hs_queries.append(
            (
                """
                SELECT *
                FROM llm_calls
                WHERE phase = 'hs_triage'
                  AND hs_run_id = ?
                ORDER BY COALESCE(timestamp, CURRENT_TIMESTAMP) DESC
                LIMIT 1
                """,
                [hs_run_id],
            )
        )

    hs_queries.append(
        (
            """
            SELECT *
            FROM llm_calls
            WHERE phase = 'hs_triage'
              AND iso3 = ?
              AND hazard_code = ?
            ORDER BY COALESCE(timestamp, CURRENT_TIMESTAMP) DESC
            LIMIT 1
            """,
            [iso3_up, hazard_up],
        )
    )

    hs_queries.append(
        (
            """
            SELECT *
            FROM llm_calls
            WHERE phase = 'hs_triage'
              AND iso3 = ?
            ORDER BY COALESCE(timestamp, CURRENT_TIMESTAMP) DESC
            LIMIT 1
            """,
            [iso3_up],
        )
    )

    for query, params in hs_queries:
        hs_rows = _fetch_llm_rows(con, query, params)
        if hs_rows:
            break

    if hs_rows:
        row = hs_rows[0]
        usage_json = _build_usage_json_from_row(row)
        calls["hs_triage"] = {
            "call_type": row.get("call_type"),
            "phase": row.get("phase"),
            "provider": row.get("provider"),
            "model_id": row.get("model_id") or row.get("model"),
            "temperature": row.get("temperature"),
            "run_id": row.get("run_id"),
            "question_id": row.get("question_id"),
            "iso3": row.get("iso3"),
            "hazard_code": row.get("hazard_code"),
            "metric": row.get("metric"),
            "prompt_text": row.get("prompt_text") or "",
            "response_text": row.get("response_text") or "",
            "error_text": row.get("error_text") or "",
            "usage_json": usage_json,
            "expected_iso3": iso3,
            "expected_hazard_code": hazard_code,
        }
    else:
        triage_rows = _fetch_llm_rows(
            con,
            """
            SELECT *
            FROM hs_triage
            WHERE run_id = ?
              AND iso3 = ?
              AND hazard_code = ?
            ORDER BY COALESCE(created_at, CURRENT_TIMESTAMP) DESC
            LIMIT 1
            """,
            [hs_run_id or run_id, iso3, hazard_code],
        )

        if triage_rows:
            triage = triage_rows[0]
            triage_payload = {
                "tier": triage.get("tier"),
                "triage_score": triage.get("triage_score"),
                "need_full_spd": triage.get("need_full_spd"),
                "drivers": triage.get("drivers_json"),
                "regime_shifts": triage.get("regime_shifts_json"),
                "data_quality": triage.get("data_quality_json"),
                "scenario_stub": triage.get("scenario_stub"),
            }

            for key in ("drivers", "regime_shifts", "data_quality"):
                raw_val = triage_payload.get(key)
                if isinstance(raw_val, str):
                    try:
                        triage_payload[key] = json.loads(raw_val)
                    except Exception:
                        continue

            calls["hs_triage"] = {
                "call_type": "hs_triage_fallback",
                "phase": "hs_triage",
                "provider": None,
                "model_id": None,
                "temperature": None,
                "run_id": triage.get("run_id") or hs_run_id or run_id,
                "question_id": question_id,
                "iso3": triage.get("iso3") or iso3,
                "hazard_code": triage.get("hazard_code") or hazard_code,
                "metric": None,
                "prompt_text": "HS triage (from hs_triage table; no llm_calls row)",
                "response_text": json.dumps(triage_payload, ensure_ascii=False, indent=2),
                "error_text": "",
                "usage_json": "{}",
                "expected_iso3": iso3,
                "expected_hazard_code": hazard_code,
            }

    return calls


def _resolve_hs_run_id_for_forecast(
    con: duckdb.DuckDBPyConnection, run_id: str
) -> str | None:
    try:
        row = con.execute(
            """
            SELECT q.hs_run_id
            FROM forecasts_ensemble fe
            JOIN questions q ON fe.question_id = q.question_id
            WHERE fe.run_id = ?
              AND q.hs_run_id IS NOT NULL
              AND q.hs_run_id <> ''
            ORDER BY COALESCE(fe.created_at, CURRENT_TIMESTAMP) DESC
            LIMIT 1
            """,
            [run_id],
        ).fetchone()
        if row and row[0]:
            return str(row[0])
    except Exception:
        return None
    return None


def _aggregate_usage_by_phase(
    con: duckdb.DuckDBPyConnection, run_id: str, hs_run_id: str | None
) -> dict[str, dict[str, float]]:
    """
    Aggregate token and cost usage by phase for a given forecast run.

    We:
      - Include all llm_calls rows where run_id = <run_id>
      - Optionally include HS triage calls (phase='hs_triage'), even if their run_id is NULL
        or not equal to <run_id>, by matching on phase alone.

    We DO NOT join to questions here to avoid schema assumptions such as q.run_id.
    """
    # Load all calls for this run (research_v2, spd_v2, scenario_v2, etc.)
    params: list[Any] = [run_id]
    if hs_run_id:
        query = """
            SELECT *
            FROM llm_calls
            WHERE run_id = ?
               OR (phase = 'hs_triage' AND hs_run_id = ?)
        """
        params.append(hs_run_id)
    else:
        query = """
            SELECT *
            FROM llm_calls
            WHERE run_id = ?
               OR (phase = 'hs_triage')
        """

    rows = _fetch_llm_rows(con, query, params)

    out: dict[str, dict[str, float]] = {}
    for row in rows:
        phase = row.get("phase") or "unknown"
        usage_raw = row.get("usage_json") or "{}"
        try:
            usage = json.loads(usage_raw)
        except Exception:
            usage = {}

        phase_acc = out.setdefault(
            phase,
            {
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "total_cost_usd": 0.0,
            },
        )
        prompt_tokens = float(usage.get("prompt_tokens") or 0.0)
        completion_tokens = float(usage.get("completion_tokens") or 0.0)
        total_tokens = float(usage.get("total_tokens") or 0.0)
        if total_tokens == 0.0 and (prompt_tokens or completion_tokens):
            total_tokens = prompt_tokens + completion_tokens
            usage = dict(usage)
            usage["total_tokens"] = total_tokens
        phase_acc["prompt_tokens"] += prompt_tokens
        phase_acc["completion_tokens"] += completion_tokens
        phase_acc["total_tokens"] += total_tokens
        # For backwards compatibility, accept either total_cost_usd or cost_usd
        cost_val = float(usage.get("total_cost_usd") or usage.get("cost_usd") or 0.0)
        if cost_val == 0.0:
            if total_tokens > 0:
                model_id = row.get("model_id") or row.get("model") or ""
                if model_id:
                    cost_val = estimate_cost_usd(str(model_id), usage)
        phase_acc["total_cost_usd"] += cost_val

    return out


def _question_lifecycle_counts(
    con: duckdb.DuckDBPyConnection, run_id: str
) -> dict[str, int]:
    counts = {
        "research": 0,
        "forecast": 0,
        "scenario": 0,
    }
    try:
        research_rows = con.execute(
            """
            SELECT COUNT(DISTINCT question_id)
            FROM llm_calls
            WHERE run_id = ?
              AND phase = 'research_v2'
            """,
            [run_id],
        ).fetchone()
        if research_rows and research_rows[0] is not None:
            counts["research"] = int(research_rows[0])
    except Exception:
        counts["research"] = 0

    try:
        forecast_llm_rows = con.execute(
            """
            SELECT COUNT(DISTINCT question_id)
            FROM llm_calls
            WHERE run_id = ?
              AND phase = 'spd_v2'
            """,
            [run_id],
        ).fetchone()
        forecast_llm = int(forecast_llm_rows[0]) if forecast_llm_rows and forecast_llm_rows[0] else 0
    except Exception:
        forecast_llm = 0

    try:
        forecast_rows = con.execute(
            """
            SELECT COUNT(DISTINCT question_id)
            FROM forecasts_ensemble
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchone()
        forecast_tbl = int(forecast_rows[0]) if forecast_rows and forecast_rows[0] else 0
    except Exception:
        forecast_tbl = 0
    counts["forecast"] = max(forecast_llm, forecast_tbl)

    try:
        scenario_rows = con.execute(
            """
            SELECT COUNT(DISTINCT question_id)
            FROM llm_calls
            WHERE run_id = ?
              AND phase = 'scenario_v2'
            """,
            [run_id],
        ).fetchone()
        if scenario_rows and scenario_rows[0] is not None:
            counts["scenario"] = int(scenario_rows[0])
    except Exception:
        counts["scenario"] = 0

    return counts


def _question_ids_researched_not_forecasted(
    con: duckdb.DuckDBPyConnection, run_id: str
) -> list[str]:
    research_ids: set[str] = set()
    forecast_ids: set[str] = set()
    try:
        research_rows = con.execute(
            """
            SELECT DISTINCT question_id
            FROM llm_calls
            WHERE run_id = ?
              AND phase = 'research_v2'
            """,
            [run_id],
        ).fetchall()
        for row in research_rows or []:
            if row and row[0]:
                research_ids.add(str(row[0]))
    except Exception:
        research_ids = set()

    try:
        forecast_llm_rows = con.execute(
            """
            SELECT DISTINCT question_id
            FROM llm_calls
            WHERE run_id = ?
              AND phase = 'spd_v2'
            """,
            [run_id],
        ).fetchall()
        for row in forecast_llm_rows or []:
            if row and row[0]:
                forecast_ids.add(str(row[0]))
    except Exception:
        forecast_ids = set()

    try:
        forecast_tbl_rows = con.execute(
            """
            SELECT DISTINCT question_id
            FROM forecasts_ensemble
            WHERE run_id = ?
            """,
            [run_id],
        ).fetchall()
        for row in forecast_tbl_rows or []:
            if row and row[0]:
                forecast_ids.add(str(row[0]))
    except Exception:
        forecast_ids = forecast_ids

    return sorted(research_ids - forecast_ids)


def _load_triage_tier(
    con: duckdb.DuckDBPyConnection, hs_run_id: str, iso3: str, hazard_code: str
) -> str | None:
    row = con.execute(
        """
        SELECT tier
        FROM hs_triage
        WHERE run_id = ? AND iso3 = ? AND hazard_code = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [hs_run_id, iso3, hazard_code],
    ).fetchone()
    return row[0] if row and row[0] else None


def _load_spd_status(con: duckdb.DuckDBPyConnection, run_id: str, question_id: str) -> str | None:
    row = con.execute(
        """
        SELECT status
        FROM forecasts_ensemble
        WHERE run_id = ? AND question_id = ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [run_id, question_id],
    ).fetchone()
    return row[0] if row and row[0] else None


def _extract_iso3_from_hs_prompt(prompt_text: str) -> tuple[str, str]:
    prompt_iso3 = ""
    prompt_label = ""
    if not prompt_text:
        return prompt_iso3, prompt_label

    match = re.search(r"assessing\s+([^()]+?)\s*\(\s*([A-Z]{3})\s*\)", prompt_text, flags=re.I)
    if match:
        prompt_label = match.group(1).strip()
        prompt_iso3 = match.group(2).upper()

    if not prompt_iso3:
        json_match = re.search(r'"country"\s*:\s*"([A-Z]{3})"', prompt_text, flags=re.I)
        if json_match:
            prompt_iso3 = json_match.group(1).upper()

    return prompt_iso3, prompt_label


def _hs_metadata_warnings(call: Dict[str, Any]) -> list[str]:
    prompt_iso3, prompt_label = _extract_iso3_from_hs_prompt(call.get("prompt_text") or "")
    logged_iso3 = str(call.get("iso3") or "").upper()
    expected_iso3 = str(call.get("expected_iso3") or "").upper()
    hazard_logged = str(call.get("hazard_code") or "").upper()
    hazard_expected = str(call.get("expected_hazard_code") or "").upper()

    warnings: list[str] = []

    if prompt_iso3 and (
        (logged_iso3 and prompt_iso3 != logged_iso3)
        or (expected_iso3 and prompt_iso3 != expected_iso3)
    ):
        pieces: list[str] = []
        if prompt_label:
            pieces.append(f"prompt country `{prompt_label}` ({prompt_iso3})")
        else:
            pieces.append(f"prompt ISO3 `{prompt_iso3}`")
        if logged_iso3:
            pieces.append(f"logged ISO3 `{logged_iso3}`")
        if expected_iso3 and expected_iso3 != logged_iso3:
            pieces.append(f"question ISO3 `{expected_iso3}`")
        warnings.append("- Warning: HS prompt location mismatch: " + "; ".join(pieces) + ".")
    elif expected_iso3 and logged_iso3 and expected_iso3 != logged_iso3:
        warnings.append(
            f"- Warning: HS logged ISO3 `{logged_iso3}` differs from question ISO3 `{expected_iso3}`."
        )

    if hazard_expected and hazard_logged and hazard_expected != hazard_logged:
        warnings.append(
            f"- Warning: HS hazard mismatch (logged `{hazard_logged}` vs question `{hazard_expected}`)."
        )

    return warnings


def _append_stage_block(lines: List[str], phase: str, call: Dict[str, Any] | None) -> None:
    if call is None:
        lines.append(f"_No LLM call recorded for phase `{phase}`._")
        return

    lines.append("")
    lines.append("##### Metadata")
    lines.append("")
    lines.append(f"- Phase: `{call.get('phase')}`")
    lines.append(f"- Provider: `{call.get('provider')}`")
    lines.append(f"- Model: `{call.get('model_id')}`")
    if call.get("temperature") is not None:
        lines.append(f"- Temperature: `{call.get('temperature')}`")
    lines.append(f"- Run ID: `{call.get('run_id')}`")
    if call.get("question_id"):
        lines.append(f"- Question ID: `{call.get('question_id')}`")
    if call.get("iso3"):
        lines.append(f"- ISO3: `{call.get('iso3')}`")
    if call.get("hazard_code"):
        lines.append(f"- Hazard: `{call.get('hazard_code')}`")
    if call.get("metric"):
        lines.append(f"- Metric: `{call.get('metric')}`")

    if call.get("phase") == "hs_triage":
        warnings = _hs_metadata_warnings(call)
        if warnings:
            lines.append("")
            lines.append("_HS metadata warnings:_")
            lines.extend(warnings)

    usage_raw = call.get("usage_json") or "{}"
    try:
        usage = json.loads(usage_raw)
    except Exception:
        usage = {}
    if usage:
        lines.append("")
        lines.append("##### Usage / Cost")
        lines.append("")
        ordered_keys = [
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "input_cost_usd",
            "output_cost_usd",
            "total_cost_usd",
            "cost_usd",
            "elapsed_ms",
        ]
        for key in ordered_keys:
            if key in usage and usage[key] is not None:
                lines.append(f"- {key}: `{usage[key]}`")

        for key, val in usage.items():
            if key in ordered_keys:
                continue
            lines.append(f"- {key}: `{val}`")

    lines.append("")
    lines.append("##### Prompt")
    lines.append("")
    lines.append("```text")
    lines.append(call.get("prompt_text") or "")
    lines.append("```")

    lines.append("")
    lines.append("##### Output")
    lines.append("")
    lines.append("```text")
    lines.append(call.get("response_text") or "")
    lines.append("```")

    lines.append("")
    lines.append("##### Errors / Failure Notes")
    lines.append("")
    error_text = call.get("error_text") or ""
    if error_text.strip():
        lines.append(error_text.strip())
    else:
        lines.append("_No error reported for this call._")


@dataclass
class BundleData:
    """Shared container for all data loaded from the DB, used by all emitters."""

    # Identity
    hs_run_id: str | None = None
    forecaster_run_id: str | None = None
    out_run_id: str = ""
    db_url: str = ""
    now: str = ""

    # Provenance
    provenance_entry: dict[str, Any] = field(default_factory=dict)
    provenance_lines: list[str] = field(default_factory=list)
    counts_before: dict[str, int | None] = field(default_factory=dict)
    counts_after: dict[str, int | None] = field(default_factory=dict)
    db_stats: dict[str, Any] = field(default_factory=dict)

    # HS manifest
    hs_manifest: dict[str, Any] | None = None
    resolved_countries_sorted: list[str] = field(default_factory=list)
    requested_countries: list[Any] = field(default_factory=list)
    skipped_entries: list[Any] = field(default_factory=list)

    # Triage
    hs_triage_rows: list[dict[str, Any]] = field(default_factory=list)
    n_hazards_triaged_total: int = 0
    missing_hs_triage: list[dict[str, str]] = field(default_factory=list)

    # Questions (forecaster only)
    questions: list[dict[str, Any]] = field(default_factory=list)
    question_ids: list[str] = field(default_factory=list)
    n_questions_by_hazard: dict[str, int] = field(default_factory=dict)
    n_questions_by_iso3: dict[str, int] = field(default_factory=dict)
    lifecycle_counts: dict[str, int] = field(default_factory=dict)
    researched_not_forecasted: list[str] = field(default_factory=list)

    # Question run metrics (forecaster only)
    question_run_metrics: list[dict[str, Any]] = field(default_factory=list)
    question_run_metrics_warning: str | None = None

    # Scenario status (forecaster only)
    scenario_status_rows: list[dict[str, Any]] = field(default_factory=list)

    # Web research
    hs_web_rows: list[dict[str, Any]] = field(default_factory=list)
    question_web_rows: list[dict[str, Any]] = field(default_factory=list)
    web_research_enabled: bool = False
    retriever_enabled: bool = False
    hs_research_web_search: str = "0"
    spd_web_search: str = "0"
    hs_web_research_active: bool = False
    research_web_research_active: bool = False
    web_research_accounting: dict[str, Any] = field(default_factory=dict)
    hs_web_research_rows: list[dict[str, Any]] = field(default_factory=list)
    hs_web_research_failures: list[dict[str, Any]] = field(default_factory=list)
    rc_grounding_rows: list[dict[str, Any]] = field(default_factory=list)
    triage_grounding_rows: list[dict[str, Any]] = field(default_factory=list)
    adversarial_grounding_rows: list[dict[str, Any]] = field(default_factory=list)
    research_web_research_rows: list[dict[str, Any]] = field(default_factory=list)
    research_web_research_failures: list[dict[str, Any]] = field(default_factory=list)
    self_search_rows: list[dict[str, Any]] = field(default_factory=list)
    self_search_failures: list[dict[str, Any]] = field(default_factory=list)
    self_search_call_total: int = 0

    # LLM call counts
    llm_call_counts: list[dict[str, Any]] = field(default_factory=list)
    batch_health: dict[str, Any] = field(default_factory=dict)
    llm_error_rows: list[dict[str, Any]] = field(default_factory=list)
    llm_calls_skip_note: str | None = None
    self_search_stats: dict[str, int] = field(default_factory=dict)
    latency_block: str = ""

    # Filter predicates (for latency queries etc.)
    predicate: str | None = None
    predicate_params: list[Any] = field(default_factory=list)
    predicate_strategy: str = ""

    # Usage/cost
    usage_by_phase: dict[str, dict[str, float]] = field(default_factory=dict)
    usage_by_phase_warning: str | None = None

    # Ensemble
    forecasts_raw_counts: list[dict[str, Any]] = field(default_factory=list)
    forecasts_ensemble_counts: list[dict[str, Any]] = field(default_factory=list)
    spd_model_ids: list[str] = field(default_factory=list)

    # CrisisWatch
    crisiswatch_entries: list[dict[str, Any]] = field(default_factory=list)
    crisiswatch_table_exists: bool = False
    crisiswatch_load_error: str | None = None

    # Run summary stats (for executive summary bottom sections)
    rc_grounding_call_stats: dict[str, Any] = field(default_factory=dict)
    triage_grounding_call_stats: dict[str, Any] = field(default_factory=dict)
    adversarial_grounding_call_stats: dict[str, Any] = field(default_factory=dict)
    # Adversarial synthesis (ADVERSARIAL_SYNTH_% generation calls) — tracked
    # separately from the search stats above: {n_calls, n_errors, cost_usd}.
    adversarial_synth_stats: dict[str, Any] = field(default_factory=dict)
    hs_triage_detail_rows: list[dict[str, Any]] = field(default_factory=list)
    structured_data_coverage: dict[str, tuple[int, int, list[str]]] = field(default_factory=dict)
    food_security_no_data_countries: list[str] = field(default_factory=list)


def _load_bundle_data(
    con: duckdb.DuckDBPyConnection,
    hs_run_id: str | None,
    forecaster_run_id: str | None,
    db_url: str,
    provenance_entry: dict[str, Any],
    provenance_lines: list[str],
    counts_before: dict[str, int | None],
    counts_after: dict[str, int | None],
    db_stats: dict[str, Any],
    questions: list[dict[str, Any]] | None = None,
) -> BundleData:
    """Load all data needed by all emitters into a shared BundleData container."""
    data = BundleData()
    data.hs_run_id = hs_run_id
    data.forecaster_run_id = forecaster_run_id
    data.out_run_id = forecaster_run_id or hs_run_id or ""
    data.db_url = db_url
    data.now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    data.provenance_entry = provenance_entry
    data.provenance_lines = provenance_lines
    data.counts_before = counts_before
    data.counts_after = counts_after
    data.db_stats = db_stats
    data.questions = questions or []

    # HS manifest
    hs_run_id_for_costs = hs_run_id
    if forecaster_run_id and not hs_run_id:
        hs_run_id_for_costs = _resolve_hs_run_id_for_forecast(con, forecaster_run_id)
    manifest_hs_run_id = hs_run_id_for_costs
    if not manifest_hs_run_id and data.questions:
        hs_run_ids = sorted({q.get("hs_run_id") for q in data.questions if q.get("hs_run_id")})
        manifest_hs_run_id = hs_run_ids[0] if hs_run_ids else None

    data.hs_manifest = _load_hs_run_metadata(con, manifest_hs_run_id)
    data.requested_countries = (
        (data.hs_manifest or {}).get("requested_countries") if data.hs_manifest else []
    ) or []
    resolved_countries = (
        (data.hs_manifest or {}).get("countries") if data.hs_manifest else []
    ) or []
    iso3s = sorted({(q.get("iso3") or "").upper() for q in data.questions if q.get("iso3")})
    if not resolved_countries:
        resolved_countries = list(iso3s)
    data.resolved_countries_sorted = sorted({str(c) for c in resolved_countries if c})
    data.skipped_entries = (
        (data.hs_manifest or {}).get("skipped_entries") if data.hs_manifest else []
    ) or []

    # Triage
    data.hs_triage_rows, data.n_hazards_triaged_total = _load_hs_triage_summary(
        con, manifest_hs_run_id
    )
    data.missing_hs_triage = _load_missing_hs_triage_combos(
        con, manifest_hs_run_id, data.resolved_countries_sorted
    )

    # Questions
    data.question_ids = sorted(
        [str(q.get("question_id")) for q in data.questions if q.get("question_id")]
    )
    for q in data.questions:
        hz = (q.get("hazard_code") or "").upper()
        iso_val = (q.get("iso3") or "").upper()
        if hz:
            data.n_questions_by_hazard[hz] = data.n_questions_by_hazard.get(hz, 0) + 1
        if iso_val:
            data.n_questions_by_iso3[iso_val] = data.n_questions_by_iso3.get(iso_val, 0) + 1

    # Web research
    data.hs_web_rows, data.question_web_rows = _load_web_research_summaries(
        con, manifest_hs_run_id, forecaster_run_id, data.resolved_countries_sorted, data.question_ids
    )
    data.web_research_enabled = os.getenv("PYTHIA_WEB_RESEARCH_ENABLED", "0") == "1"
    data.retriever_enabled = os.getenv("PYTHIA_RETRIEVER_ENABLED", "0") == "1"
    data.hs_research_web_search = os.getenv("PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED", "0")
    data.spd_web_search = os.getenv("PYTHIA_SPD_WEB_SEARCH_ENABLED", "0")
    data.hs_web_research_active = data.retriever_enabled or data.hs_research_web_search == "1"
    data.research_web_research_active = data.retriever_enabled or data.hs_research_web_search == "1"
    data.web_research_accounting = _web_research_accounting(con, forecaster_run_id, manifest_hs_run_id)
    data.hs_web_research_rows, data.hs_web_research_failures = _load_web_research_summary(
        con, "hs_web_research", forecaster_run_id, manifest_hs_run_id
    )
    data.rc_grounding_rows = _load_grounding_subsystem_stats(
        con, "GROUNDING_%", forecaster_run_id, manifest_hs_run_id
    )
    data.triage_grounding_rows = _load_grounding_subsystem_stats(
        con, "TRIAGE_GROUNDING_%", forecaster_run_id, manifest_hs_run_id
    )
    data.adversarial_grounding_rows = _load_grounding_subsystem_stats(
        con, "ADVERSARIAL_%", forecaster_run_id, manifest_hs_run_id
    )
    data.research_web_research_rows, data.research_web_research_failures = _load_web_research_summary(
        con, "research_web_research", forecaster_run_id, manifest_hs_run_id
    )
    data.self_search_rows, data.self_search_failures = _load_web_research_summary(
        con, "forecast_web_research", forecaster_run_id, manifest_hs_run_id
    )
    data.self_search_call_total = sum(
        int(row.get("n_calls") or 0) for row in (data.self_search_rows or [])
    )

    # Per-call grounding stats (for executive summary run sections)
    data.rc_grounding_call_stats = _load_grounding_call_stats(
        con, "hs_triage", "GROUNDING_%", forecaster_run_id, manifest_hs_run_id
    )
    data.triage_grounding_call_stats = _load_grounding_call_stats(
        con, "hs_triage", "TRIAGE_GROUNDING_%", forecaster_run_id, manifest_hs_run_id
    )
    # Adversarial checks: merge stats from hs_web_research (OpenAI/Gemini fallback)
    # and hs_triage with ADVERSARIAL_% hazard_code (Brave primary)
    _adv_web = _load_grounding_call_stats(
        con, "hs_web_research", None, forecaster_run_id, manifest_hs_run_id
    )
    _adv_brave = _load_grounding_call_stats(
        con, "hs_triage", "ADVERSARIAL_%", forecaster_run_id, manifest_hs_run_id
    )
    data.adversarial_grounding_call_stats = {
        "n_calls": _adv_web.get("n_calls", 0) + _adv_brave.get("n_calls", 0),
        "n_errors": _adv_web.get("n_errors", 0) + _adv_brave.get("n_errors", 0),
        "source_counts": (_adv_web.get("source_counts") or []) + (_adv_brave.get("source_counts") or []),
    }
    data.adversarial_synth_stats = _load_adversarial_synth_stats(
        con, forecaster_run_id, manifest_hs_run_id
    )

    # HS triage detail rows (RC levels, screen-outs, tiers)
    if manifest_hs_run_id:
        try:
            data.hs_triage_detail_rows = _fetch_llm_rows(
                con,
                "SELECT iso3, hazard_code, tier, track, regime_change_level, data_quality_json "
                "FROM hs_triage WHERE run_id = ?",
                [manifest_hs_run_id],
            )
        except Exception:
            LOG.warning("Failed to load hs_triage detail rows")

    # Structured data coverage
    if data.resolved_countries_sorted:
        data.structured_data_coverage = _load_structured_data_coverage(
            con, data.resolved_countries_sorted
        )

    # Food security: identify DR countries without food security data
    dr_iso3s = sorted({
        (q.get("iso3") or "").upper()
        for q in data.questions
        if (q.get("hazard_code") or "").upper() == "DR"
    })
    if dr_iso3s and _safe_table_exists(con, "facts_resolved"):
        try:
            placeholders = ", ".join("?" for _ in dr_iso3s)
            covered = {
                row[0]
                for row in con.execute(
                    f"SELECT DISTINCT UPPER(iso3) FROM facts_resolved "
                    f"WHERE UPPER(iso3) IN ({placeholders}) "
                    f"AND hazard_code = 'DR' "
                    f"AND metric IN ('phase3plus_in_need', 'phase3plus_projection')",
                    dr_iso3s,
                ).fetchall()
            }
            data.food_security_no_data_countries = sorted(
                iso3 for iso3 in dr_iso3s if iso3 not in covered
            )
        except Exception:
            LOG.warning("Failed to compute food security coverage gaps")

    # LLM call counts and latency
    llm_columns = _llm_calls_columns(con)
    if forecaster_run_id:
        data.predicate, data.predicate_params, data.predicate_strategy = _combined_llm_filter(
            llm_columns, forecaster_run_id, manifest_hs_run_id, data.question_ids,
            data.resolved_countries_sorted,
        )
    else:
        pred, pred_params, pred_strategy = _hs_llm_filter(
            llm_columns, hs_run_id, data.resolved_countries_sorted
        )
        if not pred:
            pred = "phase = 'hs_triage'"
            pred_params = []
            pred_strategy = "phase_only"
        data.predicate = pred
        data.predicate_params = pred_params
        data.predicate_strategy = pred_strategy

    try:
        data.llm_call_counts = _load_llm_call_counts(con, data.predicate, data.predicate_params)
        data.llm_error_rows = [
            row for row in data.llm_call_counts if int(row.get("n_errors") or 0) > 0
        ]
        data.self_search_stats = _load_self_search_stats(con, data.predicate, data.predicate_params)
        data.batch_health = _load_batch_health(
            con, hs_run_id, forecaster_run_id, data.predicate, data.predicate_params
        )
    except Exception as exc:
        data.llm_calls_skip_note = f"Error loading llm_calls: {exc}"
    data.latency_block = render_latency_markdown(
        con, data.predicate, data.predicate_params, data.predicate_strategy
    )

    # Forecaster-specific data
    if forecaster_run_id:
        try:
            data.usage_by_phase = _aggregate_usage_by_phase(con, forecaster_run_id, hs_run_id_for_costs)
        except Exception as exc:
            data.usage_by_phase_warning = f"Error aggregating llm_calls usage: {exc}"

        data.lifecycle_counts = _question_lifecycle_counts(con, forecaster_run_id)
        data.researched_not_forecasted = _question_ids_researched_not_forecasted(con, forecaster_run_id)
        data.forecasts_raw_counts = _load_forecasts_raw_counts(con, forecaster_run_id)
        data.forecasts_ensemble_counts = _load_forecasts_ensemble_counts(con, forecaster_run_id)
        data.spd_model_ids = _load_spd_llm_model_ids(con, forecaster_run_id)

        try:
            _compute_question_run_metrics(con, forecaster_run_id, data.questions)
        except Exception as exc:
            data.question_run_metrics_warning = f"Error computing question_run_metrics: {exc}"
        data.question_run_metrics = _load_question_run_metrics(con, forecaster_run_id)

        # Scenario status
        triage_cache: dict[tuple[str, str, str], dict[str, Any] | None] = {}
        for q in data.questions:
            qid = q.get("question_id")
            q_iso3 = q.get("iso3") or ""
            q_hz = q.get("hazard_code") or ""
            q_metric = q.get("metric") or ""
            q_hs_run_id = q.get("hs_run_id") or hs_run_id_for_costs
            triage_entry = _load_triage_entry(con, q_hs_run_id, q_iso3, q_hz, cache=triage_cache)
            triage_tier = (triage_entry or {}).get("tier")
            q_track_raw = q.get("track")
            q_track = int(q_track_raw) if q_track_raw is not None else None
            expected, reason = _scenario_expected(q_hz, q_metric, triage_entry, track=q_track)
            call_count = _load_scenario_call_count(con, forecaster_run_id, str(qid))
            row_count = _load_scenario_row_count(
                con, forecaster_run_id, q_iso3, q_hz, q_metric
            )
            # "generated" means a row landed in the scenarios table — an LLM
            # attempt whose JSON failed to parse must not be reported as
            # generated (the old call_count-based status did exactly that).
            if row_count > 0:
                status = "generated"
            elif call_count > 0 and expected:
                status = "failed_parse"
            elif not expected:
                status = f"skipped_by_design: {reason or 'not_expected'}"
            else:
                status = "missing_unexpected"
            data.scenario_status_rows.append(
                {
                    "question_id": qid,
                    "iso3": q_iso3,
                    "hazard_code": q_hz,
                    "metric": q_metric,
                    "hs_run_id": q_hs_run_id,
                    "triage_tier": triage_tier,
                    "status": status,
                    "expected": expected,
                    "call_count": call_count,
                }
            )

    # CrisisWatch
    try:
        data.crisiswatch_table_exists = _safe_table_exists(con, "crisiswatch_entries")
        if data.crisiswatch_table_exists:
            data.crisiswatch_entries = _fetch_llm_rows(
                con,
                """SELECT iso3, country_name, arrow, alert_type, summary,
                          month, year, fetched_at
                   FROM crisiswatch_entries
                   ORDER BY year DESC, month DESC, iso3""",
                [],
            )
    except Exception as exc:
        data.crisiswatch_load_error = str(exc)

    return data


# ---------------------------------------------------------------------------
# Error categorisation helpers
# ---------------------------------------------------------------------------

_AUTH_PATTERNS = re.compile(r"401|403|Unauthorized|Forbidden|AuthenticationError", re.I)
_TIMEOUT_PATTERNS = re.compile(r"timeout|timed.out|DeadlineExceeded|DEADLINE_EXCEEDED", re.I)
_RATE_LIMIT_PATTERNS = re.compile(r"429|rate.limit|RateLimitError|ResourceExhausted|quota", re.I)


def _categorise_error(error_text: str) -> str:
    """Classify an error_text string into auth/timeout/rate_limit/data_quality."""
    if not error_text:
        return ""
    if _AUTH_PATTERNS.search(error_text):
        return "auth"
    if _TIMEOUT_PATTERNS.search(error_text):
        return "timeout"
    if _RATE_LIMIT_PATTERNS.search(error_text):
        return "rate_limit"
    return "data_quality"


# ---------------------------------------------------------------------------
# Traffic-light health evaluator
# ---------------------------------------------------------------------------

def _member_gap_summary(question_run_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Count Track-1 (question, member) cells with no usable SPD.

    Reads the per-question ``missing_model_ids_json`` the metrics table
    carries (which, since 2026-09-01, requires a complete ok SPD in
    forecasts_raw, not merely an ok llm_calls row). Track-2 questions are
    excluded: their single model is judged separately.
    """

    by_model: dict[str, int] = {}
    n_missing = 0
    n_expected = 0
    n_questions = 0
    for row in question_run_metrics or []:
        expected = row.get("n_spd_models_expected")
        try:
            expected_n = int(expected or 0)
        except (TypeError, ValueError):
            expected_n = 0
        if expected_n <= 1:
            continue  # Track 2 or unknown
        n_expected += expected_n
        raw = row.get("missing_model_ids_json")
        try:
            missing = json.loads(raw) if isinstance(raw, str) and raw else (raw or [])
        except Exception:
            missing = []
        if not isinstance(missing, list) or not missing:
            continue
        n_questions += 1
        for m in missing:
            name = str(m)
            by_model[name] = by_model.get(name, 0) + 1
            n_missing += 1
    return {
        "n_cells_missing": n_missing,
        "n_cells_expected": n_expected,
        "n_questions_affected": n_questions,
        "by_model": by_model,
    }


def _evaluate_pipeline_health(data: BundleData) -> list[dict[str, Any]]:
    """Return list of {subsystem, status, detail} dicts for the executive summary."""
    checks: list[dict[str, Any]] = []

    # DB Provenance
    sha = data.provenance_entry.get("db_sha256")
    checks.append({
        "subsystem": "DB Provenance",
        "status": "OK" if sha else "FAIL",
        "detail": f"sha256={sha[:16]}..." if sha else "missing",
    })

    # HS Triage completeness
    expected_hs = len(data.resolved_countries_sorted) * len(EXPECTED_HS_HAZARDS)
    missing_count = len(data.missing_hs_triage)
    if expected_hs == 0:
        hs_status = "OK"
        hs_detail = "no countries"
    elif missing_count == 0:
        hs_status = "OK"
        hs_detail = f"{data.n_hazards_triaged_total}/{expected_hs} rows"
    elif missing_count < expected_hs * 0.05:
        hs_status = "WARN"
        hs_detail = f"{data.n_hazards_triaged_total}/{expected_hs} rows, {missing_count} missing"
    else:
        hs_status = "FAIL"
        hs_detail = f"{data.n_hazards_triaged_total}/{expected_hs} rows, {missing_count} missing"
    checks.append({"subsystem": "HS Triage", "status": hs_status, "detail": hs_detail})

    # CrisisWatch (ICG conflict arrows — ACE data source)
    if data.crisiswatch_load_error:
        cw_status = "FAIL"
        cw_detail = f"load error: {data.crisiswatch_load_error}"
    elif not data.crisiswatch_table_exists:
        cw_status = "WARN"
        cw_detail = "crisiswatch_entries table missing — schema migration needed"
    elif not data.crisiswatch_entries:
        cw_status = "WARN"
        cw_detail = "table exists but 0 entries — Gemini grounding and fallback JSON both returned nothing"
    else:
        n_entries = len(data.crisiswatch_entries)
        arrows = [e.get("arrow") or "" for e in data.crisiswatch_entries]
        n_deteriorated = sum(1 for a in arrows if a == "deteriorated")
        n_improved = sum(1 for a in arrows if a == "improved")
        n_with_arrow = sum(1 for a in arrows if a)
        n_alerts = sum(1 for e in data.crisiswatch_entries if e.get("alert_type"))
        months = sorted({(e.get("year"), e.get("month")) for e in data.crisiswatch_entries})
        latest = months[-1] if months else (None, None)
        parts = [f"{n_entries} countries"]
        if n_with_arrow:
            parts.append(f"{n_deteriorated} deteriorated, {n_improved} improved")
        else:
            parts.append("no arrows (Global Overview call may have failed)")
        if n_alerts:
            parts.append(f"{n_alerts} alerts")
        if latest[0]:
            parts.append(f"latest: {latest[0]}-{latest[1]:02d}" if latest[1] else f"latest: {latest[0]}")
        # Which editions the table actually holds: on 2026-09-01 it held
        # February and June only, and "latest 2026-06" understated that the
        # per-country rows were a mix of two editions three months apart.
        edition_labels = [
            f"{y}-{int(m):02d}" if m else str(y) for (y, m) in months if y
        ]
        if edition_labels:
            parts.append(f"editions in table: {', '.join(edition_labels)}")
        # Staleness: the edition is refreshed from the Wayback Machine by
        # resolver_update Phase 4 (monthly), the weekly structured-data
        # ingest and the hs_submit stage; refresh-crisiswatch.yml is the
        # early-month monitor. Data older than two editions means every one
        # of those paths failed to land a newer edition. Fail loudly instead
        # of the old silent WARN (the July 2026 run was serving February data).
        stale_months = False
        if latest[0]:
            try:
                _now = datetime.utcnow()
                _age_months = (_now.year - int(latest[0])) * 12 + (
                    _now.month - int(latest[1] or _now.month)
                )
                # CrisisWatch covers month M and is published early in M+1,
                # so age 1-2 months is normal cadence; >2 means the monthly
                # local refresh was missed.
                stale_months = _age_months > 2
            except Exception:
                stale_months = False
        if stale_months:
            cw_status = "FAIL"
            cw_detail = (
                f"STALE — {'; '.join(parts)}. Prompts now carry a staleness note "
                "for entries older than two editions. Dispatch Ingest Structured "
                "Data with sources=crisiswatch (Wayback refresh + DB store), or run "
                "python -m scripts.refresh_crisiswatch --source wayback locally and "
                "store the JSON."
            )
        elif n_entries < 10:
            cw_status = "WARN"
            cw_detail = f"low coverage — {'; '.join(parts)}"
        elif n_with_arrow == 0:
            cw_status = "WARN"
            cw_detail = f"no arrow data — {'; '.join(parts)}"
        else:
            cw_status = "OK"
            cw_detail = "; ".join(parts)
    checks.append({"subsystem": "CrisisWatch", "status": cw_status, "detail": cw_detail})

    # Food Security (FEWS NET IPC + IPC API sources in facts_resolved)
    dr_countries = sorted({
        (q.get("iso3") or "").upper()
        for q in (data.questions or [])
        if (q.get("hazard_code") or "").upper() == "DR"
    })
    if dr_countries:
        fewsnet_cov, *_ = data.structured_data_coverage.get("FEWS NET IPC", (0, 0, []))
        ipc_cov, *_ = data.structured_data_coverage.get("IPC API", (0, 0, []))
        total_cov = fewsnet_cov + ipc_cov
        if total_cov == 0:
            fs_status = "WARN"
            fs_detail = f"0/{len(dr_countries)} DR countries have food security data"
        else:
            fs_status = "OK"
            parts = []
            if fewsnet_cov:
                parts.append(f"{fewsnet_cov} FEWS NET")
            if ipc_cov:
                parts.append(f"{ipc_cov} IPC API")
            fs_detail = f"{' + '.join(parts)} countries with data"
        checks.append({"subsystem": "Food Security", "status": fs_status, "detail": fs_detail})

    # No-backend sentinel counter (model_id in the #808 sentinel set): each
    # such call is a country-hazard pair that got ZERO grounding evidence.
    # Written to the health JSON since #808 but previously rendered nowhere
    # — a fully breaker-tripped run still read clean in the markdown report.
    _NO_BACKEND_IDS = {
        "grounding-failed", "grounding-breaker-tripped", "grounding-unavailable",
    }

    def _no_backend_calls(rows: list) -> int:
        return sum(
            int(r.get("n_calls") or 0)
            for r in (rows or [])
            if str(r.get("model_id") or "").strip() in _NO_BACKEND_IDS
        )

    # RC Grounding health (hazard_code like GROUNDING_ACE, GROUNDING_FL, etc.)
    _rc_g_calls = sum(int(r.get("n_calls") or 0) for r in (data.rc_grounding_rows or []))
    _rc_g_errors = sum(int(r.get("n_errors") or 0) for r in (data.rc_grounding_rows or []))
    _rc_g_verified = sum(int(r.get("n_verified_sources") or 0) for r in (data.rc_grounding_rows or []))
    _rc_g_no_backend = _no_backend_calls(data.rc_grounding_rows)
    if _rc_g_calls > 0 and _rc_g_errors == 0:
        rc_g_status = "OK"
        rc_g_detail = f"{_rc_g_calls} calls, {_rc_g_verified} verified sources"
    elif _rc_g_calls > 0 and _rc_g_errors < _rc_g_calls:
        rc_g_status = "WARN"
        rc_g_detail = f"{_rc_g_calls} calls, {_rc_g_errors} returned 0 sources, {_rc_g_verified} verified sources"
    elif _rc_g_calls > 0:
        rc_g_status = "FAIL"
        rc_g_detail = f"{_rc_g_calls} calls, all returned 0 sources"
    else:
        rc_g_status = "OK"
        rc_g_detail = "no RC grounding calls in this run"
    if _rc_g_no_backend > 0:
        if rc_g_status == "OK":
            rc_g_status = "WARN"
        rc_g_detail += (
            f"; {_rc_g_no_backend} call(s) never reached ANY backend "
            "(grounding-* sentinels — those hazards got zero evidence)"
        )
    checks.append({"subsystem": "RC Grounding", "status": rc_g_status, "detail": rc_g_detail})

    # Triage Grounding health (hazard_code like TRIAGE_GROUNDING_ACE, etc.)
    _tg_calls = sum(int(r.get("n_calls") or 0) for r in (data.triage_grounding_rows or []))
    _tg_errors = sum(int(r.get("n_errors") or 0) for r in (data.triage_grounding_rows or []))
    _tg_verified = sum(int(r.get("n_verified_sources") or 0) for r in (data.triage_grounding_rows or []))
    if _tg_calls > 0 and _tg_errors == 0:
        tg_status = "OK"
        tg_detail = f"{_tg_calls} calls, {_tg_verified} verified sources"
    elif _tg_calls > 0 and _tg_errors < _tg_calls:
        tg_status = "WARN"
        tg_detail = f"{_tg_calls} calls, {_tg_errors} returned 0 sources, {_tg_verified} verified sources"
    elif _tg_calls > 0:
        tg_status = "FAIL"
        tg_detail = f"{_tg_calls} calls, all returned 0 sources"
    else:
        tg_status = "OK"
        tg_detail = "no triage grounding calls in this run"
    _tg_no_backend = _no_backend_calls(data.triage_grounding_rows)
    if _tg_no_backend > 0:
        if tg_status == "OK":
            tg_status = "WARN"
        tg_detail += (
            f"; {_tg_no_backend} call(s) never reached ANY backend "
            "(grounding-* sentinels — those hazards got zero evidence)"
        )
    checks.append({"subsystem": "Triage Grounding", "status": tg_status, "detail": tg_detail})

    # Adversarial Checks health (hs_web_research + Brave ADVERSARIAL_% rows)
    _adv_rows = (data.hs_web_research_rows or []) + (data.adversarial_grounding_rows or [])
    _hs_g_calls = sum(int(r.get("n_calls") or 0) for r in _adv_rows)
    _hs_g_errors = sum(int(r.get("n_errors") or 0) for r in _adv_rows)
    _hs_g_verified = sum(int(r.get("n_verified_sources") or 0) for r in _adv_rows)
    if _hs_g_calls > 0 and _hs_g_errors == 0:
        g_status = "OK"
        g_detail = f"{_hs_g_calls} calls, {_hs_g_verified} verified sources"
    elif _hs_g_calls > 0:
        g_status = "WARN"
        g_detail = f"{_hs_g_calls} calls, {_hs_g_errors} errors, {_hs_g_verified} verified sources"
    elif data.hs_web_research_active:
        g_status = "WARN"
        g_detail = "no adversarial check calls"
    else:
        g_status = "OK"
        g_detail = "disabled"
    checks.append({"subsystem": "Adversarial Checks", "status": g_status, "detail": g_detail})

    # Adversarial synthesis health (generation calls — separate from search)
    _synth = data.adversarial_synth_stats or {}
    _synth_calls = int(_synth.get("n_calls") or 0)
    _synth_errors = int(_synth.get("n_errors") or 0)
    _synth_cost = float(_synth.get("cost_usd") or 0.0)
    if _synth_calls == 0:
        synth_status = "OK"
        synth_detail = "no synthesis calls in this run (no RC L1+ hazards or checks disabled)"
    elif _synth_errors == 0:
        synth_status = "OK"
        synth_detail = f"{_synth_calls} calls, ${_synth_cost:.2f}"
    else:
        synth_status = "WARN"
        synth_detail = f"{_synth_calls} calls, {_synth_errors} errors, ${_synth_cost:.2f}"
    checks.append(
        {"subsystem": "Adversarial Synthesis", "status": synth_status, "detail": synth_detail}
    )

    # Grounding health — Research
    # The question-level web research pipeline is deprecated; when the retriever
    # is disabled (PYTHIA_RETRIEVER_ENABLED=0), 0 grounded questions is expected
    # because structured data injection replaces the old retriever pipeline.
    q_grounded = sum(1 for r in data.question_web_rows if r.get("grounded"))
    q_total = len(data.question_web_rows)
    if not data.retriever_enabled:
        rg_status = "OK"
        rg_detail = (
            f"retriever disabled — using structured data injection"
            f" ({q_grounded}/{q_total} legacy grounded)" if q_total else
            "retriever disabled — using structured data injection"
        )
    elif q_total == 0:
        rg_status = "OK" if not data.forecaster_run_id else "WARN"
        rg_detail = "no research grounding" if not data.forecaster_run_id else "0 questions grounded"
    elif q_grounded == q_total:
        rg_status = "OK"
        rg_detail = f"{q_grounded}/{q_total} questions grounded"
    elif q_grounded >= q_total * 0.85:
        rg_status = "WARN"
        rg_detail = f"{q_grounded}/{q_total} questions grounded"
    else:
        rg_status = "FAIL"
        rg_detail = f"{q_grounded}/{q_total} questions grounded"
    checks.append({"subsystem": "Research Grounding", "status": rg_status, "detail": rg_detail})

    # LLM call health
    total_errors = sum(int(r.get("n_errors") or 0) for r in data.llm_call_counts)
    total_calls = sum(int(r.get("n_calls") or 0) for r in data.llm_call_counts)
    if total_calls == 0:
        llm_status = "WARN"
        llm_detail = "no LLM calls"
    else:
        error_rate = total_errors / total_calls
        if error_rate == 0:
            llm_status = "OK"
            llm_detail = f"{total_calls} calls, 0 errors"
        elif error_rate < 0.05:
            llm_status = "WARN"
            llm_detail = f"{total_calls} calls, {total_errors} errors ({error_rate:.1%})"
        else:
            llm_status = "FAIL"
            llm_detail = f"{total_calls} calls, {total_errors} errors ({error_rate:.1%})"
    checks.append({"subsystem": "LLM Calls", "status": llm_status, "detail": llm_detail})

    # Batch economics. "LLM Calls: OK, 80 calls, 0 errors" was true and useless
    # for the reference run, which had lost the batch discount on two thirds of
    # its spend — an error-free run can still be an expensive one.
    bh = data.batch_health or {}
    if bh.get("available"):
        parts = [f"{bh.get('n_batches', 0)} batches"]
        if bh.get("n_terminal"):
            parts.append(
                f"{bh['fallback_pct']}% of {bh['n_terminal']} requests fell back to sync"
            )
        if bh.get("total_cost_usd"):
            parts.append(
                f"${bh['batch_cost_usd']} of ${bh['total_cost_usd']} at batch tier "
                f"({bh['non_batch_tier_pct_of_spend']}% not batch-eligible: "
                f"search APIs + synchronous phases)"
            )
        empty = bh.get("empty_batches") or []
        if empty:
            b_status = "FAIL"
            parts.append(
                f"{len(empty)} batch(es) returned no results: "
                + "; ".join(
                    f"{e['provider']}/{e['family']} ({e['error_text'] or 'no detail recorded'})"
                    for e in empty
                )
            )
        elif bh.get("fallback_pct", 0.0) > 20.0:
            b_status = "WARN"
        else:
            b_status = "OK"
        checks.append(
            {"subsystem": "Batch Economics", "status": b_status, "detail": " · ".join(parts)}
        )

    # Ensemble completeness (forecaster only)
    if data.forecaster_run_id:
        expected_models = _expected_spd_model_ids()
        present_models = set(data.spd_model_ids)
        missing_models = sorted(set(expected_models) - present_models)
        if not expected_models:
            e_status = "WARN"
            e_detail = "no ensemble configured"
        elif not missing_models:
            e_status = "OK"
            e_detail = f"{len(expected_models)}/{len(expected_models)} models"
        else:
            e_status = "FAIL"
            e_detail = f"{len(present_models)}/{len(expected_models)} models, missing: {', '.join(missing_models)}"
        # Member completeness: every model present somewhere is not the same
        # as every question aggregated from every member. Count the
        # (question, member) cells that produced no usable SPD.
        gaps = _member_gap_summary(data.question_run_metrics)
        if gaps["n_cells_missing"]:
            n_cells = gaps["n_cells_expected"]
            share = (gaps["n_cells_missing"] / n_cells) if n_cells else 0.0
            per_model = ", ".join(
                f"{m}: {c}" for m, c in sorted(gaps["by_model"].items(), key=lambda kv: (-kv[1], kv[0]))
            )
            e_detail += (
                f" · {gaps['n_cells_missing']} of {n_cells} Track-1 member SPDs missing "
                f"({per_model}) — {gaps['n_questions_affected']} question(s) aggregated from fewer members"
            )
            if e_status == "OK":
                e_status = "FAIL" if share > 0.10 else "WARN"
        checks.append({"subsystem": "SPD Ensemble", "status": e_status, "detail": e_detail})

        # Scenarios
        generated = sum(1 for r in data.scenario_status_rows if r.get("status") == "generated")
        failed_parse = sum(
            1 for r in data.scenario_status_rows if r.get("status") == "failed_parse"
        )
        missing_unexp = sum(
            1 for r in data.scenario_status_rows if r.get("status") == "missing_unexpected"
        )
        skipped = sum(
            1 for r in data.scenario_status_rows
            if str(r.get("status") or "").startswith("skipped_by_design")
        )
        if missing_unexp == 0 and failed_parse == 0:
            s_status = "OK"
        elif missing_unexp == 0:
            s_status = "WARN"
        else:
            s_status = "FAIL"
        s_detail = (
            f"{generated} generated, {failed_parse} failed parse, "
            f"{skipped} skipped, {missing_unexp} missing"
        )
        checks.append({"subsystem": "Scenarios", "status": s_status, "detail": s_detail})

    return checks


# ---------------------------------------------------------------------------
# Grounding spot-check sampler
# ---------------------------------------------------------------------------

def _sample_grounding_spot_checks(
    hs_rows: list[dict[str, Any]],
    question_rows: list[dict[str, Any]],
    n_hs: int = 3,
    n_q: int = 5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select representative samples for the executive summary."""
    import random

    def _pick_samples(rows: list[dict[str, Any]], n: int, id_key: str) -> list[dict[str, Any]]:
        if not rows or n <= 0:
            return []
        # Sort by n_verified desc to find extremes
        by_verified = sorted(rows, key=lambda r: int(r.get("n_verified") or 0), reverse=True)
        selected: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        # Pick one with most sources
        if by_verified:
            selected.append(by_verified[0])
            seen_ids.add(str(by_verified[0].get(id_key) or ""))

        # Pick one with zero/fewest sources
        zero_rows = [r for r in by_verified if int(r.get("n_verified") or 0) == 0]
        if zero_rows:
            candidate = zero_rows[0]
        else:
            candidate = by_verified[-1] if len(by_verified) > 1 else None
        if candidate and str(candidate.get(id_key) or "") not in seen_ids:
            selected.append(candidate)
            seen_ids.add(str(candidate.get(id_key) or ""))

        # Fill remaining with random picks
        remaining = [r for r in rows if str(r.get(id_key) or "") not in seen_ids]
        if remaining and len(selected) < n:
            random.shuffle(remaining)
            for r in remaining[: n - len(selected)]:
                selected.append(r)

        return selected[:n]

    hs_samples = _pick_samples(hs_rows, n_hs, "iso3")
    q_samples = _pick_samples(question_rows, n_q, "question_id")
    return hs_samples, q_samples


# ---------------------------------------------------------------------------
# Emitter: Executive Summary (for GITHUB_STEP_SUMMARY)
# ---------------------------------------------------------------------------

def _triage_breakdown(data: BundleData) -> dict[str, Any]:
    """The RC / triage / question-generation counts both markdown files use.

    Computed once and read twice: the summary prints the totals and
    coverage_detail.md prints the enumerations behind them, and two
    derivations of one number will eventually disagree.
    """

    _SILENCED_HAZARDS = {"DI", "CU", "HW"}
    rc_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    seasonal_skip_pairs: list[str] = []
    rc_promoted_pairs: list[tuple[str, str, int]] = []
    triage_quiet_pairs: list[tuple[str, str]] = []
    triage_priority_pairs: list[tuple[str, str]] = []
    rc_level_lookup: dict[tuple[str, str], int] = {}
    triage_tier_lookup: dict[tuple[str, str], str] = {}

    for row in data.hs_triage_detail_rows:
        hz = (row.get("hazard_code") or "").upper()
        iso3 = (row.get("iso3") or "").upper()
        if hz in _SILENCED_HAZARDS:
            continue

        rc_level = row.get("regime_change_level")
        dq_raw = row.get("data_quality_json") or ""
        status = ""
        if dq_raw:
            try:
                dq = json.loads(dq_raw) if isinstance(dq_raw, str) else dq_raw
                status = dq.get("status", "") if isinstance(dq, dict) else ""
            except Exception:
                pass

        if rc_level is not None:
            rc_level_lookup[(iso3, hz)] = int(rc_level)
        tier = (row.get("tier") or "").lower()
        if tier:
            triage_tier_lookup[(iso3, hz)] = tier

        if status == "seasonal_skip":
            seasonal_skip_pairs.append(f"{iso3}_{hz}")
            continue

        if rc_level is not None:
            lvl = int(rc_level)
            rc_counts[lvl] = rc_counts.get(lvl, 0) + 1
            if lvl >= 1:
                rc_promoted_pairs.append((iso3, hz, lvl))

        if status == "rc_promoted":
            continue

        if tier == "quiet":
            triage_quiet_pairs.append((iso3, hz))
        elif tier == "priority":
            triage_priority_pairs.append((iso3, hz))

    n_countries = len(data.resolved_countries_sorted)
    n_active_hazards = len(EXPECTED_HS_HAZARDS)
    total_expected = n_active_hazards * n_countries
    return {
        "rc_counts": rc_counts,
        "seasonal_skip_pairs": seasonal_skip_pairs,
        "seasonal_skip_count": len(seasonal_skip_pairs),
        "rc_promoted_pairs": rc_promoted_pairs,
        "triage_quiet_pairs": triage_quiet_pairs,
        "triage_priority_pairs": triage_priority_pairs,
        "rc_level_lookup": rc_level_lookup,
        "triage_tier_lookup": triage_tier_lookup,
        "n_countries": n_countries,
        "total_expected": total_expected,
        "n_assessed": total_expected - len(seasonal_skip_pairs),
    }


def _cost_lines(data: BundleData, collected: dict[str, Any]) -> list[str]:
    """Cost, with the batch-versus-sync delta stated rather than implied.

    A total is not actionable. What is actionable is the difference between
    what this run paid and what it would have paid at the other tier — a
    figure that in September was ten dollars and appeared in no artifact.
    """

    lines: list[str] = []
    lines.append("## Cost")
    lines.append("")
    lines.append("| Phase | Tokens | Cost (USD) |")
    lines.append("|-------|--------|------------|")
    total_tokens = 0.0
    total_cost = 0.0
    for phase in sorted(data.usage_by_phase.keys()):
        vals = data.usage_by_phase[phase]
        tokens = vals.get("total_tokens", 0.0)
        cost = vals.get("total_cost_usd", 0.0)
        total_tokens += tokens
        total_cost += cost
        lines.append(f"| {phase} | {int(tokens):,} | ${cost:.2f} |")
    lines.append(f"| **Total** | **{int(total_tokens):,}** | **${total_cost:.2f}** |")
    lines.append("")

    totals = (collected.get("batch_lifecycle") or {}).get("totals") or {}
    if totals:
        lost = float(totals.get("lost_discount_usd") or 0.0)
        lines.append(
            f"- Provider batches: {totals.get('n_batches', 0)}, "
            f"{totals.get('n_requests', 0)} requests, "
            f"{totals.get('n_fell_back_to_sync', 0)} fell back to synchronous "
            f"({totals.get('fallback_pct_of_requests', 0.0)}%)"
        )
        if lost > 0:
            lines.append(
                f"- **Batch discount lost: ${lost:.2f}** — spend on batchable families "
                "that ran at the synchronous full price. See batch_lifecycle.json."
            )
        else:
            lines.append("- Batch discount lost: $0.00 on batchable families")
        if totals.get("n_batches_yielded_nothing"):
            lines.append(
                f"- **{totals['n_batches_yielded_nothing']} batch(es) returned nothing** — "
                "every one of their requests paid full price"
            )
        if totals.get("max_queue_wall_seconds") is not None:
            lines.append(
                f"- Batch queue wall time: median "
                f"{totals.get('median_queue_wall_seconds')}s, max "
                f"{totals.get('max_queue_wall_seconds')}s"
            )
        lines.append("")
    return lines


def _cache_lines(collected: dict[str, Any]) -> list[str]:
    summary = collected.get("prompt_cache_summary") or {}
    lines = ["## Prompt cache", ""]
    if not summary.get("prompt_tokens"):
        lines.append("No prompt tokens recorded for this run.")
        lines.append("")
        return lines
    flags = summary.get("flags") or {}
    lines.append(
        f"- {summary['cache_hit_rate_pct']}% of {summary['prompt_tokens']:,} prompt tokens "
        f"read from cache (~${summary['estimated_saved_usd']:.2f} saved)"
    )
    lines.append(
        "- Flags: "
        + ", ".join(f"{k}={v}" for k, v in sorted(flags.items()))
    )
    by_provider = summary.get("by_provider") or {}
    if by_provider:
        lines.append("")
        lines.append("| Provider | Prompt tokens | Cache hits | Hit rate |")
        lines.append("|----------|---------------|------------|----------|")
        for provider, vals in sorted(by_provider.items()):
            pt = int(vals.get("prompt_tokens") or 0)
            hits = int(vals.get("hit_tokens") or 0)
            rate = f"{100.0 * hits / pt:.1f}%" if pt else "n/a"
            lines.append(f"| {provider or '(none)'} | {pt:,} | {hits:,} | {rate} |")
    lines.append("")
    lines.append(
        "_A flag being on proves nothing; only a non-zero cache read does. "
        "Detail per phase and model in prompt_cache_report.csv._"
    )
    lines.append("")
    return lines


def _connector_lines(collected: dict[str, Any]) -> list[str]:
    rows = collected.get("connector_rows") or []
    lines = ["## Connector freshness", ""]
    if not rows:
        lines.append("Not collected.")
        lines.append("")
        return lines
    # Only aged sources get a row here: an age column is the point of the
    # table, and an empty source has none. Empties are named on one line
    # below, and every one of them is in the anomaly list already.
    flagged = [r for r in rows if r.get("status") in ("stale", "warn")]
    empty = [r for r in rows if r.get("status") == "empty"]
    fresh = [r for r in rows if r.get("status") == "fresh"]
    lines.append(
        f"{len(fresh)} fresh, "
        f"{sum(1 for r in rows if r.get('status') == 'warn')} warn, "
        f"{sum(1 for r in rows if r.get('status') == 'stale')} stale, "
        f"{sum(1 for r in rows if r.get('status') == 'empty')} empty "
        f"(of {len(rows)} sources)."
    )
    lines.append("")
    if flagged:
        lines.append("| Source | Age (days) | Status | Labelled in prompt |")
        lines.append("|--------|-----------|--------|--------------------|")
        for row in sorted(flagged, key=lambda r: str(r.get("status"))):
            lines.append(
                f"| {row.get('source')} | {row.get('age_days_at_run_start')} "
                f"| {row.get('status')} | {row.get('prompt_staleness_warning')} |"
            )
        lines.append("")
    if empty:
        lines.append(
            "Empty tables: " + ", ".join(str(r.get("source")) for r in empty) + "."
        )
        lines.append("")

    cw = collected.get("crisiswatch") or {}
    if cw.get("available"):
        editions = ", ".join(
            f"{e['edition']} ({e['n_countries']})" for e in cw.get("editions_present") or []
        )
        lines.append(
            f"- **CrisisWatch (ACE inject)**: editions in table — {editions or 'none'}. "
            f"{cw.get('n_countries_with_arrow', 0)} countries carry an arrow, "
            f"{cw.get('n_countries_without_arrow', 0)} do not; "
            f"{cw.get('n_alerts', 0)} alerts."
        )
        arrows = ", ".join(
            f"{k}={v}" for k, v in sorted((cw.get("arrow_counts") or {}).items())
        )
        if arrows:
            lines.append(f"  - Arrows: {arrows}")
        lines.append(
            f"  - {cw.get('n_ace_questions_without_crisiswatch', 0)} ACE question(s) across "
            f"{cw.get('ace_countries_without_crisiswatch_row', 0)} country(ies) were forecast "
            "with no CrisisWatch row at all."
        )
        lines.append("")
    return lines


def emit_executive_summary(
    data: BundleData,
    out_dir: Path,
    *,
    collected: dict[str, Any] | None = None,
    health_checks: list[dict[str, Any]] | None = None,
) -> str:
    """The one page a reader can stop at when the run was clean.

    It leads with the anomalies because the old summary's failure was one
    of ordering as much as of content: the September run's four dead
    batches and three lost member forecasts were in the file, below a
    hundred and twenty lines of seasonal screen-outs and quiet pairs. Those
    enumerations are useful and they now live in coverage_detail.md.
    """

    collected = collected or {}
    lines: list[str] = []

    lines.append(f"# Pythia Pipeline Health — {data.out_run_id}")
    lines.append("")
    lines.append(f"_Generated at {data.now}_")
    lines.append("")
    lines.append(f"- HS run ID: `{data.hs_run_id or 'N/A'}`")
    lines.append(f"- Forecaster run ID: `{data.forecaster_run_id or 'N/A'}`")
    lines.append(f"- DB: `{data.db_url}`")
    lines.append(f"- DB SHA256: `{data.provenance_entry.get('db_sha256') or 'unknown'}`")
    lines.append("")

    # ----- Anomalies, first -----
    entries = collected.get("anomalies")
    lines.append("## Anomalies")
    lines.append("")
    if entries is None:
        lines.append("_Anomaly collection did not run._")
        lines.append("")
    elif not entries:
        lines.append("None. Every check passed and every collector ran.")
        lines.append("")
    else:
        counts = anomalies_mod.counts(entries)
        lines.append(
            f"**{counts.get('fail', 0)} fail · {counts.get('warn', 0)} warn · "
            f"{counts.get('info', 0)} info**"
        )
        lines.append("")
        lines.append("| Severity | Subsystem | What is wrong | Evidence |")
        lines.append("|----------|-----------|---------------|----------|")
        for entry in entries:
            description = str(entry.get("description") or "").replace("|", "\\|")
            lines.append(
                f"| {entry.get('severity', '').upper()} | {entry.get('subsystem')} "
                f"| {description} | `{entry.get('evidence_file')}` |"
            )
        lines.append("")

    # ----- Status -----
    checks = health_checks if health_checks is not None else _evaluate_pipeline_health(data)
    lines.append("## Pipeline status")
    lines.append("")
    lines.append("| Subsystem | Status | Detail |")
    lines.append("|-----------|--------|--------|")
    for check in checks:
        lines.append(f"| {check['subsystem']} | {check['status']} | {check['detail']} |")
    lines.append("")

    lines.extend(_cost_lines(data, collected))
    lines.extend(_cache_lines(collected))
    lines.extend(_connector_lines(collected))

    # ----- Coverage -----
    breakdown = _triage_breakdown(data)
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- Countries: {len(data.resolved_countries_sorted)}")
    lines.append(
        f"- Hazard-country pairs: {breakdown['total_expected']} expected, "
        f"{breakdown['seasonal_skip_count']} screened out seasonally, "
        f"{breakdown['n_assessed']} assessed by RC"
    )
    lines.append(
        "- RC levels (assessed): "
        + ", ".join(f"L{lvl}={breakdown['rc_counts'].get(lvl, 0)}" for lvl in range(4))
    )
    lines.append(
        f"- Triage: {len(breakdown['triage_priority_pairs'])} priority, "
        f"{len(breakdown['triage_quiet_pairs'])} quiet"
    )
    track1_qs = [q for q in data.questions if q.get("track") == 1]
    track2_qs = [q for q in data.questions if q.get("track") == 2]
    lines.append(
        f"- Questions: {len(data.questions)} "
        f"(Track 1: {len(track1_qs)}, Track 2: {len(track2_qs)})"
    )
    if data.forecaster_run_id:
        lines.append(f"- Forecasted: {data.lifecycle_counts.get('forecast', 0)}")
        lines.append(f"- Scenarios: {data.lifecycle_counts.get('scenario', 0)}")
        if data.researched_not_forecasted:
            lines.append(
                f"- Drop-offs (researched not forecasted): {len(data.researched_not_forecasted)}"
            )
    lines.append("")

    # ----- Ensemble -----
    if data.forecaster_run_id:
        lines.append("## Ensemble")
        lines.append("")
        lines.extend(_ensemble_participation_summary(data.forecasts_raw_counts, data.spd_model_ids))
        roll = collected.get("completeness_rollup") or {}
        if roll.get("n_cells_expected"):
            lines.append("")
            lines.append(
                f"- Member forecasts landed: "
                f"{roll['n_cells_expected'] - roll.get('n_cells_missing', 0)} of "
                f"{roll['n_cells_expected']} (question, model, month) cells; "
                f"{roll.get('n_question_months_short', 0)} question-months were aggregated "
                "from fewer members than expected. See model_completeness.csv."
            )
        lines.append("")

    # ----- Grounding, compact -----
    lines.append("## Grounding")
    lines.append("")
    for label, stats in [
        ("RC grounding", data.rc_grounding_call_stats),
        ("Triage grounding", data.triage_grounding_call_stats),
        ("Adversarial checks", data.adversarial_grounding_call_stats),
    ]:
        n_calls = stats.get("n_calls", 0)
        source_counts = stats.get("source_counts", [])
        with_sources = sum(1 for c in source_counts if c > 0)
        s = _compute_source_stats(source_counts)
        lines.append(
            f"- {label}: {n_calls} calls, {with_sources} with sources, "
            f"{n_calls - with_sources} empty, avg {s['avg']:.1f} sources"
        )
    breaker = collected.get("breaker") or {}
    if breaker.get("no_backend_calls"):
        lines.append(
            f"- {breaker['no_backend_calls']} call(s) reached no backend at all "
            f"({breaker.get('brave_breaker_short_circuits', 0)} of them a Brave breaker trip)"
        )
    lines.append("")

    # ----- DB provenance -----
    lines.append("## DB provenance")
    lines.append("")
    lines.append("| Table | Before | After | Delta |")
    lines.append("|-------|--------|-------|-------|")
    for tbl in KEY_TABLES:
        before = data.counts_before.get(tbl)
        after = data.counts_after.get(tbl)
        delta = ""
        if before is not None and after is not None:
            d = after - before
            delta = f"+{d}" if d >= 0 else str(d)
        lines.append(
            f"| {tbl} | {before if before is not None else 'n/a'} "
            f"| {after if after is not None else 'n/a'} | {delta} |"
        )
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Coverage enumerations (seasonal screen-outs, quiet pairs, the full Track 1 and "
        "Track 2 question tables, structured-data injects, scenarios) are in "
        "`coverage_detail__<run>.md`. Every file in this bundle is described in "
        "`BUNDLE_MANIFEST.json`."
    )

    md = "\n".join(lines)
    out_path = out_dir / f"executive_summary__{data.out_run_id}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote executive summary to {out_path}")
    return md


def emit_coverage_detail_markdown(data: BundleData, out_dir: Path) -> str:
    """The long enumerations, out of the way of the failures.

    They are worth keeping — a seasonal screen-out list is how you check
    the TC calendar did what it should — but 124 of them above a batch
    collapse is how a batch collapse goes unread.
    """

    breakdown = _triage_breakdown(data)
    lines: list[str] = []
    lines.append(f"# Coverage detail — {data.out_run_id}")
    lines.append("")
    lines.append(f"_Generated at {data.now}. Summary and failures: executive_summary__{data.out_run_id}.md_")
    lines.append("")

    # ----- RC -----
    lines.append("## RC assessment")
    lines.append("")
    lines.append("Unit: 1 row = 1 hazard-country pair.")
    lines.append(
        f"Assessed hazards: {', '.join(EXPECTED_HS_HAZARDS)} "
        f"({len(EXPECTED_HS_HAZARDS)} per country × {breakdown['n_countries']} countries "
        f"= {breakdown['total_expected']})."
    )
    lines.append(f"→ {breakdown['n_assessed']} pairs assessed by the RC LLM")
    lines.append("")
    lines.append("| RC Level | Count | Meaning |")
    lines.append("|----------|-------|---------|")
    _rc_meanings = {
        0: "Baseline → triage", 1: "Watch → Track 1",
        2: "Elevated → Track 1", 3: "Critical → Track 1",
    }
    for level in range(4):
        lines.append(f"| {level} | {breakdown['rc_counts'].get(level, 0)} | {_rc_meanings[level]} |")
    lines.append("")
    if breakdown["seasonal_skip_pairs"]:
        lines.append(f"### Seasonal screen-outs ({breakdown['seasonal_skip_count']})")
        lines.append("")
        lines.append(", ".join(sorted(breakdown["seasonal_skip_pairs"])))
        lines.append("")
    if breakdown["rc_promoted_pairs"]:
        lines.append(f"### RC Level ≥1, promoted to Track 1 ({len(breakdown['rc_promoted_pairs'])})")
        lines.append("")
        for iso3, hz, lvl in sorted(breakdown["rc_promoted_pairs"]):
            lines.append(f"- {iso3}: {hz} (L{lvl})")
        lines.append("")

    # ----- Triage -----
    lines.append("## Triage")
    lines.append("")
    n_promoted = len(breakdown["rc_promoted_pairs"])
    lines.append(
        f"Input: {breakdown['n_assessed'] - n_promoted} RC Level 0 pairs "
        f"({breakdown['n_assessed']} assessed − {n_promoted} RC-promoted)"
    )
    lines.append("")
    lines.append("| Tier | Count |")
    lines.append("|------|-------|")
    lines.append(f"| Priority | {len(breakdown['triage_priority_pairs'])} → Track 2 questions |")
    lines.append(f"| Quiet | {len(breakdown['triage_quiet_pairs'])} → no questions |")
    lines.append("")
    if breakdown["triage_priority_pairs"]:
        lines.append("### Priority")
        lines.append("")
        for iso3, hz in sorted(breakdown["triage_priority_pairs"]):
            lines.append(f"- {iso3}: {hz}")
        lines.append("")
    if breakdown["triage_quiet_pairs"]:
        lines.append(f"### Quiet ({len(breakdown['triage_quiet_pairs'])})")
        lines.append("")
        lines.append(
            ", ".join(f"{iso3}_{hz}" for iso3, hz in sorted(breakdown["triage_quiet_pairs"]))
        )
        lines.append("")

    # ----- Question generation -----
    lines.append("## Question generation")
    lines.append("")
    lines.append("Metric rules per hazard:")
    lines.append("- ACE → PA + FATALITIES (2 questions)")
    lines.append("- DR → EVENT_OCCURRENCE [+ PHASE3PLUS_IN_NEED if a food-security country] (1-2)")
    lines.append("- FL → PA + EVENT_OCCURRENCE (2 questions)")
    lines.append("- TC → PA + EVENT_OCCURRENCE (2 questions)")
    lines.append("")

    from collections import defaultdict

    track1_qs = [q for q in data.questions if q.get("track") == 1]
    track2_qs = [q for q in data.questions if q.get("track") == 2]
    _t1_by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    _t2_by_pair: dict[tuple[str, str], list[str]] = defaultdict(list)
    for q in track1_qs:
        _t1_by_pair[
            ((q.get("iso3") or "").upper(), (q.get("hazard_code") or "").upper())
        ].append(q.get("question_id") or "?")
    for q in track2_qs:
        _t2_by_pair[
            ((q.get("iso3") or "").upper(), (q.get("hazard_code") or "").upper())
        ].append(q.get("question_id") or "?")

    lines.append(
        f"### Track 1 ({len(track1_qs)} questions from {len(_t1_by_pair)} hazard-country pairs)"
    )
    lines.append("")
    lines.append("| Source pair | RC Level | Questions generated |")
    lines.append("|-------------|----------|---------------------|")
    for (iso3, hz), qids in sorted(_t1_by_pair.items()):
        lvl = breakdown["rc_level_lookup"].get((iso3, hz), 0)
        lines.append(f"| {iso3}_{hz} | L{lvl} | {', '.join(sorted(qids))} |")
    lines.append("")

    lines.append(
        f"### Track 2 ({len(track2_qs)} questions from {len(_t2_by_pair)} hazard-country pairs)"
    )
    lines.append("")
    lines.append("| Source pair | Triage tier | Questions generated |")
    lines.append("|-------------|-------------|---------------------|")
    for (iso3, hz), qids in sorted(_t2_by_pair.items()):
        tier = breakdown["triage_tier_lookup"].get((iso3, hz), "priority")
        lines.append(f"| {iso3}_{hz} | {tier.title()} | {', '.join(sorted(qids))} |")
    lines.append("")

    lines.append("### By hazard")
    lines.append("")
    lines.append("| Hazard | Total | Track 1 | Track 2 |")
    lines.append("|--------|-------|---------|---------|")
    for hz in EXPECTED_HS_HAZARDS:
        hz_qs = [q for q in data.questions if (q.get("hazard_code") or "").upper() == hz]
        lines.append(
            f"| {hz} | {len(hz_qs)} "
            f"| {sum(1 for q in hz_qs if q.get('track') == 1)} "
            f"| {sum(1 for q in hz_qs if q.get('track') == 2)} |"
        )
    lines.append("")

    # ----- Structured data injects -----
    if data.structured_data_coverage:
        lines.append("## Structured data injects")
        lines.append("")
        lines.append("| Source | Yes | No | Missing |")
        lines.append("|--------|-----|----|---------|")
        for label in [
            "ACLED fatalities", "IDMC displacement", "IFRC PA",
            "Conflict forecasts", "FEWS NET IPC", "IPC API",
            "ReliefWeb reports", "HDX Signals", "ENSO", "Seasonal TC",
            "NMME", "CrisisWatch", "GDACS events",
        ]:
            yes, no, missing = data.structured_data_coverage.get(label, (0, 0, []))
            lines.append(f"| {label} | {yes} | {no} | {', '.join(missing)} |")
        lines.append("")

    # ----- Scenarios -----
    if data.forecaster_run_id and data.scenario_status_rows:
        lines.append("## Scenarios")
        lines.append("")
        lines.append("Track 1 only. 1 scenario per Track 1 question.")
        track1_qids = {q.get("question_id") for q in data.questions if q.get("track") == 1}
        t1 = [r for r in data.scenario_status_rows if r.get("question_id") in track1_qids]
        t1_yes = sum(1 for r in t1 if r.get("status") == "generated")
        t1_failed = sum(1 for r in t1 if r.get("status") == "failed_parse")
        lines.append(f"Generated: {t1_yes} / {len(t1)}")
        lines.append(f"Failed to parse (attempted, not stored): {t1_failed}")
        lines.append(f"Missing: {len(t1) - t1_yes - t1_failed}")
        lines.append("")

    # ----- Latency + slowest questions -----
    if data.latency_block:
        lines.append(data.latency_block)
        lines.append("")

    if data.question_run_metrics:
        lines.append("## Slowest questions")
        lines.append("")
        lines.append("| Question | Wall (s) | ISO3 | Hazard |")
        lines.append("|----------|----------|------|--------|")

        def _wall_or_compute_ms(r: dict) -> float:
            return float(r.get("wall_ms") or r.get("compute_ms") or 0.0)

        for row in sorted(data.question_run_metrics, key=lambda r: -_wall_or_compute_ms(r))[:15]:
            suffix = "" if row.get("wall_ms") else " (compute)"
            lines.append(
                f"| {row.get('question_id')} | {_wall_or_compute_ms(row) / 1000.0:.1f}{suffix} "
                f"| {row.get('iso3') or ''} | {row.get('hazard_code') or ''} |"
            )
        lines.append("")

    # ----- Grounding spot-checks -----
    hs_samples, q_samples = _sample_grounding_spot_checks(
        data.hs_web_rows, data.question_web_rows, n_hs=3, n_q=5
    )
    if hs_samples or q_samples:
        lines.append("## Grounding spot-checks")
        lines.append("")
    if hs_samples:
        lines.append("### HS country packs (sample)")
        lines.append("")
        lines.append("| ISO3 | Grounded | Sources | Backend | Sample URL |")
        lines.append("|------|----------|---------|---------|------------|")
        for row in hs_samples:
            urls = row.get("top_verified_urls") or []
            lines.append(
                f"| {row.get('iso3')} | {row.get('grounded')} | {row.get('n_verified', 0)} "
                f"| {row.get('selected_backend') or ''} | {urls[0][:80] if urls else '(none)'} |"
            )
        lines.append("")
    if q_samples:
        lines.append("### Question evidence (sample)")
        lines.append("")
        lines.append("| Question | Grounded | Sources | Backend | Sample URL |")
        lines.append("|----------|----------|---------|---------|------------|")
        for row in q_samples:
            urls = row.get("top_verified_urls") or []
            lines.append(
                f"| {row.get('question_id')} | {row.get('grounded')} | {row.get('n_verified', 0)} "
                f"| {row.get('selected_backend') or ''} | {urls[0][:80] if urls else '(none)'} |"
            )
        lines.append("")

    md = "\n".join(lines)
    out_path = out_dir / f"coverage_detail__{data.out_run_id}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote coverage detail to {out_path}")
    return md


# ---------------------------------------------------------------------------
# Emitter: Health Report JSON
# ---------------------------------------------------------------------------

def _build_food_security_health(data: BundleData) -> dict[str, Any]:
    """Build food security health section for the JSON health report."""
    fewsnet_cov, *_ = data.structured_data_coverage.get("FEWS NET IPC", (0, 0, []))
    ipc_cov, *_ = data.structured_data_coverage.get("IPC API", (0, 0, []))
    return {
        "fewsnet_countries": fewsnet_cov,
        "ipc_countries": ipc_cov,
        "countries_without_food_security": data.food_security_no_data_countries,
    }


def emit_health_report_json(data: BundleData, out_dir: Path) -> None:
    """Write machine-parseable health data to JSON."""

    # Error categorisation across all LLM errors
    error_categories: dict[str, dict[str, int]] = {}
    for row in data.llm_error_rows:
        phase = row.get("phase") or "unknown"
        # We only have aggregate counts here, not individual error texts.
        # For detailed categorisation, we'd need the raw error texts.
        key = f"{phase}/{row.get('provider')}/{row.get('model_id')}"
        error_categories[key] = {
            "phase": phase,
            "provider": row.get("provider") or "",
            "model_id": row.get("model_id") or "",
            "n_errors": int(row.get("n_errors") or 0),
        }

    # Grounding health
    grounding_health: dict[str, Any] = {}
    for label, rows, failures in [
        ("hs_web_research", data.hs_web_research_rows, data.hs_web_research_failures),
        ("research_web_research", data.research_web_research_rows, data.research_web_research_failures),
        ("forecast_web_research", data.self_search_rows, data.self_search_failures),
    ]:
        total_calls = sum(int(r.get("n_calls") or 0) for r in (rows or []))
        total_errors = sum(int(r.get("n_errors") or 0) for r in (rows or []))
        total_verified = sum(int(r.get("n_verified_sources") or 0) for r in (rows or []))
        grounding_health[label] = {
            "n_calls": total_calls,
            "n_errors": total_errors,
            "n_verified_sources": total_verified,
            "failures": [
                {
                    "provider": f.get("provider"),
                    "model_id": f.get("model_id"),
                    "error_code": f.get("error_code"),
                    "error_category": _categorise_error(f.get("error_message") or ""),
                    "error_message": (f.get("error_message") or "")[:200],
                }
                for f in (failures or [])
            ],
        }

    # RC grounding health (from llm_calls with hazard_code like GROUNDING_*).
    #
    # model_id sentinels written when a grounding attempt never reached a backend.
    # A non-zero count here means those country-hazard pairs got NO grounding
    # evidence at all — a breaker-tripped count is the Brave budget gate firing.
    _GROUNDING_NO_BACKEND_REASONS = {
        GROUNDING_FAILED_MODEL_ID: "all_backends_failed",
        GROUNDING_BREAKER_MODEL_ID: "brave_circuit_breaker_tripped",
        GROUNDING_UNAVAILABLE_MODEL_ID: "backend_unavailable",
    }

    def _grounding_subsystem_health(rows: list[dict[str, Any]]) -> dict[str, Any]:
        total_calls = sum(int(r.get("n_calls") or 0) for r in (rows or []))
        total_errors = sum(int(r.get("n_errors") or 0) for r in (rows or []))
        total_verified = sum(int(r.get("n_verified_sources") or 0) for r in (rows or []))
        no_backend: dict[str, int] = {}
        for r in rows or []:
            reason = _GROUNDING_NO_BACKEND_REASONS.get(str(r.get("model_id") or "").strip())
            if reason:
                no_backend[reason] = no_backend.get(reason, 0) + int(r.get("n_calls") or 0)
        return {
            "n_calls": total_calls,
            "n_errors": total_errors,
            "n_verified_sources": total_verified,
            "n_no_backend": sum(no_backend.values()),
            "no_backend_by_reason": no_backend,
            "by_provider_model": [
                {
                    "provider": r.get("provider"),
                    "model_id": r.get("model_id"),
                    "n_calls": int(r.get("n_calls") or 0),
                    "n_errors": int(r.get("n_errors") or 0),
                    "n_verified_sources": int(r.get("n_verified_sources") or 0),
                }
                for r in (rows or [])
            ],
        }
    rc_grounding_health = _grounding_subsystem_health(data.rc_grounding_rows)
    triage_grounding_health = _grounding_subsystem_health(data.triage_grounding_rows)

    # LLM health
    llm_health = {
        "by_phase_provider_model": [
            {
                "phase": row.get("phase"),
                "provider": row.get("provider"),
                "model_id": row.get("model_id"),
                "n_calls": int(row.get("n_calls") or 0),
                "n_errors": int(row.get("n_errors") or 0),
                "error_rate": round(
                    int(row.get("n_errors") or 0) / max(int(row.get("n_calls") or 0), 1), 4
                ),
            }
            for row in data.llm_call_counts
        ],
    }

    # Cost summary
    cost_summary = {
        "by_phase": {
            phase: {
                "total_tokens": int(vals.get("total_tokens", 0)),
                "total_cost_usd": round(vals.get("total_cost_usd", 0.0), 4),
            }
            for phase, vals in data.usage_by_phase.items()
        },
        "total_tokens": int(sum(v.get("total_tokens", 0) for v in data.usage_by_phase.values())),
        "total_cost_usd": round(
            sum(v.get("total_cost_usd", 0.0) for v in data.usage_by_phase.values()), 4
        ),
    }

    # Coverage funnel
    coverage = {
        "countries": len(data.resolved_countries_sorted),
        "questions_seeded": len(data.question_ids),
        "researched": data.lifecycle_counts.get("research", 0),
        "forecasted": data.lifecycle_counts.get("forecast", 0),
        "scenarios": data.lifecycle_counts.get("scenario", 0),
        "researched_not_forecasted": data.researched_not_forecasted,
    }

    # CrisisWatch health detail
    cw_health: dict[str, Any] = {"table_exists": data.crisiswatch_table_exists}
    if data.crisiswatch_load_error:
        cw_health["error"] = data.crisiswatch_load_error
    elif data.crisiswatch_entries:
        entries = data.crisiswatch_entries
        arrows = [e.get("arrow") or "" for e in entries]
        cw_health["total_countries"] = len(entries)
        cw_health["arrow_counts"] = {
            "deteriorated": sum(1 for a in arrows if a == "deteriorated"),
            "improved": sum(1 for a in arrows if a == "improved"),
            "unchanged": sum(1 for a in arrows if a == "unchanged"),
            "missing": sum(1 for a in arrows if not a),
        }
        cw_health["alert_counts"] = {
            "conflict_risk": sum(
                1 for e in entries if e.get("alert_type") == "conflict_risk"
            ),
            "resolution_opportunity": sum(
                1 for e in entries if e.get("alert_type") == "resolution_opportunity"
            ),
        }
        months = sorted({(e.get("year"), e.get("month")) for e in entries})
        cw_health["months_covered"] = [
            f"{y}-{m:02d}" if m else str(y) for y, m in months
        ]
        # Per-country detail for deteriorated and alert countries (most actionable)
        notable = [
            {
                "iso3": e.get("iso3"),
                "country": e.get("country_name"),
                "arrow": e.get("arrow"),
                "alert_type": e.get("alert_type"),
                "summary": (e.get("summary") or "")[:200],
            }
            for e in entries
            if e.get("arrow") == "deteriorated" or e.get("alert_type")
        ]
        cw_health["notable_entries"] = notable
        # Countries in the run that have NO CrisisWatch data (potential coverage gaps)
        cw_iso3s = {(e.get("iso3") or "").upper() for e in entries}
        missing_cw = sorted(
            iso3 for iso3 in data.resolved_countries_sorted
            if iso3.upper() not in cw_iso3s
        )
        cw_health["countries_without_crisiswatch"] = missing_cw
        cw_health["n_countries_without_crisiswatch"] = len(missing_cw)
    else:
        cw_health["total_countries"] = 0
        cw_health["detail"] = (
            "No CrisisWatch entries found. Possible causes: "
            "(1) Gemini grounding returned no results for both On the Horizon "
            "and Global Overview queries; "
            "(2) fallback JSON at horizon_scanner/data/crisiswatch_latest.json "
            "is empty or missing; "
            "(3) crisiswatch.fetch_crisiswatch() was not called in this run"
        )

    # Spot checks
    hs_samples, q_samples = _sample_grounding_spot_checks(
        data.hs_web_rows, data.question_web_rows
    )

    report: dict[str, Any] = {
        "run_identity": {
            "hs_run_id": data.hs_run_id,
            "forecaster_run_id": data.forecaster_run_id,
            "db_url": data.db_url,
            "generated_at": data.now,
        },
        "db_provenance": {
            "sha256": data.provenance_entry.get("db_sha256"),
            "size_bytes": data.provenance_entry.get("db_size_bytes"),
            "row_counts_before": data.counts_before,
            "row_counts_after": {k: v for k, v in data.counts_after.items()},
        },
        "health_checks": [
            {"subsystem": c["subsystem"], "status": c["status"], "detail": c["detail"]}
            for c in _evaluate_pipeline_health(data)
        ],
        "grounding_health": grounding_health,
        "rc_grounding_health": rc_grounding_health,
        "triage_grounding_health": triage_grounding_health,
        "llm_health": llm_health,
        "cost_summary": cost_summary,
        "coverage_funnel": coverage,
        "crisiswatch_health": cw_health,
        "food_security_health": _build_food_security_health(data),
        "grounding_spot_checks": {
            "hs_countries": [
                {
                    "iso3": r.get("iso3"),
                    "grounded": r.get("grounded"),
                    "n_verified": r.get("n_verified"),
                    "backend": r.get("selected_backend"),
                    "sample_urls": (r.get("top_verified_urls") or [])[:3],
                }
                for r in hs_samples
            ],
            "questions": [
                {
                    "question_id": r.get("question_id"),
                    "grounded": r.get("grounded"),
                    "n_verified": r.get("n_verified"),
                    "backend": r.get("selected_backend"),
                    "sample_urls": (r.get("top_verified_urls") or [])[:3],
                }
                for r in q_samples
            ],
        },
    }

    out_path = out_dir / f"health_report__{data.out_run_id}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote health report to {out_path}")


# ---------------------------------------------------------------------------
# Emitter: Question Metrics CSV
# ---------------------------------------------------------------------------

def _load_batched_call_counts(
    con: duckdb.DuckDBPyConnection, run_id: str | None
) -> dict[str, dict[str, int]]:
    """{question_id: {n_calls, n_batched}} for one forecaster run.

    A batched call is replayed from the provider batch and records
    elapsed_ms=0 with the replay timestamp, so a question whose calls were
    all batched legitimately has compute_ms=0 and a wall clock that measures
    the collect stage, not the models. This count says which case a row is.
    """

    if not run_id:
        return {}
    llm_columns = _llm_calls_columns(con)
    if "question_id" not in llm_columns:
        return {}
    batched_expr = (
        "COUNT(*) FILTER (WHERE usage_json LIKE '%\"service_tier\": \"batch\"%')"
        if "usage_json" in llm_columns
        else "0"
    )
    try:
        rows = _fetch_llm_rows(
            con,
            f"""
            SELECT question_id, COUNT(*) AS n_calls, {batched_expr} AS n_batched
            FROM llm_calls
            WHERE run_id = ? AND question_id IS NOT NULL AND question_id <> ''
            GROUP BY question_id
            """,
            [run_id],
        )
    except Exception:
        return {}
    return {
        str(r.get("question_id")): {
            "n_calls": int(r.get("n_calls") or 0),
            "n_batched": int(r.get("n_batched") or 0),
        }
        for r in rows
        if r.get("question_id")
    }


def _question_target_month(question: dict[str, Any]) -> str:
    """The month the question is ABOUT — its own epoch, not its last horizon.

    ``questions.target_month`` is the sixth window month by convention, and
    that column is published as ``horizon_end_month``. What a reader means
    by the target month is the epoch in the question id (``..._2026-10``),
    which is the window start. Derived from ``window_start_date`` where
    there is one and from the id suffix otherwise, so a row is never blank
    merely because one of the two is missing.
    """

    window_start = str(question.get("window_start_date") or "")
    if len(window_start) >= 7 and window_start[4] == "-":
        return window_start[:7]
    qid = str(question.get("question_id") or "")
    match = re.search(r"_(\d{4}-\d{2})$", qid)
    return match.group(1) if match else ""


def _reported_triage_tier(scenario_tier: Any, triage_row: dict[str, Any] | None) -> str:
    """The tier, or an honest statement that triage never ran.

    An RC-promoted hazard goes straight to Track 1 and its triage LLM calls
    are skipped, so the tier column on its row is whatever the synthetic
    defaults happened to carry — ``quiet`` until 2026-09-03. Printing that
    made all 105 Track-1 questions of the September run read as quiet
    hazards that got a full ensemble. Where the row says it was promoted,
    say ``not_triaged``.
    """

    row = triage_row or {}
    tier = str(scenario_tier or row.get("tier") or "").strip()
    status = ""
    raw = row.get("data_quality_json") or ""
    if raw:
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(parsed, dict):
                status = str(parsed.get("status") or "")
        except Exception:
            status = ""
    if status == "rc_promoted" or tier == "rc_promoted":
        return "not_triaged"
    if not tier:
        return ""
    # Pre-2026-09-03 rows carry the synthetic default with no marker of
    # their own; RC level >= 1 with no triage status is that case.
    try:
        rc_level = int(row.get("rc_level")) if row.get("rc_level") is not None else 0
    except Exception:
        rc_level = 0
    if rc_level >= 1 and status == "":
        return "not_triaged"
    return tier


def emit_question_metrics_csv(
    data: BundleData, con: duckdb.DuckDBPyConnection, out_dir: Path
) -> None:
    """Write per-question metrics to CSV."""
    if not data.questions:
        return

    scenario_by_qid = {str(r["question_id"]): r for r in data.scenario_status_rows}
    metrics_by_qid = {str(r["question_id"]): r for r in data.question_run_metrics}
    web_by_qid = {str(r.get("question_id")): r for r in data.question_web_rows}

    # Build triage lookup by (iso3, hazard_code) for the new columns
    triage_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    if data.hs_run_id:
        try:
            triage_rows = _fetch_llm_rows(
                con,
                """
                SELECT iso3, hazard_code, triage_score,
                       regime_change_likelihood AS rc_likelihood,
                       regime_change_level AS rc_level,
                       regime_change_direction AS rc_direction,
                       track, tier, data_quality_json
                FROM hs_triage
                WHERE run_id = ?
                """,
                [data.hs_run_id],
            )
            for tr in triage_rows:
                key = (
                    str(tr.get("iso3") or "").upper(),
                    str(tr.get("hazard_code") or "").upper(),
                )
                triage_by_key[key] = tr
        except Exception:
            pass

    # `questions.target_month` holds the LAST month of the six-month window
    # (the month-anchoring convention: horizon 1 = window_start_date), so
    # printing it under a column called target_month put 2027-03 beside a
    # question id reading 2026-10 on all 190 rows and every reader took it
    # for a bug. It is now published as `horizon_end_month`, which is what
    # it is, and `target_month` carries the question's OWN month — the epoch
    # in its id, i.e. the first window month. No column was removed.
    fieldnames = [
        "question_id", "iso3", "hazard_code", "metric",
        "window_start_date", "target_month", "horizon_end_month",
        "triage_tier", "spd_status", "scenario_status",
        "wall_ms", "compute_ms", "queue_ms", "n_llm_calls", "n_batched_calls", "cost_usd",
        "n_spd_models_expected", "n_spd_models_ok", "missing_model_ids",
        "research_grounded", "n_verified_sources", "n_unverified_sources",
        "triage_score", "rc_likelihood", "rc_level", "rc_direction", "track",
    ]
    batched_by_qid = _load_batched_call_counts(con, data.forecaster_run_id)

    def _cell(value: Any) -> Any:
        # 0 is a measurement (a batched replay has no client-side duration);
        # only None is "unknown". `or ""` turned every Track-2 timing into
        # a blank on 2026-09-01.
        return "" if value is None else value

    out_path = out_dir / f"question_metrics__{data.out_run_id}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for q in sorted(
            data.questions,
            key=lambda r: (r.get("iso3") or "", r.get("hazard_code") or "", r.get("question_id") or ""),
        ):
            qid = str(q.get("question_id") or "")
            scenario = scenario_by_qid.get(qid, {})
            metrics = metrics_by_qid.get(qid, {})
            web = web_by_qid.get(qid, {})

            triage_key = (
                str(q.get("iso3") or "").upper(),
                str(q.get("hazard_code") or "").upper(),
            )
            triage = triage_by_key.get(triage_key, {})

            writer.writerow({
                "question_id": qid,
                "iso3": q.get("iso3") or "",
                "hazard_code": q.get("hazard_code") or "",
                "metric": q.get("metric") or "",
                "window_start_date": str(q.get("window_start_date") or ""),
                "target_month": _question_target_month(q),
                "horizon_end_month": q.get("target_month") or "",
                "triage_tier": _reported_triage_tier(
                    scenario.get("triage_tier"), triage
                ),
                "spd_status": _load_spd_status(con, data.forecaster_run_id, qid) or "",
                "scenario_status": scenario.get("status") or "",
                "wall_ms": _cell(metrics.get("wall_ms")),
                "compute_ms": _cell(metrics.get("compute_ms")),
                "queue_ms": _cell(metrics.get("queue_ms")),
                "n_llm_calls": _cell((batched_by_qid.get(qid) or {}).get("n_calls")),
                "n_batched_calls": _cell((batched_by_qid.get(qid) or {}).get("n_batched")),
                "cost_usd": _cell(metrics.get("cost_usd")),
                "n_spd_models_expected": _cell(metrics.get("n_spd_models_expected")),
                "n_spd_models_ok": _cell(metrics.get("n_spd_models_ok")),
                "missing_model_ids": metrics.get("missing_model_ids_json") or "",
                "research_grounded": web.get("grounded", ""),
                "n_verified_sources": web.get("n_verified", ""),
                "n_unverified_sources": web.get("n_unverified", ""),
                "triage_score": triage.get("triage_score") or "",
                "rc_likelihood": triage.get("rc_likelihood") or "",
                "rc_level": triage.get("rc_level") or "",
                "rc_direction": triage.get("rc_direction") or "",
                "track": triage.get("track") or q.get("track") or "",
            })
    print(f"Wrote question metrics to {out_path}")


# ---------------------------------------------------------------------------
# Emitter: Evidence Packs CSV
# ---------------------------------------------------------------------------

def emit_evidence_packs_csv(data: BundleData, out_dir: Path) -> None:
    """Write HS country + question evidence to CSV files."""

    # HS country evidence
    if data.hs_web_rows:
        hs_fieldnames = [
            "iso3", "grounded", "n_verified", "n_unverified",
            "backend", "top_3_urls", "error_code",
            "groundingSupports_count", "groundingChunks_count",
        ]
        hs_path = out_dir / f"hs_country_evidence__{data.out_run_id}.csv"
        with open(hs_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=hs_fieldnames)
            writer.writeheader()
            for row in sorted(data.hs_web_rows, key=lambda r: r.get("iso3") or ""):
                urls = (row.get("top_verified_urls") or [])[:3]
                errors = row.get("last_errors") or []
                writer.writerow({
                    "iso3": row.get("iso3") or "",
                    "grounded": row.get("grounded", False),
                    "n_verified": row.get("n_verified", 0),
                    "n_unverified": row.get("n_unverified", 0),
                    "backend": row.get("selected_backend") or "",
                    "top_3_urls": ";".join(urls),
                    "error_code": row.get("reason_code") or "",
                    "groundingSupports_count": row.get("groundingSupports_count", 0),
                    "groundingChunks_count": row.get("groundingChunks_count", 0),
                })
        print(f"Wrote HS country evidence to {hs_path}")

    # Question evidence
    if data.question_web_rows:
        q_fieldnames = [
            "question_id", "grounded", "n_verified", "n_unverified",
            "backend", "top_3_urls", "error_code",
            "groundingSupports_count", "groundingChunks_count",
        ]
        q_path = out_dir / f"question_evidence__{data.out_run_id}.csv"
        with open(q_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=q_fieldnames)
            writer.writeheader()
            for row in sorted(data.question_web_rows, key=lambda r: r.get("question_id") or ""):
                urls = (row.get("top_verified_urls") or [])[:3]
                errors = row.get("last_errors") or []
                writer.writerow({
                    "question_id": row.get("question_id") or "",
                    "grounded": row.get("grounded", False),
                    "n_verified": row.get("n_verified", 0),
                    "n_unverified": row.get("n_unverified", 0),
                    "backend": row.get("selected_backend") or "",
                    "top_3_urls": ";".join(urls),
                    "error_code": "",
                    "groundingSupports_count": row.get("groundingSupports_count", 0),
                    "groundingChunks_count": row.get("groundingChunks_count", 0),
                })
        print(f"Wrote question evidence to {q_path}")


# ---------------------------------------------------------------------------
# Emitter: LLM Calls Detail JSONL (gzipped)
# ---------------------------------------------------------------------------

def emit_llm_calls_detail_jsonl(
    data: BundleData, con: duckdb.DuckDBPyConnection, out_dir: Path
) -> None:
    """Write per-call LLM detail to gzipped JSONL, with prefixes stored once.

    This file was 9 MB of a 9.2 MB bundle and nearly all of it was the same
    static prompt prefix repeated — the V3 prompt order puts the role, the
    buckets, the hazard guidance and the schema first, identical across
    every question in a family. Each distinct prefix now goes once into
    ``prompt_prefixes__<run>.json`` and each record carries its hash plus
    its own variable tail; joining the two reproduces the prompt byte for
    byte, and each record says so. Responses are kept in full.
    """

    out_path = out_dir / f"llm_calls_detail__{data.out_run_id}.jsonl.gz"

    # Build a query that gets all calls for this run
    if data.predicate:
        query = f"SELECT * FROM llm_calls WHERE {data.predicate}"
        params = data.predicate_params
    else:
        return  # No predicate means no calls to export

    try:
        rows = _fetch_llm_rows(con, query, params)
    except Exception:
        return

    prefixes: dict[str, dict[str, Any]] = {}
    assignment: dict[Any, tuple[str, int]] = {}
    try:
        prefixes, assignment = prompt_prefixes.build_prefix_index(rows)
    except Exception as exc:  # noqa: BLE001
        # Deduplication is an optimisation; losing it must never cost the
        # file. Full prompts are written and the reason is recorded.
        LOG.warning("prompt prefix indexing failed, writing full prompts: %s", exc)

    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        for row in rows:
            record = {
                "call_id": row.get("call_id"),
                "question_id": row.get("question_id"),
                "iso3": row.get("iso3"),
                "hazard_code": row.get("hazard_code"),
                "phase": row.get("phase"),
                "call_type": row.get("call_type"),
                "provider": row.get("provider"),
                "model_id": row.get("model_id"),
                "prompt_text": row.get("prompt_text") or "",
                "response_text": row.get("response_text") or "",
                "error_text": row.get("error_text") or "",
                "elapsed_ms": row.get("elapsed_ms"),
                "cost_usd": row.get("cost_usd"),
                "usage_json": row.get("usage_json"),
                "timestamp": str(row.get("timestamp") or ""),
                "run_id": row.get("run_id"),
                "hs_run_id": row.get("hs_run_id"),
            }
            if assignment:
                record = prompt_prefixes.apply(record, row, assignment)
            f.write(json.dumps(record, default=str) + "\n")

    prefix_path = out_dir / f"prompt_prefixes__{data.out_run_id}.json"
    payload = {
        "note": (
            "Each entry is a static prompt prefix shared by every call in its group. "
            "Reconstruct a full prompt as prefixes[record.prompt_prefix_sha256].text "
            "+ record.prompt_text. A record with prompt_is_complete=true carries its "
            "whole prompt already."
        ),
        "savings": prompt_prefixes.savings(prefixes),
        "prefixes": list(prefixes.values()),
    }
    prefix_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"Wrote LLM calls detail to {out_path}")


# ---------------------------------------------------------------------------
# Emitter: SPD Tables CSV
# ---------------------------------------------------------------------------

def emit_spd_tables_csv(data: BundleData, con: duckdb.DuckDBPyConnection, out_dir: Path) -> None:
    """Write ensemble SPD probability tables to CSV."""
    if not data.forecaster_run_id:
        return

    out_path = out_dir / f"spd_tables__{data.out_run_id}.csv"

    try:
        rows = con.execute(
            """
            SELECT
                fe.question_id, q.iso3, q.hazard_code, q.metric,
                fe.model_name, fe.month_index, fe.bucket_index, fe.probability,
                fe.ev_value, fe.status
            FROM forecasts_ensemble fe
            LEFT JOIN questions q ON q.question_id = fe.question_id
            WHERE fe.run_id = ?
            ORDER BY fe.question_id, fe.month_index, fe.bucket_index
            """,
            [data.forecaster_run_id],
        ).fetchall()
    except Exception:
        return

    if not rows:
        return

    fieldnames = [
        "question_id", "iso3", "hazard_code", "metric",
        "model_name", "month_index", "bucket_index", "probability",
        "ev_value", "status",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(zip(fieldnames, row)))

    print(f"Wrote SPD tables to {out_path}")


# ---------------------------------------------------------------------------
# Emitter: RC Triage Summary CSV
# ---------------------------------------------------------------------------

def emit_rc_triage_summary_csv(
    data: BundleData, con: duckdb.DuckDBPyConnection, out_dir: Path
) -> None:
    """Export a CSV with one row per (iso3, hazard_code) showing the full RC + triage picture."""
    if not data.hs_run_id:
        return

    try:
        # Column names must match the hs_triage DDL in pythia/db/schema.py —
        # the RC fields are regime_change_*, drivers live in drivers_json and
        # the triage status inside data_quality_json.
        rows = _fetch_llm_rows(
            con,
            """
            SELECT
                iso3,
                hazard_code,
                regime_change_likelihood AS rc_likelihood,
                regime_change_magnitude AS rc_magnitude,
                regime_change_score AS rc_score,
                regime_change_level AS rc_level,
                regime_change_direction AS rc_direction,
                regime_change_window AS rc_window,
                triage_score,
                tier,
                track,
                need_full_spd,
                drivers_json AS drivers,
                data_quality_json
            FROM hs_triage
            WHERE run_id = ?
            ORDER BY triage_score DESC
            """,
            [data.hs_run_id],
        )
    except Exception as exc:
        LOG.warning("emit_rc_triage_summary_csv failed: %s", exc)
        return

    fieldnames = [
        "iso3", "hazard_code",
        "rc_likelihood", "rc_magnitude", "rc_score", "rc_level", "rc_direction", "rc_window",
        "triage_score", "tier", "track", "need_full_spd",
        "drivers", "confidence_note", "status",
    ]

    out_path = out_dir / f"rc_triage_summary__{data.out_run_id}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            drivers_raw = row.get("drivers") or ""
            if isinstance(drivers_raw, str):
                try:
                    drivers_list = json.loads(drivers_raw)
                except Exception:
                    drivers_list = [d.strip() for d in drivers_raw.split(",") if d.strip()]
            elif isinstance(drivers_raw, list):
                drivers_list = drivers_raw
            else:
                drivers_list = []
            drivers_str = "|".join(str(d) for d in drivers_list[:3])

            status = ""
            dq_raw = row.get("data_quality_json") or ""
            if dq_raw:
                try:
                    dq = json.loads(dq_raw)
                    if isinstance(dq, dict):
                        status = str(dq.get("status") or "")
                except Exception:
                    status = ""

            writer.writerow({
                "iso3": row.get("iso3") or "",
                "hazard_code": row.get("hazard_code") or "",
                "rc_likelihood": row.get("rc_likelihood") or "",
                "rc_magnitude": row.get("rc_magnitude") or "",
                "rc_score": row.get("rc_score") or "",
                "rc_level": row.get("rc_level") or "",
                "rc_direction": row.get("rc_direction") or "",
                "rc_window": row.get("rc_window") or "",
                "triage_score": row.get("triage_score") or "",
                "tier": row.get("tier") or "",
                "track": row.get("track") or "",
                "need_full_spd": row.get("need_full_spd") or "",
                "drivers": drivers_str,
                "confidence_note": "",
                "status": status,
            })

    print(f"Wrote RC triage summary to {out_path}")


# ---------------------------------------------------------------------------
# Emitter: RC Pass Detail CSV
# ---------------------------------------------------------------------------

def emit_rc_pass_detail_csv(
    data: BundleData, con: duckdb.DuckDBPyConnection, out_dir: Path
) -> None:
    """Export a CSV showing Pass 1 vs Pass 2 RC values before merge."""
    if not data.hs_run_id:
        return

    try:
        rows = _fetch_llm_rows(
            con,
            """
            SELECT
                iso3,
                hazard_code,
                model_id,
                provider,
                response_text,
                error_text,
                elapsed_ms,
                cost_usd
            FROM llm_calls
            WHERE phase = 'hs_triage'
              AND hazard_code LIKE 'RC_%'
              AND hs_run_id = ?
            ORDER BY iso3, hazard_code, timestamp
            """,
            [data.hs_run_id],
        )
    except Exception as exc:
        LOG.warning("emit_rc_pass_detail_csv failed: %s", exc)
        return

    fieldnames = [
        "iso3", "hazard_code", "pass_number",
        "model_id", "provider",
        "likelihood", "magnitude", "direction", "window",
        "elapsed_ms", "cost_usd",
        "error_text", "rationale_bullets",
    ]

    # Track pass numbers per (iso3, hazard_code)
    pass_counters: dict[tuple[str, str], int] = {}

    out_path = out_dir / f"rc_pass_detail__{data.out_run_id}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            iso3 = row.get("iso3") or ""
            hazard_code = row.get("hazard_code") or ""
            key = (iso3, hazard_code)
            pass_counters[key] = pass_counters.get(key, 0) + 1
            pass_number = min(pass_counters[key], 2)

            # Parse response JSON for RC values
            likelihood = ""
            magnitude = ""
            direction = ""
            window = ""
            rationale_bullets: list[str] = []
            resp_text = row.get("response_text") or ""
            try:
                resp = json.loads(resp_text) if resp_text.strip() else {}
                if isinstance(resp, dict):
                    likelihood = resp.get("likelihood") or resp.get("rc_likelihood") or ""
                    magnitude = resp.get("magnitude") or resp.get("rc_magnitude") or ""
                    direction = resp.get("direction") or resp.get("rc_direction") or ""
                    window = resp.get("window") or resp.get("rc_window") or ""
                    rationale = resp.get("rationale") or resp.get("rationale_bullets") or []
                    if isinstance(rationale, list):
                        rationale_bullets = [str(r) for r in rationale[:2]]
                    elif isinstance(rationale, str):
                        rationale_bullets = [rationale[:200]]
            except Exception:
                pass

            error_text = (row.get("error_text") or "")[:200]

            writer.writerow({
                "iso3": iso3,
                "hazard_code": hazard_code,
                "pass_number": pass_number,
                "model_id": row.get("model_id") or "",
                "provider": row.get("provider") or "",
                "likelihood": likelihood,
                "magnitude": magnitude,
                "direction": direction,
                "window": window,
                "elapsed_ms": row.get("elapsed_ms") or "",
                "cost_usd": row.get("cost_usd") or "",
                "error_text": error_text,
                "rationale_bullets": "|".join(rationale_bullets),
            })

    print(f"Wrote RC pass detail to {out_path}")


# ---------------------------------------------------------------------------
# Emitter: Data Inject Inventory CSV
# ---------------------------------------------------------------------------

def _safe_table_count_where(
    con: duckdb.DuckDBPyConnection, table: str, where: str, params: list[Any]
) -> int | None:
    """Return count from a table with a WHERE clause, or None if table doesn't exist."""
    try:
        row = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", params).fetchone()
        return int(row[0]) if row else None
    except Exception:
        return None


def _safe_table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {table} LIMIT 0")
        return True
    except Exception:
        return False


def _load_structured_data_coverage(
    con: duckdb.DuckDBPyConnection,
    countries: list[str],
) -> dict[str, tuple[int, int, list[str]]]:
    """Return ``{source_label: (countries_yes, countries_no, missing_iso3s)}`` for each data source."""
    n = len(countries)
    if n == 0:
        return {}

    placeholders = ", ".join("?" for _ in countries)
    upper_countries = [c.upper() for c in countries]
    upper_set = set(upper_countries)
    result: dict[str, tuple[int, int, list[str]]] = {}

    # (label, table, extra WHERE clause appended after the IN filter)
    source_queries: list[tuple[str, str, str]] = [
        ("ACLED fatalities", "acled_monthly_fatalities", ""),
        ("IDMC displacement", "facts_deltas", "AND metric = 'new_displacements'"),
        ("IFRC PA", "facts_resolved", "AND hazard_code IN ('FL','TC','DR','EQ')"),
        ("Conflict forecasts", "conflict_forecasts", ""),
        (
            "FEWS NET IPC",
            "facts_resolved",
            "AND hazard_code = 'DR' AND metric IN ('phase3plus_in_need', 'phase3plus_projection') AND UPPER(publisher) = 'FEWS NET'",
        ),
        (
            "IPC API",
            "facts_resolved",
            "AND hazard_code = 'DR' AND metric IN ('phase3plus_in_need', 'phase3plus_projection') AND UPPER(publisher) = 'IPC'",
        ),
        ("ReliefWeb reports", "reliefweb_reports", ""),
        ("HDX Signals", "hdx_signals", ""),
        ("Seasonal TC", "seasonal_tc_context_cache", "AND context_text IS NOT NULL"),
        ("NMME", "seasonal_forecasts", ""),
        ("CrisisWatch", "crisiswatch_entries", ""),
        ("GDACS events", "facts_resolved", "AND metric = 'event_occurrence'"),
    ]

    for label, table, extra_where in source_queries:
        if not _safe_table_exists(con, table):
            result[label] = (0, n, sorted(upper_set))
            continue
        try:
            sql = (
                f"SELECT DISTINCT UPPER(iso3) FROM {table} "
                f"WHERE UPPER(iso3) IN ({placeholders}) {extra_where}"
            )
            covered = {row[0] for row in con.execute(sql, upper_countries).fetchall()}
            missing = sorted(upper_set - covered)
            result[label] = (len(covered), n - len(covered), missing)
        except Exception:
            result[label] = (0, n, sorted(upper_set))

    # ENSO is global (not per-country)
    try:
        if _safe_table_exists(con, "enso_state"):
            row_count = con.execute("SELECT COUNT(*) FROM enso_state").fetchone()[0]
            result["ENSO"] = (n, 0, []) if row_count > 0 else (0, n, sorted(upper_set))
        else:
            result["ENSO"] = (0, n, sorted(upper_set))
    except Exception:
        result["ENSO"] = (0, n, sorted(upper_set))

    return result


def emit_data_inject_inventory_csv(
    data: BundleData, con: duckdb.DuckDBPyConnection, out_dir: Path
) -> None:
    """Export a CSV showing which structured data sources were available for each country."""
    if not data.resolved_countries_sorted:
        return

    fieldnames = [
        "iso3", "country_name",
        "acled_fatalities_months", "idmc_displacement_months",
        "ifrc_pa_months", "conflict_forecasts_available",
        "fewsnet_ipc_rows", "ipc_api_rows", "food_security_source",
        "reliefweb_reports_count",
        "hdx_signals_count", "enso_loaded",
        "seasonal_tc_loaded", "nmme_available",
        "crisiswatch_arrow", "gdacs_event_months",
    ]

    # Check which tables exist once
    has_acled = _safe_table_exists(con, "acled_monthly_fatalities")
    has_facts_deltas = _safe_table_exists(con, "facts_deltas")
    has_facts_resolved = _safe_table_exists(con, "facts_resolved")
    has_conflict_forecasts = _safe_table_exists(con, "conflict_forecasts")
    has_acaps = _safe_table_exists(con, "acaps_inform_severity")
    has_reliefweb = _safe_table_exists(con, "reliefweb_reports")
    has_seasonal_forecasts = _safe_table_exists(con, "seasonal_forecasts")
    has_seasonal_tc_cache = _safe_table_exists(con, "seasonal_tc_context_cache")

    # Check ENSO and seasonal TC by calling the actual loaders
    try:
        from horizon_scanner.enso import get_enso_prompt_context
        enso_loaded = bool(get_enso_prompt_context())
    except Exception:
        enso_loaded = False

    # Load COUNTRY_TO_BASINS for TC basin exposure check
    try:
        from horizon_scanner.seasonal_tc import COUNTRY_TO_BASINS
    except Exception:
        COUNTRY_TO_BASINS = {}

    # seasonal_tc_loaded is checked per-country below (varies by basin exposure)

    # Load HDX Signals cached CSV to count signals per country
    hdx_signals_by_iso3: dict[str, int] = {}
    try:
        from horizon_scanner.hdx_signals import CACHE_FILE as _hdx_cache_file
        if _hdx_cache_file.exists():
            import csv as _csv
            text = _hdx_cache_file.read_text(encoding="utf-8")
            for row in _csv.DictReader(io.StringIO(text)):
                iso = (row.get("iso3") or "").upper()
                if iso:
                    hdx_signals_by_iso3[iso] = hdx_signals_by_iso3.get(iso, 0) + 1
    except Exception:
        pass  # HDX Signals cache unavailable — leave counts empty

    # Build CrisisWatch per-country lookup from already-loaded data
    cw_by_iso3: dict[str, dict[str, Any]] = {}
    for entry in data.crisiswatch_entries:
        iso = (entry.get("iso3") or "").upper()
        if iso and iso not in cw_by_iso3:
            cw_by_iso3[iso] = entry  # most recent first (ORDER BY year DESC, month DESC)

    # Try loading country names from hs_triage
    country_names: dict[str, str] = {}
    try:
        name_rows = con.execute(
            "SELECT DISTINCT iso3, country_name FROM hs_triage WHERE country_name IS NOT NULL"
        ).fetchall()
        for iso3, name in name_rows:
            if iso3 and name:
                country_names[str(iso3).upper()] = str(name)
    except Exception:
        pass

    out_path = out_dir / f"data_inject_inventory__{data.out_run_id}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for iso3 in data.resolved_countries_sorted:
            acled_months = _safe_table_count_where(
                con, "acled_monthly_fatalities", "upper(iso3) = ?", [iso3.upper()]
            ) if has_acled else None

            idmc_months = _safe_table_count_where(
                con, "facts_deltas",
                "upper(iso3) = ? AND metric = 'new_displacements'",
                [iso3.upper()],
            ) if has_facts_deltas else None
            # Also check facts_resolved if facts_deltas had nothing
            if not idmc_months and has_facts_resolved:
                idmc_months = _safe_table_count_where(
                    con, "facts_resolved",
                    "upper(iso3) = ? AND metric = 'new_displacements'",
                    [iso3.upper()],
                )

            ifrc_months = _safe_table_count_where(
                con, "facts_resolved",
                "upper(iso3) = ? AND hazard_code IN ('FL','TC','DR','HW','EQ')",
                [iso3.upper()],
            ) if has_facts_resolved else None

            conflict_avail = bool(_safe_table_count_where(
                con, "conflict_forecasts", "upper(iso3) = ?", [iso3.upper()]
            )) if has_conflict_forecasts else False

            # Food security: per-source row counts from facts_resolved
            fewsnet_ipc_rows = 0
            ipc_api_rows = 0
            if has_facts_resolved:
                fewsnet_ipc_rows = _safe_table_count_where(
                    con, "facts_resolved",
                    "upper(iso3) = ? AND hazard_code = 'DR' AND metric IN ('phase3plus_in_need', 'phase3plus_projection') AND UPPER(publisher) = 'FEWS NET'",
                    [iso3.upper()],
                ) or 0
                ipc_api_rows = _safe_table_count_where(
                    con, "facts_resolved",
                    "upper(iso3) = ? AND hazard_code = 'DR' AND metric IN ('phase3plus_in_need', 'phase3plus_projection') AND UPPER(publisher) = 'IPC'",
                    [iso3.upper()],
                ) or 0
            if fewsnet_ipc_rows and ipc_api_rows:
                food_security_source = "both"
            elif fewsnet_ipc_rows:
                food_security_source = "FEWS NET"
            elif ipc_api_rows:
                food_security_source = "IPC"
            else:
                food_security_source = ""

            reliefweb_count = _safe_table_count_where(
                con, "reliefweb_reports", "upper(iso3) = ?", [iso3.upper()]
            ) if has_reliefweb else None

            nmme_avail = bool(_safe_table_count_where(
                con, "seasonal_forecasts", "upper(iso3) = ?", [iso3.upper()]
            )) if has_seasonal_forecasts else False

            # Per-country seasonal TC check — query the DB directly using
            # the bundle's connection instead of calling
            # get_seasonal_tc_context_for_country() which opens its own
            # connection via schema.connect() and may hit a different DB.
            if iso3.upper() not in COUNTRY_TO_BASINS:
                seasonal_tc_loaded = ""  # not TC-exposed
            elif has_seasonal_tc_cache:
                seasonal_tc_loaded = bool(_safe_table_count_where(
                    con, "seasonal_tc_context_cache",
                    "upper(iso3) = ? AND context_text IS NOT NULL AND context_text != ''",
                    [iso3.upper()],
                ))
            else:
                seasonal_tc_loaded = False

            # CrisisWatch arrow for this country (ACE-only data source)
            cw_entry = cw_by_iso3.get(iso3.upper())
            if cw_entry:
                cw_arrow = cw_entry.get("arrow") or "present"
                if cw_entry.get("alert_type"):
                    cw_arrow += f" [{cw_entry['alert_type']}]"
            else:
                cw_arrow = ""

            # GDACS event occurrence months
            gdacs_event_months = _safe_table_count_where(
                con, "facts_resolved",
                "upper(iso3) = ? AND lower(metric) = 'event_occurrence'",
                [iso3.upper()],
            ) if has_facts_resolved else 0

            writer.writerow({
                "iso3": iso3,
                "country_name": country_names.get(iso3.upper(), ""),
                "acled_fatalities_months": acled_months if acled_months is not None else "",
                "idmc_displacement_months": idmc_months if idmc_months is not None else "",
                "ifrc_pa_months": ifrc_months if ifrc_months is not None else "",
                "conflict_forecasts_available": conflict_avail,
                "fewsnet_ipc_rows": fewsnet_ipc_rows,
                "ipc_api_rows": ipc_api_rows,
                "food_security_source": food_security_source,
                "reliefweb_reports_count": reliefweb_count if reliefweb_count is not None else "",
                "hdx_signals_count": hdx_signals_by_iso3.get(iso3.upper(), "N/A" if not hdx_signals_by_iso3 else 0),
                "enso_loaded": enso_loaded,
                "seasonal_tc_loaded": seasonal_tc_loaded,
                "nmme_available": nmme_avail,
                "crisiswatch_arrow": cw_arrow,
                "gdacs_event_months": gdacs_event_months or 0,
            })

    print(f"Wrote data inject inventory to {out_path}")


# ---------------------------------------------------------------------------
# Emitter: Timing Breakdown CSV
# ---------------------------------------------------------------------------

def _timing_stage_for_call(phase: str, call_type: str, hazard_code: str) -> str | None:
    """Map one llm_calls row to a timing stage: rc / triage / adversarial /
    research / spd. ``call_type`` is authoritative (rc_pass_N, rc_grounding,
    triage_pass_N, triage_grounding, adversarial_search, adversarial_synthesis,
    spd_v2, binary_v2, ...); the synthetic UPPERCASED hazard_code is the
    fallback for rows written before call_type existed. Every HS row carries
    phase='hs_triage', which is why phase alone cannot split RC from triage —
    the 2026-09-01 bundle reported rc_elapsed_ms == triage_elapsed_ms for all
    122 countries because it did exactly that."""

    ct = (call_type or "").lower()
    hz = (hazard_code or "").upper()
    ph = (phase or "").lower()
    if ct.startswith("rc_"):
        return "rc"
    if ct.startswith("triage_"):
        return "triage"
    if ct.startswith("adversarial_"):
        return "adversarial"
    if ct in ("spd_v2", "binary_v2", "track2_spd") or ph in ("spd_v2", "binary_v2"):
        return "spd"
    if ct in ("research_v2", "research_web_research", "hs_web_research") or ph in (
        "research_v2", "research_web_research", "hs_web_research"
    ):
        return "research"
    if hz.startswith("ADVERSARIAL_"):
        return "adversarial"
    if hz.startswith("TRIAGE_"):
        return "triage"
    if hz.startswith("RC_") or hz.startswith("GROUNDING_"):
        return "rc"
    if ph == "hs_triage":
        # Legacy row with no call_type and a plain hazard code: a triage
        # pass (RC rows always carried the RC_ prefix).
        return "triage"
    return None


def _fill_spd_stage_from_forecasts(
    con: duckdb.DuckDBPyConnection,
    data: BundleData,
    iso3_data: dict[str, dict[str, Any]],
    stages: tuple[str, ...],
) -> None:
    """Give every question-producing country a row, and an spd span if it has none.

    Never raises: a DB with no forecasts_raw simply leaves the entries as
    llm_calls found them, and the ``spd_timing_source`` column says which
    case each row is.
    """

    def _blank(iso3: str) -> dict[str, Any]:
        entry: dict[str, Any] = {"iso3": iso3, "total_elapsed_ms": 0}
        for st in stages:
            entry[f"{st}_start_ts"] = None
            entry[f"{st}_end_ts"] = None
            entry[f"{st}_elapsed_ms"] = 0
            entry[f"n_{st}_calls"] = 0
            entry[f"n_{st}_batched"] = 0
        return entry

    questions_by_iso3: dict[str, list[str]] = {}
    for q in data.questions or []:
        iso3 = str(q.get("iso3") or "").upper()
        qid = str(q.get("question_id") or "")
        if iso3 and qid:
            questions_by_iso3.setdefault(iso3, []).append(qid)

    for entry in iso3_data.values():
        entry.setdefault("n_questions", 0)
        entry.setdefault("spd_timing_source", "llm_calls" if entry.get("spd_start_ts") else "")
    for iso3 in questions_by_iso3:
        entry = iso3_data.setdefault(iso3, _blank(iso3))
        entry["n_questions"] = len(questions_by_iso3[iso3])
        entry.setdefault("spd_timing_source", "llm_calls" if entry.get("spd_start_ts") else "")

    if not data.forecaster_run_id or not questions_by_iso3:
        return
    needs = [iso3 for iso3, e in iso3_data.items() if e.get("spd_start_ts") is None]
    if not needs:
        return
    qids = [qid for iso3 in needs for qid in questions_by_iso3.get(iso3, [])]
    if not qids:
        return
    spans = _forecast_write_spans(con, data.forecaster_run_id, qids)
    if not spans:
        return
    qid_to_iso3 = {qid: iso3 for iso3, qids_ in questions_by_iso3.items() for qid in qids_}
    for qid, span in spans.items():
        iso3 = qid_to_iso3.get(qid)
        if not iso3 or iso3 not in iso3_data:
            continue
        entry = iso3_data[iso3]
        start, end = span.get("started_at_utc"), span.get("finished_at_utc")
        if start is not None and (entry["spd_start_ts"] is None or start < entry["spd_start_ts"]):
            entry["spd_start_ts"] = start
        if end is not None and (entry["spd_end_ts"] is None or end > entry["spd_end_ts"]):
            entry["spd_end_ts"] = end
        entry["spd_timing_source"] = "forecasts_raw.created_at (no spd llm_calls row)"


def emit_timing_breakdown_csv(
    data: BundleData, con: duckdb.DuckDBPyConnection, out_dir: Path
) -> None:
    """Export a CSV with per-country timing breakdown from llm_calls.

    Each stage's elapsed is the span from its first to its last call for the
    country (never a sum of per-group spans, which double counts overlapping
    calls). Batched calls record elapsed_ms=0 and their timestamps are the
    replay time, so under the staged pipeline the spd stage measures the
    collect stage's replay, not model latency — n_*_batched says how much of
    each stage that applies to.
    """
    if not data.hs_run_id:
        return

    predicate_parts: list[str] = []
    params: list[Any] = []
    if data.hs_run_id:
        predicate_parts.append("hs_run_id = ?")
        params.append(data.hs_run_id)
    if data.forecaster_run_id:
        predicate_parts.append("run_id = ?")
        params.append(data.forecaster_run_id)

    if not predicate_parts:
        return

    scope = " OR ".join(predicate_parts)
    llm_columns = _llm_calls_columns(con)
    call_type_expr = "call_type" if "call_type" in llm_columns else "NULL AS call_type"
    batched_expr = (
        "COUNT(*) FILTER (WHERE usage_json LIKE '%\"service_tier\": \"batch\"%') AS n_batched"
        if "usage_json" in llm_columns
        else "0 AS n_batched"
    )

    try:
        rows = _fetch_llm_rows(
            con,
            f"""
            SELECT
                iso3,
                phase,
                {call_type_expr},
                hazard_code,
                MIN(timestamp) AS min_ts,
                MAX(timestamp) AS max_ts,
                COUNT(*) AS n_calls,
                {batched_expr}
            FROM llm_calls
            WHERE ({scope})
              AND iso3 IS NOT NULL
              AND timestamp IS NOT NULL
            GROUP BY iso3, phase, call_type, hazard_code
            ORDER BY iso3, phase
            """,
            params,
        )
    except Exception as exc:
        LOG.warning("emit_timing_breakdown_csv failed: %s", exc)
        return

    stages = ("rc", "triage", "adversarial", "research", "spd")
    iso3_data: dict[str, dict[str, Any]] = {}
    for row in rows:
        iso3 = str(row.get("iso3") or "")
        if not iso3:
            continue
        entry = iso3_data.get(iso3)
        if entry is None:
            entry = {"iso3": iso3, "total_elapsed_ms": 0}
            for st in stages:
                entry[f"{st}_start_ts"] = None
                entry[f"{st}_end_ts"] = None
                entry[f"{st}_elapsed_ms"] = 0
                entry[f"n_{st}_calls"] = 0
                entry[f"n_{st}_batched"] = 0
            iso3_data[iso3] = entry

        stage = _timing_stage_for_call(
            str(row.get("phase") or ""),
            str(row.get("call_type") or ""),
            str(row.get("hazard_code") or ""),
        )
        if stage is None:
            continue
        min_ts = row.get("min_ts")
        max_ts = row.get("max_ts")
        start_key, end_key = f"{stage}_start_ts", f"{stage}_end_ts"
        if min_ts is not None and (entry[start_key] is None or min_ts < entry[start_key]):
            entry[start_key] = min_ts
        if max_ts is not None and (entry[end_key] is None or max_ts > entry[end_key]):
            entry[end_key] = max_ts
        entry[f"n_{stage}_calls"] += int(row.get("n_calls") or 0)
        entry[f"n_{stage}_batched"] += int(row.get("n_batched") or 0)

    # A country whose questions all took the Track-2 path can have no
    # spd-stage llm_calls row at all (one batched call, replayed, and on
    # 2026-09-01 fourteen countries were missing from this file entirely
    # despite having produced questions). forecasts_raw dates what was
    # WRITTEN, which is the fallback, and every country that produced a
    # question gets a row whether or not any stage could be timed.
    _fill_spd_stage_from_forecasts(con, data, iso3_data, stages)

    def _span_ms(a: Any, b: Any) -> int:
        if a is None or b is None:
            return 0
        try:
            return max(int((b - a).total_seconds() * 1000), 0)
        except Exception:
            return 0

    for entry in iso3_data.values():
        total = 0
        for st in stages:
            elapsed = _span_ms(entry[f"{st}_start_ts"], entry[f"{st}_end_ts"])
            entry[f"{st}_elapsed_ms"] = elapsed
            total += elapsed
        entry["total_elapsed_ms"] = total

    fieldnames = ["iso3"]
    for st in stages:
        fieldnames += [f"{st}_start_ts", f"{st}_end_ts", f"{st}_elapsed_ms"]
    fieldnames.append("total_elapsed_ms")
    for st in stages:
        fieldnames.append(f"n_{st}_calls")
    for st in stages:
        fieldnames.append(f"n_{st}_batched")
    fieldnames += ["n_questions", "spd_timing_source"]

    out_path = out_dir / f"timing_breakdown__{data.out_run_id}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for iso3 in sorted(iso3_data.keys()):
            entry = iso3_data[iso3]
            writer.writerow({fn: ("" if entry.get(fn) is None else entry.get(fn)) for fn in fieldnames})

    print(f"Wrote timing breakdown to {out_path}")


# ---------------------------------------------------------------------------
# Emitter: Model Config Snapshot
# ---------------------------------------------------------------------------

def emit_model_config_snapshot(
    data: BundleData, con: duckdb.DuckDBPyConnection, out_dir: Path
) -> None:
    """Export a JSON file capturing the runtime model configuration."""
    env_var_names = [
        "PYTHIA_HS_FALLBACK_MODEL_SPECS",
        "PYTHIA_WEB_RESEARCH_MODEL_ID",
        "PYTHIA_RETRIEVER_ENABLED",
        "PYTHIA_SPD_WEB_SEARCH_ENABLED",
        "PYTHIA_SPD_GOOGLE_WEB_SEARCH_ENABLED",
        "PYTHIA_SPD_V2_USE_BAYESMC",
        "PYTHIA_SPD_V2_WRITE_BOTH",
        "PYTHIA_SPD_V2_DUAL_RUN",
        "PYTHIA_CONFIG_PROFILE",
        "PYTHIA_ADVERSARIAL_CHECK_ENABLED",
        "PYTHIA_HS_ADVERSARIAL_MAX_PER_COUNTRY",
        "PYTHIA_FORECASTER_SELF_SEARCH",
        "PYTHIA_MODEL_SELF_SEARCH_ENABLED",
    ]

    env_vars = {}
    for name in env_var_names:
        val = os.getenv(name)
        env_vars[name] = val if val is not None else "<unset>"

    # Query actual models used from llm_calls
    actual_models: dict[str, list[dict[str, str]]] = {}
    predicate_parts: list[str] = []
    params: list[Any] = []
    if data.hs_run_id:
        predicate_parts.append("hs_run_id = ?")
        params.append(data.hs_run_id)
    if data.forecaster_run_id:
        predicate_parts.append("run_id = ?")
        params.append(data.forecaster_run_id)

    if predicate_parts:
        scope = " OR ".join(predicate_parts)
        try:
            rows = con.execute(
                f"""
                SELECT DISTINCT phase, provider, model_id
                FROM llm_calls
                WHERE ({scope})
                  AND provider IS NOT NULL
                  AND model_id IS NOT NULL
                ORDER BY phase, provider, model_id
                """,
                params,
            ).fetchall()
            for phase, provider, model_id in rows:
                phase_str = str(phase or "")
                actual_models.setdefault(phase_str, []).append({
                    "provider": str(provider or ""),
                    "model_id": str(model_id or ""),
                })
        except Exception:
            pass

    # Build the SPD ensemble models list (config-declared, key-independent)
    spd_ensemble_list: list[str] = []
    try:
        spd_ensemble_list = _expected_spd_model_ids()
    except Exception:
        pass

    config = {
        "hs_run_id": data.hs_run_id or "",
        "forecaster_run_id": data.forecaster_run_id or "",
        "generated_at": data.now,
        "models": {
            "spd_track1_ensemble": spd_ensemble_list,
        },
        "env_vars": env_vars,
        "actual_models_used": actual_models,
    }

    out_path = out_dir / f"model_config__{data.out_run_id}.json"
    out_path.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    print(f"Wrote model config snapshot to {out_path}")


# ---------------------------------------------------------------------------
# Emitter: Grounding Detail CSV
# ---------------------------------------------------------------------------

def _grounding_stage_for_call(call_type: str, hazard_code: str) -> str:
    """rc_grounding / triage_grounding / adversarial_search / other, from
    call_type first and the synthetic hazard_code second."""

    ct = (call_type or "").lower()
    if ct in ("rc_grounding", "triage_grounding", "adversarial_search"):
        return ct
    hz = (hazard_code or "").upper()
    if hz.startswith("TRIAGE_GROUNDING_"):
        return "triage_grounding"
    if hz.startswith("GROUNDING_"):
        return "rc_grounding"
    if hz.startswith("ADVERSARIAL_") and not hz.startswith("ADVERSARIAL_SYNTH_"):
        return "adversarial_search"
    return "other"


def emit_grounding_detail_csv(
    data: BundleData, con: duckdb.DuckDBPyConnection, out_dir: Path
) -> None:
    """Export a CSV with per-call grounding results."""
    predicate_parts: list[str] = []
    params: list[Any] = []
    if data.hs_run_id:
        predicate_parts.append("hs_run_id = ?")
        params.append(data.hs_run_id)
    if data.forecaster_run_id:
        predicate_parts.append("run_id = ?")
        params.append(data.forecaster_run_id)

    if not predicate_parts:
        return

    scope = " OR ".join(predicate_parts)
    gd_call_type_expr = (
        "call_type" if "call_type" in _llm_calls_columns(con) else "NULL AS call_type"
    )

    try:
        rows = _fetch_llm_rows(
            con,
            f"""
            SELECT
                iso3,
                hazard_code,
                phase,
                {gd_call_type_expr},
                model_id,
                response_text,
                error_text,
                prompt_text,
                elapsed_ms,
                timestamp
            FROM llm_calls
            WHERE ({scope})
              AND (
                phase IN ('hs_web_research', 'research_web_research')
                OR phase LIKE '%web_research%'
                OR LOWER(hazard_code) LIKE '%grounding%'
                OR (
                  UPPER(hazard_code) LIKE 'ADVERSARIAL_%'
                  -- synthesis rows are generation calls, not searches
                  AND UPPER(hazard_code) NOT LIKE 'ADVERSARIAL_SYNTH_%'
                )
              )
            ORDER BY iso3, hazard_code, timestamp
            """,
            params,
        )
    except Exception as exc:
        LOG.warning("emit_grounding_detail_csv failed: %s", exc)
        return

    # `phase` is 'hs_triage' for EVERY HS row (RC grounding, triage grounding
    # and adversarial searches alike — 797 of 797 on 2026-09-01), so the
    # stage is derived from call_type, with the synthetic hazard_code prefix
    # as the fallback for rows that predate call_type.
    fieldnames = [
        "iso3", "hazard_code", "phase", "call_type", "stage",
        "model_id", "grounded", "n_sources",
        "query", "error_code", "elapsed_ms", "timestamp", "db_phase",
    ]

    out_path = out_dir / f"grounding_detail__{data.out_run_id}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            resp_text = row.get("response_text") or ""
            grounded = False
            n_sources = 0
            try:
                resp = json.loads(resp_text) if resp_text.strip() else {}
                if isinstance(resp, dict):
                    grounded = bool(resp.get("grounded") or resp.get("sources"))
                    # Compact format: pre-computed n_sources count
                    if "n_sources" in resp:
                        n_sources = int(resp["n_sources"])
                    elif isinstance(resp.get("source_urls"), list):
                        n_sources = len(resp["source_urls"])
                    else:
                        sources = resp.get("sources") or []
                        n_sources = len(sources) if isinstance(sources, list) else 0
            except Exception:
                pass

            # Fallback: check for markdown evidence pack pattern
            if not grounded and resp_text:
                if "Grounded: True" in resp_text or "Sources:" in resp_text:
                    grounded = True
                import re as _re
                urls = _re.findall(r"https?://[^\s\]\"')>]+", resp_text)
                if urls and n_sources == 0:
                    n_sources = len(set(urls))

            error_text = (row.get("error_text") or "")[:200]
            prompt_text = (row.get("prompt_text") or "")[:200]

            stage = _grounding_stage_for_call(
                str(row.get("call_type") or ""), str(row.get("hazard_code") or "")
            )
            writer.writerow({
                "iso3": row.get("iso3") or "",
                "hazard_code": row.get("hazard_code") or "",
                # `phase` carries the DERIVED stage. The raw column is
                # 'hs_triage' on every HS row — rc grounding, triage
                # grounding and adversarial searches alike, 797 of 797 on
                # 2026-09-01 — so as published it distinguished nothing. The
                # raw value is kept beside it as `db_phase` so nothing is
                # lost and the mapping stays auditable.
                "phase": stage,
                "db_phase": row.get("phase") or "",
                "call_type": row.get("call_type") or "",
                "stage": stage,
                "model_id": row.get("model_id") or "",
                "grounded": grounded,
                "n_sources": n_sources,
                "query": prompt_text,
                "error_code": error_text,
                "elapsed_ms": row.get("elapsed_ms") or "",
                "timestamp": str(row.get("timestamp") or ""),
            })

    print(f"Wrote grounding detail to {out_path}")


# ---------------------------------------------------------------------------
# Operational emitters (scripts/debug_bundle/*)
#
# Everything below is additive and everything below is wrapped. The rule the
# whole section is written to: nothing added here may fail the phase. A
# collector that raises writes a stub recording the error, the anomaly list
# gains a line saying the bundle itself failed, and the run continues.
# ---------------------------------------------------------------------------

_COLLECTOR_ERRORS: list[dict[str, Any]] = []


def _write_stub(path: Path, collector: str, error: str) -> None:
    """A failed collector leaves evidence, never an absent file.

    An absent file is indistinguishable from a file that had nothing to
    say, and the two want different responses.
    """

    payload = {
        "collector": collector,
        "error": error,
        "note": (
            "This collector failed. The bundle is otherwise complete; the "
            "phase was not failed over it."
        ),
        "generated_at": datetime.utcnow().isoformat(),
    }
    try:
        if path.suffix == ".csv":
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["collector", "error", "note"])
                writer.writeheader()
                writer.writerow({k: payload[k] for k in ("collector", "error", "note")})
        else:
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:  # pragma: no cover - defensive
        LOG.warning("could not write stub for %s", collector)


def _run_collector(name: str, path: Path, fn: Callable[[], Any]) -> Any:
    """Run one collector; on failure write a stub and record the error."""

    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        message = f"{type(exc).__name__}: {exc}"
        LOG.warning("collector %s failed: %s", name, message)
        LOG.debug("%s", traceback.format_exc())
        _COLLECTOR_ERRORS.append(
            {"collector": name, "error": message, "file": path.name}
        )
        _write_stub(path, name, message)
        return None


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})


def emit_operational_files(
    data: BundleData,
    con: duckdb.DuckDBPyConnection,
    out_dir: Path,
    *,
    pipeline_id: str | None,
    repo_root: Path,
    health_checks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write the operational half of the bundle; return what the summary needs."""

    run = data.out_run_id
    collected: dict[str, Any] = {}

    # --- batch lifecycle -------------------------------------------------
    bl_path = out_dir / f"batch_lifecycle__{run}.json"

    def _batches() -> dict[str, Any]:
        payload = batch_lifecycle.collect(
            con,
            hs_run_id=data.hs_run_id,
            forecaster_run_id=data.forecaster_run_id,
            pipeline_id=pipeline_id,
        )
        bl_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    collected["batch_lifecycle"] = _run_collector("batch_lifecycle", bl_path, _batches) or {}

    # --- raw provider objects -------------------------------------------
    po_path = out_dir / f"provider_batch_objects__{run}.json"
    fetch_enabled = os.getenv("PYTHIA_BUNDLE_FETCH_PROVIDER_OBJECTS", "1").strip().lower() not in (
        "0", "false", "no",
    )

    def _provider_objects() -> dict[str, Any]:
        payload = provider_objects.collect(
            (collected.get("batch_lifecycle") or {}).get("batches") or [],
            enabled=fetch_enabled,
        )
        po_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    _run_collector("provider_batch_objects", po_path, _provider_objects)

    # --- workflow logs ---------------------------------------------------
    wl_path = out_dir / "workflow_logs" / "INDEX.json"
    logs_enabled = os.getenv("PYTHIA_BUNDLE_WORKFLOW_LOGS", "1").strip().lower() not in (
        "0", "false", "no",
    )

    def _logs() -> dict[str, Any]:
        batches = (collected.get("batch_lifecycle") or {}).get("batches") or []
        earliest = min(
            (b.get("submitted_at") for b in batches if b.get("submitted_at")),
            default=None,
        )
        return workflow_logs.collect(
            out_dir,
            pipeline_id=pipeline_id,
            earliest_batch_submitted_at=earliest,
            enabled=logs_enabled,
        )

    collected["workflow_logs"] = _run_collector("workflow_logs", wl_path, _logs) or {}

    # --- env + config ----------------------------------------------------
    ec_path = out_dir / f"env_and_config__{run}.json"

    def _env() -> dict[str, Any]:
        models_used = sorted(
            {
                str(m)
                for m in (data.spd_model_ids or [])
                if m
            }
            | {
                str(r.get("model_id"))
                for r in (collected.get("prompt_cache_rows") or [])
                if r.get("model_id")
            }
        )
        payload = env_config.collect(repo_root=repo_root, models_used=models_used)
        ec_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return payload

    # --- prompt cache ----------------------------------------------------
    pc_path = out_dir / f"prompt_cache_report__{run}.csv"

    def _cache() -> list[dict[str, Any]]:
        rows = prompt_cache.collect(
            con, predicate=data.predicate, params=list(data.predicate_params)
        )
        _write_csv(pc_path, prompt_cache.FIELDNAMES, rows)
        return rows

    collected["prompt_cache_rows"] = _run_collector("prompt_cache_report", pc_path, _cache) or []
    try:
        collected["prompt_cache_summary"] = prompt_cache.summarise(collected["prompt_cache_rows"])
    except Exception as exc:  # noqa: BLE001
        LOG.warning("prompt cache summary failed: %s", exc)
        collected["prompt_cache_summary"] = {}
    _run_collector("env_and_config", ec_path, _env)

    # --- code snapshot ---------------------------------------------------
    cs_path = out_dir / "code_snapshot" / "INDEX.json"

    def _code() -> dict[str, Any]:
        return code_snapshot.collect(
            out_dir,
            repo_root=repo_root,
            con=con,
            current_hs_run_id=data.hs_run_id,
        )

    _run_collector("code_snapshot", cs_path, _code)

    # --- connector freshness ---------------------------------------------
    cf_path = out_dir / f"connector_freshness__{run}.csv"

    def _connectors() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        rows, crisiswatch = connector_freshness.collect(
            con,
            countries=data.resolved_countries_sorted,
            questions=data.questions,
        )
        fieldnames = list(rows[0].keys()) if rows else []
        if fieldnames:
            _write_csv(cf_path, fieldnames, rows)
        (out_dir / f"crisiswatch_inject__{run}.json").write_text(
            json.dumps(crisiswatch, indent=2, default=str), encoding="utf-8"
        )
        return rows, crisiswatch

    result = _run_collector("connector_freshness", cf_path, _connectors) or ([], {})
    collected["connector_rows"], collected["crisiswatch"] = result

    # --- retries ---------------------------------------------------------
    rr_path = out_dir / f"retry_report__{run}.csv"

    def _retries() -> list[dict[str, Any]]:
        rows = retry_report.collect(
            con, predicate=data.predicate, params=list(data.predicate_params)
        )
        _write_csv(rr_path, retry_report.FIELDNAMES, rows)
        return rows

    collected["retry_rows"] = _run_collector("retry_report", rr_path, _retries) or []
    try:
        collected["breaker"] = retry_report.breaker_summary(
            con, predicate=data.predicate, params=list(data.predicate_params)
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("breaker summary failed: %s", exc)
        collected["breaker"] = {}

    # --- model completeness ----------------------------------------------
    mc_path = out_dir / f"model_completeness__{run}.csv"

    def _completeness() -> dict[str, Any]:
        rows, rollup = model_completeness.collect(
            con,
            run_id=data.forecaster_run_id,
            questions=data.questions,
            expected_models=_expected_spd_model_ids(),
            # forecasts_raw stores Track 2 under the stable aggregate name
            # "track2_flash" whatever model backs the track2_spd role, so
            # PYTHIA_TRACK2_MODEL_ID must NOT be read here — it names the
            # model id llm_calls records, which is a different column.
            track2_model="track2_flash",
        )
        _write_csv(mc_path, model_completeness.FIELDNAMES, rows)
        (out_dir / f"model_completeness_rollup__{run}.json").write_text(
            json.dumps(rollup, indent=2, default=str), encoding="utf-8"
        )
        return rollup

    collected["completeness_rollup"] = _run_collector(
        "model_completeness", mc_path, _completeness
    ) or {}

    # --- run comparison ---------------------------------------------------
    rc_path = out_dir / f"run_comparison__{run}.csv"

    def _comparison() -> list[dict[str, Any]]:
        rows = run_comparison.collect(con, current_hs_run_id=data.hs_run_id)
        if rows:
            _write_csv(rc_path, list(rows[0].keys()), rows)
        return rows

    collected["run_comparison"] = _run_collector("run_comparison", rc_path, _comparison) or []

    # --- anomalies (last: it reads everything above) ----------------------
    an_path = out_dir / f"anomalies__{run}.json"

    def _anomalies() -> list[dict[str, Any]]:
        entries = anomalies_mod.build(
            health_checks=health_checks,
            batch_lifecycle=collected.get("batch_lifecycle"),
            prompt_cache_summary=collected.get("prompt_cache_summary"),
            connector_rows=collected.get("connector_rows"),
            crisiswatch=collected.get("crisiswatch"),
            completeness_rollup=collected.get("completeness_rollup"),
            workflow_logs_index=collected.get("workflow_logs"),
            retry_rows=collected.get("retry_rows"),
            breaker=collected.get("breaker"),
            collector_errors=_COLLECTOR_ERRORS,
            file_names={
                "health_report": f"health_report__{run}.json",
                "batch_lifecycle": f"batch_lifecycle__{run}.json",
                "prompt_cache_report": f"prompt_cache_report__{run}.csv",
                "connector_freshness": f"connector_freshness__{run}.csv",
                "model_completeness": f"model_completeness__{run}.csv",
                "retry_report": f"retry_report__{run}.csv",
            },
        )
        an_path.write_text(
            json.dumps(
                {
                    "hs_run_id": data.hs_run_id,
                    "forecaster_run_id": data.forecaster_run_id,
                    "counts": anomalies_mod.counts(entries),
                    "anomalies": entries,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return entries

    collected["anomalies"] = _run_collector("anomalies", an_path, _anomalies) or []
    return collected


# ---------------------------------------------------------------------------
# Flat ZIP packaging
# ---------------------------------------------------------------------------

EXCLUDE_EXTENSIONS = {".duckdb", ".db", ".wal", ".duckdb.wal"}
EXCLUDE_SUFFIXES = {".pyc"}
# The zip the pipeline uploads as pythia-debug-bundle. Above this the
# workflow logs are split into their own artifact rather than dropped —
# content that only exists in the bundle must not be traded for size.
BUNDLE_SIZE_TARGET_BYTES = 25 * 1024 * 1024
SUBDIRS = ("workflow_logs", "code_snapshot")


def _bundle_files(out_dir: Path, zip_paths: set[Path]) -> list[Path]:
    out: list[Path] = []
    for file_path in sorted(out_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix in EXCLUDE_EXTENSIONS or file_path.suffix in EXCLUDE_SUFFIXES:
            continue
        if "__pycache__" in str(file_path) or ".git" in file_path.parts:
            continue
        if file_path in zip_paths or file_path.suffix == ".zip":
            continue
        out.append(file_path)
    return out


def build_flat_zip(
    out_dir: Path,
    zip_path: Path,
    *,
    exclude_top_level: tuple[str, ...] = (),
) -> Path:
    """Package the artifact files into one zip.

    Paths are stored relative to ``out_dir``, so files at the root stay
    flat (the layout every existing consumer parses) while
    ``workflow_logs/`` and ``code_snapshot/`` keep the directory they were
    written into. Never includes the DuckDB database.
    """

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in _bundle_files(out_dir, {zip_path}):
            rel = file_path.relative_to(out_dir)
            if exclude_top_level and rel.parts and rel.parts[0] in exclude_top_level:
                continue
            zf.write(file_path, rel.as_posix())
    return zip_path


def build_bundle_zips(
    out_dir: Path, run_id: str, *, size_target: int = BUNDLE_SIZE_TARGET_BYTES
) -> dict[str, Any]:
    """Build the bundle zip, splitting the workflow logs out if it is too big.

    The split is a last resort and it is recorded: a reader who finds the
    logs missing must be told they exist in a second artifact, not left to
    conclude the collector failed.
    """

    main_zip = out_dir / f"pythia_debug_bundle__{run_id}.zip"
    build_flat_zip(out_dir, main_zip)
    result: dict[str, Any] = {
        "bundle_zip": main_zip.name,
        "bundle_bytes": main_zip.stat().st_size,
        "split": False,
        "size_target_bytes": size_target,
    }
    if main_zip.stat().st_size <= size_target:
        return result

    logs_zip = out_dir / f"pythia_debug_bundle_workflow_logs__{run_id}.zip"
    with zipfile.ZipFile(logs_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in _bundle_files(out_dir, {main_zip, logs_zip}):
            rel = file_path.relative_to(out_dir)
            if rel.parts and rel.parts[0] == "workflow_logs":
                zf.write(file_path, rel.as_posix())
    main_zip.unlink(missing_ok=True)
    build_flat_zip(out_dir, main_zip, exclude_top_level=("workflow_logs",))
    result.update(
        {
            "split": True,
            "bundle_bytes": main_zip.stat().st_size,
            "workflow_logs_zip": logs_zip.name,
            "workflow_logs_bytes": logs_zip.stat().st_size,
            "split_note": (
                "The bundle exceeded its size target, so workflow_logs/ was moved into "
                f"{logs_zip.name} and uploaded as the pythia-debug-bundle-workflow-logs "
                "artifact. Nothing was dropped."
            ),
        }
    )
    return result


def build_triage_only_bundle_markdown(
    con: duckdb.DuckDBPyConnection,
    db_url: str,
    hs_run_id: str,
    provenance_lines: List[str],
) -> str:
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    hs_manifest = _load_hs_run_metadata(con, hs_run_id)
    resolved_countries = (hs_manifest or {}).get("countries") if hs_manifest else []
    requested_countries = (hs_manifest or {}).get("requested_countries") if hs_manifest else []
    skipped_entries = (hs_manifest or {}).get("skipped_entries") if hs_manifest else []
    resolved_countries_sorted = sorted({str(c) for c in (resolved_countries or []) if c})

    hs_triage_rows, n_hazards_triaged_total = _load_hs_triage_summary(con, hs_run_id)
    missing_hs_triage = _load_missing_hs_triage_combos(
        con, hs_run_id, resolved_countries_sorted
    )
    hs_web_rows, question_web_rows = _load_web_research_summaries(
        con, hs_run_id, None, resolved_countries_sorted, []
    )
    web_research_enabled = os.getenv("PYTHIA_WEB_RESEARCH_ENABLED", "0") == "1"
    retriever_enabled = os.getenv("PYTHIA_RETRIEVER_ENABLED", "0") == "1"
    hs_research_web_search = os.getenv("PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED", "0")
    spd_web_search = os.getenv("PYTHIA_SPD_WEB_SEARCH_ENABLED", "0")
    hs_web_research_active = retriever_enabled or hs_research_web_search == "1"
    research_web_research_active = retriever_enabled or hs_research_web_search == "1"
    web_research_accounting = _web_research_accounting(con, None, hs_run_id)
    hs_web_research_rows, hs_web_research_failures = _load_web_research_summary(
        con, "hs_web_research", None, hs_run_id
    )
    research_web_research_rows, research_web_research_failures = _load_web_research_summary(
        con, "research_web_research", None, hs_run_id
    )
    self_search_rows, self_search_failures = _load_web_research_summary(
        con, "forecast_web_research", None, hs_run_id
    )
    self_search_call_total = sum(int(row.get("n_calls") or 0) for row in (self_search_rows or []))
    web_research_lines = _web_research_markdown(
        hs_web_rows,
        question_web_rows,
        web_research_enabled=web_research_enabled,
        retriever_enabled=retriever_enabled,
        self_search_call_total=self_search_call_total,
        hs_web_research_rows=hs_web_research_rows,
        hs_web_research_failures=hs_web_research_failures,
        research_web_research_rows=research_web_research_rows,
        research_web_research_failures=research_web_research_failures,
        self_search_rows=self_search_rows,
        self_search_failures=self_search_failures,
        accounting=web_research_accounting,
    )

    llm_columns = _llm_calls_columns(con)
    predicate, params, predicate_strategy = _hs_llm_filter(
        llm_columns, hs_run_id, resolved_countries_sorted
    )
    if not predicate:
        predicate = "phase = 'hs_triage'"
        params = []
        predicate_strategy = "phase_only"

    llm_call_counts = _load_llm_call_counts(con, predicate, params)
    llm_error_rows = [row for row in llm_call_counts if int(row.get("n_errors") or 0) > 0]
    latency_block = render_latency_markdown(con, predicate, params, strategy_label=predicate_strategy)

    lines: List[str] = []
    lines.append(f"# Pythia v2 Debug Bundle — HS run {hs_run_id}")
    lines.append("")
    lines.append(f"_Generated at {now}_")
    lines.append("")
    lines.append("## Run manifest")
    lines.append("")
    lines.append(f"- Database URL: `{db_url}`")
    lines.append(
        "- HS/Research web search enabled (PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED): "
        f"`{hs_research_web_search}`"
    )
    lines.append(f"- Retriever enabled (PYTHIA_RETRIEVER_ENABLED): `{int(retriever_enabled)}`")
    lines.append(f"- HS web research active (flag or retriever): `{int(hs_web_research_active)}`")
    lines.append(f"- Research web research active (flag or retriever): `{int(research_web_research_active)}`")
    lines.append(f"- HS evidence active (retriever or HS flag): `{int(hs_web_research_active)}`")
    lines.append(f"- Research evidence active (retriever or HS flag): `{int(research_web_research_active)}`")
    lines.append(
        "- SPD web search enabled (PYTHIA_SPD_WEB_SEARCH_ENABLED): "
        f"`{spd_web_search}`"
    )
    lines.append(f"- HS run_id: `{hs_run_id}`")
    lines.append("- Forecaster run_id: (none; triage-only bundle)")
    lines.append(
        "- Requested countries (as provided): "
        + (", ".join(requested_countries) if requested_countries else "(none)")
    )
    lines.append(
        "- Resolved ISO3s (authoritative): "
        + (", ".join(resolved_countries_sorted) if resolved_countries_sorted else "(none)")
    )
    lines.append(f"- Skipped country entries: `{len(skipped_entries or [])}`")
    lines.append(f"- n_countries_resolved: `{len(resolved_countries_sorted)}`")
    lines.append(f"- n_hazards_triaged_total: `{n_hazards_triaged_total}`")
    lines.append("")
    lines.append("### Run scope")
    lines.append("")
    lines.append(
        f"- PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED: `{hs_research_web_search}`"
    )
    lines.append(f"- PYTHIA_RETRIEVER_ENABLED: `{int(retriever_enabled)}`")
    lines.append(f"- HS web research active (flag or retriever): `{int(hs_web_research_active)}`")
    lines.append(f"- Research web research active (flag or retriever): `{int(research_web_research_active)}`")
    lines.append(f"- HS evidence active (retriever or HS flag): `{int(hs_web_research_active)}`")
    lines.append(f"- Research evidence active (retriever or HS flag): `{int(research_web_research_active)}`")
    lines.append(f"- PYTHIA_SPD_WEB_SEARCH_ENABLED: `{spd_web_search}`")
    lines.append("")
    lines.extend(provenance_lines)
    lines.extend(web_research_lines)
    lines.append("### Skipped country entries")
    lines.append("")
    lines.append("| raw | normalized | reason |")
    lines.append("| --- | --- | --- |")
    if skipped_entries:
        for entry in sorted(
            skipped_entries,
            key=lambda e: (str(e.get("raw") or ""), str(e.get("normalized") or "")),
        ):
            lines.append(
                f"| {entry.get('raw', '')} | {entry.get('normalized', '')} | {entry.get('reason', '')} |"
            )
    else:
        lines.append("| (none) | (none) | (none) |")
    lines.append("")

    lines.append("### Hazards triaged by country")
    lines.append("")
    lines.append("| iso3 | n_hazards | hazards |")
    lines.append("| ---- | --------- | -------- |")
    if hs_triage_rows:
        for row in hs_triage_rows:
            hazards_list = row.get("hazards_sorted") or []
            lines.append(
                f"| {row.get('iso3')} | {row.get('n_hazards')} | "
                f"{', '.join(hazards_list) if hazards_list else ''} |"
            )
    else:
        lines.append("| (none) | 0 | (none) |")
    lines.append("")

    lines.append("### Missing HS triage combos")
    lines.append("")
    lines.append("| iso3 | hazard_code |")
    lines.append("| ---- | ----------- |")
    if missing_hs_triage:
        for row in missing_hs_triage:
            lines.append(f"| {row.get('iso3')} | {row.get('hazard_code')} |")
    else:
        lines.append("| (none) | (none) |")
    lines.append("")

    lines.append("### Missing HS triage combos")
    lines.append("")
    lines.append("| iso3 | hazard_code |")
    lines.append("| ---- | ----------- |")
    if missing_hs_triage:
        for row in missing_hs_triage:
            lines.append(f"| {row.get('iso3')} | {row.get('hazard_code')} |")
    else:
        lines.append("| (none) | (none) |")
    lines.append("")

    lines.append("### LLM calls by phase/provider/model_id (hs_run only)")
    lines.append("")
    lines.append("| phase | provider | model_id | n_calls | n_errors |")
    lines.append("| ----- | -------- | -------- | ------- | -------- |")
    if llm_call_counts:
        for row in llm_call_counts:
            lines.append(
                f"| {row.get('phase')} | {row.get('provider')} | {row.get('model_id')} | "
                f"{row.get('n_calls')} | {row.get('n_errors')} |"
            )
    else:
        lines.append("| (none) | (none) | (none) | 0 | 0 |")
    lines.append("")

    lines.append("### LLM error summary (hs_run only)")
    lines.append("")
    lines.append("| phase | provider | model_id | n_errors |")
    lines.append("| ----- | -------- | -------- | -------- |")
    if llm_error_rows:
        for row in llm_error_rows:
            lines.append(
                f"| {row.get('phase')} | {row.get('provider')} | {row.get('model_id')} | {row.get('n_errors')} |"
            )
    else:
        lines.append("| (none) | (none) | (none) | 0 |")
    lines.append("")
    lines.append("### Latency (hs_run only)")
    lines.append("")
    lines.append(latency_block)
    lines.append("")
    lines.append("_No forecaster run detected for this HS run._")
    lines.append("")
    return "\n".join(lines)


def build_debug_bundle_markdown(
    con: duckdb.DuckDBPyConnection,
    db_url: str,
    forecaster_run_id: str,
    hs_run_id: str | None,
    questions: list[dict[str, Any]],
    provenance_lines: List[str],
) -> str:
    lines: List[str] = []

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    hs_run_id_for_costs = hs_run_id or _resolve_hs_run_id_for_forecast(con, forecaster_run_id)
    usage_by_phase_warning: str | None = None
    try:
        usage_by_phase = _aggregate_usage_by_phase(con, forecaster_run_id, hs_run_id_for_costs)
    except Exception as exc:  # pragma: no cover - defensive
        usage_by_phase = {}
        usage_by_phase_warning = f"Error aggregating llm_calls usage: {exc}"
    triage_cache: dict[tuple[str, str, str], dict[str, Any] | None] = {}

    hazards = sorted(
        {(q.get("hazard_code") or "").upper() for q in questions if q.get("hazard_code")}
    )
    iso3s = sorted({(q.get("iso3") or "").upper() for q in questions if q.get("iso3")})
    metrics = sorted({(q.get("metric") or "").upper() for q in questions if q.get("metric")})
    hs_run_ids = sorted({q.get("hs_run_id") for q in questions if q.get("hs_run_id")})

    scenario_status_rows: list[dict[str, Any]] = []
    for q in questions:
        qid = q.get("question_id")
        iso3 = q.get("iso3") or ""
        hz = q.get("hazard_code") or ""
        metric = q.get("metric") or ""
        hs_run_id = q.get("hs_run_id") or hs_run_id_for_costs
        triage_entry = _load_triage_entry(con, hs_run_id, iso3, hz, cache=triage_cache)
        triage_tier = (triage_entry or {}).get("tier")
        q_track_raw = q.get("track")
        q_track = int(q_track_raw) if q_track_raw is not None else None
        expected, reason = _scenario_expected(hz, metric, triage_entry, track=q_track)
        call_count = _load_scenario_call_count(con, forecaster_run_id, str(qid))
        if call_count > 0:
            status = "generated"
        elif not expected:
            status = f"skipped_by_design: {reason or 'not_expected'}"
        else:
            status = "missing_unexpected"
        scenario_status_rows.append(
            {
                "question_id": qid,
                "iso3": iso3,
                "hazard_code": hz,
                "metric": metric,
                "hs_run_id": hs_run_id,
                "triage_tier": triage_tier,
                "status": status,
                "expected": expected,
                "call_count": call_count,
            }
        )
    scenario_status_by_qid = {row["question_id"]: row for row in scenario_status_rows}
    forecasts_raw_counts = _load_forecasts_raw_counts(con, forecaster_run_id)
    forecasts_ensemble_counts = _load_forecasts_ensemble_counts(con, forecaster_run_id)
    spd_model_ids = _load_spd_llm_model_ids(con, forecaster_run_id)
    question_run_metrics_warning: str | None = None
    try:
        _compute_question_run_metrics(con, forecaster_run_id, questions)
    except Exception as exc:  # pragma: no cover - defensive
        question_run_metrics_warning = f"Error computing question_run_metrics: {exc}"
    question_run_metrics = _load_question_run_metrics(con, forecaster_run_id)
    manifest_hs_run_id = hs_run_id_for_costs or (hs_run_ids[0] if hs_run_ids else None)
    hs_manifest = _load_hs_run_metadata(con, manifest_hs_run_id)
    question_ids = sorted([str(q.get("question_id")) for q in questions if q.get("question_id")])
    requested_countries = (
        (hs_manifest or {}).get("requested_countries") if hs_manifest is not None else []
    ) or []
    resolved_countries = (
        (hs_manifest or {}).get("countries") if hs_manifest is not None else list(iso3s)
    ) or list(iso3s)
    resolved_countries_sorted = sorted({str(c) for c in resolved_countries if c})
    skipped_entries = ((hs_manifest or {}).get("skipped_entries") if hs_manifest else []) or []
    hs_triage_rows, n_hazards_triaged_total = _load_hs_triage_summary(con, manifest_hs_run_id)
    missing_hs_triage = _load_missing_hs_triage_combos(
        con, manifest_hs_run_id, resolved_countries_sorted
    )
    n_questions_by_hazard: dict[str, int] = {}
    n_questions_by_iso3: dict[str, int] = {}
    for q in questions:
        hz = (q.get("hazard_code") or "").upper()
        iso_val = (q.get("iso3") or "").upper()
        if hz:
            n_questions_by_hazard[hz] = n_questions_by_hazard.get(hz, 0) + 1
        if iso_val:
            n_questions_by_iso3[iso_val] = n_questions_by_iso3.get(iso_val, 0) + 1

    hs_web_rows, question_web_rows = _load_web_research_summaries(
        con, manifest_hs_run_id, forecaster_run_id, resolved_countries_sorted, question_ids
    )
    web_research_enabled = os.getenv("PYTHIA_WEB_RESEARCH_ENABLED", "0") == "1"
    retriever_enabled = os.getenv("PYTHIA_RETRIEVER_ENABLED", "0") == "1"
    hs_research_web_search = os.getenv("PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED", "0")
    spd_web_search = os.getenv("PYTHIA_SPD_WEB_SEARCH_ENABLED", "0")
    hs_web_research_active = retriever_enabled or hs_research_web_search == "1"
    research_web_research_active = retriever_enabled or hs_research_web_search == "1"
    web_research_accounting = _web_research_accounting(con, forecaster_run_id, manifest_hs_run_id)
    hs_web_research_rows, hs_web_research_failures = _load_web_research_summary(
        con, "hs_web_research", forecaster_run_id, manifest_hs_run_id
    )
    research_web_research_rows, research_web_research_failures = _load_web_research_summary(
        con, "research_web_research", forecaster_run_id, manifest_hs_run_id
    )
    self_search_rows, self_search_failures = _load_web_research_summary(
        con, "forecast_web_research", forecaster_run_id, manifest_hs_run_id
    )
    self_search_call_total = sum(int(row.get("n_calls") or 0) for row in (self_search_rows or []))
    web_research_lines = _web_research_markdown(
        hs_web_rows,
        question_web_rows,
        web_research_enabled=web_research_enabled,
        retriever_enabled=retriever_enabled,
        self_search_call_total=self_search_call_total,
        hs_web_research_rows=hs_web_research_rows,
        hs_web_research_failures=hs_web_research_failures,
        research_web_research_rows=research_web_research_rows,
        research_web_research_failures=research_web_research_failures,
        self_search_rows=self_search_rows,
        self_search_failures=self_search_failures,
        accounting=web_research_accounting,
    )

    llm_columns = _llm_calls_columns(con)
    predicate, params, predicate_strategy = _combined_llm_filter(
        llm_columns, forecaster_run_id, manifest_hs_run_id, question_ids, resolved_countries_sorted
    )
    llm_call_counts: list[dict[str, Any]] = []
    llm_calls_skip_note: str | None = None
    llm_error_rows: list[dict[str, Any]] = []
    self_search_stats = {"requests": 0, "sources": 0}
    self_search_warning: str | None = None
    try:
        llm_call_counts = _load_llm_call_counts(con, predicate, params)
        llm_error_rows = [row for row in llm_call_counts if int(row.get("n_errors") or 0) > 0]
        self_search_stats = _load_self_search_stats(con, predicate, params)
    except Exception as exc:  # pragma: no cover - defensive
        llm_calls_skip_note = f"Error loading llm_calls: {exc}"
        llm_call_counts = []
        llm_error_rows = []
        self_search_warning = llm_calls_skip_note
    latency_block = render_latency_markdown(con, predicate, params, predicate_strategy)

    lines.append(f"# Pythia v2 Debug Bundle — Run {forecaster_run_id}")
    lines.append("")
    lines.append(f"_Generated at {now}_")
    lines.append("")

    lines.append("## Run manifest")
    lines.append("")
    lines.append(f"- Database URL: `{db_url}`")
    lines.append(
        "- HS/Research web search enabled (PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED): "
        f"`{hs_research_web_search}`"
    )
    lines.append(f"- Retriever enabled (PYTHIA_RETRIEVER_ENABLED): `{int(retriever_enabled)}`")
    lines.append(f"- HS web research active (flag or retriever): `{int(hs_web_research_active)}`")
    lines.append(f"- Research web research active (flag or retriever): `{int(research_web_research_active)}`")
    lines.append(f"- HS evidence active (retriever or HS flag): `{int(hs_web_research_active)}`")
    lines.append(f"- Research evidence active (retriever or HS flag): `{int(research_web_research_active)}`")
    lines.append(
        "- SPD web search enabled (PYTHIA_SPD_WEB_SEARCH_ENABLED): "
        f"`{spd_web_search}`"
    )
    lines.append(f"- Forecast run_id: `{forecaster_run_id}`")
    lines.append(f"- HS run_id: `{manifest_hs_run_id or 'unknown'}`")
    lines.append(
        "- Requested countries (as provided): "
        + (", ".join(requested_countries) if requested_countries else "(none)")
    )
    lines.append(
        "- Resolved ISO3s (authoritative): "
        + (", ".join(resolved_countries_sorted) if resolved_countries_sorted else "(none)")
    )
    lines.append(f"- Skipped country entries: `{len(skipped_entries)}`")
    lines.append(f"- n_countries_resolved: `{len(resolved_countries_sorted)}`")
    lines.append(f"- n_questions_total: `{len(question_ids)}`")
    lifecycle_counts = _question_lifecycle_counts(con, forecaster_run_id)
    lines.append(f"- n_question_ids_researched: `{lifecycle_counts.get('research', 0)}`")
    lines.append(f"- n_question_ids_forecasted: `{lifecycle_counts.get('forecast', 0)}`")
    lines.append(f"- n_question_ids_scenarios: `{lifecycle_counts.get('scenario', 0)}`")
    researched_not_forecasted = _question_ids_researched_not_forecasted(con, forecaster_run_id)
    lines.append(f"- n_question_ids_researched_not_forecasted: `{len(researched_not_forecasted)}`")
    lines.append(
        "- question_ids_researched_not_forecasted: "
        + (", ".join(researched_not_forecasted) if researched_not_forecasted else "(none)")
    )
    lines.append(
        "- n_questions_by_hazard_code: "
        + (
            ", ".join(f"{hz}:{n_questions_by_hazard[hz]}" for hz in sorted(n_questions_by_hazard))
            if n_questions_by_hazard
            else "(none)"
        )
    )
    lines.append(
        "- n_questions_by_iso3: "
        + (
            ", ".join(f"{iso3}:{n_questions_by_iso3[iso3]}" for iso3 in sorted(n_questions_by_iso3))
            if n_questions_by_iso3
            else "(none)"
        )
    )
    lines.append(f"- n_hazards_triaged_total: `{n_hazards_triaged_total}`")
    lines.append("")
    lines.append("### Run scope")
    lines.append("")
    lines.append(
        f"- PYTHIA_HS_RESEARCH_WEB_SEARCH_ENABLED: `{hs_research_web_search}`"
    )
    lines.append(f"- PYTHIA_RETRIEVER_ENABLED: `{int(retriever_enabled)}`")
    lines.append(f"- HS web research active (flag or retriever): `{int(hs_web_research_active)}`")
    lines.append(f"- Research web research active (flag or retriever): `{int(research_web_research_active)}`")
    lines.append(f"- HS evidence active (retriever or HS flag): `{int(hs_web_research_active)}`")
    lines.append(f"- Research evidence active (retriever or HS flag): `{int(research_web_research_active)}`")
    lines.append(f"- PYTHIA_SPD_WEB_SEARCH_ENABLED: `{spd_web_search}`")
    lines.append("")
    lines.extend(provenance_lines)
    lines.extend(web_research_lines)
    lines.append("### Skipped country entries")
    lines.append("")
    lines.append("| raw | normalized | reason |")
    lines.append("| --- | --- | --- |")
    if skipped_entries:
        for entry in sorted(
            skipped_entries,
            key=lambda e: (str(e.get("raw") or ""), str(e.get("normalized") or "")),
        ):
            lines.append(
                f"| {entry.get('raw', '')} | {entry.get('normalized', '')} | {entry.get('reason', '')} |"
            )
    else:
        lines.append("| (none) | (none) | (none) |")
    lines.append("")
    lines.append("### Hazards triaged by country")
    lines.append("")
    lines.append("| iso3 | n_hazards | hazards |")
    lines.append("| ---- | --------- | -------- |")
    if hs_triage_rows:
        for row in hs_triage_rows:
            hazards_list = row.get("hazards_sorted") or []
            lines.append(
                f"| {row.get('iso3')} | {row.get('n_hazards')} | "
                f"{', '.join(hazards_list) if hazards_list else ''} |"
            )
    else:
        lines.append("| (none) | 0 | (none) |")
    lines.append("")
    lines.append("### Question list")
    lines.append("")
    lines.append(f"- Total questions: `{len(question_ids)}`")
    lines.append("- Question IDs: " + (", ".join(question_ids) if question_ids else "(none)"))
    lines.append("")
    lines.append("| question_id | iso3 | hazard | metric | target_month (window end) | wording |")
    lines.append("| ----------- | ---- | ------ | ------ | ------------ | ------- |")
    if questions:
        for q in sorted(
            questions,
            key=lambda r: (
                str(r.get("iso3") or ""),
                str(r.get("hazard_code") or ""),
                str(r.get("metric") or ""),
                str(r.get("question_id") or ""),
            ),
        ):
            lines.append(
                f"| {q.get('question_id')} | {q.get('iso3')} | {q.get('hazard_code')} | "
                f"{q.get('metric')} | {q.get('target_month')} | "
                f"{(q.get('wording') or '').strip()} |"
            )
    else:
        lines.append("| (none) | (none) | (none) | (none) | (none) | (none) |")
    lines.append("")
    lines.append("### Question counts by hazard_code")
    lines.append("")
    lines.append("| hazard_code | n_questions |")
    lines.append("| ----------- | ----------- |")
    if n_questions_by_hazard:
        for hz in sorted(n_questions_by_hazard):
            lines.append(f"| {hz} | {n_questions_by_hazard[hz]} |")
    else:
        lines.append("| (none) | 0 |")
    lines.append("")
    lines.append("### Question counts by ISO3")
    lines.append("")
    lines.append("| iso3 | n_questions |")
    lines.append("| ---- | ------------ |")
    if n_questions_by_iso3:
        for iso_val in sorted(n_questions_by_iso3):
            lines.append(f"| {iso_val} | {n_questions_by_iso3[iso_val]} |")
    else:
        lines.append("| (none) | 0 |")
    lines.append("")
    lines.append("### Question run metrics")
    lines.append("")
    if question_run_metrics_warning:
        lines.append(f"_Note: {question_run_metrics_warning}_")
        lines.append("")
    lines.append(
        "_wall_ms spans a question's first to last llm_calls timestamp; under the "
        "staged pipeline that includes provider batch latency and the poller's "
        "ticks, so queue_ms (wall minus compute) measures batch wait, not model "
        "queueing. compute_ms sums the slowest call per phase; a batched call "
        "records elapsed_ms=0, so a fully batched question (every Track-2 "
        "question on a normal cycle) legitimately reads 0._"
    )
    lines.append("")
    wall_values = [
        float(row.get("wall_ms"))
        for row in question_run_metrics
        if row.get("wall_ms") is not None
    ]
    compute_values = [
        float(row.get("compute_ms"))
        for row in question_run_metrics
        if row.get("compute_ms") is not None
    ]
    queue_values = [
        float(row.get("queue_ms"))
        for row in question_run_metrics
        if row.get("queue_ms") is not None
    ]
    cost_values = [
        float(row.get("cost_usd"))
        for row in question_run_metrics
        if row.get("cost_usd") is not None
    ]
    wall_median = _percentile(wall_values, 50)
    wall_p95 = _percentile(wall_values, 95)
    compute_median = _percentile(compute_values, 50)
    compute_p95 = _percentile(compute_values, 95)
    queue_median = _percentile(queue_values, 50)
    queue_p95 = _percentile(queue_values, 95)
    cost_median = _percentile(cost_values, 50)
    cost_p95 = _percentile(cost_values, 95)
    lines.append("| metric | value |")
    lines.append("| ------ | ----- |")
    lines.append(f"| wall_ms_median | {int(wall_median) if wall_median is not None else 0} |")
    lines.append(f"| wall_ms_p95 | {int(wall_p95) if wall_p95 is not None else 0} |")
    lines.append(
        f"| compute_ms_median | {int(compute_median) if compute_median is not None else 0} |"
    )
    lines.append(
        f"| compute_ms_p95 | {int(compute_p95) if compute_p95 is not None else 0} |"
    )
    lines.append(
        f"| queue_ms_median | {int(queue_median) if queue_median is not None else 0} |"
    )
    lines.append(
        f"| queue_ms_p95 | {int(queue_p95) if queue_p95 is not None else 0} |"
    )
    lines.append(f"| cost_usd_median | {cost_median:.4f} |" if cost_median is not None else "| cost_usd_median | 0.0000 |")
    lines.append(f"| cost_usd_p95 | {cost_p95:.4f} |" if cost_p95 is not None else "| cost_usd_p95 | 0.0000 |")
    lines.append("")

    lines.append("#### Top 10 slowest questions")
    lines.append("")
    lines.append("| question_id | wall_ms | iso3 | hazard_code | metric |")
    lines.append("| ----------- | ------- | ---- | ----------- | ------ |")
    if question_run_metrics:
        for row in sorted(
            question_run_metrics,
            key=lambda r: (
                -(float(r.get("wall_ms") or 0.0)),
                str(r.get("question_id") or ""),
            ),
        )[:10]:
            lines.append(
                f"| {row.get('question_id')} | {int(row.get('wall_ms') or 0)} | "
                f"{row.get('iso3') or ''} | {row.get('hazard_code') or ''} | {row.get('metric') or ''} |"
            )
    else:
        lines.append("| (none) | 0 | (none) | (none) | (none) |")
    lines.append("")

    lines.append("#### Top 10 slowest questions by compute_ms")
    lines.append("")
    lines.append("| question_id | compute_ms | iso3 | hazard_code | metric |")
    lines.append("| ----------- | ---------- | ---- | ----------- | ------ |")
    if question_run_metrics:
        for row in sorted(
            question_run_metrics,
            key=lambda r: (
                -(float(r.get("compute_ms") or 0.0)),
                str(r.get("question_id") or ""),
            ),
        )[:10]:
            lines.append(
                f"| {row.get('question_id')} | {int(row.get('compute_ms') or 0)} | "
                f"{row.get('iso3') or ''} | {row.get('hazard_code') or ''} | {row.get('metric') or ''} |"
            )
    else:
        lines.append("| (none) | 0 | (none) | (none) | (none) |")
    lines.append("")

    lines.append("#### Top 10 slowest questions by queue_ms")
    lines.append("")
    lines.append("| question_id | queue_ms | iso3 | hazard_code | metric |")
    lines.append("| ----------- | -------- | ---- | ----------- | ------ |")
    if question_run_metrics:
        for row in sorted(
            question_run_metrics,
            key=lambda r: (
                -(float(r.get("queue_ms") or 0.0)),
                str(r.get("question_id") or ""),
            ),
        )[:10]:
            lines.append(
                f"| {row.get('question_id')} | {int(row.get('queue_ms') or 0)} | "
                f"{row.get('iso3') or ''} | {row.get('hazard_code') or ''} | {row.get('metric') or ''} |"
            )
    else:
        lines.append("| (none) | 0 | (none) | (none) | (none) |")
    lines.append("")

    lines.append("#### Top 10 most expensive questions")
    lines.append("")
    lines.append("| question_id | cost_usd | iso3 | hazard_code | metric |")
    lines.append("| ----------- | -------- | ---- | ----------- | ------ |")
    if question_run_metrics:
        for row in sorted(
            question_run_metrics,
            key=lambda r: (
                -(float(r.get("cost_usd") or 0.0)),
                str(r.get("question_id") or ""),
            ),
        )[:10]:
            lines.append(
                f"| {row.get('question_id')} | {float(row.get('cost_usd') or 0.0):.4f} | "
                f"{row.get('iso3') or ''} | {row.get('hazard_code') or ''} | {row.get('metric') or ''} |"
            )
    else:
        lines.append("| (none) | 0.0000 | (none) | (none) | (none) |")
    lines.append("")

    lines.append("#### Questions with missing SPD model_ids")
    lines.append("")
    missing_rows: list[dict[str, Any]] = []
    for row in question_run_metrics:
        raw = row.get("missing_model_ids_json") or "[]"
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else list(raw)
        except Exception:
            parsed = []
        if parsed:
            missing_rows.append(
                {
                    "question_id": row.get("question_id"),
                    "missing_model_ids": parsed,
                }
            )
    lines.append(f"- Questions with missing SPD model contributions: `{len(missing_rows)}`")
    lines.append("")
    lines.append("| question_id | missing_model_ids |")
    lines.append("| ----------- | ------------------ |")
    if missing_rows:
        for row in sorted(missing_rows, key=lambda r: str(r.get("question_id") or "")):
            missing_ids = row.get("missing_model_ids") or []
            missing_str = ", ".join(str(mid) for mid in missing_ids)
            lines.append(f"| {row.get('question_id')} | {missing_str} |")
    else:
        lines.append("| (none) | (none) |")
    lines.append("")
    lines.extend(latency_block.splitlines())
    lines.append("")
    lines.append("### Forecast rows by model")
    lines.append("")
    lines.append("#### forecasts_ensemble")
    lines.append("")
    lines.append("| model_name | n_rows |")
    lines.append("|------------|--------|")
    if forecasts_ensemble_counts:
        for row in forecasts_ensemble_counts:
            lines.append(f"| {row.get('model_name')} | {row.get('n_rows')} |")
    else:
        lines.append("| (none) | 0 |")
    lines.append("")
    lines.append("#### forecasts_raw")
    lines.append("")
    lines.append("| model_name | n_rows |")
    lines.append("|------------|--------|")
    if forecasts_raw_counts:
        for row in forecasts_raw_counts:
            lines.append(f"| {row.get('model_name')} | {row.get('n_rows')} |")
    else:
        lines.append("| (none) | 0 |")
    lines.append("")

    lines.append("## 1. Overview")
    lines.append("")
    lines.append(f"- Question types included (by hazard_code, metric): {len(questions)}")
    lines.append(f"- Hazards present: {', '.join(hazards) if hazards else '(none)'}")
    lines.append(f"- ISO3s present: {', '.join(iso3s) if iso3s else '(none)'}")
    lines.append(f"- Metrics present: {', '.join(metrics) if metrics else '(none)'}")
    lines.append(f"- Linked HS run IDs: {', '.join(hs_run_ids) if hs_run_ids else '(none)'}")
    lines.append("")

    total_prompt = sum(v["prompt_tokens"] for v in usage_by_phase.values())
    total_completion = sum(v["completion_tokens"] for v in usage_by_phase.values())
    total_tokens = sum(v["total_tokens"] for v in usage_by_phase.values())
    total_cost = sum(v["total_cost_usd"] for v in usage_by_phase.values())

    lines.append("### 1.1 Token & Cost Summary")
    lines.append("")
    lines.append(f"- Total prompt tokens (all phases): `{int(total_prompt)}`")
    lines.append(f"- Total completion tokens (all phases): `{int(total_completion)}`")
    lines.append(f"- Total tokens (all phases): `{int(total_tokens)}`")
    lines.append(f"- Total cost (USD, all phases): `{total_cost:.4f}`")
    lines.append("")

    lines.append("### 1.2 Cost by Phase")
    lines.append("")
    if usage_by_phase_warning:
        lines.append(f"_Note: {usage_by_phase_warning}_")
        lines.append("")
    lines.append("| phase | prompt_tokens | completion_tokens | total_tokens | total_cost_usd |")
    lines.append("|-------|---------------|-------------------|--------------|----------------|")
    for phase, agg in sorted(usage_by_phase.items()):
        lines.append(
            f"| {phase} | {int(agg['prompt_tokens'])} | {int(agg['completion_tokens'])} | "
            f"{int(agg['total_tokens'])} | {agg['total_cost_usd']:.4f} |"
        )
    lines.append("")

    lines.append("### 1.3 Scenario status (per question)")
    lines.append("")
    lines.append(
        "| question_id | iso3 | hazard | metric | hs_run_id | triage_tier | scenario_status | calls_logged |"
    )
    lines.append("|-------------|------|--------|--------|----------|-------------|-----------------|--------------|")
    for row in sorted(
        scenario_status_rows,
        key=lambda r: (
            str(r.get("iso3") or ""),
            str(r.get("hazard_code") or ""),
            str(r.get("metric") or ""),
            str(r.get("question_id") or ""),
        ),
    ):
        lines.append(
            f"| {row.get('question_id')} | {row.get('iso3')} | {row.get('hazard_code')} | "
            f"{row.get('metric')} | {row.get('hs_run_id')} | {row.get('triage_tier') or ''} | "
            f"{row.get('status')} | {row.get('call_count')} |"
        )
    lines.append("")

    lines.append("### 1.4 LLM calls by phase/provider/model_id (this run)")
    lines.append("")
    if llm_calls_skip_note:
        lines.append(f"_Note: {llm_calls_skip_note}_")
        lines.append("")
    lines.append("| phase | provider | model_id | n_calls | n_errors |")
    lines.append("| ----- | -------- | -------- | ------- | -------- |")
    if llm_call_counts:
        for row in llm_call_counts:
            lines.append(
                f"| {row.get('phase')} | {row.get('provider')} | {row.get('model_id')} | "
                f"{row.get('n_calls')} | {row.get('n_errors')} |"
            )
    else:
        lines.append("| (none) | (none) | (none) | 0 | 0 |")
    lines.append("")

    lines.append("### 1.5 LLM error summary (this run)")
    lines.append("")
    lines.append("| phase | provider | model_id | n_errors |")
    lines.append("| ----- | -------- | -------- | -------- |")
    if llm_error_rows:
        for row in llm_error_rows:
            lines.append(
                f"| {row.get('phase')} | {row.get('provider')} | {row.get('model_id')} | {row.get('n_errors')} |"
            )
    else:
        lines.append("| (none) | (none) | (none) | 0 |")
    lines.append("")
    lines.append("### 1.6 forecasts_raw model writes (DB truth)")
    lines.append("")
    lines.append("| model_name | n_rows |")
    lines.append("|------------|--------|")
    if forecasts_raw_counts:
        for row in forecasts_raw_counts:
            lines.append(f"| {row.get('model_name')} | {row.get('n_rows')} |")
    else:
        lines.append("| (none) | 0 |")
    lines.append("")

    lines.append("### 1.7 Ensemble participation summary")
    lines.append("")
    lines.extend(_ensemble_participation_summary(forecasts_raw_counts, spd_model_ids))
    lines.append("")

    lines.append("### 1.8 Self-search (forecast_web_research) summary")
    lines.append("")
    if self_search_warning:
        lines.append(f"_Note: {self_search_warning}_")
        lines.append("")
    lines.append("_Note: Self-search is model-specific and may be empty when the shared retriever handles evidence._")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    lines.append(f"| self_search_requests_count | {self_search_stats.get('requests', 0)} |")
    lines.append(f"| self_search_sources_count | {self_search_stats.get('sources', 0)} |")
    lines.append("")

    lines.append("## 2. Question Types")
    lines.append("")

    for idx, question in enumerate(questions, start=1):
        qid = question.get("question_id")
        hs_run_id = question.get("hs_run_id")
        iso3_val = question.get("iso3")
        hz_val = question.get("hazard_code")
        metric_val = question.get("metric")
        target_month = question.get("target_month")
        window_start_date = question.get("window_start_date")
        window_end_date = question.get("window_end_date")
        wording = question.get("wording")

        triage_entry = _load_triage_entry(
            con, hs_run_id or hs_run_id_for_costs, iso3_val, hz_val, cache=triage_cache
        )
        triage_tier = (triage_entry or {}).get("tier") or _load_triage_tier(
            con, hs_run_id or hs_run_id_for_costs, iso3_val, hz_val
        )
        spd_status = _load_spd_status(con, forecaster_run_id, qid)
        scenario_meta = scenario_status_by_qid.get(qid) or {}

        section_label = f"{iso3_val} / {hz_val} / {metric_val}"
        lines.append(f"### 2.{idx} {section_label} (question_id={qid})")
        lines.append("")
        lines.append(f"- ISO3: `{iso3_val}`")
        lines.append(f"- Hazard: `{hz_val}`")
        lines.append(f"- Metric: `{metric_val}`")
        lines.append(f"- HS run_id: `{hs_run_id or 'N/A'}`")
        lines.append(f"- Triaged tier: `{triage_tier or 'N/A'}`")
        lines.append(f"- SPD ensemble status: `{spd_status or 'N/A'}`")
        lines.append(
            f"- Scenario status: `{scenario_meta.get('status', 'unknown')}` "
            f"(calls logged: {scenario_meta.get('call_count', 0)}, "
            f"expected: {'yes' if scenario_meta.get('expected') else 'no'})"
        )
        lines.append(f"- Target month: `{target_month}`")
        lines.append(f"- Window: `{window_start_date}` → `{window_end_date}`")
        lines.append(f"- Wording: {wording}")
        lines.append("")

        try:
            calls = _load_llm_calls_for_question(
                con, forecaster_run_id, qid, iso3_val, hz_val, hs_run_id
            )
        except Exception as exc:  # pragma: no cover - defensive
            calls = {}
            lines.append(f"_Note: unable to load llm_calls for this question ({exc})._")
            lines.append("")

        lines.append(f"#### 2.{idx}.1 Horizon Scanner (HS)")
        _append_stage_block(lines, "hs_triage", calls.get("hs_triage"))
        lines.append("")

        lines.append(f"#### 2.{idx}.2 Research (Research v2)")
        _append_stage_block(lines, "research_v2", calls.get("research_v2"))
        lines.append("")

        lines.append(f"#### 2.{idx}.3 Forecaster (SPD v2)")
        _append_stage_block(lines, "spd_v2", calls.get("spd_v2"))
        lines.append("")

        lines.append(f"#### 2.{idx}.4 Scenarios (Scenario v2)")
        scenario_details: list[str] = []
        if scenario_meta:
            scenario_details.append(f"status={scenario_meta.get('status')}")
            scenario_details.append(f"calls={scenario_meta.get('call_count')}")
            scenario_details.append(f"expected={'yes' if scenario_meta.get('expected') else 'no'}")
            if scenario_meta.get("triage_tier"):
                scenario_details.append(f"triage_tier={scenario_meta.get('triage_tier')}")
        if scenario_details:
            lines.append(f"_Scenario diagnostics: {', '.join(scenario_details)}_")
            lines.append("")
        _append_stage_block(lines, "scenario_v2", calls.get("scenario_v2"))
        lines.append("")

        lines.append(f"#### 2.{idx}.5 Ensemble SPD & EV (post-BayesMC)")
        lines.append("")

        q_dict = {
            "question_id": qid,
            "iso3": iso3_val,
            "hazard_code": hz_val,
            "metric": metric_val,
            "wording": wording,
        }
        bucket_labels = _get_bucket_labels_for_question(q_dict)
        centroids = _load_bucket_centroids_for_question(con, hz_val, metric_val, bucket_labels)

        lines.append("##### Buckets & Centroids")
        lines.append("")
        lines.append("| index | bucket_label | centroid |")
        lines.append("|-------|--------------|----------|")
        for i, label in enumerate(bucket_labels):
            centroid_val = centroids[i] if i < len(centroids) else 0.0
            lines.append(f"| {i + 1} | {label} | {centroid_val} |")
        lines.append("")

        ensemble_rows = con.execute(
            """
            SELECT month_index, bucket_index, probability, ev_value, status, human_explanation
            FROM forecasts_ensemble
            WHERE run_id = ? AND question_id = ?
            ORDER BY month_index, bucket_index
            """,
            [forecaster_run_id, qid],
        ).fetchall()

        if not ensemble_rows:
            lines.append("_No ensemble SPD rows found for this question/run._")
            lines.append("")
        else:
            statuses = {row[4] for row in ensemble_rows if row}
            if statuses == {"no_forecast"}:
                reason = (ensemble_rows[0][5] or "unknown") if ensemble_rows else "unknown"
                lines.append("_SPD status: `no_forecast`._")
                lines.append(f"_SPD failure reason: {reason}_")
                lines.append("")
                lines.append("_No ensemble SPD rows found for this question/run._")
                lines.append("")
            else:
                ensemble = _load_ensemble_spd_for_question(con, forecaster_run_id, qid, centroids)
                lines.append("##### Ensemble SPD and EV by Month")
                lines.append("")
                bucket_count = max(len(bucket_labels), 1)
                lines.append(
                    "| month_index | "
                    + " | ".join(f"p(bucket {i + 1})" for i in range(bucket_count))
                    + " | EV (units of centroid) |"
                )
                lines.append("|------------|" + "|".join(["--------------"] * (bucket_count + 1)) + "|")
                for month_idx in sorted(ensemble.keys()):
                    entry = ensemble[month_idx]
                    probs = entry.get("probs") or [0.0] * bucket_count
                    ev_val = entry.get("ev_value")
                    prob_cells = " | ".join(f"{p:.3f}" for p in probs[:bucket_count])
                    ev_cell = f"{ev_val:.1f}" if ev_val is not None else ""
                    lines.append(f"| {month_idx} | {prob_cells} | {ev_cell} |")
                lines.append("")

                lines.append("##### EV Calculation Notes")
                lines.append("")
                lines.append(
                    "For each month, the expected value (EV) is computed as:\n"
                    "- EV = sum_{i=1..5} p_i * centroid_i\n"
                    "where centroid_i is the representative value for bucket i (from `bucket_centroids` or defaults), "
                    "and p_i are the ensemble bucket probabilities after Bayes-MC aggregation."
                )
                lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    db_url = args.db
    db_path = _resolve_db_path(db_url)
    if db_path not in {":memory:"}:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    hs_run_id = args.hs_run_id
    forecaster_run_id = args.forecaster_run_id or args.run_id

    con = duckdb_io.get_db(db_url)
    try:
        db_stats = _file_stats(db_path)
        counts_after = _row_counts(con, KEY_TABLES)
        counts_before = _load_counts_before(args.counts_before_json)
        provenance_entry = _record_run_provenance(
            con,
            run_id=forecaster_run_id or hs_run_id,
            forecaster_run_id=forecaster_run_id,
            hs_run_id=hs_run_id,
            artifact_run_id=args.artifact_run_id,
            artifact_workflow=args.artifact_workflow,
            artifact_name=args.artifact_name,
            db_stats=db_stats,
            counts_before=counts_before,
            counts_after=counts_after,
        )
        provenance_lines = _provenance_markdown(provenance_entry, counts_before, counts_after, db_stats)

        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if args.legacy:
            # Legacy path: monolithic markdown bundle
            if forecaster_run_id:
                if not _forecast_run_exists(con, forecaster_run_id):
                    raise SystemExit(
                        f"Forecaster run_id {forecaster_run_id} not found in forecasts_ensemble; cannot build debug bundle."
                    )
                questions = _load_questions_for_run(con, forecaster_run_id)
                if not questions:
                    raise SystemExit(f"No active questions found for run_id={forecaster_run_id}.")
                markdown = build_debug_bundle_markdown(
                    con, db_url, forecaster_run_id, hs_run_id, questions, provenance_lines
                )
                out_run_id = forecaster_run_id
            else:
                markdown = build_triage_only_bundle_markdown(con, db_url, hs_run_id, provenance_lines)
                out_run_id = hs_run_id
            out_path = out_dir / f"pytia_debug_bundle__{out_run_id}.md"
            out_path.write_text(markdown, encoding="utf-8")
            print(f"Wrote Pythia debug bundle to {out_path}")
        else:
            # New path: split artifacts
            questions: list[dict[str, Any]] = []
            if forecaster_run_id:
                if not _forecast_run_exists(con, forecaster_run_id):
                    raise SystemExit(
                        f"Forecaster run_id {forecaster_run_id} not found in forecasts_ensemble; cannot build debug bundle."
                    )
                questions = _load_questions_for_run(con, forecaster_run_id)
                if not questions:
                    raise SystemExit(f"No active questions found for run_id={forecaster_run_id}.")

            data = _load_bundle_data(
                con,
                hs_run_id=hs_run_id,
                forecaster_run_id=forecaster_run_id,
                db_url=db_url,
                provenance_entry=provenance_entry,
                provenance_lines=provenance_lines,
                counts_before=counts_before,
                counts_after=counts_after,
                db_stats=db_stats,
                questions=questions,
            )

            emit_health_report_json(data, out_dir)
            emit_question_metrics_csv(data, con, out_dir)
            emit_evidence_packs_csv(data, out_dir)
            emit_llm_calls_detail_jsonl(data, con, out_dir)
            emit_spd_tables_csv(data, con, out_dir)

            # New emitters — each wrapped in try/except so failures are non-fatal
            for emitter_name, emitter_fn in [
                ("rc_triage_summary", lambda: emit_rc_triage_summary_csv(data, con, out_dir)),
                ("rc_pass_detail", lambda: emit_rc_pass_detail_csv(data, con, out_dir)),
                ("data_inject_inventory", lambda: emit_data_inject_inventory_csv(data, con, out_dir)),
                ("timing_breakdown", lambda: emit_timing_breakdown_csv(data, con, out_dir)),
                ("model_config_snapshot", lambda: emit_model_config_snapshot(data, con, out_dir)),
                ("grounding_detail", lambda: emit_grounding_detail_csv(data, con, out_dir)),
                ("coverage_detail", lambda: emit_coverage_detail_markdown(data, out_dir)),
            ]:
                try:
                    emitter_fn()
                except Exception as exc:
                    LOG.warning("Emitter %s failed: %s", emitter_name, exc)

            # The operational half. Health checks are evaluated once here and
            # handed to both the anomaly list and the summary — three
            # derivations of one verdict will eventually disagree, and the
            # reader cannot tell which is right.
            health_checks: list[dict[str, Any]] = []
            try:
                health_checks = _evaluate_pipeline_health(data)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("health evaluation failed: %s", exc)
            collected: dict[str, Any] = {}
            try:
                collected = emit_operational_files(
                    data,
                    con,
                    out_dir,
                    pipeline_id=args.pipeline_id or None,
                    repo_root=Path(__file__).resolve().parents[1],
                    health_checks=health_checks,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.warning("operational collectors failed: %s", exc)

            # The summary is written LAST because it leads with the
            # anomalies, and the anomalies are assembled from everything
            # above.
            try:
                emit_executive_summary(
                    data, out_dir, collected=collected, health_checks=health_checks
                )
            except Exception as exc:  # noqa: BLE001
                LOG.warning("executive summary failed: %s", exc)

            # Package all artifacts, then describe what landed.
            run_id = data.out_run_id
            try:
                zip_result = build_bundle_zips(out_dir, run_id)
                manifest = manifest_mod.build(
                    out_dir,
                    hs_run_id=data.hs_run_id,
                    forecaster_run_id=data.forecaster_run_id,
                    pipeline_id=args.pipeline_id or None,
                    extra={
                        "packaging": zip_result,
                        "collector_errors": _COLLECTOR_ERRORS,
                        "anomaly_counts": anomalies_mod.counts(collected.get("anomalies") or []),
                    },
                )
                manifest_mod.annotate_archives(manifest, zip_result)
                manifest_mod.write(out_dir, manifest)
                # The manifest must travel inside the bundle, so the zip is
                # rebuilt once it exists. The second build repeats the split
                # decision, which is why the archive annotation is applied
                # again against its result.
                zip_result = build_bundle_zips(out_dir, run_id)
                manifest_mod.annotate_archives(manifest, zip_result)
                manifest_mod.write(out_dir, manifest)
                zip_path = out_dir / zip_result["bundle_zip"]
                print(
                    f"Debug bundle zip: {zip_path} "
                    f"({zip_path.stat().st_size / 1024:.1f} KB)"
                )
                if zip_result.get("split"):
                    print(f"Workflow logs split into: {zip_result['workflow_logs_zip']}")
            except Exception as exc:
                LOG.warning("Failed to build debug bundle zip: %s", exc)
    finally:
        duckdb_io.close_db(con)


if __name__ == "__main__":
    main()
