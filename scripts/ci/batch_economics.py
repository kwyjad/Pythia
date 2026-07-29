# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

"""Report whether a staged pipeline actually realised the provider batch discount.

``llm_batches`` carries per-batch outcome counters (n_requests / n_succeeded /
n_errored / n_expired / n_fallback_sync) and ``llm_batch_requests.status``
carries the per-request terminal state, but before this script nothing read
either into an artifact or a step summary. A pipeline in which every request
silently fell through to the synchronous full-price path therefore produced
artifacts byte-indistinguishable from a fully-batched one — the entire economic
premise of the staged pipeline went unverified.

Emits ``diagnostics/batch_economics__<stage>.json`` and, when running under
Actions, a Markdown table into ``$GITHUB_STEP_SUMMARY``. Raises a
``::warning::`` when the fallback-sync rate crosses a threshold, mirroring the
prompt-cache zero-reads warning in ``post_run_diagnostics.py``.

DIAGNOSTIC ONLY: this script must never fail a pipeline stage. Every DB access
is guarded and ``main`` always returns 0 — a stage that cannot report its
economics still has real work to hand off to the next stage.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import duckdb

DEFAULT_DB_URL = "duckdb:///data/resolver.duckdb"

# Above this share of terminal requests taking the synchronous fallback, the
# batch discount is materially not being realised and the run is worth a look.
DEFAULT_FALLBACK_WARN_PCT = 20.0

# Terminal request states. 'succeeded' is the only one that was actually billed
# at the batch rate; fallback_sync was re-run synchronously at full price.
_TERMINAL = ("succeeded", "fallback_sync", "errored", "expired")


def _table_exists(con, table: str) -> bool:
    try:
        con.execute(f"SELECT 1 FROM {table} LIMIT 1")
        return True
    except Exception:
        return False


def _rows(con, sql: str, params: Optional[list] = None) -> List[tuple]:
    try:
        return list(con.execute(sql, params or []).fetchall())
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] batch_economics query failed: {type(exc).__name__}: {exc}")
        return []


def _collect_batches(con, pipeline_id: str) -> List[Dict[str, Any]]:
    if not _table_exists(con, "llm_batches"):
        print("[warn] llm_batches table not present; skipping batch rollup")
        return []
    rows = _rows(
        con,
        """
        SELECT batch_id, provider, family, stage, model_id, status,
               COALESCE(n_requests, 0), COALESCE(n_succeeded, 0),
               COALESCE(n_errored, 0), COALESCE(n_expired, 0),
               COALESCE(n_fallback_sync, 0)
        FROM llm_batches
        WHERE pipeline_id = ?
        ORDER BY submitted_at
        """,
        [pipeline_id],
    )
    return [
        {
            "batch_id": r[0],
            "provider": r[1],
            "family": r[2],
            "stage": r[3],
            "model_id": r[4],
            "status": r[5],
            "n_requests": int(r[6]),
            "n_succeeded": int(r[7]),
            "n_errored": int(r[8]),
            "n_expired": int(r[9]),
            "n_fallback_sync": int(r[10]),
        }
        for r in rows
    ]


def _collect_request_states(con, pipeline_id: str) -> Dict[str, Dict[str, int]]:
    """Per-family request status counts — the authoritative fallback signal.

    Batch-level counters can be stale when a stage dies mid-collect; the
    per-request rows are updated transactionally, so they are what the
    fallback-rate warning is computed from.
    """
    if not _table_exists(con, "llm_batch_requests"):
        print("[warn] llm_batch_requests table not present; skipping request rollup")
        return {}
    rows = _rows(
        con,
        """
        SELECT COALESCE(family, '(none)'), COALESCE(status, '(null)'), COUNT(*)
        FROM llm_batch_requests
        WHERE pipeline_id = ?
        GROUP BY 1, 2
        """,
        [pipeline_id],
    )
    out: Dict[str, Dict[str, int]] = {}
    for family, status, count in rows:
        out.setdefault(str(family), {})[str(status)] = int(count)
    return out


def _collect_batch_tier_calls(con, pipeline_id: str) -> Dict[str, Any]:
    """How much logged spend actually carried the batch service tier.

    ``collect_batch`` stamps ``usage.service_tier="batch"`` so the shared cost
    helper halves the price. Counting those rows is the end-to-end confirmation
    that the discount reached the ledger, not just the batch tables.
    """
    if not _table_exists(con, "llm_calls"):
        return {"available": False}
    # json_extract_string keeps this robust to usage_json shape drift.
    rows = _rows(
        con,
        """
        SELECT
            SUM(CASE WHEN json_extract_string(usage_json, '$.service_tier') = 'batch'
                     THEN 1 ELSE 0 END),
            COUNT(*),
            SUM(CASE WHEN json_extract_string(usage_json, '$.service_tier') = 'batch'
                     THEN COALESCE(cost_usd, 0) ELSE 0 END),
            SUM(COALESCE(cost_usd, 0))
        FROM llm_calls
        """,
    )
    if not rows or rows[0][1] is None:
        return {"available": False}
    n_batch, n_total, cost_batch, cost_total = rows[0]
    return {
        "available": True,
        "n_batch_tier_calls": int(n_batch or 0),
        "n_calls_total": int(n_total or 0),
        "cost_batch_tier_usd": round(float(cost_batch or 0.0), 4),
        "cost_total_usd": round(float(cost_total or 0.0), 4),
        "note": "llm_calls is cumulative across the travelling DB, not stage-scoped",
    }


def _summarize(batches: List[Dict[str, Any]], requests: Dict[str, Dict[str, int]]) -> Dict[str, Any]:
    totals = {k: 0 for k in ("n_requests", "n_succeeded", "n_errored", "n_expired", "n_fallback_sync")}
    for b in batches:
        for k in totals:
            totals[k] += b[k]

    req_totals: Dict[str, int] = {}
    for family_counts in requests.values():
        for status, count in family_counts.items():
            req_totals[status] = req_totals.get(status, 0) + count

    terminal = sum(req_totals.get(s, 0) for s in _TERMINAL)
    fallback = req_totals.get("fallback_sync", 0)
    succeeded = req_totals.get("succeeded", 0)
    fallback_pct = (100.0 * fallback / terminal) if terminal else 0.0
    batched_pct = (100.0 * succeeded / terminal) if terminal else 0.0

    return {
        "batch_totals": totals,
        "request_status_totals": req_totals,
        "n_terminal_requests": terminal,
        "fallback_sync_pct": round(fallback_pct, 1),
        "batched_pct": round(batched_pct, 1),
    }


def _markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    lines: List[str] = []
    lines.append(f"### Batch economics — stage `{report['stage']}`")
    lines.append("")
    lines.append(f"pipeline: `{report['pipeline_id']}`")
    lines.append("")

    if not report["batches"]:
        lines.append(
            "**No provider batches recorded for this pipeline.** Every request "
            "ran (or will run) synchronously at full price — expected when "
            "`PYTHIA_BATCH_PROVIDERS` excludes the stage's provider, and a "
            "signal worth checking otherwise."
        )
        lines.append("")

    if s["n_terminal_requests"]:
        lines.append(
            f"**{s['batched_pct']}% of {s['n_terminal_requests']} terminal requests "
            f"were served from a batch** (discounted); "
            f"{s['fallback_sync_pct']}% took the synchronous fallback (full price)."
        )
        lines.append("")

    if report["batches"]:
        lines.append("| batch | provider | family | status | req | ok | err | exp | fallback |")
        lines.append("|---|---|---|---|--:|--:|--:|--:|--:|")
        for b in report["batches"]:
            lines.append(
                f"| `{b['batch_id']}` | {b['provider']} | {b['family']} | {b['status']} "
                f"| {b['n_requests']} | {b['n_succeeded']} | {b['n_errored']} "
                f"| {b['n_expired']} | {b['n_fallback_sync']} |"
            )
        lines.append("")

    if report["request_status_by_family"]:
        lines.append("| family | " + " | ".join(_TERMINAL) + " | other |")
        lines.append("|---|" + "--:|" * (len(_TERMINAL) + 1))
        for family, counts in sorted(report["request_status_by_family"].items()):
            cells = [str(counts.get(st, 0)) for st in _TERMINAL]
            other = sum(v for k, v in counts.items() if k not in _TERMINAL)
            lines.append(f"| {family} | " + " | ".join(cells) + f" | {other} |")
        lines.append("")

    tier = report.get("ledger", {})
    if tier.get("available"):
        lines.append(
            f"Ledger: {tier['n_batch_tier_calls']} of {tier['n_calls_total']} logged "
            f"calls carry `service_tier=batch` "
            f"(${tier['cost_batch_tier_usd']} of ${tier['cost_total_usd']} total). "
            f"_{tier['note']}._"
        )
        lines.append("")

    return "\n".join(lines)


def _emit_step_summary(text: str) -> None:
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(text + "\n")
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] could not write step summary: {type(exc).__name__}: {exc}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB_URL)
    parser.add_argument("--pipeline-id", required=True)
    parser.add_argument("--stage", default=os.getenv("PIPELINE_STAGE", "unknown"))
    parser.add_argument("--out", default="")
    parser.add_argument("--fallback-warn-pct", type=float, default=DEFAULT_FALLBACK_WARN_PCT)
    args = parser.parse_args(argv)

    pipeline_id = (args.pipeline_id or "").strip()
    if not pipeline_id:
        print("[warn] no pipeline id supplied; nothing to report")
        return 0

    db = args.db
    db_path = db[len("duckdb:///"):] if db.startswith("duckdb:///") else db
    if not os.path.exists(db_path):
        print(f"[warn] DuckDB database not found at {db_path}")
        return 0

    try:
        con = duckdb.connect(db_path, read_only=True)
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] could not open DuckDB at {db_path}: {type(exc).__name__}: {exc}")
        return 0

    try:
        batches = _collect_batches(con, pipeline_id)
        requests = _collect_request_states(con, pipeline_id)
        ledger = _collect_batch_tier_calls(con, pipeline_id)
    finally:
        try:
            con.close()
        except Exception:
            pass

    report = {
        "pipeline_id": pipeline_id,
        "stage": args.stage,
        "batches": batches,
        "request_status_by_family": requests,
        "ledger": ledger,
        "summary": _summarize(batches, requests),
    }

    out = args.out or f"diagnostics/batch_economics__{args.stage}.json"
    try:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"wrote {out}")
    except Exception as exc:  # pragma: no cover - diagnostics only
        print(f"[warn] could not write {out}: {type(exc).__name__}: {exc}")

    md = _markdown(report)
    print(md)
    _emit_step_summary(md)

    s = report["summary"]
    if s["n_terminal_requests"] and s["fallback_sync_pct"] > args.fallback_warn_pct:
        print(
            f"::warning title=Batch discount not realised::{s['fallback_sync_pct']}% of "
            f"{s['n_terminal_requests']} terminal requests in {pipeline_id} took the "
            f"synchronous fallback (full price) at stage {args.stage}. Check provider "
            f"batch submission and PYTHIA_BATCH_PROVIDERS."
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
