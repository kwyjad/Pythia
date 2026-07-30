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

# Above this share of BATCHABLE SPEND paying full price, the discount is
# materially not being realised. Separate from the request-count threshold:
# the reference run was 32% of requests but 67% of forecaster spend.
DEFAULT_COST_WARN_PCT = 20.0

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
               COALESCE(n_fallback_sync, 0), COALESCE(error_text, '')
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
            "error_text": r[11],
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


def _collect_batch_tier_calls(
    con,
    hs_run_id: str = "",
    fc_run_id: str = "",
) -> Dict[str, Any]:
    """How much logged spend actually carried the batch service tier.

    ``collect_batch`` stamps ``usage.service_tier="batch"`` so the shared cost
    helper halves the price. Counting those rows is the end-to-end confirmation
    that the discount reached the ledger, not just the batch tables.

    MUST be scoped by run id. ``llm_calls`` accumulates across the travelling
    DB, so an unscoped query reports every pipeline that ever touched this file
    — which is how this section once reported a meaningless "$1.37 of $65.26"
    for a run that had spent $3.29.
    """
    if not _table_exists(con, "llm_calls"):
        return {"available": False}

    where, params = _run_scope(hs_run_id, fc_run_id)
    if not where:
        return {
            "available": False,
            "reason": "no run id supplied; refusing to report unscoped cumulative spend",
        }

    # json_extract_string keeps this robust to usage_json shape drift.
    rows = _rows(
        con,
        f"""
        SELECT
            SUM(CASE WHEN json_extract_string(usage_json, '$.service_tier') = 'batch'
                     THEN 1 ELSE 0 END),
            COUNT(*),
            SUM(CASE WHEN json_extract_string(usage_json, '$.service_tier') = 'batch'
                     THEN COALESCE(cost_usd, 0) ELSE 0 END),
            SUM(COALESCE(cost_usd, 0))
        FROM llm_calls
        WHERE {where}
        """,
        params,
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
        "scope": {"hs_run_id": hs_run_id or None, "fc_run_id": fc_run_id or None},
    }


def _run_scope(hs_run_id: str, fc_run_id: str):
    """SQL predicate + params restricting llm_calls to this pipeline's runs."""
    clauses: List[str] = []
    params: List[str] = []
    if hs_run_id:
        clauses.append("hs_run_id = ?")
        params.append(hs_run_id)
    if fc_run_id:
        clauses.append("run_id = ?")
        params.append(fc_run_id)
    if not clauses:
        return "", []
    return "(" + " OR ".join(clauses) + ")", params


def _has_column(con, table: str, column: str) -> bool:
    try:
        rows = con.execute(f"PRAGMA table_info('{table}')").fetchall()
        return any(str(r[1]) == column for r in rows)
    except Exception:
        return False


def _batchable_sql(con) -> str:
    """SQL predicate for llm_calls rows that a batch family could have carried.

    Only four families are ever batched: spd_v2, binary_v2 (keyed by `phase`)
    and hs_rc / hs_triage (the RC and triage model passes, identifiable only by
    `call_type` — every HS row shares phase='hs_triage', grounding included).

    Everything else has no batch family at all: grounding (rc_grounding /
    triage_grounding), adversarial_search / adversarial_synthesis, scenario_v2
    and sibyl. Counting those as "lost discount" — which a provider-level flag
    does, because Gemini serves both batched RC passes and unbatchable grounding
    — inflates the loss and makes the headline unactionable. On the 2026-07-30
    SOM run it reported google losing 41.6% of batchable spend when none of that
    spend was batchable in the first place.
    """
    phase_part = "phase IN ('spd_v2', 'binary_v2')"
    if _has_column(con, "llm_calls", "call_type"):
        return (
            f"({phase_part} OR call_type LIKE 'rc_pass_%' "
            "OR call_type LIKE 'triage_pass_%')"
        )
    # Pre-call_type DBs: phase alone cannot separate an RC pass from grounding,
    # so count only the forecaster families rather than over-claim.
    return f"({phase_part})"


def _collect_cost_by_provider_tier(
    con,
    batches: List[Dict[str, Any]],
    hs_run_id: str = "",
    fc_run_id: str = "",
) -> Dict[str, Any]:
    """Cost-weighted view of the discount — the number that actually matters.

    A request count understates a cost problem whenever the batched and
    unbatched members differ in price, and they always do: on the reference run
    the fallback rate was 32% of requests but **67% of forecaster spend**,
    because the OpenAI members are the expensive ones. Reconstructing that took
    a manual llm_calls analysis after the fact.

    "Lost discount" is deliberately restricted to providers for which this
    pipeline actually created a batch. Grounding (Brave), and any provider
    excluded via PYTHIA_BATCH_PROVIDERS, were never going to be discounted;
    counting them as losses would make the headline unactionable.
    """
    if not _table_exists(con, "llm_calls"):
        return {"available": False}

    where, params = _run_scope(hs_run_id, fc_run_id)
    if not where:
        return {
            "available": False,
            "reason": "no run id supplied; cost is only meaningful scoped to a run",
        }

    batchable = _batchable_sql(con)
    rows = _rows(
        con,
        f"""
        SELECT
            COALESCE(provider, '(none)'),
            CASE WHEN json_extract_string(usage_json, '$.service_tier') = 'batch'
                 THEN 'batch' ELSE 'full_price' END,
            CASE WHEN {batchable} THEN 1 ELSE 0 END,
            COUNT(*),
            SUM(COALESCE(cost_usd, 0))
        FROM llm_calls
        WHERE {where}
        GROUP BY 1, 2, 3
        ORDER BY 5 DESC
        """,
        params,
    )
    if not rows:
        return {"available": False}

    batched_providers = {
        str(b["provider"]).lower() for b in batches if b.get("provider")
    }

    by_provider: Dict[str, Dict[str, Any]] = {}
    run_total = 0.0
    for provider, tier, is_batchable, n_calls, cost in rows:
        cost = float(cost or 0.0)
        run_total += cost
        entry = by_provider.setdefault(
            str(provider),
            {
                "batch_cost_usd": 0.0,
                "full_price_cost_usd": 0.0,
                "non_batchable_cost_usd": 0.0,
                "n_batch_calls": 0,
                "n_full_price_calls": 0,
                "batch_attempted": str(provider).lower() in batched_providers,
            },
        )
        if tier == "batch":
            entry["batch_cost_usd"] += cost
            entry["n_batch_calls"] += int(n_calls or 0)
        elif int(is_batchable or 0):
            # Could have been batched and was not — this is the real loss.
            entry["full_price_cost_usd"] += cost
            entry["n_full_price_calls"] += int(n_calls or 0)
        else:
            # Grounding, adversarial, scenario, sibyl: no batch family exists,
            # so this is expected full-price spend, not a lost discount.
            entry["non_batchable_cost_usd"] += cost

    batched_spend = sum(e["batch_cost_usd"] for e in by_provider.values())
    # Batchable spend on providers we actually created a batch for, which
    # nevertheless paid full price.
    lost = sum(
        e["full_price_cost_usd"] for e in by_provider.values() if e["batch_attempted"]
    )
    batchable_spend = lost + sum(
        e["batch_cost_usd"] for e in by_provider.values() if e["batch_attempted"]
    )
    non_batchable = sum(e["non_batchable_cost_usd"] for e in by_provider.values())

    for entry in by_provider.values():
        for key in ("batch_cost_usd", "full_price_cost_usd", "non_batchable_cost_usd"):
            entry[key] = round(entry[key], 4)

    return {
        "available": True,
        "by_provider": by_provider,
        "run_total_cost_usd": round(run_total, 4),
        "batched_cost_usd": round(batched_spend, 4),
        "batchable_cost_usd": round(batchable_spend, 4),
        "non_batchable_cost_usd": round(non_batchable, 4),
        "lost_discount_cost_usd": round(lost, 4),
        "lost_discount_pct_of_batchable": (
            round(100.0 * lost / batchable_spend, 1) if batchable_spend else 0.0
        ),
        "lost_discount_pct_of_run": (
            round(100.0 * lost / run_total, 1) if run_total else 0.0
        ),
        "scope": {"hs_run_id": hs_run_id or None, "fc_run_id": fc_run_id or None},
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

    # A batch that yielded nothing is why a fallback rate is high. Naming it
    # here turns an unexplained percentage into an actionable cause.
    empty = [b for b in report["batches"] if b["n_requests"] and not b["n_succeeded"]]
    if empty:
        lines.append("**Batches that returned no results** (their requests paid full price):")
        lines.append("")
        for b in empty:
            lines.append(f"- `{b['batch_id']}` ({b['provider']}/{b['family']}): {b['error_text'] or 'no detail recorded'}")
        lines.append("")

    # The cost view leads, because a request share understates a cost problem
    # whenever the batched and unbatched members differ in price.
    cost = report.get("cost", {})
    if cost.get("available"):
        lines.append("#### Cost")
        lines.append("")
        if cost["batchable_cost_usd"]:
            lines.append(
                f"**${cost['lost_discount_cost_usd']} of ${cost['batchable_cost_usd']} "
                f"batchable spend paid full price "
                f"({cost['lost_discount_pct_of_batchable']}% of batchable, "
                f"{cost['lost_discount_pct_of_run']}% of the "
                f"${cost['run_total_cost_usd']} run).**"
            )
        else:
            lines.append(
                f"No batchable spend recorded for this run "
                f"(${cost['run_total_cost_usd']} total)."
            )
        lines.append("")
        lines.append(
            "| provider | batch attempted | batch $ | lost $ | not batchable $ "
            "| calls (batch/lost) |"
        )
        lines.append("|---|---|--:|--:|--:|---|")
        for provider, e in sorted(
            cost["by_provider"].items(),
            key=lambda kv: -(
                kv[1]["batch_cost_usd"]
                + kv[1]["full_price_cost_usd"]
                + kv[1]["non_batchable_cost_usd"]
            ),
        ):
            lines.append(
                f"| {provider} | {'yes' if e['batch_attempted'] else 'no'} "
                f"| {e['batch_cost_usd']} | {e['full_price_cost_usd']} "
                f"| {e['non_batchable_cost_usd']} "
                f"| {e['n_batch_calls']}/{e['n_full_price_calls']} |"
            )
        lines.append("")
        lines.append(
            "_`not batchable` is grounding, adversarial checks, scenarios and Sibyl — "
            "no batch family exists for them, so that spend is expected at full price "
            "and is NOT counted as lost discount. Full-price spend on a provider with "
            "`batch attempted = no` (excluded via `PYTHIA_BATCH_PROVIDERS`) is likewise "
            f"excluded. Not batchable this run: ${cost['non_batchable_cost_usd']}._"
        )
        lines.append("")
    elif cost.get("reason"):
        lines.append(f"_Cost section unavailable: {cost['reason']}._")
        lines.append("")

    tier = report.get("ledger", {})
    if tier.get("available"):
        lines.append(
            f"Ledger: {tier['n_batch_tier_calls']} of {tier['n_calls_total']} calls "
            f"logged for this run carry `service_tier=batch` "
            f"(${tier['cost_batch_tier_usd']} of ${tier['cost_total_usd']})."
        )
        lines.append("")
    elif tier.get("reason"):
        lines.append(f"_Ledger unavailable: {tier['reason']}._")
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
    parser.add_argument(
        "--cost-warn-pct",
        type=float,
        default=DEFAULT_COST_WARN_PCT,
        help="Warn when this share of batchable SPEND paid full price.",
    )
    parser.add_argument(
        "--hs-run-id",
        default=os.getenv("HS_RUN_ID", ""),
        help="Scopes the llm_calls cost sections. Without it they are omitted.",
    )
    parser.add_argument("--fc-run-id", default=os.getenv("FC_RUN_ID", ""))
    args = parser.parse_args(argv)

    hs_run_id = (args.hs_run_id or "").strip()
    fc_run_id = (args.fc_run_id or "").strip()

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
        ledger = _collect_batch_tier_calls(con, hs_run_id, fc_run_id)
        cost = _collect_cost_by_provider_tier(con, batches, hs_run_id, fc_run_id)
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
        "cost": cost,
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

    for b in report["batches"]:
        if b["n_requests"] and not b["n_succeeded"]:
            print(
                f"::warning title=Batch returned no results::{b['provider']} batch "
                f"{b['batch_id']} ({b['family']}) yielded 0 of {b['n_requests']} results — "
                f"those requests paid full price. {b['error_text'] or 'no detail recorded'}"
            )

    c = report.get("cost", {})
    if (
        c.get("available")
        and c.get("batchable_cost_usd")
        and c["lost_discount_pct_of_batchable"] > args.cost_warn_pct
    ):
        print(
            f"::warning title=Batch discount not realised (cost)::"
            f"${c['lost_discount_cost_usd']} of ${c['batchable_cost_usd']} batchable "
            f"spend paid full price in {pipeline_id} "
            f"({c['lost_discount_pct_of_batchable']}% of batchable, "
            f"{c['lost_discount_pct_of_run']}% of the ${c['run_total_cost_usd']} run) "
            f"at stage {args.stage}."
        )

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
