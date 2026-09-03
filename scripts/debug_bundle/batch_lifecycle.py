# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""One record per provider batch submitted anywhere in the cycle.

``llm_batches`` already held the outcome counters and ``llm_batch_requests``
the per-request terminal state, and until now the bundle read neither into
a file. So a cycle in which four OpenAI batches were refused wholesale and
210 requests fell back to full-price synchronous calls produced a bundle
byte-indistinguishable from a fully batched one.

The record is deliberately wide: batch ids on both sides (ours and the
provider's), the three file ids, the wall clock the batch spent in the
provider's queue, the terminal counts, and the verbatim provider error.
Beside it sits the money — realised spend at the tier it actually ran on,
against the counterfactual at the other tier, per phase and model, because
a request-count fallback rate understates the cost whenever the batched and
unbatched members differ in price, and they always do.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

# A provider batch halves input AND output token price. The counterfactual
# figures below are that factor applied to spend already recorded, never a
# re-derivation from the price table: the recorded cost is what was billed.
BATCH_DISCOUNT = 0.5

# Call families this pipeline actually submits as provider batches. Spend
# outside them (grounding, scenarios, adversarial synthesis, search APIs)
# is not a lost discount and must not be counted as one — the same rule
# scripts/ci/batch_economics.py works to.
BATCHABLE_PHASES = ("spd_v2", "binary_v2")
BATCHABLE_CALL_TYPE_PREFIXES = ("rc_pass_", "triage_pass_", "spd_v2", "binary_v2", "track2_spd")


def _rows(con, sql: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
    cur = con.execute(sql, params or [])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _columns(con, table: str) -> set[str]:
    try:
        # table_info yields (cid, name, ...) — the NAME is column 1, and reading
        # column 0 gives the ordinal, so every lookup silently missed.
        return {r[1] for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}
    except Exception:
        return set()


def _seconds_between(start: Any, end: Any) -> float | None:
    if start is None or end is None:
        return None
    try:
        return round((end - start).total_seconds(), 1)
    except Exception:
        return None


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return value.isoformat()
    except Exception:
        return str(value)


def _parse_error_payload(error_text: str | None) -> Any:
    """``llm_batches.error_text`` holds JSON when a batch ended empty."""

    if not error_text:
        return None
    text = error_text.strip()
    if not text.startswith("{") and not text.startswith("["):
        return text
    try:
        return json.loads(text)
    except Exception:
        return text


def collect(
    con,
    *,
    hs_run_id: str | None,
    forecaster_run_id: str | None,
    pipeline_id: str | None = None,
) -> dict[str, Any]:
    """Return the batch_lifecycle.json payload. Never raises."""

    ids = [i for i in (hs_run_id, forecaster_run_id) if i]
    out: dict[str, Any] = {
        "hs_run_id": hs_run_id,
        "forecaster_run_id": forecaster_run_id,
        "pipeline_id": pipeline_id,
        "batches": [],
        "cost_by_phase_model": [],
        "totals": {},
    }
    if not ids and not pipeline_id:
        out["note"] = "no run ids supplied; nothing to scope batches to"
        return out

    cols = _columns(con, "llm_batches")
    if not cols:
        out["note"] = "llm_batches table absent (DB predates the Batch-API pipeline)"
        return out

    def _sel(name: str, default: str = "NULL") -> str:
        return name if name in cols else f"{default} AS {name}"

    where_parts: list[str] = []
    params: list[Any] = []
    if ids:
        ph = ",".join("?" for _ in ids)
        where_parts.append(f"(hs_run_id IN ({ph}) OR run_id IN ({ph}))")
        params.extend(ids + ids)
    if pipeline_id and "pipeline_id" in cols:
        where_parts.append("pipeline_id = ?")
        params.append(pipeline_id)
    where = " OR ".join(where_parts) if where_parts else "1=1"

    batches = _rows(
        con,
        f"""
        SELECT batch_id, {_sel('provider')}, {_sel('provider_batch_id')},
               {_sel('family')}, {_sel('stage')}, {_sel('model_id')},
               {_sel('pipeline_id')}, {_sel('run_id')}, {_sel('hs_run_id')},
               {_sel('status')}, {_sel('n_requests', '0')},
               {_sel('n_succeeded', '0')}, {_sel('n_errored', '0')},
               {_sel('n_expired', '0')}, {_sel('n_fallback_sync', '0')},
               {_sel('input_file_id')}, {_sel('output_file_id')},
               {_sel('error_file_id')}, {_sel('results_url')},
               {_sel('submitted_at')}, {_sel('first_polled_at')},
               {_sel('ended_at')}, {_sel('collected_at')}, {_sel('error_text')}
        FROM llm_batches
        WHERE {where}
        ORDER BY submitted_at
        """,
        params,
    )
    if not batches:
        out["note"] = "no batches recorded for this run"
        return out

    batch_ids = [str(b["batch_id"]) for b in batches]
    ph = ",".join("?" for _ in batch_ids)
    by_batch_status: dict[str, dict[str, int]] = {}
    try:
        for row in _rows(
            con,
            f"""
            SELECT batch_id, COALESCE(status,'') AS status, COUNT(*) AS n
            FROM llm_batch_requests WHERE batch_id IN ({ph})
            GROUP BY batch_id, status
            """,
            batch_ids,
        ):
            by_batch_status.setdefault(str(row["batch_id"]), {})[str(row["status"])] = int(row["n"])
    except Exception:
        by_batch_status = {}

    records: list[dict[str, Any]] = []
    for b in batches:
        bid = str(b["batch_id"])
        statuses = by_batch_status.get(bid, {})
        submitted = b.get("submitted_at")
        # "Ended" is when the provider stopped working on it; "collected" is
        # when we read it. Queue seconds is the first, because the second
        # includes however long the poller took to come back.
        record = {
            "batch_id": bid,
            "provider_batch_id": b.get("provider_batch_id"),
            "provider": b.get("provider"),
            "phase": b.get("family"),
            "stage": b.get("stage"),
            "model_id": b.get("model_id"),
            "pipeline_id": b.get("pipeline_id"),
            "run_id": b.get("run_id"),
            "hs_run_id": b.get("hs_run_id"),
            "submitted_at": _iso(submitted),
            "first_polled_at": _iso(b.get("first_polled_at")),
            "ended_at": _iso(b.get("ended_at")),
            "completed_at": _iso(b.get("collected_at")),
            "queue_wall_seconds": _seconds_between(submitted, b.get("ended_at")),
            "collect_wall_seconds": _seconds_between(submitted, b.get("collected_at")),
            "seconds_to_first_poll": _seconds_between(submitted, b.get("first_polled_at")),
            "provider_state": b.get("status"),
            "n_requests": int(b.get("n_requests") or 0),
            "n_succeeded": int(b.get("n_succeeded") or 0),
            "n_errored": int(b.get("n_errored") or 0),
            "n_expired": int(b.get("n_expired") or 0),
            "n_fell_back_to_sync": int(
                b.get("n_fallback_sync") or statuses.get("fallback_sync", 0) or 0
            ),
            "request_status_counts": statuses,
            "input_file_id": b.get("input_file_id"),
            "output_file_id": b.get("output_file_id"),
            "error_file_id": b.get("error_file_id"),
            "results_url": b.get("results_url"),
            "provider_error_payload": _parse_error_payload(b.get("error_text")),
        }
        record["yielded_nothing"] = bool(
            record["n_requests"] and not record["n_succeeded"]
        )
        records.append(record)

    out["batches"] = records
    out["cost_by_phase_model"] = _cost_by_phase_model(con, ids)
    out["totals"] = _totals(records, out["cost_by_phase_model"])
    return out


def _cost_by_phase_model(con, run_ids: Iterable[str]) -> list[dict[str, Any]]:
    """Realised spend per (phase, model) with the other tier's counterfactual.

    ``service_tier='batch'`` is stamped onto the usage of every collected
    batch item, so it is the ground truth for which tier a call was billed
    at — not which family it belongs to, and not whether a batch existed.
    """

    ids = [i for i in run_ids if i]
    if not ids:
        return []
    llm_cols = _columns(con, "llm_calls")
    if not {"cost_usd", "usage_json"} <= llm_cols:
        return []
    ph = ",".join("?" for _ in ids)
    call_type_sel = "COALESCE(call_type,'')" if "call_type" in llm_cols else "''"
    try:
        rows = _rows(
            con,
            f"""
            SELECT
                COALESCE(phase,'') AS phase,
                {call_type_sel} AS call_type,
                COALESCE(provider,'') AS provider,
                COALESCE(model_id,'') AS model_id,
                SUM(CASE WHEN json_extract_string(usage_json,'$.service_tier') = 'batch'
                         THEN 1 ELSE 0 END) AS n_batch_tier,
                SUM(CASE WHEN json_extract_string(usage_json,'$.service_tier') = 'batch'
                         THEN 0 ELSE 1 END) AS n_sync_tier,
                SUM(CASE WHEN json_extract_string(usage_json,'$.service_tier') = 'batch'
                         THEN COALESCE(cost_usd,0) ELSE 0 END) AS cost_batch_tier,
                SUM(CASE WHEN json_extract_string(usage_json,'$.service_tier') = 'batch'
                         THEN 0 ELSE COALESCE(cost_usd,0) END) AS cost_sync_tier
            FROM llm_calls
            WHERE run_id IN ({ph}) OR hs_run_id IN ({ph})
            GROUP BY 1,2,3,4
            """,
            ids + ids,
        )
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for r in rows:
        phase = str(r["phase"])
        call_type = str(r["call_type"])
        batchable = phase in BATCHABLE_PHASES or any(
            call_type.startswith(p) for p in BATCHABLE_CALL_TYPE_PREFIXES
        )
        cost_batch = float(r["cost_batch_tier"] or 0.0)
        cost_sync = float(r["cost_sync_tier"] or 0.0)
        realised = cost_batch + cost_sync
        # What this spend would have cost had every call run at the OTHER
        # tier. Both directions are printed because the useful question
        # changes with the failure: after a batch collapse you want the
        # discount that was lost, and when weighing a new family you want
        # the discount that is on the table.
        all_batch = cost_batch + cost_sync * BATCH_DISCOUNT
        all_sync = cost_batch / BATCH_DISCOUNT + cost_sync
        out.append(
            {
                "phase": phase,
                "call_type": call_type,
                "provider": r["provider"],
                "model_id": r["model_id"],
                "batchable_family": batchable,
                "n_calls_batch_tier": int(r["n_batch_tier"] or 0),
                "n_calls_sync_tier": int(r["n_sync_tier"] or 0),
                "realised_cost_usd": round(realised, 6),
                "cost_at_batch_tier_usd": round(cost_batch, 6),
                "cost_at_sync_tier_usd": round(cost_sync, 6),
                "counterfactual_all_batch_usd": round(all_batch, 6),
                "counterfactual_all_sync_usd": round(all_sync, 6),
                # Only a batchable family can lose a discount; a scenario
                # call running synchronously has lost nothing.
                "lost_discount_usd": round(cost_sync * BATCH_DISCOUNT, 6) if batchable else 0.0,
            }
        )
    out.sort(key=lambda r: (-r["realised_cost_usd"], r["phase"], r["model_id"]))
    return out


def _totals(records: list[dict[str, Any]], cost_rows: list[dict[str, Any]]) -> dict[str, Any]:
    n_requests = sum(r["n_requests"] for r in records)
    n_fallback = sum(r["n_fell_back_to_sync"] for r in records)
    return {
        "n_batches": len(records),
        "n_batches_yielded_nothing": sum(1 for r in records if r["yielded_nothing"]),
        "n_requests": n_requests,
        "n_succeeded": sum(r["n_succeeded"] for r in records),
        "n_errored": sum(r["n_errored"] for r in records),
        "n_expired": sum(r["n_expired"] for r in records),
        "n_fell_back_to_sync": n_fallback,
        "fallback_pct_of_requests": round(100.0 * n_fallback / n_requests, 1) if n_requests else 0.0,
        "realised_cost_usd": round(sum(r["realised_cost_usd"] for r in cost_rows), 4),
        "lost_discount_usd": round(sum(r["lost_discount_usd"] for r in cost_rows), 4),
        "median_queue_wall_seconds": _median(
            [r["queue_wall_seconds"] for r in records if r["queue_wall_seconds"] is not None]
        ),
        "max_queue_wall_seconds": max(
            [r["queue_wall_seconds"] for r in records if r["queue_wall_seconds"] is not None],
            default=None,
        ),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return round((ordered[mid - 1] + ordered[mid]) / 2.0, 1)
