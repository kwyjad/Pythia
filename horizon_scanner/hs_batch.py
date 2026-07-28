# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Stage state + Batch-API seam for the staged Horizon Scanner.

The staged HS decomposes the per-country RC → triage flow into three
waves so the two LLM families run through provider Batch APIs (50% off):

    hs_submit       RC grounding (sync Brave, persisted) + RC prompts built
                    and enqueued as the ``hs_rc`` family; triage not run.
    hs_rc_collect   RC results replayed from the collected ``hs_rc`` batch
                    (sync fallback per miss); merged RC dicts persisted to
                    ``hs_stage_state``; triage grounding (sync) + triage
                    prompts enqueued as the ``hs_triage`` family.
    hs_finalize     RC results loaded from ``hs_stage_state`` (batch replay
                    as fallback); triage replayed from the collected
                    ``hs_triage`` batch; the country flow then continues
                    into the unmodified ``_write_hs_triage`` + adversarial
                    checks + coverage/stdout contract.

Default stage "full" is byte-for-byte today's synchronous behavior — the
seams below are inert.

Design mirrors the forecaster's ``--phase submit/collect`` seam
(``forecaster/cli.py::_try_batch_phase``): the interception point is the
single model-call site inside each per-hazard pass loop, so parsing,
JSON-repair, merging, defaults, skip logic and DB writers are the
unmodified existing code paths.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date as date_type
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Country workers (HS_MAX_WORKERS threads) share one DuckDB connection.
# duckdb-python stores the pending result set ON the connection, so even a
# lone execute(...).fetchone() is not atomic against a concurrent execute —
# ALL DB access through batch_con() must hold this lock, reads included
# (an unlocked read can return another country's row).
_DB_LOCK = threading.Lock()

STAGES = ("full", "hs_submit", "hs_rc_collect", "hs_finalize")

# family_mode() per stage: what each LLM family does in each stage.
_FAMILY_MODES: Dict[str, Dict[str, str]] = {
    "full": {"hs_rc": "full", "hs_triage": "full"},
    "hs_submit": {"hs_rc": "submit", "hs_triage": "skip"},
    "hs_rc_collect": {"hs_rc": "collect", "hs_triage": "submit"},
    "hs_finalize": {"hs_rc": "collect", "hs_triage": "collect"},
}

_STAGE = "full"
_PIPELINE_ID: Optional[str] = None
_CON = None  # lazy DuckDB connection for batch/stage state


def set_stage(stage: str, pipeline_id: Optional[str] = None) -> None:
    global _STAGE, _PIPELINE_ID
    if stage not in STAGES:
        raise ValueError(f"unknown HS stage: {stage!r} (expected one of {STAGES})")
    _STAGE = stage
    _PIPELINE_ID = (pipeline_id or "").strip() or None


def stage() -> str:
    return _STAGE


def pipeline_id() -> Optional[str]:
    return _PIPELINE_ID


def staged() -> bool:
    return _STAGE != "full"


def family_mode(family: str) -> str:
    """One of full | submit | collect | skip for this family in this stage."""

    return _FAMILY_MODES[_STAGE].get(family, "full")


def triage_enabled() -> bool:
    """False only in hs_submit (RC results are pending — no triage inputs)."""

    return family_mode("hs_triage") != "skip"


def finalize_enabled() -> bool:
    """True when _write_hs_triage / adversarial checks / coverage may run."""

    return _STAGE in ("full", "hs_finalize")


def reset(close: bool = True) -> None:
    """Test/support helper: back to full mode, drop the cached connection."""

    global _STAGE, _PIPELINE_ID, _CON
    _STAGE = "full"
    _PIPELINE_ID = None
    if _CON is not None and close:
        try:
            _CON.close()
        except Exception:  # noqa: BLE001
            pass
    _CON = None


def batch_con():
    """Lazy shared read-write connection for llm_batch + stage state."""

    global _CON
    if _CON is None:
        from pythia.db.schema import connect, ensure_schema

        _CON = connect(read_only=False)
        ensure_schema(_CON)
    return _CON


# ---------------------------------------------------------------------------
# Enqueue / replay (the model-call seam)
# ---------------------------------------------------------------------------


def enqueue_hazard_call(
    family: str,
    prompt: str,
    *,
    spec,
    iso3: str,
    hazard_code: str,
    pass_idx: int,
    temperature: float,
) -> bool:
    """Enqueue one per-hazard model call as a batch request (submit mode).

    The body comes from the same ``build_body_for_spec`` the sync path
    resolves through, so the batch payload is byte-identical to what
    ``call_chat_ms`` would send. Providers outside PYTHIA_BATCH_PROVIDERS
    are not enqueued (returns False) — their calls run sync in the collect
    stage via the ordinary replay-miss fallback.
    """

    from forecaster.providers import build_body_for_spec
    from pythia import llm_batch

    if not llm_batch.provider_batchable(spec.provider):
        return False
    body = build_body_for_spec(spec, prompt, temperature)
    with _DB_LOCK:
        llm_batch.enqueue_request(
            batch_con(),
            family=family,
            provider=spec.provider,
            model_id=spec.model_id,
            request_body=body,
            prompt_text=prompt,
            iso3=iso3,
            hazard_code=hazard_code,
            pass_idx=pass_idx,
            pipeline_id=_PIPELINE_ID,
        )
    return True


def replay_hazard_call(
    family: str,
    *,
    iso3: str,
    hazard_code: str,
    pass_idx: int,
):
    """Replay a collected batch result for one per-hazard pass.

    Returns (text, usage, error, ModelSpec) on a hit, or None — the caller
    then runs the existing synchronous model call (the per-item fallback)
    after ``mark_fallback``.
    """

    from pythia import llm_batch

    with _DB_LOCK:
        hit = llm_batch.get_result(
            batch_con(),
            family,
            iso3=iso3,
            hazard_code=hazard_code,
            pass_idx=pass_idx,
            pipeline_id=_PIPELINE_ID,
        )
    if hit is not None and hit.get("status", "succeeded") != "succeeded":
        # Errored batch item: fall back to sync. The burned tokens are
        # visible here but not rich-logged (HS logging is label-keyed per
        # pass); surface them in the log at least.
        usage = hit.get("usage") or {}
        logger.warning(
            "hs_batch: %s %s/%s p%d errored in batch (%s tokens in / %s out) — sync fallback",
            family, iso3, hazard_code, pass_idx,
            usage.get("prompt_tokens"), usage.get("completion_tokens"),
        )
        hit = None
    if hit is None:
        return None

    from forecaster.providers import ModelSpec

    usage = dict(hit.get("usage") or {})
    usage.setdefault("elapsed_ms", 0)
    model_id = usage.get("model_id") or _request_model_id(hit["custom_id"]) or "unknown"
    provider = usage.get("provider") or _request_provider(hit["custom_id"]) or "google"
    spec = ModelSpec(
        name=str(model_id),
        provider=str(provider),
        model_id=str(model_id),
        active=True,
        purpose="hs_regime_change" if family == "hs_rc" else "hs_triage",
    )
    usage.setdefault("model_selected", f"{spec.provider}:{spec.model_id}")
    usage.setdefault("fallback_used", False)
    return hit.get("text") or "", usage, (hit.get("error") or ""), spec


def mark_fallback(family: str, *, iso3: str, hazard_code: str, pass_idx: int) -> None:
    from pythia import llm_batch

    with _DB_LOCK:
        llm_batch.mark_fallback_sync(
            batch_con(),
            family,
            iso3=iso3,
            hazard_code=hazard_code,
            pass_idx=pass_idx,
            pipeline_id=_PIPELINE_ID,
        )


def _request_row(custom_id: str):
    with _DB_LOCK:
        return batch_con().execute(
            "SELECT provider, model_id FROM llm_batch_requests WHERE custom_id = ?",
            [custom_id],
        ).fetchone()


def _request_provider(custom_id: str) -> Optional[str]:
    row = _request_row(custom_id)
    return row[0] if row else None


def _request_model_id(custom_id: str) -> Optional[str]:
    row = _request_row(custom_id)
    return row[1] if row else None


def llm_call_already_logged(run_id: str, iso3: str, hazard_label: str) -> bool:
    """Idempotency guard for replayed results.

    A replayed batch result is rich-logged to llm_calls exactly once even
    when a collect stage is re-run (poller double-fire) or hs_finalize
    re-replays RC after an hs_rc_collect crash — otherwise cost would
    double-count. Sync-fallback calls always log (a real call happened).
    """

    try:
        # log_hs_llm_call uppercases both iso3 and the hazard label on write.
        with _DB_LOCK:
            row = batch_con().execute(
                """
                SELECT 1 FROM llm_calls
                WHERE hs_run_id = ? AND UPPER(iso3) = UPPER(?) AND UPPER(hazard_code) = UPPER(?)
                LIMIT 1
                """,
                [run_id, iso3, hazard_label],
            ).fetchone()
        return row is not None
    except Exception:  # noqa: BLE001
        logger.warning(
            "llm_call_already_logged guard failed for %s/%s — a duplicate "
            "llm_calls row may be written", iso3, hazard_label, exc_info=True,
        )
        return False


# ---------------------------------------------------------------------------
# Grounding pack reuse (collect stages must not re-fetch or re-persist)
# ---------------------------------------------------------------------------


def load_grounding_pack(
    run_id: str,
    iso3: str,
    hazard_code: str,
    prefer_prefix: str,
) -> Optional[dict[str, Any]]:
    """Load a persisted grounding pack from hs_hazard_tail_packs.

    ``prefer_prefix`` is "rc_grounding" or "triage_grounding". Since the
    July 2026 audit the writer's idempotency delete is scoped by pack kind,
    so BOTH packs coexist per (run, iso3, hazard) and the preferred one is
    normally found. A non-preferred pack may still be all that exists
    (pre-fix rows, or a hazard whose triage grounding was skipped); it is
    returned as a fallback.
    """

    try:
        with _DB_LOCK:
            rows = batch_con().execute(
                """
                SELECT query, report_markdown, sources_json, grounded,
                       grounding_debug_json, structural_context, recent_signals_json
                FROM hs_hazard_tail_packs
                WHERE hs_run_id = ? AND UPPER(iso3) = ? AND UPPER(hazard_code) = ?
                ORDER BY created_at DESC
                """,
                [run_id, (iso3 or "").upper(), (hazard_code or "").upper()],
            ).fetchall()
    except Exception as exc:  # noqa: BLE001
        logger.warning("load_grounding_pack failed for %s %s: %s", iso3, hazard_code, exc)
        return None
    if not rows:
        return None

    chosen = rows[0]
    for row in rows:
        if str(row[0] or "").startswith(f"{prefer_prefix}:"):
            chosen = row
            break

    query, markdown, sources_json, grounded, debug_json, structural, signals_json = chosen
    query = str(query or "")
    for prefix in ("rc_grounding: ", "triage_grounding: "):
        if query.startswith(prefix):
            query = query[len(prefix):]
            break

    def _loads(raw, default):
        try:
            return json.loads(raw) if raw else default
        except json.JSONDecodeError:
            return default

    return {
        "query": query,
        "markdown": markdown or "",
        "sources": _loads(sources_json, []),
        "grounded": bool(grounded),
        "debug": _loads(debug_json, {}),
        "structural_context": structural or "",
        "recent_signals": _loads(signals_json, []),
    }


# ---------------------------------------------------------------------------
# Cross-stage carry state (merged RC results)
# ---------------------------------------------------------------------------


def save_stage_state(run_id: str, iso3: str, stage_key: str, payload: dict[str, Any]) -> None:
    from pythia.test_mode import is_test_mode

    with _DB_LOCK:
        con = batch_con()
        con.execute(
            "DELETE FROM hs_stage_state WHERE run_id = ? AND iso3 = ? AND stage_key = ?",
            [run_id, (iso3 or "").upper(), stage_key],
        )
        con.execute(
            """
            INSERT INTO hs_stage_state (run_id, iso3, stage_key, payload_json, created_at, is_test)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                (iso3 or "").upper(),
                stage_key,
                json.dumps(payload, ensure_ascii=False, default=str),
                datetime.now(timezone.utc).replace(tzinfo=None),
                is_test_mode(),
            ],
        )


def load_stage_state(run_id: str, iso3: str, stage_key: str) -> Optional[dict[str, Any]]:
    try:
        with _DB_LOCK:
            row = batch_con().execute(
                """
                SELECT payload_json FROM hs_stage_state
                WHERE run_id = ? AND iso3 = ? AND stage_key = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                [run_id, (iso3 or "").upper(), stage_key],
            ).fetchone()
    except Exception as exc:  # noqa: BLE001
        logger.warning("load_stage_state failed for %s %s: %s", iso3, stage_key, exc)
        return None
    if not row or not row[0]:
        return None
    try:
        payload = json.loads(row[0])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


# Run-scoped stage-state rows (not per-country) use this sentinel iso3.
_RUN_SCOPE_ISO3 = "__RUN__"
_BREAKER_STATE_KEY = "brave_breaker"


def save_breaker_state(run_id: str) -> None:
    """Persist the Brave circuit-breaker stats for cross-stage carry.

    The breaker is a per-process object; without this, a trip during
    hs_submit/hs_rc_collect is invisible to hs_finalize (which does no
    grounding), so _write_hs_triage's blocked-no-grounding safety gate
    could never fire in staged mode.
    """

    try:
        from pythia.web_research.brave_circuit_breaker import get_breaker

        save_stage_state(run_id, _RUN_SCOPE_ISO3, _BREAKER_STATE_KEY, get_breaker().stats())
    except Exception:  # noqa: BLE001
        logger.warning("save_breaker_state failed for %s", run_id, exc_info=True)


def restore_breaker_state(run_id: str) -> bool:
    """Restore persisted breaker stats into this process's breaker."""

    payload = load_stage_state(run_id, _RUN_SCOPE_ISO3, _BREAKER_STATE_KEY)
    if not payload:
        return False
    try:
        from pythia.web_research.brave_circuit_breaker import get_breaker

        get_breaker().restore(payload)
        return True
    except Exception:  # noqa: BLE001
        logger.warning("restore_breaker_state failed for %s", run_id, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Run identity + submission helpers (used by horizon_scanner.main)
# ---------------------------------------------------------------------------


def resolve_run_id_for_pipeline(pipe_id: str) -> Optional[str]:
    """The hs_run_id minted by this pipeline's hs_submit stage."""

    with _DB_LOCK:
        row = batch_con().execute(
            """
            SELECT hs_run_id FROM llm_batches
            WHERE pipeline_id = ? AND hs_run_id IS NOT NULL
            ORDER BY submitted_at DESC LIMIT 1
            """,
            [pipe_id],
        ).fetchone()
    return str(row[0]) if row and row[0] else None


def run_date_from_run_id(run_id: str) -> date_type:
    """Seasonal-filter date anchored to the run id (hs_YYYYMMDDTHHMMSS).

    Collect stages run days after hs_submit; deriving run_date from the
    run id keeps get_active_hazards() consistent across stages so a month
    boundary between stages cannot flip a hazard's seasonal skip.
    """

    raw = str(run_id or "")
    if raw.startswith("hs_"):
        try:
            return datetime.strptime(raw[3:11], "%Y%m%d").date()
        except ValueError:
            pass
    return datetime.now().date()


def submit_family(family: str, *, run_id: str, stage_name: str) -> list[str]:
    """Submit all pending rows of a family as provider batches."""

    from pythia import llm_batch

    pid = _PIPELINE_ID or run_id
    return llm_batch.submit_pending(
        batch_con(),
        family=family,
        pipeline_id=pid,
        stage=stage_name,
        hs_run_id=run_id,
    )


def collect_pending_batches(pipe_id: Optional[str] = None) -> None:
    """Ingest finished provider batches for this pipeline (collect stages).

    Mirrors the forecaster collect phase: poll each pending batch, collect
    terminal ones, cancel-and-expire ones past PYTHIA_BATCH_MAX_WAIT_H so
    their items take the sync fallback path.
    """

    from pythia import llm_batch

    con = batch_con()
    for binfo in llm_batch.pending_batches(con, pipe_id or _PIPELINE_ID):
        status = llm_batch.poll_batch(con, binfo["batch_id"])
        state = status.state if status else "unknown"
        if state == "ended" or (status and status.terminal):
            counts = llm_batch.collect_batch(con, binfo["batch_id"])
            logger.info("hs_batch: collected %s: %s", binfo["batch_id"], counts)
            continue
        age_h = 0.0
        try:
            age_h = (
                datetime.now(timezone.utc).replace(tzinfo=None) - binfo["submitted_at"]
            ).total_seconds() / 3600.0
        except Exception:  # noqa: BLE001
            pass
        if age_h >= llm_batch.max_wait_hours():
            logger.warning(
                "hs_batch: %s still %s after %.1fh — canceling; items fall back to sync",
                binfo["batch_id"], state, age_h,
            )
            llm_batch.cancel_batch(con, binfo["batch_id"])
            llm_batch.collect_batch(con, binfo["batch_id"])
        else:
            logger.warning(
                "hs_batch: %s not finished (%s); its items fall back to sync this stage",
                binfo["batch_id"], state,
            )


def batch_flags_summary() -> str:
    enabled = (os.getenv("PYTHIA_BATCH_API_ENABLED", "0") or "0").strip().lower() in ("1", "true", "yes")
    providers = os.getenv("PYTHIA_BATCH_PROVIDERS", "openai,anthropic,google")
    return f"batch_enabled={enabled} providers={providers} stage={_STAGE} pipeline={_PIPELINE_ID}"
