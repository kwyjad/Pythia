# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The regime-change shadow arm: a second model family, stored, never used.

For every hazard production RC assessed (pass 1, gemini_flash), the shadow
model (role ``rc_shadow``, GPT-6.1 Sol at medium effort) is sent the
byte-identical RC prompt, grounding and evidence included. Its answer is
parsed with the same schema and normaliser, its level computed with the same
thresholds, and the result is written to ``hs_rc_shadow`` and to
``llm_calls`` (``is_shadow = TRUE``, ``call_type = 'rc_shadow'``,
``hazard_code = RC_<HZ>_SHADOW``). Nothing here writes ``hs_triage``,
``need_full_spd``, a track or a question: the shadow is a comparison, and
promotion is the owner's decision after at least two live months of recall
data (``pythia/tools/rc_recall_check.py``).

This is NOT pass 2. ``_rc_pass_count``, ``PYTHIA_HS_RC_PASSES`` and the pass
merge are untouched.

**Never delays the HS stage.** In the staged pipeline the shadow calls go to
their own batch family, ``hs_rc_shadow``, which is non-blocking
(``llm_batch.NON_BLOCKING_FAMILIES``): the poller does not wait for it, the
collect drain does not wait for it, and it is collected whenever a later
stage finds it finished (``collect_shadow``). The last stage
(fc_collect_finalize) collects what has finished and cancels the rest, which
is recorded ``unfinished``. A request that was never submitted (no batch, a
refused submit) runs synchronously at the first collect stage, inside the
cost cap. In the sync (legacy) path the shadow calls run on a background
thread pool that the run waits on, bounded, at its end.

**Cost cap.** ``PYTHIA_RC_SHADOW_MAX_USD`` (default 15) per process: once the
shadow has spent, or at submit is estimated to spend, that much, no further
shadow call is issued and each one skipped is counted (``skipped_cap``).
Batch replays already paid for are always recorded.

**Failures are logged, counted, and carried on.** Every public function here
catches its own errors.

Env: ``PYTHIA_RC_SHADOW_ENABLED`` (default on in production, off in test mode
unless set), ``PYTHIA_RC_SHADOW_MODEL`` (``provider:model_id``),
``PYTHIA_RC_SHADOW_THINKING`` (default the config's ``role_params``, medium),
``PYTHIA_RC_SHADOW_MAX_USD``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SHADOW_FAMILY = "hs_rc_shadow"
SHADOW_PASS_IDX = 1
CALL_TYPE = "rc_shadow"
DEFAULT_MAX_USD = 15.0
# A submit-time cost estimate: the RC answer is a short JSON, and medium
# effort thinks for a few thousand tokens.
_EST_OUTPUT_TOKENS = 3000

_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {"spent": 0.0, "estimated": 0.0, "calls": 0, "failures": 0, "skipped_cap": 0}
_EXECUTOR: Optional[concurrent.futures.ThreadPoolExecutor] = None
_FUTURES: List[concurrent.futures.Future] = []


def hazard_label(hazard_code: str) -> str:
    """The llm_calls hazard_code label; uppercased by the logger to RC_<HZ>_SHADOW."""

    return f"rc_{(hazard_code or '').upper()}_shadow"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def enabled() -> bool:
    raw = os.getenv("PYTHIA_RC_SHADOW_ENABLED")
    if raw is not None and raw.strip() != "":
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if os.getenv("PYTEST_CURRENT_TEST"):
        # Unset under pytest: off, so suites that exercise the RC path make
        # no shadow calls unless a test asks for them (the API's rate limits
        # use the same rule).
        return False
    try:
        from pythia.test_mode import is_test_mode

        return not is_test_mode()
    except Exception:  # noqa: BLE001
        return True


def wait_seconds() -> float:
    """How long a stage waits, at its end, for sync-path shadow calls."""

    try:
        return float(os.getenv("PYTHIA_RC_SHADOW_WAIT_SEC", "600") or 600)
    except ValueError:
        return 600.0


def max_usd() -> float:
    try:
        return float(os.getenv("PYTHIA_RC_SHADOW_MAX_USD", str(DEFAULT_MAX_USD)) or DEFAULT_MAX_USD)
    except ValueError:
        return DEFAULT_MAX_USD


def shadow_spec():
    """The shadow ModelSpec: env > config role ``rc_shadow`` (+ role_params)."""

    from forecaster.providers import ModelSpec
    from pythia.llm_profiles import get_role_model, get_role_params, split_model_ref

    ref = (os.getenv("PYTHIA_RC_SHADOW_MODEL") or "").strip() or get_role_model("rc_shadow")
    provider, model_id = split_model_ref(ref, "openai")
    thinking = (os.getenv("PYTHIA_RC_SHADOW_THINKING") or "").strip().lower() or None
    if thinking is None:
        thinking = str(get_role_params("rc_shadow").get("thinking") or "").strip().lower() or None
    return ModelSpec(
        name=model_id,
        provider=provider,
        model_id=model_id,
        active=True,
        purpose="hs_regime_change",
        thinking=thinking,
    )


def stats() -> Dict[str, Any]:
    with _LOCK:
        return dict(_STATE)


def reset() -> None:
    """Tests and a new run: clear the per-process counters."""

    with _LOCK:
        _STATE.update({"spent": 0.0, "estimated": 0.0, "calls": 0, "failures": 0, "skipped_cap": 0})


def _reserve(estimate: float) -> bool:
    """True when a call costing ``estimate`` still fits the cap; records it."""

    cap = max_usd()
    with _LOCK:
        committed = _STATE["spent"] + _STATE["estimated"]
        if cap > 0 and committed + max(estimate, 0.0) > cap:
            _STATE["skipped_cap"] += 1
            return False
        _STATE["estimated"] += max(estimate, 0.0)
        return True


def _settle(estimate: float, actual: float) -> None:
    with _LOCK:
        _STATE["estimated"] = max(0.0, _STATE["estimated"] - max(estimate, 0.0))
        _STATE["spent"] += max(actual, 0.0)


def estimate_cost(prompt: str, model_id: str) -> float:
    from forecaster.providers import estimate_cost_usd

    usage = {"prompt_tokens": len(prompt or "") // 4, "completion_tokens": _EST_OUTPUT_TOKENS}
    try:
        return float(estimate_cost_usd(model_id, usage))
    except Exception:  # noqa: BLE001
        return 0.0


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS hs_rc_shadow (
    hs_run_id TEXT,
    iso3 TEXT,
    hazard_code TEXT,
    provider TEXT,
    model_id TEXT,
    status TEXT,
    likelihood DOUBLE,
    magnitude DOUBLE,
    score DOUBLE,
    level INTEGER,
    direction TEXT,
    "window" TEXT,
    rationale_json TEXT,
    trigger_signals_json TEXT,
    error_text TEXT,
    cost_usd DOUBLE,
    source TEXT,
    is_shadow BOOLEAN DEFAULT TRUE,
    is_replay BOOLEAN DEFAULT FALSE,
    is_test BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP
)
"""


def ensure_table(con) -> None:
    con.execute(_TABLE_DDL)


def parse_answer(text: str) -> Dict[str, Any]:
    """The production RC parser and normaliser, plus the production level."""

    from horizon_scanner._utils import parse_json_response
    from horizon_scanner.regime_change import compute_level, compute_score, coerce_regime_change

    raw = (text or "").strip()
    if not raw:
        return {"status": "error", "error_text": "empty response"}
    try:
        parsed = parse_json_response(raw)
    except Exception as exc:  # noqa: BLE001
        return {"status": "parse_error", "error_text": f"{type(exc).__name__}: {exc}"[:300]}
    rc = coerce_regime_change(parsed)
    likelihood = rc.get("likelihood")
    magnitude = rc.get("magnitude")
    score = rc.get("score")
    if score is None:
        score = compute_score(likelihood, magnitude)
    level = rc.get("level")
    if level is None:
        level = compute_level(likelihood, magnitude, score)
    return {
        "status": "ok" if rc.get("valid", True) else "parse_error",
        "likelihood": likelihood,
        "magnitude": magnitude,
        "score": score,
        "level": int(level) if level is not None else None,
        "direction": rc.get("direction"),
        "window": rc.get("window"),
        "rationale_bullets": rc.get("rationale_bullets") or [],
        "trigger_signals": rc.get("trigger_signals") or [],
    }


def write_row(
    con,
    *,
    hs_run_id: str,
    iso3: str,
    hazard_code: str,
    provider: str,
    model_id: str,
    result: Dict[str, Any],
    cost_usd: float = 0.0,
    source: str = "sync",
    is_replay: bool = False,
    is_test: Optional[bool] = None,
) -> None:
    if is_test is None:
        try:
            from pythia.test_mode import is_test_mode

            is_test = is_test_mode()
        except Exception:  # noqa: BLE001
            is_test = False
    ensure_table(con)
    con.execute(
        "DELETE FROM hs_rc_shadow WHERE hs_run_id = ? AND iso3 = ? AND hazard_code = ? AND is_replay = ?",
        [hs_run_id, iso3.upper(), hazard_code.upper(), bool(is_replay)],
    )
    con.execute(
        """
        INSERT INTO hs_rc_shadow (
            hs_run_id, iso3, hazard_code, provider, model_id, status, likelihood,
            magnitude, score, level, direction, "window", rationale_json,
            trigger_signals_json, error_text, cost_usd, source, is_shadow,
            is_replay, is_test, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, TRUE, ?, ?, ?)
        """,
        [
            hs_run_id, iso3.upper(), hazard_code.upper(), provider, model_id,
            result.get("status"), result.get("likelihood"), result.get("magnitude"),
            result.get("score"), result.get("level"), result.get("direction"),
            result.get("window"),
            json.dumps(result.get("rationale_bullets") or [], ensure_ascii=False),
            json.dumps(result.get("trigger_signals") or [], ensure_ascii=False, default=str),
            (result.get("error_text") or None), float(cost_usd or 0.0), source,
            bool(is_replay), bool(is_test), datetime.now(timezone.utc).replace(tzinfo=None),
        ],
    )


def _record(
    *,
    hs_run_id: str,
    iso3: str,
    hazard_code: str,
    spec,
    prompt: str,
    text: str,
    usage: Dict[str, Any],
    error: str,
    source: str,
    con=None,
) -> Dict[str, Any]:
    """Parse, write hs_rc_shadow, write the llm_calls row. Never raises."""

    from forecaster.providers import estimate_cost_usd

    usage = dict(usage or {})
    cost = float(usage.get("cost_usd") or 0.0)
    if not cost and (usage.get("prompt_tokens") or usage.get("completion_tokens")):
        try:
            cost = float(estimate_cost_usd(spec.model_id, usage))
        except Exception:  # noqa: BLE001
            cost = 0.0
        usage["cost_usd"] = cost
    result = {"status": "error", "error_text": str(error)[:300]} if error else parse_answer(text)
    usage["rc_shadow"] = {k: result.get(k) for k in ("status", "likelihood", "magnitude", "level", "direction")}
    try:
        from horizon_scanner.llm_logging import log_hs_llm_call

        log_hs_llm_call(
            hs_run_id=hs_run_id,
            iso3=iso3,
            hazard_code=hazard_label(hazard_code),
            model_spec=spec,
            prompt_text=prompt or "",
            response_text=text or "",
            usage=usage,
            error_text=(str(error) if error else None),
            call_type=CALL_TYPE,
            is_shadow=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("rc_shadow: llm_calls log failed for %s %s: %s", iso3, hazard_code, exc)
    try:
        if con is None:
            from pythia.db.schema import connect

            con = connect(read_only=False)
        write_row(
            con, hs_run_id=hs_run_id, iso3=iso3, hazard_code=hazard_code,
            provider=spec.provider, model_id=spec.model_id, result=result,
            cost_usd=cost, source=source,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("rc_shadow: hs_rc_shadow write failed for %s %s: %s", iso3, hazard_code, exc)
    with _LOCK:
        _STATE["calls"] += 1
        if result.get("status") != "ok":
            _STATE["failures"] += 1
    return result


def record_skipped(hs_run_id: str, iso3: str, hazard_code: str, status: str, reason: str, con=None) -> None:
    try:
        spec = shadow_spec()
        if con is None:
            from pythia.db.schema import connect

            con = connect(read_only=False)
        write_row(
            con, hs_run_id=hs_run_id, iso3=iso3, hazard_code=hazard_code,
            provider=spec.provider, model_id=spec.model_id,
            result={"status": status, "error_text": reason}, source="none",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("rc_shadow: could not record %s for %s %s: %s", status, iso3, hazard_code, exc)


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------


def _sync_call(prompt: str, *, hs_run_id: str, iso3: str, hazard_code: str, source: str) -> None:
    """One synchronous shadow call inside the cap. Never raises."""

    try:
        spec = shadow_spec()
        estimate = estimate_cost(prompt, spec.model_id)
        if not _reserve(estimate):
            record_skipped(hs_run_id, iso3, hazard_code, "skipped_cap",
                           f"PYTHIA_RC_SHADOW_MAX_USD={max_usd():g} reached")
            return
        from forecaster.providers import call_chat_ms

        t0 = time.time()
        try:
            text, usage, error = asyncio.run(
                call_chat_ms(
                    spec, prompt, temperature=0.0,
                    prompt_key="hs.regime_change.shadow", prompt_version="2.0.0",
                    component="HorizonScanner", run_id=hs_run_id, log_call=False,
                )
            )
        except Exception as exc:  # noqa: BLE001
            text, usage, error = "", {}, f"provider call error: {exc}"
        usage = dict(usage or {})
        usage.setdefault("elapsed_ms", int((time.time() - t0) * 1000))
        result_cost = 0.0
        try:
            from forecaster.providers import estimate_cost_usd

            result_cost = float(usage.get("cost_usd") or estimate_cost_usd(spec.model_id, usage))
        except Exception:  # noqa: BLE001
            pass
        _settle(estimate, result_cost)
        _record(hs_run_id=hs_run_id, iso3=iso3, hazard_code=hazard_code, spec=spec,
                prompt=prompt, text=text, usage=usage, error=error, source=source)
    except Exception as exc:  # noqa: BLE001
        logger.warning("rc_shadow: call failed for %s %s: %s", iso3, hazard_code, exc)
        with _LOCK:
            _STATE["failures"] += 1


def run_for_hazard(prompt: str, *, hs_run_id: str, iso3: str, hazard_code: str, mode: str) -> None:
    """Hook called once per hazard pass 1 assessed, with pass 1's prompt.

    ``mode`` is ``hs_batch.family_mode('hs_rc')``: ``submit`` enqueues to
    the non-blocking batch family, ``full`` runs on the background pool,
    ``collect`` does nothing here (``collect_shadow`` handles it). Never
    raises and never returns anything production reads.
    """

    try:
        if not enabled():
            return
        if mode == "submit":
            _enqueue(prompt, iso3=iso3, hazard_code=hazard_code, hs_run_id=hs_run_id)
        elif mode == "full":
            _submit_background(prompt, hs_run_id=hs_run_id, iso3=iso3, hazard_code=hazard_code)
    except Exception as exc:  # noqa: BLE001
        logger.warning("rc_shadow: hook failed for %s %s: %s", iso3, hazard_code, exc)


def _enqueue(prompt: str, *, iso3: str, hazard_code: str, hs_run_id: str) -> None:
    from forecaster.providers import build_body_for_spec
    from horizon_scanner import hs_batch
    from pythia import llm_batch

    spec = shadow_spec()
    # A provider outside PYTHIA_BATCH_PROVIDERS is still enqueued: submit
    # skips it, the row stays 'pending' with no batch, and collect_shadow runs
    # it synchronously (inside the cap) at the first collect stage.
    if not _reserve(estimate_cost(prompt, spec.model_id)):
        record_skipped(hs_run_id, iso3, hazard_code, "skipped_cap",
                       f"PYTHIA_RC_SHADOW_MAX_USD={max_usd():g} reached (estimated at submit)")
        return
    body = build_body_for_spec(spec, prompt, 0.0)
    with hs_batch._DB_LOCK:
        llm_batch.enqueue_request(
            hs_batch.batch_con(),
            family=SHADOW_FAMILY,
            provider=spec.provider,
            model_id=spec.model_id,
            request_body=body,
            prompt_text=prompt,
            iso3=iso3,
            hazard_code=hazard_code,
            pass_idx=SHADOW_PASS_IDX,
            pipeline_id=hs_batch.pipeline_id(),
        )


def _submit_background(prompt: str, *, hs_run_id: str, iso3: str, hazard_code: str,
                       source: str = "sync") -> None:
    global _EXECUTOR
    with _LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="rc_shadow")
        _FUTURES.append(
            _EXECUTOR.submit(_sync_call, prompt, hs_run_id=hs_run_id, iso3=iso3,
                             hazard_code=hazard_code, source=source)
        )


def wait_background(timeout_sec: float = 900.0) -> Dict[str, Any]:
    """Wait, bounded, for the sync-path shadow calls. Never raises."""

    global _EXECUTOR
    with _LOCK:
        futures = list(_FUTURES)
        _FUTURES.clear()
        executor = _EXECUTOR
        _EXECUTOR = None
    done, not_done = concurrent.futures.wait(futures, timeout=timeout_sec) if futures else (set(), set())
    for fut in not_done:
        fut.cancel()
    if executor is not None:
        executor.shutdown(wait=False, cancel_futures=True)
    out = {"waited": len(futures), "done": len(done), "abandoned": len(not_done), **stats()}
    logger.info("rc_shadow: background calls finished: %s", out)
    return out


# ---------------------------------------------------------------------------
# Staged pipeline: submit and opportunistic collect
# ---------------------------------------------------------------------------


def submit(run_id: str, stage_name: str) -> None:
    """Submit the enqueued shadow rows as their own provider batches."""

    try:
        if not enabled():
            return
        from horizon_scanner import hs_batch

        created = hs_batch.submit_family(SHADOW_FAMILY, run_id=run_id, stage_name=stage_name)
        print(f"[rc_shadow] submitted {len(created)} shadow batch(es) (non-blocking)", flush=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("rc_shadow: submit failed (shadow rows stay pending): %s", exc)


def _prompt_from_body(body_json: Optional[str]) -> str:
    try:
        body = json.loads(body_json or "{}")
    except (TypeError, ValueError):
        return ""
    msgs = body.get("messages") or []
    if msgs and isinstance(msgs[0], dict):
        content = msgs[0].get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(str(c.get("text") or "") for c in content if isinstance(c, dict))
    contents = body.get("contents") or []
    if contents:
        parts = (contents[0] or {}).get("parts") or []
        return "".join(str(p.get("text") or "") for p in parts if isinstance(p, dict))
    return ""


def collect_shadow(con, pipeline_id: str, hs_run_id: str, *, final: bool = False,
                   run_unsubmitted: bool = True) -> Dict[str, Any]:
    """Collect whatever shadow work has finished; never wait for what has not.

    * A shadow batch that has ended is collected and each item recorded.
    * One still running is left alone, unless ``final``: then it is
      cancelled (no settle wait) and its items are recorded ``unfinished``.
    * A request never submitted (no batch id) runs synchronously, inside the
      cap, when ``run_unsubmitted``.
    Never raises.
    """

    from pythia import llm_batch

    report: Dict[str, Any] = {"collected_batches": 0, "recorded": 0, "unfinished": 0,
                              "ran_sync": 0, "left_running": 0}
    if not pipeline_id:
        return report
    try:
        ensure_table(con)
        for binfo in llm_batch.pending_batches(con, pipeline_id):
            if binfo.get("family") != SHADOW_FAMILY:
                continue
            bid = binfo["batch_id"]
            status = None
            try:
                status = llm_batch.poll_batch(con, bid)
            except Exception as exc:  # noqa: BLE001
                logger.warning("rc_shadow: poll failed for %s: %s", bid, exc)
            state = status.state if status else "unknown"
            if status is not None and (state == "ended" or status.terminal):
                llm_batch.collect_batch(con, bid)
                report["collected_batches"] += 1
            elif final:
                try:
                    llm_batch._adapter(binfo["provider"]).cancel(binfo["provider_batch_id"])
                except Exception:  # noqa: BLE001
                    pass
                con.execute(
                    "UPDATE llm_batches SET status = 'canceled' WHERE batch_id = ?", [bid]
                )
            else:
                report["left_running"] += 1

        done = {
            (r[0], r[1])
            for r in con.execute(
                "SELECT iso3, hazard_code FROM hs_rc_shadow WHERE hs_run_id = ? AND is_replay = FALSE",
                [hs_run_id],
            ).fetchall()
        }
        rows = con.execute(
            """
            SELECT r.custom_id, r.iso3, r.hazard_code, r.status, r.batch_id,
                   r.request_body_json, r.provider, r.model_id
            FROM llm_batch_requests r
            WHERE r.family = ? AND r.pipeline_id = ?
            """,
            [SHADOW_FAMILY, pipeline_id],
        ).fetchall()
        for custom_id, iso3, hz, status, batch_id, body_json, provider, model_id in rows:
            iso3_u, hz_u = (iso3 or "").upper(), (hz or "").upper()
            if (iso3_u, hz_u) in done:
                continue
            prompt = _prompt_from_body(body_json)
            if status in ("succeeded", "errored"):
                hit = llm_batch.get_result(con, SHADOW_FAMILY, iso3=iso3, hazard_code=hz,
                                           pass_idx=SHADOW_PASS_IDX, pipeline_id=pipeline_id)
                if hit is None:
                    continue
                usage = dict(hit.get("usage") or {})
                usage["service_tier"] = "batch"
                used_spec = shadow_spec()
                used_spec.provider, used_spec.model_id, used_spec.name = provider, model_id, model_id
                _record(hs_run_id=hs_run_id, iso3=iso3_u, hazard_code=hz_u, spec=used_spec,
                        prompt=hit.get("sent_prompt") or prompt, text=hit.get("text") or "",
                        usage=usage, error=(hit.get("error") or "") if hit.get("status") != "succeeded" else "",
                        source="batch", con=con)
                report["recorded"] += 1
            elif batch_id is None and status == "pending" and run_unsubmitted:
                # Never submitted (provider not batchable, or a refused
                # submit): run it on the background pool, which the stage
                # waits on, bounded, at its end.
                _submit_background(prompt, hs_run_id=hs_run_id, iso3=iso3_u,
                                   hazard_code=hz_u, source="sync_fallback")
                report["ran_sync"] += 1
            elif final:
                record_skipped(hs_run_id, iso3_u, hz_u, "unfinished",
                               "shadow batch had not finished by the last stage; cancelled", con=con)
                report["unfinished"] += 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("rc_shadow: collect failed: %s", exc)
        report["error"] = f"{type(exc).__name__}: {exc}"[:300]
    report.update(stats())
    print(f"[rc_shadow] collect (final={final}): {report}", flush=True)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    """CLI for the workflow's stage steps. Always exits 0."""

    import argparse

    ap = argparse.ArgumentParser(description="Collect the RC shadow arm's batches")
    ap.add_argument("command", choices=["collect"])
    ap.add_argument("--pipeline-id", required=True)
    ap.add_argument("--hs-run-id", default="")
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args(argv)
    try:
        from pythia.db.schema import connect, ensure_schema

        con = connect(read_only=False)
        ensure_schema(con)
        hs_run_id = args.hs_run_id
        if not hs_run_id:
            from horizon_scanner import hs_batch

            hs_run_id = hs_batch.resolve_run_id_for_pipeline(args.pipeline_id) or ""
        if not hs_run_id:
            print("[rc_shadow] no hs_run_id for this pipeline; nothing to collect", flush=True)
            return 0
        collect_shadow(con, args.pipeline_id, hs_run_id, final=args.final)
        try:
            con.close()
        except Exception:  # noqa: BLE001
            pass
    except Exception as exc:  # noqa: BLE001
        print(f"::warning title=RC shadow collect failed::{type(exc).__name__}: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
