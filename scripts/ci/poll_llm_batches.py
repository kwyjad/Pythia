# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Poller for provider Batch-API jobs in the staged pipeline.

Runs from poll_llm_batches.yml every 15 minutes. Convergent by design: each
tick re-derives all state from Actions artifacts, so a missed cron tick or a
crashed poller run only delays a pipeline, never loses it.

Per tick:
1. List recent successful runs of the submit-capable workflows and download
   each run's tiny ``pythia-batch-state`` artifact (KBs — never the DB).
2. Keep the newest state per pipeline_id.
3. Dispatch-once guard: skip a pipeline when a "Pythia Pipeline Stage" run
   for (pipeline_id, next_stage) is already queued/in_progress/succeeded
   (the collect stages are idempotent, so a rare double-fire is harmless).
4. Poll every pending provider batch via the pythia.llm_batch adapters
   (needs the three provider API-key secrets).
5. When all batches are terminal — or the oldest exceeds
   PYTHIA_BATCH_MAX_WAIT_H — dispatch the next stage via
   ``gh workflow run`` (the collect stage cancels/expires stragglers and
   falls back to sync per item).

Exit code is always 0 unless the gh CLI itself is unusable: a provider
hiccup on one pipeline must not block polling the others.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

# Workflows whose successful runs may carry a pythia-batch-state artifact.
# Only pythia_pipeline_stage.yml calls emit_batch_state — listing other
# workflows here just wastes one artifact-download attempt per run per tick.
SUBMIT_WORKFLOWS = ("Pythia Pipeline Stage",)
STAGE_WORKFLOW_FILE = "pythia_pipeline_stage.yml"
STAGE_WORKFLOW_NAME = "Pythia Pipeline Stage"
# 30 (not 15): the failure-attempt count below is derived from this window;
# a shorter window would both under-count failures and let repeated failed
# re-dispatches push the submit run (the one carrying the batch-state
# artifact) out of discovery, silently dropping the pipeline.
RUNS_PER_WORKFLOW = 30

# A stage run that ended one of these ways counts as a spent dispatch
# attempt for the stall cap.
FAILED_CONCLUSIONS = ("failure", "cancelled", "timed_out", "startup_failure")

# Run-shaping workflow_dispatch inputs that must survive a stage hand-off.
# A schedule event carries none of them, so a production pipeline sees the
# workflow defaults either way; a smoke run (only_countries=IRN,SOM,
# disable_brave, a single batch provider) would otherwise quietly revert to
# full production settings the moment the poller dispatched stage 2.
CARRIED_INPUTS = ("batch_providers", "only_countries", "grounding_primary", "disable_brave")

_DECISION_PATH = os.getenv("POLLER_DECISION_PATH", "diagnostics/poll_decision.json")


def _dispatch_input_args(state: dict) -> list[str]:
    """`gh workflow run` -f args for the inputs carried by *state*.

    Only keys the state actually carries are forwarded, so a state artifact
    written before this field existed dispatches exactly as it used to.
    """

    carried = state.get("dispatch_inputs") or {}
    if not isinstance(carried, dict):
        return []
    args: list[str] = []
    for key in CARRIED_INPUTS:
        if key not in carried:
            continue
        args += ["-f", f"{key}={carried.get(key) or ''}"]
    return args


def _max_dispatch_attempts() -> int:
    try:
        return int(os.getenv("PYTHIA_STAGE_MAX_DISPATCH_ATTEMPTS", "3") or 3)
    except ValueError:
        return 3


def _retry_cooldown_minutes() -> float:
    try:
        return float(os.getenv("PYTHIA_STAGE_RETRY_COOLDOWN_MIN", "60") or 60)
    except ValueError:
        return 60.0


def _dispatch_decision(
    pid: str,
    next_stage: str,
    stage_runs: list[dict],
    now: datetime,
    *,
    max_attempts: int,
    min_retry_minutes: float,
) -> tuple[bool, str]:
    """(dispatch?, reason) for one pipeline's next stage.

    The marker "{pid} — {next_stage}" is embedded in the stage workflow's
    run-name. Rules, in order:
      1. queued/in_progress/success marker run → skip (already handled).
      2. >= max_attempts failed marker runs → STALLED: never re-dispatch
         (before this cap existed, a red stage was re-dispatched every 15
         minutes forever).
      3. newest failed marker run younger than the cooldown → wait a tick.
      4. otherwise → dispatch.
    """

    marker = f"{pid} — {next_stage}"
    with_marker = [r for r in stage_runs if marker in str(r.get("displayTitle") or "")]
    # "completed" and "running" are split because the self-rescheduling chain
    # keys off it: a running stage means come back later, a completed one means
    # this pipeline is finished (the final stage emits no batch state, so its
    # last state lingers for the artifact's 14-day retention and would
    # otherwise re-arm the poller forever).
    if any(r.get("conclusion") == "success" for r in with_marker):
        return False, "already completed"
    if any(r.get("status") in ("queued", "in_progress") for r in with_marker):
        return False, "already running"

    failed = [r for r in with_marker if r.get("conclusion") in FAILED_CONCLUSIONS]
    if len(failed) >= max_attempts:
        return False, "STALLED"
    if failed:
        def _age_min(r: dict) -> float:
            try:
                ts = datetime.fromisoformat(str(r.get("createdAt") or "").replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return (now - ts).total_seconds() / 60.0
            except ValueError:
                return float("inf")

        newest_age_min = min(_age_min(r) for r in failed)
        if newest_age_min < min_retry_minutes:
            return False, (
                f"cooling down after failure ({len(failed)}/{max_attempts} attempts, "
                f"last {newest_age_min:.0f}m ago)"
            )
    return True, f"dispatch (attempt {len(failed) + 1}/{max_attempts})"


# Actions that mean this pipeline still needs another poll. Everything else
# ("already completed", "STALLED", a failed dispatch) is terminal for the
# self-rescheduling chain — see _should_rearm.
REARM_ACTIONS = frozenset({"waiting", "dispatched", "already running", "cooling down"})


def _chain_depth() -> int:
    try:
        return int(os.getenv("POLLER_CHAIN_DEPTH", "0") or "0")
    except ValueError:
        return 0


def _max_chain() -> int:
    """Backstop against a self-dispatch loop that a logic bug keeps alive.

    At the workflow's ~7-minute cadence, 200 links is a little over 24h, which
    is already past PYTHIA_BATCH_MAX_WAIT_H — so a healthy pipeline can never
    reach it.
    """
    try:
        return int(os.getenv("POLLER_MAX_CHAIN", "200") or "200")
    except ValueError:
        return 200


def _should_rearm(decisions: list[dict]) -> tuple[bool, str]:
    """(rearm?, why) — should the poller dispatch another tick of itself?

    Exists because the */15 cron is not honoured: GitHub throttles frequent
    schedules to roughly hourly, which left staged pipelines parked for hours
    between stages. The chain makes the cron an ignition source rather than the
    mechanism of progress.
    """
    live = [d for d in decisions if d.get("action") in REARM_ACTIONS]
    if not live:
        return False, "no pipeline needs another poll"
    depth, cap = _chain_depth(), _max_chain()
    if depth >= cap:
        return False, f"chain cap reached ({depth}/{cap})"
    return True, f"{len(live)} pipeline(s) still in flight: " + ", ".join(
        f"{d['pipeline_id']}={d['action']}" for d in live[:5]
    )


def _write_decisions(decisions: list[dict], rearm: bool, why: str, path: str) -> None:
    """Persist why the poller did what it did (gap G6).

    Dispatch decisions previously existed only in job logs, which expire — so
    "why did this pipeline stall?" had no durable evidence trail.
    """
    payload = {
        "polled_at": datetime.now(timezone.utc).isoformat(),
        "chain_depth": _chain_depth(),
        "rearm": rearm,
        "rearm_reason": why,
        "decisions": decisions,
    }
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"wrote {path}")
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        print(f"[warn] could not write {path}: {type(exc).__name__}: {exc}")


def _write_output(name: str, value: str) -> None:
    path = os.getenv("GITHUB_OUTPUT")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{name}={value}\n")
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        print(f"[warn] could not write GITHUB_OUTPUT: {type(exc).__name__}: {exc}")


def _gh(*args: str) -> str:
    result = subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=True
    )
    return result.stdout


def _gh_ok(*args: str) -> bool:
    return subprocess.run(["gh", *args], capture_output=True, text=True).returncode == 0


def _list_runs(workflow: str) -> list[dict]:
    try:
        out = _gh(
            "run", "list", "--workflow", workflow, "--branch", "main",
            "--json", "databaseId,createdAt,status,conclusion,displayTitle",
            "--limit", str(RUNS_PER_WORKFLOW),
        )
        return json.loads(out or "[]")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] gh run list failed for {workflow!r}: {exc}")
        return []


def _download_state(run_id: int, dest: str) -> dict | None:
    if not _gh_ok("run", "download", str(run_id), "-n", "pythia-batch-state", "--dir", dest):
        return None
    for root, _dirs, files in os.walk(dest):
        for name in files:
            if name.endswith(".json"):
                try:
                    with open(os.path.join(root, name), encoding="utf-8") as fh:
                        return json.load(fh)
                except Exception:  # noqa: BLE001
                    return None
    return None


def _max_wait_hours() -> float:
    try:
        return float(os.getenv("PYTHIA_BATCH_MAX_WAIT_H", "24") or 24)
    except ValueError:
        return 24.0


def _age_hours(iso_ts: str | None) -> float:
    if not iso_ts:
        return 0.0
    try:
        ts = datetime.fromisoformat(str(iso_ts).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
    except ValueError:
        return 0.0


def _poll_provider(provider: str, provider_batch_id: str) -> str:
    """Return the adapter's state string, or 'poll_error' on failure."""

    try:
        from pythia.llm_batch import _adapter

        status = _adapter(provider).poll(provider_batch_id)
        return status.state
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] poll failed {provider}:{provider_batch_id}: {exc}")
        return "poll_error"


def main() -> int:
    # 1-2. Newest batch-state per pipeline across submit-capable workflows.
    states: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for workflow in SUBMIT_WORKFLOWS:
            for run in _list_runs(workflow):
                if run.get("conclusion") != "success":
                    continue
                run_id = run["databaseId"]
                state = _download_state(run_id, os.path.join(tmp, str(run_id)))
                if not state or not state.get("pipeline_id") or not state.get("next_stage"):
                    continue
                pid = str(state["pipeline_id"])
                if pid not in states or str(state.get("created_at") or "") > str(
                    states[pid].get("created_at") or ""
                ):
                    states[pid] = state

    if not states:
        print("No pending pipelines (no pythia-batch-state artifacts found).")
        _write_output("rearm", "false")
        _write_decisions([], False, "no batch-state artifacts", _DECISION_PATH)
        return 0

    # 3. Dispatch-once guard data: existing stage runs (any status).
    stage_runs = _list_runs(STAGE_WORKFLOW_NAME)

    summary_lines: list[str] = []
    decisions: list[dict] = []
    for pid, state in sorted(states.items()):
        next_stage = str(state["next_stage"])
        record: dict = {
            "pipeline_id": pid,
            "next_stage": next_stage,
            "db_run_id": state.get("db_run_id"),
            "state_created_at": state.get("created_at"),
            "batches": [],
        }
        decisions.append(record)
        dispatch, reason = _dispatch_decision(
            pid,
            next_stage,
            stage_runs,
            datetime.now(timezone.utc),
            max_attempts=_max_dispatch_attempts(),
            min_retry_minutes=_retry_cooldown_minutes(),
        )
        if not dispatch:
            if reason == "STALLED":
                print(
                    f"::error title=Pythia pipeline stalled::{pid}: stage "
                    f"{next_stage} failed {_max_dispatch_attempts()}x — the poller "
                    f"will NOT re-dispatch. Recover by re-running 'Pythia Pipeline "
                    f"Stage' manually with stage={next_stage} pipeline_id={pid} "
                    f"db_run_id={state.get('db_run_id') or ''}."
                )
            record["action"] = reason
            record["reason"] = reason
            summary_lines.append(f"{pid}: {next_stage} — {reason}")
            continue

        pending = state.get("pending") or []
        unfinished: list[str] = []
        for batch in pending:
            batch_state = _poll_provider(
                str(batch.get("provider") or ""), str(batch.get("provider_batch_id") or "")
            )
            record["batches"].append({
                "batch_id": batch.get("batch_id"),
                "provider": batch.get("provider"),
                "state": batch_state,
            })
            if batch_state not in ("ended", "failed", "expired", "canceled"):
                unfinished.append(f"{batch.get('batch_id')}={batch_state}")

        age_h = _age_hours(state.get("created_at"))
        record["age_hours"] = round(age_h, 2)
        if unfinished and age_h < _max_wait_hours():
            record["action"] = "waiting"
            record["reason"] = f"{len(unfinished)}/{len(pending)} batch(es) unfinished"
            summary_lines.append(
                f"{pid}: waiting on {len(unfinished)}/{len(pending)} batch(es) "
                f"(age {age_h:.1f}h): {', '.join(unfinished[:5])}"
            )
            continue

        if unfinished:
            print(
                f"[warn] {pid}: {len(unfinished)} batch(es) still unfinished after "
                f"{age_h:.1f}h — dispatching {next_stage} anyway (collect cancels + "
                "falls back to sync per item)"
            )

        try:
            _gh(
                "workflow", "run", STAGE_WORKFLOW_FILE, "--ref", "main",
                "-f", f"stage={next_stage}",
                "-f", f"pipeline_id={pid}",
                "-f", f"db_run_id={state.get('db_run_id') or ''}",
                "-f", f"test_mode={'true' if state.get('test_mode') else 'false'}",
                *_dispatch_input_args(state),
            )
            record["action"] = "dispatched"
            record["reason"] = reason
            summary_lines.append(f"{pid}: dispatched {next_stage} (db_run_id={state.get('db_run_id')})")
        except Exception as exc:  # noqa: BLE001
            record["action"] = "dispatch failed"
            record["reason"] = str(exc)
            summary_lines.append(f"{pid}: dispatch FAILED: {exc}")

    rearm, why = _should_rearm(decisions)
    _write_output("rearm", "true" if rearm else "false")
    _write_output("chain_depth_next", str(_chain_depth() + 1))
    _write_decisions(decisions, rearm, why, _DECISION_PATH)

    print("=== poll_llm_batches summary ===")
    for line in summary_lines:
        print(f"  {line}")
    print(f"  rearm={rearm} ({why})")
    step_summary = os.getenv("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as fh:
            fh.write("### LLM batch poller\n")
            for line in summary_lines:
                fh.write(f"- {line}\n")
            fh.write(f"- rearm: `{rearm}` — {why}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
