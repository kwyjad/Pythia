# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Gate: is a staged Batch-API pipeline currently in flight?

Three answers, never two (Oct 2026): ``in_flight``, ``idle`` and ``unknown``.

On 5 October 2026 at 13:32 this gate answered "in flight" although the
2 October pipeline had finished at 15:2x that day: the newest batch-state
artifact was 70.6 hours old (inside the 72-hour window) and the run listing
came back without the final stage, which the old two-answer core read as
"no final stage yet". An empty or partial listing is not evidence of a
pipeline; it is the absence of an answer. The listing is now paged, checked
for completeness (it must hold the run that uploaded the newest batch-state
artifact, and as many runs as the API says exist), and retried with backoff;
what it still cannot settle is ``unknown``.

Consumers and what each does with ``unknown``:

* ``--emit proceed`` (default): resolver_update, ingest-structured-data,
  haz_backcast and interpreter_backfill, before touching the canonical
  ``pythia-resolver-db`` artifact. On a ``schedule`` event an unknown answer
  PROCEEDS with a warning: the 11th is the only refresh of the resolution
  sources and no pipeline should be in flight then. On any other event (a
  person's dispatch) an unknown answer stops the run RED, because a skip
  must never report success to someone who asked for the work.
* ``--emit active``: poll_llm_batches.yml's activity gate. Unknown polls
  (``active=true``): a broken gate must never stop the poller advancing a
  live pipeline.

A run the gate skips for a real in-flight pipeline uploads no canonical DB,
and the chained workflows read exactly that (``canonical_guard
trigger-did-work``), so a skip starts nothing downstream.

Detection:
1. Newest trusted ``pythia-batch-state`` artifact. None, or older than
   ``PYTHIA_PIPELINE_ACTIVE_WINDOW_H`` (default 72h) -> idle.
2. Otherwise the stage workflow's runs created since that artifact's run.
   A successful ``fc_collect_finalize`` created after the artifact -> idle;
   a complete listing without one -> in flight; an incomplete one -> unknown.

Writes ``state=in_flight|idle|unknown`` and ``proceed=`` (or ``active=``) to
$GITHUB_OUTPUT. FORCE=true (the force_during_pipeline input) bypasses it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

STAGE_WORKFLOW_NAME = "Pythia Pipeline Stage"
STAGE_WORKFLOW_FILE = "pythia_pipeline_stage.yml"
FINAL_STAGE_MARKER = "fc_collect_finalize"

IN_FLIGHT = "in_flight"
IDLE = "idle"
UNKNOWN = "unknown"


def _window_hours() -> float:
    try:
        return float(os.getenv("PYTHIA_PIPELINE_ACTIVE_WINDOW_H", "72") or 72)
    except ValueError:
        return 72.0


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def pipeline_state(
    latest_state_created_at: str | None,
    stage_runs: list[dict] | None,
    now: datetime,
    *,
    window_hours: float = 72.0,
    state_run_id: Any = None,
    listing_complete: bool = True,
) -> str:
    """Pure decision core: ``in_flight`` | ``idle`` | ``unknown``.

    ``stage_runs`` carry ``createdAt``, ``conclusion``, ``displayTitle`` and
    (optionally) ``databaseId``. ``None`` means the listing could not be read.
    When ``state_run_id`` is given, a listing that does not contain that run
    is incomplete by construction."""

    state_ts = _parse_ts(latest_state_created_at)
    if state_ts is None:
        return IDLE
    if (now - state_ts).total_seconds() / 3600.0 > window_hours:
        return IDLE
    if stage_runs is None:
        return UNKNOWN
    for run in stage_runs:
        if run.get("conclusion") != "success":
            continue
        if FINAL_STAGE_MARKER not in str(run.get("displayTitle") or ""):
            continue
        run_ts = _parse_ts(run.get("createdAt"))
        if run_ts is not None and run_ts > state_ts:
            return IDLE
    if not listing_complete:
        return UNKNOWN
    if state_run_id not in (None, "") and not any(
        str(r.get("databaseId")) == str(state_run_id) for r in stage_runs
    ):
        return UNKNOWN
    return IN_FLIGHT


def pipeline_in_flight(
    latest_state_created_at: str | None,
    stage_runs: list[dict],
    now: datetime,
    *,
    window_hours: float = 72.0,
) -> bool:
    """Legacy two-answer form: True only for a definite ``in_flight``."""
    return pipeline_state(
        latest_state_created_at, stage_runs, now, window_hours=window_hours,
    ) == IN_FLIGHT


def decide(state: str, emit: str, event: str) -> tuple[bool, int, str]:
    """``(value, exit_code, note)`` for a state, an output mode and the event
    that started the run (pure; tested)."""
    if emit == "active":
        if state == IN_FLIGHT:
            return True, 0, ""
        if state == UNKNOWN:
            return True, 0, "could not tell whether a pipeline is in flight; polling anyway"
        return False, 0, "nothing to poll, exiting early"
    if state == IDLE:
        return True, 0, ""
    if state == IN_FLIGHT:
        return False, 0, (
            "a staged pipeline is in flight; an ingest now would be silently discarded "
            "by the pipeline's final canonical upload"
        )
    if event == "schedule":
        return True, 0, (
            "could not tell whether a pipeline is in flight; proceeding because this is "
            "the scheduled run, and no pipeline should be in flight at this point in the month"
        )
    return False, 1, (
        "could not tell whether a pipeline is in flight; stopping, because a skip must "
        "never report success to a dispatch that asked for the work. Re-run, or dispatch "
        "with force_during_pipeline=true once you have checked"
    )


def _gh_json(*args: str) -> object:
    out = subprocess.run(
        ["gh", *args], capture_output=True, text=True, check=True
    ).stdout
    return json.loads(out or "null")


def _retry(fn: Callable[[], Any], *, attempts: int, backoff: float,
           sleep: Callable[[float], None]) -> Any:
    last: Exception | None = None
    for i in range(max(1, attempts)):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if i + 1 < attempts:
                sleep(backoff * (2 ** i))
    raise RuntimeError(f"gh failed after {attempts} attempt(s): {last}")


# Runs our own triggers start; see scripts/ci/poll_llm_batches.py.
TRUSTED_EVENTS = frozenset({"schedule", "workflow_dispatch", "workflow_run", "push"})


def _trusted(artifacts: list[dict], repo_id: int | str | None) -> list[dict]:
    if repo_id in (None, ""):
        return []
    return [
        a for a in artifacts or []
        if str((a.get("workflow_run") or {}).get("head_repository_id")) == str(repo_id)
        and (a.get("workflow_run") or {}).get("head_branch") == "main"
        and a.get("created_at")
    ]


def newest_trusted_artifact(artifacts: list[dict], repo_id: int | str | None) -> str | None:
    """created_at of the newest artifact a run of this repository's main uploaded.

    A fork's pull-request run can upload a pythia-batch-state artifact too, and
    would then hold the weekly ingest back or keep the poller busy for three
    days. An artifact whose run came from another repository is not ours.
    """
    ours = _trusted(artifacts, repo_id)
    return max(str(a["created_at"]) for a in ours) if ours else None


def newest_trusted(artifacts: list[dict], repo_id: int | str | None) -> dict | None:
    ours = _trusted(artifacts, repo_id)
    return max(ours, key=lambda a: str(a["created_at"])) if ours else None


def read_state(now: datetime, *, gh: Callable[..., object] = None,
               attempts: int = 4, backoff: float = 15.0,
               sleep: Callable[[float], None] = time.sleep) -> tuple[str, str]:
    """Ask the API; return ``(state, reason)``. Never raises."""
    gh = gh or _gh_json
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    try:
        repo_id = (_retry(lambda: gh("api", f"repos/{repo}"), attempts=attempts,
                          backoff=backoff, sleep=sleep) or {}).get("id")
        artifacts = _retry(
            lambda: gh("api", f"repos/{repo}/actions/artifacts?name=pythia-batch-state&per_page=30"),
            attempts=attempts, backoff=backoff, sleep=sleep,
        )
    except Exception as exc:  # noqa: BLE001
        return UNKNOWN, f"could not read the batch-state artifacts ({exc})"
    newest = newest_trusted((artifacts or {}).get("artifacts") or [], repo_id)
    created = newest.get("created_at") if newest else None
    window = _window_hours()
    created_ts = _parse_ts(created)
    if created_ts is None or (now - created_ts) > timedelta(hours=window):
        return IDLE, f"newest batch-state artifact {created or 'absent'} is outside the {window:g}h window"
    state_run_id = ((newest or {}).get("workflow_run") or {}).get("id")
    since = (created_ts - timedelta(days=4)).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _listing() -> tuple[list[dict], bool]:
        runs: list[dict] = []
        total = None
        page = 1
        while True:
            body = gh(
                "api",
                f"repos/{repo}/actions/workflows/{STAGE_WORKFLOW_FILE}/runs"
                f"?branch=main&created=%3E%3D{since}&per_page=100&page={page}",
            ) or {}
            total = body.get("total_count", total)
            batch = body.get("workflow_runs") or []
            runs.extend(batch)
            if len(batch) < 100:
                break
            page += 1
            if page > 10:
                break
        complete = total is not None and len(runs) >= int(total)
        return runs, complete

    try:
        raw, complete = _retry(_listing, attempts=attempts, backoff=backoff, sleep=sleep)
    except Exception as exc:  # noqa: BLE001
        return UNKNOWN, f"could not list {STAGE_WORKFLOW_NAME} runs ({exc})"
    runs = [
        {
            "databaseId": r.get("id"),
            "createdAt": r.get("created_at"),
            "conclusion": r.get("conclusion"),
            "displayTitle": r.get("display_title"),
        }
        for r in raw
        if r.get("event") in TRUSTED_EVENTS
    ]
    state = pipeline_state(
        created, runs, now, window_hours=window,
        state_run_id=state_run_id, listing_complete=complete,
    )
    if state == IDLE:
        reason = f"a successful {FINAL_STAGE_MARKER} ran after the batch-state artifact of {created}"
    elif state == IN_FLIGHT:
        reason = (f"batch-state artifact from {created} (run {state_run_id}), and the complete "
                  f"listing of {len(runs)} stage run(s) since holds no later successful "
                  f"{FINAL_STAGE_MARKER}")
    else:
        reason = (f"batch-state artifact from {created} (run {state_run_id}), but the stage-run "
                  f"listing ({len(runs)} run(s), complete={complete}) does not hold that run")
    return state, reason


def main(argv: list[str] | None = None, *, sleep: Callable[[float], None] = time.sleep) -> int:
    emit = "proceed"
    args = list(sys.argv[1:] if argv is None else argv)
    if "--emit" in args:
        idx = args.index("--emit")
        if idx + 1 >= len(args) or args[idx + 1] not in ("proceed", "active"):
            print("usage: check_pipeline_active.py [--emit proceed|active]", file=sys.stderr)
            return 2
        emit = args[idx + 1]

    force = (os.getenv("FORCE", "false") or "false").strip().lower() in ("1", "true", "yes")
    event = os.getenv("GITHUB_EVENT_NAME", "")
    if force:
        state, reason = "forced", "FORCE set; skipping the gate"
        value, code, note = True, 0, ""
    else:
        attempts = int(os.getenv("PYTHIA_GATE_ATTEMPTS", "4") or 4)
        backoff = float(os.getenv("PYTHIA_GATE_BACKOFF_SEC", "15") or 15)
        state, reason = read_state(
            datetime.now(timezone.utc), gh=lambda *a: _gh_json(*a),
            attempts=attempts, backoff=backoff, sleep=sleep,
        )
        value, code, note = decide(state, emit, event)

    key = "active" if emit == "active" else "proceed"
    full = f"{reason}" + (f" — {note}" if note else "")
    if code:
        print(f"::error title=Pipeline gate could not decide::{full}")
    elif state == UNKNOWN:
        print(f"::warning title=Pipeline gate could not decide::{full}")
    elif not value:
        title = "Ingest skipped" if key == "proceed" else "Poller idle"
        print(f"::notice title={title}::{full}")
    print(f"state={state} {key}={value} — {full}")
    out_path = os.getenv("GITHUB_OUTPUT")
    if out_path:
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(f"state={state}\n")
            fh.write(f"{key}={'true' if value else 'false'}\n")
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(f"### Pipeline-active gate\n- state: {state}\n- {key}: {value}\n- {full}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
