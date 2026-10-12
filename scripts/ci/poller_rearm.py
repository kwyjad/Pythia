# Pythia / Copyright (c) 2025 Kevin Wyjad
"""Re-arm the batch poller's self-dispatch chain, and do not let it end on
a refused dispatch.

The poller keeps itself alive by dispatching its own next run. At 15:12 UTC
on 7 October 2026 GitHub answered HTTP 500 to all four re-arm attempts the
step made over ninety seconds, the chain ended, and nothing polled until a
person restarted it at 15:39. The cron is only a backstop (and since
2026-10-12 fires only from the 13th to the 17th), and in this repository a
scheduled tick can arrive hours late. On a production
run with a 24-hour batch window that delay can matter.

Two layers:

1. **Retry with backoff, for a bounded time.** A refused dispatch is asked
   again after 30 s, then 60 s, 120 s, 240 s, capped at 300 s between
   attempts, for up to ``--retry-budget-min`` (10 minutes). An API outage of
   a few minutes is now absorbed here.

2. **If it is still refused, keep polling inside this job.** Rather than
   ending, the job runs the poller itself every ``--poll-interval-sec``
   (330 s, the chain's own cadence), so stages are still dispatched when
   batches finish, and after each poll it tries the re-arm again. It stops
   when the poller says nothing is in flight (the chain's natural end), when
   a re-arm succeeds (the chain resumes), or near the job's time limit.

The limit chosen: the poll job's ``timeout-minutes`` is 240, and the
in-job loop stops starting new polls at ``--job-limit-min`` (225) minutes
after the job began, leaving room for the last poll and the error. 240
minutes is four cron intervals: GitHub's scheduled ticks here have arrived
up to several hours late, but a job held for longer than that only to
cover a very rare outage costs a runner for no evidence. In the 30 days to
2026-10-08 the chain was broken this way once in 647 poller runs.

Exits 0 when the chain re-armed or legitimately ended, 1 when it could not
re-arm before the job limit (a red run is the signal a person should see).

``--ignite`` is the other entry point: a submit stage starting the chain at
depth 0. Since 2026-10-12 the poller's cron fires only in the production
window (the 13th to the 17th), so outside it this dispatch is the only thing
that starts a poll. It retries with the same backoff for ``--retry-budget-min``
and always exits 0: a stage that has done its work must not go red because
GitHub refused one dispatch, and the warning names the manual recovery.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

Dispatch = Callable[[int], bool]
Sleep = Callable[[float], None]
Clock = Callable[[], float]
Poll = Callable[[int], tuple[bool, int]]


def dispatch_with_backoff(
    dispatch: Dispatch,
    depth: int,
    *,
    sleep: Sleep,
    clock: Clock,
    budget_s: float,
    first_delay_s: float = 30.0,
    max_delay_s: float = 300.0,
    log: Callable[[str], None] = print,
) -> bool:
    """Try ``dispatch(depth)`` until it succeeds or ``budget_s`` has passed.

    Never sleeps past the budget, so a caller's own deadline is honoured.
    """
    start = clock()
    delay = first_delay_s
    attempt = 0
    while True:
        attempt += 1
        if dispatch(depth):
            if attempt > 1:
                log(f"re-arm dispatched on attempt {attempt}")
            return True
        remaining = budget_s - (clock() - start)
        if remaining <= 0:
            log(f"::warning title=Poller re-arm refused::{attempt} attempt(s) refused over "
                f"{int(clock() - start)} s")
            return False
        wait = min(delay, max_delay_s, remaining)
        log(f"::warning title=Poller re-arm dispatch failed::attempt {attempt}; "
            f"retrying in {int(wait)} s")
        sleep(wait)
        delay = min(delay * 2, max_delay_s)


def run_rearm(
    *,
    next_depth: int,
    dispatch: Dispatch,
    poll: Poll,
    sleep: Sleep,
    clock: Clock,
    deadline: float,
    initial_sleep_s: float = 300.0,
    retry_budget_s: float = 600.0,
    poll_interval_s: float = 330.0,
    in_job_retry_budget_s: float = 120.0,
    log: Callable[[str], None] = print,
) -> int:
    """The whole re-arm step. See the module docstring."""
    sleep(initial_sleep_s)
    if dispatch_with_backoff(dispatch, next_depth, sleep=sleep, clock=clock,
                             budget_s=min(retry_budget_s, max(0.0, deadline - clock())), log=log):
        return 0

    log("::warning title=Poller chain held in-job::GitHub refused the re-arm dispatch; "
        "this job keeps polling until a re-arm succeeds, the pipelines finish, or the job limit")
    depth = next_depth
    n_polls = 0
    while clock() + poll_interval_s + in_job_retry_budget_s < deadline:
        sleep(poll_interval_s)
        n_polls += 1
        rearm, depth_next = poll(depth)
        if not rearm:
            log(f"::notice title=Poller chain ended in-job::after {n_polls} in-job poll(s) "
                "nothing is left in flight")
            return 0
        depth = depth_next
        if dispatch_with_backoff(dispatch, depth, sleep=sleep, clock=clock,
                                 budget_s=in_job_retry_budget_s, log=log):
            log(f"::notice title=Poller chain re-armed::after {n_polls} in-job poll(s)")
            return 0
    log(f"::error title=Poller chain broken::the re-arm dispatch was refused until the job "
        f"limit ({n_polls} in-job poll(s)); dispatch poll_llm_batches.yml by hand (the cron backstop fires only 13th-17th)")
    return 1


# ---------------------------------------------------------------------------
# The real seams
# ---------------------------------------------------------------------------

def _gh_dispatch(depth: int) -> bool:
    try:
        res = subprocess.run(
            ["gh", "workflow", "run", "poll_llm_batches.yml", "--ref", "main",
             "-f", f"chain_depth={depth}"],
            capture_output=True, text=True, timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"gh workflow run raised {type(exc).__name__}: {exc}")
        return False
    if res.returncode != 0:
        print(f"gh workflow run exited {res.returncode}: {(res.stderr or res.stdout).strip()[:300]}")
    return res.returncode == 0


def read_outputs(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _subprocess_poll(depth: int) -> tuple[bool, int]:
    """Run the poller once, as the poll step does, and read its outputs.

    A poll that crashes is read as "still in flight": stopping the chain on
    our own error would be the fault this module exists to prevent.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "out"
        out.touch()
        env = dict(os.environ, POLLER_CHAIN_DEPTH=str(depth), GITHUB_OUTPUT=str(out))
        try:
            res = subprocess.run([sys.executable, "-m", "scripts.ci.poll_llm_batches"],
                                 env=env, timeout=600)
            rc = res.returncode
        except Exception as exc:  # noqa: BLE001
            print(f"in-job poll raised {type(exc).__name__}: {exc}")
            rc = 1
        outputs = read_outputs(out)
    if rc != 0 or "rearm" not in outputs:
        return True, depth + 1
    try:
        nxt = int(outputs.get("chain_depth_next") or depth + 1)
    except ValueError:
        nxt = depth + 1
    return outputs["rearm"] == "true", nxt


def ignite(dispatch: Dispatch, *, sleep: Sleep, clock: Clock, budget_s: float,
           log: Callable[[str], None] = print) -> int:
    """Start the poller chain at depth 0. Always returns 0; says so when refused."""
    if dispatch_with_backoff(dispatch, 0, sleep=sleep, clock=clock, budget_s=budget_s, log=log):
        log("Ignited poll_llm_batches.yml")
        return 0
    log("::warning title=Poller not ignited::GitHub refused the dispatch for "
        f"{int(budget_s)} s. Outside the 13th-17th no cron will start it: run "
        "`gh workflow run poll_llm_batches.yml --ref main` by hand.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Re-arm the poller chain; hold it in-job if refused.")
    ap.add_argument("--ignite", action="store_true",
                    help="start the chain at depth 0 from a submit stage, with retries; never fails")
    ap.add_argument("--next-depth", type=int, default=None)
    ap.add_argument("--job-started-at", type=float, default=None,
                    help="epoch seconds the job started (default: now)")
    ap.add_argument("--job-limit-min", type=float, default=225.0)
    ap.add_argument("--initial-sleep-sec", type=float, default=300.0)
    ap.add_argument("--retry-budget-min", type=float, default=10.0)
    ap.add_argument("--poll-interval-sec", type=float, default=330.0)
    args = ap.parse_args(argv)
    if args.ignite:
        return ignite(_gh_dispatch, sleep=time.sleep, clock=time.time,
                      budget_s=args.retry_budget_min * 60.0)
    if args.next_depth is None:
        ap.error("--next-depth is required unless --ignite is given")
    started = args.job_started_at if args.job_started_at else time.time()
    return run_rearm(
        next_depth=args.next_depth,
        dispatch=_gh_dispatch,
        poll=_subprocess_poll,
        sleep=time.sleep,
        clock=time.time,
        deadline=started + args.job_limit_min * 60.0,
        initial_sleep_s=args.initial_sleep_sec,
        retry_budget_s=args.retry_budget_min * 60.0,
        poll_interval_s=args.poll_interval_sec,
    )


if __name__ == "__main__":
    sys.exit(main())
