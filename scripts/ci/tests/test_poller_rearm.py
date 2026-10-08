# Pythia / Copyright (c) 2025 Kevin Wyjad
"""The poller re-arm: a refused dispatch is retried, then held in-job.

On 7 Oct 2026 four HTTP 500s in ninety seconds ended the chain. These tests
drive ``run_rearm`` with a fake clock, so the timing is exact.
"""
from __future__ import annotations

import re
from pathlib import Path

from scripts.ci import poller_rearm as pr

REPO = Path(__file__).resolve().parents[3]


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        assert s >= 0
        self.sleeps.append(s)
        self.t += s


def _dispatcher(results):
    calls = []
    it = iter(results)

    def dispatch(depth: int) -> bool:
        calls.append(depth)
        try:
            return next(it)
        except StopIteration:
            return False

    return dispatch, calls


def test_first_dispatch_succeeds_after_the_pacing_sleep():
    fc = FakeClock()
    dispatch, calls = _dispatcher([True])
    rc = pr.run_rearm(next_depth=7, dispatch=dispatch, poll=lambda d: (True, d + 1),
                      sleep=fc.sleep, clock=fc.clock, deadline=225 * 60, log=lambda m: None)
    assert rc == 0 and calls == [7] and fc.sleeps == [300.0]


def test_four_refusals_are_survived_by_the_backoff():
    """The 7 Oct shape: four refusals. The old step gave up after them."""
    fc = FakeClock()
    dispatch, calls = _dispatcher([False, False, False, False, True])
    rc = pr.run_rearm(next_depth=3, dispatch=dispatch, poll=lambda d: (True, d + 1),
                      sleep=fc.sleep, clock=fc.clock, deadline=225 * 60, log=lambda m: None)
    assert rc == 0
    assert len(calls) == 5
    assert fc.sleeps == [300.0, 30.0, 60.0, 120.0, 240.0]


def test_backoff_never_sleeps_past_its_budget():
    fc = FakeClock()
    dispatch, _ = _dispatcher([False] * 50)
    ok = pr.dispatch_with_backoff(dispatch, 1, sleep=fc.sleep, clock=fc.clock,
                                  budget_s=600, log=lambda m: None)
    assert ok is False
    assert fc.t == 600.0
    assert max(fc.sleeps) <= 300.0


def test_refused_until_the_pipelines_finish_ends_cleanly_in_job():
    fc = FakeClock()
    dispatch, calls = _dispatcher([False] * 100)
    polls = []

    def poll(depth):
        polls.append(depth)
        return (len(polls) < 3, depth + 1)

    rc = pr.run_rearm(next_depth=5, dispatch=dispatch, poll=poll, sleep=fc.sleep,
                      clock=fc.clock, deadline=225 * 60, log=lambda m: None)
    assert rc == 0
    assert polls == [5, 6, 7]  # the chain depth still advances, so the cap binds
    assert fc.t < 225 * 60


def test_a_later_in_job_re_arm_resumes_the_chain():
    fc = FakeClock()
    # Initial window refuses everything (10 minutes); the first in-job retry works.
    results = iter([False] * 6 + [True])
    calls = []

    def dispatch(depth):
        calls.append(depth)
        return next(results, False)

    msgs = []
    rc = pr.run_rearm(next_depth=2, dispatch=dispatch, poll=lambda d: (True, d + 1),
                      sleep=fc.sleep, clock=fc.clock, deadline=225 * 60, log=msgs.append)
    assert rc == 0
    assert calls[-1] == 3
    assert any("re-armed" in m for m in msgs)


def test_still_refused_at_the_job_limit_fails_red_inside_it():
    fc = FakeClock()
    dispatch, _ = _dispatcher([False] * 10_000)
    msgs = []
    deadline = 225 * 60
    rc = pr.run_rearm(next_depth=1, dispatch=dispatch, poll=lambda d: (True, d + 1),
                      sleep=fc.sleep, clock=fc.clock, deadline=deadline, log=msgs.append)
    assert rc == 1
    assert fc.t <= deadline
    assert any("Poller chain broken" in m for m in msgs)


def test_read_outputs_parses_github_output(tmp_path):
    p = tmp_path / "o"
    p.write_text("rearm=true\nchain_depth_next=9\n")
    assert pr.read_outputs(p) == {"rearm": "true", "chain_depth_next": "9"}


def test_workflow_uses_the_helper_and_its_job_limit_covers_the_loop():
    wf = (REPO / ".github/workflows/poll_llm_batches.yml").read_text()
    step = wf[wf.index("- name: Reschedule next poll"):]
    assert "python -m scripts.ci.poller_rearm" in step
    m = re.search(r"timeout-minutes: \$\{\{ \(inputs\.canary == true \|\| inputs\.canary == 'true'\) && 30 \|\| (\d+) \}\}", wf)
    assert m, "poll job timeout expression changed"
    job_limit = int(m.group(1))
    loop = re.search(r"--job-limit-min (\d+)", step)
    assert loop and int(loop.group(1)) + 10 <= job_limit
