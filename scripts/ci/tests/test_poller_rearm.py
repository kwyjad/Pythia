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


def test_a_held_poll_shares_no_group_with_the_stages_or_the_diagnostics():
    """A poll held in-job for up to 225 minutes must not block or cancel a
    pipeline stage (pythia-resolver-db), and a diagnostic dispatch must not
    queue behind it where the next cron tick would cancel it."""
    poll = (REPO / ".github/workflows/poll_llm_batches.yml").read_text()
    stage = (REPO / ".github/workflows/pythia_pipeline_stage.yml").read_text()
    head = poll[poll.index("\nconcurrency:"):poll.index("\njobs:")]
    assert "pythia-resolver-db" not in head
    assert "cancel-in-progress: false" in head
    assert "'poll-llm-batches'" in head and "'poll-llm-batches-diagnostic'" in head
    for name in ("inputs.canary", "inputs.inspect_batches", "inputs.inspect_recent"):
        assert name in head
    assert re.search(r"concurrency:\s*\n\s*group: pythia-resolver-db", stage)
    assert "pythia-resolver-db" not in poll.split("\njobs:", 1)[1]


# ---------------------------------------------------------------------------
# Ignition from a submit stage, and the narrowed cron (2026-10-12)
# ---------------------------------------------------------------------------

def test_ignite_retries_a_refused_dispatch_at_depth_zero():
    fc = FakeClock()
    dispatch, calls = _dispatcher([False, False, True])
    rc = pr.ignite(dispatch, sleep=fc.sleep, clock=fc.clock, budget_s=600, log=lambda m: None)
    assert rc == 0 and calls == [0, 0, 0]
    assert fc.sleeps == [30.0, 60.0]


def test_ignite_never_fails_the_stage_and_names_the_recovery():
    fc = FakeClock()
    dispatch, _ = _dispatcher([])
    msgs: list[str] = []
    rc = pr.ignite(dispatch, sleep=fc.sleep, clock=fc.clock, budget_s=600, log=msgs.append)
    assert rc == 0
    assert fc.t <= 600
    assert any("Poller not ignited" in m and "gh workflow run poll_llm_batches.yml" in m for m in msgs)


def test_main_requires_a_depth_unless_igniting():
    import pytest
    with pytest.raises(SystemExit):
        pr.main([])


def _poller_crons() -> list[str]:
    wf = (REPO / ".github/workflows/poll_llm_batches.yml").read_text()
    head = wf[:wf.index("\njobs:")]
    return re.findall(r'^\s*-\s*cron:\s*"([^"]+)"', head, re.M)


def test_poller_cron_fires_only_in_the_production_window():
    """It ran four to six times a day all month to find nothing in flight. The
    production pipeline starts on the 13th (pythia_pipeline_stage.yml) and
    its three batch waits of at most 24h end it by the 17th."""
    crons = _poller_crons()
    assert crons, "keep a backstop cron: it recovers a failed ignition"
    stage = (REPO / ".github/workflows/pythia_pipeline_stage.yml").read_text()
    stage_dom = re.search(r'cron:\s*"\S+\s+\S+\s+(\d+)\s', stage)
    assert stage_dom, "pipeline stage cron not found"
    start = int(stage_dom.group(1))
    for cron in crons:
        dom = cron.split()[2]
        m = re.fullmatch(r"(\d+)-(\d+)", dom)
        assert m, f"poller cron {cron!r} must name a day-of-month range, not run all month"
        lo, hi = int(m.group(1)), int(m.group(2))
        assert lo <= start and hi - start >= 4 and hi - lo <= 6


def test_every_submit_stage_ignites_the_poller_with_retries():
    stage = (REPO / ".github/workflows/pythia_pipeline_stage.yml").read_text()
    n_ignite = stage.count("python3 -m scripts.ci.poller_rearm --ignite")
    assert n_ignite == stage.count("- name: Ignite batch poller") == 4
