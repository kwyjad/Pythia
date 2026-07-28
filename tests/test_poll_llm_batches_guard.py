# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the poller's dispatch-once guard (scripts/ci/poll_llm_batches.py).

Pure-function tests over _dispatch_decision — no gh CLI, no network. The
regression pinned here: before the attempt cap existed, a failed stage was
re-dispatched every 15 minutes forever (the guard only recognized
queued/in_progress/success), and the churn eventually pushed the submit run
carrying the batch-state artifact out of the discovery window.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.ci.poll_llm_batches import _dispatch_decision

NOW = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
PID = "pl_123"
STAGE = "fc_collect_finalize"
MARKER_TITLE = f"Pythia Pipeline Stage: {PID} — {STAGE}"


def _run(status="completed", conclusion="success", created_min_ago=30, title=MARKER_TITLE):
    return {
        "displayTitle": title,
        "status": status,
        "conclusion": conclusion,
        "createdAt": (NOW - timedelta(minutes=created_min_ago)).isoformat(),
    }


def _decide(runs, max_attempts=3, min_retry_minutes=60):
    return _dispatch_decision(
        PID, STAGE, runs, NOW,
        max_attempts=max_attempts, min_retry_minutes=min_retry_minutes,
    )


def test_success_run_skips_dispatch():
    dispatch, reason = _decide([_run(conclusion="success")])
    assert not dispatch
    assert "already" in reason


def test_in_progress_run_skips_dispatch():
    dispatch, _ = _decide([_run(status="in_progress", conclusion=None)])
    assert not dispatch


def test_no_prior_runs_dispatches():
    dispatch, _ = _decide([])
    assert dispatch


def test_three_failures_stall_permanently():
    runs = [
        _run(conclusion="failure", created_min_ago=300),
        _run(conclusion="cancelled", created_min_ago=200),
        _run(conclusion="timed_out", created_min_ago=100),
    ]
    dispatch, reason = _decide(runs)
    assert not dispatch
    assert reason == "STALLED"


def test_recent_failure_cools_down_then_retries():
    fresh_fail = [_run(conclusion="failure", created_min_ago=10)]
    dispatch, reason = _decide(fresh_fail)
    assert not dispatch
    assert "cooling down" in reason

    old_fail = [_run(conclusion="failure", created_min_ago=90)]
    dispatch, reason = _decide(old_fail)
    assert dispatch
    assert "attempt 2/3" in reason


def test_other_stage_failures_do_not_count():
    other = f"Pythia Pipeline Stage: {PID} — hs_rc_collect"
    runs = [
        _run(conclusion="failure", title=other),
        _run(conclusion="failure", title=other),
        _run(conclusion="failure", title=other),
    ]
    dispatch, _ = _decide(runs)
    assert dispatch


def test_other_pipeline_failures_do_not_count():
    other = f"Pythia Pipeline Stage: pl_other — {STAGE}"
    runs = [_run(conclusion="failure", title=other)] * 3
    dispatch, _ = _decide(runs)
    assert dispatch
