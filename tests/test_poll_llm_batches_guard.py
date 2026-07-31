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

import pytest

from scripts.ci import poll_llm_batches
from scripts.ci.emit_batch_state import dispatch_inputs_from_env
from scripts.ci.poll_llm_batches import _dispatch_decision, _dispatch_input_args

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


# ---------------------------------------------------------------------------
# Run-shaping input carry-forward (smoke runs must survive a stage hand-off)
# ---------------------------------------------------------------------------


def test_dispatch_inputs_are_forwarded_verbatim():
    state = {
        "dispatch_inputs": {
            "batch_providers": "google",
            "only_countries": "IRN,SOM",
            "grounding_primary": "openai",
            "disable_brave": "true",
        }
    }
    args = _dispatch_input_args(state)
    assert args == [
        "-f", "batch_providers=google",
        "-f", "only_countries=IRN,SOM",
        "-f", "grounding_primary=openai",
        "-f", "disable_brave=true",
    ]


def test_empty_input_values_still_forward_as_empty_strings():
    # An empty only_countries is meaningful: it says "full country list",
    # which is what the workflow default resolves to anyway.
    args = _dispatch_input_args({"dispatch_inputs": {"only_countries": ""}})
    assert args == ["-f", "only_countries="]


def test_state_without_dispatch_inputs_forwards_nothing():
    # Back-compat: a batch-state artifact written before this field existed
    # must dispatch exactly as it did before.
    assert _dispatch_input_args({}) == []
    assert _dispatch_input_args({"dispatch_inputs": None}) == []
    assert _dispatch_input_args({"dispatch_inputs": "nonsense"}) == []


def test_dispatch_inputs_from_env_reflects_effective_settings():
    inputs = dispatch_inputs_from_env(
        {
            "PYTHIA_BATCH_PROVIDERS": "google",
            "PYTHIA_HS_ONLY_COUNTRIES": "IRN,SOM",
            "PYTHIA_GROUNDING_PRIMARY_BACKEND": "openai",
            "BRAVE_SEARCH_API_KEY": "",
        }
    )
    assert inputs == {
        "batch_providers": "google",
        "only_countries": "IRN,SOM",
        "grounding_primary": "openai",
        # An empty Brave key IS the disable_brave input — re-supplying the
        # secret on the next stage would let a 402ing key trip the breaker.
        "disable_brave": "true",
    }


def test_dispatch_inputs_from_env_production_defaults():
    inputs = dispatch_inputs_from_env(
        {
            "PYTHIA_BATCH_PROVIDERS": "openai,anthropic,google",
            "PYTHIA_HS_ONLY_COUNTRIES": "",
            "PYTHIA_GROUNDING_PRIMARY_BACKEND": "brave",
            "BRAVE_SEARCH_API_KEY": "sk-live",
        }
    )
    assert inputs["only_countries"] == ""
    assert inputs["disable_brave"] == "false"


# ---------------------------------------------------------------------------
# Self-rescheduling chain.
#
# The regression pinned here: the */15 cron is not honoured by GitHub (observed
# firing roughly hourly), so staged pipelines sat parked between stages for
# hours. Progress now comes from the poller chaining itself. The danger of a
# chain is that it never stops, so most of these tests are about NOT re-arming.
# ---------------------------------------------------------------------------

import pytest

from scripts.ci.poll_llm_batches import _should_rearm


def _d(action, pid="pl_1"):
    return {"pipeline_id": pid, "action": action}


@pytest.mark.parametrize("action", ["waiting", "dispatched", "already running", "cooling down"])
def test_rearms_while_a_pipeline_is_in_flight(action, monkeypatch):
    monkeypatch.delenv("POLLER_CHAIN_DEPTH", raising=False)
    rearm, why = _should_rearm([_d(action)])
    assert rearm is True
    assert action in why


def test_does_not_rearm_when_stalled(monkeypatch):
    """A STALLED pipeline is terminal — chaining would retry it forever."""
    monkeypatch.delenv("POLLER_CHAIN_DEPTH", raising=False)
    rearm, why = _should_rearm([_d("STALLED")])
    assert rearm is False
    assert "no pipeline" in why


def test_does_not_rearm_when_pipeline_completed(monkeypatch):
    """The final stage emits no batch state, so its last state lingers for the
    artifact's 14-day retention. Re-arming on it would spin forever."""
    monkeypatch.delenv("POLLER_CHAIN_DEPTH", raising=False)
    assert _should_rearm([_d("already completed")])[0] is False


def test_does_not_rearm_on_empty_decisions(monkeypatch):
    monkeypatch.delenv("POLLER_CHAIN_DEPTH", raising=False)
    assert _should_rearm([])[0] is False


def test_does_not_rearm_on_failed_dispatch(monkeypatch):
    monkeypatch.delenv("POLLER_CHAIN_DEPTH", raising=False)
    assert _should_rearm([_d("dispatch failed")])[0] is False


def test_chain_cap_stops_a_runaway_chain(monkeypatch):
    monkeypatch.setenv("POLLER_CHAIN_DEPTH", "200")
    monkeypatch.setenv("POLLER_MAX_CHAIN", "200")
    rearm, why = _should_rearm([_d("waiting")])
    assert rearm is False
    assert "chain cap" in why


def test_under_the_cap_still_rearms(monkeypatch):
    monkeypatch.setenv("POLLER_CHAIN_DEPTH", "199")
    monkeypatch.setenv("POLLER_MAX_CHAIN", "200")
    assert _should_rearm([_d("waiting")])[0] is True


def test_mixed_pipelines_rearm_if_any_is_live(monkeypatch):
    monkeypatch.delenv("POLLER_CHAIN_DEPTH", raising=False)
    assert _should_rearm([_d("STALLED", "pl_a"), _d("waiting", "pl_b")])[0] is True


def test_malformed_chain_depth_does_not_crash(monkeypatch):
    monkeypatch.setenv("POLLER_CHAIN_DEPTH", "not-a-number")
    assert _should_rearm([_d("waiting")])[0] is True


def test_completed_and_running_are_distinguishable():
    """The chain keys off this split, so collapsing them would break it."""
    assert _decide([_run(conclusion="success")])[1] == "already completed"
    assert _decide([_run(status="in_progress", conclusion=None)])[1] == "already running"


# --------------------------------------------------------------------------
# Chain cap vs wait ceiling
#
# The self-dispatch chain exists to keep polling until PYTHIA_BATCH_MAX_WAIT_H
# resolves a pipeline whose batches never finish. If the cap is reached first,
# the chain dies while the batches are still legitimately in flight and the
# pipeline parks until an hourly cron tick re-ignites it.
#
# That is not hypothetical: POLLER_MAX_CHAIN was hardcoded to 200 under a
# comment claiming "~7-minute cadence, a little over 24h". 200 x 7min is 23.3h
# — already under the 24h ceiling — and the measured cadence is ~5.4 min, giving
# 18.0h. On 2026-07-30 a run with slow gpt-5.6-sol batches hit the cap at
# exactly 18.16h with rearm=false while both batches were still in_progress.
# --------------------------------------------------------------------------


def _cap_hours(monkeypatch, wait_h: str) -> float:
    monkeypatch.delenv("POLLER_MAX_CHAIN", raising=False)
    monkeypatch.setenv("PYTHIA_BATCH_MAX_WAIT_H", wait_h)
    return poll_llm_batches._max_chain() * poll_llm_batches.POLLER_TICK_MINUTES / 60.0


@pytest.mark.parametrize("wait_h", ["12", "24", "36", "48"])
def test_chain_cap_always_outlives_the_batch_wait_ceiling(monkeypatch, wait_h):
    """The invariant: cap x tick interval MUST exceed the wait ceiling."""
    covered = _cap_hours(monkeypatch, wait_h)
    assert covered > float(wait_h), (
        f"chain covers only {covered:.1f}h but PYTHIA_BATCH_MAX_WAIT_H={wait_h}h — "
        "the chain would die while batches are still in flight"
    )


def test_chain_cap_never_shorter_than_the_historical_floor(monkeypatch):
    """Deriving the cap must not shorten it for small wait ceilings."""
    monkeypatch.delenv("POLLER_MAX_CHAIN", raising=False)
    monkeypatch.setenv("PYTHIA_BATCH_MAX_WAIT_H", "1")
    assert poll_llm_batches._max_chain() >= 200


def test_explicit_poller_max_chain_still_wins(monkeypatch):
    """Operators keep a hard stop for a runaway loop."""
    monkeypatch.setenv("PYTHIA_BATCH_MAX_WAIT_H", "48")
    monkeypatch.setenv("POLLER_MAX_CHAIN", "42")
    assert poll_llm_batches._max_chain() == 42


def test_malformed_wait_ceiling_falls_back_to_24h_basis(monkeypatch):
    monkeypatch.delenv("POLLER_MAX_CHAIN", raising=False)
    monkeypatch.setenv("PYTHIA_BATCH_MAX_WAIT_H", "not-a-number")
    covered = poll_llm_batches._max_chain() * poll_llm_batches.POLLER_TICK_MINUTES / 60.0
    assert covered > 24.0
