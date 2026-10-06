# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the ingest-vs-pipeline gate (scripts/ci/check_pipeline_active.py).

Pure-function tests over pipeline_in_flight — no gh CLI. The scenario
guarded: the staged pipeline forks the canonical DB at hs_submit and
re-uploads canonical days later; a weekly ingest landing inside that window
is silently discarded by the final upload.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from scripts.ci.check_pipeline_active import pipeline_in_flight

NOW = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)


def _iso(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def _final_run(hours_ago: float, conclusion: str = "success") -> dict:
    return {
        "displayTitle": "Pythia Pipeline Stage: pl_1 — fc_collect_finalize",
        "conclusion": conclusion,
        "createdAt": _iso(hours_ago),
        "event": "workflow_dispatch",
    }


def test_no_state_artifact_means_not_in_flight():
    assert pipeline_in_flight(None, [], NOW) is False


def test_recent_state_and_no_final_stage_is_in_flight():
    # hs_submit ran 3h ago; nothing has published canonical since → gate.
    # (The listing carries no run ids here, so completeness is not checked.)
    assert pipeline_in_flight(_iso(3), [], NOW) is True


def test_final_stage_after_state_means_published():
    # fc_collect_finalize succeeded AFTER the newest batch-state artifact —
    # canonical is fresh, the ingest may proceed.
    assert pipeline_in_flight(_iso(30), [_final_run(2)], NOW) is False


def test_final_stage_before_state_does_not_clear_the_gate():
    # A PREVIOUS pipeline's final stage predating this pipeline's newest
    # batch-state artifact proves nothing — still in flight.
    assert pipeline_in_flight(_iso(3), [_final_run(48)], NOW) is True


def test_failed_final_stage_does_not_clear_the_gate():
    assert pipeline_in_flight(_iso(3), [_final_run(1, conclusion="failure")], NOW) is True


def test_stale_state_artifact_expires_the_gate():
    # 80h-old state with no resolution: the pipeline is dead/stalled — the
    # weekly refresh must not be silenced forever.
    assert pipeline_in_flight(_iso(80), [], NOW) is False
    assert pipeline_in_flight(_iso(80), [], NOW, window_hours=100) is True


# ---------------------------------------------------------------------------
# Output modes: the same decision core drives two gates with OPPOSITE senses.
#   --emit proceed  ingest-structured-data.yml  proceed = NOT in flight
#   --emit active   poll_llm_batches.yml        active  = in flight
# The poller gate used to be a weaker inline shell copy that only checked the
# artifact age, so every tick for ~72h after a pipeline finished did a full
# checkout + pip install + provider poll to conclude "already completed".
# ---------------------------------------------------------------------------

import pytest

from scripts.ci import check_pipeline_active as cpa


def _live_iso(hours_ago: float) -> str:
    """A timestamp relative to the clock ``main()`` itself reads.

    The pure-function tests above pass ``now`` explicitly and so are frozen
    at :data:`NOW`. ``main()`` cannot be: it reads
    ``datetime.now(timezone.utc)`` itself, so a fixture anchored to the
    frozen NOW ages out of the 72h window as real time moves on — these
    tests passed for exactly 72 hours after they were written and then
    failed permanently, blocking an unrelated PR. What they actually guard
    is the output SENSE (active vs proceed); the window arithmetic is
    already covered above.

    Deliberately ``cpa.datetime`` rather than this module's ``datetime``:
    the fixture and the code under test must share one clock, or a test
    that moves the clock moves only one of them.
    """

    return (cpa.datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _live_final_run(hours_ago: float, conclusion: str = "success") -> dict:
    return {**_final_run(0, conclusion), "createdAt": _live_iso(hours_ago)}


_OURS = {"head_repository_id": 42, "head_branch": "main", "id": 7001}


def _stage_runs(final_hours_ago=None, include_state_run=True, event="workflow_dispatch"):
    runs = []
    if include_state_run:
        runs.append({"id": 7001, "created_at": _live_iso(3.5), "conclusion": "success",
                     "display_title": "Pythia Pipeline Stage: pl_1 — hs_finalize_fc_submit",
                     "event": "workflow_dispatch"})
    if final_hours_ago is not None:
        runs.append({"id": 7002, "created_at": _live_iso(final_hours_ago), "conclusion": "success",
                     "display_title": "Pythia Pipeline Stage: pl_1 — fc_collect_finalize",
                     "event": event})
    return runs


def _run(monkeypatch, tmp_path, argv, *, in_flight=None, gh_raises=False, force=None,
         runs=None, event="schedule", expect_code=0, total=None):
    """Drive main() with gh mocked out; return the parsed $GITHUB_OUTPUT."""

    if runs is None:
        runs = _stage_runs(final_hours_ago=None if in_flight else 1)

    def fake_gh_json(*args):
        if gh_raises:
            raise RuntimeError("gh exploded")
        path = args[-1]
        if "pythia_pipeline_stage.yml/runs" in path:
            return {"total_count": len(runs) if total is None else total, "workflow_runs": runs}
        if "artifacts" in path:
            # A batch-state artifact 3h old (inside the 72h window).
            return {"artifacts": [{"created_at": _live_iso(3), "workflow_run": _OURS}]}
        return {"id": 42}

    out = tmp_path / "gh_output"
    out.write_text("", encoding="utf-8")
    monkeypatch.setattr(cpa, "_gh_json", fake_gh_json)
    monkeypatch.setenv("GITHUB_REPOSITORY", "kwyjad/Pythia")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    monkeypatch.setenv("PYTHIA_GATE_ATTEMPTS", "2")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    if force is None:
        monkeypatch.delenv("FORCE", raising=False)
    else:
        monkeypatch.setenv("FORCE", force)

    assert cpa.main(argv, sleep=lambda _s: None) == expect_code
    got = dict(
        line.split("=", 1) for line in out.read_text(encoding="utf-8").splitlines() if line
    )
    got.pop("state", None)
    return got


@pytest.mark.parametrize(
    "argv,in_flight,expected",
    [
        # Poller: poll while a pipeline is in flight, exit early once it isn't.
        (["--emit", "active"], True, {"active": "true"}),
        (["--emit", "active"], False, {"active": "false"}),
        # Ingest: the mirror image, and the default mode.
        (["--emit", "proceed"], True, {"proceed": "false"}),
        (["--emit", "proceed"], False, {"proceed": "true"}),
        ([], True, {"proceed": "false"}),
        ([], False, {"proceed": "true"}),
    ],
)
def test_emit_modes_have_opposite_senses(monkeypatch, tmp_path, argv, in_flight, expected):
    assert _run(monkeypatch, tmp_path, argv, in_flight=in_flight) == expected


def test_a_broken_api_on_a_schedule_proceeds_and_polls(monkeypatch, tmp_path):
    """On a scheduled run "could not tell" proceeds (the 11th is the only
    refresh of the resolution sources), and the poller keeps polling."""
    assert _run(monkeypatch, tmp_path, [], gh_raises=True) == {"proceed": "true"}
    assert _run(monkeypatch, tmp_path, ["--emit", "active"], gh_raises=True) == {"active": "true"}


def test_a_broken_api_on_a_dispatch_stops_red(monkeypatch, tmp_path):
    """A skip never reports success to a person who asked for the work."""
    assert _run(monkeypatch, tmp_path, [], gh_raises=True, event="workflow_dispatch",
                expect_code=1) == {"proceed": "false"}


@pytest.mark.parametrize("argv,key", [(["--emit", "active"], "active"), ([], "proceed")])
def test_force_bypasses_the_gate_in_both_modes(monkeypatch, tmp_path, argv, key):
    assert _run(monkeypatch, tmp_path, argv, in_flight=True, force="true",
                event="workflow_dispatch") == {key: "true"}


def test_unknown_emit_mode_is_rejected():
    assert cpa.main(["--emit", "bogus"]) == 2


def test_the_gate_still_decides_correctly_a_year_from_now(monkeypatch, tmp_path):
    """The main()-driven tests must not be anchored to a wall-clock constant."""

    class _FutureDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime.now(tz) + timedelta(days=365)

    monkeypatch.setattr(cpa, "datetime", _FutureDatetime)
    assert _run(monkeypatch, tmp_path, ["--emit", "active"], in_flight=True) == {
        "active": "true"
    }
    assert _run(monkeypatch, tmp_path, ["--emit", "proceed"], in_flight=False) == {
        "proceed": "true"
    }


# --- Oct 2026: three answers, never two ------------------------------------

STATE_5_OCT = "2026-10-02T14:53:21Z"   # hs_finalize_fc_submit's batch state
STATE_RUN_5_OCT = 37021737600
NOW_5_OCT = datetime(2026, 10, 5, 13, 32, tzinfo=timezone.utc)
FINAL_2_OCT = {
    "databaseId": 37023709231, "createdAt": "2026-10-02T14:59:02Z", "conclusion": "success",
    "displayTitle": "Pythia Pipeline Stage: hs_20261002T142011 — fc_collect_finalize",
}
SUBMIT_2_OCT = {
    "databaseId": STATE_RUN_5_OCT, "createdAt": "2026-10-02T14:42:27Z", "conclusion": "success",
    "displayTitle": "Pythia Pipeline Stage: hs_20261002T142011 — hs_finalize_fc_submit",
}


def test_replay_of_5_october_13_32_an_empty_listing_is_unknown_not_in_flight():
    """The 2 October pipeline had finished; the listing came back empty and the
    old core read that as "in flight", skipping a dispatched Resolver Update."""
    assert cpa.pipeline_state(STATE_5_OCT, [], NOW_5_OCT, state_run_id=STATE_RUN_5_OCT) == cpa.UNKNOWN
    assert cpa.pipeline_state(STATE_5_OCT, None, NOW_5_OCT) == cpa.UNKNOWN


def test_replay_of_5_october_13_32_a_complete_listing_is_idle():
    assert cpa.pipeline_state(
        STATE_5_OCT, [SUBMIT_2_OCT, FINAL_2_OCT], NOW_5_OCT, state_run_id=STATE_RUN_5_OCT,
    ) == cpa.IDLE


def test_replay_of_5_october_13_32_without_the_final_stage_is_in_flight():
    assert cpa.pipeline_state(
        STATE_5_OCT, [SUBMIT_2_OCT], NOW_5_OCT, state_run_id=STATE_RUN_5_OCT,
    ) == cpa.IN_FLIGHT
    # ...but not when the API says the listing is short.
    assert cpa.pipeline_state(
        STATE_5_OCT, [SUBMIT_2_OCT], NOW_5_OCT, state_run_id=STATE_RUN_5_OCT,
        listing_complete=False,
    ) == cpa.UNKNOWN


def test_main_reads_a_partial_listing_as_unknown(monkeypatch, tmp_path):
    # The run that uploaded the batch state is absent: the listing is partial.
    partial = _stage_runs(final_hours_ago=None, include_state_run=False)
    assert _run(monkeypatch, tmp_path, [], runs=partial, event="schedule") == {"proceed": "true"}
    assert _run(monkeypatch, tmp_path, [], runs=partial, event="workflow_dispatch",
                expect_code=1) == {"proceed": "false"}
    # The API says more runs exist than were returned.
    assert _run(monkeypatch, tmp_path, [], runs=_stage_runs(), total=250,
                event="workflow_dispatch", expect_code=1) == {"proceed": "false"}


def test_decide_table():
    assert cpa.decide(cpa.IDLE, "proceed", "workflow_dispatch")[:2] == (True, 0)
    assert cpa.decide(cpa.IN_FLIGHT, "proceed", "schedule")[:2] == (False, 0)
    assert cpa.decide(cpa.UNKNOWN, "proceed", "schedule")[:2] == (True, 0)
    assert cpa.decide(cpa.UNKNOWN, "proceed", "workflow_dispatch")[:2] == (False, 1)
    assert cpa.decide(cpa.UNKNOWN, "active", "workflow_dispatch")[:2] == (True, 0)
    assert cpa.decide(cpa.IDLE, "active", "schedule")[:2] == (False, 0)


# A fork's pull-request run can upload an artifact under the same name.
def test_an_artifact_from_another_repository_is_not_ours():
    fork = {"created_at": "2026-10-02T10:00:00Z",
            "workflow_run": {"head_repository_id": 99, "head_branch": "main"}}
    ours = {"created_at": "2026-09-01T10:00:00Z", "workflow_run": _OURS}
    assert cpa.newest_trusted_artifact([fork, ours], 42) == "2026-09-01T10:00:00Z"
    assert cpa.newest_trusted_artifact([fork], 42) is None
    side = {"created_at": "2026-10-02T11:00:00Z",
            "workflow_run": {"head_repository_id": 42, "head_branch": "feature"}}
    assert cpa.newest_trusted_artifact([side, ours], 42) == "2026-09-01T10:00:00Z"
    # No repository id means no artifact can be shown to be ours.
    assert cpa.newest_trusted_artifact([ours], None) is None


def test_a_forks_final_stage_run_cannot_clear_the_gate(monkeypatch, tmp_path):
    runs = _stage_runs(final_hours_ago=1, event="pull_request")
    assert _run(monkeypatch, tmp_path, ["--emit", "active"], runs=runs) == {"active": "true"}
