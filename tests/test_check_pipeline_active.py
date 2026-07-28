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
    }


def test_no_state_artifact_means_not_in_flight():
    assert pipeline_in_flight(None, [], NOW) is False


def test_recent_state_and_no_final_stage_is_in_flight():
    # hs_submit ran 3h ago; nothing has published canonical since → gate.
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
