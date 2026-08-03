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


def _run(monkeypatch, tmp_path, argv, *, in_flight, gh_raises=False, force=None):
    """Drive main() with gh mocked out; return the parsed $GITHUB_OUTPUT."""

    def fake_gh_json(*args):
        if gh_raises:
            raise RuntimeError("gh exploded")
        if "api" in args:
            # A batch-state artifact 3h old (inside the 72h window).
            return {"artifacts": [{"created_at": _iso(3)}]}
        # A successful final stage AFTER the artifact clears the gate.
        return [_final_run(1)] if not in_flight else []

    out = tmp_path / "gh_output"
    out.write_text("", encoding="utf-8")
    monkeypatch.setattr(cpa, "_gh_json", fake_gh_json)
    monkeypatch.setenv("GITHUB_REPOSITORY", "kwyjad/Pythia")
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    if force is None:
        monkeypatch.delenv("FORCE", raising=False)
    else:
        monkeypatch.setenv("FORCE", force)

    assert cpa.main(argv) == 0
    return dict(
        line.split("=", 1) for line in out.read_text(encoding="utf-8").splitlines() if line
    )


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


@pytest.mark.parametrize("argv,key", [(["--emit", "active"], "active"), ([], "proceed")])
def test_both_modes_fail_open_to_true(monkeypatch, tmp_path, argv, key):
    """A broken gate must never stop the work — in EITHER direction.

    The fail-open VALUE is true for both modes, but it means different things:
    the ingest still runs, and the poller still advances a live pipeline. This
    is why the two senses cannot be collapsed into a single negation.
    """
    assert _run(monkeypatch, tmp_path, argv, in_flight=True, gh_raises=True) == {key: "true"}


@pytest.mark.parametrize("argv,key", [(["--emit", "active"], "active"), ([], "proceed")])
def test_force_bypasses_the_gate_in_both_modes(monkeypatch, tmp_path, argv, key):
    assert _run(monkeypatch, tmp_path, argv, in_flight=True, force="true") == {key: "true"}


def test_unknown_emit_mode_is_rejected():
    assert cpa.main(["--emit", "bogus"]) == 2
