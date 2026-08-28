# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the cron watchdog (scripts/ci/check_workflow_freshness.py).

Pure-function tests over ``evaluate`` — no gh CLI. The scenario guarded is the
one that actually happened: Resolver Update stopped running on 2026-07-15, the
28 Aug cron tick was never delivered, and the six weeks of missing ground
truth, scoring and calibration that followed went unreported.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from scripts.ci.check_workflow_freshness import (
    WATCHED,
    Watched,
    evaluate,
    main,
    render,
)

NOW = datetime(2026, 8, 28, 12, 0, tzinfo=timezone.utc)
MONTHLY = Watched("Resolver Update", 35, "monthly, 28th")
NIGHTLY = Watched("Hazard Backcast", 3, "nightly")


def _iso(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).isoformat()


def _run(days_ago: float, conclusion: str = "success") -> dict:
    return {"createdAt": _iso(days_ago), "conclusion": conclusion}


def test_a_recent_success_is_fresh():
    result = evaluate(MONTHLY, [_run(2)], NOW)
    assert result.stale is False
    assert result.verdict == "ok"
    assert result.age_days == pytest.approx(2.0, abs=0.01)


def test_the_incident_this_watchdog_exists_for():
    # Resolver Update's real state on 2026-08-28: last success 2026-07-15,
    # 44 days earlier, against a 35-day limit.
    result = evaluate(MONTHLY, [_run(44)], NOW)
    assert result.stale is True
    assert result.verdict == "STALE"
    assert "limit 35d" in result.detail


def test_a_workflow_that_fires_but_keeps_failing_is_stale():
    # Hazard Backcast failed 11 nights running in August while its cron fired
    # perfectly on time. Age is measured from the last SUCCESS precisely so
    # that "it ran" cannot pass for "it worked".
    runs = [_run(n, "failure") for n in range(1, 12)] + [_run(12)]
    assert evaluate(NIGHTLY, runs, NOW).stale is True

    # The same run list with a fresh success at the head is fine.
    assert evaluate(NIGHTLY, [_run(0.5)] + runs, NOW).stale is False


def test_the_newest_success_wins_regardless_of_list_order():
    runs = [_run(40), _run(1), _run(20)]
    assert evaluate(MONTHLY, runs, NOW).stale is False


def test_a_failed_lookup_is_stale_and_says_so():
    # runs=None means gh itself failed. A watchdog that cannot see must not
    # report "fine" — that is the failing-open mode this script must not have.
    result = evaluate(MONTHLY, None, NOW)
    assert result.stale is True
    assert "gh lookup failed" in result.detail


def test_no_successful_run_is_distinguished_from_a_failed_lookup():
    result = evaluate(MONTHLY, [], NOW)
    assert result.stale is True
    assert result.verdict == "NO SUCCESSFUL RUN"
    assert "gh lookup failed" not in result.detail


def test_unparseable_timestamps_do_not_crash_or_pass():
    runs = [{"createdAt": "not-a-date", "conclusion": "success"}]
    assert evaluate(MONTHLY, runs, NOW).stale is True


def test_boundary_is_inclusive_of_the_limit():
    assert evaluate(MONTHLY, [_run(35)], NOW).stale is False
    assert evaluate(MONTHLY, [_run(35.1)], NOW).stale is True


def test_render_emits_one_row_per_result():
    results = [evaluate(MONTHLY, [_run(1)], NOW), evaluate(NIGHTLY, None, NOW)]
    table = render(results)
    assert table.count("\n") == 3  # header + separator + 2 rows
    assert "Resolver Update" in table and "Hazard Backcast" in table


def test_every_workflow_watched_actually_exists():
    """A typo in WATCHED would fail the run forever with 'no successful run'."""
    import pathlib
    import re

    names = set()
    for path in pathlib.Path(".github/workflows").glob("*.y*ml"):
        text = path.read_text(encoding="utf-8")
        match = re.search(r'(?m)^name:\s*"?([^"\n]+?)"?\s*$', text)
        if match:
            names.add(match.group(1).strip())
    missing = sorted(w.name for w in WATCHED if w.name not in names)
    assert not missing, f"WATCHED names not found in .github/workflows: {missing}"


def test_the_whole_forecast_and_scoring_chain_is_watched():
    """The six-week silence spanned ingest AND the three compute workflows.

    Watching only the ingest would have reported this incident, but a future
    break anywhere else in the chain would be just as invisible.
    """
    watched = {w.name for w in WATCHED}
    for required in (
        "Resolver Update",
        "Pythia — Compute Resolutions",
        "Pythia — Compute SPD Scores",
        "Pythia — Compute Calibration Weights & Advice",
        "Pythia Pipeline Stage",
        "Publish Latest Data (Release)",
    ):
        assert required in watched, f"{required} is not watched"


def test_main_refuses_an_unknown_workflow_name(monkeypatch):
    monkeypatch.setenv("GITHUB_REPOSITORY", "kwyjad/Pythia")
    assert main(["--workflow", "No Such Workflow"]) == 2


def test_main_fails_when_the_repo_is_unset(monkeypatch):
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    assert main([]) == 1
