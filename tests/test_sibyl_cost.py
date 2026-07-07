# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl cost accounting and the hard run budget cut-off.

The cap must fire at question/trial boundaries (never mid-unit), remaining
questions must be marked ``skipped: run budget cap``, the run-level
``budget_capped`` flag must be set, and completed work must be persisted.
"""

from __future__ import annotations

import json

import pytest

import sibyl.agent as sibyl_agent
import sibyl.run as sibyl_run
from sibyl.cost import COST_KIND_BRAVE, COST_KIND_OPUS, CostBreakdown, CostTracker
from tests.sibyl_test_utils import (
    HS_RUN_ID,
    Q1,
    Q2,
    make_submit_response,
    seed_db,
    stub_base_rate,
)

pytestmark = pytest.mark.db


# --- CostTracker unit behaviour ----------------------------------------------

def test_tracker_accounts_per_question_and_per_kind():
    tracker = CostTracker(run_hard_cap_usd=100.0, budget_usd_per_question=None)
    tracker.add("q1", COST_KIND_OPUS, 1.5)
    tracker.add("q1", COST_KIND_BRAVE, 0.01)
    tracker.add("q2", COST_KIND_OPUS, 2.0)

    assert tracker.run_cost_usd == pytest.approx(3.51)
    q1 = tracker.question_breakdown("q1")
    assert q1.opus_usd == pytest.approx(1.5)
    assert q1.brave_usd == pytest.approx(0.01)
    assert tracker.question_cost_usd("q2") == pytest.approx(2.0)
    run = tracker.run_breakdown()
    assert run.opus_usd == pytest.approx(3.5)
    assert run.brave_usd == pytest.approx(0.01)


def test_hard_cap_boundary_semantics():
    tracker = CostTracker(run_hard_cap_usd=1.0)
    tracker.add("q1", COST_KIND_OPUS, 0.99)
    assert tracker.hard_cap_reached() is False  # next unit may still start
    tracker.add("q1", COST_KIND_OPUS, 0.02)  # in-flight unit overshoots
    assert tracker.hard_cap_reached() is True


def test_per_question_cap_optional_guard():
    unlimited = CostTracker(run_hard_cap_usd=100.0, budget_usd_per_question=None)
    unlimited.add("q1", COST_KIND_OPUS, 50.0)
    assert unlimited.question_cap_reached("q1") is False

    guarded = CostTracker(run_hard_cap_usd=100.0, budget_usd_per_question=2.0)
    guarded.add("q1", COST_KIND_OPUS, 2.5)
    assert guarded.question_cap_reached("q1") is True
    assert guarded.question_cap_reached("q2") is False


def test_negative_costs_never_reduce_totals():
    tracker = CostTracker(run_hard_cap_usd=10.0)
    tracker.add("q1", COST_KIND_OPUS, -5.0)
    assert tracker.run_cost_usd == 0.0


def test_breakdown_dict_shape():
    b = CostBreakdown(opus_usd=1.234567, brave_usd=0.005)
    d = b.to_dict()
    assert d["total_usd"] == pytest.approx(1.239567)
    assert set(d) == {"opus_usd", "brave_usd", "total_usd"}


# --- Hard cut-off end-to-end ----------------------------------------------------

def test_run_hard_cap_fires_at_boundary_and_persists_completed_work(
    tmp_path, monkeypatch
):
    """With a $1 cap and $0.60 per model call: Q1's first trial completes
    (0.60 < cap), its second completes (in-flight overshoot to 1.20), the
    third never starts; Q1's pooled forecast from the two completed trials
    is persisted; Q2 never starts and is marked 'skipped: run budget cap';
    the run record carries budget_capped=TRUE."""
    seed_db(tmp_path, monkeypatch)

    calls = {"n": 0}

    def fake_model_call(prompt: str):
        calls["n"] += 1
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "cost_usd": 0.6,
        }
        return make_submit_response(), usage, ""

    monkeypatch.setattr(sibyl_run, "load_base_rate",
                        lambda *a, **k: stub_base_rate())
    monkeypatch.setattr(
        sibyl_run, "CostTracker",
        lambda *a, **k: CostTracker(run_hard_cap_usd=1.0,
                                    budget_usd_per_question=None),
    )
    # Ledger writes are exercised in the smoke test; no-op here for speed.
    monkeypatch.setattr(sibyl_agent, "log_sibyl_call", lambda **kwargs: None)

    summary = sibyl_run.run_sibyl(HS_RUN_ID, n_questions=2,
                                  model_call=fake_model_call)

    assert summary["budget_capped"] is True
    assert summary["n_forecast"] == 1
    assert summary["n_skipped"] == 1
    assert summary["run_cost_usd"] == pytest.approx(1.2)
    assert calls["n"] == 2  # trial 3 of Q1 and all of Q2 never started

    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        rows = con.execute(
            "SELECT question_id, status, skip_reason, k FROM sibyl_forecasts "
            "ORDER BY question_id"
        ).fetchall()
        by_qid = {r[0]: r for r in rows}
        assert by_qid[Q1][1] == "ok"
        assert by_qid[Q1][3] == 2  # pooled from the two completed trials
        assert by_qid[Q2][1] == "skipped"
        assert by_qid[Q2][2] == "run budget cap"

        # Completed work persisted in the native format.
        n_spd = con.execute(
            "SELECT COUNT(*) FROM forecasts_raw "
            "WHERE question_id = ? AND model_name = 'sibyl'",
            [Q1],
        ).fetchone()[0]
        assert n_spd == 6 * 7  # 6 months x 7 FATALITIES buckets

        capped = con.execute(
            "SELECT budget_capped, n_forecast, n_skipped FROM sibyl_runs"
        ).fetchone()
        assert capped[0] is True
        assert (capped[1], capped[2]) == (1, 1)
    finally:
        con.close()


def test_run_without_cap_pressure_is_not_flagged(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)

    def cheap_model_call(prompt: str):
        return make_submit_response(), {"cost_usd": 0.01}, ""

    monkeypatch.setattr(sibyl_run, "load_base_rate",
                        lambda *a, **k: stub_base_rate())
    monkeypatch.setattr(sibyl_agent, "log_sibyl_call", lambda **kwargs: None)

    summary = sibyl_run.run_sibyl(HS_RUN_ID, n_questions=2,
                                  model_call=cheap_model_call)
    assert summary["budget_capped"] is False
    assert summary["n_forecast"] == 2
    assert summary["n_skipped"] == 0
