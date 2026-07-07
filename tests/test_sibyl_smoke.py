# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl end-to-end smoke test: one question forecast with the search tool
mocked (deterministic), asserting a valid native SPD lands in the parallel
tables beside the standard track."""

from __future__ import annotations

import json

import pytest

from pythia.web_research.types import EvidencePack, EvidenceSource

import sibyl.run as sibyl_run
import sibyl.tools as sibyl_tools
from tests.sibyl_test_utils import (
    HS_RUN_ID,
    Q1,
    STANDARD_RUN_ID,
    make_search_response,
    make_submit_response,
    seed_db,
    stub_base_rate,
)

pytestmark = pytest.mark.db


@pytest.fixture()
def smoke_env(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)
    monkeypatch.setattr(sibyl_run, "load_base_rate", lambda *a, **k: stub_base_rate())

    def fake_brave(query, **kwargs):
        pack = EvidencePack(query=query, backend="brave", grounded=True)
        pack.sources = [
            EvidenceSource(
                title="Situation report",
                url="https://news.example.com/report",
                summary="Clashes intensified across the region.",
                date="2026-06-20",
            ),
        ]
        pack.debug = {
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.005,
            }
        }
        return pack

    monkeypatch.setattr(sibyl_tools, "fetch_via_brave_search", fake_brave)

    # Deterministic agent: each trial searches once, then submits.
    state = {"calls": 0}

    def fake_model_call(prompt: str):
        state["calls"] += 1
        usage = {
            "prompt_tokens": 500,
            "completion_tokens": 200,
            "total_tokens": 700,
            "cost_usd": 0.10,
        }
        if state["calls"] % 2 == 1:
            return make_search_response("Ethiopia conflict latest"), usage, ""
        return make_submit_response(), usage, ""

    return fake_model_call


def test_end_to_end_single_question(smoke_env):
    summary = sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=smoke_env)

    assert summary["n_forecast"] == 1
    assert summary["n_skipped"] == 0
    assert summary["budget_capped"] is False
    sibyl_run_id = summary["sibyl_run_id"]

    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        # --- valid native SPD in forecasts_raw (what compute_scores reads) --
        raw = con.execute(
            """
            SELECT month_index, bucket_index, probability
            FROM forecasts_raw
            WHERE question_id = ? AND model_name = 'sibyl' AND run_id = ?
            ORDER BY month_index, bucket_index
            """,
            [Q1, STANDARD_RUN_ID],
        ).fetchall()
        months = sorted({r[0] for r in raw})
        buckets = sorted({r[1] for r in raw})
        assert months == [1, 2, 3, 4, 5, 6]
        assert buckets == [1, 2, 3, 4, 5, 6, 7]  # FATALITIES bucket scheme
        for month in months:
            total = sum(r[2] for r in raw if r[0] == month)
            assert total == pytest.approx(1.0, abs=1e-6)
        assert all(r[2] >= 0.0 for r in raw)

        # --- mirrored into forecasts_ensemble with the track marker ---------
        ens = con.execute(
            """
            SELECT COUNT(*), MIN(weights_profile), MIN(iso3), MIN(metric)
            FROM forecasts_ensemble
            WHERE question_id = ? AND model_name = 'sibyl' AND run_id = ?
            """,
            [Q1, STANDARD_RUN_ID],
        ).fetchone()
        assert ens[0] == 6 * 7
        assert ens[1] == "sibyl"
        assert (ens[2], ens[3]) == ("ETH", "FATALITIES")

        # --- full provenance in sibyl_forecasts ------------------------------
        rec = con.execute(
            """
            SELECT status, k, aggregation, pooled_quantiles_json, trials_json,
                   js_divergence_vs_standard, js_divergence_inter_trial,
                   cost_usd, opus_cost_usd, brave_cost_usd, as_of
            FROM sibyl_forecasts
            WHERE sibyl_run_id = ? AND question_id = ?
            """,
            [sibyl_run_id, Q1],
        ).fetchone()
        assert rec[0] == "ok"
        assert rec[1] == 3  # K trials completed
        assert rec[2] == "linear_pool"

        pooled = json.loads(rec[3])
        assert set(pooled) == {"0.1", "0.25", "0.5", "0.75", "0.9", "0.95", "0.99"}

        trials = json.loads(rec[4])
        assert len(trials) == 3
        for trial in trials:
            assert trial["quantiles"] is not None
            steps = trial["belief_trace"]
            assert [s["action"] for s in steps] == ["brave_search", "submit"]
            assert steps[0]["belief"]["quantiles"]
            assert trial["source_urls"] == ["https://news.example.com/report"]

        # Divergences computed (identical trials -> inter-trial JSD of 0.0,
        # but present; standard track differs -> positive JSD).
        assert rec[5] is not None and rec[5] > 0.0
        assert rec[6] is not None and rec[6] == pytest.approx(0.0, abs=1e-9)

        # Costs: 6 Opus calls x $0.10 + 3 Brave queries x $0.005.
        assert rec[7] == pytest.approx(0.615, abs=1e-6)
        assert rec[8] == pytest.approx(0.6, abs=1e-6)
        assert rec[9] == pytest.approx(0.015, abs=1e-6)
        assert rec[10] is not None  # asOf persisted for deferred calibration

        # --- run-level record -------------------------------------------------
        run_row = con.execute(
            "SELECT hs_run_id, n_selected, n_forecast, budget_capped, "
            "aggregation, k FROM sibyl_runs WHERE sibyl_run_id = ?",
            [sibyl_run_id],
        ).fetchone()
        assert run_row[0] == HS_RUN_ID
        assert (run_row[1], run_row[2]) == (1, 1)
        assert run_row[3] is False

        # --- spend itemised in the existing cost ledger ----------------------
        ledger = con.execute(
            """
            SELECT provider, COUNT(*), SUM(cost_usd)
            FROM llm_calls
            WHERE phase = 'sibyl' AND question_id = ?
            GROUP BY provider ORDER BY provider
            """,
            [Q1],
        ).fetchall()
        by_provider = {r[0]: (r[1], r[2]) for r in ledger}
        assert by_provider["anthropic"][0] == 6
        assert by_provider["brave"][0] == 3
        assert by_provider["anthropic"][1] == pytest.approx(0.6, abs=1e-6)
        assert by_provider["brave"][1] == pytest.approx(0.015, abs=1e-6)
    finally:
        con.close()


def test_rerun_overwrites_native_rows_not_duplicates(smoke_env, monkeypatch):
    """DELETE-then-INSERT convention: re-running Sibyl for the same question
    must not duplicate (run_id, question_id, model_name) rows."""
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=smoke_env)
    sibyl_run.run_sibyl(HS_RUN_ID, n_questions=1, model_call=smoke_env)

    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM forecasts_raw "
            "WHERE question_id = ? AND model_name = 'sibyl'",
            [Q1],
        ).fetchone()[0]
        assert n == 6 * 7
    finally:
        con.close()
