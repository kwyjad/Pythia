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
    research_script,
    seed_db,
    stub_reference,
    stub_tools,
)

pytestmark = pytest.mark.db


@pytest.fixture()
def smoke_env(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)
    monkeypatch.setattr(sibyl_run, "build_reference", stub_reference)

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

    stub_tools(monkeypatch)

    # Deterministic agent: each trial runs three searches (two Brave, one
    # ReliefWeb), reads three documents, then submits with its plan done.
    # Trials run on worker threads (Oct 2026), so the script position is kept
    # per trial, keyed by the lane its prompt names.
    import re
    import threading

    script = research_script()
    state: dict = {}
    lock = threading.Lock()

    def fake_model_call(prompt: str):
        usage = {
            "prompt_tokens": 500,
            "completion_tokens": 200,
            "total_tokens": 700,
            "cost_usd": 0.10,
        }
        lane = re.search(r"Lane ([A-E]),", prompt).group(1)
        with lock:
            n = state.get(lane, 0)
            state[lane] = n + 1
        return script[n % len(script)], usage, ""

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
        raw_rows = raw = con.execute(
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
        assert rec[2] == "linear_pool_by_month"

        pooled = json.loads(rec[3])
        assert set(pooled) == {"0.1", "0.25", "0.5", "0.75", "0.9", "0.95", "0.99"}

        trials = json.loads(rec[4])
        assert len(trials) == 3
        for trial in trials:
            assert trial["quantiles"] is not None
            steps = trial["belief_trace"]
            assert [s["action"] for s in steps] == ["brave_search", "fetch_url", "submit"]
            assert [c["action"] for c in steps[0]["calls"]] == [
                "brave_search", "brave_search", "reliefweb_search"]
            assert steps[0]["calls"][1]["options"]["lane"] == "reference"
            assert (trial["n_search_ok"], trial["n_docs_read"]) == (3, 3)
            assert steps[2]["belief"]["plan"]["disconfirm"]["status"] == "done"
            assert steps[0]["belief"]["month_1"]["quantiles_positive"]
            assert trial["month_1"]["p_zero"] == pytest.approx(0.1)
            assert trial["month_6"]["p_zero"] == pytest.approx(0.15)
            assert "https://news.example.com/report" in trial["source_urls"]

        # Divergences computed (identical trials -> inter-trial JSD of 0.0,
        # but present; standard track differs -> positive JSD).
        assert rec[5] is not None and rec[5] > 0.0
        assert rec[6] is not None and rec[6] == pytest.approx(0.0, abs=1e-9)

        # Costs: 9 Opus calls x $0.10 + 6 Brave queries x $0.005.
        assert rec[7] == pytest.approx(0.93, abs=1e-6)
        assert rec[8] == pytest.approx(0.9, abs=1e-6)
        assert rec[9] == pytest.approx(0.03, abs=1e-6)
        assert rec[10] is not None  # asOf persisted for deferred calibration

        # --- reference, raw pool and published vectors (Oct 2026) -----------
        ref_j, raw_j, fin_j = con.execute(
            "SELECT reference_json, raw_by_month_json, final_by_month_json "
            "FROM sibyl_forecasts WHERE sibyl_run_id = ? AND question_id = ?",
            [sibyl_run_id, Q1],
        ).fetchone()
        ref = json.loads(ref_j)
        raw = json.loads(raw_j)
        fin = json.loads(fin_j)
        assert ref["source"] == "stub" and ref["weight"] == pytest.approx(0.5)
        assert set(raw["quantiles"]["1"]) >= {"0.05", "0.5", "0.95"}
        for m in ("1", "6"):
            stated = [0.5 * a + 0.5 * b for a, b in zip(ref["by_month"][m], raw["vectors"][m])]
            assert fin[m] == pytest.approx(stated, abs=0.01)  # before/after the floor
        by_month_written = {}
        for month, bucket, p in raw_rows:
            by_month_written.setdefault(month, []).append(p)
        assert by_month_written[1] != pytest.approx(by_month_written[6])

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
        assert by_provider["anthropic"][0] == 9
        assert by_provider["brave"][0] == 6
        assert by_provider["anthropic"][1] == pytest.approx(0.9, abs=1e-6)
        assert by_provider["brave"][1] == pytest.approx(0.03, abs=1e-6)
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
