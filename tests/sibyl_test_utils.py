# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Shared fixtures/helpers for the Sibyl test suite (tests/test_sibyl_*)."""

from __future__ import annotations

import json
from pathlib import Path

from sibyl.base_rates import BaseRate
from sibyl.config import QUANTILE_LEVELS

HS_RUN_ID = "hs_sibyl_test"
STANDARD_RUN_ID = "fc_sibyl_test"
Q1 = "ETH_ACE_FATALITIES_2026-08"
Q2 = "SOM_ACE_FATALITIES_2026-08"


def seed_db(tmp_path: Path, monkeypatch) -> str:
    """Create a temp Pythia DB with two eligible Sibyl questions.

    Q1 (ETH) has higher volatility than Q2 (SOM); Q1 also has a standard
    ensemble_bayesmc_v2 forecast so track divergence can be computed.
    Returns the DB URL (also exported via PYTHIA_DB_URL).
    """
    db_path = tmp_path / "sibyl_test.duckdb"
    db_url = f"duckdb:///{db_path}"
    monkeypatch.setenv("PYTHIA_DB_URL", db_url)
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)

    from pythia.db.schema import connect, ensure_schema

    ensure_schema()
    con = connect(read_only=False)
    try:
        con.execute(
            "INSERT INTO hs_runs (hs_run_id, generated_at) VALUES (?, CURRENT_TIMESTAMP)",
            [HS_RUN_ID],
        )
        for iso3, rc_score, triage_score in (("ETH", 0.8, 0.9), ("SOM", 0.2, 0.3)):
            con.execute(
                """
                INSERT INTO hs_triage (
                    run_id, iso3, hazard_code, tier, triage_score,
                    need_full_spd, regime_change_score, track
                ) VALUES (?, ?, 'ACE', 'priority', ?, TRUE, ?, 1)
                """,
                [HS_RUN_ID, iso3, triage_score, rc_score],
            )
        for qid, iso3 in ((Q1, "ETH"), (Q2, "SOM")):
            con.execute(
                """
                INSERT INTO questions (
                    question_id, hs_run_id, iso3, hazard_code, metric,
                    target_month, window_start_date, wording, status, track
                ) VALUES (?, ?, ?, 'ACE', 'FATALITIES', '2027-01',
                          DATE '2026-08-01',
                          'How many conflict fatalities per month?', 'active', 1)
                """,
                [qid, HS_RUN_ID, iso3],
            )
        # Standard-track aggregate for Q1 (6 months x 7 buckets).
        probs = [0.35, 0.25, 0.2, 0.1, 0.05, 0.03, 0.02]
        for month in range(1, 7):
            for bucket, p in enumerate(probs, start=1):
                con.execute(
                    """
                    INSERT INTO forecasts_ensemble (
                        run_id, question_id, iso3, hazard_code, metric,
                        model_name, month_index, bucket_index, probability,
                        weights_profile, created_at, status
                    ) VALUES (?, ?, 'ETH', 'ACE', 'FATALITIES',
                              'ensemble_bayesmc_v2', ?, ?, ?, 'ensemble',
                              CURRENT_TIMESTAMP, 'ok')
                    """,
                    [STANDARD_RUN_ID, Q1, month, bucket, p],
                )
    finally:
        con.close()
    return db_url


def stub_base_rate() -> BaseRate:
    """Deterministic outside-view stub (no Resolver DB / forecaster import)."""
    anchor = {0.1: 0.0, 0.25: 2.0, 0.5: 10.0, 0.75: 40.0, 0.9: 150.0, 0.95: 400.0, 0.99: 1500.0}
    return BaseRate(
        summary={"type": "conflict_trajectory", "fatalities": {"trailing_3m_avg": 10}},
        prompt_text="BASE RATE: test anchor",
        anchor_quantiles=anchor,
        framing_notes=["test framing"],
    )


def make_submit_response(quantiles: dict[float, float] | None = None) -> str:
    """A valid single-step 'submit' model response."""
    q = quantiles or {0.1: 0, 0.25: 3, 0.5: 12, 0.75: 60, 0.9: 250, 0.95: 700, 0.99: 2500}
    assert set(q) == set(QUANTILE_LEVELS)
    return json.dumps(
        {
            "action": "submit",
            "action_input": "",
            "belief_state": {
                "quantiles": {str(k): v for k, v in q.items()},
                "confidence": "medium",
                "evidence_higher": ["escalating clashes reported"],
                "evidence_lower": ["ceasefire talks ongoing"],
                "open_questions": [],
                "baserate_reconciliation": "slightly above the anchor",
                "step_rationale": "final submission",
            },
        }
    )


def make_search_response(query: str = "test query") -> str:
    """A valid 'brave_search' step response."""
    return json.dumps(
        {
            "action": "brave_search",
            "action_input": query,
            "belief_state": {
                "quantiles": {
                    "0.1": 0, "0.25": 2, "0.5": 10, "0.75": 50,
                    "0.9": 200, "0.95": 500, "0.99": 2000,
                },
                "confidence": "low",
                "evidence_higher": [],
                "evidence_lower": [],
                "open_questions": ["current intensity"],
                "baserate_reconciliation": "at the anchor",
                "step_rationale": "need recent reporting",
            },
        }
    )
