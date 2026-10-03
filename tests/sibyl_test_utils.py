# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Shared fixtures/helpers for the Sibyl test suite (tests/test_sibyl_*)."""

from __future__ import annotations

import json
from pathlib import Path

from sibyl.base_rates import BaseRate

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


def disable_submit_gate(monkeypatch) -> None:
    """Accept any submit: the research gate (sibyl.agent.submit_gate_missing)
    is not what the calling test is about."""
    import sibyl.agent as sibyl_agent

    monkeypatch.setattr(sibyl_agent, "submit_gate_missing", lambda *a, **k: [])


def disable_evidence_gate(monkeypatch) -> None:
    """For tests about caps, arms or plumbing whose fake agent submits at
    step 1 without searching: such a trial has no evidence, the evidence gate
    (sibyl.config.MIN_SEARCH_OK / MIN_DOCS_READ) would correctly fail the
    question, and the submit gate would refuse the submit."""
    import sibyl.config as sibyl_config

    monkeypatch.setattr(sibyl_config, "MIN_SEARCH_OK", 0)
    monkeypatch.setattr(sibyl_config, "MIN_DOCS_READ", 0)
    disable_submit_gate(monkeypatch)


ALL_DONE_PLAN = {
    slot: {"status": "done", "finding": f"{slot} checked"}
    for slot in ("resolver", "nowcast", "drivers", "calendar", "reversion", "disconfirm")
}


def make_actions_response(actions: list, *, plan: dict | None = None, **belief_extra) -> str:
    """A step with several tool calls: actions = [(action, input_dict_or_str), ...]."""
    belief = make_belief(**belief_extra)
    if plan is not None:
        belief["plan"] = plan
    return json.dumps({
        "actions": [{"action": a, "action_input": i} for a, i in actions],
        "belief_state": belief,
    })


def make_plan_submit_response(plan: dict | None = None) -> str:
    """A submit that carries a plan (all slots done by default)."""
    return json.dumps({
        "action": "submit", "action_input": "",
        "belief_state": make_belief(plan=plan if plan is not None else ALL_DONE_PLAN),
    })


def research_script():
    """A fake agent that does real research: three searches, three reads,
    then a submit with every plan slot done. Cycles per trial."""
    return [
        make_actions_response([
            ("brave_search", {"query": "conflict latest", "lane": "news"}),
            ("brave_search", {"query": "conflict history", "lane": "reference"}),
            ("reliefweb_search", {"query": "situation report"}),
        ]),
        make_actions_response([
            ("fetch_url", {"url": "https://news.example.com/a", "extraction_request": "deaths"}),
            ("fetch_url", {"url": "https://news.example.com/b", "extraction_request": "deaths"}),
            ("fetch_url", {"url": "https://news.example.com/c", "extraction_request": "deaths"}),
        ], plan=ALL_DONE_PLAN),
        make_plan_submit_response(),
    ]


def stub_tools(monkeypatch) -> None:
    """Deterministic fetch_url and reliefweb_search for the agent."""
    import sibyl.agent as sibyl_agent
    from pythia.web_research.types import EvidenceSource
    from sibyl.tools import ToolResult

    def fake_fetch(url, as_of, **kw):
        return ToolResult(tool="fetch_url", ok=True, text=f"Content of {url}:\nclashes; 12 killed",
                          doc_text="clashes; 12 killed", url=url)

    def fake_rw(query, as_of, **kw):
        return ToolResult(tool="reliefweb_search", ok=True, text="1. Sitrep",
                          sources=[EvidenceSource(title="Sitrep", url="https://reliefweb.int/report/x")])

    monkeypatch.setattr(sibyl_agent, "fetch_url", fake_fetch)
    monkeypatch.setattr(sibyl_agent, "reliefweb_search", fake_rw)


def stub_base_rate() -> BaseRate:
    """Deterministic outside-view stub (no Resolver DB / forecaster import)."""
    anchor = {0.1: 0.0, 0.25: 2.0, 0.5: 10.0, 0.75: 40.0, 0.9: 150.0, 0.95: 400.0, 0.99: 1500.0}
    return BaseRate(
        summary={"type": "conflict_trajectory", "fatalities": {"trailing_3m_avg": 10}},
        prompt_text="BASE RATE: test anchor",
        anchor_quantiles=anchor,
        framing_notes=["test framing"],
    )


DEFAULT_M1 = {"p_zero": 0.1, "q": {0.05: 2, 0.25: 6, 0.5: 15, 0.75: 60, 0.95: 400}}
DEFAULT_M6 = {"p_zero": 0.15, "q": {0.05: 2, 0.25: 5, 0.5: 12, 0.75: 50, 0.95: 500}}


def _month(spec: dict) -> dict:
    return {
        "p_zero": spec["p_zero"],
        "quantiles_positive": {str(k): v for k, v in spec["q"].items()},
    }


def make_belief(m1: dict | None = None, m6: dict | None = None, **extra) -> dict:
    """A valid belief_state object in the two-horizon shape (Oct 2026)."""
    belief = {
        "month_1": _month(m1 or DEFAULT_M1),
        "month_6": _month(m6 or DEFAULT_M6),
        "confidence": "medium",
        "evidence_higher": ["escalating clashes reported"],
        "evidence_lower": ["ceasefire talks ongoing"],
        "open_questions": [],
        "baserate_reconciliation": "slightly above the reference",
        "step_rationale": "final submission",
    }
    belief.update(extra)
    return belief


def make_submit_response(m1: dict | None = None, m6: dict | None = None) -> str:
    """A valid single-step 'submit' model response."""
    return json.dumps({"action": "submit", "action_input": "", "belief_state": make_belief(m1, m6)})


def make_search_response(query: str = "test query") -> str:
    """A valid 'brave_search' step response."""
    return json.dumps({
        "action": "brave_search",
        "action_input": query,
        "belief_state": make_belief(
            {"p_zero": 0.2, "q": {0.05: 1, 0.25: 4, 0.5: 10, 0.75: 50, 0.95: 300}},
            confidence="low", step_rationale="need recent reporting",
        ),
    })


def stub_reference(*args, **kwargs):
    """A deterministic Reference (no Resolver DB): FATALITIES, months 1-6."""
    from sibyl.reference import Reference

    m1 = [0.2, 0.15, 0.3, 0.2, 0.1, 0.03, 0.02]
    m6 = [0.25, 0.15, 0.25, 0.2, 0.1, 0.03, 0.02]
    by_month = {m: [(1 - (m - 1) / 5) * a + ((m - 1) / 5) * b for a, b in zip(m1, m6)]
                for m in range(1, 7)}
    return Reference(by_month=by_month, source="stub", detail={},
                     history=[("2026-06", 12.0)], current_value=12.0,
                     prompt_text="REFERENCE: test stub")
