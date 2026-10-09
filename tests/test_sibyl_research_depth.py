# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Research depth (Oct 2026, review Part 1): the submit gate at five
documents, a document counted once, and the depth measures a run carries."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import duckdb
import pytest

from pythia.web_research.types import EvidenceSource

import sibyl.agent as sibyl_agent
import sibyl.config as sibyl_config
import sibyl.measure as measure
import sibyl.tools as sibyl_tools
from sibyl.belief_state import empty_plan
from sibyl.cost import CostTracker
from sibyl.select_questions import SibylQuestion
from tests.sibyl_test_utils import (
    ALL_DONE_PLAN,
    HS_RUN_ID,
    Q1,
    make_actions_response,
    make_plan_submit_response,
    stub_reference,
    stub_tools,
)

TODAY = date(2026, 10, 9)


def _q():
    return SibylQuestion(
        question_id=Q1, hs_run_id=HS_RUN_ID, iso3="ETH", hazard_code="ACE",
        metric="FATALITIES", target_month="2027-01", window_start_date=date(2026, 11, 1),
        wording="How many?", volatility_score=0.8, triage_score=0.9,
    )


def _run(script, monkeypatch, *, fetch=None):
    stub_tools(monkeypatch)
    if fetch is not None:
        monkeypatch.setattr(sibyl_agent, "fetch_url", fetch)

    def fake_brave(query, as_of, **kw):
        return sibyl_tools.ToolResult(tool="brave_search", ok=True, text="r",
                                      sources=[EvidenceSource(title="t", url="https://n/x")])

    monkeypatch.setattr(sibyl_agent, "brave_search", fake_brave)
    it = iter(script)

    def model(prompt):
        return next(it), {"cost_usd": 0.1}, ""

    return sibyl_agent.run_trial(
        _q(), stub_reference(), as_of=TODAY, trial_index=0, run_id="sr",
        tracker=CostTracker(run_hard_cap_usd=100),
        forecast_months=["2026-11", "2026-12"] + ["2027-01"] * 4,
        country_name="Ethiopia", model_call=model,
    )


def _fetches(*urls):
    return make_actions_response([("fetch_url", u) for u in urls], plan=ALL_DONE_PLAN)


# --- the gate at five ------------------------------------------------------------

def test_the_submit_gate_asks_for_five_documents():
    assert sibyl_config.SUBMIT_MIN_DOCS == 5
    assert sibyl_agent.submit_gate_missing(ALL_DONE_PLAN, 5, 0) == []
    missing = sibyl_agent.submit_gate_missing(ALL_DONE_PLAN, 4, 0)
    assert missing == ["documents read: 4 of 5 (use fetch_url)"]


def test_the_evidence_gate_is_unchanged():
    # It decides whether a forecast is valid; raising it would fail questions.
    assert (sibyl_config.MIN_SEARCH_OK, sibyl_config.MIN_DOCS_READ) == (3, 2)


def test_a_submit_after_four_documents_is_refused(monkeypatch):
    script = [
        _fetches("https://a", "https://b", "https://c"),
        _fetches("https://d"),
        make_plan_submit_response(),  # refused: 4 of 5
        _fetches("https://e"),
        make_plan_submit_response(),  # accepted
    ]
    trial = _run(script, monkeypatch)
    assert trial.submitted and trial.steps_used == 5
    assert trial.belief_trace[2].gate_rejected == ["documents read: 4 of 5 (use fetch_url)"]
    assert trial.n_docs_read == 5 and not trial.submit_gate_unmet


# --- a document counts once ------------------------------------------------------

def test_a_second_read_of_the_same_url_does_not_count(monkeypatch):
    script = [_fetches("https://a", "https://a", "https://b"), _fetches("https://a"),
              make_plan_submit_response()]
    monkeypatch.setattr(sibyl_agent, "MAX_STEPS", 3)
    trial = _run(script, monkeypatch)
    assert trial.n_docs_read == 2
    assert trial.docs_read_urls == ["https://a", "https://b"]
    assert trial.n_tool_calls == 4


def test_the_same_text_under_another_url_does_not_count(monkeypatch):
    def same_text(url, as_of, **kw):
        return sibyl_tools.ToolResult(tool="fetch_url", ok=True, text=f"Content of {url}",
                                      doc_text="one and the same report", url=url)

    monkeypatch.setattr(sibyl_agent, "MAX_STEPS", 2)
    trial = _run([_fetches("https://a", "https://mirror.example/a"),
                  make_plan_submit_response()], monkeypatch, fetch=same_text)
    assert trial.n_docs_read == 1


def test_a_failed_fetch_does_not_count(monkeypatch):
    def fails(url, as_of, **kw):
        if url.endswith("bad"):
            return sibyl_tools.ToolResult(tool="fetch_url", ok=False, text="[failed]",
                                          error="http_404", status_code=404)
        return sibyl_tools.ToolResult(tool="fetch_url", ok=True, text=f"Content of {url}",
                                      doc_text=f"text of {url}", url=url)

    monkeypatch.setattr(sibyl_agent, "MAX_STEPS", 2)
    trial = _run([_fetches("https://bad", "https://good"), make_plan_submit_response()],
                 monkeypatch, fetch=fails)
    assert trial.n_docs_read == 1 and trial.docs_read_urls == ["https://good"]


def test_the_step_limit_with_the_gate_unmet_is_recorded(monkeypatch):
    monkeypatch.setattr(sibyl_agent, "MAX_STEPS", 3)
    trial = _run([make_plan_submit_response(empty_plan())] * 3, monkeypatch)
    assert trial.ok and trial.submit_gate_unmet
    assert trial.to_dict()["submit_gate_unmet"] is True


# --- the measures -----------------------------------------------------------------

def _trial(docs, steps, calls, urls=(), unmet=False):
    return SimpleNamespace(resolver_status="done", n_docs_read=docs, steps_used=steps,
                           n_tool_calls=calls, docs_read_urls=list(urls),
                           submit_gate_unmet=unmet, ledger=[])


def test_the_depth_measures_over_a_fixture_run():
    trials = [
        _trial(6, 8, 14, ["https://en.wikipedia.org/wiki/X", "https://a.org/1", "https://b.org/2",
                          "https://c.org/3", "https://d.org/4", "https://e.org/5"]),
        _trial(2, 12, 9, ["https://fr.wikipedia.org/wiki/Y", "https://a.org/9"], unmet=True),
        _trial(5, 6, 10, ["https://a.org/7"] * 5),
    ]
    m = measure.process_measures([SimpleNamespace(trials=trials, final_by_month={},
                                                  raw_month1=None, reference_month1=None)])
    assert m["median_docs_per_trial"] == 5.0
    assert m["share_trials_under_doc_gate"] == pytest.approx(1 / 3)
    assert m["steps_per_trial"] == pytest.approx(26 / 3)
    assert m["tool_calls_per_trial"] == 11.0
    assert m["share_docs_wikipedia"] == pytest.approx(2 / 13)
    assert m["n_submit_gate_unmet"] == 1


def test_an_even_number_of_trials_takes_the_middle_two():
    m = measure.process_measures([SimpleNamespace(
        trials=[_trial(2, 1, 1), _trial(7, 1, 1)], final_by_month={},
        raw_month1=None, reference_month1=None)])
    assert m["median_docs_per_trial"] == 4.5


def test_wikipedia_is_matched_on_the_host_alone():
    assert measure._is_wikipedia("https://en.wikipedia.org/wiki/Sudan")
    assert not measure._is_wikipedia("https://notwikipedia.org.example.com/x")
    assert not measure._is_wikipedia("https://example.com/wikipedia.org")


def test_a_run_with_no_trials_reads_none():
    m = measure.process_measures([])
    for key in ("median_docs_per_trial", "share_trials_under_doc_gate", "steps_per_trial",
                "tool_calls_per_trial", "share_docs_wikipedia", "n_submit_gate_unmet"):
        assert m[key] is None


# --- stage health: warn, never fail ---------------------------------------------

def _health_db(tmp_path, **depth):
    path = str(tmp_path / "h.duckdb")
    con = duckdb.connect(path)
    con.execute(
        "CREATE TABLE sibyl_runs (sibyl_run_id TEXT, hs_run_id TEXT, model TEXT, k INTEGER, "
        "aggregation TEXT, run_hard_cap_usd DOUBLE, budget_capped BOOLEAN, run_cost_usd DOUBLE, "
        "opus_cost_usd DOUBLE, brave_cost_usd DOUBLE, n_selected INTEGER, n_forecast INTEGER, "
        "n_skipped INTEGER, created_at TIMESTAMP, median_docs_per_trial DOUBLE, "
        "share_docs_wikipedia DOUBLE, n_submit_gate_unmet INTEGER)"
    )
    con.execute(
        "INSERT INTO sibyl_runs VALUES ('sr', 'hs', 'm', 3, 'lp', 60, FALSE, 10, 9, 1, 25, 25, 0, "
        "now(), ?, ?, ?)",
        [depth.get("median"), depth.get("wiki"), depth.get("unmet")],
    )
    con.close()
    return path


def _stage_health(path, tmp_path, monkeypatch):
    import sys

    from scripts.ci import stage_health

    out = tmp_path / "rep.json"
    monkeypatch.setattr(sys, "argv", ["stage_health", "--db", path, "--stage", "sibyl",
                                      "--hs-run-id", "hs", "--sibyl-run-id", "sr",
                                      "--out", str(out)])
    assert stage_health.main() == 0
    import json

    return json.loads(out.read_text())["sibyl"]


def test_a_shallow_run_is_warned_about_and_the_stage_stays_green(tmp_path, monkeypatch, capsys):
    sb = _stage_health(_health_db(tmp_path, median=3.0, wiki=0.6, unmet=4), tmp_path, monkeypatch)
    out = capsys.readouterr().out
    assert len(sb["depth_warnings"]) == 2
    assert "::warning title=Sibyl research shallow::median documents per trial 3.0" in out
    assert "60% of documents read were Wikipedia" in out


def test_a_deep_run_is_not_warned_about(tmp_path, monkeypatch, capsys):
    sb = _stage_health(_health_db(tmp_path, median=6.0, wiki=0.1, unmet=0), tmp_path, monkeypatch)
    assert sb["depth_warnings"] == []
    assert "Sibyl research shallow" not in capsys.readouterr().out


def test_an_older_run_without_the_measures_is_not_judged(tmp_path, monkeypatch):
    sb = _stage_health(_health_db(tmp_path), tmp_path, monkeypatch)
    assert sb["depth_warnings"] == []
