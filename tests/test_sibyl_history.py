# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Part 4 of the Oct 2026 Sibyl work: nothing the agent reads is lost.

The append-only transcript, the cache breakpoints, the evidence ledger, the
size guard and the delta logging of ``llm_calls.prompt_text``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date

import pytest

import sibyl.agent as sibyl_agent
import sibyl.config as sibyl_config
from sibyl.belief_state import parse_step_response
from sibyl.cost import CostTracker
from sibyl.ledger import EvidenceLedger, parse_ledger_add
from sibyl.select_questions import SibylQuestion
from sibyl.tools import ToolResult
from sibyl.transcript import ToolOutput, Transcript, TranscriptEntry
from tests.sibyl_test_utils import (
    HS_RUN_ID,
    Q1,
    make_actions_response,
    make_plan_submit_response,
    stub_reference,
    stub_tools,
)

TODAY = date(2026, 10, 3)
MONTHS = ["2026-11", "2026-12", "2027-01", "2027-02", "2027-03", "2027-04"]
FIGURE = "Amhara: 1,250 people killed in August 2026"


def _q():
    return SibylQuestion(
        question_id=Q1, hs_run_id=HS_RUN_ID, iso3="ETH", hazard_code="ACE",
        metric="FATALITIES", target_month="2027-04", window_start_date=date(2026, 11, 1),
        wording="How many?", volatility_score=0.8, triage_score=0.9,
    )


def _with_ledger(response: str, items: list) -> str:
    obj = json.loads(response)
    obj["ledger_add"] = items
    return json.dumps(obj)


def _search(q):
    return make_actions_response([("brave_search", {"query": q})])


@pytest.fixture()
def trial_env(monkeypatch):
    """A deterministic tool layer; the document at https://doc/2 holds FIGURE."""
    stub_tools(monkeypatch)
    monkeypatch.setattr(sibyl_agent, "submit_gate_missing", lambda *a, **k: [])

    def fake_brave(query, as_of, **kw):
        return ToolResult(tool="brave_search", ok=True, text=f"results for {query}")

    def fake_fetch(url, as_of, **kw):
        body = FIGURE if url.endswith("/2") else f"page {url}"
        return ToolResult(tool="fetch_url", ok=True, text=body, doc_text=body, url=url)

    monkeypatch.setattr(sibyl_agent, "brave_search", fake_brave)
    monkeypatch.setattr(sibyl_agent, "fetch_url", fake_fetch)
    logged = []
    monkeypatch.setattr(sibyl_agent, "log_sibyl_call", lambda **kw: logged.append(kw))
    return logged


def _run(script):
    it = iter(script)
    prompts = []

    def model(prompt):
        prompts.append(prompt)
        return next(it), {"cost_usd": 0.1}, ""

    trial = sibyl_agent.run_trial(
        _q(), stub_reference(), as_of=TODAY, trial_index=0, run_id="sr",
        tracker=CostTracker(run_hard_cap_usd=100), forecast_months=MONTHS,
        country_name="Ethiopia", model_call=model,
    )
    return trial, prompts


def _six_steps():
    return [
        _search("first"),
        _with_ledger(
            make_actions_response([("fetch_url", {"url": "https://doc/2"})]),
            [{"url": "https://doc/2", "date": "2026-08-31", "tier": 2, "kind": "measurement",
              "quote": FIGURE, "direction": "higher"}],
        ),
        _search("third"),
        _search("fourth"),
        _search("fifth"),
        make_plan_submit_response(),
    ]


def _tail_start(prompt: str, step: int) -> int:
    i = prompt.rfind(f"=== STEP {step} of ")
    assert i > 0
    return i


# --- the transcript ------------------------------------------------------------

def test_each_prompt_begins_with_the_last_one_less_its_tail(trial_env):
    _, prompts = _run(_six_steps())
    assert len(prompts) == 6
    for n in range(1, 6):
        prev = prompts[n - 1]
        prefix = prev[: _tail_start(prev, n)]
        assert prompts[n].startswith(prefix), f"step {n + 1} re-rendered an earlier step"
        assert len(prompts[n]) > len(prev)


def test_a_figure_read_at_step_two_is_in_the_prompt_at_step_six(trial_env):
    _, prompts = _run(_six_steps())
    assert FIGURE not in prompts[1]  # not read yet at step 2's prompt
    for p in prompts[2:]:
        assert FIGURE in p
    assert "results for first" in prompts[5]


def test_the_model_json_is_kept_word_for_word(trial_env):
    script = _six_steps()
    _, prompts = _run(script)
    assert script[0] in prompts[5] and script[1] in prompts[5]


def test_a_refused_submit_stays_in_the_transcript(monkeypatch, trial_env):
    calls = {"n": 0}

    def gate(*a, **k):
        calls["n"] += 1
        return ["the 'resolver' slot"] if calls["n"] == 1 else []

    monkeypatch.setattr(sibyl_agent, "submit_gate_missing", gate)
    trial, prompts = _run([make_plan_submit_response(), _search("x"), make_plan_submit_response()])
    assert trial.submitted and len(prompts) == 3
    assert "Your submit was NOT accepted" in prompts[1]
    assert "Your submit was NOT accepted" in prompts[2]


# --- segments and caching ---------------------------------------------------------

def _segments(transcript_text: str, step: int = 1):
    from sibyl.belief_state import initial_belief

    return sibyl_agent.build_step_prompt(
        _q(), stub_reference(), initial_belief(None, "FATALITIES"), step=step, as_of=TODAY,
        perspective="P", forecast_months=MONTHS, transcript_text=transcript_text,
        country_name="Ethiopia", return_segments=True,
    )


def test_breakpoints_sit_on_question_trial_and_transcript():
    first = _segments("")
    assert [bp for _, bp in first] == [False, True, True, False]
    assert "=== QUESTION ===" in first[1][0] and "TRIAL PERSPECTIVE" in first[2][0]
    later = _segments("=== STEP 1: YOUR RESPONSE ===\n{}\n", step=2)
    assert [bp for _, bp in later] == [False, True, True, True, False]
    assert later[3][0].startswith("=== STEP 1") and later[4][0].startswith("=== STEP 2 of ")


def test_the_legacy_single_template_is_gone():
    assert not hasattr(sibyl_agent, "SIBYL_STEP_PROMPT_TEMPLATE")
    assert not hasattr(sibyl_agent, "SIBYL_STEP_STEP_V3")


def test_the_default_call_passes_segments(monkeypatch, trial_env):
    sent = []

    def fake_call_model(prompt, *, cache_segments=None):
        sent.append(cache_segments)
        return make_plan_submit_response(), {"cost_usd": 0.0}, ""

    monkeypatch.setattr(sibyl_agent, "_call_model", fake_call_model)
    sibyl_agent.run_trial(
        _q(), stub_reference(), as_of=TODAY, trial_index=0, run_id="sr",
        tracker=CostTracker(run_hard_cap_usd=100), forecast_months=MONTHS,
        country_name="Ethiopia",
    )
    assert sent and sum(1 for _, bp in sent[0] if bp) == 2


# --- delta logging --------------------------------------------------------------------

def test_later_steps_log_a_prefix_hash_and_the_new_tail(trial_env):
    _, prompts = _run(_six_steps())
    rows = [r for r in trial_env if r.get("provider") == "anthropic"]
    assert len(rows) == 6
    assert rows[0]["prompt_text"] == prompts[0]
    for n in range(1, 6):
        logged = rows[n]["prompt_text"]
        assert logged.startswith("[prefix sha256=")
        prefix = prompts[n - 1][: _tail_start(prompts[n - 1], n)]
        digest = hashlib.sha256(prefix.encode("utf-8")).hexdigest()
        assert f"sha256={digest} chars={len(prefix)}" in logged.splitlines()[0]
        body = logged.split("\n", 1)[1]
        assert prefix + body == prompts[n]
        assert len(logged) < len(prompts[n])


def test_prompt_for_log_stores_whole_when_the_prefix_does_not_match():
    assert sibyl_agent.prompt_for_log("abc", None) == "abc"
    assert sibyl_agent.prompt_for_log("abc", "xyz") == "abc"
    assert sibyl_agent.prompt_for_log("abcdef", "abc").endswith("]\ndef")


# --- the ledger ------------------------------------------------------------------------

def test_ledger_items_are_cleaned():
    items = parse_ledger_add({"ledger_add": [
        {"url": "u", "date": "2026-08", "tier": "2", "kind": "Measurement",
         "quote": "12 killed", "direction": "Higher"},
        {"url": "u", "tier": 9, "kind": "rumour", "figure": "about 40", "direction": "up"},
        {"url": "u", "quote": ""},
        "not an object",
    ]})
    assert len(items) == 2
    assert items[0] == {"url": "u", "date": "2026-08", "tier": 2, "kind": "measurement",
                        "quote": "12 killed", "direction": "higher"}
    assert items[1]["tier"] is None and items[1]["kind"] is None
    assert items[1]["quote"] == "about 40" and items[1]["direction"] == "neutral"


def test_a_response_without_ledger_add_still_parses():
    assert parse_step_response(make_plan_submit_response()).ledger_add == []


def test_the_ledger_numbers_items_and_drops_repeats():
    led = EvidenceLedger()
    a = led.add([{"url": "u", "quote": "12 killed", "direction": "higher"}], step=1)
    b = led.add([{"url": "u", "quote": "12  KILLED", "direction": "higher"},
                 {"url": "v", "quote": "40 displaced", "direction": "neutral"}], step=2)
    assert [x["id"] for x in a] == ["E1"] and [x["id"] for x in b] == ["E2"]
    assert led.to_list()[1]["step"] == 2


def test_the_trial_keeps_its_ledger_and_shows_the_ids(trial_env):
    trial, prompts = _run(_six_steps())
    assert len(trial.ledger) == 1 and trial.ledger[0]["id"] == "E1"
    assert trial.to_dict()["ledger"][0]["quote"] == FIGURE
    assert "[E1] tier 2, measurement, 2026-08-31, higher" in prompts[2]


# --- size guard ----------------------------------------------------------------------------

def test_the_guard_stubs_the_oldest_results_and_keeps_the_url():
    tr = Transcript()
    for step in (1, 2, 3):
        tr.append(TranscriptEntry(step=step, response="{}", ledger_block="-",
                                  outputs=[ToolOutput("fetch_url", f"https://d/{step}", "x" * 1000,
                                                      url=f"https://d/{step}")]))
    assert tr.n_stubbed == 0
    tr._guard(max_chars=2000)
    assert tr.n_stubbed >= 1 and tr.size() <= 2000
    first = tr.entries[0].text
    assert "https://d/1" in first and "x" * 1000 not in first
    assert "x" * 1000 in tr.entries[-1].text


def test_the_guard_inside_a_trial(monkeypatch, trial_env):
    monkeypatch.setattr(sibyl_config, "TRANSCRIPT_MAX_CHARS", 3000)

    def big_fetch(url, as_of, **kw):
        body = "y" * 1500
        return ToolResult(tool="fetch_url", ok=True, text=body, doc_text=body, url=url)

    monkeypatch.setattr(sibyl_agent, "fetch_url", big_fetch)
    script = [make_actions_response([("fetch_url", {"url": f"https://big/{i}"})]) for i in range(3)]
    trial, prompts = _run(script + [make_plan_submit_response()])
    assert trial.n_transcript_stubbed >= 1
    assert "https://big/0" in prompts[-1]
    assert trial.to_dict()["transcript_stubbed"] == trial.n_transcript_stubbed
