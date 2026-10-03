# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Part 3 of the Oct 2026 Sibyl work: give the agent documents to read.

The reader (HTML main content, PDF page selection), the extraction step on a
cheaper model, ReliefWeb through its API, the two search lanes, several
actions in a step, the research plan and the submit gate, and the resolver
cards.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from pythia.web_research.types import EvidencePack, EvidenceSource

import sibyl.agent as sibyl_agent
import sibyl.config as sibyl_config
import sibyl.tools as sibyl_tools
from sibyl import extract as sibyl_extract
from sibyl import reader
from sibyl.belief_state import BeliefStateError, empty_plan, parse_step_response
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

FIX = Path(__file__).resolve().parent / "fixtures" / "sibyl"
TODAY = date(2026, 10, 3)
PAST = date(2026, 6, 1)


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(sibyl_agent, "log_sibyl_call", lambda **kwargs: None)
    sibyl_tools.reset_run_state()


# --- reader ---------------------------------------------------------------------

def test_html_keeps_the_main_content_and_renders_tables_as_rows():
    text = reader.html_main_text((FIX / "report.html").read_text())
    assert "Clashes continued in Amhara" in text
    assert "Region | Killed | Displaced" in text
    assert "Amhara | 1,250 | 40,000" in text
    for dropped in ("Home | About", "Site header", "Copyright footer", "Subscribe", "var x"):
        assert dropped not in text


def test_pdf_keeps_the_first_two_pages_and_the_best_scoring_ones():
    body = (FIX / "report.pdf").read_bytes()
    assert reader.is_pdf(body)
    text = reader.pdf_text(body, ["Ethiopia", "killed"], max_pages=3)
    assert "[page 1 of 6]" in text and "[page 2 of 6]" in text
    assert "1,250 people killed" in text
    assert "funding requirements" not in text


def test_a_short_pdf_is_read_whole():
    text = reader.pdf_text((FIX / "report.pdf").read_bytes(), [], max_pages=40)
    assert text.count("[page ") == 6


def test_an_unreadable_pdf_is_a_failed_read_not_a_crash():
    with pytest.raises(ValueError):
        reader.pdf_text(b"%PDF-1.4 garbage", [])


def test_documents_are_capped(monkeypatch):
    monkeypatch.setattr(sibyl_config, "DOC_MAX_CHARS", 50)
    text = reader.document_text(("<main>" + "word " * 200 + "</main>").encode())
    assert len(text) <= 50


# --- fetch_url uses the reader -----------------------------------------------------

class _Resp:
    def __init__(self, body: bytes, ctype: str, status: int = 200):
        self._body = body
        self.status_code = status
        self.headers = {"Content-Type": ctype}

    def iter_content(self, chunk_size=65536):
        yield self._body

    def close(self):
        pass


@pytest.fixture()
def web(monkeypatch):
    routes = {}
    monkeypatch.setattr(sibyl_tools, "_resolve", lambda host: ["93.184.216.34"])
    monkeypatch.setattr(sibyl_tools.requests, "get", lambda url, **kw: routes[url]())
    return routes


def test_fetch_url_reads_a_pdf(web):
    web["https://example.org/r.pdf"] = lambda: _Resp((FIX / "report.pdf").read_bytes(), "application/pdf")
    r = sibyl_tools.fetch_url("https://example.org/r.pdf", TODAY, today=TODAY, terms=["killed"])
    assert r.ok and "1,250 people killed" in r.doc_text


def test_fetch_url_reads_html_main_content(web):
    web["https://example.org/s"] = lambda: _Resp((FIX / "report.html").read_bytes(), "text/html")
    r = sibyl_tools.fetch_url("https://example.org/s", TODAY, today=TODAY)
    assert r.ok and "Amhara | 1,250 | 40,000" in r.text and "Donate" not in r.text


# --- extraction ---------------------------------------------------------------------

def test_a_short_document_is_shown_whole_at_no_cost():
    calls = []
    ex = sibyl_extract.extract("12 killed", "deaths", url="u", question="q", country="c",
                               call=lambda p, m: calls.append(p) or ("x", {}, ""))
    assert ex.text == "12 killed" and not ex.extracted and calls == []


def test_a_long_document_goes_to_the_extraction_model():
    logged = []
    seen = {}

    def fake(prompt, model_id):
        seen["prompt"], seen["model"] = prompt, model_id
        return "- 1,250 killed (Aug 2026, OCHA)", {"cost_usd": 0.002}, ""

    doc = "filler " * 2000 + "1,250 killed"
    ex = sibyl_extract.extract(doc, "monthly deaths", url="https://x", question="How many?",
                               country="Ethiopia", call=fake, log=lambda **k: logged.append(k))
    assert ex.extracted and ex.text.startswith("- 1,250 killed")
    assert ex.cost_usd == pytest.approx(0.002)
    assert "monthly deaths" in seen["prompt"]
    assert "nothing relevant" in seen["prompt"] and "WORD FOR WORD" in seen["prompt"]
    assert "haiku" in seen["model"]
    assert logged and logged[0]["model_id"] == seen["model"]


def test_a_failed_extraction_shows_the_first_characters():
    doc = "A" * 7000 + "tail"
    ex = sibyl_extract.extract(doc, "", url="u", question="q", country="c",
                               call=lambda p, m: ("", {"cost_usd": 0.0}, "HTTP 500"))
    assert not ex.extracted and ex.text == "A" * sibyl_config.EXTRACTION_SKIP_CHARS
    assert ex.error == "HTTP 500"


def test_the_extraction_role_resolves_to_a_priced_model():
    from forecaster.providers import resolve_price_per_1m

    model = sibyl_extract.extraction_model()
    assert model.startswith("claude-haiku")
    assert resolve_price_per_1m(model) is not None


# --- ReliefWeb ------------------------------------------------------------------------

def _rw_item(rid, title, url, when="2026-09-20"):
    return {"id": rid, "fields": {"id": rid, "title": title, "url_alias": url,
                                  "source": [{"shortname": "OCHA"}],
                                  "date": {"original": when}, "format": [{"name": "Situation Report"}]}}


def test_reliefweb_search_lists_reports_and_remembers_ids(monkeypatch):
    sent = {}

    def fake_post(payload):
        sent["payload"] = payload
        return {"data": [_rw_item(42, "Ethiopia sitrep", "https://reliefweb.int/report/ethiopia/sitrep")]}

    monkeypatch.setattr(sibyl_tools, "_rw_post", fake_post)
    r = sibyl_tools.reliefweb_search("Amhara clashes", TODAY, today=TODAY, country_iso3="eth")
    assert r.ok and "Ethiopia sitrep" in r.text and "OCHA" in r.text and "2026-09-20" in r.text
    conds = sent["payload"]["filter"]["conditions"]
    assert {"field": "primary_country.iso3", "value": "ETH"} in conds
    assert not any(c["field"] == "date.created" for c in conds)  # live: no date cap
    assert sibyl_tools._RW_IDS["https://reliefweb.int/report/ethiopia/sitrep"] == 42


def test_reliefweb_search_caps_dates_in_backtest(monkeypatch):
    sent = {}
    monkeypatch.setattr(sibyl_tools, "_rw_post", lambda p: sent.setdefault("p", p) and {"data": []})
    sibyl_tools.reliefweb_search("x", PAST, today=TODAY)
    conds = sent["p"]["filter"]["conditions"]
    assert any(c["field"] == "date.created" and "2026-06-01" in c["value"]["to"] for c in conds)


def test_reliefweb_failure_says_so_and_points_at_brave(monkeypatch):
    monkeypatch.delenv("RELIEFWEB_APPNAME", raising=False)
    r = sibyl_tools.reliefweb_search("x", TODAY, today=TODAY)
    assert not r.ok and "RELIEFWEB_APPNAME" in r.text and "brave_search" in r.text
    assert r.search_failed


def test_a_reliefweb_page_is_read_through_the_api(monkeypatch, web):
    sibyl_tools._RW_IDS["https://reliefweb.int/report/ethiopia/sitrep"] = 42
    sent = {}

    def fake_post(payload):
        sent["payload"] = payload
        return {"data": [{"fields": {"title": "Sitrep", "body": "Clashes in Amhara.",
                                     "file": [{"url": "https://reliefweb.int/attachments/r.pdf",
                                               "mimetype": "application/pdf"}]}}]}

    monkeypatch.setattr(sibyl_tools, "_rw_post", fake_post)
    web["https://reliefweb.int/attachments/r.pdf"] = lambda: _Resp(
        (FIX / "report.pdf").read_bytes(), "application/pdf")
    r = sibyl_tools.fetch_url("https://reliefweb.int/report/ethiopia/sitrep", TODAY, today=TODAY,
                              terms=["killed"])
    assert r.ok and "Clashes in Amhara." in r.doc_text and "1,250 people killed" in r.doc_text
    assert sent["payload"]["filter"]["conditions"][0] == {"field": "id", "value": 42}


# --- search lanes -----------------------------------------------------------------------

def test_lanes_set_the_window_and_pass_hints(monkeypatch):
    seen = []

    def fake(query, **kw):
        seen.append(kw)
        pack = EvidencePack(query=query, backend="brave", grounded=True)
        pack.sources = [EvidenceSource(title="t", url="https://n.example/x", summary="s")]
        pack.debug = {"usage": {"cost_usd": 0.005}, "status_code": 200}
        return pack

    monkeypatch.setattr(sibyl_tools, "fetch_via_brave_search", fake)
    sibyl_tools.brave_search("q", TODAY, today=TODAY)
    sibyl_tools.brave_search("q", TODAY, today=TODAY, lane="reference", language="fr", country="ML")
    assert seen[0]["freshness_override"].startswith("2026-06")  # 120 days
    assert seen[1]["freshness_override"].startswith("2016-")    # ten years
    assert seen[1]["freshness_override"].endswith("2026-10-03")
    assert seen[1]["search_lang"] == "fr" and seen[1]["country"] == "ML"
    assert "search_lang" not in seen[0]


# --- actions, plan, gate ----------------------------------------------------------------

def test_a_step_may_carry_three_calls():
    d = parse_step_response(make_actions_response([
        ("brave_search", {"query": "a", "lane": "reference"}),
        ("reliefweb_search", {"query": "b"}),
        ("fetch_url", {"url": "https://x", "extraction_request": "deaths"}),
    ]))
    assert [c.action for c in d.calls] == ["brave_search", "reliefweb_search", "fetch_url"]
    assert d.calls[0].options["lane"] == "reference"
    assert d.calls[2].action_input == "https://x"
    assert d.calls[2].options["extraction_request"] == "deaths"


def test_more_than_three_calls_are_cut_and_flagged():
    d = parse_step_response(make_actions_response([("brave_search", f"q{i}") for i in range(5)]))
    assert len(d.calls) == 3 and d.repaired


def test_a_submit_beside_tools_is_dropped():
    d = parse_step_response(make_actions_response([("brave_search", "q"), ("submit", "")]))
    assert [c.action for c in d.calls] == ["brave_search"] and d.submit_dropped


def test_a_tool_call_without_input_is_refused():
    with pytest.raises(BeliefStateError):
        parse_step_response(make_actions_response([("fetch_url", {"url": ""})]))


def test_plan_is_parsed_and_unknown_status_becomes_pending():
    plan = dict(ALL_DONE_PLAN, nowcast={"status": "maybe", "finding": "x"})
    d = parse_step_response(make_plan_submit_response(plan))
    assert d.plan_given
    assert d.belief.plan["resolver"]["status"] == "done"
    assert d.belief.plan["nowcast"]["status"] == "pending"


def test_submit_gate_names_what_is_missing():
    missing = sibyl_agent.submit_gate_missing(empty_plan(), 1, 0)
    assert len(missing) == 3
    assert any("resolver" in m for m in missing) and any("1 of 3" in m for m in missing)
    assert sibyl_agent.submit_gate_missing(ALL_DONE_PLAN, 3, 0) == []
    failed_twice = dict(ALL_DONE_PLAN, resolver={"status": "failed", "finding": ""})
    assert sibyl_agent.submit_gate_missing(failed_twice, 3, 2) == []
    assert len(sibyl_agent.submit_gate_missing(failed_twice, 3, 1)) == 1


def _q():
    return SibylQuestion(
        question_id=Q1, hs_run_id=HS_RUN_ID, iso3="ETH", hazard_code="ACE",
        metric="FATALITIES", target_month="2027-01", window_start_date=date(2026, 11, 1),
        wording="How many?", volatility_score=0.8, triage_score=0.9,
    )


def _run(script, monkeypatch, *, extraction_call=None, fetch=None):
    stub_tools(monkeypatch)
    if fetch is not None:
        monkeypatch.setattr(sibyl_agent, "fetch_url", fetch)

    def fake_brave(query, as_of, **kw):
        return sibyl_tools.ToolResult(tool="brave_search", ok=True, text="r",
                                      sources=[EvidenceSource(title="t", url="https://n/x")])

    monkeypatch.setattr(sibyl_agent, "brave_search", fake_brave)
    it = iter(script)
    prompts = []

    def model(prompt):
        prompts.append(prompt)
        return next(it), {"cost_usd": 0.1}, ""

    trial = sibyl_agent.run_trial(
        _q(), stub_reference(), as_of=TODAY, trial_index=0, run_id="sr",
        tracker=CostTracker(run_hard_cap_usd=100), forecast_months=[f"2026-{m}" for m in
                                                                     ("11", "12")] + ["2027-01"] * 4,
        country_name="Ethiopia", model_call=model, extraction_call=extraction_call,
    )
    return trial, prompts


def test_an_early_submit_is_refused_and_the_trial_goes_on(monkeypatch):
    script = [
        make_plan_submit_response(empty_plan()),       # refused: nothing done
        make_actions_response([("fetch_url", "https://a"), ("fetch_url", "https://b"),
                               ("fetch_url", "https://c")], plan=ALL_DONE_PLAN),
        make_plan_submit_response(),                    # accepted
    ]
    trial, prompts = _run(script, monkeypatch)
    assert trial.submitted and trial.steps_used == 3
    assert trial.belief_trace[0].gate_rejected
    assert "Your submit was NOT accepted" in prompts[1]
    assert trial.n_docs_read == 3


def test_the_step_limit_ends_a_trial_that_never_satisfies_the_gate(monkeypatch):
    monkeypatch.setattr(sibyl_config, "MAX_STEPS", 3)
    monkeypatch.setattr(sibyl_agent, "MAX_STEPS", 3)
    trial, _ = _run([make_plan_submit_response(empty_plan())] * 3, monkeypatch)
    assert trial.ok and trial.steps_used == 3 and trial.submitted
    assert trial.belief_trace[0].gate_rejected and trial.belief_trace[1].gate_rejected
    assert trial.belief_trace[2].gate_rejected is None


def test_a_long_document_is_extracted_inside_the_trial(monkeypatch):
    def long_fetch(url, as_of, **kw):
        doc = "filler " * 2000 + "1,250 killed"
        return sibyl_tools.ToolResult(tool="fetch_url", ok=True, text=doc, doc_text=doc, url=url)

    script = [
        make_actions_response([("fetch_url", {"url": "https://a", "extraction_request": "deaths in August"})]),
        make_plan_submit_response(),
    ]
    seen = {}

    def extraction(prompt, model_id):
        seen["prompt"] = prompt
        return "- 1,250 killed", {"cost_usd": 0.003}, ""

    monkeypatch.setattr(sibyl_agent, "submit_gate_missing", lambda *a, **k: [])
    trial, prompts = _run(script, monkeypatch, extraction_call=extraction, fetch=long_fetch)
    assert "deaths in August" in seen["prompt"]
    assert "- 1,250 killed" in prompts[1]
    assert "filler filler filler filler" not in prompts[1]
    assert trial.cost.extraction_usd == pytest.approx(0.003)


# --- prompt -----------------------------------------------------------------------------

def test_the_prompt_carries_the_card_the_plan_and_no_early_submit(monkeypatch):
    trial, prompts = _run([make_plan_submit_response(empty_plan())] * 12 + [make_plan_submit_response()],
                          monkeypatch)
    p = prompts[0]
    assert "=== HOW THIS RESOLVES ===" in p and "ALL event types" in p
    assert '"disconfirm"' in p and "reliefweb_search" in p and "reference" in p
    assert "as soon as further research would not materially change" not in p
    assert "REFERENCE: test stub" in p


@pytest.mark.parametrize("hz,metric", sorted(sibyl_config.ELIGIBLE_HAZARD_METRICS))
def test_every_eligible_class_has_a_resolver_card(hz, metric):
    card = sibyl_agent.resolver_card(hz, metric)
    assert card.startswith("How this question resolves")
