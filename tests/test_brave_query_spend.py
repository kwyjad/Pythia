# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Brave spend in the Horizon Scanner (Oct 2026).

A grounding pack keeps the first ``max_results`` unique URLs in query
order, but every query formulation was sent anyway, so when the first
query filled the pack the second and third were paid for and dropped
(998 RC and ~650 triage queries a run). The fetch now stops once the pack
is full, and each stage's source and query counts can be set.
"""

from __future__ import annotations

import json

import pytest

from pythia.web_research.backends import brave_search as bs


def _results(prefix, n):
    return [{"url": f"https://{prefix}/{i}", "title": f"{prefix} story {i}", "description": "d"}
            for i in range(n)]


@pytest.fixture
def fake_brave(monkeypatch):
    calls = []
    pages = {}

    def _run(q, api_key, freshness, timeout_sec, count=10, search_lang=None, country=None):
        calls.append(q)
        return pages.get(q, [])[:count], 200

    monkeypatch.setenv("BRAVE_SEARCH_API_KEY", "k")
    monkeypatch.setattr(bs, "_run_single_query", _run)
    monkeypatch.setattr(bs, "_get_brave_breaker", lambda: type("B", (), {"is_tripped": lambda self: False})())
    return calls, pages


def _fetch(**kw):
    return bs.fetch_via_brave_search("q", recency_days=30, include_structural=False,
                                     timeout_sec=5, hazard_code="DR", country_name="Somalia", **kw)


def test_a_full_first_page_sends_no_more_queries(fake_brave):
    calls, pages = fake_brave
    queries = bs.build_brave_queries("DR", "Somalia")
    assert len(queries) >= 2
    pages[queries[0]] = _results("a", 10)
    pages[queries[1]] = _results("b", 10)
    pack = _fetch(max_results=10)
    assert calls == [queries[0]]
    assert [s.url for s in pack.sources] == [f"https://a/{i}" for i in range(10)]
    assert pack.debug["total_queries"] == 1
    assert pack.debug["queries_skipped_pack_full"] == len(queries) - 1
    assert pack.debug["queries_used"] == [queries[0]]


def test_a_short_first_page_is_filled_from_the_next_query(fake_brave):
    calls, pages = fake_brave
    queries = bs.build_brave_queries("DR", "Somalia")
    pages[queries[0]] = _results("a", 4)
    pages[queries[1]] = _results("a", 2) + _results("b", 10)  # two repeats
    pack = _fetch(max_results=10)
    assert calls == queries[:2]
    urls = [s.url for s in pack.sources]
    assert urls[:4] == [f"https://a/{i}" for i in range(4)]
    assert len(urls) == 10 and len(set(urls)) == 10
    assert pack.debug["results_per_query"] == [4, 10]


def test_the_pack_is_what_the_old_concatenation_kept(fake_brave):
    """Stopping early changes no pack: same URLs in the same order."""
    calls, pages = fake_brave
    queries = bs.build_brave_queries("ACE", "Sudan")
    for i, q in enumerate(queries):
        pages[q] = _results(f"q{i}", 3 + i)
    old = []
    for q in queries:
        for r in pages[q][:10]:
            if r["url"] not in old:
                old.append(r["url"])
    pack = bs.fetch_via_brave_search("q", recency_days=30, include_structural=False, timeout_sec=5,
                                     max_results=10, hazard_code="ACE", country_name="Sudan")
    assert [s.url for s in pack.sources] == old[:10]


def test_stage_limits_are_env_overridable(monkeypatch):
    for stage in bs.GROUNDING_STAGES:
        monkeypatch.delenv(f"PYTHIA_{stage}_GROUNDING_MAX_SOURCES", raising=False)
        monkeypatch.delenv(f"PYTHIA_{stage}_GROUNDING_MAX_QUERIES", raising=False)
        assert bs.stage_grounding_limits(stage) == {"max_sources": 10, "max_queries": 0}
    monkeypatch.setenv("PYTHIA_TRIAGE_GROUNDING_MAX_SOURCES", "6")
    monkeypatch.setenv("PYTHIA_TRIAGE_GROUNDING_MAX_QUERIES", "1")
    assert bs.stage_grounding_limits("triage") == {"max_sources": 6, "max_queries": 1}
    monkeypatch.setenv("PYTHIA_RC_GROUNDING_MAX_SOURCES", "nonsense")
    assert bs.stage_grounding_limits("RC")["max_sources"] == 10


def test_max_queries_caps_the_formulations(fake_brave):
    calls, pages = fake_brave
    queries = bs.build_brave_queries("DR", "Somalia")
    pack = _fetch(max_results=10, max_queries=1)
    assert calls == [queries[0]]
    assert pack.debug["total_queries"] == 1


def test_the_citation_report_reads_a_run(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    from scripts.ci import grounding_citation as gc

    con = duckdb.connect(str(tmp_path / "c.duckdb"))
    con.execute("CREATE TABLE hs_runs (hs_run_id TEXT, generated_at TIMESTAMP, is_test BOOLEAN)")
    con.execute("INSERT INTO hs_runs VALUES ('hs_1', TIMESTAMP '2026-10-13 04:00:00', FALSE),"
                " ('hs_t', TIMESTAMP '2026-10-14 04:00:00', TRUE)")
    con.execute("CREATE TABLE llm_calls (hs_run_id TEXT, call_type TEXT, provider TEXT, hazard_code TEXT,"
                " iso3 TEXT, cost_usd DOUBLE, usage_json TEXT, response_text TEXT)")
    con.execute("INSERT INTO llm_calls VALUES ('hs_1','rc_grounding','brave','GROUNDING_DR','SOM',0.015,"
                "'{\"web_search_requests\": 3}',NULL)")
    con.execute("INSERT INTO llm_calls VALUES ('hs_1','rc_pass_1','google','RC_DR_PASS_1','SOM',0.01,'{}',"
                "'evidence_refs: Somalia drought deepens across Gedo region, per OCHA')")
    con.execute("CREATE TABLE hs_hazard_tail_packs (hs_run_id TEXT, iso3 TEXT, hazard_code TEXT,"
                " query TEXT, sources_json TEXT)")
    srcs = [{"title": "Unrelated headline about markets today"}] * 6 + [
        {"title": "Somalia drought deepens across Gedo region | ReliefWeb"}]
    con.execute("INSERT INTO hs_hazard_tail_packs VALUES ('hs_1','SOM','DR','rc_grounding: x', ?)",
                [json.dumps(srcs)])
    assert gc.latest_production_runs(con, 2) == ["hs_1"]
    res = gc.measure_run(con, "hs_1")
    rc = res["stages"]["rc_grounding"]
    assert rc["brave_queries"] == 3 and rc["packs"] == 1
    assert rc["share_packs_citing"] == 1.0
    assert rc["citations_by_rank"] == {"rank_6_10": 1}
    assert "rc_grounding" in gc.render([res])
