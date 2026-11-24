# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The Prompt Cache and Adversarial Sync health lines (Nov 2026)."""

from __future__ import annotations

import pytest

from scripts.debug_bundle import anomalies
from scripts.debug_bundle import health_extra as he


def _row(provider, prompt, read=0, cached=0, enabled="1"):
    return {"provider": provider, "prompt_tokens": prompt, "cache_read_input_tokens": read,
            "cached_tokens": cached, "prompt_cache_enabled": enabled}


def test_the_thirteen_november_cache_figures_warn():
    # 13 Nov 2026: Google 0.0% of 6.1M prompt tokens, OpenAI 0.1%, Anthropic well cached.
    rows = [_row("google", 6_100_000), _row("openai", 2_000_000, cached=2_000),
            _row("anthropic", 3_000_000, read=1_200_000)]
    out = he.prompt_cache_check(rows)
    assert out["status"] == "WARN"
    assert "google 0.0%" in out["detail"] and "openai 0.1%" in out["detail"]
    assert "anthropic" not in out["detail"].split("(")[0]


def test_small_or_uncached_providers_do_not_warn():
    rows = [_row("google", 900_000), _row("openai", 5_000_000, enabled="0"),
            _row("anthropic", 3_000_000, read=600_000)]
    assert he.prompt_cache_check(rows)["status"] == "OK"
    assert he.prompt_cache_check([])["status"] == "N/A"


def _counts(**kw):
    base = dict(rc1_spd_questions={"Q1": "SOM|ACE", "Q2": "SDN|ACE", "Q3": "PHL|TC"},
                generated_pairs={"SOM|ACE", "SDN|ACE"}, injected={"Q1"}, withheld={"Q2"},
                spd_question_pairs={"Q1": "SOM|ACE", "Q2": "SDN|ACE", "Q3": "PHL|TC", "Q4": "AFG|FL"})
    base.update(kw)
    return he.adversarial_counts(**base)


def test_adversarial_sync_is_ok_when_every_check_is_shown_or_withheld():
    c = _counts()
    assert c["rc1_plus_with_a_check"] == 2 and c["withheld"] == 1
    assert he.adversarial_check(c)["status"] == "OK"


def test_a_check_neither_shown_nor_withheld_fails():
    c = _counts(withheld=set())
    assert c["generated_not_injected_not_withheld"] == ["Q2"]
    out = he.adversarial_check(c)
    assert out["status"] == "FAIL" and "Q2" in out["detail"]


def test_a_block_with_no_check_behind_it_fails():
    c = _counts(injected={"Q1", "Q4"})
    assert c["injected_without_check"] == ["Q4"]
    assert he.adversarial_check(c)["status"] == "FAIL"


def test_a_run_with_no_rc1_questions_is_na():
    c = he.adversarial_counts(rc1_spd_questions={}, generated_pairs=set(), injected=set(),
                              withheld=set(), spd_question_pairs={})
    assert he.adversarial_check(c)["status"] == "N/A"


def test_collect_adversarial_reads_the_db():
    duckdb = pytest.importorskip("duckdb")
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE hs_adversarial_checks (hs_run_id TEXT, iso3 TEXT, hazard_code TEXT)")
    con.execute("CREATE TABLE questions (question_id TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT)")
    con.execute("CREATE TABLE hs_triage (hs_run_id TEXT, iso3 TEXT, hazard_code TEXT, regime_change_level INT)")
    con.execute("CREATE TABLE forecasts_ensemble (run_id TEXT, question_id TEXT, adversarial_arm TEXT)")
    con.execute("CREATE TABLE llm_calls (run_id TEXT, question_id TEXT, phase TEXT, prompt_text TEXT)")
    con.execute("INSERT INTO hs_adversarial_checks VALUES ('hs', 'SOM', 'ACE'), ('hs', 'SDN', 'ACE')")
    for q, iso, hz, metric, lvl, arm in [("Q1", "SOM", "ACE", "FATALITIES", 2, "shown"),
                                         ("Q2", "SDN", "ACE", "PA", 1, "withheld"),
                                         ("Q3", "PHL", "TC", "PA", 0, "n/a"),
                                         ("Q4", "SOM", "ACE", "EVENT_OCCURRENCE", 2, "n/a")]:
        con.execute("INSERT INTO questions VALUES (?, ?, ?, ?)", [q, iso, hz, metric])
        con.execute("INSERT INTO hs_triage VALUES ('hs', ?, ?, ?)", [iso, hz, lvl])
        con.execute("INSERT INTO forecasts_ensemble VALUES ('fc', ?, ?)", [q, arm])
    con.execute("INSERT INTO llm_calls VALUES ('fc', 'Q1', 'spd_v2', 'x ADVERSARIAL EVIDENCE CHECK y')")
    c = he.collect_adversarial(con, "hs", "fc")
    assert c["questions_rc1_plus"] == 2 and c["injected"] == 1 and c["withheld"] == 1
    assert he.adversarial_check(c)["status"] == "OK"


def test_na_lines_raise_no_anomaly():
    checks = [{"subsystem": "Research Grounding", "status": "N/A", "detail": "retriever disabled"},
              {"subsystem": "Adversarial Sync", "status": "FAIL", "detail": "1 generated, not injected"}]
    found = anomalies.build(health_checks=checks)
    subsystems = {a.get("subsystem") for a in found}
    assert "Research Grounding" not in subsystems and "Adversarial Sync" in subsystems
