# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""The adversarial check says what a regime change is a change in (Oct 2026).

The prompt never defined a regime change or named the metric, its schema
offered "moderate" while models wrote "moderate_counter" (2 of 15 ACE checks
on 1 Oct 2026), nothing validated the verdict, and no row recorded which
prompt produced it.
"""

from __future__ import annotations

import asyncio
import json

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia import adversarial_check as ac


@pytest.mark.parametrize(
    "raw, want",
    [("moderate_counter", ("moderate_counter", None)),
     ("moderate", ("moderate_counter", "moderate")),
     ("Strong_Counter", ("strong_counter", None)),
     ("very strong", ("inconclusive", "very strong")),
     (None, ("inconclusive", None))],
)
def test_verdicts_are_held_to_the_enum(raw, want):
    assert ac.normalise_net_assessment(raw) == want


def test_the_prompt_defines_the_change_and_names_the_metric(monkeypatch):
    seen = {}

    async def _fake(spec, prompt, **kw):
        seen["prompt"] = prompt
        seen["version"] = kw.get("prompt_version")
        return json.dumps({"net_assessment": "moderate", "summary": "s"}), {}, None

    monkeypatch.setattr(ac, "call_chat_ms", _fake)
    monkeypatch.setattr(ac, "resolve_hs_model", lambda: "gemini-3.5-flash")
    out = asyncio.run(ac._synthesize_counter_evidence(
        "Sudan", "SDN", "ACE", {"direction": "up", "likelihood": 0.4, "magnitude": 0.5},
        "evidence", "hs_x",
    ))
    prompt = seen["prompt"]
    assert "departure of monthly conflict deaths (ACLED, all event types)" in prompt
    assert "armed conflict (ACE)" in prompt
    assert "strong_counter|moderate_counter|weak_counter|inconclusive" in prompt
    assert seen["version"] == ac.ADVERSARIAL_PROMPT_VERSION
    assert out["net_assessment"] == "moderate_counter"
    assert out["net_assessment_raw"] == "moderate"
    assert out["prompt_version"] == ac.ADVERSARIAL_PROMPT_VERSION


def test_the_row_carries_the_prompt_version(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{tmp_path / 'p.duckdb'}")
    from horizon_scanner.db_writer import log_hs_adversarial_checks_to_db
    from pythia.db.schema import connect

    log_hs_adversarial_checks_to_db(
        "hs_x", [{"iso3": "SDN", "hazard_code": "ACE", "net_assessment": "weak_counter",
                  "prompt_version": ac.ADVERSARIAL_PROMPT_VERSION}],
        is_test=False,
    )
    con = connect(read_only=False)
    row = con.execute("SELECT net_assessment, prompt_version FROM hs_adversarial_checks").fetchone()
    con.close()
    assert row == ("weak_counter", ac.ADVERSARIAL_PROMPT_VERSION)
