# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""SPD prompt structured-section rendering tests.

Covers the July 2026 inject fixes:
- the tail-pack subsystem is removed (no HAZARD TAIL PACK guidance),
- CrisisWatch text renders when loaded,
- RC/triage grounding packs render via their `report_markdown` key,
- adversarial checks render at RC Level 1 (not just 2+),
- the NEED_WEB_EVIDENCE escape is absent when self-search is disabled.
"""

from __future__ import annotations

import pytest

import forecaster.prompts as prompts


QUESTION = {
    "question_id": "q-test",
    "iso3": "ETH",
    "hazard_code": "ACE",
    "metric": "PA",
    "resolution_source": "ACLED",
    "window_start_date": "2026-08-01",
    "target_month": "2027-01",
}
HISTORY = {"source": "ACLED"}
RESEARCH: dict = {"sources": []}


def _build(hs_triage_entry: dict, structured_data: dict | None = None) -> str:
    return prompts.build_spd_prompt_v2(
        question=QUESTION,
        history_summary=HISTORY,
        hs_triage_entry=hs_triage_entry,
        research_json=RESEARCH,
        structured_data=structured_data,
    )


def test_tail_pack_guidance_removed() -> None:
    prompt_text = _build({"tier": "priority", "triage_score": 0.8})
    assert "HAZARD TAIL PACK" not in prompt_text
    assert "hs_hazard_tail_pack" not in prompt_text


def test_no_need_web_evidence_escape_when_self_search_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for var in (
        "PYTHIA_FORECASTER_SELF_SEARCH",
        "PYTHIA_SPD_WEB_SEARCH_ENABLED",
        "PYTHIA_MODEL_SELF_SEARCH_ENABLED",
    ):
        monkeypatch.delenv(var, raising=False)
    prompt_text = _build({"tier": "priority", "triage_score": 0.8})
    assert "Do NOT output" in prompt_text
    assert "If you need more evidence before forecasting" not in prompt_text
    assert "WEB SEARCH: If you need to verify" not in prompt_text


def test_need_web_evidence_escape_when_self_search_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHIA_FORECASTER_SELF_SEARCH", "1")
    monkeypatch.setenv("PYTHIA_SPD_WEB_SEARCH_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_MODEL_SELF_SEARCH_ENABLED", "1")
    prompt_text = _build({"tier": "priority", "triage_score": 0.8})
    assert "NEED_WEB_EVIDENCE: <your query>" in prompt_text


def test_crisiswatch_section_renders() -> None:
    cw_text = "ICG CRISISWATCH — Ethiopia (2026-06):\nArrow: Deteriorated"
    prompt_text = _build(
        {"tier": "priority", "triage_score": 0.8},
        structured_data={"crisiswatch": cw_text},
    )
    assert "ICG CRISISWATCH — Ethiopia" in prompt_text


def test_hazard_grounding_renders_report_markdown_key() -> None:
    pack = {"report_markdown": "## Evidence\n- signal one", "sources": []}
    prompt_text = _build(
        {"tier": "priority", "triage_score": 0.8},
        structured_data={"hazard_grounding": pack},
    )
    assert "HS GROUNDING EVIDENCE:" in prompt_text
    assert "signal one" in prompt_text


def test_adversarial_check_renders_at_rc_level_1() -> None:
    check = {
        "net_assessment": "strong_counter",
        "summary": "Counter-evidence indicates de-escalation.",
        "sources": [],
    }
    triage_entry = {
        "tier": "priority",
        "triage_score": 0.8,
        "regime_change_level": 1,
        "regime_change_score": 0.05,
    }
    prompt_text = _build(triage_entry, structured_data={"adversarial_check": check})
    assert "ADVERSARIAL EVIDENCE CHECK" in prompt_text
