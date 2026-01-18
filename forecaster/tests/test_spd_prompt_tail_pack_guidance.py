# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import forecaster.prompts as prompts


def test_spd_prompt_includes_tail_pack_guidance() -> None:
    question = {
        "question_id": "q-test",
        "iso3": "ETH",
        "hazard_code": "ACE",
        "metric": "PA",
        "resolution_source": "ACLED",
    }
    history_summary = {"source": "ACLED"}
    hs_triage_entry = {"tier": "priority", "triage_score": 0.8}
    research_json = {"sources": []}

    prompt_text = prompts.build_spd_prompt_v2(
        question=question,
        history_summary=history_summary,
        hs_triage_entry=hs_triage_entry,
        research_json=research_json,
    )

    assert "hs_hazard_tail_pack" in prompt_text
    assert "TRIGGER" in prompt_text
    assert "DAMPENER" in prompt_text
    assert "BASELINE" in prompt_text
