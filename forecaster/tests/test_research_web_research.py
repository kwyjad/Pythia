# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

import forecaster.cli as cli  # type: ignore
from pythia.db.schema import connect, ensure_schema


@pytest.mark.asyncio
async def test_research_v2_writes_grounded_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "research.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db_path}")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_ENABLED", "1")

    con = connect(read_only=False)
    ensure_schema(con)
    con.execute(
        """
        INSERT INTO hs_triage (run_id, iso3, hazard_code, tier, triage_score, drivers_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ["hs-run", "KEN", "ACE", "priority", 0.8, json.dumps(["driver"])],
    )
    con.execute(
        """
        INSERT INTO hs_country_reports (hs_run_id, iso3, report_markdown, sources_json, grounded, structural_context, recent_signals_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "hs-run",
            "KEN",
            "# report",
            json.dumps([{"title": "hs", "url": "http://hs.com"}]),
            True,
            "HS structural",
            json.dumps(["hs-signal"]),
        ],
    )
    con.close()

    def fake_fetch_evidence_pack(
        query: str,
        purpose: str,
        run_id: str | None = None,
        question_id: str | None = None,
        hs_run_id: str | None = None,
    ):
        return {
            "query": query,
            "recency_days": 120,
            "structural_context": "struct",
            "recent_signals": ["q-signal"],
            "sources": [{"title": "q", "url": "http://question.com"}],
            "grounded": True,
        }

    async def fake_call_chat_ms(ms, prompt, **_kwargs):
        payload = {"sources": ["http://question.com", "http://hs.com"], "grounded": True}
        return json.dumps(payload), {"total_tokens": 5}, None

    captured_prompts: list[str] = []
    real_build = cli.build_research_prompt_v2

    def capture_prompt(*args, **kwargs):
        prompt_text = real_build(*args, **kwargs)
        captured_prompts.append(prompt_text)
        return prompt_text

    monkeypatch.setattr(cli, "fetch_evidence_pack", fake_fetch_evidence_pack)
    monkeypatch.setattr(cli, "call_chat_ms", fake_call_chat_ms)
    monkeypatch.setattr(cli, "build_research_prompt_v2", capture_prompt)

    question_row = {
        "question_id": "Q1",
        "hs_run_id": "hs-run",
        "iso3": "KEN",
        "hazard_code": "ACE",
        "metric": "PA",
        "wording": "Outlook?",
        "target_month": "2026-01",
    }

    await cli._run_research_for_question(run_id="fc-run", question_row=question_row)

    con_check = connect(read_only=True)
    row = con_check.execute(
        "SELECT research_json, hs_evidence_json, question_evidence_json, merged_evidence_json FROM question_research WHERE run_id = ? AND question_id = ?",
        ["fc-run", "Q1"],
    ).fetchone()
    con_check.close()

    assert row is not None
    research_obj = json.loads(row[0])
    assert research_obj.get("grounded") is True
    assert "http://question.com" in (research_obj.get("sources") or [])
    assert captured_prompts and "http://question.com" in captured_prompts[0]
    assert "http://hs.com" in captured_prompts[0]
    hs_evidence = json.loads(row[1])
    question_evidence = json.loads(row[2])
    merged_evidence = json.loads(row[3])
    assert hs_evidence.get("sources")
    assert question_evidence.get("sources")
    assert any("question.com" in src.get("url", "") for src in merged_evidence.get("sources", []))


def test_normalize_grounding_moves_model_urls_to_unverified() -> None:
    research = {"sources": ["http://model.com"], "grounded": True}
    merged_pack: dict[str, object] = {"sources": []}

    enforced = cli._normalize_and_enforce_grounding(research, merged_pack)

    assert enforced["grounded"] is False
    assert enforced["verified_sources"] == []
    assert enforced["unverified_sources"] == ["http://model.com"]
    assert enforced["_grounding_enforced"] is True


def test_should_run_research_skips_when_triage_disables_spd(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_triage(*_args, **_kwargs):
        return {"need_full_spd": False}

    monkeypatch.setattr(cli, "load_hs_triage_entry", fake_triage)

    question_row = {
        "question_id": "Q1",
        "hs_run_id": "hs-run",
        "iso3": "KEN",
        "hazard_code": "ACE",
    }

    assert cli._should_run_research("fc-run", question_row) is False
    assert enforced["_grounding_verified_sources_count"] == 0


def test_normalize_grounding_prefers_verified_pack_sources() -> None:
    research = {"sources": ["http://model.com"], "grounded": False}
    merged_pack = {
        "sources": [{"url": "http://hs.com"}, {"url": "https://question.com"}],
        "unverified_sources": [{"url": "http://lead.com"}],
    }

    enforced = cli._normalize_and_enforce_grounding(research, merged_pack)

    assert enforced["grounded"] is True
    assert enforced["verified_sources"] == ["http://hs.com", "https://question.com"]
    assert "http://lead.com" in enforced["unverified_sources"]
    assert "http://model.com" in enforced["unverified_sources"]
