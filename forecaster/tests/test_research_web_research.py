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
        INSERT INTO hs_country_reports (hs_run_id, iso3, report_markdown, sources_json)
        VALUES (?, ?, ?, ?)
        """,
        ["hs-run", "KEN", "# report", json.dumps([{"title": "hs", "url": "http://hs.com"}])],
    )
    con.close()

    def fake_fetch_evidence_pack(query: str, purpose: str, run_id: str | None = None, question_id: str | None = None):
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

    monkeypatch.setattr(cli, "fetch_evidence_pack", fake_fetch_evidence_pack)
    monkeypatch.setattr(cli, "call_chat_ms", fake_call_chat_ms)

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
        "SELECT research_json FROM question_research WHERE run_id = ? AND question_id = ?",
        ["fc-run", "Q1"],
    ).fetchone()
    con_check.close()

    assert row is not None
    research_obj = json.loads(row[0])
    assert research_obj.get("grounded") is True
    assert "http://question.com" in (research_obj.get("sources") or [])
