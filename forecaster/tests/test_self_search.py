# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import asyncio

import pytest

duckdb = pytest.importorskip("duckdb")

import forecaster.cli as cli  # type: ignore
from forecaster.self_search import extract_self_search_query
from forecaster.providers import ModelSpec


def test_extract_self_search_query_basic() -> None:
    text = "NEED_WEB_EVIDENCE: flood risk Niger"
    assert extract_self_search_query(text) == "flood risk Niger"


@pytest.mark.asyncio
async def test_spd_self_search_triggers_follow_up(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    def fake_call_chat_ms(ms, prompt, **_kwargs):
        prompts.append(prompt)
        if len(prompts) == 1:
            return (
                "NEED_WEB_EVIDENCE: flood risk",
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                None,
            )
        return (
            '{"spds": {"month_1": {"probs": [0.2,0.2,0.2,0.2,0.2]}}}',
            {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            None,
        )

    monkeypatch.setattr(cli, "call_chat_ms", fake_call_chat_ms)
    monkeypatch.setattr(cli, "run_self_search", lambda *args, **kwargs: {"sources": [{"url": "http://example.com"}]})
    monkeypatch.setenv("PYTHIA_FORECASTER_SELF_SEARCH", "1")
    monkeypatch.setenv("PYTHIA_SPD_WEB_SEARCH_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_FORECASTER_SELF_SEARCH_MAX_CALLS_PER_MODEL", "1")

    text, usage, error, _ = await cli._call_spd_model_for_spec(
        ModelSpec(name="Test", provider="google", model_id="gpt-test", active=True),
        "PROMPT",
        run_id="run1",
        question_id="Q1",
        iso3="KEN",
        hazard_code="FL",
    )

    assert error is None
    assert "spds" in text
    assert usage.get("self_search", {}).get("attempted") is True
    assert usage.get("self_search", {}).get("succeeded") is True
    assert any("SELF-SEARCH RESULTS" in p for p in prompts)


@pytest.mark.asyncio
async def test_spd_self_search_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_call_chat_ms(ms, prompt, **_kwargs):
        return "NEED_WEB_EVIDENCE: more info", {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}, None

    monkeypatch.setattr(cli, "call_chat_ms", fake_call_chat_ms)
    monkeypatch.setattr(cli, "run_self_search", lambda *args, **kwargs: {"sources": []})
    monkeypatch.setenv("PYTHIA_FORECASTER_SELF_SEARCH", "0")
    monkeypatch.setenv("PYTHIA_SPD_WEB_SEARCH_ENABLED", "1")

    text, usage, error, _ = await cli._call_spd_model_for_spec(
        ModelSpec(name="Test", provider="google", model_id="gpt-test", active=True),
        "PROMPT",
        run_id="run2",
        question_id="Q2",
        iso3="KEN",
        hazard_code="FL",
    )

    assert "NEED_WEB_EVIDENCE" in text
    assert usage.get("self_search", {}).get("reason") == "self_search_disabled"
    assert error == "self_search_disabled"
