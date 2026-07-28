# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The exact prompt sent to an SPD member must reach llm_calls.prompt_text.

_call_spd_model_for_spec can send a prompt that differs from the base prompt
(appended self-search evidence, or the NEED_WEB_EVIDENCE retry instruction).
The true sent prompt travels via usage["sent_prompt_text"], which
_call_spd_members_v2 pops into raw_call_entry["sent_prompt"] so the multi-KB
text never lands in usage_json.

NOTE: tests are synchronous and wrap the async entry points in asyncio.run()
— forecaster-ci.yml installs plain pytest without pytest-asyncio (same
convention as test_spd.py).
"""

from __future__ import annotations

import asyncio

import pytest

duckdb = pytest.importorskip("duckdb")

import forecaster.cli as cli  # type: ignore
from forecaster.providers import ModelSpec


def test_self_search_stashes_sent_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    async def fake_call_chat_ms(ms, prompt, **_kwargs):
        prompts.append(prompt)
        if len(prompts) == 1:
            return (
                "NEED_WEB_EVIDENCE: flood risk",
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                None,
            )
        return (
            '{"spds": {"month_1": {"probs": [0.1,0.1,0.2,0.2,0.2,0.2]}}}',
            {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            None,
        )

    monkeypatch.setattr(cli, "call_chat_ms", fake_call_chat_ms)
    monkeypatch.setattr(
        cli, "run_self_search", lambda *args, **kwargs: {"sources": [{"url": "http://example.com"}]}
    )
    monkeypatch.setenv("PYTHIA_FORECASTER_SELF_SEARCH", "1")
    monkeypatch.setenv("PYTHIA_SPD_WEB_SEARCH_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_MODEL_SELF_SEARCH_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_FORECASTER_SELF_SEARCH_MAX_CALLS_PER_MODEL", "1")

    text, usage, error, _ = asyncio.run(
        cli._call_spd_model_for_spec(
            ModelSpec(name="Test", provider="google", model_id="gpt-test", active=True),
            "PROMPT",
            run_id="run1",
            question_id="Q1",
            iso3="KEN",
            hazard_code="FL",
        )
    )

    assert error is None
    # The returned text came from the second call, whose prompt carried the
    # self-search evidence — that prompt must be the stashed sent prompt.
    assert usage.get("sent_prompt_text") == prompts[-1]
    assert "SELF-SEARCH RESULTS" in usage["sent_prompt_text"]
    assert usage["sent_prompt_text"] != "PROMPT"


def test_disabled_retry_stashes_retry_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []

    async def fake_call_chat_ms(ms, prompt, **_kwargs):
        prompts.append(prompt)
        if len(prompts) == 1:
            return (
                "NEED_WEB_EVIDENCE: more info",
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                None,
            )
        return (
            '{"spds": {"month_1": {"probs": [0.1,0.1,0.2,0.2,0.2,0.2]}}}',
            {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            None,
        )

    monkeypatch.setattr(cli, "call_chat_ms", fake_call_chat_ms)
    monkeypatch.setenv("PYTHIA_FORECASTER_SELF_SEARCH", "0")
    monkeypatch.setenv("PYTHIA_SPD_WEB_SEARCH_ENABLED", "0")

    text, usage, error, _ = asyncio.run(
        cli._call_spd_model_for_spec(
            ModelSpec(name="Test", provider="google", model_id="gpt-test", active=True),
            "PROMPT",
            run_id="run2",
            question_id="Q2",
            iso3="KEN",
            hazard_code="FL",
        )
    )

    assert "spds" in text
    # The recovered forecast came from the no-evidence retry — its prompt is
    # the sent prompt.
    assert usage.get("sent_prompt_text") == prompts[-1]
    assert "WEB EVIDENCE IS UNAVAILABLE THIS RUN" in usage["sent_prompt_text"]


def test_plain_call_has_no_sent_prompt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_call_chat_ms(ms, prompt, **_kwargs):
        return (
            '{"spds": {"month_1": {"probs": [0.1,0.1,0.2,0.2,0.2,0.2]}}}',
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            None,
        )

    monkeypatch.setattr(cli, "call_chat_ms", fake_call_chat_ms)
    monkeypatch.setenv("PYTHIA_FORECASTER_SELF_SEARCH", "0")
    monkeypatch.setenv("PYTHIA_SPD_WEB_SEARCH_ENABLED", "0")

    _text, usage, error, _ = asyncio.run(
        cli._call_spd_model_for_spec(
            ModelSpec(name="Test", provider="google", model_id="gpt-test", active=True),
            "PROMPT",
            run_id="run3",
            question_id="Q3",
            iso3="KEN",
            hazard_code="FL",
        )
    )

    assert error is None
    assert "sent_prompt_text" not in usage


def test_members_v2_pops_sent_prompt_out_of_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    spec = ModelSpec(name="Test", provider="google", model_id="gpt-test", active=True)

    async def fake_call_spd_model_for_spec(ms, prompt, **_kwargs):
        return (
            '{"spds": {"month_1": {"probs": [0.1,0.1,0.2,0.2,0.2,0.2]}}}',
            {"prompt_tokens": 1, "total_tokens": 1, "sent_prompt_text": "FULL SENT PROMPT"},
            None,
            ms,
        )

    monkeypatch.setattr(cli, "_call_spd_model_for_spec", fake_call_spd_model_for_spec)

    _spds, _usage, raw_calls, _meta = asyncio.run(
        cli._call_spd_members_v2(
            "BASE PROMPT",
            [spec],
            run_id="run4",
            question_id="Q4",
            metric="PA",
        )
    )

    assert len(raw_calls) == 1
    entry = raw_calls[0]
    assert entry["sent_prompt"] == "FULL SENT PROMPT"
    # Load-bearing: the multi-KB prompt must never be serialized into usage_json.
    assert "sent_prompt_text" not in entry["usage"]
