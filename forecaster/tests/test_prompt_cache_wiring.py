# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Prompt-cache WIRING tests — production call sites, not just the builders.

The July 2026 audit found PYTHIA_PROMPT_CACHE_ENABLED=1 in every production
workflow while NO forecaster call site passed cache_segments or
prompt_cache_key (only Sibyl did) — claude-opus-5 paid full input price on
every SPD call and the failure was invisible (0 cached tokens reads as
"cache not warm"). These tests pin the wiring at the seam the builders'
own unit tests cannot see.
"""

from __future__ import annotations

import asyncio
import json

import pytest

import forecaster.cli as cli
from forecaster.providers import ModelSpec

_MS_ANTHROPIC = ModelSpec(
    name="claude-opus-5", provider="anthropic",
    model_id="claude-opus-5", active=True, purpose="spd_v2",
)

# Above the builder's _ANTHROPIC_CACHE_MIN_CHARS floor (4096), like the
# real V3 static block.
_PREFIX = "STATIC ROLE/TASK/BUCKETS BLOCK " * 160
_SUFFIX = "per-question data block"


def _capture_call(monkeypatch):
    captured: dict = {}

    async def _fake_call_chat_ms(ms, prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return '{"spds": {}}', {"prompt_tokens": 10}, None

    monkeypatch.setattr(cli, "call_chat_ms", _fake_call_chat_ms)
    return captured


def test_spd_member_call_passes_cache_segments_and_key(monkeypatch):
    captured = _capture_call(monkeypatch)
    asyncio.run(
        cli._call_spd_model_for_spec(
            _MS_ANTHROPIC,
            _PREFIX + _SUFFIX,
            run_id="fc_1",
            question_id="Q1",
            cache_prefix=_PREFIX,
            prompt_cache_key="pythia:spd_v2:ACE:FATALITIES:t1",
        )
    )
    kwargs = captured["kwargs"]
    segments = kwargs["cache_segments"]
    assert segments is not None
    # The invariant the Anthropic builder enforces: segments join == prompt.
    assert "".join(text for text, _bp in segments) == captured["prompt"]
    assert segments[0] == (_PREFIX, True)
    assert kwargs["prompt_cache_key"] == "pythia:spd_v2:ACE:FATALITIES:t1"


def test_segments_survive_tail_evidence_append(monkeypatch):
    # Evidence appends mutate the sent prompt AFTER the prefix was computed;
    # segments are derived at the call site against the final prompt, so the
    # join invariant must hold for the appended prompt too.
    captured = _capture_call(monkeypatch)
    monkeypatch.setenv("PYTHIA_FORECASTER_SELF_SEARCH", "0")
    asyncio.run(
        cli._call_spd_model_for_spec(
            _MS_ANTHROPIC,
            _PREFIX + _SUFFIX,  # base prompt
            run_id="fc_1",
            question_id="Q1",
            cache_prefix=_PREFIX,
        )
    )
    segments = captured["kwargs"]["cache_segments"]
    assert "".join(t for t, _ in segments) == captured["prompt"]


def test_no_prefix_means_no_segments(monkeypatch):
    # Legacy prompt order returns prefix "" → cache_prefix None → the call
    # goes out exactly as before (no segments, no key).
    captured = _capture_call(monkeypatch)
    asyncio.run(
        cli._call_spd_model_for_spec(
            _MS_ANTHROPIC, _PREFIX + _SUFFIX, run_id="fc_1", question_id="Q1",
        )
    )
    assert captured["kwargs"]["cache_segments"] is None
    assert captured["kwargs"]["prompt_cache_key"] is None


def test_batch_body_cache_default_off(monkeypatch):
    """Batch bodies stay byte-identical plain-string by default: batch items
    already get 50% off, cache hits in batches are best-effort while the
    cache-write premium is guaranteed — opt-in via PYTHIA_BATCH_PROMPT_CACHE
    once telemetry proves reads materialize."""

    from forecaster.providers import build_body_for_spec

    monkeypatch.setenv("PYTHIA_PROMPT_CACHE_ENABLED", "1")
    monkeypatch.delenv("PYTHIA_BATCH_PROMPT_CACHE", raising=False)
    body = build_body_for_spec(
        _MS_ANTHROPIC, _PREFIX + _SUFFIX, 0.2,
        cache_prefix=_PREFIX, prompt_cache_key="pythia:spd_v2:ACE:FATALITIES:t1",
    )
    assert isinstance(body["messages"][0]["content"], str)
    assert "prompt_cache_key" not in json.dumps(body)


def test_batch_body_cache_opt_in_emits_anthropic_blocks_only(monkeypatch):
    from forecaster.providers import build_body_for_spec

    monkeypatch.setenv("PYTHIA_PROMPT_CACHE_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_BATCH_PROMPT_CACHE", "1")
    body = build_body_for_spec(
        _MS_ANTHROPIC, _PREFIX + _SUFFIX, 0.2, cache_prefix=_PREFIX,
    )
    content = body["messages"][0]["content"]
    assert isinstance(content, list)
    assert any(
        isinstance(b, dict) and b.get("cache_control") for b in content
    )
    assert "".join(b["text"] for b in content) == _PREFIX + _SUFFIX

    # OpenAI batch bodies NEVER carry prompt_cache_key (no batch discount).
    ms_openai = ModelSpec(
        name="gpt-5.6-sol", provider="openai",
        model_id="gpt-5.6-sol", active=True, purpose="spd_v2",
    )
    oai_body = build_body_for_spec(
        ms_openai, _PREFIX + _SUFFIX, 0.2,
        cache_prefix=_PREFIX, prompt_cache_key="pythia:spd_v2:ACE:FATALITIES:t1",
    )
    assert "prompt_cache_key" not in oai_body


@pytest.mark.parametrize("track", [1, 2])
def test_return_parts_join_equals_full_prompt(track, monkeypatch):
    """prefix + suffix must equal the single-string build for BOTH tracks —
    the call sites now build via return_parts and join."""

    from forecaster import prompts

    monkeypatch.setenv("PYTHIA_PROMPT_V3_ORDER", "1")
    question = {
        "question_id": "SOM_ACE_FATALITIES_2026-08",
        "iso3": "SOM",
        "hazard_code": "ACE",
        "metric": "FATALITIES",
        "wording": "How many conflict fatalities?",
        "window_start_date": "2026-08-01",
    }
    full = prompts.build_spd_prompt_v2(question, {}, {}, {}, track=track)
    prefix, suffix = prompts.build_spd_prompt_v2(
        question, {}, {}, {}, track=track, return_parts=True
    )
    assert prefix + suffix == full
    assert len(prefix) > 1000  # a real cacheable static block under V3
