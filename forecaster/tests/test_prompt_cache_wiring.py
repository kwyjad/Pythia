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

# Above every model's cacheable minimum (the table tops out at 4096 tokens
# ≈ 16k chars), like the real V3 static block.
_PREFIX = "STATIC ROLE/TASK/BUCKETS BLOCK " * 600
_SUFFIX = "per-question data block"

# ~600 tokens: over claude-opus-5's 512-token minimum, under the 1024 the
# single old constant assumed. This is the size band the real binary_v2
# prefixes (~760 tokens) sit in.
_SHORT_PREFIX = "SHORT STATIC BLOCK " * 130


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


def _cache_controls(body: dict) -> list[dict]:
    content = body["messages"][0]["content"]
    if not isinstance(content, list):
        return []
    return [b["cache_control"] for b in content if isinstance(b, dict) and b.get("cache_control")]


def test_batch_bodies_buy_an_hour_ttl_but_sync_bodies_do_not(monkeypatch):
    """A batch outlives the 5-minute default TTL — the 2026-08-01 SPD batch
    ran 41 minutes — so an unqualified `ephemeral` marker in a batch body
    expires before most of the batch is scheduled and the write premium buys
    nothing. Sync calls are answered immediately and keep the 5m default.
    """

    from forecaster.providers import build_anthropic_body, build_body_for_spec

    monkeypatch.setenv("PYTHIA_PROMPT_CACHE_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_BATCH_PROMPT_CACHE", "1")

    batch = build_body_for_spec(
        _MS_ANTHROPIC, _PREFIX + _SUFFIX, 0.2, cache_prefix=_PREFIX,
    )
    assert [c.get("ttl") for c in _cache_controls(batch)] == ["1h"]

    sync = build_anthropic_body(
        _PREFIX + _SUFFIX, "claude-opus-5", 0.2, purpose="spd_v2",
        cache_segments=[(_PREFIX, True), (_SUFFIX, False)],
    )
    sync_controls = _cache_controls(sync)
    assert sync_controls and all("ttl" not in c for c in sync_controls)

    # The TTL is the ONLY divergence: the prompt bytes a batch item carries
    # must still equal the sync prompt, or a replayed result answers a
    # different question than the sync fallback would have asked.
    def _text(body: dict) -> str:
        content = body["messages"][0]["content"]
        return content if isinstance(content, str) else "".join(b["text"] for b in content)

    assert _text(batch) == _text(sync) == _PREFIX + _SUFFIX

    # PYTHIA_BATCH_CACHE_TTL=none is the escape hatch back to the 5m default.
    monkeypatch.setenv("PYTHIA_BATCH_CACHE_TTL", "none")
    reverted = build_body_for_spec(
        _MS_ANTHROPIC, _PREFIX + _SUFFIX, 0.2, cache_prefix=_PREFIX,
    )
    assert all("ttl" not in c for c in _cache_controls(reverted))


def test_cache_minimum_is_per_model_not_one_constant(monkeypatch):
    """claude-opus-5 caches from 512 tokens; Haiku 4.5 needs 4096. A single
    constant sized for 1024 left every binary_v2 prefix (~760 tokens)
    unmarked on the model that would have cached it.
    """

    from forecaster.providers import build_anthropic_body

    monkeypatch.setenv("PYTHIA_PROMPT_CACHE_ENABLED", "1")
    monkeypatch.delenv("PYTHIA_ANTHROPIC_CACHE_MIN_CHARS", raising=False)
    segments = [(_SHORT_PREFIX, True), (_SUFFIX, False)]

    opus = build_anthropic_body(
        _SHORT_PREFIX + _SUFFIX, "claude-opus-5", 0.2,
        purpose="spd_v2", cache_segments=segments,
    )
    assert _cache_controls(opus), "opus-5 must mark a ~600-token prefix (512 minimum)"

    haiku = build_anthropic_body(
        _SHORT_PREFIX + _SUFFIX, "claude-haiku-4-5", 0.2,
        purpose="spd_v2", cache_segments=segments,
    )
    assert not _cache_controls(haiku), "haiku-4-5 needs 4096 tokens — do not mark"

    # An unlisted model falls back to the most conservative minimum.
    unknown = build_anthropic_body(
        _SHORT_PREFIX + _SUFFIX, "claude-something-99", 0.2,
        purpose="spd_v2", cache_segments=segments,
    )
    assert not _cache_controls(unknown)


def test_cache_minimum_env_override_wins(monkeypatch):
    from forecaster.providers import build_anthropic_body

    monkeypatch.setenv("PYTHIA_PROMPT_CACHE_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_ANTHROPIC_CACHE_MIN_CHARS", "1")
    body = build_anthropic_body(
        _SHORT_PREFIX + _SUFFIX, "claude-haiku-4-5", 0.2,
        purpose="spd_v2", cache_segments=[(_SHORT_PREFIX, True), (_SUFFIX, False)],
    )
    assert _cache_controls(body)


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


# ---------------------------------------------------------------------------
# Prompt-cache warm gate: the first call per (provider, model, key) runs alone
# ---------------------------------------------------------------------------


def test_warm_gate_serialises_only_the_first_call(monkeypatch):
    """On 2026-09-01 all 210 OpenAI SPD calls ran sync at once and read zero
    cached tokens: none of them could hit a cache no earlier call had written.
    The gate must let exactly one call per key finish before the rest start,
    and then let the rest overlap freely."""
    monkeypatch.setenv("PYTHIA_PROMPT_CACHE_ENABLED", "1")
    cli._reset_cache_warm_gates()

    async def _scenario():
        ms = ModelSpec(provider="openai", model_id="gpt-5.6-sol", name="sol")
        key = cli._cache_warm_key(ms, "pythia:spd_v2:ACE:PA:t1")
        assert key is not None
        in_flight = 0
        peak_before_first_done = 0
        peak_after_first_done = 0
        first_done = False
        order: list[int] = []

        async def _call(i: int):
            nonlocal in_flight, peak_before_first_done, peak_after_first_done, first_done
            async with cli._prompt_cache_warm_gate(key):
                in_flight += 1
                if first_done:
                    peak_after_first_done = max(peak_after_first_done, in_flight)
                else:
                    peak_before_first_done = max(peak_before_first_done, in_flight)
                order.append(i)
                await asyncio.sleep(0.01)
                in_flight -= 1
                first_done = True

        await asyncio.gather(*(_call(i) for i in range(6)))
        return peak_before_first_done, peak_after_first_done, order

    before, after, order = asyncio.run(_scenario())
    assert before == 1, "the first call must run alone"
    assert after >= 2, "later calls must overlap once the cache is warm"
    assert len(order) == 6


def test_warm_gate_is_a_no_op_when_caching_is_off_or_keyless(monkeypatch):
    monkeypatch.setenv("PYTHIA_PROMPT_CACHE_ENABLED", "0")
    cli._reset_cache_warm_gates()

    async def _scenario():
        ms = ModelSpec(provider="openai", model_id="gpt-5.6-sol", name="sol")
        assert cli._cache_warm_key(ms, "pythia:spd_v2:ACE:PA:t1") is None
        monkeypatch.setenv("PYTHIA_PROMPT_CACHE_ENABLED", "1")
        assert cli._cache_warm_key(ms, None) is None
        peak = 0
        in_flight = 0

        async def _call():
            nonlocal peak, in_flight
            async with cli._prompt_cache_warm_gate(None):
                in_flight += 1
                peak = max(peak, in_flight)
                await asyncio.sleep(0.01)
                in_flight -= 1

        await asyncio.gather(*(_call() for _ in range(4)))
        return peak

    assert asyncio.run(_scenario()) == 4


def test_warm_gate_releases_when_the_first_call_raises(monkeypatch):
    monkeypatch.setenv("PYTHIA_PROMPT_CACHE_ENABLED", "1")
    cli._reset_cache_warm_gates()

    async def _scenario():
        ms = ModelSpec(provider="anthropic", model_id="claude-opus-5", name="claude")
        key = cli._cache_warm_key(ms, "pythia:spd_v2:FL:PA:t1")
        ran: list[str] = []

        async def _first():
            async with cli._prompt_cache_warm_gate(key):
                raise RuntimeError("provider down")

        async def _second():
            async with cli._prompt_cache_warm_gate(key):
                ran.append("second")

        results = await asyncio.gather(_first(), _second(), return_exceptions=True)
        return results, ran

    results, ran = asyncio.run(_scenario())
    assert isinstance(results[0], RuntimeError)
    assert ran == ["second"]
