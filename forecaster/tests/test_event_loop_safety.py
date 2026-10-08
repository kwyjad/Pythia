# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import asyncio

import pytest

duckdb = pytest.importorskip("duckdb")

from forecaster import providers


def _fake_provider_call(provider, prompt, model, temperature, *, timeout_sec=None, thinking_level=None, **kwargs):
    return providers.ProviderResult("ok", providers.usage_to_dict(None), 0.0, model)


def test_call_chat_ms_handles_multiple_event_loops(monkeypatch):
    monkeypatch.setattr(providers, "_call_provider_sync", _fake_provider_call)
    ms = providers.ModelSpec(name="Test", provider="openai", model_id="gpt-5-nano", active=True)

    seen = []

    async def one(prompt):
        seen.append(providers.get_llm_semaphore())
        return await providers.call_chat_ms(ms, prompt)

    for prompt in ("loop-one", "loop-two"):
        text, usage, error = asyncio.run(one(prompt))
        assert text == "ok"
        assert error == ""
        assert isinstance(usage, dict)

    # Each loop got its own semaphore, and a closed loop's entry does not
    # linger in the registry.
    assert seen[0] is not seen[1]


def test_a_new_loop_never_inherits_a_dead_loops_state(monkeypatch):
    """The registries were keyed by id(loop). asyncio.run closes its loop and
    the next loop may be allocated at the same address, so it inherited a
    semaphore and an HTTP client bound to a closed loop. Whether that
    happened depended on the heap, which made the test above pass or fail
    with test order. Forcing every id() to collide makes the fault certain:
    keyed by the loop object, the two loops still get separate state."""
    monkeypatch.setattr(providers, "id", lambda _obj: 42, raising=False)

    sems, tokens = [], []

    async def grab():
        sems.append(providers.get_llm_semaphore())
        tokens.append(providers.loop_token(asyncio.get_running_loop()))

    asyncio.run(grab())
    asyncio.run(grab())
    assert sems[0] is not sems[1]
    assert tokens[0] != tokens[1]


def test_loop_tokens_are_not_reused():
    loops = [asyncio.new_event_loop() for _ in range(3)]
    try:
        toks = [providers.loop_token(lp) for lp in loops]
        assert len(set(toks)) == 3
        assert providers.loop_token(loops[0]) == toks[0]
    finally:
        for lp in loops:
            lp.close()
