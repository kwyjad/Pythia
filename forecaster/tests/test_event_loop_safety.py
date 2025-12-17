# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import asyncio

import pytest

duckdb = pytest.importorskip("duckdb")

from forecaster import providers


def test_call_chat_ms_handles_multiple_event_loops(monkeypatch):
    providers._LLM_SEMAPHORES.clear()

    def fake_provider_call(provider, prompt, model, temperature):
        return providers.ProviderResult("ok", providers.usage_to_dict(None), 0.0, model)

    monkeypatch.setattr(providers, "_call_provider_sync", fake_provider_call)

    ms = providers.ModelSpec(name="Test", provider="openai", model_id="gpt-5-nano", active=True)

    for prompt in ("loop-one", "loop-two"):
        text, usage, error = asyncio.run(providers.call_chat_ms(ms, prompt))
        assert text == "ok"
        assert error == ""
        assert isinstance(usage, dict)

    assert len(providers._LLM_SEMAPHORES) >= 2
