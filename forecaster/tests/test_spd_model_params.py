# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Per-model SPD params must survive the purpose stamp.

`_call_spd_model_for_spec` stamps `purpose="spd_v2"` on every ensemble member
before calling the provider. It used to do that by rebuilding the ModelSpec
field-by-field, which silently omitted `temperature` and `thinking` — so every
per-model param in config.yaml's ensemble block was discarded on the way to the
API. Config-derived specs always carry `purpose=None`, so the branch fired for
all of them: the OpenAI members ran at the provider's default reasoning effort,
and the Google members were masked by the PYTHIA_GOOGLE_SPD_THINKING_LEVEL_*
env fallback rather than using their configured level.
"""

from __future__ import annotations

import asyncio

import pytest


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _capture_spec(monkeypatch: pytest.MonkeyPatch, spec):
    """Invoke the SPD wrapper and return the ModelSpec handed to the provider."""

    from forecaster import cli

    captured: dict = {}

    async def _fake_call_chat_ms(ms, prompt, **kwargs):
        captured["spec"] = ms
        return "{}", {}, None

    monkeypatch.setattr(cli, "call_chat_ms", _fake_call_chat_ms)
    _run(cli._call_spd_model_for_spec(spec, "PROMPT"))
    return captured["spec"]


def test_purpose_stamp_preserves_thinking_and_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    from forecaster.providers import ModelSpec

    spec = ModelSpec(
        name="gemini-3.1-pro-preview",
        provider="google",
        model_id="gemini-3.1-pro-preview",
        temperature=0.2,
        thinking="medium",
        purpose=None,
    )

    sent = _capture_spec(monkeypatch, spec)

    assert sent.purpose == "spd_v2", "the SPD purpose must still be stamped"
    assert sent.thinking == "medium", "configured thinking level was dropped"
    assert sent.temperature == pytest.approx(0.2), "configured temperature was dropped"
    assert sent.model_id == "gemini-3.1-pro-preview"


def test_every_configured_ensemble_member_keeps_its_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live config.yaml ensemble, end to end through the SPD wrapper.

    Guards the whole lineup rather than one hand-built spec, so a future member
    added with params can't silently lose them.
    """

    from forecaster.providers import _load_ensemble_from_config

    configured = _load_ensemble_from_config()
    assert configured, "config.yaml ensemble failed to resolve"

    for spec in configured:
        sent = _capture_spec(monkeypatch, spec)
        assert sent.purpose == "spd_v2"
        assert sent.thinking == spec.thinking, f"{spec.model_id}: thinking dropped"
        assert sent.temperature == spec.temperature, f"{spec.model_id}: temperature dropped"


def test_already_stamped_spec_is_passed_through_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from forecaster.providers import ModelSpec

    spec = ModelSpec(
        name="gpt-5.6-sol",
        provider="openai",
        model_id="gpt-5.6-sol",
        thinking="high",
        purpose="spd_v2",
    )

    sent = _capture_spec(monkeypatch, spec)

    assert sent is spec
    assert sent.thinking == "high"
