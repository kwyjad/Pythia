# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The Claude Opus 5 -> 5.5 swap, and the September 2026 OpenAI price changes.

Three things changed together and each has a quiet failure mode:

* Opus 5.5 defaults to effort ``medium`` where Opus 5 defaulted to ``high``.
  A member or a Sibyl step that sends no effort would think less after the
  swap, and every forecast would still parse. So the depth is pinned.
* Opus 5.5 cannot switch thinking off (``thinking: disabled`` is a 400 at every
  effort level). Nothing here may send a ``thinking`` field at all.
* The batch path must send the same body the sync path does, and price the
  result at half. A batch body that drifts from the sync body replays an answer
  to a question nobody asked.
"""

from __future__ import annotations

import dataclasses

import pytest


def _claude_members():
    from forecaster.providers import _load_ensemble_from_config

    members = [ms for ms in _load_ensemble_from_config() if ms.provider == "anthropic"]
    assert members, "the ensemble has no Anthropic member"
    return members


def test_the_ensemble_claude_member_is_opus_55() -> None:
    assert [ms.model_id for ms in _claude_members()] == ["claude-opus-5-5"]


def test_every_anthropic_model_we_can_reach_is_opus_55() -> None:
    """Ensemble, Sibyl and the interpreter move together or not at all."""
    import sibyl.config as sibyl_config
    from pythia.llm_profiles import get_role_model

    assert sibyl_config.MODEL == "claude-opus-5-5"
    assert get_role_model("interpreter") == "anthropic:claude-opus-5-5"


def test_opus_55_is_in_every_literal_prefix_table() -> None:
    from forecaster import providers

    assert "claude-opus-5-5" in providers._ANTHROPIC_NO_TEMPERATURE_PREFIXES
    assert "claude-opus-5-5" in providers._ANTHROPIC_EFFORT_PREFIXES
    assert providers._anthropic_cache_min_chars("claude-opus-5-5") == providers._anthropic_cache_min_chars(
        "claude-opus-5"
    ), "Opus 5.5 keeps Opus 5's 512-token cacheable minimum"


def test_the_claude_member_pins_high_effort_rather_than_inheriting_medium() -> None:
    for ms in _claude_members():
        assert ms.thinking == "high", (
            f"{ms.model_id}: no explicit effort — Opus 5.5 would run at its "
            "default of medium, one level below the Opus 5 member it replaced"
        )


@pytest.mark.parametrize("batch_cache", ["0", "1"])
def test_batch_body_matches_sync_body_and_carries_effort(
    monkeypatch: pytest.MonkeyPatch, batch_cache: str
) -> None:
    from forecaster import providers

    monkeypatch.setenv("PYTHIA_BATCH_PROMPT_CACHE", batch_cache)
    monkeypatch.setenv("PYTHIA_PROMPT_CACHE_ENABLED", "1")
    ms = dataclasses.replace(_claude_members()[0], purpose="spd_v2")
    prefix = "STATIC " * 800
    prompt = prefix + "QUESTION"

    batch = providers.build_body_for_spec(ms, prompt, 0.2, cache_prefix=prefix)
    sync = providers.build_anthropic_body(
        prompt, ms.model_id, 0.2, purpose="spd_v2", thinking_level="high"
    )

    assert batch["model"] == "claude-opus-5-5"
    assert batch["output_config"] == {"effort": "high"}
    assert batch["max_tokens"] == sync["max_tokens"] >= 32768
    for forbidden in ("temperature", "top_p", "top_k", "thinking", "tool_choice"):
        assert forbidden not in batch, f"{forbidden} would 400 on Opus 5.5"
    if batch_cache == "0":
        assert batch == sync, "a batch body must be byte-identical to the sync body"
    else:
        # The one permitted divergence: the cache marker and its 1h TTL.
        blocks = batch["messages"][0]["content"]
        assert "".join(b["text"] for b in blocks) == prompt
        assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_sibyl_sends_high_effort_on_every_step(monkeypatch: pytest.MonkeyPatch) -> None:
    from forecaster import providers
    from sibyl import agent

    captured: dict = {}

    def _fake_call_anthropic(prompt, model, temperature, **kwargs):
        captured.update(kwargs, model=model)
        return providers.ProviderResult("{}", {}, 0.0, model, error=None)

    monkeypatch.setattr(providers, "call_anthropic", _fake_call_anthropic)
    agent._call_model("step prompt")
    assert captured["model"] == "claude-opus-5-5"
    assert captured["thinking_level"] == "high"


@pytest.mark.parametrize(
    "model_id, rates",
    [
        # platform.claude.com pricing page, read 2026-09-27.
        ("claude-opus-5-5", {"input": 4.0, "output": 20.0, "cached_input": 0.20,
                             "cache_write_5m": 5.0, "cache_write_1h": 8.0}),
        # Promotional from 2026-08-21 through at least 2026-11-21; list is $5/$30.
        ("gpt-5.6-sol", {"input": 4.0, "output": 20.0, "cached_input": 0.40}),
        # Cut 80% on 2026-07-30 from $1/$6.
        ("gpt-5.6-luna", {"input": 0.20, "output": 1.20, "cached_input": 0.02}),
    ],
)
def test_current_prices(model_id: str, rates: dict) -> None:
    from forecaster.providers import resolve_price_detail

    detail = resolve_price_detail(model_id)
    for key, value in rates.items():
        assert detail[key] == pytest.approx(value), f"{model_id}.{key}"


def test_opus_55_batch_is_half_price() -> None:
    """Anthropic's batch table lists Opus 5.5 at $2 / $10."""
    from forecaster.providers import compute_cost_split_usd

    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    assert compute_cost_split_usd("claude-opus-5-5", usage) == pytest.approx((4.0, 20.0, 24.0))
    assert compute_cost_split_usd(
        "claude-opus-5-5", dict(usage, service_tier="batch")
    ) == pytest.approx((2.0, 10.0, 12.0))


def test_superseded_opus_5_still_prices_for_a_rollback() -> None:
    from forecaster.providers import resolve_price_per_1m

    assert resolve_price_per_1m("claude-opus-5") == (5.0, 25.0)
