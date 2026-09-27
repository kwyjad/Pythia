# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The GPT-5.6 -> GPT-6 swap (Sol and Luna), and the grounding roles with it.

Each change has a quiet failure mode:

* GPT-6 accepts ``temperature`` only with reasoning effort ``none``. The
  temperature guard matches literal prefixes and "gpt-5" does not cover
  "gpt-6", so a call with no effort (the hs_fallback JSON repair) would 400.
* GPT-6 defaults to effort ``medium``. The ensemble members pin theirs, so a
  later default change cannot move forecast depth without a diff.
* OpenAI batches must hold one model each and send the body the sync path
  sends, priced at half.
* The web-search grounding role moved off gpt-4.1, a non-reasoning model, onto
  GPT-6. At the old 800-token ceiling a reasoning model can spend the whole
  budget thinking and return no answer.
"""

from __future__ import annotations

import dataclasses
import sys
import types

import pytest


def _openai_members():
    from forecaster.providers import _load_ensemble_from_config

    return [ms for ms in _load_ensemble_from_config() if ms.provider == "openai"]


def test_the_openai_members_are_gpt6_with_pinned_effort() -> None:
    got = {ms.model_id: ms.thinking for ms in _openai_members()}
    assert got == {"gpt-6-sol": "high", "gpt-6-luna": "medium"}


def test_no_openai_role_is_below_gpt6() -> None:
    from pythia.llm_profiles import _ROLE_FALLBACKS, get_role_model

    for role in ("hs_fallback", "grounding_openai", "grounding_openai_fallback"):
        ref = get_role_model(role)
        assert ref.startswith("openai:gpt-6-"), f"{role} -> {ref}"
        assert _ROLE_FALLBACKS[role].startswith("openai:gpt-6-"), role


@pytest.mark.parametrize("model_id", ["gpt-6-sol", "gpt-6-luna", "GPT-6-Sol"])
def test_gpt6_never_sends_temperature(model_id: str) -> None:
    from forecaster import providers

    assert providers._openai_drops_temperature(model_id)
    # The JSON-repair path: no effort, so the guard is the only protection.
    assert "temperature" not in providers.build_openai_body("p", model_id, 0.2)


def test_batch_body_matches_sync_body(monkeypatch: pytest.MonkeyPatch) -> None:
    from forecaster import providers

    monkeypatch.setenv("PYTHIA_PROMPT_CACHE_ENABLED", "1")
    for ms in _openai_members():
        ms = dataclasses.replace(ms, purpose="spd_v2")
        batch = providers.build_body_for_spec(ms, "PROMPT", 0.2, prompt_cache_key="k")
        sync = providers.build_openai_body(
            "PROMPT", ms.model_id, 0.2, reasoning_effort=ms.thinking
        )
        assert batch == {k: v for k, v in sync.items() if k != "prompt_cache_key"}
        assert batch["reasoning_effort"] == ms.thinking
        assert "temperature" not in batch


@pytest.mark.parametrize(
    "model_id, rates",
    [
        # GPT-6 release, 2026-09-22.
        ("gpt-6-sol", {"input": 2.0, "output": 10.0, "cached_input": 0.20}),
        ("gpt-6-luna", {"input": 0.10, "output": 0.50, "cached_input": 0.01}),
    ],
)
def test_gpt6_prices_and_half_price_batch(model_id: str, rates: dict) -> None:
    from forecaster.providers import compute_cost_split_usd, resolve_price_detail

    detail = resolve_price_detail(model_id)
    for key, value in rates.items():
        assert detail[key] == pytest.approx(value), f"{model_id}.{key}"
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    full = compute_cost_split_usd(model_id, usage)
    half = compute_cost_split_usd(model_id, dict(usage, service_tier="batch"))
    assert full == pytest.approx((rates["input"], rates["output"], rates["input"] + rates["output"]))
    assert half == pytest.approx(tuple(x / 2 for x in full))


def test_superseded_gpt56_still_prices_for_a_rollback() -> None:
    from forecaster.providers import resolve_price_per_1m

    assert resolve_price_per_1m("gpt-5.6-sol") == (4.0, 20.0)
    assert resolve_price_per_1m("gpt-5.6-luna") == (0.20, 1.20)


def _run_grounding(monkeypatch: pytest.MonkeyPatch, model_id: str) -> dict:
    captured: dict = {}

    class _Responses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return {"output": [], "usage": {}}

    class _Client:
        def __init__(self, **_kwargs):
            self.responses = _Responses()

    try:
        import openai  # noqa: F401
    except ImportError:  # CI installs no openai SDK; the backend only needs the name
        monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_Client))
    from pythia.web_research.backends import openai_web_search

    monkeypatch.setattr(openai_web_search, "OpenAI", _Client)
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("PYTHIA_WEB_RESEARCH_MODEL_ID", model_id)
    openai_web_search.fetch_via_openai_web_search(
        "q", recency_days=30, include_structural=False, timeout_sec=10, max_results=5
    )
    return captured


def test_grounding_on_gpt6_asks_for_low_effort_and_room_to_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = _run_grounding(monkeypatch, "gpt-6-sol")
    assert sent["model"] == "gpt-6-sol"
    assert sent["reasoning"] == {"effort": "low"}
    assert sent["max_output_tokens"] >= 4000
    assert "temperature" not in sent


def test_grounding_on_a_non_reasoning_model_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = _run_grounding(monkeypatch, "gpt-4.1")
    assert "reasoning" not in sent
    assert sent["max_output_tokens"] == 800
