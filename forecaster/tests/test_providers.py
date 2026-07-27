# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import importlib

import pytest


def _reload_providers(monkeypatch: pytest.MonkeyPatch, env: dict[str, str] | None = None):
    env = env or {}
    import sys
    import types

    for key in (
        "PYTHIA_BLOCK_PROVIDERS",
        "PYTHIA_SPD_ENSEMBLE_SPECS",
        "MODEL_COSTS_JSON",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    # Stub duckdb to avoid optional dependency requirement for these tests.
    if "duckdb" not in sys.modules:
        monkeypatch.setitem(
            sys.modules,
            "duckdb",
            types.SimpleNamespace(CatalogException=Exception, connect=lambda *args, **kwargs: None),
        )
    # Ensure we start from a clean state for each test.
    import forecaster.providers as providers  # noqa: WPS433

    return importlib.reload(providers)


def test_parse_ensemble_specs_allows_repeated_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    providers = _reload_providers(
        monkeypatch,
        {
            "OPENAI_API_KEY": "x",
            "ANTHROPIC_API_KEY": "x",
            "GEMINI_API_KEY": "x",
        },
    )

    specs = providers.parse_ensemble_specs("google:gemini-3-pro-preview,google:gemini-3-flash-preview")

    assert len(specs) == 2
    assert all(ms.provider == "google" for ms in specs)
    assert {ms.model_id for ms in specs} == {"gemini-3-pro-preview", "gemini-3-flash-preview"}


def test_blocked_providers_removed_from_specs(monkeypatch: pytest.MonkeyPatch) -> None:
    providers = _reload_providers(
        monkeypatch,
        {
            "OPENAI_API_KEY": "x",
            "ANTHROPIC_API_KEY": "x",
            "GEMINI_API_KEY": "x",
            "PYTHIA_BLOCK_PROVIDERS": "google",
        },
    )

    specs = providers.parse_ensemble_specs(
        "openai:gpt-5.4,google:gemini-3.1-pro-preview,anthropic:claude-opus-4-8"
    )

    assert all(ms.provider != "google" for ms in specs)
    assert any(ms.provider == "openai" for ms in specs)


def test_spd_ensemble_override_keeps_two_gemini_models(monkeypatch: pytest.MonkeyPatch) -> None:
    providers = _reload_providers(
        monkeypatch,
        {
            "OPENAI_API_KEY": "x",
            "ANTHROPIC_API_KEY": "x",
            "GEMINI_API_KEY": "x",
            "PYTHIA_SPD_ENSEMBLE_SPECS": (
                "openai:gpt-5.4,anthropic:claude-opus-4-8,"
                "google:gemini-3-pro-preview,google:gemini-3-flash-preview"
            ),
            "PYTHIA_BLOCK_PROVIDERS": "",
        },
    )

    google_specs = [ms for ms in providers.SPD_ENSEMBLE if ms.provider == "google"]

    assert len(google_specs) == 2
    assert {ms.model_id for ms in google_specs} == {"gemini-3-pro-preview", "gemini-3-flash-preview"}


def test_estimate_cost_usd_per_million_rates() -> None:
    """Cost table is per-1M tokens: 1M in + 1M out == input rate + output rate."""
    from forecaster import providers

    usage = {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 1_000_000,
        "total_tokens": 2_000_000,
    }
    cost = providers.estimate_cost_usd("gemini-3.5-flash", usage)

    # gemini-3.5-flash: $1.50/1M input + $9.00/1M output
    assert cost == pytest.approx(10.50)


def test_estimate_cost_usd_small_call() -> None:
    from forecaster import providers

    usage = {"prompt_tokens": 1000, "completion_tokens": 1000, "total_tokens": 2000}
    cost = providers.estimate_cost_usd("gpt-5.4", usage)

    # gpt-5.4: $2.50/1M input + $15.00/1M output -> (2.50 + 15.00) / 1000
    assert cost == pytest.approx(0.0175)


def test_display_name_is_specific_model_id() -> None:
    """Display names must be specific model ids, not generic family labels."""
    from forecaster import providers

    assert providers._provider_display_name("google", "gemini-3.5-flash") == "gemini-3.5-flash"
    assert providers._provider_display_name("google", "gemini-3.1-pro-preview") == "gemini-3.1-pro-preview"
    assert providers._provider_display_name("openai", "gpt-5.4") == "gpt-5.4"
    assert providers._provider_display_name("anthropic", "claude-opus-4-8") == "claude-opus-4-8"
    # Explicit config display_name still wins.
    assert (
        providers._provider_display_name("openai", "gpt-5.4", {"display_name": "Custom"})
        == "Custom"
    )


def test_openai_drops_temperature_for_gpt5_family() -> None:
    """GPT-5 reasoning models reject temperature; gpt-4.1 still accepts it."""
    from forecaster import providers

    assert providers._openai_drops_temperature("gpt-5.4")
    assert providers._openai_drops_temperature("gpt-5.4-mini")
    assert providers._openai_drops_temperature("gpt-5-mini")
    assert not providers._openai_drops_temperature("gpt-4.1")
    assert not providers._openai_drops_temperature("gpt-4.1-mini")
    assert not providers._openai_drops_temperature("")


def test_call_openai_gpt54_body_never_contains_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The hs_fallback JSON-repair path calls gpt-5.4 with no reasoning_effort;
    the request body must not carry temperature (the API rejects it)."""
    from forecaster import providers

    captured: dict = {}

    class _FakeResponse:
        ok = True
        status_code = 200
        headers: dict = {}
        text = ""

        @staticmethod
        def json() -> dict:
            return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    def _fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        captured["body"] = json
        return _FakeResponse()

    monkeypatch.setattr(providers, "_OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(providers.requests, "post", _fake_post)

    result = providers.call_openai("hi", "gpt-5.4", 0.0)
    assert result.error is None
    assert "temperature" not in captured["body"]
    assert "reasoning_effort" not in captured["body"]

    result = providers.call_openai("hi", "gpt-4.1", 0.2)
    assert result.error is None
    assert captured["body"]["temperature"] == 0.2


def test_anthropic_temperature_guard_covers_every_reachable_anthropic_model() -> None:
    """Every Anthropic model the pipeline can call must be in the no-temperature tuple.

    The tuple holds LITERAL id prefixes, not families: "claude-opus-4-8" does not
    cover "claude-opus-5". Sending `temperature` to an Opus 4.7+ model is an HTTP
    400, which would take out the SPD Claude member and every Sibyl step at once.
    Derived from the live config + Sibyl default so a future swap that forgets to
    extend the tuple fails here rather than in production.
    """
    from forecaster import providers
    from forecaster.providers import _load_ensemble_from_config
    import sibyl.config as sibyl_config

    reachable = {
        ms.model_id for ms in _load_ensemble_from_config() if ms.provider == "anthropic"
    }
    reachable.add(sibyl_config.MODEL)
    assert reachable, "expected at least one reachable Anthropic model"

    unguarded = [
        m for m in sorted(reachable)
        if not m.lower().startswith(providers._ANTHROPIC_NO_TEMPERATURE_PREFIXES)
    ]
    assert not unguarded, (
        "Anthropic models missing from _ANTHROPIC_NO_TEMPERATURE_PREFIXES "
        f"(their calls would 400): {unguarded}"
    )


def test_call_anthropic_body_omits_temperature_for_opus5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from forecaster import providers

    captured: dict = {}

    class _FakeResponse:
        ok = True
        status_code = 200
        headers: dict = {}
        text = ""

        @staticmethod
        def json() -> dict:
            return {
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

    def _fake_post(url, headers=None, json=None, timeout=None):  # noqa: A002
        captured["body"] = json
        return _FakeResponse()

    monkeypatch.setattr(providers, "_ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(providers.requests, "post", _fake_post)

    result = providers.call_anthropic("hi", "claude-opus-5", 0.2)
    assert result.error is None
    assert result.text == "ok"
    assert "temperature" not in captured["body"]

    # The cheap grounding tier still accepts sampling params — don't over-guard.
    providers.call_anthropic("hi", "claude-haiku-4-5-20251001", 0.2)
    assert captured["body"]["temperature"] == 0.2


def test_call_anthropic_skips_thinking_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adaptive thinking (on by default from Opus 5) must not leak into the answer."""
    from forecaster import providers

    class _FakeResponse:
        ok = True
        status_code = 200
        headers: dict = {}
        text = ""

        @staticmethod
        def json() -> dict:
            return {
                "content": [
                    {"type": "thinking", "thinking": "internal reasoning"},
                    {"type": "text", "text": '{"answer": 1}'},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }

    monkeypatch.setattr(providers, "_ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(providers.requests, "post", lambda *a, **k: _FakeResponse())

    result = providers.call_anthropic("hi", "claude-opus-5", 0.2)
    assert result.text == '{"answer": 1}'
    assert "internal reasoning" not in result.text


def test_anthropic_stop_reason_names_refusal_and_truncation() -> None:
    """Refusal and max_tokens are HTTP 200 with empty/partial content."""
    from forecaster import providers

    assert providers._anthropic_stop_reason_error({"stop_reason": "end_turn"}) is None
    assert providers._anthropic_stop_reason_error({}) is None
    assert providers._anthropic_stop_reason_error(None) is None

    refusal = providers._anthropic_stop_reason_error(
        {"stop_reason": "refusal", "stop_details": {"category": "cyber"}}
    )
    assert refusal is not None and "refusal" in refusal.lower()
    assert "cyber" in refusal

    # stop_details is optional — must not raise or lose the signal.
    bare = providers._anthropic_stop_reason_error({"stop_reason": "refusal"})
    assert bare is not None and "refusal" in bare.lower()

    truncated = providers._anthropic_stop_reason_error({"stop_reason": "max_tokens"})
    assert truncated is not None and "max_tokens" in truncated


def test_call_anthropic_reports_refusal_but_keeps_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused call still burned input tokens and must still be costed."""
    from forecaster import providers

    class _FakeResponse:
        ok = True
        status_code = 200
        headers: dict = {}
        text = ""

        @staticmethod
        def json() -> dict:
            return {
                "content": [],
                "stop_reason": "refusal",
                "stop_details": {"category": "cyber"},
                "usage": {"input_tokens": 1200, "output_tokens": 0},
            }

    monkeypatch.setattr(providers, "_ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(providers.requests, "post", lambda *a, **k: _FakeResponse())

    result = providers.call_anthropic("hi", "claude-opus-5", 0.2, purpose="spd_v2")
    assert result.error is not None and "refusal" in result.error.lower()
    assert result.text == ""
    assert result.usage["prompt_tokens"] == 1200


def test_spd_anthropic_budget_exceeds_default_output_cap() -> None:
    """Thinking shares the max_tokens budget from Opus 5, so SPD needs headroom."""
    from forecaster import providers

    assert providers._ANTHROPIC_SPD_MAX_OUTPUT >= 32768
    assert providers._ANTHROPIC_SPD_MAX_OUTPUT > providers._ANTHROPIC_MAX_OUTPUT


def test_current_lineup_models_are_priced() -> None:
    """A missing cost entry silently logs $0 and breaks Sibyl's budget cap."""
    from forecaster import providers

    for model_id in ("gpt-5.6-sol", "gpt-5.6-luna", "claude-opus-5"):
        assert providers.resolve_price_per_1m(model_id) is not None, model_id


def test_openai_drops_temperature_for_gpt56_lineup() -> None:
    from forecaster import providers

    assert providers._openai_drops_temperature("gpt-5.6-sol")
    assert providers._openai_drops_temperature("gpt-5.6-luna")
