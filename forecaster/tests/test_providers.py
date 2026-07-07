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
