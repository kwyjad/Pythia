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
            "XAI_API_KEY": "x",
            "PYTHIA_BLOCK_PROVIDERS": "xai",
        },
    )

    specs = providers.parse_ensemble_specs(
        "openai:gpt-5.1,xai:grok-4-0709,google:gemini-3-pro-preview"
    )

    assert all(ms.provider != "xai" for ms in specs)
    assert any(ms.provider == "openai" for ms in specs)


def test_spd_ensemble_override_keeps_two_gemini_models(monkeypatch: pytest.MonkeyPatch) -> None:
    providers = _reload_providers(
        monkeypatch,
        {
            "OPENAI_API_KEY": "x",
            "ANTHROPIC_API_KEY": "x",
            "GEMINI_API_KEY": "x",
            "PYTHIA_SPD_ENSEMBLE_SPECS": (
                "openai:gpt-5.1,anthropic:claude-opus-4-5-20251101,"
                "google:gemini-3-pro-preview,google:gemini-3-flash-preview"
            ),
            "PYTHIA_BLOCK_PROVIDERS": "",
        },
    )

    google_specs = [ms for ms in providers.SPD_ENSEMBLE if ms.provider == "google"]

    assert len(google_specs) == 2
    assert {ms.model_id for ms in google_specs} == {"gemini-3-pro-preview", "gemini-3-flash-preview"}
