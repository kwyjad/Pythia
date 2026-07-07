# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Guards for the central model registry (config.yaml llm.models + roles).

These tests enforce the "swap a model in one place" contract:
- every registry alias parses and uses a known provider;
- every role resolves through the registry;
- every model the system can call has a cost entry in model_costs.json
  (otherwise its calls silently log $0);
- the ensemble loads with aliases resolved;
- llm_logging's cost split is derived from the same table as
  providers.estimate_cost_usd (no second pricing table).
"""

from __future__ import annotations

import pytest

from pythia.llm_profiles import (
    _ROLE_FALLBACKS,
    get_ensemble_resolved,
    get_model_registry,
    get_role_model,
    resolve_model_ref,
    split_model_ref,
)

KNOWN_PROVIDERS = {"openai", "anthropic", "google"}

# Roles the codebase actually consults (call sites in horizon_scanner,
# forecaster, web_research backends). Adding a call site for a new role?
# Add it here and to config.yaml + _ROLE_FALLBACKS.
CONSUMED_ROLES = [
    "hs_default",
    "hs_triage_pass1",
    "hs_triage_pass2",
    "rc_pass1",
    "rc_pass2",
    "hs_fallback",
    "track2_spd",
    "scenario_writer",
    "grounding_gemini",
    "grounding_openai",
    "grounding_openai_fallback",
    "grounding_claude",
    "crisiswatch",
]


def test_registry_aliases_parse_with_known_providers():
    registry = get_model_registry()
    assert registry, "llm.models registry missing from config.yaml"
    for alias, ref in registry.items():
        assert ":" in ref, f"registry alias {alias!r} must be provider:model_id, got {ref!r}"
        provider, model_id = split_model_ref(ref)
        assert provider in KNOWN_PROVIDERS, f"alias {alias!r} uses unknown provider {provider!r}"
        assert model_id, f"alias {alias!r} has empty model_id"


def test_every_consumed_role_resolves():
    for role in CONSUMED_ROLES:
        ref = get_role_model(role)
        assert ref and ":" in ref, f"role {role!r} did not resolve (got {ref!r})"
        provider, model_id = split_model_ref(ref)
        assert provider in KNOWN_PROVIDERS and model_id, f"role {role!r} -> bad ref {ref!r}"


def test_role_fallbacks_cover_consumed_roles():
    missing = [r for r in CONSUMED_ROLES if r not in _ROLE_FALLBACKS]
    assert not missing, f"_ROLE_FALLBACKS missing code-level defaults for: {missing}"


def test_ensemble_resolves_with_params():
    ensemble = get_ensemble_resolved()
    assert len(ensemble) >= 2, "ensemble should have multiple members"
    for entry in ensemble:
        assert entry["provider"] in KNOWN_PROVIDERS
        assert entry["model_id"]


def test_resolve_model_ref_forms():
    assert resolve_model_ref("openai:gpt-x") == "openai:gpt-x"  # explicit passthrough
    registry = get_model_registry()
    alias = next(iter(registry))
    assert resolve_model_ref(alias) == registry[alias]
    assert resolve_model_ref("not-a-real-alias") is None
    assert resolve_model_ref("") is None
    assert resolve_model_ref(None) is None


def test_all_reachable_models_have_cost_entries():
    """Any model the system can call must have a price in model_costs.json."""
    from forecaster.providers import resolve_price_per_1m

    reachable: dict[str, str] = {}
    for alias, ref in get_model_registry().items():
        reachable[split_model_ref(ref)[1]] = f"registry alias {alias!r}"
    for role in CONSUMED_ROLES:
        reachable[split_model_ref(get_role_model(role))[1]] = f"role {role!r}"
    for entry in get_ensemble_resolved():
        reachable[entry["model_id"]] = "ensemble member"

    missing = {
        model_id: origin
        for model_id, origin in reachable.items()
        if resolve_price_per_1m(model_id) is None
    }
    assert not missing, (
        "models without a cost entry in pythia/model_costs.json "
        f"(their calls would log $0): {missing}"
    )


def test_llm_logging_cost_split_matches_cost_table():
    """llm_logging must derive its input/output split from model_costs.json.

    The cost table is USD per 1M tokens, so a 1M-in / 1M-out call costs
    exactly (input_rate + output_rate).
    """
    from forecaster.llm_logging import _compute_costs_for_usage
    from forecaster.providers import resolve_price_per_1m

    for entry in get_ensemble_resolved():
        model_id = entry["model_id"]
        prices = resolve_price_per_1m(model_id)
        assert prices is not None, f"{model_id} missing from model_costs.json"
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
        input_cost, output_cost, total = _compute_costs_for_usage(
            entry["provider"], model_id, usage
        )
        assert input_cost == pytest.approx(prices[0])
        assert output_cost == pytest.approx(prices[1])
        assert total == pytest.approx(prices[0] + prices[1])
        assert total > 0.0, f"{model_id} cost split is zero — pricing drift"


def test_env_override_beats_role(monkeypatch):
    """Purpose env vars must still win over config roles (documented contract)."""
    from horizon_scanner._utils import resolve_hs_model

    monkeypatch.setenv("HS_MODEL_ID", "test-model-override")
    assert resolve_hs_model() == "test-model-override"
    monkeypatch.delenv("HS_MODEL_ID")
    ref = get_role_model("hs_default")
    assert resolve_hs_model() == split_model_ref(ref)[1]
