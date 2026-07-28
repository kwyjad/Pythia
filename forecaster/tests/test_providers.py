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


# ---------------------------------------------------------------------------
# Cache/batch telemetry + pricing (Phase 0 of the cost-optimization work)
# ---------------------------------------------------------------------------


def test_usage_to_dict_preserves_cache_and_batch_fields() -> None:
    """Provider cache/batch usage fields must survive usage_to_dict.

    Before this passthrough existed, cache_read_input_tokens etc. were silently
    dropped, so cache hits were invisible AND overbilled in the ledger.
    """
    from forecaster import providers

    usage = providers.usage_to_dict(
        {
            "prompt_tokens": 5000,
            "completion_tokens": 700,
            "total_tokens": 5700,
            "cache_read_input_tokens": 3000,
            "cache_creation_input_tokens": 1000,
            "cached_tokens": 0,
            "service_tier": "batch",
            "batch_id": "b_x",
            "provider_batch_id": "msgbatch_1",
        }
    )
    assert usage["prompt_tokens"] == 5000
    assert usage["cache_read_input_tokens"] == 3000
    assert usage["cache_creation_input_tokens"] == 1000
    assert usage["service_tier"] == "batch"
    assert usage["batch_id"] == "b_x"
    assert usage["provider_batch_id"] == "msgbatch_1"
    # Zero-valued int fields are not required to persist; core keys always do.
    assert usage["total_tokens"] == 5700


def test_anthropic_usage_normalizes_prompt_tokens_to_total() -> None:
    """Anthropic input_tokens EXCLUDES cache reads/writes; ours must not."""
    from forecaster import providers

    usage = providers._anthropic_usage_from_payload(
        {
            "usage": {
                "input_tokens": 400,
                "output_tokens": 900,
                "cache_read_input_tokens": 2500,
                "cache_creation_input_tokens": 100,
            }
        }
    )
    assert usage["prompt_tokens"] == 3000
    assert usage["cache_read_input_tokens"] == 2500
    assert usage["cache_creation_input_tokens"] == 100
    assert usage["completion_tokens"] == 900
    assert usage["total_tokens"] == 3900


def test_openai_and_google_usage_extract_cached_tokens() -> None:
    from forecaster import providers

    openai_usage = providers._openai_usage_from_payload(
        {
            "usage": {
                "prompt_tokens": 4000,
                "completion_tokens": 500,
                "total_tokens": 4500,
                "prompt_tokens_details": {"cached_tokens": 2048},
            }
        }
    )
    assert openai_usage["prompt_tokens"] == 4000
    assert openai_usage["cached_tokens"] == 2048

    google_usage = providers._google_usage_from_payload(
        {
            "usageMetadata": {
                "promptTokenCount": 6000,
                "candidatesTokenCount": 800,
                "totalTokenCount": 6800,
                "cachedContentTokenCount": 4096,
            }
        }
    )
    assert google_usage["prompt_tokens"] == 6000
    assert google_usage["cached_tokens"] == 4096


def test_compute_cost_split_prices_cached_tokens_at_cached_rate() -> None:
    """Cached portions bill at the cached rate, writes at the write premium."""
    from forecaster import providers

    # claude-opus-5 object entry: input 5, output 25, cached 0.50, write_5m 6.25
    usage = {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 0,
        "cache_read_input_tokens": 500_000,
        "cache_creation_input_tokens": 100_000,
    }
    input_cost, output_cost, total = providers.compute_cost_split_usd("claude-opus-5", usage)
    # 400k uncached * $5 + 500k cache-read * $0.50 + 100k write * $6.25 (per 1M)
    assert input_cost == pytest.approx(0.4 * 5.0 + 0.5 * 0.50 + 0.1 * 6.25)
    assert output_cost == 0.0
    assert total == pytest.approx(input_cost)


def test_compute_cost_split_applies_batch_discount() -> None:
    from forecaster import providers

    usage = {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 1_000_000,
        "service_tier": "batch",
    }
    # gemini-3.5-flash: $1.50 in / $9.00 out, halved by the batch tier.
    _, _, total = providers.compute_cost_split_usd("gemini-3.5-flash", usage)
    assert total == pytest.approx((1.50 + 9.00) * 0.5)

    # Sync pricing is unchanged by the helper.
    usage_sync = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    _, _, total_sync = providers.compute_cost_split_usd("gemini-3.5-flash", usage_sync)
    assert total_sync == pytest.approx(10.50)


def test_resolve_price_detail_defaults_for_legacy_array_entries() -> None:
    """2-array entries get provider-default cached rates (never None)."""
    from forecaster import providers

    detail = providers.resolve_price_detail("claude-opus-4-8")
    assert detail is not None
    assert detail["input"] == pytest.approx(5.00)
    assert detail["cached_input"] == pytest.approx(0.50)   # 0.1x default
    assert detail["cache_write_5m"] == pytest.approx(6.25)  # 1.25x default

    gem = providers.resolve_price_detail("gemini-2.5-flash")
    assert gem is not None
    assert gem["cached_input"] == pytest.approx(0.30 * 0.25)


def test_body_builders_match_sync_call_bodies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batch payloads must be byte-identical to sync payloads.

    The sync callers and the batch layer share the same body builders; this
    pins that the sync path actually sends the builder output unmodified.
    """
    import json as _json

    from forecaster import providers

    captured: dict = {}

    class _FakeResp:
        ok = True
        status_code = 200
        headers: dict = {}
        text = "{}"

        def json(self):
            return {"choices": [{"message": {"content": "ok"}}], "usage": {}}

    def _fake_post(url, headers=None, json=None, timeout=None):
        captured["body"] = json
        return _FakeResp()

    monkeypatch.setattr(providers, "_OPENAI_API_KEY", "k")
    monkeypatch.setattr(providers.requests, "post", _fake_post)
    providers.call_openai("p", "gpt-5.6-sol", 0.3, reasoning_effort="high")
    assert captured["body"] == providers.build_openai_body(
        "p", "gpt-5.6-sol", 0.3, reasoning_effort="high"
    )
    assert "temperature" not in captured["body"]
    assert captured["body"]["reasoning_effort"] == "high"

    class _FakeAnthropicResp(_FakeResp):
        def json(self):
            return {"content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1, "output_tokens": 1}}

    def _fake_post_anthropic(url, headers=None, json=None, timeout=None):
        captured["body"] = json
        return _FakeAnthropicResp()

    monkeypatch.setattr(providers, "_ANTHROPIC_API_KEY", "k")
    monkeypatch.setattr(providers.requests, "post", _fake_post_anthropic)
    providers.call_anthropic("p", "claude-opus-5", 0.7, purpose="spd_v2")
    assert captured["body"] == providers.build_anthropic_body("p", "claude-opus-5", 0.7, purpose="spd_v2")
    assert captured["body"]["max_tokens"] == providers._ANTHROPIC_SPD_MAX_OUTPUT
    assert "temperature" not in captured["body"]

    class _FakeGoogleResp(_FakeResp):
        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "ok"}]}}], "usageMetadata": {}}

    def _fake_post_google(url, json=None, timeout=None):
        captured["body"] = json
        return _FakeGoogleResp()

    monkeypatch.setattr(providers, "_GEMINI_API_KEY", "k")
    monkeypatch.setattr(providers.requests, "post", _fake_post_google)
    providers.call_google("p", "gemini-3.5-flash", 0.2, thinking_level="low")
    assert captured["body"] == providers.build_google_body("p", "gemini-3.5-flash", 0.2, thinking_level="low")
    assert captured["body"]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}
    # Round-trip sanity: bodies are JSON-serializable deterministically.
    _json.dumps(captured["body"], sort_keys=True)


def test_google_usage_bills_thinking_tokens() -> None:
    """thoughtsTokenCount is billed at the output rate but reported separately
    from candidatesTokenCount — it must fold into completion_tokens or every
    thinking Gemini call (both SPD members + all HS RC/triage) underbills."""

    from forecaster.providers import _google_usage_from_payload

    payload = {
        "usageMetadata": {
            "promptTokenCount": 1000,
            "candidatesTokenCount": 200,
            "thoughtsTokenCount": 700,
            "totalTokenCount": 1900,
        }
    }
    usage = _google_usage_from_payload(payload)
    assert usage["completion_tokens"] == 900  # 200 answer + 700 thinking
    assert usage["thoughts_tokens"] == 700
    assert usage["prompt_tokens"] == 1000


def test_openai_temperature_guard_is_case_insensitive() -> None:
    from forecaster.providers import _openai_drops_temperature

    assert _openai_drops_temperature("GPT-5.6-sol")
    assert _openai_drops_temperature("Gpt-5.6-Luna")
    assert not _openai_drops_temperature("gpt-4.1")


def test_sibyl_step_gets_raised_anthropic_ceiling() -> None:
    """Sibyl runs Opus 5 (adaptive thinking shares max_tokens) and each step
    answer carries a full quantile JSON — the 16k default truncates exactly
    like SPD did, so sibyl_step must get the raised ceiling too."""

    from forecaster.providers import (
        _ANTHROPIC_SPD_MAX_OUTPUT,
        build_anthropic_body,
    )

    body = build_anthropic_body("p", "claude-opus-5", 1.0, purpose="sibyl_step")
    assert body["max_tokens"] == _ANTHROPIC_SPD_MAX_OUTPUT
    assert _ANTHROPIC_SPD_MAX_OUTPUT >= 32768


def test_combine_usage_sums_cache_fields_and_drops_mixed_tier() -> None:
    """compute_cost_split_usd derives the uncached span as prompt_tokens −
    cache fields; a merged usage that sums prompt tokens but inherits only
    the base call's cache counts prices the second call's cached span at the
    full input rate — and an inherited service_tier=batch would halve the
    sync half's price."""

    from forecaster.self_search import combine_usage

    base = {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "total_tokens": 1100,
        "cache_read_input_tokens": 800,
        "service_tier": "batch",
    }
    extra = {
        "prompt_tokens": 2000,
        "completion_tokens": 300,
        "total_tokens": 2300,
        "cache_read_input_tokens": 1500,
    }
    combined = combine_usage(base, extra)
    assert combined["prompt_tokens"] == 3000
    assert combined["cache_read_input_tokens"] == 2300
    assert "service_tier" not in combined  # mixed tiers never inherit batch

    both_batch = combine_usage(base, {**extra, "service_tier": "batch"})
    assert both_batch["service_tier"] == "batch"


def test_stop_reason_errors_do_not_feed_provider_breaker(monkeypatch) -> None:
    """Six consecutive refusals/truncations are content-driven outcomes, not
    provider-health signals — they must not trip a whole-provider cooldown
    that drops the Claude member from unrelated questions."""

    import asyncio

    import forecaster.providers as providers

    ms = providers.ModelSpec(
        name="claude-opus-5", provider="anthropic",
        model_id="claude-opus-5", active=True, purpose="spd_v2",
    )

    def _fake_sync(*args, **kwargs):
        return providers.ProviderResult(
            "", providers.usage_to_dict({"prompt_tokens": 10, "completion_tokens": 5}),
            0.0, "claude-opus-5",
            error="Anthropic refusal: request declined by safety classifiers (category=cyber)",
        )

    monkeypatch.setattr(providers, "_call_provider_sync", _fake_sync)
    noted: list = []
    monkeypatch.setattr(
        providers, "_note_provider_failure",
        lambda *a, **k: noted.append(a) or {"consecutive_failures": 1, "cooldown_until_ts": 0.0},
    )
    text, usage, error = asyncio.run(
        providers.call_chat_ms(ms, "p", log_call=False, run_id="r1")
    )
    assert "refusal" in (error or "")
    assert noted == []  # breaker never fed
    # Usage is preserved on the error result (the tokens were billed).
    assert usage["prompt_tokens"] >= 10
