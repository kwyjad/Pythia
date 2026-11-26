# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The December 2026 lineup: GPT-6.1 Sol, Gemini 4 Argon, Gemini 3.8 Flash.

Argon is a new Gemini generation, and the Google provider was written for
3.x. Each of these was a failure that raises no error:

* The thinking gate matched "gemini-3" only, so Argon's ``thinking: medium``
  would have been dropped from every request.
* Family classification looked for "flash"/"pro" in the id, so Argon got no
  SPD timeout and no env thinking fallback.
* Argon accepts low|medium|high only; "minimal" is a 400 on every call.
* Argon ignores sampling params: sending temperature claims a setting nobody
  applied.
* Argon's output ceiling is 32,768 and thinking counts against it.
* Batch bodies must equal sync bodies byte for byte.
"""

from __future__ import annotations

import dataclasses

import pytest

from forecaster import providers
from forecaster.providers import ModelSpec


def _members():
    return {ms.model_id: ms for ms in providers._load_ensemble_from_config()}


def test_the_ensemble_runs_the_december_lineup() -> None:
    got = {mid: ms.thinking for mid, ms in _members().items()}
    assert got == {
        "gpt-6.1-sol": "high",
        "claude-opus-5-5": "high",
        "gemini-4-argon": "medium",
        "gemini-3.8-flash": "low",
        "gpt-6-luna": "medium",
    }


def test_argon_carries_no_temperature_in_config() -> None:
    assert _members()["gemini-4-argon"].temperature is None


def test_new_ids_join_their_families() -> None:
    from pythia.llm_profiles import model_family

    assert model_family("gpt-6.1-sol") == "gpt"
    assert model_family("gemini-4-argon") == "gemini_pro"
    assert model_family("gemini-3.8-flash") == "gemini_flash"


# --- Argon request body -----------------------------------------------------


def test_argon_body_has_thinking_and_ceiling_and_no_temperature() -> None:
    body = providers.build_google_body("P", "gemini-4-argon", 0.2, thinking_level="medium")
    gc = body["generationConfig"]
    assert gc["thinkingConfig"] == {"thinkingLevel": "medium"}
    assert gc["maxOutputTokens"] == 32768
    assert "temperature" not in gc
    assert "thinkingBudget" not in str(body)
    assert "cachedContent" not in str(body) and "cached_content" not in str(body)


def test_argon_ceiling_is_env_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHIA_GOOGLE_ARGON_MAX_OUTPUT_TOKENS", "16000")
    body = providers.build_google_body("P", "gemini-4-argon", None, thinking_level="high")
    assert body["generationConfig"]["maxOutputTokens"] == 16000


@pytest.mark.parametrize("level", ["minimal", "xhigh", "max"])
def test_argon_rejects_unsupported_levels(level: str) -> None:
    with pytest.raises(ValueError):
        providers.build_google_body("P", "gemini-4-argon", None, thinking_level=level)
    ms = ModelSpec(name="a", provider="google", model_id="gemini-4-argon", thinking=level)
    with pytest.raises(ValueError):
        providers.resolve_request_params(ms, 0.2)


def test_argon_minimal_from_the_env_fallback_fails_before_a_body(monkeypatch) -> None:
    monkeypatch.setenv("PYTHIA_GOOGLE_SPD_THINKING_LEVEL_PRO", "minimal")
    ms = ModelSpec(name="a", provider="google", model_id="gemini-4-argon", purpose="spd_v2")
    with pytest.raises(ValueError):
        providers.build_body_for_spec(ms, "P", 0.2)


def test_argon_minimal_in_config_fails_at_load(monkeypatch: pytest.MonkeyPatch) -> None:
    import pythia.llm_profiles as lp

    monkeypatch.setattr(
        lp, "get_ensemble_resolved",
        lambda *a, **k: [{"provider": "google", "model_id": "gemini-4-argon", "thinking": "minimal"}],
    )
    with pytest.raises(ValueError):
        providers._load_ensemble_from_config()


def test_argon_classifies_as_pro_with_its_own_timeout(monkeypatch) -> None:
    assert providers._google_model_family("gemini-4-argon") == "pro"
    assert providers._google_model_family("google/gemini-4-argon") == "pro"
    assert providers._supports_thinking_level("gemini-4-argon")
    assert providers._google_spd_timeout("gemini-4-argon") == 1200.0
    monkeypatch.setenv("PYTHIA_GOOGLE_ARGON_TIMEOUT_SEC", "1500")
    assert providers._google_spd_timeout("gemini-4-argon") == 1500.0
    # Gemini 3.x pro/flash keep their 300 s defaults.
    assert providers._google_spd_timeout("gemini-3.8-flash") == 300.0


def test_gemini_25_stays_outside_the_thinking_level_gate() -> None:
    assert not providers._supports_thinking_level("gemini-2.5-flash")
    body = providers.build_google_body("P", "gemini-2.5-flash", 0.2, thinking_level="low")
    assert "thinkingConfig" not in body["generationConfig"]
    assert providers._google_model_family("gemini-2.5-flash") is None


def test_argon_temperature_is_reported_as_none() -> None:
    ms = ModelSpec(name="a", provider="google", model_id="gemini-4-argon",
                   temperature=0.2, thinking="medium")
    temp, level = providers.resolve_request_params(ms, 0.2)
    assert temp is None and level == "medium"


def test_hs_logging_records_no_temperature_for_argon() -> None:
    from horizon_scanner import llm_logging as hsl

    src = open(hsl.__file__).read()
    assert "google_drops_sampling" in src


# --- 3.8 Flash ---------------------------------------------------------------


def test_flash_38_body_is_the_36_body_apart_from_the_id() -> None:
    a = providers.build_google_body("P", "gemini-3.6-flash", 0.2, thinking_level="low")
    b = providers.build_google_body("P", "gemini-3.8-flash", 0.2, thinking_level="low")
    assert a == b
    assert providers._google_model_family("gemini-3.8-flash") == "flash"


# --- GPT-6.1 Sol --------------------------------------------------------------


@pytest.mark.parametrize("effort", ["none", "off", None])
def test_gpt61_never_receives_effort_none_or_temperature(effort) -> None:
    body = providers.build_openai_body("P", "gpt-6.1-sol", 0.2, reasoning_effort=effort)
    assert "reasoning_effort" not in body
    assert "temperature" not in body


def test_gpt61_none_from_config_is_dropped() -> None:
    ms = ModelSpec(name="g", provider="openai", model_id="gpt-6.1-sol", thinking="none")
    body = providers.build_body_for_spec(ms, "P", 0.2)
    assert "reasoning_effort" not in body and "temperature" not in body


def test_grounding_openai_uses_the_responses_api() -> None:
    # GPT-6.1 Sol needs the Responses API for tool calling; the web-search
    # backend already uses it (client.responses.create with a web_search tool).
    import pythia.web_research.backends.openai_web_search as ws

    src = open(ws.__file__).read()
    assert "client.responses.create(" in src
    assert "chat.completions" not in src


# --- batch == sync ----------------------------------------------------------


@pytest.mark.parametrize("model_id", ["gemini-4-argon", "gemini-3.8-flash"])
def test_google_batch_body_equals_sync_body(monkeypatch, model_id: str) -> None:
    ms = dataclasses.replace(_members()[model_id], purpose="spd_v2")
    batch = providers.build_body_for_spec(ms, "PROMPT", 0.2)

    sent = {}

    class _Resp:
        status_code = 200
        headers: dict = {}

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "{}"}]}, "finishReason": "STOP"}]}

    def _post(url, headers=None, json=None, timeout=None):
        sent["body"] = json
        return _Resp()

    monkeypatch.setattr(providers, "_GEMINI_API_KEY", "k")
    monkeypatch.setattr(providers.requests, "post", _post)
    temp, level = providers.resolve_request_params(ms, 0.2)
    providers.call_google("PROMPT", ms.model_id, temp, thinking_level=level)
    assert sent["body"] == batch


def test_gpt61_batch_body_equals_sync_body() -> None:
    ms = dataclasses.replace(_members()["gpt-6.1-sol"], purpose="spd_v2")
    batch = providers.build_body_for_spec(ms, "PROMPT", 0.2)
    sync = providers.build_openai_body(
        "PROMPT", ms.model_id, 0.2, reasoning_effort=ms.thinking,
        max_completion_tokens=providers.openai_max_output_tokens("spd_v2"),
    )
    assert batch == {k: v for k, v in sync.items() if k != "prompt_cache_key"}


# --- prices -------------------------------------------------------------------


def test_new_prices(monkeypatch: pytest.MonkeyPatch) -> None:
    from forecaster.providers import resolve_price_detail

    assert resolve_price_detail("gpt-6.1-sol")["cached_input"] == pytest.approx(0.10)
    argon = resolve_price_detail("gemini-4-argon")
    assert (argon["input"], argon["output"], argon["cached_input"]) == (4.0, 20.0, 0.2)
    import json
    from pathlib import Path

    from forecaster.providers import _parse_model_costs

    raw = json.loads((Path(providers.__file__).resolve().parents[1] / "pythia" / "model_costs.json").read_text())
    monkeypatch.setenv("PYTHIA_PRICING_DATE", "2026-12-15")
    pairs, _ = _parse_model_costs(raw)
    assert pairs["gemini-3.8-flash"] == (0.75, 3.75)
    monkeypatch.setenv("PYTHIA_PRICING_DATE", "2027-01-02")
    pairs, _ = _parse_model_costs(raw)
    assert pairs["gemini-3.8-flash"] == (1.50, 7.50)
