import asyncio
import sys
import types

duckdb_stub = sys.modules.setdefault("duckdb", types.ModuleType("duckdb"))
if not hasattr(duckdb_stub, "CatalogException"):
    duckdb_stub.CatalogException = type("CatalogException", (Exception,), {})

from forecaster import providers
from forecaster.providers import ModelSpec, ProviderResult, usage_to_dict


def _make_model_spec(purpose: str | None) -> ModelSpec:
    return ModelSpec(
        name="Gemini",
        provider="google",
        model_id="gemini-3-flash-preview",
        active=True,
        purpose=purpose,
    )


def test_hs_retry_after_fail_fast(monkeypatch):
    monkeypatch.setenv("PYTHIA_HS_LLM_MAX_RETRY_AFTER_SEC", "10")
    monkeypatch.setenv("PYTHIA_HS_LLM_FAIL_FAST_ON_RETRY_AFTER", "1")
    monkeypatch.setenv("PYTHIA_HS_LLM_MAX_ATTEMPTS", "2")

    sleep_calls = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    def fake_call_provider_sync(
        provider: str,
        prompt: str,
        model: str,
        temperature: float,
        *,
        timeout_sec: float | None = None,
        thinking_level: str | None = None,
    ) -> ProviderResult:
        return ProviderResult(
            "",
            usage_to_dict(None),
            0.0,
            model,
            error="Gemini HTTP 429",
            retry_after=800,
        )

    monkeypatch.setattr(providers.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(providers, "_call_provider_sync", fake_call_provider_sync)

    ms = _make_model_spec("hs_triage")
    text, usage, error = asyncio.run(providers.call_chat_ms(ms, "prompt"))

    assert text == ""
    assert error
    assert sleep_calls == []
    assert usage.get("retry_after_hint_sec") == 800
    assert usage.get("retry_after_used_sec") == 0.0
    assert usage.get("retry_after_capped") is True


def test_hs_timeout_wiring(monkeypatch):
    monkeypatch.setenv("PYTHIA_HS_GEMINI_TIMEOUT_SEC", "77")

    captured = {}

    def fake_call_provider_sync(
        provider: str,
        prompt: str,
        model: str,
        temperature: float,
        *,
        timeout_sec: float | None = None,
        thinking_level: str | None = None,
    ) -> ProviderResult:
        captured["timeout_sec"] = timeout_sec
        return ProviderResult("ok", usage_to_dict(None), 0.0, model)

    monkeypatch.setattr(providers, "_call_provider_sync", fake_call_provider_sync)

    ms = _make_model_spec("hs_triage")
    text, usage, error = asyncio.run(providers.call_chat_ms(ms, "prompt"))

    assert text == "ok"
    assert error == ""
    assert captured["timeout_sec"] == 77.0
    assert usage.get("hs_timeout_sec") == 77.0


def test_non_hs_retry_after_unclamped(monkeypatch):
    monkeypatch.setenv("PYTHIA_LLM_RETRIES", "2")

    sleep_calls = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    def fake_call_provider_sync(
        provider: str,
        prompt: str,
        model: str,
        temperature: float,
        *,
        timeout_sec: float | None = None,
        thinking_level: str | None = None,
    ) -> ProviderResult:
        return ProviderResult(
            "",
            usage_to_dict(None),
            0.0,
            model,
            error="Gemini HTTP 429",
            retry_after=800,
        )

    monkeypatch.setattr(providers.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(providers, "_call_provider_sync", fake_call_provider_sync)

    ms = _make_model_spec(None)
    text, usage, error = asyncio.run(providers.call_chat_ms(ms, "prompt"))

    assert text == ""
    assert error
    assert len(sleep_calls) == 1
    assert 20.0 <= sleep_calls[0] <= 20.5  # 20s global cap + jitter
