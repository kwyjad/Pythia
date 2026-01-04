import asyncio
from typing import Any

import pytest

from forecaster import providers
from horizon_scanner import horizon_scanner


class DummyConn:
    def close(self) -> None:
        return None


def test_hs_triage_retries_timeouts(monkeypatch):
    calls = {"count": 0}

    def fake_call_provider_sync(
        provider: str,
        prompt: str,
        model: str,
        temperature: float,
        *,
        timeout_sec: float | None = None,
        thinking_level: str | None = None,
    ) -> providers.ProviderResult:
        calls["count"] += 1
        if calls["count"] == 1:
            return providers.ProviderResult(
                "",
                providers.usage_to_dict(None),
                0.0,
                model,
                error=f"timeout after {timeout_sec}s",
            )
        return providers.ProviderResult("ok", {"total_tokens": 1}, 0.0, model)

    monkeypatch.setattr(providers, "_call_provider_sync", fake_call_provider_sync)
    monkeypatch.setenv("PYTHIA_HS_LLM_MAX_ATTEMPTS", "2")
    ms = providers.ModelSpec(
        name="Gemini",
        provider="google",
        model_id="gemini-test",
        active=True,
        purpose="hs_triage",
    )

    text, usage, error = asyncio.run(
        providers.call_chat_ms(
            ms,
            "prompt",
            prompt_key="hs.triage.v2",
            prompt_version="1.0.0",
            component="Test",
        )
    )

    assert text == "ok"
    assert error == ""
    assert calls["count"] == 2
    assert usage["attempts_used"] == 2


def test_hs_triage_fallback_used(monkeypatch):
    async def fake_call_chat_ms(
        ms: providers.ModelSpec,
        prompt: str,
        temperature: float = 0.2,
        *,
        prompt_key: str = "",
        prompt_version: str | None = None,
        component: str = "",
        run_id: str | None = None,
    ) -> tuple[str, dict[str, Any], str]:
        if ms.provider == "google":
            return "", {"total_tokens": 0}, "timeout after 60s"
        return "{\"hazards\":{}}", {"total_tokens": 1}, ""

    monkeypatch.setattr(horizon_scanner, "call_chat_ms", fake_call_chat_ms)
    fallback_specs = [
        providers.ModelSpec(
            name="OpenAI",
            provider="openai",
            model_id="gpt-5.1",
            active=True,
            purpose="hs_triage",
        )
    ]

    text, usage, error, spec = asyncio.run(
        horizon_scanner._call_hs_model(
            "prompt",
            run_id="hs_test",
            fallback_specs=fallback_specs,
        )
    )

    assert error == ""
    assert text
    assert spec.provider == "openai"
    assert usage["fallback_used"] is True


def test_hs_triage_json_repair(monkeypatch, tmp_path):
    async def fake_call_chat_ms(
        ms: providers.ModelSpec,
        prompt: str,
        temperature: float = 0.2,
        *,
        prompt_key: str = "",
        prompt_version: str | None = None,
        component: str = "",
        run_id: str | None = None,
    ) -> tuple[str, dict[str, Any], str]:
        if prompt_key == "hs.triage.json_repair":
            return "{\"hazards\":{\"ACE\":{\"triage_score\":0.5}}}", {"total_tokens": 1}, ""
        return "not json", {"total_tokens": 1}, ""

    monkeypatch.setattr(horizon_scanner, "call_chat_ms", fake_call_chat_ms)
    monkeypatch.setattr(horizon_scanner, "_build_resolver_features_for_country", lambda *_: {})
    monkeypatch.setattr(horizon_scanner, "_maybe_build_country_evidence_pack", lambda *_: None)
    monkeypatch.setattr(horizon_scanner, "_write_hs_triage", lambda *_: None)
    monkeypatch.setattr(horizon_scanner, "log_hs_llm_call", lambda **_: None)
    monkeypatch.setattr(horizon_scanner, "pythia_connect", lambda **_: DummyConn())
    monkeypatch.chdir(tmp_path)

    fallback_specs = [
        providers.ModelSpec(
            name="OpenAI",
            provider="openai",
            model_id="gpt-5.1",
            active=True,
            purpose="hs_triage",
        )
    ]
    horizon_scanner._HS_FALLBACK_SPECS = fallback_specs

    result = horizon_scanner._run_hs_for_country("hs_test", "USA", "United States")

    assert result["final_status"] in {"ok", "degraded"}
    assert result["pass_results"][0]["json_repair_used"] is True


def test_hs_triage_rerun_lists(monkeypatch, capsys, tmp_path):
    fallback_specs = [
        providers.ModelSpec(
            name="OpenAI",
            provider="openai",
            model_id="gpt-5.1",
            active=True,
            purpose="hs_triage",
        )
    ]
    monkeypatch.setattr(horizon_scanner, "_resolve_hs_fallback_specs", lambda: fallback_specs)
    monkeypatch.setattr(horizon_scanner, "ensure_schema", lambda: None)
    monkeypatch.setattr(horizon_scanner, "log_hs_run_to_db", lambda *_, **__: None)
    monkeypatch.setattr(
        horizon_scanner,
        "_load_country_list",
        lambda *_: ([("Foo", "AAA"), ("Bar", "BBB"), ("Baz", "CCC")], [], ["AAA", "BBB", "CCC"]),
    )

    def fake_run(run_id: str, iso3: str, country_name: str):
        status_map = {"AAA": "ok", "BBB": "degraded", "CCC": "failed"}
        status = status_map[iso3]
        return {
            "iso3": iso3,
            "error_text": None,
            "response_text": "",
            "pass_results": [],
            "final_status": status,
            "pass1_status": "ok",
            "pass2_status": "ok",
            "pass1_valid": status != "failed",
            "pass2_valid": status == "ok",
            "primary_model_id": "gemini-test",
            "fallback_model_id": "gpt-5.1",
        }

    monkeypatch.setattr(horizon_scanner, "_run_hs_for_country", fake_run)
    monkeypatch.chdir(tmp_path)

    horizon_scanner.main(countries=["AAA", "BBB", "CCC"])

    stdout = capsys.readouterr().out
    assert "HS_TRIAGE_RERUN_ISO3S=BBB,CCC" in stdout
