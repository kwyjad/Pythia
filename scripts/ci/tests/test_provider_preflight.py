# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The provider preflight (Oct 2026) and the terminal-refusal path."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from forecaster import providers
from scripts.ci import provider_preflight as pf

LIMIT = ("Anthropic HTTP 400: You have reached your specified API usage limits. "
         "You will regain access on 2026-11-01 at 00:00 UTC.")
REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _clean():
    providers.reset_terminal_providers()
    yield
    providers.reset_terminal_providers()


def test_the_13_october_error_is_a_usage_limit():
    assert providers.classify_provider_error("anthropic", LIMIT) == "usage_limit"
    assert providers.classify_provider_error("anthropic", "HTTP 401: invalid x-api-key") == "invalid_key"
    assert providers.classify_provider_error(
        "openai", "HTTP 429: You exceeded your current quota, insufficient_quota") == "billing"
    assert providers.classify_provider_error("openai", "HTTP 429: Rate limit reached") is None
    assert providers.classify_provider_error("google", "HTTP 503: overloaded") is None
    assert pf.regain_at(LIMIT) == "2026-11-01T00:00Z"


def _fake(answers):
    def call(provider, model):
        return answers.get(provider, ("OK", None))
    return call


def test_a_usage_limited_provider_is_blocked(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTHIA_PREFLIGHT_MODELS", "anthropic:m1,openai:m2,google:m3")
    report = pf.run(["openai", "anthropic", "google"],
                    call=_fake({"anthropic": ("", LIMIT),
                                "google": ("", "HTTP 503 overloaded")}))
    by = {r["provider"]: r for r in report["providers"]}
    assert by["anthropic"]["status"] == "usage_limit" and by["anthropic"]["blocking"]
    assert by["anthropic"]["regain_at"] == "2026-11-01T00:00Z"
    assert by["openai"]["status"] == "ok"
    assert by["google"]["status"] == "error" and not by["google"]["blocking"]
    assert report["blocked_providers"] == ["anthropic"]
    assert "anthropic usage_limit until 2026-11-01" in pf.reason_line(report)


_ORIG_RUN = pf.run


def test_main_writes_outputs_and_the_block_list(monkeypatch, tmp_path):
    out_file, env_file = tmp_path / "out.txt", tmp_path / "env.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))
    monkeypatch.setenv("GITHUB_ENV", str(env_file))
    monkeypatch.setenv("PYTHIA_BLOCK_PROVIDERS", "google")
    monkeypatch.setattr(pf, "run", lambda provs: _ORIG_RUN(provs, call=_fake({"anthropic": ("", LIMIT)})))
    report_path = tmp_path / "pre.json"
    assert pf.main(["--providers", "anthropic,openai", "--out", str(report_path), "--apply-block"]) == 0
    outputs = out_file.read_text()
    assert "anthropic_ok=false" in outputs and "openai_ok=true" in outputs
    assert "PYTHIA_BLOCK_PROVIDERS=google,anthropic" in env_file.read_text()
    assert json.loads(report_path.read_text())["blocked_providers"] == ["anthropic"]


def test_main_never_fails_and_blocks_nothing_when_all_answer(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "o"))
    monkeypatch.delenv("GITHUB_ENV", raising=False)
    monkeypatch.setattr(pf, "run", lambda provs: _ORIG_RUN(provs, call=_fake({})))
    assert pf.main(["--out", str(tmp_path / "p.json")]) == 0
    assert "blocked_providers=\n" in (tmp_path / "o").read_text()


def test_a_terminal_refusal_is_not_retried_and_skips_later_calls(monkeypatch):
    calls = []

    def fake_anthropic(prompt, model, temperature, **kw):
        calls.append(model)
        return providers.ProviderResult("", providers.usage_to_dict(None), 0.0, model, error=LIMIT)

    slept = []

    async def no_sleep(sec):
        slept.append(sec)

    monkeypatch.setattr(providers, "call_anthropic", fake_anthropic)
    monkeypatch.setattr(providers.asyncio, "sleep", no_sleep)
    ms = providers.ModelSpec(name="c", provider="anthropic", model_id="claude-opus-5-5")
    _text, usage, err = asyncio.run(providers.call_chat_ms(ms, "hi", log_call=False))
    assert err and usage.get("terminal_provider_error") == "usage_limit"
    assert len(calls) == 1 and not any(s >= 60 for s in slept)  # no credit-retry pause
    _t, usage2, err2 = asyncio.run(providers.call_chat_ms(ms, "hi", log_call=False))
    assert len(calls) == 1 and "skipped" in err2
    assert usage2["terminal_provider_error"] == "usage_limit"


def test_the_bundle_line_fails_a_blocked_provider():
    from scripts.dump_pythia_debug_bundle import _provider_preflight_check

    rep = {"providers": [{"provider": "anthropic", "status": "usage_limit", "blocking": True,
                          "regain_at": "2026-11-01T00:00Z"},
                         {"provider": "openai", "status": "ok", "blocking": False}]}
    line = _provider_preflight_check([rep], None)
    assert line["status"] == "FAIL" and "2026-11-01" in line["detail"]
    ok = {"providers": [{"provider": "openai", "status": "ok", "blocking": False}]}
    assert _provider_preflight_check([ok], None)["status"] == "OK"
    assert _provider_preflight_check([ok], "anthropic")["status"] == "FAIL"
    assert _provider_preflight_check([], None) is None


def test_the_workflows_run_the_preflight_and_honour_it():
    stage = (REPO / ".github/workflows/pythia_pipeline_stage.yml").read_text()
    assert stage.count("python -m scripts.ci.provider_preflight") == 2  # hs_submit and fc_submit
    assert "--apply-block" in stage
    assert "PYTHIA_BLOCK_PROVIDERS: ${{ inputs.block_providers || '' }}" in stage
    sibyl = (REPO / ".github/workflows/run_sibyl.yml").read_text()
    assert "python -m scripts.ci.provider_preflight" in sibyl
    assert sibyl.count('if [ "${ANTHROPIC_OK:-}" = "false" ]') >= 2  # gate and interpreter
    calib = (REPO / ".github/workflows/compute_calibration_pythia.yml").read_text()
    assert calib.count('if [ "${ANTHROPIC_OK:-}" = "false" ]') >= 2
