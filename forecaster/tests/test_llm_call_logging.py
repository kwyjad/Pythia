# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""LLM telemetry double-write fix (July 2026).

call_chat_ms used to write a generic phase-NULL llm_calls row for EVERY
generation call, alongside the rich row from log_hs_llm_call /
log_forecaster_llm_call — double-counting cost in phase-less aggregations
(the Costs page "other" bucket) and leaking test-run cost (the generic rows
carried no is_test). These tests pin: the log_call opt-out, and is_test
stamping on write_llm_call.
"""

from __future__ import annotations

import asyncio

import pytest

duckdb = pytest.importorskip("duckdb")

from forecaster import providers
from forecaster.providers import ModelSpec, ProviderResult, call_chat_ms
from pythia.db.util import write_llm_call


def _stub_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_call_provider_sync(provider, prompt, model_id, temperature, **kwargs):
        return ProviderResult(
            text='{"ok": true}',
            usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            cost_usd=0.001,
            model_id=model_id,
        )

    monkeypatch.setattr(providers, "_call_provider_sync", _fake_call_provider_sync)


def _spec() -> ModelSpec:
    return ModelSpec(
        name="stub", provider="google", model_id="stub-model", active=True
    )


def test_call_chat_ms_logs_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_provider(monkeypatch)
    logged: list[dict] = []
    monkeypatch.setattr(
        providers, "_log_llm_call", lambda **kw: logged.append(kw)
    )
    text, usage, error = asyncio.run(call_chat_ms(_spec(), "prompt"))
    assert text and not error
    assert len(logged) == 1


def test_call_chat_ms_log_call_false_suppresses_generic_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_provider(monkeypatch)
    logged: list[dict] = []
    monkeypatch.setattr(
        providers, "_log_llm_call", lambda **kw: logged.append(kw)
    )
    text, usage, error = asyncio.run(call_chat_ms(_spec(), "prompt", log_call=False))
    assert text and not error
    assert logged == []


def _make_llm_calls_table(con) -> None:
    con.execute(
        """
        CREATE TABLE llm_calls (
            call_id TEXT, component TEXT, model_name TEXT, prompt_key TEXT,
            prompt_version TEXT, tokens_in INTEGER, tokens_out INTEGER,
            cost_usd DOUBLE, latency_ms INTEGER, success BOOLEAN,
            llm_profile TEXT, hs_run_id TEXT, ui_run_id TEXT,
            forecaster_run_id TEXT
        )
        """
    )


def _write_and_fetch_is_test(con) -> bool:
    write_llm_call(
        con,
        component="Test",
        model="stub-model",
        prompt_key="test.key",
        version="1.0.0",
        usage={"prompt_tokens": 1, "completion_tokens": 1},
        cost=0.001,
        latency_ms=10,
        success=True,
    )
    return bool(con.execute("SELECT is_test FROM llm_calls").fetchone()[0])


def test_write_llm_call_stamps_is_test_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    con = duckdb.connect(":memory:")
    _make_llm_calls_table(con)
    monkeypatch.setenv("PYTHIA_TEST_MODE", "1")
    assert _write_and_fetch_is_test(con) is True


def test_write_llm_call_defaults_is_test_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    con = duckdb.connect(":memory:")
    _make_llm_calls_table(con)
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
    assert _write_and_fetch_is_test(con) is False
