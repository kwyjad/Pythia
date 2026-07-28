# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the forecaster's Batch-API phases (--phase submit/collect).

Covers the member-call seam in _call_spd_model_for_spec:
- submit phase enqueues llm_batch_requests rows (byte-identical bodies via
  build_body_for_spec) and returns the submit sentinel without calling any
  provider;
- submit phase defers non-batchable providers (no enqueue, no sync call);
- collect phase replays a collected batch result (usage carries
  service_tier=batch) without calling any provider;
- collect phase falls back to the sync call_chat_ms path on a miss and marks
  the row fallback_sync;
- full phase (default) never touches batch state.
"""

from __future__ import annotations

import asyncio
import json

import duckdb
import pytest

import forecaster.cli as cli
from forecaster.providers import ModelSpec
from pythia import llm_batch
from pythia.db.schema import ensure_schema


_MS = ModelSpec(
    name="claude-opus-5",
    provider="anthropic",
    model_id="claude-opus-5",
    active=True,
    purpose="spd_v2",
)


@pytest.fixture()
def batch_env(monkeypatch: pytest.MonkeyPatch):
    con = duckdb.connect(":memory:")
    ensure_schema(con)
    monkeypatch.setenv("PYTHIA_BATCH_API_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_BATCH_PROVIDERS", "openai,anthropic,google")
    monkeypatch.setattr(cli, "_BATCH_CON", con)
    monkeypatch.setattr(cli, "_BATCH_PIPELINE_ID", "pl_test")
    yield con
    monkeypatch.setattr(cli, "_BATCH_CON", None)
    con.close()


def _call(monkeypatch, phase, *, sync_response=None):
    monkeypatch.setattr(cli, "_FORECAST_PHASE", phase)
    calls: list = []

    async def _fake_call_chat_ms(ms, prompt, **kwargs):
        calls.append((ms, prompt, kwargs))
        if sync_response is None:
            raise AssertionError("sync provider call must not happen in this scenario")
        return sync_response

    monkeypatch.setattr(cli, "call_chat_ms", _fake_call_chat_ms)
    result = asyncio.run(
        cli._call_spd_model_for_spec(
            _MS,
            "the spd prompt",
            run_id="fc_1",
            question_id="SOM_ACE_FATALITIES_2026-08",
            iso3="SOM",
            hazard_code="ACE",
            metric="FATALITIES",
            anchor_month="2026-08",
            batch_family="spd_v2",
        )
    )
    return result, calls


def test_submit_phase_enqueues_and_returns_sentinel(batch_env, monkeypatch):
    (text, usage, error, ms), sync_calls = _call(monkeypatch, "submit")
    assert error == cli._BATCH_SENTINEL_SUBMITTED
    assert text == "" and not sync_calls

    row = batch_env.execute(
        "SELECT provider, model_id, family, question_id, model_key, status, request_body_json "
        "FROM llm_batch_requests"
    ).fetchone()
    assert row[:4] == ("anthropic", "claude-opus-5", "spd_v2", "SOM_ACE_FATALITIES_2026-08")
    assert row[5] == "pending"
    body = json.loads(row[6])
    # Byte-identical to the sync body builder (incl. SPD max_tokens switch
    # and the opus-5 temperature drop).
    from forecaster.providers import build_body_for_spec

    assert body == build_body_for_spec(_MS, "the spd prompt", 0.2)
    assert "temperature" not in body


def test_submit_phase_defers_excluded_provider(batch_env, monkeypatch):
    monkeypatch.setenv("PYTHIA_BATCH_PROVIDERS", "openai")  # anthropic excluded
    (text, usage, error, ms), sync_calls = _call(monkeypatch, "submit")
    assert error == cli._BATCH_SENTINEL_DEFERRED
    assert not sync_calls
    assert batch_env.execute("SELECT COUNT(*) FROM llm_batch_requests").fetchone()[0] == 0


def test_collect_phase_replays_batch_result(batch_env, monkeypatch):
    cid = llm_batch.enqueue_request(
        batch_env,
        family="spd_v2",
        provider="anthropic",
        model_id="claude-opus-5",
        request_body={"model": "claude-opus-5"},
        prompt_text="the spd prompt",
        question_id="SOM_ACE_FATALITIES_2026-08",
        model_key=cli._batch_model_key(_MS),
        iso3="SOM",
        hazard_code="ACE",
        metric="FATALITIES",
        pipeline_id="pl_test",
    )
    batch_env.execute(
        "UPDATE llm_batch_requests SET status='succeeded', response_text=?, usage_json=? "
        "WHERE custom_id = ?",
        [
            '{"spds": {"2026-08": {"probs": [1,0,0,0,0,0,0]}}}',
            json.dumps({"prompt_tokens": 5000, "completion_tokens": 900,
                        "service_tier": "batch", "batch_id": "b_x"}),
            cid,
        ],
    )
    (text, usage, error, ms), sync_calls = _call(monkeypatch, "collect")
    assert not sync_calls
    assert error is None
    assert json.loads(text)["spds"]
    assert usage["service_tier"] == "batch"
    assert usage["prompt_tokens"] == 5000


def test_collect_phase_miss_falls_back_to_sync(batch_env, monkeypatch):
    cid = llm_batch.enqueue_request(
        batch_env,
        family="spd_v2",
        provider="anthropic",
        model_id="claude-opus-5",
        request_body={"model": "claude-opus-5"},
        prompt_text="the spd prompt",
        question_id="SOM_ACE_FATALITIES_2026-08",
        model_key=cli._batch_model_key(_MS),
        iso3="SOM",
        hazard_code="ACE",
        metric="FATALITIES",
        pipeline_id="pl_test",
    )
    batch_env.execute(
        "UPDATE llm_batch_requests SET status='errored', error_text='boom' WHERE custom_id = ?",
        [cid],
    )
    (text, usage, error, ms), sync_calls = _call(
        monkeypatch, "collect", sync_response=("sync text", {"prompt_tokens": 10}, None)
    )
    assert len(sync_calls) == 1
    assert text == "sync text"
    assert batch_env.execute(
        "SELECT status FROM llm_batch_requests WHERE custom_id = ?", [cid]
    ).fetchone()[0] == "fallback_sync"


def test_full_phase_never_touches_batch_state(batch_env, monkeypatch):
    (text, usage, error, ms), sync_calls = _call(
        monkeypatch, "full", sync_response=("sync text", {"prompt_tokens": 10}, None)
    )
    assert len(sync_calls) == 1
    assert text == "sync text"
    assert batch_env.execute("SELECT COUNT(*) FROM llm_batch_requests").fetchone()[0] == 0


def test_collect_under_different_pipeline_misses(batch_env, monkeypatch):
    # A terminal result from ANOTHER pipeline (e.g. last month's run, still
    # in the traveling DB artifact) must never replay into this pipeline —
    # the collect misses and takes the sync fallback instead.
    cid = llm_batch.enqueue_request(
        batch_env,
        family="spd_v2",
        provider="anthropic",
        model_id="claude-opus-5",
        request_body={"model": "claude-opus-5"},
        prompt_text="the spd prompt",
        question_id="SOM_ACE_FATALITIES_2026-08",
        model_key=cli._batch_model_key(_MS),
        iso3="SOM",
        hazard_code="ACE",
        metric="FATALITIES",
        pipeline_id="pl_last_month",
    )
    batch_env.execute(
        "UPDATE llm_batch_requests SET status='succeeded', response_text='stale output' "
        "WHERE custom_id = ?", [cid],
    )
    (text, usage, error, ms), sync_calls = _call(
        monkeypatch, "collect", sync_response=("fresh sync text", {"prompt_tokens": 10}, None)
    )
    assert len(sync_calls) == 1
    assert text == "fresh sync text"
    assert "stale output" not in text
