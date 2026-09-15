# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the pre-run batch canary (scripts/ci/batch_canary.py).

The canary must classify every provider outcome into a verdict a person can
act on before dispatching the pipeline, never touch a database, and exit 1
only when a provider cannot be batched right now.
"""

from __future__ import annotations

import json

import pytest

from pythia import llm_batch
from scripts.ci import batch_canary

_FILE_ACCESS = {"object": "list", "data": [{
    "code": "invalid_request", "param": "file_id",
    "message": "Cannot find file file-x, or organization org-y does not have access to it.",
}]}


class _Adapter:
    """Scripted adapter: submit outcome, then a poll state sequence."""

    submit_exc: Exception | None = None
    poll_states: list = ["ended"]
    poll_detail: str = ""
    fetch_items: list = []
    canceled: list = []
    submitted: list = []

    def submit(self, rows, **kwargs):
        type(self).submitted.append((list(rows), kwargs))
        if type(self).submit_exc:
            raise type(self).submit_exc
        return {"provider_batch_id": "pb_1", "input_file_id": "file_1"}

    def poll(self, provider_batch_id):
        states = type(self).poll_states
        state = states.pop(0) if len(states) > 1 else states[0]
        return llm_batch.BatchStatus(provider_batch_id, state, {}, type(self).poll_detail)

    def fetch(self, provider_batch_id):
        yield from type(self).fetch_items

    def cancel(self, provider_batch_id):
        type(self).canceled.append(provider_batch_id)


@pytest.fixture()
def adapter(monkeypatch):
    _Adapter.submit_exc = None
    _Adapter.poll_states = ["ended"]
    _Adapter.poll_detail = ""
    _Adapter.fetch_items = [("c", True, "OK", {"prompt_tokens": 1}, "")]
    _Adapter.canceled = []
    _Adapter.submitted = []
    monkeypatch.setattr(llm_batch, "_ADAPTERS", {"openai": _Adapter, "anthropic": _Adapter, "google": _Adapter})
    state = {"t": 0.0}
    monkeypatch.setattr(batch_canary, "_clock", lambda: state["t"])
    monkeypatch.setattr(batch_canary, "_sleep", lambda s: state.__setitem__("t", state["t"] + s))
    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(k, "k")
    return _Adapter


_BODY = {"model": "gpt-5.6-sol", "messages": [{"role": "user", "content": "hi"}]}


def test_ok_when_the_batch_ends_with_a_result(adapter):
    r = batch_canary.run_one("openai", "gpt-5.6-sol", _BODY, wait_sec=60, poll_sec=5)
    assert r["verdict"] == batch_canary.VERDICT_OK
    assert r["item_ok"] is True and r["provider_batch_id"] == "pb_1"


def test_file_access_rejection_at_submit(adapter):
    adapter.submit_exc = llm_batch.OpenAIBatchValidationError(
        "kept rejecting", provider_batch_id="pb_bad", input_file_id="file_bad",
        errors=_FILE_ACCESS["data"], file_access=True, attempts=4, same_file_retry_failed=True,
    )
    r = batch_canary.run_one("openai", "gpt-5.6-sol", _BODY, wait_sec=60, poll_sec=5)
    assert r["verdict"] == batch_canary.VERDICT_FILE_ACCESS
    assert "same_file_retry_failed=True" in r["detail"]


def test_deterministic_rejection_at_submit(adapter):
    adapter.submit_exc = llm_batch.OpenAIBatchValidationError("mismatched_model", file_access=False)
    r = batch_canary.run_one("openai", "gpt-5.6-sol", _BODY, wait_sec=60, poll_sec=5)
    assert r["verdict"] == batch_canary.VERDICT_OTHER


def test_http_failure_is_submit_error(adapter):
    adapter.submit_exc = RuntimeError("502 bad gateway")
    r = batch_canary.run_one("anthropic", "claude-opus-5", _BODY, wait_sec=60, poll_sec=5)
    assert r["verdict"] == batch_canary.VERDICT_SUBMIT_ERROR
    assert "502" in r["detail"]


def test_failed_after_submit_is_classified_from_the_poll_detail(adapter):
    adapter.poll_states = ["in_progress", "failed"]
    adapter.poll_detail = json.dumps({"errors": _FILE_ACCESS})
    r = batch_canary.run_one("openai", "gpt-5.6-luna", _BODY, wait_sec=60, poll_sec=5)
    assert r["verdict"] == batch_canary.VERDICT_FILE_ACCESS


def test_slow_batch_is_cancelled_and_accepted(adapter):
    adapter.poll_states = ["in_progress"]
    r = batch_canary.run_one("openai", "gpt-5.6-sol", _BODY, wait_sec=30, poll_sec=10)
    assert r["verdict"] == batch_canary.VERDICT_SLOW
    assert adapter.canceled == ["pb_1"]


def test_missing_key_is_skipped(adapter, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    r = batch_canary.run_one("google", "gemini-3.5-flash", _BODY, wait_sec=60, poll_sec=5)
    assert r["verdict"] == batch_canary.VERDICT_NO_KEY
    assert adapter.submitted == []


def test_google_submit_passes_model_id(adapter):
    batch_canary.run_one("google", "gemini-3.5-flash", {"contents": []}, wait_sec=60, poll_sec=5)
    assert adapter.submitted[0][1] == {"model_id": "gemini-3.5-flash"}


def test_tiny_body_caps_output_per_provider():
    n = batch_canary.CANARY_MAX_TOKENS
    assert batch_canary.tiny_body("openai", {"model": "m", "messages": []})["max_completion_tokens"] == n
    assert batch_canary.tiny_body("anthropic", {"model": "m", "max_tokens": 32768})["max_tokens"] == n
    g = batch_canary.tiny_body("google", {"contents": [], "generationConfig": {"temperature": 0.2}})
    assert g["generationConfig"] == {"temperature": 0.2, "maxOutputTokens": n}


def test_members_dedupes_and_filters_by_provider():
    got = batch_canary.members({"openai"}, "openai:gpt-5.6-sol, openai:gpt-5.6-sol,anthropic:claude-opus-5")
    assert got == [{"provider": "openai", "model_id": "gpt-5.6-sol", "temperature": None, "thinking": None}]


def test_main_exits_one_and_annotates_on_failure(adapter, tmp_path, capsys, monkeypatch):
    adapter.submit_exc = llm_batch.OpenAIBatchValidationError(
        "kept rejecting", errors=_FILE_ACCESS["data"], file_access=True, attempts=3
    )
    monkeypatch.setattr(batch_canary, "build_body", lambda entry: dict(_BODY))
    out = tmp_path / "canary.json"
    rc = batch_canary.main(["--models", "openai:gpt-5.6-sol", "--wait-min", "1", "--out", str(out)])
    assert rc == 1
    payload = json.loads(out.read_text())
    assert payload["summary"]["by_provider"] == {"openai": batch_canary.VERDICT_FILE_ACCESS}
    assert payload["exit_code"] == 1
    assert "::error title=Batch canary failed::" in capsys.readouterr().out


def test_main_exits_zero_when_every_provider_batches(adapter, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(batch_canary, "build_body", lambda entry: dict(_BODY))
    out = tmp_path / "canary.json"
    rc = batch_canary.main([
        "--models", "openai:gpt-5.6-sol,anthropic:claude-opus-5", "--wait-min", "1", "--out", str(out),
    ])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["summary"]["n_failed"] == 0
    assert set(payload["summary"]["by_provider"]) == {"openai", "anthropic"}
    assert "::error" not in capsys.readouterr().out


def test_follow_bounds_the_whole_set_by_one_deadline(adapter, monkeypatch):
    """Five slow models must not cost five waits: one deadline for the set."""
    adapter.poll_states = ["in_progress"]
    results = [batch_canary.submit_one("openai", f"m{i}", _BODY) for i in range(5)]
    assert all("_pending" in r for r in results)
    t_before = batch_canary._clock()
    batch_canary.follow(results, wait_sec=60, poll_sec=10)
    assert batch_canary._clock() - t_before <= 60 + 10
    assert [r["verdict"] for r in results] == [batch_canary.VERDICT_SLOW] * 5
    assert len(adapter.canceled) == 5
