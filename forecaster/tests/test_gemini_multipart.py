# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.
"""Gemini answers are read whole, and a cut answer is an error (Oct 2026).

On 1 Oct 2026 Gemini billed Cambodia's TC/PA flash member 1,224 answer
tokens and the stored text was 19 characters: both readers took
``parts[0]`` only. Five members were lost and llm_calls said "ok".
"""

from __future__ import annotations

import pytest

from forecaster.providers import google_finish_problem, google_text_and_finish


def _resp(parts, finish="STOP"):
    return {"candidates": [{"content": {"parts": parts}, "finishReason": finish}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 20}}


def test_every_text_part_is_read_and_thoughts_are_not():
    text, finish = google_text_and_finish(_resp([
        {"text": "thinking...", "thought": True},
        {"text": '```json\n{"reason'},
        {"text": 'ing": "x", "spds": {}}\n```'},
    ]))
    assert text == '```json\n{"reasoning": "x", "spds": {}}\n```'
    assert finish == "STOP"
    assert google_finish_problem(finish) is None


@pytest.mark.parametrize("finish", ["MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"])
def test_a_non_terminal_finish_is_a_truncation(finish):
    assert google_finish_problem(finish) == f"truncated: Gemini finishReason={finish}"


def test_the_batch_reader_reports_a_cut_answer_as_an_errored_item(monkeypatch):
    from pythia import llm_batch

    adapter = llm_batch._GoogleBatch.__new__(llm_batch._GoogleBatch)
    payload = {"response": {"inlinedResponses": {"inlinedResponses": [
        {"metadata": {"key": "a"}, "response": _resp([{"text": '{"ok": '}, {"text": "1}"}])},
        {"metadata": {"key": "b"}, "response": _resp([{"text": '{"spds": '}], "MAX_TOKENS")},
    ]}}}
    monkeypatch.setattr(adapter, "_get", lambda _bid: payload, raising=False)
    out = {cid: (ok, text, usage, err) for cid, ok, text, usage, err in adapter.fetch("batches/x")}
    assert out["a"][0] is True and out["a"][1] == '{"ok": 1}'
    assert out["a"][2]["finish_reason"] == "STOP"
    assert out["b"][0] is False
    assert out["b"][3] == "truncated: Gemini finishReason=MAX_TOKENS"
    assert out["b"][2]["finish_reason"] == "MAX_TOKENS"
