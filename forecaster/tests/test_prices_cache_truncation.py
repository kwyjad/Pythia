# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Prices, cache breakpoints and cut answers (Oct 2026).

* gemini-3.6-flash runs at half price until a stated day, then at list.
* The Anthropic SPD breakpoint sits at the end of the text a whole
  (hazard, metric, track) group shares, so arm variants share one entry.
* binary_v2 gets a breakpoint too.
* An OpenAI answer cut at its ceiling is an error, as Gemini's already is.
* A member kept by salvaging its spds block is marked and counted.
"""

from __future__ import annotations

import json

import pytest

from forecaster import providers
from forecaster.providers import (
    ModelSpec,
    _parse_model_costs,
    build_body_for_spec,
    cache_segments_from_prefix,
    openai_finish_problem,
    openai_max_output_tokens,
    openai_text_and_finish,
)


# --------------------------------------------------------------------------
# Promotional prices
# --------------------------------------------------------------------------

def _raw():
    with open("pythia/model_costs.json", encoding="utf-8") as fh:
        return json.load(fh)


def test_flash_is_promotional_until_the_stated_day_and_list_after():
    from datetime import date

    raw = _raw()
    entry = raw["gemini-3.6-flash"]
    until = date.fromisoformat(entry["promo_until"])
    pairs, details = _parse_model_costs(raw, today=until)
    assert pairs["gemini-3.6-flash"] == (0.75, 3.75)
    pairs, details = _parse_model_costs(raw, today=date(until.year + 1, 1, 1))
    assert pairs["gemini-3.6-flash"] == (1.50, 7.50)
    assert details["gemini-3.6-flash"]["cached_input"] == pytest.approx(0.15)


def test_every_promotional_entry_states_its_list_price():
    """A promotion with no list price would keep pricing at the promo forever."""
    for key, value in _raw().items():
        if isinstance(value, dict) and value.get("promo_until"):
            assert isinstance(value.get("list"), dict), key
            assert {"input", "output"} <= set(value["list"]), key


def test_pricing_date_env_overrides_today(monkeypatch):
    monkeypatch.setenv("PYTHIA_PRICING_DATE", "2027-02-01")
    pairs, _ = _parse_model_costs(_raw())
    assert pairs["gemini-3.6-flash"] == (1.50, 7.50)
    monkeypatch.setenv("PYTHIA_PRICING_DATE", "2026-10-14")
    pairs, _ = _parse_model_costs(_raw())
    assert pairs["gemini-3.6-flash"] == (0.75, 3.75)


# --------------------------------------------------------------------------
# Cache breakpoints
# --------------------------------------------------------------------------

def _spd_prefix(qid, iso3, *, advice=True, track=1):
    from forecaster import prompts

    question = {
        "question_id": qid, "iso3": iso3, "hazard_code": "ACE",
        "metric": "FATALITIES", "wording": "How many conflict deaths?",
        "window_start_date": "2026-11-01",
    }
    prefix, suffix = prompts.build_spd_prompt_v2(
        question, {}, {}, {}, track=track, return_parts=True,
    )
    return prefix, suffix


def test_the_first_segment_is_identical_across_questions_of_a_group(monkeypatch):
    monkeypatch.setenv("PYTHIA_PROMPT_V3_ORDER", "1")
    a_prefix, a_suffix = _spd_prefix("SOM_ACE_FATALITIES_2026-11", "SOM")
    b_prefix, b_suffix = _spd_prefix("SDN_ACE_FATALITIES_2026-11", "SDN")
    seg_a = cache_segments_from_prefix(a_prefix + a_suffix, a_prefix)
    seg_b = cache_segments_from_prefix(b_prefix + b_suffix, b_prefix)
    assert "".join(t for t, _ in seg_a) == a_prefix + a_suffix
    assert seg_a[0] == seg_b[0] and seg_a[0][1] is True
    # The shared group text is long enough to cache on Opus (512 tokens
    # is about 2,048 characters).
    assert len(seg_a[0][0]) >= 2048
    # The per-question tail is never a breakpoint.
    assert seg_a[-1][1] is False


def test_a_prefix_with_no_marker_keeps_one_breakpoint():
    segs = cache_segments_from_prefix("STATIC " * 400 + "tail", "STATIC " * 400)
    assert segs == [("STATIC " * 400, True), ("tail", False)]
    assert cache_segments_from_prefix("other prompt", "STATIC ") is None


def test_batch_body_marks_the_shared_group_text(monkeypatch):
    monkeypatch.setenv("PYTHIA_PROMPT_V3_ORDER", "1")
    monkeypatch.setenv("PYTHIA_PROMPT_CACHE_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_BATCH_PROMPT_CACHE", "1")
    prefix, suffix = _spd_prefix("SOM_ACE_FATALITIES_2026-11", "SOM")
    ms = ModelSpec(name="claude-opus-5-5", provider="anthropic",
                   model_id="claude-opus-5-5", active=True, purpose="spd_v2")
    body = build_body_for_spec(ms, prefix + suffix, 0.2, cache_prefix=prefix)
    blocks = body["messages"][0]["content"]
    assert "".join(b["text"] for b in blocks) == prefix + suffix
    marked = [b for b in blocks if "cache_control" in b]
    assert 1 <= len(marked) <= 4
    first = cache_segments_from_prefix(prefix + suffix, prefix)[0][0]
    assert blocks[0]["text"] == first and "cache_control" in blocks[0]


def test_binary_prompts_have_a_cacheable_static_prefix():
    from forecaster.binary_prompts import BINARY_STATIC_END, binary_cache_prefix

    prompt = "ROLE " * 700 + BINARY_STATIC_END + "QUESTION DATA for SOM"
    prefix = binary_cache_prefix(prompt)
    assert prefix and prefix.endswith(BINARY_STATIC_END)
    assert len(prefix) >= 2048
    assert binary_cache_prefix("no marker here") is None


def test_the_binary_builder_writes_the_marker():
    import inspect

    from forecaster import binary_prompts

    src = inspect.getsource(binary_prompts.build_binary_event_prompt)
    assert "The QUESTION DATA follows below." in src


# --------------------------------------------------------------------------
# OpenAI: the ceiling and the finish reason
# --------------------------------------------------------------------------

def test_forecast_purposes_get_an_explicit_ceiling(monkeypatch):
    monkeypatch.delenv("PYTHIA_OPENAI_SPD_MAX_OUTPUT_TOKENS", raising=False)
    assert openai_max_output_tokens("spd_v2") == 65536
    assert openai_max_output_tokens("track2_spd") == 65536
    assert openai_max_output_tokens("hs_triage") is None
    monkeypatch.setenv("PYTHIA_OPENAI_SPD_MAX_OUTPUT_TOKENS", "0")
    assert openai_max_output_tokens("spd_v2") is None


def test_batch_and_sync_openai_bodies_carry_the_same_ceiling(monkeypatch):
    monkeypatch.delenv("PYTHIA_OPENAI_SPD_MAX_OUTPUT_TOKENS", raising=False)
    ms = ModelSpec(name="gpt-6-sol", provider="openai", model_id="gpt-6-sol",
                   active=True, purpose="spd_v2", thinking="high")
    batch = build_body_for_spec(ms, "prompt", 0.2)
    assert batch["max_completion_tokens"] == 65536
    sent = {}

    class _Resp:
        ok = True
        status_code = 200
        headers: dict = {}

        def json(self):
            return {"choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    def _post(url, headers=None, json=None, timeout=None):
        sent.update(json)
        return _Resp()

    monkeypatch.setattr(providers, "_OPENAI_API_KEY", "k")
    monkeypatch.setattr(providers.requests, "post", _post)
    providers._call_provider_sync("openai", "prompt", "gpt-6-sol", 0.2,
                                  thinking_level="high", purpose="spd_v2")
    assert sent == batch


def test_a_length_finish_is_a_truncation_error(monkeypatch):
    payload = {"choices": [{"message": {"content": '{"spds": {'}, "finish_reason": "length"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 8549}}
    text, finish = openai_text_and_finish(payload)
    assert finish == "length" and text.startswith("{")
    assert openai_finish_problem("length") == "truncated: OpenAI finish_reason=length"
    assert openai_finish_problem("stop") is None

    class _Resp:
        ok = True
        status_code = 200
        headers: dict = {}

        def json(self):
            return payload

    monkeypatch.setattr(providers, "_OPENAI_API_KEY", "k")
    monkeypatch.setattr(providers.requests, "post", lambda *a, **k: _Resp())
    result = providers.call_openai("p", "gpt-6-sol", 0.2)
    assert result.error == "truncated: OpenAI finish_reason=length"
    # The cut answer was billed; its usage is kept.
    assert result.usage["completion_tokens"] == 8549
    assert result.usage["finish_reason"] == "length"


def test_a_cut_openai_batch_item_is_an_errored_item(monkeypatch):
    from pythia import llm_batch

    line = json.dumps({
        "custom_id": "spd-x",
        "response": {"status_code": 200, "body": {
            "choices": [{"message": {"content": "{"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 100}}},
    })

    class _R:
        def __init__(self, payload=None, text=""):
            self._p, self.text = payload, text

        def raise_for_status(self):
            return None

        def json(self):
            return self._p

    def _get(url, headers=None, timeout=None):
        if url.endswith("/content"):
            return _R(text=line)
        return _R(payload={"output_file_id": "file-1"})

    monkeypatch.setattr(llm_batch.requests, "get", _get)
    adapter = llm_batch._OpenAIBatch.__new__(llm_batch._OpenAIBatch)
    adapter.base = "https://api.openai.com/v1"
    adapter.api_key = "k"
    monkeypatch.setattr(adapter, "_headers", lambda: {}, raising=False)
    items = list(adapter.fetch("batch_1"))
    assert len(items) == 1
    cid, ok, _text, usage, error = items[0]
    assert cid == "spd-x" and ok is False
    assert error == "truncated: OpenAI finish_reason=length"
    assert usage["completion_tokens"] == 100


# --------------------------------------------------------------------------
# Salvaged members
# --------------------------------------------------------------------------

def test_salvaged_members_are_counted_by_the_bundle(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    from scripts.dump_pythia_debug_bundle import _load_salvaged_members

    con = duckdb.connect(str(tmp_path / "s.duckdb"))
    con.execute("CREATE TABLE forecasts_raw (run_id TEXT, question_id TEXT, model_name TEXT, "
                "status TEXT, spd_json TEXT)")
    rows = [
        ("fc_1", "Q1", "gpt-6-sol", "ok", '{"parse_mode": "salvaged"}'),
        ("fc_1", "Q1", "gpt-6-sol", "ok", '{"parse_mode": "salvaged"}'),  # second bucket row
        ("fc_1", "Q2", "gpt-6-sol", "ok", '{"parse_mode": "ok"}'),
        ("fc_1", "Q3", "claude-opus-5-5", "ok", "{}"),
        ("fc_2", "Q1", "gpt-6-sol", "ok", '{"parse_mode": "salvaged"}'),
    ]
    con.executemany("INSERT INTO forecasts_raw VALUES (?,?,?,?,?)", rows)
    assert _load_salvaged_members(con, "fc_1") == {"gpt-6-sol": 1}
    con.execute("CREATE TABLE empty_db (x INT)")
    assert _load_salvaged_members(duckdb.connect(), "fc_1") == {}
