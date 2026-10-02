# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The Gemini key travels in a header, never in the URL.

Until Oct 2026 every Gemini caller sent ``?key=<GEMINI_API_KEY>``. requests
quotes the URL in every connection error and HTTPError, and that text was
stored in ``llm_calls.error_text``, ``llm_batches.error_text`` and
``grounding_debug_json`` — tables that ship in the public release DB. These
tests record every request a caller makes and fail if the key appears
anywhere but the ``x-goog-api-key`` header.
"""

from __future__ import annotations

import json

import pytest
import requests

KEY = "AIzaSyTESTKEY0123456789abcdefghij"


class _Resp:
    status_code = 500
    headers: dict = {}
    text = "{}"

    def json(self):
        return {"error": {"message": "boom"}}

    def raise_for_status(self):
        raise requests.HTTPError(f"500 Server Error for url: {self._url}")


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, url, *args, **kwargs):
        self.calls.append({"url": url, **kwargs})
        resp = _Resp()
        resp._url = url
        return resp


@pytest.fixture
def recorder(monkeypatch, tmp_path):
    rec = _Recorder()
    # Some callers log their failure to a DB; keep that off the real one.
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{tmp_path / 'p.duckdb'}")
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{tmp_path / 'r.duckdb'}")
    monkeypatch.setattr(requests, "post", rec)
    monkeypatch.setattr(requests, "get", rec)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    return rec


def _assert_key_only_in_header(calls: list[dict]) -> None:
    assert calls, "the caller made no request"
    for call in calls:
        assert KEY not in call["url"], call["url"]
        assert "key=" not in call["url"], call["url"]
        params = call.get("params") or {}
        assert KEY not in json.dumps(params), params
        assert (call.get("headers") or {}).get("x-goog-api-key") == KEY, call


def test_call_google_sends_the_key_in_a_header(recorder, monkeypatch):
    from forecaster import providers

    monkeypatch.setattr(providers, "_GEMINI_API_KEY", KEY)
    providers.call_google("hi", "gemini-3.5-flash", 0.2, timeout_sec=1)
    _assert_key_only_in_header(recorder.calls)


def test_the_batch_adapter_sends_the_key_in_a_header(recorder):
    from pythia.llm_batch import _GoogleBatch

    adapter = _GoogleBatch()
    with pytest.raises(requests.HTTPError) as exc:
        adapter.submit([("cid", {"contents": []})], model_id="gemini-3.5-flash")
    assert KEY not in str(exc.value)
    with pytest.raises(requests.HTTPError):
        adapter.poll("batches/abc")
    adapter.cancel("batches/abc")
    _assert_key_only_in_header(recorder.calls)


def test_grounding_sends_the_key_in_a_header(recorder):
    from pythia.web_research.backends import gemini_grounding

    gemini_grounding.fetch_via_gemini(
        "flood", recency_days=30, include_structural=False, timeout_sec=1,
        max_results=3, model_id="gemini-2.5-flash",
    )
    _assert_key_only_in_header(recorder.calls)


def test_research_sends_the_key_in_a_header(recorder):
    from forecaster import research

    research._grounded_search("flood", max_results=3, timeout=1)
    _assert_key_only_in_header(recorder.calls)


def test_the_debug_bundle_fetcher_sends_the_key_in_a_header(recorder):
    from scripts.debug_bundle import provider_objects

    provider_objects._fetch_google({"provider_batch_id": "batches/abc"})
    _assert_key_only_in_header(recorder.calls)


def test_a_forecaster_llm_call_error_is_stored_without_the_key(tmp_path, monkeypatch):
    """The writer scrubs error text even when a caller still quotes a key."""

    import asyncio

    from forecaster.llm_logging import log_forecaster_llm_call
    from pythia.db import schema
    from forecaster.providers import ModelSpec

    db = tmp_path / "t.duckdb"
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{db}")
    spec = ModelSpec(name="gemini-3.5-flash", provider="google", model_id="gemini-3.5-flash")
    asyncio.run(log_forecaster_llm_call(
        call_type="spd_v2",
        model_spec=spec,
        prompt_text="prompt",
        response_text="",
        usage={"note": f"retry with {KEY}"},
        error_text=f"Gemini request error: ConnectionError for url: https://x/m?key={KEY}",
        run_id="fc_1",
        question_id="Q1",
    ))
    con = schema.connect()
    try:
        rows = con.execute(
            "SELECT error_text, error_message, usage_json FROM llm_calls"
        ).fetchall()
    finally:
        con.close()
        schema.close_pooled_connections()
    assert rows, "nothing was logged"
    for row in rows:
        for value in row:
            assert KEY not in str(value or "")
