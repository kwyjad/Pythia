# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Offline tests for the provider Batch-API layer (pythia/llm_batch.py).

Everything is mocked — no network. Covers: custom-id encoding (Anthropic
charset), enqueue/no-clobber semantics, submit grouping + flag gating,
the poll/collect state machine, collect idempotency, batch usage stamping
(service_tier=batch → 50% pricing), and the sync-fallback bookkeeping.
"""

from __future__ import annotations

import json
import re

import duckdb
import pytest

from pythia import llm_batch
from pythia.db.schema import ensure_schema


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    ensure_schema(con)
    yield con
    con.close()


@pytest.fixture()
def batch_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PYTHIA_BATCH_API_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_BATCH_PROVIDERS", "openai,anthropic,google")


def _enqueue_spd(con, question_id="SOM_ACE_FATALITIES_2026-08", model_key="opus5",
                 provider="anthropic", model_id="claude-opus-5", pipeline_id="pl_1"):
    return llm_batch.enqueue_request(
        con,
        family="spd_v2",
        provider=provider,
        model_id=model_id,
        request_body={"model": model_id, "max_tokens": 32768,
                      "messages": [{"role": "user", "content": "prompt text"}]},
        prompt_text="prompt text",
        question_id=question_id,
        model_key=model_key,
        iso3="SOM",
        hazard_code="ACE",
        metric="FATALITIES",
        anchor_month="2026-08",
        pipeline_id=pipeline_id,
    )


class TestCustomIds:
    def test_anthropic_charset_and_length(self):
        cid = llm_batch.make_custom_id(
            "spd_v2", question_id="SOM_ACE_FATALITIES_2026-08", model_key="gpt5.6-sol"
        )
        assert re.match(r"^[A-Za-z0-9_-]{1,64}$", cid), cid
        # Dots are stripped by sanitization, not passed through.
        assert "." not in cid

    def test_hs_ids_encode_country_hazard_pass(self):
        cid = llm_batch.make_custom_id("hs_rc", iso3="SOM", hazard_code="ACE", pass_idx=1)
        assert cid == "hsrc-SOM-ACE-p1"
        cid2 = llm_batch.make_custom_id("hs_triage", iso3="YEM", hazard_code="TC", pass_idx=2)
        assert cid2 == "hstr-YEM-TC-p2"

    def test_same_inputs_same_id_different_model_different_id(self):
        a = llm_batch.make_custom_id("spd_v2", question_id="Q1", model_key="m1")
        b = llm_batch.make_custom_id("spd_v2", question_id="Q1", model_key="m1")
        c = llm_batch.make_custom_id("spd_v2", question_id="Q1", model_key="m2")
        assert a == b != c

    def test_unknown_family_and_missing_keys_raise(self):
        with pytest.raises(ValueError):
            llm_batch.make_custom_id("nope", question_id="Q1")
        with pytest.raises(ValueError):
            llm_batch.make_custom_id("spd_v2")  # no question_id
        with pytest.raises(ValueError):
            llm_batch.make_custom_id("hs_rc", iso3="SOM")  # no hazard


class TestEnqueue:
    def test_enqueue_writes_pending_row(self, con):
        cid = _enqueue_spd(con)
        row = con.execute(
            "SELECT status, provider, family, question_id, model_key, prompt_sha256 "
            "FROM llm_batch_requests WHERE custom_id = ?", [cid]
        ).fetchone()
        assert row[0] == "pending"
        assert row[1] == "anthropic"
        assert row[2] == "spd_v2"
        assert row[3] == "SOM_ACE_FATALITIES_2026-08"
        assert row[4] == "opus5"
        assert len(row[5]) == 64

    def test_terminal_rows_never_clobbered(self, con):
        cid = _enqueue_spd(con)
        con.execute(
            "UPDATE llm_batch_requests SET status='succeeded', response_text='kept' "
            "WHERE custom_id = ?", [cid]
        )
        cid2 = _enqueue_spd(con)
        assert cid2 == cid
        row = con.execute(
            "SELECT status, response_text FROM llm_batch_requests WHERE custom_id = ?",
            [cid],
        ).fetchone()
        assert row == ("succeeded", "kept")


class _FakeAdapter:
    """Records submits; serves canned poll/fetch results."""

    provider = "fake"
    submitted: list = []
    poll_state = "ended"
    poll_detail = ""
    fetch_results: list = []
    fail_models: set = set()      # submit raises for chunks whose first body names one of these
    fail_exc: Exception | None = None
    canceled: list = []

    def __init__(self):
        pass

    def submit(self, rows, **kwargs):
        rows = list(rows)
        first_model = rows[0][1].get("model") if rows and isinstance(rows[0][1], dict) else None
        if first_model in type(self).fail_models:
            raise type(self).fail_exc or RuntimeError(f"submit refused for {first_model}")
        type(self).submitted.append((rows, kwargs))
        return {"provider_batch_id": f"prov_{len(type(self).submitted)}", "input_file_id": None}

    def poll(self, provider_batch_id):
        return llm_batch.BatchStatus(
            provider_batch_id, type(self).poll_state, {}, type(self).poll_detail
        )

    def fetch(self, provider_batch_id):
        yield from type(self).fetch_results

    def cancel(self, provider_batch_id):
        type(self).canceled.append(provider_batch_id)


@pytest.fixture()
def fake_adapters(monkeypatch: pytest.MonkeyPatch):
    _FakeAdapter.submitted = []
    _FakeAdapter.poll_state = "ended"
    _FakeAdapter.poll_detail = ""
    _FakeAdapter.fetch_results = []
    _FakeAdapter.fail_models = set()
    _FakeAdapter.fail_exc = None
    _FakeAdapter.canceled = []
    monkeypatch.setattr(llm_batch, "_sleep", lambda _s: None)
    monkeypatch.setattr(llm_batch, "_openai_submit_budget_deadline", None)
    monkeypatch.setattr(
        llm_batch, "_ADAPTERS",
        {"openai": _FakeAdapter, "anthropic": _FakeAdapter, "google": _FakeAdapter},
    )
    return _FakeAdapter


class TestSubmit:
    def test_submit_disabled_providers_stay_pending(
        self, con, fake_adapters, monkeypatch
    ):
        monkeypatch.setenv("PYTHIA_BATCH_API_ENABLED", "1")
        monkeypatch.setenv("PYTHIA_BATCH_PROVIDERS", "openai")  # anthropic excluded
        _enqueue_spd(con)  # anthropic row
        created = llm_batch.submit_pending(
            con, family="spd_v2", pipeline_id="pl_1", stage="s3"
        )
        assert created == []
        status = con.execute(
            "SELECT status FROM llm_batch_requests"
        ).fetchone()[0]
        assert status == "pending"

    def test_submit_groups_google_by_model(self, con, fake_adapters, batch_enabled):
        _enqueue_spd(con, question_id="Q1", model_key="gflash",
                     provider="google", model_id="gemini-3.5-flash")
        _enqueue_spd(con, question_id="Q1", model_key="gpro",
                     provider="google", model_id="gemini-3.1-pro-preview")
        _enqueue_spd(con, question_id="Q1", model_key="opus5",
                     provider="anthropic", model_id="claude-opus-5")
        created = llm_batch.submit_pending(
            con, family="spd_v2", pipeline_id="pl_1", stage="s3", run_id="fc_1"
        )
        # 2 google model-groups + 1 anthropic group = 3 batches
        assert len(created) == 3
        google_kwargs = [k for rows, k in fake_adapters.submitted if k]
        assert sorted(k["model_id"] for k in google_kwargs) == [
            "gemini-3.1-pro-preview", "gemini-3.5-flash",
        ]
        statuses = {
            r[0] for r in con.execute("SELECT status FROM llm_batch_requests").fetchall()
        }
        assert statuses == {"submitted"}
        batches = con.execute(
            "SELECT family, run_id, pipeline_id, stage, status FROM llm_batches"
        ).fetchall()
        assert all(b == ("spd_v2", "fc_1", "pl_1", "s3", "submitted") for b in batches)

    def test_submit_never_mixes_models_in_one_openai_batch(
        self, con, fake_adapters, batch_enabled
    ):
        """OpenAI rejects a multi-model batch wholesale with `mismatched_model`.

        The ensemble has TWO OpenAI members (gpt-5.6-sol, gpt-5.6-luna). Before
        this, both landed in one spd_v2 batch and OpenAI failed the whole thing
        at validation — request_counts={total: 0}, no error file, every request
        silently re-run at full synchronous price. It cost roughly two thirds of
        forecaster spend on 2026-07-29 and again on 2026-07-30.
        """
        _enqueue_spd(con, question_id="Q1", model_key="sol",
                     provider="openai", model_id="gpt-5.6-sol")
        _enqueue_spd(con, question_id="Q2", model_key="sol",
                     provider="openai", model_id="gpt-5.6-sol")
        _enqueue_spd(con, question_id="Q1", model_key="luna",
                     provider="openai", model_id="gpt-5.6-luna")
        created = llm_batch.submit_pending(
            con, family="spd_v2", pipeline_id="pl_1", stage="s3", run_id="fc_1"
        )
        # One batch per model, never one batch with both.
        assert len(created) == 2

        # Every submitted chunk must be single-model. Read the model back out of
        # the request bodies actually handed to the adapter — that is what
        # OpenAI validates, not our grouping key.
        for rows, _kwargs in fake_adapters.submitted:
            models = set()
            for _cid, body in rows:
                if isinstance(body, str):
                    body = json.loads(body)
                models.add(body["model"])
            assert len(models) == 1, f"batch mixed models: {models}"

    def test_chunking_respects_request_cap(self, con, fake_adapters, batch_enabled, monkeypatch):
        monkeypatch.setattr(llm_batch, "_MAX_REQUESTS_PER_BATCH", 2)
        for i in range(5):
            _enqueue_spd(con, question_id=f"Q{i}")
        created = llm_batch.submit_pending(
            con, family="spd_v2", pipeline_id="pl_1", stage="s3"
        )
        assert len(created) == 3  # 2 + 2 + 1
        sizes = [len(rows) for rows, _ in fake_adapters.submitted]
        assert sorted(sizes) == [1, 2, 2]


class TestPollCollect:
    def _submit_one(self, con, fake_adapters):
        cid = _enqueue_spd(con)
        [batch_id] = llm_batch.submit_pending(
            con, family="spd_v2", pipeline_id="pl_1", stage="s3"
        )
        return cid, batch_id

    def test_poll_updates_status(self, con, fake_adapters, batch_enabled):
        _, batch_id = self._submit_one(con, fake_adapters)
        _FakeAdapter.poll_state = "in_progress"
        st = llm_batch.poll_batch(con, batch_id)
        assert st.state == "in_progress"
        assert con.execute(
            "SELECT status FROM llm_batches WHERE batch_id = ?", [batch_id]
        ).fetchone()[0] == "in_progress"
        _FakeAdapter.poll_state = "ended"
        st = llm_batch.poll_batch(con, batch_id)
        assert st.terminal

    def test_collect_writes_results_and_stamps_batch_tier(
        self, con, fake_adapters, batch_enabled
    ):
        cid, batch_id = self._submit_one(con, fake_adapters)
        _FakeAdapter.fetch_results = [
            (cid, True, '{"spds": {}}', {"prompt_tokens": 1000, "completion_tokens": 200}, ""),
        ]
        counts = llm_batch.collect_batch(con, batch_id)
        assert counts["succeeded"] == 1
        result = llm_batch.get_result(
            con, "spd_v2", question_id="SOM_ACE_FATALITIES_2026-08", model_key="opus5",
            pipeline_id="pl_1",
        )
        assert result is not None
        assert result["text"] == '{"spds": {}}'
        assert result["usage"]["service_tier"] == "batch"
        assert result["usage"]["batch_id"] == batch_id
        # Batch tier halves the cost in the shared cost helper.
        from forecaster.providers import compute_cost_split_usd

        _, _, batch_cost = compute_cost_split_usd("claude-opus-5", result["usage"])
        sync_usage = {k: v for k, v in result["usage"].items() if k != "service_tier"}
        _, _, sync_cost = compute_cost_split_usd("claude-opus-5", sync_usage)
        assert batch_cost == pytest.approx(sync_cost * 0.5)

    def test_collect_idempotent_and_never_clobbers(self, con, fake_adapters, batch_enabled):
        cid, batch_id = self._submit_one(con, fake_adapters)
        _FakeAdapter.fetch_results = [(cid, True, "first", {"prompt_tokens": 1}, "")]
        llm_batch.collect_batch(con, batch_id)
        _FakeAdapter.fetch_results = [(cid, True, "second", {"prompt_tokens": 1}, "")]
        counts = llm_batch.collect_batch(con, batch_id)
        assert counts["skipped_terminal"] == 1
        assert llm_batch.get_result(
            con, "spd_v2", question_id="SOM_ACE_FATALITIES_2026-08", model_key="opus5",
            pipeline_id="pl_1",
        )["text"] == "first"

    def test_errored_item_returned_with_usage_for_costing(self, con, fake_adapters, batch_enabled):
        cid, batch_id = self._submit_one(con, fake_adapters)
        _FakeAdapter.fetch_results = [
            (cid, False, "", {"prompt_tokens": 5000, "completion_tokens": 100}, "boom"),
        ]
        counts = llm_batch.collect_batch(con, batch_id)
        assert counts["errored"] == 1
        # An errored item comes back WITH status + usage so the caller can
        # cost the burned tokens before taking the sync fallback.
        hit = llm_batch.get_result(
            con, "spd_v2", question_id="SOM_ACE_FATALITIES_2026-08", model_key="opus5",
            pipeline_id="pl_1",
        )
        assert hit is not None
        assert hit["status"] == "errored"
        assert hit["usage"]["prompt_tokens"] == 5000
        assert "boom" in hit["error"]
        llm_batch.mark_fallback_sync(
            con, "spd_v2", question_id="SOM_ACE_FATALITIES_2026-08", model_key="opus5",
            pipeline_id="pl_1",
        )
        assert con.execute(
            "SELECT status FROM llm_batch_requests WHERE custom_id = ?", [cid]
        ).fetchone()[0] == "fallback_sync"

    def test_whole_batch_failure_expires_items(self, con, fake_adapters, batch_enabled):
        cid, batch_id = self._submit_one(con, fake_adapters)
        con.execute(
            "UPDATE llm_batches SET status = 'expired' WHERE batch_id = ?", [batch_id]
        )
        llm_batch.collect_batch(con, batch_id)
        assert con.execute(
            "SELECT status FROM llm_batch_requests WHERE custom_id = ?", [cid]
        ).fetchone()[0] == "expired"
        assert llm_batch.get_result(
            con, "spd_v2", question_id="SOM_ACE_FATALITIES_2026-08", model_key="opus5",
            pipeline_id="pl_1",
        ) is None

    def test_pending_batches_lists_uncollected(self, con, fake_adapters, batch_enabled):
        _, batch_id = self._submit_one(con, fake_adapters)
        pend = llm_batch.pending_batches(con)
        assert [p["batch_id"] for p in pend] == [batch_id]
        _FakeAdapter.fetch_results = []
        llm_batch.collect_batch(con, batch_id)
        assert llm_batch.pending_batches(con) == []

    def test_clear_request_bodies(self, con, fake_adapters, batch_enabled):
        cid, batch_id = self._submit_one(con, fake_adapters)
        _FakeAdapter.fetch_results = [(cid, True, "t", {}, "")]
        llm_batch.collect_batch(con, batch_id)
        llm_batch.clear_request_bodies(con, batch_id)
        assert con.execute(
            "SELECT request_body_json FROM llm_batch_requests WHERE custom_id = ?", [cid]
        ).fetchone()[0] is None


class TestBodyParity:
    """Batch request bodies must be exactly the sync builders' output."""

    def test_anthropic_spd_body_matches_sync_builder(self, con):
        from forecaster.providers import build_anthropic_body

        body = build_anthropic_body("p", "claude-opus-5", 0.2, purpose="spd_v2")
        cid = llm_batch.enqueue_request(
            con, family="spd_v2", provider="anthropic", model_id="claude-opus-5",
            request_body=body, prompt_text="p", question_id="Q1", model_key="opus5",
        )
        stored = json.loads(con.execute(
            "SELECT request_body_json FROM llm_batch_requests WHERE custom_id = ?", [cid]
        ).fetchone()[0])
        assert stored == body
        assert stored["max_tokens"] == 32768 or stored["max_tokens"] >= 16384
        assert "temperature" not in stored

    def test_openai_jsonl_line_shape(self, monkeypatch):
        # The OpenAI adapter's JSONL line must carry custom_id/method/url/body.
        captured = {}

        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"id": "file_or_batch"}

        def _fake_post(url, **kwargs):
            if url.endswith("/files"):
                captured["jsonl"] = kwargs["files"]["file"][1].decode("utf-8")
            return _Resp()

        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setattr(llm_batch.requests, "post", _fake_post)
        # The submit-time guards probe the file and the batch; answer both
        # as "ready" so this test stays about the JSONL shape.
        monkeypatch.setattr(
            llm_batch.requests, "get",
            lambda url, **kw: _FakeGet({"status": "processed"} if "/files/" in url else {"status": "in_progress"}),
        )
        monkeypatch.setattr(llm_batch, "_sleep", lambda _s: None)
        adapter = llm_batch._OpenAIBatch()
        adapter.submit([("cid-1", {"model": "gpt-5.6-sol", "messages": []})])
        line = json.loads(captured["jsonl"].splitlines()[0])
        assert line == {
            "custom_id": "cid-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-5.6-sol", "messages": []},
        }


class _FakeGet:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self.ok = ok
        self.status_code = 200 if ok else 500

    def raise_for_status(self):
        if not self.ok:
            raise RuntimeError("http error")

    def json(self):
        return self._payload


_FILE_ACCESS_ERROR = {
    "object": "list",
    "data": [{
        "code": "invalid_request",
        "message": (
            "Cannot find file file-7wYyFx3ukHp3twZpEfpXHG, or organization "
            "org-bfatceYjmHrvj9PuKL8lDweY does not have access to it."
        ),
        "param": "file_id",
        "line": None,
    }],
}


def _ladder(state):
    """The backoff sleeps only — the 3s validation-poll sleeps are noise."""
    return [s for s in state["sleeps"] if s >= 30.0]


class TestOpenAISubmitValidationGuard:
    """The 2026-09-01 failure: OpenAI accepted every batch at creation and its
    asynchronous validator then rejected the input file we had just uploaded.
    This is a provider-side incident that lasts hours, not a race a re-upload
    cures in seconds, so the guard is PATIENT (a shared time budget and a
    backoff ladder), re-creates against the SAME file first, and says what it
    did where a person looks. Giving up leaves the rows pending; the stage
    never fails on batch submission."""

    def _wire(self, monkeypatch, validation_outcomes):
        """validation_outcomes: per created batch, the status sequence the
        batch GET returns (list of payload dicts)."""
        state = {"uploads": 0, "batches": 0, "batch_polls": {}, "t": 0.0, "sleeps": []}

        def _fake_post(url, **kwargs):
            if url.endswith("/files"):
                state["uploads"] += 1
                return _FakeGet({"id": f"file-{state['uploads']}"})
            if url.endswith("/batches"):
                state["batches"] += 1
                state[f"batch-{state['batches']}_file"] = kwargs["json"]["input_file_id"]
                return _FakeGet({"id": f"batch-{state['batches']}", "status": "validating"})
            raise AssertionError(url)

        def _fake_get(url, **kwargs):
            if "/files/" in url:
                return _FakeGet({"status": "processed"})
            bid = url.rsplit("/", 1)[-1]
            seq = validation_outcomes[int(bid.split("-")[1]) - 1]
            n = state["batch_polls"].get(bid, 0)
            state["batch_polls"][bid] = n + 1
            return _FakeGet(seq[min(n, len(seq) - 1)])

        def _fake_sleep(s):
            state["sleeps"].append(s)
            state["t"] += s

        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setattr(llm_batch.requests, "post", _fake_post)
        monkeypatch.setattr(llm_batch.requests, "get", _fake_get)
        monkeypatch.setattr(llm_batch, "_sleep", _fake_sleep)
        monkeypatch.setattr(llm_batch, "_clock", lambda: state["t"])
        monkeypatch.setattr(llm_batch, "_openai_submit_budget_deadline", None)
        return state

    _ROW = [("cid-1", {"model": "gpt-5.6-sol", "messages": []})]
    _REJECT = [{"status": "validating"}, {"status": "failed", "errors": _FILE_ACCESS_ERROR}]
    _ACCEPT = [{"status": "validating"}, {"status": "in_progress"}]

    def test_first_retry_reuses_the_same_file(self, monkeypatch):
        state = self._wire(monkeypatch, [self._REJECT, self._ACCEPT])
        info = llm_batch._OpenAIBatch().submit(self._ROW)
        assert info["provider_batch_id"] == "batch-2"
        assert info["input_file_id"] == "file-1"
        # One upload: the first retry re-creates against the SAME file, which
        # is what tells "not visible yet" from "the organisation is refused".
        assert state["uploads"] == 1
        assert state["batch-2_file"] == "file-1"
        assert _ladder(state) == [30.0]

    def test_second_retry_reuploads(self, monkeypatch):
        state = self._wire(monkeypatch, [self._REJECT, self._REJECT, self._ACCEPT])
        info = llm_batch._OpenAIBatch().submit(self._ROW)
        assert info["provider_batch_id"] == "batch-3"
        assert info["input_file_id"] == "file-2"
        assert state["uploads"] == 2
        assert state["batch-3_file"] == "file-2"
        assert _ladder(state) == [30.0, 60.0]

    def test_every_rejection_is_annotated(self, monkeypatch, capsys):
        self._wire(monkeypatch, [self._REJECT, self._ACCEPT])
        llm_batch._OpenAIBatch().submit(self._ROW)
        out = capsys.readouterr().out
        assert out.count("::warning title=OpenAI batch validation rejected input file::") == 1
        assert "::error" not in out

    def test_budget_exhaustion_gives_up_with_error_annotation(self, monkeypatch, capsys):
        monkeypatch.setenv("PYTHIA_OPENAI_BATCH_SUBMIT_BUDGET_MIN", "5")
        state = self._wire(monkeypatch, [self._REJECT] * 10)
        with pytest.raises(llm_batch.OpenAIBatchValidationError) as exc_info:
            llm_batch._OpenAIBatch().submit(self._ROW)
        exc = exc_info.value
        assert exc.file_access is True
        assert exc.same_file_retry_failed is True
        # 5 minutes: sleeps 30 + 60 + 120 = 210s, then the 300s rung does not
        # fit — four attempts, and the fourth is the one that gives up.
        assert _ladder(state) == [30.0, 60.0, 120.0]
        assert exc.attempts == 4
        out = capsys.readouterr().out
        assert out.count("::warning title=OpenAI batch validation rejected input file::") == 4
        assert out.count("::error title=OpenAI batch submit gave up::") == 1
        assert "org-level rejection" in out

    def test_budget_is_shared_across_groups(self, monkeypatch):
        monkeypatch.setenv("PYTHIA_OPENAI_BATCH_SUBMIT_BUDGET_MIN", "5")
        state = self._wire(monkeypatch, [self._REJECT] * 10)
        with pytest.raises(llm_batch.OpenAIBatchValidationError):
            llm_batch._OpenAIBatch().submit(self._ROW)
        n_sleeps = len(_ladder(state))
        # A second group arriving after the budget is spent gets one create +
        # probe and gives up at once: the extra wall time is bounded ONCE per
        # process, not once per OpenAI model.
        with pytest.raises(llm_batch.OpenAIBatchValidationError) as exc_info:
            llm_batch._OpenAIBatch().submit(self._ROW)
        # Whatever the budget had left (78s here) is all it may spend: one
        # more rung at most, and never the ladder from the top again.
        assert exc_info.value.attempts <= 2
        assert sum(_ladder(state)) <= 5 * 60
        assert len(_ladder(state)) <= n_sleeps + 1

    def test_other_validation_failures_are_not_retried(self, monkeypatch):
        mismatched = {"object": "list", "data": [{"code": "mismatched_model",
                      "message": "Each batch must contain requests for a single model",
                      "param": None, "line": None}]}
        state = self._wire(monkeypatch, [
            [{"status": "failed", "errors": mismatched}],
        ])
        with pytest.raises(llm_batch.OpenAIBatchValidationError, match="mismatched_model") as exc_info:
            llm_batch._OpenAIBatch().submit(self._ROW)
        assert exc_info.value.file_access is False
        assert state["uploads"] == 1
        assert _ladder(state) == []

    def test_still_validating_at_the_deadline_is_accepted(self, monkeypatch):
        monkeypatch.setenv("PYTHIA_OPENAI_BATCH_VALIDATE_WAIT_SEC", "0")
        state = self._wire(monkeypatch, [[{"status": "validating"}]])
        info = llm_batch._OpenAIBatch().submit(self._ROW)
        assert info["provider_batch_id"] == "batch-1"
        assert state["uploads"] == 1

    def test_file_access_classifier(self):
        assert llm_batch._openai_errors_are_file_access(_FILE_ACCESS_ERROR)
        assert llm_batch._openai_errors_are_file_access(_FILE_ACCESS_ERROR["data"])
        assert not llm_batch._openai_errors_are_file_access(None)
        assert not llm_batch._openai_errors_are_file_access([])
        assert not llm_batch._openai_errors_are_file_access(
            [{"code": "mismatched_model", "message": "single model", "param": None}]
        )

    def test_backoff_ladder_repeats_its_last_rung(self):
        assert [llm_batch._openai_submit_backoff_sec(i) for i in (1, 2, 3, 4, 5, 9)] == [
            30.0, 60.0, 120.0, 300.0, 300.0, 300.0
        ]


class TestSubmitReport:
    """A submit that creates fewer batches than it wanted must say so."""

    def test_report_counts_groups_and_annotates_failures(
        self, con, fake_adapters, batch_enabled, capsys
    ):
        _enqueue_spd(con, question_id="Q1", model_key="sol", provider="openai", model_id="gpt-5.6-sol")
        _enqueue_spd(con, question_id="Q1", model_key="luna", provider="openai", model_id="gpt-5.6-luna")
        fake_adapters.fail_models = {"gpt-5.6-sol"}
        report = llm_batch.submit_pending_report(
            con, family="spd_v2", pipeline_id="pl_1", stage="fc_submit"
        )
        assert report.n_groups == 2
        assert report.n_created == 1
        assert len(report.failed) == 1
        assert report.failed[0]["model_id"] == "gpt-5.6-sol"
        assert report.failed[0]["error_class"] == "RuntimeError"
        assert "1 of 2 provider batch(es) submitted" in report.summary_line()
        statuses = dict(con.execute(
            "SELECT model_id, status FROM llm_batch_requests WHERE pipeline_id = 'pl_1'"
        ).fetchall())
        assert statuses == {"gpt-5.6-sol": "pending", "gpt-5.6-luna": "submitted"}
        assert "::warning title=Batch submit failed::" in capsys.readouterr().out

    def test_validation_error_class_is_named(self, con, fake_adapters, batch_enabled):
        _enqueue_spd(con, question_id="Q1", model_key="sol", provider="openai", model_id="gpt-5.6-sol")
        fake_adapters.fail_models = {"gpt-5.6-sol"}
        fake_adapters.fail_exc = llm_batch.OpenAIBatchValidationError(
            "kept rejecting", file_access=True, attempts=4
        )
        report = llm_batch.submit_pending_report(
            con, family="spd_v2", pipeline_id="pl_1", stage="fc_submit"
        )
        assert report.failed[0]["error_class"] == "validation:file_access"

    def test_compat_wrapper_returns_created_ids(self, con, fake_adapters, batch_enabled):
        _enqueue_spd(con, question_id="Q1", model_key="m1")
        ids = llm_batch.submit_pending(con, family="spd_v2", pipeline_id="pl_1", stage="fc_submit")
        assert len(ids) == 1 and ids[0].startswith("b_pl_1_spd_anthropic")


class TestPipelineScoping:
    """Custom ids and state queries are scoped to one pipeline execution.

    The DB artifact carries llm_batch_requests forward month to month —
    without scoping, month 2's enqueue would find month 1's succeeded rows,
    submit ZERO provider batches, and replay last month's model outputs.
    """

    def test_custom_id_scoped_by_pipeline(self):
        a = llm_batch.make_custom_id(
            "spd_v2", question_id="Q1", model_key="m1", pipeline_id="pl_A"
        )
        b = llm_batch.make_custom_id(
            "spd_v2", question_id="Q1", model_key="m1", pipeline_id="pl_A"
        )
        c = llm_batch.make_custom_id(
            "spd_v2", question_id="Q1", model_key="m1", pipeline_id="pl_B"
        )
        legacy = llm_batch.make_custom_id("spd_v2", question_id="Q1", model_key="m1")
        assert a == b
        assert a != c
        assert a != legacy
        # Worst case: full-length model key + scope hash stays in-charset.
        worst = llm_batch.make_custom_id(
            "spd_v2",
            question_id="SOM_ACE_FATALITIES_2026-08",
            model_key="gemini-3.1-pro-preview-xxxx",
            pipeline_id="pl_12345678901234567890",
        )
        assert re.match(r"^[A-Za-z0-9_-]{1,64}$", worst), worst

    def test_second_pipeline_never_replays_first(self, con, fake_adapters, batch_enabled):
        # Pipeline A submits and collects a result.
        cid_a = _enqueue_spd(con, pipeline_id="pl_A")
        [batch_a] = llm_batch.submit_pending(
            con, family="spd_v2", pipeline_id="pl_A", stage="s3"
        )
        _FakeAdapter.fetch_results = [(cid_a, True, "month-1 output", {}, "")]
        llm_batch.collect_batch(con, batch_a)

        # Pipeline B (same question, same model — e.g. next month after a
        # same-epoch re-run) must NOT see pipeline A's result...
        assert llm_batch.get_result(
            con, "spd_v2", question_id="SOM_ACE_FATALITIES_2026-08",
            model_key="opus5", pipeline_id="pl_B",
        ) is None
        # ...and its enqueue must create a FRESH pending row, not no-op.
        cid_b = _enqueue_spd(con, pipeline_id="pl_B")
        assert cid_b != cid_a
        assert con.execute(
            "SELECT status FROM llm_batch_requests WHERE custom_id = ?", [cid_b]
        ).fetchone()[0] == "pending"
        created = llm_batch.submit_pending(
            con, family="spd_v2", pipeline_id="pl_B", stage="s3"
        )
        assert len(created) == 1
        # Pipeline A's collected result is untouched.
        assert llm_batch.get_result(
            con, "spd_v2", question_id="SOM_ACE_FATALITIES_2026-08",
            model_key="opus5", pipeline_id="pl_A",
        )["text"] == "month-1 output"

    def test_submit_pending_skips_foreign_and_null_pipeline_rows(
        self, con, fake_adapters, batch_enabled
    ):
        _enqueue_spd(con, question_id="Q_old", pipeline_id="pl_old")
        llm_batch.enqueue_request(  # legacy pre-scoping row (pipeline_id NULL)
            con, family="spd_v2", provider="anthropic", model_id="claude-opus-5",
            request_body={}, prompt_text="p", question_id="Q_legacy", model_key="m",
        )
        _enqueue_spd(con, question_id="Q_new", pipeline_id="pl_new")
        created = llm_batch.submit_pending(
            con, family="spd_v2", pipeline_id="pl_new", stage="s3"
        )
        assert len(created) == 1
        [(rows, _)] = _FakeAdapter.submitted
        assert len(rows) == 1  # only pl_new's row went out

    def test_legacy_row_fallback_requires_batch_join(self, con):
        # A pre-scoping row (unscoped id) whose batch belongs to pl_A: an
        # in-flight pipeline that submitted under old code must still
        # collect after deploy — but ONLY its own pipeline.
        legacy_cid = llm_batch.enqueue_request(
            con, family="spd_v2", provider="anthropic", model_id="claude-opus-5",
            request_body={}, prompt_text="p", question_id="Q_legacy", model_key="m",
        )
        con.execute(
            """
            INSERT INTO llm_batches (batch_id, provider, family, pipeline_id, status)
            VALUES ('b_legacy', 'anthropic', 'spd_v2', 'pl_A', 'collected')
            """
        )
        con.execute(
            "UPDATE llm_batch_requests SET batch_id='b_legacy', status='succeeded', "
            "response_text='legacy result' WHERE custom_id = ?", [legacy_cid],
        )
        hit = llm_batch.get_result(
            con, "spd_v2", question_id="Q_legacy", model_key="m", pipeline_id="pl_A"
        )
        assert hit is not None and hit["text"] == "legacy result"
        assert llm_batch.get_result(
            con, "spd_v2", question_id="Q_legacy", model_key="m", pipeline_id="pl_B"
        ) is None

    def test_expire_and_purge_stale(self, con):
        cid_old_pending = _enqueue_spd(con, question_id="Q_old", pipeline_id="pl_old")
        cid_ancient = _enqueue_spd(con, question_id="Q_ancient", pipeline_id="pl_ancient")
        cid_current = _enqueue_spd(con, question_id="Q_now", pipeline_id="pl_now")
        con.execute(
            "UPDATE llm_batch_requests SET created_at = now() - INTERVAL 30 HOUR "
            "WHERE custom_id = ?", [cid_old_pending],
        )
        con.execute(
            "UPDATE llm_batch_requests SET created_at = now() - INTERVAL 60 DAY, "
            "status = 'succeeded' WHERE custom_id = ?", [cid_ancient],
        )
        counts = llm_batch.expire_and_purge_stale(con, current_pipeline_id="pl_now")
        assert con.execute(
            "SELECT status FROM llm_batch_requests WHERE custom_id = ?",
            [cid_old_pending],
        ).fetchone()[0] == "expired"
        assert con.execute(
            "SELECT COUNT(*) FROM llm_batch_requests WHERE custom_id = ?",
            [cid_ancient],
        ).fetchone()[0] == 0
        assert con.execute(
            "SELECT status FROM llm_batch_requests WHERE custom_id = ?",
            [cid_current],
        ).fetchone()[0] == "pending"
        assert counts["expired"] >= 1
        assert counts["purged"] >= 1


class TestSalvageAndCounters:
    def _submit_one(self, con):
        cid = _enqueue_spd(con)
        [batch_id] = llm_batch.submit_pending(
            con, family="spd_v2", pipeline_id="pl_1", stage="s3"
        )
        return cid, batch_id

    def test_canceled_batch_salvages_completed_results(
        self, con, fake_adapters, batch_enabled
    ):
        cid, batch_id = self._submit_one(con)
        cid2 = _enqueue_spd(con, question_id="Q_other")
        con.execute(
            "UPDATE llm_batch_requests SET batch_id = ?, status='submitted' "
            "WHERE custom_id = ?", [batch_id, cid2],
        )
        con.execute(
            "UPDATE llm_batches SET status = 'canceled' WHERE batch_id = ?", [batch_id]
        )
        # The provider preserved the item completed before cancellation.
        _FakeAdapter.fetch_results = [(cid, True, "pre-cancel result", {}, "")]
        counts = llm_batch.collect_batch(con, batch_id)
        assert counts["succeeded"] == 1
        assert counts["expired"] == 1  # the un-run remainder
        assert llm_batch.get_result(
            con, "spd_v2", question_id="SOM_ACE_FATALITIES_2026-08",
            model_key="opus5", pipeline_id="pl_1",
        )["text"] == "pre-cancel result"
        status, n_succ, n_exp = con.execute(
            "SELECT status, n_succeeded, n_expired FROM llm_batches WHERE batch_id = ?",
            [batch_id],
        ).fetchone()
        assert status == "canceled"  # terminal status preserved
        assert (n_succ, n_exp) == (1, 1)

    def test_recollect_preserves_counters(self, con, fake_adapters, batch_enabled):
        cid, batch_id = self._submit_one(con)
        _FakeAdapter.fetch_results = [(cid, True, "t", {}, "")]
        llm_batch.collect_batch(con, batch_id)
        _FakeAdapter.fetch_results = [(cid, True, "t2", {}, "")]
        llm_batch.collect_batch(con, batch_id)  # poller double-fire
        n_succ = con.execute(
            "SELECT n_succeeded FROM llm_batches WHERE batch_id = ?", [batch_id]
        ).fetchone()[0]
        assert n_succ == 1  # not zeroed by the re-collect


class TestEmptyBatchDiagnosis:
    """A batch that yields nothing must explain itself.

    On 2026-07-29 both OpenAI batches returned zero results (succeeded=0,
    errored=0, expired=12/4). Every request fell back to synchronous
    full-price calls — 67% of that run's forecaster spend — and the run still
    looked healthy, because the provider's terminal status and error payload
    were never persisted. The cause was unrecoverable after the fact.
    """

    def _submitted_batch(self, con, fake_adapters, batch_enabled):
        _enqueue_spd(con, question_id="Q1", model_key="m1")
        _enqueue_spd(con, question_id="Q2", model_key="m2")
        ids = llm_batch.submit_pending(
            con, family="spd_v2", pipeline_id="pl_1", stage="fc_submit"
        )
        assert ids
        return ids[0]

    def test_zero_result_batch_records_provider_state(
        self, con, fake_adapters, batch_enabled
    ):
        batch_id = self._submitted_batch(con, fake_adapters, batch_enabled)
        fake_adapters.fetch_results = []          # provider returns nothing
        fake_adapters.poll_state = "failed"

        counts = llm_batch.collect_batch(con, batch_id)
        assert counts["succeeded"] == 0 and counts["expired"] == 2

        err = con.execute(
            "SELECT error_text FROM llm_batches WHERE batch_id = ?", [batch_id]
        ).fetchone()[0]
        assert err, "a batch that returned nothing must record why"
        assert "failed" in err          # the provider's terminal state
        assert "n_expired" in err

    def test_zero_result_batch_warns(self, con, fake_adapters, batch_enabled, capsys):
        batch_id = self._submitted_batch(con, fake_adapters, batch_enabled)
        fake_adapters.fetch_results = []
        llm_batch.collect_batch(con, batch_id)
        assert "::warning title=Batch returned no results::" in capsys.readouterr().out

    def test_successful_batch_records_no_error(self, con, fake_adapters, batch_enabled):
        batch_id = self._submitted_batch(con, fake_adapters, batch_enabled)
        rows = con.execute(
            "SELECT custom_id FROM llm_batch_requests WHERE batch_id = ?", [batch_id]
        ).fetchall()
        fake_adapters.fetch_results = [
            (cid, True, '{"ok": 1}', {"prompt_tokens": 1, "completion_tokens": 1}, "")
            for (cid,) in rows
        ]
        counts = llm_batch.collect_batch(con, batch_id)
        assert counts["succeeded"] == len(rows)
        err = con.execute(
            "SELECT error_text FROM llm_batches WHERE batch_id = ?", [batch_id]
        ).fetchone()[0]
        assert not err

    def test_poll_failure_does_not_break_collect(
        self, con, fake_adapters, batch_enabled, monkeypatch
    ):
        """The sync fallback must still run even if the diagnosis fails."""
        batch_id = self._submitted_batch(con, fake_adapters, batch_enabled)
        fake_adapters.fetch_results = []

        def _boom(self, provider_batch_id):
            raise RuntimeError("provider unreachable")

        monkeypatch.setattr(fake_adapters, "poll", _boom, raising=False)
        counts = llm_batch.collect_batch(con, batch_id)   # must not raise
        assert counts["expired"] == 2
        err = con.execute(
            "SELECT error_text FROM llm_batches WHERE batch_id = ?", [batch_id]
        ).fetchone()[0]
        assert "poll_error" in err


class TestResubmitAtCollect:
    """The collect stage re-batches what never got a batch result before it
    falls back to sync. On 2026-09-01 it had 210 unserved OpenAI requests,
    their bodies, and 23 hours of wait budget — and one move: full price."""

    def _failed_openai_batch(self, con, fake_adapters, errors, *, n=2):
        cids = [
            _enqueue_spd(con, question_id=f"Q{i}", model_key=f"m{i}",
                         provider="openai", model_id="gpt-5.6-sol")
            for i in range(n)
        ]
        ids = llm_batch.submit_pending(con, family="spd_v2", pipeline_id="pl_1", stage="fc_submit")
        assert len(ids) == 1
        fake_adapters.poll_state = "failed"
        fake_adapters.poll_detail = json.dumps({"errors": errors}) if errors is not None else "{}"
        fake_adapters.fetch_results = []
        assert llm_batch.poll_batch(con, ids[0]).state == "failed"   # the poller's tick
        counts = llm_batch.collect_batch(con, ids[0])
        assert counts["expired"] == n
        fake_adapters.poll_state = "ended"
        fake_adapters.poll_detail = ""
        return ids[0], cids

    def test_batch_failure_class(self):
        def _text(state, errors=None, detail=True):
            d = {"provider_state": state}
            if detail:
                d["detail"] = json.dumps({"errors": errors} if errors is not None else {})
            return json.dumps(d)

        assert llm_batch._batch_failure_class(_text("failed", _FILE_ACCESS_ERROR)) == "file_access"
        assert llm_batch._batch_failure_class(_text("failed", {"object": "list", "data": [
            {"code": "mismatched_model", "message": "single model", "param": None}]})) == "validation_other"
        assert llm_batch._batch_failure_class(_text("failed")) == "failed_empty"
        assert llm_batch._batch_failure_class(_text("failed", detail=False)) == "failed_empty"
        assert llm_batch._batch_failure_class(_text("ended", _FILE_ACCESS_ERROR)) is None
        assert llm_batch._batch_failure_class("not json") is None
        assert llm_batch._batch_failure_class(None) is None

    def test_unserved_selects_pending_and_rejected_batches_only(
        self, con, fake_adapters, batch_enabled, monkeypatch
    ):
        # B: expired under a batch rejected for the file-access reason.
        _, (b,) = self._failed_openai_batch(con, fake_adapters, _FILE_ACCESS_ERROR, n=1)
        # C: expired under a deterministic validation failure — never re-batched.
        c = _enqueue_spd(con, question_id="C", model_key="c", provider="openai", model_id="gpt-5.6-sol")
        ids = llm_batch.submit_pending(con, family="spd_v2", pipeline_id="pl_1", stage="fc_submit")
        fake_adapters.poll_state = "failed"
        fake_adapters.poll_detail = json.dumps({"errors": {"data": [
            {"code": "mismatched_model", "message": "single model", "param": None}]}})
        llm_batch.collect_batch(con, ids[-1])
        fake_adapters.poll_state = "ended"
        fake_adapters.poll_detail = ""
        # D: expired under a canceled batch (the wait cap) — not a failure to re-batch.
        d = _enqueue_spd(con, question_id="D", model_key="d", provider="openai", model_id="gpt-5.6-sol")
        ids = llm_batch.submit_pending(con, family="spd_v2", pipeline_id="pl_1", stage="fc_submit")
        llm_batch.cancel_batch(con, ids[-1])
        llm_batch.collect_batch(con, ids[-1])
        # A: never submitted (its submit failed) — still pending.
        a = _enqueue_spd(con, question_id="A", model_key="a", provider="openai", model_id="gpt-5.6-luna")
        # E: pending but its provider is excluded from batching.
        e = _enqueue_spd(con, question_id="E", model_key="e", provider="anthropic", model_id="claude-opus-5")
        monkeypatch.setenv("PYTHIA_BATCH_PROVIDERS", "openai")
        # F: a row from another pipeline.
        _enqueue_spd(con, question_id="F", model_key="f", provider="openai", model_id="gpt-5.6-sol", pipeline_id="pl_2")

        rows = llm_batch.unserved_requests(con, pipeline_id="pl_1", families=("spd_v2",))
        got = {r["custom_id"]: r["reason"] for r in rows}
        assert got == {a: "never_submitted", b: "batch_failed:file_access"}
        assert c not in got and d not in got and e not in got

    def test_reset_for_resubmit_keeps_bodies(self, con, fake_adapters, batch_enabled):
        _, cids = self._failed_openai_batch(con, fake_adapters, _FILE_ACCESS_ERROR)
        assert llm_batch.reset_for_resubmit(con, cids) == 2
        rows = con.execute(
            "SELECT status, batch_id, request_body_json IS NOT NULL FROM llm_batch_requests "
            "WHERE custom_id IN (?, ?)", cids
        ).fetchall()
        assert all(r == ("pending", None, True) for r in rows)

    def test_resubmit_end_to_end_replays_from_new_batch(
        self, con, fake_adapters, batch_enabled, capsys
    ):
        old_id, cids = self._failed_openai_batch(con, fake_adapters, _FILE_ACCESS_ERROR)
        fake_adapters.fetch_results = [
            (cid, True, '{"spds": 1}', {"prompt_tokens": 3, "completion_tokens": 2}, "")
            for cid in cids
        ]
        capsys.readouterr()   # drop the setup's "returned no results" warning
        report = llm_batch.resubmit_unserved(
            con, pipeline_id="pl_1", families=("spd_v2",), stage="fc_collect_resubmit",
            run_id="fc_1", wait_min=5, poll_sec=1,
        )
        assert report["n_candidates"] == 2 and report["n_reset"] == 2
        assert len(report["created"]) == 1
        assert report["counts"] == {"succeeded": 2, "errored": 0, "expired": 0}
        new_id = report["created"][0]
        rows = con.execute(
            "SELECT status, batch_id FROM llm_batch_requests WHERE custom_id IN (?, ?)", cids
        ).fetchall()
        assert all(r == ("succeeded", new_id) for r in rows)
        stage, status = con.execute(
            "SELECT stage, status FROM llm_batches WHERE batch_id = ?", [new_id]
        ).fetchone()
        assert (stage, status) == ("fc_collect_resubmit", "collected")
        old_status, old_err = con.execute(
            "SELECT status, error_text FROM llm_batches WHERE batch_id = ?", [old_id]
        ).fetchone()
        assert old_status == "failed"
        assert json.loads(old_err)["resubmitted_as"] == [new_id]
        hit = llm_batch.get_result(con, "spd_v2", question_id="Q0", model_key="m0", pipeline_id="pl_1")
        assert hit and hit["status"] == "succeeded"
        assert hit["usage"]["service_tier"] == "batch"
        assert "::warning" not in capsys.readouterr().out

    def test_resubmit_picks_up_never_submitted_rows(self, con, fake_adapters, batch_enabled):
        cid = _enqueue_spd(con, question_id="A", model_key="a", provider="openai", model_id="gpt-5.6-luna")
        fake_adapters.fetch_results = [(cid, True, "ok", {"prompt_tokens": 1}, "")]
        report = llm_batch.resubmit_unserved(
            con, pipeline_id="pl_1", families=("spd_v2",), stage="fc_collect_resubmit",
            wait_min=5, poll_sec=1,
        )
        assert report["reasons"] == {"never_submitted": 1}
        assert report["counts"]["succeeded"] == 1

    def test_resubmit_deadline_cancels_and_salvages(
        self, con, fake_adapters, batch_enabled, monkeypatch, capsys
    ):
        _, cids = self._failed_openai_batch(con, fake_adapters, _FILE_ACCESS_ERROR)
        state = {"t": 0.0}
        monkeypatch.setattr(llm_batch, "_clock", lambda: state["t"])
        monkeypatch.setattr(llm_batch, "_sleep", lambda s: state.__setitem__("t", state["t"] + s))
        fake_adapters.poll_state = "in_progress"
        fake_adapters.fetch_results = []          # nothing completed before the cancel
        report = llm_batch.resubmit_unserved(
            con, pipeline_id="pl_1", families=("spd_v2",), stage="fc_collect_resubmit",
            wait_min=1, poll_sec=10,
        )
        assert fake_adapters.canceled == ["prov_2"]
        assert report["counts"]["expired"] == 2
        rows = con.execute(
            "SELECT status FROM llm_batch_requests WHERE custom_id IN (?, ?)", cids
        ).fetchall()
        assert {r[0] for r in rows} == {"expired"}
        assert llm_batch.get_result(con, "spd_v2", question_id="Q0", model_key="m0", pipeline_id="pl_1") is None
        assert "::warning title=Re-batched batch still running at deadline::" in capsys.readouterr().out

    def test_resubmit_disabled_by_flag(self, con, fake_adapters, batch_enabled, monkeypatch):
        self._failed_openai_batch(con, fake_adapters, _FILE_ACCESS_ERROR)
        monkeypatch.setenv("PYTHIA_BATCH_RESUBMIT_AT_COLLECT", "0")
        n_before = len(fake_adapters.submitted)
        report = llm_batch.resubmit_unserved(
            con, pipeline_id="pl_1", families=("spd_v2",), stage="fc_collect_resubmit"
        )
        assert report["enabled"] is False
        assert len(fake_adapters.submitted) == n_before

    def test_resubmit_bounded_by_max_wait_from_original_submit(
        self, con, fake_adapters, batch_enabled, capsys
    ):
        old_id, _ = self._failed_openai_batch(con, fake_adapters, _FILE_ACCESS_ERROR)
        con.execute(
            "UPDATE llm_batches SET submitted_at = submitted_at - INTERVAL 25 HOUR WHERE batch_id = ?",
            [old_id],
        )
        n_before = len(fake_adapters.submitted)
        report = llm_batch.resubmit_unserved(
            con, pipeline_id="pl_1", families=("spd_v2",), stage="fc_collect_resubmit"
        )
        assert report.get("skipped") == "past_max_wait"
        assert len(fake_adapters.submitted) == n_before
        assert "::warning title=Collect re-batch skipped::" in capsys.readouterr().out

    def test_resubmit_never_raises(self, con, fake_adapters, batch_enabled, monkeypatch, capsys):
        self._failed_openai_batch(con, fake_adapters, _FILE_ACCESS_ERROR)

        def _boom(*a, **k):
            raise RuntimeError("provider unreachable")

        monkeypatch.setattr(llm_batch, "submit_pending_report", _boom)
        report = llm_batch.resubmit_unserved(
            con, pipeline_id="pl_1", families=("spd_v2",), stage="fc_collect_resubmit"
        )
        assert "provider unreachable" in report["error"]
        assert "::warning title=Collect re-batch failed::" in capsys.readouterr().out

    def test_resubmit_no_candidates_is_quiet(self, con, fake_adapters, batch_enabled, capsys):
        report = llm_batch.resubmit_unserved(
            con, pipeline_id="pl_1", families=("spd_v2",), stage="fc_collect_resubmit"
        )
        assert report["n_candidates"] == 0 and report["created"] == []
        assert capsys.readouterr().out == ""
