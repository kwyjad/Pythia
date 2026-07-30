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
    fetch_results: list = []

    def __init__(self):
        pass

    def submit(self, rows, **kwargs):
        type(self).submitted.append((list(rows), kwargs))
        return {"provider_batch_id": f"prov_{len(type(self).submitted)}", "input_file_id": None}

    def poll(self, provider_batch_id):
        return llm_batch.BatchStatus(provider_batch_id, type(self).poll_state, {})

    def fetch(self, provider_batch_id):
        yield from type(self).fetch_results

    def cancel(self, provider_batch_id):
        pass


@pytest.fixture()
def fake_adapters(monkeypatch: pytest.MonkeyPatch):
    _FakeAdapter.submitted = []
    _FakeAdapter.poll_state = "ended"
    _FakeAdapter.fetch_results = []
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
        adapter = llm_batch._OpenAIBatch()
        adapter.submit([("cid-1", {"model": "gpt-5.6-sol", "messages": []})])
        line = json.loads(captured["jsonl"].splitlines()[0])
        assert line == {
            "custom_id": "cid-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": "gpt-5.6-sol", "messages": []},
        }


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
