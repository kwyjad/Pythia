# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Staged Horizon Scanner (Batch-API waves) — Phase 4 seam tests.

Covers horizon_scanner/hs_batch.py plus the seams wired into
regime_change_llm._run_rc_for_single_hazard and
triage._run_triage_for_single_hazard:

- family_mode matrix per stage (full is inert);
- submit mode enqueues llm_batch_requests rows whose bodies are
  byte-identical to build_body_for_spec on the pass's resolved ModelSpec,
  and never calls the sync model;
- submit mode with a non-batchable provider enqueues nothing (the call
  defers to the collect stage's sync fallback);
- collect mode replays a stored batch result through the unmodified
  parse/merge path, rich-logs it exactly once across repeated replays, and
  falls back to the sync call on a miss;
- grounding packs persisted by a submit stage round-trip through
  load_grounding_pack for the collect stages;
- hs_stage_state round-trip (merged RC results carried hs_rc_collect →
  hs_finalize);
- run_date_from_run_id anchors the seasonal filter to the hs_submit date.
"""

from __future__ import annotations

import json

import pytest

from horizon_scanner import hs_batch
from horizon_scanner import regime_change_llm as rc_llm
from horizon_scanner import triage as triage_mod
from pythia import llm_batch


_RC_RESPONSE = json.dumps(
    {
        "likelihood": 0.4,
        "magnitude": 0.5,
        "direction": "up",
        "window": "month_1-2",
        "rationale_bullets": ["signal"],
        "trigger_signals": [],
    }
)

_TRIAGE_RESPONSE = json.dumps(
    {
        "triage_score": 0.35,
        "drivers": ["driver-a"],
        "data_quality": {"status": "ok"},
        "scenario_stub": "stub",
        "confidence_note": "note",
    }
)


@pytest.fixture()
def staged_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{tmp_path}/hs-stage-test.duckdb")
    monkeypatch.setenv("PYTHIA_BATCH_API_ENABLED", "1")
    monkeypatch.setenv("PYTHIA_BATCH_PROVIDERS", "openai,anthropic,google")
    monkeypatch.delenv("PYTHIA_HS_RC_PASSES", raising=False)
    monkeypatch.delenv("PYTHIA_HS_TRIAGE_PASSES", raising=False)
    hs_batch.reset()
    yield
    hs_batch.reset()


def _forbid_sync(monkeypatch: pytest.MonkeyPatch, module, attr: str) -> list:
    calls: list = []

    async def _boom(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError(f"{attr} must not be called in this scenario")

    monkeypatch.setattr(module, attr, _boom)
    return calls


def _stub_sync(monkeypatch: pytest.MonkeyPatch, module, attr: str, response: str) -> list:
    calls: list = []

    async def _fake(prompt_text, *, run_id=None, fallback_specs=None, pass_idx=1):
        calls.append(pass_idx)
        from forecaster.providers import ModelSpec

        spec = ModelSpec(
            name="stub", provider="google", model_id="stub-model", active=True
        )
        return response, {"total_tokens": 10}, "", spec

    monkeypatch.setattr(module, attr, _fake)
    return calls


def _run_rc(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(rc_llm, "build_rc_prompt", lambda *a, **k: "RC PROMPT")
    return rc_llm._run_rc_for_single_hazard(
        "FL",
        "Testland",
        "TST",
        resolver_features={},
        evidence_pack=None,
        run_id="hs_20260701T000000",
        fallback_specs=[],
    )


def _run_triage(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(triage_mod, "build_triage_prompt", lambda *a, **k: "TRIAGE PROMPT")
    return triage_mod._run_triage_for_single_hazard(
        "FL",
        "Testland",
        "TST",
        resolver_features={},
        evidence_pack=None,
        rc_result_for_hazard={"likelihood": 0.1, "magnitude": 0.1},
        run_id="hs_20260701T000000",
        fallback_specs=[],
    )


# ---------------------------------------------------------------------------
# Stage/mode plumbing
# ---------------------------------------------------------------------------


def test_family_mode_matrix() -> None:
    try:
        hs_batch.set_stage("full")
        assert hs_batch.family_mode("hs_rc") == "full"
        assert hs_batch.family_mode("hs_triage") == "full"
        assert hs_batch.triage_enabled() and hs_batch.finalize_enabled()

        hs_batch.set_stage("hs_submit", "pl_x")
        assert hs_batch.family_mode("hs_rc") == "submit"
        assert hs_batch.family_mode("hs_triage") == "skip"
        assert not hs_batch.triage_enabled()
        assert not hs_batch.finalize_enabled()

        hs_batch.set_stage("hs_rc_collect", "pl_x")
        assert hs_batch.family_mode("hs_rc") == "collect"
        assert hs_batch.family_mode("hs_triage") == "submit"
        assert hs_batch.triage_enabled()
        assert not hs_batch.finalize_enabled()

        hs_batch.set_stage("hs_finalize", "pl_x")
        assert hs_batch.family_mode("hs_rc") == "collect"
        assert hs_batch.family_mode("hs_triage") == "collect"
        assert hs_batch.triage_enabled()
        assert hs_batch.finalize_enabled()
    finally:
        hs_batch.reset(close=False)

    with pytest.raises(ValueError):
        hs_batch.set_stage("bogus")


def test_run_date_from_run_id() -> None:
    from datetime import date

    assert hs_batch.run_date_from_run_id("hs_20260701T081500") == date(2026, 7, 1)
    # Malformed ids fall back to today (never crash the pipeline).
    assert hs_batch.run_date_from_run_id("garbage") is not None
    assert hs_batch.run_date_from_run_id("") is not None


# ---------------------------------------------------------------------------
# RC submit mode
# ---------------------------------------------------------------------------


def test_rc_submit_enqueues_and_returns_pending(staged_env, monkeypatch) -> None:
    hs_batch.set_stage("hs_submit", "pl_test")
    _forbid_sync(monkeypatch, rc_llm, "_call_rc_model")

    merged = _run_rc(monkeypatch)
    assert merged["status"] == "pending_batch"
    assert merged["valid"] is False

    row = hs_batch.batch_con().execute(
        "SELECT provider, model_id, family, iso3, hazard_code, pass_idx, status, request_body_json "
        "FROM llm_batch_requests"
    ).fetchone()
    assert row is not None
    provider, model_id, family, iso3, hazard, pass_idx, status, body_json = row
    assert (family, iso3, hazard, pass_idx, status) == ("hs_rc", "TST", "FL", 1, "pending")

    # Body byte-identical to the sync request the pass would have sent.
    from forecaster.providers import build_body_for_spec

    spec = rc_llm._rc_model_spec(1)
    assert (provider, model_id) == (spec.provider, spec.model_id)
    assert json.loads(body_json) == build_body_for_spec(spec, "RC PROMPT", rc_llm.HS_TEMPERATURE)


def test_rc_submit_skips_non_batchable_provider(staged_env, monkeypatch) -> None:
    monkeypatch.setenv("PYTHIA_BATCH_PROVIDERS", "anthropic")  # RC model is google
    hs_batch.set_stage("hs_submit", "pl_test")
    _forbid_sync(monkeypatch, rc_llm, "_call_rc_model")

    merged = _run_rc(monkeypatch)
    assert merged["status"] == "pending_batch"
    assert (
        hs_batch.batch_con().execute("SELECT COUNT(*) FROM llm_batch_requests").fetchone()[0]
        == 0
    )


# ---------------------------------------------------------------------------
# RC collect mode
# ---------------------------------------------------------------------------


def _seed_rc_result(response: str) -> str:
    spec = rc_llm._rc_model_spec(1)
    cid = llm_batch.enqueue_request(
        hs_batch.batch_con(),
        family="hs_rc",
        provider=spec.provider,
        model_id=spec.model_id,
        request_body={"model": spec.model_id},
        prompt_text="RC PROMPT",
        iso3="TST",
        hazard_code="FL",
        pass_idx=1,
    )
    hs_batch.batch_con().execute(
        "UPDATE llm_batch_requests SET status='succeeded', response_text=?, usage_json=? "
        "WHERE custom_id = ?",
        [
            response,
            json.dumps({"prompt_tokens": 900, "completion_tokens": 100,
                        "total_tokens": 1000, "service_tier": "batch"}),
            cid,
        ],
    )
    return cid


def test_rc_collect_replays_and_logs_once(staged_env, monkeypatch) -> None:
    hs_batch.set_stage("hs_finalize", "pl_test")
    _seed_rc_result(_RC_RESPONSE)
    _forbid_sync(monkeypatch, rc_llm, "_call_rc_model")

    merged_first = _run_rc(monkeypatch)
    assert merged_first["status"] == "ok"
    assert merged_first["likelihood"] == pytest.approx(0.4)
    assert merged_first["magnitude"] == pytest.approx(0.5)

    # Replaying again (stage re-run) is a no-op for llm_calls: exactly one
    # rich-logged row for the replayed batch result.
    merged_second = _run_rc(monkeypatch)
    assert merged_second["likelihood"] == pytest.approx(0.4)
    n_logged = hs_batch.batch_con().execute(
        "SELECT COUNT(*) FROM llm_calls "
        "WHERE hs_run_id = ? AND UPPER(iso3) = ? AND UPPER(hazard_code) = ?",
        ["hs_20260701T000000", "TST", "RC_FL_PASS_1"],
    ).fetchone()[0]
    assert n_logged == 1


def test_rc_collect_miss_falls_back_to_sync(staged_env, monkeypatch) -> None:
    hs_batch.set_stage("hs_rc_collect", "pl_test")
    sync_calls = _stub_sync(monkeypatch, rc_llm, "_call_rc_model", _RC_RESPONSE)

    merged = _run_rc(monkeypatch)
    assert sync_calls == [1]
    assert merged["status"] == "ok"
    assert merged["likelihood"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Triage submit / collect
# ---------------------------------------------------------------------------


def test_triage_submit_enqueues_and_returns_pending(staged_env, monkeypatch) -> None:
    hs_batch.set_stage("hs_rc_collect", "pl_test")
    _forbid_sync(monkeypatch, triage_mod, "_call_triage_model")

    merged = _run_triage(monkeypatch)
    assert merged["status"] == "pending_batch"
    assert merged["valid"] is False
    assert merged["data_quality"]["status"] == "pending_batch"

    row = hs_batch.batch_con().execute(
        "SELECT family, iso3, hazard_code, pass_idx, request_body_json FROM llm_batch_requests"
    ).fetchone()
    assert row[:4] == ("hs_triage", "TST", "FL", 1)

    from forecaster.providers import build_body_for_spec

    spec = triage_mod._triage_model_spec(1)
    assert json.loads(row[4]) == build_body_for_spec(
        spec, "TRIAGE PROMPT", triage_mod.HS_TEMPERATURE
    )


def test_triage_collect_replays_stored_result(staged_env, monkeypatch) -> None:
    hs_batch.set_stage("hs_finalize", "pl_test")
    spec = triage_mod._triage_model_spec(1)
    cid = llm_batch.enqueue_request(
        hs_batch.batch_con(),
        family="hs_triage",
        provider=spec.provider,
        model_id=spec.model_id,
        request_body={"model": spec.model_id},
        prompt_text="TRIAGE PROMPT",
        iso3="TST",
        hazard_code="FL",
        pass_idx=1,
    )
    hs_batch.batch_con().execute(
        "UPDATE llm_batch_requests SET status='succeeded', response_text=?, usage_json=? "
        "WHERE custom_id = ?",
        [_TRIAGE_RESPONSE, json.dumps({"total_tokens": 500, "service_tier": "batch"}), cid],
    )
    _forbid_sync(monkeypatch, triage_mod, "_call_triage_model")

    merged = _run_triage(monkeypatch)
    assert merged["status"] == "ok"
    assert merged["triage_score"] == pytest.approx(0.35)
    assert merged["drivers"] == ["driver-a"]


def test_triage_collect_miss_falls_back_to_sync(staged_env, monkeypatch) -> None:
    hs_batch.set_stage("hs_finalize", "pl_test")
    sync_calls = _stub_sync(monkeypatch, triage_mod, "_call_triage_model", _TRIAGE_RESPONSE)

    merged = _run_triage(monkeypatch)
    assert sync_calls == [1]
    assert merged["status"] == "ok"
    assert merged["triage_score"] == pytest.approx(0.35)


# ---------------------------------------------------------------------------
# Cross-stage carry state + grounding reuse
# ---------------------------------------------------------------------------


def test_stage_state_roundtrip(staged_env) -> None:
    hs_batch.set_stage("hs_rc_collect", "pl_test")
    payload = {
        "hazards": {"FL": {"likelihood": 0.4, "status": "ok"}},
        "status": "ok",
        "pass_results": [],
    }
    hs_batch.save_stage_state("hs_x", "tst", "rc_result", payload)
    loaded = hs_batch.load_stage_state("hs_x", "TST", "rc_result")
    assert loaded == payload
    # Overwrite (stage re-run) keeps exactly one row.
    hs_batch.save_stage_state("hs_x", "TST", "rc_result", {"status": "ok"})
    assert hs_batch.load_stage_state("hs_x", "TST", "rc_result") == {"status": "ok"}
    assert hs_batch.load_stage_state("hs_x", "TST", "missing") is None


def test_grounding_pack_roundtrip(staged_env) -> None:
    from horizon_scanner.db_writer import log_hs_hazard_tail_packs_to_db

    hs_batch.set_stage("hs_rc_collect", "pl_test")
    # Force db_writer onto the same tmp DB before it opens its own connect().
    log_hs_hazard_tail_packs_to_db(
        "hs_x",
        [{
            "iso3": "TST",
            "hazard_code": "FL",
            "rc_level": 0,
            "rc_score": 0.0,
            "rc_direction": "unknown",
            "rc_window": "",
            "query": "rc_grounding: Testland flooding",
            "markdown": "## evidence",
            "sources": [{"url": "https://example.org"}],
            "grounded": True,
            "grounding_debug": {"backend": "brave"},
            "structural_context": "context",
            "recent_signals": ["signal-1"],
        }],
    )
    pack = hs_batch.load_grounding_pack("hs_x", "TST", "FL", prefer_prefix="rc_grounding")
    assert pack is not None
    assert pack["query"] == "Testland flooding"  # prefix stripped
    assert pack["markdown"] == "## evidence"
    assert pack["sources"] == [{"url": "https://example.org"}]
    assert pack["grounded"] is True
    assert pack["recent_signals"] == ["signal-1"]
    assert hs_batch.load_grounding_pack("hs_x", "TST", "TC", prefer_prefix="rc_grounding") is None
