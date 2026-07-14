# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Single-pass RC/triage (July 2026).

With the default config (same model both passes, temperature 0.0, one prompt
built once and reused), pass 2 was a deterministic duplicate of pass 1 and
the mean-based merges were no-ops — pure 2x cost/latency. The pass count is
now env-configurable (PYTHIA_HS_RC_PASSES / PYTHIA_HS_TRIAGE_PASSES, default
1, max 2); these tests pin the env parsing, the merge idempotency the
single-pass path relies on, and the actual LLM call count per pass setting.
"""

from __future__ import annotations

import json

import pytest

from horizon_scanner import regime_change_llm as rc_llm
from horizon_scanner import triage as triage_mod


# ---------------------------------------------------------------------------
# Pass-count env parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [(None, 1), ("1", 1), ("2", 2), ("5", 2), ("0", 1), ("-3", 1), ("garbage", 1)],
)
def test_rc_pass_count_env(monkeypatch: pytest.MonkeyPatch, raw, expected) -> None:
    if raw is None:
        monkeypatch.delenv("PYTHIA_HS_RC_PASSES", raising=False)
    else:
        monkeypatch.setenv("PYTHIA_HS_RC_PASSES", raw)
    assert rc_llm._rc_pass_count() == expected


@pytest.mark.parametrize(
    "raw, expected",
    [(None, 1), ("2", 2), ("7", 2), ("0", 1), ("x", 1)],
)
def test_triage_pass_count_env(monkeypatch: pytest.MonkeyPatch, raw, expected) -> None:
    if raw is None:
        monkeypatch.delenv("PYTHIA_HS_TRIAGE_PASSES", raising=False)
    else:
        monkeypatch.setenv("PYTHIA_HS_TRIAGE_PASSES", raw)
    assert triage_mod._triage_pass_count() == expected


# ---------------------------------------------------------------------------
# Merge idempotency — the single-pass path is merge(p, p)
# ---------------------------------------------------------------------------

def test_rc_merge_is_idempotent() -> None:
    p = {
        "likelihood": 0.4,
        "magnitude": 0.5,
        "direction": "up",
        "window": "month_1-2",
        "rationale_bullets": ["a", "b"],
        "trigger_signals": [],
        "valid": True,
    }
    merged = rc_llm._merge_single_hazard_passes(p, p)
    assert merged["likelihood"] == pytest.approx(0.4)
    assert merged["magnitude"] == pytest.approx(0.5)
    assert merged["direction"] == "up"
    assert merged["window"] == "month_1-2"
    assert merged["rationale_bullets"] == ["a", "b"]
    assert merged["valid"] is True
    assert merged["status"] == "ok"


def test_rc_merge_invalid_single_pass_is_error() -> None:
    p = {"likelihood": None, "magnitude": None, "valid": False}
    merged = rc_llm._merge_single_hazard_passes(p, p)
    assert merged["valid"] is False
    assert merged["status"] == "error"


def test_triage_merge_is_idempotent() -> None:
    p = {
        "score": 0.7,
        "score_valid": True,
        "drivers": ["d1", "d2"],
        "data_quality": {"note": "x"},
        "scenario_stub": "stub",
        "confidence_note": "conf",
    }
    merged = triage_mod._merge_single_triage_passes(p, p)
    assert merged["triage_score"] == pytest.approx(0.7)
    assert merged["drivers"] == ["d1", "d2"]
    assert merged["status"] == "ok"
    assert merged["valid"] is True


# ---------------------------------------------------------------------------
# LLM call count follows the pass setting
# ---------------------------------------------------------------------------

def _run_rc_with_stub(monkeypatch: pytest.MonkeyPatch, tmp_path) -> tuple[dict, list]:
    calls: list[int] = []

    async def _fake_call_rc_model(prompt_text, *, run_id=None, fallback_specs=None, pass_idx=1):
        calls.append(pass_idx)
        response = json.dumps(
            {
                "likelihood": 0.4,
                "magnitude": 0.5,
                "direction": "up",
                "window": "month_1-2",
                "rationale_bullets": ["signal"],
                "trigger_signals": [],
            }
        )
        spec = rc_llm.ModelSpec(
            name="stub", provider="google", model_id="stub-model", active=True
        )
        return response, {"total_tokens": 10}, "", spec

    # Isolate any DB-touching best-effort loaders inside the function.
    monkeypatch.setenv("PYTHIA_DB_URL", f"duckdb:///{tmp_path}/rc-test.duckdb")
    monkeypatch.setattr(rc_llm, "_call_rc_model", _fake_call_rc_model)
    monkeypatch.setattr(rc_llm, "build_rc_prompt", lambda *a, **k: "PROMPT")
    monkeypatch.setattr(rc_llm, "log_hs_llm_call", lambda **k: None)

    merged = rc_llm._run_rc_for_single_hazard(
        "FL",
        "Testland",
        "TST",
        resolver_features={},
        evidence_pack=None,
        run_id="hs_test",
        fallback_specs=[],
    )
    return merged, calls


def test_rc_single_pass_makes_one_llm_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.delenv("PYTHIA_HS_RC_PASSES", raising=False)  # default = 1
    merged, calls = _run_rc_with_stub(monkeypatch, tmp_path)
    assert calls == [1]
    assert merged["likelihood"] == pytest.approx(0.4)
    assert merged["magnitude"] == pytest.approx(0.5)
    assert merged["status"] == "ok"
    assert merged["valid"] is True


def test_rc_two_pass_makes_two_llm_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("PYTHIA_HS_RC_PASSES", "2")
    merged, calls = _run_rc_with_stub(monkeypatch, tmp_path)
    assert calls == [1, 2]
    assert merged["likelihood"] == pytest.approx(0.4)
    assert merged["status"] == "ok"
