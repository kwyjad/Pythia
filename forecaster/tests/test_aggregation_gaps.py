# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Test-gap closure from the July 2026 audit (docs/audit_2026-07.md item 7):

- BayesMC month-loss: months with zero evidence are dropped by
  aggregate_spd_v2_bayesmc, and _build_bayesmc_spd_obj converts the
  resulting partial window into insufficient_month_coverage instead of
  writing it.
- Mixed calendar/offset month keys in _build_bayesmc_spd_obj, including
  0-based offset detection.
- Sparse-month binary aggregation: the binary pipeline never writes a
  partial window.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from forecaster import cli
from forecaster.aggregate import aggregate_spd_v2_bayesmc

N_PA = 6  # PA bucket count


def _vec(hot: int = 0) -> list[float]:
    v = [0.0] * N_PA
    v[hot] = 1.0
    return v


# ---------------------------------------------------------------------------
# aggregate_spd_v2_bayesmc month-loss behavior
# ---------------------------------------------------------------------------


def test_bayesmc_drops_months_without_evidence():
    months_a = {f"2026-{m:02d}": _vec() for m in range(4, 8)}  # Apr..Jul
    spd_by_month, diag = aggregate_spd_v2_bayesmc([months_a], n_buckets=N_PA)
    assert sorted(spd_by_month) == sorted(months_a)
    assert diag["status"] == "ok"
    # Months never mentioned by any model are simply absent, not filled.
    assert "2026-08" not in spd_by_month


def test_bayesmc_partial_model_coverage_keeps_union_of_months():
    full = {f"2026-{m:02d}": _vec() for m in range(4, 10)}  # Apr..Sep
    partial = {f"2026-{m:02d}": _vec(1) for m in range(4, 7)}  # Apr..Jun
    spd_by_month, diag = aggregate_spd_v2_bayesmc([full, partial], n_buckets=N_PA)
    assert sorted(spd_by_month) == sorted(full)
    assert diag["months"]["2026-04"]["n_evidence"] == 2
    assert diag["months"]["2026-09"]["n_evidence"] == 1


def test_bayesmc_no_evidence_returns_empty_with_status():
    spd_by_month, diag = aggregate_spd_v2_bayesmc([{}], n_buckets=N_PA)
    assert spd_by_month == {}
    assert diag["status"] == "no_evidence_all_months"


# ---------------------------------------------------------------------------
# _build_bayesmc_spd_obj: month-loss guard + mixed key normalization
# ---------------------------------------------------------------------------

_SPECS = [SimpleNamespace(name="M1", model_id="m-1"), SimpleNamespace(name="M2", model_id="m-2")]


def test_build_bayesmc_partial_window_fails_closed():
    """BayesMC losing months must surface as insufficient_month_coverage,
    never as a written partial window."""
    partial = {f"2026-{m:02d}": _vec() for m in range(4, 8)}  # Apr..Jul only
    spd_obj, diag = cli._build_bayesmc_spd_obj(
        [partial], anchor_month="2026-04", specs_used=_SPECS, n_buckets=N_PA
    )
    assert spd_obj == {}
    assert diag["status"] == "insufficient_month_coverage"
    assert diag["missing_months"] == ["2026-08", "2026-09"]


def test_build_bayesmc_mixed_calendar_and_offset_keys():
    """month_N and calendar keys in the same result must normalize to one
    complete calendar window (1-based offsets: month_1 = anchor)."""
    mixed = {
        "month_1": _vec(),
        "month_2": _vec(),
        "month_3": _vec(),
        "2026-07": _vec(),
        "2026-08": _vec(),
        "2026-09": _vec(),
    }
    spd_obj, diag = cli._build_bayesmc_spd_obj(
        [mixed], anchor_month="2026-04", specs_used=_SPECS, n_buckets=N_PA
    )
    assert diag["status"] == "ok"
    assert sorted(spd_obj["spds"].keys()) == [f"2026-{m:02d}" for m in range(4, 10)]


def test_build_bayesmc_zero_based_offsets_detected():
    """A month_0 key flips detection to 0-based: month_0 = anchor month."""
    zero_based = {f"month_{i}": _vec() for i in range(0, 6)}
    spd_obj, diag = cli._build_bayesmc_spd_obj(
        [zero_based], anchor_month="2026-04", specs_used=_SPECS, n_buckets=N_PA
    )
    assert diag["status"] == "ok"
    assert sorted(spd_obj["spds"].keys()) == [f"2026-{m:02d}" for m in range(4, 10)]


def test_build_bayesmc_offset_keys_without_anchor_fail_closed():
    offsets = {f"month_{i}": _vec() for i in range(1, 7)}
    spd_obj, diag = cli._build_bayesmc_spd_obj(
        [offsets], anchor_month=None, specs_used=_SPECS, n_buckets=N_PA
    )
    assert spd_obj == {}
    assert diag["status"] == "missing_anchor_month"


# ---------------------------------------------------------------------------
# Sparse-month binary aggregation: the pipeline never writes a partial window
# ---------------------------------------------------------------------------

_QROW = {
    "question_id": "TST_FL_EVENT_OCCURRENCE_2026-04",
    "iso3": "TST",
    "hazard_code": "FL",
    "metric": "EVENT_OCCURRENCE",
    "wording": "test question",
    "window_start_date": "2026-04-01",
    "target_month": "2026-09",
    "hs_run_id": "hs_test",
}

_WINDOW = [f"2026-{m:02d}" for m in range(4, 10)]


def _binary_json(months: list[str]) -> str:
    return json.dumps({"months": {m: {"posterior": 0.2} for m in months}})


def _run_binary(raw_texts: list[str]):
    """Drive _run_binary_forecast_for_question with canned model outputs;
    return (writes, no_forecasts) captured from the writer seams."""
    writes: list[dict] = []
    no_forecasts: list[str] = []

    async def fake_members(prompt, specs, **kwargs):
        raw_calls = [{"text": t, "usage": {}, "error": None} for t in raw_texts]
        return [], {}, raw_calls, {}

    async def fake_log(**kwargs):
        return None

    def fake_write(run_id, question_row, month_probs, *, resolution_source, usage, model_name="ensemble"):
        writes.append({"months": dict(month_probs), "model_name": model_name})

    def fake_no_forecast(run_id, qid, iso3, hz, metric, reason, model_name=None):
        no_forecasts.append(reason)

    with patch.object(cli, "_call_spd_members_v2_compat", fake_members), \
         patch.object(cli, "log_forecaster_llm_call", fake_log), \
         patch.object(cli, "_write_binary_outputs", fake_write), \
         patch.object(cli, "_record_no_forecast", fake_no_forecast), \
         patch.object(cli, "_load_structured_data", lambda *a, **k: {}), \
         patch.object(cli, "load_hs_triage_entry", lambda *a, **k: {}), \
         patch.object(cli, "build_binary_base_rate", lambda *a, **k: {}), \
         patch.object(cli, "build_binary_event_prompt", lambda **k: "PROMPT"), \
         patch.object(cli, "connect", side_effect=RuntimeError("no db in test")), \
         patch.object(cli, "_select_spd_specs_for_run", lambda: (_SPECS, [])):
        asyncio.run(cli._run_binary_forecast_for_question("run_test", _QROW, track=1))
    return writes, no_forecasts


def test_binary_complete_models_write_full_window_dropping_off_window():
    writes, no_forecasts = _run_binary(
        [
            _binary_json(_WINDOW + ["2026-12"]),  # hallucinated off-window month
            _binary_json(_WINDOW),
        ]
    )
    assert no_forecasts == []
    # Track 1 with two models writes BOTH aggregations. The bayesmc write
    # regressed silently before (MemberOutput was constructed with a
    # nonexistent 'raw' kwarg, so the block always raised) — pin it.
    assert sorted(w["model_name"] for w in writes) == [
        "ensemble_bayesmc_v2",
        "ensemble_mean_v2",
    ]
    for w in writes:
        assert sorted(w["months"]) == _WINDOW  # off-window month never written


def test_binary_incomplete_model_excluded_but_complete_model_written():
    writes, no_forecasts = _run_binary(
        [
            _binary_json(_WINDOW[:4]),  # parser rejects this model
            _binary_json(_WINDOW),
        ]
    )
    assert no_forecasts == []
    assert writes
    for w in writes:
        assert sorted(w["months"]) == _WINDOW


def test_binary_all_models_incomplete_writes_nothing():
    writes, no_forecasts = _run_binary(
        [_binary_json(_WINDOW[:4]), _binary_json(_WINDOW[1:])]
    )
    assert writes == []
    assert any("no valid responses parsed" in r for r in no_forecasts)


def test_binary_sparse_aggregation_gate_fails_closed():
    """Belt-and-braces: if a sparse month set ever reaches aggregation
    (e.g. a future parser regression), the writer gate must refuse it."""
    sparse = {m: 0.2 for m in _WINDOW[:4]}
    with patch.object(cli, "parse_binary_response", lambda *a, **k: dict(sparse)):
        writes, no_forecasts = _run_binary([_binary_json(_WINDOW)])
    assert writes == []
    assert any("insufficient_month_coverage" in r for r in no_forecasts)
