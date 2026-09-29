# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Shadow ensemble members and calibration carry-over within a model family.

A shadow member is called and written to forecasts_raw like any member, so it
is scored and builds a track record, but it never moves ensemble_mean_v2 or
ensemble_bayesmc_v2. A new version of a model family (claude-opus-5 ->
claude-opus-5-5) inherits its predecessor's calibration record as a prior
that fades as its own record grows, and a retired version leaves the weight
softmax.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from forecaster import cli
from forecaster.aggregate import aggregate_spd_v2_bayesmc
from forecaster.providers import ModelSpec, is_shadow
from pythia.tools import compute_calibration_pythia as cal

MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
LOW = [0.7, 0.2, 0.05, 0.03, 0.01, 0.01]
HIGH = [0.01, 0.01, 0.03, 0.05, 0.2, 0.7]


def _spec(model_id: str, *, shadow: bool = False, provider: str = "openai") -> ModelSpec:
    return ModelSpec(name=model_id, provider=provider, model_id=model_id,
                     active=True, purpose="spd_v2", shadow=shadow)


def _answer(vec):
    return json.dumps({"spds": {m: {"probs": vec} for m in MONTHS}})


# ---------------------------------------------------------------------------
# Shadow members
# ---------------------------------------------------------------------------


def test_config_entry_carries_the_shadow_flag(monkeypatch):
    from forecaster import providers

    monkeypatch.setattr(
        "pythia.llm_profiles.get_ensemble_resolved",
        lambda: [
            {"provider": "openai", "model_id": "gpt-6-sol", "thinking": "high"},
            {"provider": "anthropic", "model_id": "claude-candidate", "shadow": True},
        ],
    )
    specs = providers._load_ensemble_from_config()
    by_id = {s.model_id: s for s in specs}
    assert is_shadow(by_id["claude-candidate"])
    assert not is_shadow(by_id["gpt-6-sol"])


def test_voting_members_drops_the_shadow_and_keeps_alignment():
    specs = [_spec("a"), _spec("s", shadow=True), _spec("b")]
    spds = [{"x": [1]}, {"x": [2]}, {"x": [3]}]
    voting_spds, voting_specs = cli._voting_members(spds, specs)
    assert [s.model_id for s in voting_specs] == ["a", "b"]
    assert voting_spds == [{"x": [1]}, {"x": [3]}]


def test_a_zero_weight_would_not_exclude_a_member_from_bayesmc():
    """Why the shadow member is removed from the lists, not given weight 0."""
    spds = [{m: LOW for m in MONTHS}, {m: HIGH for m in MONTHS}]
    with_zero, _ = aggregate_spd_v2_bayesmc(
        spds, n_buckets=6, weights_by_model={"a": 1.0, "s": 0.0}, model_names=["a", "s"]
    )
    alone, _ = aggregate_spd_v2_bayesmc(
        spds[:1], n_buckets=6, weights_by_model={"a": 1.0}, model_names=["a"]
    )
    assert with_zero["2026-01"] != pytest.approx(alone["2026-01"])


def test_shadow_member_is_called_but_never_votes(monkeypatch):
    answers = {"a": _answer(LOW), "b": _answer(LOW), "s": _answer(HIGH)}

    async def fake(ms, prompt, **_kw):
        return answers[ms.model_id], {"total_tokens": 1}, None, ms

    monkeypatch.setattr(cli, "_call_spd_model_for_spec", fake)
    monkeypatch.setattr(cli, "_calibration_weights_enabled", lambda: False)
    specs = [_spec("a"), _spec("b", provider="google"), _spec("s", shadow=True, provider="anthropic")]

    per_model, _usage, raw_calls, meta = asyncio.run(
        cli._call_spd_members_v2("prompt", specs, metric="PA", anchor_month="2026-01")
    )
    # The shadow member was called and parsed: its forecast is written and scored.
    assert len(per_model) == 3 and len(raw_calls) == 3
    assert meta["n_shadow_members"] == 1

    kw = dict(run_id="r", question_id="q", hs_run_id=None, metric="PA",
              hazard_code="FL", anchor_month="2026-01")
    spd_obj, *_ = asyncio.run(cli._call_spd_bayesmc_v2("prompt", specs=specs, **kw))
    spd_voting, *_ = asyncio.run(cli._call_spd_bayesmc_v2("prompt", specs=specs[:2], **kw))
    got = spd_obj["spds"]["2026-01"]["probs"]
    want = spd_voting["spds"]["2026-01"]["probs"]
    assert got == pytest.approx(want)


def test_a_failed_shadow_member_is_not_a_partial_ensemble(monkeypatch):
    async def fake(ms, prompt, **_kw):
        if ms.model_id == "s":
            return "", {}, "boom", ms
        return _answer(LOW), {"total_tokens": 1}, None, ms

    monkeypatch.setattr(cli, "_call_spd_model_for_spec", fake)
    specs = [_spec("a"), _spec("b", provider="google"), _spec("s", shadow=True, provider="anthropic")]
    _p, _u, _r, meta = asyncio.run(
        cli._call_spd_members_v2("prompt", specs, metric="PA", anchor_month="2026-01")
    )
    assert meta["partial_ensemble"] is False

    async def fake_voter_fails(ms, prompt, **_kw):
        if ms.model_id == "b":
            return "", {}, "boom", ms
        return _answer(LOW), {"total_tokens": 1}, None, ms

    monkeypatch.setattr(cli, "_call_spd_model_for_spec", fake_voter_fails)
    _p, _u, _r, meta = asyncio.run(
        cli._call_spd_members_v2("prompt", specs, metric="PA", anchor_month="2026-01")
    )
    assert meta["partial_ensemble"] is True


# ---------------------------------------------------------------------------
# Calibration carry-over within a family
# ---------------------------------------------------------------------------


def _samples(model: str, brier: float, n: int, start: int = 0):
    return [
        cal.Sample(
            question_key=("ETH", "ACE", "FATALITIES", f"q{start + i}"),
            hazard_code="ACE", metric="FATALITIES", model_name=model,
            score_type="brier", value=brier, observed_month="2026-08",
        )
        for i in range(n)
    ]


def test_new_version_inherits_the_retired_versions_record():
    # claude-opus-5 was good (0.2) and is retired; claude-opus-5-5 has no
    # record yet; gpt-6-sol has a middling one (0.5).
    samples = _samples("claude-opus-5", 0.2, 25) + _samples("gpt-6-sol", 0.5, 25)
    rows, advice = cal._compute_weights_for_group(
        "2026-09", samples, current_members=["claude-opus-5-5", "gpt-6-sol"]
    )
    by = {r["model_name"]: r for r in rows}
    assert "claude-opus-5" not in by, "a retired version must leave the softmax"
    assert by["claude-opus-5-5"]["inherited_from"] == "claude-opus-5"
    assert by["claude-opus-5-5"]["avg_brier"] == pytest.approx(0.2)
    assert by["claude-opus-5-5"]["weight"] > by["gpt-6-sol"]["weight"]
    assert "claude-opus-5-5 <- claude-opus-5" in advice


def test_the_inherited_prior_fades_as_the_new_version_scores():
    samples = (
        _samples("claude-opus-5", 0.2, 25)
        + _samples("claude-opus-5-5", 0.8, 30, start=100)
        + _samples("gpt-6-sol", 0.5, 25)
    )
    # As of the samples' own month, so no time decay enters the arithmetic.
    rows, _ = cal._compute_weights_for_group(
        "2026-08", samples, current_members=["claude-opus-5-5", "gpt-6-sol"]
    )
    by = {r["model_name"]: r for r in rows}
    k = cal.FAMILY_PRIOR_QUESTIONS
    expected = (30 * 0.8 + k * 0.2) / (30 + k)
    assert by["claude-opus-5-5"]["avg_brier"] == pytest.approx(expected)
    # 30 own questions outweigh a 10-question prior.
    assert by["claude-opus-5-5"]["avg_brier"] > 0.6


def test_a_model_outside_any_family_keeps_its_place():
    samples = _samples("ad-hoc-override", 0.3, 25) + _samples("gpt-6-sol", 0.5, 25)
    rows, _ = cal._compute_weights_for_group(
        "2026-09", samples, current_members=["gpt-6-sol"]
    )
    assert {r["model_name"] for r in rows} == {"ad-hoc-override", "gpt-6-sol"}


def test_lookup_carries_the_predecessors_weight_and_treats_unknowns_as_neutral(monkeypatch):
    monkeypatch.setattr(cli, "_calibration_weights_enabled", lambda: True)
    monkeypatch.setattr(
        cli, "_load_calibration_weights_db",
        lambda hz, mt: {"claude-opus-5": 0.6, "gpt-6-sol": 0.2},
    )
    cli._CALIB_WEIGHTS_CACHE.clear()
    specs = [_spec("claude-opus-5-5", provider="anthropic"), _spec("gpt-6-sol"),
             _spec("brand-new-family", provider="google")]
    _by_key, _keys, weights = cli._resolve_member_weights(specs, "ACE", "FATALITIES")
    cli._CALIB_WEIGHTS_CACHE.clear()
    # Raw (0.6 inherited, 0.2, neutral = mean 0.4) rescaled to mean 1.0.
    assert weights == pytest.approx([1.5, 0.5, 1.0])
