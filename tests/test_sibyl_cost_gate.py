# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The offline cost gate (2026-10-14, sibyl/cost_projection.py).

One recorded eight-step trial of the 13 October 2026 run goes through the
real prompt builder and transcript with a stubbed model; every step is priced
under Anthropic's block-boundary cache rule and pythia/model_costs.json. The
build fails when the projected cost per question exceeds the stated figure in
sibyl/config.py by more than 50%.
"""

from __future__ import annotations

import pytest

import sibyl.config as sibyl_config
from sibyl.cost_projection import load_fixture, project_question, project_trial, simulate_cache


def test_projected_cost_per_question_is_within_the_stated_figure():
    cost, trials = project_question(load_fixture())
    stated = sibyl_config.STATED_COST_PER_QUESTION_USD
    assert len(trials) == sibyl_config.K
    assert cost <= 1.5 * stated, (
        f"projected ${cost:.3f} a question, stated ${stated:.2f}: the agent or the prompt "
        "got dearer; re-measure with python -m sibyl.cost_projection"
    )


def test_every_step_after_the_first_reads_the_earlier_steps_from_cache():
    trial = project_trial(load_fixture())
    for prev, cur in zip(trial.steps, trial.steps[1:]):
        # Everything the previous step marked is read back, never re-written.
        assert cur.cache_read >= prev.cache_read + prev.cache_write - 5
        assert cur.cache_write < 3000  # one new entry, not the whole transcript


def test_the_old_shape_would_cost_several_times_more():
    fixture = load_fixture()
    new = project_trial(fixture).opus_usd
    old = project_trial(fixture, legacy=True).opus_usd
    assert old > 2 * new
    # The old shape re-wrote the growing transcript every step.
    steps = project_trial(fixture, legacy=True).steps
    assert steps[-1].cache_write > 10 * steps[1].cache_write / 2


def test_the_cache_rule_reads_only_at_a_block_boundary_within_the_lookback():
    m = {"cache_control": {"type": "ephemeral"}}
    a = [{"text": "x" * 400}, {"text": "y" * 400, **m}]
    b = [{"text": "x" * 400}, {"text": "y" * 400}, {"text": "z" * 400, **m}]
    use = simulate_cache([a, b])
    assert use[0]["cache_read_input_tokens"] == 0
    assert use[1]["cache_read_input_tokens"] == 200 and use[1]["cache_creation_input_tokens"] == 100
    # One growing block never matches the previous prefix.
    c = [{"text": "x" * 400 + "y" * 400 + "z" * 400, **m}]
    assert simulate_cache([a, c])[1]["cache_read_input_tokens"] == 0
    # Beyond 20 blocks back the previous entry is not found.
    far = [{"text": "x" * 400}, {"text": "y" * 400}] + [{"text": "w"}] * 21 + [{"text": "z", **m}]
    assert simulate_cache([a, far])[1]["cache_read_input_tokens"] == 0


@pytest.mark.parametrize("k", [1, 2])
def test_a_question_costs_k_trials(k):
    cost, trials = project_question(load_fixture(), k=k)
    assert len(trials) == k and cost == pytest.approx(sum(t.total_usd for t in trials))
