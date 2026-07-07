# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl aggregation: PCHIP monotonicity, linear pooling, log-space
round-trip, tail non-truncation, and the vincent alternative."""

from __future__ import annotations

import numpy as np
import pytest

from sibyl.aggregate import (
    GRID_POINTS,
    TAIL_EXTENSION_FACTOR,
    aggregate_trials,
    cdf_from_quantiles,
    linear_pool,
    make_grid,
    vincent_average,
)
from sibyl.config import QUANTILE_LEVELS

T_NARROW = {0.1: 5.0, 0.25: 20.0, 0.5: 100.0, 0.75: 300.0, 0.9: 900.0, 0.95: 2000.0, 0.99: 8000.0}
T_WIDE = {0.1: 0.0, 0.25: 50.0, 0.5: 500.0, 0.75: 2500.0, 0.9: 9000.0, 0.95: 25000.0, 0.99: 90000.0}
T_ZEROS = {0.1: 0.0, 0.25: 0.0, 0.5: 0.0, 0.75: 10.0, 0.9: 100.0, 0.95: 400.0, 0.99: 1500.0}


def test_pchip_cdf_is_monotone_and_bounded():
    for trial in (T_NARROW, T_WIDE, T_ZEROS):
        grid = make_grid([trial])
        cdf = cdf_from_quantiles(trial, grid)
        assert np.all(cdf >= 0.0) and np.all(cdf <= 1.0)
        assert np.all(np.diff(cdf) >= -1e-12), "CDF must be non-decreasing"


def test_log_space_round_trip_hits_quantile_knots():
    """F(q_level) == level at each quantile knot, through log1p and back."""
    values = np.array([T_NARROW[lv] for lv in QUANTILE_LEVELS])
    cdf = cdf_from_quantiles(T_NARROW, values)
    for level, f in zip(QUANTILE_LEVELS, cdf):
        assert f == pytest.approx(level, abs=1e-9)


def test_leading_zero_quantiles_become_mass_at_zero():
    """Ties at 0 collapse into P(X=0): F(0) equals the highest tied level."""
    cdf_at_zero = cdf_from_quantiles(T_ZEROS, np.array([0.0]))[0]
    assert cdf_at_zero == pytest.approx(0.5, abs=1e-9)


def test_linear_pool_is_mean_of_trial_cdfs():
    pooled = linear_pool([T_NARROW, T_WIDE])
    f1 = cdf_from_quantiles(T_NARROW, pooled.grid)
    f2 = cdf_from_quantiles(T_WIDE, pooled.grid)
    np.testing.assert_allclose(pooled.cdf, (f1 + f2) / 2.0, atol=1e-9)


def test_linear_pool_of_identical_trials_is_identity():
    pooled = linear_pool([T_NARROW, dict(T_NARROW)])
    solo = cdf_from_quantiles(T_NARROW, pooled.grid)
    np.testing.assert_allclose(pooled.cdf, solo, atol=1e-9)


def test_linear_pool_widens_on_disagreement():
    """The behaviour the method is chosen for, on a representative
    disagreeing pair: the mixture keeps the fat right tail that per-level
    quantile averaging (vincent) shrinks, and every pooled quantile is
    bracketed by the component quantiles at its level (mixture CDF lies
    between the component CDFs pointwise)."""
    pooled = linear_pool([T_NARROW, T_WIDE])
    vincent = vincent_average([T_NARROW, T_WIDE])

    # Fat tail preserved: beyond the vincent p99 the mixture still holds
    # more than 1% of its mass (vincent averages the tail away).
    assert pooled.quantiles[0.99] > vincent.quantiles[0.99]
    assert pooled.cdf_at([vincent.quantiles[0.99]])[0] < 0.99

    for lv in QUANTILE_LEVELS:
        lo = min(T_NARROW[lv], T_WIDE[lv])
        hi = max(T_NARROW[lv], T_WIDE[lv])
        assert lo - 1e-6 <= pooled.quantiles[lv] <= hi + 1e-6


def test_tail_is_not_truncated():
    """The grid extends beyond the largest trial quantile and the upper-tail
    mass past p99 survives pooling."""
    pooled = linear_pool([T_NARROW, T_WIDE])
    max_q = max(max(T_NARROW.values()), max(T_WIDE.values()))
    assert pooled.grid[-1] >= max_q * TAIL_EXTENSION_FACTOR * 0.99
    # 1% of mass lies beyond the largest p99 by construction.
    f_at_max = pooled.cdf_at([max_q])[0]
    assert f_at_max < 1.0
    assert pooled.cdf[-1] == pytest.approx(1.0, abs=1e-9)
    assert len(pooled.grid) == GRID_POINTS


def test_vincent_averages_per_level():
    pooled = vincent_average([T_NARROW, T_WIDE])
    for lv in QUANTILE_LEVELS:
        assert pooled.quantiles[lv] == pytest.approx((T_NARROW[lv] + T_WIDE[lv]) / 2.0)
    assert pooled.method == "vincent"


def test_aggregate_trials_dispatch_and_validation():
    assert aggregate_trials([T_NARROW], "linear_pool").method == "linear_pool"
    assert aggregate_trials([T_NARROW], "vincent").method == "vincent"
    with pytest.raises(ValueError):
        aggregate_trials([T_NARROW], "geometric")
    with pytest.raises(ValueError):
        aggregate_trials([], "linear_pool")


def test_pooled_quantiles_are_monotone():
    pooled = linear_pool([T_NARROW, T_WIDE, T_ZEROS])
    ordered = [pooled.quantiles[lv] for lv in QUANTILE_LEVELS]
    assert ordered == sorted(ordered)
