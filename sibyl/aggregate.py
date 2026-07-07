# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl trial aggregation.

Each trial yields quantiles at ``QUANTILE_LEVELS`` (a discretized CDF).
Aggregation:

1. Each trial's quantile set becomes a full CDF via monotone (PCHIP)
   interpolation over (value, level) pairs — interpolated in
   ``log1p(value)`` space for numerical stability given the multi-order-
   of-magnitude range of affected/fatalities counts, then mapped back.
2. ``linear_pool`` (default): mean of the K CDFs on a shared value grid
   (a mixture). This widens the aggregate when trials disagree — the
   desired behaviour on the highest-volatility questions.
3. ``vincent`` (config alternative): per-level quantile averaging.

The shared grid extends well beyond the largest trial quantile
(``TAIL_EXTENSION_FACTOR``) so the heavy right tail (p95/p99) is not
truncated. PCHIP is implemented in numpy (Fritsch-Carlson/Butland slopes)
because scipy is not a repo dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from sibyl.config import QUANTILE_LEVELS

# How far past the largest trial quantile the pooled grid extends.
TAIL_EXTENSION_FACTOR = 3.0
GRID_POINTS = 513


def _monotone_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Butland slopes: monotone cubic Hermite for monotone data."""
    n = len(x)
    h = np.diff(x)
    d = np.diff(y) / h
    m = np.zeros(n)
    m[0] = d[0]
    m[-1] = d[-1]
    for k in range(1, n - 1):
        if d[k - 1] * d[k] <= 0:
            m[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k])
    return m


def _pchip_eval(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Evaluate the monotone cubic Hermite interpolant at *xq*.

    Values outside [x[0], x[-1]] clamp to the endpoint ordinates (the CDF
    anchors at exactly 0 and 1 are part of the support points).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xq = np.asarray(xq, dtype=float)
    if len(x) == 1:
        return np.where(xq < x[0], 0.0, y[0])

    m = _monotone_slopes(x, y)
    idx = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, len(x) - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    h = x1 - x0
    t = np.clip((xq - x0) / h, 0.0, 1.0)

    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    out = h00 * y[idx] + h10 * h * m[idx] + h01 * y[idx + 1] + h11 * h * m[idx + 1]

    out = np.where(xq <= x[0], y[0], out)
    out = np.where(xq >= x[-1], y[-1], out)
    return out


def _support_points(quantiles: Dict[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """(log1p(value), level) support points with 0/1 anchors, ties collapsed.

    Ties (e.g. several leading quantiles at exactly 0) collapse to one point
    carrying the highest level — that IS the probability mass at that value.
    """
    pairs = sorted((float(v), float(lv)) for lv, v in quantiles.items())
    collapsed: List[tuple[float, float]] = []
    for value, level in pairs:
        xlog = float(np.log1p(max(0.0, value)))
        if collapsed and abs(collapsed[-1][0] - xlog) < 1e-12:
            collapsed[-1] = (xlog, max(collapsed[-1][1], level))
        else:
            collapsed.append((xlog, level))

    xs = [p[0] for p in collapsed]
    ys = [p[1] for p in collapsed]

    # Left anchor: counts are supported on [0, inf). If the lowest quantile
    # sits above 0, the CDF reaches 0 at value 0.
    if xs[0] > 0.0:
        xs.insert(0, 0.0)
        ys.insert(0, 0.0)
    # Right anchor: the remaining upper-tail mass is spread out to
    # TAIL_EXTENSION_FACTOR x the largest quantile instead of truncating.
    top_value = float(np.expm1(xs[-1]))
    if ys[-1] < 1.0:
        extended = max(top_value * TAIL_EXTENSION_FACTOR, top_value + 1.0)
        xs.append(float(np.log1p(extended)))
        ys.append(1.0)
    else:
        ys[-1] = 1.0
    return np.asarray(xs), np.asarray(ys)


def cdf_from_quantiles(quantiles: Dict[float, float], values: np.ndarray) -> np.ndarray:
    """Evaluate one trial's PCHIP CDF at *values* (native units)."""
    xs, ys = _support_points(quantiles)
    xq = np.log1p(np.clip(np.asarray(values, dtype=float), 0.0, None))
    return np.clip(_pchip_eval(xs, ys, xq), 0.0, 1.0)


def make_grid(trials: Sequence[Dict[float, float]], n: int = GRID_POINTS) -> np.ndarray:
    """Shared value grid: log-spaced from 0 past the largest trial quantile."""
    top = max((max(q.values()) for q in trials if q), default=0.0)
    top = max(top * TAIL_EXTENSION_FACTOR, 10.0)
    log_top = np.log1p(top)
    return np.expm1(np.linspace(0.0, log_top, n))


@dataclass
class PooledDistribution:
    """The aggregated distribution: a CDF on a grid plus pooled quantiles."""

    grid: np.ndarray  # value space, ascending
    cdf: np.ndarray  # pooled CDF on grid, in [0, 1], non-decreasing
    quantiles: Dict[float, float]
    method: str

    def cdf_at(self, values: Sequence[float]) -> np.ndarray:
        """Pooled CDF evaluated at arbitrary values (linear on the grid)."""
        vals = np.clip(np.asarray(values, dtype=float), 0.0, None)
        return np.clip(np.interp(vals, self.grid, self.cdf, left=0.0, right=1.0), 0.0, 1.0)

    def to_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "quantiles": {str(k): float(v) for k, v in sorted(self.quantiles.items())},
        }


def _quantiles_from_cdf(
    grid: np.ndarray, cdf: np.ndarray, levels: Sequence[float]
) -> Dict[float, float]:
    """Invert a (grid, cdf) pair at the requested levels."""
    cdf_mono = np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))
    out: Dict[float, float] = {}
    for lv in levels:
        idx = int(np.searchsorted(cdf_mono, lv, side="left"))
        if idx <= 0:
            out[lv] = float(grid[0])
        elif idx >= len(grid):
            out[lv] = float(grid[-1])
        else:
            f0, f1 = cdf_mono[idx - 1], cdf_mono[idx]
            if f1 - f0 < 1e-12:
                out[lv] = float(grid[idx])
            else:
                w = (lv - f0) / (f1 - f0)
                out[lv] = float(grid[idx - 1] + w * (grid[idx] - grid[idx - 1]))
    return out


def linear_pool(trials: Sequence[Dict[float, float]]) -> PooledDistribution:
    """Mean of the K trial CDFs on a shared grid (a mixture).

    Pooling CDFs (not quantiles) widens the aggregate when trials disagree.
    """
    usable = [q for q in trials if q]
    if not usable:
        raise ValueError("linear_pool requires at least one trial quantile set")
    grid = make_grid(usable)
    stacked = np.vstack([cdf_from_quantiles(q, grid) for q in usable])
    pooled = np.maximum.accumulate(np.clip(stacked.mean(axis=0), 0.0, 1.0))
    quantiles = _quantiles_from_cdf(grid, pooled, QUANTILE_LEVELS)
    return PooledDistribution(grid=grid, cdf=pooled, quantiles=quantiles, method="linear_pool")


def vincent_average(trials: Sequence[Dict[float, float]]) -> PooledDistribution:
    """Vincent (per-level quantile averaging) alternative.

    The averaged quantile set is then expanded into a CDF with the same
    PCHIP machinery so downstream bucketization is method-agnostic.
    """
    usable = [q for q in trials if q]
    if not usable:
        raise ValueError("vincent_average requires at least one trial quantile set")
    avg = {
        lv: float(np.mean([float(q[lv]) for q in usable]))
        for lv in QUANTILE_LEVELS
    }
    grid = make_grid([avg])
    cdf = np.maximum.accumulate(cdf_from_quantiles(avg, grid))
    return PooledDistribution(grid=grid, cdf=cdf, quantiles=avg, method="vincent")


def aggregate_trials(
    trials: Sequence[Dict[float, float]], method: str = "linear_pool"
) -> PooledDistribution:
    """Aggregate K trial quantile sets using the configured method."""
    if method == "vincent":
        return vincent_average(trials)
    if method == "linear_pool":
        return linear_pool(trials)
    raise ValueError(f"unknown aggregation method {method!r}")
