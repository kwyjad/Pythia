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
from typing import Dict, List, Optional, Sequence

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


# ---------------------------------------------------------------------------
# Two-horizon beliefs with an explicit zero (Oct 2026)
# ---------------------------------------------------------------------------
#
# A trial states, for month 1 and month 6 of the window, ``p_zero`` (the
# chance the resolving source records zero; for flood and cyclone, zero or no
# record) and the 0.05..0.95 quantiles of the value GIVEN that it is
# positive. Each month becomes a distribution: mass p_zero at zero, the rest
# along a monotone curve through the positive quantiles in log space that
# starts at half a unit and reaches 1.0 at five times the 0.95 quantile.
# Bucket edges are evaluated half a unit low (4.5, 24.5, 99.5, ...), so a
# quantile of exactly 100 cannot flip a bucket. Trials are linearly pooled
# per month; months 2-5 are linear mixtures of months 1 and 6.

#: Positive quantile levels each trial reports per month.
POS_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
#: Where the positive curve reaches 1.0, as a multiple of the 0.95 quantile.
POS_TOP_FACTOR = 5.0
#: Left anchor of the positive curve: positive values are counts >= 1.
POS_LEFT = 0.5
#: Quantile levels stored for the raw pooled series (the legacy seven plus 0.05).
STORED_LEVELS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


def _pos_support(qpos: Dict[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """(log value, conditional CDF) support points for the positive curve."""
    pairs = sorted(
        (max(1.0, float(qpos[lv])), float(lv)) for lv in POS_LEVELS if lv in qpos
    )
    xs: List[float] = [float(np.log(POS_LEFT))]
    ys: List[float] = [0.0]
    for value, level in pairs:
        xl = float(np.log(value))
        if abs(xl - xs[-1]) < 1e-12:
            ys[-1] = max(ys[-1], level)
        else:
            xs.append(xl)
            ys.append(max(level, ys[-1]))
    top = max(1.0, float(qpos.get(0.95, pairs[-1][0] if pairs else 1.0))) * POS_TOP_FACTOR
    xs.append(float(np.log(top)))
    ys.append(1.0)
    return np.asarray(xs), np.asarray(ys)


@dataclass
class MonthDist:
    """One month's belief: P(zero) plus the positive quantiles."""

    p_zero: float
    qpos: Dict[float, float]

    def cdf(self, values: Sequence[float]) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        xs, ys = _pos_support(self.qpos)
        g = np.zeros_like(v)
        pos = v >= POS_LEFT
        if pos.any():
            g[pos] = np.clip(_pchip_eval(xs, ys, np.log(v[pos])), 0.0, 1.0)
        p0 = min(max(float(self.p_zero), 0.0), 1.0)
        out = np.where(v < 0, 0.0, p0 + (1.0 - p0) * g)
        return np.clip(out, 0.0, 1.0)

    def top(self) -> float:
        return max(1.0, float(self.qpos.get(0.95, 1.0))) * POS_TOP_FACTOR


def _edges(metric: str) -> List[float]:
    from pythia.buckets import interior_thresholds_for  # noqa: PLC0415

    return [float(t) - 0.5 for t in interior_thresholds_for(metric)]


def vector_from_cdf(cdf_at, metric: str) -> List[float]:
    """Bucket masses from a CDF callable, edges evaluated half a unit low."""
    edges = _edges(metric)
    if not edges:
        raise ValueError(f"no bucket scheme for metric {metric!r}")
    f = np.asarray(cdf_at(edges), dtype=float)
    f = np.maximum.accumulate(np.clip(f, 0.0, 1.0))
    full = np.concatenate([[0.0], f, [1.0]])
    probs = np.clip(np.diff(full), 0.0, None)
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("degenerate CDF produced a zero-sum bucket vector")
    return [float(p / total) for p in probs]


def month_vector(dist: MonthDist, metric: str) -> List[float]:
    return vector_from_cdf(dist.cdf, metric)


def _grid(dists: Sequence[MonthDist]) -> np.ndarray:
    top = max((d.top() for d in dists), default=10.0) * 1.05
    return np.concatenate([[0.0], np.exp(np.linspace(np.log(POS_LEFT), np.log(top), 1024))])


def quantiles_from_cdf_fn(cdf_at, grid: np.ndarray, levels: Sequence[float]) -> Dict[float, float]:
    """Invert a CDF callable on *grid* at *levels* (mass at zero respected)."""
    f = np.maximum.accumulate(np.clip(np.asarray(cdf_at(grid), dtype=float), 0.0, 1.0))
    return _quantiles_from_cdf(grid, f, levels)


def mixture_weight(month: int) -> float:
    """Weight on month 6 for month ``month`` (1 -> 0.0, 6 -> 1.0)."""
    return (min(max(int(month), 1), 6) - 1) / 5.0


@dataclass
class MonthlyPool:
    """The pooled trials before any reference: vectors and quantiles by month."""

    vectors: Dict[int, List[float]]
    quantiles: Dict[int, Dict[float, float]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "vectors": {str(m): v for m, v in sorted(self.vectors.items())},
            "quantiles": {
                str(m): {str(k): float(v) for k, v in sorted(q.items())}
                for m, q in sorted(self.quantiles.items())
            },
        }


def pool_months(
    trials: Sequence[Dict[int, MonthDist]], metric: str, months: Sequence[int] = (1, 2, 3, 4, 5, 6)
) -> MonthlyPool:
    """Linear pool per month; months 2-5 mix months 1 and 6."""
    usable = [t for t in trials if t and 1 in t and 6 in t]
    if not usable:
        raise ValueError("pool_months requires at least one trial with months 1 and 6")
    grid = _grid([d for t in usable for d in (t[1], t[6])])
    vectors: Dict[int, List[float]] = {}
    quantiles: Dict[int, Dict[float, float]] = {}
    for m in months:
        w = mixture_weight(m)

        def cdf_at(x, _w=w):
            return np.mean(
                [(1.0 - _w) * t[1].cdf(x) + _w * t[6].cdf(x) for t in usable], axis=0
            )

        vectors[int(m)] = vector_from_cdf(cdf_at, metric)
        quantiles[int(m)] = quantiles_from_cdf_fn(cdf_at, grid, STORED_LEVELS)
    return MonthlyPool(vectors=vectors, quantiles=quantiles)


def publish_vectors(
    raw: Dict[int, List[float]],
    reference: Optional[Dict[int, List[float]]],
    weight: float,
) -> Dict[int, List[float]]:
    """Per month: ``weight`` x reference + the rest x raw; raw alone without one."""
    out: Dict[int, List[float]] = {}
    for m, vec in raw.items():
        ref = (reference or {}).get(m)
        if ref and len(ref) == len(vec):
            mixed = [weight * r + (1.0 - weight) * x for r, x in zip(ref, vec)]
        else:
            mixed = list(vec)
        z = sum(mixed)
        out[m] = [x / z for x in mixed]
    return out


def dist_from_vector(vec: Sequence[float], metric: str) -> MonthDist:
    """Read P(zero) and positive quantiles off a bucket vector.

    Used to seed a trial's belief from the reference. Within a bucket the
    value is interpolated in log space between its edges; the open top
    bucket is taken to run to four times its lower edge.
    """
    from pythia.buckets import interior_thresholds_for  # noqa: PLC0415

    p = [max(0.0, float(x)) for x in vec]
    total = sum(p) or 1.0
    p = [x / total for x in p]
    p0 = p[0]
    rest = p[1:]
    rz = sum(rest)
    edges = [1.0] + [float(t) for t in interior_thresholds_for(metric)[1:]]
    if rz <= 1e-12:
        return MonthDist(p_zero=min(p0, 0.999), qpos={lv: 1.0 for lv in POS_LEVELS})
    cond = [x / rz for x in rest]
    lows = edges
    highs = edges[1:] + [edges[-1] * 4.0]
    cum = 0.0
    q: Dict[float, float] = {}
    for lv in POS_LEVELS:
        cum = 0.0
        for c, lo, hi in zip(cond, lows, highs):
            if cum + c >= lv - 1e-12 and c > 0:
                frac = (lv - cum) / c
                val = float(np.exp(np.log(lo) + frac * (np.log(hi) - np.log(lo))))
                q[lv] = max(1.0, val)
                break
            cum += c
        else:
            q[lv] = highs[-1]
    return MonthDist(p_zero=p0, qpos=q)
