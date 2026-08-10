# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Whether a forecast is worth a planner's attention, and by how much.

The report used to rank by how far a forecast sat from its base rate. That
number is dominated by whichever anchor happens to sit closest to zero: a
cyclone question whose history puts 99% of its weight on "nobody affected"
divides by almost nothing, so a modest tail produces a huge multiple. The
August report led on Indonesian cyclones and put famine in Sudan on page two.

So ranking moved to **expected excess**: how many more people the forecast
expects than history would suggest, in people. The ratio survives in the prose
as a descriptor, never as the ordering.

Selection is a two-key gate. A forecast must be **unusual** (its divergence
sits in the top slice of this run) and **material** (enough probability above a
threshold worth mobilising against). Unusual-but-small goes to a watchlist
rather than into the report or the bin: a country office wants to know a rare
storm is being tracked, and does not want it on page two ahead of Sudan.

Pure functions over plain numbers. No DB, no IO, no config reads — every
threshold arrives as an argument so the caller owns the policy and the tests
can state it.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

from pythia.buckets import centroids_for, thresholds_for

# The gate a question passed, in the words the report prints.
GATE_WORSENING = "larger than usual"
GATE_MAJOR = "heavy burden"
GATE_WATCHLIST = "unusual, small scale"
GATE_DISAGREEMENT = "scan and forecast disagree"


def exceedance(probs: Sequence[float], metric: str, threshold: float) -> float:
    """P(value >= threshold) for one month's bucket vector.

    Buckets are ranges, so a threshold usually falls inside one. The part of
    that bucket above the threshold is taken by linear interpolation on the
    bucket's own bounds: assuming the mass sits uniformly across the range is
    cruder than the centroid but it is the only assumption the bucket
    definition actually supports, and it never claims more precision than the
    scheme has. The top bucket is unbounded, so a threshold inside it counts
    whole.
    """
    if not probs:
        return 0.0
    edges = thresholds_for(metric)
    if not edges or len(edges) < len(probs) + 1:
        return 0.0
    total = 0.0
    for i, p in enumerate(probs):
        if not p:
            continue
        lo, hi = float(edges[i]), float(edges[i + 1])
        if lo >= threshold:
            total += p
        elif math.isinf(hi) or hi <= threshold:
            # Wholly below the threshold, or unbounded and starting below it.
            if math.isinf(hi) and threshold > lo:
                total += p  # unbounded top bucket: cannot rule the value out
        elif hi > threshold > lo:
            span = hi - lo
            total += p * ((hi - threshold) / span) if span > 0 else p
    return max(0.0, min(total, 1.0))


def quantile(probs: Sequence[float], metric: str, q: float) -> float | None:
    """The q-quantile of one month's bucket vector, interpolated in-bucket.

    Returns the value a planner should plan against (q=0.5) or hold
    contingency for (q=0.9). None when there is nothing to read.
    """
    if not probs or not 0.0 < q < 1.0:
        return None
    edges = thresholds_for(metric)
    if not edges or len(edges) < len(probs) + 1:
        return None
    total = sum(probs)
    if total <= 0:
        return None
    cumulative = 0.0
    for i, p in enumerate(probs):
        share = p / total
        if cumulative + share >= q:
            lo, hi = float(edges[i]), float(edges[i + 1])
            if math.isinf(hi):
                # No upper bound to interpolate against; the centroid is the
                # only stated value for the band.
                cents = centroids_for(metric)
                return float(cents[i]) if i < len(cents) else lo
            if share <= 0:
                return lo
            within = (q - cumulative) / share
            return lo + within * (hi - lo)
        cumulative += share
    return float(edges[len(probs) - 1])


def p_zero(probs: Sequence[float]) -> float:
    """The chance of nothing recorded: bucket 1, which every SPD metric leads
    with by design (`pythia/buckets.py`)."""
    return float(probs[0]) if probs else 0.0


def expected_value(probs: Sequence[float], metric: str) -> float:
    """Plain expected value over the bucket centroids."""
    if not probs:
        return 0.0
    cents = centroids_for(metric)
    return sum(p * c for p, c in zip(probs, cents))


def materiality_threshold(
    metric: str,
    population: float | None,
    *,
    absolute: float | None,
    population_share: float | None,
) -> float | None:
    """The count worth mobilising against for this country and metric.

    ``min(absolute, share × population)`` when both are configured. The share
    exists so small states are not silently excluded: a flat 50,000 floor
    means Vanuatu never appears in a cyclone report, which would be absurd.
    Returns None when the metric has no threshold (binary questions gate on
    probability alone).
    """
    if absolute is None:
        return None
    candidates = [float(absolute)]
    if population_share and population:
        candidates.append(float(population_share) * float(population))
    return min(candidates)


def horizon_exceedances(
    monthly_probs: Sequence[Sequence[float]],
    metric: str,
    threshold: float | None,
) -> list[float]:
    """P(>= threshold) per horizon, in window order.

    A binary question has no threshold; its P(event) is already the
    exceedance, so the first bucket (P(yes)) is returned per month.
    """
    if threshold is None:
        return [float(m[0]) if m else 0.0 for m in monthly_probs]
    return [exceedance(m, metric, threshold) for m in monthly_probs]


def peak_horizon(values: Sequence[float]) -> int | None:
    """1-based index of the largest value, the horizon a planner works to.

    Ties take the EARLIEST month: a decision deadline derived from this should
    fall on the side of being early.
    """
    if not values:
        return None
    best, best_i = None, None
    for i, v in enumerate(values, start=1):
        if best is None or v > best:
            best, best_i = v, i
    return best_i


def clears_materiality(
    exceedances: Sequence[float], min_probability: float
) -> tuple[bool, int | None]:
    """Does ANY horizon carry enough probability above the threshold?

    Returns (cleared, 1-based horizon). The horizon is carried forward: the
    decision calendar anchors its deadline to it, so which month cleared is
    part of the answer, not a detail.
    """
    if not exceedances:
        return False, None
    best_i = peak_horizon(exceedances)
    if best_i is None:
        return False, None
    return exceedances[best_i - 1] >= min_probability, best_i


def percentile(values: Iterable[float], q: float) -> float | None:
    """The q-percentile (0-1) of a run's values, linear interpolation.

    Written out rather than delegated so the definition is pinned by this
    repo's tests, the same reason the resolution machine writes its own.
    """
    pool = sorted(float(v) for v in values if v is not None)
    if not pool:
        return None
    if len(pool) == 1:
        return pool[0]
    pos = q * (len(pool) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(pool) - 1)
    frac = pos - lo
    return pool[lo] + frac * (pool[hi] - pool[lo])


def gate_rows(
    rows: list[dict[str, Any]],
    *,
    unusual_percentile: float,
    min_probability: float,
    thin_min_obs: int,
) -> dict[str, int]:
    """Stamp the gate result on every row. Returns the counts the report prints.

    Each row needs: ``js_vs_baserate``, ``exceedances`` (per horizon),
    ``excess_nominal``, ``baserate_n_obs``. Each row gains ``passed_unusual``,
    ``passed_material``, ``peak_horizon``, ``gate``, ``baserate_thin``.

    The counts are returned rather than recomputed by the caller, because the
    selection panel, the entry tags and the appendix table must agree and the
    only way to guarantee that is for all three to read one number.
    """
    js_values = [
        r.get("js_vs_baserate") for r in rows
        if r.get("js_vs_baserate") is not None
    ]
    cut = percentile(js_values, unusual_percentile)

    counts = {"considered": len(rows), "unusual": 0, "material": 0,
              "both": 0, "major": 0, "watchlist": 0, "thin": 0}
    for row in rows:
        js = row.get("js_vs_baserate")
        unusual = js is not None and cut is not None and float(js) >= cut
        cleared, horizon = clears_materiality(
            row.get("exceedances") or [], min_probability
        )
        n_obs = row.get("baserate_n_obs")
        thin = n_obs is not None and int(n_obs) < thin_min_obs

        # Direction still decides what "worsening" means. Unusual and
        # material describe SIZE and CONFIDENCE, neither of which has a sign,
        # so without this a forecast expecting two million FEWER people than
        # usual lands in "potentially worsening situations". Uganda did
        # exactly that on the August run: excess -2,069,829, both keys passed.
        excess = row.get("excess_nominal")
        rising = excess is not None and float(excess) > 0

        row["passed_unusual"] = unusual
        row["passed_material"] = cleared
        row["peak_horizon"] = horizon
        row["baserate_thin"] = thin
        if unusual and cleared and rising:
            row["gate"] = GATE_WORSENING
        elif cleared:
            # Material but not rising, or material but unremarkable: a heavy
            # burden the report should carry, without the claim that it is
            # getting worse.
            row["gate"] = GATE_MAJOR
        elif unusual:
            row["gate"] = GATE_WATCHLIST
        else:
            row["gate"] = None

        counts["unusual"] += int(unusual)
        counts["material"] += int(cleared)
        counts["both"] += int(row["gate"] == GATE_WORSENING)
        counts["watchlist"] += int(row["gate"] == GATE_WATCHLIST)
        counts["major"] = counts.get("major", 0) + int(row["gate"] == GATE_MAJOR)
        counts["thin"] += int(thin)
    counts["unusual_cut"] = cut
    return counts


def rank_key(row: dict[str, Any], *, per_capita: bool = False) -> tuple:
    """Sort key: clear anchors before thin ones, then by excess, descending.

    A thin anchor is demoted rather than dropped. Indonesia's near-empty
    cyclone record may be a gap in GDACS and IFRC rather than a real absence
    of impact, and the honest response to not knowing is to rank it lower and
    say so, not to delete it or to lead with it.
    """
    field = "excess_per_100k" if per_capita else "excess_nominal"
    value = row.get(field)
    return (
        1 if row.get("baserate_thin") else 0,
        -(float(value) if value is not None else float("-inf")),
        str(row.get("question_id") or ""),
    )


def calibration_table(runs: dict[str, dict[str, int]]) -> str:
    """What the gate admits and drops, per run, as a printable table.

    Task 1.6 of the v3 brief: run the gate over every run in the DB before
    shipping and print what it does. If a threshold empties a section, that
    is a finding to look at, not a silently blank page.
    """
    header = (
        f"{'run':<20}{'seen':>6}{'unusual':>9}{'material':>10}"
        f"{'worsening':>11}{'burden':>8}{'watch':>7}{'thin':>6}{'cut':>8}"
    )
    lines = [header, "-" * len(header)]
    for run_id, c in runs.items():
        cut = c.get("unusual_cut")
        lines.append(
            f"{str(run_id)[:19]:<20}{c.get('considered', 0):>6}"
            f"{c.get('unusual', 0):>9}{c.get('material', 0):>10}"
            f"{c.get('both', 0):>11}{c.get('major', 0):>8}"
            f"{c.get('watchlist', 0):>7}{c.get('thin', 0):>6}"
            f"{(f'{cut:.3f}' if cut is not None else '-'):>8}"
        )
    return "\n".join(lines)
