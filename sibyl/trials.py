# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""How a question's trials are run and which of them are pooled (Oct 2026).

* ``run_trial_batch`` runs a set of trials on worker threads. Only the main
  thread writes DuckDB: every trial buffers its ``llm_calls`` rows and the
  batch writes them, in trial order, once all of its trials have finished.
* ``extra_trials_rule`` decides whether the production trials call for
  extra ones: they disagree (largest pairwise month-1 JSD above
  ``SIBYL_EXTRA_TRIALS_JSD``) or their pool departs far from the reference
  (month-1 JSD above ``SIBYL_EXTRA_TRIALS_DEPARTURE_JSD``).
* ``outlier_indices`` names the trials the outlier guard leaves out: a
  month-1 median more than ``SIBYL_OUTLIER_LOG10`` orders of magnitude from
  the median of the other trials' medians, while two trials remain.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sibyl import config as _cfg

logger = logging.getLogger(__name__)

RULE_DISAGREEMENT = "disagreement"
RULE_DEPARTURE = "departure"


# --- Pure helpers ---------------------------------------------------------------

def month_median(p_zero: float, qpos: Dict[float, float]) -> float:
    """The median of a month's belief: zero, or the matching positive quantile.

    With P(zero) = p, the median is 0 when p >= 0.5; otherwise it is the
    positive curve's quantile at (0.5 - p) / (1 - p), interpolated in log
    space between the stated levels and clamped to the outer ones.
    """
    p = min(max(float(p_zero or 0.0), 0.0), 1.0)
    if p >= 0.5 or not qpos:
        return 0.0
    level = (0.5 - p) / (1.0 - p)
    pts = sorted((float(lv), max(1.0, float(v))) for lv, v in qpos.items())
    levels = [lv for lv, _ in pts]
    logs = [math.log(v) for _, v in pts]
    return float(math.exp(np.interp(level, levels, logs)))


def _js(p: Sequence[float], q: Sequence[float]) -> float:
    from sibyl.spd import _js_divergence  # noqa: PLC0415

    return _js_divergence(p, q)


def max_pairwise_jsd(vectors: Sequence[Sequence[float]]) -> Optional[float]:
    """The largest month-1 JSD between any two trials (None below two)."""
    vecs = [list(v) for v in vectors if v]
    if len(vecs) < 2:
        return None
    return max(
        _js(vecs[i], vecs[j]) for i in range(len(vecs)) for j in range(i + 1, len(vecs))
    )


def extra_trials_rule(
    trial_vectors: Sequence[Sequence[float]],
    pooled_vector: Optional[Sequence[float]],
    reference_vector: Optional[Sequence[float]],
    *,
    jsd_limit: Optional[float] = None,
    departure_limit: Optional[float] = None,
) -> Tuple[Optional[str], Dict[str, Optional[float]]]:
    """Which rule, if any, calls for extra trials, and the measures behind it.

    Disagreement is tested first; departure from the reference second. The
    measures are returned whatever the verdict, so the record can say how
    close a question came.
    """
    jsd_limit = _cfg.EXTRA_TRIALS_JSD if jsd_limit is None else jsd_limit
    departure_limit = (
        _cfg.EXTRA_TRIALS_DEPARTURE_JSD if departure_limit is None else departure_limit
    )
    spread = max_pairwise_jsd(trial_vectors)
    departure = (
        _js(pooled_vector, reference_vector)
        if pooled_vector and reference_vector
        else None
    )
    measures = {"max_pairwise_jsd": spread, "departure_jsd": departure}
    if spread is not None and spread > jsd_limit:
        return RULE_DISAGREEMENT, measures
    if departure is not None and departure > departure_limit:
        return RULE_DEPARTURE, measures
    return None, measures


def outlier_indices(medians: Sequence[float], *, limit: Optional[float] = None) -> List[int]:
    """Positions of the trials the outlier guard leaves out.

    A trial is an outlier when log10(1 + its month-1 median) is more than
    *limit* from log10(1 + the median of the OTHER trials' medians). The
    farthest outlier is removed first and the test re-run on the rest; it
    stops while two trials remain, so the guard never leaves fewer than two.
    """
    limit = _cfg.OUTLIER_LOG10 if limit is None else limit
    alive = list(range(len(medians)))
    dropped: List[int] = []
    while len(alive) > 2:
        worst, worst_gap = None, 0.0
        for i in alive:
            others = [medians[j] for j in alive if j != i]
            centre = float(np.median(others))
            gap = abs(math.log10(1.0 + max(medians[i], 0.0)) - math.log10(1.0 + max(centre, 0.0)))
            if gap > limit and gap > worst_gap:
                worst, worst_gap = i, gap
        if worst is None:
            break
        alive.remove(worst)
        dropped.append(worst)
    return sorted(dropped)


# --- Running trials ---------------------------------------------------------------

def run_trial_batch(
    jobs: Sequence[Tuple[int, str]],
    run_one: Callable[[int, str, List[Dict[str, Any]]], Any],
    *,
    workers: Optional[int] = None,
    write_log: Optional[Callable[..., None]] = None,
) -> List[Any]:
    """Run ``run_one(trial_index, lane, log_sink)`` for every job.

    Jobs run on up to *workers* threads (``SIBYL_TRIAL_WORKERS``). Each job
    gets its own log sink; after all have finished, the buffered rows are
    written with *write_log* on the calling (main) thread, in job order.
    Results come back in job order. A job that raises is logged and
    returns None, so one trial cannot sink the others.
    """
    if not jobs:
        return []
    workers = max(1, int(_cfg.TRIAL_WORKERS if workers is None else workers))
    sinks: List[List[Dict[str, Any]]] = [[] for _ in jobs]

    def _one(pos: int) -> Any:
        idx, lane = jobs[pos]
        try:
            return run_one(idx, lane, sinks[pos])
        except Exception as exc:  # noqa: BLE001 - one trial must not sink the batch
            logger.exception("sibyl.trials: trial %d (lane %s) raised: %s", idx, lane, exc)
            return None

    if workers == 1 or len(jobs) == 1:
        results = [_one(pos) for pos in range(len(jobs))]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            results = list(pool.map(_one, range(len(jobs))))

    if write_log is not None:
        for sink in sinks:
            for row in sink:
                try:
                    write_log(**row)
                except Exception as exc:  # noqa: BLE001 - ledger failures never stop a run
                    logger.warning("sibyl.trials: llm_calls write failed: %s", exc)
    return results
