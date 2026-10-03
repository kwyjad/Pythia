# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl SPD assembly and output.

Serializes the pooled (identity-)calibrated distribution into standard
Pythia's native SPD format: per-(month_index, bucket_index) probability
rows in ``forecasts_raw`` + ``forecasts_ensemble`` under
``model_name='sibyl'`` (``weights_profile='sibyl'`` is the track marker in
forecasts_ensemble). Rows are written under the SAME forecaster ``run_id``
as the standard track for the question, so:

* ``compute_scores`` scores Sibyl head-to-head automatically (it scores
  every DISTINCT model_name in forecasts_raw), and
* the question-detail SPD panel offers ``sibyl`` as a selectable source
  next to the ensemble aggregates.

Since Oct 2026 a trial states month 1 and month 6 of the window (an explicit
P(zero) plus positive quantiles each); months 2-5 are mixtures, and each
month's published vector is a pool with Sibyl's reference
(``sibyl/reference.py``), so the six month_index rows differ. Before then
one vector was written to all six months.

Full trial-level provenance — per-trial final quantiles, belief-state
traces and evidence lists, asOf, K, aggregation method, per-question cost,
divergences, leakage stats — is persisted to the dedicated
``sibyl_forecasts`` table (everything the deferred PIT calibration and the
dashboard need).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pythia.buckets import NUM_HORIZONS, n_buckets_for, thresholds_for
from pythia.test_mode import is_test_mode

from sibyl.aggregate import PooledDistribution, cdf_from_quantiles
from sibyl.config import BUCKET_FLOOR, SIBYL_MODEL_NAME, STANDARD_MODEL_PREFERENCE

logger = logging.getLogger(__name__)


def _probs_from_cdf_values(cdf_at_thresholds: np.ndarray) -> List[float]:
    """Bucket masses from CDF values at the bucket thresholds.

    thresholds_for(metric) is ``[0, t1, .., t_{K-1}, inf]``; bucket k covers
    ``[t_{k-1}, t_k)``. The support floor is 0, so ``F(t_0)=F(0)`` is
    replaced by 0 and the top bucket takes ``1 - F(t_{K-1})``.
    """
    f = np.clip(np.asarray(cdf_at_thresholds, dtype=float), 0.0, 1.0)
    f = np.maximum.accumulate(f)
    f[0] = 0.0
    f[-1] = 1.0
    probs = np.diff(f)
    probs = np.clip(probs, 0.0, None)
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("degenerate CDF produced a zero-sum bucket vector")
    return [float(p / total) for p in probs]


def apply_bucket_floor(probs: Sequence[float], floor: float = BUCKET_FLOOR) -> List[float]:
    """Floor every bucket at *floor* and renormalise, so none ends below it.

    A single "max then divide" pass can leave a floored bucket slightly under
    the floor; this fixes the floored buckets at exactly *floor* and scales
    the rest into what is left, repeating until nothing falls below.
    """
    p = np.clip(np.asarray(probs, dtype=float), 0.0, None)
    n = len(p)
    if n == 0:
        return []
    total = float(p.sum())
    if total <= 0:
        return [1.0 / n] * n
    p = p / total
    if floor <= 0:
        return [float(x) for x in p]
    if floor * n >= 1.0:
        return [1.0 / n] * n
    fixed = np.zeros(n, dtype=bool)
    for _ in range(n + 1):
        low = (~fixed) & (p < floor)
        if not low.any():
            break
        fixed |= low
        rest = float(p[~fixed].sum())
        p[fixed] = floor
        if rest > 0:
            p[~fixed] = p[~fixed] * (1.0 - floor * fixed.sum()) / rest
    return [float(x) for x in p / p.sum()]


def bucket_probs_from_distribution(dist: PooledDistribution, metric: str) -> List[float]:
    """Discretize a pooled distribution onto the metric's SPD buckets."""
    thresholds = thresholds_for(metric)
    if not thresholds:
        raise ValueError(f"no bucket scheme for metric {metric!r}")
    finite = thresholds[:-1]  # [0, t1, ..., t_{K-1}]
    f = dist.cdf_at(finite)
    return _probs_from_cdf_values(np.append(f, 1.0))


def bucket_probs_from_quantiles(quantiles: Dict[float, float], metric: str) -> List[float]:
    """Discretize ONE trial's quantile set onto the metric's buckets.

    Used for the inter-trial disagreement diagnostic; goes through the same
    PCHIP CDF construction as the pooled path.
    """
    thresholds = thresholds_for(metric)
    if not thresholds:
        raise ValueError(f"no bucket scheme for metric {metric!r}")
    finite = np.asarray(thresholds[:-1], dtype=float)
    f = cdf_from_quantiles(quantiles, finite)
    return _probs_from_cdf_values(np.append(f, 1.0))


def _js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence (reuses the calibration-advice helper)."""
    from pythia.tools.generate_calibration_advice import (  # noqa: PLC0415
        _js_divergence as _jsd,
    )

    return float(_jsd(np.asarray(p, dtype=float), np.asarray(q, dtype=float)))


def load_standard_spd_by_month(
    con: Any, run_id: str, question_id: str, n_buckets: int
) -> Optional[Dict[int, List[float]]]:
    """Load the standard-track SPD (preferred aggregate) per month.

    Preference order: ensemble_bayesmc_v2 > ensemble_mean_v2 > track2_flash
    (mirrors the risk-index chosen-model CTE). Returns None when no
    standard aggregate exists for the question.
    """
    for model_name in STANDARD_MODEL_PREFERENCE:
        rows = con.execute(
            """
            SELECT month_index, bucket_index, probability
            FROM forecasts_ensemble
            WHERE run_id = ? AND question_id = ? AND model_name = ?
            """,
            [run_id, question_id, model_name],
        ).fetchall()
        if not rows:
            continue
        by_month: Dict[int, List[float]] = {}
        for month_idx, bucket_idx, prob in rows:
            mi, bi = int(month_idx), int(bucket_idx)
            if not (1 <= mi <= NUM_HORIZONS and 1 <= bi <= n_buckets):
                continue
            by_month.setdefault(mi, [0.0] * n_buckets)[bi - 1] = float(prob or 0.0)
        by_month = {
            mi: vec for mi, vec in by_month.items() if sum(vec) > 0
        }
        if by_month:
            return by_month
    return None


def track_divergence(
    sibyl_probs: Any, standard_by_month: Optional[Dict[int, List[float]]]
) -> Optional[float]:
    """Mean JS divergence between Sibyl's SPD and the standard track.

    Compared month by month (Oct 2026): ``sibyl_probs`` is a {month: vector}
    dict; a single vector (the pre-Oct-2026 shape) is compared with every
    month.
    """
    if not standard_by_month:
        return None
    vals = []
    for m, vec in standard_by_month.items():
        mine = sibyl_probs.get(m) if isinstance(sibyl_probs, dict) else sibyl_probs
        if mine and len(mine) == len(vec):
            vals.append(_js_divergence(mine, vec))
    return float(np.mean(vals)) if vals else None


def inter_trial_divergence_vectors(vectors: Sequence[Sequence[float]]) -> Optional[float]:
    """Mean pairwise JS divergence across trial bucket vectors (month 1)."""
    vecs = [list(v) for v in vectors if v]
    if len(vecs) < 2:
        return None
    pair_vals = [
        _js_divergence(vecs[i], vecs[j])
        for i in range(len(vecs))
        for j in range(i + 1, len(vecs))
    ]
    return float(np.mean(pair_vals))


def inter_trial_divergence(
    trial_quantiles: Sequence[Dict[float, float]], metric: str
) -> Optional[float]:
    """Mean pairwise JS divergence across the K trial distributions."""
    vecs = []
    for q in trial_quantiles:
        if not q:
            continue
        try:
            vecs.append(bucket_probs_from_quantiles(q, metric))
        except ValueError:
            continue
    if len(vecs) < 2:
        return None
    pair_vals = [
        _js_divergence(vecs[i], vecs[j])
        for i in range(len(vecs))
        for j in range(i + 1, len(vecs))
    ]
    return float(np.mean(pair_vals))


def find_standard_run_id(con: Any, question_id: str) -> Optional[str]:
    """The forecaster run_id of the question's latest standard forecast."""
    row = con.execute(
        """
        SELECT run_id
        FROM forecasts_ensemble
        WHERE question_id = ? AND model_name <> ?
        ORDER BY created_at DESC
        LIMIT 1
        """,
        [question_id, SIBYL_MODEL_NAME],
    ).fetchone()
    return str(row[0]) if row and row[0] else None


def write_native_spd(
    con: Any,
    *,
    run_id: str,
    question: Any,  # SibylQuestion
    bucket_probs: Any,  # {month: vector} or one vector for all months
    spd_payload: Dict[str, Any],
    human_explanation: str,
    cost_usd: float,
) -> None:
    """Write the Sibyl SPD in the native format (both forecast tables).

    Mirrors ``forecaster.cli._write_spd_outputs``: DELETE-then-INSERT per
    (run_id, question_id, model_name), one row per (month, bucket).
    """
    from pythia.buckets import labels_for  # noqa: PLC0415

    metric = question.metric
    n_buckets = n_buckets_for(metric)
    # One vector per window month (Oct 2026); a bare vector is written to all six.
    if isinstance(bucket_probs, dict):
        by_month = {int(m): list(v) for m, v in bucket_probs.items()}
    else:
        by_month = {m: list(bucket_probs) for m in range(1, NUM_HORIZONS + 1)}
    missing = [m for m in range(1, NUM_HORIZONS + 1) if m not in by_month]
    if missing:
        raise ValueError(f"no bucket vector for month(s) {missing}")
    for m, vec in by_month.items():
        if len(vec) != n_buckets:
            raise ValueError(
                f"bucket vector length {len(vec)} != {n_buckets} for {metric} (month {m})"
            )
    labels = labels_for(metric)
    # Every written vector carries the floor (idempotent on a floored one).
    by_month = {m: apply_bucket_floor(v) for m, v in by_month.items()}
    is_test = is_test_mode()
    spd_json = json.dumps(spd_payload, default=str)
    trace_json = json.dumps(
        {
            "track": "sibyl",
            "as_of": spd_payload.get("as_of"),
            "k": spd_payload.get("k"),
            "aggregation": spd_payload.get("aggregation"),
            "pooled_quantiles": spd_payload.get("pooled_quantiles"),
            "trial_quantiles": spd_payload.get("trial_quantiles"),
        },
        default=str,
    )

    con.execute(
        "DELETE FROM forecasts_raw WHERE run_id = ? AND question_id = ? AND model_name = ?;",
        [run_id, question.question_id, SIBYL_MODEL_NAME],
    )
    con.execute(
        "DELETE FROM forecasts_ensemble WHERE run_id = ? AND question_id = ? AND model_name = ?;",
        [run_id, question.question_id, SIBYL_MODEL_NAME],
    )

    for month_idx in range(1, NUM_HORIZONS + 1):
        for bucket_idx, prob in enumerate(by_month[month_idx], start=1):
            label = labels[bucket_idx - 1] if bucket_idx - 1 < len(labels) else str(bucket_idx)
            con.execute(
                """
                INSERT INTO forecasts_raw (
                    run_id, question_id, model_name, month_index, bucket_index,
                    probability, ok, elapsed_ms, cost_usd, prompt_tokens,
                    completion_tokens, total_tokens, status, spd_json,
                    human_explanation, horizon_m, class_bin, p, is_test,
                    reasoning_trace_json
                ) VALUES (?, ?, ?, ?, ?, ?, TRUE, NULL, ?, NULL, NULL, NULL,
                          'ok', ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    run_id,
                    question.question_id,
                    SIBYL_MODEL_NAME,
                    month_idx,
                    bucket_idx,
                    float(prob),
                    float(cost_usd),
                    spd_json,
                    human_explanation,
                    month_idx,
                    label,
                    float(prob),
                    is_test,
                    trace_json,
                ],
            )
            con.execute(
                """
                INSERT INTO forecasts_ensemble (
                    run_id, question_id, iso3, hazard_code, metric, model_name,
                    month_index, bucket_index, probability, ev_value,
                    weights_profile, created_at, status, human_explanation,
                    is_test, reasoning_trace_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'sibyl',
                          CURRENT_TIMESTAMP, 'ok', ?, ?, ?)
                """,
                [
                    run_id,
                    question.question_id,
                    question.iso3,
                    question.hazard_code,
                    metric,
                    SIBYL_MODEL_NAME,
                    month_idx,
                    bucket_idx,
                    float(prob),
                    human_explanation,
                    is_test,
                    trace_json,
                ],
            )


def persist_sibyl_forecast(con: Any, record: Dict[str, Any]) -> None:
    """Upsert one question's full Sibyl record into ``sibyl_forecasts``."""
    con.execute(
        "DELETE FROM sibyl_forecasts WHERE sibyl_run_id = ? AND question_id = ?;",
        [record["sibyl_run_id"], record["question_id"]],
    )
    con.execute(
        """
        INSERT INTO sibyl_forecasts (
            sibyl_run_id, run_id, question_id, iso3, hazard_code, metric,
            track, status, skip_reason, as_of, k, aggregation,
            volatility_score, triage_score, pooled_quantiles_json,
            trials_json, bucket_probs_json, js_divergence_vs_standard,
            js_divergence_inter_trial, cost_usd, opus_cost_usd,
            brave_cost_usd, leakage_json, created_at, is_test, selection_pass,
            base_rate_json, advice_arm, advice_as_of_month, evidence_ok,
            reference_json, raw_by_month_json, final_by_month_json
        ) VALUES (?, ?, ?, ?, ?, ?, 'sibyl', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            record["sibyl_run_id"],
            record.get("run_id"),
            record["question_id"],
            record.get("iso3"),
            record.get("hazard_code"),
            record.get("metric"),
            record.get("status", "ok"),
            record.get("skip_reason"),
            record.get("as_of"),
            record.get("k"),
            record.get("aggregation"),
            record.get("volatility_score"),
            record.get("triage_score"),
            json.dumps(record.get("pooled_quantiles"), default=str),
            json.dumps(record.get("trials"), default=str),
            json.dumps(record.get("bucket_probs"), default=str),
            record.get("js_divergence_vs_standard"),
            record.get("js_divergence_inter_trial"),
            record.get("cost_usd", 0.0),
            record.get("opus_cost_usd", 0.0),
            record.get("brave_cost_usd", 0.0),
            json.dumps(record.get("leakage"), default=str),
            is_test_mode(),
            record.get("selection_pass"),
            (
                json.dumps(record["base_rate"], default=str)
                if record.get("base_rate") is not None else None
            ),
            record.get("advice_arm"),
            record.get("advice_as_of_month"),
            record.get("evidence_ok"),
            _json_or_none(record.get("reference")),
            _json_or_none(record.get("raw_by_month")),
            _json_or_none(record.get("final_by_month")),
        ],
    )


def _json_or_none(obj: Any) -> Optional[str]:
    return None if obj is None else json.dumps(obj, default=str)


def persist_sibyl_run(con: Any, record: Dict[str, Any]) -> None:
    """Upsert the run-level record into ``sibyl_runs``."""
    con.execute(
        "DELETE FROM sibyl_runs WHERE sibyl_run_id = ?;",
        [record["sibyl_run_id"]],
    )
    con.execute(
        """
        INSERT INTO sibyl_runs (
            sibyl_run_id, hs_run_id, as_of, model, k, max_steps, aggregation,
            run_hard_cap_usd, budget_capped, run_cost_usd, opus_cost_usd,
            brave_cost_usd, n_selected, n_forecast, n_skipped, config_json,
            created_at, is_test, time_capped, n_search_calls,
            n_search_failed, n_breaker_trips, n_docs_read
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
        """,
        [
            record["sibyl_run_id"],
            record.get("hs_run_id"),
            record.get("as_of"),
            record.get("model"),
            record.get("k"),
            record.get("max_steps"),
            record.get("aggregation"),
            record.get("run_hard_cap_usd"),
            bool(record.get("budget_capped", False)),
            record.get("run_cost_usd", 0.0),
            record.get("opus_cost_usd", 0.0),
            record.get("brave_cost_usd", 0.0),
            record.get("n_selected", 0),
            record.get("n_forecast", 0),
            record.get("n_skipped", 0),
            json.dumps(record.get("config"), default=str),
            is_test_mode(),
            bool(record.get("time_capped", False)),
            record.get("n_search_calls"),
            record.get("n_search_failed"),
            record.get("n_breaker_trips"),
            record.get("n_docs_read"),
        ],
    )
