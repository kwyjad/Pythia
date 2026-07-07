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

A Sibyl trial emits ONE quantile set describing the monthly value across
the 6-month window (the window months are treated exchangeably), so the
same bucket vector is written for each month_index 1..6.

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
from sibyl.config import SIBYL_MODEL_NAME

logger = logging.getLogger(__name__)

# Standard-track aggregate preference for the head-to-head comparison.
STANDARD_MODEL_PREFERENCE = ("ensemble_bayesmc_v2", "ensemble_mean_v2", "track2_flash")


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
    sibyl_probs: Sequence[float], standard_by_month: Optional[Dict[int, List[float]]]
) -> Optional[float]:
    """Mean JS divergence between Sibyl's SPD and the standard track.

    Sibyl's vector is identical across the window months, so this is the
    mean of per-month JSD(sibyl, standard_m).
    """
    if not standard_by_month:
        return None
    vals = [
        _js_divergence(sibyl_probs, vec) for vec in standard_by_month.values()
    ]
    return float(np.mean(vals)) if vals else None


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
    bucket_probs: Sequence[float],
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
    if len(bucket_probs) != n_buckets:
        raise ValueError(
            f"bucket vector length {len(bucket_probs)} != {n_buckets} for {metric}"
        )
    labels = labels_for(metric)
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
        for bucket_idx, prob in enumerate(bucket_probs, start=1):
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
            brave_cost_usd, leakage_json, created_at, is_test
        ) VALUES (?, ?, ?, ?, ?, ?, 'sibyl', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
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
        ],
    )


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
            created_at, is_test
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  CURRENT_TIMESTAMP, ?)
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
        ],
    )
