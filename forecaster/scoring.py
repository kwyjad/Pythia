"""Scoring utilities for SPD evaluations."""
from __future__ import annotations

import math
from typing import Iterable, List


PA_THRESHOLDS = [10000, 50000, 250000, 500000]
FATALITIES_THRESHOLDS = [5, 25, 100, 500]


def normalize_probs(probs: Iterable[float]) -> list[float]:
    prob_list: List[float] = [float(p) for p in probs]
    total = sum(prob_list)
    if total <= 0:
        if not prob_list:
            return []
        uniform = 1.0 / len(prob_list)
        return [uniform for _ in prob_list]
    return [p / total for p in prob_list]


def bucket_index_from_value(metric: str, value: float) -> int:
    metric_upper = metric.upper()
    thresholds = PA_THRESHOLDS if metric_upper == "PA" else FATALITIES_THRESHOLDS
    for idx, threshold in enumerate(thresholds, start=1):
        if value < threshold:
            return idx
    return len(thresholds) + 1


def multiclass_brier(probs: Iterable[float], true_idx_1based: int) -> float:
    probs_norm = normalize_probs(probs)
    if not probs_norm:
        return 0.0
    true_index = true_idx_1based - 1
    score = 0.0
    for idx, prob in enumerate(probs_norm):
        outcome = 1.0 if idx == true_index else 0.0
        score += (prob - outcome) ** 2
    return score


def log_score(probs: Iterable[float], true_idx_1based: int, eps: float = 1e-12) -> float:
    probs_norm = normalize_probs(probs)
    if not probs_norm:
        return 0.0
    true_index = true_idx_1based - 1
    p_true = probs_norm[true_index] if 0 <= true_index < len(probs_norm) else 0.0
    p_clipped = max(p_true, eps)
    return -math.log(p_clipped)
