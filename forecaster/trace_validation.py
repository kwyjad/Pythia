# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Validation of structured reasoning traces from SPD ensemble members.

Produces diagnostic quality scores that are logged alongside forecasts.
Never blocks or modifies forecasts — purely diagnostic.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

LOG = logging.getLogger(__name__)


def validate_reasoning_traces(
    raw_calls: list[dict],
    base_rate_summary: dict,
    hazard_code: str,
    metric: str,
) -> list[dict]:
    """Validate reasoning traces from ensemble members.

    Returns a list of validation result dicts, one per model, with:
      - model_name: str
      - has_trace: bool
      - prior_quality: dict with checks on whether the prior matches base rate
      - delta_arithmetic: dict with checks on whether deltas sum correctly
      - magnitude_consistency: dict with checks on signal magnitude claims
      - trace_quality_score: float 0-1 (1 = perfect trace)
    """
    results: list[dict] = []
    for rc in raw_calls:
        try:
            result = _validate_single_trace(rc, base_rate_summary, hazard_code, metric)
        except Exception:  # noqa: BLE001
            ms = rc.get("model_spec")
            model_name = getattr(ms, "name", str(ms)) if ms else "unknown"
            result = {
                "model_name": model_name,
                "has_trace": False,
                "prior_quality": {"score": 0.0},
                "delta_arithmetic": {"score": 0.0},
                "magnitude_consistency": {"score": 0.0},
                "trace_quality_score": 0.0,
            }
        results.append(result)
    return results


def _validate_single_trace(
    raw_call: dict,
    base_rate_summary: dict,
    hazard_code: str,
    metric: str,
) -> dict:
    ms = raw_call.get("model_spec")
    model_name = getattr(ms, "name", str(ms)) if ms else "unknown"

    trace = raw_call.get("reasoning_trace")
    if not isinstance(trace, dict):
        return {
            "model_name": model_name,
            "has_trace": False,
            "prior_quality": {"score": 0.0},
            "delta_arithmetic": {"score": 0.0},
            "magnitude_consistency": {"score": 0.0},
            "trace_quality_score": 0.0,
        }

    prior_result = _check_prior_consistency(trace, base_rate_summary, hazard_code, metric)
    delta_result = _check_delta_arithmetic(trace)
    magnitude_result = _check_magnitude_consistency(trace)

    prior_score = prior_result.get("score", 0.0)
    delta_score = delta_result.get("score", 0.0)
    magnitude_score = magnitude_result.get("score", 0.0)

    composite = 0.4 * prior_score + 0.4 * delta_score + 0.2 * magnitude_score

    return {
        "model_name": model_name,
        "has_trace": True,
        "prior_quality": prior_result,
        "delta_arithmetic": delta_result,
        "magnitude_consistency": magnitude_result,
        "trace_quality_score": round(composite, 4),
    }


def _implied_modal_bucket(base_rate_summary: dict, hazard_code: str, metric: str) -> Optional[int]:
    """Determine the implied modal bucket index (0-based) from base rate data."""
    if not base_rate_summary:
        return None

    summary_type = base_rate_summary.get("type", "")

    value: Optional[float] = None

    if summary_type == "conflict_trajectory":
        fatalities = base_rate_summary.get("fatalities", {})
        value = fatalities.get("trailing_3m_avg")
    elif summary_type == "seasonal_profile":
        monthly = base_rate_summary.get("monthly_values", {})
        if monthly:
            vals = [v for v in monthly.values() if isinstance(v, (int, float)) and v is not None]
            if vals:
                value = sum(vals) / len(vals)
    elif summary_type == "fewsnet_phase3":
        value = base_rate_summary.get("recent_mean")
    else:
        # Fallback: look for common keys
        for key in ("trailing_3m_avg", "mean", "recent_mean", "median"):
            v = base_rate_summary.get(key)
            if isinstance(v, (int, float)):
                value = v
                break

    if value is None:
        return None

    # Map value to bucket using metric-specific thresholds.
    if metric.upper() == "FATALITIES":
        thresholds = [5, 25, 100, 500]
    elif metric.upper() == "PHASE3PLUS_IN_NEED":
        thresholds = [100_000, 1_000_000, 5_000_000, 15_000_000]
    else:  # PA and default
        thresholds = [10_000, 50_000, 250_000, 500_000]

    for i, t in enumerate(thresholds):
        if value < t:
            return i
    return len(thresholds)


def _check_prior_consistency(
    trace: dict,
    base_rate_summary: dict,
    hazard_code: str,
    metric: str,
) -> dict:
    """Check whether the model's stated prior matches the base rate data."""
    prior = trace.get("prior")
    if not isinstance(prior, dict):
        return {"score": 0.0, "detail": "no prior in trace"}

    prior_spd = prior.get("spd")
    if not isinstance(prior_spd, list) or len(prior_spd) != 5:
        return {"score": 0.0, "detail": "prior.spd missing or wrong length"}

    # Determine model's modal bucket
    try:
        model_mode = max(range(len(prior_spd)), key=lambda i: prior_spd[i])
    except Exception:
        return {"score": 0.0, "detail": "could not determine prior mode"}

    implied_mode = _implied_modal_bucket(base_rate_summary, hazard_code, metric)
    if implied_mode is None:
        # Cannot validate without base rate — give benefit of doubt
        return {"score": 0.7, "detail": "no base rate to compare", "model_mode": model_mode}

    distance = abs(model_mode - implied_mode)
    if distance == 0:
        score = 1.0
    elif distance == 1:
        score = 0.7
    else:
        score = 0.3

    return {
        "score": score,
        "model_mode": model_mode,
        "implied_mode": implied_mode,
        "distance": distance,
    }


def _check_delta_arithmetic(trace: dict) -> dict:
    """Check that update deltas sum to ~0 and post_update_spd = prev + delta."""
    updates = trace.get("updates")
    if not isinstance(updates, list) or len(updates) == 0:
        # No updates to check — consider it valid if prior exists
        prior = trace.get("prior")
        if isinstance(prior, dict) and isinstance(prior.get("spd"), list):
            return {"score": 1.0, "detail": "no updates to check", "n_updates": 0}
        return {"score": 0.0, "detail": "no updates and no prior"}

    n_pass = 0
    n_total = 0
    details: list[dict] = []

    prev_spd = None
    prior = trace.get("prior")
    if isinstance(prior, dict):
        prev_spd = prior.get("spd")

    for update in updates:
        if not isinstance(update, dict):
            continue
        n_total += 1

        delta = update.get("delta")
        post_spd = update.get("post_update_spd")

        ok = True
        detail: Dict[str, Any] = {"signal": update.get("signal", "?")}

        # Check delta sums to ~0
        if isinstance(delta, list) and len(delta) == 5:
            delta_sum = sum(delta)
            if abs(delta_sum) >= 0.05:
                ok = False
                detail["delta_sum"] = round(delta_sum, 4)
        else:
            ok = False
            detail["issue"] = "delta missing or wrong length"

        # Check post_update_spd ≈ prev + delta
        if (
            ok
            and isinstance(prev_spd, list)
            and len(prev_spd) == 5
            and isinstance(post_spd, list)
            and len(post_spd) == 5
            and isinstance(delta, list)
            and len(delta) == 5
        ):
            l1 = sum(abs(post_spd[i] - (prev_spd[i] + delta[i])) for i in range(5))
            if l1 >= 0.1:
                ok = False
                detail["l1_norm"] = round(l1, 4)

        if ok:
            n_pass += 1
        details.append(detail)

        # Update prev_spd for chain checking
        if isinstance(post_spd, list) and len(post_spd) == 5:
            prev_spd = post_spd

    score = n_pass / max(n_total, 1)
    return {"score": round(score, 4), "n_updates": n_total, "n_pass": n_pass, "details": details}


def _check_magnitude_consistency(trace: dict) -> dict:
    """Check that claimed magnitude is consistent with actual delta values."""
    updates = trace.get("updates")
    if not isinstance(updates, list) or len(updates) == 0:
        prior = trace.get("prior")
        if isinstance(prior, dict) and isinstance(prior.get("spd"), list):
            return {"score": 1.0, "detail": "no updates to check", "n_updates": 0}
        return {"score": 0.0, "detail": "no updates and no prior"}

    n_pass = 0
    n_total = 0

    for update in updates:
        if not isinstance(update, dict):
            continue

        magnitude = (update.get("magnitude") or "").upper()
        delta = update.get("delta")

        if not isinstance(delta, list) or len(delta) != 5 or not magnitude:
            continue

        n_total += 1
        max_abs = max(abs(d) for d in delta)

        consistent = False
        if magnitude == "SMALL":
            consistent = max_abs < 0.10
        elif magnitude == "MODERATE":
            consistent = 0.03 <= max_abs <= 0.20
        elif magnitude == "LARGE":
            consistent = max_abs > 0.10
        else:
            # Unknown magnitude — give benefit of doubt
            consistent = True

        if consistent:
            n_pass += 1

    score = n_pass / max(n_total, 1)
    return {"score": round(score, 4), "n_updates": n_total, "n_pass": n_pass}
