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
import math
from typing import Any, Dict, List, Optional

from pythia.buckets import interior_thresholds_for, n_buckets_for

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

    expected_k = n_buckets_for(metric) or 5
    prior_result = _check_prior_consistency(trace, base_rate_summary, hazard_code, metric)
    delta_result = _check_delta_arithmetic(trace, expected_k)
    magnitude_result = _check_magnitude_consistency(trace, expected_k)

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
        # Reported beside the score and deliberately NOT in the composite, so
        # trace_quality_score keeps meaning what it meant before.
        "rc_sharpness": check_rc_sharpness(trace, expected_k),
        "trace_quality_score": round(composite, 4),
    }


# --- Regime-change shift: parsing and the sharpness check --------------------

RC_SHIFT_DIRECTIONS = ("up", "down", "none", "two_sided")
_RC_SHIFT_DIRECTION_ALIASES = {
    "up": "up", "higher": "up", "increase": "up",
    "down": "down", "lower": "down", "decrease": "down",
    "none": "none", "no": "none", "no_shift": "none", "rebutted": "none",
    "two_sided": "two_sided", "two-sided": "two_sided", "two sided": "two_sided",
    "mixed": "two_sided", "both": "two_sided", "unclear": "two_sided",
}
# A posterior that keeps less than (1 - this) of the prior's modal-bucket mass...
RC_SHARPNESS_MAX_MODAL_LOSS = 0.25
# ...while its expected bucket index moved by less than this, has spread, not shifted.
RC_SHARPNESS_MIN_SHIFT = 0.25


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None  # drop NaN


def _as_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "yes", "1"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("false", "no", "0"):
        return False
    return None


def parse_rc_shift(raw: Any) -> tuple:
    """Normalise a member's ``reasoning_trace.rc_shift``.

    Returns ``(value, status)``: ``(None, "absent")`` when the member wrote none,
    ``(dict, "ok")`` when every field parsed, and ``(dict_or_None,
    "malformed:<fields>")`` otherwise, keeping whatever did parse. Never raises:
    a missing or broken rc_shift must never cost a forecast.
    """
    if raw is None:
        return None, "absent"
    if not isinstance(raw, dict):
        return None, "malformed:not_an_object"
    bad: list[str] = []
    out: Dict[str, Any] = {}
    d = str(raw.get("direction") or "").strip().lower()
    if d in _RC_SHIFT_DIRECTION_ALIASES:
        out["direction"] = _RC_SHIFT_DIRECTION_ALIASES[d]
    else:
        bad.append("direction")
    ebc = _as_float(raw.get("expected_bucket_change"))
    if ebc is None:
        bad.append("expected_bucket_change")
    else:
        out["expected_bucket_change"] = ebc
    mm = _as_float(raw.get("mass_moved"))
    if mm is None or mm < 0 or mm > 1:
        bad.append("mass_moved")
    else:
        out["mass_moved"] = mm
    sk = _as_bool(raw.get("sharpness_kept"))
    if sk is None:
        bad.append("sharpness_kept")
    else:
        out["sharpness_kept"] = sk
    why = raw.get("why")
    if isinstance(why, str) and why.strip():
        out["why"] = why.strip()[:500]
    else:
        bad.append("why")
    if bad:
        return (out or None), "malformed:" + ",".join(bad)
    return out, "ok"


def normalise_rc_shift_in_trace(trace: Any) -> Any:
    """Replace ``trace['rc_shift']`` with its parsed form and record the status.

    A trace without ``rc_shift`` is returned untouched, so traces written under
    the legacy guidance keep their stored shape. A malformed value keeps the
    raw object under ``rc_shift_raw``.
    """
    if not isinstance(trace, dict) or "rc_shift" not in trace:
        return trace
    raw = trace.get("rc_shift")
    value, status = parse_rc_shift(raw)
    trace["rc_shift_status"] = status
    if status != "ok":
        trace["rc_shift_raw"] = raw
    trace["rc_shift"] = value
    return trace


def _norm_probs(values: Any, expected_k: int) -> Optional[List[float]]:
    if not isinstance(values, list) or len(values) != expected_k:
        return None
    vals = [_as_float(v) for v in values]
    if any(v is None or v < 0 for v in vals):
        return None
    total = sum(vals)  # type: ignore[arg-type]
    if total <= 0:
        return None
    return [v / total for v in vals]  # type: ignore[operator]


def check_rc_sharpness(
    trace: dict,
    expected_k: int,
    posterior: Optional[List[float]] = None,
    *,
    max_modal_loss: float = RC_SHARPNESS_MAX_MODAL_LOSS,
    min_shift: float = RC_SHARPNESS_MIN_SHIFT,
) -> dict:
    """Did the update step SPREAD the prior rather than SHIFT it?

    Flags a posterior that keeps less than ``1 - max_modal_loss`` of the prior's
    modal-bucket mass while its expected bucket index moved by less than
    ``min_shift``. The posterior defaults to the last ``post_update_spd`` in the
    trace. Diagnostic only: it never blocks or changes a forecast.
    """
    prior = trace.get("prior") if isinstance(trace, dict) else None
    prior_spd = _norm_probs(prior.get("spd") if isinstance(prior, dict) else None, expected_k)
    if prior_spd is None:
        return {"checked": False, "reason": "no usable prior"}
    if posterior is None:
        updates = trace.get("updates") if isinstance(trace.get("updates"), list) else []
        for u in reversed(updates):
            if isinstance(u, dict) and _norm_probs(u.get("post_update_spd"), expected_k):
                posterior = u.get("post_update_spd")
                break
    post = _norm_probs(posterior, expected_k)
    if post is None:
        return {"checked": False, "reason": "no usable posterior"}
    mode = max(range(expected_k), key=lambda i: prior_spd[i])
    modal_loss = (prior_spd[mode] - post[mode]) / prior_spd[mode] if prior_spd[mode] > 0 else 0.0
    shift = sum(i * p for i, p in enumerate(post)) - sum(i * p for i, p in enumerate(prior_spd))
    flagged = modal_loss > max_modal_loss and abs(shift) < min_shift
    return {
        "checked": True,
        "prior_modal_bucket": mode,
        "modal_mass_loss": round(modal_loss, 4),
        "expected_bucket_shift": round(shift, 4),
        "spread_without_shift": flagged,
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

    # Map value to bucket using the canonical metric thresholds.
    thresholds = interior_thresholds_for(metric.upper())
    if not thresholds:  # unknown metric: fall back to PA
        thresholds = interior_thresholds_for("PA")

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
    expected_k = n_buckets_for(metric) or 5
    if not isinstance(prior_spd, list) or len(prior_spd) != expected_k:
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


def _check_delta_arithmetic(trace: dict, expected_k: int) -> dict:
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
        if isinstance(delta, list) and len(delta) == expected_k:
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
            and len(prev_spd) == expected_k
            and isinstance(post_spd, list)
            and len(post_spd) == expected_k
            and isinstance(delta, list)
            and len(delta) == expected_k
        ):
            l1 = sum(
                abs(post_spd[i] - (prev_spd[i] + delta[i]))
                for i in range(expected_k)
            )
            if l1 >= 0.1:
                ok = False
                detail["l1_norm"] = round(l1, 4)

        if ok:
            n_pass += 1
        details.append(detail)

        # Update prev_spd for chain checking
        if isinstance(post_spd, list) and len(post_spd) == expected_k:
            prev_spd = post_spd

    score = n_pass / max(n_total, 1)
    return {"score": round(score, 4), "n_updates": n_total, "n_pass": n_pass, "details": details}


def _check_magnitude_consistency(trace: dict, expected_k: int) -> dict:
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

        if not isinstance(delta, list) or len(delta) != expected_k or not magnitude:
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


# ---------------------------------------------------------------------------
# Prior anchor (PYTHIA_PRIOR_ANCHOR_SPD)
# ---------------------------------------------------------------------------

def _declared_month1_prior(trace: Any) -> Optional[List[float]]:
    """The declared prior as a month-1 vector: a single list, or the first month."""
    if not isinstance(trace, dict):
        return None
    prior = trace.get("prior")
    spd = prior.get("spd") if isinstance(prior, dict) else None
    if isinstance(spd, dict):
        keys = sorted(spd.keys())
        spd = spd.get(keys[0]) if keys else None
        if isinstance(spd, dict):
            spd = spd.get("probs")
    if not isinstance(spd, list) or not spd:
        return None
    try:
        return [float(x) for x in spd]
    except (TypeError, ValueError):
        return None


def prior_anchor_check(trace: Any, shown_month1: List[float], version: str) -> Dict[str, Any]:
    """How far the declared prior sits from the distribution the prompt showed.

    Jensen-Shannon DISTANCE (square root of the base-2 divergence, 0..1): 0
    means the member copied the distribution as told. Stored on the trace as
    ``prior_anchor_check`` and never blocks a forecast.
    """
    declared = _declared_month1_prior(trace)
    out: Dict[str, Any] = {"version": version, "js_distance": None, "status": "ok"}
    if declared is None:
        out["status"] = "no_declared_prior"
        return out
    if len(declared) != len(shown_month1):
        out["status"] = f"length_mismatch:{len(declared)}!={len(shown_month1)}"
        return out
    p = _norm_probs(declared, len(shown_month1))
    q = _norm_probs(shown_month1, len(shown_month1))
    if p is None or q is None:
        out["status"] = "unnormalisable"
        return out
    m = [(a + b) / 2.0 for a, b in zip(p, q)]

    def _kl(x: List[float], y: List[float]) -> float:
        return sum(a * math.log2(a / b) for a, b in zip(x, y) if a > 0 and b > 0)

    jsd = max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))
    out["js_distance"] = round(math.sqrt(jsd), 6)
    return out
