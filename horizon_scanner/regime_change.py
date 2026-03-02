# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import logging
import math
import os
from typing import Any

logger = logging.getLogger(__name__)

_ALLOWED_DIRECTIONS = {"up", "down", "mixed", "unclear"}
_ALLOWED_WINDOWS = {
    "month_1",
    "month_2",
    "month_3",
    "month_4",
    "month_5",
    "month_6",
    "month_1-2",
    "month_3-4",
    "month_5-6",
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.debug("Regime-change: failed to parse env %s=%r; using default %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.debug("Regime-change: failed to parse env %s=%r; using default %s", name, raw, default)
        return default


def clamp01(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool):
        logger.debug("Regime-change: bool provided where numeric expected: %r", x)
        return None
    try:
        value = float(x)
    except (TypeError, ValueError):
        logger.debug("Regime-change: failed to coerce numeric value from %r", x)
        return None
    if not math.isfinite(value):
        logger.debug("Regime-change: non-finite numeric value from %r", x)
        return None
    if value < 0.0 or value > 1.0:
        logger.debug("Regime-change: clamping numeric value %s into [0,1]", value)
    return max(0.0, min(1.0, value))


def coerce_direction(x: Any) -> str:
    if x is None:
        return "unclear"
    value = str(x).strip().lower()
    if value in _ALLOWED_DIRECTIONS:
        return value
    logger.debug("Regime-change: invalid direction %r; using 'unclear'", x)
    return "unclear"


def coerce_window(x: Any) -> str:
    if x is None:
        return ""
    value = str(x).strip()
    if value in _ALLOWED_WINDOWS:
        return value
    if value:
        logger.debug("Regime-change: invalid window %r; using empty", x)
    return ""


def compute_score(likelihood: float | None, magnitude: float | None) -> float:
    if likelihood is None or magnitude is None:
        return 0.0
    try:
        score = float(likelihood) * float(magnitude)
    except (TypeError, ValueError):
        logger.debug("Regime-change: unable to compute score from %r, %r", likelihood, magnitude)
        return 0.0
    return max(0.0, min(1.0, score))


def compute_level(likelihood: float | None, magnitude: float | None, score: float | None) -> int:
    lvl0_likelihood = _env_float("PYTHIA_HS_RC_LEVEL0_LIKELIHOOD", 0.45)
    lvl0_score = _env_float("PYTHIA_HS_RC_LEVEL0_SCORE", 0.25)
    lvl1_likelihood = _env_float("PYTHIA_HS_RC_LEVEL1_LIKELIHOOD", 0.45)
    lvl1_score = _env_float("PYTHIA_HS_RC_LEVEL1_SCORE", 0.25)
    lvl2_likelihood = _env_float("PYTHIA_HS_RC_LEVEL2_LIKELIHOOD", 0.60)
    lvl2_magnitude = _env_float("PYTHIA_HS_RC_LEVEL2_MAGNITUDE", 0.50)
    lvl3_likelihood = _env_float("PYTHIA_HS_RC_LEVEL3_LIKELIHOOD", 0.75)
    lvl3_magnitude = _env_float("PYTHIA_HS_RC_LEVEL3_MAGNITUDE", 0.60)

    likelihood_val = likelihood if isinstance(likelihood, (int, float)) else 0.0
    magnitude_val = magnitude if isinstance(magnitude, (int, float)) else 0.0
    score_val = score if isinstance(score, (int, float)) else 0.0

    if likelihood_val >= lvl3_likelihood and magnitude_val >= lvl3_magnitude:
        return 3
    if likelihood_val >= lvl2_likelihood and magnitude_val >= lvl2_magnitude:
        return 2
    if likelihood_val >= lvl1_likelihood and score_val >= lvl1_score:
        return 1
    if likelihood_val < lvl0_likelihood or score_val < lvl0_score:
        return 0
    return 1


def should_force_full_spd(level: int | None, score: float | None) -> bool:
    level_val = int(level or 0)
    return level_val > 0


def _coerce_bullets(raw: Any, limit: int = 6) -> list[str]:
    if raw is None:
        return []
    values: list[str] = []
    if isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if not text:
            continue
        values.append(text)
        if len(values) >= limit:
            break
    return values


def _coerce_int(raw: Any) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.debug("Regime-change: failed to coerce int from %r", raw)
        return None


def _coerce_evidence_refs(raw: Any) -> list[str]:
    if raw is None:
        return []
    refs: list[str] = []
    if isinstance(raw, list):
        items = raw
    else:
        items = [raw]
    for item in items:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            refs.append(text)
    return refs


def _coerce_trigger_signals(raw: Any, limit: int = 6) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        logger.debug("Regime-change: trigger_signals expected list, got %r", raw)
        raw_items = [raw]
    else:
        raw_items = raw

    signals: list[dict[str, Any]] = []
    for item in raw_items:
        if item is None:
            continue
        if isinstance(item, dict):
            signal_text = str(item.get("signal") or "").strip()
            timeframe = _coerce_int(item.get("timeframe_months"))
            evidence_refs = _coerce_evidence_refs(item.get("evidence_refs"))
        else:
            logger.debug("Regime-change: trigger signal item not dict: %r", item)
            signal_text = str(item).strip()
            timeframe = None
            evidence_refs = []
        if not signal_text:
            continue
        signals.append(
            {
                "signal": signal_text,
                "timeframe_months": timeframe,
                "evidence_refs": evidence_refs,
            }
        )
        if len(signals) >= limit:
            break
    return signals


def coerce_regime_change(obj: Any) -> dict[str, Any]:
    if obj is None:
        raw = {}
    elif isinstance(obj, dict):
        raw = obj
    else:
        logger.debug("Regime-change: expected object, got %r", obj)
        raw = {}

    likelihood = clamp01(raw.get("likelihood"))
    magnitude = clamp01(raw.get("magnitude"))
    direction = coerce_direction(raw.get("direction"))
    window = coerce_window(raw.get("window"))
    rationale_bullets = _coerce_bullets(raw.get("rationale_bullets"))
    trigger_signals = _coerce_trigger_signals(raw.get("trigger_signals"))

    valid = likelihood is not None and magnitude is not None
    normalized: dict[str, Any] = {
        "likelihood": likelihood,
        "direction": direction,
        "magnitude": magnitude,
        "window": window,
        "rationale_bullets": rationale_bullets,
        "trigger_signals": trigger_signals,
        "valid": valid,
    }
    if raw.get("status"):
        normalized["status"] = str(raw.get("status"))
    return normalized


# ---------------------------------------------------------------------------
# Run-level distribution sanity check (B4)
# ---------------------------------------------------------------------------

# Maximum fraction of hazard-country assessments allowed at each RC level
# before a warning is emitted.  Env-overridable for tuning.
_DIST_WARN_L1_FRAC = _env_float("PYTHIA_HS_RC_DIST_WARN_L1_FRAC", 0.25)
_DIST_WARN_L2_FRAC = _env_float("PYTHIA_HS_RC_DIST_WARN_L2_FRAC", 0.15)
_DIST_WARN_L3_FRAC = _env_float("PYTHIA_HS_RC_DIST_WARN_L3_FRAC", 0.08)


def check_rc_distribution(
    levels: list[int],
    *,
    run_id: str = "",
) -> dict[str, Any]:
    """Check the distribution of RC levels across a full HS run.

    Parameters
    ----------
    levels : list[int]
        One RC level (0-3) per hazard-country assessment in the run.
    run_id : str, optional
        Identifier for log messages.

    Returns
    -------
    dict with keys:
        total, counts (dict level→count), fractions (dict level→frac),
        warnings (list[str])  — empty if distribution looks healthy.
    """
    total = len(levels)
    if total == 0:
        return {"total": 0, "counts": {}, "fractions": {}, "warnings": []}

    counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    for lvl in levels:
        counts[lvl] = counts.get(lvl, 0) + 1

    fracs = {lvl: cnt / total for lvl, cnt in counts.items()}

    warnings: list[str] = []
    warn_thresholds = {
        1: _DIST_WARN_L1_FRAC,
        2: _DIST_WARN_L2_FRAC,
        3: _DIST_WARN_L3_FRAC,
    }
    for lvl, threshold in warn_thresholds.items():
        if fracs.get(lvl, 0) > threshold:
            pct = fracs[lvl] * 100
            thr_pct = threshold * 100
            msg = (
                f"RC distribution warning: L{lvl} assigned to {counts[lvl]}/{total} "
                f"assessments ({pct:.1f}%), exceeding {thr_pct:.0f}% threshold"
            )
            warnings.append(msg)
            logger.warning("HS run %s: %s", run_id, msg)

    if not warnings:
        logger.info(
            "HS run %s: RC distribution healthy — L0=%d L1=%d L2=%d L3=%d (total=%d)",
            run_id,
            counts[0],
            counts[1],
            counts[2],
            counts[3],
            total,
        )

    return {
        "total": total,
        "counts": counts,
        "fractions": fracs,
        "warnings": warnings,
    }
