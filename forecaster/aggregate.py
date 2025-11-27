# ANCHOR: aggregate (paste whole file)
from __future__ import annotations
import math, numpy as np
from typing import List, Tuple, Dict, Optional, Any

from .ensemble import EnsembleResult, sanitize_mcq_vector, MemberOutput
from . import bayes_mc as BMC

# Default centroids (v1) for PA buckets (can be moved to config later)
SPD_BUCKET_CENTROIDS_DEFAULT = [5_000.0, 25_000.0, 120_000.0, 350_000.0, 700_000.0]

# Default centroids (v1) for conflict fatalities buckets (per month)
SPD_BUCKET_CENTROIDS_FATALITIES_DEFAULT = [2.0, 14.5, 55.0, 250.0, 750.0]

def _extract_gtmc1_prob(sig: dict | None) -> float | None:
    """
    Pull a probability-like value out of GTMC1 signal dict.
    Accepts several aliases but prefers 'exceedance_ge_50'.
    Returns a float in [0,1] or None if unavailable/invalid.

    Why this helper exists:
      - Different upstreams may emit slightly different keys for the same value
        (e.g., 'exceedance_ge_50', 'gtmc1_prob', 'prob_yes', 'p_yes').
      - Normalizing here makes the aggregator robust, so GTMC1 affects the blend
        whenever a valid probability shows up.
    """
    if not isinstance(sig, dict):
        return None
    for k in ("exceedance_ge_50", "gtmc1_prob", "prob_yes", "p_yes"):
        if k in sig:
            try:
                v = float(sig[k])
                if 0.0 <= v <= 1.0:
                    return v
            except Exception:
                pass
    return None

# ---------------- Binary ----------------

def aggregate_binary(
    ensemble_res: EnsembleResult,
    gtmc1_signal: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Aggregate binary forecasts via a Beta + Monte Carlo layer.

    NOTE: `gtmc1_signal` is now **ignored**. GTMC1 is folded into the research bundle
    and influences the LLM forecasts directly, rather than acting as a separate
    expert in the BMC layer. We keep the parameter for backwards-compatibility.
    """

    evidences: List[BMC.BinaryEvidence] = []

    # 1) LLM Ensemble Evidence
    for m in ensemble_res.members:
        if m.ok and isinstance(m.parsed, (float, int)):
            w = (weights or {}).get(m.name, 1.0)
            evidences.append(BMC.BinaryEvidence(p=float(m.parsed), w=w))

    # Weak prior to let evidence dominate
    prior = BMC.BinaryPrior(alpha=0.1, beta=0.1)

    if not evidences:
        # If nothing valid came in (neither LLMs nor GTMC1), revert to 0.5.
        return 0.5, {"method": "empty_evidence", "mean": 0.5}

    # Run the Bayesian Monte Carlo update across all evidence.
    bmc_summary = BMC.update_binary_with_mc(prior, evidences)

    # Remove the non-serializable numpy array before returning (keeps logs JSON-safe).
    bmc_summary.pop("samples", None)

    final_prob = float(bmc_summary.get("mean", 0.5))
    return final_prob, bmc_summary

# ---------------- MCQ ----------------

def aggregate_mcq(
    ensemble_res: EnsembleResult,
    n_options: int,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[float], Dict[str, Any]]:
    """
    Bayesian aggregation for MCQ using bayes_mc.
    """
    evidences: List[BMC.MCQEvidence] = []
    for m in ensemble_res.members:
        if m.ok and isinstance(m.parsed, list) and len(m.parsed) == n_options:
            w = (weights or {}).get(m.name, 1.0)
            evidences.append(BMC.MCQEvidence(probs=m.parsed, w=w))
            
    if not evidences:
        # Fallback to uniform if no valid evidence
        vec = [1.0 / n_options] * n_options
        return vec, {"method": "empty_evidence", "mean": vec}

    # Weak prior
    prior = BMC.DirichletPrior(alphas=[0.1] * n_options)
    bmc_summary = BMC.update_mcq_with_mc(prior, evidences)
    
    # Remove the non-serializable numpy array before returning
    bmc_summary.pop("samples", None)

    final_vector = bmc_summary.get("mean", [1.0 / n_options] * n_options)
    return sanitize_mcq_vector(final_vector), bmc_summary

# ---------------- SPD (5 buckets × 6 months) ----------------

def _expected_from_spd(
    spd_by_month: Dict[str, List[float]],
    centroids: List[float],
) -> Dict[str, float]:
    """
    Compute expected people affected per month given SPD and bucket centroids.
    Assumes len(centroids) == number of buckets.
    """
    ev: Dict[str, float] = {}
    v = np.array(centroids, dtype=float)
    for key, probs in spd_by_month.items():
        p = np.array(probs, dtype=float)
        if p.size != v.size:
            p = np.ones_like(v) / float(v.size)
        # Ensure normalization
        s = float(p.sum())
        if s <= 0.0:
            p = np.ones_like(v) / float(v.size)
        else:
            p = p / s
        ev[key] = float((p * v).sum())
    return ev

def aggregate_spd(
    ensemble_res: EnsembleResult,
    *,
    n_months: int = 6,
    n_buckets: int = 5,
    weights: Optional[Dict[str, float]] = None,
    bucket_centroids: Optional[List[float]] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, float], Dict[str, Any]]:
    """
    Aggregate SPD forecasts (5 PA buckets × 6 months) using the MCQ/Dirichlet layer.

    Returns:
      - spd_mean_by_month: {"month_1": [..5..], ..., "month_6": [..5..]}
      - expected_by_month: {"month_1": ev1, ..., "month_6": ev6}
      - summary:          {"month_1": {...Dirichlet stats...}, ...}
    """
    # Collect evidence per month
    per_month: Dict[str, List[BMC.MCQEvidence]] = {f"month_{i}": [] for i in range(1, n_months + 1)}

    for m in ensemble_res.members:
        if not m.ok or not isinstance(m.parsed, dict):
            continue
        w = (weights or {}).get(m.name, 1.0)
        for m_idx in range(1, n_months + 1):
            key = f"month_{m_idx}"
            raw_vec = m.parsed.get(key, [])
            vec = sanitize_mcq_vector(raw_vec if isinstance(raw_vec, list) else [], n_options=n_buckets)
            per_month[key].append(BMC.MCQEvidence(probs=vec, w=w))

    spd_mean: Dict[str, List[float]] = {}
    summary: Dict[str, Any] = {}
    centroids = bucket_centroids or SPD_BUCKET_CENTROIDS_DEFAULT

    for m_idx in range(1, n_months + 1):
        key = f"month_{m_idx}"
        evidences = per_month.get(key) or []
        if not evidences:
            # Uniform fallback if no valid evidence
            vec = [1.0 / float(n_buckets)] * n_buckets
            spd_mean[key] = vec
            summary[key] = {"method": "empty_evidence", "mean": vec}
            continue

        prior = BMC.DirichletPrior(alphas=[0.1] * n_buckets)
        bmc_summary = BMC.update_mcq_with_mc(prior, evidences)
        # Drop samples array for serializability
        bmc_summary.pop("samples", None)

        mean = bmc_summary.get("mean")
        if isinstance(mean, list) and len(mean) == n_buckets:
            vec = sanitize_mcq_vector(mean, n_options=n_buckets)
        else:
            vec = [1.0 / float(n_buckets)] * n_buckets

        spd_mean[key] = vec
        summary[key] = bmc_summary

    expected = _expected_from_spd(spd_mean, centroids)
    return spd_mean, expected, summary

# ---------------- Numeric / Discrete ----------------

def aggregate_numeric(
    ensemble_res: EnsembleResult,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Numeric aggregation via a Normal Mixture Model in bayes_mc.
    Each LLM's forecast is treated as a separate distribution.
    """
    evidences: List[BMC.NumericEvidence] = []
    
    def _coerce_p10_p50_p90(d: Dict[str, float]) -> Tuple[float, float, float]:
        p10 = float(d.get("P10", 10.0))
        p90 = float(d.get("P90", 90.0))
        if "P50" in d:
            p50 = float(d["P50"])
        else:
            p50 = 0.5 * (p10 + p90)
        # Ensure order
        p10, p50, p90 = sorted([p10, p50, p90])
        return p10, p50, p90

    for m in ensemble_res.members:
        if m.ok and isinstance(m.parsed, dict):
            p10, p50, p90 = _coerce_p10_p50_p90(m.parsed)
            w = (weights or {}).get(m.name, 1.0)
            evidences.append(BMC.NumericEvidence(p10=p10, p50=p50, p90=p90, w=w))

    if not evidences:
        # Degenerate default
        quantiles = {"P10": 10.0, "P50": 50.0, "P90": 90.0}
        return quantiles, {"method": "empty_evidence", "p10": 10.0, "p50": 50.0, "p90": 90.0}

    bmc_summary = BMC.update_numeric_with_mc(evidences)
    
    # Remove the non-serializable numpy array before returning
    bmc_summary.pop("samples", None)

    quantiles = {
        "P10": bmc_summary.get("p10"),
        "P50": bmc_summary.get("p50"),
        "P90": bmc_summary.get("p90"),
    }
    return quantiles, bmc_summary
