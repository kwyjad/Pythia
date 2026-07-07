# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl base-rate injection — the outside view.

The base rate is the ONLY structured input Sibyl shares with standard
Pythia, by design. It is framed as a step-0 anchor the agent reasons from
and away from — never a target. The underlying summaries are built by the
forecaster's existing Resolver-DB loaders (``_build_history_summary``);
Sibyl adds its own framing:

* Natural hazards (FL/TC): per-calendar-month climatology when it exists
  (the ``seasonal_profile`` months dict); when only an aggregate mean is
  available it is flagged as annualized and the agent is instructed to
  seasonally adjust — FL/TC are strongly seasonal.
* DR: FEWS NET Phase 3+ recent monthly series (null-aware).
* Conflict (ACE): recent ACLED months framed as autocorrelated recency —
  a trajectory, not climatology.
* Shape: when only central values are available the agent is told to treat
  them as the mean of a right-skewed, heavy-tailed distribution and widen
  accordingly (a mean sits above a typical month for this kind of data).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Dict, List, Optional

from sibyl.config import QUANTILE_LEVELS

logger = logging.getLogger(__name__)

# Heavy-tail multipliers used to expand a central value into seed quantiles
# when the source provides no distribution shape. Deliberately right-skewed.
_CENTRAL_VALUE_MULTIPLIERS = {
    0.1: 0.2,
    0.25: 0.5,
    0.5: 1.0,
    0.75: 1.6,
    0.9: 2.5,
    0.95: 4.0,
    0.99: 10.0,
}


@dataclass
class BaseRate:
    """Everything the agent needs from the outside view."""

    summary: Dict[str, Any]
    prompt_text: str  # rendered anchor block for the step prompt
    anchor_quantiles: Optional[Dict[float, float]]  # seed for step-0 belief
    framing_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "prompt_text": self.prompt_text,
            "anchor_quantiles": (
                {str(k): v for k, v in sorted(self.anchor_quantiles.items())}
                if self.anchor_quantiles
                else None
            ),
            "framing_notes": list(self.framing_notes),
        }


def _forecast_calendar_months(forecast_keys: List[str]) -> List[int]:
    months: List[int] = []
    for key in forecast_keys:
        parts = str(key).split("-")
        if len(parts) >= 2:
            try:
                m = int(parts[1])
            except ValueError:
                continue
            if 1 <= m <= 12:
                months.append(m)
    return months


def _quantiles_from_multipliers(central: float) -> Dict[float, float]:
    return {lv: central * _CENTRAL_VALUE_MULTIPLIERS[lv] for lv in QUANTILE_LEVELS}


def _seasonal_month_stats(
    summary: Dict[str, Any], cal_months: List[int]
) -> List[Dict[str, Any]]:
    months_data = summary.get("months") or {}
    stats: List[Dict[str, Any]] = []
    for m in cal_months:
        entry = months_data.get(m) or months_data.get(str(m))
        if isinstance(entry, dict) and entry.get("n_observations", 0):
            stats.append(entry)
    return stats


def anchor_quantiles_from_summary(
    summary: Dict[str, Any], forecast_keys: List[str]
) -> Optional[Dict[float, float]]:
    """Derive seed quantiles (monthly value, native units) from a summary.

    These are deliberately rough — a starting shape for the step-0 belief
    that the agent's research updates and replaces. Returns None when the
    summary carries no usable numbers (the belief seed is then explicit
    zero-knowledge).
    """
    stype = summary.get("type", "")

    if stype == "seasonal_profile":
        stats = _seasonal_month_stats(summary, _forecast_calendar_months(forecast_keys))
        if not stats:
            # Annualized-mean path: no per-month rows for the window.
            all_means = [
                e.get("mean") for e in (summary.get("months") or {}).values()
                if isinstance(e, dict) and e.get("mean") is not None
            ]
            if not all_means:
                return None
            annual_mean = sum(float(v) for v in all_means) / len(all_means)
            return _quantiles_from_multipliers(annual_mean) if annual_mean > 0 else None
        med = median([float(e.get("median") or e.get("mean") or 0) for e in stats])
        lo = min(float(e.get("min") or 0) for e in stats)
        hi = max(float(e.get("max") or 0) for e in stats)
        hi = max(hi, med)
        return {
            0.1: lo,
            0.25: (lo + med) / 2.0,
            0.5: med,
            0.75: (med + hi) / 2.0,
            0.9: hi,
            0.95: hi * 1.5,
            0.99: hi * 3.0,
        }

    if stype == "conflict_trajectory":
        fat = summary.get("fatalities") or {}
        central = fat.get("trailing_3m_avg")
        if central is None and isinstance(fat.get("last_month"), dict):
            central = fat["last_month"].get("value")
        if central is None:
            return None
        central = float(central)
        return _quantiles_from_multipliers(central) if central > 0 else None

    if stype == "fewsnet_phase3":
        central = summary.get("recent_mean")
        peak = summary.get("recent_max")
        if central is None:
            return None
        central = float(central)
        if central <= 0:
            return None
        q = _quantiles_from_multipliers(central)
        if peak is not None and float(peak) > 0:
            q[0.9] = max(q[0.9], float(peak))
            q[0.95] = max(q[0.95], float(peak) * 1.25)
            q[0.99] = max(q[0.99], float(peak) * 2.0)
        return q

    return None


def build_framing_notes(summary: Dict[str, Any], forecast_keys: List[str]) -> List[str]:
    """Sibyl-specific outside-view framing appended to the anchor block."""
    stype = summary.get("type", "")
    cal_months = _forecast_calendar_months(forecast_keys)
    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April", 5: "May",
        6: "June", 7: "July", 8: "August", 9: "September", 10: "October",
        11: "November", 12: "December",
    }
    target_month_str = ", ".join(month_names.get(m, str(m)) for m in cal_months)

    notes: List[str] = [
        "This base rate is your OUTSIDE VIEW: an anchor to reason from and "
        "away from, never a target to reproduce. Your research is the "
        "inside view; reconcile the two explicitly at every step.",
    ]

    if stype == "seasonal_profile":
        stats = _seasonal_month_stats(summary, cal_months)
        if stats:
            notes.append(
                "The climatology above is per-calendar-month, covering your "
                f"exact forecast months ({target_month_str}). Neighbouring-"
                "month values are shown for seasonal context."
            )
        else:
            notes.append(
                "CAUTION: only an ANNUALIZED average is available — no "
                "per-month climatology for your forecast months "
                f"({target_month_str}). This hazard is strongly seasonal: "
                "adjust the anchor up or down for the season your forecast "
                "months fall in before using it."
            )
        notes.append(
            "Affected-population data is right-skewed and heavy-tailed: the "
            "mean sits above a typical month. Treat central values as the "
            "mean of a skewed distribution and keep your upper quantiles "
            "(p95/p99) wide."
        )
    elif stype == "conflict_trajectory":
        notes.append(
            "Conflict base rates are RECENT TRAJECTORY (autocorrelated "
            "month-to-month), not long-run climatology: the last few months "
            "are the strongest predictor of the next, but escalation and "
            "de-escalation happen. Weigh trend direction, and keep the "
            "right tail wide for escalation scenarios."
        )
    elif stype == "fewsnet_phase3":
        notes.append(
            "The IPC Phase 3+ series is slow-moving and assessment-gated "
            "(null months mean no assessment, not zero need). Anchor on the "
            "recent observed values and interpolate through gaps."
        )
    else:
        notes.append(
            "No historical base rate is available for this question. Build "
            "your distribution from research alone and keep uncertainty "
            "wide across all quantiles."
        )

    notes.append(
        "Only a mean/median with no distribution? Widen: for this class of "
        "data p99 is typically several multiples of the median."
    )
    return notes


def load_base_rate(
    iso3: str,
    hazard_code: str,
    metric: str,
    forecast_keys: List[str],
) -> BaseRate:
    """Fetch + frame the outside-view anchor for one question.

    Reuses the forecaster's Resolver-DB summary builders and prompt
    renderer (lazy import — ``forecaster.cli`` pulls a large module tree).
    """
    from forecaster.cli import _build_history_summary  # noqa: PLC0415
    from forecaster.history_loaders import _format_base_rate_for_prompt  # noqa: PLC0415

    try:
        summary = _build_history_summary(iso3, hazard_code, metric)
    except Exception as exc:  # DB unavailable -> explicit no-anchor state
        logger.warning(
            "sibyl.base_rates: history summary failed for %s/%s/%s: %s",
            iso3, hazard_code, metric, exc,
        )
        summary = {"type": "no_base_rate", "note": f"base rate unavailable: {exc}"}

    try:
        rendered = _format_base_rate_for_prompt(summary, forecast_keys, iso3, hazard_code)
    except Exception as exc:
        logger.warning("sibyl.base_rates: formatting failed: %s", exc)
        rendered = "BASE RATE: unavailable."

    notes = build_framing_notes(summary, forecast_keys)
    anchor = anchor_quantiles_from_summary(summary, forecast_keys)
    prompt_text = rendered + "\n\n" + "\n".join(f"- {n}" for n in notes)
    return BaseRate(
        summary=summary,
        prompt_text=prompt_text,
        anchor_quantiles=anchor,
        framing_notes=notes,
    )
