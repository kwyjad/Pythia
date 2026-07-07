# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl belief state: the agent's running memory.

Each agent step returns structured JSON containing an action and an updated
belief state. Raw retrieved text is never accumulated into a growing
context — the belief state IS the memory carried between steps.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sibyl.config import QUANTILE_LEVELS

VALID_ACTIONS = ("brave_search", "fetch_url", "submit")
VALID_CONFIDENCE = ("low", "medium", "high")


class BeliefStateError(ValueError):
    """Raised when a model step response cannot be parsed into a valid state."""


@dataclass
class BeliefState:
    """Structured belief state, values in the question's native units."""

    quantiles: Dict[float, float]
    confidence: str = "low"
    evidence_higher: List[str] = field(default_factory=list)
    evidence_lower: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    baserate_reconciliation: str = ""
    step_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "quantiles": {str(k): v for k, v in sorted(self.quantiles.items())},
            "confidence": self.confidence,
            "evidence_higher": list(self.evidence_higher),
            "evidence_lower": list(self.evidence_lower),
            "open_questions": list(self.open_questions),
            "baserate_reconciliation": self.baserate_reconciliation,
            "step_rationale": self.step_rationale,
        }


@dataclass
class StepDecision:
    """One parsed agent step: an action plus the updated belief state."""

    action: str
    action_input: str
    belief: BeliefState
    repaired: bool = False  # quantiles needed a monotonicity repair


def _extract_json(text: str) -> Dict[str, Any]:
    """Extract the first JSON object from a model response.

    Tolerates markdown code fences and leading/trailing prose, mirroring
    the lenient parsing used elsewhere in the forecaster.
    """
    if not text or not text.strip():
        raise BeliefStateError("empty model response")

    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise BeliefStateError("no JSON object found in response")
        cleaned = cleaned[start : end + 1]

    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise BeliefStateError(f"invalid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise BeliefStateError("top-level JSON is not an object")
    return obj


def enforce_monotone_quantiles(
    quantiles: Dict[float, float],
) -> tuple[Dict[float, float], bool]:
    """Repair a quantile set to be non-decreasing in level order.

    Uses a running-maximum sweep (isotonic-lite): each quantile is raised to
    at least the value of the previous level. Returns (repaired_dict,
    was_repaired). Negative values are floored at 0 (affected/fatalities
    counts cannot be negative).
    """
    repaired = False
    out: Dict[float, float] = {}
    running = 0.0
    first = True
    for level in sorted(quantiles):
        val = float(quantiles[level])
        if val < 0.0:
            val = 0.0
            repaired = True
        if first:
            running = val
            first = False
        elif val < running:
            val = running
            repaired = True
        else:
            running = val
        running = max(running, val)
        out[level] = val
    return out, repaired


def parse_step_response(text: str) -> StepDecision:
    """Parse a model step response into a validated :class:`StepDecision`.

    Expected shape::

        {
          "action": "brave_search" | "fetch_url" | "submit",
          "action_input": "<query or url; empty for submit>",
          "belief_state": {
            "quantiles": {"0.1": n, ..., "0.99": n},
            "confidence": "low|medium|high",
            "evidence_higher": [...], "evidence_lower": [...],
            "open_questions": [...],
            "baserate_reconciliation": "...", "step_rationale": "..."
          }
        }

    Raises :class:`BeliefStateError` on malformed input (the caller
    retries). Monotonicity violations in quantiles are repaired, not
    rejected, and flagged via ``StepDecision.repaired``.
    """
    obj = _extract_json(text)

    action = str(obj.get("action", "")).strip().lower()
    if action not in VALID_ACTIONS:
        raise BeliefStateError(f"invalid action {action!r}; expected one of {VALID_ACTIONS}")

    action_input = str(obj.get("action_input", "") or "").strip()
    if action in ("brave_search", "fetch_url") and not action_input:
        raise BeliefStateError(f"action {action!r} requires a non-empty action_input")

    bs = obj.get("belief_state")
    if not isinstance(bs, dict):
        raise BeliefStateError("missing belief_state object")

    raw_q = bs.get("quantiles")
    if not isinstance(raw_q, dict) or not raw_q:
        raise BeliefStateError("belief_state.quantiles missing or empty")

    quantiles: Dict[float, float] = {}
    for key, val in raw_q.items():
        try:
            level = float(key)
            value = float(val)
        except (TypeError, ValueError) as exc:
            raise BeliefStateError(f"non-numeric quantile entry {key!r}: {val!r}") from exc
        if not (0.0 < level < 1.0):
            raise BeliefStateError(f"quantile level {level} outside (0, 1)")
        if value != value or value in (float("inf"), float("-inf")):
            raise BeliefStateError(f"non-finite quantile value at level {level}")
        quantiles[level] = value

    missing = [lv for lv in QUANTILE_LEVELS if lv not in quantiles]
    if missing:
        raise BeliefStateError(f"missing required quantile levels: {missing}")

    quantiles, repaired = enforce_monotone_quantiles(quantiles)

    confidence = str(bs.get("confidence", "low")).strip().lower()
    if confidence not in VALID_CONFIDENCE:
        confidence = "low"

    def _str_list(key: str) -> List[str]:
        raw = bs.get(key, [])
        if isinstance(raw, str):
            return [raw] if raw.strip() else []
        if isinstance(raw, list):
            return [str(x) for x in raw if str(x).strip()]
        return []

    belief = BeliefState(
        quantiles=quantiles,
        confidence=confidence,
        evidence_higher=_str_list("evidence_higher"),
        evidence_lower=_str_list("evidence_lower"),
        open_questions=_str_list("open_questions"),
        baserate_reconciliation=str(bs.get("baserate_reconciliation", "") or ""),
        step_rationale=str(bs.get("step_rationale", "") or ""),
    )
    return StepDecision(
        action=action, action_input=action_input, belief=belief, repaired=repaired
    )


def initial_belief_from_anchor(anchor_quantiles: Optional[Dict[float, float]]) -> BeliefState:
    """Seed a step-0 belief state from the outside-view anchor.

    When the base rate provides no usable numbers the seed is an explicit
    zero-knowledge state; the agent's first update replaces it.
    """
    if anchor_quantiles:
        q, _ = enforce_monotone_quantiles(dict(anchor_quantiles))
    else:
        q = {lv: 0.0 for lv in QUANTILE_LEVELS}
    return BeliefState(
        quantiles=q,
        confidence="low",
        baserate_reconciliation="Seeded from the outside-view base-rate anchor.",
        step_rationale="Step 0: prior only, no inside-view evidence yet.",
    )
