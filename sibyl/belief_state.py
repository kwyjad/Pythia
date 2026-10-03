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
from sibyl.ledger import parse_ledger_add

VALID_ACTIONS = ("brave_search", "reliefweb_search", "fetch_url", "submit")
TOOL_ACTIONS = ("brave_search", "reliefweb_search", "fetch_url")
#: The research plan's slots, in the default order (Part 5 lanes reorder them).
PLAN_SLOTS = ("resolver", "nowcast", "drivers", "calendar", "reversion", "disconfirm")
PLAN_STATUSES = ("pending", "done", "failed")
# Each step returns month_1 and month_6 objects (p_zero + positive quantiles).
VALID_CONFIDENCE = ("low", "medium", "high")


class BeliefStateError(ValueError):
    """Raised when a model step response cannot be parsed into a valid state."""


@dataclass
class MonthBelief:
    """One horizon: P(zero) and the 0.05..0.95 quantiles given a positive value.

    For flood and cyclone questions ``p_zero`` is the chance of zero OR no
    record, since a month the source has no record for does not resolve.
    """

    p_zero: float
    quantiles_positive: Dict[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p_zero": round(float(self.p_zero), 6),
            "quantiles_positive": {
                str(k): v for k, v in sorted(self.quantiles_positive.items())
            },
        }

    def dist(self):
        from sibyl.aggregate import MonthDist  # noqa: PLC0415

        return MonthDist(p_zero=self.p_zero, qpos=dict(self.quantiles_positive))


@dataclass
class BeliefState:
    """Structured belief state, values in the question's native units.

    Elicited at two horizons, month 1 and month 6 of the window (Oct 2026);
    months 2-5 are mixtures of the two. ``quantiles`` is the legacy view:
    the month-1 distribution read at the seven old levels, kept so stored
    records stay readable by older code.
    """

    month_1: MonthBelief
    month_6: MonthBelief
    confidence: str = "low"
    evidence_higher: List[str] = field(default_factory=list)
    evidence_lower: List[str] = field(default_factory=list)
    open_questions: List[str] = field(default_factory=list)
    baserate_reconciliation: str = ""
    step_rationale: str = ""
    # The research plan (Oct 2026): {slot: {"status", "finding"}}.
    plan: Dict[str, Dict[str, str]] = field(default_factory=dict)

    @property
    def quantiles(self) -> Dict[float, float]:
        return legacy_quantiles(self.month_1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "month_1": self.month_1.to_dict(),
            "month_6": self.month_6.to_dict(),
            "confidence": self.confidence,
            "evidence_higher": list(self.evidence_higher),
            "evidence_lower": list(self.evidence_lower),
            "open_questions": list(self.open_questions),
            "baserate_reconciliation": self.baserate_reconciliation,
            "step_rationale": self.step_rationale,
            "plan": {k: dict(v) for k, v in self.plan.items()},
        }


def empty_plan() -> Dict[str, Dict[str, str]]:
    return {slot: {"status": "pending", "finding": ""} for slot in PLAN_SLOTS}


def legacy_quantiles(month: MonthBelief) -> Dict[float, float]:
    """A month's distribution read at the old seven QUANTILE_LEVELS."""
    from sibyl.aggregate import _grid, quantiles_from_cdf_fn  # noqa: PLC0415

    d = month.dist()
    q = quantiles_from_cdf_fn(d.cdf, _grid([d]), QUANTILE_LEVELS)
    return {lv: float(q[lv]) for lv in QUANTILE_LEVELS}


@dataclass
class ToolCall:
    """One tool call in a step: the action, its input text and options."""

    action: str
    action_input: str
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepDecision:
    """One parsed agent step: up to MAX_ACTIONS_PER_STEP actions + the belief.

    ``action`` / ``action_input`` are the first call (or ``submit``), kept for
    older readers of the trace.
    """

    action: str
    action_input: str
    belief: BeliefState
    repaired: bool = False  # the belief needed a clamp or monotonicity repair
    calls: List[ToolCall] = field(default_factory=list)
    submit_dropped: bool = False  # a submit sent beside tool calls was ignored
    plan_given: bool = False
    # New evidence-ledger items this step (sibyl/ledger.py), cleaned.
    ledger_add: List[Dict[str, Any]] = field(default_factory=list)


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


POS_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)


def _one_call(raw: Any) -> ToolCall:
    if not isinstance(raw, dict):
        raise BeliefStateError("each action must be an object with 'action' and 'action_input'")
    action = str(raw.get("action", "")).strip().lower()
    if action not in VALID_ACTIONS:
        raise BeliefStateError(f"invalid action {action!r}; expected one of {VALID_ACTIONS}")
    inp = raw.get("action_input", "")
    options: Dict[str, Any] = {}
    if isinstance(inp, dict):
        options = {k: v for k, v in inp.items()}
        text = str(options.pop("query", None) or options.pop("url", None) or "").strip()
    else:
        text = str(inp or "").strip()
    for key in ("lane", "language", "country", "extraction_request"):
        if key in raw and key not in options:
            options[key] = raw[key]
    if action in TOOL_ACTIONS and not text:
        raise BeliefStateError(f"action {action!r} requires a non-empty action_input")
    return ToolCall(action=action, action_input=text, options=options)


def _parse_actions(obj: Dict[str, Any]) -> tuple[List[ToolCall], bool, bool]:
    """(tool calls, submit dropped, truncated). No tool call means submit.

    A step may carry up to MAX_ACTIONS_PER_STEP calls in ``actions``, or the
    single ``action`` / ``action_input`` form. A submit sent beside tool calls
    is dropped: it must be a step of its own.
    """
    from sibyl.config import MAX_ACTIONS_PER_STEP  # noqa: PLC0415

    raw = obj.get("actions")
    if isinstance(raw, list) and raw:
        items = [_one_call(x) for x in raw]
    elif "action" in obj:
        items = [_one_call(obj)]
    else:
        raise BeliefStateError("no 'action' or 'actions' in the response")
    tools = [c for c in items if c.action != "submit"]
    dropped = bool(tools) and len(tools) < len(items)
    truncated = len(tools) > MAX_ACTIONS_PER_STEP
    return tools[:MAX_ACTIONS_PER_STEP], dropped, truncated


def _parse_plan(raw: Any) -> tuple[Dict[str, Dict[str, str]], bool]:
    """The plan as {slot: {status, finding}}; (empty plan, False) when absent."""
    plan = empty_plan()
    if not isinstance(raw, dict):
        return plan, False
    for slot in PLAN_SLOTS:
        entry = raw.get(slot)
        if isinstance(entry, str):
            entry = {"status": entry}
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", "pending")).strip().lower()
        plan[slot] = {
            "status": status if status in PLAN_STATUSES else "pending",
            "finding": str(entry.get("finding", "") or "")[:300],
        }
    return plan, True


def _parse_month(raw: Any, key: str) -> tuple[MonthBelief, bool]:
    """One horizon object. Repairs (clamps, monotonicity) are flagged, not refused."""
    if not isinstance(raw, dict):
        raise BeliefStateError(f"belief_state.{key} missing or not an object")
    repaired = False
    try:
        p_zero = float(raw.get("p_zero"))
    except (TypeError, ValueError) as exc:
        raise BeliefStateError(f"belief_state.{key}.p_zero missing or not a number") from exc
    if p_zero != p_zero:
        raise BeliefStateError(f"belief_state.{key}.p_zero is not a number")
    if not 0.0 <= p_zero <= 1.0:
        p_zero = min(max(p_zero, 0.0), 1.0)
        repaired = True
    qraw = raw.get("quantiles_positive")
    if not isinstance(qraw, dict) or not qraw:
        raise BeliefStateError(f"belief_state.{key}.quantiles_positive missing or empty")
    q: Dict[float, float] = {}
    for k, v in qraw.items():
        try:
            level, value = float(k), float(v)
        except (TypeError, ValueError) as exc:
            raise BeliefStateError(f"non-numeric {key} quantile entry {k!r}: {v!r}") from exc
        if value != value or value in (float("inf"), float("-inf")):
            raise BeliefStateError(f"non-finite {key} quantile value at level {level}")
        q[round(level, 4)] = value
    missing = [lv for lv in POS_LEVELS if lv not in q]
    if missing:
        raise BeliefStateError(f"{key}: missing required positive quantile levels: {missing}")
    q = {lv: q[lv] for lv in POS_LEVELS}
    if any(v < 1.0 for v in q.values()):
        q = {lv: max(1.0, v) for lv, v in q.items()}
        repaired = True
    q, rep = enforce_monotone_quantiles(q)
    return MonthBelief(p_zero=p_zero, quantiles_positive=q), repaired or rep


def parse_step_response(text: str) -> StepDecision:
    """Parse a model step response into a validated :class:`StepDecision`.

    Expected shape::

        {
          "action": "brave_search" | "fetch_url" | "submit",
          "action_input": "<query or url; empty for submit>",
          "belief_state": {
            "month_1": {"p_zero": p, "quantiles_positive": {"0.05": n, ..., "0.95": n}},
            "month_6": {"p_zero": p, "quantiles_positive": {...}},
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

    calls, submit_dropped, repaired_actions = _parse_actions(obj)

    bs = obj.get("belief_state")
    if not isinstance(bs, dict):
        raise BeliefStateError("missing belief_state object")

    repaired = False
    months: Dict[str, MonthBelief] = {}
    for key in ("month_1", "month_6"):
        mb, rep = _parse_month(bs.get(key), key)
        months[key] = mb
        repaired = repaired or rep

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

    plan, plan_given = _parse_plan(bs.get("plan"))
    belief = BeliefState(
        plan=plan,
        month_1=months["month_1"],
        month_6=months["month_6"],
        confidence=confidence,
        evidence_higher=_str_list("evidence_higher"),
        evidence_lower=_str_list("evidence_lower"),
        open_questions=_str_list("open_questions"),
        baserate_reconciliation=str(bs.get("baserate_reconciliation", "") or ""),
        step_rationale=str(bs.get("step_rationale", "") or ""),
    )
    if calls:
        action, action_input = calls[0].action, calls[0].action_input
    else:
        action, action_input = "submit", ""
    return StepDecision(
        action=action, action_input=action_input, belief=belief,
        repaired=repaired or repaired_actions, calls=calls,
        submit_dropped=submit_dropped, plan_given=plan_given,
        ledger_add=parse_ledger_add(obj),
    )


def initial_belief(reference: Optional[Dict[int, List[float]]], metric: str) -> BeliefState:
    """Seed the step-0 belief from the reference vectors (months 1 and 6).

    With no reference the seed is a labelled placeholder, never a claimed
    base rate; the agent's first update replaces it.
    """
    from sibyl.aggregate import dist_from_vector  # noqa: PLC0415

    if reference and reference.get(1):
        months = {}
        for m in (1, 6):
            d = dist_from_vector(reference.get(m) or reference[1], metric)
            months[m] = MonthBelief(p_zero=round(d.p_zero, 4),
                                    quantiles_positive={k: round(v, 2) for k, v in d.qpos.items()})
        return BeliefState(
            month_1=months[1],
            month_6=months[6],
            confidence="low",
            baserate_reconciliation="Seeded from the reference: this is the prior.",
            step_rationale="Step 0: prior only, no inside-view evidence yet.",
        )
    placeholder = MonthBelief(p_zero=0.5, quantiles_positive={lv: 1.0 for lv in POS_LEVELS})
    return BeliefState(
        month_1=placeholder,
        month_6=MonthBelief(p_zero=0.5, quantiles_positive={lv: 1.0 for lv in POS_LEVELS}),
        confidence="low",
        baserate_reconciliation=(
            "No reference was available for this question. These values are a "
            "placeholder, not a prior: replace them from your research."
        ),
        step_rationale="Step 0: prior only, no inside-view evidence yet.",
    )


def initial_belief_from_anchor(anchor_quantiles: Optional[Dict[float, float]]) -> BeliefState:
    """Legacy entry point: no reference vectors, so the labelled placeholder."""
    return initial_belief(None, "FATALITIES")
