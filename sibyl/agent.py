# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl agentic trial loop — the inside view.

Each trial is a sequential tool-use loop: at every step the model returns,
as structured JSON, an action (``brave_search`` / ``fetch_url`` /
``submit``) and an UPDATED belief state. This sequential, belief-updating
structure — not a search-dump-then-reason-once design — is the
highest-leverage part of the method.

Raw retrieved text is never accumulated into an ever-growing context: each
step's prompt carries only the question, the outside-view anchor, the
current belief state, and the LAST tool result. The belief state is the
running memory.

Trial diversity: ``claude-opus-4-8`` rejects sampling parameters
(temperature returns HTTP 400), so the K trials are differentiated by
explicit perspective seeds in the prompt rather than temperature.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional

from sibyl.base_rates import BaseRate
from sibyl.belief_state import (
    BeliefState,
    BeliefStateError,
    StepDecision,
    initial_belief_from_anchor,
    parse_step_response,
)
from sibyl.config import (
    ANTHROPIC_MAX_ATTEMPTS,
    MAX_STEPS,
    MODEL,
    QUANTILE_LEVELS,
)
from sibyl.cost import COST_KIND_BRAVE, COST_KIND_OPUS, CostBreakdown, CostTracker, log_sibyl_call
from sibyl.leakage import LeakageStats, is_backtest
from sibyl.select_questions import SibylQuestion
from sibyl.tools import ToolResult, brave_search, fetch_url

logger = logging.getLogger(__name__)

# Per-trial perspective seeds (temperature substitute — see module docstring).
TRIAL_PERSPECTIVES = [
    (
        "Base-rate-weighted perspective: give the outside view substantial "
        "weight; demand strong evidence before departing far from it."
    ),
    (
        "Tail-risk-sensitive perspective: actively probe for escalation and "
        "compounding-shock scenarios that would put the outcome in the "
        "upper quantiles; remain calibrated, not alarmist."
    ),
    (
        "Recent-signal-driven perspective: weight the freshest ground "
        "reporting most heavily and stress-test whether the base rate is "
        "already stale."
    ),
    (
        "Contrarian-check perspective: identify the consensus narrative in "
        "the reporting and search for disconfirming evidence before "
        "settling your quantiles."
    ),
    (
        "Structural perspective: prioritize slow-moving drivers (seasonal "
        "cycles, economic strain, response capacity) over headline events."
    ),
]

_METRIC_DEFINITIONS = {
    "FATALITIES": (
        "conflict-related fatalities recorded in the calendar month "
        "(battle deaths, violence against civilians, explosions/remote "
        "violence — ACLED-style event counting)"
    ),
    "PA": (
        "people affected by the hazard in the calendar month (injured, "
        "displaced, evacuated, or otherwise requiring assistance, as "
        "reported by humanitarian sources)"
    ),
    "PHASE3PLUS_IN_NEED": (
        "population classified in IPC Phase 3 or worse (Crisis, Emergency, "
        "Famine) under the Current Situation assessment for the month"
    ),
}

SIBYL_STEP_PROMPT_TEMPLATE = """You are a superforecaster running a deep-research investigation to produce a probabilistic forecast. You reason like the best geopolitical forecasters: you start from the OUTSIDE VIEW (the historical base rate below), gather INSIDE-VIEW evidence from the open web, and explicitly reconcile the two at every step.

FORECAST AS-OF DATE: {as_of}. Treat this as "today". You must not use, cite, or rely on any information published after this date.{backtest_note}

=== QUESTION ===
{wording}

Country: {country} ({iso3}) | Hazard: {hazard_code} | Metric: {metric}
Metric definition: {metric_definition}
Forecast window (6 calendar months): {forecast_months}
You are forecasting the distribution of the MONTHLY value of this metric over the window months. Your quantiles must describe a single month drawn from this window — account for both month-to-month variation (seasonality, escalation) and your own uncertainty.

=== OUTSIDE VIEW (base-rate anchor — reason from it and away from it, never treat it as a target) ===
{base_rate_block}

=== YOUR TRIAL PERSPECTIVE ===
{perspective}

=== CURRENT BELIEF STATE (step {step} of {max_steps}) ===
{belief_json}

=== RESULT OF YOUR LAST ACTION ===
{last_tool_result}

=== YOUR TASK THIS STEP ===
Decide your next action and update your belief state.

Actions:
- "brave_search": run a web search. action_input = the query (natural language, include the country name; searches are date-filtered to the as-of date).
- "fetch_url": read a page found in earlier search results. action_input = the URL.
- "submit": finalize your forecast. Use this as soon as further research would not materially change your quantiles — do not burn steps for their own sake. You MUST submit by step {max_steps}.

Respond with ONLY a JSON object, no prose outside it:
{{
  "action": "brave_search" | "fetch_url" | "submit",
  "action_input": "<query or url; empty string for submit>",
  "belief_state": {{
    "quantiles": {{{quantile_keys}}},
    "confidence": "low" | "medium" | "high",
    "evidence_higher": ["evidence found so far that pushes the estimate HIGHER"],
    "evidence_lower": ["evidence found so far that pushes the estimate LOWER"],
    "open_questions": ["what you still need to find out"],
    "baserate_reconciliation": "how your current estimate relates to the outside-view anchor and why it departs (or does not)",
    "step_rationale": "what THIS step's information changed and why"
  }}
}}

Rules for quantiles:
- values are {metric} counts for one month, in raw units (people/fatalities), NOT thousands;
- non-decreasing across levels (q0.1 <= q0.25 <= ... <= q0.99);
- this data is right-skewed and heavy-tailed: keep q0.95/q0.99 honest — for this class of data q0.99 is typically several multiples of the median;
- 0 is a legitimate value (many country-months have zero impact);
- update the belief state EVERY step, even when the action is another search.
{parse_feedback}"""


@dataclass
class TrialStepRecord:
    step: int
    action: str
    action_input: str
    tool_ok: Optional[bool]
    belief: Dict[str, Any]
    repaired: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "action_input": self.action_input,
            "tool_ok": self.tool_ok,
            "belief": self.belief,
            "repaired": self.repaired,
        }


@dataclass
class TrialResult:
    trial_index: int
    perspective: str
    quantiles: Optional[Dict[float, float]]
    confidence: str
    belief_trace: List[TrialStepRecord] = field(default_factory=list)
    evidence_higher: List[str] = field(default_factory=list)
    evidence_lower: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)
    steps_used: int = 0
    submitted: bool = False
    cost: CostBreakdown = field(default_factory=CostBreakdown)
    leakage: LeakageStats = field(default_factory=LeakageStats)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.quantiles is not None and self.error is None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "perspective": self.perspective,
            "quantiles": (
                {str(k): v for k, v in sorted(self.quantiles.items())}
                if self.quantiles
                else None
            ),
            "confidence": self.confidence,
            "belief_trace": [s.to_dict() for s in self.belief_trace],
            "evidence_higher": list(self.evidence_higher),
            "evidence_lower": list(self.evidence_lower),
            "source_urls": list(self.source_urls),
            "steps_used": self.steps_used,
            "submitted": self.submitted,
            "cost": self.cost.to_dict(),
            "leakage": self.leakage.to_dict(),
            "error": self.error,
        }


def _quantile_keys_hint() -> str:
    return ", ".join(f'"{lv}": <number>' for lv in QUANTILE_LEVELS)


def build_step_prompt(
    question: SibylQuestion,
    base_rate: BaseRate,
    belief: BeliefState,
    *,
    step: int,
    as_of: date,
    perspective: str,
    forecast_months: List[str],
    last_tool_result: str,
    country_name: str,
    parse_feedback: str = "",
) -> str:
    backtest_note = ""
    if is_backtest(as_of):
        backtest_note = (
            " This is a retrospective evaluation: search results are "
            "date-capped, and any knowledge you have of events after this "
            "date must be ignored."
        )
    return SIBYL_STEP_PROMPT_TEMPLATE.format(
        as_of=as_of.isoformat(),
        backtest_note=backtest_note,
        wording=question.wording or "(no wording stored)",
        country=country_name,
        iso3=question.iso3,
        hazard_code=question.hazard_code,
        metric=question.metric,
        metric_definition=_METRIC_DEFINITIONS.get(
            question.metric, "monthly impact magnitude"
        ),
        forecast_months=", ".join(forecast_months),
        base_rate_block=base_rate.prompt_text,
        perspective=perspective,
        step=step,
        max_steps=MAX_STEPS,
        belief_json=json.dumps(belief.to_dict(), indent=2),
        last_tool_result=last_tool_result,
        quantile_keys=_quantile_keys_hint(),
        parse_feedback=parse_feedback,
    )


def _call_model(prompt: str) -> tuple[str, Dict[str, Any], str]:
    """One Opus call through the repo's provider layer.

    Returns (text, usage_with_cost, error). Cost is estimated from
    pythia/model_costs.json via the provider helpers.
    """
    from forecaster.providers import call_anthropic, estimate_cost_usd  # noqa: PLC0415

    result = call_anthropic(prompt, MODEL, 1.0, purpose="sibyl_step")
    usage = dict(result.usage or {})
    if not usage.get("cost_usd"):
        usage["cost_usd"] = estimate_cost_usd(MODEL, usage)
    return result.text or "", usage, result.error or ""


def _execute_tool(
    decision: StepDecision, as_of: date
) -> ToolResult:
    if decision.action == "brave_search":
        return brave_search(decision.action_input, as_of)
    if decision.action == "fetch_url":
        return fetch_url(decision.action_input, as_of)
    raise ValueError(f"not a tool action: {decision.action}")


def run_trial(
    question: SibylQuestion,
    base_rate: BaseRate,
    *,
    as_of: date,
    trial_index: int,
    run_id: str,
    tracker: CostTracker,
    forecast_months: List[str],
    country_name: str,
    model_call: Optional[Callable[[str], tuple[str, Dict[str, Any], str]]] = None,
) -> TrialResult:
    """Run one independent agentic trial for *question*.

    *model_call* is injectable for tests (deterministic smoke test); the
    default goes through ``forecaster.providers.call_anthropic``.
    """
    call = model_call or _call_model
    perspective = TRIAL_PERSPECTIVES[trial_index % len(TRIAL_PERSPECTIVES)]
    result = TrialResult(
        trial_index=trial_index,
        perspective=perspective,
        quantiles=None,
        confidence="low",
    )

    belief = initial_belief_from_anchor(base_rate.anchor_quantiles)
    last_tool_result = "(none yet — this is your first step)"
    seen_urls: set[str] = set()

    for step in range(1, MAX_STEPS + 1):
        decision: Optional[StepDecision] = None
        parse_feedback = ""
        for attempt in range(1, ANTHROPIC_MAX_ATTEMPTS + 1):
            prompt = build_step_prompt(
                question,
                base_rate,
                belief,
                step=step,
                as_of=as_of,
                perspective=perspective,
                forecast_months=forecast_months,
                last_tool_result=last_tool_result,
                country_name=country_name,
                parse_feedback=parse_feedback,
            )
            text, usage, error = call(prompt)
            cost = float(usage.get("cost_usd") or 0.0)
            result.cost.add(COST_KIND_OPUS, cost)
            tracker.add(question.question_id, COST_KIND_OPUS, cost)
            log_sibyl_call(
                run_id=run_id,
                question_id=question.question_id,
                prompt_text=prompt,
                response_text=text,
                provider="anthropic",
                model_id=MODEL,
                usage=usage,
                iso3=question.iso3,
                hazard_code=question.hazard_code,
                metric=question.metric,
                error_text=error,
                hs_run_id=question.hs_run_id,
                call_type=f"sibyl_trial{trial_index}_step{step}",
            )
            if error:
                parse_feedback = (
                    "\nNOTE: your previous response failed with a provider "
                    f"error ({error[:200]}). Respond again."
                )
                continue
            try:
                decision = parse_step_response(text)
                break
            except BeliefStateError as exc:
                logger.warning(
                    "sibyl.agent: parse failure q=%s trial=%d step=%d attempt=%d: %s",
                    question.question_id, trial_index, step, attempt, exc,
                )
                parse_feedback = (
                    "\nNOTE: your previous response was rejected "
                    f"({exc}). Output ONLY the JSON object, exactly in the "
                    "specified shape, with all required quantile levels."
                )

        if decision is None:
            result.error = "model_step_failed"
            result.steps_used = step
            break

        belief = decision.belief
        record = TrialStepRecord(
            step=step,
            action=decision.action,
            action_input=decision.action_input,
            tool_ok=None,
            belief=belief.to_dict(),
            repaired=decision.repaired,
        )
        result.belief_trace.append(record)
        result.steps_used = step

        if decision.action == "submit":
            result.submitted = True
            break

        tool_result = _execute_tool(decision, as_of)
        record.tool_ok = tool_result.ok
        result.cost.add(COST_KIND_BRAVE, tool_result.cost_usd)
        tracker.add(question.question_id, COST_KIND_BRAVE, tool_result.cost_usd)
        result.leakage.merge(tool_result.leakage)
        for src in tool_result.sources:
            if src.url and src.url not in seen_urls:
                seen_urls.add(src.url)
                result.source_urls.append(src.url)
        if tool_result.tool == "brave_search" and tool_result.cost_usd > 0:
            log_sibyl_call(
                run_id=run_id,
                question_id=question.question_id,
                prompt_text=decision.action_input,
                response_text=tool_result.text[:2000],
                provider="brave",
                model_id="brave-web-search",
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": tool_result.cost_usd,
                },
                iso3=question.iso3,
                hazard_code=question.hazard_code,
                metric=question.metric,
                error_text=tool_result.error or "",
                hs_run_id=question.hs_run_id,
                call_type=f"sibyl_trial{trial_index}_search",
            )
        last_tool_result = tool_result.text

    # A trial that ran out of steps without submitting still counts: the
    # belief state was updated every step, so the latest quantiles stand.
    if result.belief_trace and result.error is None:
        result.quantiles = dict(belief.quantiles)
        result.confidence = belief.confidence
        result.evidence_higher = list(belief.evidence_higher)
        result.evidence_lower = list(belief.evidence_lower)
    elif result.error is None:
        result.error = "no_valid_steps"
    return result
