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

Trial diversity: ``claude-opus-5-5`` rejects sampling parameters
(temperature returns HTTP 400), so the K trials are differentiated by
explicit perspective seeds in the prompt rather than temperature.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional

from sibyl.base_rates import BaseRate
from sibyl.belief_state import (
    BeliefState,
    BeliefStateError,
    StepDecision,
    MonthBelief,
    empty_plan,
    initial_belief,
    parse_step_response,
)
from sibyl import config as _cfg
from sibyl.config import (
    ANTHROPIC_MAX_ATTEMPTS,
    EFFORT,
    MAX_STEPS,
    MODEL,
)
from sibyl.cost import COST_KIND_BRAVE, COST_KIND_OPUS, CostBreakdown, CostTracker, log_sibyl_call
from sibyl.leakage import LeakageStats, is_backtest
from sibyl.select_questions import SibylQuestion
from sibyl import extract as sibyl_extract
from sibyl.cost import COST_KIND_EXTRACTION
from sibyl.tools import ToolResult, brave_search, fetch_url, reliefweb_search

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

_HEAD = """You are a superforecaster running a deep-research investigation to produce a probabilistic forecast. You start from the REFERENCE below, a mechanical forecast built from the resolving source's own history, gather evidence from the open web, and adjust the reference only as far as that evidence justifies.

FORECAST AS-OF DATE: {as_of}. Treat this as "today". You must not use, cite, or rely on any information published after this date.{backtest_note}"""

_QUESTION = """=== QUESTION ===
{wording}

Country: {country} ({iso3}) | Hazard: {hazard_code} | Metric: {metric}
Metric definition: {metric_definition}
Forecast window (6 calendar months): {forecast_months}
You forecast two months of this window: MONTH 1 ({month_1}) and MONTH 6 ({month_6}). Months 2 to 5 are taken as mixtures of the two.

=== HOW THIS RESOLVES ===
{resolver_card}

=== REFERENCE (your prior) ===
{base_rate_block}{track_record_block}"""

_TASK = """=== YOUR TASK EACH STEP ===
Decide your next actions and update your belief state.

Tools (up to {max_actions} calls in one step, listed in "actions"):
- "brave_search": web search. action_input = {{"query": "...", "lane": "news" | "reference", "language": "<optional ISO 639-1 code, e.g. fr, ar, es>", "country": "<optional 2-letter country code>"}}. The news lane covers the last four months; the reference lane covers ten years, for past episodes, seasonal patterns and structural reports. Both end at the as-of date. Search in the country's own languages as well as English.
- "reliefweb_search": search ReliefWeb's situation reports, appeals and assessments (UN, NGO, government). action_input = {{"query": "..."}}.
- "fetch_url": read a document (web page or PDF) found in results. action_input = {{"url": "...", "extraction_request": "the figures or facts you want from it"}}. A long document is read for you by a second model that returns what you asked for, figures word for word; say exactly what you need.
- "submit": finalize your forecast, as a step of its own. It is accepted only once the "resolver" slot of your plan is done (or has failed twice), you have read at least {submit_min_docs} documents, and the "disconfirm" slot is done. You MUST submit by step {max_steps}.

Your research plan has six slots. Work through all of them; report each one's status and a one-line finding every step:
- "resolver": how the question resolves, and the resolving source's latest published figures for this country.
- "nowcast": an estimate for the months between the last month in the reference table and today.
- "drivers": what is driving the level now.
- "calendar": dated events inside the window (elections, ceasefire expiries, mission withdrawals, IPC analysis dates, the seasonal climate outlook).
- "reversion": the case that the series returns to its 12-month norm.
- "disconfirm": one search aimed at evidence against your current median.

Respond with ONLY a JSON object, no prose outside it:
{{
  "actions": [{{"action": "brave_search" | "reliefweb_search" | "fetch_url" | "submit", "action_input": {{...}}}}],
  "belief_state": {{
    "month_1": {{"p_zero": <probability>, "quantiles_positive": {{{quantile_keys}}}}},
    "month_6": {{"p_zero": <probability>, "quantiles_positive": {{{quantile_keys}}}}},
    "plan": {{"resolver": {{"status": "pending" | "done" | "failed", "finding": "one line"}}, "nowcast": {{...}}, "drivers": {{...}}, "calendar": {{...}}, "reversion": {{...}}, "disconfirm": {{...}}}},
    "confidence": "low" | "medium" | "high",
    "evidence_higher": ["evidence found so far that pushes the estimate HIGHER"],
    "evidence_lower": ["evidence found so far that pushes the estimate LOWER"],
    "open_questions": ["what you still need to find out"],
    "baserate_reconciliation": "how your current forecast relates to the reference and why it departs (or does not)",
    "step_rationale": "what THIS step's information changed and why"
  }}
}}

Rules for the belief state:
- p_zero is the probability that the resolving source records ZERO for that month{zero_note};
- quantiles_positive are the 0.05, 0.25, 0.5, 0.75 and 0.95 quantiles of the month's {metric} count GIVEN that it is positive: raw units (people/fatalities, not thousands), each at least 1, non-decreasing;
- update the belief state EVERY step, even when the action is another search.

Rules for weighing evidence:
- The reference is your prior. Keep it unless dated, specific evidence says otherwise.
- A rise in the news is often already in the latest months of the table. Check before you add it again.
- Count one event once, however many articles report it.
- Heavy coverage is no evidence of escalation. Thin coverage is no evidence of calm.
- For month 6, move away from the reference only on evidence that the change will last: a dated event, a seasonal cause, or a structural shift. Most spikes fade.
- Forecast what the resolving source will record."""

SIBYL_STEP_PROMPT_TEMPLATE = (
    _HEAD + "\n\n" + _QUESTION + """

=== YOUR TRIAL PERSPECTIVE ===
{perspective}

=== CURRENT BELIEF STATE (step {step} of {max_steps}) ===
{belief_json}

=== RESULT OF YOUR LAST ACTION ===
{last_tool_result}

""" + _TASK + "\n{parse_feedback}"
)


# --- V3 (static-first) segment templates -----------------------------------
#
# Same content as SIBYL_STEP_PROMPT_TEMPLATE, reordered so the stable spans
# lead and the per-step churn trails:
#   B1 run-static (head + as-of + task/action rules + JSON schema)  — stable
#      across every step/trial/question of a run (modulo {metric})
#   B2 per-question (question block + reference)                    — stable
#      across all K trials x MAX_STEPS steps of a question  → cache breakpoint
#   B3 per-trial (perspective seed)                                 — stable
#      across the trial's steps                             → cache breakpoint
#   B4 per-step (belief state, last tool result, parse feedback)    — churns
# Active only when both PYTHIA_PROMPT_V3_ORDER and PYTHIA_PROMPT_CACHE_ENABLED
# are on.

SIBYL_STEP_RUN_STATIC_V3 = _HEAD + "\n\n" + _TASK + "\n"

SIBYL_STEP_QUESTION_V3 = "\n" + _QUESTION + "\n"

SIBYL_STEP_TRIAL_V3 = """
=== YOUR TRIAL PERSPECTIVE ===
{perspective}
"""

SIBYL_STEP_STEP_V3 = """
=== CURRENT BELIEF STATE (step {step} of {max_steps}) ===
{belief_json}

=== RESULT OF YOUR LAST ACTION ===
{last_tool_result}
{parse_feedback}
Now decide your next action per "YOUR TASK EACH STEP" above and respond with ONLY the JSON object."""


# Sibyl's own calibration feedback (sibyl/advice.py -> sibyl.calibration.
# load_advice). Rendered right after the outside view and, under V3 order,
# inside the per-question segment: it is constant across a question's trials
# and steps, so the cache breakpoints hold. Absent entirely when there is no
# advice for the class or the question is in the no-advice arm; an empty
# block leaves the prompt byte-identical to before.
TRACK_RECORD_HEADING = "=== YOUR TRACK RECORD ==="


def render_track_record(advice_text: Optional[str]) -> str:
    """The track-record block, or '' when there is nothing to show."""
    text = (advice_text or "").strip()
    if not text:
        return ""
    return (
        "\n\n" + TRACK_RECORD_HEADING + "\n"
        "Feedback on how your past forecasts of this class of question compared "
        "with what then happened. Weigh it alongside the evidence you find; it is "
        "a tendency to correct, never a target for this question.\n"
        + text
    )


@dataclass
class TrialStepRecord:
    step: int
    action: str
    action_input: str
    tool_ok: Optional[bool]
    belief: Dict[str, Any]
    repaired: bool
    # Every call of the step (Oct 2026: up to MAX_ACTIONS_PER_STEP); the
    # action / action_input / tool_ok fields above are the first call's.
    calls: List[Dict[str, Any]] = field(default_factory=list)
    # A submit the research gate refused: what was missing.
    gate_rejected: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "action_input": self.action_input,
            "tool_ok": self.tool_ok,
            "belief": self.belief,
            "repaired": self.repaired,
            "calls": list(self.calls),
            "gate_rejected": self.gate_rejected,
        }


def submit_gate_missing(
    plan: Dict[str, Dict[str, str]], docs_read: int, resolver_failed_steps: int
) -> List[str]:
    """What a submit still lacks; empty when it may stand.

    The resolver slot must be done (or have been reported failed on two
    steps), SUBMIT_MIN_DOCS documents read, and the disconfirm slot done.
    """
    missing: List[str] = []
    resolver = (plan.get("resolver") or {}).get("status")
    if resolver != "done" and resolver_failed_steps < 2:
        missing.append(
            "the 'resolver' slot: find how this resolves and the resolving source's "
            "latest figures for this country"
        )
    if docs_read < _cfg.SUBMIT_MIN_DOCS:
        missing.append(
            f"documents read: {docs_read} of {_cfg.SUBMIT_MIN_DOCS} (use fetch_url)"
        )
    if (plan.get("disconfirm") or {}).get("status") != "done":
        missing.append(
            "the 'disconfirm' slot: one search aimed at evidence against your median"
        )
    return missing


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
    # The two elicited horizons (Oct 2026); ``quantiles`` above is the legacy
    # month-1 view at the seven old levels.
    month_beliefs: Dict[int, MonthBelief] = field(default_factory=dict)
    # Searches that returned at least one source, and documents read.
    n_search_ok: int = 0
    n_docs_read: int = 0
    # Set when the trial kept its last valid belief after a later step failed
    # on every attempt (e.g. "model_step_failed"); the trial still counts.
    degraded: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.quantiles is not None and self.error is None

    @property
    def evidence_ok(self) -> bool:
        """Did the trial see anything? Thresholds read at call time."""
        return (
            self.n_search_ok >= _cfg.MIN_SEARCH_OK
            and self.n_docs_read >= _cfg.MIN_DOCS_READ
        )

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
            "month_1": (self.month_beliefs[1].to_dict() if 1 in self.month_beliefs else None),
            "month_6": (self.month_beliefs[6].to_dict() if 6 in self.month_beliefs else None),
            "n_search_ok": self.n_search_ok,
            "n_docs_read": self.n_docs_read,
            "evidence_ok": self.evidence_ok,
            "degraded": self.degraded,
        }


def _quantile_keys_hint() -> str:
    from sibyl.belief_state import POS_LEVELS  # noqa: PLC0415

    return ", ".join(f'"{lv}": <number>' for lv in POS_LEVELS)


_CARD_DIR = Path(__file__).resolve().parent / "resolver_cards"


def resolver_card(hazard_code: str, metric: str) -> str:
    """The class's resolver card (sibyl/resolver_cards/), or a plain line.

    Update the cards whenever resolution changes (CLAUDE.md says so).
    """
    path = _CARD_DIR / f"{str(hazard_code).upper()}_{str(metric).upper()}.md"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "The resolving source is named in the question above."


def _zero_note(question: SibylQuestion) -> str:
    if str(question.hazard_code).upper() in ("FL", "TC") and str(question.metric).upper() == "PA":
        return (
            " OR has no record at all (a month with no IFRC GO or IDMC record "
            "does not resolve, so treat 'no record' like zero here)"
        )
    return ""


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
    return_segments: bool = False,
    track_record: str = "",
):
    """Build the step prompt (legacy or V3 section order).

    With ``return_segments=True`` returns an ordered list of
    ``(text, is_cache_breakpoint)`` tuples whose concatenation is the prompt;
    under legacy order that's a single unmarked segment (nothing usefully
    cacheable), under V3 order the per-question and per-trial spans carry
    breakpoints for Anthropic cache_control.
    """
    from forecaster.prompts import _prompt_v3_order_enabled  # noqa: PLC0415

    backtest_note = ""
    if is_backtest(as_of):
        backtest_note = (
            " This is a retrospective evaluation: search results are "
            "date-capped, and any knowledge you have of events after this "
            "date must be ignored."
        )
    common = dict(
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
        month_1=(forecast_months[0] if forecast_months else "month 1"),
        month_6=(forecast_months[min(5, len(forecast_months) - 1)] if forecast_months else "month 6"),
        zero_note=_zero_note(question),
        base_rate_block=base_rate.prompt_text,
        track_record_block=render_track_record(track_record),
        perspective=perspective,
        step=step,
        max_steps=MAX_STEPS,
        belief_json=json.dumps(belief.to_dict(), indent=2),
        last_tool_result=last_tool_result,
        quantile_keys=_quantile_keys_hint(),
        max_actions=_cfg.MAX_ACTIONS_PER_STEP,
        submit_min_docs=_cfg.SUBMIT_MIN_DOCS,
        resolver_card=resolver_card(question.hazard_code, question.metric),
        parse_feedback=parse_feedback,
    )

    if not _prompt_v3_order_enabled():
        prompt = SIBYL_STEP_PROMPT_TEMPLATE.format(**common)
        if return_segments:
            return [(prompt, False)]
        return prompt

    segments = [
        (SIBYL_STEP_RUN_STATIC_V3.format(**common), False),
        (SIBYL_STEP_QUESTION_V3.format(**common), True),   # cache breakpoint 1
        (SIBYL_STEP_TRIAL_V3.format(**common), True),      # cache breakpoint 2
        (SIBYL_STEP_STEP_V3.format(**common), False),
    ]
    if return_segments:
        return segments
    return "".join(text for text, _ in segments)


def _call_model(
    prompt: str, *, cache_segments: Optional[List[tuple]] = None
) -> tuple[str, Dict[str, Any], str]:
    """One Opus call through the repo's provider layer.

    Returns (text, usage_with_cost, error). Cost is estimated from
    pythia/model_costs.json via the provider helpers (cache-aware since the
    Phase-0 telemetry work — cache reads bill at the cached rate, which also
    keeps the Sibyl budget cap honest when caching is on).
    """
    from forecaster.providers import call_anthropic, estimate_cost_usd  # noqa: PLC0415

    result = call_anthropic(
        prompt,
        MODEL,
        1.0,
        purpose="sibyl_step",
        cache_segments=cache_segments,
        thinking_level=EFFORT if EFFORT not in ("", "off", "none") else None,
    )
    usage = dict(result.usage or {})
    if not usage.get("cost_usd"):
        usage["cost_usd"] = estimate_cost_usd(MODEL, usage)
    return result.text or "", usage, result.error or ""


def _search_terms(question: SibylQuestion, country_name: str) -> List[str]:
    """Words a PDF page must carry to be worth reading for this question."""
    by_metric = {
        "FATALITIES": ["killed", "fatalities", "deaths", "dead"],
        "PA": ["affected", "displaced", "people", "households"],
        "PHASE3PLUS_IN_NEED": ["IPC", "Phase 3", "food insecurity", "crisis"],
    }
    by_hazard = {"FL": ["flood"], "TC": ["cyclone", "storm", "typhoon", "hurricane"],
                 "DR": ["drought"], "ACE": ["conflict", "clashes", "attack"]}
    terms = [country_name, question.iso3]
    terms += by_metric.get(str(question.metric).upper(), [])
    terms += by_hazard.get(str(question.hazard_code).upper(), [])
    return [t for t in terms if t]


def _execute_tool(call, as_of: date, *, question: SibylQuestion, terms: List[str]) -> ToolResult:
    opts = getattr(call, "options", {}) or {}
    if call.action == "brave_search":
        return brave_search(
            call.action_input, as_of,
            lane=str(opts.get("lane") or "news"),
            language=(str(opts["language"]) if opts.get("language") else None),
            country=(str(opts["country"]) if opts.get("country") else None),
        )
    if call.action == "reliefweb_search":
        return reliefweb_search(call.action_input, as_of, country_iso3=question.iso3)
    if call.action == "fetch_url":
        return fetch_url(call.action_input, as_of, terms=terms)
    raise ValueError(f"not a tool action: {call.action}")


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
    track_record: str = "",
    extraction_call: Optional[Callable[[str, str], tuple[str, Dict[str, Any], str]]] = None,
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

    # The reference (sibyl/reference.py) seeds the belief; an object without
    # reference vectors (no history) seeds a labelled placeholder.
    belief = initial_belief(getattr(base_rate, "by_month", None), question.metric)
    belief.plan = empty_plan()
    last_tool_result = "(none yet — this is your first step)"
    seen_urls: set[str] = set()
    resolver_failed_steps = 0
    terms = _search_terms(question, country_name)

    for step in range(1, MAX_STEPS + 1):
        decision: Optional[StepDecision] = None
        parse_feedback = ""
        for attempt in range(1, ANTHROPIC_MAX_ATTEMPTS + 1):
            segments = build_step_prompt(
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
                return_segments=True,
                track_record=track_record,
            )
            prompt = "".join(text for text, _ in segments)
            # The injectable test seam takes a plain prompt string; the
            # default path passes the segments so cache_control can apply.
            if model_call is not None:
                text, usage, error = call(prompt)
            else:
                text, usage, error = _call_model(prompt, cache_segments=segments)
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
            result.steps_used = step
            if result.belief_trace:
                # Salvage: the belief from the last valid step stands. The
                # step that failed added nothing, and discarding a trial that
                # had already read the web wastes what it learned.
                result.degraded = "model_step_failed"
                logger.warning(
                    "sibyl.agent: q=%s trial=%d step %d failed on every attempt; "
                    "keeping the belief from step %d",
                    question.question_id, trial_index, step, step - 1,
                )
            else:
                result.error = "model_step_failed"
            break

        if not decision.plan_given:
            decision.belief.plan = dict(belief.plan) or empty_plan()
        belief = decision.belief
        if (belief.plan.get("resolver") or {}).get("status") == "failed":
            resolver_failed_steps += 1
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

        if not decision.calls:
            missing = submit_gate_missing(belief.plan, result.n_docs_read, resolver_failed_steps)
            if missing and step < MAX_STEPS:
                record.gate_rejected = missing
                last_tool_result = (
                    "Your submit was NOT accepted. Still missing:\n"
                    + "\n".join(f"- {m}" for m in missing)
                )
                continue
            result.submitted = True
            break

        outputs: List[str] = []
        for i, tcall in enumerate(decision.calls):
            tool_result = _execute_tool(tcall, as_of, question=question, terms=terms)
            if tool_result.tool == "fetch_url" and tool_result.ok and tool_result.doc_text:
                req = str((tcall.options or {}).get("extraction_request") or "")

                def _log_extract(prompt, response, usage, model_id, error):
                    log_sibyl_call(
                        run_id=run_id, question_id=question.question_id,
                        prompt_text=prompt, response_text=response,
                        provider="anthropic", model_id=model_id, usage=usage,
                        iso3=question.iso3, hazard_code=question.hazard_code,
                        metric=question.metric, error_text=error or "",
                        hs_run_id=question.hs_run_id,
                        call_type=f"sibyl_trial{trial_index}_extract",
                    )

                ex = sibyl_extract.extract(
                    tool_result.doc_text, req, url=tcall.action_input,
                    question=question.wording or "", country=country_name,
                    call=extraction_call, log=_log_extract,
                )
                result.cost.add(COST_KIND_EXTRACTION, ex.cost_usd)
                tracker.add(question.question_id, COST_KIND_EXTRACTION, ex.cost_usd)
                note = (
                    f"(read by the extraction model from {len(tool_result.doc_text):,} characters)"
                    if ex.extracted else
                    ("" if len(tool_result.doc_text) < _cfg.EXTRACTION_SKIP_CHARS
                     else f"(extraction unavailable; first {_cfg.EXTRACTION_SKIP_CHARS:,} characters)")
                )
                tool_result.text = f"Content of {tcall.action_input} {note}:\n{ex.text}"
            call_ok = tool_result.ok
            if i == 0:
                record.tool_ok = call_ok
            record.calls.append({
                "action": tcall.action, "action_input": tcall.action_input,
                "options": {k: v for k, v in (tcall.options or {}).items()},
                "tool_ok": call_ok,
            })
            if call_ok and tool_result.tool in ("brave_search", "reliefweb_search") and tool_result.sources:
                result.n_search_ok += 1
            if call_ok and tool_result.tool == "fetch_url":
                result.n_docs_read += 1
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
                    prompt_text=tcall.action_input,
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
            outputs.append(f"--- {tcall.action}: {tcall.action_input}\n{tool_result.text}")
        if decision.submit_dropped:
            outputs.append("(Your submit was ignored: submit must be a step of its own.)")
        last_tool_result = "\n\n".join(outputs)

    # A trial that ran out of steps without submitting still counts: the
    # belief state was updated every step, so the latest quantiles stand.
    if result.belief_trace and result.error is None:
        result.quantiles = dict(belief.quantiles)
        result.month_beliefs = {1: belief.month_1, 6: belief.month_6}
        result.confidence = belief.confidence
        result.evidence_higher = list(belief.evidence_higher)
        result.evidence_lower = list(belief.evidence_lower)
    elif result.error is None:
        result.error = "no_valid_steps"
    return result
