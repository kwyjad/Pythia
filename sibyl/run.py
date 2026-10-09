# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl run orchestrator.

Per run: select N affected/fatalities questions (sibyl/select_questions.py:
floor-then-fill plus controls), and for each, in run order (floor picks,
then controls, then fill picks, so a cap removes fill picks first):

1. build Sibyl's reference (sibyl/reference.py),
2. run K independent agentic trials on lanes A, B, C in parallel threads
   (a control runs one, on lane A); when they disagree or their pool
   departs far from the reference, run up to K_MAX - K more on lanes D and E
   (sibyl/trials.py),
3. leave out an outlier trial, then linear-pool the trial CDFs,
4. calibrate (identity hook while CALIBRATION_ENABLED is off),
5. serialize to the native SPD format beside the standard track,
6. record cost, and
7. compute the JS divergence vs the standard-Pythia SPD.

The hard budget cap is checked at every question boundary and before each
batch of trials (trials already running are never cut); once reached, no new work starts, completed work is persisted,
remaining questions are marked ``skipped: run budget cap``, and the
run-level ``budget_capped`` flag is set. The wall-clock cap
(``SIBYL_MAX_RUNTIME_MIN``) behaves the same way at question boundaries:
the question in flight finishes, the rest are ``skipped: run time cap``,
and ``time_capped`` is set.

Usage: ``python -m sibyl.run [--hs-run-id RUN] [--n N]``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from pythia.db.schema import connect, ensure_schema

from sibyl import config as sibyl_config
import sibyl.agent as _agent
from sibyl.agent import LANE_IDS, TrialResult, lane_for_trial, run_trial
from sibyl.aggregate import dist_from_vector, pool_months, publish_vectors
from sibyl.belief_state import MonthBelief, legacy_quantiles
from sibyl.reference import NO_REFERENCE_TEXT, build_reference
from sibyl.advice import advice_arm
from sibyl.calibration import load_advice
from sibyl.config import (
    ADVICE_EXPERIMENT_SHARE,
    AGGREGATION,
    BACKTEST_MODE,
    K,
    MAX_PER_HAZARD,
    MAX_RUNTIME_MIN,
    MAX_STEPS,
    MIN_PER_HAZARD,
    MODEL,
    N_QUESTIONS,
    RUN_HARD_CAP_USD,
)
from sibyl import config as _cfg
from sibyl import tools as sibyl_tools
from sibyl.cost import COST_KIND_SHADOW, CostTracker
from sibyl.evidence import backfill_evidence_ok
from sibyl.leakage import LeakageStats
from sibyl.measure import process_measures, reference_weight, write_evidence
from sibyl.postmortem import lessons_block_for
from sibyl.shadow import ShadowContext, run_shadow_phase, shadow_setup
from sibyl.trials import extra_trials_rule, month_median, outlier_indices, run_trial_batch
from sibyl.select_questions import (
    SibylQuestion,
    hs_run_is_test,
    latest_hs_run_id,
    select_top_questions,
)
from sibyl.spd import (
    apply_bucket_floor,
    find_standard_run_id,
    inter_trial_divergence_vectors,
    load_standard_spd_by_month,
    persist_sibyl_forecast,
    persist_sibyl_run,
    track_divergence,
    write_native_spd,
)

logger = logging.getLogger(__name__)

SKIP_REASON_BUDGET = "run budget cap"
SKIP_REASON_TIME = "run time cap"
SKIP_REASON_NO_EVIDENCE = "no evidence"


def _runtime_cap_reached(started: float, limit_min: float, now: Optional[float] = None) -> bool:
    """True once the trial loop has run *limit_min* minutes (0 = no limit)."""
    if not limit_min or limit_min <= 0:
        return False
    elapsed = (time.monotonic() if now is None else now) - started
    return elapsed >= limit_min * 60.0


@dataclass
class _NoReference:
    """Stands in for a Reference when a question has no history."""

    prompt_text: str = NO_REFERENCE_TEXT
    by_month: Optional[Dict[int, List[float]]] = None


@dataclass
class QuestionOutcome:
    question: SibylQuestion
    status: str  # ok | skipped | failed
    skip_reason: Optional[str] = None
    trials: List[TrialResult] = field(default_factory=list)
    js_vs_standard: Optional[float] = None
    js_inter_trial: Optional[float] = None
    # The extra-trial rule that fired ('disagreement' / 'departure'), and the
    # measures behind the extra-trial and outlier decisions.
    extra_trials_rule: Optional[str] = None
    trial_checks: Dict[str, Any] = field(default_factory=dict)
    # For the process measures (sibyl/measure.py): the written vectors by
    # month, and the raw pool's and the reference's month-1 vectors.
    final_by_month: Dict[int, List[float]] = field(default_factory=dict)
    raw_month1: Optional[List[float]] = None
    reference_month1: Optional[List[float]] = None
    # What the shadow arm needs to run this question's shadow trial after the
    # production work of the whole run (sibyl/shadow.py). None for a control
    # and for a question that produced no forecast.
    shadow_ctx: Optional[ShadowContext] = None


def _write_log(**kw: Any) -> None:
    """Write one buffered llm_calls row, on the main thread.

    Looked up on the agent module at call time, so a test that replaces
    ``sibyl.agent.log_sibyl_call`` sees these rows too.
    """
    _agent.log_sibyl_call(**kw)


def resolve_as_of(question: SibylQuestion) -> date:
    """asOf resolution: live -> today; backtest -> the question's anchor.

    In backtest mode the as-of date is the question's ``window_start_date``
    (the first forecast-window month) — the moment the forecast would have
    been made.
    """
    if BACKTEST_MODE and question.window_start_date:
        ws = question.window_start_date
        if hasattr(ws, "date"):
            ws = ws.date()
        return ws
    return date.today()


def _forecast_month_keys(question: SibylQuestion) -> List[str]:
    from forecaster.month_utils import (  # noqa: PLC0415
        _anchor_month_for_question,
        _expected_months,
    )

    anchor = _anchor_month_for_question(question.to_row_dict())
    return _expected_months(anchor) if anchor else []


def _country_name(iso3: str) -> str:
    try:
        from forecaster.history_loaders import _load_country_names  # noqa: PLC0415

        return _load_country_names().get(iso3.upper(), iso3.upper())
    except Exception:
        return iso3.upper()


def _human_explanation(question: SibylQuestion, trials: List[TrialResult]) -> str:
    ok = [t for t in trials if t.ok]
    higher = [e for t in ok for e in t.evidence_higher][:4]
    lower = [e for t in ok for e in t.evidence_lower][:4]
    parts = [
        f"Sibyl deep-research forecast from {len(ok)} independent agentic "
        f"trial(s) over open-web reporting (model {MODEL}).",
    ]
    if higher:
        parts.append("Evidence pushing higher: " + "; ".join(higher))
    if lower:
        parts.append("Evidence pushing lower: " + "; ".join(lower))
    return " ".join(parts)[:2000]


def process_question(
    con: Any,
    question: SibylQuestion,
    *,
    sibyl_run_id: str,
    tracker: CostTracker,
    model_call: Any = None,
    reference_weight: Optional[float] = None,
) -> QuestionOutcome:
    """Forecast one question end-to-end. Returns the outcome (never raises).

    *reference_weight* is the reference's share of the published pool
    (``sibyl.measure.reference_weight``, read once per run); None means
    ``SIBYL_REFERENCE_WEIGHT``.
    """
    outcome = QuestionOutcome(question=question, status="failed")
    as_of = resolve_as_of(question)
    forecast_keys = _forecast_month_keys(question)
    if not forecast_keys:
        outcome.skip_reason = "no forecast window (missing window_start_date/target_month)"
        logger.error(
            "sibyl.run: question %s has no resolvable forecast window; skipping",
            question.question_id,
        )
        return outcome

    country = _country_name(question.iso3)
    # Sibyl's own prior (sibyl/reference.py): one bucket vector per window
    # month, the block the prompt shows, and the seed of every trial's belief.
    reference = build_reference(con, question, forecast_keys, as_of, known_at=as_of)
    base_rate = reference or _NoReference()

    # Sibyl's own track record for this class (sibyl/advice.py). Loaded once
    # per question: the text is constant across its trials and steps. No
    # advice for the class -> no arm and no section; otherwise the question's
    # arm (hash of "sibyl:" + question_id) decides whether the prompt shows it.
    advice = load_advice(question.hazard_code, question.metric, as_of, con=con)
    # Lessons and notes on similar past questions (sibyl/postmortem.py) ride
    # with the advice: shown only in the track-record arm, never in backtest.
    lessons_block = lessons_block_for(con, question, as_of)
    arm: Optional[str] = None
    track_record = ""
    lessons = ""
    if advice is not None or lessons_block:
        arm = advice_arm(question.question_id, ADVICE_EXPERIMENT_SHARE)
        if arm == "advice":
            track_record = advice.text if advice is not None else ""
            lessons = lessons_block

    is_control = question.is_control

    def _run_one(trial_index: int, lane: str, sink: List[Dict[str, Any]]) -> TrialResult:
        return run_trial(
            question,
            base_rate,
            as_of=as_of,
            trial_index=trial_index,
            run_id=sibyl_run_id,
            tracker=tracker,
            forecast_months=forecast_keys,
            country_name=country,
            model_call=model_call,
            track_record=track_record,
            lane=lane,
            lessons=lessons,
            log_sink=sink,
        )

    def _batch(jobs, role: str) -> None:
        if not jobs:
            return
        if tracker.hard_cap_reached() or tracker.question_cap_reached(question.question_id):
            logger.warning(
                "sibyl.run: budget reached before %s trials of %s; %d not started",
                role, question.question_id, len(jobs),
            )
            return
        results = run_trial_batch(jobs, _run_one, write_log=_write_log)
        for trial in results:
            if trial is None:
                continue
            trial.role = role
            outcome.trials.append(trial)

    # Production trials: K on lanes A, B, C (a control: one, on lane A).
    n_production = 1 if is_control else K
    _batch([(i, "A" if is_control else lane_for_trial(i)) for i in range(n_production)], "production")

    def _valid() -> List[TrialResult]:
        return [t for t in outcome.trials if t.ok and t.evidence_ok]

    # Extra trials (lanes D, E) when the production trials disagree or their
    # pool departs far from the reference. Never for a control.
    metric = question.metric
    if not is_control and _cfg.K_MAX > n_production and len(_valid()) >= 2:
        from sibyl.aggregate import month_vector  # noqa: PLC0415

        valid = _valid()
        try:
            vecs = [month_vector(t.month_beliefs[1].dist(), metric) for t in valid]
            pooled1 = pool_months(
                [{1: t.month_beliefs[1].dist(), 6: t.month_beliefs[6].dist()} for t in valid],
                metric,
            ).vectors[1]
            ref1 = reference.by_month.get(1) if reference and reference.by_month else None
            rule, measures = extra_trials_rule(vecs, pooled1, ref1)
        except (ValueError, KeyError) as exc:
            rule, measures = None, {"error": str(exc)}
        outcome.trial_checks["extra_trials_measures"] = measures
        if rule:
            extra = [
                (n_production + j, LANE_IDS[(n_production + j) % len(LANE_IDS)])
                for j in range(_cfg.K_MAX - n_production)
            ]
            logger.info(
                "sibyl.run: %s calls for %d extra trial(s) (%s, %s)",
                question.question_id, len(extra), rule, measures,
            )
            outcome.extra_trials_rule = rule
            _batch(extra, rule)
    outcome.trial_checks["n_trials_run"] = len(outcome.trials)

    # Outlier guard: a trial whose month-1 median sits more than
    # OUTLIER_LOG10 orders of magnitude from the others' is left out of the
    # pool while two remain; it stays in trials_json, marked.
    valid = _valid()
    if len(valid) >= 3:
        medians = [
            month_median(t.month_beliefs[1].p_zero, t.month_beliefs[1].quantiles_positive)
            for t in valid
        ]
        dropped = outlier_indices(medians)
        for i in dropped:
            valid[i].outlier_dropped = True
        outcome.trial_checks["month1_medians"] = medians
        outcome.trial_checks["outliers_dropped"] = [valid[i].trial_index for i in dropped]

    # Evidence gate: pool only trials that finished AND saw something. A
    # question short of the required number of such trials is stored failed
    # with its trials kept, and nothing reaches the forecast tables.
    finished = [t for t in outcome.trials if t.ok]
    ok_trials = [t for t in finished if t.evidence_ok and not t.outlier_dropped]
    required = 1 if is_control else max(1, min(_cfg.MIN_VALID_TRIALS, K))
    if len([t for t in finished if t.evidence_ok]) < required:
        if not finished:
            outcome.skip_reason = "no successful trials"
        else:
            outcome.skip_reason = SKIP_REASON_NO_EVIDENCE
            logger.warning(
                "sibyl.run: %s has %d trial(s) with evidence of %d finished "
                "(need %d); stored as failed, nothing written",
                question.question_id, len(ok_trials), len(finished), required,
            )
        return outcome

    from pythia.buckets import n_buckets_for  # noqa: PLC0415

    try:
        trial_months = [
            {1: t.month_beliefs[1].dist(), 6: t.month_beliefs[6].dist()} for t in ok_trials
        ]
        pool = pool_months(trial_months, metric)
        ref_vectors = reference.by_month if reference else None
        weight = float(_cfg.REFERENCE_WEIGHT if reference_weight is None else reference_weight)
        final = {
            m: apply_bucket_floor(v)
            for m, v in publish_vectors(pool.vectors, ref_vectors, weight).items()
        }
        trial_vectors = []
        for tm in trial_months:
            from sibyl.aggregate import month_vector  # noqa: PLC0415

            trial_vectors.append(month_vector(tm[1], metric))
    except (ValueError, KeyError) as exc:
        outcome.skip_reason = f"aggregation failed: {exc}"
        logger.error("sibyl.run: aggregation failed for %s: %s", question.question_id, exc)
        return outcome

    standard_run_id = find_standard_run_id(con, question.question_id, question.hs_run_id)
    forecast_run_id = standard_run_id or sibyl_run_id
    standard = (
        load_standard_spd_by_month(
            con, standard_run_id, question.question_id, n_buckets_for(metric)
        )
        if standard_run_id
        else None
    )
    outcome.js_vs_standard = track_divergence(final, standard)
    outcome.final_by_month = {int(m): list(v) for m, v in final.items()}
    outcome.raw_month1 = list(pool.vectors.get(1) or []) or None
    outcome.reference_month1 = (
        list(reference.by_month.get(1) or []) or None
        if reference is not None and reference.by_month else None
    )
    outcome.js_inter_trial = inter_trial_divergence_vectors(trial_vectors)

    # Legacy views for older readers: the raw month-1 quantiles at the old
    # seven levels, and the final month-1 vector.
    raw_m1_legacy = {
        str(lv): float(pool.quantiles[1][lv]) for lv in sibyl_config.QUANTILE_LEVELS
    }
    reference_record = None
    base_rate_record = None
    if reference is not None:
        reference_record = dict(reference.to_dict(), weight=weight)
        ref_m1 = dist_from_vector(reference.by_month[1], metric)
        anchor = legacy_quantiles(MonthBelief(ref_m1.p_zero, ref_m1.qpos))
        base_rate_record = {
            "summary": {"type": "sibyl_reference", "source": reference.source},
            "prompt_text": reference.prompt_text,
            "anchor_quantiles": {str(k): v for k, v in sorted(anchor.items())},
            "framing_notes": [],
        }

    qcost = tracker.question_breakdown(question.question_id)
    spd_payload = {
        "track": "sibyl",
        "as_of": as_of.isoformat(),
        "k": len(ok_trials),
        "k_requested": len(outcome.trials),
        "aggregation": "linear_pool_by_month",
        "model": MODEL,
        "pooled_quantiles": raw_m1_legacy,
        "trial_quantiles": [
            {str(k): v for k, v in sorted(t.quantiles.items())} for t in ok_trials
        ],
        "forecast_months": forecast_keys,
        "reference_source": reference.source if reference else None,
        "reference_weight": weight if reference else 0.0,
        "js_divergence_vs_standard": outcome.js_vs_standard,
        "js_divergence_inter_trial": outcome.js_inter_trial,
    }
    write_native_spd(
        con,
        run_id=forecast_run_id,
        question=question,
        bucket_probs=final,
        spd_payload=spd_payload,
        human_explanation=_human_explanation(question, outcome.trials),
        cost_usd=qcost.total_usd,
    )

    leakage = LeakageStats()
    for t in outcome.trials:
        leakage.merge(t.leakage)

    persist_sibyl_forecast(
        con,
        {
            "sibyl_run_id": sibyl_run_id,
            "run_id": forecast_run_id,
            "question_id": question.question_id,
            "iso3": question.iso3,
            "hazard_code": question.hazard_code,
            "metric": question.metric,
            "status": "ok",
            "skip_reason": None,
            "as_of": as_of.isoformat(),
            "k": len(ok_trials),
            "aggregation": "linear_pool_by_month",
            "volatility_score": question.volatility_score,
            "triage_score": question.triage_score,
            "selection_pass": question.selection_pass,
            "base_rate": base_rate_record,
            "advice_arm": arm,
            "advice_as_of_month": advice.as_of_month if (track_record and advice) else None,
            "pooled_quantiles": spd_payload["pooled_quantiles"],
            "trials": [t.to_dict() for t in outcome.trials],
            "bucket_probs": list(final[1]),
            "reference": reference_record,
            "raw_by_month": pool.to_dict(),
            "final_by_month": {str(m): v for m, v in sorted(final.items())},
            "js_divergence_vs_standard": outcome.js_vs_standard,
            "js_divergence_inter_trial": outcome.js_inter_trial,
            "cost_usd": qcost.total_usd,
            "opus_cost_usd": qcost.opus_usd,
            "brave_cost_usd": qcost.brave_usd,
            "extraction_cost_usd": qcost.extraction_usd,
            "leakage": leakage.to_dict(),
            "evidence_ok": True,
            "extra_trials_rule": outcome.extra_trials_rule,
            "trial_checks": outcome.trial_checks,
        },
    )
    outcome.status = "ok"
    if not is_control:
        def _shadow_run(trial_index: int, call: Any, sink: List[Dict[str, Any]],
                        provider: str, model_id: str) -> TrialResult:
            # Same loop, same prompt, lane C; every dollar under 'shadow'.
            return run_trial(
                question,
                base_rate,
                as_of=as_of,
                trial_index=trial_index,
                run_id=sibyl_run_id,
                tracker=tracker,
                forecast_months=forecast_keys,
                country_name=country,
                model_call=call,
                track_record=track_record,
                lane="C",
                lessons=lessons,
                log_sink=sink,
                provider=provider,
                model_id=model_id,
                cost_kind=COST_KIND_SHADOW,
            )

        outcome.shadow_ctx = ShadowContext(
            question_id=question.question_id,
            run=_shadow_run,
            trial_index=max((t.trial_index for t in outcome.trials), default=-1) + 1,
            valid_trials=[t for t in finished if t.evidence_ok],
            reference=ref_vectors,
            weight=weight,
            metric=metric,
            required=required,
        )
    return outcome


def _persist_non_ok(
    con: Any,
    question: SibylQuestion,
    *,
    sibyl_run_id: str,
    status: str,
    skip_reason: str,
    tracker: CostTracker,
    trials: Optional[List[TrialResult]] = None,
    extra_trials_rule: Optional[str] = None,
    trial_checks: Optional[Dict[str, Any]] = None,
) -> None:
    qcost = tracker.question_breakdown(question.question_id)
    persist_sibyl_forecast(
        con,
        {
            "sibyl_run_id": sibyl_run_id,
            "run_id": None,
            "question_id": question.question_id,
            "iso3": question.iso3,
            "hazard_code": question.hazard_code,
            "metric": question.metric,
            "status": status,
            "skip_reason": skip_reason,
            "as_of": resolve_as_of(question).isoformat(),
            "k": len([t for t in (trials or []) if t.ok]),
            "aggregation": AGGREGATION,
            "volatility_score": question.volatility_score,
            "triage_score": question.triage_score,
            "selection_pass": question.selection_pass,
            "pooled_quantiles": None,
            "trials": [t.to_dict() for t in (trials or [])],
            "bucket_probs": None,
            "js_divergence_vs_standard": None,
            "js_divergence_inter_trial": None,
            "cost_usd": qcost.total_usd,
            "opus_cost_usd": qcost.opus_usd,
            "brave_cost_usd": qcost.brave_usd,
            "extraction_cost_usd": qcost.extraction_usd,
            "leakage": None,
            # A question that ran trials and stored no forecast had none that
            # rested on evidence; a skip that ran nothing has no verdict.
            "evidence_ok": (False if trials else None),
            "extra_trials_rule": extra_trials_rule,
            "trial_checks": trial_checks,
        },
    )


def run_sibyl(
    hs_run_id: Optional[str] = None,
    *,
    n_questions: int = N_QUESTIONS,
    model_call: Any = None,
    max_runtime_min: float = MAX_RUNTIME_MIN,
    clock: Any = None,
    shadow_call: Any = None,
) -> Dict[str, Any]:
    """Execute a full Sibyl cycle. Returns the run summary dict.

    *clock* is a monotonic-seconds callable (test seam for the time cap).
    *shadow_call* is the shadow arm's model seam (sibyl/shadow.py); without
    it the arm calls OpenAI, and with *model_call* injected and no
    *shadow_call* it does not run.
    """
    ensure_schema()
    sibyl_run_id = f"sibyl_{int(time.time() * 1000)}"
    tracker = CostTracker()
    con = connect(read_only=False)
    clock = clock or time.monotonic
    budget_capped = False
    time_capped = False
    n_forecast = 0
    n_skipped = 0
    resolved_hs_run_id = hs_run_id

    try:
        # Start-of-run state: the shared Brave breaker (a trip left over from
        # earlier in the process blinded every search of the July 2026 run)
        # and the tool counters written to sibyl_runs.
        sibyl_tools.reset_run_state()
        backfill_evidence_ok(con)

        # Derive test mode from the target HS run (workflow_run triggers
        # cannot carry the upstream run's test_mode input). Setting the env
        # var here makes every downstream is_test_mode() consumer — question
        # selection, cost ledger, SPD writers — stamp is_test consistently.
        resolved_hs_run_id = hs_run_id or latest_hs_run_id(con)
        from pythia.test_mode import is_test_mode as _is_test_mode
        if resolved_hs_run_id and hs_run_is_test(resolved_hs_run_id, con) and not _is_test_mode():
            os.environ["PYTHIA_TEST_MODE"] = "1"
            logger.warning(
                "sibyl.run: HS run %s is a test run — enabling PYTHIA_TEST_MODE "
                "so Sibyl outputs are stamped is_test.",
                resolved_hs_run_id,
            )

        questions = select_top_questions(
            resolved_hs_run_id, n=n_questions, con=con,
            n_control=_cfg.N_CONTROL,
            max_per_hazard_overrides=_cfg.MAX_PER_HAZARD_OVERRIDES,
        )
        if questions:
            resolved_hs_run_id = questions[0].hs_run_id
        logger.info(
            "sibyl.run: %s starting — %d questions, cap $%.2f, K=%d, model=%s",
            sibyl_run_id, len(questions), tracker.run_hard_cap_usd, K, MODEL,
        )

        shadow = shadow_setup(model_call_injected=model_call is not None, shadow_call=shadow_call)
        logger.info("sibyl.run: shadow arm %s%s", shadow.status,
                    f" ({shadow.provider}:{shadow.model_id})" if shadow.on else "")
        weight, weight_source = reference_weight(con, date.today().strftime("%Y-%m"))
        logger.info("sibyl.run: reference weight %.2f (%s)", weight, weight_source)
        outcomes: List[QuestionOutcome] = []
        n_evidence_rows = 0

        loop_started = clock()
        for question in questions:
            if time_capped or _runtime_cap_reached(loop_started, max_runtime_min, clock()):
                # Wall-clock cut-off: the Sibyl job is the release trigger
                # and has a hard timeout, so no new question starts once the
                # limit has passed. The question in flight already finished.
                if not time_capped:
                    logger.warning(
                        "sibyl.run: time cap (%.0f min) reached — no new "
                        "questions start; the rest are skipped.",
                        max_runtime_min,
                    )
                time_capped = True
                n_skipped += 1
                _persist_non_ok(
                    con, question,
                    sibyl_run_id=sibyl_run_id, status="skipped",
                    skip_reason=SKIP_REASON_TIME, tracker=tracker,
                )
                continue
            if tracker.hard_cap_reached():
                # Hard cut-off: no new question starts. Persist the skip so
                # the dashboard shows exactly what the cap sacrificed.
                budget_capped = True
                n_skipped += 1
                logger.warning(
                    "sibyl.run: budget cap ($%.2f) reached at $%.2f — "
                    "skipping %s",
                    tracker.run_hard_cap_usd, tracker.run_cost_usd,
                    question.question_id,
                )
                _persist_non_ok(
                    con, question,
                    sibyl_run_id=sibyl_run_id, status="skipped",
                    skip_reason=SKIP_REASON_BUDGET, tracker=tracker,
                )
                continue

            try:
                outcome = process_question(
                    con, question,
                    sibyl_run_id=sibyl_run_id, tracker=tracker,
                    model_call=model_call, reference_weight=weight,
                )
            except Exception as exc:  # noqa: BLE001 - one question must not sink the run
                logger.exception(
                    "sibyl.run: unexpected failure on %s: %s",
                    question.question_id, exc,
                )
                _persist_non_ok(
                    con, question,
                    sibyl_run_id=sibyl_run_id, status="failed",
                    skip_reason=f"exception: {exc}", tracker=tracker,
                )
                continue

            outcomes.append(outcome)
            n_evidence_rows += write_evidence(
                con, sibyl_run_id=sibyl_run_id, question_id=question.question_id,
                trials=outcome.trials, is_test=_is_test_mode(),
            )
            if outcome.status == "ok":
                n_forecast += 1
                logger.info(
                    "sibyl.run: %s forecast ok (JSD vs standard: %s, "
                    "inter-trial: %s, question cost $%.2f, run $%.2f)",
                    question.question_id,
                    f"{outcome.js_vs_standard:.4f}" if outcome.js_vs_standard is not None else "n/a",
                    f"{outcome.js_inter_trial:.4f}" if outcome.js_inter_trial is not None else "n/a",
                    tracker.question_cost_usd(question.question_id),
                    tracker.run_cost_usd,
                )
            else:
                n_skipped += 1
                _persist_non_ok(
                    con, question,
                    sibyl_run_id=sibyl_run_id, status=outcome.status,
                    skip_reason=outcome.skip_reason or "unknown",
                    tracker=tracker, trials=outcome.trials,
                    extra_trials_rule=outcome.extra_trials_rule,
                    trial_checks=outcome.trial_checks,
                )

        # The cap can also fire during the LAST question's trials (no
        # subsequent question gets skipped at the top of the loop, so the
        # flag above never flips); record it from realized spend so the
        # dashboard's BUDGET CAPPED badge reflects every capped run.
        if tracker.hard_cap_reached():
            budget_capped = True

        # The tool counters describe production research; the shadow
        # trials' searches are read after this snapshot and not counted.
        tool_counts = sibyl_tools.COUNTERS.snapshot()
        # Documents read: the trials' own count, where a repeat read of a URL
        # or of the same text counts once (the tool counter counts fetches).
        tool_counts["n_docs_read"] = sum(
            int(t.n_docs_read or 0) for o in outcomes for t in o.trials
        )

        # The shadow arm runs only after every question's production trials,
        # so it can never take budget or time production needed.
        def _minutes_left() -> float:
            if not max_runtime_min or max_runtime_min <= 0:
                return float("inf")
            return float(max_runtime_min) - (clock() - loop_started) / 60.0

        shadow_counts = run_shadow_phase(
            con,
            [o.shadow_ctx for o in outcomes if o.status == "ok" and o.shadow_ctx is not None],
            shadow,
            sibyl_run_id=sibyl_run_id,
            tracker=tracker,
            minutes_left=_minutes_left,
            write_log=_write_log,
            is_test=_is_test_mode(),
        )

        breakdown = tracker.run_breakdown()
        if tool_counts["n_search_calls"]:
            fail_share = tool_counts["n_search_failed"] / tool_counts["n_search_calls"]
            if fail_share > _cfg.DEGRADED_SEARCH_FAIL_SHARE:
                logger.warning(
                    "sibyl.run: %d of %d searches failed (%.0f%%; %d breaker "
                    "trip(s)) — this run's research is degraded",
                    tool_counts["n_search_failed"], tool_counts["n_search_calls"],
                    100 * fail_share, tool_counts["n_breaker_trips"],
                )
        run_record = {
            "sibyl_run_id": sibyl_run_id,
            "hs_run_id": resolved_hs_run_id,
            "as_of": date.today().isoformat(),
            "model": MODEL,
            "k": K,
            "max_steps": MAX_STEPS,
            "aggregation": AGGREGATION,
            "run_hard_cap_usd": tracker.run_hard_cap_usd,
            "budget_capped": budget_capped,
            "time_capped": time_capped,
            "run_cost_usd": breakdown.total_usd,
            "opus_cost_usd": breakdown.opus_usd,
            "brave_cost_usd": breakdown.brave_usd,
            "extraction_cost_usd": breakdown.extraction_usd,
            "shadow_cost_usd": breakdown.shadow_usd,
            **shadow_counts.to_record(),
            "n_selected": len(questions),
            "n_forecast": n_forecast,
            "n_skipped": n_skipped,
            **tool_counts,
            **process_measures(outcomes),
            "reference_weight": weight,
            "reference_weight_source": weight_source,
            "n_evidence_rows": n_evidence_rows,
            "config": {
                "N_QUESTIONS": n_questions,
                "MIN_PER_HAZARD": MIN_PER_HAZARD,
                "MAX_PER_HAZARD": MAX_PER_HAZARD,
                "MAX_PER_HAZARD_OVERRIDES": _cfg.MAX_PER_HAZARD_OVERRIDES,
                "N_CONTROL": _cfg.N_CONTROL,
                "K_MAX": _cfg.K_MAX,
                "EXTRA_TRIALS_JSD": _cfg.EXTRA_TRIALS_JSD,
                "EXTRA_TRIALS_DEPARTURE_JSD": _cfg.EXTRA_TRIALS_DEPARTURE_JSD,
                "OUTLIER_LOG10": _cfg.OUTLIER_LOG10,
                "TRIAL_WORKERS": _cfg.TRIAL_WORKERS,
                "MAX_RUNTIME_MIN": max_runtime_min,
                "QUANTILE_LEVELS": sibyl_config.QUANTILE_LEVELS,
                "BACKTEST_MODE": sibyl_config.BACKTEST_MODE,
                "BUDGET_USD_PER_QUESTION": sibyl_config.BUDGET_USD_PER_QUESTION,
                "RUN_HARD_CAP_USD": RUN_HARD_CAP_USD,
                "MIN_SEARCH_OK": _cfg.MIN_SEARCH_OK,
                "MIN_DOCS_READ": _cfg.MIN_DOCS_READ,
                "MIN_VALID_TRIALS": _cfg.MIN_VALID_TRIALS,
                "BUCKET_FLOOR": _cfg.BUCKET_FLOOR,
                "SHADOW_MODEL": _cfg.SHADOW_MODEL,
                "SHADOW_EFFORT": _cfg.SHADOW_EFFORT,
                "SHADOW_UNTIL": _cfg.SHADOW_UNTIL,
                "SHADOW_HEADROOM_USD": _cfg.SHADOW_HEADROOM_USD,
                "SHADOW_HEADROOM_MIN": _cfg.SHADOW_HEADROOM_MIN,
                "shadow_skip_reasons": shadow_counts.skip_reasons,
            },
        }
        persist_sibyl_run(con, run_record)
    finally:
        con.close()

    logger.info(
        "sibyl.run: %s done — %d forecast, %d skipped, $%.2f spent%s%s",
        sibyl_run_id, n_forecast, n_skipped, tracker.run_cost_usd,
        " [BUDGET CAPPED]" if budget_capped else "",
        " [TIME CAPPED]" if time_capped else "",
    )
    print(
        f"sibyl_run_id={sibyl_run_id} forecast={n_forecast} "
        f"skipped={n_skipped} cost_usd={tracker.run_cost_usd:.2f} "
        f"budget_capped={budget_capped} time_capped={time_capped}"
    )
    return run_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sibyl forecasting harness")
    parser.add_argument("--hs-run-id", default=None, help="HS run to forecast (default: latest)")
    parser.add_argument("--n", type=int, default=N_QUESTIONS, help="questions to select")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    summary = run_sibyl(args.hs_run_id, n_questions=args.n)
    print(json.dumps(summary, default=str, indent=2))


if __name__ == "__main__":
    main()
