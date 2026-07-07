# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl run orchestrator.

Per run: select the top-N volatile affected/fatalities questions, and for
each (in DESCENDING volatility order, so the budget cap sacrifices the
lowest-value work first):

1. load the Resolver base rate (outside view),
2. run K independent agentic trials (Opus over open-web research),
3. linear-pool the K trial CDFs,
4. calibrate (identity hook while CALIBRATION_ENABLED is off),
5. serialize to the native SPD format beside the standard track,
6. record cost, and
7. compute the JS divergence vs the standard-Pythia SPD.

The hard budget cap is checked at every question boundary and between
trials; once reached, no new work starts, completed work is persisted,
remaining questions are marked ``skipped: run budget cap``, and the
run-level ``budget_capped`` flag is set.

Usage: ``python -m sibyl.run [--hs-run-id RUN] [--n N]``
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from pythia.db.schema import connect, ensure_schema

from sibyl import config as sibyl_config
from sibyl.agent import TrialResult, run_trial
from sibyl.aggregate import aggregate_trials
from sibyl.base_rates import load_base_rate
from sibyl.calibration import calibrate
from sibyl.config import (
    AGGREGATION,
    BACKTEST_MODE,
    K,
    MAX_STEPS,
    MODEL,
    N_QUESTIONS,
    RUN_HARD_CAP_USD,
)
from sibyl.cost import CostTracker
from sibyl.leakage import LeakageStats
from sibyl.select_questions import SibylQuestion, select_top_questions
from sibyl.spd import (
    bucket_probs_from_distribution,
    find_standard_run_id,
    inter_trial_divergence,
    load_standard_spd_by_month,
    persist_sibyl_forecast,
    persist_sibyl_run,
    track_divergence,
    write_native_spd,
)

logger = logging.getLogger(__name__)

SKIP_REASON_BUDGET = "run budget cap"


@dataclass
class QuestionOutcome:
    question: SibylQuestion
    status: str  # ok | skipped | failed
    skip_reason: Optional[str] = None
    trials: List[TrialResult] = field(default_factory=list)
    js_vs_standard: Optional[float] = None
    js_inter_trial: Optional[float] = None


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
) -> QuestionOutcome:
    """Forecast one question end-to-end. Returns the outcome (never raises)."""
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
    base_rate = load_base_rate(
        question.iso3, question.hazard_code, question.metric, forecast_keys
    )

    for trial_index in range(K):
        if tracker.hard_cap_reached():
            logger.warning(
                "sibyl.run: hard cap reached between trials of %s "
                "(%d/%d trials done); pooling completed trials.",
                question.question_id, trial_index, K,
            )
            break
        if tracker.question_cap_reached(question.question_id):
            logger.warning(
                "sibyl.run: per-question budget reached for %s after %d trials.",
                question.question_id, trial_index,
            )
            break
        trial = run_trial(
            question,
            base_rate,
            as_of=as_of,
            trial_index=trial_index,
            run_id=sibyl_run_id,
            tracker=tracker,
            forecast_months=forecast_keys,
            country_name=country,
            model_call=model_call,
        )
        outcome.trials.append(trial)

    ok_trials = [t for t in outcome.trials if t.ok]
    if not ok_trials:
        outcome.skip_reason = "no successful trials"
        return outcome

    try:
        pooled = aggregate_trials([t.quantiles for t in ok_trials], AGGREGATION)
        # Identity hook while CALIBRATION_ENABLED is off; horizon-specific
        # application is part of the deferred PIT work (see calibration.py).
        pooled = calibrate(pooled, question.hazard_code, 0)
        bucket_probs = bucket_probs_from_distribution(pooled, question.metric)
    except ValueError as exc:
        outcome.skip_reason = f"aggregation failed: {exc}"
        logger.error("sibyl.run: aggregation failed for %s: %s", question.question_id, exc)
        return outcome

    from pythia.buckets import n_buckets_for  # noqa: PLC0415

    standard_run_id = find_standard_run_id(con, question.question_id)
    forecast_run_id = standard_run_id or sibyl_run_id
    standard = (
        load_standard_spd_by_month(
            con, standard_run_id, question.question_id, n_buckets_for(question.metric)
        )
        if standard_run_id
        else None
    )
    outcome.js_vs_standard = track_divergence(bucket_probs, standard)
    outcome.js_inter_trial = inter_trial_divergence(
        [t.quantiles for t in ok_trials], question.metric
    )

    qcost = tracker.question_breakdown(question.question_id)
    spd_payload = {
        "track": "sibyl",
        "as_of": as_of.isoformat(),
        "k": len(ok_trials),
        "k_requested": K,
        "aggregation": AGGREGATION,
        "model": MODEL,
        "pooled_quantiles": {str(k): v for k, v in sorted(pooled.quantiles.items())},
        "trial_quantiles": [
            {str(k): v for k, v in sorted(t.quantiles.items())} for t in ok_trials
        ],
        "forecast_months": forecast_keys,
        "js_divergence_vs_standard": outcome.js_vs_standard,
        "js_divergence_inter_trial": outcome.js_inter_trial,
    }
    write_native_spd(
        con,
        run_id=forecast_run_id,
        question=question,
        bucket_probs=bucket_probs,
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
            "aggregation": AGGREGATION,
            "volatility_score": question.volatility_score,
            "triage_score": question.triage_score,
            "pooled_quantiles": spd_payload["pooled_quantiles"],
            "trials": [t.to_dict() for t in outcome.trials],
            "bucket_probs": list(bucket_probs),
            "js_divergence_vs_standard": outcome.js_vs_standard,
            "js_divergence_inter_trial": outcome.js_inter_trial,
            "cost_usd": qcost.total_usd,
            "opus_cost_usd": qcost.opus_usd,
            "brave_cost_usd": qcost.brave_usd,
            "leakage": leakage.to_dict(),
        },
    )
    outcome.status = "ok"
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
            "pooled_quantiles": None,
            "trials": [t.to_dict() for t in (trials or [])],
            "bucket_probs": None,
            "js_divergence_vs_standard": None,
            "js_divergence_inter_trial": None,
            "cost_usd": qcost.total_usd,
            "opus_cost_usd": qcost.opus_usd,
            "brave_cost_usd": qcost.brave_usd,
            "leakage": None,
        },
    )


def run_sibyl(
    hs_run_id: Optional[str] = None,
    *,
    n_questions: int = N_QUESTIONS,
    model_call: Any = None,
) -> Dict[str, Any]:
    """Execute a full Sibyl cycle. Returns the run summary dict."""
    ensure_schema()
    sibyl_run_id = f"sibyl_{int(time.time() * 1000)}"
    tracker = CostTracker()
    con = connect(read_only=False)
    budget_capped = False
    n_forecast = 0
    n_skipped = 0
    resolved_hs_run_id = hs_run_id

    try:
        questions = select_top_questions(hs_run_id, n=n_questions, con=con)
        if questions:
            resolved_hs_run_id = questions[0].hs_run_id
        logger.info(
            "sibyl.run: %s starting — %d questions, cap $%.2f, K=%d, model=%s",
            sibyl_run_id, len(questions), tracker.run_hard_cap_usd, K, MODEL,
        )

        for question in questions:
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
                    model_call=model_call,
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
                )

        # The cap can also fire during the LAST question's trials (no
        # subsequent question gets skipped at the top of the loop, so the
        # flag above never flips); record it from realized spend so the
        # dashboard's BUDGET CAPPED badge reflects every capped run.
        if tracker.hard_cap_reached():
            budget_capped = True

        breakdown = tracker.run_breakdown()
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
            "run_cost_usd": breakdown.total_usd,
            "opus_cost_usd": breakdown.opus_usd,
            "brave_cost_usd": breakdown.brave_usd,
            "n_selected": len(questions),
            "n_forecast": n_forecast,
            "n_skipped": n_skipped,
            "config": {
                "N_QUESTIONS": n_questions,
                "QUANTILE_LEVELS": sibyl_config.QUANTILE_LEVELS,
                "BACKTEST_MODE": sibyl_config.BACKTEST_MODE,
                "BUDGET_USD_PER_QUESTION": sibyl_config.BUDGET_USD_PER_QUESTION,
                "RUN_HARD_CAP_USD": RUN_HARD_CAP_USD,
            },
        }
        persist_sibyl_run(con, run_record)
    finally:
        con.close()

    logger.info(
        "sibyl.run: %s done — %d forecast, %d skipped, $%.2f spent%s",
        sibyl_run_id, n_forecast, n_skipped, tracker.run_cost_usd,
        " [BUDGET CAPPED]" if budget_capped else "",
    )
    print(
        f"sibyl_run_id={sibyl_run_id} forecast={n_forecast} "
        f"skipped={n_skipped} cost_usd={tracker.run_cost_usd:.2f} "
        f"budget_capped={budget_capped}"
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
