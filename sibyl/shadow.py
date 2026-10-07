# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The shadow arm: one trial on a second model family (Oct 2026, Part 7).

The question it answers: does one trial from a different model family make
Sibyl's pool better? Production stays all Claude and nothing here changes
what Sibyl publishes.

How it runs
-----------
After every question of the run has had its production trials, each
question that produced a forecast (controls excluded) gets one more trial
on lane C, through the same loop and the same prompt, on
``SIBYL_SHADOW_MODEL`` (default the registry alias ``gpt``, i.e.
``openai:gpt-6-sol``, at reasoning effort ``SIBYL_SHADOW_EFFORT``). Every
dollar it spends is counted under its own cost kind (``shadow``).

The shadow series is the production pool with the Claude lane C trial
replaced by the shadow trial, then the same steps as production: the
evidence gate, the outlier guard, the linear pool by month, the mix with the
reference at the run's weight, and the bucket floor. It is stored in
``sibyl_forecasts.shadow_json`` and NEVER in ``forecasts_raw`` or
``forecasts_ensemble``; ``sibyl.score_variants`` scores it as
``__ext_sibyl_shadow``, and the single shadow trial against the single
Claude lane C trial in ``sibyl_variant_scores``.

When it does not run
--------------------
The run records why on ``sibyl_runs.shadow_status``: ``off``
(``SIBYL_SHADOW_MODEL=off``), ``backtest``, ``expired`` (past
``SIBYL_SHADOW_UNTIL``), ``unknown_model``, ``unsupported_provider`` (only
OpenAI is wired), ``no_key`` (``OPENAI_API_KEY`` is not set: a warning, and
the production run is untouched), or ``no_shadow_call`` (a test injected the
production model and not the shadow one, so the shadow trial would have gone
to the network). Near the caps the shadow trials stop first: none starts
within ``SIBYL_SHADOW_HEADROOM_USD`` of the run's hard cap or
``SIBYL_SHADOW_HEADROOM_MIN`` of its time cap.

Reading it
----------
``shadow_comparison`` reports shadow minus ``sibyl`` per score type, a mean
of per-question differences with a 90% interval that resamples questions,
and says "not yet" below ``SIBYL_SHADOW_MIN_QUESTIONS`` (20). It is a
finding. Adopting a second model is the owner's decision.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sibyl import config as _cfg

logger = logging.getLogger(__name__)

SHADOW_MODEL_NAME = "__ext_sibyl_shadow"
#: sibyl_variant_scores series for the single-trial comparison.
SERIES_SHADOW_TRIAL = "shadow_trial"
SERIES_CLAUDE_LANE_C = "claude_lane_c"
SHADOW_LANE = "C"
SCORE_TYPES = ("brier", "log", "crps")

ModelCall = Callable[[str], Tuple[str, Dict[str, Any], str]]


# ---------------------------------------------------------------------------
# Whether the arm runs, and on what
# ---------------------------------------------------------------------------


@dataclass
class ShadowSetup:
    status: str  # on | off | backtest | expired | unknown_model | unsupported_provider | no_key | no_shadow_call
    provider: str = ""
    model_id: str = ""
    call: Optional[ModelCall] = None

    @property
    def on(self) -> bool:
        return self.status == "on" and self.call is not None


def _until_passed(until: str, today: date) -> Optional[bool]:
    """True when *today* is in a month after *until* (YYYY-MM); None if unparseable."""
    m = re.fullmatch(r"(\d{4})-(\d{2})", (until or "").strip())
    if not m:
        return None
    return (today.year, today.month) > (int(m.group(1)), int(m.group(2)))


def resolve_shadow_model(ref: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """(provider, model_id) for ``SIBYL_SHADOW_MODEL``; None when it does not resolve."""
    ref = (_cfg.SHADOW_MODEL if ref is None else ref) or ""
    try:
        from pythia.llm_profiles import resolve_model_ref, split_model_ref  # noqa: PLC0415

        resolved = resolve_model_ref(ref)
        if not resolved:
            return None
        return split_model_ref(resolved, default_provider="openai")
    except Exception as exc:  # noqa: BLE001 - a broken registry stops the arm, never the run
        logger.warning("sibyl.shadow: could not resolve %r: %s", ref, exc)
        return None


def openai_shadow_call(model_id: str) -> ModelCall:
    """One shadow step through ``forecaster.providers.call_openai``."""

    def call(prompt: str) -> Tuple[str, Dict[str, Any], str]:
        from forecaster.providers import call_openai, estimate_cost_usd  # noqa: PLC0415

        effort = _cfg.SHADOW_EFFORT if _cfg.SHADOW_EFFORT not in ("", "off", "none") else None
        result = call_openai(
            prompt, model_id, 1.0,
            reasoning_effort=effort,
            prompt_cache_key="pythia:sibyl:shadow",
        )
        usage = dict(result.usage or {})
        if not usage.get("cost_usd"):
            usage["cost_usd"] = estimate_cost_usd(model_id, usage)
        return result.text or "", usage, result.error or ""

    return call


def shadow_setup(
    *,
    today: Optional[date] = None,
    backtest: Optional[bool] = None,
    model_call_injected: bool = False,
    shadow_call: Optional[ModelCall] = None,
) -> ShadowSetup:
    """Decide whether the shadow arm runs this run, and with which call."""
    today = today or date.today()
    backtest = _cfg.BACKTEST_MODE if backtest is None else backtest
    if (_cfg.SHADOW_MODEL or "").lower() in ("", "off", "none"):
        return ShadowSetup("off")
    if backtest:
        return ShadowSetup("backtest")
    passed = _until_passed(_cfg.SHADOW_UNTIL, today)
    if passed is None and (_cfg.SHADOW_UNTIL or "").lower() in ("off", "none"):
        return ShadowSetup("off")
    if passed is None:
        logger.warning("sibyl.shadow: SIBYL_SHADOW_UNTIL=%r is not YYYY-MM; the arm is off",
                       _cfg.SHADOW_UNTIL)
        return ShadowSetup("off")
    if passed:
        return ShadowSetup("expired")
    resolved = resolve_shadow_model()
    if resolved is None:
        logger.warning("sibyl.shadow: SIBYL_SHADOW_MODEL=%r does not resolve; the arm is off",
                       _cfg.SHADOW_MODEL)
        return ShadowSetup("unknown_model")
    provider, model_id = resolved
    if shadow_call is not None:
        return ShadowSetup("on", provider, model_id, shadow_call)
    if provider != "openai":
        logger.warning("sibyl.shadow: provider %r is not wired for the shadow arm", provider)
        return ShadowSetup("unsupported_provider", provider, model_id)
    if not (os.getenv("OPENAI_API_KEY") or "").strip():
        logger.warning(
            "sibyl.shadow: OPENAI_API_KEY is not set — the shadow trials are skipped "
            "and the run is flagged shadow_status=no_key. Production is unaffected."
        )
        return ShadowSetup("no_key", provider, model_id)
    if model_call_injected:
        return ShadowSetup("no_shadow_call", provider, model_id)
    return ShadowSetup("on", provider, model_id, openai_shadow_call(model_id))


# ---------------------------------------------------------------------------
# What a question hands the shadow phase
# ---------------------------------------------------------------------------


@dataclass
class ShadowContext:
    """Built by ``process_question`` for a question that produced a forecast.

    *run* runs the shadow trial: ``run(trial_index, model_call, log_sink,
    provider, model_id)`` with the question's own prompt inputs.
    """

    question_id: str
    run: Callable[..., Any]
    trial_index: int
    valid_trials: List[Any]  # finished production trials with evidence
    reference: Optional[Dict[int, List[float]]]
    weight: float
    metric: str
    required: int


# ---------------------------------------------------------------------------
# The shadow series
# ---------------------------------------------------------------------------


def _months_of(trial: Any) -> Dict[int, Any]:
    return {1: trial.month_beliefs[1].dist(), 6: trial.month_beliefs[6].dist()}


def single_trial_vectors(trial: Any, metric: str) -> Dict[str, List[float]]:
    """One trial's vectors by window month, floored as a published vector is."""
    from sibyl.aggregate import pool_months  # noqa: PLC0415
    from sibyl.spd import apply_bucket_floor  # noqa: PLC0415

    pool = pool_months([_months_of(trial)], metric)
    return {str(m): apply_bucket_floor(v) for m, v in sorted(pool.vectors.items())}


def build_shadow_series(ctx: ShadowContext, shadow_trial: Any, *, model: str) -> Dict[str, Any]:
    """The production pool with the Claude lane C trial replaced by the shadow trial."""
    from sibyl.aggregate import pool_months, publish_vectors  # noqa: PLC0415
    from sibyl.spd import apply_bucket_floor  # noqa: PLC0415
    from sibyl.trials import month_median, outlier_indices  # noqa: PLC0415

    claude_c = [t for t in ctx.valid_trials if getattr(t, "lane", "") == SHADOW_LANE]
    out: Dict[str, Any] = {
        "model": model,
        "lane": SHADOW_LANE,
        "shadow_trial_index": getattr(shadow_trial, "trial_index", None),
        "replaced_trial_index": claude_c[0].trial_index if claude_c else None,
        "cost_usd": round(float(shadow_trial.cost.total_usd), 6) if shadow_trial else 0.0,
        "trial": shadow_trial.to_dict() if shadow_trial is not None else None,
    }
    if shadow_trial is None or not (shadow_trial.ok and shadow_trial.evidence_ok):
        out["status"] = "no_valid_shadow_trial"
        return out
    try:
        out["shadow_trial_by_month"] = single_trial_vectors(shadow_trial, ctx.metric)
        if claude_c:
            out["claude_lane_c_by_month"] = single_trial_vectors(claude_c[0], ctx.metric)
        members = [t for t in ctx.valid_trials if t not in claude_c] + [shadow_trial]
        dropped: List[int] = []
        if len(members) >= 3:
            medians = [
                month_median(t.month_beliefs[1].p_zero, t.month_beliefs[1].quantiles_positive)
                for t in members
            ]
            drop = outlier_indices(medians)
            dropped = [members[i].trial_index for i in drop]
            members = [t for i, t in enumerate(members) if i not in set(drop)]
        out["trial_indices"] = [t.trial_index for t in members]
        out["outliers_dropped"] = dropped
        if len(members) < ctx.required:
            out["status"] = "too_few_trials"
            return out
        pool = pool_months([_months_of(t) for t in members], ctx.metric)
        final = {
            m: apply_bucket_floor(v)
            for m, v in publish_vectors(pool.vectors, ctx.reference, ctx.weight).items()
        }
        out["raw_by_month"] = pool.to_dict()
        out["final_by_month"] = {str(m): v for m, v in sorted(final.items())}
        out["reference_weight"] = ctx.weight if ctx.reference else 0.0
        out["status"] = "ok"
    except (ValueError, KeyError, AttributeError) as exc:
        out["status"] = f"aggregation failed: {exc}"
    return out


# ---------------------------------------------------------------------------
# The phase
# ---------------------------------------------------------------------------


@dataclass
class ShadowCounters:
    status: str
    model: str = ""
    n_trials: int = 0
    n_series: int = 0
    n_skipped: int = 0
    skip_reasons: Dict[str, int] = field(default_factory=dict)

    def to_record(self) -> Dict[str, Any]:
        return {
            "shadow_status": self.status,
            "shadow_model": self.model or None,
            "n_shadow_trials": self.n_trials,
            "n_shadow_series": self.n_series,
            "n_shadow_skipped": self.n_skipped,
        }


def _store(con, sibyl_run_id: str, question_id: str, payload: Dict[str, Any]) -> None:
    try:
        con.execute(
            "UPDATE sibyl_forecasts SET shadow_json = ?, shadow_cost_usd = ? "
            "WHERE sibyl_run_id = ? AND question_id = ?",
            [json.dumps(payload, default=str), float(payload.get("cost_usd") or 0.0),
             sibyl_run_id, question_id],
        )
    except Exception as exc:  # noqa: BLE001 - the shadow record never stops a run
        logger.warning("sibyl.shadow: could not store the shadow series of %s: %s",
                       question_id, exc)


def run_shadow_phase(
    con: Any,
    contexts: Sequence[ShadowContext],
    setup: ShadowSetup,
    *,
    sibyl_run_id: str,
    tracker: Any,
    minutes_left: Callable[[], float],
    write_log: Optional[Callable[..., None]] = None,
    is_test: bool = False,
    workers: Optional[int] = None,
) -> ShadowCounters:
    """Run the shadow trials after all production work. Never raises."""
    from sibyl.measure import write_evidence  # noqa: PLC0415
    from sibyl.trials import run_trial_batch  # noqa: PLC0415

    counters = ShadowCounters(status=setup.status, model=f"{setup.provider}:{setup.model_id}"
                              if setup.model_id else "")
    if not setup.on or not contexts:
        return counters
    workers = max(1, int(_cfg.TRIAL_WORKERS if workers is None else workers))

    def _skip(ctx: ShadowContext, reason: str) -> None:
        counters.n_skipped += 1
        counters.skip_reasons[reason] = counters.skip_reasons.get(reason, 0) + 1
        _store(con, sibyl_run_id, ctx.question_id,
               {"status": "skipped", "reason": reason, "model": counters.model, "cost_usd": 0.0})

    pending = list(contexts)
    while pending:
        reason = None
        if tracker.run_cost_usd + _cfg.SHADOW_HEADROOM_USD >= tracker.run_hard_cap_usd:
            reason = "run budget headroom"
        elif minutes_left() <= _cfg.SHADOW_HEADROOM_MIN:
            reason = "run time headroom"
        if reason:
            logger.warning("sibyl.shadow: %s reached — %d shadow trial(s) not started",
                           reason, len(pending))
            for ctx in pending:
                _skip(ctx, reason)
            break
        batch, pending = pending[:workers], pending[workers:]

        def _one(pos: int, _lane: str, sink: List[Dict[str, Any]], _batch=batch) -> Any:
            ctx = _batch[pos]
            return ctx.run(ctx.trial_index, setup.call, sink, setup.provider, setup.model_id)

        results = run_trial_batch([(i, SHADOW_LANE) for i in range(len(batch))], _one,
                                  workers=workers, write_log=write_log)
        for ctx, trial in zip(batch, results):
            if trial is not None:
                trial.role = "shadow"
                counters.n_trials += 1
                write_evidence(con, sibyl_run_id=sibyl_run_id, question_id=ctx.question_id,
                               trials=[trial], is_test=is_test, replace=False, role="shadow")
            payload = build_shadow_series(ctx, trial, model=counters.model)
            if trial is None:
                payload["status"] = "trial raised"
            if payload.get("status") == "ok":
                counters.n_series += 1
            _store(con, sibyl_run_id, ctx.question_id, payload)
    logger.info("sibyl.shadow: %d trial(s), %d series, %d skipped %s",
                counters.n_trials, counters.n_series, counters.n_skipped,
                counters.skip_reasons or "")
    return counters


# ---------------------------------------------------------------------------
# Reading it back: shadow minus sibyl, paired over questions
# ---------------------------------------------------------------------------


def _paired(diffs: Dict[str, Dict[str, float]], min_questions: int) -> Dict[str, Any]:
    from sibyl.advice import bootstrap_ratio  # noqa: PLC0415

    out: Dict[str, Any] = {}
    for st in SCORE_TYPES:
        vals = [(v[st], 1.0) for v in diffs.values() if st in v]
        n = len(vals)
        if n < min_questions:
            out[st] = {"status": "not_yet", "n_questions": n, "min_questions": min_questions}
            continue
        stat = bootstrap_ratio(vals).to_dict()
        out[st] = {"status": "ok", "n_questions": n, "mean_diff": stat["value"],
                   "lo": stat["lo"], "hi": stat["hi"]}
    return out


def shadow_comparison(con: Any, *, include_test: bool = False,
                      min_questions: Optional[int] = None) -> Dict[str, Any]:
    """Shadow minus production, per score type, over the questions both scored.

    ``series``: ``__ext_sibyl_shadow`` against ``sibyl`` on the latest Sibyl
    forecast of each question (the one score_variants scored). ``trial``:
    the single shadow trial against the single Claude lane C trial, from
    ``sibyl_variant_scores``. Negative means the shadow arm scored better.
    Each per-question value is the mean over its resolved horizons; the
    interval resamples questions. Below *min_questions* the report says
    "not yet" and gives no number. Never raises.
    """
    min_q = int(_cfg.SHADOW_MIN_QUESTIONS if min_questions is None else min_questions)
    empty = {"series": _paired({}, min_q), "trial": _paired({}, min_q),
             "min_questions": min_q, "model": None}
    try:
        cols = {str(r[1]).lower() for r in con.execute(
            "PRAGMA table_info('sibyl_forecasts')").fetchall()}
        if "shadow_json" not in cols:
            return empty
        has_scores = bool(con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'scores'").fetchone())
        test_f = (" AND NOT COALESCE(f.is_test, FALSE)"
                  if not include_test and "is_test" in cols else "")
        ev = " AND COALESCE(f.evidence_ok, TRUE)" if "evidence_ok" in cols else ""
        order = "f.created_at DESC NULLS LAST, " if "created_at" in cols else ""
        rows = con.execute(
            f"""
            SELECT question_id, run_id, shadow_json FROM (
                SELECT f.question_id, f.run_id, f.shadow_json,
                       ROW_NUMBER() OVER (PARTITION BY f.question_id
                           ORDER BY {order}f.sibyl_run_id DESC) AS rn
                FROM sibyl_forecasts f
                WHERE f.status = 'ok'{ev}{test_f}
            ) WHERE rn = 1
            """
        ).fetchall()
        series_diffs: Dict[str, Dict[str, float]] = {}
        model = None
        s_test = "" if include_test else " AND NOT COALESCE(a.is_test, FALSE)"
        # Indicative ACE/PA months never enter a comparison (scoring_class.py).
        from pythia.tools.scoring_class import scored_only_clause  # noqa: PLC0415

        s_test += scored_only_clause(con, "a")
        for qid, run_id, sj in rows:
            payload = json.loads(sj) if sj else {}
            if payload.get("status") != "ok":
                continue
            model = model or payload.get("model")
            if not has_scores:
                continue
            got = con.execute(
                f"""
                SELECT a.score_type, AVG(a.value - b.value)
                FROM scores a JOIN scores b
                  ON a.question_id = b.question_id AND a.horizon_m = b.horizon_m
                 AND a.score_type = b.score_type
                WHERE a.question_id = ? AND a.model_name = ? AND a.run_id IS NULL
                  AND b.model_name = 'sibyl' AND b.run_id = ?
                  {s_test}
                GROUP BY 1
                """,
                [qid, SHADOW_MODEL_NAME, run_id],
            ).fetchall()
            for st, v in got:
                if v is not None:
                    series_diffs.setdefault(str(qid), {})[str(st)] = float(v)
        trial_diffs: Dict[str, Dict[str, float]] = {}
        if con.execute("SELECT 1 FROM information_schema.tables "
                       "WHERE table_name = 'sibyl_variant_scores'").fetchone():
            v_test = "" if include_test else " AND NOT COALESCE(a.is_test, FALSE)"
            v_test += scored_only_clause(con, "a")
            for qid, st, v in con.execute(
                f"""
                SELECT a.question_id, a.score_type, AVG(a.value - b.value)
                FROM sibyl_variant_scores a JOIN sibyl_variant_scores b
                  ON a.question_id = b.question_id AND a.horizon_m = b.horizon_m
                 AND a.score_type = b.score_type AND a.sibyl_run_id = b.sibyl_run_id
                WHERE a.series = ? AND b.series = ?{v_test}
                GROUP BY 1, 2
                """,
                [SERIES_SHADOW_TRIAL, SERIES_CLAUDE_LANE_C],
            ).fetchall():
                if v is not None:
                    trial_diffs.setdefault(str(qid), {})[str(st)] = float(v)
        return {"series": _paired(series_diffs, min_q), "trial": _paired(trial_diffs, min_q),
                "min_questions": min_q, "model": model}
    except Exception as exc:  # noqa: BLE001
        logger.warning("sibyl.shadow: comparison failed: %s", exc)
        return empty
