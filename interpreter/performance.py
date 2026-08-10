# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Part B: how well the system has done, and when it will first be testable.

No forecast has resolved yet. The first resolutions arrive at the end of
September 2026, for July's forecasts, and only for the earliest horizons. So
this module is built now and mostly dormant, and its dormant state is
deliberately informative: a reader should finish the section knowing how many
question-horizons are due, on what date, for which hazards. "There is nothing
to score" is true and tells them nothing.

The part that matters most is the small-sample discipline. The first scored
run will rest on well under a hundred question-horizons, and a naive report
will announce that Fred beat climatology when it did nothing of the kind.
Three rules, enforced here rather than asked for in the prompt:

* Below `min_resolved` the report **declines the claim**. It states the count
  and says the sample is too small. It does not hedge in prose while printing
  a number, because a printed number is what a reader remembers.
* Every skill figure carries a bootstrap interval, and the interval is what
  gets printed.
* Horizons are never pooled and score families are never blended. Binary Brier
  runs 0 to 1 and SPD Brier runs 0 to 2; an average across them is arithmetic
  with no meaning.

Pure functions over plain values, with a seeded bootstrap: a confidence
interval that moves between two runs of the same data is not evidence.
"""

from __future__ import annotations

import random
from typing import Any, Iterable, Mapping, Sequence

from interpreter.decisions import add_months, month_label, parse_ym

# The forecast window is six months and month index 1 is the window's first
# month (CLAUDE.md's anchoring convention).
N_HORIZONS = 6

# The resolution pipeline resolves the previous COMPLETE month, and its
# monthly cycle starts on the 28th (resolver_update -> compute_resolutions).
# So a window month resolves on the 28th of the month after it.
RESOLUTION_DAY = 28

VERDICT_TOO_FEW = "too few to say"
VERDICT_BETTER = "better than assuming a normal year"
VERDICT_WORSE = "worse than assuming a normal year"
VERDICT_UNCLEAR = "indistinguishable from assuming a normal year"

# The plain-English score copy. Mirrored verbatim in
# web/src/lib/score_glossary.ts (REPORT_SCORE_EXPLANATIONS) and pinned by
# interpreter/tests/test_performance.py, so the report and the dashboard
# cannot drift into explaining the same score two different ways.
SCORE_EXPLANATIONS: tuple[tuple[str, str], ...] = (
    (
        "Brier score",
        "How far the forecast sat from what happened, squared and averaged. "
        "Lower is better, and zero is perfect. It rewards a forecast that put "
        "its weight in the right place and punishes one that was confident "
        "and wrong.",
    ),
    (
        "Log loss",
        "The same idea, but far harsher on confidence. A forecast that ruled "
        "something out and then watched it happen scores very badly indeed. "
        "Lower is better.",
    ),
    (
        "Ranked probability score",
        "Brier's cousin for questions about size. It gives partial credit for "
        "being close: a forecast that expected the band next door scores "
        "better than one that expected the other end of the scale. Lower is "
        "better.",
    ),
    (
        "Skill",
        "The score set against what you would have got by assuming a normal "
        "year. Above zero means the system beat that; below zero means it "
        "lost to it. This is the number that says whether the forecasting was "
        "worth doing.",
    ),
)


# ---------------------------------------------------------------------------
# The dormant state: what is coming, and when
# ---------------------------------------------------------------------------


def resolution_date_for(window_month: str) -> str | None:
    """The date the pipeline first resolves a given window month."""
    nxt = add_months(window_month, 1)
    if not nxt:
        return None
    return f"{nxt}-{RESOLUTION_DAY:02d}"


def upcoming_resolutions(
    questions: Sequence[Mapping[str, Any]],
    *,
    as_of: str,
    n_horizons: int = N_HORIZONS,
) -> dict[str, Any]:
    """What is due to resolve, when, and for which hazards.

    ``as_of`` is a YYYY-MM-DD date. A horizon whose resolution date has
    already passed is reported as overdue rather than dropped: if the calendar
    says a month should have resolved and the report has nothing to score, the
    reader should be told that too.
    """
    schedule: dict[str, dict[str, Any]] = {}
    by_hazard: dict[str, int] = {}
    total = 0
    for q in questions:
        start = str(q.get("window_start_date") or "")[:7]
        if not parse_ym(start):
            continue
        hazard = str(q.get("hazard_code") or "").upper()
        for horizon in range(1, int(n_horizons) + 1):
            window_month = add_months(start, horizon - 1)
            if not window_month:
                continue
            date = resolution_date_for(window_month)
            if not date:
                continue
            total += 1
            by_hazard[hazard] = by_hazard.get(hazard, 0) + 1
            slot = schedule.setdefault(
                date,
                {
                    "resolution_date": date,
                    "window_month": window_month,
                    "window_month_label": month_label(window_month),
                    "n_question_horizons": 0,
                },
            )
            slot["n_question_horizons"] += 1

    ordered = [schedule[k] for k in sorted(schedule)]
    upcoming = [s for s in ordered if s["resolution_date"] > str(as_of)]
    overdue = [s for s in ordered if s["resolution_date"] <= str(as_of)]
    first = upcoming[0] if upcoming else None
    return {
        "as_of": str(as_of),
        "n_question_horizons": total,
        "n_questions": len(questions),
        "by_hazard": dict(sorted(by_hazard.items())),
        "first_resolution_date": first["resolution_date"] if first else None,
        "first_resolution_window_month": first["window_month_label"] if first else None,
        "n_first_resolution_horizons": (
            first["n_question_horizons"] if first else 0
        ),
        "schedule": ordered,
        "n_overdue_slots": len(overdue),
    }


def dormant_sentences(upcoming: Mapping[str, Any]) -> list[str]:
    """The dormant Part B, in sentences the renderer prints as they are.

    Generated, not written by the model: the whole point of the section is a
    set of counts and a date, and those are exactly what a language model
    should never be asked to supply.
    """
    lines = [
        "No forecast window has closed yet, so there is nothing to score. "
        "This is a fact about the calendar, not a gap in the system."
    ]
    date = upcoming.get("first_resolution_date")
    if date:
        lines.append(
            f"The first outcomes arrive on {date}, when "
            f"{upcoming.get('n_first_resolution_horizons', 0):,} "
            "question-months covering "
            f"{upcoming.get('first_resolution_window_month')} become "
            "resolvable."
        )
    total = upcoming.get("n_question_horizons") or 0
    if total:
        by_hazard = upcoming.get("by_hazard") or {}
        parts = ", ".join(f"{k} {v:,}" for k, v in by_hazard.items())
        lines.append(
            f"In all, {total:,} question-months from this run are due to "
            f"resolve over the coming half year ({parts})."
        )
    lines.append(
        "People-affected ground truth is the thin part. The resolution "
        "machine that will fix it runs in shadow mode and does not feed "
        "scoring yet, so the first scores will lean on fatalities and on the "
        "yes-or-no disaster-alert questions."
    )
    return lines


# ---------------------------------------------------------------------------
# Small-sample discipline
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    pool = [float(v) for v in values if v is not None]
    return sum(pool) / len(pool) if pool else None


def bootstrap_skill(
    paired: Sequence[tuple[float, float]],
    *,
    samples: int,
    seed: int = 20260801,
    alpha: float = 0.05,
) -> dict[str, Any] | None:
    """Skill against a reference, with a percentile bootstrap interval.

    ``paired`` is [(model_score, reference_score)] over the SAME
    question-horizons, resampled together: the pairing is what makes the
    comparison fair, and resampling the two sides independently would inflate
    the interval by comparing different question sets.

    Seeded, because an interval that moves between two runs of one dataset is
    not evidence of anything.
    """
    pool = [
        (float(a), float(b)) for a, b in paired
        if a is not None and b is not None
    ]
    if not pool:
        return None

    def _skill(rows: Sequence[tuple[float, float]]) -> float | None:
        ref = _mean([b for _, b in rows])
        model = _mean([a for a, _ in rows])
        if ref is None or model is None or ref <= 0:
            return None
        return 1.0 - (model / ref)

    point = _skill(pool)
    if point is None:
        return None

    rng = random.Random(seed)
    n = len(pool)
    draws: list[float] = []
    for _ in range(max(0, int(samples))):
        resample = [pool[rng.randrange(n)] for _ in range(n)]
        value = _skill(resample)
        if value is not None:
            draws.append(value)
    draws.sort()
    low = high = None
    if len(draws) >= 20:
        lo_i = int((alpha / 2) * (len(draws) - 1))
        hi_i = int((1 - alpha / 2) * (len(draws) - 1))
        low, high = draws[lo_i], draws[hi_i]
    return {
        "n_resolved": n,
        "skill": point,
        "ci_low": low,
        "ci_high": high,
        "ci_level": 1 - alpha,
        "bootstrap_samples": len(draws),
    }


def skill_claim(
    paired: Sequence[tuple[float, float]],
    *,
    min_resolved: int,
    samples: int,
    label: str = "",
) -> dict[str, Any]:
    """A skill figure the report is allowed to print, or a refusal.

    Below ``min_resolved`` the verdict is ``too few to say`` and no skill
    number is offered. That is the whole discipline: the report declines,
    rather than hedging in prose beside a number the reader will keep.
    """
    result = bootstrap_skill(paired, samples=samples)
    n = result["n_resolved"] if result else len(paired)
    base = {
        "label": label,
        "n_resolved": n,
        "min_resolved": int(min_resolved),
        "claim_allowed": False,
        "verdict": VERDICT_TOO_FEW,
        "skill": None,
        "ci_low": None,
        "ci_high": None,
    }
    if result is None or n < int(min_resolved):
        base["reason"] = (
            f"{n:,} resolved question-months is below the {int(min_resolved):,} "
            "this report requires before it will make a skill claim"
        )
        return base

    low, high = result.get("ci_low"), result.get("ci_high")
    if low is not None and high is not None:
        if low > 0:
            verdict = VERDICT_BETTER
        elif high < 0:
            verdict = VERDICT_WORSE
        else:
            verdict = VERDICT_UNCLEAR
    else:
        verdict = VERDICT_UNCLEAR
    base.update({
        "claim_allowed": True,
        "verdict": verdict,
        "skill": result["skill"],
        "ci_low": low,
        "ci_high": high,
        "bootstrap_samples": result.get("bootstrap_samples"),
    })
    return base


# ---------------------------------------------------------------------------
# The forecast diary
# ---------------------------------------------------------------------------


def diary(
    previous_reports: Iterable[Mapping[str, Any]],
    resolved: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """We flagged this; here is what happened.

    ``previous_reports`` are stored interpretations, newest first, each with a
    month label and its attention entries. ``resolved`` maps question_id to
    its resolution rows. Only flagged questions that have since resolved
    appear: the diary is the record of calls that can now be checked, and
    padding it with open ones would make it a second attention list.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for report in previous_reports:
        label = str(report.get("month_label") or report.get("run_id") or "")
        for entry in report.get("entries") or []:
            for qid in entry.get("question_ids") or []:
                qid = str(qid)
                rows = resolved.get(qid)
                if not rows or qid in seen:
                    continue
                seen.add(qid)
                observed = [
                    {
                        "horizon_m": r.get("horizon_m"),
                        "observed_month": r.get("observed_month"),
                        "value": r.get("value"),
                    }
                    for r in rows
                    if r.get("value") is not None
                ]
                if not observed:
                    continue
                out.append({
                    "report_month": label,
                    "question_id": qid,
                    "iso3": entry.get("iso3"),
                    "hazard_code": entry.get("hazard_code"),
                    "metric": entry.get("metric"),
                    "what_we_said": entry.get("why_it_stands_out"),
                    "observed": sorted(
                        observed, key=lambda r: int(r.get("horizon_m") or 0)
                    ),
                })
                if len(out) >= limit:
                    return out
    return out
