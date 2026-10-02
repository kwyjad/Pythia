# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl question selection: floor, then fill.

Selects N affected/fatalities questions for a run, spread across the
hazards. There is no first-class "volatility" score in Pythia; the proxy is
the Regime Change score (``hs_triage.regime_change_score`` = likelihood x
magnitude), which measures "expected departure from the historical base
rate" (see DISCOVERY.md §1).

The rule (``floor_then_fill``):

1. rank candidates by volatility, highest first;
2. FLOOR: each hazard takes its ``MIN_PER_HAZARD`` most volatile questions;
3. FILL: take the most volatile remaining candidate whose hazard is under
   ``MAX_PER_HAZARD``, until N are chosen;
4. ties go to the hazard holding fewer picks so far, then to question_id.

A plain top-N let one hazard take the run: the 1 October 2026 run chose six
drought questions and no cyclone. The floor keeps every hazard in Sibyl's
scored record; the fill still spends most slots where the RC signal is
strongest. ``triage_score`` is NOT a tiebreak: every row with RC >= 0.1 is
tier ``rc_promoted`` and carries a placeholder 0, so the tiebreak was dead.

Scope is strict: numeric affected/fatalities magnitude questions only
(``ELIGIBLE_HAZARD_METRICS``). Binary EVENT_OCCURRENCE questions are never
eligible and are never used as padding — a hazard short of its floor takes
what it has, and a pool short of N is logged and the run proceeds with
fewer.

Run order is floor picks first, then fill picks, each by falling
volatility, so a budget or time cut removes fill picks first and the
hazard floor is the last thing a cut reaches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

from pythia.db.schema import connect
from pythia.test_mode import is_test_mode

from sibyl.config import (
    ELIGIBLE_HAZARD_METRICS,
    MAX_PER_HAZARD,
    MIN_PER_HAZARD,
    N_QUESTIONS,
)

SELECTION_FLOOR = "floor"
SELECTION_FILL = "fill"

logger = logging.getLogger(__name__)


@dataclass
class SibylQuestion:
    question_id: str
    hs_run_id: str
    iso3: str
    hazard_code: str
    metric: str
    window_start_date: Optional[date]
    target_month: str
    wording: str
    volatility_score: float
    triage_score: float
    # Which pass chose the question (SELECTION_FLOOR / SELECTION_FILL);
    # persisted on sibyl_forecasts.selection_pass.
    selection_pass: Optional[str] = None

    def to_row_dict(self) -> dict:
        """Shape compatible with forecaster month/window helpers."""
        return {
            "question_id": self.question_id,
            "hs_run_id": self.hs_run_id,
            "iso3": self.iso3,
            "hazard_code": self.hazard_code,
            "metric": self.metric,
            "window_start_date": self.window_start_date,
            "target_month": self.target_month,
            "wording": self.wording,
        }


def _eligibility_sql() -> str:
    clauses = [
        f"(upper(q.hazard_code) = '{hz}' AND upper(q.metric) = '{m}')"
        for hz, m in sorted(ELIGIBLE_HAZARD_METRICS)
    ]
    return "(" + " OR ".join(clauses) + ")"


def latest_hs_run_id(con: Any = None) -> Optional[str]:
    """The most recent HS run that produced active questions."""
    own = con is None
    if own:
        con = connect(read_only=False)
    try:
        row = con.execute(
            """
            SELECT q.hs_run_id
            FROM questions q
            LEFT JOIN hs_runs r ON r.hs_run_id = q.hs_run_id
            WHERE q.status = 'active' AND q.hs_run_id IS NOT NULL
            GROUP BY q.hs_run_id, r.generated_at
            ORDER BY r.generated_at DESC NULLS LAST, q.hs_run_id DESC
            LIMIT 1
            """
        ).fetchone()
        return str(row[0]) if row and row[0] else None
    finally:
        if own:
            con.close()


def hs_run_is_test(hs_run_id: Optional[str], con: Any = None) -> bool:
    """Whether *hs_run_id* is stamped ``is_test`` in ``hs_runs``.

    Sibyl chains off HS via ``workflow_run``, which cannot carry the upstream
    run's test-mode input — so test mode is derived from the DB instead.
    """
    if not hs_run_id:
        return False
    own = con is None
    if own:
        con = connect(read_only=False)
    try:
        row = con.execute(
            "SELECT COALESCE(is_test, FALSE) FROM hs_runs WHERE hs_run_id = ?",
            [hs_run_id],
        ).fetchone()
        return bool(row[0]) if row else False
    except Exception:
        return False
    finally:
        if own:
            con.close()


def floor_then_fill(
    candidates: Sequence[SibylQuestion],
    n: int,
    *,
    min_per_hazard: int = MIN_PER_HAZARD,
    max_per_hazard: int = MAX_PER_HAZARD,
) -> List[SibylQuestion]:
    """Choose up to *n* of *candidates* by the floor-then-fill rule.

    Pure (no DB). Returns the chosen questions in RUN order — floor picks
    first, then fill picks, each by falling volatility — with
    ``selection_pass`` stamped on each.

    When *n* is smaller than the floor would take (``n < hazards x min``),
    the floor is taken in rounds — every hazard's best, then every hazard's
    second best, ... — so as many hazards as *n* allows stay represented.
    """
    n = max(int(n), 0)
    if n == 0 or not candidates:
        return []
    cap = max(int(max_per_hazard), 1)
    floor = max(min(int(min_per_hazard), cap), 0)

    # One ordering everywhere: falling volatility, then question_id.
    ranked = sorted(candidates, key=lambda q: (-q.volatility_score, q.question_id))
    by_hazard: Dict[str, List[SibylQuestion]] = {}
    for q in ranked:
        by_hazard.setdefault(q.hazard_code, []).append(q)

    picks: Dict[str, int] = {hz: 0 for hz in by_hazard}
    chosen_ids: set = set()
    floor_picks: List[SibylQuestion] = []
    fill_picks: List[SibylQuestion] = []

    def _take(q: SibylQuestion, how: str, into: List[SibylQuestion]) -> None:
        q.selection_pass = how
        picks[q.hazard_code] += 1
        chosen_ids.add(q.question_id)
        into.append(q)

    # FLOOR, in rounds. Within a round, order by falling volatility; ties go
    # to the hazard holding fewer picks, then question_id.
    for rnd in range(floor):
        round_qs = [qs[rnd] for qs in by_hazard.values() if len(qs) > rnd]
        round_qs.sort(
            key=lambda q: (-q.volatility_score, picks[q.hazard_code], q.question_id)
        )
        for q in round_qs:
            if len(floor_picks) >= n:
                break
            _take(q, SELECTION_FLOOR, floor_picks)

    # FILL: the most volatile remaining candidate under its hazard's cap.
    while len(floor_picks) + len(fill_picks) < n:
        open_qs = [
            q for q in ranked
            if q.question_id not in chosen_ids and picks[q.hazard_code] < cap
        ]
        if not open_qs:
            break
        best = min(
            open_qs,
            key=lambda q: (-q.volatility_score, picks[q.hazard_code], q.question_id),
        )
        _take(best, SELECTION_FILL, fill_picks)

    order = lambda q: (-q.volatility_score, q.question_id)  # noqa: E731
    return sorted(floor_picks, key=order) + sorted(fill_picks, key=order)


def load_candidates(
    hs_run_id: Optional[str] = None,
    con: Any = None,
) -> List[SibylQuestion]:
    """Every eligible active question of the HS run, by falling volatility."""
    own = con is None
    if own:
        con = connect(read_only=False)
    try:
        run_id = hs_run_id or latest_hs_run_id(con)
        if not run_id:
            logger.error("sibyl.select_questions: no HS run with active questions found")
            return []

        # Run-aware test filter: a test-mode HS run stamps every question
        # is_test=TRUE, and the Sibyl workflow cannot inherit the upstream
        # test-mode env — filtering purely on env silently excluded ALL
        # questions of test runs (gate reported 0 eligible).
        run_is_test = hs_run_is_test(run_id, con)
        include_test = is_test_mode() or run_is_test
        test_filter = "" if include_test else "AND COALESCE(q.is_test, FALSE) = FALSE"
        if run_is_test and not is_test_mode():
            logger.warning(
                "sibyl.select_questions: HS run %s is a test run — including "
                "its is_test questions; Sibyl outputs should also be stamped "
                "is_test (see sibyl.run).",
                run_id,
            )
        sql = f"""
            SELECT
                q.question_id, q.hs_run_id, q.iso3,
                upper(q.hazard_code) AS hazard_code,
                upper(q.metric) AS metric,
                q.window_start_date, q.target_month, q.wording,
                COALESCE(t.regime_change_score, 0.0) AS volatility_score,
                COALESCE(t.triage_score, 0.0) AS triage_score
            FROM questions q
            LEFT JOIN hs_triage t
              ON t.run_id = q.hs_run_id
             AND upper(t.iso3) = upper(q.iso3)
             AND upper(t.hazard_code) = upper(q.hazard_code)
            WHERE q.status = 'active'
              AND q.hs_run_id = ?
              AND {_eligibility_sql()}
              {test_filter}
            ORDER BY volatility_score DESC, q.question_id
        """
        rows = con.execute(sql, [run_id]).fetchall()
    finally:
        if own:
            con.close()

    return [
        SibylQuestion(
            question_id=str(r[0]),
            hs_run_id=str(r[1]),
            iso3=str(r[2] or "").upper(),
            hazard_code=str(r[3] or "").upper(),
            metric=str(r[4] or "").upper(),
            window_start_date=r[5],
            target_month=str(r[6] or ""),
            wording=str(r[7] or ""),
            volatility_score=float(r[8] or 0.0),
            triage_score=float(r[9] or 0.0),
        )
        for r in rows
    ]


def select_top_questions(
    hs_run_id: Optional[str] = None,
    n: int = N_QUESTIONS,
    con: Any = None,
    *,
    min_per_hazard: int = MIN_PER_HAZARD,
    max_per_hazard: int = MAX_PER_HAZARD,
) -> List[SibylQuestion]:
    """The run's Sibyl questions by floor-then-fill, in run order.

    The name is kept for the workflow gate and callers; the rule is no
    longer a plain top-N (see the module docstring).
    """
    candidates = load_candidates(hs_run_id, con)
    questions = floor_then_fill(
        candidates, n,
        min_per_hazard=min_per_hazard, max_per_hazard=max_per_hazard,
    )

    counts: Dict[str, int] = {}
    for q in questions:
        counts[q.hazard_code] = counts.get(q.hazard_code, 0) + 1
    short = sorted(
        hz for hz in {hz for hz, _ in ELIGIBLE_HAZARD_METRICS}
        if counts.get(hz, 0) < min(min_per_hazard, max_per_hazard)
    )
    if short and n >= len(ELIGIBLE_HAZARD_METRICS) * min_per_hazard:
        logger.warning(
            "sibyl.select_questions: hazard(s) %s hold fewer than the floor "
            "of %d eligible questions for hs_run_id=%s; they take what they have.",
            ", ".join(short), min_per_hazard, hs_run_id or "(latest)",
        )
    if len(questions) < n:
        # Loud but expected (small runs legitimately have < N eligible
        # questions), so WARNING not ERROR: proceed with what exists —
        # never pad with binary (EVENT_OCCURRENCE) questions.
        logger.warning(
            "sibyl.select_questions: only %d of %d requested eligible "
            "affected/fatalities questions could be chosen for hs_run_id=%s "
            "(pool %d, per-hazard cap %d); proceeding without padding.",
            len(questions), n, hs_run_id or "(latest)", len(candidates),
            max_per_hazard,
        )
    logger.info(
        "sibyl.select_questions: chose %d (%s) — %d floor, %d fill",
        len(questions),
        ", ".join(f"{hz} {c}" for hz, c in sorted(counts.items())) or "none",
        sum(1 for q in questions if q.selection_pass == SELECTION_FLOOR),
        sum(1 for q in questions if q.selection_pass == SELECTION_FILL),
    )
    return questions


def eligibility_breakdown(
    hs_run_id: Optional[str] = None,
    con: Any = None,
) -> Dict[str, Any]:
    """Diagnostic counts explaining the gate outcome (logged by run_sibyl.yml).

    Returns per-(hazard, metric) active-question counts for the resolved HS
    run, plus how many rows the eligibility pair filter and the test filter
    would exclude — so a 0-eligible gate result is self-explanatory in logs.
    """
    own = con is None
    if own:
        con = connect(read_only=False)
    try:
        run_id = hs_run_id or latest_hs_run_id(con)
        if not run_id:
            return {"hs_run_id": None, "note": "no HS run with active questions"}

        run_is_test = hs_run_is_test(run_id, con)
        rows = con.execute(
            """
            SELECT upper(q.hazard_code), upper(q.metric),
                   COUNT(*),
                   SUM(CASE WHEN COALESCE(q.is_test, FALSE) THEN 1 ELSE 0 END)
            FROM questions q
            WHERE q.status = 'active' AND q.hs_run_id = ?
            GROUP BY 1, 2
            ORDER BY 1, 2
            """,
            [run_id],
        ).fetchall()
        pairs = {
            f"{hz}/{m}": {
                "active": int(cnt),
                "is_test": int(test_cnt or 0),
                "pair_eligible": (hz, m) in ELIGIBLE_HAZARD_METRICS,
            }
            for hz, m, cnt, test_cnt in rows
        }
        return {
            "hs_run_id": run_id,
            "run_is_test": run_is_test,
            "env_test_mode": is_test_mode(),
            "eligible_pairs": sorted(f"{hz}/{m}" for hz, m in ELIGIBLE_HAZARD_METRICS),
            "pairs": pairs,
        }
    finally:
        if own:
            con.close()
