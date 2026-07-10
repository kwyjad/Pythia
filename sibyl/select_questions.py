# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl question selection.

Selects the top-N highest-volatility affected/fatalities questions for a
run. There is no first-class "volatility" score in Pythia; the proxy is the
Regime Change score (``hs_triage.regime_change_score`` = likelihood x
magnitude), which measures exactly "expected departure from the historical
base rate" — with ``triage_score`` as tiebreak and ``question_id`` for
determinism (see DISCOVERY.md §1).

Scope is strict: numeric affected/fatalities magnitude questions only
(``ELIGIBLE_HAZARD_METRICS``). Binary EVENT_OCCURRENCE questions are never
eligible and are never used as padding — when fewer than N questions
qualify the shortfall is logged loudly and the run proceeds with what
exists.

Questions are returned in DESCENDING volatility order so that when the run
hard cap fires mid-cycle, the questions left unforecast are the least
volatile — the cap sacrifices the lowest-value work first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

from pythia.db.schema import connect
from pythia.test_mode import is_test_mode

from sibyl.config import ELIGIBLE_HAZARD_METRICS, N_QUESTIONS

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


def select_top_questions(
    hs_run_id: Optional[str] = None,
    n: int = N_QUESTIONS,
    con: Any = None,
) -> List[SibylQuestion]:
    """Top-*n* eligible questions for *hs_run_id*, descending volatility."""
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
            ORDER BY volatility_score DESC, triage_score DESC, q.question_id
            LIMIT {int(n)}
        """
        rows = con.execute(sql, [run_id]).fetchall()
    finally:
        if own:
            con.close()

    questions = [
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

    if len(questions) < n:
        # Fail loudly, proceed with what exists — never pad with binary
        # (EVENT_OCCURRENCE) questions.
        logger.error(
            "sibyl.select_questions: only %d of %d requested eligible "
            "affected/fatalities questions exist for hs_run_id=%s; "
            "proceeding without padding.",
            len(questions), n, hs_run_id or "(latest)",
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
