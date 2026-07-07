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
from typing import Any, List, Optional

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

        test_filter = "" if is_test_mode() else "AND COALESCE(q.is_test, FALSE) = FALSE"
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
