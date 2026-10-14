# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Sibyl question selection: regime-change picks plus controls.

There is no first-class "volatility" score in Pythia; the proxy is the
Regime Change score (``hs_triage.regime_change_score`` = likelihood x
magnitude), which measures "expected departure from the historical base
rate" (see DISCOVERY.md §1).

The rule since 2026-10-14 (``select_on_rc``, all knobs in sibyl/config.py):

1. a pick must carry an RC flag: ``regime_change_level`` at least
   ``MIN_RC_LEVEL`` (1), and at least ``HAZARD_MIN_RC_LEVEL`` for a hazard
   named there (cyclone needs 2);
2. rank RC level 2 and 3 first, then level 1, each by falling volatility;
3. FLOOR: each hazard takes its ``FLOOR_PER_HAZARD`` best (ACE 4, DR 2,
   FL 1, TC 0);
4. FILL: the best remaining candidate whose hazard is under its
   ``CAP_PER_HAZARD`` (ACE 7, DR 3, FL 2, TC 1), until the picks are full;
5. OVERFLOW: slots no hazard could fill go to the hazards in
   ``OVERFLOW_ORDER`` (ACE, then DR) beyond their cap;
6. CONTROLS: ``N_CONTROL`` (2) ACE/FATALITIES questions with no RC flag,
   drawn by hash.

Run order is controls, then floor picks, then fill picks, so a budget cut
removes fill picks and never the controls (owner decision). Sibyl went to 15
questions because the 13 October 2026 run cost $5.04 a question and hit its
cap after its 12 floor picks, skipping all five controls; a conflict focus
because that is where the scored record and the RC signal are richest.

The floor-then-fill rule of 2 to 13 October 2026 (``floor_then_fill``, 25
questions, a floor of 3 per hazard, FL and TC held at 3) is kept for replay
and behind ``SIBYL_SELECTION_RULE=floor_then_fill``.

Scope is strict: numeric affected/fatalities magnitude questions only
(``ELIGIBLE_HAZARD_METRICS``). Binary EVENT_OCCURRENCE questions are never
eligible and are never used as padding; a pool short of N is logged and the
run proceeds with fewer.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

from pythia.db.schema import connect
from pythia.test_mode import is_test_mode
from pythia.run_questions import in_run_clause, latest_hs_run_with_questions

from sibyl import config as _cfg
from sibyl.config import (
    ELIGIBLE_HAZARD_METRICS,
    MAX_PER_HAZARD,
    MAX_PER_HAZARD_OVERRIDES,
    MIN_PER_HAZARD,
    N_CONTROL,
    N_QUESTIONS,
)

SELECTION_FLOOR = "floor"
SELECTION_FILL = "fill"
SELECTION_CONTROL = "control"

# The control classes of 2 to 13 October 2026 (three conflict, two drought).
# Since 2026-10-14 the default is sibyl.config.CONTROL_CLASSES (ACE only).
LEGACY_CONTROL_CLASSES = (("ACE", "FATALITIES"), ("DR", "PHASE3PLUS_IN_NEED"))
CONTROL_CLASSES = LEGACY_CONTROL_CLASSES

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
    # hs_triage.regime_change_level (None when the hazard has no triage row).
    rc_level: Optional[int] = None

    @property
    def is_control(self) -> bool:
        return self.selection_pass == SELECTION_CONTROL

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
    """The most recent HS run that asked for active questions.

    Read through run_questions as well as the questions' origin: a test run
    in an epoch production has already opened owns none of its questions'
    rows, and the origin alone would name the production scan (Oct 2026).
    """
    own = con is None
    if own:
        con = connect(read_only=False)
    try:
        return latest_hs_run_with_questions(con)
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
    max_per_hazard_overrides: Optional[Dict[str, int]] = None,
) -> List[SibylQuestion]:
    """Choose up to *n* of *candidates* by the floor-then-fill rule.

    *max_per_hazard_overrides* sets a hazard's own cap (e.g. ``{"FL": 3}``);
    a hazard's floor never exceeds its cap.

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
    default_cap = max(int(max_per_hazard), 1)
    overrides = {str(k).upper(): max(int(v), 0) for k, v in (max_per_hazard_overrides or {}).items()}

    def cap_for(hz: str) -> int:
        return overrides.get(str(hz).upper(), default_cap)

    floor = max(int(min_per_hazard), 0)

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
        round_qs = [
            qs[rnd] for hz, qs in by_hazard.items()
            if len(qs) > rnd and rnd < cap_for(hz)
        ]
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
            if q.question_id not in chosen_ids and picks[q.hazard_code] < cap_for(q.hazard_code)
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


def _control_key(q: SibylQuestion) -> str:
    return hashlib.sha1(f"{q.hs_run_id}{q.question_id}".encode("utf-8")).hexdigest()


def draw_controls(
    candidates: Sequence[SibylQuestion],
    n: int,
    *,
    exclude: Sequence[str] = (),
    classes: Optional[Sequence[tuple]] = None,
) -> List[SibylQuestion]:
    """Draw *n* control questions: no RC flag, by hash.

    Pure (no DB). Eligible: candidates of *classes* (default the legacy pair
    ACE/FATALITIES and DR/PHASE3PLUS_IN_NEED) whose ``rc_level`` is 0 or None
    and whose id is not in *exclude*. Each class is ordered by SHA-1 of
    hs_run_id + question_id, so the draw is fixed for a run and blind to
    volatility. The quota is split round-robin in class order (3 conflict and
    2 drought at five over the legacy pair); a class short of its share is
    made up from the others. Returned in run order (class order, each in hash
    order), stamped ``control``.
    """
    n = max(int(n), 0)
    if n == 0:
        return []
    classes = tuple(tuple(c) for c in (classes or LEGACY_CONTROL_CLASSES))
    excluded = set(exclude)
    pools: Dict[tuple, List[SibylQuestion]] = {}
    for cls in classes:
        pools[cls] = sorted(
            (
                q for q in candidates
                if (q.hazard_code, q.metric) == cls
                and q.question_id not in excluded
                and not q.rc_level
            ),
            key=_control_key,
        )
    want = {cls: 0 for cls in classes}
    for i in range(n):
        want[classes[i % len(classes)]] += 1
    take = {cls: min(want[cls], len(pools[cls])) for cls in classes}
    short = n - sum(take.values())
    for cls in classes:
        if short <= 0:
            break
        extra = min(short, len(pools[cls]) - take[cls])
        take[cls] += extra
        short -= extra
    out: List[SibylQuestion] = []
    for cls in classes:
        for q in pools[cls][: take[cls]]:
            q.selection_pass = SELECTION_CONTROL
            out.append(q)
    return out


def _rc(q: SibylQuestion) -> int:
    return int(q.rc_level or 0)


def rc_eligible(
    q: SibylQuestion,
    *,
    min_rc_level: Optional[int] = None,
    hazard_min_rc_level: Optional[Dict[str, int]] = None,
) -> bool:
    """Whether *q* may be a pick: its RC level reaches the run's and its hazard's minimum."""
    base = int(_cfg.MIN_RC_LEVEL if min_rc_level is None else min_rc_level)
    per = _cfg.HAZARD_MIN_RC_LEVEL if hazard_min_rc_level is None else hazard_min_rc_level
    need = max(base, int((per or {}).get(q.hazard_code, base)))
    return _rc(q) >= need


def rc_rank_key(q: SibylQuestion) -> tuple:
    """RC level 2 and 3 first, then level 1, each by falling volatility."""
    return (0 if _rc(q) >= 2 else 1, -q.volatility_score, q.question_id)


def pool_breakdown(candidates: Sequence[SibylQuestion]) -> Dict[str, Dict[str, int]]:
    """Eligible pool by hazard and RC level (logged at selection time)."""
    out: Dict[str, Dict[str, int]] = {}
    for q in candidates:
        lvl = "none" if q.rc_level is None else str(int(q.rc_level))
        out.setdefault(q.hazard_code, {}).setdefault(lvl, 0)
        out[q.hazard_code][lvl] += 1
    return {hz: dict(sorted(v.items())) for hz, v in sorted(out.items())}


def select_on_rc(
    candidates: Sequence[SibylQuestion],
    n: int,
    *,
    floors: Optional[Dict[str, int]] = None,
    caps: Optional[Dict[str, int]] = None,
    overflow_order: Optional[Sequence[str]] = None,
    min_rc_level: Optional[int] = None,
    hazard_min_rc_level: Optional[Dict[str, int]] = None,
) -> List[SibylQuestion]:
    """Choose up to *n* regime-change picks (the 2026-10-14 rule). Pure.

    Returns floor picks then fill picks (each in rank order), stamped
    ``floor`` / ``fill``. A hazard missing from *caps* is not picked.
    """
    n = max(int(n), 0)
    if n == 0:
        return []
    floors = {k.upper(): int(v) for k, v in (_cfg.FLOOR_PER_HAZARD if floors is None else floors).items()}
    caps = {k.upper(): int(v) for k, v in (_cfg.CAP_PER_HAZARD if caps is None else caps).items()}
    overflow = [h.upper() for h in (_cfg.OVERFLOW_ORDER if overflow_order is None else overflow_order)]
    pool = sorted(
        (q for q in candidates
         if rc_eligible(q, min_rc_level=min_rc_level, hazard_min_rc_level=hazard_min_rc_level)),
        key=rc_rank_key,
    )
    by_hazard: Dict[str, List[SibylQuestion]] = {}
    for q in pool:
        by_hazard.setdefault(q.hazard_code, []).append(q)
    picks: Dict[str, int] = {hz: 0 for hz in by_hazard}
    chosen: set = set()
    floor_picks: List[SibylQuestion] = []
    fill_picks: List[SibylQuestion] = []

    def _take(q: SibylQuestion, how: str, into: List[SibylQuestion]) -> None:
        q.selection_pass = how
        picks[q.hazard_code] = picks.get(q.hazard_code, 0) + 1
        chosen.add(q.question_id)
        into.append(q)

    for hz in sorted(by_hazard):
        want = min(floors.get(hz, 0), caps.get(hz, 0))
        for q in by_hazard[hz][:want]:
            if len(floor_picks) >= n:
                break
            _take(q, SELECTION_FLOOR, floor_picks)

    def _room() -> int:
        return n - len(floor_picks) - len(fill_picks)

    for q in pool:
        if _room() <= 0:
            break
        if q.question_id in chosen or picks.get(q.hazard_code, 0) >= caps.get(q.hazard_code, 0):
            continue
        _take(q, SELECTION_FILL, fill_picks)
    for hz in overflow:
        for q in by_hazard.get(hz, []):
            if _room() <= 0:
                break
            if q.question_id not in chosen:
                _take(q, SELECTION_FILL, fill_picks)
    return sorted(floor_picks, key=rc_rank_key) + sorted(fill_picks, key=rc_rank_key)


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
        # The run's own question set and its own triage (run_questions): a
        # test run's questions may name a production scan as their origin.
        in_run, n_bind = in_run_clause(con, "q")
        sql = f"""
            SELECT
                q.question_id, ? AS hs_run_id, q.iso3,
                upper(q.hazard_code) AS hazard_code,
                upper(q.metric) AS metric,
                q.window_start_date, q.target_month, q.wording,
                COALESCE(t.regime_change_score, 0.0) AS volatility_score,
                COALESCE(t.triage_score, 0.0) AS triage_score,
                t.regime_change_level AS rc_level
            FROM questions q
            LEFT JOIN hs_triage t
              ON t.run_id = ?
             AND upper(t.iso3) = upper(q.iso3)
             AND upper(t.hazard_code) = upper(q.hazard_code)
            WHERE q.status = 'active'
              AND {in_run}
              AND {_eligibility_sql()}
              {test_filter}
            ORDER BY volatility_score DESC, q.question_id
        """
        rows = con.execute(sql, [run_id, run_id] + [run_id] * n_bind).fetchall()
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
            rc_level=(int(r[10]) if r[10] is not None else None),
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
    max_per_hazard_overrides: Optional[Dict[str, int]] = None,
    n_control: int = N_CONTROL,
    rule: Optional[str] = None,
) -> List[SibylQuestion]:
    """The run's Sibyl questions, in run order.

    ``n - n_control`` picks and ``n_control`` controls (drawn from the
    questions left). Under the default rule (``rc_focus``, since 2026-10-14)
    the run order is controls, floor picks, fill picks; under the legacy
    ``floor_then_fill`` rule it is floor, controls, fill. A run asking for no
    more than ``n_control`` questions takes no controls. The name is kept for
    the workflow gate and callers.
    """
    rule = (rule or _cfg.SELECTION_RULE or "rc_focus").strip().lower()
    overrides = MAX_PER_HAZARD_OVERRIDES if max_per_hazard_overrides is None else max_per_hazard_overrides
    n = max(int(n), 0)
    n_control = max(int(n_control), 0) if n > max(int(n_control), 0) else 0
    n_selected = n - n_control

    candidates = load_candidates(hs_run_id, con)
    breakdown = pool_breakdown(candidates)
    logger.info(
        "sibyl.select_questions: eligible pool by hazard and RC level for hs_run_id=%s: %s",
        hs_run_id or "(latest)", breakdown,
    )
    if rule == "floor_then_fill":
        chosen = floor_then_fill(
            candidates, n_selected,
            min_per_hazard=min_per_hazard, max_per_hazard=max_per_hazard,
            max_per_hazard_overrides=overrides,
        )
        control_classes = LEGACY_CONTROL_CLASSES
    else:
        chosen = select_on_rc(candidates, n_selected)
        control_classes = _cfg.CONTROL_CLASSES or LEGACY_CONTROL_CLASSES[:1]
    controls = draw_controls(
        candidates, n_control, exclude=[q.question_id for q in chosen],
        classes=control_classes,
    )
    floor_picks = [q for q in chosen if q.selection_pass == SELECTION_FLOOR]
    fill_picks = [q for q in chosen if q.selection_pass == SELECTION_FILL]
    if rule == "floor_then_fill":
        questions = floor_picks + controls + fill_picks
    else:
        questions = controls + floor_picks + fill_picks

    counts: Dict[str, int] = {}
    for q in chosen:
        counts[q.hazard_code] = counts.get(q.hazard_code, 0) + 1
    if len(chosen) < n_selected:
        # Loud but expected (small runs legitimately have < N eligible
        # questions), so WARNING not ERROR: proceed with what exists —
        # never pad with binary (EVENT_OCCURRENCE) questions.
        logger.warning(
            "sibyl.select_questions: only %d of %d requested picks could be chosen "
            "for hs_run_id=%s (rule %s, pool %d: %s); proceeding without padding.",
            len(chosen), n_selected, hs_run_id or "(latest)", rule, len(candidates), breakdown,
        )
    if len(controls) < n_control:
        logger.warning(
            "sibyl.select_questions: only %d of %d controls could be drawn "
            "(questions of %s with no RC flag, not already chosen)",
            len(controls), n_control, control_classes,
        )
    logger.info(
        "sibyl.select_questions: chose %d (%s) — %d control, %d floor, %d fill (rule %s)",
        len(questions),
        ", ".join(f"{hz} {c}" for hz, c in sorted(counts.items())) or "none",
        len(controls), len(floor_picks), len(fill_picks), rule,
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
        in_run, n_bind = in_run_clause(con, "q")
        rows = con.execute(
            f"""
            SELECT upper(q.hazard_code), upper(q.metric),
                   COUNT(*),
                   SUM(CASE WHEN COALESCE(q.is_test, FALSE) THEN 1 ELSE 0 END)
            FROM questions q
            WHERE q.status = 'active' AND {in_run}
            GROUP BY 1, 2
            ORDER BY 1, 2
            """,
            [run_id] * n_bind,
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
