# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Question-selection tests, including the test-mode HS run regression.

A test-mode HS pipeline stamps every question is_test=TRUE. Sibyl chains off
HS via workflow_run (which cannot carry the upstream test_mode input), so its
selector must derive test-run status from hs_runs.is_test — filtering purely
on the PYTHIA_TEST_MODE env silently excluded ALL questions of test runs and
the gate reported 0 eligible (observed 2026-07-09).
"""

from __future__ import annotations

import pytest

pytest.importorskip("duckdb")

from tests.sibyl_test_utils import HS_RUN_ID, Q1, Q2, seed_db


def _mark_run_as_test(db_url: str) -> None:
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        con.execute(
            "UPDATE hs_runs SET is_test = TRUE WHERE hs_run_id = ?", [HS_RUN_ID]
        )
        con.execute(
            "UPDATE questions SET is_test = TRUE WHERE hs_run_id = ?", [HS_RUN_ID]
        )
    finally:
        con.close()


def test_select_top_questions_orders_by_volatility(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)
    from sibyl.select_questions import select_top_questions

    qs = select_top_questions(HS_RUN_ID)
    assert [q.question_id for q in qs] == [Q1, Q2]


def test_test_mode_run_questions_still_selected_without_env(tmp_path, monkeypatch):
    """Regression: is_test questions of a test HS run must not be filtered out."""
    db_url = seed_db(tmp_path, monkeypatch)
    _mark_run_as_test(db_url)
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)

    from sibyl.select_questions import hs_run_is_test, select_top_questions

    assert hs_run_is_test(HS_RUN_ID) is True
    qs = select_top_questions(HS_RUN_ID)
    assert [q.question_id for q in qs] == [Q1, Q2]


def test_non_test_run_still_excludes_stray_test_questions(tmp_path, monkeypatch):
    db_url = seed_db(tmp_path, monkeypatch)
    from pythia.db.schema import connect

    con = connect(read_only=False)
    try:
        con.execute("UPDATE questions SET is_test = TRUE WHERE question_id = ?", [Q2])
    finally:
        con.close()
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)

    from sibyl.select_questions import select_top_questions

    qs = select_top_questions(HS_RUN_ID)
    assert [q.question_id for q in qs] == [Q1]


def test_run_sibyl_enables_test_mode_for_test_runs(tmp_path, monkeypatch):
    """run_sibyl must export PYTHIA_TEST_MODE when the target run is a test run."""
    db_url = seed_db(tmp_path, monkeypatch)
    _mark_run_as_test(db_url)
    monkeypatch.delenv("PYTHIA_TEST_MODE", raising=False)
    # Zero-question budget so run_sibyl resolves the run, sets the env, and
    # exits without any model/network calls.
    from sibyl.run import run_sibyl

    import os

    run_sibyl(HS_RUN_ID, n_questions=0)
    assert os.environ.get("PYTHIA_TEST_MODE") == "1"


def test_eligibility_breakdown_reports_pairs(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)
    from sibyl.select_questions import eligibility_breakdown

    info = eligibility_breakdown(HS_RUN_ID)
    assert info["hs_run_id"] == HS_RUN_ID
    assert info["run_is_test"] is False
    assert info["pairs"]["ACE/FATALITIES"]["active"] == 2
    assert info["pairs"]["ACE/FATALITIES"]["pair_eligible"] is True


# --- Floor-then-fill (Oct 2026) ---------------------------------------------
#
# The rule is a pure function (floor_then_fill), so most of it is tested on
# synthetic candidates. The shape test reads the three real candidate pools
# the rule was argued from (tests/fixtures/sibyl_candidate_pools.json).

import collections
import json
from pathlib import Path

from sibyl.select_questions import (
    SELECTION_FILL,
    SELECTION_FLOOR,
    SibylQuestion,
    floor_then_fill,
)

_HAZARD_METRIC = {
    "ACE": "FATALITIES", "DR": "PHASE3PLUS_IN_NEED", "FL": "PA", "TC": "PA",
}


def _q(qid: str, hazard: str, vol: float, iso3: str = "") -> SibylQuestion:
    return SibylQuestion(
        question_id=qid, hs_run_id="hs_x", iso3=iso3 or qid[:3],
        hazard_code=hazard, metric=_HAZARD_METRIC[hazard],
        window_start_date=None, target_month="2026-11", wording="",
        volatility_score=vol, triage_score=0.0,
    )


def _pool(spec: dict) -> list:
    """spec: {hazard: [vol, vol, ...]} -> candidates with stable ids."""
    out = []
    for hz, vols in spec.items():
        for i, v in enumerate(vols):
            out.append(_q(f"{hz}{i:02d}_{hz}", hz, v, iso3=f"C{hz}{i:02d}"))
    return out


def _counts(qs) -> dict:
    return dict(collections.Counter(q.hazard_code for q in qs))


def test_floor_gives_every_hazard_three_even_when_their_scores_are_low():
    pool = _pool({
        "DR": [0.9] * 20,          # one hazard dominates the scores
        "ACE": [0.05] * 5,
        "FL": [0.04] * 5,
        "TC": [0.01] * 5,
    })
    qs = floor_then_fill(pool, 25, min_per_hazard=3, max_per_hazard=10)
    counts = _counts(qs)
    assert len(qs) == 25
    for hz in ("ACE", "FL", "TC"):
        assert counts[hz] >= 3
    assert counts["DR"] == 10  # the cap, not the 20 it could have taken


def test_cap_holds_and_fill_takes_the_most_volatile_under_it():
    pool = _pool({
        "DR": [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.44, 0.43],
        "ACE": [0.42, 0.41, 0.40, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33],
        "FL": [0.3, 0.2, 0.1, 0.09],
        "TC": [0.3, 0.2, 0.1, 0.09],
    })
    qs = floor_then_fill(pool, 25, min_per_hazard=3, max_per_hazard=10)
    counts = _counts(qs)
    assert counts["DR"] == 10
    assert max(counts.values()) <= 10
    chosen = {q.question_id for q in qs}
    # DR's two above-cap questions (0.44, 0.43) are passed over for lower ones.
    assert "DR10_DR" not in chosen and "DR11_DR" not in chosen
    # Every chosen fill pick is at least as volatile as every unchosen
    # candidate whose hazard still had room.
    room = {hz for hz, c in counts.items() if c < 10}
    fills = [q for q in qs if q.selection_pass == SELECTION_FILL]
    left = [q for q in pool if q.question_id not in chosen and q.hazard_code in room]
    if fills and left:
        assert min(q.volatility_score for q in fills) >= max(q.volatility_score for q in left)


def test_run_order_is_floor_first_then_fill_each_by_falling_volatility():
    pool = _pool({
        "DR": [0.9, 0.8, 0.7, 0.6, 0.5],
        "ACE": [0.4, 0.3, 0.2, 0.15],
        "FL": [0.12, 0.11, 0.10],
        "TC": [0.03, 0.02, 0.01],
    })
    qs = floor_then_fill(pool, 14, min_per_hazard=3, max_per_hazard=10)
    passes = [q.selection_pass for q in qs]
    n_floor = passes.count(SELECTION_FLOOR)
    assert n_floor == 12
    assert passes == [SELECTION_FLOOR] * 12 + [SELECTION_FILL] * 2
    floor_vols = [q.volatility_score for q in qs[:12]]
    fill_vols = [q.volatility_score for q in qs[12:]]
    assert floor_vols == sorted(floor_vols, reverse=True)
    assert fill_vols == sorted(fill_vols, reverse=True)
    # A low-volatility floor pick (TC at 0.01) runs BEFORE a higher fill pick
    # (DR at 0.6): a cut removes fill picks first.
    assert qs[11].hazard_code == "TC" and qs[12].volatility_score == 0.6


def test_ties_go_to_the_hazard_with_fewer_picks_then_question_id():
    pool = _pool({
        "DR": [0.9, 0.9, 0.9],
        "FL": [0.8, 0.8, 0.8, 0.5],
        "TC": [0.7, 0.5],
    })
    qs = floor_then_fill(pool, 9, min_per_hazard=3, max_per_hazard=10)
    fills = [q for q in qs if q.selection_pass == SELECTION_FILL]
    # Floor: DR x3, FL x3, TC x2 (all it has) = 8. One fill slot, one candidate.
    assert [q.question_id for q in fills] == ["FL03_FL"]

    # Equal volatility within a floor round: question_id decides.
    tie = [
        _q("BBB_FL", "FL", 0.5), _q("AAA_FL", "FL", 0.5),
        _q("ZZZ_TC", "TC", 0.5),
        _q("DR1", "DR", 0.9), _q("DR2", "DR", 0.9),
    ]
    # floor=1: DR1, then AAA_FL and ZZZ_TC (tied at 0.5, both hazards on 0
    # picks, so id order). Fill: DR2 (0.9), then BBB_FL.
    qs2 = floor_then_fill(tie, 5, min_per_hazard=1, max_per_hazard=10)
    assert [q.question_id for q in qs2] == ["DR1", "AAA_FL", "ZZZ_TC", "DR2", "BBB_FL"]

    # Same volatility, different pick counts: the hazard with fewer picks wins
    # even though its id sorts later.
    tie2 = [
        _q("AAA_FL", "FL", 0.9), _q("AAB_FL", "FL", 0.4),
        _q("ZZZ_TC", "TC", 0.4),
    ]
    # floor=0, cap 10, n=2: AAA_FL (0.9) first; then AAB_FL vs ZZZ_TC tie at
    # 0.4 -> TC holds 0 picks, FL holds 1 -> ZZZ_TC.
    qs3 = floor_then_fill(tie2, 2, min_per_hazard=0, max_per_hazard=10)
    assert [q.question_id for q in qs3] == ["AAA_FL", "ZZZ_TC"]

    # Determinism: input order does not matter.
    import random

    shuffled = list(pool)
    random.Random(7).shuffle(shuffled)
    again = floor_then_fill(shuffled, 9, min_per_hazard=3, max_per_hazard=10)
    assert [q.question_id for q in again] == [q.question_id for q in qs]


def test_thin_pool_runs_short_and_never_pads():
    pool = _pool({"ACE": [0.5, 0.4], "DR": [0.3], "FL": [], "TC": [0.2]})
    qs = floor_then_fill(pool, 25, min_per_hazard=3, max_per_hazard=10)
    assert len(qs) == 4
    assert {q.question_id for q in qs} == {q.question_id for q in pool}
    # The cap also limits a thin pool: one hazard cannot exceed it to pad.
    one = _pool({"ACE": [0.5] * 15})
    assert len(floor_then_fill(one, 25, min_per_hazard=3, max_per_hazard=10)) == 10


def test_small_n_keeps_hazards_represented_by_taking_the_floor_in_rounds():
    pool = _pool({
        "DR": [0.9, 0.8, 0.7], "ACE": [0.6, 0.5, 0.4],
        "FL": [0.3, 0.2, 0.1], "TC": [0.05, 0.04, 0.03],
    })
    qs = floor_then_fill(pool, 4, min_per_hazard=3, max_per_hazard=10)
    assert _counts(qs) == {"DR": 1, "ACE": 1, "FL": 1, "TC": 1}
    assert floor_then_fill(pool, 0) == []


def _fixture_pools() -> dict:
    path = Path(__file__).parent / "fixtures" / "sibyl_candidate_pools.json"
    raw = json.loads(path.read_text())["runs"]
    return {
        run: [
            SibylQuestion(
                question_id=r["question_id"], hs_run_id=run, iso3=r["iso3"],
                hazard_code=r["hazard_code"], metric=r["metric"],
                window_start_date=None, target_month="", wording="",
                volatility_score=float(r["volatility_score"]), triage_score=0.0,
            )
            for r in rows
        ]
        for run, rows in raw.items()
    }


def test_shape_on_three_production_pools():
    """Pins the shape the rule gives on the 1 Aug, 15 Sep and 1 Oct 2026 runs.

    25 questions in 22-24 countries; conflict and drought 6-10 each, flood
    and cyclone 3-6 each; every hazard at its floor.
    """
    expected = {
        "hs_20260801T025754": {"ACE": 10, "DR": 6, "FL": 3, "TC": 6},
        "hs_20260915T130009": {"ACE": 7, "DR": 10, "FL": 5, "TC": 3},
        "hs_20261001T045127": {"ACE": 6, "DR": 10, "FL": 6, "TC": 3},
    }
    for run, pool in _fixture_pools().items():
        qs = floor_then_fill(pool, 25, min_per_hazard=3, max_per_hazard=10)
        counts = _counts(qs)
        assert len(qs) == 25, run
        assert counts == expected[run], (run, counts)
        assert 6 <= counts["ACE"] <= 10 and 6 <= counts["DR"] <= 10, run
        assert 3 <= counts["FL"] <= 6 and 3 <= counts["TC"] <= 6, run
        assert 22 <= len({q.iso3 for q in qs}) <= 24, run
        assert sum(q.selection_pass == SELECTION_FLOOR for q in qs) == 12, run

    # The pool limits how selective 25 can be: in August only 18 eligible
    # questions had RC >= 0.1, so 8 of the 25 picks fall below it.
    aug = floor_then_fill(_fixture_pools()["hs_20260801T025754"], 25)
    assert sum(q.volatility_score < 0.1 for q in aug) == 8


def test_shape_on_three_production_pools_under_the_october_rule():
    """The rule since Oct 2026: 20 by floor-then-fill with flood and cyclone
    held at their floor of 3 (the five controls are drawn separately).

    The fixture carries no RC level, so the control draw is tested on
    synthetic pools below rather than replayed here.
    """
    expected = {
        "hs_20260801T025754": {"ACE": 10, "DR": 4, "FL": 3, "TC": 3},
        "hs_20260915T130009": {"ACE": 6, "DR": 8, "FL": 3, "TC": 3},
        "hs_20261001T045127": {"ACE": 4, "DR": 10, "FL": 3, "TC": 3},
    }
    for run, pool in _fixture_pools().items():
        qs = floor_then_fill(
            pool, 20, min_per_hazard=3, max_per_hazard=10,
            max_per_hazard_overrides={"FL": 3, "TC": 3},
        )
        assert len(qs) == 20, run
        assert _counts(qs) == expected[run], (run, _counts(qs))
        assert sum(q.selection_pass == SELECTION_FLOOR for q in qs) == 12, run
        assert 18 <= len({q.iso3 for q in qs}) <= 19, run
    # Fewer slots below RC 0.1 than under the old 25: August drops from 8 to 3.
    aug = floor_then_fill(
        _fixture_pools()["hs_20260801T025754"], 20,
        max_per_hazard_overrides={"FL": 3, "TC": 3},
    )
    assert sum(q.volatility_score < 0.1 for q in aug) == 3


def test_selection_pass_is_persisted(tmp_path, monkeypatch):
    seed_db(tmp_path, monkeypatch)
    from sibyl.select_questions import select_top_questions

    qs = select_top_questions(HS_RUN_ID)
    assert {q.selection_pass for q in qs} == {SELECTION_FLOOR}
