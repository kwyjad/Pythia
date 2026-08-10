# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""What changed since last month, done properly.

The old "what changed" section admitted its own matching was broken. Two
faults: it compared country and hazard without the metric, so a country's
drought fatalities question could be matched against its drought
people-affected question; and it could say a risk had entered or left without
saying anything about how the ones that stayed were developing.

Three things a reader actually wants, and all three are here:

* **Persistence.** A risk flagged three months running is a different object
  from one flagged today. The count comes from the reports themselves, which
  is the only honest source: what was flagged is what the report said, not
  what a threshold would say about an old run if re-run under today's rules.
* **Movement, on the months the two runs share.** Consecutive runs overlap by
  five of their six months, so the comparison is made on those and no others.
  Comparing a whole window against a whole window would read a shift in the
  window as a change in the forecast.
* **Why something dropped out.** A risk that disappears without explanation
  reads as an error. It usually fell below a threshold, and saying so costs
  one clause.

Pure functions over plain values.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

MOVEMENT_RISEN = "risen"
MOVEMENT_HELD = "held"
MOVEMENT_FALLEN = "fallen"

DROP_NO_QUESTION = "no question was generated for it this month"
DROP_BELOW_GATE = "it no longer passes the selection tests"
DROP_NOT_SHOWN = "it still passes the tests but ranks below the entries shown"


def match_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """The key two runs are compared on.

    Question ids carry an epoch suffix and never match across runs by design,
    so the identity of a risk is the country, the hazard AND the metric. The
    metric is the part that used to be missing.
    """
    return (
        str(row.get("iso3") or "").upper(),
        str(row.get("hazard_code") or "").upper(),
        str(row.get("metric") or "").upper(),
    )


def consecutive_runs(
    key: tuple[str, str, str],
    previous_flag_sets: Sequence[Iterable[tuple[str, str, str]]],
) -> int:
    """How many runs in a row this risk has been flagged, including this one.

    ``previous_flag_sets`` are the flagged keys of earlier reports, NEWEST
    FIRST. The count stops at the first report that did not flag it: a risk
    flagged in June and August but not July has been flagged for one run, not
    two, and calling that persistence would be a lie about a gap.
    """
    count = 1
    for flagged in previous_flag_sets:
        if key in set(flagged):
            count += 1
        else:
            break
    return count


def persistence_phrase(runs: int) -> str:
    """The words the report prints for a persistence count."""
    if runs <= 1:
        return "new this month"
    return f"flagged for {runs} consecutive runs"


def movement(
    previous_by_month: Mapping[str, float],
    current_by_month: Mapping[str, float],
    *,
    tolerance: float = 0.15,
) -> dict[str, Any] | None:
    """How the planning figure moved, on the months the two runs share.

    Returns None when the runs share no month, which is the honest answer for
    a question whose window has moved past the previous one entirely.
    """
    months = sorted(set(previous_by_month) & set(current_by_month))
    if not months:
        return None
    prev_values = [float(previous_by_month[m]) for m in months]
    cur_values = [float(current_by_month[m]) for m in months]
    prev_mean = sum(prev_values) / len(prev_values)
    cur_mean = sum(cur_values) / len(cur_values)

    if prev_mean <= 0 and cur_mean <= 0:
        verdict = MOVEMENT_HELD
    elif prev_mean <= 0:
        verdict = MOVEMENT_RISEN if cur_mean > 0 else MOVEMENT_HELD
    elif cur_mean >= prev_mean * (1.0 + tolerance):
        verdict = MOVEMENT_RISEN
    elif cur_mean <= prev_mean * (1.0 - tolerance):
        verdict = MOVEMENT_FALLEN
    else:
        verdict = MOVEMENT_HELD
    return {
        "overlapping_months": months,
        "n_overlapping_months": len(months),
        "previous_mean": round(prev_mean, 1),
        "current_mean": round(cur_mean, 1),
        "movement": verdict,
    }


def drop_reason(
    current_row: Mapping[str, Any] | None,
    *,
    shown_question_ids: Iterable[str] = (),
) -> str:
    """Why a previously flagged risk is not an entry this month."""
    if current_row is None:
        return DROP_NO_QUESTION
    if not current_row.get("gate"):
        return DROP_BELOW_GATE
    if str(current_row.get("question_id") or "") in set(shown_question_ids):
        return ""  # still shown; not a drop at all
    return DROP_NOT_SHOWN
