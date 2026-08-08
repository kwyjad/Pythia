# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Which forecasts the report talks about, and under which heading.

Two questions a reader has, kept apart:

* **Potentially worsening situations** — where the forecast sits above what
  history would lead you to expect. A forecast BELOW its base rate is not
  news, so it cannot enter this section at all.
* **Major impact but roughly stable** — where the expected burden per head
  of population is large, but the forecast is not calling a departure from
  the usual. These are the places already in trouble.

Each is split into climate hazards (drought, flood, tropical cyclone) and
conflict, because the reasoning a reader needs differs.

Selection inside each of the four boxes is a dual gate: the top
``min_per_entries`` always appear, however unremarkable the month, so the
section is never empty and always names the month's worst; beyond that, an
entry appears only if it clears the threshold. A cap keeps the report short.

Pure functions over plain dicts — no DB, no IO — so the thresholds are
testable without a pack.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

CATEGORY_WORSENING = "worsening"
CATEGORY_STABLE_MAJOR = "stable_major"

CATEGORY_LABELS = {
    CATEGORY_WORSENING: "Potentially worsening situations",
    CATEGORY_STABLE_MAJOR: "Major impact but roughly stable",
}

# Section order in the report: what is changing comes before what is merely
# large, and within each, climate before conflict.
SECTION_ORDER = (
    (CATEGORY_WORSENING, "climate"),
    (CATEGORY_WORSENING, "conflict"),
    (CATEGORY_STABLE_MAJOR, "climate"),
    (CATEGORY_STABLE_MAJOR, "conflict"),
)


def ev_multiple(log_ev_ratio: float | None) -> float | None:
    """exp(log_ev_ratio): how many times the historical expectation.

    2.0 reads as "twice what history would suggest"; 0.5 as "half".
    """
    if log_ev_ratio is None:
        return None
    try:
        return math.exp(float(log_ev_ratio))
    except (TypeError, ValueError, OverflowError):
        return None


def direction(log_ev_ratio: float | None, *, deadband: float = 0.05) -> str | None:
    """``above`` | ``below`` | ``at``. None when there is no anchor at all.

    The deadband keeps forecasts that merely round off the base rate out of
    the worsening section; ~5% either way is noise, not a signal.
    """
    if log_ev_ratio is None:
        return None
    value = float(log_ev_ratio)
    if value > deadband:
        return "above"
    if value < -deadband:
        return "below"
    return "at"


def _sorted_by(rows: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Descending on ``key``, Nones last, question_id breaking ties so the
    ordering is stable across runs."""
    candidates = [r for r in rows if r.get(key) is not None]
    candidates.sort(key=lambda r: (-float(r[key]), str(r.get("question_id") or "")))
    return candidates


def select_worsening(
    rows: Iterable[dict[str, Any]],
    *,
    threshold_multiple: float,
    min_entries: int,
    max_entries: int,
) -> list[dict[str, Any]]:
    """Above-base-rate entries for one hazard family, best first.

    Ranked by how far ABOVE the base rate the forecast sits, which is
    log_ev_ratio: a signed quantity, so ranking on it cannot smuggle in a
    forecast that diverges by being far BELOW its anchor (which the
    undirected divergence score would happily do).
    """
    above = [r for r in rows if r.get("direction") == "above"]
    ordered = _sorted_by(above, "log_ev_ratio")
    threshold = math.log(threshold_multiple) if threshold_multiple > 0 else 0.0
    picked: list[dict[str, Any]] = []
    for index, row in enumerate(ordered):
        if len(picked) >= max_entries:
            break
        clears = float(row["log_ev_ratio"]) >= threshold
        if index < min_entries or clears:
            picked.append(row)
    return picked


def select_stable_major(
    rows: Iterable[dict[str, Any]],
    *,
    exclude_question_ids: set[str],
    max_entries: int,
    per_capita_floor: float,
    threshold_multiple: float,
) -> list[dict[str, Any]]:
    """Large per-capita burden that is NOT being called as worsening.

    The nominal floor is the existing per-capita gate: without it the
    ordering returns the same handful of very small states every cycle.
    Anything already named as worsening is excluded, so no country appears
    twice under two headings.

    There is no second threshold here. "Large impact" is a ranking question,
    the per-capita floor is already the gate, and a top-N over a ranking
    satisfies the same "always show the worst few" guarantee the worsening
    sections get from their minimum.
    """
    threshold = math.log(threshold_multiple) if threshold_multiple > 0 else 0.0
    eligible = [
        r for r in rows
        if str(r.get("question_id")) not in exclude_question_ids
        and r.get("eiv_per_100k") is not None
        and (r.get("eiv_nominal") or 0) >= per_capita_floor
        # A forecast that clears the worsening threshold is never described as
        # "roughly stable" just because the cap crowded it out of the section
        # above. It is simply left out of both.
        and not (
            r.get("direction") == "above"
            and r.get("log_ev_ratio") is not None
            and float(r["log_ev_ratio"]) >= threshold
        )
    ]
    return _sorted_by(eligible, "eiv_per_100k")[:max_entries]


def assign_categories(
    rows: list[dict[str, Any]],
    *,
    threshold_multiple: float,
    min_entries: int,
    max_entries: int,
    per_capita_floor: float,
) -> list[dict[str, Any]]:
    """Stamp ``category`` and ``category_rank`` on the rows the report covers.

    Mutates and returns ``rows`` (every row keeps its place in the index;
    only the selected ones gain a category), so the pack still carries the
    full picture and the selection stays auditable.
    """
    for row in rows:
        row.setdefault("category", None)
        row.setdefault("category_rank", None)

    selected_ids: set[str] = set()
    for family in ("climate", "conflict"):
        family_rows = [r for r in rows if r.get("hazard_family") == family]
        worsening = select_worsening(
            family_rows,
            threshold_multiple=threshold_multiple,
            min_entries=min_entries,
            max_entries=max_entries,
        )
        for rank, row in enumerate(worsening, start=1):
            row["category"] = CATEGORY_WORSENING
            row["category_rank"] = rank
            selected_ids.add(str(row.get("question_id")))

    for family in ("climate", "conflict"):
        family_rows = [r for r in rows if r.get("hazard_family") == family]
        stable = select_stable_major(
            family_rows,
            exclude_question_ids=selected_ids,
            max_entries=max_entries,
            per_capita_floor=per_capita_floor,
            threshold_multiple=threshold_multiple,
        )
        for rank, row in enumerate(stable, start=1):
            row["category"] = CATEGORY_STABLE_MAJOR
            row["category_rank"] = rank
            selected_ids.add(str(row.get("question_id")))

    return rows


def selected_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The categorised rows, in report order."""
    order = {pair: i for i, pair in enumerate(SECTION_ORDER)}
    picked = [r for r in rows if r.get("category")]
    picked.sort(
        key=lambda r: (
            order.get((r.get("category"), r.get("hazard_family")), 99),
            r.get("category_rank") or 99,
        )
    )
    return picked
