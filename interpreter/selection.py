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

**Which box a row belongs in is decided by the gate**, not here:
`gating.gate_rows` stamps every row `larger than usual`, `heavy burden`,
`unusual, small scale` or nothing, and this module only orders the gated rows
and cuts them to length. That is why a worsening entry can no longer be
relabelled "roughly stable" when a cap crowds it out: a row has one gate, and
the two sections read different gates.

**Two lists, two orderings, each in the units that suit its purpose.** The
worsening list answers "what has changed", so it is ordered by the change:
how far the planning and contingency figures moved above the anchor
(`gating.movement_rank_key`, in people or deaths). The stable-major list
answers "where is the need", so it is ordered by the expected burden itself,
because "already in trouble" is a statement about size and not about change.
Ranking both by one number is what let a cyclone question carrying most of
its weight on "nobody affected" lead a report. Thin anchors are demoted in
both, never dropped.

The report carries at most `max_entries` entries in total, split between the
two categories. Fifteen pages with twelve entries is not a briefing, and the
sixth-most-notable forecast in a month has never changed anyone's plan.

Pure functions over plain dicts — no DB, no IO — so the policy is testable
without a pack.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from interpreter import gating

# How the total entry budget splits between the two categories. Worsening
# leads because a change is news and a standing crisis is not, but the split
# is deliberately not all-or-nothing: a report with no standing crises in it
# would mislead a reader about where the need actually is.
WORSENING_SHARE = 0.6

CATEGORY_WORSENING = "worsening"
CATEGORY_STABLE_MAJOR = "stable_major"

CATEGORY_LABELS = {
    CATEGORY_WORSENING: "Potentially worsening situations",
    CATEGORY_STABLE_MAJOR: "Major impact but roughly stable",
}

# What each section is ordered by, printed in its own heading. A reader who
# can see the ordering can check it; a reader who cannot has to take the
# sequence on trust.
CATEGORY_ORDERINGS = {
    CATEGORY_WORSENING: "ranked by how far the planning figures have risen",
    CATEGORY_STABLE_MAJOR: "ranked by the expected number of people affected",
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
    max_entries: int,
) -> list[dict[str, Any]]:
    """The rows the gate called ``larger than usual``, biggest movement first.

    Membership is the gate's, so this cannot admit a forecast sitting BELOW
    its anchor (clearing a positive movement threshold is the direction test)
    nor one whose planning figures barely moved. Ordering is the movement
    itself, with thin anchors demoted: `gating.movement_rank_key`.
    """
    picked = [r for r in rows if r.get("gate") == gating.GATE_WORSENING]
    picked.sort(key=gating.movement_rank_key)
    return picked[:max(0, int(max_entries))]


def select_stable_major(
    rows: Iterable[dict[str, Any]],
    *,
    exclude_question_ids: set[str],
    max_entries: int,
) -> list[dict[str, Any]]:
    """The rows the gate called ``heavy burden``, largest first.

    These are the places already in trouble: material enough to mobilise
    against, without a call that things are departing from the usual. Ordered
    by the expected burden itself rather than by excess, because the reason a
    reader needs them is the size of the caseload, not its novelty.

    A worsening entry can no longer leak in here even when the cap crowds it
    out of its own section: a row carries ONE gate, and the two sections read
    different gates.
    """
    eligible = [
        r for r in rows
        if r.get("gate") == gating.GATE_MAJOR
        and str(r.get("question_id")) not in exclude_question_ids
    ]
    ordered = _sorted_by(eligible, "eiv_nominal")
    # Rows with no impact centroid (binary questions) still belong; they sort
    # after the ranked ones rather than vanishing.
    unranked = [r for r in eligible if r.get("eiv_nominal") is None]
    unranked.sort(key=lambda r: str(r.get("question_id") or ""))
    return (ordered + unranked)[:max(0, int(max_entries))]


def split_budget(max_entries: int, n_worsening: int, n_major: int) -> tuple[int, int]:
    """How the total entry budget divides between the two categories.

    Worsening takes the larger share, and whichever side has fewer candidates
    hands its unused slots to the other. A month with nothing worsening in it
    still fills the report with the heaviest burdens rather than printing a
    short page and a gap.
    """
    total = max(0, int(max_entries))
    want_worsening = min(n_worsening, math.ceil(total * WORSENING_SHARE))
    want_major = min(n_major, total - want_worsening)
    # Hand back whatever the other side could not use.
    want_worsening = min(n_worsening, total - want_major)
    return want_worsening, want_major


def assign_categories(
    rows: list[dict[str, Any]],
    *,
    max_entries: int,
) -> list[dict[str, Any]]:
    """Stamp ``category`` and ``category_rank`` on the rows the report covers.

    Mutates and returns ``rows`` (every row keeps its place in the index;
    only the selected ones gain a category), so the pack still carries the
    full picture and the selection stays auditable. ``gating.gate_rows`` must
    have run first: this reads the gate, it does not re-derive it.

    ``category_rank`` is a single ordering across the whole category, not a
    per-family one, so the report's numbering runs 1..N down the page rather
    than restarting under each heading.
    """
    for row in rows:
        row.setdefault("category", None)
        row.setdefault("category_rank", None)

    worsening_pool = [r for r in rows if r.get("gate") == gating.GATE_WORSENING]
    major_pool = [
        r for r in rows if r.get("gate") == gating.GATE_MAJOR
    ]
    n_worsening, n_major = split_budget(
        max_entries, len(worsening_pool), len(major_pool)
    )

    worsening = select_worsening(rows, max_entries=n_worsening)
    selected_ids = {str(r.get("question_id")) for r in worsening}
    major = select_stable_major(
        rows, exclude_question_ids=selected_ids, max_entries=n_major
    )

    for rank, row in enumerate(worsening, start=1):
        row["category"] = CATEGORY_WORSENING
        row["category_rank"] = rank
    for rank, row in enumerate(major, start=1):
        row["category"] = CATEGORY_STABLE_MAJOR
        row["category_rank"] = rank
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
