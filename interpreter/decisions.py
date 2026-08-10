# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""When a decision has to be taken, and the page that collects them.

The most useful sentence in the August report appeared once, by accident: the
Vietnam entry noted that the prepositioning window closes in September. A
forecast a reader cannot act on by a date is a fact, not a decision.

So every attention entry carries a decision point, and the deadlines are
collected onto one dated page near the front. The deadline is **derived, never
invented**: it is the peak horizon the materiality gate already found, minus a
lead time configured per hazard. A drought response is a procurement cycle and
needs months; a cyclone response is a warehouse that is either stocked or is
not. Those assumptions live in `interpreter/config.py` where they can be
argued with.

The model writes the ACTION, in words, and nothing else. Asking it for the
date would let it write a plausible month that no part of the system believes,
and the reader has no way to check. Where a row has no peak horizon there is
no derived deadline, and the model has to justify the entry's inclusion in
prose instead.

Pure functions over plain values: no DB, no IO, no config reads (the lead time
arrives as an argument), so the calendar's arithmetic is testable on its own.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
)

# The bases the schema allows, in the words the report prints.
BASIS_LABELS = {
    "peak_horizon": "the month with most expected impact",
    "lean_season_start": "the start of the lean season",
    "election_date": "an election",
    "seasonal_window": "a seasonal window",
}
DEFAULT_BASIS = "peak_horizon"

_YM = re.compile(r"^(\d{4})-(\d{2})$")


def parse_ym(value: Any) -> tuple[int, int] | None:
    """``2026-09`` or a date/timestamp string to (year, month)."""
    text = str(value or "")[:7]
    match = _YM.match(text)
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    if not 1 <= month <= 12:
        return None
    return year, month


def add_months(ym: str, delta: int) -> str | None:
    """Month arithmetic on a ``YYYY-MM`` string. None when unparseable."""
    parsed = parse_ym(ym)
    if parsed is None:
        return None
    year, month = parsed
    index = (year * 12 + (month - 1)) + int(delta)
    return f"{index // 12:04d}-{index % 12 + 1:02d}"


def month_label(ym: Any) -> str | None:
    """``2026-09`` to ``September 2026``, the form the report prints."""
    parsed = parse_ym(ym)
    if parsed is None:
        return None
    year, month = parsed
    return f"{MONTH_NAMES[month - 1]} {year}"


def horizon_month(window_start: Any, horizon: Any) -> str | None:
    """The ``YYYY-MM`` of a 1-based horizon in a window.

    Month index 1 IS the window_start month, the repo's anchoring convention
    everywhere (CLAUDE.md's per-horizon architecture note).
    """
    if not horizon:
        return None
    try:
        offset = int(horizon) - 1
    except (TypeError, ValueError):
        return None
    if offset < 0:
        return None
    return add_months(str(window_start or "")[:7], offset)


def deadline_for(peak_ym: Any, lead_months: int) -> str | None:
    """The decision month: the peak month less the hazard's lead time."""
    if not peak_ym:
        return None
    return add_months(str(peak_ym), -int(lead_months))


# What an already-passed deadline reads as. A drought's three-month run-up
# subtracted from a horizon-1 peak lands two months BEFORE the report was
# written, and the first live report duly printed "By June 2026" against three
# of its five entries. A date in the past is not a deadline; it is a statement
# that the run-up has already begun, and that is what the calendar should say.
OVERDUE_LABEL = "as soon as possible"


def decision_point(
    *,
    window_start: Any,
    peak_horizon: Any,
    lead_months: int,
    basis: str = DEFAULT_BASIS,
    issued_month: Any = None,
) -> dict[str, Any]:
    """The derived half of an entry's decision point.

    The ``action`` half is the model's; everything datable is computed here.
    An empty dict means the row carries no peak horizon, which the caller
    must treat as "no derived deadline", never as a reason to invent one.

    ``issued_month`` is when the report goes out. The derived month is kept
    verbatim either way (it is the honest arithmetic and the audit trail), but
    a deadline before the report exists is LABELLED as immediate rather than
    printed as a month a reader cannot act on.
    """
    peak_ym = horizon_month(window_start, peak_horizon)
    if not peak_ym:
        return {}
    deadline = deadline_for(peak_ym, lead_months)
    if not deadline:
        return {}
    issued = parse_ym(issued_month)
    overdue = bool(issued and parse_ym(deadline) and parse_ym(deadline) <= issued)
    return {
        "peak_month": peak_ym,
        "peak_month_label": month_label(peak_ym),
        "deadline_month": deadline,
        "deadline_month_label": (
            OVERDUE_LABEL if overdue else month_label(deadline)
        ),
        "derived_deadline_month_label": month_label(deadline),
        "overdue": overdue,
        "lead_time_months": int(lead_months),
        "basis": basis,
    }


def _sort_key(row: dict[str, Any]) -> tuple:
    """Earliest deadline first; rows without one sort last, by country."""
    ym = parse_ym(row.get("deadline_month"))
    return (
        1 if ym is None else 0,
        ym or (9999, 12),
        str(row.get("country") or ""),
    )


def calendar_rows(
    rows: Sequence[dict[str, Any]],
    actions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """The dated page: what decision, for where, by when, ordered by deadline.

    ``rows`` are pack attention rows carrying ``decision`` blocks; ``actions``
    maps question_id to the model's sentence. A row whose action is missing
    still appears, with an empty action, so the calendar and the report's
    entries always list the same set.
    """
    actions = actions or {}
    out: list[dict[str, Any]] = []
    for row in rows:
        decision = row.get("decision") or {}
        qid = str(row.get("question_id") or "")
        out.append({
            "question_id": qid,
            "country": row.get("country_name") or row.get("iso3"),
            "hazard": row.get("hazard_name") or row.get("hazard_code"),
            "metric": row.get("metric_name") or row.get("metric"),
            "deadline_month": decision.get("deadline_month"),
            "deadline_label": decision.get("deadline_month_label") or "not dated",
            "overdue": bool(decision.get("overdue")),
            "peak_label": decision.get("peak_month_label") or "",
            "basis": BASIS_LABELS.get(
                str(decision.get("basis") or DEFAULT_BASIS), ""
            ),
            "action": str(actions.get(qid) or "").strip(),
        })
    out.sort(key=_sort_key)
    return out


def derived_deadlines(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """{question_id: decision block} for the rows that have one."""
    return {
        str(r.get("question_id")): r["decision"]
        for r in rows
        if r.get("question_id") and r.get("decision")
    }
