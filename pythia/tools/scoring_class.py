# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""Which resolved conflict displacement months may be marked.

ACE/PA resolves from IDMC conflict displacement, which most countries
report late and irregularly. For a country IDMC reports in at least 8 of
the 12 months before a month (a regular reporter), a month resolves to its
figure or, bracketed by a later report, to zero. For every other country a
month resolves only when IDMC reports it, so its resolved months are the
months something was reported: a selected sample, and a score on it says
how the forecast did on the months that happened to be reported, not how
it did. On 7 Oct 2026, 31 of the 38 countries with a current ACE/PA
question were not regular reporters.

Owner decision (7 Oct 2026): keep asking every ACE/PA question and split
them. Superseded on 8 Oct 2026: from the 13 Oct run ACE/PA is ASKED only
for a regular reporter (``pythia/tools/ace_pa_eligibility.py``); questions
already open stay and are split as below. Each resolution row of an ACE/PA question carries a scoring class,
decided when it is resolved (so before it is scored) from the regular-
reporter rule as it stands for that month:

* ``scored`` — a regular reporter's month; marked like any other.
* ``indicative`` — still forecast, published and resolved when a figure
  arrives, but left out of calibration weights, advice, recalibration
  fitting, centroid updates, the Sibyl comparison and every headline skill
  figure. The scored bundle lists them in their own table with the reason.

* ``ace_pa_ladder`` (Oct 2026) — a country that is a regular reporter only
  once the extracted OCHA/UN figures (ladder rung 2) are counted. Its months
  are forecast, resolved and stored like any other, but every reader that
  learns from or reports skill leaves them out until at least
  ``PYTHIA_ACE_PA_LADDER_MIN_RESOLVED`` (6) such horizon-months have
  resolved, because a rung that is days old has no record of agreeing with
  IDMC yet. After that they count as scored.

A month answered by a DTM lower bound alone never counts toward the rule.

Every other hazard and metric has no class (NULL), which reads as scored.

Readers filter through :func:`scored_only_sql` / :func:`scored_only_clause`,
which join ``resolutions`` rather than copying the class onto score rows, so
every table keyed by (question_id, horizon_m) — ``scores``,
``baseline_scored_forecasts``, ``sibyl_variant_scores`` — is filtered the
same way and cannot drift from the resolution it scores.
"""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Tuple

SCORED = "scored"
INDICATIVE = "indicative"
LADDER = "ace_pa_ladder"

#: Plain words for the dashboard, the Interpreter and the bundle.
INDICATIVE_REASON = (
    "no source reports this country's conflict displacement most months "
    "(IDMC, OCHA/UN reports, IOM DTM): a month resolves only when one of "
    "them reports it, so the forecast stands but cannot be marked"
)
INDICATIVE_REASON_SHORT = "not a regular reporter on the displacement ladder"
LADDER_REASON = (
    "a regular reporter only once OCHA/UN figures extracted from ReliefWeb are "
    "counted; scored like any other once enough such months have resolved"
)


def ladder_min_resolved() -> int:
    """Horizon-months of the ladder class that must resolve before it counts."""
    try:
        return int(os.getenv("PYTHIA_ACE_PA_LADDER_MIN_RESOLVED", "") or 6)
    except ValueError:
        return 6


def conflict_scoring_class(
    reported: Mapping[str, Any], ym: str, *, idmc_reported: Optional[Mapping[str, Any]] = None,
) -> Tuple[str, Optional[str]]:
    """(class, reason) for one ACE/PA country-month, from the regular-reporter
    rule as it stands for ``ym`` (``base_rate_spd.conflict_regular_reporter``).
    ``reported`` is the months that count toward regularity
    (``base_rate_spd.conflict_regular_months``: IDMC or an extracted figure).
    With ``idmc_reported`` (IDMC's own months) a country regular only with
    the extracted figures is ``ace_pa_ladder``; without it, ``scored``."""
    from pythia.tools.base_rate_spd import (  # noqa: PLC0415
        CONFLICT_REGULAR_MIN_MONTHS,
        CONFLICT_REGULAR_WINDOW_MONTHS,
        conflict_regular_reporter,
    )

    if conflict_regular_reporter(reported, ym):
        if idmc_reported is not None and not conflict_regular_reporter(idmc_reported, ym):
            return LADDER, LADDER_REASON
        return SCORED, None
    return INDICATIVE, (
        f"{INDICATIVE_REASON_SHORT}: reported in fewer than {CONFLICT_REGULAR_MIN_MONTHS} "
        f"of the {CONFLICT_REGULAR_WINDOW_MONTHS} months before {ym}"
    )


def scored_only_sql(alias: str, resolutions: str = "resolutions") -> str:
    """Predicate keeping rows of ``alias`` (a table with question_id and
    horizon_m) whose resolution is not indicative, and not of the ladder class
    while fewer than :func:`ladder_min_resolved` ladder horizon-months exist."""
    n = ladder_min_resolved()
    return (
        f"NOT EXISTS (SELECT 1 FROM {resolutions} sc_r WHERE sc_r.question_id = {alias}.question_id "
        f"AND sc_r.horizon_m = {alias}.horizon_m AND (sc_r.scoring_class = '{INDICATIVE}' "
        f"OR (sc_r.scoring_class = '{LADDER}' AND (SELECT COUNT(*) FROM {resolutions} sc_l "
        f"WHERE sc_l.scoring_class = '{LADDER}') < {int(n)})))"
    )


def excluded_classes(con: Any) -> set:
    """The classes every learning and skill reader leaves out right now:
    ``indicative`` always, ``ace_pa_ladder`` until enough have resolved."""
    out = {INDICATIVE}
    try:
        n = con.execute(
            "SELECT COUNT(*) FROM resolutions WHERE scoring_class = ?", [LADDER]
        ).fetchone()[0]
    except Exception:  # noqa: BLE001 - no column is the pre-Oct-2026 table
        n = 0
    if int(n or 0) < ladder_min_resolved():
        out.add(LADDER)
    return out


def has_scoring_class(con: Any) -> bool:
    try:
        rows = con.execute("PRAGMA table_info('resolutions')").fetchall()
    except Exception:  # noqa: BLE001
        return False
    return any(str(r[1]).lower() == "scoring_class" for r in rows)


def scored_only_clause(con: Any, alias: str, *, prefix: str = " AND ") -> str:
    """``scored_only_sql`` with a leading ``AND``, or '' on a DB whose
    resolutions table predates the column."""
    return f"{prefix}{scored_only_sql(alias)}" if has_scoring_class(con) else ""
