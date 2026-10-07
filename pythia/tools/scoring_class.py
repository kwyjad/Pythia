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

Owner decision (Oct 2026): keep asking every ACE/PA question and split
them. Each resolution row of an ACE/PA question carries a scoring class,
decided when it is resolved (so before it is scored) from the regular-
reporter rule as it stands for that month:

* ``scored`` — a regular reporter's month; marked like any other.
* ``indicative`` — still forecast, published and resolved when a figure
  arrives, but left out of calibration weights, advice, recalibration
  fitting, centroid updates, the Sibyl comparison and every headline skill
  figure. The scored bundle lists them in their own table with the reason.

Every other hazard and metric has no class (NULL), which reads as scored.

Readers filter through :func:`scored_only_sql` / :func:`scored_only_clause`,
which join ``resolutions`` rather than copying the class onto score rows, so
every table keyed by (question_id, horizon_m) — ``scores``,
``baseline_scored_forecasts``, ``sibyl_variant_scores`` — is filtered the
same way and cannot drift from the resolution it scores.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

SCORED = "scored"
INDICATIVE = "indicative"

#: Plain words for the dashboard, the Interpreter and the bundle.
INDICATIVE_REASON = (
    "IDMC does not report this country every month: a month resolves only "
    "when IDMC reports it, so the forecast stands but cannot be marked"
)
INDICATIVE_REASON_SHORT = "not a regular IDMC reporter"


def conflict_scoring_class(reported: Mapping[str, Any], ym: str) -> Tuple[str, Optional[str]]:
    """(class, reason) for one ACE/PA country-month, from the regular-reporter
    rule as it stands for ``ym`` (``base_rate_spd.conflict_regular_reporter``)."""
    from pythia.tools.base_rate_spd import (  # noqa: PLC0415
        CONFLICT_REGULAR_MIN_MONTHS,
        CONFLICT_REGULAR_WINDOW_MONTHS,
        conflict_regular_reporter,
    )

    if conflict_regular_reporter(reported, ym):
        return SCORED, None
    return INDICATIVE, (
        f"{INDICATIVE_REASON_SHORT}: reported in fewer than {CONFLICT_REGULAR_MIN_MONTHS} "
        f"of the {CONFLICT_REGULAR_WINDOW_MONTHS} months before {ym}"
    )


def scored_only_sql(alias: str, resolutions: str = "resolutions") -> str:
    """Predicate keeping rows of ``alias`` (a table with question_id and
    horizon_m) whose resolution is not indicative."""
    return (
        f"NOT EXISTS (SELECT 1 FROM {resolutions} sc_r WHERE sc_r.question_id = {alias}.question_id "
        f"AND sc_r.horizon_m = {alias}.horizon_m AND sc_r.scoring_class = '{INDICATIVE}')"
    )


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
