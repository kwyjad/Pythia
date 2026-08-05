# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Deterministic rulebook-driven predicates for the resolution machine.

Every function here takes the value(s) it judges plus a :class:`Rulebook`
and returns a plain result — no I/O, no LLMs, no hidden state. Phase 1
detection and Phase 3 reconciliation build on these; Phase 0 ships them so
the acceptance test can prove that changing ``rulebook.yaml`` changes
behaviour without a code change.
"""

from __future__ import annotations

import calendar
import datetime as dt

from resolver.hazard_resolution.rulebook import GDACS_ALERT_LEVELS, Rulebook


def cyclone_track_qualifies(
    distance_km: float, max_wind_kt: float, rulebook: Rulebook
) -> bool:
    """True when a storm track point counts as a qualifying cyclone.

    A point qualifies when it lies within ``cyclone.buffer_km`` of the
    country and its wind speed is at least ``cyclone.min_wind_kt``.
    """

    buffer_km = float(rulebook.get("cyclone.buffer_km"))
    min_wind_kt = float(rulebook.get("cyclone.min_wind_kt"))
    return distance_km <= buffer_km and max_wind_kt >= min_wind_kt


def flood_alert_qualifies(alert_level: str, rulebook: Rulebook) -> bool:
    """True when a GDACS flood alert colour meets ``flood.gdacs_trigger_level``.

    Colours order green < orange < red; the configured level and anything
    above it trigger. Unknown colours raise rather than silently not trigger.
    """

    level = str(alert_level).strip().lower()
    if level not in GDACS_ALERT_LEVELS:
        raise ValueError(
            f"unknown GDACS alert level {alert_level!r}; expected one of {list(GDACS_ALERT_LEVELS)}"
        )
    threshold = str(rulebook.get("flood.gdacs_trigger_level")).strip().lower()
    return GDACS_ALERT_LEVELS.index(level) >= GDACS_ALERT_LEVELS.index(threshold)


def drought_delta_qualifies(
    delta: float, baseline: float | None, rulebook: Rulebook
) -> bool:
    """True when an IPC Phase 3+ increase counts as drought people-affected.

    An increase qualifies when it is strictly positive and clears both
    ``drought.min_delta_people`` (absolute) and ``drought.min_delta_pct``
    (a PERCENTAGE of the previous analysis's Phase 3+ population). At the
    shipped 0 / 0.0 any increase qualifies.

    A delta that does not qualify is not missing data — the two analyses
    were read and they state no material increase — so the caller resolves
    it as a credible zero rather than as NO_DATA.
    """

    value = float(delta)
    if value <= 0:
        return False
    if value < float(rulebook.get("drought.min_delta_people")):
        return False
    min_pct = float(rulebook.get("drought.min_delta_pct"))
    if min_pct > 0:
        if baseline is None or float(baseline) <= 0:
            # No denominator: the absolute test is all there is to apply.
            # Refusing here would drop a real increase for want of a ratio.
            return True
        if (value / float(baseline)) * 100.0 < min_pct:
            return False
    return True


def freeze_deadline(year: int, month: int, rulebook: Rulebook) -> dt.date:
    """The date a (year, month) cell freezes: month-end + ``freeze_days``.

    After this date the cell's resolution is immutable; later source
    revisions are logged to ``haz_revisions`` but never change the value.
    """

    if not 1 <= month <= 12:
        raise ValueError(f"month must be 1..12, got {month!r}")
    last_day = calendar.monthrange(year, month)[1]
    month_end = dt.date(year, month, last_day)
    return month_end + dt.timedelta(days=int(rulebook.get("freeze_days")))


def is_frozen(year: int, month: int, rulebook: Rulebook, today: dt.date | None = None) -> bool:
    """True when the (year, month) cell is past its freeze deadline."""

    reference = today if today is not None else dt.date.today()
    return reference > freeze_deadline(year, month, rulebook)


def is_provisional(
    year: int, month: int, rulebook: Rulebook, today: dt.date | None = None
) -> bool:
    """True when a resolution written now for this cell is still provisional.

    The inverse of :func:`is_frozen` at the write boundary: a row written
    before the freeze deadline is real and usable but may still change,
    so it is marked provisional. On or after the deadline it is written
    final — and immutable.
    """

    return not is_frozen(year, month, rulebook, today=today)


def within_sanity_ceiling(
    value: float, exposed_population: float | None, rulebook: Rulebook
) -> bool:
    """True when ``value`` respects the GDACS exposure ceiling.

    The ceiling is ``sanity.ceiling_multiplier`` x the GDACS exposed
    population. With no exposure estimate available there is no ceiling to
    apply and the value passes (the national population cap is a separate
    check owned by reconciliation).
    """

    if exposed_population is None:
        return True
    multiplier = float(rulebook.get("sanity.ceiling_multiplier"))
    return float(value) <= multiplier * float(exposed_population)


def within_population_cap(
    value: float, national_population: float | None, rulebook: Rulebook
) -> bool:
    """True when ``value`` respects the national-population cap.

    More people cannot be affected in a country than live in it. The cap
    is only applied when ``sanity.population_cap`` is on and a denominator
    is available; with neither, there is nothing to check and the value
    passes (absence of a denominator is not evidence against a figure).
    """

    if not bool(rulebook.get("sanity.population_cap")):
        return True
    if national_population is None:
        return True
    return float(value) <= float(national_population)


def ladder_rank(source: str, rulebook: Rulebook) -> int | None:
    """Position of ``source`` in the impact ladder (0 = highest rung).

    Returns None for a source the ladder does not walk — GDACS above all,
    which is detection and ceiling only and must never be ranked as a
    resolution source.
    """

    ladder = [str(rung) for rung in rulebook.get("ladder")]
    try:
        return ladder.index(str(source))
    except ValueError:
        return None


def is_lower_bound_rung(source: str, rulebook: Rulebook) -> bool:
    """True when a rung's figures are displacement, i.e. a lower bound.

    Displacement counts a strict subset of the people a hazard affects, so
    a figure from such a rung resolves as "at least this many".
    """

    return str(source) in {str(r) for r in rulebook.get("lower_bound_rungs", [])}


def orders_of_magnitude_apart(a: float, b: float, rulebook: Rulebook) -> bool:
    """True when two figures differ by more than the configured factor.

    This is the deterministic definition of "wild conflict" between
    adjacent ladder rungs (``conflict_detection.order_of_magnitude_factor``).
    Two zeros agree; one zero against a positive figure is maximal
    disagreement.
    """

    factor = float(rulebook.get("conflict_detection.order_of_magnitude_factor"))
    lo, hi = sorted((abs(float(a)), abs(float(b))))
    if hi == 0:
        return False  # both zero — perfect agreement
    if lo == 0:
        return True  # zero vs. a positive figure
    return (hi / lo) > factor


def event_months(start: dt.date, end: dt.date) -> list[str]:
    """Every ``YYYY-MM`` an event's date range overlaps (detection scope).

    An event that runs 28 Jan .. 4 Feb is DETECTED in both January and
    February. Its figure is a separate question — see
    :func:`attribution_month`.
    """

    if end < start:
        raise ValueError(f"event end {end} precedes start {start}")
    months: list[str] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def attribution_month(start: dt.date, end: dt.date, rulebook: Rulebook) -> str:
    """The single ``YYYY-MM`` an event's people-affected figure belongs to.

    ``event_attribution.figure = start_month``: the whole figure lands in
    the month the event STARTED. Figures are never split across months —
    a source states one number for the event, and apportioning it would
    invent a monthly breakdown nobody reported. The full span travels with
    the figure in provenance so a reader can see what it covers.
    """

    policy = str(rulebook.get("event_attribution.figure"))
    if policy != "start_month":  # pragma: no cover - validator guards this
        raise ValueError(f"unsupported event_attribution.figure: {policy!r}")
    if end < start:
        raise ValueError(f"event end {end} precedes start {start}")
    return f"{start.year:04d}-{start.month:02d}"
