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
