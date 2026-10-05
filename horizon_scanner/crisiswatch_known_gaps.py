# Pythia / Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
"""CrisisWatch editions known to be unrecoverable, with a date to look again.

Stdlib only: the resolver debug bundle and the operational debug bundle both
read it, and neither may pull in the scraper's imports.

An entry here turns the twelve-edition check's FAIL for that month into a
named, dated known gap. Any OTHER missing month still fails, and an entry
past its ``review_by`` fails again, because a suppression that never expires
is how a known gap becomes an invisible one.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

KNOWN_MISSING_EDITIONS: dict[str, dict[str, str]] = {
    "2026-05": {
        "reason": (
            "No capture of the May 2026 edition exists: the Wayback Machine holds "
            "none of the main page or the per-edition page (every capture from April "
            "to June carries April), ReliefWeb stopped reposting CrisisWatch after "
            "April 2026, and crisisgroup.org refuses CI runners, so Save Page Now "
            "cannot be asked for it (runs 37122264452, 37123118713)."
        ),
        # Chosen for the remedy: re-run the publication-days measurement
        # (refresh-crisiswatch.yml, measure_publication_days) to see whether a
        # capture has appeared. The month leaves the twelve-month window with
        # the 2027-05 edition in any case.
        "review_by": "2026-12-31",
    },
}


def known_gap(label: str, today: date) -> Optional[dict[str, str]]:
    """The entry for ``label`` (``YYYY-MM``) while it is still in force."""
    entry = KNOWN_MISSING_EDITIONS.get(label)
    if not entry:
        return None
    try:
        if today > date.fromisoformat(entry["review_by"]):
            return None
    except (KeyError, ValueError):
        return None
    return entry


def split_missing(labels: list[str], today: date) -> tuple[list[str], list[str]]:
    """``(still_failing, known)`` for a list of missing edition labels."""
    known = [x for x in labels if known_gap(x, today)]
    return [x for x in labels if x not in known], known
