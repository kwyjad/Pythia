# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Issue dates for seasonal TC outlooks: a real date, or NULL and a reason.

Every scraper used to put whatever it found into ``issue_date`` — an ISO
date from TSR, ``"October 2025"`` from BoM, nothing from the South Pacific,
SWI and NIO blocks — and the store wrote the string as it came. A date
column holding free text cannot be compared, ordered or aged, and a blank
cannot say whether the page carried no date or the parser missed it.

:func:`parse_issue_date` turns any of those forms into ``(iso_date, reason)``:
the date when one can be established (a month-only form becomes the first of
the month, and says so), otherwise ``None`` beside the reason it is missing.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Optional

#: Reasons a stored outlook carries no issue date, or an approximate one.
REASON_MONTH_PRECISION = "month_precision_only"
REASON_NO_DATE_IN_SOURCE = "no_issue_date_in_source"
REASON_UNPARSEABLE = "issue_date_unparseable"
REASON_CLIMATOLOGY = "climatology_context_has_no_issue_date"
REASON_URL_MONTH_DISAGREES = "issued_line_disagrees_with_url_month"
REASON_PDF_METADATA = "pdf_metadata_creation_date"

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    # French, for the Météo-France La Réunion article
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "août": 8, "aout": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
}

_MONTH_WORD = "|".join(sorted(_MONTHS, key=len, reverse=True))

_ISO_RE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})")
#: "5 August 2026", "5th August 2026", "21 octobre 2025"
_DMY_RE = re.compile(
    rf"(\d{{1,2}})(?:st|nd|rd|th|er)?\s+({_MONTH_WORD})\s+(\d{{4}})", re.IGNORECASE
)
#: "August 5, 2026"
_MDY_RE = re.compile(rf"({_MONTH_WORD})\s+(\d{{1,2}}),?\s+(\d{{4}})", re.IGNORECASE)
#: "October 2025"
_MY_RE = re.compile(rf"({_MONTH_WORD})\s+(\d{{4}})", re.IGNORECASE)


def month_number(word: str) -> Optional[int]:
    return _MONTHS.get((word or "").strip().lower())


def parse_issue_date(value: object) -> tuple[Optional[str], Optional[str]]:
    """``(iso_date, reason)`` for whatever a scraper put in ``issue_date``.

    * an ISO date, or a day-month-year in English or French, or a
      month-day-year: the date, no reason;
    * a month and year only: the first of that month, with
      :data:`REASON_MONTH_PRECISION`;
    * empty: ``(None, REASON_NO_DATE_IN_SOURCE)``;
    * anything else: ``(None, REASON_UNPARSEABLE)``.
    """

    if value is None:
        return None, REASON_NO_DATE_IN_SOURCE
    if isinstance(value, dt.datetime):
        return value.date().isoformat(), None
    if isinstance(value, dt.date):
        return value.isoformat(), None
    text = str(value).strip()
    if not text:
        return None, REASON_NO_DATE_IN_SOURCE

    m = _ISO_RE.match(text)
    if m:
        try:
            return dt.date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat(), None
        except ValueError:
            return None, REASON_UNPARSEABLE
    m = _DMY_RE.search(text)
    if m:
        month = month_number(m.group(2))
        try:
            return dt.date(int(m.group(3)), month, int(m.group(1))).isoformat(), None
        except (TypeError, ValueError):
            return None, REASON_UNPARSEABLE
    m = _MDY_RE.search(text)
    if m:
        month = month_number(m.group(1))
        try:
            return dt.date(int(m.group(3)), month, int(m.group(2))).isoformat(), None
        except (TypeError, ValueError):
            return None, REASON_UNPARSEABLE
    m = _MY_RE.search(text)
    if m:
        month = month_number(m.group(1))
        try:
            return dt.date(int(m.group(2)), month, 1).isoformat(), REASON_MONTH_PRECISION
        except (TypeError, ValueError):
            return None, REASON_UNPARSEABLE
    return None, REASON_UNPARSEABLE


def first_date_in(text: str) -> Optional[str]:
    """The first full date written in ``text``, ISO, or None."""

    iso, reason = parse_issue_date(text)
    return iso if iso and reason is None else None


__all__ = [
    "REASON_CLIMATOLOGY", "REASON_MONTH_PRECISION", "REASON_NO_DATE_IN_SOURCE",
    "REASON_PDF_METADATA", "REASON_UNPARSEABLE", "REASON_URL_MONTH_DISAGREES",
    "first_date_in", "month_number", "parse_issue_date",
]
