# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Turn extracted figures into ladder candidates — deterministically.

Everything the model was forbidden to do happens here, by rule:

* **Households become people** (:func:`convert_units`), and only when the
  document said households. The multiplier comes from
  ``reliefweb.household_conversion`` and the conversion is recorded on the
  candidate, so a reader can undo it.
* **Duplicates collapse** (:func:`deduplicate`). The same sentence
  republished across five situation reports is one statement, and the same
  figure for the same area restated in different words is one figure.
* **Precedence decides** (:func:`preference_key`). The rulebook orders
  attributions — government > UN agency > IFRC/NGO > media — and within a
  tier the latest-dated figure wins. This is the rung's own internal
  precedence: unlike the record-based rungs, "the largest number" is not
  the right answer when several bodies are quoting different assessments
  of the same flood.
* **Implausible figures are rejected** (:func:`apply_ceiling`), not
  flagged. This is a deliberate difference from :mod:`reconcile`, which
  flags a ceiling breach and keeps the value: there, a curated source
  disagreeing with modelled exposure is a finding worth a human's time;
  here, a figure above the exposed population is much more likely a
  mis-transcription — a national cumulative total, or a figure for a
  different emergency in the same document — and admitting it would let a
  transcription error outrank a curated EM-DAT record.

No LLM, no network, no clock beyond what the caller passes: the same
figures always produce the same candidates.
"""

from __future__ import annotations

import hashlib
import datetime as dt
import logging
import re
from typing import Any

from resolver.hazard_resolution import cell_ledger
from resolver.hazard_resolution.candidates import VALUE_AFFECTED, Candidate
from resolver.hazard_resolution.extract import (
    SOURCE,
    UNIT_HOUSEHOLDS,
    UNIT_PEOPLE,
    ExtractedFigure,
)
from resolver.hazard_resolution.rulebook import Rulebook
from resolver.hazard_resolution.rules import within_sanity_ceiling

LOG = logging.getLogger(__name__)

#: Rank given to an attribution that matches no configured tier. It sorts
#: after every listed tier but is NOT discarded: an unattributed figure in
#: an OCHA situation report is still the best evidence in the room when
#: nothing else is available.
UNRANKED = 10_000


def authority_tier(stated_by: str, rulebook: Rulebook) -> tuple[int, str]:
    """Classify an attribution into a precedence tier. Returns (rank, tier).

    Lowercase substring matching against the rulebook's keyword lists,
    first tier that matches wins. Deliberately simple: the alternative is a
    judgement call, and judgement calls are what this module exists to keep
    out of the resolution path.
    """

    text = str(stated_by or "").strip().lower()
    if not text:
        return UNRANKED, "unattributed"
    for rank, entry in enumerate(rulebook.get("reliefweb.authority_precedence")):
        keywords = [str(k).strip().lower() for k in entry.get("keywords") or []]
        if any(keyword and keyword in text for keyword in keywords):
            return rank, str(entry.get("tier"))
    return UNRANKED, "unrecognised"


def household_multiplier(iso3: str, rulebook: Rulebook) -> tuple[float, str]:
    """The households-to-people multiplier for a country. Returns (value, origin)."""

    by_iso3 = rulebook.get("reliefweb.household_conversion.by_iso3") or {}
    specific = by_iso3.get(iso3.upper())
    if specific is not None:
        return float(specific), "by_iso3"
    return float(rulebook.get("reliefweb.household_conversion.default_multiplier")), "default"


#: What a figure is attributed to when the document names nobody in the
#: text. Ordered by how much it says.
_ATTRIBUTION_FROM_TEXT = "text"
_ATTRIBUTION_FROM_DOCUMENT = "document"
_ATTRIBUTION_NONE = "none"
UNATTRIBUTED = "(unattributed)"


def _attribute(figure: "ExtractedFigure") -> tuple[str, str]:
    """``(stated_by, where it came from)`` for one extracted figure.

    The model's verbatim ``stated_by`` first; the document's own publisher
    where the text names nobody; ``(unattributed)`` where neither says
    anything. The second case is an inference and is labelled as one — a
    reader must be able to tell "the report said UNHCR" from "the report is
    an OCHA report and said nothing".
    """

    stated = str(getattr(figure, "stated_by", "") or "").strip()
    if stated:
        return stated, _ATTRIBUTION_FROM_TEXT
    publisher = str(getattr(figure, "doc_publisher", "") or "").strip()
    if publisher:
        return publisher, _ATTRIBUTION_FROM_DOCUMENT
    return UNATTRIBUTED, _ATTRIBUTION_NONE


def conversion_for_unit(
    unit: str | None, iso3: str, rulebook: Rulebook
) -> tuple[float, str]:
    """``(factor, origin)`` joining a stated unit to whole persons.

    Every ledger row carries this, accepted or rejected. Before Sept 2026
    only the accepted household rows did, so 74 household rows and 229
    person rows went out with an empty factor — and a household count with
    no factor beside it is either being multiplied by an unrecorded number
    or read as persons, which is a factor of five either way.
    """

    if str(unit or "").strip().lower() == UNIT_HOUSEHOLDS:
        multiplier, origin = household_multiplier(iso3, rulebook)
        return float(multiplier), origin
    # A person figure is one person per person. Stated outright so a blank
    # in this column means "unresolved" and never "trivially one".
    return 1.0, "identity"


def convert_units(
    figure: ExtractedFigure, iso3: str, rulebook: Rulebook
) -> tuple[float, dict[str, Any] | None]:
    """Convert a households figure to people. Returns (value, conversion).

    ``conversion`` is None when nothing was converted, so a people figure
    carries no conversion record and cannot be mistaken for one.
    """

    if figure.unit != UNIT_HOUSEHOLDS:
        return float(figure.value), None
    multiplier, origin = household_multiplier(iso3, rulebook)
    # People are counted in whole persons. 15,600 households x 4.1 is
    # 63,960 people, not 63,959.99999999 — the fractional counts in the
    # September 2026 ledger were the tell that the unit column and the
    # value column had come apart.
    converted = float(round(float(figure.value) * multiplier))
    return converted, {
        "from_unit": UNIT_HOUSEHOLDS,
        "to_unit": UNIT_PEOPLE,
        "stated_value": float(figure.value),
        "multiplier": multiplier,
        "multiplier_origin": origin,
        "converted_value": converted,
    }


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _newest_first(value: str) -> int:
    """A date as a term that sorts NEWEST first in an ascending key.

    Negated day ordinals, so a later date gives a smaller number. An
    absent or unparseable date yields 0, which sorts after every real date
    — correct, because an undated figure should never outrank a dated one.
    """

    from resolver.hazard_resolution.sources import parse_date

    parsed = parse_date(value)
    return -parsed.toordinal() if parsed is not None else 0


def preference_key(
    figure: ExtractedFigure, rulebook: Rulebook
) -> tuple[int, int, int, int, str]:
    """Sort key implementing the rung's precedence. Lower sorts better.

    Authority first, then the latest-dated figure, then the newest
    document, then the document's own source rank, then its id so ties
    never wobble between runs.
    """

    rank, _ = authority_tier(figure.stated_by, rulebook)
    return (
        rank,
        _newest_first(figure.date),
        _newest_first(figure.doc_date),
        int(figure.doc_source_rank),
        str(figure.doc_id),
    )


def deduplicate(
    figures: list[tuple[ExtractedFigure, float]], rulebook: Rulebook
) -> tuple[list[tuple[ExtractedFigure, float]], list[dict[str, Any]]]:
    """Collapse restatements of one figure. Returns (kept, dropped).

    Two passes, because a figure is republished in two different ways:
    verbatim (the same sentence copied into the next situation report) and
    paraphrased (the same number for the same area, worded differently).
    The best-ranked member of each group survives.
    """

    dropped: list[dict[str, Any]] = []

    if _switch(rulebook, "reliefweb.household_conversion.prefer_stated_people"):
        # A stated people count beats a household conversion of the same
        # sentence, BEFORE preference ordering decides: grouped by quote,
        # the ordering would otherwise pick whichever reading it ranked
        # first, and for Thailand in November 2026 that was the households
        # reading times five.
        people_quotes = {
            _normalise(f.quote) for f, _v in figures if f.unit == UNIT_PEOPLE
        }
        survivors: list[tuple[ExtractedFigure, float]] = []
        for figure, value in figures:
            if figure.unit == UNIT_HOUSEHOLDS and _normalise(figure.quote) in people_quotes:
                dropped.append({
                    "reason": REASON_HOUSEHOLD_SUPERSEDED,
                    "doc_id": figure.doc_id,
                    "value": figure.value,
                    "unit": figure.unit,
                    "quote": figure.quote,
                    "superseded_by_doc_id": figure.doc_id,
                })
                continue
            survivors.append((figure, value))
        figures = survivors

    def _collapse(
        items: list[tuple[ExtractedFigure, float]],
        key,
        reason: str,
    ) -> list[tuple[ExtractedFigure, float]]:
        groups: dict[Any, list[tuple[ExtractedFigure, float]]] = {}
        for item in items:
            groups.setdefault(key(item), []).append(item)
        kept: list[tuple[ExtractedFigure, float]] = []
        for group in groups.values():
            group.sort(key=lambda pair: preference_key(pair[0], rulebook))
            kept.append(group[0])
            for figure, _value in group[1:]:
                dropped.append(
                    {
                        "reason": reason,
                        "doc_id": figure.doc_id,
                        "value": figure.value,
                        "unit": figure.unit,
                        "quote": figure.quote,
                        "superseded_by_doc_id": group[0][0].doc_id,
                    }
                )
        return kept

    kept = _collapse(
        figures, lambda pair: _normalise(pair[0].quote), "duplicate_quote"
    )
    kept = _collapse(
        kept,
        lambda pair: (round(pair[1], 3), _normalise(pair[0].area)),
        "duplicate_value_and_area",
    )
    kept.sort(key=lambda pair: preference_key(pair[0], rulebook))
    return kept, dropped


REASON_PRIMARY_COUNTRY_MISMATCH = "document_primary_country_mismatch"
REASON_OUTSIDE_WINDOW = "outside_reporting_window"
#: The quote names years, and every one of them is before the cell's year.
REASON_QUOTE_EARLIER_PERIOD = "quote_names_earlier_period"
#: An undated figure in a document whose dated figures (or title) are about
#: an earlier event.
REASON_DOCUMENT_EARLIER_EVENT = "document_about_earlier_event"
REASON_HOUSEHOLD_SUPERSEDED = "household_superseded_by_stated_people"
REASON_HOUSEHOLD_WITH_PEOPLE = "household_with_stated_people"
REASON_CYCLONE_IN_FLOOD = "cyclone_figure_in_flood_cell"

#: Which field dated a figure. The first three are the figure's own
#: evidence; the last two are the document's, and a figure dated by them
#: is "undated" for :data:`REASON_DOCUMENT_EARLIER_EVENT`.
FIELD_FIGURE_DATE = "figure_date"
FIELD_FIGURE_AND_QUOTE = "figure_date+quote"
FIELD_QUOTE = "quote"
FIELD_DOC_ORIGINAL = "doc_date_original"
FIELD_DOC_CREATED = "doc_date"
_FIGURE_OWN_FIELDS = frozenset({FIELD_FIGURE_DATE, FIELD_FIGURE_AND_QUOTE, FIELD_QUOTE})


# ---------------------------------------------------------------------------
# Reading the time a quote names
# ---------------------------------------------------------------------------

_EN_MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11,
    "December": 12, "Sept": 9,
}
_FR_MONTHS = {
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "août": 8, "aout": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
}
_ES_MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}


def _month_number(name: str) -> int | None:
    if name in _EN_MONTHS:
        return _EN_MONTHS[name]
    if name.upper() == name and name.title() in _EN_MONTHS:
        return _EN_MONTHS[name.title()]
    lowered = name.lower()
    return _FR_MONTHS.get(lowered) or _ES_MONTHS.get(lowered)


# English month names are matched only capitalised (or in capitals): "may"
# in lower case is a verb, and "The affected people may need shelter" is not
# a date. French and Spanish month names are written in lower case, so they
# are matched in any case.
_EN_ALTS = sorted(
    set(_EN_MONTHS) | {m.upper() for m in _EN_MONTHS}, key=len, reverse=True
)
_FRES_ALTS = sorted(set(_FR_MONTHS) | set(_ES_MONTHS), key=len, reverse=True)
_MONTH = (
    r"(?:" + "|".join(re.escape(m) for m in _EN_ALTS)
    + r"|(?i:" + "|".join(re.escape(m) for m in _FRES_ALTS) + r"))"
)
_YEAR = r"(?:19[9]\d|20\d\d)"
_DAY = r"(?:[0-3]?\d)(?:st|nd|rd|th|er)?"
_OF = r"(?:\s+(?:de|del))?"

_DATE_RE = re.compile(
    r"(?<![\w-])(?:"
    # [3 au] 16 September [2026] / 16 de septiembre [de 2026] / 1er octobre
    rf"(?:(?P<d0>{_DAY})\s*(?:-|–|au|al|a|to|y|et|and)\s*)?"
    rf"(?P<d1>{_DAY})\s+(?:de\s+)?(?P<m1>{_MONTH})(?:{_OF}\s*,?\s+(?P<y1>{_YEAR}))?"
    r"|"
    # September 16[,] [2026]
    rf"(?P<m2>{_MONTH})\s+(?P<d2>{_DAY})(?![\d])(?:,?\s+(?P<y2>{_YEAR}))?"
    r"|"
    # September 2026 / septiembre de 2026 / September
    rf"(?P<m3>{_MONTH})(?:{_OF}\s*,?\s+(?P<y3>{_YEAR}))?"
    r"|"
    # 2025, 2025a (a citation year), but not 2025 people
    rf"(?P<y4>{_YEAR})[a-z]?"
    r")(?![\w])"
)

#: What may stand between two dates that form a range, so the second
#: date's year belongs to the first ("from 16 September to 6 October 2026",
#: "in July and August", "du 3 au 5 mars 2026").
_RANGE_GAP_RE = re.compile(
    r"\s*(?:to|until|till|through|and|-|–|—|au|à|et|al|a|y|hasta)\s*$", re.I
)
#: A four-digit number followed by a unit is a count, not a year.
_COUNT_AFTER_RE = re.compile(
    r"\s*(?:people|persons|individuals|households|families|homes|houses|"
    r"personnes|personas|familias|familles|ménages|hogares)\b",
    re.I,
)
#: "the worst floods since 1998" compares, it does not date.
_COMPARISON_BEFORE_RE = re.compile(r"(?:since|depuis|desde)\s*$", re.I)


def _valid_day(year: int, month: int, day: int) -> dt.date | None:
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def quote_period(
    quote: str, *, reference_date: "dt.date | None"
) -> tuple["dt.date", "dt.date", list[str]] | None:
    """The period a verbatim quote names, as (start, end, evidence), or None.

    Reads explicit time references only: four-digit years from 1990 to 2099
    (a citation year such as "2025a" included), "As of 6 October",
    "From 16 September to 6 October 2026", and month names in English,
    French and Spanish, alone or with a year. A month with no year takes
    the year of a range partner that carries one, else the year of
    ``reference_date`` (the document's date), stepping back a year when
    that would put the month after the document's own month. ``evidence``
    lists the matched spans, for the ledger.

    The figure's date is the EARLIEST start and LATEST end any reference
    names; the caller applies the start-month rule to the start. A quote
    that names no time returns None.
    """

    text = str(quote or "")
    if not text.strip():
        return None
    mentions: list[dict[str, Any]] = []
    for match in _DATE_RE.finditer(text):
        groups = match.groupdict()
        if groups.get("y4"):
            after = text[match.end():match.end() + 20]
            before = text[max(0, match.start() - 12):match.start()]
            if _COUNT_AFTER_RE.match(after) or _COMPARISON_BEFORE_RE.search(before):
                continue
            # A year preceded by a digit-comma is part of a count ("1,2025").
            if match.start() >= 2 and text[match.start() - 1] in ",." and text[match.start() - 2].isdigit():
                continue
            mentions.append({
                "span": match.span(), "text": match.group(0),
                "year": int(groups["y4"]), "month": None, "day": None,
                "named_year": True,
            })
            continue
        for m_key, d_key, y_key in (("m1", "d1", "y1"), ("m2", "d2", "y2"), ("m3", None, "y3")):
            if groups.get(m_key):
                month = _month_number(groups[m_key])
                if month is None:
                    break
                day_text = groups.get(d_key) if d_key else None
                day = int(re.match(r"\d+", day_text).group(0)) if day_text else None
                year = int(groups[y_key]) if groups.get(y_key) else None
                first_day = (
                    int(re.match(r"\d+", groups["d0"]).group(0))
                    if m_key == "m1" and groups.get("d0") else None
                )
                mentions.append({
                    "span": match.span(), "text": match.group(0),
                    "year": year, "month": month, "day": day,
                    "first_day": first_day,
                    "named_year": year is not None,
                })
                break
    if not mentions:
        return None

    # Borrow a year from the right across a range connector, right to left
    # so a chain of yearless dates inherits from the one that has a year.
    for index in range(len(mentions) - 2, -1, -1):
        current, nxt = mentions[index], mentions[index + 1]
        if current["year"] is not None or current["month"] is None:
            continue
        if nxt["year"] is None or nxt["month"] is None:
            continue
        gap = text[current["span"][1]:nxt["span"][0]]
        if not _RANGE_GAP_RE.match(gap):
            continue
        year = nxt["year"]
        if current["month"] > nxt["month"]:
            year -= 1
        current["year"] = year

    starts: list[dt.date] = []
    ends: list[dt.date] = []
    evidence: list[str] = []
    for mention in mentions:
        year, month, day = mention["year"], mention["month"], mention["day"]
        if month is not None and year is None:
            if reference_date is None:
                continue
            year = reference_date.year
            if month > reference_date.month:
                year -= 1
        if month is None:
            start, end = dt.date(year, 1, 1), dt.date(year, 12, 31)
        elif day is not None:
            point = _valid_day(year, month, day)
            if point is None:
                from resolver.hazard_resolution.sources import month_bounds

                start, end = month_bounds(f"{year:04d}-{month:02d}")
            else:
                start = end = point
                first = mention.get("first_day")
                if first is not None and first <= day:
                    start = _valid_day(year, month, first) or point
        else:
            from resolver.hazard_resolution.sources import month_bounds

            start, end = month_bounds(f"{year:04d}-{month:02d}")
        starts.append(start)
        ends.append(end)
        evidence.append(mention["text"])
    if not starts:
        return None
    return min(starts), max(ends), evidence


def quote_named_years(quote: str) -> list[int]:
    """The years a quote NAMES outright (not ones inferred for a bare month)."""

    found = quote_period_mentions(quote)
    return sorted({m["year"] for m in found if m["named_year"]})


def quote_period_mentions(quote: str) -> list[dict[str, Any]]:
    """Every explicit-year mention in ``quote`` (used by the year test)."""

    text = str(quote or "")
    out: list[dict[str, Any]] = []
    for match in _DATE_RE.finditer(text):
        groups = match.groupdict()
        year_text = groups.get("y4") or groups.get("y1") or groups.get("y2") or groups.get("y3")
        if not year_text:
            continue
        if groups.get("y4"):
            after = text[match.end():match.end() + 20]
            before = text[max(0, match.start() - 12):match.start()]
            if _COUNT_AFTER_RE.match(after) or _COMPARISON_BEFORE_RE.search(before):
                continue
            if match.start() >= 2 and text[match.start() - 1] in ",." and text[match.start() - 2].isdigit():
                continue
        out.append({"text": match.group(0), "year": int(year_text), "named_year": True})
    return out


def names_only_earlier_years(text: str, cell_year: int) -> list[int]:
    """The years ``text`` names when EVERY one of them is before ``cell_year``.

    Empty when the text names no year, or names this year or a later one.
    """

    years = quote_named_years(text)
    if years and all(year < cell_year for year in years):
        return years
    return []


def _reference_date(figure: ExtractedFigure) -> "dt.date | None":
    from resolver.hazard_resolution.sources import parse_date

    return parse_date(figure.doc_date_original) or parse_date(figure.doc_date)


def figure_period(
    figure: ExtractedFigure, *, read_quote: bool = True
) -> tuple["dt.date | None", "dt.date | None", str, list[str]]:
    """``(start, end, field, evidence)`` for the period a figure is about.

    In order: the figure's own date or period (``YYYY``, ``YYYY-MM`` and a
    full date all parse); the period its verbatim quote names; the
    document's ``date.original``; the document's ``date.created``. Where the
    figure's own date lies inside a longer period the quote names ("From
    16 September to 6 October 2026" beside a figure dated 6 October), the
    quote's start is taken, because the start-month rule attributes a
    figure to the month its event STARTED. Returns (None, None, "", [])
    when nothing dates it.
    """

    from resolver.hazard_resolution.sources import parse_date, parse_period

    if read_quote:
        own = parse_period(figure.date)
    else:
        # The pre-November rule, kept whole behind the switch: a figure's
        # own date counted only when it was a full date.
        day = parse_date(figure.date)
        own = (day, day) if day is not None else None
    quoted = (
        quote_period(figure.quote, reference_date=_reference_date(figure))
        if read_quote else None
    )
    if own is not None:
        if quoted is not None and quoted[0] < own[0] <= quoted[1]:
            return quoted[0], max(own[1], quoted[1]), FIELD_FIGURE_AND_QUOTE, quoted[2]
        return own[0], own[1], FIELD_FIGURE_DATE, []
    if quoted is not None:
        return quoted[0], quoted[1], FIELD_QUOTE, quoted[2]
    for field, raw in (
        (FIELD_DOC_ORIGINAL, figure.doc_date_original),
        (FIELD_DOC_CREATED, figure.doc_date),
    ):
        parsed = parse_date(raw)
        if parsed is not None:
            return parsed, parsed, field, []
    return None, None, "", []


def figure_window_date(
    figure: ExtractedFigure, *, read_quote: bool = True
) -> tuple["dt.date | None", str]:
    """The date that decides a figure's cell, and which field supplied it.

    The START of :func:`figure_period`: a figure belongs to the month its
    event started (``rules.attribution_month``), so "From 16 September to
    6 October 2026" is a September figure and only a September figure.
    """

    start, _end, field, _evidence = figure_period(figure, read_quote=read_quote)
    return start, field


def reporting_window(ym: str, rulebook: Rulebook) -> tuple["dt.date", "dt.date"]:
    """The dates a figure may be about and still belong to the cell ``ym``.

    Month start to month end plus ``reliefweb.documents.publication_pad_days``:
    a figure is attributed to the month its event STARTED (rules.attribution_month),
    and a report published in the pad after month end may restate it.
    """

    from resolver.hazard_resolution.sources import month_bounds

    start, end = month_bounds(ym)
    pad = int(rulebook.get("reliefweb.documents.publication_pad_days"))
    return start, end + dt.timedelta(days=pad)


def _switch(rulebook: Rulebook, key: str) -> bool:
    """A rulebook switch that is ON when absent (a fixture rulebook)."""

    return bool(rulebook.get(key, True))


def apply_attribution(
    pairs: list[tuple[ExtractedFigure, float]],
    iso3: str,
    ym: str,
    rulebook: Rulebook,
) -> tuple[list[tuple[ExtractedFigure, float]], list[dict[str, Any]]]:
    """Keep only figures that are about THIS country and THIS period.

    * **country** — the document's ``primary_country`` must be the cell's
      country (``reliefweb.documents.require_primary_country``). ReliefWeb
      files a regional round-up under every country it mentions.
    * **earlier years** — a quote that names years, all of them before the
      cell's year, is about an earlier event
      (``reliefweb.documents.enforce_quote_period``): "The 2022 floods ...
      affecting 33 million people" resolved PAK / FL / 2026-10 in the
      November 2026 review.
    * **window** — the START of the period the figure is about
      (:func:`figure_period`) must fall inside :func:`reporting_window`
      (``reliefweb.documents.enforce_reporting_window``). The figure's own
      date comes first, then the time its quote names, then the document's
      dates. A figure with nothing to date it is kept.
    * **document about an earlier event** — once a document's dated figures
      in this cell are rejected as dated BEFORE the window, or its title
      names only earlier years, its undated figures (those dated only by the
      document's publication date) are rejected too
      (``reliefweb.documents.reject_undated_in_dated_document``). Cuba's
      645,000 people, from a document about Hurricane Melissa (October
      2025), is the case.
    """

    iso3 = iso3.upper()
    require_primary = bool(rulebook.get("reliefweb.documents.require_primary_country"))
    enforce_window = bool(rulebook.get("reliefweb.documents.enforce_reporting_window"))
    read_quote = _switch(rulebook, "reliefweb.documents.enforce_quote_period")
    reject_undated = _switch(rulebook, "reliefweb.documents.reject_undated_in_dated_document")
    start, end = reporting_window(ym, rulebook)
    cell_year = int(ym.split("-")[0])
    kept: list[tuple[ExtractedFigure, float, str]] = []
    rejected: list[dict[str, Any]] = []
    #: Documents shown to be about an event before this cell's window.
    earlier_docs: dict[str, str] = {}
    for figure, value in pairs:
        primary = (figure.doc_primary_country or "").upper()
        if require_primary and primary and primary != iso3:
            rejected.append({
                "doc_id": figure.doc_id,
                "reason": REASON_PRIMARY_COUNTRY_MISMATCH,
                "value": value,
                "unit": figure.unit,
                "quote": figure.quote,
                "doc_primary_country": primary,
                "doc_country_iso3s": list(figure.doc_country_iso3s),
                "cell_iso3": iso3,
            })
            continue
        if read_quote:
            earlier = names_only_earlier_years(figure.quote, cell_year)
            if earlier:
                rejected.append({
                    "doc_id": figure.doc_id,
                    "reason": REASON_QUOTE_EARLIER_PERIOD,
                    "value": value,
                    "unit": figure.unit,
                    "quote": figure.quote,
                    "quote_years": earlier,
                    "cell_ym": ym,
                })
                earlier_docs.setdefault(str(figure.doc_id), REASON_QUOTE_EARLIER_PERIOD)
                continue
        about, _about_end, field, evidence = figure_period(figure, read_quote=read_quote)
        if enforce_window and about is not None and not (start <= about <= end):
            rejected.append({
                "doc_id": figure.doc_id,
                "reason": REASON_OUTSIDE_WINDOW,
                "value": value,
                "unit": figure.unit,
                "quote": figure.quote,
                "figure_about": about.isoformat(),
                "figure_about_field": field,
                "quote_period_evidence": evidence,
                "window_start": start.isoformat(),
                "window_end": end.isoformat(),
                "cell_ym": ym,
            })
            if about < start and field in _FIGURE_OWN_FIELDS:
                earlier_docs.setdefault(str(figure.doc_id), REASON_OUTSIDE_WINDOW)
            continue
        kept.append((figure, value, field))

    out: list[tuple[ExtractedFigure, float]] = []
    for figure, value, field in kept:
        undated = field not in _FIGURE_OWN_FIELDS
        if reject_undated and undated:
            doc = str(figure.doc_id)
            why = earlier_docs.get(doc)
            title_years = names_only_earlier_years(figure.doc_title, cell_year) if read_quote else []
            if why or title_years:
                rejected.append({
                    "doc_id": figure.doc_id,
                    "reason": REASON_DOCUMENT_EARLIER_EVENT,
                    "value": value,
                    "unit": figure.unit,
                    "quote": figure.quote,
                    "figure_about_field": field,
                    "document_evidence": (
                        f"a dated figure in this document was rejected ({why})"
                        if why else f"title names only earlier years {title_years}"
                    ),
                    "cell_ym": ym,
                })
                continue
        out.append((figure, value))
    if rejected:
        LOG.info(
            "[figures] %s %s: %d figure(s) rejected as not attributable to this cell "
            "(%s)",
            iso3, ym, len(rejected),
            ", ".join(sorted({str(r["reason"]) for r in rejected})),
        )
    return out, rejected


# ---------------------------------------------------------------------------
# Households against a stated people count
# ---------------------------------------------------------------------------

_STATED_PEOPLE_RE = re.compile(
    r"(?P<num>\d{1,3}(?:[,.\s]\d{3})+|\d+(?:[.,]\d+)?)\s*"
    r"(?P<mult>million|thousand|m\b)?\s*"
    r"(?:people|persons|individuals|personnes|personas|habitants)\b",
    re.I,
)


def stated_people_counts(quote: str) -> list[float]:
    """Every people count the quote states outright ("4,188,452 people")."""

    out: list[float] = []
    for match in _STATED_PEOPLE_RE.finditer(str(quote or "")):
        raw, mult = match.group("num"), (match.group("mult") or "").lower()
        try:
            if mult:
                number = float(raw.replace(",", ".")) * (
                    1_000_000 if mult in ("million", "m") else 1_000
                )
            else:
                number = float(re.sub(r"[,.\s]", "", raw))
        except ValueError:
            continue
        out.append(number)
    return out


def apply_household_rules(
    pairs: list[tuple[ExtractedFigure, float]], rulebook: Rulebook
) -> tuple[list[tuple[ExtractedFigure, float]], list[dict[str, Any]]]:
    """A households figure whose own quote states the people count is dropped.

    "1,474,981 households (4,188,452 people)" is a statement of 4,188,452
    people; multiplying the households by five gave Thailand 7,374,905 in
    two months of the November 2026 review. When the extraction returned
    the people reading too, :func:`deduplicate` keeps it
    (:data:`REASON_HOUSEHOLD_SUPERSEDED`); when it returned only the
    households reading, the figure is rejected here
    (:data:`REASON_HOUSEHOLD_WITH_PEOPLE`) and no people figure is made up
    from the quote, because this module transcribes and never infers. The
    stated people count must be at least the household count, so "5,000
    households, including 300 people with disabilities" is not read as the
    same group. Switch: ``reliefweb.household_conversion.prefer_stated_people``.
    """

    if not _switch(rulebook, "reliefweb.household_conversion.prefer_stated_people"):
        return pairs, []
    people_quotes = {
        _normalise(figure.quote) for figure, _v in pairs if figure.unit == UNIT_PEOPLE
    }
    kept: list[tuple[ExtractedFigure, float]] = []
    rejected: list[dict[str, Any]] = []
    for figure, value in pairs:
        if figure.unit == UNIT_HOUSEHOLDS and _normalise(figure.quote) not in people_quotes:
            counts = [c for c in stated_people_counts(figure.quote) if c >= float(figure.value)]
            if counts:
                rejected.append({
                    "doc_id": figure.doc_id,
                    "reason": REASON_HOUSEHOLD_WITH_PEOPLE,
                    "value": value,
                    "stated_value": figure.value,
                    "unit": figure.unit,
                    "quote": figure.quote,
                    "stated_people": counts,
                })
                continue
        kept.append((figure, value))
    return kept, rejected


# ---------------------------------------------------------------------------
# One figure, one hazard
# ---------------------------------------------------------------------------

_NAMED_STORM_RE = re.compile(
    r"\b(?:Hurricane|Typhoon|Super Typhoon|Cyclone|Tropical Storm|Tropical Depression|"
    r"Huracán|Huracan|Ouragan|Tifón|Tifon|Tormenta Tropical|Tempête tropicale)\s+[A-Z][a-zà-ÿ]+"
)
_FLOOD_NAMED_RE = re.compile(
    r"\b(?:flood(?:s|ing|ed)?|flash[- ]floods?|inundations?|inondations?|inundaciones)\b",
    re.I,
)


def _cyclone_keywords(rulebook: Rulebook) -> list[str]:
    return [
        str(k).strip().lower()
        for k in (rulebook.get("cyclone.reliefweb_sweep.keywords", []) or [])
        if str(k).strip()
    ]


def names_a_cyclone(text: str, keywords: list[str]) -> bool:
    """True when ``text`` names a cyclone by keyword or by a storm's name."""

    lowered = str(text or "").lower()
    for keyword in keywords:
        if re.search(rf"(?<![\w-]){re.escape(keyword)}s?(?![\w])", lowered):
            return True
    return bool(_NAMED_STORM_RE.search(str(text or "")))


def is_cyclone_figure(figure: ExtractedFigure, keywords: list[str]) -> bool:
    """A figure belongs to the cyclone cell when its document is about a cyclone.

    The document's ReliefWeb disaster types say so ("Tropical Cyclone"),
    or its title or the quote names one. Documents cached before disaster
    types were kept fall back to the keyword test alone.
    """

    types = {str(t).strip().lower() for t in (figure.doc_disaster_types or ())}
    if "tropical cyclone" in types:
        return True
    return names_a_cyclone(figure.doc_title, keywords) or names_a_cyclone(
        figure.quote, keywords
    )


def apply_one_hazard(
    pairs: list[tuple[ExtractedFigure, float]],
    hazard: str,
    rulebook: Rulebook,
    all_figures: list[ExtractedFigure] | None = None,
) -> tuple[list[tuple[ExtractedFigure, float]], list[dict[str, Any]]]:
    """Keep a cyclone document's figures out of the flood cell.

    In the November 2026 review one document filed under both Flood and
    Tropical Cyclone resolved both hazards (Laos 403,133 people; Cuba
    645,000; Viet Nam 100,000), so the same people were counted twice. A
    cyclone figure is admitted to the TC cell only. The exception: the
    quote attributes the figure to flooding by name and the same document
    states a separate figure for the cyclone; then the flood figure is a
    flood figure. Switch: ``reliefweb.documents.one_hazard_per_figure``.
    """

    if str(hazard).upper() != "FL" or not _switch(
        rulebook, "reliefweb.documents.one_hazard_per_figure"
    ):
        return pairs, []
    keywords = _cyclone_keywords(rulebook)
    pool = list(all_figures if all_figures is not None else [f for f, _ in pairs])
    kept: list[tuple[ExtractedFigure, float]] = []
    rejected: list[dict[str, Any]] = []
    for figure, value in pairs:
        if not is_cyclone_figure(figure, keywords):
            kept.append((figure, value))
            continue
        if _FLOOD_NAMED_RE.search(figure.quote or "") and not names_a_cyclone(
            figure.quote, keywords
        ):
            separate = any(
                str(other.doc_id) == str(figure.doc_id)
                and float(other.value) != float(figure.value)
                and names_a_cyclone(other.quote, keywords)
                for other in pool
            )
            if separate:
                kept.append((figure, value))
                continue
        rejected.append({
            "doc_id": figure.doc_id,
            "reason": REASON_CYCLONE_IN_FLOOD,
            "value": value,
            "unit": figure.unit,
            "quote": figure.quote,
            "doc_disaster_types": list(figure.doc_disaster_types or ()),
        })
    return kept, rejected


def apply_ceiling(
    figures: list[tuple[ExtractedFigure, float]],
    exposure_ceiling: float | None,
    rulebook: Rulebook,
    ceiling_basis: dict[str, Any] | None = None,
) -> tuple[list[tuple[ExtractedFigure, float]], list[dict[str, Any]]]:
    """Drop figures above the GDACS exposure ceiling. Returns (kept, rejected).

    ``ceiling_basis`` (from :func:`candidates.exposure_ceiling_basis`) names
    the event and the response field the ceiling came from. It is optional
    only so existing callers keep working: without it a rejection records
    the number and not its origin, and a ceiling of 2 against a reported
    40,000 cannot be told apart from a genuine mis-transcription.

    It takes no hazard on purpose. A hazard whose GDACS figure is not a
    population exposure arrives here with no ceiling at all, because
    :func:`candidates.exposure_ceiling_basis` declines to supply one — and
    the ceiling that DOES arrive may be the population share, which is a
    real bound whatever the hazard.
    """

    basis = ceiling_basis or {}
    kept: list[tuple[ExtractedFigure, float]] = []
    rejected: list[dict[str, Any]] = []
    for figure, value in figures:
        if within_sanity_ceiling(value, exposure_ceiling, rulebook):
            kept.append((figure, value))
            continue
        rejected.append(
            {
                "reason": "exceeds_gdacs_exposure_ceiling",
                "doc_id": figure.doc_id,
                "value": value,
                "unit": figure.unit,
                "quote": figure.quote,
                "exposed_population": exposure_ceiling,
                "ceiling_multiplier": rulebook.get("sanity.ceiling_multiplier"),
                "ceiling_source": basis.get("source"),
                "ceiling_source_ref": basis.get("source_ref"),
                "ceiling_field": basis.get("field"),
                "ceiling_basis": basis.get("basis"),
                "ceiling_events_seen": basis.get("n_events"),
                "ceiling_events_with_exposure": basis.get("n_events_with_exposure"),
            }
        )
    if rejected:
        LOG.warning(
            "[figures] rejected %d extracted figure(s) above the GDACS exposure "
            "ceiling (%.0f x %s) — an extracted figure above modelled exposure is "
            "far more likely a mis-transcription than a discovery",
            len(rejected),
            float(rulebook.get("sanity.ceiling_multiplier")),
            f"{exposure_ceiling:.0f}" if exposure_ceiling is not None else "unknown",
        )
    return kept, rejected


def _source_ref(figure: ExtractedFigure) -> str:
    """A stable per-figure identifier: the document plus the quote's digest.

    Several figures can come from one document, so the document id alone
    would collide on the candidate table's uniqueness constraint. Hashing
    the quote keeps the reference stable across runs — the same sentence
    always yields the same reference.
    """

    digest = hashlib.sha1(_normalise(figure.quote).encode("utf-8")).hexdigest()[:8]
    return f"{figure.doc_id}#{digest}"


def build_candidates(
    extraction,
    *,
    iso3: str,
    ym: str,
    hazard: str,
    rulebook: Rulebook,
    exposure_ceiling: float | None = None,
    ceiling_basis: dict[str, Any] | None = None,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """The whole deterministic pipeline: figures in, ranked candidates out.

    ``preference_rank`` is 0 for the figure the rulebook's precedence
    prefers, and the reconciler reads it instead of falling back to
    "largest wins" on this rung.
    """

    iso3 = iso3.upper()
    rejected: list[dict[str, Any]] = list(extraction.rejected)

    converted: list[tuple[ExtractedFigure, float, dict[str, Any] | None]] = []
    for figure in extraction.figures:
        value, conversion = convert_units(figure, iso3, rulebook)
        converted.append((figure, value, conversion))

    conversions = {id(figure): conversion for figure, _v, conversion in converted}
    pairs = [(figure, value) for figure, value, _c in converted]

    pairs, attribution_rejects = apply_attribution(pairs, iso3, ym, rulebook)
    rejected.extend(attribution_rejects)

    pairs, hazard_rejects = apply_one_hazard(
        pairs, hazard, rulebook, all_figures=list(extraction.figures)
    )
    rejected.extend(hazard_rejects)

    pairs, household_rejects = apply_household_rules(pairs, rulebook)
    rejected.extend(household_rejects)

    pairs, ceiling_rejects = apply_ceiling(
        pairs, exposure_ceiling, rulebook, ceiling_basis
    )
    rejected.extend(ceiling_rejects)

    pairs, duplicates = deduplicate(pairs, rulebook)
    rejected.extend(duplicates)

    candidates: list[Candidate] = []
    for rank, (figure, value) in enumerate(pairs):
        # A figure the model found no in-text attribution for is still
        # attributable: the DOCUMENT is the attribution. A figure in an OCHA
        # situation report that names nobody is OCHA's, and treating it as
        # unattributed put 558 of 950 accepted figures — 59% — outside
        # `preference_rank`, which exists precisely to rank a named
        # authority above an unnamed one.
        #
        # It is an INFERENCE, not a transcription, so it is labelled: the
        # detail says whether the attribution came from the text or from the
        # document, and only the first is what the source stated.
        stated_by, attribution_source = _attribute(figure)
        tier_rank, tier = authority_tier(stated_by, rulebook)
        conversion = conversions.get(id(figure))
        candidates.append(
            Candidate(
                iso3=iso3,
                ym=ym,
                hazard=hazard,
                value=float(value),
                value_type=VALUE_AFFECTED,
                source=SOURCE,
                source_ref=_source_ref(figure),
                stated_by=stated_by,
                doc_url=figure.doc_url or None,
                extraction_model=figure.model,
                preference_rank=rank,
                detail={
                    "quote": figure.quote,
                    "unit_as_stated": figure.unit,
                    "stated_value": figure.value,
                    "area": figure.area,
                    "figure_date": figure.date,
                    "cumulative_or_new": figure.cumulative_or_new,
                    "authority_tier": tier,
                    "stated_by_source": attribution_source,
                    "authority_rank": tier_rank,
                    "doc_id": figure.doc_id,
                    "doc_title": figure.doc_title,
                    "doc_date": figure.doc_date,
                    "doc_date_original": figure.doc_date_original,
                    "doc_primary_country": figure.doc_primary_country,
                    "household_conversion": conversion,
                },
            )
        )

    LOG.info(
        "[figures] %s %s %s: %d extracted figure(s) -> %d candidate(s), %d rejected",
        iso3, hazard, ym, len(extraction.figures), len(candidates), len(rejected),
    )
    _record_figures(
        iso3=iso3, ym=ym, hazard=hazard,
        candidates=candidates, rejected=rejected,
        exposure_ceiling=exposure_ceiling, ceiling_basis=ceiling_basis or {},
        rulebook=rulebook,
    )
    return candidates, rejected


def _record_figures(
    *,
    iso3: str,
    ym: str,
    hazard: str,
    candidates: list[Candidate],
    rejected: list[dict[str, Any]],
    exposure_ceiling: float | None,
    ceiling_basis: dict[str, Any],
    rulebook: Rulebook,
) -> None:
    """Append every figure and its fate to the run's evidence stream.

    Kept AND rejected: a ledger listing only the survivors cannot answer
    "what was rejected, against what ceiling", which is the question a
    reader has when the machine resolves less than it should.
    """

    if not cell_ledger.enabled():
        return
    multiplier = rulebook.get("sanity.ceiling_multiplier")
    for candidate in candidates:
        detail = candidate.detail or {}
        conversion = detail.get("household_conversion") or {}
        cell_ledger.record_figure(
            iso3=iso3, hazard=hazard, ym=ym, outcome="accepted",
            doc_id=str(detail.get("doc_id") or ""),
            doc_url=candidate.doc_url,
            value=candidate.value,
            unit=str(detail.get("unit_as_stated") or ""),
            stated_value=detail.get("stated_value"),
            stated_unit=str(detail.get("unit_as_stated") or ""),
            value_persons=int(round(float(candidate.value))),
            conversion_factor=(
                float(conversion["multiplier"]) if conversion
                else conversion_for_unit(
                    detail.get("unit_as_stated"), iso3, rulebook
                )[0]
            ),
            conversion_factor_origin=(
                str(conversion.get("multiplier_origin") or "") if conversion
                else conversion_for_unit(
                    detail.get("unit_as_stated"), iso3, rulebook
                )[1]
            ),
            figure_date=str(detail.get("figure_date") or ""),
            doc_date=str(detail.get("doc_date") or ""),
            doc_date_original=str(detail.get("doc_date_original") or ""),
            doc_primary_country=str(detail.get("doc_primary_country") or ""),
            stated_by=candidate.stated_by,
            quote=str(detail.get("quote") or ""),
            ceiling=exposure_ceiling,
            ceiling_multiplier=multiplier if exposure_ceiling is not None else None,
            ceiling_source=ceiling_basis.get("source"),
            ceiling_source_ref=ceiling_basis.get("source_ref"),
            ceiling_field=ceiling_basis.get("field"),
            ceiling_basis=ceiling_basis.get("basis"),
            preference_rank=candidate.preference_rank,
            detail={
                "authority_tier": detail.get("authority_tier"),
                "household_conversion": detail.get("household_conversion"),
                "figure_date": detail.get("figure_date"),
                # Why a blank ceiling is blank. Without these a reader
                # cannot tell "GDACS listed no event" from "GDACS listed
                # events and described none of them", and only the second
                # is an enrichment failure to chase.
                "ceiling_events_seen": ceiling_basis.get("n_events"),
                "ceiling_events_with_exposure": ceiling_basis.get(
                    "n_events_with_exposure"
                ),
                "ceiling_events_below_plausible_floor": ceiling_basis.get(
                    "n_events_below_plausible_floor"
                ),
            },
        )
    for entry in rejected:
        # A rejected figure gets the same unit accounting as an accepted
        # one. It is still evidence — "306 houses were affected, impacting
        # 1,917 people" rejected against a ceiling of 20 is a reading a
        # human has to be able to check — and a blank unit column made 74
        # household rejections unreadable.
        factor, factor_origin = conversion_for_unit(
            entry.get("unit"), iso3, rulebook
        )
        stated = entry.get("stated_value", entry.get("value"))
        persons: int | None = None
        try:
            if stated is not None:
                persons = int(round(float(stated) * factor))
        except (TypeError, ValueError):
            persons = None
        cell_ledger.record_figure(
            iso3=iso3, hazard=hazard, ym=ym, outcome="rejected",
            doc_id=str(entry.get("doc_id") or ""),
            value=entry.get("value"),
            unit=str(entry.get("unit") or ""),
            stated_value=stated,
            stated_unit=str(entry.get("unit") or ""),
            value_persons=persons,
            conversion_factor=factor,
            conversion_factor_origin=factor_origin,
            figure_date=str(entry.get("figure_about") or entry.get("figure_date") or ""),
            doc_primary_country=str(entry.get("doc_primary_country") or ""),
            quote=str(entry.get("quote") or ""),
            reason=str(entry.get("reason") or ""),
            ceiling=entry.get("exposed_population", exposure_ceiling),
            ceiling_multiplier=entry.get("ceiling_multiplier", multiplier),
            ceiling_source=entry.get("ceiling_source", ceiling_basis.get("source")),
            ceiling_source_ref=entry.get(
                "ceiling_source_ref", ceiling_basis.get("source_ref")
            ),
            ceiling_field=entry.get("ceiling_field", ceiling_basis.get("field")),
            ceiling_basis=entry.get("ceiling_basis", ceiling_basis.get("basis")),
            detail={
                # Why a blank ceiling is blank, on a rejected row too. Only
                # apply_ceiling's own rejections carried these; every other
                # reason (a duplicate quote, a document about another
                # country) rendered the counts as blank, so a reader
                # comparing rejections could not see that the cell had no
                # bound at all.
                "ceiling_events_seen": ceiling_basis.get("n_events"),
                "ceiling_events_with_exposure": ceiling_basis.get(
                    "n_events_with_exposure"
                ),
                "ceiling_events_below_plausible_floor": ceiling_basis.get(
                    "n_events_below_plausible_floor"
                ),
                **{
                    k: v
                    for k, v in entry.items()
                    if k
                    not in {
                        "doc_id", "value", "unit", "quote", "reason",
                        "exposed_population", "ceiling_multiplier",
                        "ceiling_source", "ceiling_source_ref", "ceiling_field",
                        "ceiling_basis",
                    }
                },
            },
        )
