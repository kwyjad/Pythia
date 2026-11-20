# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""A country's GDACS alert history, counted in CALENDAR MONTHS.

Binary EVENT_OCCURRENCE questions ask whether GDACS will issue an Orange or
Red alert naming the country in a month. Two prompt blocks describe the
history behind that question: the base-rate section
(``binary_prompts._query_base_rate``) and the GDACS event history
(``history_loaders._build_gdacs_event_history``). Until Oct 2026 both counted
the country's ROWS in ``facts_resolved``:

* the denominator was the number of rows, not of months. GDACS writes a row
  only where an event was listed, so a quiet month had no row and did not
  count, and a month carried by several rows (re-ingests, several
  ``series_semantics``) counted several times;
* the window was the span of the COUNTRY's own rows, so a country with three
  Green drought months in the table read "2026-05 to 2026-07 ... 0 of 9
  months" — a base rate built from nine rows, three months and no quiet
  month at all, printed as though it were a history.

Five Track-1 drought questions (ETH, SLV, SSD, HND, SOM) resolved "yes" in
August 2026 against ensemble probabilities of 1.7% to 15%, with exactly that
block in front of the models.

This module counts the months the SOURCE covered: the window runs from the
first month GDACS has any row for the hazard (in any country) to the last
complete month it has reached, every calendar month in it counts once, and a
month is an event month when any row for it is Orange or Red (value >= 1). A
country with no row in a covered month had no qualifying event — GDACS is
satellite-global — which is the rule ``compute_resolutions`` applies.

Where the source's own window is shorter than ``MIN_HISTORY_MONTHS`` the
result says so (``history_available = False``) and the renderers print a
"history unavailable" line: a confident 0% built from a few months is the
fault this module exists to end.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Optional

#: Below this many covered months the history is reported as unavailable
#: rather than printed as a rate. Two years is the least that can show one
#: season twice.
MIN_HISTORY_MONTHS = 24

_ALERT_RANK = {"RED": 3, "ORANGE": 2, "GREEN": 1}


def _ym(value: Any) -> str:
    return str(value or "")[:7]


def _month_add(ym: str, n: int) -> str:
    y, m = int(ym[:4]), int(ym[5:7])
    idx = y * 12 + (m - 1) + n
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def _months_between(start: str, end: str) -> list[str]:
    out: list[str] = []
    cur = start
    while cur <= end:
        out.append(cur)
        cur = _month_add(cur, 1)
    return out


def last_complete_month(today: Optional[date] = None) -> str:
    today = today or datetime.now(timezone.utc).date()
    return _month_add(f"{today.year:04d}-{today.month:02d}", -1)


def gdacs_calendar_series(
    con,
    iso3: str,
    hazard_code: str,
    *,
    today: Optional[date] = None,
    min_months: int = MIN_HISTORY_MONTHS,
    written_before: Optional[datetime] = None,
) -> dict[str, Any]:
    """Return the country's alert history over the source's calendar window.

    Keys: ``months`` (a list of ``{"ym", "occurred", "alertlevel"}``, one per
    calendar month, oldest first), ``window_start``/``window_end``,
    ``total_months``, ``event_months``, ``country_rows`` (rows the country
    has in the window), ``history_available`` and ``unavailable_reason``.
    Never raises: an unreadable table is an unavailable history.

    ``written_before`` restricts the read to rows whose ``created_at`` is
    earlier, which is how a diagnostic rebuilds what the database held on a
    forecast date (``created_at`` never moves for a row a later MERGE
    matched). Production callers leave it None.
    """

    iso3_up = (iso3 or "").upper().strip()
    hz_up = (hazard_code or "").upper().strip()
    empty: dict[str, Any] = {
        "months": [], "window_start": None, "window_end": None,
        "total_months": 0, "event_months": 0, "country_rows": 0,
        "history_available": False, "unavailable_reason": "",
    }
    cutoff = last_complete_month(today)
    written_sql = ""
    written_args: list[Any] = []
    if written_before is not None:
        written_sql = " AND (created_at IS NULL OR created_at < ?)"
        written_args = [written_before]
    try:
        span = con.execute(
            """
            SELECT MIN(substr(CAST(ym AS VARCHAR), 1, 7)),
                   MAX(substr(CAST(ym AS VARCHAR), 1, 7))
            FROM facts_resolved
            WHERE upper(hazard_code) = ? AND lower(metric) = 'event_occurrence'
              AND substr(CAST(ym AS VARCHAR), 1, 7) <= ?
            """ + written_sql,
            [hz_up, cutoff, *written_args],
        ).fetchone()
    except Exception as exc:  # noqa: BLE001
        empty["unavailable_reason"] = f"GDACS rows unreadable ({type(exc).__name__})"
        return empty
    start, end = (span or (None, None))
    if not start or not end:
        empty["unavailable_reason"] = f"no GDACS {hz_up} alert rows in the database"
        return empty

    try:
        rows = con.execute(
            """
            SELECT substr(CAST(ym AS VARCHAR), 1, 7) AS ym,
                   MAX(COALESCE(value, 0)) AS v,
                   string_agg(DISTINCT upper(COALESCE(alertlevel, '')), ',') AS levels,
                   COUNT(*) AS n
            FROM facts_resolved
            WHERE upper(iso3) = ? AND upper(hazard_code) = ?
              AND lower(metric) = 'event_occurrence'
              AND substr(CAST(ym AS VARCHAR), 1, 7) BETWEEN ? AND ?
            """ + written_sql + """
            GROUP BY 1
            """,
            [iso3_up, hz_up, start, end, *written_args],
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        empty["unavailable_reason"] = f"GDACS rows unreadable ({type(exc).__name__})"
        return empty

    by_month: dict[str, tuple[bool, Optional[str]]] = {}
    country_rows = 0
    for ym, v, levels, n in rows:
        country_rows += int(n or 0)
        best = None
        for lvl in str(levels or "").split(","):
            lvl = lvl.strip().upper()
            if lvl in _ALERT_RANK and (best is None or _ALERT_RANK[lvl] > _ALERT_RANK[best]):
                best = lvl
        occurred = float(v or 0) >= 1.0
        by_month[_ym(ym)] = (occurred, best.title() if best else None)

    months = [
        {
            "ym": ym,
            "occurred": by_month.get(ym, (False, None))[0],
            "alertlevel": by_month.get(ym, (False, None))[1],
        }
        for ym in _months_between(start, end)
    ]
    total = len(months)
    out = {
        "months": months,
        "window_start": start,
        "window_end": end,
        "total_months": total,
        "event_months": sum(1 for m in months if m["occurred"]),
        "country_rows": country_rows,
        "history_available": total >= min_months,
        "unavailable_reason": "",
    }
    if total < min_months:
        out["unavailable_reason"] = (
            f"the database holds GDACS {hz_up} alerts for only {total} month(s) "
            f"({start} to {end}); fewer than {min_months}"
        )
    return out


def recent_alert_months(
    con,
    iso3: str,
    hazard_code: str,
    *,
    limit: int = 6,
    today: Optional[date] = None,
) -> list[dict[str, Any]]:
    """The country's latest Orange/Red alert MONTHS, newest first, once each.

    The binary prompt's "Recent significant GDACS event months" list. Until
    Nov 2026 it read the newest six ROWS, and GDACS writes several rows per
    month, so Nicaragua's prompt on 13 Nov 2026 listed 2026-10 four times
    and 2026-11 twice: the repeated-month fault this module fixed in the
    history block, one section higher. Each month appears once with its
    strongest level; the month in progress is kept (an alert in it is news)
    and marked ``in_progress`` so the renderer can say it is incomplete.
    Raises on an unreadable table; the caller decides what that costs.
    """

    current = (today or datetime.now(timezone.utc).date()).strftime("%Y-%m")
    rows = con.execute(
        """
        SELECT substr(CAST(ym AS VARCHAR), 1, 7) AS ym,
               MAX(CASE upper(alertlevel) WHEN 'RED' THEN 3 WHEN 'ORANGE' THEN 2 ELSE 0 END) AS rank,
               MAX(COALESCE(value, 0)) AS v
        FROM facts_resolved
        WHERE upper(iso3) = ? AND upper(hazard_code) = ?
          AND lower(metric) = 'event_occurrence'
          AND COALESCE(value, 0) >= 1
          AND upper(COALESCE(alertlevel, '')) IN ('ORANGE', 'RED')
          AND substr(CAST(ym AS VARCHAR), 1, 7) <= ?
        GROUP BY 1
        ORDER BY 1 DESC
        LIMIT ?
        """,
        [(iso3 or "").upper(), (hazard_code or "").upper(), current, int(limit)],
    ).fetchall()
    return [
        {
            "ym": ym,
            "value": float(v or 0),
            "alertlevel": "Red" if rank == 3 else "Orange",
            "in_progress": ym == current,
        }
        for ym, rank, v in rows
    ]


def seasonal_frequency(months: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Per calendar month: years observed, years with an event, percentage."""

    out: dict[int, dict[str, Any]] = {}
    for cal in range(1, 13):
        obs = [m for m in months if int(m["ym"][5:7]) == cal]
        evt = sum(1 for m in obs if m["occurred"])
        out[cal] = {
            "years_observed": len(obs),
            "years_with_event": evt,
            "frequency_pct": round(evt / len(obs) * 100, 0) if obs else 0.0,
        }
    return out


def unavailable_line(hazard_name: str, country: str, reason: str) -> str:
    """The line printed instead of a rate when the history is too thin."""

    return (
        f"History unavailable for {country} / {hazard_name}: {reason}. "
        "Do NOT read this as a 0% base rate; reason from seasonality, the "
        "climate outlook and current conditions instead."
    )



# ---------------------------------------------------------------------------
# Reading the block back out of a stored prompt
# ---------------------------------------------------------------------------
#
# The writer of the "GDACS EVENT HISTORY" block is
# ``forecaster.prompts._format_gdacs_event_history_for_prompt``; the reader
# lives here, beside the series it renders, so a change to one is made with
# the other in view. It reads both the current format ("N calendar months;
# a month with no alert counts as no event") and the format before Oct 2026
# ("(N months)"), which counted rows and repeated months in its timeline.

_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

_HEADER_RE = re.compile(r"^GDACS EVENT HISTORY \(([A-Za-z]{3})\s*[—\-–]+\s*(.+?)\):\s*$")
_COVERAGE_RE = re.compile(
    r"^Coverage:\s*(\d{4}-\d{2})\s+to\s+(\d{4}-\d{2})\s*\((\d+)\s+(calendar\s+)?months"
)
_RATE_RE = re.compile(
    r"^Overall event rate:\s*(\d+)\s+events?\s+in\s+(\d+)\s+months\s*\(([\d.]+)%\)"
)
_SEASON_RE = re.compile(
    r"^\s+([A-Z][a-z]{2}):\s*(?:(\d+)/(\d+)\s+years had events|no historical data)"
)
_TIMELINE_CELL_RE = re.compile(r"(\d{4}-\d{2}):\s*(--|[A-Za-z?]+)")
_STOP_PREFIXES = ("INTERPRETATION", "END OF QUESTION DATA")


def _timeline_gaps(timeline: list[tuple[str, Optional[str]]]) -> list[str]:
    """Months missing between the first and last timeline entry."""

    yms = sorted({ym for ym, _ in timeline})
    if len(yms) < 2:
        return []
    present = set(yms)
    return [ym for ym in _months_between(yms[0], yms[-1]) if ym not in present]


def parse_gdacs_prompt_block(prompt_text: Optional[str]) -> Optional[dict[str, Any]]:
    """Parse the "GDACS EVENT HISTORY" block of a stored prompt.

    Returns None when the prompt carries no block. Otherwise a dict with
    ``iso3``, ``hazard_label``, ``format`` (``current`` | ``legacy`` |
    ``unavailable``), ``coverage_start``, ``coverage_end``, ``total_months``,
    ``event_months``, ``event_rate_pct``, ``history_unavailable``,
    ``unavailable_reason``, ``seasonal`` (calendar month -> years observed
    and years with events, the forecast months only, as printed),
    ``timeline`` (list of ``(ym, level or None)``), ``repeated_months``
    (timeline months listed more than once) and ``timeline_gaps``.
    Never raises.
    """

    if not prompt_text or "GDACS EVENT HISTORY" not in prompt_text:
        return None
    lines = prompt_text.splitlines()
    start = None
    header = None
    for i, line in enumerate(lines):
        m = _HEADER_RE.match(line.strip())
        if m:
            start, header = i, m
            break
    if start is None or header is None:
        return None

    out: dict[str, Any] = {
        "iso3": header.group(1).upper(),
        "hazard_label": header.group(2).strip(),
        "format": None,
        "coverage_start": None,
        "coverage_end": None,
        "total_months": None,
        "event_months": None,
        "event_rate_pct": None,
        "history_unavailable": False,
        "unavailable_reason": None,
        "seasonal": {},
        "timeline": [],
        "repeated_months": [],
        "timeline_gaps": [],
    }
    section = None
    for raw in lines[start + 1:]:
        line = raw.rstrip()
        stripped = line.strip()
        if stripped.startswith(_STOP_PREFIXES):
            break
        if stripped.startswith("History unavailable for"):
            out["history_unavailable"] = True
            out["format"] = "unavailable"
            m = re.match(r"History unavailable for [^:]+:\s*(.*?)\.\s*Do NOT", stripped)
            out["unavailable_reason"] = m.group(1) if m else None
            break
        m = _COVERAGE_RE.match(stripped)
        if m:
            out["coverage_start"], out["coverage_end"] = m.group(1), m.group(2)
            out["total_months"] = int(m.group(3))
            out["format"] = "current" if m.group(4) else "legacy"
            continue
        m = _RATE_RE.match(stripped)
        if m:
            out["event_months"] = int(m.group(1))
            if out["total_months"] is None:
                out["total_months"] = int(m.group(2))
            out["event_rate_pct"] = float(m.group(3))
            continue
        if stripped.startswith("Seasonal event frequency"):
            section = "seasonal"
            continue
        if stripped.startswith("Recent event timeline"):
            section = "timeline"
            continue
        if not stripped:
            continue
        if section == "seasonal":
            m = _SEASON_RE.match(line)
            if m:
                cal = _MONTH_ABBR.get(m.group(1).lower())
                if cal:
                    obs = int(m.group(3)) if m.group(3) else 0
                    evt = int(m.group(2)) if m.group(2) else 0
                    out["seasonal"][cal] = {"years_observed": obs, "years_with_event": evt}
                continue
        if section == "timeline" and raw.startswith(" "):
            cells = _TIMELINE_CELL_RE.findall(line)
            if cells:
                for ym, lvl in cells:
                    out["timeline"].append((ym, None if lvl == "--" else lvl))
                continue
        if section in ("timeline", "seasonal") and not raw.startswith(" "):
            # Unindented text that is none of the above ends the block.
            break

    counts: dict[str, int] = {}
    for ym, _ in out["timeline"]:
        counts[ym] = counts.get(ym, 0) + 1
    out["repeated_months"] = sorted(ym for ym, n in counts.items() if n > 1)
    out["timeline_gaps"] = _timeline_gaps(out["timeline"])
    return out


_BASE_RATE_HEADER_RE = re.compile(r"HISTORICAL BASE RATE \(GDACS(?:,\s*(\d{4}-\d{2}) to (\d{4}-\d{2}))?\):")
_BASE_RATE_COUNT_RE = re.compile(
    r"in\s+(\d+)\s+of\s+(?:the\s+)?(\d+)\s+(?:calendar\s+)?months[^(]*\(([\d.]+)%\)",
    re.S,
)


def parse_base_rate_section(prompt_text: Optional[str]) -> Optional[dict[str, Any]]:
    """Parse the binary prompt's "HISTORICAL BASE RATE (GDACS ...)" section.

    Returns None when absent; otherwise ``coverage_start``, ``coverage_end``,
    ``event_months``, ``total_months``, ``rate_pct`` and
    ``history_unavailable``. Reads both the current and the pre-Oct-2026
    wording. Never raises.
    """

    if not prompt_text:
        return None
    m = _BASE_RATE_HEADER_RE.search(prompt_text)
    if not m:
        return None
    tail = prompt_text[m.end(): m.end() + 600]
    out: dict[str, Any] = {
        "coverage_start": m.group(1),
        "coverage_end": m.group(2),
        "event_months": None,
        "total_months": None,
        "rate_pct": None,
        "history_unavailable": tail.lstrip().startswith("History unavailable"),
    }
    if not out["history_unavailable"]:
        c = _BASE_RATE_COUNT_RE.search(tail)
        if c:
            out["event_months"] = int(c.group(1))
            out["total_months"] = int(c.group(2))
            out["rate_pct"] = float(c.group(3))
    return out


_RECENT_ALERTS_RE = re.compile(r"Recent significant GDACS event months for [^\n]*:\n((?:\s+- \[[^\]]*\] \d{4}-\d{2}[^\n]*\n?)+)")


def parse_recent_alert_months(prompt_text: Optional[str]) -> list[str]:
    """Months listed under the binary prompt's "Recent significant GDACS
    event months" list, in order and WITH repeats (an empty list when the
    prompt has none)."""

    if not prompt_text:
        return []
    m = _RECENT_ALERTS_RE.search(prompt_text)
    if not m:
        return []
    return re.findall(r"\] (\d{4}-\d{2})", m.group(1))
