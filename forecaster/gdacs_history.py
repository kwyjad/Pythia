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
) -> dict[str, Any]:
    """Return the country's alert history over the source's calendar window.

    Keys: ``months`` (a list of ``{"ym", "occurred", "alertlevel"}``, one per
    calendar month, oldest first), ``window_start``/``window_end``,
    ``total_months``, ``event_months``, ``country_rows`` (rows the country
    has in the window), ``history_available`` and ``unavailable_reason``.
    Never raises: an unreadable table is an unavailable history.
    """

    iso3_up = (iso3 or "").upper().strip()
    hz_up = (hazard_code or "").upper().strip()
    empty: dict[str, Any] = {
        "months": [], "window_start": None, "window_end": None,
        "total_months": 0, "event_months": 0, "country_rows": 0,
        "history_available": False, "unavailable_reason": "",
    }
    cutoff = last_complete_month(today)
    try:
        span = con.execute(
            """
            SELECT MIN(substr(CAST(ym AS VARCHAR), 1, 7)),
                   MAX(substr(CAST(ym AS VARCHAR), 1, 7))
            FROM facts_resolved
            WHERE upper(hazard_code) = ? AND lower(metric) = 'event_occurrence'
              AND substr(CAST(ym AS VARCHAR), 1, 7) <= ?
            """,
            [hz_up, cutoff],
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
            GROUP BY 1
            """,
            [iso3_up, hz_up, start, end],
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
