# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""The resolving source's latest reading (Oct 2026, review Part 5).

Every trial's plan opens with a "resolver" slot (how the question resolves
and the resolving source's latest figures) and a "nowcast" slot (the months
between the reference's last month and today). The trials fill both by
searching the open web for figures the pipeline already holds, or, for
conflict deaths, for a month-to-date count that only ACLED itself can give.
So each question now carries one reading of the resolving source, taken on
the main thread before its trials start and shown to every lane:

* ACE/FATALITIES: a live ACLED read for the country from the first day of the
  month in progress to today. Deaths over all event types, the event count, a
  split by event type and the newest event date, with a note that the month
  is partial and ACLED revises recent weeks. One request, filtered by the
  numeric ``iso`` code, every event attributed by its OWN returned country
  (the ACLED API ignores filters it does not know). Needs
  ``SIBYL_LIVE_LOOKUPS_ENABLED``.
* DR/PHASE3PLUS_IN_NEED: the newest ``phase3plus_in_need`` rows held (the
  series the question resolves on, each the lower bound of FEWS NET's range),
  and the Most Likely projections for the window, labelled as projections.
* FL/TC PA: the last six months of resolving rows, a month with none stated
  as such, then the last three months of GDACS alert levels, which detect an
  event and never resolve one.

Only aggregates reach the prompt: no event-level row is shown or stored.
Nothing is read in backtest (the database holds figures published after the
as-of date). ``build_resolver_reading`` never raises: a read that fails says
so in the block, with its reason scrubbed of anything credential-shaped.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from sibyl import config as _cfg
from sibyl.leakage import is_backtest

logger = logging.getLogger(__name__)

HEADING = "=== RESOLVING SOURCE: LATEST READING (as of {as_of}) ==="
PREFACE = (
    "This is the resolving source's own data, read for you before your trial began. "
    "Use it for the \"resolver\" and \"nowcast\" slots of your plan. It is not a "
    "document you read: it counts toward neither the search nor the document "
    "requirement."
)
LIVE_OK = "ok"
LIVE_UNAVAILABLE = "unavailable"

ACLED_API_BASE_URL = "https://acleddata.com/api/acled/read"
ACLED_FIELDS = "event_date|event_type|fatalities|iso|iso3|country"
ACLED_PAGE_LIMIT = 5000
ACLED_TIMEOUT_SEC = 30
ACLED_MIN_INTERVAL_SEC = 1.0
PA_MONTHS_BACK = 6
GDACS_MONTHS_BACK = 3
PHASE3_ROWS_SHOWN = 6

# Paces the live reads across a run: the reading runs on the main thread, one
# question at a time, so a process-wide clock is enough.
_PACE_LOCK = threading.Lock()
_LAST_REQUEST_AT: List[float] = [0.0]


@dataclass
class ResolverReading:
    """The rendered block and the record stored beside the forecast.

    ``text`` is '' when nothing is shown, which leaves the prompt unchanged.
    ``live`` is None when no live read was attempted.
    """

    text: str = ""
    record: Dict[str, Any] = field(default_factory=dict)
    live: Optional[str] = None

    def evidence_row(self) -> Optional[Dict[str, Any]]:
        """The ``sibyl_evidence`` row for this reading (step 0), or None."""
        if not self.text:
            return None
        return {
            "step": 0,
            "call_index": 0,
            "tool": "resolver_reading",
            "target": str(self.record.get("source") or ""),
            "lane": None,
            "retrieved_at": datetime.now(timezone.utc),
            "http_status": self.record.get("http_status"),
            "ok": self.live != LIVE_UNAVAILABLE,
            "sha256": hashlib.sha256(self.text.encode("utf-8", "replace")).hexdigest(),
            "shown_text": self.text,
            "doc_text": None,
            "doc_chars": None,
        }


def _scrub(text: Any) -> str:
    try:
        from pythia.secret_scrub import scrub_text  # noqa: PLC0415

        return scrub_text(str(text))[:300]
    except Exception:  # noqa: BLE001
        return "error text withheld"


def _num(x: Any) -> str:
    try:
        return f"{float(x):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


def _ym_add(ym: str, k: int) -> str:
    y, m = int(ym[:4]), int(ym[5:7])
    idx = y * 12 + (m - 1) + k
    return f"{idx // 12:04d}-{idx % 12 + 1:02d}"


def _render(as_of: date, body: Sequence[str]) -> str:
    return "\n\n" + "\n".join([HEADING.format(as_of=as_of.isoformat()), PREFACE, *body])


# --- ACE/FATALITIES: a live ACLED read ------------------------------------------------

def _default_get(url: str, params: Dict[str, Any], headers: Dict[str, str], timeout: int) -> Any:
    import requests  # noqa: PLC0415

    return requests.get(url, params=params, headers=headers, timeout=timeout)


def _pace(sleep: Callable[[float], None], clock: Callable[[], float]) -> None:
    with _PACE_LOCK:
        wait = ACLED_MIN_INTERVAL_SEC - (clock() - _LAST_REQUEST_AT[0])
        if wait > 0:
            sleep(wait)
        _LAST_REQUEST_AT[0] = clock()


def acled_month_to_date(
    iso3: str,
    as_of: date,
    *,
    http_get: Optional[Callable[..., Any]] = None,
    token_fn: Optional[Callable[[], str]] = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Dict[str, Any]:
    """One month-to-date ACLED read for *iso3*. Never raises.

    Returns ``{"status": "ok", ...aggregates}`` or ``{"status":
    "unavailable", "reason": ...}``. One request (one retry on a network
    error, 429 or 5xx), at least ``ACLED_MIN_INTERVAL_SEC`` after the last.
    A 403, a refused token or an HTML body is unavailable, never zero deaths.
    """
    from pythia.acled_political import _event_iso3, _iso_numeric  # noqa: PLC0415
    from resolver.ingestion.acled_auth import (  # noqa: PLC0415
        AcledResponseError,
        parse_json_response,
    )

    start = as_of.replace(day=1)
    base: Dict[str, Any] = {"from": start.isoformat(), "to": as_of.isoformat()}
    iso_num = _iso_numeric(iso3)
    if iso_num is None:
        return {**base, "status": LIVE_UNAVAILABLE, "reason": f"no numeric ISO code for {iso3}"}
    try:
        if token_fn is None:
            from resolver.ingestion.acled_auth import get_access_token as token_fn  # noqa: PLC0415
        token = token_fn()
    except Exception as exc:  # noqa: BLE001
        return {**base, "status": LIVE_UNAVAILABLE, "reason": "auth failed: " + _scrub(exc)}

    params = {
        "iso": iso_num,
        "event_date": f"{start.isoformat()}|{as_of.isoformat()}",
        "event_date_where": "BETWEEN",
        "fields": ACLED_FIELDS,
        "limit": ACLED_PAGE_LIMIT,
        "cursor": 0,
        "_format": "json",
    }
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    get = http_get or _default_get
    resp = None
    reason = ""
    for attempt in (1, 2):
        _pace(sleep, clock)
        try:
            resp = get(ACLED_API_BASE_URL, params, headers, ACLED_TIMEOUT_SEC)
        except Exception as exc:  # noqa: BLE001
            resp, reason = None, "network error: " + _scrub(exc)
            continue
        status = getattr(resp, "status_code", None)
        if status == 429 or (isinstance(status, int) and status >= 500):
            reason = f"HTTP {status}"
            resp = None
            continue
        break
    if resp is None:
        return {**base, "status": LIVE_UNAVAILABLE, "reason": reason or "no response"}
    http_status = getattr(resp, "status_code", None)
    base["http_status"] = http_status
    try:
        payload = parse_json_response(resp, what="Sibyl resolver reading")
    except AcledResponseError as exc:
        return {**base, "status": LIVE_UNAVAILABLE, "reason": _scrub(exc)}
    except Exception as exc:  # noqa: BLE001
        return {**base, "status": LIVE_UNAVAILABLE, "reason": _scrub(exc)}
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return {**base, "status": LIVE_UNAVAILABLE, "reason": "response carried no data list"}

    iso_up = iso3.upper()
    mine = [ev for ev in data if isinstance(ev, dict) and _event_iso3(ev) == iso_up]
    by_type: Dict[str, Dict[str, int]] = {}
    deaths = 0
    newest = ""
    for ev in mine:
        try:
            f = int(float(ev.get("fatalities") or 0))
        except (TypeError, ValueError):
            f = 0
        deaths += f
        t = str(ev.get("event_type") or "unclassified")
        slot = by_type.setdefault(t, {"events": 0, "deaths": 0})
        slot["events"] += 1
        slot["deaths"] += f
        d = str(ev.get("event_date") or "")[:10]
        if d > newest:
            newest = d
    # One request per question: a full page means more events exist than
    # were read, and the counts are then a floor, said so in the block.
    truncated = len(data) >= ACLED_PAGE_LIMIT
    return {
        **base,
        "status": LIVE_OK,
        "deaths": deaths,
        "events": len(mine),
        "events_returned": len(data),
        "events_other_country": len(data) - len(mine),
        "by_event_type": by_type,
        "newest_event_date": newest or None,
        "truncated": bool(truncated),
    }


def _acled_lines(country: str, as_of: date, read: Dict[str, Any]) -> List[str]:
    if read.get("status") != LIVE_OK:
        return [
            f"ACLED could not be read for {country} this run ({read.get('reason') or 'no reason given'}). "
            "No month-to-date count is available; do not read its absence as a quiet month."
        ]
    days = monthrange(as_of.year, as_of.month)[1]
    lines = [
        f"ACLED, read live for {country} from {read['from']} to {read['to']} "
        f"(month to date, {as_of.day} of {days} days):",
        f"- reported deaths, all event types: {_num(read['deaths'])}",
        f"- events: {_num(read['events'])}, newest event dated "
        f"{read.get('newest_event_date') or 'none in this period'}",
    ]
    if read.get("by_event_type"):
        parts = [
            f"{t} {v['events']} events, {v['deaths']} deaths"
            for t, v in sorted(read["by_event_type"].items(), key=lambda kv: -kv[1]["deaths"])
        ]
        lines.append("- by event type: " + "; ".join(parts))
    if read.get("truncated"):
        lines.append(f"- the read stopped at {ACLED_PAGE_LIMIT:,} events; the counts above are a floor")
    lines.append(
        "This is a part month. ACLED publishes weekly and revises recent weeks, so the "
        "count will rise and may change; the question resolves on the settled count for "
        "the whole month."
    )
    return lines


# --- DR/PHASE3PLUS_IN_NEED -------------------------------------------------------------

def _cols(con: Any, table: str) -> set:
    """The table's columns; empty when it does not exist. A connection that
    cannot answer raises, so an unreadable database is never reported as a
    table that is absent."""
    rows = con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ?", [table]
    ).fetchall()
    return {r[0] for r in rows}


def _phase3_lines(con: Any, iso3: str, forecast_keys: Sequence[str],
                  as_of: date) -> Tuple[List[str], Dict[str, Any]]:
    cols = _cols(con, "facts_resolved")
    if not cols:
        return ["No phase3plus_in_need record is held (the facts table is absent)."], {"rows": 0}
    high = "value_high" if "value_high" in cols else "CAST(NULL AS DOUBLE)"
    pub = "publication_date" if "publication_date" in cols else "CAST(NULL AS VARCHAR)"
    publisher = "publisher" if "publisher" in cols else "CAST(NULL AS VARCHAR)"
    cutoff = _ym_add(as_of.strftime("%Y-%m"), -36)
    rows = con.execute(
        f"SELECT ym, {publisher}, value, {high}, {pub} FROM facts_resolved "
        "WHERE upper(iso3) = ? AND lower(metric) = 'phase3plus_in_need' AND ym >= ? "
        "AND ym <= ? ORDER BY ym DESC LIMIT ?",
        [iso3.upper(), cutoff, as_of.strftime("%Y-%m"), PHASE3_ROWS_SHOWN],
    ).fetchall()
    lines: List[str] = []
    if rows:
        lines.append("IPC Phase 3+ population (phase3plus_in_need), the series this question "
                     "resolves on, newest rows held:")
        for ym, who, value, hi, published in rows:
            fig = _num(value)
            if hi is not None and value is not None and float(hi) > float(value):
                fig += f" to {_num(hi)}"
            when = f", published {str(published)[:10]}" if published else ""
            lines.append(f"- {ym} ({who or 'publisher not recorded'}{when}): {fig}")
        lines.append("Each figure is the LOWER bound of the range FEWS NET publishes; the "
                     "question resolves on the lower bound.")
    else:
        lines.append("No phase3plus_in_need record is held for this country in the last 36 months.")
    proj: List[Any] = []
    if forecast_keys:
        proj = con.execute(
            f"SELECT ym, value, {high} FROM facts_resolved WHERE upper(iso3) = ? "
            "AND lower(metric) = 'phase3plus_projection' AND ym >= ? AND ym <= ? ORDER BY ym",
            [iso3.upper(), forecast_keys[0], forecast_keys[-1]],
        ).fetchall()
    if proj:
        lines.append("Most Likely projections for the window (FEWS NET's own forecast, not a "
                     "measurement):")
        for ym, value, hi in proj:
            fig = _num(value)
            if hi is not None and value is not None and float(hi) > float(value):
                fig += f" to {_num(hi)}"
            lines.append(f"- {ym}: {fig}")
    else:
        lines.append("No Most Likely projection is held for the window's months.")
    return lines, {"rows": len(rows), "projection_rows": len(proj),
                   "newest_month": rows[0][0] if rows else None}


# --- FL/TC PA ----------------------------------------------------------------------

def _pa_lines(con: Any, iso3: str, hazard: str,
              as_of: date) -> Tuple[List[str], Dict[str, Any]]:
    from pythia.tools.compute_resolutions import PA_FACTS_RESOLVED_METRICS  # noqa: PLC0415

    cols = _cols(con, "facts_resolved")
    if not cols:
        return ["No resolving record is held (the facts table is absent)."], {"months_with_record": 0}
    this_month = as_of.strftime("%Y-%m")
    months = [_ym_add(this_month, -k) for k in range(PA_MONTHS_BACK, 0, -1)]
    publisher = "publisher" if "publisher" in cols else "CAST(NULL AS VARCHAR)"
    marks = ", ".join("?" for _ in PA_FACTS_RESOLVED_METRICS)
    rows = con.execute(
        f"SELECT ym, {publisher}, lower(metric), value FROM facts_resolved "
        f"WHERE upper(iso3) = ? AND upper(hazard_code) = ? AND lower(metric) IN ({marks}) "
        "AND ym >= ? AND ym <= ? ORDER BY ym",
        [iso3.upper(), hazard.upper(), *PA_FACTS_RESOLVED_METRICS, months[0], months[-1]],
    ).fetchall()
    by_month: Dict[str, List[str]] = {}
    for ym, who, metric, value in rows:
        by_month.setdefault(ym, []).append(f"{_num(value)} {metric} ({who or 'publisher not recorded'})")
    lines = [f"People affected, as the resolver reads them, {months[0]} to {months[-1]}:"]
    for ym in months:
        lines.append(f"- {ym}: " + ("; ".join(by_month[ym]) if ym in by_month
                                     else "no record held for this month"))
    gd_months = months[-GDACS_MONTHS_BACK:]
    alert = "alertlevel" if "alertlevel" in cols else "CAST(NULL AS VARCHAR)"
    gd = con.execute(
        f"SELECT ym, {alert}, value FROM facts_resolved WHERE upper(iso3) = ? "
        "AND upper(hazard_code) = ? AND lower(metric) = 'event_occurrence' "
        "AND ym >= ? AND ym <= ? ORDER BY ym",
        [iso3.upper(), hazard.upper(), gd_months[0], gd_months[-1]],
    ).fetchall()
    gd_by = {ym: (lvl or "Green") for ym, lvl, _v in gd}
    lines.append(f"GDACS alert levels, {gd_months[0]} to {gd_months[-1]}: "
                 + "; ".join(f"{ym} {gd_by.get(ym, 'no alert listed')}" for ym in gd_months))
    lines.append("GDACS alerts detect an event; they are not the figure this question resolves on.")
    return lines, {"months_with_record": len(by_month), "gdacs_months_listed": len(gd_by)}


# --- entry point -------------------------------------------------------------------

def build_resolver_reading(
    con: Any,
    question: Any,
    as_of: date,
    *,
    forecast_keys: Sequence[str] = (),
    country_name: str = "",
    http_get: Optional[Callable[..., Any]] = None,
    token_fn: Optional[Callable[[], str]] = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ResolverReading:
    """The reading for one question. Never raises; '' text when switched off."""
    if not _cfg.RESOLVER_READING or _cfg.BACKTEST_MODE or is_backtest(as_of):
        return ResolverReading()
    hazard = str(getattr(question, "hazard_code", "") or "").upper()
    metric = str(getattr(question, "metric", "") or "").upper()
    iso3 = str(getattr(question, "iso3", "") or "").upper()
    country = country_name or iso3
    try:
        if hazard == "ACE" and metric == "FATALITIES":
            if not _cfg.LIVE_LOOKUPS_ENABLED:
                return ResolverReading()
            read = acled_month_to_date(iso3, as_of, http_get=http_get, token_fn=token_fn,
                                       sleep=sleep, clock=clock)
            record = {"source": "acled_live", **read}
            if read.get("status") != LIVE_OK:
                logger.warning("sibyl.resolver_reading: ACLED unavailable for %s: %s",
                               iso3, read.get("reason"))
            return ResolverReading(text=_render(as_of, _acled_lines(country, as_of, read)),
                                   record=record, live=read.get("status"))
        if hazard == "DR" and metric == "PHASE3PLUS_IN_NEED":
            lines, info = _phase3_lines(con, iso3, list(forecast_keys), as_of)
            return ResolverReading(text=_render(as_of, lines),
                                   record={"source": "facts_resolved:phase3plus", **info})
        if hazard in ("FL", "TC") and metric == "PA":
            lines, info = _pa_lines(con, iso3, hazard, as_of)
            return ResolverReading(text=_render(as_of, lines),
                                   record={"source": "facts_resolved:pa+gdacs", **info})
    except Exception as exc:  # noqa: BLE001 - the reading never stops a question
        logger.warning("sibyl.resolver_reading: %s failed: %s",
                       getattr(question, "question_id", "?"), exc)
        return ResolverReading(record={"source": "error", "reason": _scrub(exc)})
    return ResolverReading()
