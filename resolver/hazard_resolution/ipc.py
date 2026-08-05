# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""IPC/CH analyses into ``haz_raw_ipc`` — the drought path's figure source.

Drought skips the impact ladder, so this is the only place a drought
figure can come from. What it stores is not a monthly series but a list of
ANALYSES: each one a Phase 3+ population, the validity window the analysis
declares it for, and when it was published. The drought rule is a
difference between two consecutive analyses, so the analysis — not the
month — is the unit that has to survive into the cache.

**Two acquisition paths, deliberately.**

``ipc_api``
    The IPC public API (``drought.ipc.api_url``), authenticated with the
    ``key=`` query parameter the repo's existing IPC connector already
    uses. Richest: it states the current-period window explicitly.
``facts_resolved``
    The ``phase3plus_in_need`` rows the repo's FEWS NET and IPC connectors
    already write. Needs no key, and — importantly — covers the FEWS NET
    countries that ``resolver/connectors/ipc_api.py`` deliberately EXCLUDES
    to avoid double-writing facts_resolved. Those are exactly the countries
    where drought resolution matters most, so excluding them here would
    have made the drought path useless where it is most needed.

Both paths land in the same cache; :func:`analyses_for_country`
deduplicates by validity window, preferring whichever source
``drought.ipc.source_priority`` lists first.

Period-date parsing is REUSED from ``resolver/connectors/ipc_api.py``
rather than re-implemented — the IPC API states its windows as free text
("Jun 2025 - Sep 2025"), and two parsers for one format would eventually
disagree about which months an analysis covers. A guard test asserts the
borrowed helper still exists, so an upstream rename fails in CI rather
than at runtime.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import requests

from resolver.hazard_resolution.rulebook import Rulebook
from resolver.hazard_resolution.rules import event_months
from resolver.hazard_resolution.sources import (
    FetchOutcome,
    RawRecord,
    fetch_window,
    load_raw_records,
    month_bounds,
    parse_date,
    parse_number,
    store_raw_records,
    utcnow_iso,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

SOURCE = "ipc"
HAZARD = "DR"
API_KEY_ENV = "IPC_API_KEY"

#: Acquisition path labels (also the vocabulary of
#: ``drought.ipc.source_priority``).
PATH_API = "ipc_api"
PATH_FACTS = "facts_resolved"

#: Injectable transport seam for tests: (url, params, timeout) -> parsed JSON.
GetFn = Callable[[str, dict, float], Any]


@dataclass
class IpcAnalysis:
    """One IPC/CH current-period analysis for one country.

    ``months`` is every ``YYYY-MM`` the validity window covers. The drought
    rule needs it twice: to find the analysis covering a target month, and
    to decide which months a delta speaks for.
    """

    iso3: str
    analysis_id: str
    window_start: str
    window_end: str
    months: list[str]
    value: float
    path: str
    analysis_date: str | None = None
    source_url: str | None = None
    retrieved_at: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def provenance(self) -> dict[str, Any]:
        """This analysis's audit trail, as embedded in a resolution.

        The spec requires a drought resolution to record WHICH analysis
        window it rests on; this is that record.
        """

        return {
            "analysis_id": self.analysis_id,
            "path": self.path,
            "window": {"start": self.window_start, "end": self.window_end},
            "months_covered": list(self.months),
            "phase3plus_population": self.value,
            "analysis_date": self.analysis_date,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at,
        }


def _ipc_connector_api():
    """The period-date parser owned by ``resolver/connectors/ipc_api.py``.

    Borrowed rather than copied (the same reuse contract the GDACS
    connector has). Raises a clear AttributeError if the helper is renamed,
    so CI fails loudly instead of the drought path silently losing every
    analysis window.
    """

    from resolver.connectors import ipc_api as connector

    parse_period_dates = getattr(connector, "_parse_period_dates", None)
    if parse_period_dates is None:  # pragma: no cover - guarded by a test
        raise AttributeError(
            "resolver.connectors.ipc_api._parse_period_dates has been renamed or "
            "removed; resolver/hazard_resolution/ipc.py reuses it to parse IPC "
            "validity windows and must be updated in the same change"
        )
    return parse_period_dates


def _default_get(url: str, params: dict, timeout: float) -> Any:
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _api_key() -> str | None:
    key = os.getenv(API_KEY_ENV, "").strip()
    return key or None


def _analysis_record(
    *,
    iso3: str,
    path: str,
    window_start: dt.date,
    window_end: dt.date,
    value: float,
    analysis_date: str | None,
    source_url: str | None,
    raw: dict[str, Any] | None = None,
) -> RawRecord:
    """One analysis on its way into ``haz_raw_ipc``.

    ``record_id`` keys on (path, country, window start): a re-fetch of the
    same analysis collides on content_hash and is a no-op, while a REVISED
    Phase 3+ figure for the same window appends a row that readers prefer.
    """

    months = event_months(window_start, window_end)
    payload = {
        "iso3": iso3,
        "path": path,
        "period": "current",
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "months_covered": months,
        "phase3plus_population": float(value),
        "analysis_date": analysis_date or "",
        "fetched_at": utcnow_iso(),
        "raw": raw or {},
    }
    return RawRecord(
        record_id=f"ipc-{path}-{iso3}-{window_start.isoformat()}",
        payload=payload,
        iso3=iso3,
        ym=window_start.strftime("%Y-%m"),
        hazard=HAZARD,
        source_url=source_url,
    )


def _iso3_from_api_record(record: dict[str, Any]) -> str | None:
    """ISO3 for an IPC API record.

    The API reports ISO2 in ``country``, so the primary path is pycountry —
    the same conversion ``resolver/connectors/ipc_api.py`` makes. The name
    fallback is not decoration: without it, an environment missing pycountry
    (or a payload that only carries a country name) resolves nothing and the
    whole drought path silently reports "IPC has no analyses" rather than
    failing.
    """

    code = record.get("country") or record.get("country_code") or record.get("iso2") or ""
    if isinstance(code, dict):
        code = code.get("iso2") or code.get("code") or ""
    code = str(code).strip().upper()
    if len(code) == 3 and code.isalpha():
        return code

    if len(code) == 2:
        try:
            import pycountry
        except ImportError:
            LOG.debug("[ipc] pycountry unavailable — trying the country name instead")
        else:
            country = pycountry.countries.get(alpha_2=code)
            if country:
                return country.alpha_3

    try:
        from resolver.ingestion.utils.iso_normalize import resolve_iso3
    except Exception:  # pragma: no cover - ingestion utils always import
        return None
    iso3, _reason = resolve_iso3(
        dict(record), name_keys=("country_name", "name", "title", "country")
    )
    return iso3


def _records_from_api(rows: list[Any], window: tuple[dt.date, dt.date]) -> list[RawRecord]:
    """Current-period analyses out of an IPC API payload.

    Only the CURRENT period is stored. The projection is a forecast, and a
    forecast is not a resolution source — the same reason
    ``compute_resolutions`` resolves against ``phase3plus_in_need`` and
    never against ``phase3plus_projection``.
    """

    parse_period_dates = _ipc_connector_api()
    start_bound, end_bound = window
    records: list[RawRecord] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        iso3 = _iso3_from_api_record(row)
        if not iso3:
            continue
        value = parse_number(row.get("p3plus"))
        if value is None:
            continue
        raw_dates = str(row.get("current_period_dates") or "").strip()
        start_text, end_text = parse_period_dates(raw_dates)
        start = parse_date(start_text)
        end = parse_date(end_text) or start
        if start is None or end is None or end < start:
            continue
        if end < start_bound or start > end_bound:
            continue
        records.append(
            _analysis_record(
                iso3=iso3,
                path=PATH_API,
                window_start=start,
                window_end=end,
                value=value,
                analysis_date=str(row.get("analysis_date") or "").strip() or None,
                source_url="https://www.ipcinfo.org",
                raw=row,
            )
        )
    return records


def _records_from_facts_resolved(
    con: "duckdb.DuckDBPyConnection", window: tuple[dt.date, dt.date], rulebook: Rulebook
) -> tuple[list[RawRecord], str | None]:
    """Analyses reconstructed from the repo's existing Phase 3+ rows.

    ``facts_resolved`` stores one row per analysis with ``ym`` = the
    window's first month and ``as_of_date`` = the window's end, which is
    exactly the pair this rule needs. Returns (records, error) — a missing
    table on a fresh DB is reported, never raised.
    """

    metric = str(rulebook.get("drought.ipc.facts_resolved_metric")).lower()
    publishers = [
        str(p).strip().lower()
        for p in rulebook.get("drought.ipc.facts_resolved_publishers")
    ]
    start_bound, end_bound = window
    placeholders = ", ".join("?" for _ in publishers)
    try:
        rows = con.execute(
            f"""
            SELECT ym, iso3, value, as_of_date, publication_date, publisher, source_url
            FROM facts_resolved
            WHERE hazard_code = ?
              AND lower(metric) = ?
              AND lower(publisher) IN ({placeholders})
              AND ym >= ? AND ym <= ?
            ORDER BY iso3, ym
            """,
            [
                HAZARD,
                metric,
                *publishers,
                start_bound.strftime("%Y-%m"),
                end_bound.strftime("%Y-%m"),
            ],
        ).fetchall()
    except Exception as exc:
        LOG.warning(
            "[ipc] facts_resolved is unavailable (%s) — the drought path falls "
            "back to the IPC API alone",
            exc,
        )
        return [], str(exc)

    records: list[RawRecord] = []
    for ym, iso3, value, as_of_date, publication_date, publisher, source_url in rows:
        number = parse_number(value)
        if number is None or not ym or not iso3:
            continue
        start, month_last = month_bounds(str(ym))
        end = parse_date(as_of_date) or month_last
        if end < start:
            end = month_last
        records.append(
            _analysis_record(
                iso3=str(iso3).upper(),
                path=PATH_FACTS,
                window_start=start,
                window_end=end,
                value=number,
                analysis_date=str(publication_date or "") or None,
                source_url=str(source_url or "") or None,
                raw={"publisher": publisher, "metric": metric, "ym": ym},
            )
        )
    return records, None


def fetch_ipc(
    con: "duckdb.DuckDBPyConnection",
    ym: str,
    rulebook: Rulebook,
    *,
    get: GetFn | None = None,
) -> FetchOutcome:
    """Refresh IPC analyses around ``ym`` into ``haz_raw_ipc``. Never raises.

    ``ok`` is true when at least one acquisition path answered. Both paths
    failing means "we could not ask", which the drought rule must never
    read as "IPC has nothing" — the difference decides whether a NO_DATA is
    honest.
    """

    url = str(rulebook.get("drought.ipc.api_url"))
    timeout = float(rulebook.get("drought.ipc.request_timeout_sec"))
    window = fetch_window(ym, rulebook, "drought.ipc")
    outcome = FetchOutcome(source=SOURCE, ok=False, source_urls=[url])
    detail: dict[str, Any] = {
        "window": {"from": window[0].isoformat(), "to": window[1].isoformat()},
        "paths": {},
    }

    records: list[RawRecord] = []
    paths_ok = 0

    key = _api_key()
    if key:
        getter = get or _default_get
        try:
            payload = getter(
                url,
                {"key": key, "start": window[0].year, "end": window[1].year},
                timeout,
            )
            rows = payload if isinstance(payload, list) else [payload]
            api_records = _records_from_api(rows, window)
            records.extend(api_records)
            paths_ok += 1
            detail["paths"][PATH_API] = {
                "ok": True,
                "rows_returned": len(rows),
                "analyses": len(api_records),
            }
        except Exception as exc:
            LOG.error("[ipc] IPC API fetch failed for %s: %s", ym, exc)
            detail["paths"][PATH_API] = {"ok": False, "error": str(exc)}
    else:
        LOG.warning(
            "[ipc] %s is not set — falling back to the Phase 3+ rows already in "
            "facts_resolved",
            API_KEY_ENV,
        )
        detail["paths"][PATH_API] = {"ok": False, "error": f"{API_KEY_ENV} not set"}

    if bool(rulebook.get("drought.ipc.use_facts_resolved")):
        facts_records, error = _records_from_facts_resolved(con, window, rulebook)
        if error is None:
            records.extend(facts_records)
            paths_ok += 1
            detail["paths"][PATH_FACTS] = {"ok": True, "analyses": len(facts_records)}
        else:
            detail["paths"][PATH_FACTS] = {"ok": False, "error": error}

    if paths_ok == 0:
        outcome.error = "no IPC acquisition path answered"
        outcome.detail = detail
        LOG.error("[ipc] %s: %s", ym, outcome.error)
        return outcome

    stored = store_raw_records(con, SOURCE, records)
    outcome.ok = True
    outcome.records = stored["records"]
    outcome.inserted = stored["inserted"]
    outcome.detail = detail
    LOG.info(
        "[ipc] %s: %d analyses stored (%d new) from %d path(s)",
        ym, stored["records"], stored["inserted"], paths_ok,
    )
    return outcome


def _to_analysis(payload: dict[str, Any]) -> IpcAnalysis | None:
    start = payload.get("window_start")
    end = payload.get("window_end")
    value = parse_number(payload.get("phase3plus_population"))
    iso3 = str(payload.get("iso3") or "").upper()
    if not start or not end or value is None or not iso3:
        return None
    return IpcAnalysis(
        iso3=iso3,
        analysis_id=str(payload.get("_record_id") or ""),
        window_start=str(start),
        window_end=str(end),
        months=[str(m) for m in payload.get("months_covered") or []],
        value=float(value),
        path=str(payload.get("path") or ""),
        analysis_date=str(payload.get("analysis_date") or "") or None,
        source_url=payload.get("_source_url"),
        retrieved_at=payload.get("_retrieved_at"),
        detail={"period": payload.get("period")},
    )


def analyses_for_country(
    con: "duckdb.DuckDBPyConnection", iso3: str, rulebook: Rulebook
) -> list[IpcAnalysis]:
    """Every cached analysis for a country, oldest window first.

    Analyses found by both acquisition paths describe one real analysis, so
    they are collapsed per validity window, keeping the path
    ``drought.ipc.source_priority`` lists first. Two paths disagreeing about
    one window would otherwise show up as a spurious delta of zero — or,
    worse, as a baseline the covering analysis is diffed against.
    """

    priority = [str(p) for p in rulebook.get("drought.ipc.source_priority")]

    def rank(analysis: IpcAnalysis) -> int:
        try:
            return priority.index(analysis.path)
        except ValueError:
            return len(priority)

    by_window: dict[str, IpcAnalysis] = {}
    for payload in load_raw_records(con, SOURCE, iso3=iso3.upper(), hazard=HAZARD):
        analysis = _to_analysis(payload)
        if analysis is None:
            continue
        incumbent = by_window.get(analysis.window_start)
        if incumbent is None or rank(analysis) < rank(incumbent):
            by_window[analysis.window_start] = analysis
    return sorted(by_window.values(), key=lambda a: (a.window_start, a.window_end))


def covering_analysis(analyses: list[IpcAnalysis], ym: str) -> IpcAnalysis | None:
    """The analysis whose validity window covers ``ym``, newest first.

    IPC windows overlap when a new analysis supersedes a running one, so
    "newest window start" is what decides, with the analysis date as the
    tiebreak.
    """

    covering = [a for a in analyses if ym in a.months]
    if not covering:
        return None
    return max(covering, key=lambda a: (a.window_start, a.analysis_date or ""))


def previous_analysis(
    analyses: list[IpcAnalysis], current: IpcAnalysis
) -> IpcAnalysis | None:
    """The current-period analysis immediately preceding ``current``.

    "Preceding" is by validity window, not publication date: the rule
    compares the situation this analysis describes with the situation the
    last one described, and IPC occasionally publishes those out of order.
    """

    earlier = [a for a in analyses if a.window_start < current.window_start]
    if not earlier:
        return None
    return max(earlier, key=lambda a: (a.window_start, a.analysis_date or ""))


def store_summary(con: "duckdb.DuckDBPyConnection") -> dict[str, Any]:
    """Provenance summary of the IPC cache (for evidence records)."""

    from resolver.hazard_resolution.sources import raw_store_summary

    return raw_store_summary(con, SOURCE, HAZARD)
