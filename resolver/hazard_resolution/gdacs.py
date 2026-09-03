# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""GDACS events into ``haz_raw_gdacs`` — flood/cyclone detection + ceiling.

**GDACS is never a resolution value.** It answers two questions and no
others: *did a qualifying event occur here?* (Layer 1 detection, floods)
and *how many people could plausibly have been affected?* (the sanity
ceiling). Its population figure is MODELLED EXPOSURE — hazard footprint
times gridded population — not reported impact, and the repo has already
been burned once by letting exposure into a people-affected series (see
the 2026-08-04 entry in CLAUDE.md's failure modes). It therefore enters
``haz_impact_candidates`` only as ``value_type='exposed_ceiling'``, which
the reconciler reads as a bound and never as an answer.

**Reuse, not duplication.** Event discovery and per-event enrichment come
from :mod:`resolver.connectors.gdacs`, the repo's existing GDACS client —
it already owns the search endpoint, the quarterly chunking, the RSS
namespaces, the alert-level parsing and the threaded per-event
enrichment. This module adapts that client's output into the machine's
raw-cache shape; it does not re-implement any of it. The endpoints
consequently live in that connector, NOT in ``rulebook.yaml``, so there
is exactly one place to change a GDACS URL — the rulebook comment says so
explicitly, and :func:`_connector_api` fails loudly if the borrowed
helpers ever move.

What the machine stores is the per-event record with its country list
intact. The core connector aggregates to country-months and splits
exposure by population weight for ``facts_resolved``; that aggregation is
lossy for our purposes, because reconciliation needs the event identity,
its date span and its alert level to attribute a figure to a month.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import TYPE_CHECKING, Any

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
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

SOURCE = "gdacs"

#: Helpers borrowed from the core connector. Named here so the reuse is
#: explicit and a rename upstream fails with a clear message (and a test)
#: rather than silently falling back to some other behaviour.
_BORROWED = (
    "GdacsConnector",
    "_build_session",
    "_load_countries",
    "_SEARCH_API",
    "_EVENT_RSS_PATTERN",
)


def _connector_api() -> Any:
    """The core GDACS connector module, checked for the helpers we borrow."""

    from resolver.connectors import gdacs as core

    missing = [name for name in _BORROWED if not hasattr(core, name)]
    if missing:
        raise AttributeError(
            "resolver.connectors.gdacs no longer provides "
            f"{missing} — the hazard-resolution GDACS adapter reuses these "
            "deliberately rather than duplicating the client; update "
            "resolver/hazard_resolution/gdacs.py to match the new API"
        )
    return core


def _event_record(
    event: dict[str, Any], hazard: str,
    geometries: "_CountryGeometries | None" = None,
) -> RawRecord | None:
    """One discovered GDACS event as a raw-cache record."""

    event_id = str(event.get("eventid") or "").strip()
    if not event_id:
        return None
    start = parse_date(event.get("fromdate"))
    end = parse_date(event.get("todate")) or start
    if start is None:
        return None

    # GDACS occasionally publishes todate BEFORE fromdate (seen live in
    # 2021: todate 2021-09-01 against fromdate 2021-09-28), and
    # event_months rejects a reversed range. Clamp to the start day — the
    # figure is attributed to the start month anyway — and keep the raw
    # todate in the payload so provenance records what GDACS actually said.
    end_date_raw = None
    if end is not None and end < start:
        LOG.warning(
            "[gdacs] event %s (%s): todate %s precedes fromdate %s — "
            "clamping end to start; raw todate kept as end_date_raw",
            event_id,
            event.get("eventtype"),
            end.isoformat(),
            start.isoformat(),
        )
        end_date_raw = end.isoformat()
        end = start

    iso3_list = [
        str(code).strip().upper()
        for code in (event.get("iso3_list") or [])
        if str(code).strip()
    ]
    primary = str(event.get("iso3") or "").strip().upper()
    if primary and primary not in iso3_list:
        iso3_list.insert(0, primary)

    # GDACS names no country for an event whose affectedcountries list is
    # empty — routine for a tropical cyclone still over open ocean — and a
    # record with no country is written and then never read by any cell. The
    # 2026-08 run dropped 16 TC events this way (ids 1001273..1001315).
    # The event's own position resolves it, against the same vendored
    # boundaries cyclone detection already uses.
    geo_resolved: list[str] = []
    if not iso3_list:
        geo_resolved = _iso3s_near_event(event, hazard, geometries)
        iso3_list = list(geo_resolved)
        primary = iso3_list[0] if iso3_list else primary

    payload = {
        "event_id": event_id,
        "event_type": event.get("eventtype"),
        "hazard": hazard,
        "iso3_list": iso3_list,
        "country": event.get("country") or "",
        "alert_level": str(event.get("alertlevel") or "Green"),
        "alert_score": event.get("alertscore"),
        # GDACS "population" is MODELLED EXPOSURE, never reported impact.
        # The key name says so, so no downstream reader can mistake it.
        "exposed_population": parse_number(event.get("population")) or 0.0,
        "start_date": start.isoformat(),
        "end_date": (end or start).isoformat(),
        # Always present; None unless the reversed-dates clamp above fired.
        "end_date_raw": end_date_raw,
        "months_overlapped": event_months(start, end or start),
        # Provenance: which countries came from GDACS itself and which the
        # machine derived from the event's position. A derived attribution
        # is a weaker claim than a stated one and the row must say so.
        "iso3_from_geometry": geo_resolved,
        "published_at": (
            parse_date(event.get("pub_date")).isoformat()
            if parse_date(event.get("pub_date"))
            else None
        ),
    }
    core = _connector_api()
    return RawRecord(
        record_id=f"{event.get('eventtype')}-{event_id}",
        payload=payload,
        iso3=primary or (iso3_list[0] if iso3_list else None),
        # ym anchors the event to its START month — the month its figure
        # would be attributed to (rules.attribution_month). Detection reads
        # months_overlapped inside the payload, not this column.
        ym=start.strftime("%Y-%m"),
        hazard=hazard,
        source_url=core._EVENT_RSS_PATTERN.format(
            type=event.get("eventtype"), eventid=event_id
        ),
    )


#: How far from a country a GDACS point may sit and still be attributed to
#: it. Deliberately generous: the point is an event CENTRE (a cyclone eye, a
#: flood's reference location), not its footprint, so a storm affecting a
#: coast sits well offshore. Narrower than the cyclone rulebook's own buffer
#: would drop the events this exists to recover.
_GEOMETRY_ATTRIBUTION_KM = 500.0


class _CountryGeometries:
    """A once-per-fetch loader for the vendored boundaries.

    Parsing the 1:50m layer costs a few hundred milliseconds, and the naive
    version paid it per EVENT — fine for the handful of uncountried events a
    live month sees, and not fine for a backcast month full of them. Loading
    is still LAZY: a month in which GDACS named every country never touches
    the file at all.

    Not a module-level cache, deliberately: the tests inject synthetic
    geometries by monkeypatching the loader, and a process-wide cache would
    serve the first test's boundaries to every later one.
    """

    def __init__(self) -> None:
        self._loaded = False
        self._countries: dict[str, Any] = {}

    def get(self) -> dict[str, Any]:
        if not self._loaded:
            from resolver.hazard_resolution.geometry import load_country_geometries

            self._countries = load_country_geometries()
            self._loaded = True
        return self._countries


def _iso3s_near_event(
    event: dict[str, Any], hazard: str, geometries: "_CountryGeometries | None" = None
) -> list[str]:
    """Countries within :data:`_GEOMETRY_ATTRIBUTION_KM` of the event's point.

    Never raises: an event that cannot be placed keeps no country, which is
    the state it was already in. Returned nearest-first, so the first entry
    is the primary attribution.
    """

    lat, lon = event.get("lat"), event.get("lon")
    if lat is None or lon is None:
        return []
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return []

    try:
        from resolver.hazard_resolution.geometry import distance_km

        countries = (geometries or _CountryGeometries()).get()
    except Exception as exc:  # noqa: BLE001 - boundaries absent is not fatal
        LOG.warning(
            "[gdacs] cannot resolve %s %s by geometry: %s",
            event.get("eventid"), hazard, exc,
        )
        return []

    hits: list[tuple[float, str]] = []
    for iso3, country in countries.items():
        try:
            distance = distance_km(country, lat_f, lon_f, _GEOMETRY_ATTRIBUTION_KM)
        except Exception:  # noqa: BLE001 - one bad polygon must not lose the event
            continue
        if distance is not None and distance <= _GEOMETRY_ATTRIBUTION_KM:
            hits.append((distance, str(iso3).upper()))

    hits.sort()
    resolved = [iso3 for _d, iso3 in hits]
    if resolved:
        LOG.info(
            "[gdacs] event %s (%s) named no country; resolved %s from its "
            "position (%.2f, %.2f)",
            event.get("eventid"), hazard, ",".join(resolved[:5]), lat_f, lon_f,
        )
    return resolved


def fetch_gdacs_events(
    con: "duckdb.DuckDBPyConnection",
    ym: str,
    hazard: str,
    rulebook: Rulebook,
    *,
    session: Any = None,
) -> FetchOutcome:
    """Fetch GDACS events around ``ym`` for ``hazard`` into ``haz_raw_gdacs``.

    A failure returns ``ok=False`` rather than raising: the caller must be
    able to tell "GDACS says nothing happened" from "GDACS did not answer",
    because only the first of those can justify a zero.
    """

    core = _connector_api()
    start, end = fetch_window(ym, rulebook, "flood.gdacs")
    delay = float(rulebook.get("flood.gdacs.request_delay_sec"))

    outcome = FetchOutcome(source=SOURCE, ok=False, source_urls=[core._SEARCH_API])
    try:
        session = session or core._build_session()
        connector = core.GdacsConnector()
        name_to_iso3, _ = core._load_countries()

        events = connector._search_events(
            session, start, end, delay, event_types=[hazard]
        )
        # Discovery carries no exposure figure; the per-event RSS does.
        events = connector._enrich_with_population(
            session, events, delay, name_to_iso3
        )
    except Exception as exc:
        LOG.error(
            "[gdacs] fetch failed for %s %s (%s .. %s): %s",
            hazard, ym, start, end, exc,
        )
        outcome.error = str(exc)
        return outcome

    # Per-event guard: the no-raise contract above applies per RECORD too.
    # One malformed upstream event (a reversed date range, a garbled field)
    # must never kill the month — GDACS answered; that event is skipped
    # with a logged warning and counted in the outcome detail.
    records = []
    skipped_malformed = 0
    # One loader for the whole month, loaded only if an event needs it.
    geometries = _CountryGeometries()
    for event in events:
        try:
            record = _event_record(event, hazard, geometries)
        except Exception as exc:  # noqa: BLE001 - one bad event must not kill the month
            skipped_malformed += 1
            LOG.warning(
                "[gdacs] skipping malformed event %r (%s %s): %s",
                event.get("eventid"),
                hazard,
                ym,
                exc,
            )
            continue
        if record is not None:
            records.append(record)
    stored = store_raw_records(con, SOURCE, records)
    outcome.ok = True
    outcome.records = stored["records"]
    outcome.inserted = stored["inserted"]
    outcome.detail = {
        "window": {"from": start.isoformat(), "to": end.isoformat()},
        "events_discovered": len(events),
        "events_skipped_malformed": skipped_malformed,
        "hazard": hazard,
    }
    LOG.info(
        "[gdacs] %s %s: %d events in %s..%s (%d stored, %d new)",
        hazard, ym, len(events), start, end, stored["records"], stored["inserted"],
    )
    return outcome


def events_for_country_month(
    con: "duckdb.DuckDBPyConnection", iso3: str, ym: str, hazard: str
) -> list[dict[str, Any]]:
    """Cached GDACS events overlapping ``ym`` that name ``iso3``.

    Detection scope, not attribution scope: an event is returned for every
    month its span overlaps (``event_attribution.detection = overlap``),
    so a flood spanning a month boundary is visible to both months.
    """

    iso3 = iso3.upper()
    out = []
    for event in load_raw_records(con, SOURCE, hazard=hazard):
        if iso3 not in (event.get("iso3_list") or []):
            continue
        if ym not in (event.get("months_overlapped") or []):
            continue
        out.append(event)
    return out


def coverage(
    con: "duckdb.DuckDBPyConnection", ym: str, hazard: str, rulebook: Rulebook
) -> tuple[bool, str]:
    """Does the GDACS store demonstrably cover this month?

    The same zero-safety gate cyclone detection applies to IBTrACS: unless
    the newest stored event reaches ``month_end - coverage_grace_days``, an
    ingestion gap is indistinguishable from a quiet month, and quiet months
    are the ones that become zeros. Fail closed.
    """

    grace = int(rulebook.get("flood.gdacs.coverage_grace_days"))
    _, month_end = month_bounds(ym)
    required = month_end - dt.timedelta(days=grace)

    row = con.execute(
        """
        SELECT MAX(json_extract_string(payload_json, '$.end_date'))
        FROM haz_raw_gdacs WHERE hazard = ?
        """,
        [hazard],
    ).fetchone()
    newest = parse_date(row[0]) if row and row[0] else None
    if newest is None:
        return False, "haz_raw_gdacs holds no events for this hazard"
    if newest < required:
        return False, (
            f"newest stored GDACS event {newest.isoformat()} predates "
            f"month_end - {grace}d ({required.isoformat()}) — zeros suppressed"
        )
    return True, "ok"
