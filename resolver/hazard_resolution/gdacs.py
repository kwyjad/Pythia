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
        # What the feed's gdacs:population element actually said, verbatim,
        # so a ceiling of 5 can be traced to "5 people" or "5 Million".
        "exposed_population_unit": str(event.get("population_unit") or ""),
        "exposed_population_text": str(event.get("population_text") or "")[:200],
        "exposed_population_raw": str(event.get("population_raw") or ""),
        "exposed_population_parse": str(event.get("population_parse") or ""),
        # Whether this run READ the figure or was served it from the cache.
        # A stale figure is worth more than none and only worth anything if
        # the row says it is stale.
        "exposed_population_source": str(event.get("population_source") or ""),
        "exposed_population_cached_at": str(event.get("population_cached_at") or ""),
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
        from resolver.hazard_resolution.geometry import distance_km, point_near_bounds

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
            # The bounding-box test is the cheap pre-filter distance_km's
            # projection exists to be spared by: ~250 polygons per event
            # otherwise each pay for a clip.
            if not point_near_bounds(country, lat_f, lon_f, _GEOMETRY_ATTRIBUTION_KM):
                continue
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


def _rb_int(rulebook: Rulebook | None, dotted: str, default: int) -> int:
    """A rulebook integer, or the default when an older rulebook lacks it."""

    if rulebook is None:
        return default
    try:
        return int(rulebook.get(dotted))
    except Exception:  # noqa: BLE001 - a missing key is not a failure here
        return default


def _rb_float(rulebook: Rulebook | None, dotted: str, default: float) -> float:
    if rulebook is None:
        return default
    try:
        return float(rulebook.get(dotted))
    except Exception:  # noqa: BLE001
        return default


#: How recently an event must have ENDED for its exposure to be worth
#: asking about again. GDACS revises a figure while an event is live and
#: leaves it alone once it is over, so an event that finished last year has
#: a settled exposure and re-asking for it buys nothing but a request. The
#: backcast walks 2000..2026, which is where nearly every request went.
_DEFAULT_EXPOSURE_REFRESH_DAYS = 21


def _exposure_refresh_days(rulebook: Rulebook | None) -> int:
    return max(
        0,
        _rb_int(
            rulebook,
            "flood.gdacs.exposure_refresh_days",
            _DEFAULT_EXPOSURE_REFRESH_DAYS,
        ),
    )


def seed_exposure_memo(
    con: "duckdb.DuckDBPyConnection",
    *,
    refresh_days: int,
    today: dt.date | None = None,
) -> dict[str, int]:
    """Tell the connector what the database already knows, before it asks.

    The cache was already read, and read too late: ``_carry_forward_exposure``
    ran AFTER every request had been made, so it repaired a refusal and
    never prevented one. Run 34124705852 asked GDACS for 291 events and was
    refused on 138 of them while the cache held figures for events it was
    about to ask for again.

    Seeding the connector's own per-run memo is deliberately the same
    mechanism, not a second one: a hit returns through ``_apply_exposure``
    exactly as a fetched answer does, so a cache-served figure and a live
    one cannot diverge in how they reach the record. The figure is labelled
    ``cache`` with the time it was read, because a stale figure is worth
    more than none and only worth anything if it says it is stale.

    An event that ended within ``refresh_days`` is NOT seeded: while an
    event is live its exposure is still being revised, and serving last
    week's figure as this week's would be the fallback-that-serves-old-data
    failure in another costume.

    Never raises: a cache that cannot be read seeds nothing and the run
    behaves exactly as it did before.
    """

    counts = {"seeded": 0, "still_live": 0, "no_usable_figure": 0}
    today = today or dt.date.today()
    cutoff = today - dt.timedelta(days=max(0, int(refresh_days)))
    core = _connector_api()
    try:
        rows = load_raw_records(con, SOURCE)
    except Exception as exc:  # noqa: BLE001 - a cache we cannot read decides nothing
        LOG.warning("[gdacs] could not read the exposure cache to seed it: %s", exc)
        return counts

    for row in rows:
        try:
            value = float(row.get("exposed_population") or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        if value <= 0:
            counts["no_usable_figure"] += 1
            continue
        ended = parse_date(row.get("end_date"))
        if ended is None or ended > cutoff:
            # Still live, or undated and therefore not provably settled.
            counts["still_live"] += 1
            continue
        record_id = str(row.get("_record_id") or "")
        etype, _, event_id = record_id.partition("-")
        if not etype or not event_id:
            continue
        episode = {
            "population": value,
            "population_unit": row.get("exposed_population_unit") or "",
            "population_text": row.get("exposed_population_text") or "",
            "population_raw": row.get("exposed_population_raw") or "",
            "population_parse": row.get("exposed_population_parse") or "",
            "population_source": "cache",
            "population_cached_at": str(row.get("_retrieved_at") or ""),
        }
        with core._EXPOSURE_MEMO_LOCK:
            core._EXPOSURE_MEMO[(etype, event_id)] = (episode, None)
        counts["seeded"] += 1
    return counts


def _carry_forward_exposure(
    con: "duckdb.DuckDBPyConnection", records: list[RawRecord]
) -> int:
    """Fill a record's missing exposure from the newest stored revision.

    Returns how many records were filled. A record whose exposure this run
    could not read (the per-event RSS was refused, 404ed, or carried no
    figure) keeps whatever the cache already knows, labelled
    ``exposed_population_source = "cache"`` with the retrieval timestamp of
    the revision it came from — a STALE figure is worth more than none, and
    only worth anything if it says it is stale.

    Never raises: an unreadable cache leaves the records exactly as they are.
    """

    missing = [r for r in records if not float(r.payload.get("exposed_population") or 0.0) > 0]
    if not missing:
        return 0
    try:
        cached = {
            str(row.get("_record_id")): row
            for row in load_raw_records(con, SOURCE)
        }
    except Exception as exc:  # noqa: BLE001 - a cache we cannot read decides nothing
        LOG.warning("[gdacs] could not read the exposure cache: %s", exc)
        return 0

    filled = 0
    for record in missing:
        previous = cached.get(record.record_id)
        if previous is None:
            continue
        try:
            value = float(previous.get("exposed_population") or 0.0)
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        record.payload["exposed_population"] = value
        for key in (
            "exposed_population_unit", "exposed_population_text",
            "exposed_population_raw", "exposed_population_parse",
        ):
            if previous.get(key):
                record.payload[key] = previous[key]
        record.payload["exposed_population_source"] = "cache"
        record.payload["exposed_population_cached_at"] = str(
            previous.get("_retrieved_at") or ""
        )
        filled += 1
    return filled


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
        core.reset_search_property_keys()
        connector = core.GdacsConnector()
        name_to_iso3, _ = core._load_countries()

        events = connector._search_events(
            session, start, end, delay, event_types=[hazard]
        )
        # Discovery carries no exposure figure; the per-event RSS does. But
        # ask only for the ones we do not already have. An event that ended
        # more than `exposure_refresh_days` ago has a settled figure, so the
        # cache answers for it and no request is spent.
        seeded = seed_exposure_memo(
            con, refresh_days=_exposure_refresh_days(rulebook)
        )
        events = connector._enrich_with_population(
            session, events, delay, name_to_iso3,
            # The rulebook owns the pacing it claims to own.
            workers=_rb_int(rulebook, "flood.gdacs.enrich_workers", 1),
            min_interval=_rb_float(
                rulebook, "flood.gdacs.enrich_min_interval_sec", 2.0
            ),
            max_seconds=_rb_float(rulebook, "flood.gdacs.enrich_max_seconds", 900.0),
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
    # A run GDACS refused to enrich must not overwrite a stored exposure
    # with a zero. The raw cache keeps every revision and readers take the
    # newest, so a throttled run appending a figure-less revision of an
    # event we already have a figure for is data loss dressed as a refresh.
    carried = _carry_forward_exposure(con, records)

    stored = store_raw_records(con, SOURCE, records)
    outcome.ok = True
    outcome.records = stored["records"]
    outcome.inserted = stored["inserted"]
    refused = sum(1 for e in events if e.get("population_refused"))
    from_cache = sum(
        1 for e in events if str(e.get("population_source") or "") == "cache"
    )
    from_search = sum(
        1 for e in events if str(e.get("population_source") or "") == "search"
    )
    outcome.detail = {
        "window": {"from": start.isoformat(), "to": end.isoformat()},
        "events_discovered": len(events),
        "events_skipped_malformed": skipped_malformed,
        "events_enrichment_refused": refused,
        # Answered from the cache without a request. The number that says
        # whether the cache-first read is doing its job.
        "events_exposure_from_cache": from_cache,
        "exposure_cache_seeded": seeded,
        # Answered by the SEARCH response, with no per-event request at all.
        "events_exposure_from_search": from_search,
        # What that response actually carried, so "is there a bulk route"
        # is settled by evidence rather than by the assumption the code
        # was written on.
        "search_property_keys": core.observed_search_property_keys(),
        "events_exposure_carried_from_cache": carried,
        "hazard": hazard,
    }
    LOG.info(
        "[gdacs] %s %s: %d event(s) discovered, %d answered from the cache "
        "and %d by the search response with no per-event request (%d settled "
        "figures seeded, %d still live, %d with no usable figure), %d refused",
        hazard, ym, len(events), from_cache, from_search, seeded["seeded"],
        seeded["still_live"], seeded["no_usable_figure"], refused,
    )
    if from_search:
        LOG.info(
            "[gdacs] the search response stated an exposure for %d of %d "
            "events — the per-event route may be unnecessary for those",
            from_search, len(events),
        )
    else:
        LOG.info(
            "[gdacs] the search response stated no exposure; its property "
            "keys were %s",
            ",".join(core.observed_search_property_keys()) or "(none seen)",
        )
    if refused or carried:
        LOG.warning(
            "[gdacs] %s %s: %d event(s) refused enrichment; %d kept an "
            "exposure figure from the cache rather than losing it",
            hazard, ym, refused, carried,
        )
    LOG.info(
        "[gdacs] %s %s: %d events in %s..%s (%d stored, %d new)",
        hazard, ym, len(events), start, end, stored["records"], stored["inserted"],
    )
    return outcome


#: Per-process memo of the parsed GDACS cache: (connection id, hazard, row
#: count) -> events. The ladder asks for a cell's events twice per triggered
#: cell (candidates, then the ceiling basis), and every ask used to load and
#: JSON-parse the whole hazard's cache. The row count is part of the key so
#: any insert or retention delete invalidates it; the connection id keeps
#: one test's database from answering for the next.
_EVENTS_MEMO: dict[tuple[int, str, int], list[dict[str, Any]]] = {}


def _all_events(con: "duckdb.DuckDBPyConnection", hazard: str) -> list[dict[str, Any]]:
    try:
        count = int(
            con.execute(
                "SELECT COUNT(*) FROM haz_raw_gdacs WHERE hazard = ?", [hazard]
            ).fetchone()[0]
        )
    except Exception:  # noqa: BLE001 - a missing table is an empty cache
        return load_raw_records(con, SOURCE, hazard=hazard)
    key = (id(con), hazard, count)
    events = _EVENTS_MEMO.get(key)
    if events is None:
        events = load_raw_records(con, SOURCE, hazard=hazard)
        _EVENTS_MEMO.clear()  # one hazard at a time; never let the memo grow
        _EVENTS_MEMO[key] = events
    return events


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
    for event in _all_events(con, hazard):
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
