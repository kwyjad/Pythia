# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Layer-1 cyclone detection: IBTrACS track points → ``haz_triggers``.

For each country-month, a cyclone triggers when any stored track point
qualifies under :func:`resolver.hazard_resolution.rules.cyclone_track_qualifies`
(within ``cyclone.buffer_km`` of the country's territory at
``cyclone.min_wind_kt`` or stronger; distances are exact AEQD — see
``geometry.py``).  A row is written for BOTH triggered and non-triggered
country-months, carrying the parameters and per-storm evidence that
produced the verdict.

Zero-resolution safety: zeros may only be written for months IBTrACS
demonstrably covers.  ``coverage_ok`` is true when the newest track
point in the store is no earlier than ``month_end -
cyclone.ibtracs.coverage_grace_days`` — months inside ingestion gaps
stay unresolved instead of becoming false zeros (the same month-gate
principle compute_resolutions applies to its zero defaults).
"""

from __future__ import annotations

import calendar
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

from resolver.hazard_resolution import ibtracs as ibtracs_mod
from resolver.hazard_resolution.geometry import Country, distance_km, point_near_bounds
from resolver.hazard_resolution.rulebook import (
    GDACS_ALERT_LEVELS,
    HAZARD_CODE_BY_RULEBOOK_NAME,
    Rulebook,
)
from resolver.hazard_resolution.rules import cyclone_track_qualifies
from resolver.hazard_resolution.schema import ensure_haz_schema

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

TRIGGER_SOURCE_IBTRACS = "ibtracs"
TRIGGER_SOURCE_GDACS = "gdacs"
TRIGGER_SOURCE_RELIEFWEB = "reliefweb_sweep"
TRIGGER_SOURCE_NONE = "none"

_WIND_COLUMNS = {
    "usa_wind": "usa_wind_kt",
    "wmo_wind": "wmo_wind_kt",
}


def month_bounds(ym: str) -> tuple[date, date]:
    """(first day, last day) of a ``YYYY-MM`` month."""
    year, month = ym.split("-")
    y, m = int(year), int(month)
    return date(y, m, 1), date(y, m, calendar.monthrange(y, m)[1])


def ym_to_year_month(ym: str) -> tuple[int, int]:
    """Split a ``YYYY-MM`` key into the (year, month) ints haz_* tables use."""
    year, month = ym.split("-")
    return int(year), int(month)


@dataclass
class CountryTrigger:
    iso3: str
    triggered: bool
    trigger_source: str
    detail: dict = field(default_factory=dict)
    sweep_evidence: dict | None = None


@dataclass
class DetectionResult:
    hazard: str
    ym: str
    coverage_ok: bool
    coverage_note: str
    n_points_month: int
    n_points_qualifying: int
    max_point_time: str | None
    rows: list[CountryTrigger] = field(default_factory=list)


def _resolve_wind(rulebook: Rulebook, record: dict) -> float | None:
    """First non-null wind per ``cyclone.wind_source_priority``."""
    for name in rulebook.get("cyclone.wind_source_priority"):
        col = _WIND_COLUMNS.get(str(name))
        if col is None:
            raise ValueError(
                f"unknown wind source {name!r} in cyclone.wind_source_priority "
                f"(known: {sorted(_WIND_COLUMNS)})"
            )
        value = record.get(col)
        if value is not None and value == value:  # not None, not NaN
            return float(value)
    return None


def _coverage(con, ym: str, rulebook: Rulebook) -> tuple[bool, str, str | None]:
    """Does the IBTrACS store demonstrably cover this month?"""
    grace_days = int(rulebook.get("cyclone.ibtracs.coverage_grace_days"))
    _, month_end = month_bounds(ym)
    summary = ibtracs_mod.store_summary(con)
    max_point = summary["max_point_time"]
    if not max_point:
        return False, "haz_raw_ibtracs is empty", None
    max_dt = datetime.fromisoformat(max_point)
    required = datetime.combine(
        month_end - timedelta(days=grace_days), datetime.min.time()
    )
    if max_dt < required:
        return (
            False,
            (
                f"newest stored track point {max_dt.isoformat()} predates "
                f"month_end - {grace_days}d grace ({required.date().isoformat()}) — "
                "zeros suppressed for this month"
            ),
            max_point,
        )
    return True, "ok", max_point


def detect_cyclone_month(
    con: "duckdb.DuckDBPyConnection",
    ym: str,
    rulebook: Rulebook,
    countries: dict[str, Country],
    iso3_filter: list[str] | None = None,
) -> DetectionResult:
    """Run cyclone detection for one month across the country universe.

    ``countries`` is the geometry universe (see
    ``geometry.load_country_geometries``); ``iso3_filter`` optionally
    restricts it.  Returns one :class:`CountryTrigger` per country —
    triggered and non-triggered alike.
    """
    ensure_haz_schema(con)
    hazard = HAZARD_CODE_BY_RULEBOOK_NAME["cyclone"]
    buffer_km = float(rulebook.get("cyclone.buffer_km"))
    min_wind = float(rulebook.get("cyclone.min_wind_kt"))
    # Distances are evaluated out to 2x the buffer so non-triggered rows
    # can carry nearest-approach evidence; any factor >= 1 produces
    # identical trigger decisions (the decision itself is
    # rules.cyclone_track_qualifies on the exact distance).
    report_km = buffer_km * 2.0

    month_start, month_end = month_bounds(ym)
    points = ibtracs_mod.load_track_points(
        con, month_start.isoformat(), month_end.isoformat()
    )
    # A storm can span a month boundary; only its points inside the
    # target month count for this month's verdict.
    if not points.empty:
        in_month = points["iso_time"].str.startswith(ym)
        points = points[in_month]
    n_month = len(points)

    point_records: list[dict] = []
    for rec in points.to_dict("records"):
        wind = _resolve_wind(rulebook, rec)
        if wind is None or wind < min_wind:
            continue
        rec["wind_kt"] = wind
        point_records.append(rec)

    coverage_ok, coverage_note, max_point_time = _coverage(con, ym, rulebook)

    universe = sorted(iso3_filter) if iso3_filter else sorted(countries)
    missing_geom = [c for c in universe if c not in countries]
    if missing_geom:
        LOG.warning(
            "[detect] %d countries have no boundary geometry and are skipped: %s",
            len(missing_geom),
            ",".join(missing_geom),
        )
    universe = [c for c in universe if c in countries]

    result = DetectionResult(
        hazard=hazard,
        ym=ym,
        coverage_ok=coverage_ok,
        coverage_note=coverage_note,
        n_points_month=n_month,
        n_points_qualifying=len(point_records),
        max_point_time=max_point_time,
    )

    for iso3 in universe:
        country = countries[iso3]
        storms: dict[str, dict] = {}
        for rec in point_records:
            lat, lon = float(rec["lat"]), float(rec["lon"])
            if not point_near_bounds(country, lat, lon, report_km):
                continue
            d_km = distance_km(country, lat, lon, report_km)
            if d_km is None:
                continue
            sid = str(rec["sid"])
            point_evidence = {
                "iso_time": str(rec["iso_time"]),
                "lat": lat,
                "lon": lon,
                "wind_kt": float(rec["wind_kt"]),
            }
            entry = storms.setdefault(
                sid,
                {
                    "sid": sid,
                    "name": str(rec.get("storm_name") or ""),
                    "basin": str(rec.get("basin") or ""),
                    "min_distance_km": d_km,
                    "max_wind_kt": float(rec["wind_kt"]),
                    "qualifies": False,
                    "closest_point": point_evidence,
                },
            )
            if d_km < entry["min_distance_km"]:
                entry["min_distance_km"] = d_km
                entry["closest_point"] = point_evidence
            entry["max_wind_kt"] = max(entry["max_wind_kt"], float(rec["wind_kt"]))
            # The trigger decision itself goes through the shared
            # rulebook predicate, per point.
            if cyclone_track_qualifies(d_km, float(rec["wind_kt"]), rulebook):
                entry["qualifies"] = True

        in_buffer = {sid: s for sid, s in storms.items() if s["qualifies"]}
        triggered = bool(in_buffer)
        detail = {
            "params": {
                "min_wind_kt": min_wind,
                "buffer_km": buffer_km,
                "wind_source_priority": list(
                    rulebook.get("cyclone.wind_source_priority")
                ),
            },
            "ibtracs": {
                "n_points_month_global": n_month,
                "n_points_qualifying_global": len(point_records),
                "max_point_time": max_point_time,
                "coverage_ok": coverage_ok,
                "coverage_note": coverage_note,
            },
            "storms_in_buffer": sorted(
                (dict(s) for s in in_buffer.values()),
                key=lambda s: s["min_distance_km"],
            ),
            "near_misses": sorted(
                (dict(s) for sid, s in storms.items() if sid not in in_buffer),
                key=lambda s: s["min_distance_km"],
            ),
        }
        result.rows.append(
            CountryTrigger(
                iso3=iso3,
                triggered=triggered,
                trigger_source=TRIGGER_SOURCE_IBTRACS if triggered else TRIGGER_SOURCE_NONE,
                detail=detail,
            )
        )

    n_triggered = sum(1 for r in result.rows if r.triggered)
    LOG.info(
        "[detect] %s %s: %d/%d countries triggered "
        "(%d qualifying points of %d in month; coverage_ok=%s)",
        hazard,
        ym,
        n_triggered,
        len(result.rows),
        len(point_records),
        n_month,
        coverage_ok,
    )
    return result


def write_triggers(
    con: "duckdb.DuckDBPyConnection", result: DetectionResult, rulebook: Rulebook
) -> int:
    """Upsert the month's trigger rows (idempotent per iso3/year/month/hazard).

    Same delete-then-insert transaction idiom as
    ``population.load_population_into_db``.
    """
    ensure_haz_schema(con)
    year, month = ym_to_year_month(result.ym)
    iso3s = [row.iso3 for row in result.rows]
    if not iso3s:
        return 0
    placeholders = ", ".join("?" for _ in iso3s)
    try:
        con.execute("BEGIN TRANSACTION")
        con.execute(
            f"""
            DELETE FROM haz_triggers
            WHERE hazard = ? AND year = ? AND month = ? AND iso3 IN ({placeholders})
            """,
            [result.hazard, year, month, *iso3s],
        )
        for row in result.rows:
            con.execute(
                """
                INSERT INTO haz_triggers
                    (iso3, year, month, hazard, triggered, trigger_source,
                     trigger_detail_json, evidence_of_absence_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                [
                    row.iso3,
                    year,
                    month,
                    result.hazard,
                    row.triggered,
                    row.trigger_source,
                    json.dumps(row.detail),
                ],
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    LOG.info(
        "[detect] wrote %d haz_triggers rows for %s %s",
        len(result.rows),
        result.hazard,
        result.ym,
    )
    return len(result.rows)


def detect_flood_month(
    con: "duckdb.DuckDBPyConnection",
    ym: str,
    rulebook: Rulebook,
    iso3_filter: list[str] | None = None,
) -> DetectionResult:
    """Run flood detection for one month from the cached GDACS events.

    A country-month triggers when any GDACS flood event naming that
    country, overlapping that month, carries an alert level at or above
    ``flood.gdacs_trigger_level`` (green < orange < red) — the decision
    itself goes through :func:`rules.flood_alert_qualifies`, so changing
    the rulebook colour changes behaviour with no code change.

    Unlike cyclone detection there is no geometry step: GDACS already
    attributes each event to countries. What is shared is the shape — a
    row for EVERY country in the universe, triggered or not, and the same
    coverage gate in front of any zero.
    """

    from resolver.hazard_resolution import gdacs as gdacs_mod
    from resolver.hazard_resolution.rules import flood_alert_qualifies

    ensure_haz_schema(con)
    hazard = HAZARD_CODE_BY_RULEBOOK_NAME["flood"]
    threshold = str(rulebook.get("flood.gdacs_trigger_level"))

    coverage_ok, coverage_note = gdacs_mod.coverage(con, ym, hazard, rulebook)

    # Index the cached events by country once, rather than re-scanning the
    # store per country (the universe is ~200 countries).
    events = load_raw_events_for_month(con, hazard, ym)
    by_country: dict[str, list[dict]] = {}
    for event in events:
        for iso3 in event.get("iso3_list") or []:
            by_country.setdefault(str(iso3).upper(), []).append(event)

    universe = sorted(iso3_filter) if iso3_filter else sorted(by_country)
    result = DetectionResult(
        hazard=hazard,
        ym=ym,
        coverage_ok=coverage_ok,
        coverage_note=coverage_note,
        n_points_month=len(events),
        n_points_qualifying=0,
        max_point_time=max((e.get("end_date") or "") for e in events) if events else None,
    )

    qualifying_total = 0
    for iso3 in universe:
        country_events = by_country.get(iso3, [])
        qualifying: list[dict] = []
        below: list[dict] = []
        for event in country_events:
            summary = {
                "event_id": event.get("event_id"),
                "alert_level": event.get("alert_level"),
                "exposed_population": event.get("exposed_population"),
                "start_date": event.get("start_date"),
                "end_date": event.get("end_date"),
                "source_url": event.get("_source_url"),
            }
            try:
                qualifies = flood_alert_qualifies(event.get("alert_level"), rulebook)
            except ValueError as exc:
                # An unrecognised colour is a data problem, not a silent
                # non-trigger: record it and treat the event as below
                # threshold rather than inventing a verdict.
                LOG.warning("[detect] %s %s: %s", iso3, ym, exc)
                summary["alert_level_error"] = str(exc)
                below.append(summary)
                continue
            (qualifying if qualifies else below).append(summary)

        qualifying_total += len(qualifying)
        result.rows.append(
            CountryTrigger(
                iso3=iso3,
                triggered=bool(qualifying),
                trigger_source=TRIGGER_SOURCE_GDACS if qualifying else TRIGGER_SOURCE_NONE,
                detail={
                    "params": {
                        "gdacs_trigger_level": threshold,
                        "alert_order": list(GDACS_ALERT_LEVELS),
                    },
                    "gdacs": {
                        "n_events_month_global": len(events),
                        "coverage_ok": coverage_ok,
                        "coverage_note": coverage_note,
                    },
                    "events_qualifying": qualifying,
                    "events_below_threshold": below,
                },
            )
        )

    result.n_points_qualifying = qualifying_total
    n_triggered = sum(1 for r in result.rows if r.triggered)
    LOG.info(
        "[detect] %s %s: %d/%d countries triggered "
        "(%d qualifying events of %d in month; coverage_ok=%s)",
        hazard, ym, n_triggered, len(result.rows), qualifying_total, len(events), coverage_ok,
    )
    return result


def load_raw_events_for_month(
    con: "duckdb.DuckDBPyConnection", hazard: str, ym: str
) -> list[dict]:
    """Cached GDACS events whose span overlaps ``ym`` (newest revision each)."""

    from resolver.hazard_resolution.sources import load_raw_records

    return [
        event
        for event in load_raw_records(con, "gdacs", hazard=hazard)
        if ym in (event.get("months_overlapped") or [])
    ]


def flip_trigger_from_sweep(
    con: "duckdb.DuckDBPyConnection",
    *,
    hazard: str,
    iso3: str,
    ym: str,
    sweep_evidence: dict,
) -> None:
    """Promote a non-triggered country-month after a non-silent sweep.

    The ReliefWeb sweep found cyclone reporting despite no qualifying
    track (remnant systems, naming edge cases): set triggered=true with
    trigger_source='reliefweb_sweep' and leave the month for the impact
    ladder (Phase 2+).  The sweep record joins the trigger detail.
    """
    year, month = ym_to_year_month(ym)
    row = con.execute(
        """
        SELECT trigger_detail_json FROM haz_triggers
        WHERE hazard = ? AND iso3 = ? AND year = ? AND month = ?
        """,
        [hazard, iso3, year, month],
    ).fetchone()
    detail = json.loads(row[0]) if row and row[0] else {}
    detail["reliefweb_sweep"] = sweep_evidence
    con.execute(
        """
        UPDATE haz_triggers
        SET triggered = TRUE,
            trigger_source = ?,
            trigger_detail_json = ?,
            evidence_of_absence_json = NULL
        WHERE hazard = ? AND iso3 = ? AND year = ? AND month = ?
        """,
        [TRIGGER_SOURCE_RELIEFWEB, json.dumps(detail), hazard, iso3, year, month],
    )
    if row is None:
        # No detection row (unexpected) — record the sweep verdict alone.
        con.execute(
            """
            INSERT INTO haz_triggers
                (iso3, year, month, hazard, triggered, trigger_source,
                 trigger_detail_json, evidence_of_absence_json)
            VALUES (?, ?, ?, ?, TRUE, ?, ?, NULL)
            """,
            [iso3, year, month, hazard, TRIGGER_SOURCE_RELIEFWEB, json.dumps(detail)],
        )


def record_sweep_on_trigger(
    con: "duckdb.DuckDBPyConnection",
    *,
    hazard: str,
    iso3: str,
    ym: str,
    sweep_evidence: dict,
) -> None:
    """Attach sweep evidence to a (still non-triggered) row.

    A silent sweep is evidence of absence, so it lands in the trigger
    row's ``evidence_of_absence_json`` (the column exists for exactly
    this — see schema.py).
    """
    year, month = ym_to_year_month(ym)
    con.execute(
        """
        UPDATE haz_triggers SET evidence_of_absence_json = ?
        WHERE hazard = ? AND iso3 = ? AND year = ? AND month = ?
        """,
        [json.dumps(sweep_evidence), hazard, iso3, year, month],
    )
