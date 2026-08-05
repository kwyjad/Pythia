# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Layer-1 cyclone detection: IBTrACS track points → ``haz_triggers``.

For each country-month, a cyclone triggers when any stored track point
at >= ``cyclone.min_wind_kt`` lies within ``cyclone.buffer_km`` of the
country's territory (exact AEQD distance — see ``geometry.py``).  A row
is written for BOTH triggered and non-triggered country-months, carrying
the parameters and per-storm evidence that produced the verdict.

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

import pandas as pd

from resolver.resolution_machine.geometry import Country, distance_km, point_near_bounds
from resolver.resolution_machine.rulebook import Rulebook
from resolver.resolution_machine.schema import ensure_schema, utcnow_iso

LOG = logging.getLogger(__name__)

TRIGGER_SOURCE_IBTRACS = "ibtracs"
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


@dataclass
class CountryTrigger:
    iso3: str
    triggered: bool
    trigger_source: str
    detail: dict = field(default_factory=dict)


@dataclass
class DetectionResult:
    hazard_code: str
    ym: str
    coverage_ok: bool
    coverage_note: str
    n_points_month: int
    n_points_qualifying: int
    max_iso_time: str | None
    rows: list[CountryTrigger] = field(default_factory=list)


def _wind_expression(rulebook: Rulebook) -> str:
    """COALESCE expression implementing ``cyclone.wind_source_priority``."""
    priority = rulebook["cyclone.wind_source_priority"]
    cols = []
    for name in priority:
        col = _WIND_COLUMNS.get(str(name))
        if col is None:
            raise ValueError(
                f"unknown wind source '{name}' in cyclone.wind_source_priority "
                f"(known: {sorted(_WIND_COLUMNS)})"
            )
        cols.append(col)
    return f"COALESCE({', '.join(cols)})"


def _load_month_points(con, ym: str, rulebook: Rulebook) -> tuple[pd.DataFrame, int]:
    """Qualifying track points for the month + total point count (any wind)."""
    start, end = month_bounds(ym)
    end_excl = end + timedelta(days=1)
    wind_expr = _wind_expression(rulebook)
    min_wind = float(rulebook["cyclone.min_wind_kt"])

    n_month = int(
        con.execute(
            "SELECT COUNT(*) FROM haz_raw_ibtracs WHERE iso_time >= ? AND iso_time < ?",
            [start.isoformat(), end_excl.isoformat()],
        ).fetchone()[0]
    )
    points = con.execute(
        f"""
        SELECT sid, storm_name, basin, CAST(iso_time AS VARCHAR) AS iso_time,
               lat, lon, {wind_expr} AS wind_kt
        FROM haz_raw_ibtracs
        WHERE iso_time >= ? AND iso_time < ?
          AND {wind_expr} >= ?
        ORDER BY sid, iso_time
        """,
        [start.isoformat(), end_excl.isoformat(), min_wind],
    ).fetchdf()
    return points, n_month


def _coverage(con, ym: str, rulebook: Rulebook) -> tuple[bool, str, str | None]:
    """Does the IBTrACS store demonstrably cover this month?"""
    grace_days = int(rulebook["cyclone.ibtracs.coverage_grace_days"])
    _, month_end = month_bounds(ym)
    row = con.execute("SELECT MAX(iso_time) FROM haz_raw_ibtracs").fetchone()
    max_iso = row[0]
    if max_iso is None:
        return False, "haz_raw_ibtracs is empty", None
    if isinstance(max_iso, str):
        max_dt = datetime.fromisoformat(max_iso)
    else:
        max_dt = max_iso
    required = datetime.combine(month_end - timedelta(days=grace_days), datetime.min.time())
    if max_dt < required:
        return (
            False,
            (
                f"newest stored track point {max_dt.isoformat()} predates "
                f"month_end - {grace_days}d grace ({required.date().isoformat()}) — "
                "zeros suppressed for this month"
            ),
            max_dt.isoformat(),
        )
    return True, "ok", max_dt.isoformat()


def detect_cyclone_month(
    con,
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
    ensure_schema(con)
    hazard_code = str(rulebook["cyclone.hazard_code"])
    buffer_km = float(rulebook["cyclone.buffer_km"])
    min_wind = float(rulebook["cyclone.min_wind_kt"])
    # Distances are evaluated out to 2x the buffer so non-triggered rows
    # can carry nearest-approach evidence; any factor >= 1 produces
    # identical trigger decisions.
    report_km = buffer_km * 2.0

    points, n_month = _load_month_points(con, ym, rulebook)
    coverage_ok, coverage_note, max_iso_time = _coverage(con, ym, rulebook)

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
        hazard_code=hazard_code,
        ym=ym,
        coverage_ok=coverage_ok,
        coverage_note=coverage_note,
        n_points_month=n_month,
        n_points_qualifying=len(points),
        max_iso_time=max_iso_time,
    )

    point_records = points.to_dict("records") if not points.empty else []

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
            entry = storms.setdefault(
                sid,
                {
                    "sid": sid,
                    "name": str(rec.get("storm_name") or ""),
                    "basin": str(rec.get("basin") or ""),
                    "min_distance_km": d_km,
                    "max_wind_kt": float(rec["wind_kt"]),
                    "closest_point": {
                        "iso_time": str(rec["iso_time"]),
                        "lat": lat,
                        "lon": lon,
                        "wind_kt": float(rec["wind_kt"]),
                    },
                },
            )
            if d_km < entry["min_distance_km"]:
                entry["min_distance_km"] = d_km
                entry["closest_point"] = {
                    "iso_time": str(rec["iso_time"]),
                    "lat": lat,
                    "lon": lon,
                    "wind_kt": float(rec["wind_kt"]),
                }
            entry["max_wind_kt"] = max(entry["max_wind_kt"], float(rec["wind_kt"]))

        in_buffer = {
            sid: s for sid, s in storms.items() if s["min_distance_km"] <= buffer_km
        }
        triggered = bool(in_buffer)
        detail = {
            "params": {
                "min_wind_kt": min_wind,
                "buffer_km": buffer_km,
                "wind_source_priority": list(rulebook["cyclone.wind_source_priority"]),
            },
            "ibtracs": {
                "n_points_month_global": n_month,
                "n_points_qualifying_global": len(point_records),
                "max_iso_time": max_iso_time,
                "coverage_ok": coverage_ok,
                "coverage_note": coverage_note,
            },
            "storms_in_buffer": sorted(
                in_buffer.values(), key=lambda s: s["min_distance_km"]
            ),
            "near_misses": sorted(
                (s for sid, s in storms.items() if sid not in in_buffer),
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
        hazard_code,
        ym,
        n_triggered,
        len(result.rows),
        len(point_records),
        n_month,
        coverage_ok,
    )
    return result


def write_triggers(con, result: DetectionResult, rulebook: Rulebook) -> int:
    """Upsert the month's trigger rows (idempotent per hazard/ym/iso3)."""
    ensure_schema(con)
    now = utcnow_iso()
    written = 0
    for row in result.rows:
        con.execute(
            """
            INSERT OR REPLACE INTO haz_triggers
                (hazard_code, iso3, ym, triggered, trigger_source,
                 detail_json, rulebook_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                result.hazard_code,
                row.iso3,
                result.ym,
                row.triggered,
                row.trigger_source,
                json.dumps(row.detail),
                rulebook.version,
                now,
            ],
        )
        written += 1
    LOG.info("[detect] wrote %d haz_triggers rows for %s %s", written, result.hazard_code, result.ym)
    return written


def flip_trigger_from_sweep(
    con,
    *,
    hazard_code: str,
    iso3: str,
    ym: str,
    sweep_evidence: dict,
    rulebook: Rulebook,
) -> None:
    """Promote a non-triggered country-month after a non-silent sweep.

    The ReliefWeb sweep found cyclone reporting despite no qualifying
    track (remnant systems, naming edge cases): set triggered=true with
    trigger_source='reliefweb_sweep' and leave the month for the impact
    ladder (Phase 2+).
    """
    row = con.execute(
        """
        SELECT detail_json FROM haz_triggers
        WHERE hazard_code = ? AND iso3 = ? AND ym = ?
        """,
        [hazard_code, iso3, ym],
    ).fetchone()
    detail = json.loads(row[0]) if row and row[0] else {}
    detail["reliefweb_sweep"] = sweep_evidence
    con.execute(
        """
        INSERT OR REPLACE INTO haz_triggers
            (hazard_code, iso3, ym, triggered, trigger_source,
             detail_json, rulebook_version, created_at)
        VALUES (?, ?, ?, TRUE, ?, ?, ?, ?)
        """,
        [
            hazard_code,
            iso3,
            ym,
            TRIGGER_SOURCE_RELIEFWEB,
            json.dumps(detail),
            rulebook.version,
            utcnow_iso(),
        ],
    )


def record_sweep_on_trigger(
    con,
    *,
    hazard_code: str,
    iso3: str,
    ym: str,
    sweep_evidence: dict,
) -> None:
    """Attach sweep evidence to an existing (still non-triggered) row."""
    row = con.execute(
        """
        SELECT triggered, trigger_source, detail_json, rulebook_version
        FROM haz_triggers
        WHERE hazard_code = ? AND iso3 = ? AND ym = ?
        """,
        [hazard_code, iso3, ym],
    ).fetchone()
    if row is None:
        return
    detail = json.loads(row[2]) if row[2] else {}
    detail["reliefweb_sweep"] = sweep_evidence
    con.execute(
        """
        UPDATE haz_triggers SET detail_json = ?
        WHERE hazard_code = ? AND iso3 = ? AND ym = ?
        """,
        [json.dumps(detail), hazard_code, iso3, ym],
    )
