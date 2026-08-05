# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Phase-1 acceptance tests for the cyclone detection + zero path.

The three acceptance months from the design, all network-free:

- a month with a major landfall (Haiyan, Nov 2013, PHL) must trigger;
- a quiet basin month must produce zeros with complete
  evidence_of_absence;
- a near-miss track at ~250 km must NOT trigger at the rulebook's
  200 km buffer (and must trigger when the rulebook says 300 km — the
  buffer is honoured, not hard-coded).
"""

from __future__ import annotations

import json
from datetime import date

import duckdb
import pytest

from resolver.resolution_machine import cli as cli_mod
from resolver.resolution_machine.detect import (
    TRIGGER_SOURCE_IBTRACS,
    TRIGGER_SOURCE_NONE,
    TRIGGER_SOURCE_RELIEFWEB,
    detect_cyclone_month,
    flip_trigger_from_sweep,
    write_triggers,
)
from resolver.resolution_machine.geometry import load_country_geometries
from resolver.resolution_machine.resolutions import (
    WRITE_FROZEN_SKIP,
    WRITE_WRITTEN,
    freeze_date_for_ym,
    write_zero_resolution,
)
from resolver.tests.resolution_machine_utils import (
    SYNTHETIC_COUNTRIES_GEOJSON,
    make_rulebook,
    seed_ibtracs,
    silent_sweep_evidence,
)


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    seed_ibtracs(con)
    return con


@pytest.fixture(scope="module")
def real_geoms():
    return load_country_geometries()


@pytest.fixture(scope="module")
def synthetic_geoms():
    return load_country_geometries(SYNTHETIC_COUNTRIES_GEOJSON)


# ---------------------------------------------------------------------------
# Acceptance month 1: major landfall must trigger
# ---------------------------------------------------------------------------


def test_haiyan_landfall_triggers_phl(con, real_geoms):
    result = detect_cyclone_month(
        con, "2013-11", make_rulebook(), real_geoms, iso3_filter=["PHL", "MDG"]
    )
    rows = {r.iso3: r for r in result.rows}
    phl = rows["PHL"]
    assert phl.triggered is True
    assert phl.trigger_source == TRIGGER_SOURCE_IBTRACS
    storms = phl.detail["storms_in_buffer"]
    assert [s["sid"] for s in storms] == ["2013306N07162"]
    assert storms[0]["name"] == "HAIYAN"
    # Closest fixture point sits in Leyte Gulf, a few km off the 1:50m
    # coastline — comfortably a landfall-scale approach.  (The exact
    # over-land == 0.0 semantic is pinned in the geometry tests.)
    assert storms[0]["min_distance_km"] < 25.0
    assert storms[0]["max_wind_kt"] >= 150
    # Same month, other side of the world: no trigger.
    assert rows["MDG"].triggered is False
    assert rows["MDG"].trigger_source == TRIGGER_SOURCE_NONE


def test_triggers_written_for_both_outcomes(con, real_geoms):
    rb = make_rulebook()
    result = detect_cyclone_month(
        con, "2013-11", rb, real_geoms, iso3_filter=["PHL", "MDG"]
    )
    write_triggers(con, result, rb)
    stored = con.execute(
        "SELECT iso3, triggered, trigger_source FROM haz_triggers "
        "WHERE hazard_code = 'TC' AND ym = '2013-11' ORDER BY iso3"
    ).fetchall()
    assert stored == [("MDG", False, "none"), ("PHL", True, "ibtracs")]
    # Idempotent: re-running detection + write leaves one row per country.
    result2 = detect_cyclone_month(
        con, "2013-11", rb, real_geoms, iso3_filter=["PHL", "MDG"]
    )
    write_triggers(con, result2, rb)
    n = con.execute(
        "SELECT COUNT(*) FROM haz_triggers WHERE hazard_code='TC' AND ym='2013-11'"
    ).fetchone()[0]
    assert n == 2


# ---------------------------------------------------------------------------
# Acceptance month 2: near-miss at ~250 km must NOT trigger at 200 km
# ---------------------------------------------------------------------------


def test_near_miss_honours_rulebook_buffer(con, synthetic_geoms):
    # Default rulebook: buffer_km = 200. The NEARMISS track point sits
    # 250 km west of Squareland — must NOT trigger.
    result = detect_cyclone_month(
        con, "2013-06", make_rulebook(), synthetic_geoms
    )
    aaa = {r.iso3: r for r in result.rows}["AAA"]
    assert aaa.triggered is False
    near = aaa.detail["near_misses"]
    assert [s["sid"] for s in near] == ["TEST01NEAR250"]
    assert near[0]["min_distance_km"] == pytest.approx(250.0, abs=1.5)

    # Same data, wider rulebook buffer: the SAME point must trigger —
    # proof the threshold comes from the rulebook, not code.
    wide = make_rulebook({"cyclone": {"buffer_km": 300}})
    result_wide = detect_cyclone_month(con, "2013-06", wide, synthetic_geoms)
    aaa_wide = {r.iso3: r for r in result_wide.rows}["AAA"]
    assert aaa_wide.triggered is True
    assert aaa_wide.detail["storms_in_buffer"][0]["sid"] == "TEST01NEAR250"


def test_min_wind_threshold_is_honoured(con, synthetic_geoms):
    # WEAKLING (20 kt) passes 11 km from Squareland in 2013-06 — below
    # min_wind_kt=34 it must not appear at all, let alone trigger.
    result = detect_cyclone_month(con, "2013-06", make_rulebook(), synthetic_geoms)
    aaa = {r.iso3: r for r in result.rows}["AAA"]
    sids = {s["sid"] for s in aaa.detail["storms_in_buffer"] + aaa.detail["near_misses"]}
    assert "TEST02LOWWIND" not in sids

    # Lower the rulebook threshold and the same track point triggers.
    low = make_rulebook({"cyclone": {"min_wind_kt": 15}})
    result_low = detect_cyclone_month(con, "2013-06", low, synthetic_geoms)
    aaa_low = {r.iso3: r for r in result_low.rows}["AAA"]
    assert "TEST02LOWWIND" in {s["sid"] for s in aaa_low.detail["storms_in_buffer"]}


def test_wind_source_priority_falls_back_to_wmo(con, synthetic_geoms):
    # TEST03INSIDE (2013-07, 150 km from Squareland) has no USA wind,
    # only WMO 60 kt — the priority chain must pick it up and trigger.
    result = detect_cyclone_month(con, "2013-07", make_rulebook(), synthetic_geoms)
    aaa = {r.iso3: r for r in result.rows}["AAA"]
    assert aaa.triggered is True
    storm = aaa.detail["storms_in_buffer"][0]
    assert storm["sid"] == "TEST03INSIDE"
    assert storm["max_wind_kt"] == pytest.approx(60.0)


# ---------------------------------------------------------------------------
# Acceptance month 3: quiet month resolves zero with evidence of absence
# ---------------------------------------------------------------------------


def test_quiet_month_zero_carries_complete_evidence(con):
    rb = make_rulebook()
    evidence = {
        "ibtracs": {
            "total_points": 10,
            "max_iso_time": "2013-12-15 00:00:00",
            "last_fetched_at": "2026-01-15T00:00:00+00:00",
            "source_scopes": ["last3years"],
            "source_urls": ["https://example.test/ibtracs.csv"],
            "query": {
                "ym": "2013-01",
                "min_wind_kt": rb["cyclone.min_wind_kt"],
                "buffer_km": rb["cyclone.buffer_km"],
                "n_points_month_global": 0,
                "n_points_qualifying_global": 0,
                "n_storms_in_buffer": 0,
                "coverage_note": "ok",
            },
        },
        "reliefweb": silent_sweep_evidence("PHL", "2013-01"),
        "retrieved_at": "2026-01-15T00:00:00+00:00",
    }
    outcome = write_zero_resolution(
        con,
        iso3="PHL",
        hazard_code="TC",
        ym="2013-01",
        evidence_of_absence=evidence,
        rulebook=rb,
        today=date(2013, 2, 15),
    )
    assert outcome == WRITE_WRITTEN

    row = con.execute(
        """
        SELECT status, value, unit, source, source_record_ids, source_urls,
               rule_fired, evidence_json, retrieved_at, freeze_at
        FROM pa_resolutions
        WHERE iso3='PHL' AND hazard_code='TC' AND ym='2013-01'
        """
    ).fetchone()
    assert row is not None
    status, value, unit, source, record_ids, urls, rule, ev_json, retrieved, freeze = row
    # Hard rule 1: full provenance on every resolution row.
    assert status == "RESOLVED_ZERO" and value == 0.0 and unit == "persons"
    assert source == "resolution_machine:absence"
    assert json.loads(record_ids) == []
    assert "https://api.reliefweb.int/v2/reports" in json.loads(urls)
    assert "no IBTrACS trigger and ReliefWeb sweep silent" in rule
    assert retrieved
    ev = json.loads(ev_json)
    # Evidence of absence: IBTrACS query summary + ReliefWeb queries with
    # hit counts + timestamps.
    assert ev["ibtracs"]["query"]["buffer_km"] == rb["cyclone.buffer_km"]
    assert ev["reliefweb"]["total_hits"] == 0
    assert [q["hits"] for q in ev["reliefweb"]["queries"]] == [0, 0]
    assert ev["retrieved_at"]
    assert freeze == freeze_date_for_ym("2013-01", rb).isoformat()


def test_freeze_is_immutable_and_logs_revisions(con):
    rb = make_rulebook()
    evidence = {"ibtracs": {}, "reliefweb": silent_sweep_evidence("MDG", "2013-01"),
                "retrieved_at": "2013-02-01T00:00:00+00:00"}
    kwargs = dict(con=con, iso3="MDG", hazard_code="TC", ym="2013-01",
                  evidence_of_absence=evidence, rulebook=rb)

    # Initial write and a pre-freeze re-run both succeed (idempotent).
    assert write_zero_resolution(**kwargs, today=date(2013, 2, 15)) == WRITE_WRITTEN
    assert write_zero_resolution(**kwargs, today=date(2013, 3, 20)) == WRITE_WRITTEN
    before = con.execute(
        "SELECT resolved_at FROM pa_resolutions WHERE iso3='MDG' AND ym='2013-01'"
    ).fetchone()[0]

    # 2013-01 froze on 2013-04-01 (Jan 31 + 60d). After that: immutable.
    assert write_zero_resolution(**kwargs, today=date(2013, 6, 1)) == WRITE_FROZEN_SKIP
    after = con.execute(
        "SELECT resolved_at FROM pa_resolutions WHERE iso3='MDG' AND ym='2013-01'"
    ).fetchone()[0]
    assert after == before  # resolved row untouched

    revisions = con.execute(
        "SELECT observed_status, note FROM pa_resolution_revisions "
        "WHERE iso3='MDG' AND ym='2013-01'"
    ).fetchall()
    assert len(revisions) == 1
    assert revisions[0][0] == "RESOLVED_ZERO"
    assert "freeze" in revisions[0][1]

    n_rows = con.execute(
        "SELECT COUNT(*) FROM pa_resolutions WHERE iso3='MDG' AND ym='2013-01'"
    ).fetchone()[0]
    assert n_rows == 1


# ---------------------------------------------------------------------------
# Sweep outcomes drive the trigger table
# ---------------------------------------------------------------------------


def test_sweep_hits_flip_trigger_for_the_ladder(con, real_geoms):
    rb = make_rulebook()
    result = detect_cyclone_month(
        con, "2013-01", rb, real_geoms, iso3_filter=["PHL"]
    )
    write_triggers(con, result, rb)
    sweep = silent_sweep_evidence("PHL", "2013-01")
    sweep.update({"silent": False, "total_hits": 4})
    flip_trigger_from_sweep(
        con, hazard_code="TC", iso3="PHL", ym="2013-01",
        sweep_evidence=sweep, rulebook=rb,
    )
    row = con.execute(
        "SELECT triggered, trigger_source, detail_json FROM haz_triggers "
        "WHERE hazard_code='TC' AND iso3='PHL' AND ym='2013-01'"
    ).fetchone()
    assert row[0] is True
    assert row[1] == TRIGGER_SOURCE_RELIEFWEB
    assert json.loads(row[2])["reliefweb_sweep"]["total_hits"] == 4


# ---------------------------------------------------------------------------
# End-to-end through the CLI runner (the acceptance command)
# ---------------------------------------------------------------------------


def _seeded_file_db(tmp_path):
    from resolver.db.duckdb_io import get_db

    db_path = str(tmp_path / "resolution_machine_test.duckdb")
    con = get_db(db_path)
    seed_ibtracs(con)
    return db_path, con


def test_cli_landfall_and_quiet_month_end_to_end(tmp_path, monkeypatch):
    db_path, con = _seeded_file_db(tmp_path)
    monkeypatch.setattr(
        "resolver.resolution_machine.reliefweb_sweep.sweep_country_month",
        lambda iso3, ym, rulebook, **kw: silent_sweep_evidence(iso3, ym),
    )
    rc = cli_mod.run_cyclone_month(
        ym="2013-11",
        db_url=db_path,
        countries_filter=["PHL", "MDG"],
        scope="auto",
        skip_fetch=True,
        no_sweep=False,
        dry_run=False,
    )
    assert rc == 0

    triggers = dict(
        con.execute(
            "SELECT iso3, triggered FROM haz_triggers "
            "WHERE hazard_code='TC' AND ym='2013-11'"
        ).fetchall()
    )
    assert triggers == {"PHL": True, "MDG": False}

    # The landfall country must NOT get a zero; the quiet country must.
    resolutions = con.execute(
        "SELECT iso3, status, value FROM pa_resolutions "
        "WHERE hazard_code='TC' AND ym='2013-11' ORDER BY iso3"
    ).fetchall()
    assert resolutions == [("MDG", "RESOLVED_ZERO", 0.0)]

    ev = json.loads(
        con.execute(
            "SELECT evidence_json FROM pa_resolutions "
            "WHERE iso3='MDG' AND hazard_code='TC' AND ym='2013-11'"
        ).fetchone()[0]
    )
    # The CLI assembles evidence from the live store: IBTrACS provenance
    # and the sweep record must both be present.
    assert ev["ibtracs"]["total_points"] == 10
    assert ev["ibtracs"]["query"]["n_storms_in_buffer"] == 0
    assert ev["reliefweb"]["silent"] is True
    # The non-triggered row also carries the sweep evidence.
    detail = json.loads(
        con.execute(
            "SELECT detail_json FROM haz_triggers "
            "WHERE hazard_code='TC' AND iso3='MDG' AND ym='2013-11'"
        ).fetchone()[0]
    )
    assert detail["reliefweb_sweep"]["silent"] is True


def test_cli_sweep_hits_promote_to_ladder(tmp_path, monkeypatch):
    db_path, con = _seeded_file_db(tmp_path)

    def noisy_sweep(iso3, ym, rulebook, **kw):
        ev = silent_sweep_evidence(iso3, ym)
        ev.update({"silent": False, "total_hits": 7})
        return ev

    monkeypatch.setattr(
        "resolver.resolution_machine.reliefweb_sweep.sweep_country_month", noisy_sweep
    )
    rc = cli_mod.run_cyclone_month(
        ym="2013-01",
        db_url=db_path,
        countries_filter=["PHL"],
        scope="auto",
        skip_fetch=True,
        no_sweep=False,
        dry_run=False,
    )
    assert rc == 0
    row = con.execute(
        "SELECT triggered, trigger_source FROM haz_triggers "
        "WHERE hazard_code='TC' AND iso3='PHL' AND ym='2013-01'"
    ).fetchone()
    assert row == (True, TRIGGER_SOURCE_RELIEFWEB)
    # Promoted to the ladder — never a zero.
    n_zero = con.execute(
        "SELECT COUNT(*) FROM pa_resolutions WHERE iso3='PHL' AND ym='2013-01'"
    ).fetchone()[0]
    assert n_zero == 0


def test_cli_coverage_gate_suppresses_zeros(tmp_path, monkeypatch):
    # The fixture's newest track point is 2013-12-15; December's coverage
    # requirement is Dec 31 - 7d = Dec 24 — not met, so a silent quiet
    # country stays UNRESOLVED instead of becoming a false zero.
    db_path, con = _seeded_file_db(tmp_path)
    monkeypatch.setattr(
        "resolver.resolution_machine.reliefweb_sweep.sweep_country_month",
        lambda iso3, ym, rulebook, **kw: silent_sweep_evidence(iso3, ym),
    )
    rc = cli_mod.run_cyclone_month(
        ym="2013-12",
        db_url=db_path,
        countries_filter=["MDG"],
        scope="auto",
        skip_fetch=True,
        no_sweep=False,
        dry_run=False,
    )
    assert rc == 0
    n_zero = con.execute(
        "SELECT COUNT(*) FROM pa_resolutions WHERE ym='2013-12'"
    ).fetchone()[0]
    assert n_zero == 0
    # But the trigger row still records the (non-)detection.
    row = con.execute(
        "SELECT triggered FROM haz_triggers "
        "WHERE hazard_code='TC' AND iso3='MDG' AND ym='2013-12'"
    ).fetchone()
    assert row == (False,)


def test_cli_inconclusive_sweep_never_writes_zero(tmp_path, monkeypatch):
    db_path, con = _seeded_file_db(tmp_path)

    def broken_sweep(iso3, ym, rulebook, **kw):
        ev = silent_sweep_evidence(iso3, ym)
        ev.update({"silent": False, "inconclusive": True, "error": "boom"})
        return ev

    monkeypatch.setattr(
        "resolver.resolution_machine.reliefweb_sweep.sweep_country_month", broken_sweep
    )
    rc = cli_mod.run_cyclone_month(
        ym="2013-11",
        db_url=db_path,
        countries_filter=["MDG"],
        scope="auto",
        skip_fetch=True,
        no_sweep=False,
        dry_run=False,
    )
    assert rc == 0
    n = con.execute("SELECT COUNT(*) FROM pa_resolutions").fetchone()[0]
    assert n == 0
