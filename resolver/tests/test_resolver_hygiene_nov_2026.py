# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Resolver hygiene from the November 2026 review (run 42999258374).

1. Flood and cyclone write no answer for a month that has not ended. On 11
   November the machine wrote 14 FL and 3 TC values for 2026-11.
2. A cell for a territory with no population reads ``no_population``,
   never ``unexplained_no_row``.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from resolver.hazard_resolution import cell_ledger
from resolver.hazard_resolution import cli as cli_mod
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.tests.hazard_resolution_utils import (
    make_rulebook,
    seed_gdacs_event,
    seed_population,
    silent_sweep_evidence,
)


@pytest.fixture()
def rulebook():
    return make_rulebook()


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    seed_population(con)
    return con


def _insert_row(con, iso3, year, month, hazard, *, provisional=True, created_at=None):
    con.execute(
        "INSERT INTO haz_resolutions (iso3, year, month, hazard, status, value, "
        "provenance_json, rule_fired, flagged, provisional, run_type, frozen_at, created_at) "
        "VALUES (?, ?, ?, ?, 'RESOLVED_VALUE', 1000.0, '{}', 'ladder:test', FALSE, ?, "
        "'live', NULL, ?)",
        [iso3, year, month, hazard, provisional,
         created_at or dt.datetime(year, month, 11, 4, 0)],
    )


def _run_flood(con, rulebook, monkeypatch, *, today):
    swept: list[str] = []

    def _sweep(iso3, ym, rulebook, **kw):
        swept.append(iso3)
        return silent_sweep_evidence(iso3, ym)

    monkeypatch.setattr(cli_mod, "_load_universe", lambda: ["VNM", "IDN"])
    monkeypatch.setattr(
        "resolver.hazard_resolution.reliefweb_sweep.sweep_country_month", _sweep
    )
    monkeypatch.setattr(
        "resolver.hazard_resolution.gdacs.coverage", lambda *a, **k: (True, "ok (test)")
    )
    rc = cli_mod.run_flood_month(
        ym="2024-03", db_url=None, countries_filter=None, skip_fetch=True,
        no_sweep=False, dry_run=False, no_ladder=False, rulebook=rulebook,
        con=con, today=today,
    )
    return rc, swept


def test_a_flood_walk_on_the_11th_writes_nothing_for_the_month(con, rulebook, monkeypatch):
    seed_gdacs_event(con, iso3="VNM", alert_level="Orange", event_id="901")
    _insert_row(con, "IDN", 2024, 3, "FL")  # an earlier mid-month answer

    rc, swept = _run_flood(con, rulebook, monkeypatch, today=dt.date(2024, 3, 11))

    assert rc == 0
    assert swept == [], "a month in progress is not swept"
    assert con.execute(
        "SELECT COUNT(*) FROM haz_resolutions WHERE hazard='FL'"
    ).fetchone()[0] == 0
    # Detection stands, and each trigger row says why it has no answer.
    rows = con.execute(
        "SELECT iso3, triggered, trigger_detail_json FROM haz_triggers "
        "WHERE hazard='FL' ORDER BY iso3"
    ).fetchall()
    assert [r[0] for r in rows] == ["IDN", "VNM"]
    assert {r[0]: r[1] for r in rows}["VNM"] is True
    for _, _, detail in rows:
        assert json.loads(detail)["no_row_reason"] == cell_ledger.REASON_MONTH_IN_PROGRESS


def test_the_same_walk_after_the_month_ends_writes_zeros(con, rulebook, monkeypatch):
    rc, swept = _run_flood(con, rulebook, monkeypatch, today=dt.date(2024, 4, 11))
    assert rc == 0
    assert swept == ["IDN", "VNM"]
    assert con.execute(
        "SELECT COUNT(*) FROM haz_resolutions WHERE hazard='FL' AND status='RESOLVED_ZERO'"
    ).fetchone()[0] == 2


def test_rows_written_mid_month_are_retracted_and_others_kept(con):
    _insert_row(con, "PHL", 2026, 11, "TC", created_at=dt.datetime(2026, 11, 11, 5))
    _insert_row(con, "VNM", 2026, 10, "TC", created_at=dt.datetime(2026, 11, 11, 5))
    _insert_row(con, "CUB", 2026, 11, "TC", provisional=False,
                created_at=dt.datetime(2026, 11, 11, 5))
    _insert_row(con, "PHL", 2026, 11, "FL", created_at=dt.datetime(2026, 11, 11, 5))

    assert cli_mod.retract_rows_written_mid_month(con, "TC") == 1
    left = con.execute(
        "SELECT iso3, hazard, month FROM haz_resolutions ORDER BY hazard, iso3"
    ).fetchall()
    assert left == [("PHL", "FL", 11), ("CUB", "TC", 11), ("VNM", "TC", 10)]
    assert cli_mod.retract_rows_written_mid_month(con, "TC") == 0
