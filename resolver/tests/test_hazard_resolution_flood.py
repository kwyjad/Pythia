# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Phase-2 acceptance tests for the flood path, end to end.

The acceptance criterion: run the full flood path for three past months
and check that **every triggered country-month has candidates or a flagged
NO_DATA, and every zero has evidence of absence**. The bottom of this file
asserts exactly that, over three months, against a real DuckDB.

All network-free: GDACS events are seeded into the raw cache, the ladder
sources use their injectable transport seams, and the ReliefWeb sweep is
monkeypatched the way the Phase 1 suite does it.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest

from resolver.hazard_resolution import candidates as cand_mod
from resolver.hazard_resolution import cli as cli_mod
from resolver.hazard_resolution import impact as impact_mod
from resolver.hazard_resolution.detect import (
    TRIGGER_SOURCE_GDACS,
    TRIGGER_SOURCE_NONE,
    detect_flood_month,
    write_triggers,
)
from resolver.hazard_resolution.schema import ensure_haz_schema
from resolver.tests.hazard_resolution_utils import (
    make_rulebook,
    seed_gdacs_event,
    seed_population,
    silent_sweep_evidence,
)

# Long past every freeze deadline, so NO_DATA (not PENDING) is the
# expected outcome for a triggered cell nobody reported on.
TODAY = dt.date(2024, 12, 1)


@pytest.fixture()
def rulebook():
    return make_rulebook()


@pytest.fixture()
def con():
    con = duckdb.connect(":memory:")
    ensure_haz_schema(con)
    seed_population(con)
    return con


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def test_orange_alert_triggers_green_does_not(con, rulebook):
    seed_gdacs_event(con, iso3="PHL", alert_level="Orange", event_id="1")
    seed_gdacs_event(con, iso3="VNM", alert_level="Green", event_id="2")

    result = detect_flood_month(con, "2024-03", rulebook, iso3_filter=["PHL", "VNM"])
    verdicts = {row.iso3: (row.triggered, row.trigger_source) for row in result.rows}

    assert verdicts["PHL"] == (True, TRIGGER_SOURCE_GDACS)
    assert verdicts["VNM"] == (False, TRIGGER_SOURCE_NONE)


def test_red_alert_triggers_at_the_orange_threshold(con, rulebook):
    seed_gdacs_event(con, iso3="PHL", alert_level="Red")
    result = detect_flood_month(con, "2024-03", rulebook, iso3_filter=["PHL"])
    assert result.rows[0].triggered is True


def test_trigger_level_is_honoured_from_the_rulebook(con):
    """Raising the bar to red stops an orange event triggering — no code change."""
    seed_gdacs_event(con, iso3="PHL", alert_level="Orange")
    assert detect_flood_month(
        con, "2024-03", make_rulebook(), iso3_filter=["PHL"]
    ).rows[0].triggered is True
    assert detect_flood_month(
        con, "2024-03", make_rulebook({"flood": {"gdacs_trigger_level": "red"}}),
        iso3_filter=["PHL"],
    ).rows[0].triggered is False


def test_rows_are_written_for_triggered_and_non_triggered_alike(con, rulebook):
    seed_gdacs_event(con, iso3="PHL", alert_level="Orange")
    result = detect_flood_month(con, "2024-03", rulebook, iso3_filter=["PHL", "VNM"])
    write_triggers(con, result, rulebook)

    rows = dict(
        con.execute(
            "SELECT iso3, triggered FROM haz_triggers "
            "WHERE hazard='FL' AND year=2024 AND month=3"
        ).fetchall()
    )
    assert rows == {"PHL": True, "VNM": False}


def test_an_event_spanning_a_month_boundary_triggers_both_months(con, rulebook):
    """Detection scope is overlap — the flood did happen in both months."""
    seed_gdacs_event(
        con, iso3="PHL", alert_level="Orange",
        start_date="2024-01-28", end_date="2024-02-04", ym="2024-01",
    )
    for ym in ("2024-01", "2024-02"):
        result = detect_flood_month(con, ym, rulebook, iso3_filter=["PHL"])
        assert result.rows[0].triggered is True, ym


def test_an_unknown_alert_colour_does_not_silently_trigger(con, rulebook):
    """A data problem is recorded, not converted into a verdict."""
    seed_gdacs_event(con, iso3="PHL", alert_level="Chartreuse")
    result = detect_flood_month(con, "2024-03", rulebook, iso3_filter=["PHL"])
    row = result.rows[0]
    assert row.triggered is False
    assert "alert_level_error" in row.detail["events_below_threshold"][0]


def test_coverage_gate_suppresses_zeros_when_gdacs_is_stale(con, rulebook):
    """An ingestion gap must never read as a quiet month."""
    seed_gdacs_event(
        con, iso3="PHL", start_date="2024-01-05", end_date="2024-01-09", ym="2024-01",
    )
    # March's requirement is Mar 31 - 7d; the newest stored event is January.
    result = detect_flood_month(con, "2024-03", rulebook, iso3_filter=["PHL"])
    assert result.coverage_ok is False
    assert "predates" in result.coverage_note


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def test_gdacs_enters_candidates_only_as_a_ceiling(con, rulebook):
    seed_gdacs_event(con, iso3="PHL", exposed_population=500_000)
    found = cand_mod.build_candidates(con, "PHL", "2024-03", "FL", rulebook)

    assert [c.value_type for c in found] == [cand_mod.VALUE_CEILING]
    assert cand_mod.ladder_candidates(found) == []  # never choosable
    assert "NOT reported impact" in found[0].stated_by


def test_candidates_are_rewritten_idempotently(con, rulebook):
    seed_gdacs_event(con, iso3="PHL")
    found = cand_mod.build_candidates(con, "PHL", "2024-03", "FL", rulebook)
    for _ in range(3):
        cand_mod.write_candidates(con, found, "PHL", "2024-03", "FL")
    assert con.execute("SELECT COUNT(*) FROM haz_impact_candidates").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Population denominator
# ---------------------------------------------------------------------------


def test_population_lookup_takes_the_most_recent_year_at_or_before(con):
    con.execute(
        "INSERT INTO haz_raw_population (iso3, year, population, source) "
        "VALUES ('PHL', 2020, 109000000, 'test')"
    )
    assert impact_mod.national_population(con, "PHL", 2024) == 115_000_000
    assert impact_mod.national_population(con, "PHL", 2021) == 109_000_000
    assert impact_mod.national_population(con, "XXX", 2024) is None


def test_population_falls_back_to_the_earliest_year_for_old_cells(con):
    """A denominator from the wrong decade still bounds a figure usefully."""
    assert impact_mod.national_population(con, "PHL", 1990) == 115_000_000


# ---------------------------------------------------------------------------
# End-to-end: the three-month acceptance run
# ---------------------------------------------------------------------------


def _run_three_months(con, rulebook, monkeypatch, *, go_reports):
    """Run the flood ladder over three past months, network-free."""
    monkeypatch.setenv("EMDAT_API_KEY", "")
    monkeypatch.setenv("IDMC_API_KEY", "")

    months = ["2024-01", "2024-02", "2024-03"]
    # PHL floods in Jan and Mar; VNM is quiet throughout.
    seed_gdacs_event(
        con, iso3="PHL", ym="2024-01", event_id="101",
        start_date="2024-01-05", end_date="2024-01-09", exposed_population=400_000,
    )
    seed_gdacs_event(
        con, iso3="PHL", ym="2024-03", event_id="103",
        start_date="2024-03-05", end_date="2024-03-09", exposed_population=600_000,
    )

    for ym in months:
        result = detect_flood_month(con, ym, rulebook, iso3_filter=["PHL", "VNM"])
        # Force the coverage gate open: the seeded store is deliberately
        # sparse, and this test is about the ladder, not the gate (which
        # has its own test above).
        result.coverage_ok = True
        result.coverage_note = "ok (test)"
        write_triggers(con, result, rulebook)

        cli_mod.sweep_and_resolve_zeros(
            con, result, rulebook,
            hazard_key="flood",
            detector_evidence={"gdacs": {"source_urls": ["https://example.test/gdacs"]}},
            dry_run=False,
        )

        triggered = impact_mod.triggered_iso3s(con, ym, "FL")
        if triggered:
            impact_mod.resolve_triggered_cells(
                con,
                ym=ym,
                hazard="FL",
                iso3s=triggered,
                rulebook=rulebook,
                fetches={"emdat": {"ok": True}, "ifrc_go": {"ok": True},
                         "idmc_idu": {"ok": True}, "gdacs": {"ok": True}},
                today=TODAY,
            )
    return months


def test_three_month_flood_run_meets_the_acceptance_criteria(con, rulebook, monkeypatch):
    """Every triggered cell has candidates or a flagged NO_DATA;
    every zero has evidence of absence."""
    monkeypatch.setattr(
        "resolver.hazard_resolution.reliefweb_sweep.sweep_country_month",
        lambda iso3, ym, rulebook, **kw: silent_sweep_evidence(iso3, ym),
    )
    # One GO report, for the March flood only.
    go_report = {
        "id": 9001, "summary": "Philippines: Floods", "dtype": {"name": "Flood"},
        "report_date": "2024-03-11", "num_affected": 30000,
        "countries_details": [{"iso3": "PHL", "name": "Philippines"}],
    }
    from resolver.hazard_resolution import ifrc_go as go_mod

    go_mod.fetch_ifrc_go(
        con, "2024-03", "FL", rulebook,
        get=lambda *a: {"results": [go_report], "next": None},
    )

    _run_three_months(con, rulebook, monkeypatch, go_reports=[go_report])

    rows = con.execute(
        """
        SELECT iso3, year, month, status, value, flagged, provenance_json
        FROM haz_resolutions WHERE hazard='FL' ORDER BY year, month, iso3
        """
    ).fetchall()
    assert rows, "the three-month run produced no resolutions at all"

    triggered_cells = {
        (iso3, year, month)
        for iso3, year, month in con.execute(
            "SELECT iso3, year, month FROM haz_triggers WHERE hazard='FL' AND triggered"
        ).fetchall()
    }
    assert triggered_cells, "no cell triggered — the fixture is not exercising the path"

    for iso3, year, month, status, value, flagged, provenance_json in rows:
        provenance = json.loads(provenance_json)
        cell = (iso3, year, month)

        if cell in triggered_cells:
            # ACCEPTANCE: candidates, or a flagged NO_DATA.
            n_candidates = con.execute(
                """
                SELECT COUNT(*) FROM haz_impact_candidates
                WHERE iso3=? AND year=? AND month=? AND hazard='FL'
                  AND value_type <> 'exposed_ceiling'
                """,
                [iso3, year, month],
            ).fetchone()[0]
            if status == "NO_DATA":
                assert flagged is True, f"{cell} NO_DATA must be flagged"
            else:
                assert status == "RESOLVED_VALUE"
                assert n_candidates > 0, f"{cell} resolved a value with no candidate"
            assert provenance["rule_fired"]
        else:
            # ACCEPTANCE: every zero carries evidence of absence.
            assert status == "RESOLVED_ZERO"
            assert value == 0.0
            evidence = provenance["evidence_of_absence"]
            assert evidence["reliefweb"]["silent"] is True
            assert evidence["gdacs"]
            assert provenance["rule_fired"] == (
                "flood_zero:no_gdacs_trigger+reliefweb_silent"
            )


def test_march_flood_resolves_from_the_go_rung_under_its_ceiling(con, rulebook, monkeypatch):
    monkeypatch.setattr(
        "resolver.hazard_resolution.reliefweb_sweep.sweep_country_month",
        lambda iso3, ym, rulebook, **kw: silent_sweep_evidence(iso3, ym),
    )
    from resolver.hazard_resolution import ifrc_go as go_mod

    go_mod.fetch_ifrc_go(
        con, "2024-03", "FL", rulebook,
        get=lambda *a: {
            "results": [{
                "id": 9001, "summary": "Philippines: Floods", "dtype": {"name": "Flood"},
                "report_date": "2024-03-11", "num_affected": 30000,
                "countries_details": [{"iso3": "PHL", "name": "Philippines"}],
            }],
            "next": None,
        },
    )
    _run_three_months(con, rulebook, monkeypatch, go_reports=[])

    row = con.execute(
        "SELECT status, value, flagged, rule_fired FROM haz_resolutions "
        "WHERE iso3='PHL' AND hazard='FL' AND year=2024 AND month=3"
    ).fetchone()
    assert row == ("RESOLVED_VALUE", 30000.0, False, "ladder:ifrc_go")


def test_january_flood_with_no_reporting_is_flagged_no_data(con, rulebook, monkeypatch):
    """Triggered, past freeze, nobody reported — a legible gap, not a zero."""
    monkeypatch.setattr(
        "resolver.hazard_resolution.reliefweb_sweep.sweep_country_month",
        lambda iso3, ym, rulebook, **kw: silent_sweep_evidence(iso3, ym),
    )
    _run_three_months(con, rulebook, monkeypatch, go_reports=[])

    status, flagged = con.execute(
        "SELECT status, flagged FROM haz_resolutions "
        "WHERE iso3='PHL' AND hazard='FL' AND year=2024 AND month=1"
    ).fetchone()
    assert (status, flagged) == ("NO_DATA", True)


def test_a_quiet_country_gets_zeros_in_all_three_months(con, rulebook, monkeypatch):
    monkeypatch.setattr(
        "resolver.hazard_resolution.reliefweb_sweep.sweep_country_month",
        lambda iso3, ym, rulebook, **kw: silent_sweep_evidence(iso3, ym),
    )
    _run_three_months(con, rulebook, monkeypatch, go_reports=[])

    zeros = con.execute(
        "SELECT month FROM haz_resolutions WHERE iso3='VNM' AND hazard='FL' "
        "AND status='RESOLVED_ZERO' ORDER BY month"
    ).fetchall()
    assert [m for (m,) in zeros] == [1, 2, 3]


def test_unavailable_sources_are_recorded_on_the_row(con, rulebook):
    """A rung that could not be READ is not a rung that was empty."""
    seed_gdacs_event(con, iso3="PHL")
    result = detect_flood_month(con, "2024-03", rulebook, iso3_filter=["PHL"])
    write_triggers(con, result, rulebook)

    impact_mod.resolve_triggered_cells(
        con,
        ym="2024-03",
        hazard="FL",
        iso3s=["PHL"],
        rulebook=rulebook,
        fetches={
            "emdat": {"ok": False, "error": "EMDAT_API_KEY not set"},
            "ifrc_go": {"ok": True},
            "idmc_idu": {"ok": True},
            "gdacs": {"ok": True},
        },
        today=TODAY,
    )
    provenance = json.loads(
        con.execute(
            "SELECT provenance_json FROM haz_resolutions WHERE iso3='PHL'"
        ).fetchone()[0]
    )
    assert provenance["decision"]["sources_unavailable"] == ["emdat"]
    assert provenance["source_fetches"]["emdat"]["error"] == "EMDAT_API_KEY not set"
