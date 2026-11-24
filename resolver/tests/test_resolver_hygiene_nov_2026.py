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


# --- 2. territories with no population --------------------------------------


def test_no_population_is_a_known_no_row_reason():
    assert cell_ledger.REASON_NO_POPULATION == "no_population"
    assert cell_ledger.REASON_NO_POPULATION in cell_ledger.NO_ROW_REASONS
    assert cell_ledger.REASON_SWEEP_HIT in cell_ledger.NO_ROW_REASONS


def test_unpopulated_trigger_rows_are_stamped_no_population(con):
    from resolver.hazard_resolution import detect as detect_mod

    for iso3 in ("ATA", "BVT", "VNM"):
        con.execute(
            "INSERT INTO haz_triggers (iso3, year, month, hazard, triggered, "
            "trigger_source, trigger_detail_json, run_type) "
            "VALUES (?, 2013, 4, 'FL', FALSE, 'none', '{}', 'backcast')",
            [iso3],
        )
    n = detect_mod.stamp_unpopulated_territories(con, cli_mod.UNPOPULATED_TERRITORIES)
    assert n == 2
    reasons = dict(con.execute(
        "SELECT iso3, json_extract_string(trigger_detail_json, '$.no_row_reason') "
        "FROM haz_triggers"
    ).fetchall())
    assert reasons == {"ATA": "no_population", "BVT": "no_population", "VNM": None}
    assert detect_mod.stamp_unpopulated_territories(
        con, cli_mod.UNPOPULATED_TERRITORIES
    ) == 0


# --- 5. conflictforecast.org: no new vintage vs a failure --------------------


def test_the_issue_source_maps_no_new_vintage_to_info():
    from resolver.diagnostics.issue_sources import issues_from_reconciliation
    from resolver.diagnostics.issues import DEGRADED, INFO

    issues = issues_from_reconciliation([
        {"connector": "conflictforecast_forecasts", "status": "ok", "agrees": "—",
         "reason": "no_new_vintage: newest stored issue date 2026-11-01"},
        {"connector": "views_forecasts", "status": "error", "agrees": "—",
         "reason": "fetch-returned-empty"},
    ])
    by_id = {i.id: i for i in issues}
    assert by_id["no_new_vintage_conflictforecast_forecasts"].severity == INFO
    assert by_id["connector_failed_views_forecasts"].severity == DEGRADED


def test_an_empty_answer_is_classified_by_the_stored_vintage():
    from resolver.tools.fetch_conflict_forecasts import classify_empty_answer

    today = dt.date(2026, 11, 11)
    failed = {"outcome": "download_failed", "errors": ["CERTIFICATE_VERIFY_FAILED"]}
    # November's vintage is stored: nothing is lost this run.
    assert classify_empty_answer(failed, dt.date(2026, 11, 1), today) == "no_new_vintage"
    # October's is still the previous month: no new vintage yet.
    assert classify_empty_answer(failed, dt.date(2026, 10, 1), today) == "no_new_vintage"
    # Nothing for the current or previous month: a failure.
    assert classify_empty_answer(failed, dt.date(2026, 9, 1), today) == "fetch_failed"
    assert classify_empty_answer(failed, None, today) == "fetch_failed"
    assert classify_empty_answer({"outcome": "no_new_vintage"}, None, today) == "no_new_vintage"


def test_the_connector_skips_the_download_when_the_listing_names_a_stored_vintage(monkeypatch):
    from resolver.connectors.conflictforecast import ConflictForecastOrgConnector

    connector = ConflictForecastOrgConnector()
    connector.newest_stored = dt.date(2026, 11, 1)
    listing = [{"name": "conflictforecast_ons_armedconf_03_11-2026.csv", "publicUrl": "x"}]
    monkeypatch.setattr(connector, "_get_file_listing", lambda: listing)

    def _boom(url):
        raise AssertionError("must not download a vintage already stored")

    monkeypatch.setattr(ConflictForecastOrgConnector, "_download_csv", staticmethod(_boom))
    assert connector.fetch_forecasts().empty
    assert connector.summary["outcome"] == "no_new_vintage"
    assert connector.summary["listing_issue_date"] == "2026-11-01"


def test_a_failed_download_is_recorded_with_its_error(monkeypatch):
    from resolver.connectors.conflictforecast import ConflictForecastOrgConnector

    connector = ConflictForecastOrgConnector()
    connector.newest_stored = dt.date(2026, 10, 1)
    listing = [{"name": "conflictforecast_ons_armedconf_03_11-2026.csv", "publicUrl": "x"}]
    monkeypatch.setattr(connector, "_get_file_listing", lambda: listing)

    def _tls(url):
        raise OSError("CERTIFICATE_VERIFY_FAILED")

    monkeypatch.setattr(ConflictForecastOrgConnector, "_download_csv", staticmethod(_tls))
    assert connector.fetch_forecasts().empty
    assert connector.summary["outcome"] == "download_failed"
    assert any("CERTIFICATE_VERIFY_FAILED" in e for e in connector.summary["errors"])


# --- 6. overdue known-issue reviews ------------------------------------------


def test_overdue_entries_are_read_from_the_register(tmp_path):
    from resolver.diagnostics.issues import KnownIssues

    path = tmp_path / "known.yml"
    path.write_text(
        "issues:\n"
        "  - id: late\n    review_by: 2026-11-01\n"
        "  - id: fine\n    review_by: 2026-12-15\n"
        "  - id: forever\n    review_by: never\n    no_review_reason: no expiry\n"
        "  - id: undated\n"
    )
    known = KnownIssues.load(path)
    assert known.overdue_entries(dt.date(2026, 11, 24)) == [
        ("late", "2026-11-01"), ("undated", ""),
    ]


def test_the_shipped_register_has_nothing_overdue_today_and_the_moved_dates():
    from resolver.diagnostics.issues import KnownIssues

    known = KnownIssues.load()
    assert str(known.entries["acled_cast_stale_vintage"]["review_by"]) == "2027-01-15"
    assert str(known.entries["views_stale_vintage"]["review_by"]) == "2026-12-15"
    assert "77 of 120" in known.entries["acled_cast_stale_vintage"]["note"]
    assert "December 2026 vintage" in known.entries["views_stale_vintage"]["note"]
    assert known.overdue_entries(dt.date(2026, 11, 24)) == []


def test_an_overdue_review_reaches_the_register_at_info():
    from resolver.diagnostics.issue_sources import issues_from_checks
    from resolver.diagnostics.issues import INFO

    issues = issues_from_checks([{
        "name": "no_known_issue_is_past_its_review_date", "verdict": "FAIL",
        "left": "1 of 5", "right": "0", "detail": "late (review by 2026-11-01)",
        "issues": [{"id": "known_issue_review_overdue_late", "severity": "info",
                    "title": "Known issue late is past its review date (2026-11-01)."}],
    }])
    assert [(i.id, i.severity) for i in issues] == [("known_issue_review_overdue_late", INFO)]
