# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the extracted Resolver DuckDB inspection report generator.

The report logic previously lived as a ~1,400-line heredoc inside
``inspect_resolver_duckdb.yml`` and was untestable; these fixture-based tests
pin the July-2026 fixes: chronological ACAPS "%b%Y" date handling, the
observation/projection freshness split, the retired ipc_phases false alarm,
the informational (not warning) IDMC/IDU note, and the target-list inject
readiness section.
"""

from __future__ import annotations

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from scripts.ci.inspect_resolver_db import build_report


@pytest.fixture()
def fixture_paths(tmp_path):
    db = tmp_path / "fixture.duckdb"
    con = duckdb.connect(str(db))
    con.execute(
        "CREATE TABLE facts_resolved (ym TEXT, iso3 TEXT, hazard_code TEXT, "
        "metric TEXT, value DOUBLE, publisher TEXT)"
    )
    con.execute(
        "INSERT INTO facts_resolved VALUES "
        "('2026-07','SOM','ACE','fatalities',10,'ACLED'),"
        "('2026-06','ETH','DR','phase3plus_in_need',100000,'FEWS NET'),"
        "('2027-02','ETH','DR','phase3plus_projection',120000,'FEWS NET'),"
        "('2026-07','SOM','FL','event_occurrence',1,'GDACS / JRC')"
    )
    con.execute(
        "CREATE TABLE facts_deltas (ym TEXT, iso3 TEXT, hazard_code TEXT, "
        "metric TEXT, value DOUBLE)"
    )
    con.execute(
        "INSERT INTO facts_deltas VALUES ('2026-07','SOM','IDU','new_displacements',500)"
    )
    con.execute(
        "CREATE TABLE acaps_humanitarian_access (iso3 TEXT, crisis_id TEXT, "
        "snapshot_date TEXT, access_score DOUBLE, fetched_at TEXT)"
    )
    # "Sep2025" sorts AFTER "Apr2026"/"Jul2026" alphabetically — the exact
    # shape that produced the nonsense "Apr2026 → Sep2025" range.
    con.execute(
        "INSERT INTO acaps_humanitarian_access VALUES "
        "('SOM','c1','Sep2025',3.0,'2026-07-15'),"
        "('ETH','c2','Apr2026',4.0,'2026-07-15'),"
        "('AFG','c3','Jul2026',3.5,'2026-07-15')"
    )
    con.execute(
        "CREATE TABLE acaps_inform_severity (iso3 TEXT, crisis_id TEXT, "
        "snapshot_date TEXT, severity_score DOUBLE, fetched_at TEXT)"
    )
    con.execute(
        "INSERT INTO acaps_inform_severity VALUES ('SOM','c1','Jul2026',4.1,'2026-07-15')"
    )
    con.execute(
        "CREATE TABLE conflict_forecasts (source TEXT, iso3 TEXT, "
        "forecast_issue_date DATE, metric TEXT, value DOUBLE)"
    )
    con.execute(
        "INSERT INTO conflict_forecasts VALUES "
        "('VIEWS','SOM',DATE '2020-05-01','views_predicted_fatalities',100),"
        "('conflictforecast_org','ETH',CURRENT_DATE,'cf_armed_conflict_risk_3m',0.9)"
    )
    con.execute(
        "CREATE TABLE crisiswatch_entries (iso3 TEXT, year INT, month INT, "
        "arrow TEXT, alert_type TEXT)"
    )
    con.execute(
        "INSERT INTO crisiswatch_entries VALUES "
        "('SOM',2026,6,'unchanged',NULL),('ETH',2026,6,'deteriorated',NULL),"
        "('SOM',2026,2,'unchanged',NULL)"
    )
    con.execute("CREATE TABLE ipc_phases (iso3 TEXT, analysis_date TEXT)")  # dead code
    con.execute("CREATE TABLE reliefweb_reports (iso3 TEXT, published_date TEXT)")
    con.execute(
        "INSERT INTO reliefweb_reports VALUES ('SOM','2026-07-15T07:00:00+00:00')"
    )
    con.execute(
        "CREATE TABLE seasonal_forecasts (iso3 TEXT, variable TEXT, "
        "forecast_issue_date DATE)"
    )
    con.execute(
        "INSERT INTO seasonal_forecasts VALUES "
        "('SOM','prate',DATE '2026-07-08'),('ETH','tmp2m',DATE '2026-07-08')"
    )
    con.execute("CREATE TABLE seasonal_tc_context_cache (iso3 TEXT, context TEXT)")
    con.execute("INSERT INTO seasonal_tc_context_cache VALUES ('SOM','tc ctx')")
    con.close()

    country_list = tmp_path / "hs_list.txt"
    country_list.write_text("Somalia\nEthiopia\nAfghanistan\nGuinea Conakry\n# comment\nNarnia\n")
    countries_csv = tmp_path / "countries.csv"
    countries_csv.write_text(
        "country_name,iso3\nSomalia,SOM\nEthiopia,ETH\nAfghanistan,AFG\nGuinea Conakry,GIN\n"
    )
    return db, country_list, countries_csv


@pytest.fixture()
def report(fixture_paths):
    db, country_list, countries_csv = fixture_paths
    return build_report(db, country_list_path=country_list, countries_csv_path=countries_csv)


def test_missing_db_renders_stub():
    out = build_report(Path("/nonexistent/nope.duckdb"))
    assert "Database not found" in out


def test_acaps_monyyyy_range_is_chronological(report):
    # Pre-fix: lexicographic MIN/MAX rendered "Apr2026 → Sep2025".
    assert "| 3 | 3 | Sep2025 | Jul2026 |" in report
    assert "Apr2026 | Sep2025" not in report


def test_acaps_monyyyy_freshness_parses(report):
    # Pre-fix: "%b%Y" was unparseable → "—" status in the freshness table.
    for line in report.splitlines():
        if line.startswith("| ACAPS humanitarian access"):
            assert "Jul2026" in line and "—" not in line.split("|")[-2]
            assert "✅" in line or "⚠️" in line
            break
    else:
        pytest.fail("ACAPS humanitarian access row missing from freshness table")


def test_freshness_splits_observations_from_projections(report):
    # Newest observation (2026-07) must not be masked by the 2027-02 projection.
    assert "Resolution facts · observations" in report
    assert "Resolution facts · projections (phase3plus_projection)" in report
    obs_line = next(
        l for l in report.splitlines() if l.startswith("| Resolution facts · observations")
    )
    assert "2026-07" in obs_line
    proj_line = next(
        l for l in report.splitlines() if l.startswith("| Resolution facts · projections")
    )
    assert "2027-02" in proj_line and "future-dated" in proj_line


def test_ipc_phases_not_flagged_as_empty_connector(report):
    # The dead-code table must not raise a permanent false alarm.
    assert "EMPTY CONNECTOR TABLES" not in report or "ipc_phases" not in report.split(
        "EMPTY CONNECTOR TABLES"
    )[1].split("\n")[0]
    assert "legacy table from the retired" in report


def test_idmc_idu_is_informational_not_warning(report):
    assert "forecaster matches `IN ('ACE','IDU')`" in report
    assert "Forecaster queries filtering on ACE will miss this data" not in report


def test_stale_conflict_forecast_flagged(report):
    views_line = next(
        l for l in report.splitlines() if l.startswith("| Conflict forecast · VIEWS")
    )
    assert "⚠️" in views_line
    cf_line = next(
        l for l in report.splitlines()
        if l.startswith("| Conflict forecast · conflictforecast_org")
    )
    assert "✅" in cf_line


def test_inject_readiness_section(report):
    assert "## Inject Readiness — 4-Country Target List" in report
    # Unresolvable target entry is surfaced.
    assert "did not resolve to ISO3: Narnia" in report
    # NMME covers SOM+ETH of the 4 targets (search within the readiness
    # section — the freshness table has a similarly-labelled row).
    section = report.split("## Inject Readiness")[1].split("## 1. Table Inventory")[0]
    nmme_line = next(
        l for l in section.splitlines() if l.startswith("| NMME seasonal forecasts")
    )
    assert "2/4" in nmme_line
    # Missing-countries detail names the gaps.
    assert "Missing target countries per source" in report
    assert "**NMME seasonal forecasts** (2): AFG, GIN" in report
    # Absent tables degrade to "no table", not a crash.
    assert "| ACLED monthly fatalities | _no table_" in report


def test_inject_readiness_skipped_without_country_list(fixture_paths, tmp_path):
    db, _, countries_csv = fixture_paths
    out = build_report(
        db,
        country_list_path=tmp_path / "missing.txt",
        countries_csv_path=countries_csv,
    )
    assert "## Inject Readiness" not in out
    # Rest of the report still renders.
    assert "## Data Freshness — At a Glance" in out
