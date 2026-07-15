# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Tests for the Resolver Update run summary (pythia.tools.summarize_all_phases).

The summary is DB-backed on purpose: the JSONL ``counts`` are inconsistent
across connector types, so authoritative row/country counts come from the
just-written resolver DB. These tests pin that behaviour + the problems
section, using a tiny temp DuckDB (no network).
"""

from __future__ import annotations

import pytest

pytest.importorskip("duckdb")

import duckdb

from pythia.tools.summarize_all_phases import build_phase_summary


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "r.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE facts_resolved (iso3 VARCHAR, publisher VARCHAR)")
    con.execute(
        "INSERT INTO facts_resolved VALUES "
        "('SOM','ACLED'),('IRN','ACLED'),('NGA','IFRC')"
    )
    con.execute("CREATE TABLE conflict_forecasts (iso3 VARCHAR, source VARCHAR)")
    con.execute(
        "INSERT INTO conflict_forecasts VALUES ('SOM','VIEWS'),('IRN','VIEWS')"
    )
    # acled_political_events deliberately empty — simulates a silent failure.
    con.execute("CREATE TABLE acled_political_events (iso3 VARCHAR)")
    con.execute("CREATE TABLE crisiswatch_entries (iso3 VARCHAR, year INTEGER, month INTEGER)")
    con.execute("INSERT INTO crisiswatch_entries VALUES ('SOM',2026,6),('IRN',2026,6)")
    con.close()
    return duckdb.connect(str(path), read_only=True)


ENTRIES = [
    {"connector_id": "acled_client", "status": "ok",
     "counts": {"fetched": 2, "normalized": 2, "written": 2}, "duration_ms": 39000},
    {"connector_id": "views_forecasts", "status": "ok",
     "counts": {"fetched": 2292, "written": 2, "empty": 250},
     "extras": {"conflict_rows": 2292, "self_storing": True}, "duration_ms": 8000},
    {"connector_id": "acled_political_events", "status": "ok",
     "counts": {"fetched": 0, "written": 0, "empty": 252}, "duration_ms": 60000},
    {"connector_id": "seasonal_tc", "status": "error",
     "reason": "fetch-or-store-failed", "counts": {"written": 0}, "duration_ms": 0},
    {"connector_id": "idmc", "status": "skipped",
     "reason": "disabled via RESOLVER_SKIP_IDMC", "counts": {}, "duration_ms": 0},
]


def test_problems_section_surfaces_error_skip_and_silent_empty(db):
    out = build_phase_summary(ENTRIES, con=db)
    assert "## Problems & Warnings" in out
    # Hard error is flagged with its reason.
    assert "❌ **seasonal_tc**" in out and "fetch-or-store-failed" in out
    # Skip is flagged.
    assert "⏭️ **idmc**" in out
    # "OK but table empty" is caught as a silent failure.
    assert "⚠️ **acled_political_events**" in out and "empty" in out


def test_counts_come_from_db_not_misleading_jsonl(db):
    out = build_phase_summary(ENTRIES, con=db)
    # VIEWS: JSONL written=2 (nominal) must NOT be shown as the row count —
    # the DB has 2 conflict_forecasts rows, and Δ this run = 2292 (extras).
    assert "2,292" in out  # Δ this run from extras.conflict_rows
    # facts_resolved headline (3 rows, 3 countries) is present.
    assert "`facts_resolved`" in out
    # CrisisWatch latest edition surfaced.
    assert "2026-06" in out


def test_verdict_reflects_error_and_empty(db):
    out = build_phase_summary(ENTRIES, con=db)
    # 1 error + 1 skip + 1 silent-empty → not all-OK.
    assert "Result: ❌" in out
    assert "1 error" in out and "1 skipped" in out and "1 wrote 0 rows" in out


def test_graceful_without_db():
    # No DB → no false "empty" flags, still renders status/problems.
    out = build_phase_summary(ENTRIES, con=None)
    assert "## Problems & Warnings" in out
    assert "❌ **seasonal_tc**" in out
    # acled_political must NOT be flagged empty when DB is unavailable.
    assert "⚠️ **acled_political_events**" not in out


def test_empty_entries():
    out = build_phase_summary([], con=None)
    assert "No connector diagnostics" in out


# ---------------------------------------------------------------------------
# Cross-cutting checks (July 2026): vintage staleness, list shrink, drops
# ---------------------------------------------------------------------------


@pytest.fixture()
def db_with_vintages(tmp_path):
    path = tmp_path / "v.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE facts_resolved (iso3 VARCHAR, publisher VARCHAR)")
    con.execute(
        "CREATE TABLE conflict_forecasts (iso3 VARCHAR, source VARCHAR, forecast_issue_date DATE)"
    )
    con.execute(
        "INSERT INTO conflict_forecasts VALUES "
        "('SOM','VIEWS', CURRENT_DATE - INTERVAL 10 DAY),"
        "('SOM','ACLED_CAST', CURRENT_DATE - INTERVAL 200 DAY)"
    )
    con.close()
    return duckdb.connect(str(path), read_only=True)


OK_ENTRIES = [
    {"connector_id": "acled_client", "status": "ok",
     "counts": {"fetched": 2, "normalized": 2, "written": 2}, "duration_ms": 1000},
]


def test_stale_conflict_forecast_vintage_flagged(db_with_vintages):
    out = build_phase_summary(OK_ENTRIES, con=db_with_vintages)
    assert "## Conflict Forecast Vintages" in out
    assert "⚠️ **ACLED_CAST** conflict forecast vintage is" in out
    # Fresh VIEWS vintage is listed but not warned about.
    assert "⚠️ **VIEWS** conflict forecast vintage" not in out
    # Warnings (no hard errors) → ⚠️ verdict.
    assert "Result: ⚠️" in out
    assert "stale forecast vintage" in out


def test_attribution_drops_flagged():
    entries = OK_ENTRIES + [
        {"connector_id": "acled_political_events", "status": "ok",
         "counts": {"fetched": 141, "written": 141, "empty": 111},
         "extras": {"attribution_dropped_countries": ["GIN", "COG"],
                    "attribution_dropped_events": 27},
         "duration_ms": 48000},
    ]
    out = build_phase_summary(entries, con=None)
    assert "attributed NONE" in out
    assert "COG, GIN" in out
    assert "27" in out


def test_ingest_windows_line(monkeypatch):
    monkeypatch.setenv("FEWSNET_MONTHS", "3")
    monkeypatch.delenv("IPC_API_MONTHS", raising=False)
    monkeypatch.delenv("GDACS_MONTHS", raising=False)
    out = build_phase_summary(OK_ENTRIES, con=None)
    assert "FEWSNET_MONTHS=3" in out
    assert "IPC_API_MONTHS=24 (default)" in out


def test_country_list_shrink_detected(tmp_path):
    import json as _json
    import subprocess

    from pythia.tools.summarize_all_phases import _country_list_shrink

    root = tmp_path / "repo"
    (root / "resolver" / "data").mkdir(parents=True)
    f = root / "resolver" / "data" / "fewsnet_countries.json"
    f.write_text(_json.dumps([f"C{i:02d}" for i in range(48)]))
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=root, check=True,
    )
    # Simulate the short-window overwrite.
    f.write_text(_json.dumps([f"C{i:02d}" for i in range(27)]))

    res = _country_list_shrink(repo_root=root)
    assert res == {"on_disk": 27, "in_git": 48, "shrunk": True}

    # Restored file → not shrunk.
    f.write_text(_json.dumps([f"C{i:02d}" for i in range(48)]))
    res = _country_list_shrink(repo_root=root)
    assert res["shrunk"] is False
