# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""DR Phase 3+ questions only where the source still publishes (12 Oct 2026).

The newest Current Situation row must be no more than
PHASE3_MAX_STALENESS_MONTHS before the window's first month; the IPC
connector lets a FEWS NET country through once its FEWS NET rows are stale;
resolution takes FEWS NET first where both cover a month.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.db.schema import ensure_schema
from pythia.tools import phase3_eligibility as p3

_DDL = """
CREATE TABLE IF NOT EXISTS facts_resolved (
    ym TEXT NOT NULL, iso3 TEXT NOT NULL, hazard_code TEXT NOT NULL,
    metric TEXT NOT NULL, series_semantics TEXT NOT NULL DEFAULT '',
    value DOUBLE, publisher TEXT, created_at TIMESTAMP DEFAULT now()
)
"""


def _facts(con, rows):
    con.execute(_DDL)
    for ym, iso3, pub, metric in rows:
        con.execute(
            "INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, publisher) "
            "VALUES (?, ?, 'DR', ?, 100000, ?)",
            [ym, iso3, metric, pub],
        )


# --- the rule ---------------------------------------------------------------

def test_months_between_and_window_start():
    assert p3.months_between("2026-11", "2026-01") == 10
    assert p3.months_between("2026-11", "2026-02") == 9
    assert p3.window_start_for(date(2026, 10, 13)) == "2026-11"
    assert p3.window_start_for(date(2026, 12, 13)) == "2027-01"


def test_env_override(monkeypatch):
    monkeypatch.delenv(p3.ENV_MAX_STALENESS, raising=False)
    assert p3.max_staleness_months() == 9
    monkeypatch.setenv(p3.ENV_MAX_STALENESS, "12")
    assert p3.max_staleness_months() == 12
    monkeypatch.setenv(p3.ENV_MAX_STALENESS, "abc")
    assert p3.max_staleness_months() == 9


def test_decide_drops_stale_and_keeps_fresh(monkeypatch):
    monkeypatch.delenv(p3.ENV_MAX_STALENESS, raising=False)
    con = duckdb.connect()
    # The 12 Oct 2026 shape: ETH FEWS NET newest 2026-01, NAM IPC 2025-09,
    # SOM FEWS NET 2026-09, ZMB IPC stale but FEWS NET fresh.
    _facts(con, [
        ("2026-01", "ETH", "FEWS NET", "phase3plus_in_need"),
        ("2026-09", "ETH", "FEWS NET", "phase3plus_projection"),  # projections never count
        ("2025-09", "NAM", "IPC", "phase3plus_in_need"),
        ("2026-09", "SOM", "FEWS NET", "phase3plus_in_need"),
        ("2025-09", "ZMB", "IPC", "phase3plus_in_need"),
        ("2026-09", "ZMB", "FEWS NET", "phase3plus_in_need"),
        ("2026-02", "KEN", "FEWS NET", "phase3plus_in_need"),  # exactly 9 months: kept
    ])
    d = p3.decide(con, window_start="2026-11")
    assert d.used_rule
    assert not d.admits("ETH")
    assert not d.admits("NAM")
    assert d.admits("SOM")
    assert d.admits("ZMB")
    assert d.admits("KEN")
    assert not d.admits("ERI")  # no row at all
    assert d.stale["ETH"] == ("2026-01", "FEWS NET", 10)
    assert "2026-01" in d.why_not("ETH") and "FEWS NET" in d.why_not("ETH")
    # A wider bound keeps ETH.
    assert p3.decide(con, window_start="2026-11", max_staleness=12).admits("ETH")


def test_no_rows_is_a_guard_not_a_drop():
    con = duckdb.connect()
    con.execute(_DDL)
    d = p3.decide(con, window_start="2026-11")
    assert d.source == p3.SOURCE_NO_ROWS and d.admits("ETH")
    d2 = p3.decide(duckdb.connect(), window_start="2026-11")  # no table at all
    assert d2.source == p3.SOURCE_UNREADABLE and d2.admits("ETH")


def test_table_without_publisher_column_still_decides():
    con = duckdb.connect()
    con.execute("CREATE TABLE facts_resolved (ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT, value DOUBLE)")
    con.execute("INSERT INTO facts_resolved VALUES ('2024-11', 'NIC', 'DR', 'phase3plus_in_need', 1)")
    d = p3.decide(con, window_start="2026-11")
    assert d.used_rule and not d.admits("NIC")


# --- question creation -----------------------------------------------------

def _seed_triage(db_path: Path, rows, facts):
    con = duckdb.connect(str(db_path))
    try:
        ensure_schema(con)
        con.execute("DROP TABLE IF EXISTS facts_resolved")
        _facts(con, facts)
        for iso3 in rows:
            con.execute(
                """
                INSERT INTO hs_triage (
                    run_id, iso3, hazard_code, tier, triage_score, need_full_spd,
                    drivers_json, regime_shifts_json, data_quality_json, scenario_stub
                ) VALUES ('hs_run_p3', ?, 'DR', 'priority', 0.7, TRUE, '[]', '[]', '{}', '')
                """,
                [iso3],
            )
    finally:
        con.close()


def test_question_creation_drops_stale_country_and_keeps_fresh(monkeypatch, tmp_path):
    import scripts.create_questions_from_triage as cq

    class _Oct(date):
        @classmethod
        def today(cls):
            return date(2026, 10, 13)

    monkeypatch.setattr(cq, "date", _Oct)
    monkeypatch.delenv(p3.ENV_MAX_STALENESS, raising=False)
    cq._FOOD_SECURITY_DB_CACHE.clear()
    db_path = tmp_path / "p3.duckdb"
    _seed_triage(db_path, ["ETH", "SOM"], [
        ("2026-01", "ETH", "FEWS NET", "phase3plus_in_need"),
        ("2026-09", "SOM", "FEWS NET", "phase3plus_in_need"),
    ])
    cq.create_questions_from_triage(f"duckdb:///{db_path}", hs_run_id="hs_run_p3")
    cq._FOOD_SECURITY_DB_CACHE.clear()

    con = duckdb.connect(str(db_path))
    try:
        got = {(r[0], r[1]) for r in con.execute("SELECT iso3, metric FROM questions").fetchall()}
        stored = p3.latest_decision(con, "hs_run_p3")
    finally:
        con.close()
    assert ("SOM", "PHASE3PLUS_IN_NEED") in got
    assert ("ETH", "PHASE3PLUS_IN_NEED") not in got
    # DR/EVENT_OCCURRENCE is unaffected.
    assert ("ETH", "EVENT_OCCURRENCE") in got
    assert stored["window_start"] == "2026-11"
    assert stored["asked"] == ["SOM"]
    assert list(stored["dropped"]) == ["ETH"]
    assert stored["stale"]["ETH"]["newest_month"] == "2026-01"


# --- the IPC connector's FEWS NET exclusion ---------------------------------

def test_fewsnet_exclusions_let_stale_country_through():
    con = duckdb.connect()
    _facts(con, [
        ("2024-11", "MRT", "FEWS NET", "phase3plus_in_need"),
        ("2026-09", "SOM", "FEWS NET", "phase3plus_in_need"),
        ("2026-09", "MRT", "IPC", "phase3plus_in_need"),  # IPC rows do not keep MRT excluded
    ])
    excluded, let_through = p3.fewsnet_exclusions(
        con, {"MRT", "SOM", "ERI"}, reference_ym="2026-11")
    assert excluded == {"SOM"}
    assert set(let_through) == {"MRT", "ERI"}
    assert "2024-11" in let_through["MRT"]


def test_fewsnet_exclusions_fall_back_without_fewsnet_rows():
    con = duckdb.connect()
    _facts(con, [("2026-09", "NAM", "IPC", "phase3plus_in_need")])
    assert p3.fewsnet_exclusions(con, {"SOM"}, reference_ym="2026-11") is None
    assert p3.fewsnet_exclusions(duckdb.connect(), {"SOM"}, reference_ym="2026-11") is None


def test_ipc_connector_exclusion_reads_the_db(monkeypatch, tmp_path):
    from resolver.connectors import ipc_api

    db_path = tmp_path / "res.duckdb"
    con = duckdb.connect(str(db_path))
    _facts(con, [
        ("2024-11", "MRT", "FEWS NET", "phase3plus_in_need"),
        ("2026-09", "SOM", "FEWS NET", "phase3plus_in_need"),
    ])
    con.close()
    monkeypatch.setattr(ipc_api, "_load_fewsnet_country_list", lambda: {"MRT", "SOM"})
    monkeypatch.setenv("RESOLVER_DB_URL", f"duckdb:///{db_path}")
    try:
        assert ipc_api._fewsnet_exclusion_set(date(2026, 10, 13)) == {"SOM"}
    finally:
        from resolver.db import duckdb_io

        duckdb_io.close_db(duckdb_io.get_db(f"duckdb:///{db_path}"))
    monkeypatch.delenv("RESOLVER_DB_URL")
    assert ipc_api._fewsnet_exclusion_set(date(2026, 10, 13)) == {"MRT", "SOM"}


# --- resolution precedence ---------------------------------------------------

def test_resolution_takes_fewsnet_first_where_both_cover_a_month():
    from pythia.tools.compute_resolutions import _try_phase3plus

    con = duckdb.connect()
    con.execute(_DDL)
    con.execute("INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, publisher, created_at) "
                "VALUES ('2026-07', 'UGA', 'DR', 'phase3plus_in_need', 500000, 'FEWS NET', '2026-08-01')")
    con.execute("INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, publisher, created_at) "
                "VALUES ('2026-07', 'UGA', 'DR', 'phase3plus_in_need', 900000, 'IPC', '2026-09-01')")
    value, _ts, desc = _try_phase3plus(con, "UGA", "DR", "2026-07")
    assert value == 500000 and "FEWS NET" in desc


# --- the debug bundle's health line ------------------------------------------

def test_debug_bundle_reports_the_decision():
    pytest.importorskip("pandas")
    from scripts.dump_pythia_debug_bundle import _phase3_eligibility_check

    con = duckdb.connect()
    _facts(con, [("2026-01", "ETH", "FEWS NET", "phase3plus_in_need"),
                 ("2026-09", "SOM", "FEWS NET", "phase3plus_in_need")])
    d = p3.decide(con, window_start="2026-11", max_staleness=9)
    p3.record(con, d, hs_run_id="hs_x", asked=["SOM"], dropped=["ETH"], is_test=False)
    check = _phase3_eligibility_check(p3.latest_decision(con, "hs_x"))
    assert check["status"] == "OK" and "ETH" in check["detail"] and "dropped 1" in check["detail"]
    assert _phase3_eligibility_check({"absent": True})["status"] == "WARN"
