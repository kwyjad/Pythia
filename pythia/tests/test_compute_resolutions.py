# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.tools.compute_resolutions import (
    _resolve_value,
    _try_facts_deltas,
    _try_facts_resolved,
    _try_emdat_pa,
    _try_acled_fatalities,
    _try_gdacs_binary,
    _try_fewsnet_ipc,
    _data_freshness_cutoff,
    _calendar_cutoff,
    _purge_stale_resolutions,
    _should_default_to_zero,
    horizon_to_calendar_month,
    compute_resolutions,
)


# ---------------------------------------------------------------------------
# _resolve_value against the real facts_resolved schema
# ---------------------------------------------------------------------------


@pytest.fixture
def resolver_db(tmp_path: Path):
    """Create a DuckDB with the resolver facts_resolved schema."""
    db_path = tmp_path / "resolver.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE facts_resolved (
            ym TEXT NOT NULL,
            iso3 TEXT NOT NULL,
            hazard_code TEXT NOT NULL,
            hazard_label TEXT,
            hazard_class TEXT,
            metric TEXT NOT NULL,
            series_semantics TEXT NOT NULL DEFAULT '',
            value DOUBLE,
            unit TEXT,
            as_of DATE,
            as_of_date VARCHAR,
            publication_date VARCHAR,
            publisher TEXT,
            source_id TEXT,
            source_type TEXT,
            source_url TEXT,
            doc_title TEXT,
            definition_text TEXT,
            precedence_tier TEXT,
            event_id TEXT,
            proxy_for TEXT,
            confidence TEXT,
            provenance_source TEXT,
            provenance_rank INTEGER,
            series TEXT,
            alertlevel TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT facts_resolved_unique
                UNIQUE (ym, iso3, hazard_code, metric, series_semantics)
        )
        """
    )
    yield con
    con.close()


class TestResolveValue:
    def test_pa_found(self, resolver_db):
        resolver_db.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, created_at)
            VALUES ('2025-01', 'MLI', 'FL', 'affected', 12345.0, '2025-02-10 08:00:00')
            """
        )
        result = _resolve_value(resolver_db, "MLI", "FL", "2025-01", "PA")
        assert result is not None
        value, source_ts = result
        assert value == 12345.0
        assert source_ts is not None

    def test_fatalities_found(self, resolver_db):
        resolver_db.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, created_at)
            VALUES ('2025-01', 'SOM', 'ACE', 'fatalities', 200.0, '2025-02-10 08:00:00')
            """
        )
        result = _resolve_value(resolver_db, "SOM", "ACE", "2025-01", "FATALITIES")
        assert result is not None
        value, _ = result
        assert value == 200.0

    def test_no_match_returns_none(self, resolver_db):
        result = _resolve_value(resolver_db, "ZZZ", "XX", "2099-01", "PA")
        assert result is None

    def test_unknown_metric_returns_none(self, resolver_db):
        result = _resolve_value(resolver_db, "MLI", "FL", "2025-01", "UNKNOWN")
        assert result is None

    def test_people_affected_alias(self, resolver_db):
        resolver_db.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, created_at)
            VALUES ('2025-01', 'BGD', 'FL', 'people_affected', 50000.0, '2025-02-10 08:00:00')
            """
        )
        result = _resolve_value(resolver_db, "BGD", "FL", "2025-01", "PA")
        assert result is not None
        assert result[0] == 50000.0

    def test_created_at_ordering(self, resolver_db):
        """When multiple rows match via different series_semantics, the most
        recently created row should be returned."""
        resolver_db.execute(
            """
            INSERT INTO facts_resolved
                (ym, iso3, hazard_code, metric, series_semantics, value, created_at)
            VALUES
                ('2025-01', 'ETH', 'ACE', 'fatalities', 'old', 100.0, '2025-01-01 00:00:00'),
                ('2025-01', 'ETH', 'ACE', 'fatalities', 'new', 200.0, '2025-02-01 00:00:00')
            """
        )
        result = _resolve_value(resolver_db, "ETH", "ACE", "2025-01", "FATALITIES")
        assert result is not None
        assert result[0] == 200.0

    def test_null_created_at(self, resolver_db):
        """created_at may be NULL in edge cases; the function should still work."""
        resolver_db.execute(
            """
            INSERT INTO facts_resolved
                (ym, iso3, hazard_code, metric, value, created_at)
            VALUES ('2025-06', 'NGA', 'FL', 'pa', 7777.0, CURRENT_TIMESTAMP)
            """
        )
        result = _resolve_value(resolver_db, "NGA", "FL", "2025-06", "PA")
        assert result is not None
        assert result[0] == 7777.0

    def test_event_occurrence_found(self, resolver_db):
        """EVENT_OCCURRENCE metric dispatches to _try_gdacs_binary."""
        resolver_db.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, created_at)
            VALUES ('2025-01', 'BGD', 'FL', 'event_occurrence', 1.0, '2025-02-10 08:00:00')
            """
        )
        result = _resolve_value(resolver_db, "BGD", "FL", "2025-01", "EVENT_OCCURRENCE")
        assert result is not None
        assert result[0] == 1.0

    def test_event_occurrence_no_match_returns_none(self, resolver_db):
        result = _resolve_value(resolver_db, "ZZZ", "FL", "2099-01", "EVENT_OCCURRENCE")
        assert result is None

    def test_phase3plus_in_need_found(self, resolver_db):
        """PHASE3PLUS_IN_NEED metric dispatches to _try_fewsnet_ipc."""
        resolver_db.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, created_at)
            VALUES ('2025-06', 'ETH', 'DR', 'phase3plus_in_need', 17000000.0, '2025-07-01 08:00:00')
            """
        )
        result = _resolve_value(resolver_db, "ETH", "DR", "2025-06", "PHASE3PLUS_IN_NEED")
        assert result is not None
        assert result[0] == 17000000.0

    def test_phase3plus_in_need_wrong_hazard(self, resolver_db):
        """FEWS NET IPC only applies to DR hazard."""
        resolver_db.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, created_at)
            VALUES ('2025-06', 'ETH', 'FL', 'phase3plus_in_need', 100.0, '2025-07-01 08:00:00')
            """
        )
        result = _resolve_value(resolver_db, "ETH", "FL", "2025-06", "PHASE3PLUS_IN_NEED")
        assert result is None


# ---------------------------------------------------------------------------
# _try_facts_deltas — IDMC new_displacements / flow data
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_table_db(tmp_path: Path):
    """DuckDB with facts_resolved, facts_deltas, emdat_pa, and
    acled_monthly_fatalities tables for multi-table resolution tests."""
    db_path = tmp_path / "multi.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE facts_resolved (
            ym TEXT NOT NULL,
            iso3 TEXT NOT NULL,
            hazard_code TEXT NOT NULL,
            hazard_label TEXT,
            hazard_class TEXT,
            metric TEXT NOT NULL,
            series_semantics TEXT NOT NULL DEFAULT '',
            value DOUBLE,
            unit TEXT,
            as_of DATE,
            as_of_date VARCHAR,
            publication_date VARCHAR,
            publisher TEXT,
            source_id TEXT,
            source_type TEXT,
            source_url TEXT,
            doc_title TEXT,
            definition_text TEXT,
            precedence_tier TEXT,
            event_id TEXT,
            proxy_for TEXT,
            confidence TEXT,
            provenance_source TEXT,
            provenance_rank INTEGER,
            series TEXT,
            alertlevel TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT facts_resolved_unique
                UNIQUE (ym, iso3, hazard_code, metric, series_semantics)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE facts_deltas (
            ym TEXT NOT NULL,
            iso3 TEXT NOT NULL,
            hazard_code TEXT NOT NULL,
            metric TEXT NOT NULL,
            value_new DOUBLE,
            value_stock DOUBLE,
            series_semantics TEXT NOT NULL DEFAULT 'new',
            as_of VARCHAR,
            source_id TEXT,
            series TEXT,
            rebase_flag INTEGER,
            first_observation INTEGER,
            delta_negative_clamped INTEGER,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT facts_deltas_unique UNIQUE (ym, iso3, hazard_code, metric)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE emdat_pa (
            iso3 TEXT,
            ym TEXT,
            shock_type TEXT,
            pa BIGINT,
            as_of_date TEXT,
            source_id TEXT
        )
        """
    )
    con.execute(
        """
        CREATE TABLE acled_monthly_fatalities (
            iso3 TEXT,
            month DATE,
            fatalities BIGINT,
            source TEXT,
            updated_at TIMESTAMP
        )
        """
    )
    yield con
    con.close()


class TestTryFactsDeltas:
    def test_idmc_new_displacements(self, multi_table_db):
        multi_table_db.execute(
            """
            INSERT INTO facts_deltas (ym, iso3, hazard_code, metric, value_new, created_at)
            VALUES ('2024-08', 'MLI', 'ACE', 'new_displacements', 1500.0, '2024-09-10 00:00:00')
            """
        )
        result = _try_facts_deltas(multi_table_db, "MLI", "ACE", "2024-08", "PA")
        assert result is not None
        assert result[0] == 1500.0

    def test_deltas_fatalities(self, multi_table_db):
        multi_table_db.execute(
            """
            INSERT INTO facts_deltas (ym, iso3, hazard_code, metric, value_new, created_at)
            VALUES ('2024-09', 'SOM', 'ACE', 'fatalities', 88.0, '2024-10-10 00:00:00')
            """
        )
        result = _try_facts_deltas(multi_table_db, "SOM", "ACE", "2024-09", "FATALITIES")
        assert result is not None
        assert result[0] == 88.0

    def test_value_stock_fallback(self, multi_table_db):
        """When value_new is NULL, should fall back to value_stock."""
        multi_table_db.execute(
            """
            INSERT INTO facts_deltas (ym, iso3, hazard_code, metric, value_new, value_stock, created_at)
            VALUES ('2024-10', 'ETH', 'FL', 'affected', NULL, 9999.0, '2024-11-10 00:00:00')
            """
        )
        result = _try_facts_deltas(multi_table_db, "ETH", "FL", "2024-10", "PA")
        assert result is not None
        assert result[0] == 9999.0

    def test_no_match(self, multi_table_db):
        result = _try_facts_deltas(multi_table_db, "ZZZ", "XX", "2099-01", "PA")
        assert result is None

    def test_unknown_metric(self, multi_table_db):
        result = _try_facts_deltas(multi_table_db, "MLI", "FL", "2024-08", "UNKNOWN")
        assert result is None


class TestTryEmdatPa:
    def test_flood_lookup(self, multi_table_db):
        multi_table_db.execute(
            """
            INSERT INTO emdat_pa (iso3, ym, shock_type, pa, as_of_date)
            VALUES ('BGD', '2024-07', 'flood', 250000, '2024-08-15')
            """
        )
        result = _try_emdat_pa(multi_table_db, "BGD", "FL", "2024-07")
        assert result is not None
        assert result[0] == 250000.0
        assert result[1] == "2024-08-15"

    def test_drought_lookup(self, multi_table_db):
        multi_table_db.execute(
            """
            INSERT INTO emdat_pa (iso3, ym, shock_type, pa, as_of_date)
            VALUES ('ETH', '2024-06', 'drought', 100000, '2024-07-20')
            """
        )
        result = _try_emdat_pa(multi_table_db, "ETH", "DR", "2024-06")
        assert result is not None
        assert result[0] == 100000.0

    def test_tropical_cyclone_lookup(self, multi_table_db):
        multi_table_db.execute(
            """
            INSERT INTO emdat_pa (iso3, ym, shock_type, pa, as_of_date)
            VALUES ('PHL', '2024-11', 'tropical_cyclone', 500000, '2024-12-01')
            """
        )
        result = _try_emdat_pa(multi_table_db, "PHL", "TC", "2024-11")
        assert result is not None
        assert result[0] == 500000.0

    def test_unmapped_hazard_returns_none(self, multi_table_db):
        """ACE (armed conflict) has no EM-DAT shock_type mapping."""
        multi_table_db.execute(
            """
            INSERT INTO emdat_pa (iso3, ym, shock_type, pa, as_of_date)
            VALUES ('SOM', '2024-07', 'flood', 1000, '2024-08-01')
            """
        )
        result = _try_emdat_pa(multi_table_db, "SOM", "ACE", "2024-07")
        assert result is None

    def test_no_match(self, multi_table_db):
        result = _try_emdat_pa(multi_table_db, "ZZZ", "FL", "2099-01")
        assert result is None


class TestTryAcledFatalities:
    def test_fatalities_lookup(self, multi_table_db):
        multi_table_db.execute(
            """
            INSERT INTO acled_monthly_fatalities (iso3, month, fatalities, updated_at)
            VALUES ('SOM', '2024-08-01', 312, '2024-09-15 00:00:00')
            """
        )
        result = _try_acled_fatalities(multi_table_db, "SOM", "2024-08")
        assert result is not None
        assert result[0] == 312.0
        assert result[1] is not None

    def test_no_match(self, multi_table_db):
        result = _try_acled_fatalities(multi_table_db, "ZZZ", "2099-01")
        assert result is None

    def test_null_fatalities_returns_none(self, multi_table_db):
        multi_table_db.execute(
            """
            INSERT INTO acled_monthly_fatalities (iso3, month, fatalities, updated_at)
            VALUES ('NGA', '2024-10-01', NULL, '2024-11-01 00:00:00')
            """
        )
        result = _try_acled_fatalities(multi_table_db, "NGA", "2024-10")
        assert result is None


# ---------------------------------------------------------------------------
# _try_gdacs_binary
# ---------------------------------------------------------------------------


class TestTryGdacsBinary:
    def test_event_found(self, resolver_db):
        resolver_db.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, created_at)
            VALUES ('2025-01', 'PHL', 'TC', 'event_occurrence', 1.0, '2025-02-01 00:00:00')
            """
        )
        result = _try_gdacs_binary(resolver_db, "PHL", "TC", "2025-01")
        assert result is not None
        assert result[0] == 1.0

    def test_no_event(self, resolver_db):
        resolver_db.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, created_at)
            VALUES ('2025-01', 'PHL', 'TC', 'event_occurrence', 0.0, '2025-02-01 00:00:00')
            """
        )
        result = _try_gdacs_binary(resolver_db, "PHL", "TC", "2025-01")
        assert result is not None
        assert result[0] == 0.0

    def test_no_match(self, resolver_db):
        result = _try_gdacs_binary(resolver_db, "ZZZ", "TC", "2099-01")
        assert result is None


# ---------------------------------------------------------------------------
# _try_fewsnet_ipc
# ---------------------------------------------------------------------------


class TestTryFewsnetIpc:
    def test_dr_found(self, resolver_db):
        resolver_db.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, created_at)
            VALUES ('2025-06', 'ETH', 'DR', 'phase3plus_in_need', 17000000.0, '2025-07-01 00:00:00')
            """
        )
        result = _try_fewsnet_ipc(resolver_db, "ETH", "DR", "2025-06")
        assert result is not None
        assert result[0] == 17000000.0

    def test_non_dr_returns_none(self, resolver_db):
        """FEWS NET IPC only applies to DR hazard."""
        result = _try_fewsnet_ipc(resolver_db, "ETH", "FL", "2025-06")
        assert result is None

    def test_no_match(self, resolver_db):
        result = _try_fewsnet_ipc(resolver_db, "ZZZ", "DR", "2099-01")
        assert result is None


# ---------------------------------------------------------------------------
# _should_default_to_zero
# ---------------------------------------------------------------------------


class TestShouldDefaultToZero:
    def test_fatalities_ace_is_zero(self):
        assert _should_default_to_zero("FATALITIES", "ACE") is True

    def test_fatalities_aco_is_zero(self):
        assert _should_default_to_zero("FATALITIES", "ACO") is True

    def test_fatalities_fl_not_zero(self):
        """FL fatalities are not from ACLED — no continuous coverage."""
        assert _should_default_to_zero("FATALITIES", "FL") is False

    def test_event_occurrence_fl_is_zero(self):
        assert _should_default_to_zero("EVENT_OCCURRENCE", "FL") is True

    def test_event_occurrence_dr_is_zero(self):
        assert _should_default_to_zero("EVENT_OCCURRENCE", "DR") is True

    def test_pa_any_hazard_not_zero(self):
        """PA metric should never default to zero."""
        assert _should_default_to_zero("PA", "FL") is False
        assert _should_default_to_zero("PA", "DR") is False
        assert _should_default_to_zero("PA", "ACE") is False

    def test_phase3plus_not_zero(self):
        assert _should_default_to_zero("PHASE3PLUS_IN_NEED", "DR") is False


# ---------------------------------------------------------------------------
# _resolve_value multi-table priority cascade
# ---------------------------------------------------------------------------


class TestResolveValueMultiTable:
    def test_facts_resolved_takes_priority(self, multi_table_db):
        """facts_resolved should be preferred over facts_deltas for the same
        (iso3, hazard_code, month)."""
        multi_table_db.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, created_at)
            VALUES ('2024-08', 'MLI', 'FL', 'affected', 5000.0, '2024-09-01 00:00:00')
            """
        )
        multi_table_db.execute(
            """
            INSERT INTO facts_deltas (ym, iso3, hazard_code, metric, value_new, created_at)
            VALUES ('2024-08', 'MLI', 'FL', 'affected', 3000.0, '2024-09-02 00:00:00')
            """
        )
        result = _resolve_value(multi_table_db, "MLI", "FL", "2024-08", "PA")
        assert result is not None
        # Should pick facts_resolved value (5000), not facts_deltas (3000)
        assert result[0] == 5000.0

    def test_falls_through_to_facts_deltas(self, multi_table_db):
        """When facts_resolved has no match, should find data in facts_deltas."""
        multi_table_db.execute(
            """
            INSERT INTO facts_deltas (ym, iso3, hazard_code, metric, value_new, created_at)
            VALUES ('2024-09', 'ETH', 'ACE', 'new_displacements', 2000.0, '2024-10-10 00:00:00')
            """
        )
        result = _resolve_value(multi_table_db, "ETH", "ACE", "2024-09", "PA")
        assert result is not None
        assert result[0] == 2000.0

    def test_falls_through_to_emdat_pa(self, multi_table_db):
        """When both facts tables have no match, should find PA in emdat_pa."""
        multi_table_db.execute(
            """
            INSERT INTO emdat_pa (iso3, ym, shock_type, pa, as_of_date)
            VALUES ('BGD', '2024-07', 'flood', 250000, '2024-08-15')
            """
        )
        result = _resolve_value(multi_table_db, "BGD", "FL", "2024-07", "PA")
        assert result is not None
        assert result[0] == 250000.0

    def test_falls_through_to_acled_fatalities(self, multi_table_db):
        """When facts tables have no match, should find FATALITIES in acled table."""
        multi_table_db.execute(
            """
            INSERT INTO acled_monthly_fatalities (iso3, month, fatalities, updated_at)
            VALUES ('SOM', '2024-08-01', 312, '2024-09-15 00:00:00')
            """
        )
        result = _resolve_value(multi_table_db, "SOM", "ACE", "2024-08", "FATALITIES")
        assert result is not None
        assert result[0] == 312.0

    def test_emdat_not_checked_for_fatalities(self, multi_table_db):
        """emdat_pa should not be consulted for FATALITIES metric."""
        multi_table_db.execute(
            """
            INSERT INTO emdat_pa (iso3, ym, shock_type, pa, as_of_date)
            VALUES ('SOM', '2024-08', 'flood', 1000, '2024-09-01')
            """
        )
        result = _resolve_value(multi_table_db, "SOM", "FL", "2024-08", "FATALITIES")
        assert result is None

    def test_acled_not_checked_for_pa(self, multi_table_db):
        """acled_monthly_fatalities should not be consulted for PA metric."""
        multi_table_db.execute(
            """
            INSERT INTO acled_monthly_fatalities (iso3, month, fatalities, updated_at)
            VALUES ('SOM', '2024-08-01', 312, '2024-09-15 00:00:00')
            """
        )
        result = _resolve_value(multi_table_db, "SOM", "ACE", "2024-08", "PA")
        assert result is None


# ---------------------------------------------------------------------------
# horizon_to_calendar_month
# ---------------------------------------------------------------------------


class TestHorizonToCalendarMonth:
    def test_h1(self):
        assert horizon_to_calendar_month(date(2024, 8, 1), 1) == "2024-08"

    def test_h6(self):
        assert horizon_to_calendar_month(date(2024, 8, 1), 6) == "2025-01"

    def test_year_rollover(self):
        assert horizon_to_calendar_month(date(2024, 11, 1), 3) == "2025-01"


# ---------------------------------------------------------------------------
# End-to-end: compute_resolutions with a real facts_resolved table
# ---------------------------------------------------------------------------


def _setup_e2e_db(con, extra_tables=True):
    """Create the minimal Pythia + Resolver schema for E2E resolution tests."""
    con.execute(
        """
        CREATE TABLE facts_resolved (
            ym TEXT NOT NULL,
            iso3 TEXT NOT NULL,
            hazard_code TEXT NOT NULL,
            hazard_label TEXT,
            hazard_class TEXT,
            metric TEXT NOT NULL,
            series_semantics TEXT NOT NULL DEFAULT '',
            value DOUBLE,
            unit TEXT,
            as_of DATE,
            as_of_date VARCHAR,
            publication_date VARCHAR,
            publisher TEXT,
            source_id TEXT,
            source_type TEXT,
            source_url TEXT,
            doc_title TEXT,
            definition_text TEXT,
            precedence_tier TEXT,
            event_id TEXT,
            proxy_for TEXT,
            confidence TEXT,
            provenance_source TEXT,
            provenance_rank INTEGER,
            series TEXT,
            alertlevel TEXT,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CONSTRAINT facts_resolved_unique
                UNIQUE (ym, iso3, hazard_code, metric, series_semantics)
        )
        """
    )
    if extra_tables:
        con.execute(
            """
            CREATE TABLE facts_deltas (
                ym TEXT NOT NULL,
                iso3 TEXT NOT NULL,
                hazard_code TEXT NOT NULL,
                metric TEXT NOT NULL,
                value_new DOUBLE,
                value_stock DOUBLE,
                series_semantics TEXT NOT NULL DEFAULT 'new',
                as_of VARCHAR,
                source_id TEXT,
                series TEXT,
                rebase_flag INTEGER,
                first_observation INTEGER,
                delta_negative_clamped INTEGER,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT facts_deltas_unique UNIQUE (ym, iso3, hazard_code, metric)
            )
            """
        )
        con.execute(
            """
            CREATE TABLE acled_monthly_fatalities (
                iso3 TEXT,
                month DATE,
                fatalities BIGINT,
                source TEXT,
                updated_at TIMESTAMP
            )
            """
        )
    con.execute("CREATE TABLE hs_runs (hs_run_id TEXT PRIMARY KEY)")
    con.execute("INSERT INTO hs_runs VALUES ('run1')")
    con.execute(
        """
        CREATE TABLE questions (
            question_id TEXT,
            hs_run_id TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            target_month TEXT,
            window_start_date DATE,
            status TEXT
        )
        """
    )


@pytest.mark.db
def test_compute_resolutions_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify compute_resolutions writes resolution rows when facts_resolved has data."""
    db_path = tmp_path / "e2e_resolutions.duckdb"
    db_url = f"duckdb:///{db_path}"

    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}

    monkeypatch.setattr("pythia.tools.compute_resolutions.load_cfg", _fake_load_cfg)

    con = duckdb.connect(str(db_path))
    try:
        _setup_e2e_db(con)
        # Question with window starting 2024-08 (h1=2024-08 .. h6=2025-01)
        con.execute(
            """
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status)
            VALUES ('Q1', 'run1', 'MLI', 'FL', 'PA', '2025-01', '2024-08-01', 'active')
            """
        )
        # Ground truth for horizons 1-3 (2024-08, 2024-09, 2024-10)
        con.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value)
            VALUES
                ('2024-08', 'MLI', 'FL', 'affected', 5000.0),
                ('2024-09', 'MLI', 'FL', 'affected', 15000.0),
                ('2024-10', 'MLI', 'FL', 'affected', 30000.0)
            """
        )
    finally:
        con.close()

    # Run with today = 2025-01-15 so that months up to 2024-12 are eligible
    compute_resolutions(db_url=db_url, today=date(2025, 1, 15))

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT question_id, horizon_m, observed_month, value "
            "FROM resolutions ORDER BY horizon_m"
        ).fetchall()
        # We should have resolutions for h1, h2, h3 (the months with ground truth
        # that are <= cutoff 2024-12).  h4 and h5 have no data and PA does NOT
        # default to zero — they should be absent.
        assert len(rows) == 3
        assert rows[0] == ("Q1", 1, "2024-08", 5000.0)
        assert rows[1] == ("Q1", 2, "2024-09", 15000.0)
        assert rows[2] == ("Q1", 3, "2024-10", 30000.0)
    finally:
        con.close()


@pytest.mark.db
def test_compute_resolutions_multi_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify compute_resolutions resolves from facts_deltas and
    acled_monthly_fatalities when facts_resolved has no relevant data."""
    db_path = tmp_path / "e2e_multi.duckdb"
    db_url = f"duckdb:///{db_path}"

    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}

    monkeypatch.setattr("pythia.tools.compute_resolutions.load_cfg", _fake_load_cfg)

    con = duckdb.connect(str(db_path))
    try:
        _setup_e2e_db(con)
        # Question 1: ACE FATALITIES (should resolve from acled_monthly_fatalities)
        con.execute(
            """
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status)
            VALUES ('Q_FAT', 'run1', 'SOM', 'ACE', 'FATALITIES', '2025-01', '2024-08-01', 'active')
            """
        )
        # Question 2: FL PA (should resolve from facts_deltas)
        con.execute(
            """
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status)
            VALUES ('Q_PA', 'run1', 'ETH', 'FL', 'PA', '2025-01', '2024-08-01', 'active')
            """
        )
        # ACLED data for SOM: h1=2024-08, h2=2024-09
        con.execute(
            """
            INSERT INTO acled_monthly_fatalities (iso3, month, fatalities, updated_at)
            VALUES
                ('SOM', '2024-08-01', 120, '2024-09-15 00:00:00'),
                ('SOM', '2024-09-01', 95, '2024-10-15 00:00:00')
            """
        )
        # IDMC data for ETH: h1=2024-08
        con.execute(
            """
            INSERT INTO facts_deltas (ym, iso3, hazard_code, metric, value_new, created_at)
            VALUES ('2024-08', 'ETH', 'FL', 'new_displacements', 7500.0, '2024-09-10 00:00:00')
            """
        )
    finally:
        con.close()

    compute_resolutions(db_url=db_url, today=date(2025, 1, 15))

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT question_id, horizon_m, observed_month, value "
            "FROM resolutions ORDER BY question_id, horizon_m"
        ).fetchall()
        fat_rows = [r for r in rows if r[0] == "Q_FAT"]
        pa_rows = [r for r in rows if r[0] == "Q_PA"]

        # Q_FAT: ACLED data for h1, h2 from source; h3-h5 default to 0.0
        # (FATALITIES+ACE defaults to zero when no data)
        assert fat_rows[0] == ("Q_FAT", 1, "2024-08", 120.0)
        assert fat_rows[1] == ("Q_FAT", 2, "2024-09", 95.0)
        # Remaining horizons within cutoff default to 0.0
        for r in fat_rows[2:]:
            assert r[3] == 0.0, f"FATALITIES h{r[1]} should default to 0.0"

        # Q_PA: only h1 has data; h2+ have no PA data and should NOT resolve
        assert len(pa_rows) == 1, f"Expected 1 PA resolution, got {len(pa_rows)}"
        assert pa_rows[0] == ("Q_PA", 1, "2024-08", 7500.0)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# _data_freshness_cutoff
# ---------------------------------------------------------------------------


class TestDataFreshnessCutoff:
    def test_pa_cutoff_from_facts_resolved(self, multi_table_db):
        multi_table_db.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, created_at)
            VALUES
                ('2025-09', 'MLI', 'FL', 'affected', 100.0, '2025-10-01 00:00:00'),
                ('2025-12', 'ETH', 'DR', 'pa', 200.0, '2026-01-01 00:00:00')
            """
        )
        cutoff = _data_freshness_cutoff(multi_table_db, "PA")
        assert cutoff == "2025-12"

    def test_fatalities_cutoff_from_acled(self, multi_table_db):
        multi_table_db.execute(
            """
            INSERT INTO acled_monthly_fatalities (iso3, month, fatalities, updated_at)
            VALUES
                ('SOM', '2025-10-01', 50, '2025-11-15 00:00:00'),
                ('SOM', '2026-01-01', 80, '2026-02-15 00:00:00')
            """
        )
        cutoff = _data_freshness_cutoff(multi_table_db, "FATALITIES")
        assert cutoff == "2026-01"

    def test_empty_tables_return_none(self, multi_table_db):
        assert _data_freshness_cutoff(multi_table_db, "PA") is None
        assert _data_freshness_cutoff(multi_table_db, "FATALITIES") is None

    def test_pa_cutoff_includes_displaced(self, multi_table_db):
        """The 'displaced' metric from IFRC GO should contribute to the PA cutoff."""
        multi_table_db.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, created_at)
            VALUES ('2026-01', 'MLI', 'FL', 'displaced', 5000.0, '2026-02-01 00:00:00')
            """
        )
        cutoff = _data_freshness_cutoff(multi_table_db, "PA")
        assert cutoff == "2026-01"

    def test_event_occurrence_cutoff(self, multi_table_db):
        multi_table_db.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, created_at)
            VALUES ('2025-11', 'PHL', 'TC', 'event_occurrence', 1.0, '2025-12-01 00:00:00')
            """
        )
        cutoff = _data_freshness_cutoff(multi_table_db, "EVENT_OCCURRENCE")
        assert cutoff == "2025-11"

    def test_phase3plus_cutoff(self, multi_table_db):
        multi_table_db.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, created_at)
            VALUES ('2025-10', 'ETH', 'DR', 'phase3plus_in_need', 17000000.0, '2025-11-01 00:00:00')
            """
        )
        cutoff = _data_freshness_cutoff(multi_table_db, "PHASE3PLUS_IN_NEED")
        assert cutoff == "2025-10"


# ---------------------------------------------------------------------------
# 'displaced' metric resolution
# ---------------------------------------------------------------------------


class TestDisplacedMetric:
    def test_displaced_resolves_via_facts_resolved(self, resolver_db):
        """The 'displaced' metric from IFRC GO should be matched by PA lookups."""
        resolver_db.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value, created_at)
            VALUES ('2025-01', 'MLI', 'FL', 'displaced', 42000.0, '2025-02-10 08:00:00')
            """
        )
        result = _resolve_value(resolver_db, "MLI", "FL", "2025-01", "PA")
        assert result is not None
        assert result[0] == 42000.0

    def test_displaced_resolves_via_facts_deltas(self, multi_table_db):
        """The 'displaced' metric should also be found in facts_deltas."""
        multi_table_db.execute(
            """
            INSERT INTO facts_deltas (ym, iso3, hazard_code, metric, value_new, created_at)
            VALUES ('2025-06', 'ETH', 'FL', 'displaced', 8500.0, '2025-07-10 00:00:00')
            """
        )
        result = _try_facts_deltas(multi_table_db, "ETH", "FL", "2025-06", "PA")
        assert result is not None
        assert result[0] == 8500.0


# ---------------------------------------------------------------------------
# Source-aware null handling (E2E)
# ---------------------------------------------------------------------------


@pytest.mark.db
def test_pa_no_data_produces_null_not_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When PA source data exists for the period (setting the cutoff) but not
    for this specific country/hazard, no resolution row should be written.
    This is the key behavioral change: PA no-data = unresolvable, NOT zero."""
    db_path = tmp_path / "e2e_null_pa.duckdb"
    db_url = f"duckdb:///{db_path}"

    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}
    monkeypatch.setattr("pythia.tools.compute_resolutions.load_cfg", _fake_load_cfg)

    con = duckdb.connect(str(db_path))
    try:
        _setup_e2e_db(con)
        # Question for MLI/FL/PA with window 2025-07 to 2025-12
        con.execute(
            """
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status)
            VALUES ('Q_NULL', 'run1', 'MLI', 'FL', 'PA', '2025-12', '2025-07-01', 'active')
            """
        )
        # Source data exists for ANOTHER country, setting the PA cutoff to 2025-08.
        # No data for MLI/FL at all.
        con.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value)
            VALUES ('2025-08', 'ETH', 'DR', 'affected', 999.0)
            """
        )
    finally:
        con.close()

    compute_resolutions(db_url=db_url)

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT question_id, horizon_m, value "
            "FROM resolutions ORDER BY horizon_m"
        ).fetchall()
        # PA with no data should NOT produce resolution rows (null, not zero).
        assert len(rows) == 0, (
            f"PA no-data should produce 0 resolution rows (null), got {len(rows)}: {rows}"
        )
    finally:
        con.close()


@pytest.mark.db
def test_fatalities_no_data_defaults_to_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """FATALITIES for ACE with no source data should default to 0.0 (ACLED
    covers continuously)."""
    db_path = tmp_path / "e2e_zero_fat.duckdb"
    db_url = f"duckdb:///{db_path}"

    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}
    monkeypatch.setattr("pythia.tools.compute_resolutions.load_cfg", _fake_load_cfg)

    con = duckdb.connect(str(db_path))
    try:
        _setup_e2e_db(con)
        con.execute(
            """
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status)
            VALUES ('Q_FATZ', 'run1', 'SOM', 'ACE', 'FATALITIES', '2025-12', '2025-07-01', 'active')
            """
        )
        # ACLED data for another country to set the FATALITIES cutoff
        con.execute(
            """
            INSERT INTO acled_monthly_fatalities (iso3, month, fatalities, updated_at)
            VALUES ('ETH', '2025-08-01', 50, '2025-09-15 00:00:00')
            """
        )
    finally:
        con.close()

    compute_resolutions(db_url=db_url)

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT question_id, horizon_m, value "
            "FROM resolutions ORDER BY horizon_m"
        ).fetchall()
        # h1=2025-07 and h2=2025-08 are <= cutoff 2025-08
        # Both should default to 0.0 (FATALITIES+ACE)
        assert len(rows) == 2, f"Expected 2 resolution rows, got {len(rows)}: {rows}"
        for qid, h, val in rows:
            assert val == 0.0, f"Expected 0.0 for FATALITIES h{h}, got {val}"
    finally:
        con.close()


@pytest.mark.db
def test_event_occurrence_no_data_defaults_to_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """EVENT_OCCURRENCE with no GDACS data should default to 0.0."""
    db_path = tmp_path / "e2e_zero_eo.duckdb"
    db_url = f"duckdb:///{db_path}"

    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}
    monkeypatch.setattr("pythia.tools.compute_resolutions.load_cfg", _fake_load_cfg)

    con = duckdb.connect(str(db_path))
    try:
        _setup_e2e_db(con)
        con.execute(
            """
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status)
            VALUES ('Q_EO', 'run1', 'PHL', 'TC', 'EVENT_OCCURRENCE', '2025-12', '2025-07-01', 'active')
            """
        )
        # GDACS data for another country to set EVENT_OCCURRENCE cutoff
        con.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value)
            VALUES ('2025-08', 'BGD', 'FL', 'event_occurrence', 1.0)
            """
        )
    finally:
        con.close()

    compute_resolutions(db_url=db_url)

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT question_id, horizon_m, value "
            "FROM resolutions ORDER BY horizon_m"
        ).fetchall()
        assert len(rows) == 2, f"Expected 2 resolution rows, got {len(rows)}: {rows}"
        for qid, h, val in rows:
            assert val == 0.0, f"Expected 0.0 for EVENT_OCCURRENCE h{h}, got {val}"
    finally:
        con.close()


@pytest.mark.db
def test_phase3plus_no_data_produces_null(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """PHASE3PLUS_IN_NEED with no FEWS NET data should NOT produce resolution
    rows (unresolvable, not zero)."""
    db_path = tmp_path / "e2e_null_ipc.duckdb"
    db_url = f"duckdb:///{db_path}"

    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}
    monkeypatch.setattr("pythia.tools.compute_resolutions.load_cfg", _fake_load_cfg)

    con = duckdb.connect(str(db_path))
    try:
        _setup_e2e_db(con)
        con.execute(
            """
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status)
            VALUES ('Q_IPC', 'run1', 'ETH', 'DR', 'PHASE3PLUS_IN_NEED', '2025-12', '2025-07-01', 'active')
            """
        )
        # FEWS NET data for another country to set the cutoff
        con.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value)
            VALUES ('2025-08', 'SOM', 'DR', 'phase3plus_in_need', 5000000.0)
            """
        )
    finally:
        con.close()

    compute_resolutions(db_url=db_url)

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT question_id, horizon_m, value "
            "FROM resolutions ORDER BY horizon_m"
        ).fetchall()
        assert len(rows) == 0, (
            f"PHASE3PLUS_IN_NEED no-data should produce 0 rows, got {len(rows)}: {rows}"
        )
    finally:
        con.close()


@pytest.mark.db
def test_mixed_pa_real_data_and_null(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """h1 has real PA source data, h2 should have NO resolution row (null, not zero)."""
    db_path = tmp_path / "e2e_mixed_null.duckdb"
    db_url = f"duckdb:///{db_path}"

    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}
    monkeypatch.setattr("pythia.tools.compute_resolutions.load_cfg", _fake_load_cfg)

    con = duckdb.connect(str(db_path))
    try:
        _setup_e2e_db(con)
        con.execute(
            """
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status)
            VALUES ('Q_MIX', 'run1', 'MLI', 'FL', 'PA', '2025-12', '2025-07-01', 'active')
            """
        )
        # Real data for h1 (2025-07), plus another row setting cutoff to 2025-08.
        con.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value)
            VALUES
                ('2025-07', 'MLI', 'FL', 'affected', 5000.0),
                ('2025-08', 'ETH', 'DR', 'affected', 100.0)
            """
        )
    finally:
        con.close()

    compute_resolutions(db_url=db_url)

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT horizon_m, value FROM resolutions ORDER BY horizon_m"
        ).fetchall()
        # Only h1 should have a resolution row (real data).
        # h2 has no PA data for MLI/FL → null, no row.
        assert len(rows) == 1, f"Expected 1 row, got {len(rows)}: {rows}"
        assert rows[0] == (1, 5000.0), f"h1 should have real value, got {rows[0]}"
    finally:
        con.close()


@pytest.mark.db
def test_di_hazard_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """DI (Displacement Influx) questions should be skipped — no resolution source."""
    db_path = tmp_path / "e2e_di.duckdb"
    db_url = f"duckdb:///{db_path}"

    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}
    monkeypatch.setattr("pythia.tools.compute_resolutions.load_cfg", _fake_load_cfg)

    con = duckdb.connect(str(db_path))
    try:
        _setup_e2e_db(con)
        con.execute(
            """
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status)
            VALUES ('Q_DI', 'run1', 'MLI', 'DI', 'PA', '2025-12', '2025-07-01', 'active')
            """
        )
        # Provide PA data so the cutoff is set.
        con.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value)
            VALUES ('2025-12', 'ETH', 'FL', 'affected', 100.0)
            """
        )
    finally:
        con.close()

    compute_resolutions(db_url=db_url)

    con = duckdb.connect(str(db_path))
    try:
        count = con.execute("SELECT COUNT(*) FROM resolutions").fetchone()[0]
        assert count == 0, f"DI questions should not resolve, got {count} rows"
    finally:
        con.close()


@pytest.mark.db
def test_hw_hazard_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """HW (Heatwave) questions should be skipped — no resolution source."""
    db_path = tmp_path / "e2e_hw.duckdb"
    db_url = f"duckdb:///{db_path}"

    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}
    monkeypatch.setattr("pythia.tools.compute_resolutions.load_cfg", _fake_load_cfg)

    con = duckdb.connect(str(db_path))
    try:
        _setup_e2e_db(con)
        con.execute(
            """
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status)
            VALUES ('Q_HW', 'run1', 'IND', 'HW', 'PA', '2025-12', '2025-07-01', 'active')
            """
        )
        con.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value)
            VALUES ('2025-12', 'ETH', 'FL', 'affected', 100.0)
            """
        )
    finally:
        con.close()

    compute_resolutions(db_url=db_url)

    con = duckdb.connect(str(db_path))
    try:
        count = con.execute("SELECT COUNT(*) FROM resolutions").fetchone()[0]
        assert count == 0, f"HW questions should not resolve, got {count} rows"
    finally:
        con.close()


@pytest.mark.db
def test_ineligible_horizons_not_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Horizons beyond the data freshness cutoff should not be resolved."""
    db_path = tmp_path / "e2e_cutoff.duckdb"
    db_url = f"duckdb:///{db_path}"

    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}
    monkeypatch.setattr("pythia.tools.compute_resolutions.load_cfg", _fake_load_cfg)

    con = duckdb.connect(str(db_path))
    try:
        _setup_e2e_db(con)
        # Window: 2025-07 to 2025-12 (h1=Jul, h2=Aug, ..., h6=Dec)
        con.execute(
            """
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status)
            VALUES ('Q_CUT', 'run1', 'SOM', 'ACE', 'FATALITIES', '2025-12', '2025-07-01', 'active')
            """
        )
        # ACLED data sets cutoff to 2025-09, plus real data for SOM
        con.execute(
            """
            INSERT INTO acled_monthly_fatalities (iso3, month, fatalities, updated_at)
            VALUES
                ('SOM', '2025-07-01', 100, '2025-08-15 00:00:00'),
                ('SOM', '2025-08-01', 80, '2025-09-15 00:00:00'),
                ('SOM', '2025-09-01', 60, '2025-10-15 00:00:00')
            """
        )
    finally:
        con.close()

    compute_resolutions(db_url=db_url)

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT horizon_m FROM resolutions ORDER BY horizon_m"
        ).fetchall()
        resolved_horizons = [r[0] for r in rows]
        # h1=2025-07, h2=2025-08, h3=2025-09 are <= cutoff 2025-09
        assert resolved_horizons == [1, 2, 3], (
            f"Only h1-h3 should be resolved, got {resolved_horizons}"
        )
    finally:
        con.close()


@pytest.mark.db
def test_partial_resolution_some_horizons_null(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When some horizons have data and others don't for PA, only horizons
    with actual data should get resolution rows."""
    db_path = tmp_path / "e2e_partial.duckdb"
    db_url = f"duckdb:///{db_path}"

    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}
    monkeypatch.setattr("pythia.tools.compute_resolutions.load_cfg", _fake_load_cfg)

    con = duckdb.connect(str(db_path))
    try:
        _setup_e2e_db(con)
        # Window: 2025-07 to 2025-12; cutoff covers all 6 months
        con.execute(
            """
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status)
            VALUES ('Q_PART', 'run1', 'BGD', 'FL', 'PA', '2025-12', '2025-07-01', 'active')
            """
        )
        # Data only for h1 (Jul) and h3 (Sep), not h2, h4, h5, h6
        con.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value)
            VALUES
                ('2025-07', 'BGD', 'FL', 'affected', 10000.0),
                ('2025-09', 'BGD', 'FL', 'affected', 25000.0),
                ('2025-12', 'ETH', 'FL', 'affected', 1.0)
            """
        )
    finally:
        con.close()

    compute_resolutions(db_url=db_url, today=date(2026, 2, 1))

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT horizon_m, value FROM resolutions ORDER BY horizon_m"
        ).fetchall()
        # Only h1 and h3 should have resolution rows
        assert len(rows) == 2, f"Expected 2 partial resolution rows, got {len(rows)}: {rows}"
        assert rows[0] == (1, 10000.0)
        assert rows[1] == (3, 25000.0)
    finally:
        con.close()
