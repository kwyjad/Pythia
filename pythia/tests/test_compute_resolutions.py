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
    _data_freshness_cutoff,
    _llm_derived_window_starts,
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
        # Resolver schema: facts_resolved
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
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT facts_resolved_unique
                    UNIQUE (ym, iso3, hazard_code, metric, series_semantics)
            )
            """
        )
        # Pythia schema
        con.execute(
            """
            CREATE TABLE hs_runs (hs_run_id TEXT PRIMARY KEY)
            """
        )
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
        # that are <= cutoff 2024-12)
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
        # Resolver schema: facts_resolved (empty for this test)
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
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT facts_resolved_unique
                    UNIQUE (ym, iso3, hazard_code, metric, series_semantics)
            )
            """
        )
        # facts_deltas with IDMC displacement data
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
        # acled_monthly_fatalities
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
        # Pythia schema
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
        # Q_FAT should have 2 resolutions from ACLED (h1=2024-08, h2=2024-09)
        # Q_PA should have 1 resolution from facts_deltas (h1=2024-08)
        fat_rows = [r for r in rows if r[0] == "Q_FAT"]
        pa_rows = [r for r in rows if r[0] == "Q_PA"]

        assert len(fat_rows) == 2, f"Expected 2 FATALITIES resolutions, got {len(fat_rows)}"
        assert fat_rows[0] == ("Q_FAT", 1, "2024-08", 120.0)
        assert fat_rows[1] == ("Q_FAT", 2, "2024-09", 95.0)

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
# Zero-default and data-driven eligibility (E2E)
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
def test_zero_default_when_no_source_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When source data exists for the period (setting the cutoff) but not
    for this specific country/hazard, the resolution should default to 0.0."""
    db_path = tmp_path / "e2e_zero.duckdb"
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
            VALUES ('Q_ZERO', 'run1', 'MLI', 'FL', 'PA', '2025-12', '2025-07-01', 'active')
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
        # Horizons h1=2025-07 and h2=2025-08 are <= cutoff 2025-08
        assert len(rows) == 2, f"Expected 2 resolution rows, got {len(rows)}: {rows}"
        for qid, h, val in rows:
            assert val == 0.0, f"Expected 0.0 for h{h}, got {val}"
    finally:
        con.close()


@pytest.mark.db
def test_zero_default_mixed_with_real_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """h1 has real source data, h2 should default to zero."""
    db_path = tmp_path / "e2e_mixed.duckdb"
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
        assert len(rows) == 2
        assert rows[0] == (1, 5000.0), f"h1 should have real value, got {rows[0]}"
        assert rows[1] == (2, 0.0), f"h2 should default to 0.0, got {rows[1]}"
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
            VALUES ('Q_CUT', 'run1', 'MLI', 'FL', 'PA', '2025-12', '2025-07-01', 'active')
            """
        )
        # Data freshness: only up to 2025-09 (h1=Jul, h2=Aug, h3=Sep eligible)
        con.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value)
            VALUES ('2025-09', 'ETH', 'FL', 'affected', 100.0)
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
def test_status_update_after_full_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """When all 6 horizons are resolved, questions.status should be updated to 'resolved'."""
    db_path = tmp_path / "e2e_status.duckdb"
    db_url = f"duckdb:///{db_path}"

    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}
    monkeypatch.setattr("pythia.tools.compute_resolutions.load_cfg", _fake_load_cfg)

    con = duckdb.connect(str(db_path))
    try:
        _setup_e2e_db(con)
        # Window: 2024-08 to 2025-01 (h1=Aug, ..., h6=Jan)
        con.execute(
            """
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status)
            VALUES ('Q_ST', 'run1', 'MLI', 'FL', 'PA', '2025-01', '2024-08-01', 'active')
            """
        )
        # Data covers all 6 months (cutoff >= 2025-01)
        con.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value)
            VALUES ('2025-01', 'ETH', 'FL', 'affected', 100.0)
            """
        )
    finally:
        con.close()

    compute_resolutions(db_url=db_url)

    con = duckdb.connect(str(db_path))
    try:
        # All 6 horizons should be resolved (zero-default)
        count = con.execute("SELECT COUNT(*) FROM resolutions WHERE question_id = 'Q_ST'").fetchone()[0]
        assert count == 6, f"Expected 6 resolutions, got {count}"

        status = con.execute(
            "SELECT status FROM questions WHERE question_id = 'Q_ST'"
        ).fetchone()[0]
        assert status == "resolved", f"Expected status='resolved', got '{status}'"
    finally:
        con.close()


@pytest.mark.db
def test_displaced_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Questions should resolve from facts_deltas rows with metric='displaced'."""
    db_path = tmp_path / "e2e_displaced.duckdb"
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
            VALUES ('Q_DISP', 'run1', 'MLI', 'FL', 'PA', '2025-12', '2025-07-01', 'active')
            """
        )
        # IFRC GO displacement data for h1 and h2
        con.execute(
            """
            INSERT INTO facts_deltas (ym, iso3, hazard_code, metric, value_new, created_at)
            VALUES
                ('2025-07', 'MLI', 'FL', 'displaced', 3000.0, '2025-08-10 00:00:00'),
                ('2025-08', 'MLI', 'FL', 'displaced', 7000.0, '2025-09-10 00:00:00')
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
        # h1 and h2 should resolve from displaced data (cutoff = 2025-08)
        assert len(rows) == 2, f"Expected 2 resolutions, got {len(rows)}"
        assert rows[0] == (1, 3000.0)
        assert rows[1] == (2, 7000.0)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# _llm_derived_window_starts — unit tests
# ---------------------------------------------------------------------------


class TestLlmDerivedWindowStarts:
    def test_basic(self, tmp_path: Path):
        """Two questions with different timestamps → correct window starts."""
        db_path = tmp_path / "llm_ws.duckdb"
        con = duckdb.connect(str(db_path))
        con.execute(
            """
            CREATE TABLE llm_calls (
                question_id TEXT,
                timestamp TIMESTAMP
            )
            """
        )
        con.execute(
            """
            INSERT INTO llm_calls (question_id, timestamp) VALUES
                ('Q_A', '2025-12-15 10:00:00'),
                ('Q_A', '2025-12-20 12:00:00'),
                ('Q_B', '2026-01-05 08:00:00')
            """
        )
        result = _llm_derived_window_starts(con)
        con.close()
        # Q_A earliest = Dec 2025 → window_start = Jan 2026
        assert result["Q_A"] == date(2026, 1, 1)
        # Q_B earliest = Jan 2026 → window_start = Feb 2026
        assert result["Q_B"] == date(2026, 2, 1)

    def test_empty_table(self, tmp_path: Path):
        """Empty llm_calls → empty dict."""
        db_path = tmp_path / "llm_empty.duckdb"
        con = duckdb.connect(str(db_path))
        con.execute(
            "CREATE TABLE llm_calls (question_id TEXT, timestamp TIMESTAMP)"
        )
        result = _llm_derived_window_starts(con)
        con.close()
        assert result == {}

    def test_no_llm_calls_table(self, tmp_path: Path):
        """When llm_calls table doesn't exist → empty dict (no crash)."""
        db_path = tmp_path / "no_llm.duckdb"
        con = duckdb.connect(str(db_path))
        result = _llm_derived_window_starts(con)
        con.close()
        assert result == {}

    def test_null_timestamps_skipped(self, tmp_path: Path):
        """Rows with NULL timestamps should be ignored."""
        db_path = tmp_path / "llm_null.duckdb"
        con = duckdb.connect(str(db_path))
        con.execute(
            "CREATE TABLE llm_calls (question_id TEXT, timestamp TIMESTAMP)"
        )
        con.execute(
            """
            INSERT INTO llm_calls (question_id, timestamp) VALUES
                ('Q_OK', '2025-12-15 10:00:00'),
                ('Q_NULL', NULL)
            """
        )
        result = _llm_derived_window_starts(con)
        con.close()
        assert "Q_OK" in result
        assert "Q_NULL" not in result

    def test_year_rollover(self, tmp_path: Path):
        """Forecast in December → window_start in January of next year."""
        db_path = tmp_path / "llm_rollover.duckdb"
        con = duckdb.connect(str(db_path))
        con.execute(
            "CREATE TABLE llm_calls (question_id TEXT, timestamp TIMESTAMP)"
        )
        con.execute(
            """
            INSERT INTO llm_calls (question_id, timestamp) VALUES
                ('Q_DEC', '2025-12-31 23:59:59')
            """
        )
        result = _llm_derived_window_starts(con)
        con.close()
        assert result["Q_DEC"] == date(2026, 1, 1)


# ---------------------------------------------------------------------------
# LLM-window priority: overrides overwritten questions table window
# ---------------------------------------------------------------------------


def _setup_e2e_db_with_llm_calls(con, extra_tables=True):
    """Like _setup_e2e_db but also creates llm_calls table."""
    _setup_e2e_db(con, extra_tables=extra_tables)
    con.execute(
        """
        CREATE TABLE llm_calls (
            call_id TEXT,
            run_id TEXT,
            hs_run_id TEXT,
            question_id TEXT,
            timestamp TIMESTAMP
        )
        """
    )


@pytest.mark.db
def test_resolution_uses_llm_window_over_question_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When llm_calls has a December forecast but questions table was
    overwritten by a February HS run, resolutions should use the
    December-derived window (2026-01-01), not the February one (2026-03-01)."""
    db_path = tmp_path / "e2e_llm_override.duckdb"
    db_url = f"duckdb:///{db_path}"

    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}
    monkeypatch.setattr("pythia.tools.compute_resolutions.load_cfg", _fake_load_cfg)

    con = duckdb.connect(str(db_path))
    try:
        _setup_e2e_db_with_llm_calls(con)
        # Question's window_start_date was OVERWRITTEN by Feb HS run → 2026-03-01
        # Original forecast was December → window should be 2026-01-01
        con.execute(
            """
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status)
            VALUES ('Q_OVR', 'run1', 'MLI', 'FL', 'PA', '2026-08', '2026-03-01', 'active')
            """
        )
        # llm_calls shows the REAL forecast was in December 2025
        con.execute(
            """
            INSERT INTO llm_calls (call_id, question_id, timestamp)
            VALUES ('call1', 'Q_OVR', '2025-12-15 14:00:00')
            """
        )
        # Source data available up to 2026-02
        con.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value)
            VALUES
                ('2026-01', 'MLI', 'FL', 'affected', 3500.0),
                ('2026-02', 'ETH', 'FL', 'affected', 100.0)
            """
        )
    finally:
        con.close()

    compute_resolutions(db_url=db_url)

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT horizon_m, observed_month, value "
            "FROM resolutions ORDER BY horizon_m"
        ).fetchall()
        # LLM forecast in Dec → window_start = 2026-01-01
        # h1=2026-01 (data exists: 3500), h2=2026-02 (zero-default, cutoff=2026-02)
        # If the overwritten window (2026-03-01) were used, h1=2026-03 > cutoff → 0 rows
        assert len(rows) >= 1, (
            f"Expected at least 1 resolution row (using LLM window), got {len(rows)}: {rows}"
        )
        assert rows[0] == (1, "2026-01", 3500.0), (
            f"h1 should be 2026-01 (from LLM window), got {rows[0]}"
        )
    finally:
        con.close()


@pytest.mark.db
def test_resolution_falls_back_to_question_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When no llm_calls row exists, fall back to q.window_start_date."""
    db_path = tmp_path / "e2e_fallback_q.duckdb"
    db_url = f"duckdb:///{db_path}"

    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}
    monkeypatch.setattr("pythia.tools.compute_resolutions.load_cfg", _fake_load_cfg)

    con = duckdb.connect(str(db_path))
    try:
        _setup_e2e_db_with_llm_calls(con)
        con.execute(
            """
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status)
            VALUES ('Q_FB', 'run1', 'MLI', 'FL', 'PA', '2025-12', '2025-07-01', 'active')
            """
        )
        # No llm_calls row for Q_FB → should use window_start_date from questions
        con.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value)
            VALUES ('2025-08', 'MLI', 'FL', 'affected', 2000.0)
            """
        )
    finally:
        con.close()

    compute_resolutions(db_url=db_url)

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT horizon_m, observed_month FROM resolutions ORDER BY horizon_m"
        ).fetchall()
        # window_start=2025-07-01 → h1=2025-07, h2=2025-08
        # cutoff=2025-08 → h1 and h2 eligible
        assert len(rows) == 2
        assert rows[0][1] == "2025-07"
        assert rows[1][1] == "2025-08"
    finally:
        con.close()


@pytest.mark.db
def test_resolution_falls_back_to_target_month(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """When no llm_calls row exists AND no window_start_date, derive from target_month."""
    db_path = tmp_path / "e2e_fallback_tm.duckdb"
    db_url = f"duckdb:///{db_path}"

    def _fake_load_cfg():
        return {"app": {"db_url": db_url}}
    monkeypatch.setattr("pythia.tools.compute_resolutions.load_cfg", _fake_load_cfg)

    con = duckdb.connect(str(db_path))
    try:
        _setup_e2e_db_with_llm_calls(con)
        # No window_start_date, only target_month
        con.execute(
            """
            INSERT INTO questions
                (question_id, hs_run_id, iso3, hazard_code, metric,
                 target_month, window_start_date, status)
            VALUES ('Q_TM', 'run1', 'MLI', 'FL', 'PA', '2025-12', NULL, 'active')
            """
        )
        # Data up to 2025-08
        con.execute(
            """
            INSERT INTO facts_resolved (ym, iso3, hazard_code, metric, value)
            VALUES ('2025-08', 'ETH', 'FL', 'affected', 100.0)
            """
        )
    finally:
        con.close()

    compute_resolutions(db_url=db_url)

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT horizon_m, observed_month FROM resolutions ORDER BY horizon_m"
        ).fetchall()
        # target_month=2025-12 → window_start = 2025-12 - 5 months = 2025-07
        # h1=2025-07, h2=2025-08, cutoff=2025-08 → 2 rows
        assert len(rows) == 2
        assert rows[0][1] == "2025-07"
        assert rows[1][1] == "2025-08"
    finally:
        con.close()
