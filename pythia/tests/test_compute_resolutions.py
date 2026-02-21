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
