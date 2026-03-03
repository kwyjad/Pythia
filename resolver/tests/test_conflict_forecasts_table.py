# Pythia / Copyright (c) 2025 Kevin Wyjad

"""Tests for the conflict_forecasts DuckDB table.

Validates that ``ensure_schema`` creates the table with the expected
columns and PRIMARY KEY constraint using an in-memory DuckDB connection.
"""

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")
if getattr(duckdb, "__pythia_stub__", False):
    pytest.skip("duckdb not installed", allow_module_level=True)

from pythia.db.schema import ensure_schema


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    """Return lower-cased column names for *table*."""
    return {
        str(row[1]).lower()
        for row in con.execute(f"PRAGMA table_info('{table}')").fetchall()
    }


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_name = ?",
        [table],
    ).fetchall()
    return len(rows) > 0


class TestConflictForecastsTable:
    """Tests for the conflict_forecasts table created by ensure_schema."""

    def test_ensure_schema_creates_conflict_forecasts(self):
        """ensure_schema should create the conflict_forecasts table."""
        con = duckdb.connect(":memory:")
        try:
            ensure_schema(con)
            assert _table_exists(con, "conflict_forecasts")
        finally:
            con.close()

    def test_conflict_forecasts_has_expected_columns(self):
        """The table should contain all expected columns."""
        con = duckdb.connect(":memory:")
        try:
            ensure_schema(con)
            cols = _columns(con, "conflict_forecasts")

            expected = {
                "source",
                "iso3",
                "hazard_code",
                "metric",
                "lead_months",
                "value",
                "forecast_issue_date",
                "target_month",
                "model_version",
                "created_at",
            }
            assert expected.issubset(cols), f"Missing columns: {expected - cols}"
        finally:
            con.close()

    def test_insert_row_with_expected_columns(self):
        """Inserting a well-formed row should succeed."""
        con = duckdb.connect(":memory:")
        try:
            ensure_schema(con)

            con.execute(
                """
                INSERT INTO conflict_forecasts (
                    source, iso3, hazard_code, metric, lead_months,
                    value, forecast_issue_date, target_month, model_version
                ) VALUES (
                    'VIEWS', 'SOM', 'ACE', 'views_predicted_fatalities', 1,
                    42.5, '2025-06-01', '2025-07-01', 'fatalities003'
                )
                """
            )

            rows = con.execute("SELECT COUNT(*) FROM conflict_forecasts").fetchone()
            assert rows[0] == 1

            row = con.execute(
                "SELECT source, iso3, value FROM conflict_forecasts"
            ).fetchone()
            assert row[0] == "VIEWS"
            assert row[1] == "SOM"
            assert row[2] == pytest.approx(42.5)
        finally:
            con.close()

    def test_primary_key_prevents_duplicate_insert(self):
        """Inserting two rows with the same PK should raise a constraint error."""
        con = duckdb.connect(":memory:")
        try:
            ensure_schema(con)

            insert_sql = """
                INSERT INTO conflict_forecasts (
                    source, iso3, hazard_code, metric, lead_months,
                    value, forecast_issue_date, target_month
                ) VALUES (
                    'VIEWS', 'ETH', 'ACE', 'views_predicted_fatalities', 1,
                    10.0, '2025-06-01', '2025-07-01'
                )
            """
            con.execute(insert_sql)

            with pytest.raises(duckdb.ConstraintException):
                con.execute(insert_sql)

            # Confirm only one row survived.
            count = con.execute(
                "SELECT COUNT(*) FROM conflict_forecasts"
            ).fetchone()[0]
            assert count == 1
        finally:
            con.close()

    def test_different_keys_allow_multiple_rows(self):
        """Rows with different PK combinations should coexist."""
        con = duckdb.connect(":memory:")
        try:
            ensure_schema(con)

            con.execute(
                """
                INSERT INTO conflict_forecasts (
                    source, iso3, hazard_code, metric, lead_months,
                    value, forecast_issue_date, target_month
                ) VALUES
                    ('VIEWS', 'ETH', 'ACE', 'views_predicted_fatalities', 1,
                     10.0, '2025-06-01', '2025-07-01'),
                    ('VIEWS', 'ETH', 'ACE', 'views_predicted_fatalities', 2,
                     15.0, '2025-06-01', '2025-08-01'),
                    ('conflictforecast_org', 'ETH', 'ACE', 'cf_armed_conflict_risk_3m', 3,
                     0.45, '2025-06-01', '2025-09-01')
                """
            )

            count = con.execute(
                "SELECT COUNT(*) FROM conflict_forecasts"
            ).fetchone()[0]
            assert count == 3
        finally:
            con.close()

    def test_ensure_schema_is_idempotent(self):
        """Calling ensure_schema twice should not error or lose data."""
        con = duckdb.connect(":memory:")
        try:
            ensure_schema(con)

            con.execute(
                """
                INSERT INTO conflict_forecasts (
                    source, iso3, hazard_code, metric, lead_months,
                    value, forecast_issue_date, target_month
                ) VALUES (
                    'VIEWS', 'SSD', 'ACE', 'views_p_gte25_brd', 1,
                    0.8, '2025-05-01', '2025-06-01'
                )
                """
            )

            # Second call should be harmless.
            ensure_schema(con)

            count = con.execute(
                "SELECT COUNT(*) FROM conflict_forecasts"
            ).fetchone()[0]
            assert count == 1
        finally:
            con.close()
