# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import pytest

duckdb = pytest.importorskip("duckdb")

from pythia.tools.source_coverage import (
    countries_with_source_data,
    months_with_source_data,
    refresh_source_coverage,
)


@pytest.fixture
def cov_db():
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE facts_resolved (ym TEXT, iso3 TEXT, hazard_code TEXT, "
        "metric TEXT, value DOUBLE)"
    )
    con.execute(
        "CREATE TABLE acled_monthly_fatalities (iso3 TEXT, month DATE, "
        "fatalities INTEGER, updated_at TIMESTAMP)"
    )
    yield con
    con.close()


def test_refresh_builds_cells_across_source_tables(cov_db):
    # A facts 'fatalities' row is an IFRC death count or a legacy battles-only
    # ACLED row, so it never widens FATALITIES coverage (Sept 2026).
    cov_db.execute(
        "INSERT INTO facts_resolved VALUES "
        "('2026-01', 'ETH', 'ACE', 'fatalities', 10.0), "
        "('2026-01', 'PHL', 'TC', 'event_occurrence', 1.0)"
    )
    cov_db.execute(
        "INSERT INTO acled_monthly_fatalities VALUES "
        "('KEN', DATE '2026-03-01', 2, TIMESTAMP '2026-04-01 00:00:00'), "
        "('SOM', DATE '2026-02-01', 5, TIMESTAMP '2026-03-02 00:00:00'), "
        # Written before April ended: a partial count, not a covered month.
        "('SOM', DATE '2026-04-01', 1, TIMESTAMP '2026-04-28 00:00:00')"
    )
    written = refresh_source_coverage(cov_db)
    assert written["FATALITIES"] == 2  # SOM/2026-02, KEN/2026-03
    assert written["EVENT_OCCURRENCE"] == 1

    assert months_with_source_data(cov_db, "FATALITIES") == {"2026-02", "2026-03"}
    assert countries_with_source_data(cov_db, "FATALITIES") == {"SOM", "KEN"}
    assert months_with_source_data(cov_db, "EVENT_OCCURRENCE") == {"2026-01"}
    assert countries_with_source_data(cov_db, "EVENT_OCCURRENCE") == {"PHL"}


def test_refresh_is_idempotent_and_replaces_stale_cells(cov_db):
    cov_db.execute(
        "INSERT INTO acled_monthly_fatalities VALUES "
        "('ETH', DATE '2026-01-01', 10, TIMESTAMP '2026-02-03 00:00:00')"
    )
    refresh_source_coverage(cov_db)
    refresh_source_coverage(cov_db)  # second run must not raise on the PK
    n = cov_db.execute(
        "SELECT COUNT(*) FROM source_coverage WHERE metric = 'FATALITIES'"
    ).fetchone()[0]
    assert n == 1

    # Source rows removed -> refresh clears the stale cell.
    cov_db.execute("DELETE FROM acled_monthly_fatalities")
    refresh_source_coverage(cov_db)
    assert months_with_source_data(cov_db, "FATALITIES") == set()


def test_missing_source_tables_are_skipped():
    con = duckdb.connect(":memory:")
    try:
        written = refresh_source_coverage(con)
        assert written == {"FATALITIES": 0, "EVENT_OCCURRENCE": 0}
        assert months_with_source_data(con, "FATALITIES") == set()
        assert countries_with_source_data(con, "EVENT_OCCURRENCE") == set()
    finally:
        con.close()
