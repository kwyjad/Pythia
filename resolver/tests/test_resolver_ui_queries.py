import duckdb

from resolver.query.resolver_ui import get_connector_last_updated, get_country_facts


def _seed_facts(conn):
    conn.execute(
        """
        CREATE TABLE facts_resolved (
            iso3 TEXT,
            ym TEXT,
            hazard_code TEXT,
            hazard_label TEXT,
            metric TEXT,
            value DOUBLE,
            as_of_date DATE,
            source_id TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO facts_resolved VALUES
          ('AFG', '2024-03', 'ACE', 'Armed Conflict', 'fatalities', 123, '2024-03-15', 'ACLED'),
          ('AFG', '2024-03', 'ACE', 'Armed Conflict', 'pa', 4567, '2024-03-05', 'IDMC'),
          ('AFG', '2023-09', 'FL', 'Flood', 'pa', 89, '2023-09-30', 'EM-DAT')
        """
    )


def test_connector_last_updated_and_rows_scanned():
    conn = duckdb.connect(":memory:")
    _seed_facts(conn)

    rows, diagnostics = get_connector_last_updated(conn)

    rows_by_source = {row["source"]: row for row in rows}
    assert rows_by_source["ACLED"]["last_updated"] == "2024-03-15"
    assert rows_by_source["IDMC"]["last_updated"] == "2024-03-05"
    assert rows_by_source["EM-DAT"]["last_updated"] == "2023-09-30"
    assert rows_by_source["ACLED"]["rows_scanned"] == 1
    assert rows_by_source["IDMC"]["rows_scanned"] == 1
    assert rows_by_source["EM-DAT"]["rows_scanned"] == 1
    assert diagnostics["facts_source_table"] == "facts_resolved"
    assert diagnostics["fallback_used"] is False


def test_country_facts_conflict_rows_and_parsing():
    conn = duckdb.connect(":memory:")
    _seed_facts(conn)

    rows, diagnostics = get_country_facts(conn, "AFG")

    sources = {row["source_id"] for row in rows if row["hazard_code"] == "ACE"}
    assert sources == {"ACLED", "IDMC"}

    conflict_rows = [row for row in rows if row["hazard_code"] == "ACE"]
    assert all(row["hazard"] == "Armed Conflict" for row in conflict_rows)
    assert {row["metric"] for row in conflict_rows} == {"FATALITIES", "PA"}
    assert {row["year"] for row in rows} == {2023, 2024}
    assert {row["month"] for row in rows} == {3, 9}
    assert diagnostics["facts_source_table"] == "facts_resolved"
    assert diagnostics["fallback_used"] is False


def test_resolver_ui_fallback_to_facts_deltas():
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE facts_deltas (
            ym TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            value_new DOUBLE,
            value_stock DOUBLE,
            series_semantics TEXT,
            as_of TEXT,
            source_id TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO facts_deltas VALUES
          ('2024-03', 'AFG', 'ACE', 'fatalities', 10, NULL, 'new', '2024-03-20', 'ACLED'),
          ('2024-03', 'AFG', 'ACE', 'pa', 20, NULL, 'new', '2024-03-18', 'IDMC'),
          ('2024-03', 'AFG', 'FL', 'pa', 5, NULL, 'new', '2024-03-01', 'EM-DAT')
        """
    )

    rows, diagnostics = get_country_facts(conn, "AFG")
    assert {row["source_id"] for row in rows} == {"ACLED", "IDMC", "EM-DAT"}
    assert diagnostics["facts_source_table"] == "facts_deltas"
    assert diagnostics["fallback_used"] is True

    status_rows, status_diagnostics = get_connector_last_updated(conn)
    status_by_source = {row["source"]: row for row in status_rows}
    assert status_by_source["ACLED"]["last_updated"] == "2024-03-20"
    assert status_by_source["IDMC"]["last_updated"] == "2024-03-18"
    assert status_by_source["EM-DAT"]["last_updated"] == "2024-03-01"
    assert status_diagnostics["facts_source_table"] == "facts_deltas"
    assert status_diagnostics["fallback_used"] is True


def test_resolver_ui_fallback_when_facts_resolved_empty():
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE facts_resolved (
            iso3 TEXT,
            ym TEXT,
            hazard_code TEXT,
            hazard_label TEXT,
            metric TEXT,
            value DOUBLE,
            as_of_date DATE,
            source_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE facts_deltas (
            ym TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            value_new DOUBLE,
            value_stock DOUBLE,
            series_semantics TEXT,
            as_of TEXT,
            source_id TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO facts_deltas VALUES
          ('2024-04', 'AFG', 'ACE', 'fatalities', 33, NULL, 'new', '2024-04-12', 'ACLED')
        """
    )

    rows, diagnostics = get_country_facts(conn, "AFG")
    assert len(rows) == 1
    assert diagnostics["facts_source_table"] == "facts_deltas"
    assert diagnostics["fallback_used"] is True


def test_resolver_ui_acled_monthly_fatalities_union():
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE facts_deltas (
            ym TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            value_new DOUBLE,
            value_stock DOUBLE,
            series_semantics TEXT,
            as_of TEXT,
            source_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE acled_monthly_fatalities (
            iso3 TEXT,
            month DATE,
            fatalities DOUBLE,
            updated_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO facts_deltas VALUES
          ('2024-03', 'AFG', 'FL', 'pa', 5, NULL, 'new', '2024-03-01', 'EM-DAT'),
          ('2024-03', 'AFG', 'ACE', 'pa', 20, NULL, 'new', '2024-03-18', 'IDMC')
        """
    )
    conn.execute(
        """
        INSERT INTO acled_monthly_fatalities VALUES
          ('AFG', '2024-03-01', 12, '2024-03-20 00:00:00')
        """
    )

    rows, diagnostics = get_country_facts(conn, "AFG")
    sources = {row["source_id"] for row in rows}
    assert "ACLED" in sources
    assert diagnostics["acled_table_present"] is True
    assert diagnostics["acled_rows_added"] == 1

    status_rows, status_diagnostics = get_connector_last_updated(conn)
    status_by_source = {row["source"]: row for row in status_rows}
    assert status_by_source["ACLED"]["rows_scanned"] == 1
    assert status_by_source["ACLED"]["last_updated"] == "2024-03-20"
    assert status_diagnostics["acled_status_source_table"] == "acled_monthly_fatalities"
    assert status_diagnostics["acled_status_date_column_used"] == "updated_at"


def test_resolver_ui_acled_deduplicates_monthly_rows():
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE facts_deltas (
            ym TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            value_new DOUBLE,
            value_stock DOUBLE,
            series_semantics TEXT,
            as_of TEXT,
            source_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE acled_monthly_fatalities (
            iso3 TEXT,
            month DATE,
            fatalities DOUBLE
        )
        """
    )
    conn.execute(
        """
        INSERT INTO facts_deltas VALUES
          ('2024-03', 'AFG', 'ACE', 'fatalities', 10, NULL, 'new', '2024-03-20', 'ACLED')
        """
    )
    conn.execute(
        """
        INSERT INTO acled_monthly_fatalities VALUES
          ('AFG', '2024-03-01', 10)
        """
    )

    rows, diagnostics = get_country_facts(conn, "AFG")
    acled_rows = [row for row in rows if row["source_id"] == "ACLED"]
    assert len(acled_rows) == 1
    assert diagnostics["acled_rows_added"] == 0


def test_connector_last_updated_uses_created_at_when_present():
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE facts_deltas (
            ym TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            value_new DOUBLE,
            value_stock DOUBLE,
            series_semantics TEXT,
            as_of TEXT,
            created_at TIMESTAMP,
            source_id TEXT
        )
        """
    )
    conn.execute(
        """
        INSERT INTO facts_deltas VALUES
          ('2026-01', 'AFG', 'ACE', 'fatalities', 10, NULL, 'new', '2026-01-01', '2026-01-17 12:00:00', 'ACLED')
        """
    )

    rows, diagnostics = get_connector_last_updated(conn)
    rows_by_source = {row["source"]: row for row in rows}

    assert rows_by_source["ACLED"]["last_updated"] == "2026-01-17"
    assert diagnostics["date_column_used"] == "created_at"


def test_connector_last_updated_prefers_facts_over_acled_monthly_when_facts_present_even_if_source_id_blank():
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE facts_deltas (
            ym TEXT,
            iso3 TEXT,
            hazard_code TEXT,
            metric TEXT,
            value_new DOUBLE,
            value_stock DOUBLE,
            series_semantics TEXT,
            as_of TEXT,
            created_at TIMESTAMP,
            source_id TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE acled_monthly_fatalities (
            iso3 TEXT,
            month DATE,
            fatalities DOUBLE,
            updated_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO facts_deltas VALUES
          ('2026-01', 'AFG', 'ACE', 'events', 1, NULL, 'new', '2026-01-01', '2026-01-17 12:00:00', '')
        """
    )
    conn.execute(
        """
        INSERT INTO acled_monthly_fatalities VALUES
          ('AFG', '2025-12-01', 12, '2025-12-18 00:00:00')
        """
    )

    rows, diagnostics = get_connector_last_updated(conn)
    rows_by_source = {row["source"]: row for row in rows}

    assert rows_by_source["ACLED"]["last_updated"] == "2026-01-17"
    assert rows_by_source["ACLED"]["rows_scanned"] == 1
    assert diagnostics["acled_status_source_table"] == "facts_deltas"
