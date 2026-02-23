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


def test_resolver_ui_status_prefers_facts_deltas_over_monthly_view():
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
          ('2026-01', 'AFG', 'ACE', 'pa', 20, NULL, 'new', NULL, '2026-01-17 12:00:00', 'IDMC'),
          ('2026-01', 'AFG', 'FL', 'pa', 5, NULL, 'new', NULL, '2026-01-17 09:00:00', 'EM-DAT')
        """
    )
    conn.execute(
        """
        CREATE VIEW facts_monthly_deltas AS
        SELECT
          ym,
          iso3,
          hazard_code,
          metric,
          COALESCE(value_new, value_stock) AS value,
          source_id
        FROM facts_deltas
        """
    )

    rows, _ = get_connector_last_updated(conn)
    rows_by_source = {row["source"]: row for row in rows}

    assert rows_by_source["IDMC"]["last_updated"] == "2026-01-17"
    assert rows_by_source["EM-DAT"]["last_updated"] == "2026-01-17"
    assert rows_by_source["IDMC"]["diagnostics"]["table_used"] == "facts_deltas"
    assert rows_by_source["EM-DAT"]["diagnostics"]["table_used"] == "facts_deltas"


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
    assert (
        rows_by_source["ACLED"]["diagnostics"]["date_expr"]
        == "coalesce(created_at, publication_date, as_of_date, as_of, ym_proxy)"
    )


def test_connector_last_updated_uses_acled_table_rows_over_facts():
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
          ('AFG', '2025-12-01', 12, '2025-12-18 00:00:00'),
          ('AFG', '2025-11-01', 9, '2025-11-18 00:00:00')
        """
    )

    rows, diagnostics = get_connector_last_updated(conn)
    rows_by_source = {row["source"]: row for row in rows}

    assert rows_by_source["ACLED"]["last_updated"] == "2026-01-17"
    assert rows_by_source["ACLED"]["rows_scanned"] == 2
    assert diagnostics["acled_status_source_table"] == "acled_monthly_fatalities"


def test_connector_last_updated_uses_created_at_for_idmc_and_emdat():
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
          ('2026-01', 'AFG', 'ACE', 'pa', 20, NULL, 'new', NULL, '2026-01-17 12:00:00', 'IDMC'),
          ('2026-01', 'AFG', 'FL', 'pa', 5, NULL, 'new', NULL, '2026-01-17 09:00:00', 'EM-DAT')
        """
    )

    rows, _ = get_connector_last_updated(conn)
    rows_by_source = {row["source"]: row for row in rows}

    assert rows_by_source["IDMC"]["last_updated"] == "2026-01-17"
    assert rows_by_source["EM-DAT"]["last_updated"] == "2026-01-17"


def test_connector_last_updated_acled_month_fallback_when_updated_at_missing():
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
        INSERT INTO acled_monthly_fatalities VALUES
          ('AFG', '2026-01-01', 12, NULL)
        """
    )

    rows, diagnostics = get_connector_last_updated(conn)
    rows_by_source = {row["source"]: row for row in rows}

    assert rows_by_source["ACLED"]["last_updated"] == "2026-01-01"
    assert diagnostics["acled_status_date_column_used"] == "month"


def test_connector_last_updated_acled_prefers_facts_created_at_when_acled_table_stale():
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

    rows, _ = get_connector_last_updated(conn)
    rows_by_source = {row["source"]: row for row in rows}

    # The ACLED facts row (source_id='', metric='events') is counted from
    # facts_deltas via _status_from_facts plus acled_monthly_fatalities via
    # _status_from_acled_table.  _status_from_acled_table counts 1 row.
    assert rows_by_source["ACLED"]["rows_scanned"] == 1
    assert rows_by_source["ACLED"]["last_updated"] == "2026-01-17"


def test_status_from_facts_multi_table_ifrc_and_idmc():
    """After the backfill fix, IFRC data in facts_resolved AND IDMC data in
    facts_deltas should *both* appear in the connector status cards."""
    conn = duckdb.connect(":memory:")
    # IFRC data in facts_resolved (stock rows)
    conn.execute(
        """
        CREATE TABLE facts_resolved (
            iso3 TEXT,
            ym TEXT,
            hazard_code TEXT,
            metric TEXT,
            value DOUBLE,
            source_id TEXT,
            as_of_date DATE,
            created_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO facts_resolved VALUES
          ('MLI', '2024-08', 'FL', 'affected', 5000, 'ifrc_go:field_report:1234', '2024-09-01', '2024-09-05 12:00:00'),
          ('BGD', '2024-09', 'FL', 'affected', 12000, 'ifrc_go:appeal:5678', '2024-10-01', '2024-10-02 08:00:00')
        """
    )
    # IDMC data in facts_deltas (flow rows)
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
            source_id TEXT,
            created_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO facts_deltas VALUES
          ('2024-08', 'MLI', 'ACE', 'new_displacements', 1500, NULL, 'new', NULL, 'idmc_idu', '2024-09-10 00:00:00'),
          ('2024-09', 'MLI', 'ACE', 'new_displacements', 2000, NULL, 'new', NULL, 'idmc_idu', '2024-10-10 00:00:00'),
          ('2024-08', 'ETH', 'FL', 'new_displacements', 3000, NULL, 'new', NULL, 'idmc_idu', '2024-09-15 00:00:00')
        """
    )

    rows, diagnostics = get_connector_last_updated(conn)
    rows_by_source = {row["source"]: row for row in rows}

    # IFRC should see its 2 rows in facts_resolved
    assert rows_by_source["IFRC"]["rows_scanned"] == 2
    assert rows_by_source["IFRC"]["last_updated"] is not None

    # IDMC should see its 3 rows in facts_deltas (not zero!)
    assert rows_by_source["IDMC"]["rows_scanned"] == 3
    assert rows_by_source["IDMC"]["last_updated"] == "2024-10-10"

    # Both tables were checked, so IDMC used fallback (no facts_resolved rows)
    # and IFRC used facts_resolved
    assert rows_by_source["IFRC"]["diagnostics"]["table_used"] is not None
    assert "facts_resolved" in rows_by_source["IFRC"]["diagnostics"]["table_used"]


def test_status_from_facts_aggregates_across_tables():
    """A connector with data in BOTH tables should get combined counts."""
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE facts_resolved (
            iso3 TEXT, ym TEXT, hazard_code TEXT, metric TEXT,
            value DOUBLE, source_id TEXT, as_of_date DATE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE facts_deltas (
            ym TEXT, iso3 TEXT, hazard_code TEXT, metric TEXT,
            value_new DOUBLE, value_stock DOUBLE, series_semantics TEXT,
            as_of TEXT, source_id TEXT
        )
        """
    )
    # Same connector (EM-DAT) has rows in both tables
    conn.execute(
        """
        INSERT INTO facts_resolved VALUES
          ('AFG', '2024-03', 'FL', 'pa', 89, 'emdat_2024', '2024-04-01')
        """
    )
    conn.execute(
        """
        INSERT INTO facts_deltas VALUES
          ('2024-04', 'AFG', 'DR', 'pa', 50, NULL, 'new', '2024-05-01', 'emdat_2024')
        """
    )

    rows, _ = get_connector_last_updated(conn)
    rows_by_source = {row["source"]: row for row in rows}

    # EM-DAT should see rows from BOTH tables aggregated
    assert rows_by_source["EM-DAT"]["rows_scanned"] == 2
