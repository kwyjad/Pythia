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

    rows = get_connector_last_updated(conn)

    rows_by_source = {row["source"]: row for row in rows}
    assert rows_by_source["ACLED"]["last_updated"] == "2024-03-15"
    assert rows_by_source["IDMC"]["last_updated"] == "2024-03-05"
    assert rows_by_source["EM-DAT"]["last_updated"] == "2023-09-30"
    assert rows_by_source["ACLED"]["rows_scanned"] == 1
    assert rows_by_source["IDMC"]["rows_scanned"] == 1
    assert rows_by_source["EM-DAT"]["rows_scanned"] == 1


def test_country_facts_conflict_rows_and_parsing():
    conn = duckdb.connect(":memory:")
    _seed_facts(conn)

    rows = get_country_facts(conn, "AFG")

    sources = {row["source_id"] for row in rows if row["hazard_code"] == "ACE"}
    assert sources == {"ACLED", "IDMC"}

    conflict_rows = [row for row in rows if row["hazard_code"] == "ACE"]
    assert all(row["hazard"] == "Armed Conflict" for row in conflict_rows)
    assert {row["metric"] for row in conflict_rows} == {"FATALITIES", "PA"}
    assert {row["year"] for row in rows} == {2023, 2024}
    assert {row["month"] for row in rows} == {3, 9}
