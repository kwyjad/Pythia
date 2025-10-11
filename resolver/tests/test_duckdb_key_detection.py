import duckdb

from resolver.db import duckdb_io as dio


def test_detector_accepts_unique_index(tmp_path):
    db_path = tmp_path / "d.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE facts_resolved (
            ym VARCHAR,
            iso3 VARCHAR,
            hazard_code VARCHAR,
            metric VARCHAR,
            series_semantics VARCHAR,
            value DOUBLE
        )
        """
    )
    keys = ["ym", "iso3", "hazard_code", "metric", "series_semantics"]
    assert not dio._has_declared_key(conn, "facts_resolved", keys)
    conn.execute(
        """
        CREATE UNIQUE INDEX ux_facts_resolved_series
        ON facts_resolved (ym, iso3, hazard_code, metric, series_semantics)
        """
    )
    assert dio._has_declared_key(conn, "facts_resolved", keys)
