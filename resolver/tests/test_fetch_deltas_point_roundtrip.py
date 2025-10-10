from __future__ import annotations

import pytest

from resolver.db.conn_shared import get_shared_duckdb_conn
from resolver.query import db_reader

pytest.importorskip(
    "duckdb",
    reason=(
        "duckdb not installed. Install via extras: `pip install .[db]` or use "
        "`scripts/install_db_extra_offline.(sh|ps1)`"
    ),
)


def test_fetch_deltas_point_finds_row(tmp_path, monkeypatch):
    db_path = tmp_path / "roundtrip.duckdb"
    url = f"duckdb:///{db_path}"
    monkeypatch.setenv("RESOLVER_DB_URL", url)

    conn, _, _ = get_shared_duckdb_conn(url)
    conn.execute("DROP TABLE IF EXISTS facts_deltas")
    conn.execute(
        """
        CREATE TABLE facts_deltas (
            ym VARCHAR,
            iso3 VARCHAR,
            hazard_code VARCHAR,
            metric VARCHAR,
            value_new DOUBLE,
            value_stock DOUBLE,
            series_semantics VARCHAR,
            as_of DATE,
            source_id VARCHAR,
            series VARCHAR,
            rebase_flag INTEGER,
            first_observation INTEGER,
            delta_negative_clamped INTEGER,
            created_at TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        INSERT INTO facts_deltas (
            ym, iso3, hazard_code, metric, value_new, value_stock, series_semantics,
            as_of, source_id, series, rebase_flag, first_observation, delta_negative_clamped, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "2024-02",
            "PHL",
            "TC",
            "in_need",
            500.0,
            500.0,
            "new",
            "2024-02-28",
            "source-1",
            "new",
            0,
            1,
            0,
            "2024-02-28 00:00:00",
        ],
    )

    row = db_reader.fetch_deltas_point(
        None,
        ym="2024-02",
        iso3="PHL",
        hazard_code="TC",
        cutoff="2024-02-28",
        preferred_metric="in_need",
    )
    assert row is not None
    assert pytest.approx(row["value_new"], rel=0, abs=1e-9) == 500.0
    assert row["metric"] == "in_need"
