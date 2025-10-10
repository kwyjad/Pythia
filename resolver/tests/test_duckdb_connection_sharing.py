from __future__ import annotations

import pytest

from resolver.db import duckdb_io

pytest.importorskip(
    "duckdb",
    reason=(
        "duckdb not installed. Install via extras: `pip install .[db]` or use "
        "`scripts/install_db_extra_offline.(sh|ps1)`"
    ),
)


def test_same_url_returns_same_connection(tmp_path, monkeypatch):
    db_path = tmp_path / "shared.duckdb"
    url = f"duckdb:///{db_path}"
    monkeypatch.setenv("RESOLVER_DB_URL", url)

    conn_w = duckdb_io.get_db(url)
    conn_w.execute(
        """
        CREATE TABLE IF NOT EXISTS facts_deltas (
            ym VARCHAR,
            iso3 VARCHAR,
            hazard_code VARCHAR,
            metric VARCHAR,
            value_new DOUBLE,
            as_of DATE
        )
        """
    )
    conn_w.execute(
        "INSERT INTO facts_deltas VALUES (?, ?, ?, ?, ?, ?)",
        ["2024-02", "PHL", "TC", "in_need", 500.0, "2024-02-28"],
    )

    conn_r = duckdb_io.get_db(url)

    assert conn_w is conn_r
    count = conn_r.execute(
        "SELECT COUNT(*) FROM facts_deltas WHERE ym = ? AND iso3 = ? AND hazard_code = ?",
        ["2024-02", "PHL", "TC"],
    ).fetchone()[0]
    assert count == 1
