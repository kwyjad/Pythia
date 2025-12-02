from __future__ import annotations

import os
from typing import Sequence

from resolver.db import duckdb_io

PA_CENTROIDS: Sequence[float] = (0.0, 30_000.0, 150_000.0, 375_000.0, 700_000.0)
FATALITIES_CENTROIDS: Sequence[float] = (0.0, 15.0, 62.0, 300.0, 700.0)


def _ensure_table(conn: "duckdb_io.duckdb.DuckDBPyConnection") -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bucket_centroids (
            metric TEXT NOT NULL,
            hazard_code TEXT NOT NULL,
            bucket_index INTEGER NOT NULL,
            centroid DOUBLE NOT NULL
        );
        """
    )


def _upsert_centroids(
    conn: "duckdb_io.duckdb.DuckDBPyConnection", metric: str, values: Sequence[float]
) -> None:
    metric_upper = str(metric).upper()
    conn.execute(
        "DELETE FROM bucket_centroids WHERE upper(metric) = ? AND hazard_code = '*'",
        [metric_upper],
    )
    rows = [(metric_upper, "*", idx + 1, float(value)) for idx, value in enumerate(values)]
    conn.executemany(
        "INSERT INTO bucket_centroids (metric, hazard_code, bucket_index, centroid) VALUES (?, ?, ?, ?)",
        rows,
    )


def main() -> None:
    db_url = os.getenv("RESOLVER_DB_URL", duckdb_io.DEFAULT_DB_URL)
    print(f"[update_bucket_centroids] Connecting to {db_url}")
    conn = duckdb_io.get_db(db_url)
    try:
        _ensure_table(conn)
        _upsert_centroids(conn, "PA", PA_CENTROIDS)
        _upsert_centroids(conn, "FATALITIES", FATALITIES_CENTROIDS)
    finally:
        duckdb_io.close_db(conn)
    print("[update_bucket_centroids] Done.")


if __name__ == "__main__":
    main()
