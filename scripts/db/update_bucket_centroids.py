# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import os
from typing import Sequence

from pythia.db.schema import PA_CENTROIDS, FATALITIES_CENTROIDS, connect, get_db_url


def _ensure_table(conn) -> None:
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


def _upsert_centroids(conn, metric: str, values: Sequence[float]) -> None:
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
    db_url = os.getenv("PYTHIA_DB_URL") or get_db_url()
    print(f"[update_bucket_centroids] Connecting to {db_url}")
    conn = connect(read_only=False)
    try:
        _ensure_table(conn)
        _upsert_centroids(conn, "PA", PA_CENTROIDS)
        _upsert_centroids(conn, "FATALITIES", FATALITIES_CENTROIDS)
    finally:
        conn.close()
    print("[update_bucket_centroids] Done.")


if __name__ == "__main__":
    main()
