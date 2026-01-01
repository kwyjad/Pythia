# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import os
from typing import Sequence

from pythia.buckets import BUCKET_SPECS
from pythia.db.schema import connect, get_db_url


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


def _ensure_bucket_definitions_table(conn) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bucket_definitions (
            metric TEXT NOT NULL,
            bucket_index INTEGER NOT NULL,
            label TEXT NOT NULL,
            lower_bound DOUBLE,
            upper_bound DOUBLE
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


def _upsert_bucket_definitions(conn, metric: str) -> None:
    metric_upper = str(metric).upper()
    specs = BUCKET_SPECS.get(metric_upper, [])
    conn.execute(
        "DELETE FROM bucket_definitions WHERE upper(metric) = ?",
        [metric_upper],
    )
    rows = [
        (
            metric_upper,
            int(spec.idx),
            spec.label,
            None if spec.lower is None else float(spec.lower),
            None if spec.upper is None else float(spec.upper),
        )
        for spec in specs
    ]
    if rows:
        conn.executemany(
            """
            INSERT INTO bucket_definitions (
                metric, bucket_index, label, lower_bound, upper_bound
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )


def main() -> None:
    db_url = os.getenv("PYTHIA_DB_URL") or get_db_url()
    print(f"[update_bucket_centroids] Connecting to {db_url}")
    conn = connect(read_only=False)
    try:
        _ensure_table(conn)
        _ensure_bucket_definitions_table(conn)
        for metric, specs in BUCKET_SPECS.items():
            _upsert_bucket_definitions(conn, metric)
            _upsert_centroids(conn, metric, [spec.centroid for spec in specs])
    finally:
        conn.close()
    print("[update_bucket_centroids] Done.")


if __name__ == "__main__":
    main()
