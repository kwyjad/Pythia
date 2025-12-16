# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import argparse
import logging

from resolver.db import duckdb_io

from pythia.config import load as load_cfg


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())


SPD_CLASS_BINS = [
    "<10k",
    "10k-<50k",
    "50k-<250k",
    "250k-<500k",
    ">=500k",
]


def _get_db_url_from_config() -> str:
    cfg = load_cfg()
    app_cfg = cfg.get("app", {}) if isinstance(cfg, dict) else {}
    db_url = str(app_cfg.get("db_url", "")).strip()
    if not db_url:
        db_url = "duckdb:///data/resolver.duckdb"
        LOGGER.warning("app.db_url missing in config; falling back to %s", db_url)
    else:
        LOGGER.info("Using app.db_url from config: %s", db_url)
    return db_url


def _open_db(db_url: str | None):
    if not duckdb_io.DUCKDB_AVAILABLE:
        raise RuntimeError(duckdb_io.duckdb_unavailable_reason())
    return duckdb_io.get_db(db_url or duckdb_io.DEFAULT_DB_URL)


def _close_db(conn) -> None:
    try:
        duckdb_io.close_db(conn)
    except Exception:
        pass


def compute_bucket_centroids(db_url: str, metric: str = "PA") -> None:
    """
    Compute hazard-specific bucket centroids from historical facts_resolved.

    - Reads PA-like rows from resolver.facts_resolved.
    - Bins `value` into the SPD buckets.
    - Computes E[value | hazard_code, bucket].
    - Writes results into pythia.bucket_centroids (overwriting existing rows for this metric).

    Assumes:
      - `facts_resolved` table exists in the same DuckDB database.
      - `metric` is the Pythia metric name (e.g. 'PA'), not the raw facts_resolved.metric.
    """
    conn = _open_db(db_url)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bucket_centroids (
              hazard_code TEXT,
              metric      TEXT,
              class_bin   TEXT,
              ev          DOUBLE,
              n_obs       BIGINT,
              updated_at  TIMESTAMP DEFAULT now(),
              PRIMARY KEY (hazard_code, metric, class_bin)
            )
            """
        )

        LOGGER.info("Deleting existing bucket_centroids rows for metric=%r", metric)
        conn.execute("DELETE FROM bucket_centroids WHERE metric = ?", [metric])

        LOGGER.info("Computing centroids from facts_resolved (PA-like metrics).")
        conn.execute(
            """
            WITH base AS (
              SELECT
                COALESCE(hazard_code, '') AS hazard_code,
                CAST(value AS DOUBLE) AS v
              FROM facts_resolved
              WHERE value IS NOT NULL
                AND lower(metric) IN ('affected','people_affected','pa')
            ),
            binned AS (
              SELECT
                hazard_code,
                CASE
                  WHEN v < 10000 THEN '<10k'
                  WHEN v < 50000 THEN '10k-<50k'
                  WHEN v < 250000 THEN '50k-<250k'
                  WHEN v < 500000 THEN '250k-<500k'
                  ELSE '>=500k'
                END AS class_bin,
                v
              FROM base
            ),
            agg AS (
              SELECT
                hazard_code,
                class_bin,
                AVG(v) AS ev,
                COUNT(*) AS n_obs
              FROM binned
              GROUP BY hazard_code, class_bin
            )
            INSERT INTO bucket_centroids (hazard_code, metric, class_bin, ev, n_obs, updated_at)
            SELECT
              hazard_code,
              ? AS metric,
              class_bin,
              ev,
              n_obs,
              now()
            FROM agg
            """,
            [metric],
        )

        rows = conn.execute(
            """
            SELECT hazard_code, class_bin, ev, n_obs
            FROM bucket_centroids
            WHERE metric = ?
            ORDER BY hazard_code, CASE class_bin
              WHEN '<10k' THEN 1
              WHEN '10k-<50k' THEN 2
              WHEN '50k-<250k' THEN 3
              WHEN '250k-<500k' THEN 4
              WHEN '>=500k' THEN 5
              ELSE 6
            END
            """,
            [metric],
        ).fetchall()
        bin_order = {bin_label: idx for idx, bin_label in enumerate(SPD_CLASS_BINS)}
        rows = sorted(rows, key=lambda r: (r[0], bin_order.get(r[1], len(bin_order))))
        LOGGER.info("bucket_centroids: wrote %d rows for metric=%s", len(rows), metric)
        if len(rows) <= 20:
            for hc, bin_label, ev, n_obs in rows:
                LOGGER.info("  %s | %s -> ev=%.1f (n=%d)", hc or "<ALL>", bin_label, ev, n_obs)
    finally:
        _close_db(conn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute data-driven SPD bucket centroids from facts_resolved.")
    parser.add_argument(
        "--db-url",
        default=None,
        help="DuckDB URL (default: app.db_url from pythia.config, or duckdb:///data/resolver.duckdb)",
    )
    parser.add_argument(
        "--metric",
        default="PA",
        help="Pythia metric name to label centroids with (default: PA).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    db_url = args.db_url or _get_db_url_from_config()
    compute_bucket_centroids(db_url=db_url, metric=args.metric)


if __name__ == "__main__":
    main()
