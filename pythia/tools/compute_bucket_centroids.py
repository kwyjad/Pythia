# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import argparse
import logging

from resolver.db import duckdb_io

from pythia.buckets import get_bucket_specs
from pythia.config import load as load_cfg


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    LOGGER.addHandler(logging.NullHandler())


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
              bucket_index INTEGER,
              centroid    DOUBLE
            )
            """
        )

        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS ux_bucket_centroids
            ON bucket_centroids (hazard_code, metric, bucket_index)
            """
        )

        LOGGER.info("Deleting existing bucket_centroids rows for metric=%r", metric)
        conn.execute("DELETE FROM bucket_centroids WHERE metric = ?", [metric])

        specs = list(get_bucket_specs(metric))
        if not specs:
            raise ValueError(f"No bucket specs configured for metric={metric!r}")

        if any(spec.lower is None for spec in specs):
            raise ValueError("Bucket specs must include lower bounds for all buckets")

        case_lines = []
        for spec in specs:
            if spec.upper is None:
                case_lines.append(f"WHEN v >= {float(spec.lower)} THEN {int(spec.idx)}")
            else:
                case_lines.append(
                    f"WHEN v >= {float(spec.lower)} AND v < {float(spec.upper)} THEN {int(spec.idx)}"
                )
        case_sql = "\n                ".join(case_lines)

        LOGGER.info("Computing centroids from facts_resolved (PA-like metrics).")
        conn.execute(
            f"""
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
                  {case_sql}
                  ELSE NULL
                END AS bucket_index,
                v
              FROM base
            ),
            agg AS (
              SELECT
                hazard_code,
                bucket_index,
                AVG(v) AS centroid
              FROM binned
              WHERE bucket_index IS NOT NULL
              GROUP BY hazard_code, bucket_index
            )
            INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid)
            SELECT
              hazard_code,
              ? AS metric,
              bucket_index,
              centroid
            FROM agg
            """,
            [metric],
        )

        rows = conn.execute(
            """
            SELECT hazard_code, bucket_index, centroid
            FROM bucket_centroids
            WHERE metric = ?
            ORDER BY hazard_code, bucket_index
            """,
            [metric],
        ).fetchall()
        LOGGER.info("bucket_centroids: wrote %d rows for metric=%s", len(rows), metric)
        if len(rows) <= 20:
            for hc, bucket_index, centroid in rows:
                label = next(
                    (spec.label for spec in specs if spec.idx == bucket_index),
                    str(bucket_index),
                )
                LOGGER.info(
                    "  %s | %s -> centroid=%.1f",
                    hc or "<ALL>",
                    label,
                    centroid,
                )
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
