# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

from resolver.db import duckdb_io

from pythia.buckets import bucket_schema_version, get_bucket_specs
from pythia.config import load as load_cfg
from pythia.tools.scoring_class import exact_outcome_clause, scored_only_clause


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


def _ensure_schema_version_column(conn) -> None:
    """Add the bucket-boundary version stamp on DBs that predate it."""
    cols = {
        str(r[1]).lower()
        for r in conn.execute("PRAGMA table_info('bucket_centroids')").fetchall()
    }
    if "schema_version" not in cols:
        conn.execute("ALTER TABLE bucket_centroids ADD COLUMN schema_version TEXT")


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
        _ensure_schema_version_column(conn)

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

        # Build metric-aware filter for facts_resolved
        if metric == "PHASE3PLUS_IN_NEED":
            metric_filter = "lower(metric) = 'phase3plus_in_need'"
        elif metric == "PA":
            metric_filter = "lower(metric) IN ('affected','people_affected','pa','displaced')"
        elif metric == "FATALITIES":
            # facts_resolved carries no all-types conflict series: its
            # ACLED rows are the battles-only 'fatalities_battle_month' and
            # its 'fatalities' rows are IFRC natural-hazard deaths. ACE
            # centroids are learned from acled_monthly_fatalities below,
            # the series ACE/FATALITIES questions resolve against.
            metric_filter = "lower(metric) = 'fatalities' AND upper(hazard_code) NOT IN ('ACE','ACO')"
        else:
            metric_filter = f"lower(metric) = '{metric.lower()}'"

        conflict_union = ""
        if metric == "FATALITIES":
            has_acled = conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'acled_monthly_fatalities'"
            ).fetchone()[0]
            if has_acled:
                conflict_union = (
                    "UNION ALL SELECT 'ACE' AS hazard_code, CAST(fatalities AS DOUBLE) AS v "
                    "FROM acled_monthly_fatalities WHERE fatalities IS NOT NULL"
                )
        LOGGER.info("Computing centroids from facts_resolved (metric=%s, filter=%s).", metric, metric_filter)
        conn.execute(
            f"""
            WITH base AS (
              SELECT
                COALESCE(hazard_code, '') AS hazard_code,
                CAST(value AS DOUBLE) AS v
              FROM facts_resolved
              WHERE value IS NOT NULL
                AND {metric_filter}
              {conflict_union}
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
            INSERT INTO bucket_centroids (hazard_code, metric, bucket_index, centroid, schema_version)
            SELECT
              hazard_code,
              ? AS metric,
              bucket_index,
              centroid,
              ? AS schema_version
            FROM agg
            """,
            [metric, bucket_schema_version(metric)],
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


# --- EMA centroid update (Phase 2) ---

EMA_LEARNING_RATE = 0.1
MIN_RESOLUTIONS_PER_BUCKET = 10  # auto-activation threshold


NOVEMBER_2026_PRIORS = Path(__file__).with_name("data") / "centroid_priors_2026_11.csv"
NOVEMBER_2026_BACKFILL_MARK = "pre-2026-11 (backfill)"


def backfill_november_2026_priors(conn, path: Path | None = None) -> int:
    """Fill ``prior_centroid`` on the 2026-11 rows, once. Returns rows filled.

    The 11 November 2026 calibration chain moved these centroids before the
    column existed, so the value each row replaced is known only from that
    chain's ``centroid_ema_updates.csv``, committed as
    ``data/centroid_priors_2026_11.csv``. Applied only where
    ``as_of_month = '2026-11'`` and ``prior_centroid`` is still NULL, so a
    second call changes nothing.
    """

    import csv

    path = path or NOVEMBER_2026_PRIORS
    if not path.is_file():
        return 0
    filled = 0
    with path.open(encoding="utf-8") as handle:
        rows = [line for line in handle if not line.startswith("#")]
    for record in csv.DictReader(rows):
        params = [
            float(record["old"]), NOVEMBER_2026_BACKFILL_MARK,
            str(record["hazard_code"]).upper(), str(record["metric"]).upper(),
            int(record["bucket_index"]),
        ]
        hit = conn.execute(
            """
            SELECT COUNT(*) FROM bucket_centroids
            WHERE as_of_month = '2026-11' AND prior_centroid IS NULL
              AND upper(hazard_code) = ? AND upper(metric) = ? AND bucket_index = ?
            """,
            params[2:],
        ).fetchone()[0]
        if not hit:
            continue
        conn.execute(
            """
            UPDATE bucket_centroids SET prior_centroid = ?, prior_as_of_month = ?
            WHERE as_of_month = '2026-11' AND prior_centroid IS NULL
              AND upper(hazard_code) = ? AND upper(metric) = ? AND bucket_index = ?
            """,
            params,
        )
        filled += int(hit)
        LOGGER.info(
            "EMA centroid backfill: %s/%s bucket %s prior_centroid = %s (from the "
            "November 2026 calibration chain)",
            record["hazard_code"], record["metric"], record["bucket_index"], record["old"],
        )
    if filled:
        LOGGER.info("EMA centroid backfill: filled prior_centroid on %d 2026-11 row(s)", filled)
    return filled


def _restore_unqualified(conn, metric: str, as_of_month: str, qualifying: set) -> list[dict]:
    """Return this month's moved rows that no longer qualify to their prior."""

    restored: list[dict] = []
    stale = conn.execute(
        """
        SELECT hazard_code, bucket_index, centroid, prior_centroid
        FROM bucket_centroids
        WHERE upper(metric) = ? AND as_of_month = ? AND prior_centroid IS NOT NULL
          AND hazard_code <> '*' AND centroid <> prior_centroid
        """,
        [metric.upper(), as_of_month],
    ).fetchall()
    for hazard_code, bucket_index, centroid, prior in stale:
        if (str(hazard_code).upper(), int(bucket_index)) in qualifying:
            continue
        conn.execute(
            """
            UPDATE bucket_centroids SET centroid = prior_centroid
            WHERE upper(hazard_code) = ? AND upper(metric) = ? AND bucket_index = ?
              AND as_of_month = ?
            """,
            [str(hazard_code).upper(), metric.upper(), int(bucket_index), as_of_month],
        )
        LOGGER.warning(
            "EMA centroid update: %s/%s bucket %d no longer has %d resolutions; "
            "restored this month's step %.1f -> %.1f (the value the month started from)",
            hazard_code, metric, bucket_index, MIN_RESOLUTIONS_PER_BUCKET, centroid, prior,
        )
        restored.append({
            "hazard_code": str(hazard_code).upper(), "metric": metric,
            "bucket_index": int(bucket_index), "old": float(centroid), "new": float(prior),
            "empirical": None, "n_resolutions": None, "as_of_month": as_of_month,
            "restored_to_prior": True,
        })
    return restored


def update_bucket_centroids_ema(
    db_url: str,
    metric: str = "PA",
    as_of: date | None = None,
) -> list[dict]:
    """EMA-update bucket centroids from resolution data.

    Returns one dict per centroid moved (old -> new), for the chain
    diagnostics; an empty list when nothing moved.

    For each (hazard_code, metric, bucket_index) with >= MIN_RESOLUTIONS_PER_BUCKET
    resolutions, compute the empirical mean of resolved values in that bucket,
    then blend with the current centroid using exponential moving average:

        new_centroid = (1 - alpha) * current_centroid + alpha * empirical_mean

    where alpha = EMA_LEARNING_RATE.

    Writes new rows with as_of_month set. Does NOT overwrite default/seed rows.
    """
    if as_of is None:
        as_of = date.today()
    as_of_month = as_of.strftime("%Y-%m")

    updates: list[dict] = []
    conn = _open_db(db_url)
    try:
        # Ensure stamp columns exist (PRAGMA-first — a failed ALTER inside an
        # implicit transaction aborts the connection on DuckDB >= 1.5).
        cols = {
            str(r[1]).lower()
            for r in conn.execute("PRAGMA table_info('bucket_centroids')").fetchall()
        }
        if "as_of_month" not in cols:
            conn.execute("ALTER TABLE bucket_centroids ADD COLUMN as_of_month TEXT")
        if "schema_version" not in cols:
            conn.execute("ALTER TABLE bucket_centroids ADD COLUMN schema_version TEXT")
        if "prior_centroid" not in cols:
            conn.execute("ALTER TABLE bucket_centroids ADD COLUMN prior_centroid DOUBLE")
        if "prior_as_of_month" not in cols:
            conn.execute("ALTER TABLE bucket_centroids ADD COLUMN prior_as_of_month TEXT")
        backfill_november_2026_priors(conn)

        specs = list(get_bucket_specs(metric))
        if not specs:
            LOGGER.warning("No bucket specs for metric=%s; skipping EMA update.", metric)
            return updates

        # Build bucket CASE expression
        case_lines = []
        for spec in specs:
            if spec.upper is None:
                case_lines.append(f"WHEN v >= {float(spec.lower)} THEN {int(spec.idx)}")
            else:
                case_lines.append(
                    f"WHEN v >= {float(spec.lower)} AND v < {float(spec.upper)} THEN {int(spec.idx)}"
                )
        case_sql = "\n                ".join(case_lines)

        # Compute empirical mean per (hazard_code, bucket) from resolutions.
        # An indicative ACE/PA month (pythia/tools/scoring_class.py) is a
        # selected sample and never moves a centroid.
        empirical_sql = f"""
            WITH res_with_bucket AS (
                SELECT
                    q.hazard_code,
                    r.value AS v,
                    CASE {case_sql} ELSE NULL END AS bucket_index
                FROM resolutions r
                JOIN questions q ON q.question_id = r.question_id
                WHERE upper(q.metric) = ?
                  AND r.value IS NOT NULL{scored_only_clause(conn, "r", prefix=chr(10) + "                  AND ")}{exact_outcome_clause(conn, "r", prefix=chr(10) + "                  AND ")}
            )
            SELECT
                COALESCE(UPPER(hazard_code), '*') AS hazard_code,
                bucket_index,
                COUNT(*) AS n_resolutions,
                AVG(v) AS empirical_mean
            FROM res_with_bucket
            WHERE bucket_index IS NOT NULL
            GROUP BY hazard_code, bucket_index
            HAVING COUNT(*) >= {MIN_RESOLUTIONS_PER_BUCKET}
        """

        rows = conn.execute(empirical_sql, [metric.upper()]).fetchall()

        # A bucket that moved earlier THIS month and no longer has enough
        # resolutions goes back to the value the month started from: the
        # step it took was learned from resolutions that have since been
        # withdrawn, and "one step a month from the month's start" means no
        # step when the bucket does not qualify. FL/PA bucket 6 on 23 Nov
        # 2026: the 11 Nov step (837,934 to 1,238,385) rested on eight
        # resolutions the figure rules later withdrew, leaving seven.
        qualifying = {(str(h).upper(), int(b)) for h, b, _n, _m in rows}
        updates.extend(_restore_unqualified(conn, metric, as_of_month, qualifying))

        if not rows:
            LOGGER.info(
                "EMA centroid update: no (hazard, bucket) pairs with >= %d resolutions "
                "for metric=%s. Phase 2 not yet active.",
                MIN_RESOLUTIONS_PER_BUCKET, metric,
            )
            return updates

        n_updated = 0
        for hazard_code, bucket_index, n_res, empirical_mean in rows:
            # Load current centroid (newest as_of_month, or seed/default)
            current_row = conn.execute(
                """
                SELECT centroid, as_of_month, prior_centroid, prior_as_of_month
                FROM bucket_centroids
                WHERE upper(hazard_code) = ? AND upper(metric) = ? AND bucket_index = ?
                ORDER BY as_of_month DESC NULLS LAST
                LIMIT 1
                """,
                [hazard_code.upper(), metric.upper(), int(bucket_index)],
            ).fetchone()

            # One EMA step per month, however many times calibration runs in
            # it. The newest row may be THIS month's own output; blending
            # from it again is a second step, and the DELETE below would then
            # also lose the value the month started from. So a row written
            # this month blends from the value it replaced (prior_centroid),
            # and a row from an earlier month becomes the prior.
            prior_centroid: float | None
            prior_month: str | None
            if current_row is None:
                spec = next((s for s in specs if s.idx == bucket_index), None)
                current_centroid = float(spec.centroid) if spec else float(empirical_mean)
                prior_centroid, prior_month = current_centroid, None
            elif current_row[1] == as_of_month and current_row[2] is not None:
                current_centroid = float(current_row[2])
                prior_centroid, prior_month = float(current_row[2]), current_row[3]
            else:
                if current_row[1] == as_of_month:
                    LOGGER.warning(
                        "EMA centroid update: %s/%s bucket %d already moved in %s and "
                        "carries no prior_centroid; blending from its current value "
                        "(a second step this month)",
                        hazard_code, metric, bucket_index, as_of_month,
                    )
                current_centroid = float(current_row[0])
                prior_centroid, prior_month = current_centroid, current_row[1]

            new_centroid = (
                (1.0 - EMA_LEARNING_RATE) * current_centroid
                + EMA_LEARNING_RATE * float(empirical_mean)
            )

            # Replace any existing row for this (hazard, metric, bucket).
            # The unique index ux_bucket_centroids is on (hazard_code, metric,
            # bucket_index) — it does NOT include as_of_month — so we must
            # delete any prior row (regardless of as_of_month) before inserting
            # the new EMA value, or the INSERT will violate the unique
            # constraint on the second and subsequent EMA runs.
            conn.execute(
                """
                DELETE FROM bucket_centroids
                WHERE upper(hazard_code) = ? AND upper(metric) = ?
                  AND bucket_index = ?
                """,
                [hazard_code.upper(), metric.upper(), int(bucket_index)],
            )
            conn.execute(
                """
                INSERT INTO bucket_centroids
                    (hazard_code, metric, bucket_index, centroid, as_of_month,
                     schema_version, prior_centroid, prior_as_of_month)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [hazard_code.upper(), metric.upper(), int(bucket_index),
                 new_centroid, as_of_month, bucket_schema_version(metric),
                 prior_centroid, prior_month],
            )

            LOGGER.info(
                "EMA centroid update: %s/%s bucket %d: %.1f -> %.1f "
                "(empirical=%.1f, n=%d)",
                hazard_code, metric, bucket_index, current_centroid,
                new_centroid, empirical_mean, n_res,
            )
            n_updated += 1
            updates.append({
                "hazard_code": hazard_code, "metric": metric,
                "bucket_index": int(bucket_index), "old": float(current_centroid),
                "new": float(new_centroid), "empirical": float(empirical_mean),
                "n_resolutions": int(n_res), "as_of_month": as_of_month,
            })

        LOGGER.info(
            "EMA centroid update: updated %d centroids for metric=%s as_of=%s.",
            n_updated, metric, as_of_month,
        )
    finally:
        _close_db(conn)
    return updates


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
    parser.add_argument(
        "--ema",
        action="store_true",
        help=(
            "Run EMA centroid update instead of static recompute. "
            "Auto-activates per bucket when >= 10 resolutions."
        ),
    )
    parser.add_argument("--report-json", default=None,
                        help="With --ema: write the centroids moved here (chain diagnostics)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    db_url = args.db_url or _get_db_url_from_config()
    if args.ema:
        updates = update_bucket_centroids_ema(db_url=db_url, metric=args.metric)
        if args.report_json:
            from pythia.tools.compute_resolutions import write_report_json

            write_report_json(args.report_json, {"metric": args.metric, "updates": updates})
    else:
        compute_bucket_centroids(db_url=db_url, metric=args.metric)


if __name__ == "__main__":
    main()
