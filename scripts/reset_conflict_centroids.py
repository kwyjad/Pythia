# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Remove ACE/FATALITIES bucket centroids learned from battles-only outcomes.

Until Sept 2026 ACE/FATALITIES questions resolved to the ACLED battles-only
series (see ``compute_resolutions.ACE_FATALITIES_SERIES``). Every hazard-
specific ACE centroid was learned from that wrong quantity: the EMA rows
(``as_of_month`` set) from resolutions, the historical rows (``as_of_month``
NULL, written by ``compute_bucket_centroids``) from ``facts_resolved``
'fatalities', which held the same battles-only series.

Re-running the calibration chain does not undo them. ``update_bucket_centroids_ema``
blends each run's empirical mean into the CURRENT centroid (alpha 0.1), so a
plain re-run moves a wrong centroid a tenth of the way and keeps the rest.
This script deletes every hazard-specific ACE/FATALITIES row, so
``compute_scores._load_centroids`` falls back to the wildcard seed rows
(``hazard_code = '*'``, written by ``scripts/db/update_bucket_centroids.py``)
or to the ``BucketSpec`` defaults, and the next EMA run starts from the seed.

Idempotent: a second run finds nothing to delete. Wildcard rows and every
other hazard and metric are untouched.

Usage::

    python -m scripts.reset_conflict_centroids --db duckdb:///path/resolver.duckdb [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

HAZARDS = ("ACE", "ACO")
METRIC = "FATALITIES"


def reset(con, *, dry_run: bool = False) -> dict[str, Any]:
    """Delete hazard-specific ACE/ACO FATALITIES centroids. Never raises on a
    missing table."""

    present = {
        str(r[0]).lower()
        for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }
    if "bucket_centroids" not in present:
        return {"deleted": 0, "rows": [], "reason": "bucket_centroids absent"}
    cols = {str(r[1]).lower() for r in con.execute("PRAGMA table_info('bucket_centroids')").fetchall()}
    aom = "as_of_month" if "as_of_month" in cols else "NULL AS as_of_month"
    placeholders = ", ".join("?" for _ in HAZARDS)
    where = f"upper(hazard_code) IN ({placeholders}) AND upper(metric) = ?"
    params = [*HAZARDS, METRIC]
    rows = con.execute(
        f"SELECT hazard_code, bucket_index, centroid, {aom} FROM bucket_centroids "
        f"WHERE {where} ORDER BY hazard_code, bucket_index",
        params,
    ).fetchall()
    listing = [
        {"hazard_code": r[0], "bucket_index": int(r[1]), "centroid": float(r[2]),
         "as_of_month": r[3]}
        for r in rows
    ]
    if rows and not dry_run:
        con.execute(f"DELETE FROM bucket_centroids WHERE {where}", params)
    for item in listing:
        LOG.info(
            "[reset_conflict_centroids] %s %s bucket %d centroid=%.1f as_of_month=%s",
            "would delete" if dry_run else "deleted",
            item["hazard_code"], item["bucket_index"], item["centroid"], item["as_of_month"],
        )
    LOG.info(
        "[reset_conflict_centroids] %d ACE/FATALITIES centroid row(s) %s; "
        "EIV now falls back to the wildcard seeds.",
        len(rows), "found (dry run)" if dry_run else "deleted",
    )
    return {"deleted": 0 if dry_run else len(rows), "rows": listing, "dry_run": dry_run}


#: What the audit copy exports, per table: the ACE/FATALITIES rows as they
#: stand before the post-fix rerun replaces them.
AUDIT_QUERIES: dict[str, str] = {
    "resolutions": (
        "SELECT r.* FROM resolutions r JOIN questions q ON q.question_id = r.question_id "
        "WHERE upper(q.hazard_code) = 'ACE' AND upper(q.metric) = 'FATALITIES' "
        "ORDER BY r.question_id, r.horizon_m"
    ),
    "scores": (
        "SELECT s.* FROM scores s JOIN questions q ON q.question_id = s.question_id "
        "WHERE upper(q.hazard_code) = 'ACE' AND upper(q.metric) = 'FATALITIES' "
        "ORDER BY s.question_id, s.horizon_m, s.model_name, s.score_type"
    ),
    "eiv_scores": (
        "SELECT e.* FROM eiv_scores e JOIN questions q ON q.question_id = e.question_id "
        "WHERE upper(q.hazard_code) = 'ACE' AND upper(q.metric) = 'FATALITIES' "
        "ORDER BY e.question_id"
    ),
    "calibration_advice": (
        "SELECT * FROM calibration_advice "
        "WHERE (upper(hazard_code) = 'ACE' AND upper(metric) = 'FATALITIES') "
        "OR model_name LIKE '__ext_%' OR hazard_code = '*' "
        "ORDER BY as_of_month, hazard_code, metric, model_name"
    ),
    "bucket_centroids": (
        "SELECT * FROM bucket_centroids WHERE upper(metric) = 'FATALITIES' "
        "ORDER BY hazard_code, bucket_index"
    ),
}


def export_audit(con, out_dir: Path) -> dict[str, int]:
    """Write the current ACE/FATALITIES rows to CSV, one file per table.

    Read-only. A table that does not exist is recorded as -1 rather than
    raising, so a partial database still yields the files it can.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for table, sql in AUDIT_QUERIES.items():
        path = out_dir / f"{table}__ace_fatalities.csv"
        try:
            cur = con.execute(sql)
            cols = [d[0] for d in (cur.description or [])]
            rows = cur.fetchall()
        except Exception as exc:  # noqa: BLE001
            path.write_text(f"# not exported: {type(exc).__name__}: {exc}\n")
            counts[table] = -1
            continue
        import csv

        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(cols)
            writer.writerows(rows)
        counts[table] = len(rows)
        LOG.info("[reset_conflict_centroids] audit: %s -> %d row(s)", path.name, len(rows))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", default=None, help="DuckDB URL or path")
    parser.add_argument("--dry-run", action="store_true", help="List, delete nothing")
    parser.add_argument("--summary-out", default=None, help="Write the report as JSON here")
    parser.add_argument(
        "--audit-dir", default=None,
        help="Before changing anything, export the current ACE/FATALITIES rows "
        "(resolutions, scores, eiv_scores, calibration_advice, bucket_centroids) to CSV here",
    )
    parser.add_argument(
        "--audit-only", action="store_true", help="Export the audit copy and change nothing"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from resolver.db.duckdb_io import close_db, get_db

    con = get_db(args.db)
    try:
        audit = export_audit(con, Path(args.audit_dir)) if args.audit_dir else {}
        report = reset(con, dry_run=args.dry_run or args.audit_only)
    finally:
        close_db(con)
    report["audit"] = audit
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
