# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Undo what the false conflict displacement zeros taught the loop.

Until Oct 2026 an ACE/PA month with no IDMC row resolved to zero whenever
IDMC had reported ANY country that month. IDMC reports late and irregularly,
so 60 of the 64 zero-defaults on the 5 October 2026 release were months the
country simply had not been reported for YET. Those zeros were scored, and
the scores reached every learned table:

* ``family_recalibration``: ACE/PA factors of about [2.0, 0.6, 0.58, 0.5,
  0.5, 0.5] for all five families, at the clip limit on four of six buckets,
  in ``apply`` mode;
* ``calibration_weights``: the first ACE/PA weights ever written;
* ``calibration_advice``: shared, per-model and family rows ("actual rate is
  0.0%" for the top two buckets);
* ``bucket_centroids``: EMA-moved ACE/PA rows.

Re-running the chain does not undo them: the newest ``as_of_month`` wins, a
group short of the fit threshold writes nothing new, and the EMA keeps nine
tenths of a wrong centroid. So this deletes every ACE/PA row from the
resolution, score and learned tables (writing an audit CSV of each first);
``compute_resolutions`` then re-resolves under the settled rule and the chain
rebuilds whatever the corrected outcomes support.

Idempotent: a second run finds nothing. Usage::

    python -m scripts.reset_conflict_displacement --db duckdb:///path/resolver.duckdb \\
        --audit-dir audit/conflict_displacement [--audit-only] [--summary-out plan.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

HAZARD = "ACE"
METRIC = "PA"

#: Tables keyed by question, reached through ``questions``.
QUESTION_TABLES = (
    "resolutions", "scores", "eiv_scores", "resolution_vintages",
    "baseline_scored_forecasts",
)
#: Tables keyed by (hazard_code, metric) directly. ``bucket_centroids`` keeps
#: its wildcard seed rows (``hazard_code = '*'``), which the reset falls back to.
GROUP_TABLES = (
    "family_recalibration", "calibration_weights", "calibration_advice",
    "bucket_centroids",
)


def _tables(con) -> set[str]:
    return {
        str(r[0]).lower()
        for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }


def _columns(con, table: str) -> set[str]:
    return {str(r[1]).lower() for r in con.execute(f"PRAGMA table_info('{table}')").fetchall()}


def _where(con, table: str) -> str | None:
    cols = _columns(con, table)
    if table in QUESTION_TABLES:
        if "question_id" not in cols:
            return None
        return (
            "question_id IN (SELECT question_id FROM questions "
            f"WHERE upper(hazard_code) = '{HAZARD}' AND upper(metric) = '{METRIC}')"
        )
    if {"hazard_code", "metric"} <= cols:
        return f"upper(hazard_code) = '{HAZARD}' AND upper(metric) = '{METRIC}'"
    return None


def reset(con, *, audit_dir: Path | None = None, audit_only: bool = False) -> dict[str, Any]:
    """Write the audit CSVs, then (unless ``audit_only``) delete the rows.

    Returns ``{table: rows}`` for every table touched. Never raises on a
    missing table or column."""

    present = _tables(con)
    summary: dict[str, Any] = {"audit_only": audit_only, "tables": {}}
    if "questions" not in present:
        summary["reason"] = "questions table absent"
        return summary
    if audit_dir is not None:
        audit_dir.mkdir(parents=True, exist_ok=True)
    for table in QUESTION_TABLES + GROUP_TABLES:
        if table not in present:
            continue
        where = _where(con, table)
        if where is None:
            continue
        cur = con.execute(f"SELECT * FROM {table} WHERE {where}")
        names = [d[0] for d in cur.description]
        rows = cur.fetchall()
        summary["tables"][table] = len(rows)
        if audit_dir is not None:
            with (audit_dir / f"{table}.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(names)
                writer.writerows(rows)
        if rows and not audit_only:
            con.execute(f"DELETE FROM {table} WHERE {where}")
    if "resolutions" in present and "questions" in present:
        by_source = con.execute(
            f"""
            SELECT COALESCE(r.source_desc, 'null'), COUNT(*)
            FROM resolutions r JOIN questions q USING (question_id)
            WHERE upper(q.hazard_code) = '{HAZARD}' AND upper(q.metric) = '{METRIC}'
            GROUP BY 1 ORDER BY 1
            """
        ).fetchall()
        summary["resolutions_left_by_source"] = {str(k): int(v) for k, v in by_source}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True)
    parser.add_argument("--audit-dir", default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    import duckdb

    path = args.db.replace("duckdb:///", "", 1) if args.db.startswith("duckdb:///") else args.db
    con = duckdb.connect(path)
    try:
        summary = reset(
            con,
            audit_dir=Path(args.audit_dir) if args.audit_dir else None,
            audit_only=args.audit_only,
        )
    finally:
        con.close()
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
