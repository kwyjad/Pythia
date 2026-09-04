# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Repair facts whose ``publication_date`` lies in the future.

A publication date after today is false in the plain sense: nothing has been
published on a day that has not happened. Two paths wrote such rows.
``enrich`` raised every publication date to ``as_of_date`` — the END of the
period a figure describes, which for an IPC projection window or the current
month is in the future — and then clamped it to the run date, so a FEWS NET
projection about March 2027 said it was published today, every run. And the
storage layer filled a missing publication date on ``facts_deltas`` from the
same period end, uncapped, so the deltas table carried ``2027-03-01`` as a
publication date and the freshness report called the table fresh.

Both writers are fixed. This pass repairs what they already wrote, in place,
and runs on every scheduled Resolver Update, idempotently:

* ``facts_deltas`` takes the publication date of the fact it was derived from
  (same ``iso3``, ``hazard_code``, ``metric``, ``ym``) when that date has
  passed;
* otherwise, on any table, the row's own ``created_at`` — the day it was
  first stored, which is the latest it can have been published;
* otherwise ``today``.

Nothing is deleted: the figure is right, the date beside it was not. Counts
of repaired and untouched rows are logged per table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

LOG = logging.getLogger(__name__)

#: Tables with a publication_date column the pipeline writes.
TABLES: tuple[str, ...] = ("facts_resolved", "facts_deltas", "emdat_pa")

#: Columns that identify the fact a delta was derived from.
_DELTA_KEY = ("iso3", "hazard_code", "metric", "ym")


@dataclass
class RepairReport:
    today: str
    tables: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def repaired(self) -> int:
        return sum(t.get("repaired", 0) for t in self.tables.values())

    @property
    def untouched(self) -> int:
        return sum(t.get("untouched", 0) for t in self.tables.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "today": self.today,
            "repaired": self.repaired,
            "untouched": self.untouched,
            "tables": dict(self.tables),
        }


def _tables(con) -> set[str]:
    return {
        str(r[0])
        for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }


def _columns(con, table: str) -> set[str]:
    return {str(r[1]) for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()}


def _future_predicate(column: str) -> str:
    # TRY_CAST tolerates the odd unparseable string; an unparseable date is
    # not a future date and is left alone.
    return f'TRY_CAST("{column}" AS DATE) > CAST(? AS DATE)'


def repair_table(con, table: str, today: dt.date, *, dry_run: bool = False) -> dict[str, int]:
    """Repair one table. Returns ``{rows, future, repaired, untouched, ...}``."""

    columns = _columns(con, table)
    if "publication_date" not in columns:
        return {"rows": 0, "future": 0, "repaired": 0, "untouched": 0, "skipped": 1}
    iso_today = today.isoformat()
    total = int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
    future = int(
        con.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE {_future_predicate("publication_date")}',
            [iso_today],
        ).fetchone()[0]
    )
    counts = {
        "rows": total, "future": future, "repaired": 0, "untouched": total - future,
        "from_fact": 0, "from_created_at": 0, "from_today": 0,
    }
    if future == 0 or dry_run:
        return counts

    # 1. A delta takes the date of the fact it came from, when that has passed.
    if table == "facts_deltas" and "facts_resolved" in _tables(con):
        fact_columns = _columns(con, "facts_resolved")
        if set(_DELTA_KEY).issubset(columns) and set(_DELTA_KEY).issubset(fact_columns):
            before = future
            con.execute(
                f"""
                UPDATE facts_deltas AS d
                SET publication_date = src.pub
                FROM (
                    SELECT {", ".join(f'f."{c}" AS "{c}"' for c in _DELTA_KEY)},
                           MAX(f.publication_date) AS pub
                    FROM facts_resolved f
                    WHERE TRY_CAST(f.publication_date AS DATE) <= CAST(? AS DATE)
                    GROUP BY {", ".join(f'f."{c}"' for c in _DELTA_KEY)}
                ) AS src
                WHERE {" AND ".join(f'd."{c}" = src."{c}"' for c in _DELTA_KEY)}
                  AND TRY_CAST(d.publication_date AS DATE) > CAST(? AS DATE)
                """,
                [iso_today, iso_today],
            )
            remaining = int(
                con.execute(
                    f'SELECT COUNT(*) FROM "{table}" WHERE {_future_predicate("publication_date")}',
                    [iso_today],
                ).fetchone()[0]
            )
            counts["from_fact"] = before - remaining

    # 2. The row's own created_at, when it has one and it has passed.
    if "created_at" in columns:
        before = int(
            con.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE {_future_predicate("publication_date")}',
                [iso_today],
            ).fetchone()[0]
        )
        con.execute(
            f"""
            UPDATE "{table}"
            SET publication_date = strftime(CAST(created_at AS DATE), '%Y-%m-%d')
            WHERE {_future_predicate("publication_date")}
              AND created_at IS NOT NULL
              AND CAST(created_at AS DATE) <= CAST(? AS DATE)
            """,
            [iso_today, iso_today],
        )
        remaining = int(
            con.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE {_future_predicate("publication_date")}',
                [iso_today],
            ).fetchone()[0]
        )
        counts["from_created_at"] = before - remaining

    # 3. Today: the day this pass first saw a row nothing else can date.
    before = int(
        con.execute(
            f'SELECT COUNT(*) FROM "{table}" WHERE {_future_predicate("publication_date")}',
            [iso_today],
        ).fetchone()[0]
    )
    if before:
        con.execute(
            f'UPDATE "{table}" SET publication_date = ? WHERE {_future_predicate("publication_date")}',
            [iso_today, iso_today],
        )
    counts["from_today"] = before
    counts["repaired"] = counts["from_fact"] + counts["from_created_at"] + counts["from_today"]
    return counts


def repair(con, *, today: dt.date | None = None, dry_run: bool = False) -> RepairReport:
    today = today or dt.date.today()
    report = RepairReport(today=today.isoformat())
    present = _tables(con)
    for table in TABLES:
        if table not in present:
            continue
        counts = repair_table(con, table, today, dry_run=dry_run)
        report.tables[table] = counts
        LOG.info(
            "[repair_publication_dates] %s: %d rows, %d dated after %s, %d repaired "
            "(%d from the source fact, %d from created_at, %d set to today), %d untouched%s",
            table, counts["rows"], counts["future"], today, counts["repaired"],
            counts.get("from_fact", 0), counts.get("from_created_at", 0),
            counts.get("from_today", 0), counts["untouched"],
            " [dry run: nothing written]" if dry_run else "",
        )
    return report


def count_future(con, *, today: dt.date | None = None) -> dict[str, int]:
    """Rows per table still dated after ``today`` — the acceptance query."""

    today = today or dt.date.today()
    out: dict[str, int] = {}
    present = _tables(con)
    for table in TABLES:
        if table not in present or "publication_date" not in _columns(con, table):
            continue
        out[table] = int(
            con.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE {_future_predicate("publication_date")}',
                [today.isoformat()],
            ).fetchone()[0]
        )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", default=None, help="DuckDB URL or path (default: resolver config)")
    parser.add_argument("--dry-run", action="store_true", help="Count, log, write nothing")
    parser.add_argument("--today", default=None, help="Override today's date (YYYY-MM-DD; tests)")
    parser.add_argument("--summary-out", default=None, help="Write the report as JSON here")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()

    from resolver.db.duckdb_io import close_db, get_db

    con = get_db(args.db)
    try:
        report = repair(con, today=today, dry_run=args.dry_run)
        remaining = count_future(con, today=today)
    finally:
        close_db(con)

    for table, n in remaining.items():
        if n:
            LOG.error(
                "[repair_publication_dates] %s still holds %d rows dated after %s",
                table, n, today,
            )
    if args.summary_out:
        import json
        from pathlib import Path

        payload = report.as_dict()
        payload["remaining_future"] = remaining
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(payload, indent=2, sort_keys=True))
    # A dry run reports and never fails; a real run that leaves future rows
    # behind has not done its job.
    return 1 if (not args.dry_run and any(remaining.values())) else 0


if __name__ == "__main__":
    sys.exit(main())
