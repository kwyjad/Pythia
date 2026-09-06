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

The repair is collision-safe. ``facts_resolved`` and ``facts_deltas`` carry
a UNIQUE index that INCLUDES ``publication_date``, so clamping a future date
can land on a key another row already holds. Such a row is MERGED AWAY — the
correct row is already there and the future-dated one is the same fact
wearing a date that has not happened — rather than raising, which is what
killed the 2026-09 pass halfway through and left 296 future-dated rows in
``facts_deltas`` while the step reported completed. A table whose repair
fails no longer costs the other tables theirs.

Beyond the merge nothing is deleted: the figure is right, the date beside it
was not. Counts of repaired, merged and untouched rows are logged per table.
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


#: Unique keys the repair can collide with. ``duckdb_io`` declares these as
#: UNIQUE INDEXes rather than table constraints, so ``duckdb_constraints()``
#: does not list them — the index SQL is parsed instead, with these literals
#: as the fallback for a database whose index is named differently.
_KNOWN_UNIQUE_KEYS: dict[str, tuple[str, ...]] = {
    "facts_resolved": (
        "event_id", "iso3", "hazard_code", "metric", "as_of_date",
        "publication_date", "source_id", "series_semantics", "ym",
    ),
    "facts_deltas": (
        "event_id", "iso3", "hazard_code", "metric", "as_of_date",
        "publication_date", "source_id", "ym",
    ),
}


def _unique_key(con, table: str) -> list[str]:
    """The unique key columns of ``table`` that include ``publication_date``.

    Empty when the table has no such key, in which case the repair cannot
    collide and no merge is needed. Never raises.
    """

    columns = _columns(con, table)
    candidates: list[list[str]] = []
    try:
        rows = con.execute(
            "SELECT sql FROM duckdb_indexes() WHERE lower(table_name) = ? "
            "AND is_unique",
            [table.lower()],
        ).fetchall()
    except Exception:  # noqa: BLE001 - introspection is best effort
        rows = []
    for (sql,) in rows:
        text = str(sql or "")
        start, end = text.find("("), text.rfind(")")
        if start == -1 or end <= start:
            continue
        parsed = [c.strip().strip('"') for c in text[start + 1 : end].split(",")]
        candidates.append([c for c in parsed if c])
    try:
        for (cols,) in con.execute(
            "SELECT constraint_column_names FROM duckdb_constraints() "
            "WHERE lower(table_name) = ? AND constraint_type = 'UNIQUE'",
            [table.lower()],
        ).fetchall():
            candidates.append([str(c) for c in (cols or [])])
    except Exception:  # noqa: BLE001
        pass
    candidates.append(list(_KNOWN_UNIQUE_KEYS.get(table, ())))
    for key in candidates:
        if "publication_date" in key and set(key).issubset(columns):
            return list(key)
    return []


def _apply_repair(
    con, table: str, key: list[str], target_sql: str, params: list[Any]
) -> int:
    """Move every future-dated row to the date ``target_sql`` gives it.

    ``target_sql`` selects ``(rid, new_pub)`` for the rows to move, where
    ``rid`` is the row's ``rowid``. Where a move would land on a key another
    row already occupies, the future-dated row is MERGED AWAY — deleted —
    rather than inserted beside it: the correct row is already there, and
    the future-dated one is the same fact wearing a date that has not
    happened. Two future rows repairing onto one key collapse the same way,
    keeping the lower rowid so a re-run is deterministic.

    Returns the number of rows whose date changed or that were merged away.
    The deletes run BEFORE the update and the target set is rebuilt in
    between, because a DuckDB rowid is a physical position and a delete can
    move it. Before Sept 2026 the repair did a bare UPDATE and the first
    collision raised a ConstraintException, which killed the pass halfway
    through — facts_resolved was repaired, facts_deltas was not, and 296
    future-dated rows survived to the next run to do it again.
    """

    con.execute("CREATE OR REPLACE TEMP TABLE _pubfix AS " + target_sql, params)
    planned = int(con.execute("SELECT COUNT(*) FROM _pubfix").fetchone()[0])
    if not planned:
        con.execute("DROP TABLE IF EXISTS _pubfix")
        return 0

    if key:
        others = " AND ".join(
            'o."{c}" IS NOT DISTINCT FROM r."{c}"'.format(c=c)
            for c in key
            if c != "publication_date"
        ) or "TRUE"
        pairs = " AND ".join(
            'o."{c}" IS NOT DISTINCT FROM r."{c}"'.format(c=c)
            for c in key
            if c != "publication_date"
        ) or "TRUE"
        # (a) a repaired row whose key another row already holds is merged away
        con.execute(
            'DELETE FROM "{t}" WHERE rowid IN ('
            "  SELECT f.rid FROM _pubfix f"
            '  JOIN "{t}" r ON r.rowid = f.rid'
            "  WHERE EXISTS ("
            '    SELECT 1 FROM "{t}" o'
            "    WHERE o.rowid <> f.rid AND {others}"
            "      AND o.publication_date IS NOT DISTINCT FROM f.new_pub"
            "  )"
            ")".format(t=table, others=others)
        )
        # (b) two future rows repairing onto one key: keep the lower rowid
        con.execute(
            'DELETE FROM "{t}" WHERE rowid IN ('
            "  SELECT f.rid FROM _pubfix f"
            '  JOIN "{t}" r ON r.rowid = f.rid'
            "  WHERE EXISTS ("
            "    SELECT 1 FROM _pubfix g"
            '    JOIN "{t}" o ON o.rowid = g.rid'
            "    WHERE g.rid < f.rid AND {pairs}"
            "      AND g.new_pub IS NOT DISTINCT FROM f.new_pub"
            "  )"
            ")".format(t=table, pairs=pairs)
        )
        # The rowids in _pubfix may have moved under the deletes above, so
        # the surviving targets are recomputed rather than reused.
        con.execute("DROP TABLE IF EXISTS _pubfix")
        con.execute("CREATE OR REPLACE TEMP TABLE _pubfix AS " + target_sql, params)

    remaining = int(con.execute("SELECT COUNT(*) FROM _pubfix").fetchone()[0])
    if remaining:
        con.execute(
            'UPDATE "{t}" SET publication_date = '
            "(SELECT new_pub FROM _pubfix WHERE rid = \"{t}\".rowid) "
            'WHERE rowid IN (SELECT rid FROM _pubfix)'.format(t=table)
        )
    con.execute("DROP TABLE IF EXISTS _pubfix")
    return planned


def _future_predicate(column: str, *, alias: str | None = None) -> str:
    # TRY_CAST tolerates the odd unparseable string; an unparseable date is
    # not a future date and is left alone. ``alias`` qualifies the column
    # for a joined statement — quoting "d.publication_date" whole names a
    # column that does not exist.
    qualified = f'{alias}."{column}"' if alias else f'"{column}"'
    return f"TRY_CAST({qualified} AS DATE) > CAST(? AS DATE)"


def repair_table(con, table: str, today: dt.date, *, dry_run: bool = False) -> dict[str, int]:
    """Repair one table. Returns ``{rows, future, repaired, untouched, ...}``."""

    columns = _columns(con, table)
    if "publication_date" not in columns:
        return {"rows": 0, "future": 0, "repaired": 0, "untouched": 0, "skipped": 1}
    iso_today = today.isoformat()
    total = int(con.execute('SELECT COUNT(*) FROM "{t}"'.format(t=table)).fetchone()[0])

    def _future_count() -> int:
        return int(
            con.execute(
                'SELECT COUNT(*) FROM "{t}" WHERE {p}'.format(
                    t=table, p=_future_predicate("publication_date")
                ),
                [iso_today],
            ).fetchone()[0]
        )

    future = _future_count()
    counts = {
        "rows": total, "future": future, "repaired": 0, "untouched": total - future,
        "from_fact": 0, "from_created_at": 0, "from_today": 0, "merged_away": 0,
    }
    if future == 0 or dry_run:
        return counts

    key = _unique_key(con, table)
    rows_before = total

    # 1. A delta takes the date of the fact it came from, when that has passed.
    if table == "facts_deltas" and "facts_resolved" in _tables(con):
        fact_columns = _columns(con, "facts_resolved")
        if set(_DELTA_KEY).issubset(columns) and set(_DELTA_KEY).issubset(fact_columns):
            join = " AND ".join(
                'd."{c}" = src."{c}"'.format(c=c) for c in _DELTA_KEY
            )
            target = (
                "SELECT d.rowid AS rid, src.pub AS new_pub FROM facts_deltas d JOIN ("
                "  SELECT {sel}, MAX(f.publication_date) AS pub FROM facts_resolved f"
                "  WHERE TRY_CAST(f.publication_date AS DATE) <= CAST(? AS DATE)"
                "  GROUP BY {grp}"
                ") AS src ON {join} WHERE {pred}"
            ).format(
                sel=", ".join('f."{c}" AS "{c}"'.format(c=c) for c in _DELTA_KEY),
                grp=", ".join('f."{c}"'.format(c=c) for c in _DELTA_KEY),
                join=join,
                pred=_future_predicate("publication_date", alias="d"),
            )
            before = future
            _apply_repair(con, table, key, target, [iso_today, iso_today])
            counts["from_fact"] = before - _future_count()

    # 2. The row's own created_at, when it has one and it has passed.
    if "created_at" in columns:
        before = _future_count()
        target = (
            "SELECT rowid AS rid, "
            "strftime(CAST(created_at AS DATE), '%Y-%m-%d') AS new_pub "
            'FROM "{t}" WHERE {pred} AND created_at IS NOT NULL '
            "AND CAST(created_at AS DATE) <= CAST(? AS DATE)"
        ).format(t=table, pred=_future_predicate("publication_date"))
        _apply_repair(con, table, key, target, [iso_today, iso_today])
        counts["from_created_at"] = before - _future_count()

    # 3. Today: the day this pass first saw a row nothing else can date.
    before = _future_count()
    if before:
        target = (
            "SELECT rowid AS rid, CAST(? AS VARCHAR) AS new_pub "
            'FROM "{t}" WHERE {pred}'
        ).format(t=table, pred=_future_predicate("publication_date"))
        _apply_repair(con, table, key, target, [iso_today, iso_today])
    counts["from_today"] = before - _future_count()
    counts["repaired"] = counts["from_fact"] + counts["from_created_at"] + counts["from_today"]
    rows_after = int(con.execute('SELECT COUNT(*) FROM "{t}"'.format(t=table)).fetchone()[0])
    # A merged-away row is repaired too — the correct row was already there
    # and the future-dated one was the same fact wearing an impossible date.
    counts["merged_away"] = max(0, rows_before - rows_after)
    counts["rows"] = rows_after
    return counts


def repair(con, *, today: dt.date | None = None, dry_run: bool = False) -> RepairReport:
    today = today or dt.date.today()
    report = RepairReport(today=today.isoformat())
    present = _tables(con)
    for table in TABLES:
        if table not in present:
            continue
        # One table's failure must never cost the others their repair. The
        # 2026-09 run died on the first facts_deltas collision and left 296
        # future-dated rows in that table alone, having already fixed
        # facts_resolved — and the step it ran in was marked completed.
        try:
            counts = repair_table(con, table, today, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001
            LOG.error(
                "[repair_publication_dates] %s: repair FAILED (%s: %s) — "
                "the remaining tables are still repaired",
                table, type(exc).__name__, exc,
            )
            report.tables[table] = {
                "rows": 0, "future": 0, "repaired": 0, "untouched": 0,
                "error": f"{type(exc).__name__}: {exc}",
            }
            continue
        report.tables[table] = counts
        LOG.info(
            "[repair_publication_dates] %s: %d rows, %d dated after %s, %d repaired "
            "(%d from the source fact, %d from created_at, %d set to today, "
            "%d merged into a row that already held the repaired key), %d untouched%s",
            table, counts["rows"], counts["future"], today, counts["repaired"],
            counts.get("from_fact", 0), counts.get("from_created_at", 0),
            counts.get("from_today", 0), counts.get("merged_away", 0),
            counts["untouched"],
            " [dry run: nothing written]" if dry_run else "",
        )
    failed = [t for t, c in report.tables.items() if c.get("error")]
    if failed:
        LOG.error(
            "[repair_publication_dates] %d table(s) could not be repaired: %s",
            len(failed), ", ".join(failed),
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
