# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Relabel ACLED battle-only fatality rows that were stored as ``fatalities``.

The ACLED connector writes ``fatalities_battle_month``: deaths in events of
type *Battles* only. Until Sept 2026 the ACLED adapter renamed that metric to
``fatalities`` on its way into ``facts_resolved``, and ``compute_resolutions``
read ``facts_resolved`` BEFORE ``acled_monthly_fatalities`` — the all-types
series every conflict question is worded against and every conflict base
rate is drawn from. So 27 of 32 August ACE/FATALITIES questions resolved to a
battle-only count, a median 0.42 of the all-types level the prompt showed.

The adapter no longer renames. This pass repairs what it already wrote, in
place, and runs on every scheduled Resolver Update, idempotently:

* an ACLED ``fatalities`` row in ``facts_resolved`` or ``facts_deltas`` is
  relabelled ``fatalities_battle_month`` — the figure is right, its name was
  not;
* where a correctly named twin already holds the same
  ``(ym, iso3, hazard_code[, series_semantics])`` key, the mislabelled row is
  MERGED AWAY (deleted) rather than raising on the unique constraint: the
  twin is the same fact, written by a run after the fix.

An ACLED row is one whose ``publisher`` or ``source_id`` names ACLED, or whose
hazard is a conflict hazard (``ACE``/``ACO``) — no other connector writes a
fatalities figure for those, and none of them would be an all-types count.
Counts of relabelled, merged and untouched rows are logged per table.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

OLD_METRIC = "fatalities"
NEW_METRIC = "fatalities_battle_month"
CONFLICT_HAZARDS = ("ACE", "ACO")
TABLES = ("facts_resolved", "facts_deltas")


def _tables(con) -> set[str]:
    return {
        str(r[0])
        for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    }


def _columns(con, table: str) -> set[str]:
    return {str(r[1]) for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()}


def _acled_predicate(columns: set[str], alias: str = "") -> str:
    p = f"{alias}." if alias else ""
    parts = [f"upper({p}hazard_code) IN ({', '.join(repr(h) for h in CONFLICT_HAZARDS)})"]
    for col in ("publisher", "source_id"):
        if col in columns:
            parts.append(f"lower(COALESCE({p}{col}, '')) LIKE '%acled%'")
    return "(" + " OR ".join(parts) + ")"


def repair_table(con, table: str, *, dry_run: bool = False) -> dict[str, int]:
    """Relabel one table. Never raises for a missing table or column."""

    columns = _columns(con, table)
    if not {"metric", "ym", "iso3", "hazard_code"}.issubset(columns):
        return {"relabelled": 0, "merged": 0, "untouched": 0}
    acled = _acled_predicate(columns)
    target = f"lower(metric) = '{OLD_METRIC}' AND {acled}"
    planned = int(con.execute(f'SELECT COUNT(*) FROM "{table}" WHERE {target}').fetchone()[0])
    untouched = int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]) - planned
    if not planned:
        return {"relabelled": 0, "merged": 0, "untouched": untouched}

    key = ["ym", "iso3", "hazard_code"]
    if "series_semantics" in columns:
        key.append("series_semantics")
    same_key = " AND ".join(f"t.{c} IS NOT DISTINCT FROM r.{c}" for c in key)
    twin = (
        f'EXISTS (SELECT 1 FROM "{table}" t WHERE lower(t.metric) = \'{NEW_METRIC}\' '
        f"AND {same_key})"
    )
    old_r = f"lower(r.metric) = '{OLD_METRIC}' AND {_acled_predicate(columns, 'r')}"
    merged = int(
        con.execute(f'SELECT COUNT(*) FROM "{table}" r WHERE {old_r} AND {twin}').fetchone()[0]
    )
    if dry_run:
        return {"relabelled": planned - merged, "merged": merged, "untouched": untouched}

    con.execute(f'DELETE FROM "{table}" r WHERE {old_r} AND {twin}')
    stamp = ", updated_at = now()" if "updated_at" in columns else ""
    con.execute(f"UPDATE \"{table}\" SET metric = '{NEW_METRIC}'{stamp} WHERE {target}")
    return {"relabelled": planned - merged, "merged": merged, "untouched": untouched}


def count_mislabelled(con) -> dict[str, int]:
    """ACLED rows still named ``fatalities`` per table — the acceptance query."""

    present = _tables(con)
    out: dict[str, int] = {}
    for table in TABLES:
        if table not in present:
            continue
        columns = _columns(con, table)
        if "metric" not in columns:
            continue
        out[table] = int(
            con.execute(
                f'SELECT COUNT(*) FROM "{table}" WHERE lower(metric) = \'{OLD_METRIC}\' '
                f"AND {_acled_predicate(columns)}"
            ).fetchone()[0]
        )
    return out


def repair(con, *, dry_run: bool = False) -> dict[str, Any]:
    present = _tables(con)
    report: dict[str, Any] = {"tables": {}, "errors": {}}
    for table in TABLES:
        if table not in present:
            continue
        try:
            counts = repair_table(con, table, dry_run=dry_run)
        except Exception as exc:  # noqa: BLE001 - one table never costs the other
            report["errors"][table] = repr(exc)
            LOG.error("[repair_acled_fatalities_metric] %s failed: %r", table, exc)
            continue
        report["tables"][table] = counts
        LOG.info(
            "[repair_acled_fatalities_metric] %s: relabelled=%d merged=%d untouched=%d",
            table, counts["relabelled"], counts["merged"], counts["untouched"],
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--db", default=None, help="DuckDB URL or path (default: resolver config)")
    parser.add_argument("--dry-run", action="store_true", help="Count, log, write nothing")
    parser.add_argument("--summary-out", default=None, help="Write the report as JSON here")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from resolver.db.duckdb_io import close_db, get_db

    con = get_db(args.db)
    try:
        report = repair(con, dry_run=args.dry_run)
        remaining = count_mislabelled(con)
    finally:
        close_db(con)
    report["remaining_mislabelled"] = remaining
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(report, indent=2, sort_keys=True))
    failed = bool(report["errors"]) or (not args.dry_run and any(remaining.values()))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
