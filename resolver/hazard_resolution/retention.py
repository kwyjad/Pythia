# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Retention for the machine's raw caches.

Why this exists: the ``haz_raw_*`` caches were append-only in practice. A
wall clock inside the hashed payload defeated the content-hash dedup (fixed
in ``sources.py``), so every nightly backcast and every monthly live run
appended a full duplicate of every record it touched -- ReliefWeb bodies
included, at up to 720 KB a cell. The canonical DB went from 3.0 GB on
2026-08-06 to 17.7 GB on 2026-09-01, and the release publish aborted.

Fixing the hash stops the growth. It does not undo it, and it never will:
DuckDB reuses freed blocks but does not truncate the file, so even deleting
every duplicate row leaves the bytes on disk. Reclaiming needs two separate
things, and this module does the first:

1. collapse each cache to the row a reader would actually take, then
2. copy the database to a fresh file (``scripts.ci.build_release_db``'s
   ``compact_database``) -- which is what returns the space.

Ordering matters for more than tidiness: rebuild first and the copy's source
is ~2 GB rather than 17.7 GB, which is the difference between a routine job
and one that needs ~40 GB of runner scratch.

    THE COMPACTION KEY IS (record_id, iso3, ym, hazard), NOT record_id.

``reliefweb_docs`` sets ``record_id = f"rw-{doc_id}"``, so the SAME document
is legitimately cached once per cell it was fetched for, with a different
payload each time (the payload embeds iso3/ym/hazard).
:func:`sources.load_raw_records` filters on the COLUMNS first and only then
applies ``PARTITION BY record_id`` within that filtered set -- so partitioning
globally on ``record_id`` alone would keep one cell's copy and delete every
other cell's documents. Keeping the newest per the finer key always retains
the global newest per ``record_id`` too, so every coarser filtered read
returns exactly what it returned before.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Iterable

from resolver.hazard_resolution.schema import (
    _CORE_TABLE_DDL,
    RAW_SOURCES,
    ensure_haz_schema,
    raw_ddl,
    raw_table_name,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

LOG = logging.getLogger(__name__)

# population has a different shape (no record_id, no content_hash, UNIQUE on
# (iso3, year, source)) and is already delete-then-reloaded by population.py,
# so it is not a revision cache and there is nothing here to collapse.
COMPACTABLE_SOURCES = tuple(s for s in RAW_SOURCES if s != "population")

# The key a reader would group by. See the module docstring: the finer key is
# the whole point, and narrowing it deletes data.
_PARTITION = ("record_id", "iso3", "ym", "hazard")


def _assert_compactable(source: str) -> str:
    """Hard allowlist.

    This module must be structurally incapable of touching the machine's
    verdicts (``haz_resolutions``), its audit log (``haz_revisions``), its
    extraction cache (``haz_doc_extractions``) or -- above all -- its resume
    ledger (``haz_backcast_progress``), whose loss would make the backcast
    re-walk history it has already paid for.
    """
    if source not in COMPACTABLE_SOURCES:
        raise ValueError(
            f"refusing to compact {source!r}; compactable sources are "
            f"{list(COMPACTABLE_SOURCES)}"
        )
    table = raw_table_name(source)
    if not table.startswith("haz_raw_"):  # pragma: no cover - belt and braces
        raise ValueError(f"refusing to compact non-raw table {table!r}")
    return table


def _table_exists(con: "duckdb.DuckDBPyConnection", table: str) -> bool:
    row = con.execute(
        "SELECT COUNT(*) FROM information_schema.tables "
        "WHERE table_schema = 'main' AND table_name = ?",
        [table],
    ).fetchone()
    return bool(row and row[0])


def _keep_revisions(rulebook: Any | None) -> int:
    """How many revisions per cell to keep — ``raw_cache.keep_revisions_per_record``.

    The rulebook owns the number; this module used to hard-code 1 while the
    key was validated and read by nothing, which is the "hard-coding a
    rulebook value in Python" bug the rulebook's own docstring names.
    """

    if rulebook is None:
        try:
            from resolver.hazard_resolution.rulebook import load_rulebook

            rulebook = load_rulebook()
        except Exception as exc:  # noqa: BLE001 - keep 1 is the safe floor
            LOG.warning("[retention] rulebook unavailable (%s); keeping 1 revision", exc)
            return 1
    try:
        return max(1, int(rulebook.get("raw_cache.keep_revisions_per_record")))
    except Exception:  # noqa: BLE001 - older rulebooks have no such key
        return 1


def compact_raw_cache(
    con: "duckdb.DuckDBPyConnection",
    source: str,
    *,
    apply: bool = False,
    keep: int = 1,
) -> dict[str, Any]:
    """Collapse one raw cache to the newest ``keep`` row(s) per cell.

    ``apply=False`` (the default) reports the plan and writes nothing.
    """
    table = _assert_compactable(source)
    if not _table_exists(con, table):
        return {"source": source, "table": table, "present": False,
                "rows_before": 0, "rows_after": 0, "removed": 0, "applied": False}

    keep = max(1, int(keep))
    partition = ", ".join(_PARTITION)
    rows_before = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    # What a reader would keep. Reported alongside rows_after so the two can be
    # eyeballed against each other in a dry run.
    distinct_cells = int(
        con.execute(
            f"SELECT COUNT(*) FROM (SELECT DISTINCT {partition} FROM {table})"
        ).fetchone()[0]
    )
    kept_plan = int(
        con.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT ROW_NUMBER() OVER (
                    PARTITION BY {partition}
                    ORDER BY retrieved_at DESC, content_hash DESC
                ) AS rn FROM {table}
            ) WHERE rn <= {keep}
            """
        ).fetchone()[0]
    )

    if not apply:
        return {
            "source": source, "table": table, "present": True,
            "rows_before": rows_before, "rows_after": kept_plan,
            "removed": rows_before - kept_plan,
            "distinct_cells": distinct_cells, "keep": keep, "applied": False,
        }

    tmp = f"{table}__compact"
    old = f"{table}__old"
    # A previous run killed mid-swap can leave either scratch table behind.
    con.execute(f"DROP TABLE IF EXISTS {tmp}")
    con.execute(f"DROP TABLE IF EXISTS {old}")
    # Recreate from the SAME DDL, so UNIQUE (record_id, content_hash) survives
    # the rebuild. CREATE TABLE AS SELECT would silently drop it.
    con.execute(raw_ddl(source).replace(table, tmp, 1))
    con.execute(
        f"""
        INSERT INTO {tmp}
        SELECT record_id, iso3, ym, hazard, payload_json, content_hash,
               source_url, retrieved_at
        FROM {table}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY {partition}
            -- content_hash breaks ties so two rows sharing a timestamp
            -- collapse the same way on every run.
            ORDER BY retrieved_at DESC, content_hash DESC
        ) <= {keep}
        """
    )
    rows_after = int(con.execute(f"SELECT COUNT(*) FROM {tmp}").fetchone()[0])
    # The swap is one transaction and the live name is never unbound: a
    # kill between a DROP and a RENAME used to leave the DB with no
    # haz_raw_reliefweb_docs at all, which ensure_haz_schema then recreated
    # empty — the whole document cache gone, and the job carried on.
    #
    # The lookup index (schema._INDEX_DDL) depends on the live table, and
    # DuckDB refuses to rename or drop a table something depends on — so it
    # is dropped first and recreated on the new table, inside the same
    # transaction, under the same name.
    index_name = f"idx_{table}_lookup"
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(f"DROP INDEX IF EXISTS {index_name}")
        con.execute(f"ALTER TABLE {table} RENAME TO {old}")
        con.execute(f"ALTER TABLE {tmp} RENAME TO {table}")
        con.execute(f"DROP TABLE {old}")
        con.execute(
            f"CREATE INDEX IF NOT EXISTS {index_name} ON {table} (hazard, ym, iso3)"
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise

    LOG.info(
        "[retention] %s: %d -> %d rows (%d duplicate revisions removed)",
        table, rows_before, rows_after, rows_before - rows_after,
    )
    return {
        "source": source, "table": table, "present": True,
        "rows_before": rows_before, "rows_after": rows_after,
        "removed": rows_before - rows_after,
        "distinct_cells": distinct_cells, "keep": keep, "applied": True,
    }


REVISION_LOG_DDL = """
CREATE TABLE IF NOT EXISTS haz_revision_collapse_log (
    collapsed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    rows_before BIGINT,
    rows_after BIGINT,
    rows_removed BIGINT
)
"""

# The rule an old revision row fired, read from its provenance when the
# rule_fired column (added Oct 2026) is empty.
_REVISION_RULE_SQL = (
    "COALESCE(rule_fired, json_extract_string(detail_json, '$.observed_rule_fired'), "
    "json_extract_string(detail_json, '$.observed_status'))"
)


# Physical rows read per committed batch when the table is rebuilt. A
# pre-October row is ~75 KB, so a batch holds at most ~75 MB of JSON in
# Python and writes a few hundred KB.
REVISION_BATCH_ROWS = 1000

_REVISION_COLUMNS = (
    "iso3", "year", "month", "hazard", "source", "source_ref",
    "old_value", "new_value", "detail_json", "observed_at", "rule_fired",
)


def _narrow_revisions(con: "duckdb.DuckDBPyConnection") -> None:
    """The small columns of every revision, with its rule read out of the JSON.

    A window over ``haz_revisions`` itself carries each row's 75 KB of JSON
    through the sort; over this table it carries a few dozen bytes.
    """

    con.execute(
        "CREATE OR REPLACE TEMP TABLE _haz_revision_narrow AS "
        "SELECT rowid AS rid, iso3, year, month, hazard, source, observed_at, "
        f"new_value, source_ref, {_REVISION_RULE_SQL} AS rule FROM haz_revisions"
    )


def _revision_dup_sql() -> str:
    """Row ids of revisions that repeat the row before them in their cell.

    Reads ``_haz_revision_narrow``; call ``_narrow_revisions`` first.
    """

    return """
        WITH l AS (
            SELECT rid,
                   LAG(rid) OVER w IS NOT NULL
                   AND LAG(new_value) OVER w IS NOT DISTINCT FROM new_value
                   AND LAG(source_ref) OVER w IS NOT DISTINCT FROM source_ref
                   AND LAG(rule) OVER w IS NOT DISTINCT FROM rule AS dup
            FROM _haz_revision_narrow
            WINDOW w AS (PARTITION BY iso3, year, month, hazard, source ORDER BY observed_at, rid)
        )
    """


def collapse_revisions(
    con: "duckdb.DuckDBPyConnection",
    *,
    apply: bool = False,
    batch_rows: int = REVISION_BATCH_ROWS,
) -> dict[str, Any]:
    """Collapse ``haz_revisions`` to the rule its writer now follows.

    Per (cell, source), in observed order, a row is kept only when its
    value, source reference or rule differs from the row before it, and a
    kept row's provenance is replaced by its hash. The nightly backcast and
    every Resolver Update had re-logged the same frozen cells with the full
    provenance on each row: 40% of a 30.6 GB canonical DB in October 2026.

    The table is REBUILT, never updated in place: the first applied run
    (37609780161) did the DELETE and the UPDATE of 12 GB of JSON inside one
    transaction, DuckDB held every old version for the rollback, and it ran
    out of memory at 12.4 GiB with nothing changed. Now the rows to keep are
    marked once over the small columns, copied slimmed into
    ``haz_revisions_rebuild`` in committed batches, and the names are swapped
    in one transaction, the old table renamed aside before it is dropped.
    Idempotent; an applied collapse is recorded in
    ``haz_revision_collapse_log`` with the rows removed.
    """
    out: dict[str, Any] = {"source": "haz_revisions", "present": _table_exists(con, "haz_revisions"),
                           "applied": False}
    if not out["present"]:
        return out
    before = int(con.execute("SELECT COUNT(*) FROM haz_revisions").fetchone()[0])
    _narrow_revisions(con)
    dup_sql = _revision_dup_sql()
    dups = int(con.execute(f"{dup_sql} SELECT COUNT(*) FROM l WHERE dup").fetchone()[0])
    out.update(rows_before=before, rows_after=before - dups, removed=dups)
    if not apply:
        return out
    # A string match, not a JSON parse: this only decides whether to rebuild,
    # and a slimmed row says "provenance_sha256", never "provenance":.
    to_slim = int(con.execute(
        "SELECT COUNT(*) FROM haz_revisions WHERE detail_json LIKE '%\"provenance\":%' "
        "OR detail_json LIKE '%\"evidence_of_absence\":%'").fetchone()[0])
    no_rule = int(con.execute(
        "SELECT COUNT(*) FROM haz_revisions WHERE rule_fired IS NULL").fetchone()[0])
    slimmed = 0
    if dups or to_slim or no_rule:
        slimmed = _rebuild_revisions(con, dup_sql, batch_rows=max(1, int(batch_rows)))
    con.execute(REVISION_LOG_DDL)
    after = int(con.execute("SELECT COUNT(*) FROM haz_revisions").fetchone()[0])
    con.execute(
        "INSERT INTO haz_revision_collapse_log (rows_before, rows_after, rows_removed) VALUES (?, ?, ?)",
        [before, after, before - after],
    )
    out.update(rows_after=after, removed=before - after, applied=True, slimmed=slimmed)
    LOG.info("[retention] haz_revisions: %d -> %d rows (%d removed, %d slimmed)",
             before, after, before - after, slimmed)
    return out


def _slim_row_detail(detail_json: Any) -> tuple[Any, bool]:
    """The writer's own slimming, applied to a stored row.

    ``resolutions.slim_revision_detail`` is what every revision written since
    October 2026 went through, so a compacted row and a new row carry the
    same digest for the same provenance. Returns (detail, slimmed).
    """
    import json

    from resolver.hazard_resolution.resolutions import (
        _BULKY_REVISION_KEYS,
        slim_revision_detail,
    )

    if not isinstance(detail_json, str):
        return detail_json, False
    try:
        detail = json.loads(detail_json)
    except ValueError:
        return detail_json, False
    if not isinstance(detail, dict) or not any(k in detail for k in _BULKY_REVISION_KEYS):
        return detail_json, False
    return json.dumps(slim_revision_detail(detail), default=str), True


def _row_rule(rule_fired: Any, detail_json: Any) -> Any:
    """``rule_fired``, else the rule the pre-column rows kept in their JSON."""
    import json

    if rule_fired is not None or not isinstance(detail_json, str):
        return rule_fired
    try:
        detail = json.loads(detail_json)
    except ValueError:
        return None
    if not isinstance(detail, dict):
        return None
    for key in ("observed_rule_fired", "observed_status"):
        value = detail.get(key)
        if value is not None:
            return value if isinstance(value, str) else json.dumps(value)
    return None


def _rebuild_revisions(con: "duckdb.DuckDBPyConnection", dup_sql: str, *, batch_rows: int) -> int:
    """Copy the kept rows, slimmed, into a fresh table; swap it into place.

    The slimming runs in Python: DuckDB's JSON functions held about fifty
    times the row's size in memory per row, which is how a 12 GB table needed
    more than the runner's 16 GB.
    """
    import pandas as pd

    cols = ", ".join(_REVISION_COLUMNS)
    # A run killed mid-copy leaves the rebuild table; it is never read.
    con.execute("DROP TABLE IF EXISTS haz_revisions_rebuild")
    # From the schema's own DDL: a CREATE TABLE AS would drop the NOT NULL
    # constraints and observed_at's default, which every writer relies on.
    con.execute(_CORE_TABLE_DDL["haz_revisions"].replace(
        "haz_revisions (", "haz_revisions_rebuild (", 1))
    # Nothing writes haz_revisions on this connection until the swap, so a
    # rowid read here still names the same row when the batch copies it.
    con.execute(f"CREATE OR REPLACE TEMP TABLE _haz_revision_keep AS {dup_sql} "
                "SELECT rid FROM l WHERE NOT dup")
    rid_min, rid_max = con.execute(
        "SELECT MIN(rid), MAX(rid) FROM _haz_revision_keep").fetchone()
    # A batch is a window of PHYSICAL rows, not of kept rows: kept rows are
    # sparse, and a window of N kept rows can span many times N rows that are
    # read only to be dropped. The window bounds what one batch holds.
    windows = [] if rid_min is None else range(int(rid_min), int(rid_max) + 1, batch_rows)
    slimmed = 0
    for lo in windows:
        hi = lo + batch_rows - 1
        rows = con.execute(
            f"SELECT {cols} FROM haz_revisions WHERE rowid BETWEEN {lo} AND {hi} "
            f"AND rowid IN (SELECT rid FROM _haz_revision_keep WHERE rid BETWEEN {lo} AND {hi})"
        ).fetchall()
        if not rows:
            continue
        out_rows = []
        i_detail = _REVISION_COLUMNS.index("detail_json")
        i_rule = _REVISION_COLUMNS.index("rule_fired")
        for row in rows:
            row = list(row)
            row[i_rule] = _row_rule(row[i_rule], row[i_detail])
            row[i_detail], did = _slim_row_detail(row[i_detail])
            slimmed += int(did)
            out_rows.append(row)
        frame = pd.DataFrame(out_rows, columns=list(_REVISION_COLUMNS))
        con.register("_haz_revision_batch", frame)
        try:
            con.execute("BEGIN TRANSACTION")
            try:
                con.execute(f"INSERT INTO haz_revisions_rebuild ({cols}) "
                            f"SELECT {cols} FROM _haz_revision_batch")
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
        finally:
            con.unregister("_haz_revision_batch")
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute("DROP TABLE IF EXISTS haz_revisions_old")
        con.execute("ALTER TABLE haz_revisions RENAME TO haz_revisions_old")
        con.execute("ALTER TABLE haz_revisions_rebuild RENAME TO haz_revisions")
        con.execute("DROP TABLE haz_revisions_old")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    con.execute("DROP TABLE IF EXISTS _haz_revision_keep")
    con.execute("DROP TABLE IF EXISTS _haz_revision_narrow")
    return slimmed


def compact_all(
    con: "duckdb.DuckDBPyConnection",
    *,
    sources: Iterable[str] | None = None,
    apply: bool = False,
    rulebook: Any | None = None,
) -> dict[str, dict[str, Any]]:
    """Compact every raw cache. One bad table must not lose the rest."""

    ensure_haz_schema(con)
    keep = _keep_revisions(rulebook)
    wanted = tuple(sources) if sources is not None else COMPACTABLE_SOURCES
    out: dict[str, dict[str, Any]] = {}
    for source in wanted:
        try:
            out[source] = compact_raw_cache(con, source, apply=apply, keep=keep)
        except Exception as exc:
            LOG.error("[retention] %s failed: %s", source, exc)
            out[source] = {"source": source, "error": str(exc), "applied": False}
    if sources is None:
        try:
            out["haz_revisions"] = collapse_revisions(con, apply=apply)
        except Exception as exc:
            LOG.error("[retention] haz_revisions failed: %s", exc)
            out["haz_revisions"] = {"source": "haz_revisions", "error": str(exc), "applied": False}
    return out


def summarize(results: dict[str, dict[str, Any]]) -> list[str]:
    """Human-readable plan/outcome, for a step summary."""

    lines = [
        f"{'source':<22} {'rows before':>14} {'rows after':>14} {'removed':>14}",
        "-" * 68,
    ]
    total_before = total_after = 0
    for source in sorted(results):
        r = results[source]
        if r.get("error"):
            lines.append(f"{source:<22} {'ERROR: ' + r['error']:>44}")
            continue
        if not r.get("present"):
            lines.append(f"{source:<22} {'(table absent)':>44}")
            continue
        total_before += r["rows_before"]
        total_after += r["rows_after"]
        lines.append(
            f"{source:<22} {r['rows_before']:>14,} {r['rows_after']:>14,} "
            f"{r['removed']:>14,}"
        )
    lines.append("-" * 68)
    lines.append(
        f"{'TOTAL':<22} {total_before:>14,} {total_after:>14,} "
        f"{total_before - total_after:>14,}"
    )
    lines.append("")
    lines.append(
        "Row deletion alone does not shrink the file: DuckDB reuses freed "
        "blocks but never truncates. A copy to a fresh database is what "
        "reclaims the space."
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m resolver.hazard_resolution.retention [--db PATH] [--apply]``.

    Dry run by default. The caller is expected to read the plan before
    applying, which is why the compaction workflow gates on an ``apply``
    input rather than doing both in one go.

    ``--db`` resolves the same way every other machine entry point does
    (``duckdb_io.get_db``, env-first), so a workflow that points the machine
    at a non-default database points retention at the same one.
    """
    import argparse

    from resolver.db.duckdb_io import close_db, get_db

    parser = argparse.ArgumentParser(description="Compact the machine's raw caches")
    parser.add_argument("--db", default=None, help="DuckDB path or duckdb:/// URL")
    parser.add_argument("--apply", action="store_true",
                        help="Write the changes (default: report the plan only)")
    parser.add_argument("--sources", nargs="*", default=None,
                        help=f"Subset of {list(COMPACTABLE_SOURCES)}")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[retention] %(message)s")
    con = get_db(args.db)
    try:
        results = compact_all(con, sources=args.sources, apply=args.apply)
        if args.apply:
            # Flush, so the file the compaction step copies reflects the
            # rebuild rather than a WAL the next reader has to replay.
            con.execute("CHECKPOINT")
    finally:
        close_db(con)

    print("\n".join(summarize(results)))
    failed = [s for s, r in results.items() if r.get("error")]
    if failed:
        print(f"::warning::retention could not compact: {', '.join(failed)}")
    if not args.apply:
        print("::notice::dry run — nothing written. Re-run with --apply.")
    # A retention failure must not fail the job that carries the canonical DB:
    # the worst case is a file that stays large, not one that is damaged.
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
