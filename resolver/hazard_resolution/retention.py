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


def compact_raw_cache(
    con: "duckdb.DuckDBPyConnection", source: str, *, apply: bool = False
) -> dict[str, Any]:
    """Collapse one raw cache to the newest row per cell.

    ``apply=False`` (the default) reports the plan and writes nothing.
    """
    table = _assert_compactable(source)
    if not _table_exists(con, table):
        return {"source": source, "table": table, "present": False,
                "rows_before": 0, "rows_after": 0, "removed": 0, "applied": False}

    partition = ", ".join(_PARTITION)
    rows_before = int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    # What a reader would keep. Reported alongside rows_after so the two can be
    # eyeballed against each other in a dry run.
    distinct_cells = int(
        con.execute(
            f"SELECT COUNT(*) FROM (SELECT DISTINCT {partition} FROM {table})"
        ).fetchone()[0]
    )

    if not apply:
        return {
            "source": source, "table": table, "present": True,
            "rows_before": rows_before, "rows_after": distinct_cells,
            "removed": rows_before - distinct_cells,
            "distinct_cells": distinct_cells, "applied": False,
        }

    tmp = f"{table}__compact"
    con.execute(f"DROP TABLE IF EXISTS {tmp}")
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
        ) = 1
        """
    )
    rows_after = int(con.execute(f"SELECT COUNT(*) FROM {tmp}").fetchone()[0])
    con.execute(f"DROP TABLE {table}")
    con.execute(f"ALTER TABLE {tmp} RENAME TO {table}")

    LOG.info(
        "[retention] %s: %d -> %d rows (%d duplicate revisions removed)",
        table, rows_before, rows_after, rows_before - rows_after,
    )
    return {
        "source": source, "table": table, "present": True,
        "rows_before": rows_before, "rows_after": rows_after,
        "removed": rows_before - rows_after,
        "distinct_cells": distinct_cells, "applied": True,
    }


def compact_all(
    con: "duckdb.DuckDBPyConnection",
    *,
    sources: Iterable[str] | None = None,
    apply: bool = False,
) -> dict[str, dict[str, Any]]:
    """Compact every raw cache. One bad table must not lose the rest."""

    ensure_haz_schema(con)
    wanted = tuple(sources) if sources is not None else COMPACTABLE_SOURCES
    out: dict[str, dict[str, Any]] = {}
    for source in wanted:
        try:
            out[source] = compact_raw_cache(con, source, apply=apply)
        except Exception as exc:
            LOG.error("[retention] %s failed: %s", source, exc)
            out[source] = {"source": source, "error": str(exc), "applied": False}
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
    """CLI: ``python -m resolver.hazard_resolution.retention --db ... [--apply]``.

    Dry run by default. The caller is expected to read the plan before
    applying, which is why the compaction workflow gates on an ``apply``
    input rather than doing both in one go.
    """
    import argparse

    import duckdb

    parser = argparse.ArgumentParser(description="Compact the machine's raw caches")
    parser.add_argument("--db", required=True)
    parser.add_argument("--apply", action="store_true",
                        help="Write the changes (default: report the plan only)")
    parser.add_argument("--sources", nargs="*", default=None,
                        help=f"Subset of {list(COMPACTABLE_SOURCES)}")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[retention] %(message)s")
    con = duckdb.connect(args.db)
    try:
        results = compact_all(con, sources=args.sources, apply=args.apply)
        if args.apply:
            # Flush, so the file the next step copies reflects the rebuild.
            con.execute("CHECKPOINT")
    finally:
        con.close()

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
