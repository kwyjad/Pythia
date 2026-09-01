# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Per-table on-disk size attribution for a DuckDB file.

Why this exists: on 2026-09-01 the publish aborted because the release copy
was 5.1 GB after stripping 12.5 GB of ``haz_raw_*`` from a 17.7 GB canonical
DB -- and nothing anywhere could say what the remaining 5.1 GB WAS. There was
no per-table size reporting in the repo at all, so the only way to decide what
to cut was to guess. A size crisis you cannot attribute is one you fix twice.

How the estimate works, and what it is honest about:

``pragma_storage_info('<table>')`` lists one row per column segment with the
``block_id`` it lives in. Counting DISTINCT block ids and multiplying by
``block_size`` attributes on-disk bytes to a table. It reads catalog metadata
only -- no table scan -- so it is cheap even on a 17.7 GB file.

It is an ESTIMATE, and the two ways it can mislead are both reported rather
than hidden:

* Blocks holding the catalog, the free list and other metadata belong to no
  table, so the per-table numbers do not sum to the file size. The shortfall
  is printed as its own ``unattributed`` row instead of being silently
  dropped -- a size report whose numbers do not add up is not usable.
* A large free list means the file is holding space that no live data needs.
  DuckDB reuses freed blocks but never truncates, so deleting rows stops
  growth without reclaiming bytes; only a copy to a fresh database reclaims.
  ``free_blocks`` is therefore the signal that a compaction is due, and it is
  reported alongside the tables.

Alternatives considered and rejected as defaults: ``duckdb_tables()``'s
``estimated_size`` is a ROW estimate, not bytes. ``SUM(octet_length(col))`` is
a full scan and measures uncompressed logical bytes -- the right number for
"what would retention save", the wrong one for "what is on disk" -- so it is
available behind ``logical_bytes()`` but never runs by default.

    python -m scripts.ci.db_table_sizes --db data/resolver.duckdb --top 30
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Tables are enumerated from information_schema, but the name is interpolated
# into a pragma call, so it is quoted defensively rather than trusted.
_IDENT_SAFE = set(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
)


@dataclass
class TableSize:
    table: str
    rows: int
    blocks: int
    est_bytes: int
    pct_of_file: float


def _quote(name: str) -> str:
    if not name or not set(name) <= _IDENT_SAFE:
        raise ValueError(f"refusing to introspect unusual table name: {name!r}")
    return name


def resolve_catalog(con: Any, catalog: str = "main") -> str:
    """The name this database actually answers to.

    ``duckdb.connect(path)`` names the catalog after the FILE (``resolver``),
    while ``ATTACH ... AS work`` names it ``work`` -- and neither is ``main``.
    Both call sites exist (the inspect report connects directly, the release
    builder attaches), so resolve rather than assume: guessing wrong makes
    every size read zero, which looks like an empty database rather than a
    bad catalog name.
    """
    names = {
        str(r[0])
        for r in con.execute(
            "SELECT database_name FROM pragma_database_size()"
        ).fetchall()
    }
    if catalog in names:
        return catalog
    try:
        current = str(con.execute("SELECT current_database()").fetchone()[0])
    except Exception:
        current = ""
    if current in names:
        return current
    # Ignore DuckDB's built-in scratch catalogs when picking a fallback.
    real = sorted(names - {"memory", "system", "temp"})
    return real[0] if real else catalog


def whole_file(con: Any, catalog: str = "main") -> dict[str, int]:
    """Block accounting for one attached database."""

    catalog = resolve_catalog(con, catalog)
    row = con.execute(
        "SELECT block_size, total_blocks, used_blocks, free_blocks, wal_size "
        "FROM pragma_database_size() WHERE database_name = ?",
        [catalog],
    ).fetchone()
    if row is None:
        return {}
    block_size, total, used, free, wal = row
    block_size = int(block_size or 0)
    return {
        "block_size": block_size,
        "total_blocks": int(total or 0),
        "used_blocks": int(used or 0),
        "free_blocks": int(free or 0),
        "file_bytes": int(total or 0) * block_size,
        "used_bytes": int(used or 0) * block_size,
        "free_bytes": int(free or 0) * block_size,
        "wal_size": wal,
    }


def _tables(con: Any, catalog: str) -> list[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_catalog = ? AND table_schema = 'main' ORDER BY table_name",
        [catalog],
    ).fetchall()
    return [str(r[0]) for r in rows]


def _blocks_for(con: Any, catalog: str, table: str) -> int:
    """DISTINCT blocks backing one table.

    The aggregation runs IN SQL: pragma_storage_info emits one row per column
    segment, which on a multi-GB table is millions of rows -- materialising
    them in Python would make the size report more expensive than the problem
    it is diagnosing.

    ``additional_block_ids`` carries the overflow blocks of segments that span
    more than one block (large strings), so it is unioned in; ignoring it
    would under-count exactly the text-heavy tables this exists to find.
    """
    target = f"{_quote(catalog)}.{_quote(table)}"
    row = con.execute(
        f"""
        SELECT COUNT(DISTINCT b) FROM (
            SELECT block_id AS b
              FROM pragma_storage_info('{target}') WHERE block_id >= 0
            UNION
            SELECT UNNEST(additional_block_ids) AS b
              FROM pragma_storage_info('{target}')
        )
        """
    ).fetchone()
    return int(row[0] or 0) if row else 0


def size_report(
    con: Any, *, catalog: str = "main", top: int | None = None
) -> list[TableSize]:
    """Per-table byte attribution, largest first."""

    catalog = resolve_catalog(con, catalog)
    info = whole_file(con, catalog)
    block_size = info.get("block_size") or 0
    file_bytes = info.get("file_bytes") or 0

    out: list[TableSize] = []
    for table in _tables(con, catalog):
        try:
            blocks = _blocks_for(con, catalog, table)
            rows = int(
                con.execute(
                    f"SELECT COUNT(*) FROM {_quote(catalog)}.{_quote(table)}"
                ).fetchone()[0]
            )
        except Exception:
            # A view, or a table this build of DuckDB will not introspect.
            # One awkward table must not cost the whole report.
            continue
        est = blocks * block_size
        out.append(
            TableSize(
                table=table,
                rows=rows,
                blocks=blocks,
                est_bytes=est,
                pct_of_file=(100.0 * est / file_bytes) if file_bytes else 0.0,
            )
        )

    out.sort(key=lambda t: t.est_bytes, reverse=True)
    return out[:top] if top else out


def logical_bytes(con: Any, table: str, *, catalog: str = "main") -> int:
    """Uncompressed bytes of the text/blob columns. Opt-in: FULL SCAN.

    This answers "how much would dropping these columns save", which is not
    the same question as "what is on disk" -- compression means the two differ,
    often by a lot. Never called by the report.
    """
    catalog = resolve_catalog(con, catalog)
    cols = con.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_catalog = ? AND table_schema = 'main' AND table_name = ?",
        [catalog, table],
    ).fetchall()
    wide = [
        c for c, t in cols if str(t).upper() in {"VARCHAR", "BLOB", "TEXT", "JSON"}
    ]
    if not wide:
        return 0
    # CAST to BLOB first: octet_length takes BLOB/BIT in current DuckDB, and
    # the cast is what makes this a BYTE count rather than a character count.
    expr = " + ".join(
        f"COALESCE(octet_length(CAST({_quote(c)} AS BLOB)), 0)" for c in wide
    )
    row = con.execute(
        f"SELECT COALESCE(SUM({expr}), 0) FROM {_quote(catalog)}.{_quote(table)}"
    ).fetchone()
    return int(row[0] or 0)


def _mb(n: float) -> str:
    return f"{n / (1024 * 1024):,.1f}"


def format_markdown(
    report: list[TableSize], info: dict[str, int], *, title: str = "Table sizes"
) -> str:
    """Render the report, remainder row included.

    The remainder is not decoration: without it the per-table numbers appear
    not to add up and the reader cannot tell whether that is metadata or a bug.
    """
    file_bytes = info.get("file_bytes") or 0
    attributed = sum(t.est_bytes for t in report)
    remainder = file_bytes - attributed

    lines = [
        f"### {title}",
        "",
        f"File {_mb(file_bytes)} MB | used {_mb(info.get('used_bytes', 0))} MB | "
        f"free list {_mb(info.get('free_bytes', 0))} MB "
        f"({info.get('free_blocks', 0):,} blocks)",
        "",
        "| Table | Rows | Est. MB | % of file |",
        "|---|---:|---:|---:|",
    ]
    for t in report:
        lines.append(
            f"| `{t.table}` | {t.rows:,} | {_mb(t.est_bytes)} | {t.pct_of_file:.1f}% |"
        )
    pct_rem = (100.0 * remainder / file_bytes) if file_bytes else 0.0
    lines.append(
        f"| _unattributed (free list, catalog, metadata)_ | — | "
        f"{_mb(remainder)} | {pct_rem:.1f}% |"
    )
    lines += [
        "",
        "Estimated from DISTINCT storage blocks per table (metadata only, no "
        "table scan). A large free list means the file is holding space no "
        "live data needs — DuckDB reuses freed blocks but never truncates, so "
        "only a copy to a fresh database reclaims it.",
    ]
    return "\n".join(lines)


def report_lines(report: list[TableSize], info: dict[str, int]) -> list[str]:
    """Compact one-line-per-table form, for a workflow log."""

    file_bytes = info.get("file_bytes") or 0
    attributed = sum(t.est_bytes for t in report)
    out = [
        f"file {_mb(file_bytes)} MB, free list {_mb(info.get('free_bytes', 0))} MB"
    ]
    for t in report:
        out.append(
            f"  {t.table:<40} {_mb(t.est_bytes):>10} MB  {t.pct_of_file:>5.1f}%  "
            f"{t.rows:,} rows"
        )
    out.append(
        f"  {'unattributed (free list, catalog)':<40} "
        f"{_mb(file_bytes - attributed):>10} MB"
    )
    return out


def main(argv: list[str] | None = None) -> int:
    import duckdb

    parser = argparse.ArgumentParser(description="Per-table sizes for a DuckDB file")
    parser.add_argument("--db", required=True)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)

    path = Path(args.db)
    if not path.exists():
        print(f"::error::DB not found: {path}")
        return 1

    con = duckdb.connect()
    try:
        con.execute(f"ATTACH '{path}' AS probe (READ_ONLY)")
        report = size_report(con, catalog="probe", top=args.top)
        info = whole_file(con, "probe")
    finally:
        con.close()

    if args.markdown:
        print(format_markdown(report, info, title=f"Table sizes — {path.name}"))
    else:
        for line in report_lines(report, info):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
