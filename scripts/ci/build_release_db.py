# Pythia
# Copyright (c) 2025 Kevin Wyjad
# Licensed under the Pythia Non-Commercial Public License v1.0.
# See the LICENSE file in the project root for details.

"""Build the DB copy that gets published to the release.

Why this exists: GitHub caps a release asset at 2 GiB. On 2026-08-06 the
canonical DB had grown to 3.0 GB and ``gh release upload --clobber`` DELETED
the working 453 MB asset before the oversized replacement was rejected,
leaving the release with no database at all. Two defects, one incident: the
published copy carried data nothing serves, and the upload was destructive
before it was validated.

What the release is FOR is the API and dashboard. Those read facts, forecasts,
scores, questions and interpretations — they never read one ``haz_*`` table
(the PA resolution machine's own raw caches and working tables). Those live in
the canonical ARTIFACT, which the machine resumes from and which this script
never touches. Dropping them from the published copy is what makes the asset
serveable again; compaction is a free bonus on top.

On 2026-09-01 it happened again and the narrow prefix was why: this stripped
only ``haz_raw_``, so the machine's WORKING tables still shipped — and
``haz_triggers`` alone carries one ``evidence_of_absence_json`` per assessed
cell across ~200 countries x 3 hazards x ~360 months. The publish aborted at
5.1 GB. ``grep haz_ pythia/api/ web/src/`` returns nothing, and the release
asset's only consumer is ``pythia/api/db_sync.py`` (every pipeline job takes
the canonical artifact via ``.github/actions/download-canonical-db``), so the
prefix is now the whole ``haz_`` family.

It also reports per-table sizes on the way through. The 2026-09-01 abort could
not be attributed to any table because nothing measured one, which meant the
only way to choose what to cut was to guess. Now the same log block that says
"over the limit" says what is over it.

    python -m scripts.ci.build_release_db \
        --src data/resolver.duckdb --out data_out/resolver.duckdb \
        [--max-bytes 2147483648] [--keep-all]

Exit codes: 0 on success, 1 when the result still exceeds --max-bytes (so the
publish workflow stops BEFORE it clobbers a working asset).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# GitHub's per-asset ceiling for a release.
GITHUB_ASSET_LIMIT_BYTES = 2 * 1024 * 1024 * 1024

# Prefixes of tables excluded from the PUBLISHED copy: the PA resolution
# machine's raw source caches (documents, tracks, provider payloads) AND its
# working, ledger and verdict tables. Nothing in pythia/api/** or web/src/**
# references any of them — verified by grep, and pinned by a test so a future
# consumer cannot quietly start depending on a table this strips.
#
# Widening this is a decision about what the dashboard can ever show, so it
# stays deliberate: if a future view wants the machine's verdicts, narrow this
# back to the tables that view needs and say so here. Until then the release
# is a serving copy and the machine's state belongs in the canonical artifact.
EXCLUDED_TABLE_PREFIXES = ("haz_",)


def _connect():
    import duckdb

    return duckdb.connect()


def tables_in(con, alias: str) -> list[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_catalog = ? AND table_schema = 'main' ORDER BY table_name",
        [alias],
    ).fetchall()
    return [str(r[0]) for r in rows]


def is_excluded(table: str) -> bool:
    return any(table.lower().startswith(p) for p in EXCLUDED_TABLE_PREFIXES)


def compact_database(src: str, out: str) -> dict[str, int]:
    """Copy src -> out to reclaim space. The ONLY thing that shrinks a file.

    DuckDB reuses freed blocks but never truncates, so deleting rows stops a
    file growing without giving any bytes back. Reclaiming means writing a
    fresh database, which is what ``COPY FROM DATABASE`` does here.

    Shared with the retention workflow rather than reimplemented there: a
    second copy of this would eventually diverge from the release path, and
    this is the operation with the ~40 GB disk peak if it is run in the wrong
    order (compact the ROWS first, then the file).
    """
    src_path, out_path = Path(src), Path(out)
    if not src_path.exists():
        raise FileNotFoundError(f"source DB not found: {src_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.unlink(missing_ok=True)
    Path(str(out_path) + ".wal").unlink(missing_ok=True)

    src_bytes = src_path.stat().st_size
    con = _connect()
    try:
        con.execute(f"ATTACH '{src_path}' AS csrc (READ_ONLY)")
        con.execute(f"ATTACH '{out_path}' AS cdst")
        con.execute("COPY FROM DATABASE csrc TO cdst")
        con.execute("CHECKPOINT cdst")
    finally:
        con.close()
    out_bytes = out_path.stat().st_size
    return {
        "src_bytes": src_bytes,
        "out_bytes": out_bytes,
        "saved_bytes": src_bytes - out_bytes,
    }


def _size_lines(con, alias: str, top: int) -> list[str]:
    """Per-table sizes, or a note saying why there are none.

    Never fatal: this is a diagnostic wrapped around the one job that must not
    fail for a diagnostic's sake. A publish that stops because the size REPORT
    broke would be a worse outcome than a publish with no size report.
    """
    if top <= 0:
        return []
    try:
        from scripts.ci.db_table_sizes import report_lines, size_report, whole_file

        return report_lines(
            size_report(con, catalog=alias, top=top), whole_file(con, alias)
        )
    except Exception as exc:  # pragma: no cover - diagnostics must never block
        return [f"(size report unavailable: {exc})"]


def build_release_db(
    src: str, out: str, *, keep_all: bool = False, report_top: int = 15
) -> dict[str, object]:
    """Copy src -> out, dropping excluded tables and compacting.

    Two passes on purpose: ``COPY FROM DATABASE`` reproduces the schema
    faithfully, but DROPping afterwards leaves the freed blocks in the file —
    so the second copy is what actually reclaims them.
    """
    src_path, out_path = Path(src), Path(out)
    if not src_path.exists():
        raise FileNotFoundError(f"source DB not found: {src_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_path = out_path.with_suffix(out_path.suffix + ".work")
    for stale in (out_path, work_path):
        stale.unlink(missing_ok=True)
        Path(str(stale) + ".wal").unlink(missing_ok=True)

    src_bytes = src_path.stat().st_size
    dropped: list[str] = []
    before: list[str] = []
    after: list[str] = []

    con = _connect()
    try:
        con.execute(f"ATTACH '{src_path}' AS src (READ_ONLY)")
        con.execute(f"ATTACH '{work_path}' AS work")
        con.execute("COPY FROM DATABASE src TO work")
        # Blocks are not allocated until the write is flushed, so an
        # un-checkpointed database measures as zero bytes.
        con.execute("CHECKPOINT work")
        before = _size_lines(con, "work", report_top)
        if not keep_all:
            for table in tables_in(con, "work"):
                if is_excluded(table):
                    con.execute(f"DROP TABLE IF EXISTS work.{table}")
                    dropped.append(table)
        con.execute("DETACH src")
        con.execute(f"ATTACH '{out_path}' AS dst")
        con.execute("COPY FROM DATABASE work TO dst")
        con.execute("CHECKPOINT dst")
        after = _size_lines(con, "dst", report_top)
    finally:
        con.close()
    work_path.unlink(missing_ok=True)
    Path(str(work_path) + ".wal").unlink(missing_ok=True)

    out_bytes = out_path.stat().st_size
    return {
        "src_bytes": src_bytes,
        "out_bytes": out_bytes,
        "dropped_tables": dropped,
        "saved_bytes": src_bytes - out_bytes,
        "sizes_before": before,
        "sizes_after": after,
    }


def _mb(n: float) -> str:
    return f"{n / (1024 * 1024):.1f} MB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-bytes", type=int, default=GITHUB_ASSET_LIMIT_BYTES)
    parser.add_argument("--keep-all", action="store_true",
                        help="Compact only; keep every table (diagnostics)")
    parser.add_argument("--report-top", type=int, default=15,
                        help="Per-table sizes to log before/after (0 disables)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[build_release_db] %(message)s")
    stats = build_release_db(
        args.src, args.out, keep_all=args.keep_all, report_top=args.report_top
    )
    LOGGER.info(
        "source %s -> published %s (saved %s)",
        _mb(stats["src_bytes"]), _mb(stats["out_bytes"]), _mb(stats["saved_bytes"]),
    )
    if stats["dropped_tables"]:
        LOGGER.info(
            "dropped %d machine-cache table(s) the API never reads: %s",
            len(stats["dropped_tables"]), ", ".join(stats["dropped_tables"]),
        )
    # Log the attribution in the SAME block as any size error, so whoever
    # reads the failure also reads what caused it.
    for label, key in (("canonical", "sizes_before"), ("published", "sizes_after")):
        for line in stats.get(key) or []:
            LOGGER.info("%s | %s", label, line)
    out_bytes = int(stats["out_bytes"])
    if out_bytes > args.max_bytes:
        # Loud and non-destructive: the caller must NOT proceed to a clobbering
        # upload. This is the guard whose absence cost the release its asset.
        LOGGER.error(
            "published DB is %s, over the %s release-asset limit — refusing to "
            "publish rather than delete the working asset and fail the upload",
            _mb(out_bytes), _mb(args.max_bytes),
        )
        print(f"::error::Release DB {_mb(out_bytes)} exceeds the "
              f"{_mb(args.max_bytes)} GitHub asset limit; publish aborted "
              "before clobbering the existing asset.")
        return 1
    print(f"RELEASE_DB_BYTES={out_bytes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
